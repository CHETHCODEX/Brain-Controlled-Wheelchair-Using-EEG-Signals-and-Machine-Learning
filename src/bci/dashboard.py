"""Local BCI dashboard, classified-intent input, and caregiver safety bridge."""

import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
import time
from urllib.parse import urlparse

from src.bci.adapters import ExternalClassifier, SimulatedHeadset
from src.bci.runtime import EMERGENCY, ROOT, TELEMETRY, read_state, write_state
from src.ipc.command_schema import make_command
from src.ipc.zmq_publisher import ENDPOINT, Publisher, intent_probabilities
from src.webots_controller.safety_override import SafetyController


class DashboardService:
    def __init__(self, mode="simulated", emergency_path=EMERGENCY, telemetry_path=TELEMETRY):
        if mode not in ("simulated", "external"):
            raise ValueError("Input mode must be simulated or external")
        self.mode = mode
        self.adapter = SimulatedHeadset() if mode == "simulated" else ExternalClassifier()
        self.emergency_path, self.telemetry_path = emergency_path, telemetry_path
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.thread = None
        self.state = {"source": mode, "starting": True}
        self.history = deque(maxlen=40)
        self.gate = SafetyController()
        self.previous_emergency = None
        self.rearm_until = 0.0
        try:
            self.emergency = read_state(emergency_path)["active"]
            if type(self.emergency) is not bool:
                raise ValueError("Invalid latch state")
        except FileNotFoundError:
            self.emergency = False
        except (OSError, ValueError, KeyError):
            self.emergency = True

    def set_emergency(self, active):
        if type(active) is not bool:
            raise ValueError("Emergency state must be boolean")
        with self.lock:
            write_state(self.emergency_path, {"active": active, "timestamp": time.time(), "actor": "caregiver"})
            self.emergency = active
            if not active:
                self.rearm_until = time.monotonic() + 2.0

    def telemetry(self):
        try:
            payload = read_state(self.telemetry_path)
            if not -0.25 <= time.time() - payload["timestamp"] <= 1.5:
                return None
            return payload
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def tick(self, publisher):
        sample = self.adapter.read()
        with self.lock:
            probabilities = sample["probabilities"]
            if self.emergency or time.monotonic() < self.rearm_until:
                probabilities = intent_probabilities("REST")
            packet = publisher.send(probabilities, signal_quality=sample["signal_quality"],
                                    headset_connected=sample["headset_connected"], safety_flag=self.emergency)
            displayed = make_command(sample["probabilities"], packet["sequence_id"], source=self.mode,
                                     signal_quality=sample["signal_quality"],
                                     headset_connected=sample["headset_connected"])
            if self.previous_emergency != self.emergency:
                self.gate.emergency_stop(self.emergency)
                self.previous_emergency = self.emergency
            self.gate.accept(packet)
            self.history.appendleft({"timestamp": displayed["timestamp"], "intent": displayed["intent"],
                                     "confidence": displayed["confidence"], "source": self.mode,
                                     "accepted": self.gate.intent == displayed["intent"] and not self.emergency})
            self.state = {
                "source": self.mode, "simulated": self.mode == "simulated", "packet": displayed,
                "output_packet": packet,
                "waves": sample["waves"], "features": sample["features"], "classifier": sample["classifier"],
                "validated_intent": self.gate.intent, "gate_reason": self.gate.reason,
                "stable": self.gate.consecutive >= 2, "emergency": self.emergency,
                "webots": self.telemetry(), "history": list(self.history),
                "timestamp": time.time(), "rearming": time.monotonic() < self.rearm_until,
            }

    def start(self, endpoint=ENDPOINT):
        def loop():
            publisher = None
            try:
                publisher = Publisher(endpoint, source=self.mode)
                self.ready.set()
                while not self.stop_event.is_set():
                    self.tick(publisher)
                    self.stop_event.wait(0.25)
            except Exception as error:
                self.error = str(error)
                self.ready.set()
            finally:
                if publisher is not None:
                    publisher.send(intent_probabilities("REST"), safety_flag=self.emergency)
                    publisher.close()
        self.thread = threading.Thread(target=loop, name="bci-input", daemon=True)
        self.thread.start()
        if not self.ready.wait(5) or self.error:
            raise RuntimeError(self.error or "BCI publisher did not start")

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(5)

    def snapshot(self):
        with self.lock:
            return {**self.state, "service_error": self.error}


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *arguments):
        return

    def reply(self, status, payload):
        encoded = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            return self.reply(200, self.server.service.snapshot())
        if path == "/assets/hospital_floor.png":
            target = ROOT / "simulation" / "textures" / "hospital_floor.png"
        else:
            target = (ROOT / "dashboard" / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
            if not target.is_relative_to(ROOT / "dashboard"):
                return self.reply(403, {"error": "Forbidden"})
        if not target.is_file():
            return self.reply(404, {"error": "Not found"})
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if host.split(":")[0] not in ("localhost", "127.0.0.1") or origin != "http://" + host:
            return self.reply(403, {"error": "Same-origin local requests only"})
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self.reply(415, {"error": "JSON is required"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 131072:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length))
            if urlparse(self.path).path == "/api/emergency":
                self.server.service.set_emergency(payload["active"])
                return self.reply(200, {"active": self.server.service.emergency})
            if urlparse(self.path).path == "/api/prediction":
                if self.server.service.mode != "external":
                    return self.reply(409, {"error": "Restart with --input external to accept headset predictions"})
                self.server.service.adapter.ingest(payload)
                return self.reply(202, {"accepted": True})
            return self.reply(404, {"error": "Unknown endpoint"})
        except (ValueError, KeyError, TypeError, OSError) as error:
            return self.reply(400, {"error": str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", choices=["simulated", "external"], default="simulated")
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    service = DashboardService(arguments.input)
    server = None
    try:
        for port in range(arguments.port, arguments.port + 10):
            try:
                server = ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)
                break
            except OSError:
                continue
        if server is None:
            raise RuntimeError("No free dashboard port in the requested range")
        server.service = service
        service.start()
        print(f"BCI dashboard: http://127.0.0.1:{server.server_port} | input={arguments.input}", flush=True)
        print("Simulation only. Stop other publishers using TCP port 5555 before launching.", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
        if server:
            server.server_close()


if __name__ == "__main__":
    main()