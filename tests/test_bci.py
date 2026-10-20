import tempfile
import json
import threading
from pathlib import Path
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

import numpy as np

from src.bci.adapters import ExternalClassifier, SignalProcessor, SimulatedHeadset
from src.bci.runtime import read_state, write_state
from src.bci.dashboard import DashboardHandler, DashboardService
from src.ipc.command_schema import make_command
from src.ipc.zmq_publisher import intent_probabilities


class InputTests(unittest.TestCase):
    def test_emergency_inhibits_output_without_hiding_detected_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            service = DashboardService("external", emergency_path=Path(directory) / "stop.json")
            service.adapter.ingest(make_command(intent_probabilities("FORWARD"), 0, session_id="headset"))
            service.set_emergency(True)
            publisher = Mock()
            publisher.send.side_effect = lambda probabilities, **status: make_command(probabilities, 0, **status)
            service.tick(publisher)
            state = service.snapshot()
            self.assertEqual(state["packet"]["intent"], "FORWARD")
            self.assertEqual(state["output_packet"]["intent"], "REST")
            self.assertTrue(state["output_packet"]["safety_flag"])
            self.assertFalse(state["history"][0]["accepted"])

    def test_caregiver_latch_api_and_cross_origin_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            service = DashboardService(emergency_path=Path(directory) / "emergency.json")
            server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
            server.service = service
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            origin = f"http://127.0.0.1:{server.server_port}"
            try:
                for active in (True, False):
                    request = Request(origin + "/api/emergency", data=json.dumps({"active": active}).encode(),
                                      headers={"Content-Type": "application/json", "Origin": origin})
                    with urlopen(request, timeout=2) as response:
                        self.assertEqual(response.status, 200)
                    self.assertEqual(read_state(service.emergency_path)["active"], active)
                request = Request(origin + "/api/emergency", data=b'{"active":true}',
                                  headers={"Content-Type": "application/json", "Origin": "http://untrusted.invalid"})
                with self.assertRaises(HTTPError) as error:
                    urlopen(request, timeout=2)
                self.assertEqual(error.exception.code, 403)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(2)

    def test_synthetic_input_is_labeled_and_all_five_intents_occur(self):
        headset = SimulatedHeadset()
        intents = set()
        for step in range(150):
            sample = headset.read()
            self.assertEqual(sample["source"], "simulated")
            self.assertAlmostEqual(sum(sample["probabilities"].values()), 1)
            self.assertTrue(np.isfinite(sample["waves"]).all())
            intents.add(max(sample["probabilities"], key=sample["probabilities"].get))
        self.assertEqual(intents, {"REST", "FORWARD", "TURN_LEFT", "TURN_RIGHT", "BACKWARD"})

    def test_external_input_fails_closed_on_dropout_and_duplicate(self):
        adapter = ExternalClassifier()
        packet = make_command(intent_probabilities("FORWARD"), 0, session_id="headset")
        adapter.ingest(packet, now=0)
        self.assertTrue(adapter.read(now=0.2)["headset_connected"])
        self.assertFalse(adapter.read(now=0.8)["headset_connected"])
        with self.assertRaises(ValueError):
            adapter.ingest(packet, now=0.2)
        packet["sequence_id"] = 1
        packet["safety_flag"] = True
        with self.assertRaises(ValueError):
            adapter.ingest(packet)

    def test_atomic_runtime_state_and_bad_signal_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            write_state(path, {"active": True})
            self.assertTrue(read_state(path)["active"])
            self.assertEqual(len(list(Path(directory).glob("*.tmp"))), 0)
        with self.assertRaises(ValueError):
            SignalProcessor().process(np.zeros((7, 320)))


if __name__ == "__main__":
    unittest.main()