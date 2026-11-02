"""Recorded-EEG-triggered autonomous navigation, exclusively for Webots."""

import argparse
import heapq
import math
from pathlib import Path
import re
import time

from src.bci.runtime import EMERGENCY, ROOT, RUNTIME, TELEMETRY, read_state, write_state
from src.ingestion.preprocessor import filter_raw, load_recording
from src.ipc.zmq_publisher import Publisher, intent_probabilities
from src.streaming.inference_engine import load_decoder, windows

STATUS = RUNTIME / "navigation.json"
GOAL = (6.0, -3.2)
CLEARANCE = 0.65
RESOLUTION = 0.1


class HospitalMap:
    def __init__(self, world):
        text = Path(world).read_text(encoding="utf-8")
        self.obstacles = []
        for block in re.findall(r"ClinicBlock\s*\{([^{}]*)\}", text):
            translation = re.search(r"translation\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", block)
            size = re.search(r"size\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", block)
            if translation is None or size is None:
                raise ValueError("Hospital obstacle lacks translation or size")
            x, y, _ = map(float, translation.groups())
            width, depth, _ = map(float, size.groups())
            self.obstacles.append((x - width / 2, y - depth / 2,
                                   x + width / 2, y + depth / 2))
        if not self.obstacles:
            raise ValueError("No hospital obstacles were loaded")

    def clear(self, point):
        x, y = point
        return (-8.2 <= x <= 8.2 and -5.2 <= y <= 5.2
                and all(not (left - CLEARANCE <= x <= right + CLEARANCE
                             and bottom - CLEARANCE <= y <= top + CLEARANCE)
                        for left, bottom, right, top in self.obstacles))

    def segment_clear(self, start, end):
        steps = max(1, math.ceil(math.dist(start, end) / (RESOLUTION / 2)))
        return all(self.clear((start[0] + (end[0] - start[0]) * i / steps,
                               start[1] + (end[1] - start[1]) * i / steps))
                   for i in range(steps + 1))

    def plan(self, start, goal=GOAL):
        if not self.clear(start) or not self.clear(goal):
            raise ValueError("Start or destination lacks wheelchair clearance")
        cell = lambda p: (round(p[0] / RESOLUTION), round(p[1] / RESOLUTION))
        point = lambda c: (c[0] * RESOLUTION, c[1] * RESOLUTION)
        first, last = cell(start), cell(goal)
        queue = [(0.0, first)]
        costs, parents = {first: 0.0}, {}
        while queue:
            _, current = heapq.heappop(queue)
            if current == last:
                route = [goal]
                while current != first:
                    route.append(point(current))
                    current = parents[current]
                route.append(start)
                route.reverse()
                simplified = [route[0]]
                index = 0
                while index < len(route) - 1:
                    next_index = index + 1
                    for candidate in range(index + 2, len(route)):
                        if self.segment_clear(route[index], route[candidate]):
                            next_index = candidate
                    simplified.append(route[next_index])
                    index = next_index
                if not all(self.segment_clear(a, b) for a, b in zip(simplified, simplified[1:])):
                    raise ValueError("Planned route fails clearance validation")
                return simplified
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.segment_clear(point(current), point(neighbor)):
                    continue
                cost = costs[current] + RESOLUTION
                if cost < costs.get(neighbor, math.inf):
                    costs[neighbor], parents[neighbor] = cost, current
                    heapq.heappush(queue, (cost + math.dist(point(neighbor), goal), neighbor))
        raise ValueError("No wheelchair-clear route to caregiver station")


def checked_telemetry():
    state = read_state(TELEMETRY)
    if state.get("engine") != "Webots" or not -0.25 <= time.time() - state["timestamp"] <= 0.75:
        raise RuntimeError("Webots telemetry is stale; start the world in real-time mode")
    position, yaw = state.get("position"), state.get("yaw")
    if (not isinstance(position, list) or len(position) != 3
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in position)
            or not isinstance(yaw, (int, float)) or not math.isfinite(yaw)):
        raise ValueError("Webots position/heading is invalid")
    if state.get("emergency_stop") or (EMERGENCY.exists() and read_state(EMERGENCY)["active"]):
        raise RuntimeError("Emergency stop is latched; reset it explicitly before restarting")
    return state


def status(phase, **details):
    payload = {"timestamp": time.time(), "phase": phase,
               "mode": "Recorded EEG request + confirmed autonomous Webots navigation",
               "destination": "Caregiver station", **details}
    write_state(STATUS, payload)
    print(f"{phase}: {details}", flush=True)


def request_trip(model, recording, publisher):
    metadata, report, predict = load_decoder(model)
    if report["task"] != "multiclass":
        raise ValueError("Trip request requires the four-state EEG model")
    samples = filter_raw(load_recording(recording), metadata.get("notch", 60.0)).get_data()
    consecutive = 0
    status("Replaying recorded EEG", recording=str(recording), model=str(model))
    for index, window in enumerate(windows(samples)):
        checked_telemetry()
        probabilities = predict(window)
        ordered = sorted(probabilities.values(), reverse=True)
        intent = max(probabilities, key=probabilities.get)
        qualifying = intent == "FORWARD" and ordered[0] >= 0.65 and ordered[0] - ordered[1] >= 0.15
        consecutive = consecutive + 1 if qualifying else 0
        publisher.send(intent_probabilities("REST"))
        if index % 4 == 0 or consecutive:
            status("Replaying recorded EEG", detected_intent=intent, confidence=ordered[0],
                   recording_seconds=2 + index * 0.25)
        if consecutive >= 2:
            status("EEG trip request accepted", confidence=ordered[0],
                   warning="EEG requests the trip; the planner chooses subsequent movements")
            return
        time.sleep(0.25)
    raise RuntimeError("Recording ended without a qualifying FORWARD request; no trip started")


def steering(position, yaw, target):
    error = math.atan2(target[1] - position[1], target[0] - position[0]) - yaw
    error = math.atan2(math.sin(error), math.cos(error))
    if abs(error) > 0.09:
        return "TURN_LEFT" if error > 0 else "TURN_RIGHT"
    return "FORWARD"


def navigate(hospital, publisher, timeout=180):
    state = checked_telemetry()
    route = hospital.plan(tuple(state["position"][:2]))
    status("Navigating autonomously", route=route, source="autonomous planner (not EEG predictions)")
    index, previous, settle_until = 1, "REST", 0.0
    deadline = time.monotonic() + timeout
    progress_time, best_distance = time.monotonic(), math.inf
    while time.monotonic() < deadline:
        state = checked_telemetry()
        position = tuple(state["position"][:2])
        if not hospital.clear(position):
            raise RuntimeError("Wheelchair departed the clearance-safe map region")
        distance = math.dist(position, route[index])
        if distance < 0.13:
            publisher.send(intent_probabilities("REST"))
            if index == len(route) - 1:
                # Hold REST until deceleration is complete before reporting arrival.
                for _ in range(15):
                    checked_telemetry()
                    publisher.send(intent_probabilities("REST"))
                    time.sleep(0.1)
                stopped = checked_telemetry()
                position = tuple(stopped["position"][:2])
                if (math.dist(position, GOAL) > 0.2
                        or abs(stopped["linear_velocity"]) > 0.01
                        or abs(stopped["angular_velocity"]) > 0.01):
                    raise RuntimeError("Arrival position or stopped-state verification failed")
                status("Destination reached", position=position, distance_to_goal=math.dist(position, GOAL))
                return
            index += 1
            previous, settle_until = "REST", time.monotonic() + 0.8
            progress_time, best_distance = time.monotonic(), math.inf
            status("Waypoint reached", waypoint=index - 1, position=position)
            continue
        if distance < best_distance - 0.04:
            best_distance, progress_time = distance, time.monotonic()
        if time.monotonic() - progress_time > 18:
            raise RuntimeError("No route progress; obstacle or controller fault. Trip stopped.")
        if state["reason"] in ("sensor_fault", "telemetry_fault", "invalid_command", "invalid_json"):
            raise RuntimeError(f"Controller fault: {state['reason']}")
        command = steering(position, state["yaw"], route[index])
        if time.monotonic() < settle_until:
            command = "REST"
        elif command != previous and previous != "REST":
            command, settle_until = "REST", time.monotonic() + 0.8
        if command == "FORWARD" and state["reason"] == "obstacle":
            raise RuntimeError("Unmapped obstacle blocks the route; stopped for operator review")
        publisher.send(intent_probabilities(command))
        previous = command
        time.sleep(0.1)
    raise RuntimeError("Navigation timed out; trip stopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "multiclass_lda.joblib")
    parser.add_argument("--recording", type=Path, default=ROOT / "data" / "raw" / "S001" / "S001R06.edf")
    parser.add_argument("--confirm-start", action="store_true",
                        help="Explicitly authorize autonomous simulation travel after the EEG request")
    args = parser.parse_args()
    if not args.confirm_start:
        parser.error("Review the simulation route and supply --confirm-start to authorize travel")
    publisher = None
    try:
        hospital = HospitalMap(ROOT / "simulation" / "worlds" / "hospital_rehab.wbt")
        state = checked_telemetry()
        hospital.plan(tuple(state["position"][:2]))
        publisher = Publisher(source="recorded_eeg_assisted_navigation")
        request_trip(args.model, args.recording, publisher)
        navigate(hospital, publisher)
    except (OSError, ValueError, RuntimeError) as error:
        status("Trip stopped", error=str(error))
        raise
    except KeyboardInterrupt:
        status("Trip cancelled by operator")
    finally:
        if publisher is not None:
            try:
                publisher.send(intent_probabilities("REST"))
            finally:
                publisher.close()


if __name__ == "__main__":
    main()
