import math
import time
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.bci.navigation_demo import GOAL, HospitalMap, checked_telemetry, steering
from src.bci.runtime import ROOT, read_state, write_state


class NavigationTests(unittest.TestCase):
    def test_stale_invalid_or_emergency_telemetry_cannot_start_trip(self):
        state = {"engine": "Webots", "timestamp": time.time(), "position": [-7, 0, 0.15],
                 "yaw": 0.0, "emergency_stop": False}
        with patch("src.bci.navigation_demo.read_state", return_value=state), \
                patch.object(Path, "exists", return_value=False):
            self.assertEqual(checked_telemetry(), state)
            state["timestamp"] -= 5
            with self.assertRaises(RuntimeError):
                checked_telemetry()
            state["timestamp"] = time.time()
            state["yaw"] = float("nan")
            with self.assertRaises(ValueError):
                checked_telemetry()
            state["yaw"], state["emergency_stop"] = 0.0, True
            with self.assertRaises(RuntimeError):
                checked_telemetry()

    def test_hospital_route_clears_obstacles_and_ends_at_station(self):
        hospital = HospitalMap(ROOT / "simulation" / "worlds" / "hospital_rehab.wbt")
        route = hospital.plan((-7.0, 0.0))
        self.assertEqual(route[0], (-7.0, 0.0))
        self.assertEqual(route[-1], GOAL)
        self.assertGreater(len(route), 2)
        self.assertTrue(all(hospital.segment_clear(a, b) for a, b in zip(route, route[1:])))
        self.assertFalse(hospital.clear((6.2, -4.5)))
        with self.assertRaises(ValueError):
            hospital.plan((-7, 0), (6.2, -4.5))

    def test_steering_and_heading_wrap(self):
        self.assertEqual(steering((0, 0), 0, (1, 0)), "FORWARD")
        self.assertEqual(steering((0, 0), 0, (0, 1)), "TURN_LEFT")
        self.assertEqual(steering((0, 0), 0, (0, -1)), "TURN_RIGHT")
        self.assertEqual(steering((0, 0), math.pi - 0.01, (-1, -0.01)), "FORWARD")

    def test_runtime_retries_only_transient_file_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "state.json"
            original = Path.replace
            attempts = []

            def replace(path, destination):
                attempts.append(path)
                if len(attempts) < 3:
                    raise PermissionError("File temporarily in use")
                return original(path, destination)

            with patch.object(Path, "replace", replace):
                write_state(target, {"phase": "test"})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(read_state(target), {"phase": "test"})
            with patch.object(Path, "replace", side_effect=PermissionError("Locked")):
                with self.assertRaises(PermissionError):
                    write_state(target, {"phase": "failed"})
            self.assertEqual(read_state(target), {"phase": "test"})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            with patch.object(Path, "read_text", side_effect=[
                PermissionError("Locked"), '{"phase": "test"}'
            ]):
                self.assertEqual(read_state(target), {"phase": "test"})
            with patch.object(Path, "read_text", side_effect=PermissionError("Locked")):
                with self.assertRaises(PermissionError):
                    read_state(target)


if __name__ == "__main__":
    unittest.main()
