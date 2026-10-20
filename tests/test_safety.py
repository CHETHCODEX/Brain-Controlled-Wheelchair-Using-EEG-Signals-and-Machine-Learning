import copy
import unittest

from src.ipc.command_schema import INTENTS, NAVIGATION_INTENTS, make_command, validate_command
from src.webots_controller.safety_override import SafetyController


class SafetyTests(unittest.TestCase):
    def test_obstacle_lock_prevents_repeated_prediction_creep(self):
        controller = self.moving()
        self.assertEqual(controller.wheel_velocities(self.surround(front_center=0.2), now=0.3, dt=0.016), (0, 0))
        for sequence in range(2, 8):
            controller.accept(self.command(sequence), now=0.4, wall_time=100)
            self.assertEqual(controller.wheel_velocities(self.surround(front_center=0.45), now=0.4, dt=0.016), (0, 0))
        controller.wheel_velocities(self.surround(), now=0.5, dt=0.016)
        self.assertNotIn("FORWARD", controller.obstacle_locks)

    def surround(self, **changes):
        distances = {name: 2.0 for name in ("front_left", "front_center", "front_right",
                     "rear_left", "rear_center", "rear_right", "side_left", "side_right")}
        return {**distances, **changes}

    def test_smooth_acceleration_reverse_and_emergency_latch(self):
        controller = self.moving()
        first = controller.wheel_velocities(self.surround(), now=0.3, dt=0.016)
        self.assertGreater(first[0], 0)
        self.assertLess(controller.linear_velocity, 0.01)
        self.assertEqual(controller.wheel_velocities(self.surround(front_center=0.2), now=0.3, dt=0.016), (0, 0))
        for sequence in (2, 3):
            packet = make_command({name: float(name == "BACKWARD") for name in NAVIGATION_INTENTS},
                                  sequence, wall_time=100)
            controller.accept(packet, now=0.3, wall_time=100)
        for step in range(25):
            controller.wheel_velocities(self.surround(front_center=0.2), now=0.4, dt=0.016)
        self.assertLess(controller.linear_velocity, 0)
        self.assertEqual(controller.wheel_velocities(self.surround(rear_center=0.2), now=0.4, dt=0.016), (0, 0))
        controller.emergency_stop()
        controller.accept(self.command(4), now=0.5, wall_time=100)
        self.assertEqual(controller.wheel_velocities(self.surround(), now=0.5, dt=0.016), (0, 0))
        self.assertTrue(controller.emergency_latched)
        controller.emergency_stop(False)
        self.assertEqual(controller.intent, "REST")

    def test_session_handover_and_headset_loss(self):
        controller = SafetyController()
        for session in ("first", "second"):
            packet = make_command({name: float(name == "FORWARD") for name in NAVIGATION_INTENTS},
                                  0, wall_time=100, session_id=session)
            self.assertTrue(controller.accept(packet, now=0, wall_time=100))
            self.assertEqual(controller.intent, "REST")
        packet["session_id"] = "first"
        self.assertFalse(controller.accept(packet, now=0, wall_time=100))
        packet["session_id"], packet["sequence_id"] = "second", 1
        packet["headset_connected"] = False
        self.assertTrue(controller.accept(packet, now=0, wall_time=100))
        self.assertEqual(controller.reason, "signal_lost_or_poor")

    def test_five_intent_protocol_and_signal_metadata(self):
        probabilities = {name: float(name == "BACKWARD") for name in NAVIGATION_INTENTS}
        packet = make_command(probabilities, 0, wall_time=100, session_id="new-session",
                              source="simulated", signal_quality=0.9)
        self.assertEqual(validate_command(packet, wall_time=100)["intent"], "BACKWARD")
        packet["signal_quality"] = float("nan")
        with self.assertRaises(ValueError):
            validate_command(packet, wall_time=100)

    def command(self, sequence, intent="FORWARD", confidence=1.0):
        probabilities = {name: (1 - confidence) / 3 for name in INTENTS}
        probabilities[intent] = confidence
        return make_command(probabilities, sequence, wall_time=100.0)

    def moving(self):
        controller = SafetyController()
        controller.accept(self.command(0), now=0, wall_time=100)
        self.assertEqual(controller.wheel_velocities([2, 2, 2], now=0), (0, 0))
        controller.accept(self.command(1), now=0.25, wall_time=100)
        return controller

    def test_debounce_and_kinematics(self):
        controller = self.moving()
        left, right = controller.wheel_velocities([2, 2, 2], now=0.5)
        self.assertAlmostEqual(left, 0.4 / 0.15)
        self.assertAlmostEqual(right, left)
        for sequence in [2, 3]:
            controller.accept(self.command(sequence, "TURN_LEFT"), now=0.5, wall_time=100)
        left, right = controller.wheel_velocities([2, 2, 2], now=0.5)
        self.assertLess(left, 0)
        self.assertGreater(right, 0)

    def test_obstacle_watchdog_and_sensor_fault_stop(self):
        for distances, now, reason in [([2, 0.35, 2], 0.5, "obstacle"),
                                       ([2, 2, 2], 1.25, "heartbeat_timeout"),
                                       ([2, float("nan"), 2], 0.5, "sensor_fault")]:
            controller = self.moving()
            self.assertEqual(controller.wheel_velocities(distances, now), (0, 0))
            self.assertEqual(controller.reason, reason)

    def test_low_confidence_duplicate_and_invalid_stop(self):
        controller = self.moving()
        controller.accept(self.command(2, confidence=0.6), now=0.5, wall_time=100)
        self.assertEqual(controller.intent, "REST")
        self.assertFalse(controller.accept(self.command(2), now=0.5, wall_time=100))
        self.assertFalse(controller.accept({"intent": "FORWARD"}, now=0.5, wall_time=100))

    def test_schema_rejects_corrupt_or_stale_fields(self):
        packet = self.command(0)
        for field, value in [("timestamp", 98), ("timestamp", 101),
                             ("confidence", float("nan")), ("sequence_id", True),
                             ("intent", []), ("safety_flag", "false")]:
            invalid = copy.deepcopy(packet)
            invalid[field] = value
            with self.assertRaises(ValueError):
                validate_command(invalid, wall_time=100)
        packet["probabilities"]["REST"] = 1.0
        with self.assertRaises(ValueError):
            validate_command(packet, wall_time=100)


if __name__ == "__main__":
    unittest.main()