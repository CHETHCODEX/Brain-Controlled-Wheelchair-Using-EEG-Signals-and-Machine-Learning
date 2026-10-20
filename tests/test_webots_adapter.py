import json
import unittest
from unittest.mock import Mock

import zmq

from src.ipc.command_schema import make_command
from src.ipc.zmq_publisher import intent_probabilities
from src.webots_controller.safety_override import SafetyController
from src.webots_controller.wheelchair_driver import drive_step


class AdapterTests(unittest.TestCase):
    def test_surround_sensor_adapter_uses_motion_ramp(self):
        motors = [Mock(), Mock()]
        sensors = [Mock(getValue=Mock(return_value=2.0)) for index in range(8)]
        subscriber = Mock()
        safety = SafetyController()
        for sequence in (0, 1):
            packet = make_command(intent_probabilities("BACKWARD"), sequence)
            subscriber.recv.side_effect = [json.dumps(packet).encode(), zmq.Again()]
            velocity = drive_step(motors, sensors, subscriber, safety, dt=0.016)
        self.assertLess(velocity[0], 0)
        self.assertGreater(velocity[0], -0.1)
        sensors[4].getValue.return_value = 0.2
        subscriber.recv.side_effect = zmq.Again()
        self.assertEqual(drive_step(motors, sensors, subscriber, safety, dt=0.016), (0, 0))

    def test_messages_drive_motors_and_sensor_overrides(self):
        motors = [Mock(), Mock()]
        sensors = [Mock(), Mock(), Mock()]
        for sensor in sensors:
            sensor.getValue.return_value = 2.0
        safety = SafetyController()
        subscriber = Mock()
        for sequence in [0, 1]:
            packet = make_command(intent_probabilities("FORWARD"), sequence)
            subscriber.recv.side_effect = [json.dumps(packet).encode(), zmq.Again()]
            velocities = drive_step(motors, sensors, subscriber, safety)
        self.assertGreater(velocities[0], 0)
        motors[0].setVelocity.assert_called_with(velocities[0])
        sensors[1].getValue.return_value = 0.2
        subscriber.recv.side_effect = zmq.Again()
        self.assertEqual(drive_step(motors, sensors, subscriber, safety), (0, 0))
        motors[0].setVelocity.assert_called_with(0)

    def test_invalid_json_brakes(self):
        motors = [Mock(), Mock()]
        sensors = [Mock(getValue=Mock(return_value=2.0)) for index in range(3)]
        subscriber = Mock(recv=Mock(return_value=b"not JSON"))
        safety = SafetyController()
        safety.intent = "FORWARD"
        self.assertEqual(drive_step(motors, sensors, subscriber, safety), (0, 0))


if __name__ == "__main__":
    unittest.main()