import unittest

import numpy as np
import zmq

from src.ipc.command_schema import validate_command
from src.ipc.zmq_publisher import Publisher, intent_probabilities
from src.streaming.inference_engine import smooth_probabilities, windows
from src.webots_controller.safety_override import SafetyController


class IPCTests(unittest.TestCase):
    def test_loopback_packet_reaches_safety_controller(self):
        publisher = Publisher("tcp://127.0.0.1:*")
        context = zmq.Context()
        subscriber = context.socket(zmq.SUB)
        subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        subscriber.setsockopt(zmq.LINGER, 0)
        subscriber.connect(publisher.socket.getsockopt_string(zmq.LAST_ENDPOINT))
        try:
            controller = SafetyController()
            received = 0
            for attempt in range(20):
                publisher.send(intent_probabilities("FORWARD"))
                if subscriber.poll(50):
                    packet = subscriber.recv_json()
                    validate_command(packet)
                    self.assertTrue(controller.accept(packet))
                    received += 1
                    if received == 2:
                        break
            self.assertEqual(received, 2)
            self.assertGreater(controller.wheel_velocities([2, 2, 2])[0], 0)
        finally:
            subscriber.close()
            context.term()
            publisher.close()

    def test_replay_window_stride(self):
        samples = np.tile(np.arange(400), (7, 1))
        result = list(windows(samples))
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].shape, (7, 320))
        np.testing.assert_array_equal(result[1][0], np.arange(40, 360))

    def test_probability_smoothing_window(self):
        history = []
        p1 = {"REST": 0.8, "TURN_LEFT": 0.2, "TURN_RIGHT": 0.0, "FORWARD": 0.0}
        p2 = {"REST": 0.4, "TURN_LEFT": 0.6, "TURN_RIGHT": 0.0, "FORWARD": 0.0}
        p3 = {"REST": 0.2, "TURN_LEFT": 0.8, "TURN_RIGHT": 0.0, "FORWARD": 0.0}
        out1 = smooth_probabilities(history, p1, 2)
        out2 = smooth_probabilities(history, p2, 2)
        out3 = smooth_probabilities(history, p3, 2)
        self.assertAlmostEqual(out1["REST"], 0.8)
        self.assertAlmostEqual(out2["REST"], 0.6)
        self.assertAlmostEqual(out2["TURN_LEFT"], 0.4)
        self.assertAlmostEqual(out3["REST"], 0.3)
        self.assertAlmostEqual(out3["TURN_LEFT"], 0.7)
        with self.assertRaises(ValueError):
            smooth_probabilities([], p1, 0)


if __name__ == "__main__":
    unittest.main()
