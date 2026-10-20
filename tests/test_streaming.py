import unittest

import numpy as np

from src.streaming.ring_buffer import RingBuffer


class BufferTests(unittest.TestCase):
    def test_warmup_wraparound_and_copy(self):
        buffer = RingBuffer(channels=2, capacity=5)
        buffer.append(np.tile(np.arange(3), (2, 1)))
        with self.assertRaises(ValueError):
            buffer.window()
        buffer.append(np.tile(np.arange(3, 7), (2, 1)))
        np.testing.assert_array_equal(buffer.window()[0], [2, 3, 4, 5, 6])
        window = buffer.window()
        window[:] = -1
        np.testing.assert_array_equal(buffer.window()[0], [2, 3, 4, 5, 6])

    def test_large_chunk_and_invalid_samples(self):
        buffer = RingBuffer(channels=1, capacity=3)
        buffer.append([[0, 1, 2, 3, 4]])
        np.testing.assert_array_equal(buffer.window(), [[2, 3, 4]])
        with self.assertRaises(ValueError):
            buffer.append([[np.nan]])
        with self.assertRaises(ValueError):
            buffer.append([[1], [2]])


if __name__ == "__main__":
    unittest.main()