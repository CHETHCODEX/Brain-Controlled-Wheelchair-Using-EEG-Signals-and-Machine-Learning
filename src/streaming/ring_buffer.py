"""Fixed-size channel-first EEG window."""

import numpy as np


class RingBuffer:
    def __init__(self, channels=7, capacity=320):
        if channels < 1 or capacity < 1:
            raise ValueError("Buffer dimensions must be positive")
        self.data = np.zeros((channels, capacity), dtype=np.float64)
        self.position = 0
        self.count = 0

    @property
    def ready(self):
        return self.count == self.data.shape[1]

    def append(self, samples):
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[0] != self.data.shape[0]:
            raise ValueError("Samples must have shape (channels, time)")
        if not np.isfinite(samples).all():
            raise ValueError("Samples must be finite")
        capacity = self.data.shape[1]
        if samples.shape[1] >= capacity:
            self.data[:] = samples[:, -capacity:]
            self.position = 0
            self.count = capacity
            return
        first = min(samples.shape[1], capacity - self.position)
        self.data[:, self.position:self.position + first] = samples[:, :first]
        remainder = samples.shape[1] - first
        if remainder:
            self.data[:, :remainder] = samples[:, first:]
        self.position = (self.position + samples.shape[1]) % capacity
        self.count = min(capacity, self.count + samples.shape[1])

    def window(self):
        if not self.ready:
            raise ValueError("Buffer does not yet contain a full window")
        return np.concatenate((self.data[:, self.position:], self.data[:, :self.position]), axis=1)