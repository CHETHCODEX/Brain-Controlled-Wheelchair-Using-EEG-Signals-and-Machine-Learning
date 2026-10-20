"""Compact EEGNet for seven-channel, two-second motor-imagery windows."""

import numpy as np
import torch
from torch import nn

from src.ingestion.edf_loader import CHANNELS


def validate_windows(data):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3 or data.shape[1:] != (len(CHANNELS), 320) or len(data) == 0:
        raise ValueError("Expected nonempty EEG windows shaped (trials, 7, 320)")
    if not np.isfinite(data).all():
        raise ValueError("EEG windows must contain finite samples in volts")
    return data


class ChannelStandardizer:
    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, training_data):
        training_data = validate_windows(training_data)
        self.mean = training_data.mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
        self.scale = np.maximum(training_data.std(axis=(0, 2), keepdims=True, dtype=np.float64),
                                1e-12).astype(np.float32)
        return self

    def transform(self, data):
        data = validate_windows(data)
        if self.mean is None or self.scale is None:
            raise ValueError("Fit normalization on training data before transforming windows")
        normalized = (data - self.mean) / self.scale
        if not np.isfinite(normalized).all():
            raise ValueError("Normalization produced non-finite values")
        return normalized


class EEGNet(nn.Module):
    def __init__(self, classes=4):
        super().__init__()
        if classes not in (2, 4):
            raise ValueError("EEGNet supports two or four classes")
        self.features = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, kernel_size=(1, 32), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(len(CHANNELS), 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout2d(0.25),
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(16, 16, kernel_size=(1, 16), groups=16, bias=False),
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout2d(0.25),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(16 * 10, classes)

    def forward(self, samples):
        if samples.ndim != 4 or tuple(samples.shape[1:]) != (1, len(CHANNELS), 320):
            raise ValueError("EEGNet expects tensors shaped (trials, 1, 7, 320)")
        return self.classifier(self.features(samples))