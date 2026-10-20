"""Synthetic test input and a classified-intent interface for future EEG hardware."""

from collections import deque
import threading
import time

import numpy as np
from scipy.signal import butter, sosfilt

from src.ingestion.edf_loader import CHANNELS
from src.ipc.command_schema import NAVIGATION_INTENTS, validate_command

SCHEDULE = (("REST", 4), ("FORWARD", 6), ("REST", 2), ("TURN_LEFT", 3),
            ("REST", 2), ("FORWARD", 5), ("REST", 2), ("TURN_RIGHT", 3),
            ("REST", 2), ("BACKWARD", 4), ("REST", 3))


class SignalProcessor:
    def __init__(self):
        self.sos = butter(4, [8, 30], btype="bandpass", fs=160, output="sos")
        self.filter_state = np.zeros((len(self.sos), len(CHANNELS), 2))
        self.samples = deque(maxlen=320)

    def process(self, samples):
        samples = np.asarray(samples, dtype=float)
        if samples.shape != (7, 40) or not np.isfinite(samples).all():
            raise ValueError("EEG chunks must be finite (7, 40) samples in volts at 160 Hz")
        filtered, self.filter_state = sosfilt(self.sos, samples, axis=-1, zi=self.filter_state)
        self.samples.extend(filtered.T.tolist())
        spectrum = np.abs(np.fft.rfft(filtered, axis=-1)) ** 2
        frequencies = np.fft.rfftfreq(40, d=1 / 160)
        features = {
            "mu_power": float(np.mean(spectrum[:, (frequencies >= 8) & (frequencies <= 12)])),
            "beta_power": float(np.mean(spectrum[:, (frequencies >= 16) & (frequencies <= 30)])),
        }
        return (np.asarray(self.samples).T * 1e6).tolist(), features


class SimulatedHeadset:
    source = "simulated"

    def __init__(self, seed=42):
        self.generator = np.random.default_rng(seed)
        self.sample = 0
        self.processor = SignalProcessor()

    def read(self):
        elapsed = self.sample / 160
        position = elapsed % sum(duration for intent, duration in SCHEDULE)
        intent = "REST"
        for candidate, duration in SCHEDULE:
            if position < duration:
                intent = candidate
                break
            position -= duration
        times = (np.arange(40) + self.sample) / 160
        signals = []
        for channel in range(7):
            amplitude = 12e-6 * (0.6 if intent != "REST" and channel in (0, 2) else 1)
            signals.append(amplitude * np.sin(2 * np.pi * (10 + channel * 0.15) * times)
                           + 5e-6 * np.sin(2 * np.pi * 20 * times + channel)
                           + self.generator.normal(0, 2e-6, 40))
        self.sample += 40
        waves, features = self.processor.process(signals)
        confidence = 0.90 + 0.04 * np.sin(elapsed * 0.7)
        probabilities = {name: float((1 - confidence) / 4) for name in NAVIGATION_INTENTS}
        probabilities[intent] = float(confidence)
        return {"probabilities": probabilities, "signal_quality": 0.96,
                "headset_connected": True, "waves": waves, "features": features,
                "source": self.source, "classifier": "Synthetic intent sequence (not a trained EEG decoder)"}


class ExternalClassifier:
    source = "external"

    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None
        self.received = None
        self.last_sequences = {}
        self.processor = SignalProcessor()

    def ingest(self, packet, now=None):
        validate_command(packet)
        session = packet.get("session_id")
        if session is None:
            raise ValueError("External predictions require session_id")
        if packet.get("safety_flag") or packet.get("caregiver_reset"):
            raise ValueError("The headset adapter cannot issue caregiver commands")
        waves, features = None, {}
        with self.lock:
            if packet["sequence_id"] <= self.last_sequences.get(session, -1):
                raise ValueError("External prediction is duplicated or out of order")
            if len(self.last_sequences) >= 32 and session not in self.last_sequences:
                raise ValueError("Too many input sessions; restart the adapter")
            if "eeg" in packet:
                waves, features = self.processor.process(packet["eeg"])
            self.last_sequences[session] = packet["sequence_id"]
            self.latest = {"probabilities": dict(packet["probabilities"]),
                           "signal_quality": packet.get("signal_quality", 1.0),
                           "headset_connected": packet.get("headset_connected", True),
                           "waves": waves, "features": features, "source": self.source,
                           "classifier": "External classifier (not independently validated)"}
            self.received = time.monotonic() if now is None else now

    def read(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.latest is None or now - self.received >= 0.75:
                return {"probabilities": {name: float(name == "REST") for name in NAVIGATION_INTENTS},
                        "signal_quality": 0.0, "headset_connected": False, "waves": None,
                        "features": {}, "source": self.source, "classifier": "External classifier offline"}
            return dict(self.latest)