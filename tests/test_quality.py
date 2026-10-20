import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.image as image
import mne
import numpy as np
from scipy.signal import welch

from src.ingestion.edf_loader import CHANNELS
from src.ingestion.preprocessor import filter_raw
from src.ingestion.quality_report import channel_statistics, write_report


class QualityTests(unittest.TestCase):
    def signal(self):
        times = np.arange(160 * 12) / 160
        samples = 20e-6 * np.sin(2 * np.pi * 10 * times) + 10e-6 * np.sin(2 * np.pi * 60 * times)
        return mne.io.RawArray(np.tile(samples, (7, 1)),
                              mne.create_info(list(CHANNELS), 160, "eeg"), verbose=False)

    def test_filter_preserves_motor_band_and_suppresses_line_noise(self):
        raw = self.signal()
        frequency, before = welch(raw.get_data()[0, 160:-160], fs=160, nperseg=320)
        _, after = welch(filter_raw(raw).get_data()[0, 160:-160], fs=160, nperseg=320)
        motor = np.argmin(abs(frequency - 10))
        line = np.argmin(abs(frequency - 60))
        self.assertGreater(after[motor] / before[motor], 0.8)
        self.assertLess(after[line] / before[line], 0.01)

    def test_report_artifacts_and_raw_data_preserved(self):
        raw = self.signal()
        original = raw.get_data().copy()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = write_report(raw, output, "synthetic")
            stored = json.loads((output / "synthetic_quality.json").read_text())
            self.assertEqual(stored["channels"], list(CHANNELS))
            self.assertEqual(report["samples"], 1920)
            self.assertEqual(report["before"]["flat_channels"], [])
            for suffix in ["psd", "traces"]:
                pixels = image.imread(output / f"synthetic_{suffix}.png")
                self.assertGreater(pixels.shape[0], 500)
                self.assertGreater(float(pixels[:, :, :3].std()), 0.05)
        np.testing.assert_array_equal(raw.get_data(), original)

    def test_invalid_samples_and_flat_channel(self):
        with self.assertRaises(ValueError):
            channel_statistics(np.full((7, 10), np.nan))
        self.assertEqual(channel_statistics(np.zeros((1, 10)))["flat_channels"], [0])


if __name__ == "__main__":
    unittest.main()