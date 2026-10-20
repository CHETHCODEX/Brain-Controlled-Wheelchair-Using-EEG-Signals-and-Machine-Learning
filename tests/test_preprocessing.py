import unittest

import mne
import numpy as np

from src.ingestion.edf_loader import CHANNELS
from src.ingestion.preprocessor import extract_epochs, filter_raw, model_windows


class PreprocessingTests(unittest.TestCase):
    def raw(self):
        generator = np.random.default_rng(42)
        data = generator.normal(0, 10e-6, (7, 160 * 20))
        raw = mne.io.RawArray(data, mne.create_info(list(CHANNELS), 160, "eeg"), verbose=False)
        raw.set_annotations(mne.Annotations([0, 5, 10, 15], [4, 4, 4, 2], ["T0", "T1", "T2", "T1"]))
        return raw

    def test_epoch_shape_and_labels(self):
        epochs = extract_epochs(self.raw(), 4)
        self.assertEqual(epochs.get_data().shape, (3, 7, 400))
        np.testing.assert_array_equal(epochs.events[:, 2] - 1, [0, 1, 2])
        self.assertEqual(model_windows(epochs.get_data()).shape, (3, 7, 320))

    def test_filter_does_not_mutate_input(self):
        raw = self.raw()
        original = raw.get_data().copy()
        cleaned = filter_raw(raw)
        np.testing.assert_array_equal(raw.get_data(), original)
        self.assertTrue(np.isfinite(cleaned.get_data()).all())
        self.assertFalse(np.allclose(cleaned.get_data(), original, atol=1e-12))

    def test_model_windows_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            model_windows(np.zeros((2, 7, 200)))
        with self.assertRaises(ValueError):
            model_windows(np.full((2, 7, 320), np.nan))


if __name__ == "__main__":
    unittest.main()