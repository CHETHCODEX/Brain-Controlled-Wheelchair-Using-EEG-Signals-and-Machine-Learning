import unittest
from pathlib import Path

from src.ingestion.edf_loader import annotation_label, recording_path


class LabelTests(unittest.TestCase):
    def test_hand_imagery_labels(self):
        for run in (4, 8, 12):
            self.assertEqual(annotation_label(run, "T0"), 0)
            self.assertEqual(annotation_label(run, "T1"), 1)
            self.assertEqual(annotation_label(run, "T2"), 2)

    def test_bilateral_imagery_labels(self):
        for run in (6, 10, 14):
            self.assertEqual(annotation_label(run, "T0"), 0)
            self.assertEqual(annotation_label(run, "T1"), 3)
            self.assertEqual(annotation_label(run, "T2"), 3)

    def test_rejects_movement_runs_and_unknown_events(self):
        with self.assertRaises(ValueError):
            annotation_label(3, "T1")
        with self.assertRaises(ValueError):
            annotation_label(4, "T9")

    def test_recording_path(self):
        self.assertEqual(
            recording_path(Path("data/raw"), 1, 4),
            Path("data/raw/S001/S001R04.edf"),
        )
        with self.assertRaises(ValueError):
            recording_path(Path("data/raw"), 0, 4)


if __name__ == "__main__":
    unittest.main()