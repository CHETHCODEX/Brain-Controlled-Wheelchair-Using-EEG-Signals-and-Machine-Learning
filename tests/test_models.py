import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from src.ingestion.edf_loader import CHANNELS
from src.models.csp_classifier import predict_window, train
from src.streaming.inference_engine import load_decoder


class ModelTests(unittest.TestCase):
    def test_grouped_training_and_serialized_prediction(self):
        generator = np.random.default_rng(42)
        labels = np.tile([1, 2], 18)
        data = generator.normal(0, 1e-6, (36, 7, 400))
        data[labels == 1, 0] *= 5
        data[labels == 2, 2] *= 5
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "epochs.npz"
            model = Path(directory) / "model.joblib"
            np.savez(dataset, X=data, y=labels, subjects=np.ones(36),
                     runs=np.repeat([4, 8, 12], 12),
                     metadata=json.dumps({"channels": list(CHANNELS), "sfreq": 160.0}))
            report = train(dataset, model)
            self.assertEqual(report["folds"], 3)
            self.assertGreater(report["accuracy"], 0.9)
            self.assertEqual(report["class_counts"], [18, 18])
            self.assertEqual(report["majority_class_accuracy"], 0.5)
            for split in report["run_splits"]:
                self.assertFalse(set(split["training_runs"]) & set(split["validation_runs"]))
            bundle = joblib.load(model)
            probabilities = predict_window(bundle, data[0, :, :320])
            self.assertAlmostEqual(sum(probabilities.values()), 1.0)
            self.assertEqual(probabilities["FORWARD"], 0.0)
            with patch.dict("sys.modules", {"torch": None}):
                _, replay_report, predict = load_decoder(model)
                self.assertEqual(replay_report["task"], "binary")
                self.assertEqual(predict(data[0, :, :320]), probabilities)
            with self.assertRaises(ValueError):
                predict_window(bundle, data[0])

    def test_multiclass_and_svm_alternative(self):
        generator = np.random.default_rng(7)
        labels = np.tile([0, 1, 2, 3], 12)
        data = generator.normal(0, 1e-6, (48, 7, 400))
        for label in range(4):
            data[labels == label, label] *= 6
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "epochs.npz"
            np.savez(dataset, X=data, y=labels, subjects=np.ones(48),
                     runs=np.repeat([4, 8, 12, 6, 10, 14], 8),
                     metadata=json.dumps({"channels": list(CHANNELS), "sfreq": 160.0}))
            for classifier in ["lda", "svm"]:
                with self.subTest(classifier=classifier):
                    model = Path(directory) / f"{classifier}.joblib"
                    report = train(dataset, model, task="multiclass", classifier=classifier)
                    self.assertEqual(report["class_order"], [0, 1, 2, 3])
                    self.assertEqual(report["folds"], 5)
                    self.assertTrue(model.with_suffix(".json").exists())
                    result = predict_window(joblib.load(model), data[0, :, :320])
                    self.assertAlmostEqual(sum(result.values()), 1.0)
                    self.assertTrue(all(0 <= value <= 1 for value in result.values()))


if __name__ == "__main__":
    unittest.main()