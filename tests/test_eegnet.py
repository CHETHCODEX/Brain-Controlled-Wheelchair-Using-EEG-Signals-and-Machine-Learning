import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from src.ingestion.edf_loader import CHANNELS
    from src.models.eegnet import ChannelStandardizer, EEGNet
    from src.models.trainer import EEGNetPredictor, fit_model, train
    from src.streaming.inference_engine import load_decoder


@unittest.skipUnless(TORCH_AVAILABLE, "Install requirements-dl.txt for EEGNet tests")
class EEGNetTests(unittest.TestCase):
    def test_logits_gradients_and_input_contract(self):
        torch.manual_seed(42)
        for classes in (2, 4):
            model = EEGNet(classes)
            inputs = torch.randn(4, 1, 7, 320)
            logits = model(inputs)
            self.assertEqual(tuple(logits.shape), (4, classes))
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(4) % classes)
            loss.backward()
            self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())
            model.eval()
            with torch.inference_mode():
                probabilities = model(inputs).softmax(dim=1)
            torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(4))
            with self.assertRaises(ValueError):
                model(torch.randn(4, 1, 7, 400))

    def test_normalization_uses_only_training_samples(self):
        generator = np.random.default_rng(42)
        training = generator.normal(0, 1e-5, (6, 7, 320)).astype(np.float32)
        evaluation = training + 1e-3
        normalizer = ChannelStandardizer().fit(training)
        mean, scale = normalizer.mean.copy(), normalizer.scale.copy()
        normalized = normalizer.transform(training)
        np.testing.assert_allclose(normalized.mean(axis=(0, 2)), 0, atol=1e-6)
        np.testing.assert_allclose(normalized.std(axis=(0, 2)), 1, atol=1e-6)
        self.assertGreater(float(normalizer.transform(evaluation).mean()), 50)
        np.testing.assert_array_equal(normalizer.mean, mean)
        np.testing.assert_array_equal(normalizer.scale, scale)
        with self.assertRaises(ValueError):
            ChannelStandardizer().transform(training)
        with self.assertRaises(ValueError):
            normalizer.transform(np.full((1, 7, 320), np.nan))

    def test_seeded_training_is_repeatable(self):
        generator = np.random.default_rng(7)
        data = generator.normal(0, 1e-5, (8, 7, 320)).astype(np.float32)
        labels = np.tile([0, 1], 4)
        first, _, first_history = fit_model(data, labels, classes=2, epochs=2, seed=9)
        second, _, second_history = fit_model(data, labels, classes=2, epochs=2, seed=9)
        self.assertEqual(first_history, second_history)
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            fit_model(data, labels, classes=2, epochs=0)

    def test_run_disjoint_training_and_safe_checkpoint(self):
        generator = np.random.default_rng(42)
        data = generator.normal(0, 1e-5, (24, 7, 400)).astype(np.float32)
        labels = np.tile([1, 2], 12)
        runs = np.repeat([4, 8, 12], 8)
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "epochs.npz"
            output = Path(directory) / "binary.pt"
            np.savez(dataset, X=data, y=labels, subjects=np.ones(24), runs=runs,
                     metadata=json.dumps({"channels": list(CHANNELS), "sfreq": 160.0}))
            with patch("src.models.trainer.fit_model", wraps=fit_model) as fit:
                report = train(dataset, output, epochs=1)
            self.assertEqual(report["folds"], 3)
            self.assertEqual(fit.call_count, 4)
            for split, call in zip(report["run_splits"], fit.call_args_list):
                self.assertFalse(set(split["training_runs"]) & set(split["validation_runs"]))
                expected = data[np.isin(runs, split["training_runs"]), :, :320]
                np.testing.assert_array_equal(call.args[0], expected)
            predictor = EEGNetPredictor.load(output)
            values = predictor.probabilities(data[0, :, :320])
            self.assertAlmostEqual(sum(values.values()), 1.0, places=6)
            self.assertEqual(values["REST"], 0)
            self.assertEqual(values["FORWARD"], 0)
            metadata, replay_report, predict = load_decoder(output)
            self.assertEqual(metadata["channels"], list(CHANNELS))
            self.assertEqual(replay_report["task"], "binary")
            self.assertEqual(predict(data[0, :, :320]), values)
            self.assertTrue(output.with_suffix(".json").exists())
            with self.assertRaises(ValueError):
                predictor.probabilities(np.full((7, 320), np.nan))

    def test_multiclass_checkpoint_supports_all_navigation_intents(self):
        generator = np.random.default_rng(7)
        data = generator.normal(0, 1e-5, (24, 7, 400)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "epochs.npz"
            output = Path(directory) / "multiclass.pt"
            np.savez(dataset, X=data, y=np.tile([0, 1, 2, 3], 6),
                     subjects=np.ones(24), runs=np.repeat([4, 8, 12], 8),
                     metadata=json.dumps({"channels": list(CHANNELS), "sfreq": 160.0}))
            report = train(dataset, output, task="multiclass", epochs=1)
            self.assertEqual(report["class_order"], [0, 1, 2, 3])
            _, replay_report, predict = load_decoder(output)
            self.assertEqual(replay_report["task"], "multiclass")
            values = predict(data[0, :, :320])
            self.assertEqual(set(values), {"REST", "TURN_LEFT", "TURN_RIGHT", "FORWARD"})
            self.assertTrue(all(0 < value < 1 for value in values.values()))
            self.assertAlmostEqual(sum(values.values()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()