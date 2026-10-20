"""Fixed-epoch CPU EEGNet training with recording-disjoint evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.ingestion.edf_loader import CHANNELS, LABELS
from src.ingestion.preprocessor import model_windows
from src.models.eegnet import ChannelStandardizer, EEGNet, validate_windows


def fit_model(data, labels, classes, epochs=50, batch_size=16, learning_rate=0.001, seed=42):
    data = validate_windows(data)
    labels = np.asarray(labels)
    if labels.shape != (len(data),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("One integer class index is required per EEG window")
    if sorted(np.unique(labels).tolist()) != list(range(classes)):
        raise ValueError("Every model class must occur in training data")
    if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0 or seed < 0:
        raise ValueError("Epochs, batch size, and learning rate must be positive; seed must be non-negative")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    normalizer = ChannelStandardizer().fit(data)
    samples = torch.from_numpy(normalizer.transform(data)).unsqueeze(1)
    targets = torch.from_numpy(labels.astype(np.int64))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(samples, targets), batch_size=batch_size,
                        shuffle=True, generator=generator, num_workers=0)
    model = EEGNet(classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        history.append(total_loss / len(data))
    model.eval()
    return model, normalizer, history


def predict_probabilities(model, normalizer, data):
    normalized = normalizer.transform(data)
    model.eval()
    with torch.inference_mode():
        probabilities = model(torch.from_numpy(normalized).unsqueeze(1)).softmax(dim=1).numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced non-finite probabilities")
    return probabilities


class EEGNetPredictor:
    def __init__(self, model, normalizer, class_order, metadata, report):
        self.model = model
        self.normalizer = normalizer
        self.class_order = class_order
        self.metadata = metadata
        self.report = report

    @classmethod
    def load(cls, path):
        torch.set_num_threads(1)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "eegnet-v1":
            raise ValueError("Unsupported EEGNet checkpoint format")
        class_order = checkpoint["class_order"]
        if class_order not in ([1, 2], [0, 1, 2, 3]):
            raise ValueError("Invalid checkpoint class order")
        metadata = json.loads(checkpoint["metadata_json"])
        if metadata["channels"] != list(CHANNELS) or metadata["sfreq"] != 160.0:
            raise ValueError("Checkpoint channels or sampling rate do not match the EEG interface")
        model = EEGNet(len(class_order))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        normalizer = ChannelStandardizer()
        normalizer.mean = checkpoint["normalization_mean"].numpy().copy()
        normalizer.scale = checkpoint["normalization_scale"].numpy().copy()
        if (normalizer.mean.shape != (1, len(CHANNELS), 1)
                or normalizer.scale.shape != (1, len(CHANNELS), 1)
                or not np.isfinite(normalizer.mean).all()
                or not np.isfinite(normalizer.scale).all()
                or np.any(normalizer.scale <= 0)):
            raise ValueError("Invalid saved normalization statistics")
        return cls(model, normalizer, class_order, metadata, json.loads(checkpoint["report_json"]))

    def probabilities(self, window):
        window = np.asarray(window, dtype=np.float32)
        if window.shape != (len(CHANNELS), 320):
            raise ValueError("Expected a finite (7, 320) EEG window in volts")
        values = predict_probabilities(self.model, self.normalizer, window[np.newaxis])[0]
        names = {label: name for name, label in LABELS.items()}
        result = {name: 0.0 for name in LABELS}
        for label, value in zip(self.class_order, values):
            result[names[label]] = float(value)
        return result


def train(dataset: Path, output: Path, task="binary", subject=1,
          epochs=50, batch_size=16, learning_rate=0.001, seed=42):
    if task not in ("binary", "multiclass"):
        raise ValueError("Task must be binary or multiclass")
    if output.suffix != ".pt":
        raise ValueError("EEGNet checkpoints must use the .pt extension")
    with np.load(dataset, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata["channels"] != list(CHANNELS) or metadata["sfreq"] != 160.0:
            raise ValueError("Dataset channel order or sampling rate differs from the HLD")
        data = validate_windows(model_windows(archive["X"]))
        labels = archive["y"].copy()
        subjects, runs = archive["subjects"].copy(), archive["runs"].copy()
    selected = subjects == subject
    expected = [1, 2] if task == "binary" else [0, 1, 2, 3]
    if task == "binary":
        selected &= np.isin(labels, expected)
    data, labels, groups = data[selected], labels[selected], runs[selected]
    if sorted(np.unique(labels).tolist()) != expected:
        raise ValueError(f"Subject {subject} must contain all task labels: {expected}")
    encoded = np.searchsorted(expected, labels).astype(np.int64)
    folds = min(5, len(np.unique(groups)))
    if folds < 2:
        raise ValueError("At least two distinct runs are required for held-out evaluation")
    predictions = np.empty_like(labels)
    scores, run_splits = [], []
    for fold, (training, validation) in enumerate(GroupKFold(n_splits=folds).split(data, labels, groups)):
        model, normalizer, history = fit_model(data[training], encoded[training], len(expected),
                                               epochs, batch_size, learning_rate, seed + fold)
        values = predict_probabilities(model, normalizer, data[validation])
        predictions[validation] = np.asarray(expected)[values.argmax(axis=1)]
        accuracy = float(accuracy_score(labels[validation], predictions[validation]))
        scores.append(accuracy)
        run_splits.append({
            "training_runs": np.unique(groups[training]).tolist(),
            "validation_runs": np.unique(groups[validation]).tolist(),
            "validation_trials": len(validation), "seed": seed + fold,
            "final_training_loss": history[-1],
        })
        print(f"EEGNet fold {fold + 1}/{folds}: accuracy={accuracy:.4f}", flush=True)
    model, normalizer, history = fit_model(data, encoded, len(expected), epochs, batch_size, learning_rate, seed)
    counts = [int(np.sum(labels == label)) for label in expected]
    report = {
        "task": task, "classifier": "eegnet", "subject": subject,
        "evaluation": "recording-disjoint GroupKFold", "folds": folds,
        "fold_accuracy": scores, "run_splits": run_splits,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=expected).tolist(),
        "class_order": expected, "class_counts": counts, "trials": len(labels),
        "majority_class_accuracy": max(counts) / len(labels),
        "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "seed": seed, "device": "cpu", "cpu_threads": 1, "torch_version": str(torch.__version__),
        "normalization": "Per-channel mean/std fitted on training-fold samples only",
        "loss": "Unweighted cross entropy", "gradient_clip_norm": 1.0,
        "training_loss_history": history,
        "note": "Fixed epochs; no early stopping or hyperparameter selection on held-out runs. Final model refitted on all task trials.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "eegnet-v1", "state_dict": model.state_dict(), "class_order": expected,
        "normalization_mean": torch.from_numpy(normalizer.mean),
        "normalization_scale": torch.from_numpy(normalizer.scale),
        "metadata_json": json.dumps(metadata), "report_json": json.dumps(report),
    }, output)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/epochs.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/eegnet.pt"))
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    print(json.dumps(train(arguments.dataset, arguments.output, arguments.task, arguments.subject,
                           arguments.epochs, arguments.batch_size, arguments.learning_rate,
                           arguments.seed), indent=2))