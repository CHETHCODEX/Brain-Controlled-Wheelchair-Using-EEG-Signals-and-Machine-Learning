"""CSP baselines with recording-disjoint validation."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from mne.decoding import CSP
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.ingestion.edf_loader import CHANNELS, LABELS
from src.ingestion.preprocessor import model_windows


def make_pipeline(classifier="lda"):
    if classifier == "lda":
        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif classifier == "svm":
        estimator = CalibratedClassifierCV(SVC(C=1.0, kernel="rbf"), cv=3, ensemble=False)
    else:
        raise ValueError("Classifier must be lda or svm")
    return Pipeline([
        ("csp", CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False)),
        ("classifier", estimator),
    ])


def train(dataset: Path, output: Path, task="binary", classifier="lda", subject=1):
    if task not in ("binary", "multiclass"):
        raise ValueError("Task must be binary or multiclass")
    with np.load(dataset, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata["channels"] != list(CHANNELS) or metadata["sfreq"] != 160.0:
            raise ValueError("Dataset channel order or sampling rate differs from the HLD")
        data = model_windows(archive["X"]).astype(np.float64)
        labels = archive["y"].copy()
        subjects, runs = archive["subjects"].copy(), archive["runs"].copy()
    selected = subjects == subject
    if task == "binary":
        selected &= np.isin(labels, [LABELS["TURN_LEFT"], LABELS["TURN_RIGHT"]])
    data, labels, groups = data[selected], labels[selected], runs[selected]
    expected = [1, 2] if task == "binary" else [0, 1, 2, 3]
    if sorted(np.unique(labels).tolist()) != expected:
        raise ValueError(f"Subject {subject} must contain all task labels: {expected}")
    folds = min(5, len(np.unique(groups)))
    if folds < 2:
        raise ValueError("At least two distinct runs are required for held-out evaluation")
    predictions = np.empty_like(labels)
    scores = []
    run_splits = []
    for training, validation in GroupKFold(n_splits=folds).split(data, labels, groups):
        if sorted(np.unique(labels[training]).tolist()) != expected:
            raise ValueError("A training fold lacks a task class; provide more recording runs")
        pipeline = make_pipeline(classifier)
        pipeline.fit(data[training], labels[training])
        predictions[validation] = pipeline.predict(data[validation])
        scores.append(float(accuracy_score(labels[validation], predictions[validation])))
        run_splits.append({
            "training_runs": np.unique(groups[training]).tolist(),
            "validation_runs": np.unique(groups[validation]).tolist(),
            "validation_trials": len(validation),
        })
    class_counts = [int(np.sum(labels == label)) for label in expected]
    report = {
        "task": task, "classifier": classifier, "subject": subject,
        "evaluation": "recording-disjoint GroupKFold", "folds": folds,
        "fold_accuracy": scores,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=expected).tolist(),
        "class_order": expected, "trials": len(labels),
        "class_counts": class_counts, "majority_class_accuracy": max(class_counts) / len(labels),
        "run_splits": run_splits,
        "note": ("Three hand-imagery runs permit three disjoint folds, not five."
             if task == "binary" else "Five folds hold out entire runs across six imagery recordings."),
    }
    pipeline = make_pipeline(classifier)
    pipeline.fit(data, labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata, "report": report}, output)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def predict_window(bundle, window):
    if window.shape != (len(CHANNELS), 320) or not np.isfinite(window).all():
        raise ValueError("Expected a finite (7, 320) EEG window in volts")
    probabilities = bundle["pipeline"].predict_proba(window[np.newaxis].astype(np.float64))[0]
    names = {value: name for name, value in LABELS.items()}
    result = {name: 0.0 for name in LABELS}
    for label, probability in zip(bundle["pipeline"].classes_, probabilities):
        result[names[int(label)]] = float(probability)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/epochs.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/csp_lda.joblib"))
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--classifier", choices=["lda", "svm"], default="lda")
    parser.add_argument("--subject", type=int, default=1)
    arguments = parser.parse_args()
    print(json.dumps(train(arguments.dataset, arguments.output, arguments.task,
                           arguments.classifier, arguments.subject), indent=2))