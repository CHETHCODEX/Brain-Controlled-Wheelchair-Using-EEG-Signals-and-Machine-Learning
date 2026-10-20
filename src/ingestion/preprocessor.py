"""HLD offline preprocessing; all EEG amplitudes remain in volts."""

import argparse
import json
from pathlib import Path

import mne
import numpy as np

from .edf_loader import CHANNELS, IMAGERY_RUNS, LABELS, annotation_label, recording_path

SFREQ = 160.0
EPOCH_START = 1.0
EPOCH_SECONDS = 2.5
MODEL_SECONDS = 2.0
FILTER = {"order": 4, "ftype": "butter", "output": "sos"}


def load_recording(path: Path):
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    mne.datasets.eegbci.standardize(raw)
    if not np.isclose(raw.info["sfreq"], SFREQ):
        raise ValueError(f"Expected {SFREQ} Hz, found {raw.info['sfreq']}")
    missing = set(CHANNELS) - set(raw.ch_names)
    if missing:
        raise ValueError(f"Missing sensorimotor channels: {sorted(missing)}")
    raw.pick(list(CHANNELS))
    if not np.isfinite(raw.get_data()).all():
        raise ValueError("Recording contains non-finite EEG samples")
    if np.any(np.ptp(raw.get_data(), axis=1) < 1e-12):
        raise ValueError("Recording contains a flat sensorimotor channel")
    raw.set_montage("standard_1005")
    return raw


def filter_raw(raw, notch: float = 60.0):
    cleaned = raw.copy()
    cleaned.notch_filter(
        [notch], method="iir", iir_params=FILTER, phase="zero", verbose="ERROR"
    )
    cleaned.filter(
        8.0, 30.0, method="iir", iir_params=FILTER, phase="zero", verbose="ERROR"
    )
    return cleaned


def extract_epochs(raw, run: int):
    event_rows = []
    for onset, duration, description in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description
    ):
        if description not in ("T0", "T1", "T2"):
            continue
        label = annotation_label(run, description)
        if duration < EPOCH_START + EPOCH_SECONDS:
            continue
        sample = int(raw.time_as_index(onset - raw.first_time, use_rounding=True)[0])
        event_rows.append([sample + raw.first_samp, 0, label + 1])
    if not event_rows:
        raise ValueError(f"No usable motor-imagery events in run {run}")
    epochs = mne.Epochs(
        raw,
        np.asarray(event_rows, dtype=int),
        event_id={name: label + 1 for name, label in LABELS.items()},
        tmin=EPOCH_START,
        tmax=EPOCH_START + EPOCH_SECONDS - 1.0 / SFREQ,
        baseline=None,
        detrend=None,
        preload=True,
        reject_by_annotation=True,
        on_missing="ignore",
        verbose="ERROR",
    )
    if not len(epochs):
        raise ValueError("All epochs were rejected")
    return epochs


def model_windows(data):
    samples = round(MODEL_SECONDS * SFREQ)
    if data.ndim != 3 or data.shape[1] != len(CHANNELS) or data.shape[2] < samples:
        raise ValueError("Expected epochs shaped (trials, 7 channels, >=320 samples)")
    if not np.isfinite(data).all():
        raise ValueError("Epochs must contain only finite samples")
    return data[:, :, :samples]


def prepare(subjects, runs, root: Path, output: Path, notch: float = 60.0):
    blocks, labels, subject_ids, run_ids = [], [], [], []
    for subject in subjects:
        for run in runs:
            path = recording_path(root, subject, run)
            raw = load_recording(path)
            epochs = extract_epochs(filter_raw(raw, notch), run)
            blocks.append(epochs.get_data(copy=True).astype(np.float32))
            labels.extend(epochs.events[:, 2] - 1)
            subject_ids.extend([subject] * len(epochs))
            run_ids.extend([run] * len(epochs))
            print(f"S{subject:03d} R{run:02d}: {len(epochs)} epochs", flush=True)
    if not blocks:
        raise ValueError("Select at least one subject and run")
    data = np.concatenate(blocks)
    metadata = {
        "dataset": "eegmmidb/1.0.0", "sfreq": SFREQ, "units": "V",
        "channels": list(CHANNELS), "bandpass": [8.0, 30.0],
        "filter_order": 4, "notch": notch, "phase": "zero",
        "epoch_start": EPOCH_START, "epoch_seconds": EPOCH_SECONDS,
        "model_seconds": MODEL_SECONDS, "labels": LABELS,
        "warning": "Offline zero-phase processing is not causal live-headset processing",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, X=data, y=np.asarray(labels), subjects=np.asarray(subject_ids),
        runs=np.asarray(run_ids), metadata=json.dumps(metadata),
    )
    print(f"Saved {data.shape} to {output}", flush=True)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", type=int, default=[1])
    parser.add_argument("--runs", nargs="+", type=int, default=list(IMAGERY_RUNS))
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/epochs.npz"))
    parser.add_argument("--notch", type=float, choices=[50.0, 60.0], default=60.0)
    arguments = parser.parse_args()
    prepare(arguments.subjects, arguments.runs, arguments.root, arguments.output, arguments.notch)