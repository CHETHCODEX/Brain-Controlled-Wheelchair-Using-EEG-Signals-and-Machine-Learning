"""Run-aware labels for PhysioNet EEGMMIDB imagery recordings."""

from pathlib import Path


HAND_RUNS = (4, 8, 12)
BILATERAL_RUNS = (6, 10, 14)
IMAGERY_RUNS = HAND_RUNS + BILATERAL_RUNS
CHANNELS = ("C3", "Cz", "C4", "FC3", "FC4", "CP3", "CP4")
LABELS = {"REST": 0, "TURN_LEFT": 1, "TURN_RIGHT": 2, "FORWARD": 3}


def annotation_label(run: int, annotation: str) -> int:
    if run not in IMAGERY_RUNS:
        raise ValueError(f"Run {run} is not a supported motor-imagery run")
    if annotation == "T0":
        return LABELS["REST"]
    if annotation not in ("T1", "T2"):
        raise ValueError(f"Unknown EEG annotation: {annotation}")
    if run in BILATERAL_RUNS:
        return LABELS["FORWARD"]
    return LABELS["TURN_LEFT" if annotation == "T1" else "TURN_RIGHT"]


def recording_path(root: Path, subject: int, run: int) -> Path:
    if not 1 <= subject <= 109:
        raise ValueError("Subject must be between 1 and 109")
    if run not in IMAGERY_RUNS:
        raise ValueError("Only motor-imagery runs are supported")
    return root / f"S{subject:03d}" / f"S{subject:03d}R{run:02d}.edf"