"""Offline signal-quality statistics and before/after filter plots."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from src.ingestion.preprocessor import filter_raw, load_recording


def channel_statistics(samples):
    if samples.ndim != 2 or samples.shape[1] < 2 or not np.isfinite(samples).all():
        raise ValueError("Expected finite channel-first EEG with at least two samples")
    microvolts = samples * 1e6
    return {
        "rms_uv": np.sqrt(np.mean(microvolts ** 2, axis=1)).tolist(),
        "peak_to_peak_uv": np.ptp(microvolts, axis=1).tolist(),
        "mean_uv": np.mean(microvolts, axis=1).tolist(),
        "flat_channels": np.flatnonzero(np.ptp(samples, axis=1) < 1e-12).tolist(),
    }


def write_report(raw, output: Path, name="recording", notch=60.0):
    before = raw.get_data()
    before_stats = channel_statistics(before)
    cleaned = filter_raw(raw, notch)
    after = cleaned.get_data()
    after_stats = channel_statistics(after)
    sampling_rate = float(raw.info["sfreq"])
    frequency, before_psd = welch(before * 1e6, fs=sampling_rate,
                                  nperseg=min(round(sampling_rate * 2), before.shape[1]), axis=-1)
    _, after_psd = welch(after * 1e6, fs=sampling_rate,
                         nperseg=min(round(sampling_rate * 2), after.shape[1]), axis=-1)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "recording": name, "sfreq": sampling_rate, "channels": list(raw.ch_names),
        "samples": before.shape[1], "duration_seconds": before.shape[1] / sampling_rate,
        "bandpass_hz": [8.0, 30.0], "notch_hz": notch,
        "before": before_stats, "after": after_stats,
        "annotation_counts": {description: int(sum(raw.annotations.description == description))
                              for description in np.unique(raw.annotations.description)},
        "warning": "Descriptive inspection only; no automatic artifact-rejection threshold applied.",
    }
    (output / f"{name}_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    try:
        for axis, spectrum, title in zip(axes, [before_psd, after_psd], ["Before filtering", "After filtering"]):
            for channel, values in zip(raw.ch_names, spectrum):
                axis.semilogy(frequency, np.maximum(values, 1e-12), label=channel)
            axis.axvspan(8, 30, alpha=0.12, color="green")
            axis.set(title=title, ylabel="PSD (uV^2/Hz)", xlim=(0, sampling_rate / 2))
            axis.grid(alpha=0.2)
        axes[0].legend(ncol=4, fontsize=8)
        axes[-1].set_xlabel("Frequency (Hz)")
        figure.savefig(output / f"{name}_psd.png", dpi=140)
    finally:
        plt.close(figure)
    visible = min(before.shape[1], round(10 * sampling_rate))
    times = np.arange(visible) / sampling_rate
    figure, axes = plt.subplots(len(raw.ch_names), 1, figsize=(11, 10), sharex=True,
                                squeeze=False, constrained_layout=True)
    try:
        for index, channel in enumerate(raw.ch_names):
            axis = axes[index, 0]
            axis.plot(times, before[index, :visible] * 1e6, alpha=0.55, linewidth=0.7, label="Raw")
            axis.plot(times, after[index, :visible] * 1e6, linewidth=0.8, label="Filtered")
            axis.set_ylabel(channel + " (uV)")
            axis.grid(alpha=0.2)
        axes[0, 0].legend(loc="upper right")
        axes[-1, 0].set_xlabel("Time (s), first 10 seconds")
        figure.savefig(output / f"{name}_traces.png", dpi=140)
    finally:
        plt.close(figure)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/quality"))
    parser.add_argument("--notch", type=float, choices=[50.0, 60.0], default=60.0)
    arguments = parser.parse_args()
    result = write_report(load_recording(arguments.recording), arguments.output,
                          arguments.recording.stem, arguments.notch)
    print(json.dumps(result, indent=2))