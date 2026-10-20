"""Paced replay of offline-filtered EDF recordings, not live headset acquisition."""

import argparse
import json
from pathlib import Path
import time

import joblib

from src.ingestion.preprocessor import filter_raw, load_recording
from src.ipc.zmq_publisher import Publisher, intent_probabilities
from src.models.csp_classifier import predict_window
from src.streaming.ring_buffer import RingBuffer


def windows(samples):
    buffer = RingBuffer()
    for start in range(0, samples.shape[1] - 39, 40):
        buffer.append(samples[:, start:start + 40])
        if buffer.ready:
            yield buffer.window()


def load_decoder(path: Path):
    if path.suffix == ".pt":
        from src.models.trainer import EEGNetPredictor

        predictor = EEGNetPredictor.load(path)
        return predictor.metadata, predictor.report, predictor.probabilities
    if path.suffix != ".joblib":
        raise ValueError("Models must use .joblib for CSP or .pt for EEGNet")
    bundle = joblib.load(path)
    return bundle["metadata"], bundle["report"], lambda window: predict_window(bundle, window)


def replay(model: Path, recording: Path, publish=False, seconds=10.0):
    if seconds <= 0:
        raise ValueError("Replay duration must be positive")
    metadata, report, predict = load_decoder(model)
    if publish and report["task"] != "multiclass":
        raise ValueError("Simulator replay requires a multiclass model with a REST class")
    raw = load_recording(recording)
    samples = filter_raw(raw, metadata.get("notch", 60.0)).get_data()
    publisher = Publisher() if publish else None
    try:
        deadline = time.monotonic()
        count = 0
        for window in windows(samples):
            if count * 0.25 >= seconds:
                break
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            started = time.perf_counter()
            probabilities = predict(window)
            elapsed = (time.perf_counter() - started) * 1000
            packet = publisher.send(probabilities) if publisher else {"probabilities": probabilities}
            print(json.dumps({**packet, "inference_ms": elapsed}), flush=True)
            count += 1
            deadline = max(deadline + 0.25, time.monotonic())
    finally:
        if publisher:
            publisher.send(intent_probabilities("REST"))
            publisher.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/csp_lda.joblib"))
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--seconds", type=float, default=10.0)
    arguments = parser.parse_args()
    try:
        replay(arguments.model, arguments.recording, arguments.publish, arguments.seconds)
    except KeyboardInterrupt:
        pass