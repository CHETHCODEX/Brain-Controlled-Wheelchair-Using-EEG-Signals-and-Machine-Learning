"""Download selected EDF files and verify official SHA-256 checksums."""

import argparse
import hashlib
from pathlib import Path
import sys
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.edf_loader import IMAGERY_RUNS, recording_path

BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0"
SOURCES = {
    "physionet": BASE_URL,
    "aws": "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0",
}


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(subjects, runs, root: Path, source="physionet"):
    if source not in SOURCES:
        raise ValueError("Source must be physionet or aws")
    base_url = SOURCES[source]
    paths = [recording_path(root, subject, run) for subject in subjects for run in runs]
    with urlopen(f"{base_url}/SHA256SUMS.txt", timeout=30) as response:
        lines = response.read().decode("utf-8").splitlines()
    hashes = {}
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            hashes[parts[1].lstrip("*").removeprefix("./")] = parts[0]
    for path in paths:
        relative = path.relative_to(root).as_posix()
        expected = hashes.get(relative)
        if expected is None:
            raise ValueError(f"Official checksum is missing for {relative}")
        if path.exists() and checksum(path) == expected:
            print(f"Verified cached {relative}", flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".edf.part")
        for attempt in range(3):
            try:
                print(f"Downloading {relative} from {source} (attempt {attempt + 1}/3)", flush=True)
                with urlopen(f"{base_url}/{relative}", timeout=30) as response:
                    with partial.open("wb") as target:
                        while chunk := response.read(64 * 1024):
                            target.write(chunk)
                if checksum(partial) != expected:
                    raise ValueError(f"Checksum mismatch for {relative}")
                partial.replace(path)
                print(f"Verified downloaded {relative}", flush=True)
                break
            except (OSError, ValueError):
                partial.unlink(missing_ok=True)
                if attempt == 2:
                    raise
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", type=int, default=[1])
    parser.add_argument("--runs", nargs="+", type=int, default=list(IMAGERY_RUNS))
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--source", choices=list(SOURCES), default="physionet")
    arguments = parser.parse_args()
    download(arguments.subjects, arguments.runs, arguments.root, arguments.source)