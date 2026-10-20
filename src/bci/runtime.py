"""Atomic local state shared by the dashboard and Webots controller."""

import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "artifacts" / "runtime"
TELEMETRY = RUNTIME / "wheelchair.json"
EMERGENCY = RUNTIME / "emergency.json"


def write_state(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_state(path):
    return json.loads(path.read_text(encoding="utf-8"))