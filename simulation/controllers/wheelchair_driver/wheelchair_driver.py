"""Webots entry point; select the project's .venv Python in Webots preferences."""

import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.webots_controller.wheelchair_driver import run


if __name__ == "__main__":
    run(importlib.import_module("controller").Robot())