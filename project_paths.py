"""Repository-relative locations shared by runnable applications."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = REPOSITORY_DIR / "artifacts" / "runtime"
DEFAULT_SIMULATION_OUTPUT_DIR = REPOSITORY_DIR / "outputs" / "simulations"


def artifact_dir() -> Path:
    """Return the configured runtime-artifact directory."""
    configured = os.environ.get("FOOTBALL_ARTIFACT_DIR")
    return Path(configured).expanduser() if configured else DEFAULT_ARTIFACT_DIR