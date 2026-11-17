from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
APP_NAME = "nfl-ridge-game-sim"
VOLUME_NAME = "nfl-ridge-game-sim-artifacts"


def inside(path: Path, *, root: Path = ROOT) -> Path:
    resolved = Path(path).resolve()
    boundary = root.resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise ValueError(f"Write outside experiment rejected: {resolved}")
    return resolved


def output_dir(path: Path) -> Path:
    path = inside(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, value: object) -> None:
    path = inside(path)
    output_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=json_default), encoding="utf-8")
    os.replace(temporary, path)


def protected_hashes() -> dict[str, str]:
    candidates = [REPO / name for name in (
        "model_feature_config.json", "bayesian_decay_rates.json", "Torch Modeling.py",
        "backtest_2025_results.csv", "backtest_2025_summary.json",
    )]
    candidates.extend((REPO / "artifacts" / "runtime").glob("*"))
    return {str(path.relative_to(REPO)): digest(path) for path in candidates if path.is_file()}


def assert_protected(before: dict[str, str]) -> None:
    changed = [name for name, checksum in before.items() if not (REPO / name).exists() or digest(REPO / name) != checksum]
    if changed:
        raise RuntimeError(f"Protected files changed: {changed}")


def check_remote_names(app: str, volume: str) -> None:
    if app != APP_NAME or volume != VOLUME_NAME:
        raise ValueError("Only isolated ridge Modal app/volume names are permitted")