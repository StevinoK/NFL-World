from pathlib import Path

import json
import numpy as np

import pytest

from ridge_game_sim.paths import APP_NAME, ROOT, VOLUME_NAME, check_remote_names, inside
from ridge_game_sim.paths import write_json


def test_manifest_numpy_values_round_trip(tmp_path):
    path = tmp_path / "manifest.json"
    write_json(path, {"classes": [np.float32(-2.5), np.int64(3)], "ready": np.bool_(False),
                      "metrics": {"loss": np.float32(0.25)}, "array": np.array([1, 2])})
    assert json.loads(path.read_text()) == {"classes": [-2.5, 3], "ready": False,
                                         "metrics": {"loss": 0.25}, "array": [1, 2]}


def test_manifest_nonfinite_values_preserve_existing_file(tmp_path):
    path = tmp_path / "manifest.json"
    write_json(path, {"valid": True})
    for value in (np.float32(np.nan), np.float64(np.inf)):
        with pytest.raises(ValueError):
            write_json(path, {"loss": value})
        assert json.loads(path.read_text()) == {"valid": True}


def test_write_boundary():
    assert inside(ROOT / "artifacts" / "runtime").is_absolute()
    with pytest.raises(ValueError):
        inside(ROOT / ".." / "params.pt")
    with pytest.raises(ValueError):
        inside(Path("C:/outside-experiment"))


def test_remote_boundary():
    check_remote_names(APP_NAME, VOLUME_NAME)
    with pytest.raises(ValueError):
        check_remote_names("nfl-game-sim", VOLUME_NAME)