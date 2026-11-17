from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from ridge_game_sim.paths import ROOT


@pytest.fixture
def tmp_path():
    directory = ROOT / "tests" / "_tmp"
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="pytest_", dir=directory) as temporary:
        yield Path(temporary)