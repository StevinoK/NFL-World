"""Isolated ridge simulation research pipeline."""

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "artifacts/cache/matplotlib"))