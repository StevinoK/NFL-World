from __future__ import annotations

import os
import sys
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "feature_dashboard.py"

if __name__ == "__main__":
    os.execv(
        sys.executable,
        [sys.executable, "-m", "streamlit", "run", str(APP_PATH)],
    )
