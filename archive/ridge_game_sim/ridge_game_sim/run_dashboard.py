from __future__ import annotations

import os
import sys
from pathlib import Path


if __name__ == "__main__":
    app = Path(__file__).with_name("dashboard.py")
    os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", str(app)])