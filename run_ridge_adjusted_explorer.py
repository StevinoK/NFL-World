"""Launch the weekly ridge ratings explorer with Streamlit."""

import subprocess
import sys
from pathlib import Path


APP_PATH = Path(__file__).resolve().parent / "ridge_adjusted_explorer.py"

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP_PATH), *sys.argv[1:]],
        check=True,
    )