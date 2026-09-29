"""Launch the DFS Showdown optimizer dashboard with Streamlit."""

import subprocess
import sys
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "dfs_dashboard.py"

if __name__ == "__main__":
    sys.exit(subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP_PATH)], check=False).returncode)
