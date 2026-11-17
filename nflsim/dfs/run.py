"""Launch the DFS Showdown optimizer dashboard with Streamlit."""

import os
import subprocess
import sys
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "dashboard.py"
REPO = APP_PATH.parents[2]

if __name__ == "__main__":
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))}
    sys.exit(subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP_PATH)], env=environment, cwd=str(REPO),
                            check=False).returncode)
