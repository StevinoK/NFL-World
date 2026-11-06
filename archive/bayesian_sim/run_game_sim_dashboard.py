"""Launch the weekly game simulation dashboard with Streamlit."""

import subprocess
import sys
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "game_sim_dashboard.py"

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP_PATH)], check=False)
