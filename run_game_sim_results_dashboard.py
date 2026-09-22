"""Launch the weekly game simulation results dashboard with Streamlit."""

import os
import sys
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "game_sim_results_dashboard.py"

if __name__ == "__main__":
    os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", str(APP_PATH)])
