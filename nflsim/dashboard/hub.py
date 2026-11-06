"""One app for every dashboard, with tabs across the top.

    streamlit run nflsim/dashboard/hub.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import streamlit as st

DASHBOARD = REPO / "nflsim/dashboard"
PAGES = {
    "Simulate": [
        st.Page(DASHBOARD / "app.py", title="Game Simulator", url_path="simulator"),
        st.Page(DASHBOARD / "season_experiments.py", title="Season QB Experiments", url_path="season"),
    ],
    "Results": [
        st.Page(DASHBOARD / "weekly_results.py", title="Weekly Matchups", url_path="matchups", default=True),
        st.Page(DASHBOARD / "synthetic_results.py", title="Synthetic Report", url_path="synthetic"),
    ],
    "Tools": [
        st.Page(REPO / "nflsim/dfs/dashboard.py", title="DFS Optimizer", url_path="dfs"),
        st.Page(REPO / "nflsim/ridge/explorer.py", title="Ridge Graph Builder", url_path="ridge"),
        st.Page(REPO / "player_value/explorer.py", title="Player Estimates", url_path="players"),
    ],
}

st.navigation(PAGES, position="top").run()
