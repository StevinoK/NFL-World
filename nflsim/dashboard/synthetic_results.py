"""Show the synthetic-opponent report for a saved run, building it the first time."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from nflsim import ui
from nflsim.dashboard.runs import synthetic_runs
from nflsim.dashboard.synthetic_report import build_report

CHARTS = (("Point spread distributions", "synthetic_spread_violins.png"), ("Points for and against", "synthetic_points_grid.png"),
          ("Score heatmaps", "synthetic_scorigami_grid.png"))
TABLE = "synthetic_summary_table.html"


def main() -> None:
    ui.page("Synthetic Report")
    ui.header("Synthetic report", "Every team against the same synthetic opponent, for power rankings.", eyebrow="Game results")
    runs = synthetic_runs()
    if not runs:
        st.info("No synthetic runs yet. Run Synthetic opponents on the Game Simulator page.")
        return
    folder = Path(st.selectbox("Simulation run", [str(path) for path in runs], format_func=lambda path: Path(path).name, help='A saved synthetic-opponent run, newest first. The report is built the first time you open a run.'))
    report = folder / "report"
    if not (report / TABLE).is_file():
        with st.spinner("Building the report for this run..."):
            build_report(folder, report, table_png=False)
    summary = pd.read_csv(report / "synthetic_report_summary.csv")
    st.caption(f"{len(summary)} teams, {int(summary.sims.max()):,} simulations each. Saved in {report}")
    components.html((report / TABLE).read_text(encoding="utf-8"), height=44 * len(summary) + 160, scrolling=True)
    for title, name in CHARTS:
        ui.section(title)
        st.image(str(report / name), width="stretch")


main()
