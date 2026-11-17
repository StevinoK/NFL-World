"""Show the league ratings report for a saved run, building it the first time."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from nflsim import ui
from nflsim.dashboard.runs import synthetic_runs
from nflsim.dashboard.power_ratings import RATINGS_FILE
from nflsim.dashboard.synthetic_report import build_report, rebuild_tables, refresh_ratings

CHARTS = (("Point spread distributions", "synthetic_spread_violins.png"), ("Points for and against", "synthetic_points_grid.png"),
          ("Score heatmaps", "synthetic_scorigami_grid.png"))
TABLE = "synthetic_summary_table.html"


def main() -> None:
    ui.page("League Ratings")
    ui.header("League ratings", "Every team against every team, itself included, for power rankings.", eyebrow="Game results")
    runs = synthetic_runs()
    if not runs:
        st.info("No league runs yet. Run League ratings on the Game Simulator page.")
        return
    folder = Path(st.selectbox("Simulation run", [str(path) for path in runs], format_func=lambda path: Path(path).name, help='A saved league run, newest first. The report is built the first time you open a run.'))
    report = folder / "report"
    if not (report / TABLE).is_file():
        with st.spinner("Building the report for this run..."):
            build_report(folder, report, table_png=False)
    elif not (report / RATINGS_FILE).is_file() or st.button("Refresh outside ratings", help="Pull the latest outside ratings into this table. Saved ratings stay as pulled until you do."):
        with st.spinner("Pulling outside ratings..."):
            refresh_ratings(folder, report)
    elif "nfl-table" not in (report / TABLE).read_text(encoding="utf-8"):
        rebuild_tables(folder, report)
    summary = pd.read_csv(report / "synthetic_report_summary.csv")
    st.caption(f"{len(summary)} teams, {int(summary.sims.max()):,} simulations each. Saved in {report}")
    components.html(ui.html_with_image_actions((report / TABLE).read_text(encoding="utf-8"), f"{folder.name} ratings table"),
                    height=35 * len(summary) + 230, scrolling=True)
    for title, name in CHARTS:
        ui.section(title)
        st.image(str(report / name), width="stretch")
        ui.image_actions((report / name).read_bytes(), f"{folder.name} {title}")


main()
