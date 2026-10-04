"""Interactive weekly ridge coefficients, fitted strictly before each league week."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


if __name__ == "__main__" and not st.runtime.exists():
    import subprocess

    raise SystemExit(subprocess.call([
        sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]
    ]))


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "ridge_adjusted_weekly_bakeoff"
BLOCK_NAMES = {
    "subject": "Player / offensive team",
    "defense": "Defensive team",
    "offense_team": "Offensive team context",
    "teammate": "Teammate",
    "venue": "Venue",
    "position": "Position",
    "home": "Home advantage",
    "rest": "Rest differential",
    "intercept": "League baseline",
}


@st.cache_data(show_spinner="Loading coefficient history...")
def load_data(directory: str) -> tuple[pd.DataFrame, dict[str, str]]:
    path = Path(directory)
    history = pd.read_parquet(path / "weekly_coefficients.parquet")
    names_file = path / "player_names.csv"
    names = pd.read_csv(names_file) if names_file.exists() else pd.DataFrame(columns=["label", "name"])
    return history, dict(zip(names["label"].astype(str), names["name"].astype(str)))


def main() -> None:
    st.set_page_config(page_title="Adjusted Skill Ratings", layout="wide")
    st.markdown("""
        <style>
        .block-container { max-width: 1180px; padding-top: 2rem; }
        h1, h2, h3 { letter-spacing: 0; }
        [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
        </style>
    """, unsafe_allow_html=True)
    st.title("Adjusted skill ratings")
    st.caption("Pregame ridge estimates · weighted by recency and exposure")

    directory = sys.argv[sys.argv.index("--output-dir") + 1] if "--output-dir" in sys.argv else str(DEFAULT_DATA_DIR)
    try:
        data, names = load_data(directory)
    except (FileNotFoundError, OSError) as error:
        st.error(f"No weekly coefficient history found in {directory}: {error}")
        st.stop()
    if data.empty:
        st.info("No weekly coefficients available yet.")
        st.stop()

    with st.sidebar:
        st.header("Explore")
        metric = st.selectbox("Metric", sorted(data["metric"].unique()), index=0)
        metric_rows = data[data["metric"].eq(metric)]
        blocks = [block for block in BLOCK_NAMES if block in set(metric_rows["block"])]
        default_block = "defense" if metric_rows["side"].iloc[0] == "defense" else "subject"
        if st.session_state.get("ridge_block") not in blocks:
            st.session_state["ridge_block"] = default_block
        block = st.selectbox("Estimate type", blocks, key="ridge_block", format_func=BLOCK_NAMES.get)
        group = metric_rows[metric_rows["block"].eq(block)]
        options = sorted(group["label"].astype(str).unique(), key=lambda label: (names.get(label, label), label))

        def display_label(label: str) -> str:
            return f"{names[label]} ({label})" if block in {"subject", "teammate"} and label in names else label

        latest_season = group["season"].max()
        recent = group[group["season"].eq(latest_season)].sort_values("weighted_exposure", ascending=False)
        initial = str(recent.iloc[0]["label"])
        if st.session_state.get("ridge_entity") not in options:
            st.session_state["ridge_entity"] = initial
        label = st.selectbox("Player, team or venue", options, key="ridge_entity", format_func=display_label)
        entity = group[group["label"].eq(label)].sort_values(["season", "week"])
        seasons = sorted(entity["season"].unique())
        selected_seasons = st.multiselect("Season", seasons, default=seasons)
        weeks = st.slider("Weeks", 1, 18, (1, 18))

    visible = entity[entity["season"].isin(selected_seasons) & entity["week"].between(*weeks)].copy()
    if visible.empty:
        st.info("No observed pregame estimates match these filters.")
        return
    visible["week_label"] = visible["season"].astype(str) + " W" + visible["week"].astype(str).str.zfill(2)
    is_entity = block in {"subject", "defense", "offense_team", "teammate"}
    plotted = "estimate" if is_entity else "coefficient"
    title = display_label(label)
    st.subheader(f"{metric} · {title}")
    latest = visible.iloc[-1]
    left, center, right = st.columns(3)
    left.metric("Latest estimate" if is_entity else "Latest effect", f"{latest[plotted]:.3f}")
    center.metric("Latest week", f"{int(latest['season'])} W{int(latest['week']):02d}")
    right.metric("Weighted exposure", f"{latest['weighted_exposure']:,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=visible["week_label"], y=visible[plotted], mode="lines+markers",
        line={"color": "#126a70", "width": 2.5}, marker={"size": 5, "color": "#d07045"},
        customdata=visible[["weighted_exposure", "coefficient"]].to_numpy(),
        hovertemplate="%{x}<br>Estimate: %{y:.3f}<br>Coefficient: %{customdata[1]:.3f}"
                      "<br>Weighted exposure: %{customdata[0]:,.1f}<extra></extra>",
    ))
    if not is_entity:
        fig.add_hline(y=0, line_color="#adb6b3", line_dash="dot")
    fig.update_layout(
        height=480, margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="League week", yaxis_title="Adjusted estimate" if is_entity else "Marginal effect",
        xaxis=dict(type="category", tickangle=-45, nticks=24, showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(100,120,115,0.15)", zeroline=False),
        hovermode="x unified", showlegend=False,
    )
    st.plotly_chart(fig)
    st.caption(
        "Each point uses only games before that week. Player estimates include the league baseline, position, "
        "and player effect; team estimates include the baseline and team effect. Venue and other context values "
        "are marginal effects. Weeks without prior weighted observations are omitted."
    )
    st.dataframe(
        visible[["season", "week", "estimate", "coefficient", "weighted_exposure"]].sort_values(
            ["season", "week"], ascending=False
        ),
        hide_index=True, width="stretch",
    )
    st.download_button(
        "Download selected history", visible.to_csv(index=False).encode("utf-8"),
        file_name=f"{metric}_{block}_{label}_weekly.csv", mime="text/csv",
    )


if __name__ == "__main__":
    main()