"""Explore a saved weekly simulation run: comparison with the market, then each game in detail.

Reads a run folder's games.csv and players.csv and renders, per selected game, a
LightSource-shaded discrete scorigami grid (matplotlib) plus player stat
histograms and binned heatmaps with explicit player+stat+range filters (no graph zoom/brush).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import nfl_data_py as nfl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from matplotlib.colors import BoundaryNorm, LightSource, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, PercentFormatter
from scipy.ndimage import gaussian_filter

from nflsim import ui
from nflsim.dashboard.runs import weekly_runs
from nflsim.sim.market_lines import market_lines

ID_COLUMNS = {"game_id", "week", "simulation", "team", "player_name", "player_id", "position"}
ui.style_matplotlib()
LEFT_COLOR = ui.SERIES[0]
RIGHT_COLOR = ui.SERIES[1]
# team_stats_wide uses LA/LV; nflverse team_desc historically carries both spellings.
TEAM_DESC_ALIASES = {"LA": "LAR", "LAR": "LA", "LV": "OAK", "OAK": "LV"}
# Normalized 0..1 relief over 1-point grid cells needs exaggeration for visible hillshading.
RELIEF_EXAGGERATION = 15.0
# Bare soil at zero sims up to lush meadow at the peak. Stops are explicitly positioned so
# green takes over early (~38%); no near-white stops, since the overlay light blend bleaches
# pale midtones into a white ring.
SCORIGAMI_CMAP = LinearSegmentedColormap.from_list(
    "deep_soil_to_meadow",
    [
        (0.00, "#8a6a3c"),
        (0.15, "#9c7f45"),
        (0.28, "#948b40"),
        (0.38, "#74933a"),
        (0.55, "#539030"),
        (0.72, "#378026"),
        (0.87, "#20701f"),
        (1.00, "#0f5a1a"),
    ],
)


@st.cache_data(show_spinner=False)
def _team_color_lookup() -> Dict[str, str]:
    try:
        desc = nfl.import_team_desc()[["team_abbr", "team_color"]]
        return dict(zip(desc["team_abbr"], desc["team_color"]))
    except Exception:
        return {}


def team_color(team: str, fallback: str) -> str:
    colors = _team_color_lookup()
    return colors.get(team) or colors.get(TEAM_DESC_ALIASES.get(team, "")) or fallback


@st.cache_data(show_spinner=False)
def _team_logo_lookup() -> Dict[str, str]:
    try:
        desc = nfl.import_team_desc()[["team_abbr", "team_logo_espn"]]
        return dict(zip(desc["team_abbr"], desc["team_logo_espn"]))
    except Exception:
        return {}


def team_logo(team: str) -> Optional[str]:
    logos = _team_logo_lookup()
    return logos.get(team) or logos.get(TEAM_DESC_ALIASES.get(team, ""))


# ------------------------------------------------------------------ data loading


@st.cache_resource(show_spinner="Loading simulation results...", max_entries=3)
def load_week_results(folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target_dir = Path(folder)
    return pd.read_csv(target_dir / "games.csv"), pd.read_csv(target_dir / "players.csv")


@st.cache_data(show_spinner=False)
def load_market_lines(seasons: Tuple[int, ...]) -> pd.DataFrame:
    schedule = nfl.import_schedules(list(seasons))
    columns = ["game_id", "spread_line", "total_line", "gameday", "gametime"]
    return schedule[[column for column in columns if column in schedule.columns]].drop_duplicates("game_id")


@st.cache_data(show_spinner="Reading Kalshi and Polymarket...", ttl=300)
def load_prediction_markets(games: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], str]:
    lines, errors = market_lines(games)
    return lines, errors, pd.Timestamp.now(tz="America/New_York").strftime("%b %d, %I:%M %p ET")


def attach_market_lines(games_df: pd.DataFrame) -> pd.DataFrame:
    games = games_df.copy()
    if "spread_line" not in games.columns:
        games["spread_line"] = np.nan
    if "total_line" not in games.columns:
        games["total_line"] = np.nan

    seasons = games["game_id"].astype(str).str.extract(r"^(\d{4})_", expand=False)
    seasons = tuple(sorted(seasons.dropna().astype(int).unique()))
    if not seasons:
        return games

    market_lines = load_market_lines(seasons)
    if market_lines.empty:
        return games
    games = games.merge(market_lines, on="game_id", how="left", suffixes=("", "_schedule"))
    for column in ("spread_line", "total_line"):
        schedule_column = f"{column}_schedule"
        if schedule_column in games.columns:
            games[column] = pd.to_numeric(games[column], errors="coerce").fillna(
                pd.to_numeric(games[schedule_column], errors="coerce")
            )
            games = games.drop(columns=schedule_column)
    return games


def build_market_summary(games_df: pd.DataFrame) -> pd.DataFrame:
    games = games_df.copy()
    games["predicted_spread"] = pd.to_numeric(games["home_score"], errors="coerce") - pd.to_numeric(
        games["away_score"], errors="coerce"
    )
    games["predicted_total"] = pd.to_numeric(games["home_score"], errors="coerce") + pd.to_numeric(
        games["away_score"], errors="coerce"
    )
    return (
        games.groupby("game_id", as_index=False, sort=False)
        .agg(
            week=("week", "first"),
            away_team=("away_team", "first"),
            home_team=("home_team", "first"),
            simulations=("simulation", "nunique"),
            predicted_spread=("predicted_spread", "median"),
            vegas_spread=("spread_line", "first"),
            predicted_total=("predicted_total", "median"),
            vegas_total=("total_line", "first"),
        )
        .assign(
            spread_difference=lambda frame: frame["predicted_spread"] - frame["vegas_spread"],
            total_difference=lambda frame: frame["predicted_total"] - frame["vegas_total"],
        )
        .merge(
            games.assign(home_win=(games["predicted_spread"] > 0) + 0.5 * (games["predicted_spread"] == 0))
            .groupby("game_id", as_index=False)["home_win"].mean(),
            on="game_id",
        )
        .sort_values(["week", "away_team", "home_team"], kind="stable")
        .reset_index(drop=True)
    )


def _market_comparison_chart(
    summary: pd.DataFrame, market_column: str, prediction_column: str, title: str, value_format: str = ".1f"
) -> go.Figure:
    points = summary[[market_column, prediction_column, "away_team", "home_team"]].dropna()
    fig = go.Figure()
    if not points.empty:
        low = float(min(points[market_column].min(), points[prediction_column].min()))
        high = float(max(points[market_column].max(), points[prediction_column].max()))
        bounds = [low - 2, high + 2]
        gaps = points[prediction_column] - points[market_column]
        widest = gaps.abs().nlargest(3).index
        names = points["away_team"] + " @ " + points["home_team"]
        fig.add_trace(go.Scatter(x=bounds, y=bounds, mode="lines", name="Agreement", hoverinfo="skip", line=dict(color=ui.MUTED, dash="dash", width=1)))
        fig.add_trace(
            go.Scatter(
                x=points[market_column],
                y=points[prediction_column],
                mode="markers+text",
                name="Games",
                text=names.where(points.index.isin(widest), ""),
                textposition=["top center" if gap > 0 else "middle left" for gap in gaps],
                textfont=dict(size=11, color=ui.MUTED),
                customdata=names,
                hovertemplate=f"%{{customdata}}<br>Vegas: %{{x:{value_format}}}<br>Ours: %{{y:{value_format}}}<extra></extra>",
                marker=dict(color=ui.ACCENT, size=9, line=dict(color="white", width=2)),
            )
        )
        fig.update_xaxes(range=bounds)
        fig.update_yaxes(range=bounds)
    fig.update_layout(
        title=title,
        xaxis_title="Vegas",
        yaxis_title="Our simulations",
        height=390,
        showlegend=False,
        margin=dict(t=55, r=20, b=55, l=55),
    )
    return ui.style_plotly(fig)


def render_market_summary(games_df: pd.DataFrame) -> None:
    summary = build_market_summary(games_df)
    market_columns = ["kalshi_spread", "kalshi_home_win", "polymarket_spread", "polymarket_home_win"]
    schedule = games_df[["game_id", "gameday", "gametime"]].drop_duplicates("game_id") if {"gameday", "gametime"} <= set(games_df.columns) else None
    markets_note = "Prediction market prices need schedule kickoff times, which aren't available."
    if schedule is not None:
        try:
            lines, errors, read_at = load_prediction_markets(summary[["game_id", "home_team", "away_team"]].merge(schedule, on="game_id"))
            summary = summary.merge(lines, on="game_id", how="left")
            markets_note = (
                f"Kalshi and Polymarket read {read_at}: live before kickoff, the last price before kickoff after that. "
                "Their spread is the home margin their ladder prices at 50%. Blank means no market or too thin a market."
            )
            if errors:
                st.warning(f"Some prediction market prices couldn't be read: {', '.join(errors)}")
        except Exception as error:
            st.warning(f"Could not read prediction markets: {error}")
    for column in market_columns:
        if column not in summary.columns:
            summary[column] = np.nan
    available = summary.dropna(subset=["vegas_spread", "vegas_total"])
    if available.empty:
        st.info("Vegas spread and total lines are not available for these games in the schedule data.")
        return

    metric_columns = st.columns(3)
    metric_columns[0].metric("Games with Vegas lines", f"{len(available)} / {len(summary)}")
    metric_columns[1].metric(
        "Average spread gap",
        f"{available['spread_difference'].abs().mean():.1f} pts",
        help="Mean absolute difference between predicted and Vegas home margin.",
    )
    metric_columns[2].metric(
        "Average total gap",
        f"{available['total_difference'].abs().mean():.1f} pts",
        help="Mean absolute difference between predicted and Vegas total.",
    )

    st.caption(
        f"{int(available['simulations'].max()):,} simulations per game. Spread is home score minus away score, so positive favors the home team. "
        "Win columns are the home team's, and Poly is Polymarket. Blue favors the home team and orange the away team; totals are blue above the average Vegas total and orange below it. "
        f"Gaps are ours minus Vegas: blue means we are higher than Vegas, orange lower. {markets_note}"
    )
    display = available.assign(
        away_logo=available["away_team"].map(team_logo),
        home_logo=available["home_team"].map(team_logo),
        Game=available["away_team"] + " @ " + available["home_team"],
    ).rename(columns={
        "week": "Week",
        "predicted_spread": "Our spread",
        "vegas_spread": "Vegas spread",
        "kalshi_spread": "Kalshi spread",
        "polymarket_spread": "Poly spread",
        "spread_difference": "Spread gap",
        "home_win": "Our win",
        "kalshi_home_win": "Kalshi win",
        "polymarket_home_win": "Poly win",
        "predicted_total": "Our total",
        "vegas_total": "Vegas total",
        "total_difference": "Total gap",
    })
    spreads = ["Our spread", "Vegas spread", "Kalshi spread", "Poly spread"]
    wins = ["Our win", "Kalshi win", "Poly win"]
    totals = ["Our total", "Vegas total"]
    gaps = ["Spread gap", "Total gap"]
    average_total = float(available["vegas_total"].mean())
    columns = ["away_logo", "home_logo", "Game", *spreads, "Spread gap", *wins, *totals, "Total gap"]
    if display["Week"].nunique() > 1:
        columns.insert(0, "Week")
    st.dataframe(
        ui.styled_frame(
            display[columns],
            {**{column: ui.signed for column in spreads + gaps}, **{column: ui.percent for column in wins}, **{column: ui.number for column in totals}},
            {
                **{column: (0.0, ui.SPREAD_SCALE) for column in spreads},
                **{column: (0.5, ui.WIN_SCALE) for column in wins},
                **{column: (average_total, ui.TOTAL_SCALE) for column in totals},
                **{column: (0.0, ui.GAP_SCALE) for column in gaps},
            },
        ),
        hide_index=True,
        width="stretch",
        height=35 * (len(display) + 1) + 3,
        column_config={
            "away_logo": st.column_config.ImageColumn("Away", width="small"),
            "home_logo": st.column_config.ImageColumn("Home", width="small"),
        },
    )

    spread_column, total_column = st.columns(2)
    spread_chart = _market_comparison_chart(available, "vegas_spread", "predicted_spread", "Home spread", "+.1f")
    total_chart = _market_comparison_chart(available, "vegas_total", "predicted_total", "Total")
    spread_column.plotly_chart(spread_chart, use_container_width=True, theme=None, key="market_spread_comparison")
    ui.image_actions(spread_chart, "spread vs market", container=spread_column)
    total_column.plotly_chart(total_chart, use_container_width=True, theme=None, key="market_total_comparison")
    ui.image_actions(total_chart, "total vs market", container=total_column)


# ------------------------------------------------------------ scorigami contour


def render_scorigami(game_games: pd.DataFrame, game_label: str, selected_sims: Optional[Set[int]] = None) -> None:
    st.header(f"Score Distribution: {game_label}")
    if selected_sims is not None:
        game_games = game_games[game_games["simulation"].isin(selected_sims)]
    if game_games.empty:
        st.info("No simulated games for this selection.")
        return

    home_team = str(game_games["home_team"].iloc[0])
    away_team = str(game_games["away_team"].iloc[0])
    home_scores = game_games["home_score"].round().astype(int)
    away_scores = game_games["away_score"].round().astype(int)
    axis_max = int(max(home_scores.max(), away_scores.max())) + 2

    # Every whole score 0..axis_max gets its own 1x1 cell, even if it never occurred, so the
    # grid squares are uniform (hockeyviz-style) instead of only spanning observed scores.
    all_scores = list(range(0, axis_max + 1))
    counts = (
        pd.crosstab(away_scores, home_scores)
        .reindex(index=all_scores, columns=all_scores, fill_value=0)
    )
    # Splat each simulated score as a small 2D normal onto its neighbors (Gaussian-blurred
    # counts), so the plot shows the overall shape rather than isolated exact-score cells.
    z = gaussian_filter(counts.to_numpy(dtype=float), sigma=1.5)

    dx = dy = 1.0
    # Discrete color buckets (BoundaryNorm) instead of continuous blending, plus a
    # LightSource shade pass for the 3D-relief illumination look.
    levels = MaxNLocator(nbins=10).tick_values(0, max(float(z.max()), 1.0))
    cmap = SCORIGAMI_CMAP
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    light = LightSource(azdeg=315, altdeg=45)
    # shade_rgb keeps the discrete bucket colors while taking relief from a 0..1 normalized
    # surface; shading raw counts directly saturates the hillshade because the per-cell slope
    # (counts per 1 point of score) dwarfs the grid spacing.
    elevation = z / max(float(z.max()), 1e-9)
    rgb = light.shade_rgb(
        cmap(norm(z))[:, :, :3], elevation, vert_exag=RELIEF_EXAGGERATION, dx=dx, dy=dy, blend_mode="overlay"
    )

    extent = [
        all_scores[0] - dx / 2, all_scores[-1] + dx / 2,
        all_scores[0] - dy / 2, all_scores[-1] + dy / 2,
    ]

    fig, ax = plt.subplots(figsize=(4, 3.3))
    ax.imshow(rgb, extent=extent, origin="lower", interpolation="nearest", aspect="equal")

    # Thin gridlines aligned exactly to each 1x1 cell edge.
    x_edges = np.linspace(extent[0], extent[1], len(all_scores) + 1)
    y_edges = np.linspace(extent[2], extent[3], len(all_scores) + 1)
    ax.set_xticks(x_edges, minor=True)
    ax.set_yticks(y_edges, minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.3, alpha=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks(np.arange(0, axis_max + 1, 5))
    ax.set_yticks(np.arange(0, axis_max + 1, 5))

    # Bold border around each tied-score cell instead of a plain diagonal line.
    for score in all_scores:
        ax.add_patch(
            plt.Rectangle((score - dx / 2, score - dy / 2), dx, dy, fill=False, edgecolor="black", linewidth=0.6)
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, extend="max", ticks=levels)
    cbar.set_label("Sim Density")

    ax.set_title(f"{away_team} @ {home_team}")
    ax.set_xlabel(f"{home_team} Points")
    ax.set_ylabel(f"{away_team} Points")
    fig.tight_layout()

    st.pyplot(fig)
    ui.image_actions(fig, f"{away_team} at {home_team} score heatmap")
    plt.close(fig)


# --------------------------------------------------- total score vs. margin


def render_total_score_margin(
    game_games: pd.DataFrame, game_label: str, selected_sims: Optional[Set[int]] = None
) -> None:
    """One total-score bin (width 5) per column; each column is its own horizontal margin histogram."""
    st.header(f"Total Score vs. Margin: {game_label}")
    if selected_sims is not None:
        game_games = game_games[game_games["simulation"].isin(selected_sims)]
    if game_games.empty:
        st.info("No simulated games for this selection.")
        return

    home_team = str(game_games["home_team"].iloc[0])
    away_team = str(game_games["away_team"].iloc[0])
    home_color = team_color(home_team, LEFT_COLOR)
    away_color = team_color(away_team, RIGHT_COLOR)
    total_score = game_games["home_score"] + game_games["away_score"]
    home_margin = game_games["home_score"] - game_games["away_score"]

    n_sims = len(home_margin)
    home_win_pct = float((home_margin > 0).mean() * 100)
    away_win_pct = float((home_margin < 0).mean() * 100)
    push_pct = 100.0 - home_win_pct - away_win_pct
    avg_margin = float(home_margin.mean())
    favorite, spread = (home_team, avg_margin) if avg_margin >= 0 else (away_team, -avg_margin)
    push_note = f", push {push_pct:.1f}%" if push_pct > 0 else ""
    st.caption(
        f"{n_sims} filtered sims: {home_team} win {home_win_pct:.1f}% / {away_team} win {away_win_pct:.1f}%"
        f"{push_note}, spread: {favorite} \u2212{spread:.1f}"
    )

    bin_size = 5
    x_start = int((total_score.min() // bin_size) * bin_size)
    x_end = int(total_score.max()) + bin_size
    bin_edges = list(range(x_start, x_end + bin_size, bin_size))
    bin_labels = [f"{lo}-{lo + bin_size - 1}" for lo in bin_edges[:-1]]

    margin_max = int(max(abs(home_margin.min()), abs(home_margin.max()), 1)) + 1
    margin_bin_size = 1
    margin_edges = np.arange(-margin_max - 0.5, margin_max + 0.5 + margin_bin_size, margin_bin_size)
    margin_centers = (margin_edges[:-1] + margin_edges[1:]) / 2

    fig = make_subplots(
        rows=1, cols=len(bin_labels), shared_yaxes=True, horizontal_spacing=0.0,
        subplot_titles=bin_labels,
    )
    for col, lo in enumerate(bin_edges[:-1], start=1):
        hi = lo + bin_size
        margins_in_bin = home_margin[(total_score >= lo) & (total_score < hi)]
        counts, _ = np.histogram(margins_in_bin, bins=margin_edges)
        colors = [away_color if center < 0 else home_color for center in margin_centers]
        fig.add_trace(
            go.Bar(x=counts, y=margin_centers, orientation="h", marker_color=colors, showlegend=False),
            row=1, col=col,
        )
        fig.add_hline(y=0, line_dash="dash", line_color=ui.MUTED, line_width=1, row=1, col=col)
        fig.update_xaxes(showticklabels=False, row=1, col=col)

    fig.update_yaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=True, side="right", row=1, col=len(bin_labels))
    fig.add_annotation(
        xref="paper", yref="y", x=1.0, y=margin_max, text=f"\u2191 {home_team} ({home_win_pct:.1f}%)",
        showarrow=False, xanchor="left", yanchor="top", font=dict(color=home_color, size=13),
    )
    fig.add_annotation(
        xref="paper", yref="y", x=1.0, y=-margin_max, text=f"\u2193 {away_team} ({away_win_pct:.1f}%)",
        showarrow=False, xanchor="left", yanchor="bottom", font=dict(color=away_color, size=13),
    )
    fig.update_layout(
        bargap=0,
        height=480,
        margin=dict(r=130),
        annotations=list(fig.layout.annotations) + [
            dict(
                text="Total score (5-point bins)", xref="paper", yref="paper",
                x=0.5, y=-0.12, showarrow=False, font=dict(size=13),
            ),
        ],
    )
    st.plotly_chart(ui.style_plotly(fig), use_container_width=True, theme=None, key="total_score_margin_hist")
    ui.image_actions(fig, "total score and margin")


# --------------------------------------------------------- player stat histograms


def stat_columns(players_df: pd.DataFrame) -> List[str]:
    return [
        column
        for column in players_df.columns
        if column not in ID_COLUMNS and pd.api.types.is_numeric_dtype(players_df[column])
    ]


def wide_series(game_players: pd.DataFrame, player: str, stat: str) -> pd.Series:
    subset = game_players.loc[game_players["player_name"] == player, ["simulation", stat]].dropna()
    return subset.groupby("simulation")[stat].mean()


def valid_stats_for_player(game_players: pd.DataFrame, player: str) -> List[str]:
    """Stats that actually vary for this player (e.g. a QB has no receiving_yards)."""
    subset = game_players.loc[game_players["player_name"] == player]
    valid = []
    for stat in stat_columns(game_players):
        values = pd.to_numeric(subset[stat], errors="coerce").dropna()
        if not values.empty and (values != 0).any():
            valid.append(stat)
    return valid


def sims_in_range(series: pd.Series, value_range: Optional[Tuple[float, float]]) -> Optional[Set[int]]:
    if value_range is None:
        return None
    low, high = value_range
    return set(series[(series >= low) & (series <= high)].index)


def combine_constraints(*constraints: Optional[Set[int]]) -> Optional[Set[int]]:
    active = [c for c in constraints if c is not None]
    if not active:
        return None
    combined = active[0]
    for extra in active[1:]:
        combined = combined & extra
    return combined


def build_histogram(full_series: pd.Series, selected_sims: Optional[Set[int]], label: str, color: str) -> go.Figure:
    """A single trace over the currently filtered sims, redrawn fresh (no fade overlay)."""
    display_series = full_series if selected_sims is None else full_series[full_series.index.isin(selected_sims)]
    fig = go.Figure()
    if not display_series.empty:
        low, high = float(full_series.min()), float(full_series.max())
        bin_size = (high - low) / 30 or 1.0
        fig.add_trace(
            go.Histogram(
                x=display_series.values,
                marker_color=color,
                xbins=dict(start=low, end=high + bin_size, size=bin_size),
            )
        )
    fig.update_layout(xaxis_title=label, yaxis_title="Count", height=420, showlegend=False)
    return ui.style_plotly(fig)


def _stat_heatmap_edges(values: pd.Series, bins: int = 12, max_integer_bins: int = 24) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    low, high = float(numeric.min()), float(numeric.max())
    integer_values = np.all(np.isclose(numeric, np.rint(numeric)))
    if integer_values:
        low_integer, high_integer = round(low), round(high)
        value_count = high_integer - low_integer + 1
        bin_width = max(1, int(np.ceil(value_count / max_integer_bins)))
        offsets = np.arange(0, value_count, bin_width)
        edges = low_integer - 0.5 + offsets
        return np.append(edges, high_integer + 0.5)
    if low == high:
        return np.linspace(low - 0.5, high + 0.5, bins + 1)
    return np.linspace(low, high, bins + 1)


def build_stat_heatmap(
    left_series: pd.Series, right_series: pd.Series, selected_sims: Optional[Set[int]], left_label: str, right_label: str
) -> plt.Figure:
    joined = pd.concat([left_series.rename("left"), right_series.rename("right")], axis=1).dropna()
    if selected_sims is not None:
        joined = joined[joined.index.isin(selected_sims)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if joined.empty:
        ax.text(0.5, 0.5, "No simulations match the current filters.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    x_edges = _stat_heatmap_edges(joined["left"])
    y_edges = _stat_heatmap_edges(joined["right"])
    counts, x_edges, y_edges = np.histogram2d(
        joined["left"], joined["right"], bins=(x_edges, y_edges)
    )
    outcome_percentages = (counts / len(joined) * 100).T
    levels = MaxNLocator(nbins=10).tick_values(0, max(float(outcome_percentages.max()), 1.0))
    cmap = plt.get_cmap("Blues")
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    elevation = outcome_percentages / max(float(outcome_percentages.max()), 1e-9)
    rgb = LightSource(azdeg=315, altdeg=45).shade_rgb(
        cmap(norm(outcome_percentages))[:, :, :3],
        elevation,
        vert_exag=RELIEF_EXAGGERATION,
        dx=1.0,
        dy=1.0,
        blend_mode="overlay",
    )
    ax.pcolormesh(
        x_edges, y_edges, rgb, shading="flat", edgecolors="black", linewidth=0.4
    )
    ax.set_aspect("auto")
    for axis, edges in ((ax.xaxis, x_edges), (ax.yaxis, y_edges)):
        integer_bins = np.allclose(edges + 0.5, np.rint(edges + 0.5)) and np.allclose(
            np.diff(edges), np.rint(np.diff(edges))
        )
        if integer_bins:
            if len(edges) - 1 <= 10:
                axis.set_ticks((edges[:-1] + edges[1:]) / 2)
            else:
                axis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.set_xlabel(left_label)
    ax.set_ylabel(right_label)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(
        sm,
        ax=ax,
        label="Percentage of outcomes",
        ticks=levels,
        format=PercentFormatter(xmax=100, decimals=1),
    )
    fig.tight_layout()
    return fig


def render_filters(game_players: pd.DataFrame, players: List[str]) -> Optional[Set[int]]:
    """Explicit player/stat/range filters, not graph zoom or brushing, that highlight matching sims above."""
    st.markdown("**Filters**")
    st.session_state.setdefault("filter_ids", [0])
    st.session_state.setdefault("next_filter_id", 1)

    constraints: List[Optional[Set[int]]] = []
    for filter_id in list(st.session_state.filter_ids):
        player_col, stat_col, min_col, max_col, slider_col, remove_col = st.columns([2, 2, 1, 1, 3, 1])
        player = player_col.selectbox("Player", players, key=f"filter_player_{filter_id}", help='The player this filter applies to.')
        valid_stats = valid_stats_for_player(game_players, player)
        if not valid_stats:
            stat_col.info("No valid stats")
            continue
        stat = stat_col.selectbox("Stat", valid_stats, key=f"filter_stat_{filter_id}_{player}", help='The stat this filter applies to.')
        series = wide_series(game_players, player, stat)
        if series.empty:
            slider_col.info("No data")
            continue
        low, high = int(np.floor(series.min())), int(np.ceil(series.max()))
        if low >= high:
            slider_col.caption(f"Constant value: {low} (no filtering)")
            continue

        suffix = f"{filter_id}_{player}_{stat}"
        slider_key, min_key, max_key = f"filter_range_{suffix}", f"filter_min_{suffix}", f"filter_max_{suffix}"

        def sync_from_slider(slider_key=slider_key, min_key=min_key, max_key=max_key) -> None:
            lo, hi = st.session_state[slider_key]
            st.session_state[min_key], st.session_state[max_key] = int(lo), int(hi)

        def sync_from_text(slider_key=slider_key, min_key=min_key, max_key=max_key) -> None:
            lo, hi = int(st.session_state[min_key]), int(st.session_state[max_key])
            st.session_state[slider_key] = (min(lo, hi), max(lo, hi))

        min_col.number_input(
            "Min", min_value=low, max_value=high, value=low, step=1, key=min_key, on_change=sync_from_text
        , help='Lowest value to keep.')
        max_col.number_input(
            "Max", min_value=low, max_value=high, value=high, step=1, key=max_key, on_change=sync_from_text
        , help='Highest value to keep.')
        slider_col.slider(
            "Range", min_value=low, max_value=high, value=(low, high), step=1, key=slider_key, on_change=sync_from_slider
        , help="Only keep simulations where this player's stat falls in this range.")
        selected_range = st.session_state[slider_key]

        if len(st.session_state.filter_ids) > 1 and remove_col.button("-", key=f"filter_remove_{filter_id}", help='Remove this filter.'):
            st.session_state.filter_ids.remove(filter_id)
            st.rerun()

        constraints.append(sims_in_range(series, selected_range))

    if st.button("+ Add filter", key="add_filter_button", help='Add another condition. Charts then only use simulations that meet every condition.'):
        st.session_state.filter_ids.append(st.session_state.next_filter_id)
        st.session_state.next_filter_id += 1
        st.rerun()

    return combine_constraints(*constraints)


def render_receptions_chart(game_players: pd.DataFrame, selected_sims: Optional[Set[int]]) -> None:
    """Average receptions per pass catcher, grouped by team, over the currently filtered sims."""
    if "receptions" not in game_players.columns:
        return

    subset = game_players
    if selected_sims is not None:
        subset = subset[subset["simulation"].isin(selected_sims)]
    if subset.empty:
        st.info("No simulations match the current filters.")
        return

    averages = (
        subset.groupby(["team", "player_name"])["receptions"]
        .mean()
        .reset_index()
        .query("receptions > 0")
        .sort_values("receptions")
    )
    if averages.empty:
        st.info("No receptions recorded for the current filters.")
        return

    fig = go.Figure()
    for team, group in averages.groupby("team"):
        fig.add_trace(
            go.Bar(x=group["receptions"], y=group["player_name"], orientation="h", name=str(team))
        )
    fig.update_layout(
        barmode="group",
        xaxis_title="Average receptions",
        yaxis_title="",
        height=max(320, 26 * len(averages)),
        legend_title="Team",
    )
    fig.update_yaxes(automargin=True)

    sims_label = len(subset["simulation"].unique())
    st.markdown(f"**Average receptions by player ({sims_label} simulations)**")
    st.plotly_chart(ui.style_plotly(fig), use_container_width=True, theme=None, key="receptions_chart")
    ui.image_actions(fig, "average receptions")


@st.fragment  # Picking a player or stat here doesn't redraw the score charts.
def render_stat_histograms(
    game_players: pd.DataFrame, game_label: str, players: List[str], selected_sims: Optional[Set[int]]
) -> None:
    st.subheader(f"Player Stat Distributions: {game_label}")
    if not players:
        st.info("No player stat data available for this game.")
        return

    left_col, right_col = st.columns(2)
    with left_col:
        left_player = st.selectbox("Left player", players, key="left_player_select", help='Player for the left chart.')
        left_stats = valid_stats_for_player(game_players, left_player)
        if not left_stats:
            st.info(f"No valid stats for {left_player}.")
            return
        left_stat = st.selectbox("Left stat", left_stats, key=f"left_stat_select_{left_player}", help='Stat for the left chart.')
    with right_col:
        default_right_index = min(1, len(players) - 1)
        right_player = st.selectbox("Right player", players, index=default_right_index, key="right_player_select", help='Player for the right chart.')
        right_stats = valid_stats_for_player(game_players, right_player)
        if not right_stats:
            st.info(f"No valid stats for {right_player}.")
            return
        right_stat = st.selectbox("Right stat", right_stats, key=f"right_stat_select_{right_player}", help='Stat for the right chart.')

    left_series = wide_series(game_players, left_player, left_stat)
    right_series = wide_series(game_players, right_player, right_stat)

    with left_col:
        left_chart = build_histogram(left_series, selected_sims, f"{left_player} {left_stat}", LEFT_COLOR)
        st.plotly_chart(left_chart, use_container_width=True, theme=None, key="hist_left")
        ui.image_actions(left_chart, f"{left_player} {left_stat}")
    with right_col:
        right_chart = build_histogram(right_series, selected_sims, f"{right_player} {right_stat}", RIGHT_COLOR)
        st.plotly_chart(right_chart, use_container_width=True, theme=None, key="hist_right")
        ui.image_actions(right_chart, f"{right_player} {right_stat}")

    st.markdown(f"**{left_player} {left_stat} vs. {right_player} {right_stat}**")
    heatmap = build_stat_heatmap(
        left_series, right_series, selected_sims, f"{left_player} {left_stat}", f"{right_player} {right_stat}"
    )
    st.pyplot(heatmap, width="stretch")
    ui.image_actions(heatmap, f"{left_player} {left_stat} vs {right_player} {right_stat}")
    plt.close(heatmap)

    render_receptions_chart(game_players, selected_sims)


# ------------------------------------------------------------------------- main


@st.fragment
def game_explorer(games_df: pd.DataFrame, players_df: pd.DataFrame) -> None:
    """Runs on its own, so choices here don't redraw the market summary."""
    game_labels: Dict[str, str] = {}
    for game_id in games_df["game_id"].unique():
        row = games_df.loc[games_df["game_id"] == game_id].iloc[0]
        game_labels[f"{row['away_team']} @ {row['home_team']} ({game_id})"] = game_id
    game_label = st.selectbox("Game", list(game_labels), key="results_game_select", help='The game to explore.')
    selected_game_id = game_labels[game_label]
    game_games = games_df[games_df["game_id"] == selected_game_id]
    game_players = players_df[players_df["game_id"] == selected_game_id]
    players = sorted(game_players["player_name"].dropna().unique())

    selected_sims = render_filters(game_players, players) if players else None

    render_scorigami(game_games, game_label, selected_sims)
    st.divider()
    render_total_score_margin(game_games, game_label, selected_sims)
    st.divider()
    render_stat_histograms(game_players, game_label, players, selected_sims)


def main() -> None:
    ui.page("Weekly Matchups")
    ui.header("Weekly matchups", "Saved simulation runs compared with the market, game by game.", eyebrow="Game results")
    runs = weekly_runs()
    if not runs:
        st.info("No weekly simulation runs yet. Run a full week on the Game Simulator page.")
        return
    folder = st.selectbox("Simulation run", [str(path) for path in runs], format_func=lambda path: Path(path).name, help='A saved full-week or single-game run, newest first. The name comes from its output folder.')
    games_df, players_df = load_week_results(folder)

    try:
        games_df = attach_market_lines(games_df)
    except Exception as error:
        st.warning(f"Could not load Vegas lines from the schedule: {error}")
        if "spread_line" not in games_df.columns:
            games_df["spread_line"] = np.nan
        if "total_line" not in games_df.columns:
            games_df["total_line"] = np.nan

    summary_tab, explorer_tab = st.tabs(["Market Summary", "Game Explorer"])
    with summary_tab:
        render_market_summary(games_df)

    with explorer_tab:
        game_explorer(games_df, players_df)


main()
