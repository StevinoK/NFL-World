"""Single-game flow: win-probability worm, cumulative EPA worm and drive summary.

Works on nflverse play-by-play for one game. Used by the "Single game" view of
`game_sim_results_dashboard.py`; the per-player outcome chart lives in `game_report.py`.

Definitions:

* Game clock: seconds elapsed since kickoff, so overtime extends past 60:00
  (10-minute regular-season OT, 15-minute playoff OT, read from the data).
* Offensive plays: passes, runs, kneels, spikes and penalty-nullified scrimmage
  plays. Special-teams plays (kickoffs, punts, field goals, extra points and
  penalties on them) count only when ``include_special_teams`` is set. EPA is
  credited to ``posteam`` either way, as nflverse does.
* Win probability is nflverse's pre-snap ``home_wp`` (or ``vegas_home_wp``, which
  also uses the closing spread); the worm ends at the final result.
* Drives use nflverse ``fixed_drive``; drive EPA is the sum of EPA on the drive's
  counted plays.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

GAME_COLUMNS = [
    "season", "season_type", "week", "game_id", "play_id", "order_sequence", "home_team", "away_team",
    "posteam", "defteam", "qtr", "quarter_seconds_remaining", "game_seconds_remaining",
    "time", "desc", "play_type", "epa", "success", "yards_gained", "penalty",
    "home_wp", "vegas_home_wp", "total_home_score", "total_away_score",
    "fixed_drive", "fixed_drive_result", "drive_play_count", "drive_time_of_possession",
    "drive_first_downs", "drive_start_yard_line", "drive_game_clock_start",
    "touchdown", "interception", "fumble_lost",
]

OFFENSE_PLAY_TYPES = {"pass", "run", "qb_kneel", "qb_spike"}
SPECIAL_PLAY_TYPES = {"kickoff", "punt", "field_goal", "extra_point"}
_SPECIAL_DESC = r"\b(?:punts|kicks|field goal|extra point)\b"

RESULT_SHORT = {
    "Touchdown": "TD",
    "Field goal": "FG",
    "Punt": "Punt",
    "Turnover": "TO",
    "Turnover on downs": "Downs",
    "End of half": "Half",
    "Missed field goal": "Missed FG",
    "Opp touchdown": "Opp TD",
    "Safety": "Safety",
}


# ------------------------------------------------------------------ game selection


def game_catalog(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per game with final score and a display label, most recent first."""
    if pbp.empty:
        return pd.DataFrame(columns=["game_id", "label"])
    order = "order_sequence" if "order_sequence" in pbp else "play_id"
    last = pbp.sort_values(["game_id", order]).groupby("game_id").tail(1)
    cat = last[["game_id", "season", "season_type", "week", "home_team", "away_team",
                "total_home_score", "total_away_score"]].copy()
    cat = cat.sort_values(["season", "week", "game_id"], ascending=[False, False, True]).reset_index(drop=True)
    suffix = np.where(cat["season_type"] == "REG", "", " (P)")
    cat["label"] = [
        f"{int(r.season)} W{int(r.week)}{s} · {r.away_team} {int(r.total_away_score)} @ "
        f"{r.home_team} {int(r.total_home_score)}"
        for r, s in zip(cat.itertuples(), suffix)
    ]
    return cat


def game_plays(pbp: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """Plays of one game in order, with elapsed game seconds and play-kind flags."""
    plays = pbp[pbp["game_id"] == game_id]
    order = "order_sequence" if "order_sequence" in plays and plays["order_sequence"].notna().all() else "play_id"
    plays = plays.sort_values(order, kind="stable").copy()
    # A few source rows carry a stale clock; never let game time run backwards.
    plays["elapsed"] = elapsed_seconds(plays).cummax()
    desc = plays["desc"].astype(str)
    is_no_play = plays["play_type"].eq("no_play")
    special = plays["play_type"].isin(SPECIAL_PLAY_TYPES) | (
        is_no_play & desc.str.contains(_SPECIAL_DESC, case=False, regex=True)
    )
    plays["is_special"] = special
    plays["is_offense"] = (plays["play_type"].isin(OFFENSE_PLAY_TYPES) | is_no_play) & ~special
    return plays


def elapsed_seconds(plays: pd.DataFrame) -> pd.Series:
    """Seconds since kickoff; overtime periods continue past 3600."""
    qtr = plays["qtr"].astype(float)
    qsr = plays["quarter_seconds_remaining"].astype(float)
    regulation = (qtr - 1) * 900 + (900 - qsr)
    # OT length differs (600 regular season, 900 playoffs): take each period's start value.
    ot_len = plays.groupby(qtr)["quarter_seconds_remaining"].transform("max").astype(float)
    overtime = 3600 + (qtr - 5) * 900 + (ot_len - qsr)
    return pd.Series(np.where(qtr <= 4, regulation, overtime), index=plays.index)


def counted_plays(plays: pd.DataFrame, include_special_teams: bool = False) -> pd.DataFrame:
    mask = plays["is_offense"] | (plays["is_special"] if include_special_teams else False)
    return plays[mask & plays["epa"].notna() & plays["posteam"].notna()]


# --------------------------------------------------------------------------- drives


def drive_summary(plays: pd.DataFrame, include_special_teams: bool = False) -> pd.DataFrame:
    """One row per offensive drive, in game order."""
    counted = counted_plays(plays, include_special_teams)
    scrimmage = plays[plays["is_offense"] & plays["posteam"].notna() & plays["fixed_drive"].notna()]
    if scrimmage.empty:
        return pd.DataFrame(columns=["drive", "seq", "team", "label"])
    first = scrimmage.groupby("fixed_drive").head(1).set_index("fixed_drive")
    agg = scrimmage.groupby("fixed_drive").agg(
        team=("posteam", lambda s: s.mode().iloc[0]),
        start_elapsed=("elapsed", "min"),
        end_elapsed=("elapsed", "max"),
        yards=("yards_gained", lambda s: float(s.fillna(0).sum())),
        success_rate=("success", "mean"),
    )
    epa = counted.groupby("fixed_drive")["epa"].agg(["sum", "size"])
    drives = agg.join(epa).rename(columns={"sum": "epa", "size": "epa_plays"})
    drives["epa"] = drives["epa"].fillna(0.0)
    drives["epa_plays"] = drives["epa_plays"].fillna(0).astype(int)
    drives["epa_per_play"] = drives["epa"] / drives["epa_plays"].where(drives["epa_plays"] > 0)
    drives["qtr"] = first["qtr"].astype(int)
    drives["clock"] = first["drive_game_clock_start"].fillna(first["time"])
    drives["start"] = first["drive_start_yard_line"]
    drives["plays"] = first["drive_play_count"]
    drives["top"] = first["drive_time_of_possession"]
    drives["first_downs"] = first["drive_first_downs"]
    drives["result"] = first["fixed_drive_result"].fillna("")
    drives = drives.sort_values("start_elapsed").reset_index().rename(columns={"fixed_drive": "drive"})
    drives["drive"] = drives["drive"].astype(int)
    drives["seq"] = np.arange(1, len(drives) + 1)
    drives["result_short"] = drives["result"].map(RESULT_SHORT).fillna(drives["result"])
    drives["label"] = [
        f"{r.seq}. {r.team} Q{r.qtr} {r.clock} · {r.result_short or '—'} · {r.epa:+.1f} EPA"
        for r in drives.itertuples()
    ]
    return drives


def drive_table(drives: pd.DataFrame) -> pd.DataFrame:
    """Display version of ``drive_summary``."""
    table = drives[[
        "seq", "team", "qtr", "clock", "start", "plays", "yards", "first_downs", "top",
        "result", "epa", "epa_per_play", "success_rate",
    ]].rename(columns={
        "seq": "Drive", "team": "Team", "qtr": "Q", "clock": "Clock", "start": "Start",
        "plays": "Plays", "yards": "Yards", "first_downs": "1st downs", "top": "TOP",
        "result": "Result", "epa": "EPA", "epa_per_play": "EPA/play", "success_rate": "Success",
    })
    table["Yards"] = table["Yards"].astype(int)
    return table.set_index("Drive").round({"EPA": 2, "EPA/play": 2, "Success": 2})


# -------------------------------------------------------------------------- figures


def _quarter_axis(fig: go.Figure, end: float) -> None:
    ticks = [0, 900, 1800, 2700, 3600]
    labels = ["Q1", "Q2", "Q3", "Q4", "End"]
    if end > 3600:
        labels[-1] = "OT"
    for x in ticks[1:4] + ([3600] if end > 3600 else []):
        fig.add_vline(x=x, line=dict(color="rgba(128,128,128,0.35)", width=1, dash="dot"))
    fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=labels, range=[0, max(end, 3600)],
                     showgrid=False, zeroline=False)


def _hover_text(plays: pd.DataFrame) -> List[str]:
    from game_report import _hover_desc  # shared wrapper for long descriptions

    return [
        f"Q{int(q)} {t} · {pt or ''}<br>{_hover_desc(d)}"
        for q, t, pt, d in zip(plays["qtr"], plays["time"], plays["posteam"].fillna(""), plays["desc"])
    ]


def build_wp_figure(
    plays: pd.DataFrame,
    home_color: str,
    away_color: str,
    spread_adjusted: bool = False,
    height: int = 340,
) -> go.Figure:
    """Home win probability over game time, filled toward whichever team leads."""
    home, away = plays["home_team"].iloc[0], plays["away_team"].iloc[0]
    column = "vegas_home_wp" if spread_adjusted else "home_wp"
    wp_plays = plays[plays[column].notna()]
    x = wp_plays["elapsed"].to_numpy(dtype=float)
    wp = wp_plays[column].to_numpy(dtype=float)
    hover = _hover_text(wp_plays)
    final_home = plays["total_home_score"].iloc[-1]
    final_away = plays["total_away_score"].iloc[-1]
    end = float(plays["elapsed"].max())
    final_wp = 1.0 if final_home > final_away else 0.0 if final_home < final_away else 0.5
    x = np.append(x, end)
    wp = np.append(wp, final_wp)
    hover.append(f"Final: {away} {int(final_away)} – {home} {int(final_home)}")

    fig = go.Figure()
    for team, color, clipped in (
        (home, home_color, np.maximum(wp, 0.5)),
        (away, away_color, np.minimum(wp, 0.5)),
    ):
        fig.add_trace(go.Scatter(x=x, y=np.full_like(wp, 0.5), mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=x, y=clipped, mode="lines", line=dict(width=0, shape="hv"), fill="tonexty",
            fillcolor=_with_alpha(color, 0.28), hoverinfo="skip", name=f"{team} ahead", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=x, y=wp, mode="lines", line=dict(color="#3d3d3a", width=2, shape="hv"), name=f"{home} win %",
        text=hover, hovertemplate=f"{home} win %{{y:.0%}}<br>%{{text}}<extra></extra>", showlegend=False,
    ))
    fig.add_hline(y=0.5, line=dict(color="rgba(128,128,128,0.6)", width=1))
    _quarter_axis(fig, end)
    fig.update_layout(
        title=f"Win probability{' (spread-adjusted)' if spread_adjusted else ''}",
        height=height, margin=dict(t=50, b=30, l=60, r=60), hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1],
                   ticktext=[f"{away} 100%", f"{away} 75%", "50%", f"{home} 75%", f"{home} 100%"],
                   gridcolor="rgba(128,128,128,0.15)", zeroline=False),
    )
    return fig


def build_epa_worm(
    plays: pd.DataFrame,
    colors: Dict[str, str],
    include_special_teams: bool = False,
    height: int = 340,
) -> go.Figure:
    """Cumulative EPA for each offense over game time."""
    counted = counted_plays(plays, include_special_teams)
    end = float(plays["elapsed"].max())
    fig = go.Figure()
    for team in (plays["away_team"].iloc[0], plays["home_team"].iloc[0]):
        own = counted[counted["posteam"] == team]
        x = np.concatenate([[0.0], own["elapsed"].to_numpy(dtype=float), [end]])
        cum = own["epa"].cumsum().to_numpy(dtype=float)
        y = np.concatenate([[0.0], cum, [cum[-1] if len(cum) else 0.0]])
        text = ["Kickoff"] + [
            f"{h}<br>Play EPA {e:+.2f}" for h, e in zip(_hover_text(own), own["epa"])
        ] + ["End"]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=f"{team} offense", line=dict(color=colors[team], width=2.5, shape="hv"),
            text=text, hovertemplate=f"<b>{team}</b> cumulative %{{y:+.1f}} EPA<br>%{{text}}<extra></extra>",
        ))
        fig.add_annotation(x=end, y=y[-1], text=f"{team} {y[-1]:+.1f}", showarrow=False, xanchor="left",
                           xshift=6, font=dict(size=12))
    fig.add_hline(y=0, line=dict(color="rgba(128,128,128,0.6)", width=1))
    _quarter_axis(fig, end)
    fig.update_layout(
        title="Cumulative EPA" + (" (incl. special teams)" if include_special_teams else " (offense)"),
        height=height, margin=dict(t=50, b=30, l=60, r=70), hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
        yaxis=dict(title="EPA", gridcolor="rgba(128,128,128,0.15)", zeroline=False),
    )
    return fig


def build_drive_figure(drives: pd.DataFrame, colors: Dict[str, str], height: int = 360) -> go.Figure:
    """Bar per drive in game order: height = drive EPA, color = offense, label = result."""
    fig = go.Figure()
    for team in drives["team"].unique():
        own = drives[drives["team"] == team]
        custom = np.stack([
            own["drive"], own["qtr"], own["clock"].astype(str), own["start"].fillna("").astype(str),
            own["plays"].fillna(0).astype(int), own["yards"].astype(int), own["top"].fillna("").astype(str),
            own["result"], own["epa_per_play"].fillna(0.0).round(2),
            (own["success_rate"].fillna(0) * 100).round(0),
        ], axis=-1)
        fig.add_trace(go.Bar(
            x=own["seq"], y=own["epa"], name=team, marker=dict(color=colors[team]),
            text=own["result_short"], textposition="outside", cliponaxis=False,
            textfont=dict(size=11), customdata=custom,
            hovertemplate=(
                f"<b>{team}</b> drive %{{x}} · Q%{{customdata[1]}} %{{customdata[2]}}<br>"
                "Start %{customdata[3]} · %{customdata[4]} plays, %{customdata[5]} yds, TOP %{customdata[6]}<br>"
                "%{customdata[7]}<br>EPA %{y:+.2f} (%{customdata[8]:+.2f}/play) · success %{customdata[9]:.0f}%"
                "<extra></extra>"
            ),
        ))
    # Mark where each quarter's first drive starts.
    firsts = drives.groupby("qtr")["seq"].min()
    for qtr, seq in firsts.items():
        if seq > 1:
            fig.add_vline(x=seq - 0.5, line=dict(color="rgba(128,128,128,0.35)", width=1, dash="dot"))
        fig.add_annotation(x=seq - 0.45, y=1.0, yref="paper", text="OT" if qtr > 4 else f"Q{qtr}",
                           showarrow=False, xanchor="left", font=dict(size=11, color="#8f8e88"))
    span = max(float(drives["epa"].abs().max()), 1.0)
    fig.add_hline(y=0, line=dict(color="#52514e", width=1))
    fig.update_layout(
        title="EPA per drive", height=height, barmode="overlay", bargap=0.25,
        margin=dict(t=60, b=40, l=60, r=20), plot_bgcolor="rgba(0,0,0,0)", hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Drive", tickmode="linear", dtick=1, showgrid=False, zeroline=False),
        yaxis=dict(title="EPA", range=[-span * 1.25, span * 1.25], gridcolor="rgba(128,128,128,0.15)",
                   zeroline=False),
    )
    return fig


def _with_alpha(color: str, alpha: float) -> str:
    color = color.lstrip("#")
    if len(color) != 6:
        return f"rgba(128,128,128,{alpha})"
    r, g, b = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def distinct_team_colors(
    home: str, away: str, primary: Dict[str, str], secondary: Dict[str, str],
    fallback_home: str = "#2a78d6", fallback_away: str = "#eb6834",
) -> Dict[str, str]:
    """Team colors for a matchup, switching the home team to its secondary color if both
    primaries are too close to tell apart (e.g. two navy teams)."""
    away_color = primary.get(away) or fallback_away
    home_color = primary.get(home) or fallback_home
    if _distance(home_color, away_color) < 90:
        alt = secondary.get(home)
        home_color = alt if alt and _distance(alt, away_color) >= 90 else fallback_home
        if _distance(home_color, away_color) < 90:
            away_color = fallback_away
    return {home: home_color, away: away_color}


def _distance(a: str, b: str) -> float:
    try:
        ca = [int(a.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
        cb = [int(b.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    except ValueError:
        return 255.0
    return float(np.sqrt(sum((x - y) ** 2 for x, y in zip(ca, cb))))
