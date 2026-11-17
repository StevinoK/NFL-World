"""Post-game EPA report: per-player target EPA split by outcome.

Builds, from nflverse play-by-play, one segment per (target play, outcome component)
and draws them as boxes stacked above (positive EPA) or below (negative EPA) a
horizontal axis of player names. Used by the "Game report" view of
`game_sim_results_dashboard.py`.

Attribution rules for a targeted player (the receiver on a pass play):

* Completion   -> ``air_epa`` is "Reception", ``yac_epa`` is "YAC".
* Completion followed by a lost fumble by the target -> ``air_epa`` is "Reception",
  the rest of the play's EPA is "Fumble lost".
* Incompletion -> play EPA is "Incomplete".
* Interception -> play EPA is "Interception".
* Penalty that nullified the play (``play_type == "no_play"``, e.g. DPI, offensive
  holding) -> play EPA is "Penalty". nflverse does not fill ``receiver_player_id`` on
  these plays, so the target is parsed from the play description and resolved by
  jersey number against the same team's other targets in that game.

Two-point attempts are excluded (their EPA is not comparable to scrimmage plays).
Touchdowns are marked with a pattern: "/" when the offense scored, "x" when the
defense returned the play for a touchdown (pick-six, fumble return).
"""

from __future__ import annotations

import re
import textwrap
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PBP_COLUMNS = [
    "season", "season_type", "week", "game_id", "play_id", "posteam", "defteam",
    "home_team", "away_team", "qtr", "time", "desc", "play_type", "two_point_attempt",
    "epa", "air_epa", "yac_epa", "complete_pass", "incomplete_pass", "interception",
    "penalty", "fumble_lost", "fumbled_1_player_id", "fumbled_2_player_id",
    "receiver_player_id", "receiver_player_name", "touchdown", "td_team",
]

# Stacking order away from the axis, and legend order.
CATEGORIES = ["Reception", "YAC", "Penalty", "Incomplete", "Fumble lost", "Interception"]
CATEGORY_COLORS: Dict[str, str] = {
    "Reception": "#2a78d6",
    "YAC": "#1baf7a",
    "Penalty": "#eda100",
    "Incomplete": "#8f8e88",
    "Fumble lost": "#6250d6",
    "Interception": "#e34948",
}
TD_PATTERNS = {"none": "", "offense": "/", "defense": "x"}
GROUP_PLAYER_GAME = "Player per game"
GROUP_PLAYER = "Player (combined)"

# "pass short left to 18-M.Harrison" / "pass deep right intended for 22-R.Shaheed"
_TARGET_RE = re.compile(r"(?:\bto|intended for) (\d{1,2})-([^\s(\[]+)")
_PENALTY_SPLIT_RE = re.compile(r"penalty", re.IGNORECASE)
_DESC_JERSEY_RE = re.compile(r"(?:\bto|intended for) (\d{1,2})-")


def _flag(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(0).astype(float).eq(1)


def parse_target(desc: object) -> Optional[tuple]:
    """Return (jersey, short_name) of the last target named before any penalty text."""
    if not isinstance(desc, str):
        return None
    before_penalty = _PENALTY_SPLIT_RE.split(desc, maxsplit=1)[0]
    matches = _TARGET_RE.findall(before_penalty)
    if not matches:
        return None
    jersey, name = matches[-1]
    return jersey, name.rstrip(".,;")


def _attach_penalty_targets(pbp: pd.DataFrame) -> pd.DataFrame:
    """Fill receiver id/name on nullified pass plays by parsing the description."""
    pbp = pbp.copy()
    targeted = pbp["receiver_player_id"].notna()
    jerseys = pbp.loc[targeted, "desc"].astype(str).str.extract(_DESC_JERSEY_RE, expand=False)
    roster = (
        pd.DataFrame({
            "game_id": pbp.loc[targeted, "game_id"],
            "posteam": pbp.loc[targeted, "posteam"],
            "jersey": jerseys,
            "rid": pbp.loc[targeted, "receiver_player_id"],
            "rname": pbp.loc[targeted, "receiver_player_name"],
        })
        .dropna(subset=["jersey"])
        .drop_duplicates(["game_id", "posteam", "jersey"])
        .set_index(["game_id", "posteam", "jersey"])
    )

    nullified = (pbp["play_type"] == "no_play") & ~targeted & _flag(pbp, "penalty")
    nullified &= pbp["desc"].astype(str).str.contains(r"\bpass\b", regex=True)
    for idx in pbp.index[nullified]:
        parsed = parse_target(pbp.at[idx, "desc"])
        if parsed is None:
            continue
        jersey, name = parsed
        key = (pbp.at[idx, "game_id"], pbp.at[idx, "posteam"], jersey)
        if key in roster.index:
            pbp.at[idx, "receiver_player_id"] = roster.at[key, "rid"]
            pbp.at[idx, "receiver_player_name"] = roster.at[key, "rname"]
        else:  # target never otherwise targeted this game: stable synthetic id
            pbp.at[idx, "receiver_player_id"] = f"{key[1]}#{jersey}:{name}"
            pbp.at[idx, "receiver_player_name"] = name
    return pbp


def build_target_segments(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per EPA component of every target play, tagged with its category."""
    pbp = _attach_penalty_targets(pbp)
    plays = pbp[pbp["receiver_player_id"].notna() & pbp["epa"].notna()]
    plays = plays[~_flag(plays, "two_point_attempt")].copy()
    if plays.empty:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)

    rid = plays["receiver_player_id"]
    target_fumbled = (plays["fumbled_1_player_id"] == rid) | (plays["fumbled_2_player_id"] == rid)
    complete = _flag(plays, "complete_pass")
    interception = _flag(plays, "interception")
    incomplete = _flag(plays, "incomplete_pass")
    fumble_lost = complete & _flag(plays, "fumble_lost") & target_fumbled
    nullified = plays["play_type"].eq("no_play")

    air = plays["air_epa"].fillna(0.0)
    epa = plays["epa"]
    parts: List[pd.DataFrame] = []

    def add(mask: pd.Series, category: str, values: pd.Series) -> None:
        if mask.any():
            part = plays.loc[mask].copy()
            part["category"] = category
            part["seg_epa"] = values[mask]
            parts.append(part)

    add(nullified, "Penalty", epa)
    live = ~nullified
    add(live & interception, "Interception", epa)
    add(live & ~interception & complete, "Reception", air)
    add(live & ~interception & complete & fumble_lost, "Fumble lost", epa - air)
    add(live & ~interception & complete & ~fumble_lost, "YAC", epa - air)
    add(live & ~interception & ~complete & incomplete, "Incomplete", epa)
    # Live target plays with no pass outcome (penalty enforced on the play itself).
    add(live & ~interception & ~complete & ~incomplete, "Penalty", epa)

    seg = pd.concat(parts, ignore_index=True)
    td = _flag(seg, "touchdown")
    seg["td"] = np.where(~td, "none", np.where(seg["td_team"] == seg["posteam"], "offense", "defense"))
    seg = seg.rename(columns={
        "receiver_player_id": "player_id",
        "receiver_player_name": "player_name",
        "posteam": "team",
        "defteam": "opponent",
    })
    seg["epa"] = seg["seg_epa"].astype(float)
    seg["week"] = seg["week"].astype(int)
    seg["season"] = seg["season"].astype(int)
    return seg[SEGMENT_COLUMNS].reset_index(drop=True)


SEGMENT_COLUMNS = [
    "season", "season_type", "week", "game_id", "play_id", "team", "opponent", "qtr",
    "time", "desc", "player_id", "player_name", "category", "epa", "td",
]


def week_label(season: int, week: int, season_type: str = "REG") -> str:
    suffix = "" if season_type == "REG" else " (P)"
    return f"{season} W{week}{suffix}"


def team_game_catalog(segments: pd.DataFrame) -> pd.DataFrame:
    """Unique (game, offense) pairs, most recent first, with a display label."""
    cat = (
        segments[["season", "season_type", "week", "game_id", "team", "opponent"]]
        .drop_duplicates(["game_id", "team"])
        .sort_values(["season", "week", "team"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    cat["label"] = [
        f"{week_label(r.season, r.week, r.season_type)} · {r.team} vs {r.opponent}"
        for r in cat.itertuples()
    ]
    return cat


def player_game_catalog(segments: pd.DataFrame) -> pd.DataFrame:
    """Unique (player, game) pairs with targets and net EPA, most recent first."""
    plays = segments.drop_duplicates(["game_id", "play_id", "player_id"])
    targets = plays.groupby(["player_id", "game_id"]).size().rename("targets")
    net = segments.groupby(["player_id", "game_id"])["epa"].sum().rename("net_epa")
    base = (
        segments[["player_id", "player_name", "season", "season_type", "week", "game_id", "team", "opponent"]]
        .drop_duplicates(["player_id", "game_id"])
        .set_index(["player_id", "game_id"])
        .join(targets)
        .join(net)
        .reset_index()
        .sort_values(["season", "week", "player_name"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    base["label"] = [
        f"{r.player_name} ({r.team}) · {week_label(r.season, r.week, r.season_type)} vs {r.opponent}"
        f" · {r.targets} tgt, {r.net_epa:+.1f} EPA"
        for r in base.itertuples()
    ]
    return base


def _unit_columns(segments: pd.DataFrame, group_by: str) -> pd.DataFrame:
    seg = segments.copy()
    if group_by == GROUP_PLAYER:
        seg["unit"] = seg["player_id"].astype(str)
        teams = seg.groupby("unit")["team"].agg(lambda s: "/".join(dict.fromkeys(s)))
        seg["unit_label"] = seg["player_name"] + "<br>" + seg["unit"].map(teams)
    else:
        seg["unit"] = seg["player_id"].astype(str) + "|" + seg["game_id"].astype(str)
        seg["unit_label"] = [
            f"{r.player_name}<br>{r.team} W{r.week}" + ("" if r.season_type == "REG" else "P")
            + f" v {r.opponent}"
            for r in seg.itertuples()
        ]
    # Labels must be unique per unit (two players can share a short name).
    dup = seg.groupby("unit_label")["unit"].transform("nunique") > 1
    seg.loc[dup, "unit_label"] = seg.loc[dup, "unit_label"] + " #" + seg.loc[dup, "player_id"].astype(str).str[-4:]
    return seg


def stack_segments(segments: pd.DataFrame, group_by: str = GROUP_PLAYER_GAME) -> pd.DataFrame:
    """Add unit, base and order so each segment is a box stacked away from zero.

    Positive segments stack upward from 0 and negative ones downward; within each
    direction boxes are grouped by category (CATEGORIES order) then chronologically.
    """
    if segments.empty:
        return segments.assign(unit=[], unit_label=[], base=[])
    seg = _unit_columns(segments, group_by)
    seg = seg[seg["epa"] != 0].copy()
    seg["cat_order"] = seg["category"].map({c: i for i, c in enumerate(CATEGORIES)})
    seg["positive"] = seg["epa"] > 0
    seg = seg.sort_values(["unit", "positive", "cat_order", "season", "week", "play_id"])
    seg["top"] = seg.groupby(["unit", "positive"])["epa"].cumsum()
    seg["base"] = seg["top"] - seg["epa"]
    return seg.drop(columns=["top"]).reset_index(drop=True)


def unit_order(stacked: pd.DataFrame) -> List[str]:
    net = stacked.groupby("unit_label")["epa"].sum().sort_values(ascending=False)
    return list(net.index)


def _hover_desc(desc: object, width: int = 70) -> str:
    return "<br>".join(textwrap.wrap(str(desc), width)[:5])


def build_epa_figure(
    segments: pd.DataFrame,
    group_by: str = GROUP_PLAYER_GAME,
    title: str = "Target EPA by outcome",
) -> go.Figure:
    stacked = stack_segments(segments, group_by)
    fig = go.Figure()
    if stacked.empty:
        fig.update_layout(title=title)
        return fig
    order = unit_order(stacked)
    # Box outlines separate plays; thin them when bars hold many plays (combined view).
    max_boxes = int(stacked.groupby("unit").size().max())
    outline = 1.5 if max_boxes <= 30 else 0.75 if max_boxes <= 80 else 0.4

    for category in CATEGORIES:
        part = stacked[stacked["category"] == category]
        if part.empty:
            continue
        # Plotly draws the legend swatch from the first bar; keep that one unpatterned.
        part = part.sort_values("td", key=lambda s: s.ne("none"), kind="stable")
        custom = np.stack([
            part["player_name"],
            [f"{week_label(s, w, t)} {tm} vs {o}" for s, w, t, tm, o in
             zip(part["season"], part["week"], part["season_type"], part["team"], part["opponent"])],
            [f"Q{int(q)} {t}" if pd.notna(q) else "" for q, t in zip(part["qtr"], part["time"])],
            [_hover_desc(d) for d in part["desc"]],
            np.where(part["td"] == "none", "", np.where(part["td"] == "offense", " · TD", " · Defensive TD")),
        ], axis=-1)
        fig.add_trace(go.Bar(
            name=category,
            legendgroup=category,
            x=part["unit_label"],
            y=part["epa"],
            base=part["base"],
            marker=dict(
                color=CATEGORY_COLORS[category],
                line=dict(color="rgba(255,255,255,0.85)", width=outline),
                pattern=dict(
                    shape=[TD_PATTERNS[t] for t in part["td"]],
                    fillmode="overlay",
                    fgcolor="rgba(255,255,255,0.9)",
                    size=7,
                    solidity=0.3,
                ),
            ),
            width=0.7,
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b> · %{customdata[1]}<br>%{customdata[2]}<br>"
                f"{category}: " + "%{y:+.2f} EPA%{customdata[4]}<br><br>%{customdata[3]}<extra></extra>"
            ),
        ))

    # Legend keys for the touchdown patterns.
    for td_kind, label in (("offense", "Touchdown"), ("defense", "Defensive TD (return)")):
        if (stacked["td"] == td_kind).any():
            fig.add_trace(go.Bar(
                name=label, x=[order[0]], y=[0], base=[0], showlegend=True, hoverinfo="skip",
                marker=dict(color="rgba(0,0,0,0)", line=dict(color="#8f8e88", width=1),
                            pattern=dict(shape=TD_PATTERNS[td_kind], fillmode="replace", fgcolor="#52514e", size=7)),
            ))

    net = stacked.groupby("unit_label").agg(
        net=("epa", "sum"), player=("player_name", "first"),
        plays=("play_id", "nunique"),
    ).reindex(order)
    fig.add_trace(go.Scatter(
        name="Net EPA", x=net.index, y=net["net"], mode="markers",
        marker=dict(symbol="diamond", size=10, color="#0b0b0b", line=dict(color="white", width=2)),
        customdata=np.stack([net["player"], net["plays"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>Net %{y:+.2f} EPA on %{customdata[1]} targets<extra></extra>",
    ))

    fig.add_hline(y=0, line=dict(color="#52514e", width=1.5))
    fig.update_layout(
        title=title,
        barmode="overlay",
        bargap=0.25,
        height=620,
        hovermode="closest",
        title_y=0.97,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(t=110, b=40, l=60, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(categoryorder="array", categoryarray=order, tickangle=-45 if len(order) > 12 else 0,
                   automargin=True),
        yaxis=dict(title="EPA", zeroline=False, gridcolor="rgba(128,128,128,0.18)"),
    )
    return fig


def category_table(segments: pd.DataFrame, group_by: str = GROUP_PLAYER_GAME) -> pd.DataFrame:
    """Per-unit EPA by category (the table view of the chart)."""
    if segments.empty:
        return pd.DataFrame()
    seg = _unit_columns(segments, group_by)
    seg["unit_label"] = seg["unit_label"].str.replace("<br>", " · ", regex=False)
    table = seg.pivot_table(index="unit_label", columns="category", values="epa", aggfunc="sum", fill_value=0.0)
    table = table.reindex(columns=[c for c in CATEGORIES if c in table.columns])
    plays = seg.drop_duplicates(["unit", "game_id", "play_id"]).groupby("unit_label").size()
    tds = seg[seg["td"] == "offense"].drop_duplicates(["unit", "game_id", "play_id"]).groupby("unit_label").size()
    table.insert(0, "Targets", plays)
    table["TDs"] = tds.reindex(table.index).fillna(0).astype(int)
    table["Net EPA"] = table[[c for c in CATEGORIES if c in table.columns]].sum(axis=1)
    table.index.name = "Player"
    table.columns.name = None
    return table.sort_values("Net EPA", ascending=False).round(2)


def filter_min_targets(segments: pd.DataFrame, group_by: str, min_targets: int) -> pd.DataFrame:
    """Drop units (player-games or players) with fewer than ``min_targets`` targets."""
    if segments.empty or min_targets <= 1:
        return segments
    units = _unit_columns(segments, group_by)["unit"]
    plays = segments.assign(_unit=units.values).drop_duplicates(["_unit", "game_id", "play_id"])
    counts = plays.groupby("_unit").size()
    keep = units.map(counts).fillna(0).values >= min_targets
    return segments[keep]


def select_team_games(segments: pd.DataFrame, game_team_pairs: Iterable[Sequence[str]]) -> pd.DataFrame:
    pairs = pd.DataFrame(list(game_team_pairs), columns=["game_id", "team"])
    return segments.merge(pairs, on=["game_id", "team"])


def select_player_games(segments: pd.DataFrame, player_game_pairs: Iterable[Sequence[str]]) -> pd.DataFrame:
    pairs = pd.DataFrame(list(player_game_pairs), columns=["player_id", "game_id"])
    return segments.merge(pairs, on=["player_id", "game_id"])
