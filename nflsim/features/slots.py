"""Skill-slot assignment shared by training, backtests and the dashboard.

Eligible players are the team-game's QB/RB/WR/TE roster rows that are on the active roster (status
ACT, when the roster reports any) and not listed Out. QBs are ordered by depth chart, RBs by
carry share (rush_rate) and WRs/TEs by medium-depth target share; ties go to the depth chart, then
player id.

A slot with no eligible player is empty: training gives it zero usage shares (and typical values
for the other stats), and the simulator gives its placeholder player the same zero shares.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SLOT_LIMITS = {"QB": 1, "WR": 4, "RB": 3, "TE": 3}
RANK_STAT = {"RB": "rush_rate", "WR": "medium", "TE": "medium"}
USAGE_COLUMNS = ("rush_rate", "deep", "medium", "short")
EMPTY_PREFIXES = ("placeholder_", "ridge_placeholder_")


def is_empty_slot(player_id) -> bool:
    return str(player_id).startswith(EMPTY_PREFIXES)


def eligible(roster: pd.DataFrame, keys=("game_id", "team")) -> pd.DataFrame:
    keys = list(keys)
    rows = roster[roster.position.replace({"FB": "RB"}).isin(SLOT_LIMITS)].copy()
    rows["position"] = rows.position.replace({"FB": "RB"})
    if "report_status" in rows:
        rows = rows[~rows.report_status.eq("Out")]
    if "status" in rows:
        active = rows.status.eq("ACT")
        any_active = active.groupby([rows[key] for key in keys]).transform("any")
        rows = rows[active | ~any_active]
    return rows


def rank_slots(roster: pd.DataFrame, keys=("game_id", "team")) -> pd.DataFrame:
    """Return the slotted rows of `roster` with a `slot` column (QB1, WR1..WR4, RB1..RB3, TE1..TE3)."""
    keys = list(keys)
    rows = eligible(roster, keys)
    depth = pd.to_numeric(rows.get("depth_team", pd.Series(np.nan, index=rows.index)), errors="coerce").fillna(99)
    score = pd.Series(-depth.to_numpy(), index=rows.index, dtype=float)
    for position, column in RANK_STAT.items():
        mask = rows.position.eq(position)
        values = pd.to_numeric(rows[column], errors="coerce") if column in rows else pd.Series(np.nan, index=rows.index)
        score[mask] = values[mask].fillna(-1.0)
    rows = rows.assign(_score=score, _depth=depth, player_id=rows.player_id.astype(str))
    rows = rows.sort_values([*keys, "position", "_score", "_depth", "player_id"], ascending=[*[True]*len(keys), True, False, True, True])
    rows["rank"] = rows.groupby([*keys, "position"]).cumcount() + 1
    rows = rows[rows["rank"] <= rows.position.map(SLOT_LIMITS)]
    rows["slot"] = rows.position + rows["rank"].astype(str)
    return rows.drop(columns=["_score", "_depth"])


def depth_ranks(depth: pd.DataFrame, team: str, kickoff: pd.Timestamp) -> dict:
    """player_id -> pos_rank from the team's latest depth-chart snapshot on or before kickoff day."""
    if depth.empty or "dt" not in depth:
        return {}
    rows = depth[depth.team.astype(str).eq(str(team))].copy()
    if "pos_abb" in rows:
        rows = rows[rows.pos_abb.isin(["QB", "RB", "FB", "WR", "TE"])]
    rows["dt"] = pd.to_datetime(rows.dt, errors="coerce", utc=True).dt.tz_convert(None)
    rows = rows[rows.dt.le(pd.Timestamp(kickoff) + pd.Timedelta(days=1))]
    if rows.empty:
        return {}
    rows = rows[rows.dt.eq(rows.dt.max())]
    rank_column = next((name for name in ("pos_rank", "depth_chart_position") if name in rows), None)
    if rank_column is None:
        return {}
    rows["rank"] = pd.to_numeric(rows[rank_column], errors="coerce")
    rows = rows.dropna(subset=["gsis_id"]).sort_values("rank").drop_duplicates("gsis_id")
    return dict(zip(rows.gsis_id.astype(str), rows["rank"]))
