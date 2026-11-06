"""Fantasy scoring and player-attribute helpers shared by the simulation dashboard.

Scoring is ported from the inline block in `Torch Game Simulations.py` so that
dashboard output matches the legacy simulation loop.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

PLAYER_STAT_COLUMNS = [
    "passing_yards",
    "rushing_yards",
    "rushing_attempts",
    "receiving_yards",
    "passing_touchdowns",
    "receiving_touchdowns",
    "rushing_touchdowns",
    "interceptions",
    "receptions",
    "sacks",
    "punts",
    "punt_yards",
    "pat_attempts",
    "pat_made",
    "pass_attempts",
    "completions",
    "air_yards",
    "fumble_lost",
]

DST_STAT_COLUMNS = ["sack", "interception", "pa", "fumble_recovered", "defensive_touchdown"]
KICKER_STAT_COLUMNS = ["pat_made", "fg 0-39", "fg 40-49", "fg 50-59"]

POINTS_ALLOWED_TIERS = [
    (0, 10),
    (7, 7),
    (14, 4),
    (21, 1),
    (28, 0),
    (35, -1),
]
POINTS_ALLOWED_FLOOR = -4


def _numeric_columns(frame: pd.DataFrame, columns: Iterable[str]) -> dict[str, pd.Series]:
    zeros = pd.Series(0.0, index=frame.index)
    return {
        column: pd.to_numeric(frame[column], errors="coerce").fillna(0.0) if column in frame.columns else zeros
        for column in columns
    }


def skill_fantasy_points(frame: pd.DataFrame) -> pd.Series:
    """Full-PPR DraftKings scoring with yardage bonuses."""
    values = _numeric_columns(frame, PLAYER_STAT_COLUMNS)
    return (
        values["passing_yards"] * 0.04
        + (values["passing_yards"] > 300) * 3
        + values["rushing_yards"] * 0.1
        + (values["rushing_yards"] > 100) * 3
        + values["receiving_yards"] * 0.1
        + (values["receiving_yards"] > 100) * 3
        + values["passing_touchdowns"] * 4
        + values["receiving_touchdowns"] * 6
        + values["rushing_touchdowns"] * 6
        + values["interceptions"] * -1
        + values["receptions"] * 1
        + values["fumble_lost"] * -1
    )


def _points_allowed_score(points_allowed: pd.Series) -> pd.Series:
    conditions = [points_allowed == 0] + [points_allowed < bound for bound, _ in POINTS_ALLOWED_TIERS[1:]]
    choices = [score for _, score in POINTS_ALLOWED_TIERS]
    return pd.Series(np.select(conditions, choices, default=POINTS_ALLOWED_FLOOR), index=points_allowed.index)


def dst_fantasy_points(frame: pd.DataFrame) -> pd.Series:
    values = _numeric_columns(frame, DST_STAT_COLUMNS)
    return (
        values["defensive_touchdown"] * 6
        + values["sack"] * 1
        + values["interception"] * 2
        + values["fumble_recovered"] * 2
        + _points_allowed_score(values["pa"])
    )


def kicker_fantasy_points(frame: pd.DataFrame) -> pd.Series:
    values = _numeric_columns(frame, KICKER_STAT_COLUMNS)
    return values["pat_made"] * 1 + values["fg 0-39"] * 3 + values["fg 40-49"] * 4 + values["fg 50-59"] * 5


def add_fantasy_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Score a mixed frame of skill players, DST rows (position=='DST') and kicker rows (position=='K')."""
    result = frame.copy()
    if result.empty:
        result["fantasy_points"] = pd.Series(dtype=float)
        return result

    result["fantasy_points"] = skill_fantasy_points(result)
    dst_mask = result["position"] == "DST"
    if dst_mask.any():
        result.loc[dst_mask, "fantasy_points"] = dst_fantasy_points(result.loc[dst_mask])
    kicker_mask = result["position"] == "K"
    if kicker_mask.any():
        result.loc[kicker_mask, "fantasy_points"] = kicker_fantasy_points(result.loc[kicker_mask])
    return result


def team_unit_rows(team, simulation: int) -> list[dict[str, object]]:
    """Build the DST and kicker rows for one simulated game."""
    dst = {
        "simulation": simulation,
        "team": team.name,
        "player_name": f"{team.name} Defense",
        "player_id": "NA",
        "position": "DST",
    }
    dst.update({column: team.stats.get(column, 0) for column in DST_STAT_COLUMNS})

    kicker = {
        "simulation": simulation,
        "team": team.name,
        "player_name": f"{team.name} Kicker",
        "player_id": "NA",
        "position": "K",
    }
    kicker.update({column: team.stats.get(column, 0) for column in KICKER_STAT_COLUMNS})
    return [dst, kicker]


def fantasy_correlation_matrix(player_frame: pd.DataFrame) -> pd.DataFrame:
    """Correlation of fantasy points across players over the simulation set."""
    if player_frame.empty:
        return pd.DataFrame()
    matrix = player_frame.pivot_table(
        index="simulation",
        columns="player_name",
        values="fantasy_points",
        aggfunc="first",
    )
    return matrix.corr()


def prior_fill_values(player_stats_wide: pd.DataFrame, columns: Iterable[str]) -> dict[str, float]:
    """Value the Bayesian smoother emits for a player with no observations yet.

    Training saw this prior in every unused slot, and some slot columns are so close to
    constant there (scale_ ~5e-5) that substituting 0.0 yields z-scores in the thousands.
    """
    first_rows = player_stats_wide.sort_values("game_date").groupby("player_id").head(1)
    values: dict[str, float] = {}
    for column in columns:
        if column not in player_stats_wide.columns:
            values[column] = 0.0
            continue
        for source in (first_rows[column], player_stats_wide[column]):
            mode = source.mode()
            if not mode.empty:
                values[column] = float(mode.iloc[0])
                break
        else:
            values[column] = 0.0
    return values


def latest_player_row(
    player_stats_wide: pd.DataFrame,
    player_id: str,
    before_date: pd.Timestamp,
    feature_columns: Iterable[str],
    fill_values: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Pregame attribute row for a player, backfilled to the runtime schema.

    Rows are already pregame estimates (the smoother emits before folding in that game), and
    training paired each play with `merge_asof(direction="backward")` on game_date. So the row
    dated on kickoff is the right one; taking the last row strictly before it is a game stale.
    """
    feature_columns = list(feature_columns)
    fill_values = fill_values or {}
    defaults = pd.Series({column: float(fill_values.get(column, 0.0)) for column in feature_columns})

    matches = player_stats_wide[player_stats_wide["player_id"].astype(str) == str(player_id)]
    if not matches.empty and "game_date" in matches.columns:
        dates = pd.to_datetime(matches["game_date"], errors="coerce")
        matches = matches[dates <= pd.Timestamp(before_date)]
        matches = matches.assign(_order=dates.loc[matches.index]).sort_values("_order").drop(columns="_order")

    if matches.empty:
        row = pd.DataFrame([defaults.to_dict()])
    else:
        row = matches.tail(1).reset_index(drop=True)
        for column in feature_columns:
            if column not in row.columns:
                row[column] = np.nan
        row[feature_columns] = row[feature_columns].astype(float).fillna(defaults)

    row["player_id"] = str(player_id)
    return row
