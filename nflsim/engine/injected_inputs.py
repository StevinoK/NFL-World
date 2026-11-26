"""Inputs a refit can add to a play model without a new column group in the simulator.

A model trained with extra columns (``fit_minimal --added-inputs``) expects them in its preprocessor, but the
simulator builds each model's row from fixed column groups. Just before a model call, the runtime fills every
expected column that the row lacks and that this module knows how to supply:

- the play's situation, published by the game before each snap (quarter, seconds left in the half, both teams'
  timeouts, whether the offense gets the second-half kickoff, the score margin scaled by elapsed time);
- values derived from the row itself (air yards past the line to gain, a throw at the line of scrimmage);
- pregame team columns read from the offense's attributes (weather, and the tables in
  ``nflsim.features.pregame_tables``).

Training computes the same columns from play-by-play (``add_training_columns``), so a model sees each input
defined one way in both places.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from nflsim.features.weather import WEATHER_COLUMNS

TIMEOUT_COLUMNS = ("posteam_timeouts_remaining", "defteam_timeouts_remaining")
SITUATION_COLUMNS = ("qtr", "half_seconds_remaining", "receive_2h_ko", "diff_time_ratio", *TIMEOUT_COLUMNS)
DERIVED_COLUMNS = ("distance_to_sticks", "air_is_zero")
# Pregame team columns the offense's attributes supply (extended by nflsim.features.pregame_tables).
ATTRIBUTE_COLUMNS: set[str] = set(WEATHER_COLUMNS)


def injectable(columns) -> list[str]:
    known = set(SITUATION_COLUMNS) | set(DERIVED_COLUMNS) | ATTRIBUTE_COLUMNS
    return [column for column in columns if column in known]


def diff_time_ratio(margin: float, game_seconds_remaining: float) -> float:
    """nflfastR's Diff_Time_Ratio: the margin grows in weight as the game runs out."""
    elapsed = min(max((3600.0 - float(game_seconds_remaining)) / 3600.0, 0.0), 1.0)
    return float(margin) * math.exp(4.0 * elapsed)


def derived(column: str, row) -> float | None:
    air = row.get("air_yards")
    if air is None or air != air:
        return None
    if column == "air_is_zero":
        return float(float(air) == 0.0)
    if column == "distance_to_sticks":
        distance = row.get("ydstogo")
        return None if distance is None or distance != distance else float(air) - float(distance)
    return None


def fill(columns, row, state: dict) -> dict:
    """Values for ``columns`` missing from ``row``: from the row (derived), the published situation, then the
    offense's pregame attributes. Columns nothing supplies are left out (the model call then fails on them)."""
    values = {}
    attributes = state.get("_attributes")
    for column in columns:
        if column in row:
            continue
        if column in DERIVED_COLUMNS:
            value = derived(column, row)
        elif column in SITUATION_COLUMNS:
            value = state.get(column)
        else:
            value = attributes.get(column) if attributes is not None else None
        if value is not None:
            values[column] = value
    return values


def add_training_columns(pbp: pd.DataFrame) -> pd.DataFrame:
    """The added inputs that come from play-by-play, defined as the simulator defines them."""
    pbp = pbp.copy()
    if {"score_differential", "game_seconds_remaining"} <= set(pbp):
        elapsed = ((3600.0 - pbp.game_seconds_remaining.astype(float)) / 3600.0).clip(0.0, 1.0)
        pbp["diff_time_ratio"] = pbp.score_differential.astype(float) * np.exp(4.0 * elapsed)
    if {"home_opening_kickoff", "posteam", "home_team", "away_team", "qtr"} <= set(pbp):
        second_half = np.where(pbp.home_opening_kickoff.eq(1), pbp.away_team, pbp.home_team)
        pbp["receive_2h_ko"] = (pbp.qtr.le(2) & pbp.posteam.eq(second_half)).astype(float)
    if {"air_yards", "ydstogo"} <= set(pbp):
        pbp["distance_to_sticks"] = pbp.air_yards.astype(float) - pbp.ydstogo.astype(float)
        pbp["air_is_zero"] = pbp.air_yards.astype(float).eq(0.0).astype(float).where(pbp.air_yards.notna())
    return pbp
