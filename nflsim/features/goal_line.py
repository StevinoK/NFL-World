"""Yardage that the goal line cuts short.

The play-by-play stops air yards, yards after catch and run yards at the goal line, so a 15-yard throw from the 5 is
recorded as 5. Those rows say the play went at least that far, not exactly that far, and the yardage models train on
them that way. The sim draws a full-length yardage and cuts it at the goal line, which matches.

The goal-line inputs repeat the distance on a closer scale. Over the whole field the 3 and the 5 are only .08 standard
deviations apart, too close for a network to separate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GOAL_CAPPED_MODELS = frozenset({"ay", "yac", "rush", "scramble_yards"})
GOAL_LINE_COLUMNS = ("goal_log", "goal_3", "goal_5", "goal_10", "goal_20")


def goal_line_inputs(yards) -> dict:
    """Goal-line inputs for the yards still available to the play; a scalar or an array."""
    yards = np.maximum(yards, 0)
    return {"goal_log": np.log1p(yards), **{f"goal_{limit}": np.minimum(yards, limit) for limit in (3, 5, 10, 20)}}


def yards_available(model: str, frame: pd.DataFrame) -> np.ndarray:
    """Yards this model's outcome can reach before the goal line: yards after catch start where the throw ended."""
    yards = pd.to_numeric(frame["yardline_100"]).to_numpy(dtype=float)
    return yards - pd.to_numeric(frame["air_yards"]).to_numpy(dtype=float) if model == "yac" else yards


def with_goal_line_inputs(model: str, frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(**goal_line_inputs(yards_available(model, frame)))
