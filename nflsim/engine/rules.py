"""Field-position and play-call rules the simulator applies between model draws.

Yardlines follow the simulator's convention: the offense's distance from its own goal line (0-100), so the
opponent's goal line is 100. A "start" is the receiving team's own yardline after a change of possession.
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np

TOUCHBACK_START = 20
# The holder sets the ball about seven yards behind the line of scrimmage; the posts stand ten yards past the goal line.
KICK_SPOT_BEHIND_LINE = 7
GOAL_POSTS_PAST_GOAL_LINE = 10
# Longer than any NFL field goal attempt; the play model is only overruled beyond this.
MAX_FIELD_GOAL_DISTANCE = 70

PLAY_TYPES = ("kick", "pass", "punt", "run")
MODEL_PLAY_CALLS_ENV = "NFLSIM_MODEL_PLAY_CALLS"


def field_goal_distance(yardline: float) -> float:
    return (100 - yardline) + KICK_SPOT_BEHIND_LINE + GOAL_POSTS_PAST_GOAL_LINE


def missed_field_goal_start(yardline: float) -> float:
    """The defense takes over at the spot of the kick, or at its 20 if that is farther from its goal line."""
    return max(float(TOUCHBACK_START), 100.0 - (yardline - KICK_SPOT_BEHIND_LINE))


def end_zone_turnover_start(start: float) -> float:
    """A turnover caught or recovered in the defense's own end zone and not returned past its 20 is a touchback."""
    return max(float(start), float(TOUCHBACK_START))


def model_play_calls() -> bool:
    """With NFLSIM_MODEL_PLAY_CALLS=1 the play model's kick and punt choices stand (an experiment).

    Off by default: the simulator keeps its older overrides, which turn early-down kicks and punts into passes and
    runs (removing end-of-half field goals) and force a field goal on 4th-and-2 or longer within 54 yards.
    """
    return os.environ.get(MODEL_PLAY_CALLS_ENV) == "1"


def impossible_play_call(play_type: str, down: int, yardline: float) -> bool:
    """Calls the simulator refuses even when it otherwise follows the play model."""
    if play_type == "kick":
        return field_goal_distance(yardline) > MAX_FIELD_GOAL_DISTANCE
    return False


def without_play_type(probabilities: Sequence[float], play_type: str) -> np.ndarray:
    """The play-call probabilities with one call removed and the rest renormalized (order as PLAY_TYPES)."""
    probabilities = np.asarray(probabilities, dtype=float).copy()
    probabilities[PLAY_TYPES.index(play_type)] = 0.0
    total = probabilities.sum()
    if total <= 0:
        raise ValueError(f"No play call left after removing {play_type}")
    return probabilities / total
