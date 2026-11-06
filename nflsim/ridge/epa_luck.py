"""Luck-adjusted play EPA for the team EPA ridge targets.

Raw EPA carries outcomes that do not repeat from week to week:

- Fumble recoveries are close to a coin flip, and a returned fumble can swing a single play by
  more than a touchdown. Every fumble play is charged the league-average EPA of fumbles of the
  same play kind (pass/rush) from the same 10-yard field-position bucket, so the fumble is still
  penalized but the recovery and the return are not.
- Interception returns: interceptions are charged the league-average interception EPA for the
  field-position bucket, so a pick-six counts the same as a pick at the spot.
- Single-play extremes are clipped to the league's 0.5% / 99.5% EPA quantiles.
- Optionally, garbage time (win probability outside [garbage_wp, 1 - garbage_wp] at the snap) is
  dropped.

League averages and quantiles for season S come only from seasons before S (the earliest
season uses itself), so nothing is learned from the games being predicted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nflsim.ridge.metrics import _numeric

LUCK_COLUMN = "epa_luck"
LUCK_GARBAGE_COLUMN = "epa_luck_gt"
CLIP_QUANTILE = 0.005
GARBAGE_WP = 0.05


def _seasons(pbp: pd.DataFrame) -> pd.Series:
    return pbp["game_id"].astype(str).str[:4].astype(int)


def luck_adjusted_epa(pbp: pd.DataFrame, garbage_wp: float | None = None) -> pd.Series:
    epa = _numeric(pbp, "epa")
    seasons = _seasons(pbp)
    bucket = (_numeric(pbp, "yardline_100").clip(0, 99) // 10).fillna(5)
    kind = np.where(_numeric(pbp, "pass").eq(1), "pass", np.where(_numeric(pbp, "rush").eq(1), "rush", "other"))
    event = pd.Series(
        np.where(_numeric(pbp, "interception").eq(1), "int", np.where(_numeric(pbp, "fumble").eq(1), "fumble", "")),
        index=pbp.index,
    )
    keys = pd.DataFrame({"event": event, "kind": kind, "bucket": bucket, "epa": epa})
    adjusted = epa.copy()
    for season in sorted(seasons.unique()):
        current = seasons.eq(season)
        history = seasons.lt(season) if seasons.lt(season).any() else current
        low, high = epa[history].quantile([CLIP_QUANTILE, 1 - CLIP_QUANTILE])
        turnovers = keys[history & event.ne("")]
        means = turnovers.groupby(["event", "kind", "bucket"])["epa"].mean()
        fallback = turnovers.groupby("event")["epa"].mean()
        rows = current & event.ne("")
        index = pd.MultiIndex.from_frame(keys.loc[rows, ["event", "kind", "bucket"]])
        replaced = pd.Series(means.reindex(index).to_numpy(), index=keys.index[rows])
        replaced = replaced.fillna(event[rows].map(fallback))
        adjusted[rows] = replaced
        adjusted[current] = adjusted[current].clip(low, high)
    if garbage_wp is not None:
        wp = _numeric(pbp, "wp")
        adjusted[wp.lt(garbage_wp) | wp.gt(1 - garbage_wp)] = np.nan
    return adjusted


def add_luck_adjusted_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    """Add ``epa_luck`` (turnovers neutralized, extremes clipped) and ``epa_luck_gt`` (also drops garbage time)."""
    pbp = pbp.copy()
    pbp[LUCK_COLUMN] = luck_adjusted_epa(pbp)
    pbp[LUCK_GARBAGE_COLUMN] = luck_adjusted_epa(pbp, garbage_wp=GARBAGE_WP)
    return pbp
