"""Pass-rush pressure as ridge terms and as context for passer targets.

Two weekly-updated measures, both per dropback of the offense:

- ``hit``: nflverse play-by-play ``qb_hit`` (every play since 2016; sacks are nearly all hits).
- ``pfr``: Pro Football Reference ``times_pressured`` summed over the offense's passers in the
  team-game (nflverse ``pfr_advstats`` weekly pass files, 2018+), divided by the offense's dropbacks
  and spread evenly over those dropbacks. Charted pressure (``was_pressure``) is not used: nflverse
  publishes it only after the season.

Each measure is a team-scope ridge target on dropbacks: the subject coefficient is the offense's
pressure allowed and the defense coefficient the defense's pressure rate. Its pregame reconstruction
(offense + defense + venue/home/rest, fit on strictly earlier weeks) is the game's pressure forecast.
Passer targets take pressure as a covariate: fit on the pressure the offense actually faced, predict
with the forecast (``build_design(pressure=...)``), so subject estimates are net of pressure faced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nflsim.features.builder import canonicalize_teams
from nflsim.ridge.metrics import (
    RidgeConfig, RidgeTarget, actual_output_frame, aggregate_output_predictions, build_design, build_target_rows,
    rolling_predictions,
)

MEASURES = ("hit", "pfr")
PFR_FIRST_SEASON = 2018
PRESSURE_CONFIG = RidgeConfig(within_decay=0.97, season_carryover=0.5, lookback_seasons=1,
                              lambda_subject=300.0, lambda_defense=300.0)
PRESSURE_RECENCY_GRID = {"within_decay": (0.9, 0.97, 1.0), "season_carryover": (0.25, 0.5, 0.75)}
PRESSURE_PENALTY_GRID = {"lambda_subject": (100.0, 300.0, 1000.0), "lambda_defense": (100.0, 300.0, 1000.0)}


def _dropback(pbp: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(pbp["qb_dropback"], errors="coerce").eq(1)


PRESSURE_TARGETS = {
    measure: RidgeTarget(f"pressure_{measure}", "normal", "team", "posteam", f"pressure_{measure}", _dropback,
                         f"pressure_allowed_{measure}", f"pressure_rate_{measure}")
    for measure in MEASURES
}


def load_pfr_pressures(pfr_dir: Path) -> pd.DataFrame:
    """Offense team-game ``times_pressured`` from the weekly PFR pass files (game_id, team, pressures)."""
    files = sorted(Path(pfr_dir).glob("advstats_week_pass_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No advstats_week_pass_*.parquet in {pfr_dir}")
    passing = canonicalize_teams(pd.concat([pd.read_parquet(file) for file in files], ignore_index=True))
    return (passing.groupby(["game_id", "team"])["times_pressured"].sum(min_count=1)
            .rename("pressures").reset_index().astype({"game_id": str, "team": str}))


def add_pressure_columns(pbp: pd.DataFrame, pfr_dir: Path | None) -> pd.DataFrame:
    """``pressure_hit`` and ``pressure_pfr`` on dropbacks (NaN elsewhere and for games PFR has not charted)."""
    dropback = _dropback(pbp)
    hit = pd.to_numeric(pbp["qb_hit"], errors="coerce").where(dropback)
    pbp = pbp.assign(pressure_hit=hit, pressure_pfr=np.nan)
    if pfr_dir is None:
        return pbp
    keys = pd.DataFrame({"game_id": pbp["game_id"].astype(str), "team": pbp["posteam"]})
    keys = canonicalize_teams(keys).astype({"game_id": str})
    dropbacks = dropback.groupby([keys["game_id"], keys["team"]]).transform("sum")
    pressures = keys.merge(load_pfr_pressures(pfr_dir), on=["game_id", "team"], how="left")["pressures"].to_numpy()
    rate = np.clip(pressures / dropbacks.to_numpy(dtype=float), 0.0, 1.0)
    pbp["pressure_pfr"] = np.where(dropback, rate, np.nan)
    return pbp


def team_game_pressure(pbp: pd.DataFrame, games: pd.DataFrame, measure: str,
                       config: RidgeConfig = PRESSURE_CONFIG) -> pd.DataFrame:
    """Per offense team-game: realized pressure rate faced and its pregame ridge forecast.

    Columns: game_id, posteam, season, pressure (realized), pressure_forecast.
    """
    target = PRESSURE_TARGETS[measure]
    rows = build_target_rows(pbp, target, games)
    design = build_design(rows, "team")
    seasons = sorted(rows["season"].unique())
    predictions = rolling_predictions(rows, design, config, seasons, isolated=False)
    forecast = aggregate_output_predictions(rows, predictions, "subject").rename(
        columns={"entity": "posteam", "pred_full": "pressure_forecast"})
    realized = actual_output_frame(rows, "subject").rename(columns={"entity": "posteam", "rate": "pressure"})
    frame = realized[["game_id", "posteam", "season", "pressure"]].merge(
        forecast[["game_id", "posteam", "pressure_forecast"]], on=["game_id", "posteam"], how="left")
    return frame


def attach_pressure(rows: pd.DataFrame, pressure: pd.DataFrame) -> pd.DataFrame:
    """Add the offense's realized and forecast pressure for each (game_id, posteam) target row."""
    columns = ["pressure", "pressure_forecast"]
    rows = rows.drop(columns=[column for column in columns if column in rows])
    return rows.merge(pressure[["game_id", "posteam", *columns]], on=["game_id", "posteam"], how="left")
