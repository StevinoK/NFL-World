"""Are the ridge posterior SDs of team EPA ratings the size of their real pregame errors?

For each week of the test seasons, fit the production pass/rush EPA ridge (config/ridge_parameters.json) on
strictly earlier games, predict every team-game, and compare the realized game rate with the posterior
predictive: rating uncertainty x Cov x' plus play noise sigma2 / plays. Prints var(z) (1 = calibrated) and the
scale on the rating variance that would make it 1.

    PBP_DIR=<folder of pbp_<season>.parquet> python docs/analysis/2026-11-accuracy/exp_strength_uncertainty.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from nflsim.features.builder import parse_config
from nflsim.ridge.metrics import (TARGETS_BY_NAME, _row_weights, build_design, build_game_context, build_target_rows,
                                  ridge_posterior)

PBP_DIR = Path(os.environ.get("PBP_DIR", "data/pbp"))
TEST_SEASONS = (2023, 2024, 2025)
PARAMETERS = json.loads(Path("nflsim/config/ridge_parameters.json").read_text())["parameters"]


def load(seasons):
    frames = [pd.read_parquet(PBP_DIR / f"pbp_{season}.parquet") for season in seasons]
    pbp = pd.concat(frames, ignore_index=True)
    return pbp[pbp.season_type.eq("REG")]


def evaluate(pbp, target_name, metric):
    target = TARGETS_BY_NAME[target_name]
    config = parse_config(PARAMETERS[metric])
    games = build_game_context(pbp)
    rows = build_target_rows(pbp, target, games)
    design = build_design(rows, target.scope)
    rate, counts = rows["rate"].to_numpy(float), rows["count"].to_numpy(float)
    out = []
    for season, week in sorted(set(zip(rows.season, rows.week))):
        if season not in TEST_SEASONS or week < 2:
            continue
        weights = _row_weights(rows, int(season), int(week), config)
        beta, covariance, sigma2 = ridge_posterior(design.matrix, rate, weights, design.penalty(config), counts)
        index = np.flatnonzero((rows.season == season) & (rows.week == week))
        x = design.matrix[index].toarray()
        offense = np.flatnonzero(design.blocks == "subject")
        defense = np.flatnonzero(design.blocks == "defense")
        for row, i in zip(x, index):
            o, d = offense[row[offense] > 0], defense[row[defense] > 0]
            out.append({"season": season, "week": week, "actual": rate[i], "plays": counts[i], "pred": row @ beta,
                        "rating_var": row @ covariance @ row, "noise_var": sigma2 / counts[i],
                        "offense_sd": float(np.sqrt(covariance[o, o].sum())) if len(o) else np.nan,
                        "defense_sd": float(np.sqrt(covariance[d, d].sum())) if len(d) else np.nan})
    frame = pd.DataFrame(out)
    error2 = (frame.actual - frame.pred) ** 2
    z = (frame.actual - frame.pred) / np.sqrt(frame.rating_var + frame.noise_var)
    scale = (error2.mean() - frame.noise_var.mean()) / frame.rating_var.mean()
    return {"target": target_name, "team_games": len(frame), "var_z": float(z.var()),
            "rating_share_of_variance": float(frame.rating_var.mean() / (frame.rating_var + frame.noise_var).mean()),
            "rating_variance_scale": float(scale), "offense_sd": float(frame.offense_sd.mean()),
            "defense_sd": float(frame.defense_sd.mean()), "rmse": float(np.sqrt(error2.mean())),
            "early_weeks_var_z": float(z[frame.week <= 6].var()), "late_weeks_var_z": float(z[frame.week > 6].var())}


if __name__ == "__main__":
    pbp = load(range(min(TEST_SEASONS) - 3, max(TEST_SEASONS) + 1))
    results = [evaluate(pbp, "pass_epa", "epa_rate"), evaluate(pbp, "rush_epa", "rushing_epa_rate")]
    print(pd.DataFrame(results).round(4).to_string(index=False))
