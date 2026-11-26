"""Proper scores for backtest runs, compared game by game.

Per game: Brier score of the home win probability, CRPS of the spread (home minus away) and of the total, and the
energy score of the two teams' points together (energy_points), which also rewards getting their dependence right.
Per team-game: energy score and variogram score (order 1/2) of the joint forecast of the team's stat line: points,
rushing attempts and yards, the QB1 passing line, and the RB1, WR1, WR2 and TE1 lines (nflsim.sim.joint_outcomes picks
the slots and the actual stats). Every stat is divided by one fixed scale, its standard deviation across all team-games
in the comparison, so the multivariate scores weigh stats evenly and every run is scored on the same scale. Stats with no
actual value (a slot player who did not play) are left out of that team-game. All scores are lower-is-better.

Runs are compared on the games they share; a difference's standard error treats each game as one sample. A calibration
table (forecast_calibration.csv) adds each run's simulated spread and total SD against its errors, mean squared z, and
the simulated against the observed home/away score correlation.

    python -m nflsim.sim.forecast_scores --season 2026 --run live=<run dir> --run pruned=<run dir> [--output <dir>]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nflsim.features.bayesian_history import load_pbp_data
from nflsim.paths import output_dir
from nflsim.sim.joint_outcomes import DRIVE_STATS, observed, simulated
from nflsim.sim.proper_scores import crps, energy_score, variogram_score

SCORES = ["brier", "crps_spread", "crps_total", "energy_points", "energy", "variogram"]


def game_scores(run: Path) -> pd.DataFrame:
    results = pd.read_csv(run / "backtest_2025_results.csv")
    rows = []
    for record in results.to_dict("records"):
        path = run / "raw_games" / f"{record['game_id']}.csv"
        if not path.is_file():
            continue
        sims = pd.read_csv(path)
        spread = (sims.home - sims.away).to_numpy(float)
        total = (sims.home + sims.away).to_numpy(float)
        actual_spread = record["actual_home_score"] - record["actual_away_score"]
        home_win = np.mean(spread > 0) + 0.5 * np.mean(spread == 0)
        outcome = 1.0 if actual_spread > 0 else 0.0 if actual_spread < 0 else 0.5
        rows.append({"game_id": record["game_id"], "week": record["week"], "brier": float((home_win - outcome) ** 2),
                     "crps_spread": crps(spread, actual_spread),
                     "crps_total": crps(total, record["actual_home_score"] + record["actual_away_score"]),
                     # Both teams' points together, so the home/away dependence counts (unlike spread and total alone).
                     "energy_points": energy_score(sims[["home", "away"]].to_numpy(float),
                                                   np.array([record["actual_home_score"], record["actual_away_score"]], float)),
                     # Moments for the calibration table (not scores).
                     "actual_home": float(record["actual_home_score"]), "actual_away": float(record["actual_away_score"]),
                     "sim_home_mean": float(sims.home.mean()), "sim_away_mean": float(sims.away.mean()),
                     "sim_spread_sd": float(spread.std(ddof=1)), "sim_total_sd": float(total.std(ddof=1)),
                     "sim_home_away_corr": float(np.corrcoef(sims.home, sims.away)[0, 1])
                     if sims.home.std() > 0 and sims.away.std() > 0 else np.nan})
    return pd.DataFrame(rows)


def calibration(per_game: pd.DataFrame) -> pd.DataFrame:
    """Per run: is the spread of the simulated scores as large as the run's own errors, and do the two teams' scores
    move together as much as real ones do?

    A calibrated run has sim SD close to error SD (RMSE of the sim mean) and a mean squared z near 1; above 1 the
    simulated scores are too narrow. ``sim_home_away_corr`` averages the within-game correlation of the simulated
    scores; ``observed_home_away_corr`` correlates the two teams' errors (actual minus sim mean) across games.
    """
    rows = []
    for label, frame in per_game.groupby("label", sort=False):
        spread_error = (frame.actual_home - frame.actual_away) - (frame.sim_home_mean - frame.sim_away_mean)
        total_error = (frame.actual_home + frame.actual_away) - (frame.sim_home_mean + frame.sim_away_mean)
        rows.append({
            "label": label, "games": len(frame),
            "spread_bias": float(spread_error.mean()), "spread_rmse": float(np.sqrt((spread_error ** 2).mean())),
            "spread_sim_sd": float(np.sqrt((frame.sim_spread_sd ** 2).mean())),
            "spread_mean_sq_z": float(((spread_error / frame.sim_spread_sd) ** 2).mean()),
            "total_bias": float(total_error.mean()), "total_rmse": float(np.sqrt((total_error ** 2).mean())),
            "total_sim_sd": float(np.sqrt((frame.sim_total_sd ** 2).mean())),
            "total_mean_sq_z": float(((total_error / frame.sim_total_sd) ** 2).mean()),
            "sim_home_away_corr": float(frame.sim_home_away_corr.mean()),
            "observed_home_away_corr": float(np.corrcoef(frame.actual_home - frame.sim_home_mean,
                                                         frame.actual_away - frame.sim_away_mean)[0, 1]),
        })
    return pd.DataFrame(rows)


def team_line_scores(sims: pd.DataFrame, actual: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    stats = [column for column in scale.index]
    groups = dict(tuple(sims.groupby(["game_id", "team"])))
    rows = []
    for record in actual.to_dict("records"):
        draws = groups.get((record["game_id"], record["team"]))
        if draws is None:
            continue
        present = [stat for stat in stats if pd.notna(record.get(stat)) and stat in draws]
        if len(present) < 2:
            continue
        x = draws[present].to_numpy(float) / scale[present].to_numpy()
        y = np.array([record[stat] for stat in present], float) / scale[present].to_numpy()
        rows.append({"game_id": record["game_id"], "team": record["team"], "stats": len(present),
                     "energy": energy_score(x, y), "variogram": variogram_score(x, y)})
    return pd.DataFrame(rows)


def evaluate(runs: dict[str, Path], season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pbp = load_pbp_data([season])
    built = {}
    for label, run in runs.items():
        sims, slots = simulated(run)
        built[label] = (sims, observed(run, slots, pbp))
    # Drive stats are calibration targets in joint_outcomes, not part of the stat line (older runs lack them).
    stats = [column for column in built[next(iter(built))][1] if "." in column
             and not column.startswith("opp.") and column != "team.win" and column not in DRIVE_STATS]
    actual_all = pd.concat([actual for _, actual in built.values()])
    scale = actual_all[stats].std().replace(0, np.nan).dropna()
    per_game = []
    for label, run in runs.items():
        sims, actual = built[label]
        games = game_scores(run)
        lines = team_line_scores(sims, actual, scale).groupby("game_id")[["energy", "variogram"]].sum().reset_index()
        per_game.append(games.merge(lines, on="game_id").assign(label=label))
    per_game = pd.concat(per_game, ignore_index=True)
    shared = set.intersection(*(set(frame.game_id) for _, frame in per_game.groupby("label")))
    per_game = per_game[per_game.game_id.isin(shared)]
    reference = next(iter(runs))
    base = per_game[per_game.label.eq(reference)].set_index("game_id")[SCORES]
    rows = []
    for label, frame in per_game.groupby("label", sort=False):
        frame = frame.set_index("game_id")[SCORES]
        diff = frame - base.loc[frame.index]
        for score in SCORES:
            rows.append({"label": label, "score": score, "games": len(frame), "mean": frame[score].mean(),
                         f"minus_{reference}": diff[score].mean(),
                         "se": diff[score].std(ddof=1) / np.sqrt(len(diff)) if label != reference else np.nan})
    summary = pd.DataFrame(rows)
    summary["z"] = summary[f"minus_{reference}"] / summary.se
    return per_game, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, help="label=run_dir; the first is the reference")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = dict(spec.split("=", 1) for spec in args.run)
    runs = {label: Path(path) for label, path in runs.items()}
    per_game, summary = evaluate(runs, args.season)
    if args.output:
        output = output_dir(args.output)
        per_game.to_csv(output / "forecast_scores_by_game.csv", index=False)
        summary.to_csv(output / "forecast_scores.csv", index=False)
        calibration(per_game).to_csv(output / "forecast_calibration.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(summary.pivot(index="score", columns="label", values="mean")[list(runs)])
        print(calibration(per_game).set_index("label").T)
        print(summary[summary.label.ne(next(iter(runs)))].to_string(index=False))


if __name__ == "__main__":
    main()
