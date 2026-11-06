"""Check that batched simulation matches sequential simulation on a fixed test set of games.

Three arms are run on the same games:

* A: sequential, seeds from run id "equivalence-a"
* B: batched, the same seeds as A. Each simulation draws the same random numbers, so outcomes should agree
  simulation by simulation; only last-bit differences in batched matrix products can make them diverge.
* C: batched, independent seeds ("equivalence-c"). Its distributions are compared with A's by Welch t-tests on
  means and two-sample Kolmogorov-Smirnov tests, per game and metric. If batching changes nothing, the p-values
  are uniform: about 5% fall below 0.05.

    python -m nflsim.sim.batch_equivalence --simulations 400 --parallelism 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from nflsim.features.game_inputs import build_real_game
from nflsim.paths import ROOT
from nflsim.sim.execution import execute_ridge_games
from nflsim.sim.synthetic_games import _load_inputs, build_synthetic_game, synthetic_game_row

# Outdoor, dome and neutral-site games, plus a synthetic game whose home and away teams swap halfway.
TEST_GAMES = ("2026_09_JAX_BAL", "2026_09_LV_SF", "2026_09_CLE_NO", "2026_09_BUF_MIN", "2026_09_CIN_ATL", "2026_10_NE_DET")
SYNTHETIC = ("KC", 9)
TEAM_STATS = ("passing_yards", "rushing_yards", "receiving_yards", "receptions", "pass_attempts", "completions",
              "rushing_attempts", "interceptions", "sacks", "fumble_lost", "passing_touchdowns", "rushing_touchdowns")


def test_games(runtime, artifact_dir: Path, season: int, simulations: int):
    schedule, roster, depth = _load_inputs(season, artifact_dir / "weekly_inputs")
    games = []
    for game_id in TEST_GAMES:
        row = schedule[schedule.game_id.eq(game_id)].iloc[0]
        games.append(build_real_game(runtime, runtime.player_stats_wide, runtime.team_stats_wide, roster, depth,
                                     schedule, row, simulations))
    team, week = SYNTHETIC
    games.append(build_synthetic_game(runtime, runtime.player_stats_wide, runtime.team_stats_wide, roster, depth,
                                      schedule, team, synthetic_game_row(schedule, team, week), simulations))
    return games


def outcomes(results) -> pd.DataFrame:
    """One row per game and simulation: scores plus each side's summed player stats."""
    rows = []
    for game_id, (games, players) in results.items():
        frame = games[["game_id", "simulation", "home_team", "away_team", "home_score", "away_score"]].copy()
        frame["total"] = frame.home_score + frame.away_score
        frame["spread"] = frame.home_score - frame.away_score
        frame["home_win"] = (frame.home_score > frame.away_score).astype(float)
        teams = players.groupby(["simulation", "team"])[list(TEAM_STATS)].sum()
        for side in ("home", "away"):
            keys = pd.MultiIndex.from_arrays([frame.simulation, frame[f"{side}_team"]])
            values = teams.reindex(keys).reset_index(drop=True).add_prefix(f"{side}_")
            frame = pd.concat([frame.reset_index(drop=True), values], axis=1)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def metrics(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in {"game_id", "simulation", "home_team", "away_team"}]


def paired_agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    merged = a.merge(b, on=["game_id", "simulation"], suffixes=("_a", "_b"))
    columns = metrics(a)
    same = np.all([np.isclose(merged[f"{column}_a"], merged[f"{column}_b"], equal_nan=True) for column in columns], axis=0)
    return {"simulations": int(len(merged)), "identical": int(same.sum()), "identical_share": float(same.mean()),
            "by_game": merged.assign(same=same).groupby("game_id").same.mean().round(4).to_dict()}


def distribution_tests(a: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for game_id in sorted(a.game_id.unique()):
        left, right = a[a.game_id.eq(game_id)], c[c.game_id.eq(game_id)]
        for column in metrics(a):
            x, y = left[column].to_numpy(float), right[column].to_numpy(float)
            if np.nanstd(x) == 0 and np.nanstd(y) == 0:
                continue
            welch = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
            ks = stats.ks_2samp(x[~np.isnan(x)], y[~np.isnan(y)])
            pooled = np.sqrt((np.nanvar(x, ddof=1) + np.nanvar(y, ddof=1)) / 2)
            rows.append({"game_id": game_id, "metric": column, "mean_sequential": np.nanmean(x), "mean_batched": np.nanmean(y),
                         "standardized_difference": (np.nanmean(y) - np.nanmean(x)) / pooled if pooled else 0.0,
                         "welch_p": float(welch.pvalue), "ks_p": float(ks.pvalue)})
    return pd.DataFrame(rows)


def summarize(tests: pd.DataFrame, paired: dict, timings: dict) -> dict:
    summary = {"paired_same_seeds": paired, "timings_seconds": timings, "tests": int(len(tests))}
    for column in ("welch_p", "ks_p"):
        p = tests[column].to_numpy()
        summary[column] = {
            "share_below_0.05": float((p < 0.05).mean()), "expected_share": 0.05,
            "min": float(p.min()),
            # Uniformity of all p-values; metrics within a game are correlated, so this is indicative only.
            "uniformity_ks_p": float(stats.kstest(p, "uniform").pvalue),
        }
    summary["max_abs_standardized_difference"] = float(tests.standardized_difference.abs().max())
    summary["mean_abs_standardized_difference"] = float(tests.standardized_difference.abs().mean())
    return summary


def main():
    import time

    from nflsim.bundle import validate_bundle, verify_bundle_hash
    from nflsim.sim.runtime import RidgeRuntime
    from nflsim.sim.season import latest_bundle

    parser = argparse.ArgumentParser(description="Compare batched with sequential simulation on test games")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--parallelism", type=int, default=6)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/batch_equivalence")
    args = parser.parse_args()
    artifact_dir = (args.artifact_dir or latest_bundle()).resolve()
    validate_bundle(artifact_dir)
    bundle_hash = verify_bundle_hash(artifact_dir)
    runtime = RidgeRuntime(artifact_dir, strict_schema=True)
    games = test_games(runtime, artifact_dir, args.season, args.simulations)
    arms, timings = {}, {}
    for name, run_id, batched in (("A", "equivalence-a", False), ("B", "equivalence-a", True), ("C", "equivalence-c", True)):
        start = time.perf_counter()
        results = execute_ridge_games(games, artifact_dir, bundle_hash, backend="local", parallelism=args.parallelism,
                                      run_id=run_id, batched=batched)
        timings[name] = round(time.perf_counter() - start, 1)
        arms[name] = outcomes(results)
        print(f"arm {name}: {timings[name]}s", flush=True)
    output = args.output / f"{bundle_hash[:12]}-sims{args.simulations}"
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in arms.items():
        frame.to_csv(output / f"arm_{name}.csv", index=False)
    tests = distribution_tests(arms["A"], arms["C"])
    tests.to_csv(output / "distribution_tests.csv", index=False)
    summary = summarize(tests, paired_agreement(arms["A"], arms["B"]), timings)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
