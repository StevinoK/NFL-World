from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from nflsim.paths import REPO, ROOT, digest, output_dir, write_json
from nflsim.features.builder import canonicalize_teams
from nflsim.storage import json_value


def metrics(frame):
    if frame.empty:
        raise ValueError("No compatible games to compare")
    spread = (frame.sim_spread-frame.actual_spread).to_numpy(float)
    total = (frame.sim_total-frame.actual_total).to_numpy(float)
    result = {"brier": float(np.mean((frame.sim_home_win_prob-(frame.actual_spread > 0).astype(float))**2)), "spread_rmse": float(np.sqrt(np.mean(spread**2))), "spread_mae": float(np.mean(abs(spread))), "total_rmse": float(np.sqrt(np.mean(total**2))), "total_mae": float(np.mean(abs(total))), "spread_bias": float(np.mean(spread)), "total_bias": float(np.mean(total)), "home_score_bias": float((frame.sim_home_median-frame.actual_home_score).mean()), "away_score_bias": float((frame.sim_away_median-frame.actual_away_score).mean())}
    return result


def calibration(frame):
    data = frame.assign(probability_bin=pd.cut(frame.sim_home_win_prob, np.linspace(0, 1, 11), include_lowest=True).astype(str), actual_home_win=(frame.actual_spread > 0).astype(int))
    return data.groupby("probability_bin", observed=True).agg(games=("game_id", "count"), predicted=("sim_home_win_prob", "mean"), actual=("actual_home_win", "mean")).reset_index().to_dict("records")


def market_metrics(frame):
    result = {}
    for column, target in (("spread_line", "actual_spread"), ("total_line", "actual_total")):
        if column in frame:
            valid = frame[[column, target]].dropna()
            if len(valid):
                error = valid[column]-valid[target]
                result[column] = {"games": len(valid), "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(abs(error))), "bias": float(error.mean())}
    if {"home_moneyline", "away_moneyline"} <= set(frame):
        def implied(values):
            values = pd.to_numeric(values, errors="coerce")
            return np.where(values < 0, -values/(-values+100), 100/(values+100))
        home, away = implied(frame.home_moneyline), implied(frame.away_moneyline)
        probability = home/(home+away)
        valid = np.isfinite(probability)
        if valid.any():
            result["brier"] = float(np.mean((probability[valid]-(frame.actual_spread.to_numpy()[valid] > 0))**2))
    return result


def compare(ridge, baseline, bootstrap_samples=1000, seed=42):
    ridge, baseline = canonicalize_teams(ridge), canonicalize_teams(baseline)
    required = {"game_id", "week", "home_team", "away_team", "actual_home_score", "actual_away_score", "actual_total", "actual_spread", "sim_spread", "sim_total", "sim_home_win_prob", "sim_home_median", "sim_away_median", "simulations_run"}
    for label, frame in (("ridge", ridge), ("baseline", baseline)):
        if not required <= set(frame):
            raise ValueError(f"{label}: missing comparison columns {sorted(required-set(frame))}")
        if frame.game_id.duplicated().any():
            raise ValueError(f"{label}: duplicate games")
        numeric = frame[list(required-{"game_id", "home_team", "away_team"})].to_numpy(float)
        if not np.isfinite(numeric).all() or not frame.sim_home_win_prob.between(0, 1).all() or not frame.simulations_run.gt(0).all():
            raise ValueError(f"{label}: invalid numeric values")
        if not np.allclose(frame.actual_spread, frame.actual_home_score-frame.actual_away_score) or not np.allclose(frame.actual_total, frame.actual_home_score+frame.actual_away_score):
            raise ValueError(f"{label}: actual score sign/total mismatch")
    common = sorted(set(ridge.game_id) & set(baseline.game_id))
    ridge = ridge.set_index("game_id").loc[common].reset_index()
    baseline_common = baseline.set_index("game_id").loc[common].reset_index()
    for column in ("home_team", "away_team", "week", "actual_home_score", "actual_away_score", "actual_spread", "actual_total"):
        if not ridge[column].equals(baseline_common[column]) and not (pd.api.types.is_numeric_dtype(ridge[column]) and np.allclose(ridge[column], baseline_common[column])):
            raise ValueError(f"Incompatible paired games: {column}")
    for column in ("spread_line", "total_line", "home_moneyline", "away_moneyline"):
        if column in ridge and column in baseline_common and not np.allclose(ridge[column], baseline_common[column], equal_nan=True):
            raise ValueError(f"Incompatible market reference {column}")
    ridge_metrics, baseline_metrics = metrics(ridge), metrics(baseline_common)
    rng = np.random.default_rng(seed)
    deltas = {name: [] for name in ridge_metrics}
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, len(common), len(common))
        sample_ridge, sample_baseline = metrics(ridge.iloc[indices]), metrics(baseline_common.iloc[indices])
        for name in deltas:
            deltas[name].append(sample_ridge[name]-sample_baseline[name])
    comparison = pd.DataFrame([{"metric": name, "ridge": ridge_metrics[name], "baseline": baseline_metrics[name], "delta_ridge_minus_baseline": ridge_metrics[name]-baseline_metrics[name], "paired_ci_low": float(np.quantile(deltas[name], 0.025)) if deltas[name] else None, "paired_ci_high": float(np.quantile(deltas[name], 0.975)) if deltas[name] else None} for name in deltas])
    week_slices = []
    for week in sorted(ridge.week.unique()):
        mask = ridge.week.eq(week)
        left, right = metrics(ridge[mask]), metrics(baseline_common[mask])
        week_slices.append({"week": int(week), "games": int(mask.sum()), "ridge": left, "baseline": right, "delta": {name: left[name]-right[name] for name in left}})
    summary = {"common_games": len(common), "game_ids": common, "baseline_excluded": sorted(set(baseline.game_id)-set(common)), "ridge_metrics": ridge_metrics, "baseline_metrics": baseline_metrics, "ridge_simulations": ridge.simulations_run.astype(int).tolist(), "baseline_simulations": baseline_common.simulations_run.astype(int).tolist(), "unequal_simulation_counts": bool((ridge.simulations_run != baseline_common.simulations_run).any()), "calibration": {"ridge": calibration(ridge), "baseline": calibration(baseline_common)}, "by_week": week_slices, "market_common_games": market_metrics(baseline_common), "bootstrap_samples": bootstrap_samples, "seed": seed, "confounds": ["Ridge estimator and context feature set both changed", "Saved baseline training/source provenance is not established", "Training timestamp and runtime/platform versions may differ; not a causal comparison"], "optional_player_distribution_comparisons": "not computed; require compatible baseline raw simulations"}
    return comparison, summary


def qb_change_games(player_value: pd.DataFrame, threshold_sd: float = 1.0) -> set[str]:
    """Games where either team's QB PV moved more than ``threshold_sd`` league SDs from its previous game."""
    frame = canonicalize_teams(player_value).copy()
    frame["game_id"] = frame.game_id.astype(str)
    frame = frame.sort_values(["team", "game_id"])
    change = frame.groupby("team").pv_off_qb.diff().abs()
    scale = frame.pv_off_qb.std()
    return set(frame.loc[change > threshold_sd * scale, "game_id"])


def slice_metrics(ridge: pd.DataFrame, baseline: pd.DataFrame, games: set[str]) -> dict:
    left, right = ridge[ridge.game_id.isin(games)], baseline[baseline.game_id.isin(games)]
    common = sorted(set(left.game_id) & set(right.game_id))
    if not common:
        return {"games": 0}
    left, right = left.set_index("game_id").loc[common].reset_index(), right.set_index("game_id").loc[common].reset_index()
    a, b = metrics(left), metrics(right)
    return {"games": len(common), "ridge": a, "baseline": b, "delta": {name: a[name] - b[name] for name in a}}


def main():
    parser = argparse.ArgumentParser(description="Paired saved-baseline comparison; never modifies baseline")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, default=REPO / "archive/bayesian_sim/backtest_2025_results.csv")
    parser.add_argument("--baseline-summary", type=Path, default=REPO / "archive/bayesian_sim/backtest_2025_summary.json")
    parser.add_argument("--extra-baseline", action="append", default=[], metavar="LABEL=RESULTS_CSV",
                        help="Additional saved results to compare against, e.g. ridge=ridge_game_sim/benchmarks/2025/ridge-context-2025/backtest_2025_results.csv")
    parser.add_argument("--player-value", type=Path, default=None, help="player_value.csv for QB-change slices (default: run bundle)")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_samples < 0:
        parser.error("bootstrap samples cannot be negative")
    output = output_dir(args.run_dir)
    baseline_hash = digest(args.baseline_results)
    ridge = pd.read_csv(output / "backtest_2025_results.csv")
    baseline = pd.read_csv(args.baseline_results)
    comparison, summary = compare(ridge, baseline, args.bootstrap_samples, args.seed)
    summary["ridge_excluded"] = sorted(set(ridge.game_id)-set(summary["game_ids"]))
    summary["baseline_sha256"] = baseline_hash
    summary["run_manifest"] = json.loads((output / "run_manifest.json").read_text())
    snapshots = output_dir(output / "baseline")
    shutil.copy2(args.baseline_results, snapshots / "results.csv")
    if args.baseline_summary.exists():
        summary["baseline_summary_sha256"] = digest(args.baseline_summary)
        shutil.copy2(args.baseline_summary, snapshots / "summary.json")
    comparison.to_csv(output / "comparison.csv", index=False)
    pv_path = args.player_value or ROOT / "artifacts/runtime/player_value.csv"
    qb_games = qb_change_games(pd.read_csv(pv_path, dtype={"game_id": str})) if pv_path.is_file() else set()
    summary["qb_change_slice"] = {"definition": "either team's QB PV changed > 1 league SD from its previous game",
                                  "bayesian": slice_metrics(canonicalize_teams(ridge), canonicalize_teams(baseline), qb_games)}
    html_sections = ["<h2>vs Bayesian saved baseline</h2>" + comparison.to_html(index=False)]
    for item in args.extra_baseline:
        label, _, path = item.partition("=")
        other = pd.read_csv(Path(path))
        extra, extra_summary = compare(ridge, other, args.bootstrap_samples, args.seed)
        extra.to_csv(output / f"comparison_vs_{label}.csv", index=False)
        summary[f"vs_{label}"] = {key: extra_summary[key] for key in ("common_games", "ridge_metrics", "baseline_metrics", "by_week")}
        summary[f"vs_{label}"]["results_sha256"] = digest(Path(path))
        summary["qb_change_slice"][label] = slice_metrics(canonicalize_teams(ridge), canonicalize_teams(other), qb_games)
        html_sections.append(f"<h2>vs {html.escape(label)}</h2>" + extra.to_html(index=False))
    write_json(output / "comparison_summary.json", json_value(summary))
    (output / "comparison_report.html").write_text("<!doctype html><html><head><meta charset='utf-8'><title>Ridge PV 2025 Comparison</title></head><body><h1>Ridge + Player Value vs Saved Baselines</h1>" + "".join(html_sections) + "<h2>Coverage and Provenance</h2><pre>" + html.escape(json.dumps(json_value(summary), indent=2)) + "</pre></body></html>", encoding="utf-8")
    if digest(args.baseline_results) != baseline_hash:
        raise RuntimeError("Baseline changed")


if __name__ == "__main__":
    main()