"""Tune recency-weighted ridge metrics on a holdout season and compare with rolling Bayesian baselines."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from bayesian_feature_builder import (
    DEFAULT_CACHE_PATH,
    MetricSpec,
    _config_for_metric,
    load_decay_rates,
    load_pbp_data,
)
from compare_weighted_rolling_features import _bootstrap_delta, _frozen_bayesian_history
from ridge_adjusted_metrics import (
    TARGETS,
    TARGETS_BY_NAME,
    RidgeConfig,
    RidgeTarget,
    actual_output_frame,
    aggregate_output_predictions,
    build_design,
    build_game_context,
    build_target_rows,
    fit_coefficients,
    rolling_predictions,
    weekly_coefficients,
)
from run_historical_feature_bakeoff import _score


DEFAULT_OUTPUT_DIR = Path("ridge_adjusted_bakeoff")
DEFAULT_BAKEOFF_SELECTION = Path("historical_feature_bakeoff/historical_feature_production_selection.csv")
Side = Literal["subject", "defense"]

RECENCY_GRID = {
    "within_decay": (0.85, 0.92, 0.97, 1.0),
    "across_decay": (0.98, 1.0),
    "season_carryover": (0.25, 0.5, 0.75),
    "lookback_seasons": (1, 2),
}
PENALTY_GRID = {
    "lambda_subject": (10.0, 30.0, 100.0, 300.0),
    "lambda_defense": (30.0, 100.0, 300.0, 1000.0),
    "lambda_venue": (100.0, 1000.0),
    "lambda_offense_team": (100.0, 300.0, 1000.0),
}
QUICK_RECENCY_GRID = {"within_decay": (0.9, 1.0), "season_carryover": (0.5,), "lookback_seasons": (1,)}
QUICK_PENALTY_GRID = {"lambda_subject": (30.0, 300.0), "lambda_defense": (100.0,), "lambda_offense_team": (300.0,)}
EXTRA_BLOCK_PENALTIES = (30.0, 100.0, 300.0, 1000.0)
WEEK_BUCKETS = ((1, 4, "1-4"), (5, 12, "5-12"), (13, 18, "13-18"))


def _expand(base: RidgeConfig, grid: dict[str, Iterable[float]]) -> list[RidgeConfig]:
    names = list(grid)
    return [replace(base, **dict(zip(names, values))) for values in itertools.product(*(grid[name] for name in names))]


def _load_positions() -> dict[str, str]:
    try:
        from weighted_rolling_feature_builder import load_player_metadata

        metadata = load_player_metadata()
    except Exception as error:
        print(f"Player metadata unavailable, using a single position: {error}")
        return {}
    return dict(zip(metadata["player_id"].astype(str), metadata["position"].astype(str)))


def _outputs(target: RidgeTarget) -> list[tuple[Side, str]]:
    outputs: list[tuple[Side, str]] = []
    if target.player_output is not None:
        outputs.append(("subject", target.player_output))
    if target.defense_output is not None:
        outputs.append(("defense", target.defense_output))
    return outputs


class TargetEvaluator:
    """Caches pregame ridge predictions per (config, season) for one design."""

    def __init__(self, rows: pd.DataFrame, design, sides: Iterable[Side], validation_season: int) -> None:
        self.rows = rows
        self.design = design
        self.validation_season = validation_season
        self.actual = {side: actual_output_frame(rows, side) for side in sides}
        self._cache: dict[tuple[RidgeConfig, int], dict[Side, pd.DataFrame]] = {}

    def predict(self, config: RidgeConfig, season: int) -> dict[Side, pd.DataFrame]:
        key = (config, season)
        if key not in self._cache:
            predictions = rolling_predictions(self.rows, self.design, config, [season])
            self._cache[key] = {
                side: actual[actual["season"] == season].merge(
                    aggregate_output_predictions(self.rows, predictions, side),
                    on=["entity", "game_id"],
                    how="inner",
                )
                for side, actual in self.actual.items()
            }
        return self._cache[key]

    def validation_loss(self, config: RidgeConfig, side: Side) -> float:
        frame = self.predict(config, self.validation_season)[side]
        score = _score(
            frame["rate"].to_numpy(dtype=float),
            frame["pred_full"].to_numpy(dtype=float),
            frame["count"].to_numpy(dtype=float),
        )
        loss = score["exposure_rmse"]
        return float(loss) if np.isfinite(loss) else float("inf")


def tune_config(
    evaluator: TargetEvaluator,
    side: Side,
    recency_grid: dict[str, Iterable[float]],
    penalty_grid: dict[str, Iterable[float]],
    rounds: int,
    base: RidgeConfig = RidgeConfig(),
) -> RidgeConfig:
    """Coordinate search: recency block, then penalty block, repeated."""
    best = base
    for _ in range(rounds):
        for grid in (recency_grid, penalty_grid):
            best = min(_expand(best, grid), key=lambda config: evaluator.validation_loss(config, side))
    return best


def _bayesian_predictions(
    actual: pd.DataFrame,
    kind: str,
    output_name: str,
    fit_through: int,
    decay_rates: dict[str, dict[str, float]],
) -> pd.DataFrame:
    spec = MetricSpec(output_name, kind, actual, "entity", None)
    history = _frozen_bayesian_history(spec, actual, _config_for_metric(kind, output_name, decay_rates), fit_through)
    return history.rename(columns={"bayesian_prediction": "pred_bayesian"})[["entity", "game_id", "pred_bayesian"]]


def _split_frame(
    ridge_frames: dict[str, pd.DataFrame],
    bayesian: pd.DataFrame,
) -> pd.DataFrame:
    """Join every method on identical entity-game rows."""
    base_name, base = next(iter(ridge_frames.items()))
    merged = base[["entity", "game_id", "week", "rate", "count"]].merge(bayesian, on=["entity", "game_id"], how="inner")
    for name, frame in ridge_frames.items():
        renamed = frame[["entity", "game_id", "pred_full", "pred_neutral"]].rename(
            columns={"pred_full": f"pred_{name}_full", "pred_neutral": f"pred_{name}_neutral"}
        )
        merged = merged.merge(renamed, on=["entity", "game_id"], how="inner")
    return merged.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def _method_score(frame: pd.DataFrame, column: str) -> dict[str, float]:
    return _score(
        frame["rate"].to_numpy(dtype=float),
        frame[column].to_numpy(dtype=float),
        frame["count"].to_numpy(dtype=float),
    )


def compare_methods(
    meta: dict[str, object],
    splits: dict[str, pd.DataFrame],
    configs: dict[str, str],
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    test = splits["test"]
    methods = ["bayesian", *(column[len("pred_"):] for column in test.columns if column.startswith("pred_") and column != "pred_bayesian")]
    rows: list[dict[str, object]] = []
    slices: list[dict[str, object]] = []
    bayesian_test = _method_score(test, "pred_bayesian")
    for method in methods:
        column = f"pred_{method}"
        row = {**meta, "method": method, "config": configs.get(method.rsplit("_", 1)[0], "current_decay")}
        for split, frame in splits.items():
            row.update({f"{split}_{key}": value for key, value in _method_score(frame, column).items()})
        row["rmse_ratio_vs_bayesian"] = row["test_rmse"] / bayesian_test["rmse"] if bayesian_test["rmse"] > 0 else np.nan
        row["exposure_rmse_ratio_vs_bayesian"] = (
            row["test_exposure_rmse"] / bayesian_test["exposure_rmse"] if bayesian_test["exposure_rmse"] > 0 else np.nan
        )
        if method == "bayesian":
            row["rmse_delta_ci_low"] = row["rmse_delta_ci_high"] = 0.0
        else:
            paired = test.rename(columns={column: "weighted_prediction", "pred_bayesian": "bayesian_prediction"})
            row["rmse_delta_ci_low"], row["rmse_delta_ci_high"] = _bootstrap_delta(
                paired, "entity", bootstrap_samples, seed
            )
        rows.append(row)

        for low, high, label in WEEK_BUCKETS:
            bucket = test[test["week"].between(low, high)]
            if bucket.empty:
                continue
            method_score = _method_score(bucket, column)
            bayesian_score = _method_score(bucket, "pred_bayesian")
            slices.append(
                {
                    **meta,
                    "method": method,
                    "week_bucket": label,
                    "rows": method_score["rows"],
                    "exposure_rmse": method_score["exposure_rmse"],
                    "bayesian_exposure_rmse": bayesian_score["exposure_rmse"],
                    "exposure_rmse_ratio_vs_bayesian": method_score["exposure_rmse"] / bayesian_score["exposure_rmse"]
                    if bayesian_score["exposure_rmse"] > 0
                    else np.nan,
                }
            )
    return rows, slices


def evaluate_target(
    target: RidgeTarget,
    pbp: pd.DataFrame,
    games: pd.DataFrame,
    positions: dict[str, str],
    validation_season: int,
    test_season: int,
    decay_rates: dict[str, dict[str, float]],
    bootstrap_samples: int,
    seed: int,
    recency_grid: dict[str, Iterable[float]],
    penalty_grid: dict[str, Iterable[float]],
    rounds: int,
    run_ablation: bool,
) -> dict[str, object]:
    outputs = _outputs(target)
    rows = build_target_rows(pbp, target, games, positions)
    empty = {"results": [], "slices": [], "predictions": [], "coefficients": [], "weekly_coefficients": [], "params": {}}
    if rows.empty or not outputs or not rows["season"].isin([validation_season]).any():
        print(f"{target.name}: no usable rows, skipped")
        return empty
    design = build_design(rows, target.scope)
    evaluator = TargetEvaluator(rows, design, [side for side, _ in outputs], validation_season)

    result: dict[str, object] = {**empty, "results": [], "slices": [], "predictions": [], "coefficients": [], "weekly_coefficients": []}
    params: dict[str, dict[str, object]] = {}
    for index, (side, output_name) in enumerate(outputs):
        active_penalties = penalty_grid if target.scope == "player" else {
            name: values for name, values in penalty_grid.items() if name != "lambda_offense_team"
        }
        config = tune_config(evaluator, side, recency_grid, active_penalties, rounds)
        params[output_name] = {"target": target.name, "side": side, **config.__dict__}
        ridge_by_split = {
            split: {"ridge": evaluator.predict(config, season)[side]}
            for split, season in (("validation", validation_season), ("test", test_season))
        }
        configs = {"ridge": config.label()}

        if run_ablation and side == "subject" and target.scope == "player" and target.teammate_col is not None:
            block_rows = build_target_rows(pbp, target, games, positions, extra_blocks=("teammate",))
            block_evaluator = TargetEvaluator(
                block_rows,
                build_design(block_rows, target.scope, extra_blocks=("teammate",)),
                [side],
                validation_season,
            )
            block_config = min(
                (replace(config, lambda_teammate=value) for value in EXTRA_BLOCK_PENALTIES),
                key=lambda candidate: block_evaluator.validation_loss(candidate, side),
            )
            method = "ridge+teammate"
            configs[method] = block_config.label()
            for split, season in (("validation", validation_season), ("test", test_season)):
                ridge_by_split[split][method] = block_evaluator.predict(block_config, season)[side]

        actual = evaluator.actual[side]
        splits = {
            split: _split_frame(
                ridge_by_split[split],
                _bayesian_predictions(actual, target.kind, output_name, fit_through, decay_rates),
            )
            for split, fit_through in (("validation", validation_season - 1), ("test", validation_season))
        }
        meta = {"metric": output_name, "target": target.name, "side": side, "scope": target.scope, "kind": target.kind}
        metric_rows, slice_rows = compare_methods(meta, splits, configs, bootstrap_samples, seed + index)
        result["results"].extend(metric_rows)
        result["slices"].extend(slice_rows)
        result["predictions"].append(splits["test"].assign(metric=output_name))

        coefficients = fit_coefficients(rows, design, config, test_season + 1, 1)
        result["coefficients"].append(coefficients.assign(metric=output_name, target=target.name))
        result["weekly_coefficients"].append(
            weekly_coefficients(rows, design, config).assign(metric=output_name, target=target.name, side=side)
        )

        ridge_row = next(row for row in metric_rows if row["method"] == "ridge_full")
        print(
            f"{output_name} ({target.name}/{side}): {config.label()} | "
            f"test exposure ratio full={ridge_row['exposure_rmse_ratio_vs_bayesian']:.4f}"
        )
    result["params"] = params
    return result


def build_summary(results: pd.DataFrame, bakeoff_selection: Path) -> dict[str, object]:
    summary: dict[str, object] = {"outputs": int(results["metric"].nunique()), "methods": {}}
    for method, group in results.groupby("method"):
        ratios = pd.to_numeric(group["rmse_ratio_vs_bayesian"], errors="coerce")
        exposure = pd.to_numeric(group["exposure_rmse_ratio_vs_bayesian"], errors="coerce")
        summary["methods"][method] = {
            "outputs": int(group["metric"].nunique()),
            "mean_rmse_ratio_vs_bayesian": float(ratios.mean()),
            "median_rmse_ratio_vs_bayesian": float(ratios.median()),
            "median_exposure_rmse_ratio_vs_bayesian": float(exposure.median()),
            "wins_vs_bayesian": int((ratios < 1.0).sum()),
            "significant_wins": int((pd.to_numeric(group["rmse_delta_ci_high"], errors="coerce") < 0).sum()),
            "significant_losses": int((pd.to_numeric(group["rmse_delta_ci_low"], errors="coerce") > 0).sum()),
        }
    selected = results.loc[results.groupby("metric")["validation_exposure_rmse"].idxmin()]
    summary["validation_selected"] = {
        "method_counts": {str(key): int(value) for key, value in selected["method"].value_counts().items()},
        "median_rmse_ratio_vs_bayesian": float(selected["rmse_ratio_vs_bayesian"].median()),
        "significant_wins": int((selected["rmse_delta_ci_high"] < 0).sum()),
        "significant_losses": int((selected["rmse_delta_ci_low"] > 0).sum()),
    }
    if bakeoff_selection.exists():
        reference = pd.read_csv(bakeoff_selection)
        reference = reference[reference["metric"].isin(results["metric"])]
        summary["historical_bakeoff_reference"] = {
            str(row.metric): {"method": str(row.method), "rmse_ratio_vs_bayesian": float(row.rmse_ratio_vs_bayesian)}
            for row in reference.itertuples(index=False)
        }
    return summary


def player_names(pbp: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for role in ("receiver", "passer", "rusher"):
        player_id = f"{role}_player_id"
        player_name = f"{role}_player_name"
        if player_id in pbp and player_name in pbp:
            frames.append(
                pbp[[player_id, player_name, "game_date"]]
                .rename(columns={player_id: "label", player_name: "name"})
                .dropna(subset=["label", "name"])
            )
    if not frames:
        return pd.DataFrame(columns=["label", "name"])
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("game_date")
        .drop_duplicates("label", keep="last")[["label", "name"]]
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ridge-adjusted skill metrics vs rolling Bayesian baselines.")
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2016, 2026)))
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--targets", nargs="+", choices=sorted(TARGETS_BY_NAME), default=None)
    parser.add_argument("--rounds", type=int, default=2, help="Coordinate-search passes over recency then penalties.")
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--quick", action="store_true", help="Tiny grid for smoke tests.")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--bakeoff-selection", type=Path, default=DEFAULT_BAKEOFF_SELECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    targets = [TARGETS_BY_NAME[name] for name in args.targets] if args.targets else list(TARGETS)
    recency_grid = QUICK_RECENCY_GRID if args.quick else RECENCY_GRID
    penalty_grid = QUICK_PENALTY_GRID if args.quick else PENALTY_GRID
    rounds = 1 if args.quick else args.rounds

    pbp = load_pbp_data(args.years, cache_path=DEFAULT_CACHE_PATH)
    games = build_game_context(pbp)
    positions = _load_positions()
    decay_rates = load_decay_rates()

    collected: dict[str, list] = {"results": [], "slices": [], "predictions": [], "coefficients": [], "weekly_coefficients": []}
    params: dict[str, object] = {}
    for index, target in enumerate(targets):
        started = time.perf_counter()
        outcome = evaluate_target(
            target,
            pbp,
            games,
            positions,
            args.validation_season,
            args.test_season,
            decay_rates,
            args.bootstrap_samples,
            7301 + 10 * index,
            recency_grid,
            penalty_grid,
            rounds,
            not args.skip_ablation,
        )
        for key in collected:
            collected[key].extend(outcome[key])
        params.update(outcome["params"])
        print(f"  {target.name} finished in {time.perf_counter() - started:.1f}s")

    if not collected["results"]:
        print("No results produced.")
        return
    results = pd.DataFrame(collected["results"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "results.csv", index=False)
    pd.DataFrame(collected["slices"]).to_csv(args.output_dir / "week_slices.csv", index=False)
    pd.concat(collected["predictions"], ignore_index=True).to_csv(args.output_dir / "test_predictions.csv", index=False)
    pd.concat(collected["coefficients"], ignore_index=True).to_csv(args.output_dir / "coefficients.csv", index=False)
    pd.concat(collected["weekly_coefficients"], ignore_index=True).to_parquet(
        args.output_dir / "weekly_coefficients.parquet", index=False
    )
    player_names(pbp).to_csv(args.output_dir / "player_names.csv", index=False)
    (args.output_dir / "best_params.json").write_text(json.dumps(params, indent=2, sort_keys=True), encoding="utf-8")
    summary = build_summary(results, args.bakeoff_selection)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary["methods"], indent=2))
    print(f"Saved {len(results)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
