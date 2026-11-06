"""Compare alternative rolling-history models on a blocked NFL holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from nflsim.features.bayesian_history import (
    DEFAULT_CACHE_PATH,
    MetricSpec,
    _config_for_metric,
    collect_player_metric_specs,
    collect_team_metric_specs,
    load_decay_rates,
    load_pbp_data,
)
from nflsim.features.rolling_comparison import _bootstrap_delta, _frozen_bayesian_history, _season
from nflsim.features.historical_estimators import EstimatorConfig, candidate_grid, config_name, predict_history
from nflsim.features.weighted_rolling import (
    DraftPriorModel,
    _fast_weighted_predictions,
    attach_prior_means,
    build_player_game_context,
    enrich_metric_frame,
    fit_draft_prior,
    load_parameters,
)


DEFAULT_OUTPUT_DIR = Path("historical_feature_bakeoff")


def _fit_prior(spec: MetricSpec, frame: pd.DataFrame) -> DraftPriorModel:
    if spec.scope == "player":
        return fit_draft_prior(frame, spec.entity_col)
    values = pd.to_numeric(frame["rate"], errors="coerce")
    counts = pd.to_numeric(frame[spec.count_col], errors="coerce").fillna(1.0)
    valid = values.notna() & counts.gt(0)
    mean = float(np.average(values[valid], weights=counts[valid])) if valid.any() else 0.0
    return DraftPriorModel(mean)


def _score(actual: np.ndarray, predicted: np.ndarray, exposure: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(exposure)
    if not valid.any():
        return {"rows": 0, "mae": float("nan"), "rmse": float("nan"), "exposure_rmse": float("nan")}
    error = predicted[valid] - actual[valid]
    weights = np.clip(exposure[valid], 1.0, None)
    return {
        "rows": int(valid.sum()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "exposure_rmse": float(np.sqrt(np.average(np.square(error), weights=weights))),
    }


def _family_candidates(kind: str) -> dict[str, list[EstimatorConfig]]:
    families: dict[str, list[EstimatorConfig]] = {}
    for config in candidate_grid(kind):
        families.setdefault(config.method, []).append(config)
    return families


def _screening_frame(
    frame: pd.DataFrame,
    entity_col: str,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()
    sizes = frame.groupby(entity_col, dropna=False).size()
    entities = sizes.index.to_numpy()
    rng = np.random.default_rng(seed)
    selected: list[object] = []
    row_count = 0
    for entity in rng.permutation(entities):
        selected.append(entity)
        row_count += int(sizes.loc[entity])
        if row_count >= max_rows:
            break
    return frame[frame[entity_col].isin(selected)].copy().reset_index(drop=True)


def _baseline_predictions(
    spec: MetricSpec,
    enriched: pd.DataFrame,
    fit_through: int,
    weighted_parameters: dict[str, object],
    decay_rates: dict[str, dict[str, float]],
) -> dict[str, np.ndarray]:
    bayesian = _frozen_bayesian_history(
        spec,
        enriched,
        _config_for_metric(spec.kind, spec.source, decay_rates),
        fit_through,
    )
    keys = [spec.entity_col, "game_id", spec.date_col]
    aligned = enriched[keys].merge(bayesian, on=keys, how="left")
    result = {"bayesian": aligned["bayesian_prediction"].to_numpy(dtype=float)}
    parameters = weighted_parameters.get(spec.source)
    if parameters is not None:
        weighted_input = attach_prior_means(enriched, parameters.prior, spec.scope == "player")
        result["weighted_rolling"] = _fast_weighted_predictions(
            weighted_input,
            spec.entity_col,
            parameters.weighting,
            count_col=spec.count_col,
            date_col=spec.date_col,
            use_player_context=spec.scope == "player",
        )
    return result


def evaluate_metric(
    spec: MetricSpec,
    enriched: pd.DataFrame,
    train_through: int,
    validation_season: int,
    test_season: int,
    weighted_parameters: dict[str, object],
    decay_rates: dict[str, dict[str, float]],
    bootstrap_samples: int,
    seed: int,
    screening_rows: int = 8000,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    enriched = (
        enriched[enriched[spec.entity_col].notna()]
        .sort_values([spec.entity_col, spec.date_col, "game_id"])
        .drop_duplicates([spec.entity_col, "game_id", spec.date_col], keep="last")
        .reset_index(drop=True)
    )
    train = enriched[_season(enriched) <= train_through].copy()
    refit = enriched[_season(enriched) <= validation_season].copy()
    validation_mask = (_season(enriched) == validation_season).to_numpy()
    test_mask = (_season(enriched) == test_season).to_numpy()
    actual = pd.to_numeric(enriched["rate"], errors="coerce").to_numpy(dtype=float)
    exposure = pd.to_numeric(enriched[spec.count_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    if not validation_mask.any() or not test_mask.any() or train.empty:
        return [], []

    train_prior = _fit_prior(spec, train)
    refit_prior = _fit_prior(spec, refit)
    screening = _screening_frame(
        enriched[_season(enriched) <= validation_season],
        spec.entity_col,
        screening_rows,
        seed,
    )
    screening_train = screening[_season(screening) <= train_through]
    screening_prior = _fit_prior(spec, screening_train)
    screening = attach_prior_means(screening, screening_prior, spec.scope == "player")
    full_candidate_frame = attach_prior_means(enriched, refit_prior, spec.scope == "player")
    screening_validation = (_season(screening) == validation_season).to_numpy()
    screening_actual = pd.to_numeric(screening["rate"], errors="coerce").to_numpy(dtype=float)
    screening_exposure = (
        pd.to_numeric(screening[spec.count_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    )
    baselines = _baseline_predictions(spec, enriched, validation_season, weighted_parameters, decay_rates)
    rows: list[dict[str, object]] = []
    prediction_records: list[dict[str, object]] = []

    for method, predictions in baselines.items():
        validation_score = _score(actual[validation_mask], predictions[validation_mask], exposure[validation_mask])
        test_score = _score(actual[test_mask], predictions[test_mask], exposure[test_mask])
        rows.append(
            {
                "metric": spec.source,
                "scope": spec.scope,
                "kind": spec.kind,
                "method": method,
                "config": "saved" if method == "weighted_rolling" else "current_decay",
                **{f"validation_{key}": value for key, value in validation_score.items()},
                **{f"test_{key}": value for key, value in test_score.items()},
            }
        )
        for index in np.flatnonzero(test_mask):
            prediction_records.append(
                {
                    "metric": spec.source,
                    "row_id": int(index),
                    "entity": enriched.iloc[index][spec.entity_col],
                    "method": method,
                    "actual": actual[index],
                    "prediction": predictions[index],
                }
            )

    for family, configs in _family_candidates(spec.kind).items():
        best_config: EstimatorConfig | None = None
        best_loss = float("inf")
        for config in configs:
            predictions = predict_history(
                screening,
                spec,
                screening_prior,
                config,
                variance_frame=screening_train,
            )
            score = _score(
                screening_actual[screening_validation],
                predictions[screening_validation],
                screening_exposure[screening_validation],
            )
            if np.isfinite(score["exposure_rmse"]) and score["exposure_rmse"] < best_loss:
                best_loss = score["exposure_rmse"]
                best_config = config
        if best_config is None:
            continue
        predictions = predict_history(
            full_candidate_frame,
            spec,
            refit_prior,
            best_config,
            variance_frame=refit,
        )
        validation_score = _score(actual[validation_mask], predictions[validation_mask], exposure[validation_mask])
        test_score = _score(actual[test_mask], predictions[test_mask], exposure[test_mask])
        rows.append(
            {
                "metric": spec.source,
                "scope": spec.scope,
                "kind": spec.kind,
                "method": family,
                "config": config_name(best_config),
                **{f"validation_{key}": value for key, value in validation_score.items()},
                **{f"test_{key}": value for key, value in test_score.items()},
            }
        )
        for index in np.flatnonzero(test_mask):
            prediction_records.append(
                {
                    "metric": spec.source,
                    "row_id": int(index),
                    "entity": enriched.iloc[index][spec.entity_col],
                    "method": family,
                    "actual": actual[index],
                    "prediction": predictions[index],
                }
            )

    row_frame = pd.DataFrame(rows)
    if row_frame.empty or "bayesian" not in set(row_frame["method"]):
        return rows, prediction_records
    bayesian_rmse = float(row_frame.loc[row_frame["method"] == "bayesian", "test_rmse"].iloc[0])
    for row in rows:
        row["rmse_ratio_vs_bayesian"] = float(row["test_rmse"]) / bayesian_rmse if bayesian_rmse > 0 else np.nan

    prediction_frame = pd.DataFrame(prediction_records)
    bayesian_rows = prediction_frame[prediction_frame["method"] == "bayesian"].rename(
        columns={"prediction": "bayesian_prediction"}
    )
    for row in rows:
        if row["method"] == "bayesian":
            row["rmse_delta_ci_low"] = 0.0
            row["rmse_delta_ci_high"] = 0.0
            continue
        method_rows = prediction_frame[prediction_frame["method"] == row["method"]].rename(
            columns={"prediction": "weighted_prediction"}
        )
        paired = method_rows.merge(
            bayesian_rows[["metric", "row_id", "entity", "actual", "bayesian_prediction"]],
            on=["metric", "row_id", "entity", "actual"],
            how="inner",
        ).rename(columns={"actual": "rate"})
        if paired.empty:
            row["rmse_delta_ci_low"] = np.nan
            row["rmse_delta_ci_high"] = np.nan
        else:
            low, high = _bootstrap_delta(paired, "entity", bootstrap_samples, seed)
            row["rmse_delta_ci_low"] = low
            row["rmse_delta_ci_high"] = high
    return rows, prediction_records


def build_summary(results: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "metrics": int(results["metric"].nunique()) if not results.empty else 0,
        "methods": {},
        "selected_method_counts": {},
    }
    if results.empty:
        return summary
    for method, group in results.groupby("method"):
        ratios = pd.to_numeric(group["rmse_ratio_vs_bayesian"], errors="coerce")
        summary["methods"][method] = {
            "metrics": int(group["metric"].nunique()),
            "mean_rmse_ratio_vs_bayesian": float(ratios.mean()),
            "median_rmse_ratio_vs_bayesian": float(ratios.median()),
            "wins_vs_bayesian": int((ratios < 1.0).sum()),
            "significant_wins": int((pd.to_numeric(group["rmse_delta_ci_high"], errors="coerce") < 0).sum()),
            "significant_losses": int((pd.to_numeric(group["rmse_delta_ci_low"], errors="coerce") > 0).sum()),
        }
    portfolios = {
        "alternatives_only": results[~results["method"].isin(["bayesian", "weighted_rolling"])],
        "production_safe": results[results["method"] != "weighted_rolling"],
        "all_methods": results,
    }
    summary["portfolios"] = {}
    for name, candidates in portfolios.items():
        if candidates.empty:
            continue
        winners = candidates.loc[candidates.groupby("metric")["validation_exposure_rmse"].idxmin()]
        winner_ratios = winners["rmse_ratio_vs_bayesian"]
        summary["portfolios"][name] = {
            "mean_rmse_ratio_vs_bayesian": float(winner_ratios.mean()),
            "median_rmse_ratio_vs_bayesian": float(winner_ratios.median()),
            "wins_vs_bayesian": int((winner_ratios < 1.0).sum()),
            "losses_vs_bayesian": int((winner_ratios > 1.0).sum()),
            "significant_wins": int((winners["rmse_delta_ci_high"] < 0).sum()),
            "significant_losses": int((winners["rmse_delta_ci_low"] > 0).sum()),
            "method_counts": {
                str(key): int(value) for key, value in winners["method"].value_counts().items()
            },
        }
    return summary


def select_portfolio(results: pd.DataFrame, include_weighted: bool = False) -> pd.DataFrame:
    candidates = results.copy()
    if not include_weighted:
        candidates = candidates[candidates["method"] != "weighted_rolling"]
    return (
        candidates.loc[candidates.groupby("metric")["validation_exposure_rmse"].idxmin()]
        .sort_values(["scope", "kind", "metric"])
        .reset_index(drop=True)
    )


def run_bakeoff(
    years: Iterable[int],
    train_through: int,
    validation_season: int,
    test_season: int,
    weights_file: Path,
    bootstrap_samples: int,
    screening_rows: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    years = list(years)
    pbp = load_pbp_data(years, cache_path=DEFAULT_CACHE_PATH)
    context = build_player_game_context(pbp, years)
    specs = collect_player_metric_specs(pbp) + collect_team_metric_specs(pbp)
    weighted_parameters = load_parameters(weights_file)
    decay_rates = load_decay_rates()
    rows: list[dict[str, object]] = []
    for metric_index, spec in enumerate(specs):
        if spec.kind == "direct":
            continue
        enriched = enrich_metric_frame(spec, context).reset_index(drop=True)
        metric_rows, _ = evaluate_metric(
            spec,
            enriched,
            train_through,
            validation_season,
            test_season,
            weighted_parameters,
            decay_rates,
            bootstrap_samples,
            9109 + metric_index,
            screening_rows,
        )
        rows.extend(metric_rows)
        if metric_rows:
            best = min(
                (row for row in metric_rows if row["method"] not in {"bayesian", "weighted_rolling"}),
                key=lambda row: row["validation_exposure_rmse"],
            )
            print(
                f"{spec.source}: selected={best['method']} ({best['config']}), "
                f"test ratio={best['rmse_ratio_vs_bayesian']:.4f}"
            )
    results = pd.DataFrame(rows)
    return results, build_summary(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a broad rolling-feature estimator bake-off.")
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2016, 2026)))
    parser.add_argument("--train-through", type=int, default=2023)
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument(
        "--weights-file",
        type=Path,
        default=Path("weighted_rolling_full/weighted_rolling_params.json"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument(
        "--screening-rows",
        type=int,
        default=3000,
        help="Approximate number of complete entity-history rows used to select each family configuration.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    results, summary = run_bakeoff(
        args.years,
        args.train_through,
        args.validation_season,
        args.test_season,
        args.weights_file,
        args.bootstrap_samples,
        args.screening_rows,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "historical_feature_bakeoff.csv", index=False)
    select_portfolio(results, include_weighted=False).to_csv(
        args.output_dir / "historical_feature_production_selection.csv",
        index=False,
    )
    select_portfolio(results, include_weighted=True).to_csv(
        args.output_dir / "historical_feature_all_method_selection.csv",
        index=False,
    )
    (args.output_dir / "historical_feature_bakeoff_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Saved {len(results)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()