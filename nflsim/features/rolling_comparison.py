"""Blocked temporal comparison of Bayesian and context-weighted feature estimates."""

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
    SmoothingConfig,
    _config_for_metric,
    _game_gap_decay,
    collect_player_metric_specs,
    collect_team_metric_specs,
    load_decay_rates,
    load_pbp_data,
)
from nflsim.features.weighted_rolling import (
    DEFAULT_WEIGHTS_FILE,
    DraftPriorModel,
    MetricParameters,
    _fast_weighted_predictions,
    attach_prior_means,
    build_player_game_context,
    enrich_metric_frame,
    fit_draft_prior,
    fit_weighting_config,
    load_parameters,
)


DEFAULT_OUTPUT_CSV = Path("weighted_rolling_comparison.csv")
DEFAULT_OUTPUT_JSON = Path("weighted_rolling_comparison_summary.json")


def _season(frame: pd.DataFrame) -> pd.Series:
    if "season" in frame.columns:
        return pd.to_numeric(frame["season"], errors="coerce")
    return pd.to_numeric(frame["game_id"].astype(str).str.split("_").str[0], errors="coerce")


def _frozen_bayesian_history(
    spec: MetricSpec,
    frame: pd.DataFrame,
    config: SmoothingConfig,
    fit_through_season: int,
) -> pd.DataFrame:
    """Reproduce current smoothing while freezing prior moments before test."""
    ordered = frame.sort_values([spec.entity_col, spec.date_col, "game_id"]).copy()
    training = ordered[_season(ordered) <= fit_through_season]
    values = pd.to_numeric(training["rate"], errors="coerce")
    counts = pd.to_numeric(training[spec.count_col], errors="coerce")
    global_mean = float(values.mean())
    prior_weight = max(float(counts.mean()) * config.prior_strength, config.prior_games)
    global_variance = float(max(values.var(ddof=0), 1e-9))

    records: list[dict[str, object]] = []
    for _, entity_rows in ordered.groupby(spec.entity_col, sort=False, dropna=False):
        state_weight = 0.0
        state_sum = 0.0
        state_sum_sq = 0.0
        last_season_week: tuple[int | None, int | None] | None = None
        for row in entity_rows.to_dict(orient="records"):
            parts = str(row["game_id"]).split("_")
            current_season = int(parts[0]) if parts and parts[0].isdigit() else None
            current_week = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            if last_season_week is not None:
                decay = _game_gap_decay(
                    config.decay,
                    config.season_carryover,
                    last_season_week[0],
                    last_season_week[1],
                    current_season,
                    current_week,
                )
                state_weight *= decay
                state_sum *= decay
                state_sum_sq *= decay

            if spec.kind == "binary":
                prior_success = global_mean * prior_weight
                estimate = (prior_success + state_sum) / (prior_weight + state_weight)
            elif state_weight <= 0:
                estimate = global_mean
            else:
                empirical_mean = state_sum / state_weight
                empirical_variance = max(
                    state_sum_sq / state_weight - empirical_mean * empirical_mean,
                    1e-9,
                )
                prior_precision = prior_weight / global_variance
                posterior_precision = prior_precision + state_weight / empirical_variance
                estimate = (
                    prior_precision * global_mean + state_weight * empirical_mean / empirical_variance
                ) / posterior_precision

            records.append(
                {
                    spec.entity_col: row[spec.entity_col],
                    "game_id": row["game_id"],
                    spec.date_col: row[spec.date_col],
                    "bayesian_prediction": float(estimate),
                }
            )
            value = pd.to_numeric(pd.Series([row["rate"]]), errors="coerce").iloc[0]
            count = pd.to_numeric(pd.Series([row[spec.count_col]]), errors="coerce").iloc[0]
            last_season_week = (current_season, current_week)
            if pd.isna(value) or pd.isna(count) or float(count) <= 0:
                continue
            if spec.kind == "binary":
                state_sum += float(value) * float(count)
                state_weight += float(count)
            else:
                state_weight += float(count)
                state_sum += float(value) * float(count)
                state_sum_sq += float(value) * float(value) * float(count)
    return pd.DataFrame.from_records(records)


def _prediction_frame(
    spec: MetricSpec,
    enriched: pd.DataFrame,
    parameters: MetricParameters,
    bayesian_config: SmoothingConfig,
    fit_through_season: int,
) -> pd.DataFrame:
    weighted_input = attach_prior_means(enriched, parameters.prior, spec.scope == "player")
    weighted_input["weighted_prediction"] = _fast_weighted_predictions(
        weighted_input,
        spec.entity_col,
        parameters.weighting,
        count_col=spec.count_col,
        date_col=spec.date_col,
        use_player_context=spec.scope == "player",
    )
    bayesian = _frozen_bayesian_history(spec, enriched, bayesian_config, fit_through_season)
    context_columns = [
        spec.entity_col,
        "game_id",
        spec.date_col,
        "rate",
        spec.count_col,
        "season",
        "week",
    ]
    for column in ("position", "team", "injury_designation", "entity_game_number"):
        if column in enriched.columns and column not in context_columns:
            context_columns.append(column)
    actual = enriched[context_columns].copy()
    if "team" in actual.columns:
        actual = actual.sort_values([spec.entity_col, spec.date_col, "game_id"])
        previous_team = actual.groupby(spec.entity_col)["team"].shift(1)
        actual["team_continuity"] = np.where(
            previous_team.isna() | actual["team"].isna(),
            "unknown",
            np.where(actual["team"].astype(str) == previous_team.astype(str), "same_team", "changed_team"),
        )
    keys = [spec.entity_col, "game_id", spec.date_col]
    return (
        actual.merge(weighted_input[keys + ["weighted_prediction"]], on=keys, how="inner")
        .merge(bayesian, on=keys, how="inner")
        .dropna(subset=["rate", "weighted_prediction", "bayesian_prediction"])
    )


def _weighted_mse(frame: pd.DataFrame, prediction_col: str, count_col: str) -> float:
    weights = pd.to_numeric(frame[count_col], errors="coerce").fillna(1.0).clip(lower=1.0)
    return float(np.average(np.square(frame[prediction_col] - frame["rate"]), weights=weights))


def _metric_summary(frame: pd.DataFrame, spec: MetricSpec, label: str, value: object) -> dict[str, object]:
    weighted_error = frame["weighted_prediction"] - frame["rate"]
    bayesian_error = frame["bayesian_prediction"] - frame["rate"]
    entity_weighted = frame.assign(error=np.square(weighted_error)).groupby(spec.entity_col)["error"].mean()
    entity_bayesian = frame.assign(error=np.square(bayesian_error)).groupby(spec.entity_col)["error"].mean()
    return {
        "metric": spec.source,
        "scope": spec.scope,
        "kind": spec.kind,
        "slice": label,
        "slice_value": str(value),
        "rows": int(len(frame)),
        "entities": int(frame[spec.entity_col].nunique()),
        "weighted_mae": float(weighted_error.abs().mean()),
        "bayesian_mae": float(bayesian_error.abs().mean()),
        "weighted_rmse": float(np.sqrt(np.mean(np.square(weighted_error)))),
        "bayesian_rmse": float(np.sqrt(np.mean(np.square(bayesian_error)))),
        "weighted_exposure_rmse": float(np.sqrt(_weighted_mse(frame, "weighted_prediction", spec.count_col))),
        "bayesian_exposure_rmse": float(np.sqrt(_weighted_mse(frame, "bayesian_prediction", spec.count_col))),
        "weighted_entity_macro_rmse": float(np.sqrt(entity_weighted.mean())),
        "bayesian_entity_macro_rmse": float(np.sqrt(entity_bayesian.mean())),
    }


def _bootstrap_delta(
    frame: pd.DataFrame,
    entity_col: str,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    entities = frame[entity_col].dropna().unique()
    if len(entities) < 2 or samples <= 0:
        return float("nan"), float("nan")
    grouped = {
        entity: group
        for entity, group in frame.groupby(entity_col, dropna=False)
    }
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(samples):
        selected = rng.choice(entities, size=len(entities), replace=True)
        sample = pd.concat([grouped[entity] for entity in selected], ignore_index=True)
        weighted_rmse = np.sqrt(np.mean(np.square(sample["weighted_prediction"] - sample["rate"])))
        bayesian_rmse = np.sqrt(np.mean(np.square(sample["bayesian_prediction"] - sample["rate"])))
        deltas.append(float(weighted_rmse - bayesian_rmse))
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def _slice_rows(frame: pd.DataFrame, spec: MetricSpec) -> list[dict[str, object]]:
    rows = [_metric_summary(frame, spec, "overall", "all")]
    slices: list[tuple[str, pd.Series]] = []
    if "position" in frame:
        slices.append(("position", frame["position"].fillna("missing")))
    if "injury_designation" in frame:
        injury = frame["injury_designation"].astype("boolean").fillna(False)
        slices.append(("injury", injury.astype(str)))
    if "team_continuity" in frame:
        slices.append(("team_continuity", frame["team_continuity"].fillna("unknown")))
    if "week" in frame:
        week = pd.to_numeric(frame["week"], errors="coerce")
        slices.append(("source_week", pd.cut(week, [0, 4, 12, 16, 18], labels=["1-4", "5-12", "13-16", "17-18"])))
    if "entity_game_number" in frame:
        career_game = pd.to_numeric(frame["entity_game_number"], errors="coerce") + 1
        slices.append(("career_games", pd.cut(career_game, [0, 4, 16, 48, np.inf], labels=["1-4", "5-16", "17-48", "49+"])))
    for label, values in slices:
        for value, group in frame.assign(_slice_value=values).groupby("_slice_value", observed=True, dropna=False):
            if len(group) >= 5:
                rows.append(_metric_summary(group, spec, label, value))
    return rows


def compare_features(
    years: Iterable[int],
    train_through: int,
    validation_season: int,
    test_season: int,
    regularization_grid: Iterable[float],
    bootstrap_samples: int,
    weights_file: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    years = list(years)
    pbp = load_pbp_data(years, cache_path=DEFAULT_CACHE_PATH)
    player_context = build_player_game_context(pbp, years)
    specs = collect_player_metric_specs(pbp) + collect_team_metric_specs(pbp)
    decay_rates = load_decay_rates()
    saved_parameters = load_parameters(weights_file) if weights_file is not None else {}
    output_rows: list[dict[str, object]] = []
    metric_results: list[dict[str, object]] = []

    for metric_index, spec in enumerate(specs):
        if spec.kind == "direct":
            continue
        enriched = enrich_metric_frame(spec, player_context)
        training = enriched[_season(enriched) <= train_through]
        if training.empty:
            continue
        prior = fit_draft_prior(training, spec.entity_col) if spec.scope == "player" else DraftPriorModel(
            float(pd.to_numeric(training["rate"], errors="coerce").mean())
        )
        bayesian_config = _config_for_metric(spec.kind, spec.source, decay_rates)
        parameters = saved_parameters.get(spec.source)
        best_regularization: float | None = None
        if parameters is None:
            best_validation_loss = float("inf")
            for regularization in regularization_grid:
                weighting, _ = fit_weighting_config(spec, training, prior, float(regularization))
                candidate = MetricParameters(weighting, prior, len(training), None)
                predictions = _prediction_frame(spec, enriched, candidate, bayesian_config, train_through)
                validation = predictions[_season(predictions) == validation_season]
                if validation.empty:
                    continue
                loss = _weighted_mse(validation, "weighted_prediction", spec.count_col)
                if loss < best_validation_loss:
                    best_validation_loss = loss
                    best_regularization = float(regularization)
            if best_regularization is None:
                continue

            refit = enriched[_season(enriched) <= validation_season]
            refit_prior = fit_draft_prior(refit, spec.entity_col) if spec.scope == "player" else DraftPriorModel(
                float(pd.to_numeric(refit["rate"], errors="coerce").mean())
            )
            weighting, training_loss = fit_weighting_config(spec, refit, refit_prior, best_regularization)
            parameters = MetricParameters(weighting, refit_prior, len(refit), training_loss)
        predictions = _prediction_frame(spec, enriched, parameters, bayesian_config, validation_season)
        test = predictions[_season(predictions) == test_season].copy()
        if test.empty:
            continue
        metric_rows = _slice_rows(test, spec)
        ci_low, ci_high = _bootstrap_delta(test, spec.entity_col, bootstrap_samples, 1729 + metric_index)
        metric_rows[0]["rmse_delta_ci_low"] = ci_low
        metric_rows[0]["rmse_delta_ci_high"] = ci_high
        metric_rows[0]["regularization"] = best_regularization
        output_rows.extend(metric_rows)
        metric_results.append(metric_rows[0])
        print(
            f"{spec.source}: weighted={metric_rows[0]['weighted_rmse']:.6f}, "
            f"bayesian={metric_rows[0]['bayesian_rmse']:.6f}"
        )

    results = pd.DataFrame(output_rows)
    overall = pd.DataFrame(metric_results)
    summary = {
        "train_through": train_through,
        "validation_season": validation_season,
        "test_season": test_season,
        "metrics_scored": int(len(overall)),
        "weighted_mean_normalized_rmse": None,
        "bayesian_mean_normalized_rmse": None,
        "weighted_wins": 0,
        "bayesian_wins": 0,
    }
    if not overall.empty:
        scale = overall["bayesian_rmse"].replace(0, np.nan)
        summary["weighted_mean_normalized_rmse"] = float((overall["weighted_rmse"] / scale).mean())
        summary["bayesian_mean_normalized_rmse"] = 1.0
        summary["weighted_wins"] = int((overall["weighted_rmse"] < overall["bayesian_rmse"]).sum())
        summary["bayesian_wins"] = int((overall["weighted_rmse"] > overall["bayesian_rmse"]).sum())
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare weighted rolling and Bayesian feature estimates.")
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2016, 2026)))
    parser.add_argument("--train-through", type=int, default=2023)
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--regularization-grid", type=float, nargs="+", default=[0.005, 0.02, 0.10])
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument(
        "--weights-file",
        type=Path,
        help=f"Use pre-fitted weighted parameters instead of tuning again (default fit path: {DEFAULT_WEIGHTS_FILE}).",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    results, summary = compare_features(
        args.years,
        args.train_through,
        args.validation_season,
        args.test_season,
        args.regularization_grid,
        args.bootstrap_samples,
        args.weights_file,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} comparison rows to {args.output_csv}")


if __name__ == "__main__":
    main()