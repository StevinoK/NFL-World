"""Compare ridge-metric variants out of sample: expectation targets, team-health PV and regime decay.

Every variant is tuned on the validation season and scored on the test season with pregame
predictions (each week refit on strictly earlier games). Variants:

- ``production``: current targets and the five player-value columns.
- ``health``: adds offensive-line, linebacker and secondary player value (``pv_off_ol``,
  ``pv_def_lb``, ``pv_def_db``) so, e.g., a running back is not charged for an injured line.
- ``regime``: tunes extra carryover for games before a team change or head-coach change.
- ``xoe``: the target is ``outcome - expected(game state, weather)`` (``nflsim.ridge.expectations``).
- ``all``: ``xoe`` + ``health`` + ``regime``.
- ``live``: the target as the feature build fits it (over-expectation for adopted targets).
- ``hit`` / ``pfr``: ``live`` plus pass-rush pressure faced as a covariate, fit on realized and
  predicted with the pregame pressure forecast (``nflsim.ridge.pressure``). Compare with
  ``--reference live``.
- ``luck`` (EPA targets only): turnovers charged at league-average EPA and extremes clipped
  (``nflsim.ridge.epa_luck``); ``luck_gt`` also drops garbage time. Both are scored against
  raw EPA, so they win only if the cleaned history predicts future EPA better.

Scoring. The simulator already conditions every play on game state and weather, so what a
ridge estimate must supply is the player/team effect *beyond* that context. Each entity-game
is therefore scored as ``expected outcome (same context model for all variants) + estimated
effect``. Production-scale effects are centered within each league week so only the
skill estimate differs between variants. ``no_context_rmse`` scores production predictions
alone, without the context term, for reference.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from nflsim.features.bayesian_history import DEFAULT_CACHE_PATH, load_pbp_data
from nflsim.features.rolling_comparison import _bootstrap_delta
from nflsim.features.historical_bakeoff import _score
from nflsim.ridge.bakeoff import (
    QUICK_PENALTY_GRID, QUICK_RECENCY_GRID, PENALTY_GRID, RECENCY_GRID, TargetEvaluator, _load_positions, tune_config,
)
from nflsim.ridge.epa_luck import LUCK_COLUMN, LUCK_GARBAGE_COLUMN, add_luck_adjusted_epa
from nflsim.ridge.expectations import (
    EXPECTATIONS, add_expectations, expected_target, over_expectation_target, production_targets, raw_target,
    weather_effects,
)
from nflsim.ridge.pressure import (
    MEASURES, PRESSURE_PENALTY_GRID, PRESSURE_RECENCY_GRID, PRESSURE_TARGETS, add_pressure_columns, attach_pressure,
    team_game_pressure,
)
from nflsim.ridge.metrics import (
    TARGETS_BY_NAME, RidgeConfig, RidgeTarget, actual_output_frame, build_design, build_game_context, build_target_rows,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "variant_bakeoff"
DEFAULT_TARGETS = ("cpoe", "yac_oe", "rush_yards", "sack", "interception", "first_down", "pass_epa", "rush_epa")
REGIME_GRID = {"team_change_carryover": (1.0, 0.7, 0.4), "coach_change_carryover": (1.0, 0.7, 0.4)}
VARIANTS = ("production", "health", "regime", "xoe", "all", "luck", "luck_gt", "live", *MEASURES)
LIVE_VARIANTS = ("live", *MEASURES)
LUCK_VARIANTS = {"luck": LUCK_COLUMN, "luck_gt": LUCK_GARBAGE_COLUMN}
EPA_TARGETS = ("pass_epa", "rush_epa")
# nflverse CPOE is in percentage points; the raw outcome (complete_pass) is a 0/1 rate.
PRODUCTION_SCALE = {"cpoe": 0.01, "cpoe_short": 0.01, "cpoe_medium": 0.01, "cpoe_deep": 0.01}


def _player_value(inputs_dir: Path, extended: bool) -> pd.DataFrame:
    from nflsim.features.builder import canonicalize_teams
    from nflsim.features.player_value import (
        EXTENDED_GROUP_LIMITS, GROUP_LIMITS, build_player_value_tables, load_pm_estimates,
    )

    groups = EXTENDED_GROUP_LIMITS if extended else GROUP_LIMITS
    frames = {name: canonicalize_teams(pd.read_parquet(inputs_dir / f"{name}.parquet"))
              for name in ("roster", "schedule", "snap_counts")}
    values, *_ = build_player_value_tables(frames["roster"], frames["schedule"], frames["snap_counts"],
                                           load_pm_estimates(groups=groups), groups=groups)
    return values


def _sides(target: RidgeTarget) -> list[str]:
    return [side for side, output in (("subject", target.player_output), ("defense", target.defense_output))
            if output is not None]


def _entity_frame(pbp, target, games, positions, side) -> pd.DataFrame:
    """Entity-game observed outcome and context expectation (the same for every variant)."""
    raw = actual_output_frame(build_target_rows(pbp, raw_target(target), games, positions), side)
    expected = actual_output_frame(build_target_rows(pbp, expected_target(target), games, positions), side)
    frame = raw[["entity", "game_id", "season", "week", "rate", "count"]].merge(
        expected[["entity", "game_id", "rate"]].rename(columns={"rate": "expected"}), on=["entity", "game_id"])
    return frame


def _variant_predictions(target, pbp, games, positions, player_value, side, validation, test, grids, regime, rounds,
                         pressure=None):
    """Tune on validation, return test-season entity-game predictions and the chosen config."""
    rows = build_target_rows(pbp, target, games, positions, player_value=player_value)
    if pressure is not None:
        rows = attach_pressure(rows, pressure)
    design = build_design(rows, target.scope, include_qb_offense=side == "subject",
                          pressure=None if pressure is None else ("pressure", "pressure_forecast"))
    evaluator = TargetEvaluator(rows, design, [side], validation)
    recency, penalties = grids
    if player_value is None:
        penalties = {name: values for name, values in penalties.items() if name != "lambda_player_value"}
    if target.scope != "player":
        penalties = {name: values for name, values in penalties.items() if name != "lambda_offense_team"}
    config = tune_config(evaluator, side, recency, penalties, rounds)
    if regime:
        config = tune_config(evaluator, side, REGIME_GRID, {}, 1, base=config)
    return evaluator.predict(config, test)[side][["entity", "game_id", "pred_full"]], config


def _centered(frame: pd.DataFrame, column: str) -> pd.Series:
    weights = frame["count"]
    week_mean = (frame[column] * weights).groupby([frame["season"], frame["week"]]).transform("sum") \
        / weights.groupby([frame["season"], frame["week"]]).transform("sum")
    return frame[column] - week_mean


def _ridge_target(target: RidgeTarget, variant: str) -> RidgeTarget:
    if variant in ("xoe", "all"):
        return over_expectation_target(target)
    if variant in LIVE_VARIANTS:
        return production_targets([target])[0]
    if variant in LUCK_VARIANTS:
        return replace(target, name=f"{target.name}_{variant}", value_col=LUCK_VARIANTS[variant])
    return target


def _is_over_expectation(target: RidgeTarget) -> bool:
    return target.value_col.endswith("__xoe")


def evaluate(target_name: str, pbp, games, positions, values, validation, test, grids, rounds, bootstrap, seed,
             variants=VARIANTS, reference="production", pressure=None):
    target = TARGETS_BY_NAME[target_name]
    variants = [variant for variant in variants if variant not in LUCK_VARIANTS or target_name in EPA_TARGETS]
    if reference not in variants:
        variants = [reference, *variants]
    xoe = {variant: _is_over_expectation(_ridge_target(target, variant)) for variant in variants}
    results, configs = [], {}
    for side in _sides(target):
        base = _entity_frame(pbp, target, games, positions, side)
        base = base[base["season"] == test]
        predictions = {}
        for variant in variants:
            ridge_target = _ridge_target(target, variant)
            pv = values["extended"] if variant in ("health", "all") else values["base"]
            frame, config = _variant_predictions(ridge_target, pbp, games, positions, pv, side, validation, test,
                                                 grids, variant in ("regime", "all"), rounds,
                                                 (pressure or {}).get(variant))
            if not xoe[variant]:
                frame = frame.assign(pred_full=frame["pred_full"] * PRODUCTION_SCALE.get(target_name, 1.0))
            predictions[variant] = frame.rename(columns={"pred_full": variant})
            configs[f"{target_name}/{side}/{variant}"] = config.__dict__
        scored = base
        for variant, frame in predictions.items():
            scored = scored.merge(frame, on=["entity", "game_id"], how="inner")
        scored = scored.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        no_context = _score(scored["rate"].to_numpy(), scored[reference].to_numpy(), scored["count"].to_numpy())
        context_only = _score(scored["rate"].to_numpy(), scored["expected"].to_numpy(), scored["count"].to_numpy())
        with_context = {}
        for variant in variants:
            effect = scored[variant] if xoe[variant] else _centered(scored, variant)
            with_context[variant] = scored["expected"] + effect
        baseline = with_context[reference]
        for variant in variants:
            score = _score(scored["rate"].to_numpy(), with_context[variant].to_numpy(), scored["count"].to_numpy())
            paired = pd.DataFrame({"entity": scored["entity"], "rate": scored["rate"],
                                   "weighted_prediction": with_context[variant], "bayesian_prediction": baseline})
            low, high = (0.0, 0.0) if variant == reference else _bootstrap_delta(paired, "entity", bootstrap, seed)
            results.append({
                "target": target_name, "side": side, "variant": variant, "reference": reference, "rows": score["rows"],
                "exposure_rmse": score["exposure_rmse"],
                "skill_vs_reference": 1 - score["exposure_rmse"] / _score(
                    scored["rate"].to_numpy(), baseline.to_numpy(), scored["count"].to_numpy())["exposure_rmse"],
                "rmse_delta_ci_low": low, "rmse_delta_ci_high": high,
                "context_only_rmse": context_only["exposure_rmse"], "no_context_rmse": no_context["exposure_rmse"],
            })
        best = min((row for row in results if row["target"] == target_name and row["side"] == side),
                   key=lambda row: row["exposure_rmse"])
        print(f"{target_name}/{side}: best={best['variant']} skill_vs_{reference}={best['skill_vs_reference']:+.4f}",
              flush=True)
    return results, configs


def pressure_term(pbp, games, measure, validation, test):
    """Tune the pressure ridge on the validation season; return all-season team-game forecasts and test skill.

    Skill is against each team-game's league-week mean pressure rate, for the offense (pressure allowed)
    and the defense (pressure rate) outputs.
    """
    rows = build_target_rows(pbp, PRESSURE_TARGETS[measure], games)
    evaluator = TargetEvaluator(rows, build_design(rows, "team"), ["subject", "defense"], validation)
    config = tune_config(evaluator, "subject", PRESSURE_RECENCY_GRID, PRESSURE_PENALTY_GRID, 1)
    terms = []
    for side, frame in evaluator.predict(config, test).items():
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["pred_full"])
        groups = [frame["season"], frame["week"]]
        week_mean = (frame["rate"] * frame["count"]).groupby(groups).transform("sum") \
            / frame["count"].groupby(groups).transform("sum")
        model = _score(frame["rate"].to_numpy(), frame["pred_full"].to_numpy(), frame["count"].to_numpy())
        naive = _score(frame["rate"].to_numpy(), week_mean.to_numpy(), frame["count"].to_numpy())
        terms.append({"measure": measure, "side": side, "test_season": test, "rows": model["rows"],
                      "exposure_rmse": model["exposure_rmse"], "week_mean_rmse": naive["exposure_rmse"],
                      "skill_vs_week_mean": 1 - model["exposure_rmse"] / naive["exposure_rmse"], **config.__dict__})
    return team_game_pressure(pbp, games, measure, config), terms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2016, 2026)))
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--targets", nargs="+", choices=sorted(EXPECTATIONS), default=list(DEFAULT_TARGETS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--reference", choices=VARIANTS, default="production", help="Variant the others are scored against")
    parser.add_argument("--pfr-dir", type=Path, help="Directory with nflverse advstats_week_pass_*.parquet (pfr variant)")
    parser.add_argument("--quick", action="store_true", help="Small tuning grid")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--inputs-dir", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "inputs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    grids = (QUICK_RECENCY_GRID, QUICK_PENALTY_GRID) if args.quick else (RECENCY_GRID, PENALTY_GRID)
    started = time.perf_counter()
    pbp = add_luck_adjusted_epa(add_expectations(load_pbp_data(args.years, cache_path=DEFAULT_CACHE_PATH), args.targets))
    print(f"Expectations fit in {time.perf_counter() - started:.0f}s", flush=True)
    games = build_game_context(pbp)
    positions = _load_positions()
    measures = [variant for variant in args.variants if variant in MEASURES]
    if "pfr" in measures and args.pfr_dir is None:
        raise SystemExit("--pfr-dir is required for the pfr variant")
    if measures:
        pbp = add_pressure_columns(pbp, args.pfr_dir)
    pressure, pressure_terms = {}, []
    for measure in measures:
        pressure[measure], terms = pressure_term(pbp, games, measure, args.validation_season, args.test_season)
        pressure_terms.extend(terms)
        print(f"pressure_{measure}: " + ", ".join(f"{row['side']} skill={row['skill_vs_week_mean']:+.4f}" for row in terms),
              flush=True)
    values = {"base": _player_value(args.inputs_dir, False), "extended": _player_value(args.inputs_dir, True)}

    results, configs = [], {}
    for index, name in enumerate(args.targets):
        rows, chosen = evaluate(name, pbp, games, positions, values, args.validation_season, args.test_season,
                                grids, args.rounds, args.bootstrap_samples, 9100 + index, args.variants,
                                args.reference, pressure)
        results.extend(rows)
        configs.update(chosen)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(results)
    table.to_csv(args.output_dir / "results.csv", index=False)
    if pressure_terms:
        pd.DataFrame(pressure_terms).to_csv(args.output_dir / "pressure_terms.csv", index=False)
    (args.output_dir / "configs.json").write_text(json.dumps(configs, indent=2, sort_keys=True), encoding="utf-8")
    weather = pd.concat([weather_effects(pbp[pbp["game_id"].astype(str).str[:4].astype(int) < args.test_season], name) for name in args.targets],
                        ignore_index=True)
    weather.to_csv(args.output_dir / "weather_effects.csv", index=False)
    print(table.pivot_table(index=["target", "side"], columns="variant", values="skill_vs_reference").round(4))
    print(f"Saved to {args.output_dir} in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
