from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from nflsim.configuration import snapshot
from nflsim.features.builder import build_estimates, overlay, prepare_pbp, wide_tables, skill_roster
from nflsim.paths import ROOT, assert_protected, digest, output_dir, protected_hashes, write_json
from nflsim.features.player_value import (
    DEFAULT_PM_DIR, PV_COLUMNS, build_player_value_tables, load_pm_estimates, pm_input_hashes,
)
from nflsim.features.weather import WEATHER_COLUMNS
from nflsim.semantics import (
    ISOLATED_MU_CONTEXT_V2, ISOLATED_MU_CONTEXT_V2_DEFINITION, ISOLATED_MU_CONTEXT_V3,
    ISOLATED_MU_CONTEXT_V3_DEFINITION, ISOLATED_MU_DEFINITION, ISOLATED_MU_V1,
    ISOLATED_SEMANTICS, semantics_of,
)
from nflsim.features.retained import regenerate_retained_features
from nflsim.ridge.metrics import TARGETS
from nflsim.ridge.expectations import PRODUCTION_XOE_TARGETS, production_targets, with_production_expectations


def _season(game_ids: pd.Series) -> pd.Series:
    return pd.to_numeric(game_ids.astype(str).str.extract(r"^(\d{4})_", expand=False), errors="coerce")


def _frame_hash(frame: pd.DataFrame) -> str:
    frame = frame.copy().reset_index(drop=True)
    string_columns = {
        "game_id", "home_team", "away_team", "stadium_id", "game_stadium", "stadium", "location",
        "posteam", "defteam", "receiver_player_id", "passer_player_id", "rusher_player_id", "depth_bucket", "position",
    }
    date_columns = {"game_date", "gameday"}
    for column in frame:
        if column in string_columns:
            frame[column] = frame[column].astype("string").fillna("<NA>")
        elif column in date_columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce").astype("int64")
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    hasher = hashlib.sha256()
    hasher.update("\0".join(map(str, frame.columns)).encode("utf-8"))
    hasher.update(pd.util.hash_pandas_object(frame, index=False, categorize=True).values.tobytes())
    return hasher.hexdigest()


SELECTION_PBP_COLUMNS = (
    "game_id", "play_id", "game_date", "gameday", "home_team", "away_team", "stadium_id", "game_stadium",
    "stadium", "location", "posteam", "defteam", "receiver_player_id", "passer_player_id",
    "rusher_player_id", "pass_attempt", "complete_pass", "target", "first_down", "yards_gained",
    "rushing_yards", "rush_attempt", "rush", "pass", "qb_scramble", "sack", "interception",
    "is_interception_worthy", "cpoe", "yac_oe", "yards_after_catch", "xyac_mean_yardage",
    "air_yards", "depth_bucket", "explosive",
)
SELECTION_SCHEDULE_COLUMNS = (
    "game_id", "game_date", "gameday", "home_team", "away_team", "stadium_id", "game_stadium",
    "stadium", "location",
)


def _stable_projection(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return frame.reindex(columns=columns)


def selection_input_hashes(pbp: pd.DataFrame, schedule: pd.DataFrame, roster: pd.DataFrame,
                           player_value: pd.DataFrame | None = None) -> dict[str, str]:
    schedule = schedule.loc[_season(schedule.game_id).le(2024)].copy()
    game_ids = set(schedule.game_id.astype(str))
    pbp = pbp.loc[pbp.game_id.astype(str).isin(game_ids)].copy()
    roster, _ = skill_roster(roster)
    roster = roster.loc[roster.game_id.astype(str).isin(game_ids)].copy()
    hashes = {
        "pbp_through_2024": _frame_hash(_stable_projection(pbp, SELECTION_PBP_COLUMNS).sort_values(["game_id", "play_id"]).reset_index(drop=True)),
        "schedule_through_2024": _frame_hash(_stable_projection(schedule, SELECTION_SCHEDULE_COLUMNS).sort_values("game_id").reset_index(drop=True)),
        "roster_through_2024": _frame_hash(roster.sort_values(["game_id", "team", "player_id"]).reset_index(drop=True)),
    }
    if player_value is not None:
        values = player_value.loc[player_value.game_id.astype(str).isin(game_ids), ["game_id", "team", *PV_COLUMNS]]
        values = values.astype({"game_id": str, "team": str}).sort_values(["game_id", "team"]).reset_index(drop=True)
        hashes["player_value_through_2024"] = _frame_hash(values.rename(columns={"team": "posteam"}).round(10))
    return hashes


def select_parameters(pbp, schedule, roster, quick=False, player_value=None):
    roster, _ = skill_roster(roster)
    from nflsim.ridge.metrics import build_design, build_game_context, build_target_rows
    from nflsim.ridge.bakeoff import (
        PENALTY_GRID, RECENCY_GRID, QUICK_PENALTY_GRID, QUICK_RECENCY_GRID,
        TargetEvaluator, tune_config,
    )
    schedule = schedule[_season(schedule.game_id).le(2024)]
    history = with_production_expectations(prepare_pbp(pbp))
    history = history[history.game_id.isin(schedule.game_id)]
    positions = roster[roster.game_id.isin(schedule.game_id)].drop_duplicates("player_id", keep="last").set_index("player_id").position.to_dict()
    games = build_game_context(schedule)
    parameters = {}
    for target in production_targets(TARGETS):
        rows = build_target_rows(history, target, games, positions, player_value=player_value)
        if rows.empty or not rows.season.eq(2024).any():
            raise ValueError(f"Cannot select {target.name} on 2024")
        outputs = [(side, metric) for side, metric in (("subject", target.player_output), ("defense", target.defense_output)) if metric]
        designs = {side: build_design(rows, target.scope, include_qb_offense=side == "subject")
               for side, _ in outputs}
        evaluator = TargetEvaluator(rows, designs, [side for side, _ in outputs], 2024)
        for side, metric in outputs:
            penalties = QUICK_PENALTY_GRID if quick else PENALTY_GRID
            if target.scope == "team":
                penalties = {name: values for name, values in penalties.items() if name != "lambda_offense_team"}
            if player_value is None:
                penalties = {name: values for name, values in penalties.items() if name != "lambda_player_value"}
            config = tune_config(evaluator, side, QUICK_RECENCY_GRID if quick else RECENCY_GRID, penalties, 1 if quick else 2)
            parameters[metric] = {"target": target.name, "side": side, **config.__dict__}
            print(f"[select] {metric}: {config.label()}", flush=True)
    return {"validation_season": 2024, "max_selection_season": 2024, "method": "reproduced_2024_coordinate_search", "quick": quick,
            "player_value": player_value is not None, "over_expectation_targets": list(PRODUCTION_XOE_TARGETS),
            "parameters": parameters}


def player_value_inputs(schedule, roster, snap_counts, pm_estimates=None, pm_dir=DEFAULT_PM_DIR):
    """Pregame team PV and the lookup tables simulation needs to recompute custom lineups."""
    estimates = load_pm_estimates(pm_dir) if pm_estimates is None else pm_estimates
    values, components, history, defaults = build_player_value_tables(roster, schedule, snap_counts, estimates)
    return {"values": values, "components": components, "history": history, "defaults": defaults, "estimates": estimates}


def reuse_retained_tables(directory, output, input_paths, seasons, excluded_sources):
    directory = Path(directory)
    manifest = json.loads((directory / "retained_manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Retained feature source is incomplete")
    expected = {Path(name).name: checksum for name, checksum in manifest["input_hashes"].items()}
    actual = {Path(path).name: digest(path) for path in input_paths}
    if actual != expected:
        raise ValueError("Retained feature inputs do not match frozen source")
    if not set(seasons) <= set(manifest["prediction_seasons"]):
        raise ValueError("Retained source does not cover requested seasons")
    if set(excluded_sources) != set(manifest["excluded_sources"]):
        raise ValueError("Retained source excludes different ridge metrics")
    for name, checksum in manifest["implementation_hashes"].items():
        if digest(Path(name)) != checksum:
            raise ValueError(f"Retained implementation changed: {name}")
    for name, checksum in manifest["tables"].items():
        if digest(directory / name) != checksum:
            raise ValueError(f"Retained table changed: {name}")
        shutil.copy2(directory / name, output / name)
    manifest["reused_from"] = str(directory.resolve())
    write_json(output / "retained_manifest.json", manifest)
    players = pd.read_csv(output / "retained_player_stats.csv", dtype={"player_id": str, "game_id": str})
    teams = pd.read_csv(output / "retained_team_stats.csv", dtype={"team": str, "game_id": str})
    diagnostics = pd.read_csv(directory / "retained_diagnostics.csv", low_memory=False)
    return players.loc[_season(players.game_id).isin(seasons)], teams.loc[_season(teams.game_id).isin(seasons)], diagnostics


def build_feature_tables(pbp, schedule, roster, snap_counts, parameters_path, output_path,
                         *, reselect_if_training_history_changed=False, source_hashes=None, input_paths=(),
                         prediction_seasons=None, prior_bundle=None, pm_estimates=None, pm_dir=DEFAULT_PM_DIR,
                         feature_semantics=ISOLATED_MU_V1, retained_from=None):
    if prior_bundle is not None and semantics_of(prior_bundle) not in ISOLATED_SEMANTICS:
        raise ValueError("Cannot refresh from a legacy-conditioned bundle; historical tables would be mislabelled as isolated mus")
    if feature_semantics not in ISOLATED_SEMANTICS:
        raise ValueError(f"Unsupported feature semantics {feature_semantics}")
    output = output_dir(output_path)
    if (output / "bundle_manifest.json").exists() or (output / "training_manifest.json").exists():
        raise ValueError("Cannot rebuild features beneath trained/sealed models; use a new isolated output directory")
    parameters_path = Path(parameters_path)
    if prior_bundle is not None and semantics_of(prior_bundle) != feature_semantics:
        raise ValueError(f"Cannot refresh {feature_semantics} features from a {semantics_of(prior_bundle)} bundle")
    if not (ROOT / "config/model_manifest.json").exists():
        snapshot()
    if prior_bundle is not None:
        prior_xoe = json.loads((Path(prior_bundle) / "feature_manifest.json").read_text()).get("over_expectation_targets", [])
        if prior_xoe != list(PRODUCTION_XOE_TARGETS):
            raise ValueError("Prior bundle's ridge targets use different over-expectation definitions; rebuild all seasons")
    pbp = with_production_expectations(prepare_pbp(pbp))
    pv = player_value_inputs(schedule, roster, snap_counts, pm_estimates, pm_dir)
    expected_inputs = selection_input_hashes(pbp, schedule, roster, pv["values"])
    if parameters_path.exists():
        selection = json.loads(parameters_path.read_text())
    else:
        selection = {}
    selection_valid = (
        selection.get("max_selection_season", 9999) <= 2024
        and selection.get("validation_season") == 2024
        and selection.get("over_expectation_targets", []) == list(PRODUCTION_XOE_TARGETS)
        and (selection.get("input_hashes") == expected_inputs
             or bool(selection.get("attested_from_bundle_selection")))
    )
    if not selection_valid:
        if not reselect_if_training_history_changed:
            raise ValueError("Verified frozen 2016-2024 ridge parameters required; run parameter selection once")
        selection = {**select_parameters(pbp, schedule, roster, player_value=pv["values"]), "input_hashes": expected_inputs}
        write_json(parameters_path, selection)
    latest_season = int(_season(schedule.game_id).max())
    seasons_to_build = sorted(set(int(value) for value in prediction_seasons)) if prediction_seasons is not None else list(range(2017, latest_season + 1))
    player, team, context, diagnostics = build_estimates(
        pbp, schedule, roster, selection["parameters"], targets=production_targets(TARGETS), predict_seasons=seasons_to_build,
        player_value=pv["values"], feature_semantics=feature_semantics,
    )
    if player.empty or team.empty or set(player.source) | set(team.source) != set(selection["parameters"]):
        raise ValueError("Incomplete ridge metric coverage; more warm-up observations required")
    refresh_from = min(seasons_to_build)
    if retained_from is not None:
        retained_player, retained_team, retained_diagnostics = reuse_retained_tables(
            retained_from, output, input_paths, seasons_to_build, selection["parameters"])
    else:
        retained_player, retained_team, retained_diagnostics = regenerate_retained_features(
            pbp, schedule, skill_roster(roster)[0], snap_counts, output,
            excluded_sources=selection["parameters"], seasons=seasons_to_build,
            input_paths=input_paths, prior_bundle=prior_bundle,
        )
    if prior_bundle is not None:
        retained_player = retained_player.loc[_season(retained_player.game_id).isin(seasons_to_build)]
        retained_team = retained_team.loc[_season(retained_team.game_id).isin(seasons_to_build)]
    player_current = overlay(retained_player, player, "player_id")
    team_current = overlay(retained_team, team, "team")
    if prior_bundle is not None:
        prior_bundle = Path(prior_bundle)
        prior_player = pd.read_csv(prior_bundle / "player_stats.csv", dtype={"player_id": str, "game_id": str})
        prior_team = pd.read_csv(prior_bundle / "Team_Stats.csv", dtype={"team": str, "game_id": str})
        prior_player = prior_player.loc[_season(prior_player.game_id).lt(refresh_from)]
        prior_team = prior_team.loc[_season(prior_team.game_id).lt(refresh_from)]
        player_long = pd.concat([prior_player, player_current], ignore_index=True, sort=False)
        team_long = pd.concat([prior_team, team_current], ignore_index=True, sort=False)
        prior_context_path = prior_bundle / "ridge_context.csv"
        prior_context = pd.read_csv(prior_context_path, dtype={"game_id": str, "team": str})
        prior_context = prior_context.loc[_season(prior_context.game_id).lt(refresh_from)]
        context = pd.concat([prior_context, context], ignore_index=True, sort=False)
        prior_fallbacks_path = prior_bundle / "ridge_fallbacks.csv"
        if prior_fallbacks_path.exists():
            prior_fallbacks = pd.read_csv(prior_fallbacks_path, low_memory=False)
            if "season" in prior_fallbacks:
                prior_fallbacks = prior_fallbacks.loc[pd.to_numeric(prior_fallbacks.season, errors="coerce").lt(refresh_from)]
            diagnostics = pd.concat([prior_fallbacks, diagnostics], ignore_index=True, sort=False)
    else:
        player_long, team_long = player_current, team_current
    covered_games = set(player_long.game_id) & set(team_long.game_id)
    player_long = player_long[player_long.game_id.isin(covered_games)]
    team_long = team_long[team_long.game_id.isin(covered_games)]
    player_wide, team_wide = wide_tables(player_long, team_long, context, pv["values"])
    coefficient_columns = ["season", "week", "metric", "side", "term", "beta"]
    if "reason" in diagnostics:
        is_pv_coefficient = diagnostics.reason.eq("player_value_coefficient")
        is_qb_coefficient = diagnostics.reason.eq("offense_qb_coefficient")
        is_coefficient = is_pv_coefficient | is_qb_coefficient
        pv_coefficients = diagnostics.loc[is_coefficient, ["season", "week", "metric", "side", "entity", "estimate"]].rename(
            columns={"entity": "term", "estimate": "beta"})
        pv_coefficients.loc[is_qb_coefficient.loc[pv_coefficients.index], "term"] = (
            "offense_qb:" + pv_coefficients.loc[is_qb_coefficient.loc[pv_coefficients.index], "term"].astype(str))
        diagnostics = diagnostics.loc[~is_coefficient]
    else:
        pv_coefficients = pd.DataFrame(columns=coefficient_columns)
    if prior_bundle is not None and (Path(prior_bundle) / "ridge_pv_coefficients.csv").exists():
        prior_coefficients = pd.read_csv(Path(prior_bundle) / "ridge_pv_coefficients.csv")
        pv_coefficients = pd.concat([prior_coefficients[prior_coefficients.season.lt(refresh_from)], pv_coefficients], ignore_index=True)
    for name, frame in (("player_value.csv", pv["values"]), ("player_value_components.csv", pv["components"]),
                        ("pm_estimates.csv", pv["estimates"]), ("snap_share_history.csv", pv["history"]),
                        ("ridge_pv_coefficients.csv", pv_coefficients)):
        frame.to_csv(output / name, index=False)
    write_json(output / "pv_share_defaults.json", {f"{group}|{rank}": value for (group, rank), value in pv["defaults"].items()})
    for name, frame in (("player_stats.csv", player_long), ("Team_Stats.csv", team_long),
                        ("player_stats_wide.csv", player_wide), ("team_stats_wide.csv", team_wide),
                        ("ridge_context.csv", context), ("ridge_fallbacks.csv", diagnostics)):
        frame.to_csv(output / name, index=False)
    (output / "retained_diagnostics.csv").write_text(retained_diagnostics.to_csv(index=False), encoding="utf-8")
    v2 = feature_semantics in {ISOLATED_MU_CONTEXT_V2, ISOLATED_MU_CONTEXT_V3}
    v2_definition = ISOLATED_MU_CONTEXT_V3_DEFINITION if feature_semantics == ISOLATED_MU_CONTEXT_V3 else ISOLATED_MU_CONTEXT_V2_DEFINITION
    paired = {spec.get("target") for spec in selection["parameters"].values() if spec.get("side") == "subject"} if v2 else set()
    manifest = {"estimator": "ridge", "feature_semantics": feature_semantics,
                "isolated_mu": {**(v2_definition if v2 else ISOLATED_MU_DEFINITION),
                                "metrics": {metric: {"side": spec.get("side"), "target": spec.get("target"),
                                                   "block": "defense" if spec.get("side") == "defense" else "subject",
                                                   **({"fit": "subject"} if v2 and spec.get("side") == "defense" and spec.get("target") in paired else {})}
                                            for metric, spec in sorted(selection["parameters"].items())}},
                "covered_metrics": sorted(selection["parameters"]),
                "over_expectation_targets": list(PRODUCTION_XOE_TARGETS),
                "retained_player_metrics": sorted(set(player_long.source)-set(player.source)),
                "retained_team_metrics": sorted(set(team_long.source)-set(team.source)),
                "selection_sha256": digest(parameters_path),
                "selection_input_hashes": expected_inputs,
                "tables": {name: digest(output/name) for name in ("player_stats_wide.csv", "team_stats_wide.csv", "ridge_context.csv")},
                "input_hashes": source_hashes or {},
                "latest_schedule_season": latest_season,
                "prediction_seasons_rebuilt": seasons_to_build,
                "historical_tables_reused_from": str(prior_bundle) if prior_bundle is not None else None,
                "ridge_context": "metric-specific same-subject-fit venue/offense marginal coefficients (separate context, not folded into mus)",
                "uncertainty": "unavailable", "direct_tendency_lag_count": 1,
                "qb_team_games_played_training_caveat": "same-game pass-attempt participation selected the historical player association",
                "player_value": {"columns": PV_COLUMNS, "pm_inputs": pm_input_hashes(pm_dir) if Path(pm_dir).is_dir() else "unavailable",
                                 "definition": "strictly prior player_value box estimate, snap-share weights normalized per group, "
                                               "cross-season decay 0.8, 25th pct replacement; raw values are direct Torch inputs and joint-ridge regressors",
                                 "tables": {name: digest(output / name) for name in ("player_value.csv", "ridge_pv_coefficients.csv")}},
                "weather": {"columns": list(WEATHER_COLUMNS),
                            "definition": "outdoor-only temperature (F, centered at 60 and scaled by 20) and wind (mph scaled by 15), clipped to [-2, 2]; missing outdoor weather is flagged",
                            "source": "historical game-observed PBP/schedule weather; future games need forecast inputs"}}
    write_json(output / "feature_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build isolated preweek ridge tables from explicit frozen input files")
    parser.add_argument("--pbp", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--snap-counts", type=Path, required=True,
                        help="Offline joined snap counts: game_type/offense_snaps/position/gsis_id/name/team/game_id")
    parser.add_argument("--parameters", type=Path, default=ROOT / "config/ridge_parameters.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/runtime")
    parser.add_argument("--select-2024", action="store_true")
    parser.add_argument("--quick-selection", action="store_true")
    parser.add_argument("--pm-dir", type=Path, default=DEFAULT_PM_DIR, help="Saved player_value weekly estimate CSVs (read-only)")
    parser.add_argument("--feature-semantics", choices=ISOLATED_SEMANTICS, default=ISOLATED_MU_V1)
    parser.add_argument("--retained-from", type=Path)
    args = parser.parse_args()
    before = protected_hashes()
    try:
        if not (ROOT / "config/model_manifest.json").exists():
            snapshot()
        load = lambda path: pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype={"game_id": str, "player_id": str})
        pbp, schedule, roster, snap_counts = [load(path) for path in (args.pbp, args.schedule, args.roster, args.snap_counts)]
        estimates = load_pm_estimates(args.pm_dir)
        if args.select_2024:
            values = player_value_inputs(schedule, roster, snap_counts, estimates)["values"]
            write_json(args.parameters, {**select_parameters(pbp, schedule, roster, args.quick_selection, player_value=values),
                                         "input_hashes": selection_input_hashes(pbp, schedule, roster, values)})
        build_feature_tables(pbp, schedule, roster, snap_counts, args.parameters, args.output_dir,
                     source_hashes={"pbp": digest(args.pbp), "schedule": digest(args.schedule),
                                            "roster": digest(args.roster), "snap_counts": digest(args.snap_counts),
                                            **pm_input_hashes(args.pm_dir)},
                             input_paths=(args.pbp, args.schedule, args.roster, args.snap_counts),
                             pm_estimates=estimates, pm_dir=args.pm_dir, feature_semantics=args.feature_semantics,
                             retained_from=args.retained_from)
    finally:
        assert_protected(before)


if __name__ == "__main__":
    main()