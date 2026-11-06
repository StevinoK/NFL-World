from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ridge_game_sim.configuration import snapshot
from ridge_game_sim.feature_builder import build_estimates, overlay, prepare_pbp, wide_tables, skill_roster
from ridge_game_sim.paths import ROOT, assert_protected, digest, output_dir, protected_hashes, write_json
from ridge_game_sim.retained_features import regenerate_retained_features


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


def selection_input_hashes(pbp: pd.DataFrame, schedule: pd.DataFrame, roster: pd.DataFrame) -> dict[str, str]:
    schedule = schedule.loc[_season(schedule.game_id).le(2024)].copy()
    game_ids = set(schedule.game_id.astype(str))
    pbp = pbp.loc[pbp.game_id.astype(str).isin(game_ids)].copy()
    roster, _ = skill_roster(roster)
    roster = roster.loc[roster.game_id.astype(str).isin(game_ids)].copy()
    return {
        "pbp_through_2024": _frame_hash(_stable_projection(pbp, SELECTION_PBP_COLUMNS).sort_values(["game_id", "play_id"]).reset_index(drop=True)),
        "schedule_through_2024": _frame_hash(_stable_projection(schedule, SELECTION_SCHEDULE_COLUMNS).sort_values("game_id").reset_index(drop=True)),
        "roster_through_2024": _frame_hash(roster.sort_values(["game_id", "team", "player_id"]).reset_index(drop=True)),
    }


def select_parameters(pbp, schedule, roster, quick=False):
    roster, _ = skill_roster(roster)
    from ridge_adjusted_metrics import TARGETS, build_design, build_game_context, build_target_rows
    from run_ridge_adjusted_bakeoff import (
        PENALTY_GRID, RECENCY_GRID, QUICK_PENALTY_GRID, QUICK_RECENCY_GRID,
        TargetEvaluator, tune_config,
    )
    schedule = schedule[_season(schedule.game_id).le(2024)]
    history = prepare_pbp(pbp[pbp.game_id.isin(schedule.game_id)])
    positions = roster[roster.game_id.isin(schedule.game_id)].drop_duplicates("player_id", keep="last").set_index("player_id").position.to_dict()
    games = build_game_context(schedule)
    parameters = {}
    for target in TARGETS:
        rows = build_target_rows(history, target, games, positions)
        if rows.empty or not rows.season.eq(2024).any():
            raise ValueError(f"Cannot select {target.name} on 2024")
        outputs = [(side, metric) for side, metric in (("subject", target.player_output), ("defense", target.defense_output)) if metric]
        evaluator = TargetEvaluator(rows, build_design(rows, target.scope), [side for side, _ in outputs], 2024)
        for side, metric in outputs:
            penalties = QUICK_PENALTY_GRID if quick else PENALTY_GRID
            if target.scope == "team":
                penalties = {name: values for name, values in penalties.items() if name != "lambda_offense_team"}
            config = tune_config(evaluator, side, QUICK_RECENCY_GRID if quick else RECENCY_GRID, penalties, 1 if quick else 2)
            parameters[metric] = {"target": target.name, "side": side, **config.__dict__}
    return {"validation_season": 2024, "max_selection_season": 2024, "method": "reproduced_2024_coordinate_search", "quick": quick, "parameters": parameters}


def build_feature_tables(pbp, schedule, roster, snap_counts, parameters_path, output_path,
                         *, reselect_if_training_history_changed=False, source_hashes=None, input_paths=(),
                         prediction_seasons=None, prior_bundle=None):
    output = output_dir(output_path)
    if (output / "bundle_manifest.json").exists() or (output / "training_manifest.json").exists():
        raise ValueError("Cannot rebuild features beneath trained/sealed models; use a new isolated output directory")
    parameters_path = Path(parameters_path)
    if not (ROOT / "config/model_manifest.json").exists():
        snapshot()
    expected_inputs = selection_input_hashes(pbp, schedule, roster)
    if parameters_path.exists():
        selection = json.loads(parameters_path.read_text())
    else:
        selection = {}
    selection_valid = (
        selection.get("max_selection_season", 9999) <= 2024
        and selection.get("validation_season") == 2024
        and selection.get("input_hashes") == expected_inputs
    )
    if not selection_valid:
        if not reselect_if_training_history_changed:
            raise ValueError("Verified frozen 2016-2024 ridge parameters required; run parameter selection once")
        selection = {**select_parameters(pbp, schedule, roster), "input_hashes": expected_inputs}
        write_json(parameters_path, selection)
    latest_season = int(_season(schedule.game_id).max())
    seasons_to_build = sorted(set(int(value) for value in prediction_seasons)) if prediction_seasons is not None else list(range(2017, latest_season + 1))
    player, team, context, diagnostics = build_estimates(
        pbp, schedule, roster, selection["parameters"], predict_seasons=seasons_to_build,
    )
    if player.empty or team.empty or set(player.source) | set(team.source) != set(selection["parameters"]):
        raise ValueError("Incomplete ridge metric coverage; more warm-up observations required")
    refresh_from = min(seasons_to_build)
    retained_player, retained_team, retained_diagnostics = regenerate_retained_features(
        pbp, schedule, skill_roster(roster)[0], snap_counts, output,
        excluded_sources=selection["parameters"],
        seasons=seasons_to_build,
        input_paths=input_paths,
        prior_bundle=prior_bundle,
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
    player_wide, team_wide = wide_tables(player_long, team_long, context)
    for name, frame in (("player_stats.csv", player_long), ("Team_Stats.csv", team_long),
                        ("player_stats_wide.csv", player_wide), ("team_stats_wide.csv", team_wide),
                        ("ridge_context.csv", context), ("ridge_fallbacks.csv", diagnostics)):
        frame.to_csv(output / name, index=False)
    (output / "retained_diagnostics.csv").write_text(retained_diagnostics.to_csv(index=False), encoding="utf-8")
    manifest = {"estimator": "ridge", "covered_metrics": sorted(selection["parameters"]),
                "retained_player_metrics": sorted(set(player_long.source)-set(player.source)),
                "retained_team_metrics": sorted(set(team_long.source)-set(team.source)),
                "selection_sha256": digest(parameters_path),
                "selection_input_hashes": expected_inputs,
                "tables": {name: digest(output/name) for name in ("player_stats_wide.csv", "team_stats_wide.csv", "ridge_context.csv")},
                "input_hashes": source_hashes or {},
                "latest_schedule_season": latest_season,
                "prediction_seasons_rebuilt": seasons_to_build,
                "historical_tables_reused_from": str(prior_bundle) if prior_bundle is not None else None,
                "ridge_context": "metric-specific same-subject-fit venue/offense marginal coefficients",
                "uncertainty": "unavailable", "direct_tendency_lag_count": 1,
                "qb_team_games_played_training_caveat": "same-game pass-attempt participation selected the historical player association"}
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
    args = parser.parse_args()
    before = protected_hashes()
    try:
        if not (ROOT / "config/model_manifest.json").exists():
            snapshot()
        load = lambda path: pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype={"game_id": str, "player_id": str})
        pbp, schedule, roster, snap_counts = [load(path) for path in (args.pbp, args.schedule, args.roster, args.snap_counts)]
        if args.select_2024:
            write_json(args.parameters, {**select_parameters(pbp, schedule, roster, args.quick_selection), "input_hashes": selection_input_hashes(pbp, schedule, roster)})
        build_feature_tables(pbp, schedule, roster, snap_counts, args.parameters, args.output_dir,
                     source_hashes={"pbp": digest(args.pbp), "schedule": digest(args.schedule),
                                            "roster": digest(args.roster), "snap_counts": digest(args.snap_counts)},
                             input_paths=(args.pbp, args.schedule, args.roster, args.snap_counts))
    finally:
        assert_protected(before)


if __name__ == "__main__":
    main()