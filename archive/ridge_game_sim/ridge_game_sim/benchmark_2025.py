from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from game_sim_execution import PreparedGame, build_chunk_tasks
from ridge_game_sim.feature_builder import canonicalize_teams
from ridge_game_sim.bundle import validate_bundle, verify_bundle_hash
from ridge_game_sim.modal_worker import simulate_chunk
from ridge_game_sim.paths import APP_NAME, ROOT, VOLUME_NAME, assert_protected, check_remote_names, digest, inside, output_dir, protected_hashes, write_json
from ridge_game_sim.runtime import RidgeRuntime
from ridge_game_sim.storage import ChunkStore, json_value, task_hash


def redirect_outputs(module, output):
    output = output_dir(output)
    for name in vars(module):
        value = getattr(module, name)
        if name.endswith("_PATH") and isinstance(value, Path):
            setattr(module, name, output / value.name)
    module.RAW_RESULTS_DIR = output_dir(output / "raw_games")


def validate_frozen_venue(game_row, team_stats):
    schedule_column = next((column for column in ("stadium_id", "game_stadium", "stadium") if hasattr(game_row, column)), None)
    scheduled = getattr(game_row, schedule_column) if schedule_column else None
    if scheduled is None or pd.isna(scheduled) or not str(scheduled).strip():
        raise ValueError("Scheduled venue identity missing; refusing home-team stadium assumption")
    frozen_column = next((column for column in ("venue", "stadium_id", "game_stadium", "stadium") if column in team_stats), None)
    status = {"game_id": str(game_row.game_id), "scheduled_venue": str(scheduled), "schedule_column": schedule_column, "frozen_column": frozen_column}
    if frozen_column is None:
        return {**status, "status": "unavailable", "reason": "Frozen feature files lack venue identity; persist the feature-generation venue to validate the fresh schedule"}
    current = team_stats[team_stats.game_id.astype(str).eq(str(game_row.game_id))]
    missing = []
    for team in (str(game_row.home_team), str(game_row.away_team)):
        values = current.loc[current.team.astype(str).eq(team), frozen_column]
        present = values.dropna().astype(str)
        present = present[present.str.strip().ne("")]
        if present.ne(str(scheduled)).any():
            raise ValueError(f"{game_row.game_id}/{team}: scheduled venue {scheduled!r} differs from frozen venue {present.tolist()!r}")
        if len(values) != 1 or len(present) != 1:
            missing.append(team)
    if missing:
        return {**status, "status": "unavailable", "reason": f"Frozen venue identity missing or ambiguous for teams {missing}"}
    return {**status, "status": "validated"}


def make_collector(args, output):
    os.environ["FOOTBALL_ARTIFACT_DIR"] = str(args.artifact_dir)
    import torch_game_simulations_2025_backtest as source
    redirect_outputs(source, output)
    class HistoricalCollector(source.GameBacktester):
        def _load_schedule(self):
            return canonicalize_teams(super()._load_schedule())

        def _load_rosters(self):
            return canonicalize_teams(super()._load_rosters())

        def _load_depth_charts(self):
            return canonicalize_teams(super()._load_depth_charts())

        def _get_pbp_data(self):
            return canonicalize_teams(super()._get_pbp_data())

        def _get_actual_player_weekly(self):
            return canonicalize_teams(super()._get_actual_player_weekly())

        def _get_actual_player_game_stats(self):
            return canonicalize_teams(super()._get_actual_player_game_stats())

        def _get_actual_team_game_stats(self):
            return canonicalize_teams(super()._get_actual_team_game_stats())

        def __init__(self):
            self.season, self.simulations, self.workers = 2025, args.simulations, 1
            self.week_start, self.week_end, self.game_id = args.week_start, args.week_end, args.game_id
            self.resume, self.reset_artifacts = True, False
            self.print_game_summaries, self.strict_schema, self.attempt_diagnostics, self.live_pbp = False, True, True, False
            self.raw_results_dir = output_dir(output / "raw_games")
            self.runtime = RidgeRuntime(args.artifact_dir, strict_schema=True)
            self.schedule, self.rosters, self.depth_charts = self._load_schedule(), self._load_rosters(), self._load_depth_charts()
            if args.max_games:
                self.schedule = self.schedule.head(args.max_games)
            self.player_stats = self.runtime.player_stats_wide.copy()
            self.team_stats = self.runtime.team_stats_wide.copy()
            self.player_stats["game_date"] = pd.to_datetime(self.player_stats.game_date)
            self.team_stats["game_date"] = pd.to_datetime(self.team_stats.game_date)
            for name in ("_pbp_cache", "_actual_player_weekly_cache", "_actual_player_game_stats_cache", "_actual_team_game_stats_cache", "_actual_game_fumble_stats_cache", "_actual_team_penalty_stats_cache"):
                setattr(self, name, None)
            self.results, self.missing = pd.DataFrame(), pd.DataFrame()
            self.venue_validation = {}
            self.ridge_priors = pd.read_csv(args.artifact_dir / "ridge_fallbacks.csv")

        def run(self, *args, **kwargs):
            raise RuntimeError("Collector cannot run the base simulation loop")

        def _build_team_slots(self, team, week, gameday):
            base = super()._build_team_slots(team, week, gameday)
            roster = self.rosters[(self.rosters.team == team) & self.rosters.week.eq(week)].copy()
            if "report_status" in roster:
                roster = roster[~roster.report_status.eq("Out")]
            roster = roster[roster.status.eq("ACT")] if roster.status.eq("ACT").any() else roster
            game_ids = self.schedule.loc[self.schedule.week.eq(week) & (self.schedule.home_team.eq(team) | self.schedule.away_team.eq(team)), "game_id"]
            current = self.player_stats[self.player_stats.game_id.isin(game_ids)]
            roster = roster.merge(current[["player_id", "medium", "rush_rate"]], on="player_id", how="left", validate="many_to_one")
            slots = {"QB1": base["QB1"]}
            for position, limit in (("WR", 4), ("RB", 3), ("TE", 3)):
                score = "rush_rate" if position == "RB" else "medium"
                ordered = roster[roster.position.eq(position)].sort_values([score, "player_id"], ascending=[False, True]).head(limit)
                records = ordered.to_dict("records")
                for rank in range(1, limit+1):
                    if rank <= len(records):
                        row = records[rank-1]
                        slots[f"{position}{rank}"] = {"player_id": str(row["player_id"]), "player_name": str(row.get("player_name", row["player_id"])), "position": position}
                    else:
                        slots[f"{position}{rank}"] = {"player_id": f"placeholder_{team}_{position}_{week}_{rank}", "player_name": f"{team} {position} Placeholder {rank}", "position": position}
            return slots

        def _build_player_objects(self, slot_map, team, week, gameday):
            players = super()._build_player_objects(slot_map, team, week, gameday)
            from ridge_adjusted_metrics import TARGETS
            metrics = [target.player_output for target in TARGETS if target.scope == "player" and target.player_output]
            current_game = self.schedule.loc[self.schedule.week.eq(week) & (self.schedule.home_team.eq(team) | self.schedule.away_team.eq(team)), "game_id"]
            current = self.player_stats[self.player_stats.game_id.isin(current_game)].set_index("player_id")
            priors = self.ridge_priors[self.ridge_priors.season.eq(2025) & self.ridge_priors.week.eq(week) & self.ridge_priors.reason.eq("position_prior")].set_index(["metric", "entity"])
            for player in players:
                for metric in metrics:
                    value = current.loc[player.player_id, metric] if player.player_id in current.index else np.nan
                    if pd.isna(value):
                        position = player.position[:-1]
                        key = (metric, position) if (metric, position) in priors.index else (metric, "OTHER")
                        if key not in priors.index:
                            raise source.SkipGameError(f"Missing preweek ridge prior {metric}/{position}")
                        value = priors.loc[key, "estimate"]
                    player.attributes.loc[player.attributes.index[0], metric] = float(value)
            return players

        def _build_game_package(self, game_row):
            self.venue_validation[str(game_row.game_id)] = validate_frozen_venue(game_row, self.team_stats)
            package = super()._build_game_package(game_row)
            roof = getattr(game_row, "roof", None)
            if roof is None or pd.isna(roof):
                raise source.SkipGameError("Scheduled venue roof missing; refusing home-team stadium assumption")
            package["roof"] = str(roof)
            for team in (package["home_team"], package["away_team"]):
                attrs = package["team_features"].loc[team]
                for model in self.runtime.context_map:
                    self.runtime.checked_context(model, attrs)
            return package
    return HistoricalCollector(), source


def pending_tasks(games, directory, parallelism, store):
    for task in build_chunk_tasks(games, directory, parallelism):
        wanted = list(range(int(task["start_simulation"]), int(task["start_simulation"])+int(task["count"])))
        ids = [simulation for simulation in wanted if (task["game_id"], simulation) not in store.completed]
        if not ids:
            continue
        task.update(simulation_ids=ids, count=len(ids), run_id=store.identity["run_id"], bundle_hash=store.identity["bundle_hash"], seed=store.identity["seed"])
        task["task_hash"] = task_hash(task)
        yield task


def dispatch(tasks, backend, max_containers, app, environment):
    if backend == "local":
        return (simulate_chunk(task) for task in tasks)
    if backend != "modal":
        raise ValueError("Backend must be explicitly local or modal")
    check_remote_names(app, VOLUME_NAME)
    import modal
    function = modal.Function.from_name(app, "simulate_chunk", environment_name=environment)
    function.update_autoscaler(max_containers=max_containers)
    return function.map(tasks, order_outputs=False)


def summarize_game(actual, rows):
    frame = pd.DataFrame(rows)
    home, away = frame.home_score.to_numpy(float), frame.away_score.to_numpy(float)
    probability = float(np.mean(home > away) / max(np.mean(home != away), 1e-9))
    result = {key: actual.get(key) for key in ("game_id", "week", "home_team", "away_team", "spread_line", "total_line", "home_moneyline", "away_moneyline")}
    result.update(actual_home_score=float(actual["home_score"]), actual_away_score=float(actual["away_score"]), actual_spread=float(actual["home_score"]-actual["away_score"]), actual_total=float(actual["home_score"]+actual["away_score"]), sim_home_median=float(np.quantile(home, 0.5)), sim_away_median=float(np.quantile(away, 0.5)), sim_spread=float(np.quantile(home-away, 0.5)), sim_total=float(np.quantile(home+away, 0.5)), sim_home_win_prob=probability, simulations_run=len(rows), is_ot=actual.get("overtime"))
    return result


def reports(store, collector, source):
    by_game = {}
    diagnostic, buckets, nan_rows = [], [], []
    worker_schema_issues = set()
    for result in store.results():
        game, players = by_game.setdefault(result["game_id"], ([], []))
        game.extend(result["game_rows"])
        players.extend(result["player_rows"])
        diagnostic.extend(result["diagnostics"])
        buckets.extend(result["bucket_rows"])
        nan_rows.extend(result["nan_rows"])
        worker_schema_issues.update(result.get("schema_issues", []))
    runtime = collector.runtime
    if not hasattr(runtime, "_ridge_startup_schema_issues"):
        runtime._ridge_startup_schema_issues = tuple(runtime.schema_issues)
    collector.worker_schema_issues = sorted(worker_schema_issues)
    venue_validation = sorted(getattr(collector, "venue_validation", {}).values(), key=lambda record: record["game_id"])
    venue_issues = {f"{record['game_id']}: venue identity validation unavailable: {record['reason']}" for record in venue_validation if record["status"] == "unavailable"}
    runtime.schema_issues[:] = sorted(set(runtime._ridge_startup_schema_issues) | worker_schema_issues | venue_issues)
    scores = []
    for actual in collector.schedule.to_dict("records"):
        game_id = actual["game_id"]
        if game_id not in by_game:
            continue
        rows, players = by_game[game_id]
        raw = pd.DataFrame(rows).sort_values("simulation").rename(columns={"home_score": "home", "away_score": "away"})
        raw.to_csv(collector.raw_results_dir / f"{game_id}.csv", index=False)
        pd.DataFrame(players).to_csv(collector.raw_results_dir / f"{game_id}_players.csv", index=False)
        if len(rows) == collector.simulations:
            scores.append(summarize_game(actual, rows))
    collector.results = pd.DataFrame(scores)
    collector.results.to_csv(source.RESULTS_PATH, index=False)
    pd.DataFrame(diagnostic).to_csv(source.ATTEMPT_DIAGNOSTICS_PATH, index=False)
    pd.DataFrame(buckets).to_csv(source.PLAYCALL_BUCKET_DIAGNOSTICS_PATH, index=False)
    pd.DataFrame(nan_rows).to_csv(source.NAN_INPUT_LOG_PATH, index=False)
    collector._write_summary()
    summary = json.loads(source.SUMMARY_PATH.read_text(encoding="utf-8"))
    summary["worker_schema_issues"] = collector.worker_schema_issues
    summary["venue_validation"] = venue_validation
    write_json(source.SUMMARY_PATH, json_value(summary))
    if scores:
        collector._write_attempt_summary()
        collector._write_playcall_bucket_summary()
        collector._build_team_offense_calibration()


def main():
    parser = argparse.ArgumentParser(description="Isolated historical benchmark; explicit backend, no upload/deploy/fallback")
    parser.add_argument("--backend", choices=["local", "modal"], required=True)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/runtime")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", default="ridge-context-2025")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--max-containers", type=int, default=32)
    parser.add_argument("--week-start", type=int, default=1)
    parser.add_argument("--week-end", type=int, default=18)
    parser.add_argument("--game-id")
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modal-app", default=APP_NAME)
    parser.add_argument("--modal-environment")
    args = parser.parse_args()
    if args.simulations < 1 or args.max_containers < 1 or not 1 <= args.week_start <= args.week_end <= 18 or (args.max_games is not None and args.max_games < 1):
        parser.error("Invalid simulation count, concurrency, week range or max-games")
    check_remote_names(args.modal_app, VOLUME_NAME)
    args.artifact_dir = inside(args.artifact_dir)
    output = output_dir(args.output_dir or ROOT / "benchmarks/2025" / args.run_id)
    before = protected_hashes()
    old_env = os.environ.get("FOOTBALL_ARTIFACT_DIR")
    started = time.monotonic()
    lock = output / "active_bundle.json"
    try:
        validate_bundle(args.artifact_dir)
        bundle_hash = verify_bundle_hash(args.artifact_dir)
        collector, source = make_collector(args, output)
        identity = {"run_id": args.run_id, "bundle_hash": bundle_hash, "seed": args.seed, "simulations": args.simulations, "backend": args.backend, "modal_app": args.modal_app, "modal_environment": args.modal_environment, "games": json_value(collector.schedule.to_dict("records")), "implementation_sha256": {path.name: digest(path) for path in ROOT.glob("*.py")}}
        store = ChunkStore(output, identity, args.resume)
        with lock.open("x", encoding="utf-8") as stream:
            json.dump({"bundle_hash": bundle_hash, "artifact_dir": str(args.artifact_dir), "pid": os.getpid()}, stream)
        failures, games = [], []
        for row in collector.schedule.itertuples():
            try:
                package = collector._build_game_package(row)
                serialized = collector._serialize_game_package(package)
                games.append(PreparedGame(row.game_id, int(row.week), package["home_team"], package["away_team"], package["roof"], serialized["home_players"], serialized["away_players"], serialized["team_features"], args.simulations))
            except (source.SkipGameError, ValueError, KeyError) as error:
                failures.append({"game_id": row.game_id, "stage": "package", "error": str(error)})
        write_json(output / "failures.json", failures)
        tasks = iter(pending_tasks(games, args.artifact_dir, args.max_containers, store))
        while True:
            batch = list(itertools.islice(tasks, args.max_containers*4))
            if not batch:
                break
            expected = {task["task_hash"]: task for task in batch}
            try:
                for result in dispatch(batch, args.backend, args.max_containers, args.modal_app, args.modal_environment):
                    if result["task_hash"] not in expected:
                        raise ValueError("Unexpected chunk result")
                    store.save(expected[result["task_hash"]], result)
                reports(store, collector, source)
            except Exception as error:
                failures.extend({"game_id": task["game_id"], "task_hash": task["task_hash"], "simulation_ids": task["simulation_ids"], "stage": "dispatch", "error": str(error)} for task in batch if any((task["game_id"], simulation) not in store.completed for simulation in task["simulation_ids"]))
                write_json(output / "failures.json", failures)
                reports(store, collector, source)
                raise
        reports(store, collector, source)
        write_json(output / "failures.json", failures)
        write_json(output / "run_manifest.json", {"identity": identity, "status": "complete" if not failures and len(store.completed) == len(collector.schedule)*args.simulations else "incomplete", "completed_simulations": len(store.completed), "expected_simulations": len(collector.schedule)*args.simulations, "wall_seconds": time.monotonic()-started, "failures": failures, "comparison_confounds": ["ridge estimator and new context inputs", "saved baseline training provenance unverified", "platform/runtime version differences possible"]})
    finally:
        if lock.exists() and json.loads(lock.read_text()).get("pid") == os.getpid():
            lock.unlink()
        if old_env is None:
            os.environ.pop("FOOTBALL_ARTIFACT_DIR", None)
        else:
            os.environ["FOOTBALL_ARTIFACT_DIR"] = old_env
        assert_protected(before)


if __name__ == "__main__":
    main()