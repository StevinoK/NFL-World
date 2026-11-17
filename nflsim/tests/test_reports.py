import json
from types import SimpleNamespace

import pandas as pd
import pytest

from nflsim.sim import benchmark_2025
from nflsim.sim.benchmark_2025 import redirect_outputs, reports, validate_frozen_venue
from nflsim.storage import ChunkStore, task_hash


def test_report_names_resolve_ids_without_changing_identity_or_statistics():
    import nflsim.engine.backtest_reports as source

    collector = object.__new__(source.GameBacktester)
    collector.player_stats = pd.DataFrame({"player_id": ["qb"], "player_name": ["qb"]})
    collector.rosters = pd.DataFrame({"player_id": ["qb", "wr"], "full_name": ["Lamar Jackson", "Zay Flowers"]})
    players = pd.DataFrame({"player_id": ["qb", "wr", "unknown", "named"],
                            "player_name": ["qb", None, None, "Existing Name"], "passing_yards": [200, 0, 0, 0]})
    result = collector._report_player_names(players)
    assert result.player_name.tolist() == ["Lamar Jackson", "Zay Flowers", "unknown", "Existing Name"]
    pd.testing.assert_series_equal(result.player_id, players.player_id)
    pd.testing.assert_series_equal(result.passing_yards, players.passing_yards)
    assert players.player_name.iloc[0] == "qb"


def test_saved_report_names_only_change_table_labels_and_escape_names():
    from nflsim.engine.player_names import name_report_tables, player_names

    roster = pd.DataFrame({"player_id": ["00-1"], "player_name": ["00-1"],
                           "first_name": ["John"], "last_name": ["A&B"]})
    report = '<!doctype html><script>var id="00-1";</script><table><tr><td>00-1</td><td>12.5</td><td>unknown</td></tr></table>'
    result, count = name_report_tables(report, player_names(roster))
    assert count == 1
    assert '<td>John A&amp;B</td>' in result
    assert '<script>var id="00-1";</script>' in result
    assert '<td>12.5</td><td>unknown</td>' in result


def test_original_summary_and_html_with_synthetic_chunks(tmp_path, monkeypatch):
    import nflsim.engine.backtest_reports as source
    for name in list(vars(source)):
        if name.endswith("_PATH") or name == "RAW_RESULTS_DIR":
            monkeypatch.setattr(source, name, getattr(source, name))
    redirect_outputs(source, tmp_path)
    collector = object.__new__(source.GameBacktester)
    collector.season = 2025
    collector.simulations = 3
    collector.attempt_diagnostics = True
    collector.raw_results_dir = source.RAW_RESULTS_DIR
    collector.runtime = type("Runtime", (), {"schema_issues": ["startup warning"]})()
    collector.schedule = pd.DataFrame([{"game_id": "2025_01_B_A", "week": 1, "home_team": "A", "away_team": "B", "home_score": 21, "away_score": 17, "spread_line": 3, "total_line": 40, "home_moneyline": -150, "away_moneyline": 130}])
    collector.venue_validation = {"2025_01_B_A": {"game_id": "2025_01_B_A", "status": "unavailable", "reason": "Frozen feature files lack venue identity"}}
    monkeypatch.setattr(collector, "_actual_playcall_bucket_rates", lambda *args: pd.DataFrame())
    monkeypatch.setattr(collector, "_get_actual_team_game_stats", lambda: pd.DataFrame())
    monkeypatch.setattr(collector, "_get_actual_team_penalty_stats", lambda *args: pd.DataFrame())
    monkeypatch.setattr(collector, "_get_actual_game_fumble_stats", lambda *args: pd.DataFrame())
    monkeypatch.setattr(collector, "_get_actual_player_game_stats", lambda *args: pd.DataFrame())
    store = ChunkStore(tmp_path, {"run_id": "R", "bundle_hash": "B", "seed": 42})
    task = {"game_id": "2025_01_B_A", "week": 1, "home_team": "A", "away_team": "B", "run_id": "R", "bundle_hash": "B", "seed": 42, "simulation_ids": [1, 2], "count": 2}
    task["task_hash"] = task_hash(task)
    result = {key: task[key] for key in ("game_id", "week", "run_id", "bundle_hash", "task_hash")}
    result.update(completed_ids=[1, 2], completed_count=2, game_rows=[{"game_id": task["game_id"], "simulation": simulation, "home_team": "A", "away_team": "B", "home_score": 20+simulation, "away_score": 17} for simulation in (1, 2)], player_rows=[], diagnostics=[{"game_id": task["game_id"], "simulation": simulation, "play_call_pass": 2, "play_call_run": 1, "sim_pass_attempts": 2, "sim_rushing_attempts": 1, "pass_attempt_delta": 0} for simulation in (1, 2)], bucket_rows=[{"game_id": task["game_id"], "simulation": 1, "bucket_key": "early", "snaps": 3, "off_snaps": 3, "pass_calls": 2, "run_calls": 1, "punt_calls": 0, "kick_calls": 0, "pass_prob_sum": 2, "run_prob_sum": 1}], nan_rows=[], schema_issues=["sack: detected NaN inputs (down)", "sack: detected NaN inputs (down)"])
    store.save(task, result)
    extra_task = {**task, "simulation_ids": [3], "count": 1}
    extra_task["task_hash"] = task_hash(extra_task)
    extra_result = {**result, "task_hash": extra_task["task_hash"], "completed_ids": [3], "completed_count": 1,
                    "game_rows": [{**result["game_rows"][0], "simulation": 3, "home_score": 23}],
                    "diagnostics": [{**result["diagnostics"][0], "simulation": 3}], "bucket_rows": [],
                    "schema_issues": ["target: detected NaN inputs (medium)", "sack: detected NaN inputs (down)"]}
    store.save(extra_task, extra_result)
    reports(store, collector, source)
    summary = json.loads(source.SUMMARY_PATH.read_text())
    assert summary["games_scored"] == 1
    assert summary["worker_schema_issues"] == ["sack: detected NaN inputs (down)", "target: detected NaN inputs (medium)"]
    assert summary["schema_issue_count"] == 4
    assert "startup warning" in summary["schema_issues"]
    assert summary["venue_validation"][0]["status"] == "unavailable"
    assert collector.worker_schema_issues == summary["worker_schema_issues"]
    assert collector.runtime.schema_issues == summary["schema_issues"]
    assert source.HTML_REPORT_PATH.is_file()
    assert json.loads(source.ATTEMPT_SUMMARY_PATH.read_text())["rows"] == 3
    assert json.loads(source.PLAYCALL_BUCKET_SUMMARY_PATH.read_text())["unique_buckets"] == 1
    assert json.loads(source.TEAM_OFFENSE_CALIBRATION_SUMMARY_PATH.read_text())["rows"] == 0
    assert pd.read_csv(collector.raw_results_dir / f"{task['game_id']}.csv").home.tolist() == [21, 22, 23]
    reports(store, collector, source)
    assert json.loads(source.SUMMARY_PATH.read_text()) == summary


@pytest.mark.parametrize("schedule_column", ["stadium_id", "game_stadium", "stadium"])
@pytest.mark.parametrize("frozen_column", ["venue", "stadium_id", "game_stadium", "stadium"])
def test_fresh_schedule_venue_matches_both_frozen_teams(schedule_column, frozen_column):
    game = SimpleNamespace(game_id="G", home_team="A", away_team="B", **{schedule_column: "neutral_stadium"})
    frozen = pd.DataFrame({"game_id": ["G", "G"], "team": ["A", "B"], frozen_column: ["neutral_stadium", "neutral_stadium"]})
    assert validate_frozen_venue(game, frozen)["status"] == "validated"
    frozen.loc[1, frozen_column] = "old_home_stadium"
    with pytest.raises(ValueError, match="differs from frozen venue"):
        validate_frozen_venue(game, frozen)


@pytest.mark.parametrize("venue", [None, float("nan"), ""])
def test_missing_frozen_venue_is_explicitly_unvalidated(venue):
    game = SimpleNamespace(game_id="G", home_team="A", away_team="B", stadium_id="neutral_stadium")
    frozen = pd.DataFrame({"game_id": ["G", "G"], "team": ["A", "B"]})
    assert validate_frozen_venue(game, frozen)["status"] == "unavailable"
    frozen["venue"] = ["neutral_stadium", venue]
    status = validate_frozen_venue(game, frozen)
    assert status["status"] == "unavailable"
    assert "B" in status["reason"]


@pytest.mark.parametrize("venue", [None, float("nan"), "", " "])
def test_missing_fresh_schedule_venue_is_rejected(venue):
    game = SimpleNamespace(game_id="G", home_team="A", away_team="B", stadium_id=venue)
    with pytest.raises(ValueError, match="Scheduled venue identity missing"):
        validate_frozen_venue(game, pd.DataFrame())


def test_collector_validates_frozen_venue_before_building_package(tmp_path, monkeypatch):
    import nflsim.engine.backtest_reports as source
    monkeypatch.setenv("FOOTBALL_ARTIFACT_DIR", "unused")
    for name in list(vars(source)):
        if name.endswith("_PATH") or name == "RAW_RESULTS_DIR":
            monkeypatch.setattr(source, name, getattr(source, name))
    game = SimpleNamespace(game_id="G", home_team="A", away_team="B", stadium_id="neutral_stadium", roof="outdoors")
    frozen = pd.DataFrame({"game_id": ["G", "G"], "team": ["A", "B"], "game_date": ["2025-09-01"] * 2, "venue": ["neutral_stadium"] * 2})
    runtime = SimpleNamespace(team_stats_wide=frozen, player_stats_wide=pd.DataFrame({"game_date": []}), context_map={})
    monkeypatch.setattr(benchmark_2025, "RidgeRuntime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(source.GameBacktester, "_load_schedule", lambda self: pd.DataFrame([vars(game)]))
    monkeypatch.setattr(source.GameBacktester, "_load_rosters", lambda self: pd.DataFrame())
    monkeypatch.setattr(source.GameBacktester, "_load_depth_charts", lambda self: pd.DataFrame())
    packages_built = []

    def base_package(self, game_row):
        packages_built.append(game_row.game_id)
        return {"home_team": "A", "away_team": "B", "team_features": self.team_stats.set_index("team"), "roof": "dome"}

    monkeypatch.setattr(source.GameBacktester, "_build_game_package", base_package)
    pd.DataFrame(columns=["season", "week", "reason"]).to_csv(tmp_path / "ridge_fallbacks.csv", index=False)
    args = SimpleNamespace(artifact_dir=tmp_path, simulations=1, season=2025, week_start=1, week_end=1, game_id=None, max_games=None)
    collector, _ = benchmark_2025.make_collector(args, tmp_path)
    assert collector._build_game_package(game)["roof"] == "outdoors"
    assert collector.venue_validation["G"]["status"] == "validated"
    collector.team_stats.loc[1, "venue"] = "old_home_stadium"
    with pytest.raises(ValueError, match="differs from frozen venue"):
        collector._build_game_package(game)
    assert packages_built == ["G"]
    collector.team_stats = collector.team_stats.drop(columns="venue")
    collector._build_game_package(game)
    assert collector.venue_validation["G"]["status"] == "unavailable"