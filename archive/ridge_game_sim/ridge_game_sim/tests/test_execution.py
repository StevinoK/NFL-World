import numpy as np
import pandas as pd
import pytest
from concurrent.futures import Future
import ridge_game_sim.execution as execution
from ridge_game_sim.dashboard import default_save_folder, output_run_directory

from game_sim_execution import PreparedGame
from ridge_game_sim.execution import build_ridge_tasks, prepared_game


def test_prepared_game_uses_serialized_player_and_team_feature_contract():
    class Player:
        def __init__(self, player_id):
            self.name = player_id
            self.position = "QB1"
            self.player_id = player_id
            self.attributes = pd.DataFrame([{"ridge_cpoe_venue_effect": .3, "yac_bayes": 1.0}])

    row = pd.Series({"game_id": "2025_01_A_B", "week": 1, "home_team": "A", "away_team": "B"})
    teams = pd.DataFrame([{"ridge_cpoe_venue_effect": .4}], index=["A"])
    teams.loc["B"] = {"ridge_cpoe_venue_effect": -.2}
    game = prepared_game(row, teams, [Player("A1")], [Player("B1")], 3, "outdoors")
    assert isinstance(game, PreparedGame)
    assert game.home_players[0]["attributes"]["ridge_cpoe_venue_effect"] == .3
    assert game.team_features["A"]["ridge_cpoe_venue_effect"] == .4
    assert game.simulations == 3


def test_chunk_tasks_pin_bundle_run_and_deterministic_ids():
    game = PreparedGame("G", 1, "A", "B", "outdoors", [], [], {"A": {}, "B": {}}, 7)
    tasks = build_ridge_tasks([game], "C:/bundle", "bundlehash", "run-1", 17, 2)
    assert sum(task["count"] for task in tasks) == 7
    assert all(task["bundle_hash"] == "bundlehash" and task["run_id"] == "run-1" for task in tasks)
    assert [number for task in tasks for number in task["simulation_ids"]] == list(range(1, 8))
    assert all(task["task_hash"] for task in tasks)


def test_chunk_tasks_accept_parallelism_above_32():
    game = PreparedGame("G", 1, "A", "B", "outdoors", [], [], {"A": {}, "B": {}}, 100)
    tasks = build_ridge_tasks([game], "C:/bundle", "bundlehash", "run-1", 17, 64)

    assert sum(task["count"] for task in tasks) == 100
    assert [number for task in tasks for number in task["simulation_ids"]] == list(range(1, 101))


def test_ridge_dispatch_adds_unchanged_fantasy_scoring(monkeypatch, tmp_path):
    game = PreparedGame("G", 1, "A", "B", "outdoors", [], [], {"A": {}, "B": {}}, 2)
    calls = []
    def worker(task):
        calls.append(task)
        return {"game_id": "G", "completed_count": task["count"], "game_rows": [
            {"game_id": "G", "simulation": sim, "home_team": "A", "away_team": "B",
             "home_score": 24, "away_score": 17, "total": 41, "spread": 7}
            for sim in task["simulation_ids"]], "player_rows": [
            {"game_id": "G", "simulation": sim, "team": "A", "player_id": "P",
             "player_name": "Player P", "position": "QB1", "passing_yards": 300,
             "passing_touchdowns": 2}
            for sim in task["simulation_ids"]]}
    monkeypatch.setattr(execution, "simulate_chunk", worker)
    bundle = "local-test-bundle"
    result = execution.execute_ridge_games([game], tmp_path, bundle, backend="local", parallelism=1)
    games, players = result["G"]
    assert sum(call["count"] for call in calls) == 2
    assert all(call["bundle_hash"] == bundle for call in calls)
    assert games.game_id.eq("G").all()
    assert players.fantasy_points.tolist() == [20.0, 20.0]


def test_ridge_local_dispatch_uses_requested_process_workers(monkeypatch, tmp_path):
    game = PreparedGame("G", 1, "A", "B", "outdoors", [], [], {"A": {}, "B": {}}, 8)
    submitted = []
    worker_counts = []

    class InlineProcessPool:
        def __init__(self, max_workers):
            worker_counts.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, task):
            submitted.append(task)
            future = Future()
            future.set_result(function(task))
            return future

    def worker(task):
        return {
            "game_id": task["game_id"], "completed_count": task["count"],
            "game_rows": [
                {"game_id": "G", "simulation": simulation, "home_team": "A", "away_team": "B",
                 "home_score": 24, "away_score": 17, "total": 41, "spread": 7}
                for simulation in task["simulation_ids"]
            ],
            "player_rows": [
                {"game_id": "G", "simulation": simulation, "team": "A", "player_id": "P",
                 "player_name": "Player P", "position": "QB1", "passing_yards": 300,
                 "passing_touchdowns": 2}
                for simulation in task["simulation_ids"]
            ],
        }

    monkeypatch.setattr(execution, "ProcessPoolExecutor", InlineProcessPool)
    monkeypatch.setattr(execution, "simulate_chunk", worker)

    result = execution.execute_ridge_games(
        [game], tmp_path, "bundle", backend="local", parallelism=4,
    )

    games, players = result["G"]
    assert worker_counts == [4]
    assert len(submitted) > 1
    assert sum(task["count"] for task in submitted) == 8
    assert games.simulation.tolist() == list(range(1, 9))
    assert players.fantasy_points.tolist() == [20.0] * 8


def test_dashboard_output_directory_is_user_selected_and_never_overwrites(tmp_path):
    destination = output_run_directory(str(tmp_path), "my-ridge-week")
    assert destination == (tmp_path / "my-ridge-week").resolve()
    with pytest.raises(FileExistsError):
        output_run_directory(str(tmp_path), "my-ridge-week")
    with pytest.raises(ValueError, match="single folder name"):
        output_run_directory(str(tmp_path), "../outside")


def test_synthetic_save_folder_matches_previous_weekly_output_location():
    from project_paths import DEFAULT_SIMULATION_OUTPUT_DIR
    from ridge_game_sim.paths import ROOT

    assert default_save_folder("Synthetic opponents", 5) == DEFAULT_SIMULATION_OUTPUT_DIR / "synthetic_home_week_05"
    assert default_save_folder("Synthetic opponents", 12) == DEFAULT_SIMULATION_OUTPUT_DIR / "synthetic_home_week_12"
    assert default_save_folder("Single game", 5) == ROOT / "outputs"
    assert default_save_folder("Full week", 5) == ROOT / "outputs"
