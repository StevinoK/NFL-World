import copy
from types import SimpleNamespace

import pytest

from nflsim.engine.chunk_execution import PreparedGame
from nflsim.sim import benchmark_2025
from nflsim.sim.benchmark_2025 import dispatch, pending_tasks, redirect_outputs, summarize_game
from nflsim.storage import ChunkStore, task_hash


def fixture_task():
    value = {"game_id": "G", "week": 1, "home_team": "A", "away_team": "B", "simulation_ids": [1, 3], "count": 2, "bundle_hash": "B", "run_id": "R", "seed": 42}
    value["task_hash"] = task_hash(value)
    result = {key: value[key] for key in ("game_id", "week", "bundle_hash", "run_id", "task_hash")}
    result.update(completed_ids=[1, 3], completed_count=2, game_rows=[{"game_id": "G", "simulation": simulation, "home_team": "A", "away_team": "B", "home_score": 20, "away_score": 10} for simulation in (1, 3)], player_rows=[], diagnostics=[{"game_id": "G", "simulation": simulation} for simulation in (1, 3)], bucket_rows=[], nan_rows=[])
    return value, result


def test_atomic_store_retry_conflict_and_resume(tmp_path):
    identity = {"bundle_hash": "B", "run_id": "R", "seed": 42}
    task, result = fixture_task()
    store = ChunkStore(tmp_path, identity)
    store.save(task, result)
    store.save(task, result)
    resumed = ChunkStore(tmp_path, identity, resume=True)
    assert len(resumed.completed) == 2
    game = PreparedGame("G", 1, "A", "B", "outdoors", [], [], {}, 4)
    tasks = list(pending_tasks([game], tmp_path, 1, resumed))
    assert sorted(simulation for item in tasks for simulation in item["simulation_ids"]) == [2, 4]
    conflict = copy.deepcopy(result)
    conflict["game_rows"][0]["home_score"] = 99
    with pytest.raises(ValueError, match="Conflicting"):
        store.save(task, conflict)
    with pytest.raises(ValueError, match="settings"):
        ChunkStore(tmp_path, {**identity, "seed": 99}, True)


def test_bad_handshake_and_duplicate_counts(tmp_path):
    task, result = fixture_task()
    store = ChunkStore(tmp_path, {"bundle_hash": "B", "run_id": "R", "seed": 42})
    with pytest.raises(ValueError, match="handshake"):
        store.save(task, {**result, "bundle_hash": "wrong"})
    with pytest.raises(ValueError, match="IDs"):
        store.save(task, {**result, "completed_ids": [1, 1]})


def test_cross_game_tasks_and_no_local_fallback(monkeypatch, tmp_path):
    store = ChunkStore(tmp_path, {"bundle_hash": "B", "run_id": "R", "seed": 42})
    games = [PreparedGame(game_id, 1, "A", "B", "outdoors", [], [], {}, 2) for game_id in ("G", "H")]
    assert {task["game_id"] for task in pending_tasks(games, tmp_path, 32, store)} == {"G", "H"}
    with pytest.raises(ValueError):
        dispatch([], "modal", 32, "nfl-game-sim", None)


def test_summary_tie_and_sign():
    actual = {"game_id": "G", "week": 1, "home_team": "A", "away_team": "B", "home_score": 24, "away_score": 17}
    result = summarize_game(actual, [{"home_score": 20, "away_score": 20}, {"home_score": 21, "away_score": 14}, {"home_score": 10, "away_score": 13}])
    assert result["sim_home_win_prob"] == 0.5
    assert result["actual_spread"] == 7
    assert result["sim_spread"] == 0


def test_local_dispatch_uses_four_processes(monkeypatch):
    submitted = []
    worker_counts = []

    class Future:
        def __init__(self, task):
            self.task = task

        def result(self):
            return self.task

    class Pool:
        def __init__(self, max_workers):
            worker_counts.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, task):
            assert function is benchmark_2025.simulate_chunk
            submitted.append(task)
            return Future(task)

    monkeypatch.setattr(benchmark_2025, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(benchmark_2025, "as_completed", lambda futures: reversed(futures))
    tasks = [{"game_id": str(index)} for index in range(6)]
    assert list(dispatch(tasks, "local", 4, "unused", None)) == list(reversed(tasks))
    assert submitted == tasks
    assert worker_counts == [4]