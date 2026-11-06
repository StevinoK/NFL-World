import random
from types import SimpleNamespace

import numpy as np
import pytest

from ridge_game_sim import modal_worker
from ridge_game_sim.storage import ChunkStore, task_hash


class FakeRuntime:
    def __init__(self):
        self.target_embed_columns = []
        self.running_embed_columns = []
        self.schema_issues = []

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def reset_nan_input_log(self):
        pass

    def get_nan_input_log_rows(self):
        return [{"model": "sack", "column": "down"}]

    def structural_schema_issues(self):
        return []


class FakeGame:
    def __init__(self, away, home, roof, runtime):
        self.home, self.away, self.runtime = home, away, runtime

    def simulate_game(self):
        self.home.score = int(self.runtime.rng.integers(0, 50))
        self.away.score = random.randrange(50)

    def diagnostics_row(self):
        return {"pass_calls": 2}

    def diagnostics_bucket_rows(self):
        return [{"bucket": "early", "count": 2}]


def task(ids):
    return {"data_dir": "unused", "bundle_hash": "hash", "run_id": "test", "game_id": "G", "week": 1, "home_team": "A", "away_team": "B", "roof": "outdoors", "home_players": [], "away_players": [], "team_features": {"A": {}, "B": {}}, "simulation_ids": ids, "start_simulation": ids[0], "count": len(ids), "seed": 7, "task_hash": "task"}


def test_deterministic_retries_chunk_sizes_and_diagnostics(monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(modal_worker, "get_runtime", lambda *args: runtime)
    monkeypatch.setattr(modal_worker, "RidgeGame", FakeGame)
    monkeypatch.setattr(modal_worker, "team_unit_rows", lambda *args: [])
    batch = modal_worker.simulate_chunk(task([1, 2, 3]))
    retried = modal_worker.simulate_chunk(task([1, 2, 3]))
    assert batch == retried
    single = [modal_worker.simulate_chunk(task([simulation]))["game_rows"][0] for simulation in [1, 2, 3]]
    assert batch["game_rows"] == single
    assert len(batch["diagnostics"]) == len(batch["nan_rows"]) == 3


def test_synthetic_opponent_swaps_roles_at_full_game_midpoint(monkeypatch):
    runtime = FakeRuntime()

    class RoleSensitiveGame(FakeGame):
        def simulate_game(self):
            self.home.score = 30
            self.away.score = 10

    monkeypatch.setattr(modal_worker, "get_runtime", lambda *args: runtime)
    monkeypatch.setattr(modal_worker, "RidgeGame", RoleSensitiveGame)
    monkeypatch.setattr(modal_worker, "team_unit_rows", lambda *args: [])
    payload = task([1, 2, 3, 4])
    payload.update(swap_home_away_halfway=True, swap_after_simulation=2, simulations=4)

    rows = modal_worker.simulate_chunk(payload)["game_rows"]

    assert [(row["home_score"], row["away_score"]) for row in rows] == [
        (30, 10), (30, 10), (10, 30), (10, 30),
    ]
    assert [row["home_away_swapped"] for row in rows] == [False, False, True, True]
    assert all(row["home_team"] == "A" and row["away_team"] == "B" for row in rows)


def test_seed_depends_on_game_not_chunk():
    assert modal_worker.simulation_seed("b", "r", "G", 1, 7) != modal_worker.simulation_seed("b", "r", "H", 1, 7)


def test_cached_runtime_schema_is_chunk_local_and_retry_is_persistable(monkeypatch, tmp_path):
    runtime = FakeRuntime()
    runtime.schema_issues = ["startup warning"]

    class SchemaGame(FakeGame):
        def simulate_game(self):
            super().simulate_game()
            issue = f"{self.home.name}: detected NaN inputs (down)"
            if issue not in self.runtime.schema_issues:
                self.runtime.schema_issues.append(issue)

    monkeypatch.setattr(modal_worker, "get_runtime", lambda *args: runtime)
    monkeypatch.setattr(modal_worker, "RidgeGame", SchemaGame)
    monkeypatch.setattr(modal_worker, "team_unit_rows", lambda *args: [])
    original = task([1, 2])
    original["task_hash"] = task_hash(original)
    store = ChunkStore(tmp_path, {"run_id": "test", "bundle_hash": "hash", "seed": 7})
    first = modal_worker.simulate_chunk(original)
    store.save(original, first)
    other = {**task([1]), "game_id": "H", "home_team": "C", "team_features": {"C": {}, "B": {}}}
    other_result = modal_worker.simulate_chunk(other)
    assert other_result["schema_issues"] == ["startup warning", "C: detected NaN inputs (down)"]
    retried = modal_worker.simulate_chunk(original)
    assert retried == first
    assert first["schema_issues"] == ["startup warning", "A: detected NaN inputs (down)"]
    store.save(original, retried)
    assert len(list(store.results())) == 1