from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nflsim.engine.fantasy import team_unit_rows
from nflsim.bundle import verify_bundle_hash
from nflsim.engine.batching import Batcher
from nflsim.paths import canonical_hash
from nflsim.sim.runtime import RidgeGame, RidgeRuntime
from nflsim.engine.simulation_runtime import Player, Team

_RUNTIMES = {}


def simulation_seed(bundle_hash, run_id, game_id, simulation, seed):
    return int(canonical_hash([bundle_hash, run_id, game_id, int(simulation), int(seed)])[:8], 16)


def get_runtime(directory, bundle_hash):
    directory = Path(directory)
    key = (str(directory.resolve()), bundle_hash)
    if key not in _RUNTIMES:
        if verify_bundle_hash(directory) != bundle_hash:
            raise ValueError("Worker bundle hash mismatch")
        torch.set_num_threads(1)
        _RUNTIMES[key] = RidgeRuntime(directory, strict_schema=True)
        _RUNTIMES[key]._ridge_startup_schema_issues = tuple(_RUNTIMES[key].schema_issues)
    return _RUNTIMES[key]


def simulate_one(task, runtime, simulation):
    """One seeded simulation of the task's game; returns its output rows."""
    seed = simulation_seed(task["bundle_hash"], task["run_id"], task["game_id"], simulation, task["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    runtime.seed(seed)
    runtime.neutral_site = bool(task.get("neutral_site", False))
    runtime.reset_nan_input_log()
    def players(snapshots):
        return [Player(str(item["name"]), str(item["position"]), str(item["player_id"]), pd.DataFrame([item["attributes"]])) for item in snapshots]
    swapped = bool(task.get("swap_home_away_halfway", False)) and int(simulation) > int(task.get("swap_after_simulation", 0))
    if swapped:
        home = Team(task["away_team"], players(task["away_players"]), "home", pd.Series(task["team_features"][task["away_team"]]), runtime)
        away = Team(task["home_team"], players(task["home_players"]), "away", pd.Series(task["team_features"][task["home_team"]]), runtime)
    else:
        home = Team(task["home_team"], players(task["home_players"]), "home", pd.Series(task["team_features"][task["home_team"]]), runtime)
        away = Team(task["away_team"], players(task["away_players"]), "away", pd.Series(task["team_features"][task["away_team"]]), runtime)
    game = RidgeGame(away, home, str(task["roof"]), runtime)
    game.simulate_game()
    identity = {"game_id": task["game_id"], "simulation": int(simulation)}
    team_score = away.score if swapped else home.score
    synthetic_score = home.score if swapped else away.score
    player_rows = []
    for team in (home, away):
        for player in team.players:
            player_rows.append({**identity, "team": team.name, "player_id": player.player_id, "player_name": player.name, "position": player.position, **player.stats})
        player_rows.extend({**row, **identity} for row in team_unit_rows(team, simulation))
    output = {
        "game_rows": [{**identity, "home_team": task["home_team"], "away_team": task["away_team"], "neutral_site": runtime.neutral_site, "home_away_swapped": swapped, "home_score": team_score, "away_score": synthetic_score, "total": team_score+synthetic_score, "spread": team_score-synthetic_score}],
        "player_rows": player_rows,
        "diagnostics": [{**game.diagnostics_row(), **identity}],
        "bucket_rows": [{**row, **identity} for row in game.diagnostics_bucket_rows()],
        "nan_rows": [{**row, **identity} for row in runtime.get_nan_input_log_rows()],
    }
    structural = runtime.structural_schema_issues()
    if structural:
        raise ValueError(f"Inference schema issues: {structural}")
    return output


def simulate_chunk(task):
    runtime = get_runtime(task["data_dir"], task["bundle_hash"])
    if not hasattr(runtime, "_ridge_startup_schema_issues"):
        runtime._ridge_startup_schema_issues = tuple(runtime.schema_issues)
    runtime.schema_issues[:] = runtime._ridge_startup_schema_issues
    ids = task.get("simulation_ids", list(range(int(task["start_simulation"]), int(task["start_simulation"])+int(task["count"]))))
    if len(ids) != task["count"] or len(set(ids)) != len(ids) or any(int(value) < 1 for value in ids):
        raise ValueError("Invalid simulation IDs")
    if task.get("batched", True):
        # All of the chunk's simulations run together, sharing one forward pass per model call.
        outputs = Batcher(runtime).run([lambda simulation=simulation: simulate_one(task, runtime, simulation) for simulation in ids])
    else:
        outputs = [simulate_one(task, runtime, simulation) for simulation in ids]
    rows = {key: [row for output in outputs for row in output[key]]
            for key in ("game_rows", "player_rows", "diagnostics", "bucket_rows", "nan_rows")}
    return {"game_id": task["game_id"], "week": task["week"], "run_id": task["run_id"], "task_hash": task["task_hash"], "bundle_hash": task["bundle_hash"], "completed_ids": list(ids), "completed_count": len(ids), **rows, "schema_issues": list(runtime.schema_issues)}
