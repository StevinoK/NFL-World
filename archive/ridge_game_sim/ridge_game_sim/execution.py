from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from game_sim_execution import PreparedGame, build_chunk_tasks
from game_sim_fantasy import add_fantasy_points
from ridge_game_sim.modal_worker import simulate_chunk
from ridge_game_sim.paths import APP_NAME, canonical_hash, check_remote_names
from ridge_game_sim.storage import task_hash


@dataclass(frozen=True)
class RidgePreparedGame(PreparedGame):
    neutral_site: bool = False
    swap_home_away_halfway: bool = False


def prepared_game(game_row: pd.Series, team_features: pd.DataFrame, home_players, away_players, simulations: int, roof: str,
                  swap_home_away_halfway: bool = False) -> PreparedGame:
    from game_sim_execution import serialize_players

    home, away = str(game_row["home_team"]), str(game_row["away_team"])
    return RidgePreparedGame(
        game_id=str(game_row["game_id"]), week=int(game_row["week"]),
        home_team=home, away_team=away, roof=str(roof),
        home_players=serialize_players(home_players), away_players=serialize_players(away_players),
        team_features={home: team_features.loc[home].to_dict(), away: team_features.loc[away].to_dict()},
        simulations=int(simulations),
        neutral_site=str(game_row.get("location", "Home")).lower() == "neutral",
        swap_home_away_halfway=bool(swap_home_away_halfway),
    )


def build_ridge_tasks(games: Sequence[PreparedGame], artifact_dir, bundle_hash: str, run_id: str, seed: int, parallelism: int):
    tasks = build_chunk_tasks(games, artifact_dir, parallelism)
    for task in tasks:
        game = next(game for game in games if game.game_id == task["game_id"])
        task["neutral_site"] = bool(getattr(game, "neutral_site", False))
        task["swap_home_away_halfway"] = bool(getattr(game, "swap_home_away_halfway", False))
        task["swap_after_simulation"] = int(game.simulations) // 2
        task["simulations"] = int(game.simulations)
        task["simulation_ids"] = list(range(int(task["start_simulation"]), int(task["start_simulation"]) + int(task["count"])))
        task.update(run_id=run_id, bundle_hash=bundle_hash, seed=int(seed))
        task["task_hash"] = task_hash(task)
    return tasks


def execute_ridge_games(
    games: Sequence[PreparedGame], artifact_dir, bundle_hash: str, *, backend: str,
    parallelism: int = 1, run_id: str = "ridge-simulation", seed: int = 42,
    modal_app: str = APP_NAME, modal_environment: str | None = None,
    progress: Callable[[int, int], None] | None = None,
):
    tasks = build_ridge_tasks(games, artifact_dir, bundle_hash, run_id, seed, parallelism)
    if not tasks:
        return {}
    key = backend.strip().lower()
    if key == "local":
        worker_count = min(max(1, int(parallelism)), len(tasks))
        if worker_count == 1:
            results = (simulate_chunk(task) for task in tasks)
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(simulate_chunk, task) for task in tasks]
                results = (future.result() for future in as_completed(futures))
                output = _collect_ridge_results(games, results, progress)
            return output
    elif key == "modal":
        check_remote_names(modal_app, "nfl-ridge-game-sim-artifacts")
        import modal
        function = modal.Function.from_name(modal_app, "simulate_chunk", environment_name=modal_environment)
        function.update_autoscaler(max_containers=max(1, int(parallelism)))
        results = function.map(tasks, order_outputs=False)
    else:
        raise ValueError("Ridge backend must be local or modal")

    return _collect_ridge_results(games, results, progress)


def _collect_ridge_results(games, results, progress):
    expected = sum(game.simulations for game in games)
    completed = 0
    game_rows = {game.game_id: [] for game in games}
    player_rows = {game.game_id: [] for game in games}
    for result in results:
        completed += int(result["completed_count"])
        if progress is not None:
            progress(completed, expected)
        if result["game_id"] not in game_rows:
            raise ValueError(f"Unexpected Ridge game result {result['game_id']}")
        game_rows[result["game_id"]].extend(result["game_rows"])
        player_rows[result["game_id"]].extend(result["player_rows"])

    output = {}
    for game in games:
        game_id = game.game_id
        rows = game_rows[game_id]
        ids = sorted(int(row["simulation"]) for row in rows)
        if ids != list(range(1, game.simulations + 1)):
            raise ValueError(f"Incomplete or duplicate Ridge simulations for {game_id}: {ids}")
        games_frame = pd.DataFrame(rows).sort_values("simulation").reset_index(drop=True)
        players_frame = add_fantasy_points(pd.DataFrame(player_rows[game_id]).sort_values(["simulation", "team", "player_name"]).reset_index(drop=True))
        for frame in (games_frame, players_frame):
            frame.drop(columns=["game_id", "week"], errors="ignore", inplace=True)
            frame.insert(0, "game_id", game.game_id)
            frame.insert(1, "week", int(game.week))
        output[game_id] = (games_frame, players_frame)
    return output
