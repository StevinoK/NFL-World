"""Backend-neutral orchestration for local and Modal game simulation chunks."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from nflsim.engine.fantasy import add_fantasy_points
from nflsim.engine.chunk_worker import init_worker, serialize_players, simulate_chunk_task
from nflsim.engine.simulation_runtime import Player, SimulationRuntime

ProgressCallback = Callable[[int, int], None]
ChunkResult = Dict[str, object]


@dataclass(frozen=True)
class PreparedGame:
    game_id: str
    week: int
    home_team: str
    away_team: str
    roof: str
    home_players: List[Dict[str, object]]
    away_players: List[Dict[str, object]]
    team_features: Dict[str, Dict[str, object]]
    simulations: int


def prepare_game(
    runtime: SimulationRuntime,
    game_row: pd.Series,
    team_features: pd.DataFrame,
    home_players: List[Player],
    away_players: List[Player],
    simulations: int,
    roof: str,
) -> PreparedGame:
    home_team = str(game_row["home_team"])
    away_team = str(game_row["away_team"])
    return PreparedGame(
        game_id=str(game_row["game_id"]),
        week=int(game_row["week"]),
        home_team=home_team,
        away_team=away_team,
        roof=str(roof),
        home_players=serialize_players(home_players),
        away_players=serialize_players(away_players),
        team_features={
            home_team: team_features.loc[home_team].to_dict(),
            away_team: team_features.loc[away_team].to_dict(),
        },
        simulations=int(simulations),
    )


def chunk_ranges(simulations: int, parallelism: int, chunk_size: int | None = None) -> List[Tuple[int, int]]:
    if simulations < 1:
        raise ValueError("simulations must be at least 1")
    parallelism = max(1, int(parallelism))
    if chunk_size is None:
        chunk_size = max(1, min(25, int(np.ceil(simulations / (parallelism * 4)))))
    return [
        (start, min(chunk_size, simulations - start + 1))
        for start in range(1, simulations + 1, chunk_size)
    ]


def build_chunk_tasks(
    games: Sequence[PreparedGame],
    artifact_dir: Path | str,
    parallelism: int,
    chunk_size: int | None = None,
) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    for game in games:
        template: Dict[str, object] = {
            "data_dir": str(artifact_dir),
            "game_id": game.game_id,
            "week": game.week,
            "home_team": game.home_team,
            "away_team": game.away_team,
            "roof": game.roof,
            "home_players": game.home_players,
            "away_players": game.away_players,
            "team_features": game.team_features,
        }
        for start, count in chunk_ranges(game.simulations, parallelism, chunk_size):
            tasks.append({**template, "start_simulation": start, "count": count})
    return tasks


def _local_results(
    tasks: Sequence[Dict[str, object]],
    workers: int,
    artifact_dir: Path | str,
) -> Iterator[ChunkResult]:
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(str(artifact_dir),),
    ) as executor:
        futures = [executor.submit(simulate_chunk_task, task) for task in tasks]
        for future in as_completed(futures):
            yield future.result()


def _modal_results(
    tasks: Sequence[Dict[str, object]],
    max_containers: int,
    app_name: str,
    function_name: str,
    environment_name: Optional[str],
) -> Iterable[ChunkResult]:
    try:
        import modal
    except ImportError as error:
        raise RuntimeError("Modal backend requires the 'modal' package") from error

    function = modal.Function.from_name(
        app_name,
        function_name,
        environment_name=environment_name,
    )
    function.update_autoscaler(max_containers=max(1, int(max_containers)))
    return function.map(tasks, order_outputs=False)


def execute_tasks(
    tasks: Sequence[Dict[str, object]],
    backend: str,
    parallelism: int,
    artifact_dir: Path | str,
    progress: Optional[ProgressCallback] = None,
    modal_app_name: str = "nfl-game-sim",
    modal_function_name: str = "simulate_chunk",
    modal_environment: Optional[str] = None,
) -> List[ChunkResult]:
    if not tasks:
        return []
    expected = sum(int(task["count"]) for task in tasks)
    backend_key = backend.strip().lower()
    if backend_key == "local":
        results = _local_results(tasks, max(1, int(parallelism)), artifact_dir)
    elif backend_key == "modal":
        results = _modal_results(
            tasks,
            parallelism,
            modal_app_name,
            modal_function_name,
            modal_environment,
        )
    else:
        raise ValueError(f"Unsupported simulation backend: {backend}")

    completed = 0
    collected: List[ChunkResult] = []
    for result in results:
        collected.append(result)
        completed += int(result["completed_count"])
        if progress is not None:
            progress(completed, expected)
    return collected


def aggregate_results(
    games: Sequence[PreparedGame],
    results: Iterable[Mapping[str, object]],
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    expected = {game.game_id: game for game in games}
    game_rows: Dict[str, List[Dict[str, object]]] = {game_id: [] for game_id in expected}
    player_rows: Dict[str, List[Dict[str, object]]] = {game_id: [] for game_id in expected}

    for result in results:
        game_id = str(result["game_id"])
        if game_id not in expected:
            raise ValueError(f"Received result for unknown game_id {game_id}")
        game_rows[game_id].extend(result["game_rows"])
        player_rows[game_id].extend(result["player_rows"])

    aggregated: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for game_id, game in expected.items():
        rows = game_rows[game_id]
        simulation_ids = [int(row["simulation"]) for row in rows]
        wanted = list(range(1, game.simulations + 1))
        if sorted(simulation_ids) != wanted:
            raise ValueError(
                f"Incomplete or duplicate results for {game_id}: "
                f"expected simulations 1..{game.simulations}, received {len(simulation_ids)} rows"
            )

        rows.sort(key=lambda row: int(row["simulation"]))
        players = player_rows[game_id]
        players.sort(
            key=lambda row: (
                int(row["simulation"]),
                str(row["team"]),
                str(row["player_name"]),
            )
        )
        game_frame = pd.DataFrame(rows)
        player_frame = add_fantasy_points(pd.DataFrame(players))
        for frame in (game_frame, player_frame):
            frame.insert(0, "game_id", game_id)
            frame.insert(1, "week", game.week)
        aggregated[game_id] = game_frame, player_frame
    return aggregated


def run_prepared_games(
    games: Sequence[PreparedGame],
    backend: str,
    parallelism: int,
    artifact_dir: Path | str,
    progress: Optional[ProgressCallback] = None,
    modal_app_name: str = "nfl-game-sim",
    modal_function_name: str = "simulate_chunk",
    modal_environment: Optional[str] = None,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    tasks = build_chunk_tasks(games, artifact_dir, parallelism)
    results = execute_tasks(
        tasks,
        backend,
        parallelism,
        artifact_dir,
        progress,
        modal_app_name,
        modal_function_name,
        modal_environment,
    )
    return aggregate_results(games, results)