"""Process-pool worker for the weekly game simulation dashboard.

Lives in its own module so spawned children import it directly instead of
re-executing the Streamlit app script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from nflsim.engine.fantasy import team_unit_rows
from nflsim.engine.simulation_runtime import Game, Player, SimulationRuntime, Team

_WORKER_RUNTIME: Optional[SimulationRuntime] = None


def init_worker(data_dir: str) -> None:
    global _WORKER_RUNTIME
    try:
        import torch
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass
    _WORKER_RUNTIME = SimulationRuntime(Path(data_dir))


def serialize_players(players: List[Player]) -> List[Dict[str, object]]:
    snapshots: List[Dict[str, object]] = []
    for player in players:
        row = player.attributes.iloc[0].to_dict()
        snapshots.append(
            {
                "name": player.name,
                "position": player.position,
                "player_id": player.player_id,
                "attributes": row,
            }
        )
    return snapshots


def _player_from_snapshot(snapshot: Dict[str, object]) -> Player:
    return Player(
        name=str(snapshot["name"]),
        position=str(snapshot["position"]),
        player_id=str(snapshot["player_id"]),
        attributes=pd.DataFrame([snapshot["attributes"]]),
    )


def simulate_task(task: Dict[str, object]) -> Dict[str, object]:
    runtime = _WORKER_RUNTIME
    if runtime is None:
        runtime = SimulationRuntime(Path(str(task["data_dir"])))
    runtime.reset_nan_input_log()

    simulation = int(task["simulation"])
    home_team = str(task["home_team"])
    away_team = str(task["away_team"])
    team_features = task["team_features"]

    away = Team(away_team, [_player_from_snapshot(s) for s in task["away_players"]], "away", pd.Series(team_features[away_team]), runtime)
    home = Team(home_team, [_player_from_snapshot(s) for s in task["home_players"]], "home", pd.Series(team_features[home_team]), runtime)
    Game(away, home, str(task["roof"]), runtime).simulate_game()

    player_rows: List[Dict[str, object]] = []
    for team in (home, away):
        for player in team.players:
            row: Dict[str, object] = {
                "simulation": simulation,
                "team": team.name,
                "player_name": player.name,
                "player_id": player.player_id,
                "position": player.position,
            }
            row.update(player.stats)
            player_rows.append(row)
        player_rows.extend(team_unit_rows(team, simulation))

    return {
        "game_row": {
            "simulation": simulation,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home.score,
            "away_score": away.score,
            "total": home.score + away.score,
            "spread": home.score - away.score,
        },
        "player_rows": player_rows,
    }


def simulate_chunk_task(task: Dict[str, object]) -> Dict[str, object]:
    """Execute a batch of sequential simulations within a single worker task."""
    runtime = _WORKER_RUNTIME
    if runtime is None:
        runtime = SimulationRuntime(Path(str(task["data_dir"])))

    start_simulation = int(task.get("start_simulation", 1))
    count = int(task.get("count", 1))
    home_team = str(task["home_team"])
    away_team = str(task["away_team"])
    roof = str(task["roof"])
    team_features = task["team_features"]
    away_series = pd.Series(team_features[away_team])
    home_series = pd.Series(team_features[home_team])
    away_template = [_player_from_snapshot(s) for s in task["away_players"]]
    home_template = [_player_from_snapshot(s) for s in task["home_players"]]

    game_rows: List[Dict[str, object]] = []
    player_rows: List[Dict[str, object]] = []

    for sim_idx in range(start_simulation, start_simulation + count):
        runtime.reset_nan_input_log()
        away = Team(
            away_team,
            [Player(p.name, p.position, p.player_id, p.attributes) for p in away_template],
            "away",
            away_series,
            runtime,
        )
        home = Team(
            home_team,
            [Player(p.name, p.position, p.player_id, p.attributes) for p in home_template],
            "home",
            home_series,
            runtime,
        )
        Game(away, home, roof, runtime).simulate_game()

        game_rows.append(
            {
                "simulation": sim_idx,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home.score,
                "away_score": away.score,
                "total": home.score + away.score,
                "spread": home.score - away.score,
            }
        )
        for team in (home, away):
            for player in team.players:
                row: Dict[str, object] = {
                    "simulation": sim_idx,
                    "team": team.name,
                    "player_name": player.name,
                    "player_id": player.player_id,
                    "position": player.position,
                }
                row.update(player.stats)
                player_rows.append(row)
            player_rows.extend(team_unit_rows(team, sim_idx))

    return {
        "game_id": str(task["game_id"]),
        "week": int(task["week"]),
        "game_rows": game_rows,
        "player_rows": player_rows,
        "completed_count": count,
    }
