from __future__ import annotations

import unittest
from unittest.mock import patch

from game_sim_execution import (
    PreparedGame,
    aggregate_results,
    build_chunk_tasks,
    chunk_ranges,
    execute_tasks,
)


def prepared_game(game_id: str = "game-1", simulations: int = 5) -> PreparedGame:
    return PreparedGame(
        game_id=game_id,
        week=1,
        home_team="HOME",
        away_team="AWAY",
        roof="outdoors",
        home_players=[],
        away_players=[],
        team_features={"HOME": {}, "AWAY": {}},
        simulations=simulations,
    )


def chunk_result(game_id: str, simulations: list[int]) -> dict[str, object]:
    return {
        "game_id": game_id,
        "week": 1,
        "completed_count": len(simulations),
        "game_rows": [
            {
                "simulation": simulation,
                "home_team": "HOME",
                "away_team": "AWAY",
                "home_score": 20,
                "away_score": 17,
                "total": 37,
                "spread": 3,
            }
            for simulation in simulations
        ],
        "player_rows": [],
    }


class ChunkingTests(unittest.TestCase):
    def test_chunk_ranges_cover_each_simulation_once(self) -> None:
        ranges = chunk_ranges(53, 3)
        covered = [
            simulation
            for start, count in ranges
            for simulation in range(start, start + count)
        ]
        self.assertEqual(covered, list(range(1, 54)))
        self.assertTrue(all(count <= 25 for _, count in ranges))

    def test_tasks_flatten_multiple_games(self) -> None:
        tasks = build_chunk_tasks(
            [prepared_game("one", 3), prepared_game("two", 2)],
            "/artifacts",
            parallelism=2,
        )
        self.assertEqual(sum(int(task["count"]) for task in tasks), 5)
        self.assertEqual({str(task["game_id"]) for task in tasks}, {"one", "two"})
        self.assertEqual({str(task["data_dir"]) for task in tasks}, {"/artifacts"})


class DispatchTests(unittest.TestCase):
    @patch("game_sim_execution._local_results")
    def test_local_progress_tracks_completed_simulations(self, local_results) -> None:
        local_results.return_value = iter(
            [chunk_result("game-1", [3]), chunk_result("game-1", [1, 2])]
        )
        progress: list[tuple[int, int]] = []
        tasks = build_chunk_tasks([prepared_game(simulations=3)], "/artifacts", 1)

        results = execute_tasks(
            tasks,
            "Local",
            1,
            "/artifacts",
            lambda completed, total: progress.append((completed, total)),
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(progress[-1], (3, 3))

    @patch("game_sim_execution._modal_results")
    def test_modal_dispatch_uses_configured_app_and_container_limit(self, modal_results) -> None:
        modal_results.return_value = iter([chunk_result("game-1", [1])])
        tasks = build_chunk_tasks([prepared_game(simulations=1)], "/local-artifacts", 8)

        results = execute_tasks(
            tasks,
            "Modal",
            8,
            "/local-artifacts",
            modal_app_name="test-app",
            modal_function_name="test-function",
            modal_environment="staging",
        )

        self.assertEqual(len(results), 1)
        modal_results.assert_called_once_with(
            tasks,
            8,
            "test-app",
            "test-function",
            "staging",
        )

    @patch("game_sim_execution._modal_results", side_effect=RuntimeError("remote failed"))
    def test_modal_errors_propagate_without_local_fallback(self, modal_results) -> None:
        with self.assertRaisesRegex(RuntimeError, "remote failed"):
            execute_tasks(
                [{"count": 1}],
                "Modal",
                2,
                "/local-artifacts",
            )
        modal_results.assert_called_once()

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported simulation backend"):
            execute_tasks([{"count": 1}], "other", 1, "/artifacts")


class AggregationTests(unittest.TestCase):
    def test_out_of_order_chunks_are_sorted(self) -> None:
        game_frames = aggregate_results(
            [prepared_game(simulations=3)],
            [chunk_result("game-1", [3]), chunk_result("game-1", [1, 2])],
        )
        games, players = game_frames["game-1"]
        self.assertEqual(games["simulation"].tolist(), [1, 2, 3])
        self.assertEqual(games["game_id"].unique().tolist(), ["game-1"])
        self.assertIn("fantasy_points", players.columns)

    def test_missing_or_duplicate_simulations_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Incomplete or duplicate"):
            aggregate_results(
                [prepared_game(simulations=3)],
                [chunk_result("game-1", [1, 1, 3])],
            )


if __name__ == "__main__":
    unittest.main()