from __future__ import annotations

import numpy as np
import pandas as pd

from bayesian_feature_builder import MetricSpec
from historical_feature_estimators import candidate_grid, predict_history
from run_historical_feature_bakeoff import evaluate_metric
from weighted_rolling_feature_builder import DraftPriorModel


def _frame(kind: str) -> tuple[MetricSpec, pd.DataFrame]:
    rates = [0.2, 0.8, 0.4] if kind == "binary" else [2.0, 8.0, 4.0]
    frame = pd.DataFrame(
        {
            "player_id": ["p1"] * 3,
            "player_name": ["Player"] * 3,
            "game_id": [f"2024_{week:02d}_A_B" for week in (1, 2, 3)],
            "game_date": pd.date_range("2024-09-01", periods=3, freq="7D"),
            "season": [2024] * 3,
            "week": [1, 2, 3],
            "team": ["A", "A", "B"],
            "position": ["WR"] * 3,
            "draft_pick": [20] * 3,
            "injury_designation": [False, True, False],
            "entity_game_number": [0, 1, 2],
            "rate": rates,
            "count": [10.0] * 3,
        }
    )
    spec = MetricSpec("test", kind, frame, "player_id", "player_name")
    return spec, frame


def test_all_candidates_are_pregame_and_use_player_prior() -> None:
    for kind in ("binary", "normal"):
        spec, frame = _frame(kind)
        prior_value = 0.35 if kind == "binary" else 3.5
        prior = DraftPriorModel(prior_value, position_coefficients=(("WR", 0.1),))
        changed = frame.copy()
        changed.loc[1, "rate"] = 0.0 if kind == "binary" else -100.0
        for config in candidate_grid(kind):
            baseline = predict_history(frame, spec, prior, config, variance_frame=frame)
            alternate = predict_history(changed, spec, prior, config, variance_frame=frame)
            assert np.isclose(baseline[0], prior_value + 0.1), config
            assert np.isclose(baseline[1], alternate[1]), config
            assert not np.isclose(baseline[2], alternate[2]), config
            assert np.isfinite(baseline).all(), config
            if kind == "binary":
                assert ((baseline >= 0.0) & (baseline <= 1.0)).all(), config


def test_team_change_and_injury_keep_predictions_finite() -> None:
    spec, frame = _frame("normal")
    prior = DraftPriorModel(3.0)
    for config in candidate_grid("normal"):
        predictions = predict_history(frame, spec, prior, config, variance_frame=frame.iloc[:2])
        assert np.isfinite(predictions).all(), config


def test_binary_bakeoff_selects_every_method_family() -> None:
    rows = []
    for player_index in range(8):
        for season in range(2022, 2026):
            for game_index, week in enumerate((1, 8, 18)):
                rows.append(
                    {
                        "player_id": f"p{player_index}",
                        "player_name": f"Player {player_index}",
                        "game_id": f"{season}_{week:02d}_A_B",
                        "game_date": pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=week * 7),
                        "rate": 0.1 + 0.04 * player_index,
                        "count": 5.0 + player_index,
                        "season": season,
                        "week": week,
                        "team": "A",
                        "position": ("QB", "RB", "WR", "TE")[player_index % 4],
                        "draft_year": 2021,
                        "draft_pick": 10 + 25 * player_index,
                        "injury_designation": week == 8,
                        "entity_game_number": (season - 2022) * 3 + game_index,
                    }
                )
    frame = pd.DataFrame(rows).sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    spec = MetricSpec(
        "synthetic_binary",
        "binary",
        frame[["player_id", "player_name", "game_id", "game_date", "rate", "count"]],
        "player_id",
        "player_name",
    )
    result, _ = evaluate_metric(spec, frame, 2023, 2024, 2025, {}, {}, 5, 42)
    methods = {row["method"] for row in result}
    assert {
        "bayesian",
        "cumulative",
        "ewma",
        "rolling_window",
        "kalman",
        "robust_kalman",
        "logit_kalman",
        "dynamic_beta",
    }.issubset(methods)