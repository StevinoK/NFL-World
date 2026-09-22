from __future__ import annotations

import numpy as np
import pandas as pd

from bayesian_feature_builder import MetricSpec, SmoothingConfig
from compare_weighted_rolling_features import (
    _bootstrap_delta,
    _frozen_bayesian_history,
    _metric_summary,
    _prediction_frame,
    _slice_rows,
)
from weighted_rolling_feature_builder import DraftPriorModel, MetricParameters, WeightingConfig


def _comparison_fixture() -> tuple[MetricSpec, pd.DataFrame]:
    rows = []
    for player_index in range(8):
        for season in range(2022, 2026):
            for week in (1, 8, 18):
                rows.append(
                    {
                        "player_id": f"p{player_index}",
                        "player_name": f"Player {player_index}",
                        "game_id": f"{season}_{week:02d}_A_B",
                        "game_date": pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=week * 7),
                        "rate": 0.1 * player_index + 0.01 * week,
                        "count": 5.0 + player_index,
                        "season": season,
                        "week": week,
                        "team": "A" if season < 2025 or player_index % 2 == 0 else "B",
                        "position": ("QB", "RB", "WR", "TE")[player_index % 4],
                        "draft_pick": 20 + player_index * 50,
                        "injury_designation": week == 8,
                        "entity_game_number": (season - 2022) * 3 + (0 if week == 1 else 1 if week == 8 else 2),
                    }
                )
    frame = pd.DataFrame(rows).sort_values(["player_id", "game_date", "game_id"])
    spec_frame = frame[
        ["player_id", "player_name", "game_id", "game_date", "rate", "count"]
    ].copy()
    return MetricSpec("synthetic_rate", "normal", spec_frame, "player_id", "player_name"), frame


def test_frozen_bayesian_prior_does_not_use_test_outcomes() -> None:
    spec, frame = _comparison_fixture()
    config = SmoothingConfig(kind="normal", decay=0.9, season_carryover=0.8)
    baseline = _frozen_bayesian_history(spec, frame, config, 2024)
    changed = frame.copy()
    changed.loc[changed["season"] == 2025, "rate"] += 100.0
    changed_predictions = _frozen_bayesian_history(spec, changed, config, 2024)

    first_test_keys = frame[frame["season"] == 2025].groupby("player_id", as_index=False).first()[
        ["player_id", "game_id"]
    ]
    baseline_first = first_test_keys.merge(baseline, on=["player_id", "game_id"])
    changed_first = first_test_keys.merge(changed_predictions, on=["player_id", "game_id"])
    assert np.allclose(
        baseline_first["bayesian_prediction"],
        changed_first["bayesian_prediction"],
    )


def test_prediction_report_includes_requested_slices_and_intervals() -> None:
    spec, frame = _comparison_fixture()
    parameters = MetricParameters(
        WeightingConfig(kind="normal"),
        DraftPriorModel(intercept=0.25),
        training_rows=len(frame[frame["season"] <= 2024]),
    )
    predictions = _prediction_frame(
        spec,
        frame,
        parameters,
        SmoothingConfig(kind="normal", decay=0.9, season_carryover=0.8),
        2024,
    )
    test = predictions[predictions["season"] == 2025]
    assert {"weighted_prediction", "bayesian_prediction", "team_continuity"}.issubset(test.columns)
    assert test[["weighted_prediction", "bayesian_prediction"]].notna().all().all()

    summary = _metric_summary(test, spec, "overall", "all")
    assert np.isfinite(summary["weighted_rmse"])
    assert np.isfinite(summary["bayesian_rmse"])
    slices = _slice_rows(test, spec)
    labels = {row["slice"] for row in slices}
    assert {"overall", "position", "injury", "team_continuity", "source_week", "career_games"}.issubset(labels)

    low, high = _bootstrap_delta(test, "player_id", samples=25, seed=42)
    assert np.isfinite([low, high]).all()
    assert low <= high