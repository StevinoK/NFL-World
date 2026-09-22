from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bayesian_feature_builder import FUTURE_SLOT_FLAG, MetricSpec, _group_player_share
from weighted_rolling_feature_builder import (
    DraftPriorModel,
    MetricParameters,
    WeightingConfig,
    apply_metric_spec,
    attach_prior_means,
    enrich_metric_frame,
    fit_draft_prior,
    load_parameters,
    save_parameters,
    week_bucket,
    weighted_rolling_history,
)


def _history_frame(rates: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": ["p1"] * len(rates),
            "player_name": ["Player One"] * len(rates),
            "game_id": [f"2024_{week:02d}_A_B" for week in range(1, len(rates) + 1)],
            "game_date": pd.date_range("2024-09-01", periods=len(rates), freq="7D"),
            "season": [2024] * len(rates),
            "week": list(range(1, len(rates) + 1)),
            "team": ["A"] * len(rates),
            "injury_designation": [False] * len(rates),
            "prior_mean": [0.5] * len(rates),
            "rate": rates,
            "count": [10.0] * len(rates),
        }
    )


def test_estimates_are_strictly_pregame() -> None:
    config = WeightingConfig(kind="binary", prior_weight=4.0)
    baseline = weighted_rolling_history(_history_frame([0.2, 0.9, 0.4]), "player_id", config)
    changed = weighted_rolling_history(_history_frame([0.2, 0.0, 0.4]), "player_id", config)

    assert baseline.loc[0, "average"] == 0.5
    assert baseline.loc[1, "average"] == changed.loc[1, "average"]
    assert baseline.loc[2, "average"] != changed.loc[2, "average"]
    assert baseline["average"].between(0.0, 1.0).all()


def test_future_row_never_updates_history() -> None:
    config = WeightingConfig(kind="binary", game_decay=0.8, prior_game_decay=1.0)
    frame = _history_frame([0.2, 1.0, 0.4])
    frame[FUTURE_SLOT_FLAG] = [False, True, False]
    future_result = weighted_rolling_history(frame, "player_id", config)

    control = frame.copy()
    control.loc[1, "rate"] = np.nan
    control.loc[1, "count"] = 0.0
    control_result = weighted_rolling_history(control, "player_id", config)
    assert future_result.loc[2, "average"] == control_result.loc[2, "average"]


def test_team_injury_and_week_weights_change_history_contributions() -> None:
    frame = _history_frame([0.0, 1.0, 0.5])
    frame.loc[0, "team"] = "B"
    frame.loc[1, "injury_designation"] = True
    frame.loc[0, "week"] = 18
    config = WeightingConfig(
        kind="normal",
        game_decay=1.0,
        season_decay=1.0,
        same_team_multiplier=3.0,
        injury_multiplier=0.25,
        final_week_multiplier=0.5,
        prior_weight=0.0,
    )
    result = weighted_rolling_history(frame, "player_id", config)

    assert result.loc[2, "average"] > 0.5
    assert week_bucket(18) == "final"
    assert week_bucket(10) == "middle"


def test_prior_strength_declines_with_career_games() -> None:
    frame = _history_frame([0.0, 0.0, 0.0])
    config = WeightingConfig(
        kind="normal",
        game_decay=1.0,
        season_decay=1.0,
        prior_weight=10.0,
        prior_game_decay=0.5,
    )
    result = weighted_rolling_history(frame, "player_id", config)
    assert result.loc[2, "average"] < result.loc[1, "average"]


def test_draft_position_prior_is_finite_and_nonconstant() -> None:
    rows = []
    for player_index in range(10):
        for game_index in range(3):
            rows.append(
                {
                    "player_id": f"p{player_index}",
                    "game_id": f"2023_{game_index + 1:02d}_A_B",
                    "game_date": pd.Timestamp("2023-09-01") + pd.Timedelta(days=7 * game_index),
                    "rate": 1.0 - player_index / 12.0,
                    "count": 5.0,
                    "draft_pick": player_index * 25 + 1,
                    "position": SKILL_POSITION_SEQUENCE[player_index % len(SKILL_POSITION_SEQUENCE)],
                }
            )
    prior = fit_draft_prior(pd.DataFrame(rows), "player_id", ridge=0.1)
    predictions = [prior.predict(1, "QB"), prior.predict(250, "TE"), prior.predict(np.nan, "RB")]
    assert np.isfinite(predictions).all()
    assert len(set(np.round(predictions, 8))) > 1


SKILL_POSITION_SEQUENCE = ("QB", "RB", "WR", "TE")


def test_metric_application_and_parameter_round_trip(tmp_path: Path) -> None:
    frame = _history_frame([0.2, 0.4, 0.6]).drop(
        columns=["season", "week", "team", "injury_designation", "prior_mean"]
    )
    spec = MetricSpec("test_rate", "normal", frame, "player_id", "player_name")
    context = _history_frame([0.2, 0.4, 0.6])[
        ["player_id", "game_id", "team", "season", "week", "injury_designation"]
    ].copy()
    context["position"] = "WR"
    context["draft_year"] = 2024
    context["draft_pick"] = 20
    context["report_status"] = pd.NA
    enriched = enrich_metric_frame(spec, context)
    prior = DraftPriorModel(0.3, -0.02, -0.1, (("WR", 0.05),))
    parameters = MetricParameters(WeightingConfig(kind="normal"), prior, 3, 0.1)

    result = apply_metric_spec(spec, context, parameters)
    assert list(result.columns) == [
        "player_id",
        "player_name",
        "game_id",
        "game_date",
        "source",
        "average",
        "bayes_estimate_mean",
        "bayes_estimate_std",
    ]
    assert result["bayes_estimate_mean"].notna().all()
    assert attach_prior_means(enriched, prior, True)["prior_mean"].notna().all()

    path = tmp_path / "weights.json"
    save_parameters({"test_rate": parameters}, path)
    loaded = load_parameters(path)["test_rate"]
    assert loaded.weighting == parameters.weighting
    assert dict(loaded.prior.position_coefficients) == dict(parameters.prior.position_coefficients)


def test_enrichment_drops_missing_player_ids_instead_of_stringifying_them() -> None:
    frame = pd.DataFrame(
        {
            "receiver_player_id": ["p1", np.nan],
            "receiver_player_name": ["Player", np.nan],
            "game_id": ["2024_01_A_B", "2024_01_A_B"],
            "game_date": [pd.Timestamp("2024-09-01")] * 2,
            "rate": [0.2, 1.0],
            "count": [10.0, 10.0],
        }
    )
    spec = MetricSpec("test", "binary", frame, "receiver_player_id", "receiver_player_name")
    context = pd.DataFrame(
        {
            "player_id": ["p1"],
            "game_id": ["2024_01_A_B"],
            "team": ["A"],
            "position": ["WR"],
            "draft_year": [2024],
            "draft_pick": [10],
            "injury_designation": [False],
            "report_status": [pd.NA],
        }
    )
    enriched = enrich_metric_frame(spec, context)
    assert enriched["receiver_player_id"].tolist() == ["p1"]


def test_active_player_share_uses_team_denominator_and_deduplicates_aliases() -> None:
    pbp = pd.DataFrame(
        {
            "player_id": ["p1"],
            "player_name": ["A. Player"],
            "team": ["T"],
            "game_id": ["2024_01_X_T"],
            "game_date": [pd.Timestamp("2024-09-01")],
            "event": [1],
        }
    )
    active = pd.DataFrame(
        {
            "player_id": ["p1", "p1", "p2"],
            "player_name": ["Alias A", "A. Player", "B. Player"],
            "team": ["T", "T", "T"],
            "game_id": ["2024_01_X_T"] * 3,
            "game_date": [pd.Timestamp("2024-09-01")] * 3,
        }
    )
    result = _group_player_share(
        pbp,
        "player_id",
        "player_name",
        "team",
        pd.Series([True]),
        "event",
        active,
    ).set_index("player_id")
    assert len(result) == 2
    assert result.loc["p1", "rate"] == 1.0
    assert result.loc["p2", "rate"] == 0.0
    assert (result["count"] == 1.0).all()