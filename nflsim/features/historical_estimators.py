"""Alternative pregame estimators for rolling football features.

Every player estimator starts from the fitted draft-slot and position prior.
Team estimators use the corresponding league prior through the same interface.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from nflsim.features.bayesian_history import FUTURE_SLOT_FLAG, MetricSpec
from nflsim.features.weighted_rolling import DraftPriorModel, attach_prior_means


Method = Literal[
    "cumulative",
    "ewma",
    "rolling_window",
    "kalman",
    "robust_kalman",
    "logit_kalman",
    "dynamic_beta",
]


@dataclass(frozen=True)
class EstimatorConfig:
    method: Method
    prior_strength: float = 4.0
    alpha: float = 0.25
    window: int = 8
    process_scale: float = 0.02
    observation_scale: float = 1.0
    discount: float = 0.90
    innovation_clip: float = 2.5
    offseason_process_multiplier: float = 4.0
    team_change_process_multiplier: float = 3.0
    injury_observation_multiplier: float = 2.0
    final_week_observation_multiplier: float = 1.5


def candidate_grid(kind: Literal["binary", "normal"]) -> list[EstimatorConfig]:
    candidates = [
        *(EstimatorConfig("cumulative", prior_strength=strength) for strength in (2.0, 8.0, 20.0)),
        *(
            EstimatorConfig("ewma", prior_strength=strength, alpha=alpha)
            for strength in (2.0, 8.0)
            for alpha in (0.10, 0.25, 0.50)
        ),
        *(
            EstimatorConfig("rolling_window", prior_strength=strength, window=window)
            for strength in (2.0, 8.0)
            for window in (4, 8, 16)
        ),
        *(
            EstimatorConfig("kalman", prior_strength=strength, process_scale=process, observation_scale=observation)
            for strength in (2.0, 8.0)
            for process in (0.005, 0.02, 0.08)
            for observation in (0.5, 1.5)
        ),
        *(
            EstimatorConfig(
                "robust_kalman",
                prior_strength=4.0,
                process_scale=process,
                observation_scale=1.0,
                innovation_clip=clip,
            )
            for process in (0.01, 0.05)
            for clip in (1.5, 2.5)
        ),
    ]
    if kind == "binary":
        candidates.extend(
            EstimatorConfig("dynamic_beta", prior_strength=strength, discount=discount)
            for strength in (2.0, 8.0, 20.0)
            for discount in (0.80, 0.90, 0.97)
        )
        candidates.extend(
            EstimatorConfig(
                "logit_kalman",
                prior_strength=strength,
                process_scale=process,
                observation_scale=observation,
            )
            for strength in (2.0, 8.0)
            for process in (0.01, 0.05, 0.20)
            for observation in (0.5, 1.5)
        )
    return candidates


def config_name(config: EstimatorConfig) -> str:
    if config.method == "cumulative":
        return f"cumulative_p{config.prior_strength:g}"
    if config.method == "ewma":
        return f"ewma_a{config.alpha:g}_p{config.prior_strength:g}"
    if config.method == "rolling_window":
        return f"window_{config.window}_p{config.prior_strength:g}"
    if config.method in {"kalman", "logit_kalman"}:
        return (
            f"{config.method}_q{config.process_scale:g}_r{config.observation_scale:g}"
            f"_p{config.prior_strength:g}"
        )
    if config.method == "robust_kalman":
        return f"robust_kalman_q{config.process_scale:g}_c{config.innovation_clip:g}"
    return f"dynamic_beta_d{config.discount:g}_p{config.prior_strength:g}"


def _row_game_number(row: pd.Series, fallback: int) -> int:
    value = row.get("entity_game_number", fallback)
    return fallback if value is None or pd.isna(value) else int(value)


def _row_season(row: pd.Series) -> int:
    value = row.get("season")
    if value is not None and pd.notna(value):
        return int(value)
    text = str(row["game_id"]).split("_")[0]
    return int(text) if text.isdigit() else pd.Timestamp(row["game_date"]).year


def _context_process_multiplier(
    row: pd.Series,
    previous_team: object,
    previous_season: int | None,
    config: EstimatorConfig,
    use_player_context: bool,
) -> float:
    multiplier = 1.0
    season = _row_season(row)
    if previous_season is not None and season > previous_season:
        multiplier *= config.offseason_process_multiplier
    current_team = row.get("team")
    if (
        use_player_context
        and pd.notna(previous_team)
        and pd.notna(current_team)
        and str(previous_team) != str(current_team)
    ):
        multiplier *= config.team_change_process_multiplier
    return multiplier


def _observation_multiplier(row: pd.Series, config: EstimatorConfig, use_player_context: bool) -> float:
    multiplier = 1.0
    if use_player_context and bool(row.get("injury_designation", False)):
        multiplier *= config.injury_observation_multiplier
    week = row.get("week")
    if week is not None and pd.notna(week) and int(week) >= 17:
        multiplier *= config.final_week_observation_multiplier
    return multiplier


def _training_variance(frame: pd.DataFrame) -> float:
    values = pd.to_numeric(frame["rate"], errors="coerce")
    variance = float(values.var(ddof=0))
    return max(variance, 1e-6) if np.isfinite(variance) else 1.0


def predict_history(
    frame: pd.DataFrame,
    spec: MetricSpec,
    prior: DraftPriorModel,
    config: EstimatorConfig,
    variance_frame: pd.DataFrame | None = None,
) -> np.ndarray:
    """Return predictions aligned to ``frame.index`` before each row is observed."""
    use_player_context = spec.scope == "player"
    ordered = frame.copy() if "prior_mean" in frame.columns else attach_prior_means(frame, prior, use_player_context)
    ordered["_output_position"] = np.arange(len(ordered), dtype=int)
    ordered = ordered.sort_values([spec.entity_col, spec.date_col, "game_id"])
    predictions = np.full(len(ordered), np.nan, dtype=float)
    base_variance = _training_variance(variance_frame if variance_frame is not None else frame)

    for _, entity_rows in ordered.groupby(spec.entity_col, sort=False, dropna=False):
        records = entity_rows.to_dict(orient="records")
        first_prior = float(records[0]["prior_mean"])
        if spec.kind == "binary":
            first_prior = float(np.clip(first_prior, 1e-5, 1.0 - 1e-5))
        mean = (
            float(np.log(first_prior / (1.0 - first_prior)))
            if config.method == "logit_kalman"
            else first_prior
        )
        level_variance = base_variance / max(config.prior_strength, 1e-6)
        weighted_sum = first_prior * config.prior_strength
        total_weight = config.prior_strength
        alpha_state = first_prior * config.prior_strength
        beta_state = (1.0 - first_prior) * config.prior_strength
        history: deque[tuple[float, float]] = deque(maxlen=max(config.window, 1))
        previous_team: object = None
        previous_season: int | None = None
        previous_game_number: int | None = None

        for fallback, row in enumerate(records):
            current_game_number = _row_game_number(row, fallback)
            season = _row_season(row)
            gap = 1 if previous_game_number is None else max(current_game_number - previous_game_number, 1)
            process_multiplier = _context_process_multiplier(
                row,
                previous_team,
                previous_season,
                config,
                use_player_context,
            )

            if config.method in {"kalman", "robust_kalman", "logit_kalman"}:
                level_variance += base_variance * config.process_scale * gap * process_multiplier
            elif config.method == "dynamic_beta" and previous_game_number is not None:
                carry = config.discount ** gap
                if previous_season is not None and season > previous_season:
                    carry *= config.discount ** 4
                alpha_state *= carry
                beta_state *= carry

            if config.method == "rolling_window":
                prior_mean = float(row["prior_mean"])
                if spec.kind == "binary":
                    prior_mean = float(np.clip(prior_mean, 0.0, 1.0))
                history_weight = sum(weight for _, weight in history)
                prediction = (
                    sum(value * weight for value, weight in history)
                    + config.prior_strength * prior_mean
                ) / (history_weight + config.prior_strength)
            elif config.method == "dynamic_beta":
                prediction = alpha_state / max(alpha_state + beta_state, 1e-9)
            elif config.method == "logit_kalman":
                prediction = 1.0 / (1.0 + np.exp(-mean))
            else:
                prediction = mean
            output_position = int(row["_output_position"])
            predictions[output_position] = (
                np.clip(prediction, 0.0, 1.0) if spec.kind == "binary" else prediction
            )

            value = pd.to_numeric(pd.Series([row["rate"]]), errors="coerce").iloc[0]
            count = pd.to_numeric(pd.Series([row[spec.count_col]]), errors="coerce").iloc[0]
            is_future = bool(row.get(FUTURE_SLOT_FLAG, False))
            if not is_future and pd.notna(value) and pd.notna(count) and float(count) > 0:
                value = float(value)
                count = float(count)
                observation_multiplier = _observation_multiplier(row, config, use_player_context)
                if config.method == "cumulative":
                    reliability = count / observation_multiplier
                    weighted_sum += reliability * value
                    total_weight += reliability
                    mean = weighted_sum / total_weight
                elif config.method == "ewma":
                    effective_alpha = 1.0 - (1.0 - config.alpha) ** max(count / observation_multiplier, 1.0)
                    mean += effective_alpha * (value - mean)
                elif config.method == "rolling_window":
                    history.append((value, count / observation_multiplier))
                elif config.method == "dynamic_beta":
                    bounded = float(np.clip(value, 0.0, 1.0))
                    reliability = count / observation_multiplier
                    alpha_state += bounded * reliability
                    beta_state += (1.0 - bounded) * reliability
                elif config.method == "logit_kalman":
                    bounded = float(np.clip((value * count + 0.5) / (count + 1.0), 1e-5, 1.0 - 1e-5))
                    observation = float(np.log(bounded / (1.0 - bounded)))
                    observation_variance = (
                        config.observation_scale
                        * observation_multiplier
                        / max(count * bounded * (1.0 - bounded), 1e-5)
                    )
                    gain = level_variance / (level_variance + observation_variance)
                    mean += gain * (observation - mean)
                    level_variance *= 1.0 - gain
                else:
                    observation_variance = (
                        base_variance
                        * config.observation_scale
                        * observation_multiplier
                        / max(count, 1.0)
                    )
                    innovation = value - mean
                    if config.method == "robust_kalman":
                        innovation_sd = np.sqrt(max(level_variance + observation_variance, 1e-9))
                        limit = config.innovation_clip * innovation_sd
                        innovation = float(np.clip(innovation, -limit, limit))
                    gain = level_variance / (level_variance + observation_variance)
                    mean += gain * innovation
                    level_variance *= 1.0 - gain

            previous_team = row.get("team")
            previous_season = season
            previous_game_number = current_game_number
    return predictions