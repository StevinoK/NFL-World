"""Context-weighted rolling player and team features.

This module is a drop-in research alternative to ``bayesian_feature_builder``.
It keeps that builder's metric names and table schemas while replacing the
posterior update with positive, context-dependent rolling weights.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import nfl_data_py as nfl
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from bayesian_feature_builder import (
    DEFAULT_CACHE_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_YEARS,
    EPS,
    FUTURE_SLOT_FLAG,
    FutureContext,
    MetricSpec,
    _future_placeholder_frame,
    apply_metric_spec as apply_bayesian_metric_spec,
    build_future_context,
    build_wide_tables,
    collect_player_metric_specs,
    collect_team_metric_specs,
    load_pbp_data,
    standardize_player_frame,
    standardize_team_frame,
)


WEEK_BUCKETS = ("early", "middle", "late", "final")
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
DEFAULT_WEIGHTS_FILE = Path(__file__).with_name("weighted_rolling_params.json")


@dataclass(frozen=True)
class WeightingConfig:
    """Coefficients for one metric's positive observation-weight kernel."""

    kind: Literal["binary", "normal"]
    game_decay: float = 0.90
    season_decay: float = 0.80
    same_team_multiplier: float = 1.0
    injury_multiplier: float = 1.0
    early_week_multiplier: float = 1.0
    late_week_multiplier: float = 1.0
    final_week_multiplier: float = 1.0
    prior_weight: float = 4.0
    prior_game_decay: float = 0.90

    def __post_init__(self) -> None:
        unit_interval = {
            "game_decay": self.game_decay,
            "season_decay": self.season_decay,
            "injury_multiplier": self.injury_multiplier,
            "prior_game_decay": self.prior_game_decay,
        }
        for name, value in unit_interval.items():
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1].")
        if self.same_team_multiplier < 1.0:
            raise ValueError("same_team_multiplier must be at least 1.")
        if min(
            self.early_week_multiplier,
            self.late_week_multiplier,
            self.final_week_multiplier,
            self.prior_weight,
        ) < 0.0:
            raise ValueError("Week multipliers and prior_weight must be nonnegative.")


@dataclass(frozen=True)
class DraftPriorModel:
    intercept: float
    log_pick_coefficient: float = 0.0
    undrafted_coefficient: float = 0.0
    position_coefficients: tuple[tuple[str, float], ...] = ()

    def predict(self, draft_pick: object, position: object) -> float:
        try:
            numeric_pick = float(draft_pick)
        except (TypeError, ValueError):
            numeric_pick = float("nan")
        undrafted = float(not np.isfinite(numeric_pick) or numeric_pick <= 0)
        effective_pick = 270.0 if undrafted else float(numeric_pick)
        position_map = dict(self.position_coefficients)
        normalized_position = str(position) if pd.notna(position) else "OTHER"
        return float(
            self.intercept
            + self.log_pick_coefficient * np.log1p(effective_pick)
            + self.undrafted_coefficient * undrafted
            + position_map.get(normalized_position, position_map.get("OTHER", 0.0))
        )


@dataclass(frozen=True)
class MetricParameters:
    weighting: WeightingConfig
    prior: DraftPriorModel
    training_rows: int = 0
    training_loss: float | None = None


def week_bucket(week: int | float | None) -> str:
    """Map a regular-season week to the compact source-week effect buckets."""
    if week is None or pd.isna(week):
        return "middle"
    value = int(week)
    if value <= 4:
        return "early"
    if value <= 12:
        return "middle"
    if value <= 16:
        return "late"
    return "final"


def _week_multiplier(week: int | float | None, config: WeightingConfig) -> float:
    bucket = week_bucket(week)
    if bucket == "early":
        return config.early_week_multiplier
    if bucket == "late":
        return config.late_week_multiplier
    if bucket == "final":
        return config.final_week_multiplier
    return 1.0


def _history_weight(
    observation: dict[str, object],
    current_team: object,
    current_season: int,
    current_game_number: int,
    config: WeightingConfig,
    use_player_context: bool,
) -> float:
    games_ago = max(current_game_number - int(observation["game_number"]), 1)
    seasons_ago = max(current_season - int(observation["season"]), 0)
    weight = float(observation["count"])
    weight *= config.game_decay ** (games_ago - 1)
    weight *= config.season_decay ** seasons_ago
    weight *= _week_multiplier(observation.get("week"), config)

    if use_player_context:
        source_team = observation.get("team")
        if pd.notna(current_team) and pd.notna(source_team) and str(current_team) == str(source_team):
            weight *= config.same_team_multiplier
        if bool(observation.get("injury_designation", False)):
            weight *= config.injury_multiplier
    return max(float(weight), 0.0)


def weighted_rolling_history(
    frame: pd.DataFrame,
    entity_col: str,
    config: WeightingConfig,
    *,
    value_col: str = "rate",
    count_col: str = "count",
    date_col: str = "game_date",
    name_col: str | None = None,
    use_player_context: bool = True,
) -> pd.DataFrame:
    """Emit a strictly pregame weighted estimate for every entity-game row.

    Optional contextual columns are ``season``, ``week``, ``team``,
    ``injury_designation``, and ``prior_mean``. A future row marked by
    ``FUTURE_SLOT_FLAG`` receives an estimate but never enters history.
    """
    required = {entity_col, "game_id", date_col, value_col, count_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Weighted history frame is missing columns: {missing}")
    if frame.empty:
        return frame.copy()

    ordered = frame.copy()
    ordered[date_col] = pd.to_datetime(ordered[date_col], errors="coerce")
    ordered = ordered.sort_values([entity_col, date_col, "game_id"])
    observed = ordered.loc[
        ~ordered.get(FUTURE_SLOT_FLAG, pd.Series(False, index=ordered.index)).fillna(False).astype(bool)
    ]
    observed_values = pd.to_numeric(observed[value_col], errors="coerce")
    global_mean = float(observed_values.mean())
    if config.kind == "binary":
        global_mean = float(np.clip(global_mean, EPS, 1.0 - EPS))

    records: list[dict[str, object]] = []
    for _, entity_rows in ordered.groupby(entity_col, sort=False, dropna=False):
        entity_rows = entity_rows.sort_values([date_col, "game_id"]).reset_index(drop=True)
        history: list[dict[str, object]] = []
        game_number = 0

        for row in entity_rows.to_dict(orient="records"):
            row_game_number = row.get("entity_game_number", game_number)
            if row_game_number is None or pd.isna(row_game_number):
                row_game_number = game_number
            row_game_number = int(row_game_number)
            season = row.get("season")
            if season is None or pd.isna(season):
                game_text = str(row["game_id"]).split("_")
                season = int(game_text[0]) if game_text and game_text[0].isdigit() else row[date_col].year
            current_team = row.get("team")
            weighted_sum = 0.0
            total_weight = 0.0
            weighted_square_sum = 0.0
            for observation in history:
                weight = _history_weight(
                    observation,
                    current_team,
                    int(season),
                    row_game_number,
                    config,
                    use_player_context,
                )
                value = float(observation["value"])
                weighted_sum += weight * value
                weighted_square_sum += weight * value * value
                total_weight += weight

            prior_mean = row.get("prior_mean", global_mean)
            if prior_mean is None or pd.isna(prior_mean):
                prior_mean = global_mean
            prior_weight = config.prior_weight * (config.prior_game_decay ** row_game_number)
            denominator = total_weight + prior_weight
            estimate = (
                (weighted_sum + prior_weight * float(prior_mean)) / denominator
                if denominator > 0
                else float(prior_mean)
            )
            if config.kind == "binary":
                estimate = float(np.clip(estimate, 0.0, 1.0))

            if total_weight > 0:
                variance = max(weighted_square_sum / total_weight - (weighted_sum / total_weight) ** 2, 0.0)
                effective_numerator = total_weight * total_weight
                effective_denominator = sum(
                    _history_weight(
                        observation,
                        current_team,
                        int(season),
                        row_game_number,
                        config,
                        use_player_context,
                    ) ** 2
                    for observation in history
                )
                effective_n = effective_numerator / effective_denominator if effective_denominator > 0 else 0.0
                estimate_std = np.sqrt(variance / effective_n) if effective_n > 1.0 else np.nan
            else:
                estimate_std = np.nan

            record = {
                entity_col: row[entity_col],
                "game_id": row["game_id"],
                "game_date": row[date_col],
                "average": estimate,
                "bayes_estimate_mean": estimate,
                "bayes_estimate_std": estimate_std,
            }
            if name_col is not None:
                record[name_col] = row.get(name_col)
            records.append(record)

            is_future = bool(row.get(FUTURE_SLOT_FLAG, False))
            if is_future:
                game_number += 1
                continue

            count = pd.to_numeric(pd.Series([row[count_col]]), errors="coerce").iloc[0]
            value = pd.to_numeric(pd.Series([row[value_col]]), errors="coerce").iloc[0]
            if pd.notna(count) and pd.notna(value) and float(count) > 0:
                history.append(
                    {
                        "value": float(value),
                        "count": float(count),
                        "season": int(season),
                        "week": row.get("week"),
                        "team": row.get("team"),
                        "injury_designation": row.get("injury_designation", False),
                        "game_number": row_game_number,
                    }
                )
            game_number += 1

    return pd.DataFrame.from_records(records)


def _fast_weighted_predictions(
    frame: pd.DataFrame,
    entity_col: str,
    config: WeightingConfig,
    *,
    count_col: str,
    date_col: str,
    use_player_context: bool,
) -> np.ndarray:
    """Linear-time equivalent of the weighted mean used during parameter fitting."""
    ordered = frame.sort_values([entity_col, date_col, "game_id"]).copy()
    predictions = pd.Series(np.nan, index=ordered.index, dtype=float)
    for _, entity_rows in ordered.groupby(entity_col, sort=False, dropna=False):
        base_weight = 0.0
        base_sum = 0.0
        team_weight: dict[str, float] = {}
        team_sum: dict[str, float] = {}
        previous_game_number: int | None = None
        previous_season: int | None = None
        pending: dict[str, object] | None = None

        for row_index, row in entity_rows.iterrows():
            fallback_game_number = 0 if previous_game_number is None else previous_game_number + 1
            current_game_number = row.get("entity_game_number", fallback_game_number)
            if current_game_number is None or pd.isna(current_game_number):
                current_game_number = fallback_game_number
            current_game_number = int(current_game_number)
            season = row.get("season")
            if season is None or pd.isna(season):
                parts = str(row["game_id"]).split("_")
                season = int(parts[0]) if parts and parts[0].isdigit() else pd.Timestamp(row[date_col]).year
            current_season = int(season)

            if previous_game_number is not None and previous_season is not None:
                game_gap = max(current_game_number - previous_game_number, 1)
                season_gap = max(current_season - previous_season, 0)
                carry = (config.game_decay ** game_gap) * (config.season_decay ** season_gap)
                base_weight *= carry
                base_sum *= carry
                for team in list(team_weight):
                    team_weight[team] *= carry
                    team_sum[team] *= carry
                if pending is not None:
                    initial = (
                        float(pending["count"])
                        * _week_multiplier(pending.get("week"), config)
                        * (config.game_decay ** (game_gap - 1))
                        * (config.season_decay ** season_gap)
                    )
                    if use_player_context and bool(pending.get("injury_designation", False)):
                        initial *= config.injury_multiplier
                    value = float(pending["value"])
                    base_weight += initial
                    base_sum += initial * value
                    source_team = pending.get("team")
                    if use_player_context and pd.notna(source_team):
                        team_key = str(source_team)
                        team_weight[team_key] = team_weight.get(team_key, 0.0) + initial
                        team_sum[team_key] = team_sum.get(team_key, 0.0) + initial * value

            current_team = row.get("team")
            same_weight = team_weight.get(str(current_team), 0.0) if pd.notna(current_team) else 0.0
            same_sum = team_sum.get(str(current_team), 0.0) if pd.notna(current_team) else 0.0
            history_weight = base_weight + (config.same_team_multiplier - 1.0) * same_weight
            history_sum = base_sum + (config.same_team_multiplier - 1.0) * same_sum
            prior_mean = row.get("prior_mean")
            prior_mean = float(prior_mean) if prior_mean is not None and pd.notna(prior_mean) else 0.0
            prior_weight = config.prior_weight * (config.prior_game_decay ** current_game_number)
            denominator = history_weight + prior_weight
            prediction = (history_sum + prior_weight * prior_mean) / denominator if denominator > 0 else prior_mean
            predictions.loc[row_index] = np.clip(prediction, 0.0, 1.0) if config.kind == "binary" else prediction

            value = pd.to_numeric(pd.Series([row["rate"]]), errors="coerce").iloc[0]
            count = pd.to_numeric(pd.Series([row[count_col]]), errors="coerce").iloc[0]
            is_future = bool(row.get(FUTURE_SLOT_FLAG, False))
            pending = None
            if not is_future and pd.notna(value) and pd.notna(count) and float(count) > 0:
                pending = {
                    "value": float(value),
                    "count": float(count),
                    "week": row.get("week"),
                    "team": row.get("team"),
                    "injury_designation": row.get("injury_designation", False),
                }
            previous_game_number = current_game_number
            previous_season = current_season
    return predictions.reindex(frame.index).to_numpy(dtype=float)


def _season_week_columns(game_ids: pd.Series) -> pd.DataFrame:
    parts = game_ids.astype(str).str.split("_", expand=True)
    season_values = parts[0] if 0 in parts else pd.Series(np.nan, index=game_ids.index)
    week_values = parts[1] if 1 in parts else pd.Series(np.nan, index=game_ids.index)
    return pd.DataFrame(
        {
            "season": pd.to_numeric(season_values, errors="coerce"),
            "week": pd.to_numeric(week_values, errors="coerce"),
        },
        index=game_ids.index,
    )


def load_player_metadata() -> pd.DataFrame:
    players = nfl.import_players().copy()
    required = {"gsis_id", "position"}
    missing = sorted(required.difference(players.columns))
    if missing:
        raise ValueError(f"Player metadata is missing required columns: {missing}")
    for column in ("draft_pick", "draft_year", "rookie_season"):
        if column not in players.columns:
            players[column] = np.nan
    players["position"] = players["position"].replace({"FB": "RB"}).fillna("OTHER")
    players["position"] = players["position"].where(players["position"].isin(SKILL_POSITIONS), "OTHER")
    players["draft_year"] = pd.to_numeric(players["draft_year"], errors="coerce").combine_first(
        pd.to_numeric(players["rookie_season"], errors="coerce")
    )
    players["draft_pick"] = pd.to_numeric(players["draft_pick"], errors="coerce")
    return (
        players[["gsis_id", "position", "draft_year", "draft_pick"]]
        .dropna(subset=["gsis_id"])
        .drop_duplicates("gsis_id", keep="last")
        .rename(columns={"gsis_id": "player_id"})
    )


def load_injury_context(years: Iterable[int]) -> pd.DataFrame:
    columns = ["player_id", "season", "week", "injury_designation", "report_status"]
    try:
        injuries = nfl.import_injuries(list(years)).copy()
    except Exception as error:
        print(f"Skipping injury enrichment: {error}")
        return pd.DataFrame(columns=columns)
    required = {"season", "week", "gsis_id", "report_status"}
    if not required.issubset(injuries.columns):
        return pd.DataFrame(columns=columns)
    if "game_type" in injuries.columns:
        injuries = injuries[injuries["game_type"] == "REG"]
    if "date_modified" in injuries.columns:
        injuries["date_modified"] = pd.to_datetime(injuries["date_modified"], errors="coerce", utc=True)
        injuries = injuries.sort_values("date_modified")
    injuries = injuries.drop_duplicates(["season", "week", "gsis_id"], keep="last")
    status = injuries["report_status"].astype("string").str.strip()
    injuries["injury_designation"] = status.notna() & status.ne("")
    injuries["report_status"] = status
    return injuries.rename(columns={"gsis_id": "player_id"})[columns]


def build_player_game_context(
    pbp: pd.DataFrame,
    years: Iterable[int],
    future: FutureContext | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for role in ("passer", "rusher", "receiver"):
        player_col = f"{role}_player_id"
        if player_col not in pbp.columns:
            continue
        subset = pbp.loc[
            pbp[player_col].notna(),
            [player_col, "posteam", "game_id", "game_date"],
        ].copy()
        frames.append(subset.rename(columns={player_col: "player_id", "posteam": "team"}))
    if not frames:
        return pd.DataFrame(
            columns=[
                "player_id",
                "team",
                "game_id",
                "game_date",
                "season",
                "week",
                "position",
                "draft_year",
                "draft_pick",
                "injury_designation",
                "report_status",
            ]
        )

    context = pd.concat(frames, ignore_index=True)
    context["player_id"] = context["player_id"].astype(str)
    context = context.sort_values(["game_date", "game_id"]).drop_duplicates(
        ["player_id", "game_id"], keep="last"
    )
    context[["season", "week"]] = _season_week_columns(context["game_id"])

    if future is not None and not future.player_slots.empty:
        slots = future.player_slots[["player_id", "team", "game_id", "game_date"]].copy()
        slots["player_id"] = slots["player_id"].astype(str)
        slots[["season", "week"]] = _season_week_columns(slots["game_id"])
        context = pd.concat([context, slots], ignore_index=True).drop_duplicates(
            ["player_id", "game_id"], keep="last"
        )

    context = context.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    context["entity_game_number"] = context.groupby("player_id").cumcount()
    context = context.merge(load_player_metadata(), on="player_id", how="left")
    injuries = load_injury_context(years)
    if not injuries.empty:
        injuries["player_id"] = injuries["player_id"].astype(str)
        context = context.merge(injuries, on=["player_id", "season", "week"], how="left")
    if "injury_designation" not in context.columns:
        context["injury_designation"] = False
    context["injury_designation"] = context["injury_designation"].fillna(False).astype(bool)
    if "report_status" not in context.columns:
        context["report_status"] = pd.NA
    return context


def enrich_metric_frame(spec: MetricSpec, player_context: pd.DataFrame) -> pd.DataFrame:
    frame = spec.frame.copy()
    frame[["season", "week"]] = _season_week_columns(frame["game_id"])
    if spec.scope == "team":
        return frame
    context_columns = [
        "player_id",
        "game_id",
        "team",
        "position",
        "draft_year",
        "draft_pick",
        "injury_designation",
        "report_status",
    ]
    if "entity_game_number" in player_context.columns:
        context_columns.append("entity_game_number")
    context = player_context[context_columns].rename(columns={"player_id": spec.entity_col})
    context = context[context[spec.entity_col].notna()].copy()
    frame = frame[frame[spec.entity_col].notna()].copy()
    context[spec.entity_col] = context[spec.entity_col].astype("string")
    frame[spec.entity_col] = frame[spec.entity_col].astype("string")
    return frame.merge(context, on=[spec.entity_col, "game_id"], how="left")


def fit_draft_prior(frame: pd.DataFrame, entity_col: str, ridge: float = 10.0) -> DraftPriorModel:
    usable = frame.dropna(subset=[entity_col, "rate"]).sort_values(
        [entity_col, "game_date", "game_id"]
    ).copy()
    if usable.empty:
        return DraftPriorModel(intercept=0.0)
    usable["career_game"] = usable.groupby(entity_col).cumcount()
    usable = usable[usable["career_game"] < 8]
    usable["count"] = pd.to_numeric(usable["count"], errors="coerce").fillna(1.0)
    usable["weighted_value"] = pd.to_numeric(usable["rate"], errors="coerce") * usable["count"]
    entity = usable.groupby(entity_col, as_index=False).agg(
        target_sum=("weighted_value", "sum"),
        exposure=("count", "sum"),
        draft_pick=("draft_pick", "last"),
        position=("position", "last"),
    )
    entity = entity[entity["exposure"] > 0].copy()
    fallback = float(np.average(usable["rate"], weights=usable["count"]))
    if len(entity) < 8:
        return DraftPriorModel(intercept=fallback)
    target = (entity["target_sum"] / entity["exposure"]).to_numpy(dtype=float)
    pick = pd.to_numeric(entity["draft_pick"], errors="coerce")
    undrafted = (pick.isna() | (pick <= 0)).astype(float).to_numpy()
    log_pick = np.log1p(pick.where(pick.notna() & (pick > 0), 270.0)).to_numpy(dtype=float)
    position = entity["position"].fillna("OTHER")
    position = position.where(position.isin(SKILL_POSITIONS), "OTHER")
    position_names = [*SKILL_POSITIONS, "OTHER"]
    position_matrix = np.column_stack([(position == name).astype(float) for name in position_names])
    design = np.column_stack([np.ones(len(entity)), log_pick, undrafted, position_matrix])
    penalty = np.eye(design.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return DraftPriorModel(
        intercept=float(coefficients[0]),
        log_pick_coefficient=float(coefficients[1]),
        undrafted_coefficient=float(coefficients[2]),
        position_coefficients=tuple(
            (name, float(value)) for name, value in zip(position_names, coefficients[3:])
        ),
    )


def attach_prior_means(
    frame: pd.DataFrame,
    prior: DraftPriorModel,
    use_player_context: bool,
) -> pd.DataFrame:
    result = frame.copy()
    if use_player_context:
        draft_picks = result["draft_pick"] if "draft_pick" in result else pd.Series(np.nan, index=result.index)
        positions = result["position"] if "position" in result else pd.Series("OTHER", index=result.index)
        result["prior_mean"] = [
            prior.predict(pick, position) for pick, position in zip(draft_picks, positions)
        ]
    else:
        result["prior_mean"] = prior.intercept
    return result


def _config_from_vector(kind: Literal["binary", "normal"], values: np.ndarray) -> WeightingConfig:
    return WeightingConfig(
        kind=kind,
        game_decay=float(np.exp(values[0])),
        season_decay=float(np.exp(values[1])),
        same_team_multiplier=float(np.exp(values[2])),
        injury_multiplier=float(np.exp(values[3])),
        early_week_multiplier=float(np.exp(values[4])),
        late_week_multiplier=float(np.exp(values[5])),
        final_week_multiplier=float(np.exp(values[6])),
        prior_weight=float(np.exp(values[7])),
        prior_game_decay=float(np.exp(values[8])),
    )


def fit_weighting_config(
    spec: MetricSpec,
    frame: pd.DataFrame,
    prior: DraftPriorModel,
    regularization: float = 0.02,
    max_rows: int = 3000,
) -> tuple[WeightingConfig, float]:
    fit_frame = frame.dropna(subset=["rate", spec.count_col]).copy()
    if len(fit_frame) > max_rows:
        fit_frame = fit_frame.sort_values(spec.date_col).tail(max_rows)
    use_player_context = spec.scope == "player"
    fit_frame = attach_prior_means(fit_frame, prior, use_player_context)
    neutral = np.log(np.array([0.90, 0.80, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 0.90]))
    bounds = [
        (np.log(0.50), np.log(0.999)),
        (np.log(0.40), np.log(1.0)),
        (0.0, np.log(3.0)),
        (np.log(0.25), 0.0),
        (np.log(0.50), np.log(1.50)),
        (np.log(0.50), np.log(1.50)),
        (np.log(0.25), np.log(1.50)),
        (np.log(0.10), np.log(50.0)),
        (np.log(0.50), np.log(1.0)),
    ]
    if not use_player_context:
        bounds[2] = (0.0, 0.0)
        bounds[3] = (0.0, 0.0)

    def objective(values: np.ndarray) -> float:
        prediction_values = _fast_weighted_predictions(
            fit_frame,
            spec.entity_col,
            _config_from_vector(spec.kind, values),
            count_col=spec.count_col,
            date_col=spec.date_col,
            use_player_context=use_player_context,
        )
        actual = pd.to_numeric(fit_frame["rate"], errors="coerce").to_numpy(dtype=float)
        exposure = pd.to_numeric(fit_frame[spec.count_col], errors="coerce").fillna(1.0).clip(lower=1.0).to_numpy(dtype=float)
        valid = np.isfinite(prediction_values) & np.isfinite(actual) & np.isfinite(exposure)
        if not valid.any():
            return float("inf")
        squared_error = np.square(prediction_values[valid] - actual[valid])
        prediction_loss = float(np.average(squared_error, weights=exposure[valid]))
        return prediction_loss + float(regularization) * float(np.mean(np.square(values - neutral)))

    if len(fit_frame) < 20:
        return _config_from_vector(spec.kind, neutral), objective(neutral)
    result = minimize(
        objective,
        neutral,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 6, "maxfun": 30, "ftol": 1e-7},
    )
    values = result.x if np.isfinite(result.fun) else neutral
    return _config_from_vector(spec.kind, values), float(objective(values))


def _parameters_to_dict(parameters: MetricParameters) -> dict[str, object]:
    return {
        "weighting": parameters.weighting.__dict__,
        "prior": {
            "intercept": parameters.prior.intercept,
            "log_pick_coefficient": parameters.prior.log_pick_coefficient,
            "undrafted_coefficient": parameters.prior.undrafted_coefficient,
            "position_coefficients": dict(parameters.prior.position_coefficients),
        },
        "training_rows": parameters.training_rows,
        "training_loss": parameters.training_loss,
    }


def save_parameters(parameters: dict[str, MetricParameters], path: Path = DEFAULT_WEIGHTS_FILE) -> None:
    payload = {
        "metrics": {
            name: _parameters_to_dict(value)
            for name, value in sorted(parameters.items())
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_parameters(path: Path = DEFAULT_WEIGHTS_FILE) -> dict[str, MetricParameters]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, MetricParameters] = {}
    for name, value in payload.get("metrics", {}).items():
        weighting = WeightingConfig(**value["weighting"])
        prior_value = value["prior"]
        prior = DraftPriorModel(
            intercept=float(prior_value["intercept"]),
            log_pick_coefficient=float(prior_value.get("log_pick_coefficient", 0.0)),
            undrafted_coefficient=float(prior_value.get("undrafted_coefficient", 0.0)),
            position_coefficients=tuple(
                sorted(prior_value.get("position_coefficients", {}).items())
            ),
        )
        result[name] = MetricParameters(
            weighting=weighting,
            prior=prior,
            training_rows=int(value.get("training_rows", 0)),
            training_loss=value.get("training_loss"),
        )
    return result


def fit_metric_parameters(spec: MetricSpec, enriched_frame: pd.DataFrame) -> MetricParameters:
    if enriched_frame.empty:
        return MetricParameters(
            WeightingConfig(kind=spec.kind),
            DraftPriorModel(intercept=0.0),
            0,
            None,
        )
    if spec.scope == "player":
        prior = fit_draft_prior(enriched_frame, spec.entity_col)
    else:
        values = pd.to_numeric(enriched_frame["rate"], errors="coerce")
        counts = pd.to_numeric(enriched_frame[spec.count_col], errors="coerce").fillna(1.0)
        valid = values.notna() & counts.gt(0)
        intercept = float(np.average(values[valid], weights=counts[valid])) if valid.any() else 0.0
        prior = DraftPriorModel(intercept=intercept)
    weighting, loss = fit_weighting_config(spec, enriched_frame, prior)
    return MetricParameters(weighting, prior, int(len(enriched_frame)), loss)


def apply_metric_spec(
    spec: MetricSpec,
    player_context: pd.DataFrame,
    parameters: MetricParameters,
    future: FutureContext | None = None,
) -> pd.DataFrame:
    if spec.kind == "direct":
        return apply_bayesian_metric_spec(spec, future=future)

    frame = enrich_metric_frame(spec, player_context)
    if future is not None:
        placeholder = _future_placeholder_frame(spec, future)
        if not placeholder.empty:
            placeholder_spec = MetricSpec(
                source=spec.source,
                kind=spec.kind,
                frame=placeholder,
                entity_col=spec.entity_col,
                name_col=spec.name_col,
                count_col=spec.count_col,
                date_col=spec.date_col,
                scope=spec.scope,
            )
            placeholder = enrich_metric_frame(placeholder_spec, player_context)
            frame[FUTURE_SLOT_FLAG] = False
            frame = pd.concat([frame, placeholder], ignore_index=True)
    frame = attach_prior_means(frame, parameters.prior, spec.scope == "player")
    smoothed = weighted_rolling_history(
        frame,
        spec.entity_col,
        parameters.weighting,
        count_col=spec.count_col,
        date_col=spec.date_col,
        name_col=spec.name_col,
        use_player_context=spec.scope == "player",
    )
    if spec.scope == "team":
        return standardize_team_frame(smoothed, spec.source)
    return standardize_player_frame(smoothed, spec.source)


def _training_frame(frame: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    return frame[pd.to_numeric(frame["season"], errors="coerce") <= int(cutoff)].copy()


def _default_metric_parameters(spec: MetricSpec, frame: pd.DataFrame) -> MetricParameters:
    if spec.scope == "player":
        prior = fit_draft_prior(frame, spec.entity_col)
    else:
        values = pd.to_numeric(frame["rate"], errors="coerce")
        prior = DraftPriorModel(float(values.mean()) if values.notna().any() else 0.0)
    return MetricParameters(WeightingConfig(kind=spec.kind), prior, int(len(frame)), None)


def build_feature_tables(
    years: Iterable[int] = DEFAULT_YEARS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cache_path: Path = DEFAULT_CACHE_PATH,
    weights_file: Path = DEFAULT_WEIGHTS_FILE,
    fit_weights: bool = False,
    fit_through_season: int | None = None,
    future_weeks: int = 0,
    write_wide: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = list(years)
    pbp = load_pbp_data(years, cache_path=cache_path)
    future = build_future_context(pbp, weeks_ahead=future_weeks) if future_weeks > 0 else None
    player_context = build_player_game_context(pbp, years, future=future)
    player_specs = collect_player_metric_specs(pbp)
    team_specs = collect_team_metric_specs(pbp, future=future)
    parameters = load_parameters(weights_file)
    cutoff = fit_through_season if fit_through_season is not None else max(years) - 1

    enriched_frames = {
        spec.source: enrich_metric_frame(spec, player_context)
        for spec in [*player_specs, *team_specs]
        if spec.kind != "direct"
    }
    if fit_weights:
        parameters = {}
        for spec in [*player_specs, *team_specs]:
            if spec.kind == "direct":
                continue
            training = _training_frame(enriched_frames[spec.source], cutoff)
            parameters[spec.source] = fit_metric_parameters(spec, training)
            loss = parameters[spec.source].training_loss
            print(f"Fitted {spec.source}: loss={loss:.6f}" if loss is not None else f"Fitted {spec.source}")
        save_parameters(parameters, weights_file)

    for spec in [*player_specs, *team_specs]:
        if spec.kind == "direct" or spec.source in parameters:
            continue
        training = _training_frame(enriched_frames[spec.source], cutoff)
        parameters[spec.source] = _default_metric_parameters(spec, training)

    player_frames = [
        apply_metric_spec(spec, player_context, parameters.get(spec.source), future)
        if spec.kind != "direct"
        else apply_bayesian_metric_spec(spec, future=future)
        for spec in player_specs
    ]
    team_frames = [
        apply_metric_spec(spec, player_context, parameters.get(spec.source), future)
        if spec.kind != "direct"
        else apply_bayesian_metric_spec(spec, future=future)
        for spec in team_specs
    ]
    player_stats = pd.concat(player_frames, ignore_index=True).drop_duplicates(
        ["player_id", "player_name", "game_id", "source"], keep="last"
    )
    team_stats = pd.concat(team_frames, ignore_index=True).drop_duplicates(
        ["team", "game_id", "source"], keep="last"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    player_stats.to_csv(output_dir / "player_stats.csv", index=False)
    team_stats.to_csv(output_dir / "Team_Stats.csv", index=False)
    if write_wide:
        build_wide_tables(player_stats, team_stats, output_dir)
    return player_stats, team_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build context-weighted rolling football feature tables.")
    parser.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_YEARS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--weights-file", type=Path, default=DEFAULT_WEIGHTS_FILE)
    parser.add_argument("--fit-weights", action="store_true")
    parser.add_argument("--fit-through-season", type=int)
    parser.add_argument("--future-weeks", type=int, default=1)
    parser.add_argument("--write-wide", action="store_true")
    args = parser.parse_args()
    build_feature_tables(
        years=args.years,
        output_dir=args.output_dir,
        cache_path=args.cache_path,
        weights_file=args.weights_file,
        fit_weights=args.fit_weights,
        fit_through_season=args.fit_through_season,
        future_weeks=args.future_weeks,
        write_wide=args.write_wide,
    )


if __name__ == "__main__":
    main()