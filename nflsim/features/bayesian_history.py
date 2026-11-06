"""Clean feature builder for recency-weighted Bayesian player and team stats.

This is the replacement path for the older notebook-style workflow in
Bayesian Player Stats.py. The goal is to keep the modeling idea intact while
making weekly refreshes much faster and the code much easier to maintain.

The main changes are:
- one-pass recency smoothing instead of repeated history rescans;
- reusable helpers for share, average, and team-state features;
- direct CSV export for the final player and team tables.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal

import nfl_data_py as nfl
import numpy as np
import pandas as pd

from nflsim.engine.defaults import DEFAULT_ARTIFACT_DIR, REPOSITORY_DIR


DEFAULT_YEARS = tuple(range(2016, 2027))
DEFAULT_OUTPUT_DIR = DEFAULT_ARTIFACT_DIR
DEFAULT_CACHE_PATH = REPOSITORY_DIR / "data" / "cache"
DEFAULT_DECAY_FILE = REPOSITORY_DIR / "nflsim" / "config" / "bayesian_decay_rates.json"
DECAY_FILE_NOTE = ("Research baselines only (rolling_comparison, historical_bakeoff, ridge.bakeoff). No production model input reads "
                   "this file, including metrics listed here. See input_sources.csv in a bundle for how each input is built.")
EPS = 1e-9

# Marks scheduled-but-unplayed rows so smoothing emits them without folding them into state.
FUTURE_SLOT_FLAG = "is_future_slot"
FUTURE_PLAYER_ID_COLUMNS = ("passer_player_id", "rusher_player_id", "receiver_player_id")
TEAM_DIRECT_LAG_COLUMNS = (
    "pass_oe_weekly_variation",
    "target_gini_roll4",
    "rb_carry_split_gini_roll4",
    "rb_top_carry_share_roll4",
)


@dataclass(frozen=True)
class SmoothingConfig:
    kind: Literal["binary", "normal"]
    decay: float = 0.8
    season_carryover: float = 0.95
    prior_strength: float = 1.0
    prior_games: float = 4.0


@dataclass(frozen=True)
class MetricSpec:
    source: str
    kind: Literal["binary", "normal", "direct"]
    frame: pd.DataFrame
    entity_col: str
    name_col: str | None
    count_col: str = "count"
    date_col: str = "game_date"
    scope: Literal["player", "team"] = "player"


def _as_datetime(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.to_datetime(result[column])
    return result


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _gini_from_counts(values: pd.Series | np.ndarray) -> float:
    counts = np.asarray(values, dtype=float)
    counts = counts[np.isfinite(counts)]
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0

    sorted_counts = np.sort(counts)
    n = sorted_counts.size
    total = sorted_counts.sum()
    weighted = np.sum(np.arange(1, n + 1) * sorted_counts)
    return float((2.0 * weighted) / (n * total) - (n + 1) / n)


def _season_week_from_game_id(game_id: object) -> tuple[int | None, int | None]:
    text = str(game_id)
    parts = text.split("_")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        season = int(parts[0])
        week = int(parts[1])
        if 1 <= week <= 25:
            return season, week
    return None, None


def _roof_to_is_dome(roof: object) -> float | None:
    if roof is None or (isinstance(roof, float) and np.isnan(roof)):
        return None
    roof_text = str(roof).strip().lower()
    if roof_text in {"dome", "closed", "indoors", "indoor"}:
        return 1.0
    if roof_text in {"outdoors", "outdoor", "open"}:
        return 0.0
    return None


def _game_gap_decay(
    decay: float,
    season_carryover: float,
    previous_season: int | None,
    previous_week: int | None,
    current_season: int | None,
    current_week: int | None,
) -> float:
    if previous_season is None or current_season is None:
        return decay

    if previous_week is None or current_week is None:
        game_gap = 1
    elif previous_season == current_season:
        game_gap = max(current_week - previous_week, 1)
    else:
        game_gap = max((18 - previous_week) + current_week, 1)

    season_gap = max(current_season - previous_season, 0)
    return (decay ** game_gap) * (season_carryover ** season_gap)


def load_decay_rates(decay_file: Path = DEFAULT_DECAY_FILE) -> dict[str, dict[str, float]]:
    if not decay_file.exists():
        return {}

    with decay_file.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    decay_rates = payload.get("decay_rates", payload)
    cleaned: dict[str, dict[str, float]] = {}
    for metric_name, value in decay_rates.items():
        if isinstance(value, dict):
            try:
                decay = float(value.get("decay", 0.8))
                season_carryover = float(value.get("season_carryover", 0.95))
            except (TypeError, ValueError):
                continue
        else:
            try:
                decay = float(value)
                season_carryover = 0.95
            except (TypeError, ValueError):
                continue

        cleaned[metric_name] = {
            "decay": float(np.clip(decay, 0.5, 0.99)),
            "season_carryover": float(np.clip(season_carryover, 0.5, 0.99)),
        }
    return cleaned


def save_decay_rates(decay_rates: dict[str, dict[str, float]], decay_file: Path = DEFAULT_DECAY_FILE) -> None:
    serializable = {
        metric: {
            "decay": round(float(params.get("decay", 0.8)), 6),
            "season_carryover": round(float(params.get("season_carryover", 0.95)), 6),
        }
        for metric, params in sorted(decay_rates.items())
    }
    payload = {
        "note": DECAY_FILE_NOTE,
        "decay_rates": serializable,
        "updated_from": str(Path(__file__).name),
    }
    with decay_file.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)


def _config_for_metric(
    kind: Literal["binary", "normal"],
    metric_name: str,
    decay_rates: dict[str, dict[str, float]] | None,
    default_decay: float = 0.8,
) -> SmoothingConfig:
    params = decay_rates.get(metric_name) if decay_rates is not None else None
    decay = float(params.get("decay", default_decay)) if params is not None else default_decay
    season_carryover = float(params.get("season_carryover", 0.95)) if params is not None else 0.95
    return SmoothingConfig(kind=kind, decay=float(decay), season_carryover=season_carryover)


@dataclass(frozen=True)
class FutureContext:
    """Scheduled games with no play-by-play yet, plus the entities that need a pregame row."""

    games: pd.DataFrame
    team_slots: pd.DataFrame
    player_slots: pd.DataFrame


def _load_schedule_frame(seasons: list[int]) -> pd.DataFrame:
    for candidate in (seasons, seasons[:1]):
        try:
            return nfl.import_schedules(candidate)
        except Exception as error:  # nfl_data_py raises bare Exception for unpublished seasons
            print(f"Schedule fetch failed for {candidate}: {error}")
    return pd.DataFrame()


def _future_player_slots(pbp: pd.DataFrame, team_slots: pd.DataFrame, latest_season: int) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["player_id", "player_name", "team", "game_id", "game_date"])
    recent = pbp[pbp["game_id"].astype(str).str.startswith(f"{latest_season}_")]
    if recent.empty:
        recent = pbp

    frames: list[pd.DataFrame] = []
    for id_col in FUTURE_PLAYER_ID_COLUMNS:
        name_col = id_col.replace("_id", "_name")
        if id_col not in recent.columns or name_col not in recent.columns:
            continue
        subset = recent.loc[recent[id_col].notna(), [id_col, name_col, "posteam", "game_date"]].copy()
        subset.columns = ["player_id", "player_name", "team", "game_date"]
        frames.append(subset)

    if not frames:
        return empty

    stacked = pd.concat(frames, ignore_index=True).dropna(subset=["player_id", "team"])
    if stacked.empty:
        return empty

    latest = (
        stacked.sort_values("game_date")
        .drop_duplicates(subset=["player_id"], keep="last")[["player_id", "player_name", "team"]]
    )
    latest["player_id"] = latest["player_id"].astype(str)
    merged = latest.merge(team_slots, on="team", how="inner")
    return merged[["player_id", "player_name", "team", "game_id", "game_date"]]


def build_future_context(pbp: pd.DataFrame, weeks_ahead: int = 1) -> FutureContext | None:
    """Collect the next `weeks_ahead` unplayed regular season weeks as feature slots."""
    if weeks_ahead <= 0 or pbp.empty:
        return None

    played = set(pbp["game_id"].astype(str))
    seasons = sorted({season for season, _ in map(_season_week_from_game_id, played) if season is not None})
    if not seasons:
        return None

    schedule = _load_schedule_frame([seasons[-1], seasons[-1] + 1])
    if schedule.empty:
        return None

    schedule = schedule[(schedule["game_type"] == "REG") & schedule["week"].between(1, 18)].copy()
    schedule["game_id"] = schedule["game_id"].astype(str)
    schedule["game_date"] = pd.to_datetime(schedule["gameday"], errors="coerce")
    schedule = schedule.dropna(subset=["game_date", "home_team", "away_team"])
    upcoming = schedule[~schedule["game_id"].isin(played)]
    if upcoming.empty:
        return None

    week_order = (
        upcoming.groupby(["season", "week"], dropna=False)["game_date"]
        .min()
        .reset_index()
        .sort_values("game_date")
        .head(weeks_ahead)
    )
    games = (
        upcoming.merge(week_order[["season", "week"]], on=["season", "week"], how="inner")[
            ["game_id", "game_date", "home_team", "away_team", "season", "week"]
        ]
        .sort_values(["game_date", "game_id"])
        .reset_index(drop=True)
    )
    if games.empty:
        return None

    team_slots = pd.concat(
        [
            games[["home_team", "game_id", "game_date"]].rename(columns={"home_team": "team"}),
            games[["away_team", "game_id", "game_date"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    ).dropna(subset=["team"])

    player_slots = _future_player_slots(pbp, team_slots, seasons[-1])
    labels = ", ".join(f"{int(s)} wk {int(w)}" for s, w in zip(week_order["season"], week_order["week"]))
    print(f"Future feature slots: {len(games)} games ({labels}), {len(player_slots)} player rows")
    return FutureContext(games=games, team_slots=team_slots, player_slots=player_slots)


def _cached_pbp_years_present(years: Iterable[int], cache_path: Path) -> list[int]:
    """Return the subset of years that already have a populated pbp cache folder."""
    missing: list[int] = []
    for year in years:
        season_dirs = list(cache_path.glob(f"season={year}*")) if cache_path.exists() else []
        if not any(any(d.glob("*.parquet")) for d in season_dirs):
            missing.append(year)
    return missing


def load_pbp_data(years: Iterable[int], cache_path: Path = DEFAULT_CACHE_PATH) -> pd.DataFrame:
    years = list(years)
    # The current season is still being played, so its cache file is incomplete/stale each run.
    current_season = datetime.now().year
    cached_years = [year for year in years if year != current_season]
    live_years = [year for year in years if year == current_season]

    missing_years = _cached_pbp_years_present(cached_years, cache_path)
    if missing_years:
        print(f"Populating pbp cache for missing season(s) {missing_years} in {cache_path}")
        cache_path.mkdir(parents=True, exist_ok=True)
        nfl.cache_pbp(missing_years, alt_path=str(cache_path))

    frames: list[pd.DataFrame] = []
    if cached_years:
        frames.append(nfl.import_pbp_data(cached_years, cache=True, alt_path=str(cache_path)))
    if live_years:
        print(f"Loading in-progress season(s) {live_years} fresh (bypassing cache)")
        frames.append(nfl.import_pbp_data(live_years, cache=False))
    pbp = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    try:
        ftn = nfl.import_ftn_data(years[-4:] if len(years) >= 4 else years)
        pbp = pbp.merge(
            ftn,
            left_on=["game_id", "play_id"],
            right_on=["nflverse_game_id", "nflverse_play_id"],
            how="left",
        )
    except Exception as error:
        print(f"Skipping FTN enrichment: {error}")

    pbp = pbp[pbp["season_type"] == "REG"].copy()
    pbp = _as_datetime(pbp, "game_date")
    pbp["game_id"] = pbp["game_id"].astype(str)

    if "is_interception_worthy" not in pbp.columns:
        pbp["is_interception_worthy"] = 0.0
    if "route" not in pbp.columns:
        pbp["route"] = pd.NA

    pbp["half"] = np.where(pbp["qtr"] < 3, 1, 2)
    pbp["drive_in_half"] = pbp.groupby(["game_id", "posteam", "half"])["fixed_drive"].rank(method="dense")
    pbp["yac_oe"] = pbp["yards_after_catch"] - pbp["xyac_mean_yardage"]
    pbp["depth_bucket"] = np.where(
        pbp["air_yards"] >= 25,
        "deep",
        np.where(pbp["air_yards"] >= 5, "medium", "short"),
    )
    pbp["target"] = ((pbp["pass_attempt"] == 1) & pbp["receiver_player_id"].notna()).astype(int)
    pbp["is_td"] = pbp["touchdown"].fillna(0).astype(int)
    pbp["is_rz"] = (pbp["yardline_100"] <= 20).astype(int)
    pbp["rush_yards"] = pbp["rushing_yards"].fillna(0)
    pbp["explosive"] = (pbp["rush_yards"] > 15).astype(int)
    pbp["close"] = (pbp["ydstogo"] <= 2).astype(int)
    pbp["is_middle"] = (pbp["pass_location"] == "middle").astype(int)
    pbp["went_for_it"] = pbp["play_type"].isin(["run", "pass"]).astype(int)
    pbp["game_state"] = np.select(
        [pbp["score_differential"] > 3, pbp["score_differential"] < -3],
        ["winning", "losing"],
        default="tied",
    )
    return pbp


def load_snap_counts(years: Iterable[int]) -> pd.DataFrame:
    years = list(years)
    snap_counts = nfl.import_snap_counts(years)
    snap_counts["total_snaps"] = (
        snap_counts["offense_snaps"] + snap_counts["defense_snaps"] + snap_counts["st_snaps"]
    )
    ids = nfl.import_ids()[["pfr_id", "gsis_id", "name"]]
    snap_counts = snap_counts.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    return snap_counts.drop(columns=["pfr_id"])


def _observed_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if FUTURE_SLOT_FLAG not in frame.columns:
        return frame
    observed = frame.loc[~frame[FUTURE_SLOT_FLAG].fillna(False).astype(bool)]
    return observed if not observed.empty else frame


def smooth_binary_history(
    frame: pd.DataFrame,
    entity_col: str,
    value_col: str,
    count_col: str,
    config: SmoothingConfig,
    date_col: str = "game_date",
    name_col: str | None = None,
) -> pd.DataFrame:
    sort_cols = [entity_col, date_col]
    if "game_id" in frame.columns:
        sort_cols.append("game_id")
    ordered = frame.sort_values(sort_cols).copy()
    if ordered.empty:
        return ordered

    records: list[dict[str, object]] = []
    observed = _observed_rows(ordered)
    global_mean = float(np.clip(observed[value_col].mean(), EPS, 1.0 - EPS))
    prior_count = max(float(observed[count_col].mean()) * config.prior_strength, config.prior_games)
    alpha_prior = global_mean * prior_count
    beta_prior = (1.0 - global_mean) * prior_count

    for _, group in ordered.groupby(entity_col, sort=False):
        group = group.sort_values([date_col, "game_id"] if "game_id" in group.columns else [date_col]).reset_index(drop=True)
        state_success = 0.0
        state_failure = 0.0
        last_season_week: tuple[int | None, int | None] | None = None

        for row in group.itertuples(index=False):
            current_date = getattr(row, date_col)
            current_game_id = getattr(row, "game_id")
            current_season, current_week = _season_week_from_game_id(current_game_id)
            if last_season_week is not None:
                decay_factor = _game_gap_decay(
                    config.decay,
                    config.season_carryover,
                    last_season_week[0],
                    last_season_week[1],
                    current_season,
                    current_week,
                )
                state_success *= decay_factor
                state_failure *= decay_factor

            alpha_post = alpha_prior + state_success
            beta_post = beta_prior + state_failure
            total = alpha_post + beta_post

            record = {
                entity_col: getattr(row, entity_col),
                "game_id": getattr(row, "game_id"),
                "game_date": current_date,
                "alpha": alpha_post,
                "beta": beta_post,
                "bayes_estimate_mean": alpha_post / total if total > 0 else np.nan,
                "bayes_estimate_std": np.sqrt((alpha_post * beta_post) / ((total**2) * (total + 1))) if total > 1 else np.nan,
            }
            if name_col is not None:
                record[name_col] = getattr(row, name_col)
            records.append(record)

            last_season_week = (current_season, current_week)
            if getattr(row, FUTURE_SLOT_FLAG, False):
                continue

            count = float(getattr(row, count_col))
            value = float(getattr(row, value_col))
            state_success += value * count
            state_failure += (1.0 - value) * count

    return pd.DataFrame.from_records(records)


def smooth_normal_history(
    frame: pd.DataFrame,
    entity_col: str,
    value_col: str,
    count_col: str,
    config: SmoothingConfig,
    date_col: str = "game_date",
    name_col: str | None = None,
) -> pd.DataFrame:
    sort_cols = [entity_col, date_col]
    if "game_id" in frame.columns:
        sort_cols.append("game_id")
    ordered = frame.sort_values(sort_cols).copy()
    if ordered.empty:
        return ordered

    records: list[dict[str, object]] = []
    observed = _observed_rows(ordered)
    global_mean = float(observed[value_col].mean())
    global_var = float(max(observed[value_col].var(ddof=0), EPS))
    prior_weight = max(float(observed[count_col].mean()) * config.prior_strength, config.prior_games)
    prior_precision = prior_weight / global_var

    for _, group in ordered.groupby(entity_col, sort=False):
        group = group.sort_values([date_col, "game_id"] if "game_id" in group.columns else [date_col]).reset_index(drop=True)
        state_weight = 0.0
        state_sum = 0.0
        state_sum_sq = 0.0
        last_season_week: tuple[int | None, int | None] | None = None

        for row in group.itertuples(index=False):
            current_date = getattr(row, date_col)
            current_game_id = getattr(row, "game_id")
            current_season, current_week = _season_week_from_game_id(current_game_id)
            if last_season_week is not None:
                decay_factor = _game_gap_decay(
                    config.decay,
                    config.season_carryover,
                    last_season_week[0],
                    last_season_week[1],
                    current_season,
                    current_week,
                )
                state_weight *= decay_factor
                state_sum *= decay_factor
                state_sum_sq *= decay_factor

            if state_weight <= 0:
                bayes_mean = global_mean
                bayes_std = np.sqrt(global_var)
            else:
                empirical_mean = state_sum / state_weight
                empirical_var = max(state_sum_sq / state_weight - empirical_mean**2, EPS)
                posterior_precision = prior_precision + state_weight / empirical_var
                bayes_mean = (
                    prior_precision * global_mean + (state_weight * empirical_mean) / empirical_var
                ) / posterior_precision
                bayes_std = np.sqrt(1.0 / posterior_precision)

            record = {
                entity_col: getattr(row, entity_col),
                "game_id": getattr(row, "game_id"),
                "game_date": current_date,
                "bayes_estimate_mean": bayes_mean,
                "bayes_estimate_std": bayes_std,
            }
            if name_col is not None:
                record[name_col] = getattr(row, name_col)
            records.append(record)

            last_season_week = (current_season, current_week)
            if getattr(row, FUTURE_SLOT_FLAG, False):
                continue

            count = float(getattr(row, count_col))
            value = float(getattr(row, value_col))
            state_weight += count
            state_sum += value * count
            state_sum_sq += (value**2) * count

    return pd.DataFrame.from_records(records)


def standardize_player_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    result = frame.copy()
    rename_map: dict[str, str] = {}
    for column in result.columns:
        if column.endswith("_player_id"):
            rename_map[column] = "player_id"
        elif column.endswith("_player_name"):
            rename_map[column] = "player_name"
    result = result.rename(columns=rename_map)
    if "player_id" not in result.columns:
        result["player_id"] = np.nan
    if "player_name" not in result.columns:
        result["player_name"] = np.nan
    if "average" not in result.columns:
        result["average"] = result.get("bayes_estimate_mean", np.nan)
    if "bayes_estimate_mean" not in result.columns:
        result["bayes_estimate_mean"] = result["average"]
    if "bayes_estimate_std" not in result.columns:
        result["bayes_estimate_std"] = np.nan
    result["source"] = source
    return result[["player_id", "player_name", "game_id", "game_date", "source", "average", "bayes_estimate_mean", "bayes_estimate_std"]]


def standardize_team_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    result = frame.copy()
    rename_map: dict[str, str] = {}
    for column in result.columns:
        if column in {"posteam", "defteam", "team"}:
            rename_map[column] = "team"
    result = result.rename(columns=rename_map)
    if "team" not in result.columns:
        result["team"] = np.nan
    if "average" not in result.columns:
        result["average"] = result.get("bayes_estimate_mean", np.nan)
    if "bayes_estimate_mean" not in result.columns:
        result["bayes_estimate_mean"] = result["average"]
    if "bayes_estimate_std" not in result.columns:
        result["bayes_estimate_std"] = np.nan
    result["source"] = source
    return result[["team", "game_id", "game_date", "source", "average", "bayes_estimate_mean", "bayes_estimate_std"]]


def _group_player_share(
    pbp: pd.DataFrame,
    player_col: str,
    player_name_col: str,
    team_col: str,
    event_mask: pd.Series,
    source_event_col: str,
    active_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    subset = pbp.loc[event_mask].copy()
    player_totals = (
        subset.groupby([player_col, team_col, "game_id", "game_date"], dropna=False)
        .agg(
            successes=(source_event_col, "sum"),
            event_player_name=(player_name_col, "last"),
        )
        .reset_index()
    )
    team_totals = (
        subset.groupby([team_col, "game_id"], dropna=False)
        .agg(team_total=(source_event_col, "sum"))
        .reset_index()
    )
    if active_frame is not None:
        active_columns = [player_col, player_name_col, team_col, "game_id", "game_date"]
        active_players = (
            active_frame[active_columns]
            .dropna(subset=[player_col])
            .sort_values([player_col, "game_date", "game_id"])
            .drop_duplicates([player_col, team_col, "game_id", "game_date"], keep="last")
        )
        merged = active_players.merge(
            player_totals,
            on=[player_col, team_col, "game_id", "game_date"],
            how="left",
        )
        merged[player_name_col] = merged["event_player_name"].combine_first(merged[player_name_col])
        merged = merged.drop(columns=["event_player_name"])
        merged["successes"] = merged["successes"].fillna(0.0)
    else:
        merged = player_totals.rename(columns={"event_player_name": player_name_col})
    merged = merged.merge(team_totals, on=[team_col, "game_id"], how="left")
    if active_frame is not None:
        merged["team_total"] = merged["team_total"].fillna(0.0)
    merged["rate"] = _safe_divide(merged["successes"], merged["team_total"])
    if active_frame is not None:
        merged["rate"] = merged["rate"].fillna(0.0)
    merged["count"] = merged["team_total"]
    return merged[[player_col, player_name_col, "game_id", "game_date", "rate", "count"]]


def _group_player_average(
    pbp: pd.DataFrame,
    player_col: str,
    player_name_col: str,
    event_mask: pd.Series,
    value_col: str,
    count_col: str,
) -> pd.DataFrame:
    subset = pbp.loc[event_mask].copy()
    grouped = (
        subset.groupby([player_col, player_name_col, "game_id", "game_date"], dropna=False)
        .agg(value_sum=(value_col, "sum"), count=(count_col, "sum"))
        .reset_index()
    )
    grouped["rate"] = _safe_divide(grouped["value_sum"], grouped["count"])
    return grouped[[player_col, player_name_col, "game_id", "game_date", "rate", "count"]]


def _group_team_average(
    pbp: pd.DataFrame,
    team_col: str,
    event_mask: pd.Series,
    value_col: str,
    count_col: str,
) -> pd.DataFrame:
    subset = pbp.loc[event_mask].copy()
    grouped = (
        subset.groupby([team_col, "game_id", "game_date"], dropna=False)
        .agg(value_sum=(value_col, "sum"), count=(count_col, "sum"))
        .reset_index()
    )
    grouped["rate"] = _safe_divide(grouped["value_sum"], grouped["count"])
    return grouped[[team_col, "game_id", "game_date", "rate", "count"]]


def _opponent_context_frame(pbp: pd.DataFrame, opponent_metric: pd.DataFrame) -> pd.DataFrame:
    if opponent_metric.empty:
        return pd.DataFrame(columns=["team", "game_id", "game_date", "rate", "count"])

    required = {"team", "game_id", "game_date", "rate", "count"}
    if not required.issubset(opponent_metric.columns):
        return pd.DataFrame(columns=["team", "game_id", "game_date", "rate", "count"])

    game_sides = (
        pd.concat(
            [
                pbp[["game_id", "game_date", "posteam", "defteam"]].rename(
                    columns={"posteam": "team", "defteam": "opponent"}
                ),
                pbp[["game_id", "game_date", "defteam", "posteam"]].rename(
                    columns={"defteam": "team", "posteam": "opponent"}
                ),
            ],
            ignore_index=True,
        )
        .dropna(subset=["team", "opponent", "game_id", "game_date"])
        .drop_duplicates(subset=["team", "opponent", "game_id"])
    )
    if game_sides.empty:
        return pd.DataFrame(columns=["team", "game_id", "game_date", "rate", "count"])

    opponent_values = opponent_metric[["team", "game_id", "rate", "count"]].rename(
        columns={"team": "opponent", "rate": "opponent_rate", "count": "opponent_count"}
    )
    merged = game_sides.merge(opponent_values, on=["opponent", "game_id"], how="left")
    merged = merged.dropna(subset=["opponent_rate"])
    if merged.empty:
        return pd.DataFrame(columns=["team", "game_id", "game_date", "rate", "count"])

    merged["rate"] = pd.to_numeric(merged["opponent_rate"], errors="coerce")
    merged["count"] = pd.to_numeric(merged["opponent_count"], errors="coerce").fillna(1.0).clip(lower=1.0)
    return merged[["team", "game_id", "game_date", "rate", "count"]]


def _rolling_team_massey_rating(
    pbp: pd.DataFrame,
    team_col: str,
    event_mask: pd.Series,
    value_col: str,
    count_col: str,
    source: str,
    future: FutureContext | None = None,
) -> pd.DataFrame:
    subset = pbp.loc[event_mask].copy()
    if subset.empty:
        return pd.DataFrame(columns=["team", "game_id", "game_date", "rate", "count"])

    required = ["game_id", "game_date", "home_team", "away_team", team_col, value_col, count_col]
    missing = [column for column in required if column not in subset.columns]
    if missing:
        return pd.DataFrame(columns=["team", "game_id", "game_date", "rate", "count"])

    team_prefix = subset[["game_id", "game_date", "home_team", "away_team", team_col, value_col, count_col]].copy()
    team_prefix[count_col] = pd.to_numeric(team_prefix[count_col], errors="coerce").fillna(0.0)
    team_prefix[value_col] = pd.to_numeric(team_prefix[value_col], errors="coerce").fillna(0.0)
    team_game = (
        team_prefix.groupby([team_col, "game_id", "game_date"], dropna=False)
        .agg(team_value=(value_col, "sum"), plays=(count_col, "sum"))
        .reset_index()
        .rename(columns={team_col: "team"})
    )
    team_game["team_eppa"] = _safe_divide(team_game["team_value"], team_game["plays"])

    game_rows = (
        team_prefix[["game_id", "game_date", "home_team", "away_team"]]
        .drop_duplicates()
        .sort_values(["game_date", "game_id"])
        .reset_index(drop=True)
    )
    game_rows["game_date"] = pd.to_datetime(game_rows["game_date"], errors="coerce")
    game_rows = game_rows.dropna(subset=["game_date", "home_team", "away_team"])
    if game_rows.empty:
        return pd.DataFrame(columns=["team", "game_id", "game_date", "rate", "count"])

    if future is not None and not future.games.empty:
        extra = future.games[["game_id", "game_date", "home_team", "away_team"]].copy()
        extra["game_id"] = extra["game_id"].astype(str)
        extra = extra[~extra["game_id"].isin(game_rows["game_id"].astype(str))]
        if not extra.empty:
            game_rows = (
                pd.concat([game_rows, extra], ignore_index=True)
                .sort_values(["game_date", "game_id"])
                .reset_index(drop=True)
            )

    parsed = [_season_week_from_game_id(game_id) for game_id in game_rows["game_id"]]
    game_rows["season"] = [value[0] for value in parsed]
    game_rows["week"] = [value[1] for value in parsed]
    fallback = game_rows["season"].isna() | game_rows["week"].isna()
    if fallback.any():
        iso = game_rows.loc[fallback, "game_date"].dt.isocalendar()
        game_rows.loc[fallback, "season"] = iso["year"].to_numpy()
        game_rows.loc[fallback, "week"] = iso["week"].to_numpy()

    # One Massey solve per league week, not per game.
    week_keys = (
        game_rows.groupby(["season", "week"], dropna=False)["game_date"]
        .min()
        .reset_index()
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    week_keys["week_ord"] = np.arange(len(week_keys), dtype=int)
    game_rows = game_rows.merge(week_keys[["season", "week", "week_ord"]], on=["season", "week"], how="left")

    all_teams = sorted(
        {
            team
            for team in pd.concat([game_rows["home_team"], game_rows["away_team"]], ignore_index=True)
            .dropna()
            .unique()
            if str(team) != "nan"
        }
    )
    team_to_index = {team: idx for idx, team in enumerate(all_teams)}
    team_count = len(all_teams)

    eppa_lookup = {
        (str(game_id), team): float(value)
        for game_id, team, value in zip(team_game["game_id"], team_game["team"], team_game["team_eppa"])
    }

    records: list[dict[str, object]] = []
    lookback_weeks = 17
    week_decay = 0.95
    season_carryover = 0.7

    for current_ord, current_season in zip(week_keys["week_ord"], week_keys["season"]):
        current_ord = int(current_ord)
        window = game_rows[
            (game_rows["week_ord"] < current_ord) & (game_rows["week_ord"] >= current_ord - lookback_weeks)
        ]
        if window.empty:
            continue

        rows = []
        rhs = []
        weights = []
        for prior_id, prior_home, prior_away, prior_ord, prior_season in zip(
            window["game_id"], window["home_team"], window["away_team"], window["week_ord"], window["season"]
        ):
            home_rate = eppa_lookup.get((str(prior_id), prior_home))
            away_rate = eppa_lookup.get((str(prior_id), prior_away))
            if home_rate is None or away_rate is None or np.isnan(home_rate) or np.isnan(away_rate):
                continue

            season_gap = max(int(current_season) - int(prior_season), 0)
            weight = (week_decay ** (current_ord - 1 - int(prior_ord))) * (season_carryover ** season_gap)

            row_home = np.zeros(2 * team_count, dtype=float)
            row_home[team_to_index[prior_home]] = 1.0
            row_home[team_count + team_to_index[prior_away]] = -1.0
            rows.append(row_home)
            rhs.append(home_rate)
            weights.append(weight)

            row_away = np.zeros(2 * team_count, dtype=float)
            row_away[team_to_index[prior_away]] = 1.0
            row_away[team_count + team_to_index[prior_home]] = -1.0
            rows.append(row_away)
            rhs.append(away_rate)
            weights.append(weight)

        if not rows:
            continue

        design = np.vstack(rows)
        observed = np.asarray(rhs, dtype=float)
        game_weights = np.asarray(weights, dtype=float)
        sqrt_w = np.sqrt(np.clip(game_weights, 1e-12, None))
        weighted_design = design * sqrt_w[:, None]
        weighted_rhs = observed * sqrt_w
        solution = np.linalg.pinv(weighted_design) @ weighted_rhs

        offense = solution[:team_count]
        defense = solution[team_count:]

        current_games = game_rows[game_rows["week_ord"] == current_ord]
        for current_game_id, current_date, home_team, away_team in zip(
            current_games["game_id"],
            current_games["game_date"],
            current_games["home_team"],
            current_games["away_team"],
        ):
            for team in (home_team, away_team):
                if team not in team_to_index:
                    continue
                team_idx = team_to_index[team]
                records.append({
                    "team": team,
                    "game_id": str(current_game_id),
                    "game_date": current_date,
                    "offense_rating": float(offense[team_idx]),
                    "defense_rating": float(defense[team_idx]),
                    "rate": float(offense[team_idx] + defense[team_idx]),
                })

    if not records:
        return pd.DataFrame(columns=["team", "game_id", "game_date", "offense_rating", "defense_rating", "rate", "count"])

    result = pd.DataFrame.from_records(records)
    result["count"] = 1.0
    return result[["team", "game_id", "game_date", "offense_rating", "defense_rating", "rate", "count"]]


def collect_player_metric_specs(pbp: pd.DataFrame) -> list[MetricSpec]:
    specs: list[MetricSpec] = []
    targeted_receiver_games = pbp.loc[
        (pbp["pass_attempt"] == 1) & pbp["receiver_player_id"].notna(),
        ["receiver_player_id", "receiver_player_name", "posteam", "game_id", "game_date"],
    ].drop_duplicates()
    seasons = sorted(
        {
            int(str(game_id).split("_")[0])
            for game_id in pbp["game_id"].dropna().unique()
            if str(game_id).split("_")[0].isdigit()
        }
    )
    snap_active_receiver_games = pd.DataFrame(
        columns=["receiver_player_id", "receiver_player_name", "posteam", "game_id", "game_date"]
    )
    if seasons:
        snap_counts = load_snap_counts(seasons)
        snap_counts = snap_counts[
            (snap_counts["game_type"] == "REG")
            & (snap_counts["offense_snaps"].fillna(0) > 0)
            & snap_counts["position"].isin({"WR", "TE", "RB", "FB"})
            & snap_counts["gsis_id"].notna()
        ].copy()
        game_dates = pbp[["game_id", "game_date"]].drop_duplicates()
        snap_active_receiver_games = (
            snap_counts.rename(
                columns={
                    "gsis_id": "receiver_player_id",
                    "name": "receiver_player_name",
                    "team": "posteam",
                }
            )[["receiver_player_id", "receiver_player_name", "posteam", "game_id"]]
            .merge(game_dates, on="game_id", how="inner")
            .drop_duplicates()
        )
    active_receiver_games = pd.concat(
        [targeted_receiver_games, snap_active_receiver_games],
        ignore_index=True,
    ).drop_duplicates()

    for source, bucket in [("deep", "deep"), ("medium", "medium"), ("short", "short")]:
        base = _group_player_share(
            pbp,
            player_col="receiver_player_id",
            player_name_col="receiver_player_name",
            team_col="posteam",
            event_mask=(pbp["pass_attempt"] == 1) & (pbp["depth_bucket"] == bucket),
            source_event_col="target",
            active_frame=active_receiver_games,
        )
        specs.append(
            MetricSpec(
                source=source,
                kind="binary",
                frame=base,
                entity_col="receiver_player_id",
                name_col="receiver_player_name",
            )
        )

    fd = _group_player_average(
        pbp,
        player_col="receiver_player_id",
        player_name_col="receiver_player_name",
        event_mask=pbp["complete_pass"] == 1,
        value_col="first_down",
        count_col="complete_pass",
    )
    specs.append(MetricSpec("fd_bayes", "binary", fd, "receiver_player_id", "receiver_player_name"))

    rz = (
        pbp.loc[pbp["target"] == 1]
        .groupby(["receiver_player_id", "receiver_player_name", "posteam", "game_id", "game_date"], dropna=False)
        .agg(rz_targets=("is_rz", "sum"), targets=("target", "sum"))
        .reset_index()
    )
    rz["rate"] = _safe_divide(rz["rz_targets"], rz["targets"])
    specs.append(
        MetricSpec(
            source="rz_bayes",
            kind="binary",
            frame=rz[["receiver_player_id", "receiver_player_name", "game_id", "game_date", "rate", "targets"]].rename(columns={"targets": "count"}),
            entity_col="receiver_player_id",
            name_col="receiver_player_name",
        )
    )

    yac = _group_player_average(
        pbp,
        player_col="receiver_player_id",
        player_name_col="receiver_player_name",
        event_mask=pbp["complete_pass"] == 1,
        value_col="yac_oe",
        count_col="complete_pass",
    )
    specs.append(MetricSpec("yac_bayes", "normal", yac, "receiver_player_id", "receiver_player_name"))

    # Receiver efficiency by target depth.
    air = _group_player_average(
        pbp,
        player_col="receiver_player_id",
        player_name_col="receiver_player_name",
        event_mask=(pbp["target"] == 1) & pbp["air_yards"].notna(),
        value_col="air_yards",
        count_col="target",
    )
    specs.append(MetricSpec("adot", "normal", air, "receiver_player_id", "receiver_player_name"))

    ypt = _group_player_average(
        pbp,
        player_col="receiver_player_id",
        player_name_col="receiver_player_name",
        event_mask=pbp["target"] == 1,
        value_col="yards_gained",
        count_col="target",
    )
    specs.append(MetricSpec("ypt", "normal", ypt, "receiver_player_id", "receiver_player_name"))

    rush = _group_player_average(
        pbp,
        player_col="rusher_player_id",
        player_name_col="rusher_player_name",
        event_mask=pbp["rush_attempt"] == 1,
        value_col="yards_gained",
        count_col="rush_attempt",
    )
    specs.append(MetricSpec("ypc_bayes", "normal", rush, "rusher_player_id", "rusher_player_name"))

    # Rusher context and style rates used by downstream rushing models.
    rush_attempts = pbp.loc[pbp["rush_attempt"] == 1].copy()

    close = _group_player_average(
        rush_attempts,
        player_col="rusher_player_id",
        player_name_col="rusher_player_name",
        event_mask=rush_attempts["rush_attempt"] == 1,
        value_col="close",
        count_col="rush_attempt",
    )
    specs.append(MetricSpec("close_bayes", "binary", close, "rusher_player_id", "rusher_player_name"))

    explosive = _group_player_average(
        rush_attempts,
        player_col="rusher_player_id",
        player_name_col="rusher_player_name",
        event_mask=rush_attempts["rush_attempt"] == 1,
        value_col="explosive",
        count_col="rush_attempt",
    )
    specs.append(MetricSpec("explosive_bayes", "binary", explosive, "rusher_player_id", "rusher_player_name"))

    rush_share = _group_player_share(
        rush_attempts,
        player_col="rusher_player_id",
        player_name_col="rusher_player_name",
        team_col="posteam",
        event_mask=rush_attempts["rush_attempt"] == 1,
        source_event_col="rush_attempt",
    )
    specs.append(MetricSpec("rush_rate", "binary", rush_share, "rusher_player_id", "rusher_player_name"))

    scramble = _group_player_average(
        rush_attempts.assign(qb_scramble=rush_attempts["qb_scramble"].fillna(0)),
        player_col="rusher_player_id",
        player_name_col="rusher_player_name",
        event_mask=rush_attempts["rush_attempt"] == 1,
        value_col="qb_scramble",
        count_col="rush_attempt",
    )
    specs.append(MetricSpec("scramble_rate", "binary", scramble, "rusher_player_id", "rusher_player_name"))

    qb_scramble_ypc = _group_player_average(
        rush_attempts,
        player_col="rusher_player_id",
        player_name_col="rusher_player_name",
        event_mask=rush_attempts["qb_scramble"].fillna(0) == 1,
        value_col="yards_gained",
        count_col="rush_attempt",
    )
    specs.append(
        MetricSpec(
            "qb_scramble_ypc_bayes",
            "normal",
            qb_scramble_ypc,
            "rusher_player_id",
            "rusher_player_name",
        )
    )

    pass_df = pbp.loc[pbp["pass_attempt"] == 1].copy()
    pass_metrics = {
        "cpoe_bayes": ("cpoe", "normal"),
        "turnover_bayes": ("is_interception_worthy", "binary"),
        "sack_bayes": ("sack", "binary"),
    }
    for source, (value_col, kind) in pass_metrics.items():
        metric_df = (
            pass_df.groupby(["passer_player_id", "passer_player_name", "game_id", "game_date"], dropna=False)
            .agg(value_sum=(value_col, "sum"), count=("pass_attempt", "sum"))
            .reset_index()
        )
        metric_df["rate"] = _safe_divide(metric_df["value_sum"], metric_df["count"])
        specs.append(
            MetricSpec(
                source=source,
                kind=kind,
                frame=metric_df[["passer_player_id", "passer_player_name", "game_id", "game_date", "rate", "count"]],
                entity_col="passer_player_id",
                name_col="passer_player_name",
            )
        )

    # Depth-specific QB CPOE.
    for source, bucket in [("cpoe_short", "short"), ("cpoe_medium", "medium"), ("cpoe_deep", "deep")]:
        metric_df = (
            pass_df[pass_df["depth_bucket"] == bucket]
            .groupby(["passer_player_id", "passer_player_name", "game_id", "game_date"], dropna=False)
            .agg(value_sum=("cpoe", "sum"), count=("pass_attempt", "sum"))
            .reset_index()
        )
        metric_df["rate"] = _safe_divide(metric_df["value_sum"], metric_df["count"])
        specs.append(
            MetricSpec(
                source=source,
                kind="normal",
                frame=metric_df[["passer_player_id", "passer_player_name", "game_id", "game_date", "rate", "count"]],
                entity_col="passer_player_id",
                name_col="passer_player_name",
            )
        )

    for source, bucket in [("deep_pass_rate", "deep"), ("medium_pass_rate", "medium"), ("short_pass_rate", "short")]:
        qb_games = (
            pass_df.groupby(
                ["passer_player_id", "passer_player_name", "game_id", "game_date"],
                dropna=False,
            )
            .agg(total_attempts=("pass_attempt", "sum"))
            .reset_index()
        )
        bucket_attempts = (
            pass_df[pass_df["depth_bucket"] == bucket]
            .groupby(
                ["passer_player_id", "passer_player_name", "game_id", "game_date"],
                dropna=False,
            )
            .agg(bucket_attempts=("pass_attempt", "sum"))
            .reset_index()
        )
        base = qb_games.merge(
            bucket_attempts,
            on=["passer_player_id", "passer_player_name", "game_id", "game_date"],
            how="left",
        )
        base["bucket_attempts"] = base["bucket_attempts"].fillna(0.0)
        base["rate"] = _safe_divide(base["bucket_attempts"], base["total_attempts"])
        base["count"] = base["total_attempts"]
        base = base[["passer_player_id", "passer_player_name", "game_id", "game_date", "rate", "count"]]
        specs.append(MetricSpec(source, "binary", base, "passer_player_id", "passer_player_name"))

    routes = {
        "FLAG": "CORNER",
        "DEEP OUT": "OUT",
        "HITCH/CURL": "CURL",
        "IN": "DIG",
        "IN/DIG": "DIG",
    }
    final_routes = ["FLAT", "QUICK OUT", "HITCH", "SLANT", "CURL", "DIG", "POST", "GO", "OUT", "CROSS", "WHEEL", "SWING", "SCREEN", "CORNER"]
    route_pbp = pbp.copy()
    if "route" in route_pbp.columns:
        route_pbp["route"] = route_pbp["route"].replace(routes)
        route_pbp = route_pbp[route_pbp["route"].isin(final_routes)]
        for route in final_routes:
            route_df = _group_player_average(
                route_pbp.assign(is_route=(route_pbp["route"] == route).astype(int)),
                player_col="receiver_player_id",
                player_name_col="receiver_player_name",
                event_mask=route_pbp["route"].notna(),
                value_col="is_route",
                count_col="pass",
            )
            if not route_df.empty:
                specs.append(MetricSpec(f"{route}_bayes", "binary", route_df, "receiver_player_id", "receiver_player_name"))

    # Team-QB continuity: first game for a QB-team combo is 0, then increments.
    qb_games = (
        pbp.loc[pbp["pass_attempt"] == 1]
        .groupby(["posteam", "game_id", "game_date", "passer_player_id", "passer_player_name"], dropna=False)
        .agg(attempts=("pass_attempt", "sum"))
        .reset_index()
    )
    qb_games = qb_games.sort_values(["posteam", "game_date", "attempts"], ascending=[True, True, False])
    qb_games = qb_games.groupby(["posteam", "game_id", "game_date"], as_index=False).head(1)
    qb_games = qb_games[qb_games["passer_player_id"].notna()].copy()
    qb_games = qb_games.sort_values(["posteam", "game_date", "game_id"]).reset_index(drop=True)
    qb_games["rate"] = qb_games.groupby(["posteam", "passer_player_id"]).cumcount().astype(float)
    specs.append(
        MetricSpec(
            source="qb_team_games_played",
            kind="direct",
            frame=qb_games[["passer_player_id", "passer_player_name", "game_id", "game_date", "rate"]],
            entity_col="passer_player_id",
            name_col="passer_player_name",
        )
    )

    # QB dome experience: Bayesian rate of recent games played in a dome.
    # Captures how accustomed the QB is to indoor conditions.
    qb_games = (
        pbp.loc[pbp["pass_attempt"] == 1]
        .groupby(["passer_player_id", "passer_player_name", "game_id", "game_date"], dropna=False)
        .agg(attempts=("pass_attempt", "sum"))
        .reset_index()
    )
    game_roof = (
        pbp.groupby(["game_id", "game_date"], dropna=False)["roof"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode(dropna=True).empty else np.nan)
        .reset_index(name="roof")
    )
    game_roof["is_dome"] = game_roof["roof"].map(_roof_to_is_dome)
    dome_games = qb_games.merge(game_roof[["game_id", "game_date", "is_dome"]], on=["game_id", "game_date"], how="left")
    dome_games = dome_games[dome_games["passer_player_id"].notna()].copy()
    dome_games["rate"] = dome_games["is_dome"].fillna(0.0)
    dome_games["count"] = np.where(dome_games["is_dome"].notna(), 1.0, 0.0)
    specs.append(
        MetricSpec(
            source="qb_dome_rate_bayes",
            kind="binary",
            frame=dome_games[["passer_player_id", "passer_player_name", "game_id", "game_date", "rate", "count"]],
            entity_col="passer_player_id",
            name_col="passer_player_name",
        )
    )

    return specs


def collect_team_metric_specs(pbp: pd.DataFrame, future: FutureContext | None = None) -> list[MetricSpec]:
    specs: list[MetricSpec] = []

    fourth_downs = pbp[
        (pbp["down"] == 4)
        & (pbp["ydstogo"] <= 4)
        & (pbp["score_differential"].abs() <= 10)
        & (pbp["yardline_100"].between(25, 75))
    ].copy()
    team_go_rate = (
        fourth_downs.groupby(["posteam", "game_id", "game_date"], dropna=False)
        .agg(go_for_it=("went_for_it", "sum"), total=("went_for_it", "count"))
        .reset_index()
    )
    team_go_rate["rate"] = _safe_divide(team_go_rate["go_for_it"], team_go_rate["total"])
    specs.append(
        MetricSpec(
            source="team_4th_rate",
            kind="binary",
            frame=team_go_rate.rename(columns={"posteam": "team", "total": "count"})[["team", "game_id", "game_date", "rate", "count"]],
            entity_col="team",
            name_col=None,
            scope="team",
        )
    )

    game_state_rate = (
        pbp[pbp["play_type"].isin(["run", "pass"]) & pbp["down"].isin([1, 2, 3])]
        .groupby(["posteam", "game_state", "game_id", "game_date"], dropna=False)
        .agg(pass_oe=("pass_oe", "mean"), total=("pass", "count"))
        .reset_index()
    )
    for state in ["winning", "losing", "tied"]:
        state_df = game_state_rate[game_state_rate["game_state"] == state].copy()
        if not state_df.empty:
            specs.append(
                MetricSpec(
                    source=f"{state}_rate",
                    kind="normal",
                    frame=state_df.rename(columns={"posteam": "team", "pass_oe": "rate", "total": "count"})[["team", "game_id", "game_date", "rate", "count"]],
                    entity_col="team",
                    name_col=None,
                    scope="team",
                )
            )

    # Week-level team pass rate over expected.
    pass_oe_weekly = (
        pbp[pbp["play_type"].isin(["run", "pass"])]
        .groupby(["posteam", "game_id", "game_date"], dropna=False)
        .agg(pass_oe=("pass_oe", "mean"), count=("pass", "count"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    pass_oe_weekly = pass_oe_weekly.sort_values(["team", "game_date", "game_id"])
    pass_oe_weekly["rate"] = (
        pass_oe_weekly.groupby("team")["pass_oe"]
        .transform(lambda s: s.rolling(window=4, min_periods=2).std())
        .fillna(0.0)
    )
    specs.append(
        MetricSpec(
            source="pass_oe_weekly_variation",
            kind="direct",
            frame=pass_oe_weekly[["team", "game_id", "game_date", "rate"]],
            entity_col="team",
            name_col=None,
            scope="team",
        )
    )
    specs.append(
        MetricSpec(
            source="pass_oe",
            kind="normal",
            frame=pass_oe_weekly[["team", "game_id", "game_date", "pass_oe", "count"]].rename(columns={"pass_oe": "rate"}),
            entity_col="team",
            name_col=None,
            scope="team",
        )
    )

    team_games = (
        pbp.groupby(["posteam", "game_id", "game_date"], dropna=False)
        .size()
        .reset_index(name="n")
        .rename(columns={"posteam": "team"})
    )

    # Team target concentration by week (rolling Gini).
    receiver_targets = (
        pbp.loc[(pbp["target"] == 1) & pbp["receiver_player_id"].notna()]
        .groupby(["posteam", "game_id", "game_date", "receiver_player_id"], dropna=False)
        .agg(targets=("target", "sum"))
        .reset_index()
    )
    target_gini = (
        receiver_targets.groupby(["posteam", "game_id", "game_date"], dropna=False)["targets"]
        .apply(_gini_from_counts)
        .reset_index(name="target_gini")
        .rename(columns={"posteam": "team"})
    )
    target_gini = team_games[["team", "game_id", "game_date"]].merge(
        target_gini,
        on=["team", "game_id", "game_date"],
        how="left",
    )
    target_gini["target_gini"] = target_gini["target_gini"].fillna(0.0)
    target_gini = target_gini.sort_values(["team", "game_date", "game_id"])
    target_gini["rate"] = (
        target_gini.groupby("team")["target_gini"]
        .transform(lambda s: s.rolling(window=4, min_periods=1).mean())
        .fillna(0.0)
    )
    specs.append(
        MetricSpec(
            source="target_gini_roll4",
            kind="direct",
            frame=target_gini[["team", "game_id", "game_date", "rate"]],
            entity_col="team",
            name_col=None,
            scope="team",
        )
    )

    # Team RB carry split by week (rolling Gini and top-back share).
    rb_carries = (
        pbp.loc[(pbp["rush_attempt"] == 1) & (pbp["qb_scramble"].fillna(0) == 0) & pbp["rusher_player_id"].notna()]
        .groupby(["posteam", "game_id", "game_date", "rusher_player_id"], dropna=False)
        .agg(carries=("rush_attempt", "sum"))
        .reset_index()
    )
    rb_game = rb_carries.groupby(["posteam", "game_id", "game_date"], dropna=False).agg(
        rb_carry_split_gini=("carries", _gini_from_counts),
        total_carries=("carries", "sum"),
        top_rb_carries=("carries", "max"),
    ).reset_index()
    rb_game["rb_top_carry_share"] = _safe_divide(rb_game["top_rb_carries"], rb_game["total_carries"])
    rb_game = rb_game.rename(columns={"posteam": "team"})
    rb_game = team_games[["team", "game_id", "game_date"]].merge(
        rb_game,
        on=["team", "game_id", "game_date"],
        how="left",
    )
    rb_game[["rb_carry_split_gini", "rb_top_carry_share"]] = rb_game[["rb_carry_split_gini", "rb_top_carry_share"]].fillna(0.0)
    rb_game = rb_game.sort_values(["team", "game_date", "game_id"])
    rb_game["rb_carry_split_gini_roll4"] = (
        rb_game.groupby("team")["rb_carry_split_gini"]
        .transform(lambda s: s.rolling(window=4, min_periods=1).mean())
        .fillna(0.0)
    )
    rb_game["rb_top_carry_share_roll4"] = (
        rb_game.groupby("team")["rb_top_carry_share"]
        .transform(lambda s: s.rolling(window=4, min_periods=1).mean())
        .fillna(0.0)
    )
    specs.append(
        MetricSpec(
            source="rb_carry_split_gini_roll4",
            kind="direct",
            frame=rb_game[["team", "game_id", "game_date", "rb_carry_split_gini_roll4"]].rename(columns={"rb_carry_split_gini_roll4": "rate"}),
            entity_col="team",
            name_col=None,
            scope="team",
        )
    )
    specs.append(
        MetricSpec(
            source="rb_top_carry_share_roll4",
            kind="direct",
            frame=rb_game[["team", "game_id", "game_date", "rb_top_carry_share_roll4"]].rename(columns={"rb_top_carry_share_roll4": "rate"}),
            entity_col="team",
            name_col=None,
            scope="team",
        )
    )

    for source, filter_mask, team_col, value_col, count_col in [
        ("passing_massey_rating", pbp["pass"] == 1, "posteam", "epa", "pass"),
        ("rushing_massey_rating", pbp["rush"] == 1, "posteam", "epa", "rush"),
    ]:
        metric_df = _rolling_team_massey_rating(pbp, team_col, filter_mask, value_col, count_col, source, future=future)
        specs.append(MetricSpec(source, "direct", metric_df, "team", None, scope="team"))

    for base_source, base_metric in [
        ("passing_massey_rating", "passing_massey"),
        ("rushing_massey_rating", "rushing_massey"),
    ]:
        base_frame = next(
            (spec.frame.copy() for spec in reversed(specs) if spec.source == base_source),
            pd.DataFrame(columns=["team", "game_id", "game_date", "offense_rating", "defense_rating", "rate", "count"]),
        )
        if base_frame.empty:
            continue
        offense_frame = base_frame[["team", "game_id", "game_date", "offense_rating", "count"]].rename(
            columns={"offense_rating": "rate"}
        )
        defense_frame = base_frame[["team", "game_id", "game_date", "defense_rating", "count"]].rename(
            columns={"defense_rating": "rate"}
        )
        specs.append(MetricSpec(f"{base_metric}_offense", "direct", offense_frame, "team", None, scope="team"))
        specs.append(MetricSpec(f"{base_metric}_defense", "direct", defense_frame, "team", None, scope="team"))

    for source, filter_mask, team_col, value_col, count_col in [
        ("epa_rate", pbp["pass"] == 1, "posteam", "epa", "pass"),
        ("rushing_epa_rate", pbp["rush"] == 1, "posteam", "epa", "rush"),
        ("def_pass_epa_rate", pbp["pass"] == 1, "defteam", "epa", "pass"),
        ("def_rush_epa_rate", pbp["rush"] == 1, "defteam", "epa", "rush"),
    ]:
        metric_df = _group_team_average(pbp, team_col, filter_mask, value_col, count_col).rename(columns={team_col: "team"})
        specs.append(MetricSpec(source, "normal", metric_df, "team", None, scope="team"))

    def_game = pbp.groupby(["defteam", "game_id", "game_date"], dropna=False).agg(
        short_cpoe_allowed=("cpoe", lambda s: s[pbp.loc[s.index, "depth_bucket"] == "short"].mean()),
        medium_cpoe_allowed=("cpoe", lambda s: s[pbp.loc[s.index, "depth_bucket"] == "medium"].mean()),
        deep_cpoe_allowed=("cpoe", lambda s: s[pbp.loc[s.index, "depth_bucket"] == "deep"].mean()),
        rush_yards_per_carry=("yards_gained", lambda s: s[pbp.loc[s.index, "play_type"] == "run"].mean()),
        yac_oe_allowed=("yac_oe", lambda s: s[pbp.loc[s.index, "play_type"] == "pass"].mean()),
        sacks=("sack", "sum"),
        interceptions=("interception", "sum"),
        dropbacks=("pass", "sum"),
    ).reset_index()
    def_game["sack_rate"] = _safe_divide(def_game["sacks"], def_game["dropbacks"])
    def_game["int_rate"] = _safe_divide(def_game["interceptions"], def_game["dropbacks"])
    for source, value_col, kind in [
        ("deep_cpoe_allowed", "deep_cpoe_allowed", "normal"),
        ("medium_cpoe_allowed", "medium_cpoe_allowed", "normal"),
        ("short_cpoe_allowed", "short_cpoe_allowed", "normal"),
        ("rush_yards_per_carry", "rush_yards_per_carry", "normal"),
        ("yac_oe_allowed", "yac_oe_allowed", "normal"),
        ("sack_rate", "sack_rate", "binary"),
        ("int_rate", "int_rate", "binary"),
    ]:
        temp = def_game[["defteam", "game_id", "game_date", value_col, "dropbacks"]].rename(columns={"defteam": "team", value_col: "rate", "dropbacks": "count"})
        specs.append(MetricSpec(source, kind, temp, "team", None, scope="team"))

    # Defensive game-state aliases expected by downstream models.
    def_state_rate = (
        pbp[pbp["play_type"].isin(["run", "pass"]) & pbp["down"].isin([1, 2, 3])]
        .groupby(["defteam", "game_state", "game_id", "game_date"], dropna=False)
        .agg(pass_oe=("pass_oe", "mean"), total=("pass", "count"))
        .reset_index()
    )
    for state in ["winning", "losing", "tied"]:
        state_df = def_state_rate[def_state_rate["game_state"] == state].copy()
        if not state_df.empty:
            specs.append(
                MetricSpec(
                    source=f"def_{state}_rate",
                    kind="normal",
                    frame=state_df.rename(columns={"defteam": "team", "pass_oe": "rate", "total": "count"})[["team", "game_id", "game_date", "rate", "count"]],
                    entity_col="team",
                    name_col=None,
                    scope="team",
                )
            )

    # Backward-compatible aliases for older model configs.
    # Pace of play: Bayesian average seconds between offensive plays per team per game.
    pace_pbp = pbp.copy()
    pace_pbp = pace_pbp.sort_values(["game_id", "play_id"])
    pace_pbp["elapsed_time"] = pace_pbp.groupby("game_id")["game_seconds_remaining"].diff(-1)
    pace_pbp = pace_pbp[
        pace_pbp["elapsed_time"].between(1, 120)
        & pace_pbp["play_type"].isin(["pass", "run"])
    ]
    pace_game = (
        pace_pbp.groupby(["posteam", "game_id", "game_date"], dropna=False)
        .agg(avg_elapsed=("elapsed_time", "mean"), play_count=("elapsed_time", "count"))
        .reset_index()
        .rename(columns={"posteam": "team", "avg_elapsed": "rate", "play_count": "count"})
    )
    specs.append(
        MetricSpec(
            source="pace_of_play_bayes",
            kind="normal",
            frame=pace_game[["team", "game_id", "game_date", "rate", "count"]],
            entity_col="team",
            name_col=None,
            scope="team",
        )
    )

    opponent_context_sources = {
        "opp_def_pass_epa_rate": "def_pass_epa_rate",
        "opp_def_rush_epa_rate": "def_rush_epa_rate",
        "opp_pass_epa_rate": "epa_rate",
        "opp_rushing_epa_rate": "rushing_epa_rate",
    }
    for source, base_source in opponent_context_sources.items():
        base_spec = next((spec for spec in reversed(specs) if spec.source == base_source), None)
        if base_spec is None:
            continue
        specs.append(
            MetricSpec(
                source=source,
                kind=base_spec.kind,
                frame=_opponent_context_frame(pbp, base_spec.frame),
                entity_col="team",
                name_col=None,
                count_col="count",
                date_col="game_date",
                scope="team",
            )
        )

    return specs


def _future_placeholder_frame(spec: MetricSpec, future: FutureContext) -> pd.DataFrame:
    """One zero-count, NaN-rate row per entity for each scheduled future game."""
    if spec.frame.empty:
        return pd.DataFrame()

    if spec.scope == "team":
        slots = future.team_slots.rename(columns={"team": spec.entity_col}).copy()
    else:
        slots = future.player_slots.rename(columns={"player_id": spec.entity_col}).copy()
        known = pd.DataFrame({spec.entity_col: spec.frame[spec.entity_col].dropna().astype(str).unique()})
        slots[spec.entity_col] = slots[spec.entity_col].astype(str)
        slots = slots.merge(known, on=spec.entity_col, how="inner")
        if spec.name_col is not None:
            slots = slots.rename(columns={"player_name": spec.name_col})
            known_names = (
                spec.frame[[spec.entity_col, spec.name_col]]
                .dropna(subset=[spec.entity_col])
                .assign(**{spec.entity_col: lambda f: f[spec.entity_col].astype(str)})
                .drop_duplicates(subset=[spec.entity_col], keep="last")
                .rename(columns={spec.name_col: "_known_name"})
            )
            slots = slots.merge(known_names, on=spec.entity_col, how="left")
            slots[spec.name_col] = slots["_known_name"].combine_first(slots[spec.name_col])

    if slots.empty:
        return pd.DataFrame()

    existing = set(zip(spec.frame[spec.entity_col].astype(str), spec.frame["game_id"].astype(str)))
    keep = [
        (str(entity), str(game_id)) not in existing
        for entity, game_id in zip(slots[spec.entity_col], slots["game_id"])
    ]
    slots = slots.loc[keep]
    if slots.empty:
        return pd.DataFrame()

    placeholder = pd.DataFrame(index=range(len(slots)))
    for column in spec.frame.columns:
        if column == spec.entity_col:
            placeholder[column] = slots[spec.entity_col].to_numpy()
        elif column == "game_id":
            placeholder[column] = slots["game_id"].to_numpy()
        elif column == spec.date_col:
            placeholder[column] = slots["game_date"].to_numpy()
        elif spec.name_col is not None and column == spec.name_col:
            placeholder[column] = slots[spec.name_col].to_numpy()
        elif column == spec.count_col:
            placeholder[column] = 0.0
        else:
            placeholder[column] = np.nan
    placeholder[FUTURE_SLOT_FLAG] = True
    return placeholder


def _carry_forward_direct(spec: MetricSpec, future: FutureContext) -> pd.DataFrame:
    """Direct metrics are same-game aggregates, so a future week reuses the last observed value."""
    frame = spec.frame.copy()
    if frame.empty or spec.scope != "team" or "game_id" not in frame.columns:
        return frame

    slots = future.team_slots.rename(columns={"team": spec.entity_col}).copy()
    existing = set(zip(frame[spec.entity_col].astype(str), frame["game_id"].astype(str)))
    keep = [
        (str(entity), str(game_id)) not in existing
        for entity, game_id in zip(slots[spec.entity_col], slots["game_id"])
    ]
    slots = slots.loc[keep]
    if slots.empty:
        return frame

    last_rows = (
        frame.sort_values([spec.date_col, "game_id"])
        .drop_duplicates(subset=[spec.entity_col], keep="last")
        .drop(columns=["game_id", spec.date_col])
    )
    carried = slots.merge(last_rows, on=spec.entity_col, how="inner")
    if carried.empty:
        return frame
    return pd.concat([frame, carried[frame.columns]], ignore_index=True)


def apply_metric_spec(
    spec: MetricSpec,
    decay_rates: dict[str, dict[str, float]] | None = None,
    future: FutureContext | None = None,
) -> pd.DataFrame:
    if spec.kind == "direct":
        direct = _carry_forward_direct(spec, future) if future is not None else spec.frame.copy()
        if "average" not in direct.columns:
            direct["average"] = direct.get("rate", np.nan)
        if "bayes_estimate_mean" not in direct.columns:
            direct["bayes_estimate_mean"] = direct["average"]
        if "bayes_estimate_std" not in direct.columns:
            direct["bayes_estimate_std"] = 0.0

        if spec.scope == "team":
            return standardize_team_frame(direct, source=spec.source)
        return standardize_player_frame(direct, source=spec.source)

    frame = spec.frame
    if future is not None:
        placeholder = _future_placeholder_frame(spec, future)
        if not placeholder.empty:
            frame = pd.concat(
                [frame.assign(**{FUTURE_SLOT_FLAG: False}), placeholder],
                ignore_index=True,
            )

    config = _config_for_metric(spec.kind, spec.source, decay_rates)
    smoother = smooth_binary_history if spec.kind == "binary" else smooth_normal_history
    smoothed = smoother(
        frame,
        entity_col=spec.entity_col,
        value_col="rate",
        count_col=spec.count_col,
        config=config,
        date_col=spec.date_col,
        name_col=spec.name_col,
    )

    if spec.scope == "team":
        return standardize_team_frame(smoothed, source=spec.source)
    return standardize_player_frame(smoothed, source=spec.source)


def evaluate_decay_for_metric(
    spec: MetricSpec,
    decay: float,
    season_carryover: float,
    aggregation: Literal["row", "player"] = "row",
) -> float:
    candidate_config = SmoothingConfig(
        kind=spec.kind,
        decay=float(decay),
        season_carryover=float(season_carryover),
    )
    smoother = smooth_binary_history if spec.kind == "binary" else smooth_normal_history
    smoothed = smoother(
        spec.frame,
        entity_col=spec.entity_col,
        value_col="rate",
        count_col=spec.count_col,
        config=candidate_config,
        date_col=spec.date_col,
        name_col=spec.name_col,
    )

    evaluation = smoothed.merge(
        spec.frame[[spec.entity_col, "game_id", spec.date_col, "rate"]],
        on=[spec.entity_col, "game_id", spec.date_col],
        how="inner",
        suffixes=("_pred", "_actual"),
    )
    evaluation = evaluation.dropna(subset=["bayes_estimate_mean", "rate"])
    if evaluation.empty:
        return float("inf")

    error = evaluation["bayes_estimate_mean"] - evaluation["rate"]
    squared_error = np.square(error)
    if aggregation == "player":
        squared_error = squared_error.groupby(evaluation[spec.entity_col]).mean()
    return float(np.mean(squared_error))


def evaluate_decay_grid_for_metric(
    spec: MetricSpec,
    decay_grid: Iterable[float] | None = None,
    season_carryover_grid: Iterable[float] | None = None,
) -> pd.DataFrame:
    if spec.kind == "direct":
        raise ValueError("Decay evaluation is only available for binary and normal metrics.")

    decays = (
        np.round(np.arange(0.5, 1.0, 0.05), 2)
        if decay_grid is None
        else np.array(sorted({round(float(value), 4) for value in decay_grid}))
    )
    season_carryovers = (
        np.round(np.arange(0.75, 1.0, 0.05), 2)
        if season_carryover_grid is None
        else np.array(sorted({round(float(value), 4) for value in season_carryover_grid}))
    )

    rows = [
        {
            "metric": spec.source,
            "decay": float(decay),
            "season_carryover": float(season_carryover),
            "mse": evaluate_decay_for_metric(spec, float(decay), float(season_carryover)),
            "player_mse": evaluate_decay_for_metric(
                spec,
                float(decay),
                float(season_carryover),
                aggregation="player",
            ),
        }
        for decay in decays
        for season_carryover in season_carryovers
    ]
    return pd.DataFrame(rows).sort_values("mse", ignore_index=True)


def fit_decay_rates(
    metric_specs: list[MetricSpec],
    decay_grid: Iterable[float] | None = None,
) -> dict[str, dict[str, float]]:
    if decay_grid is None:
        coarse_grid = np.round(np.arange(0.5, 1.0, 0.05), 2)
    else:
        coarse_grid = np.array(sorted({round(float(value), 4) for value in decay_grid}))

    coarse_season_grid = np.round(np.arange(0.75, 1.0, 0.05), 2)
    tuned: dict[str, dict[str, float]] = {}
    for spec in metric_specs:
        if spec.kind == "direct":
            continue

        best_decay = None
        best_season_carryover = None
        best_score = float("inf")
        for decay in coarse_grid:
            for season_carryover in coarse_season_grid:
                score = evaluate_decay_for_metric(spec, float(decay), float(season_carryover))
                if score < best_score:
                    best_score = score
                    best_decay = float(decay)
                    best_season_carryover = float(season_carryover)

        if best_decay is not None:
            lower = max(0.5, best_decay - 0.04)
            upper = min(0.99, best_decay + 0.04)
            fine_grid = np.round(np.arange(lower, upper + 0.001, 0.01), 2)
            center = best_season_carryover if best_season_carryover is not None else 0.95
            season_lower = max(0.5, center - 0.04)
            season_upper = min(0.99, center + 0.04)
            fine_season_grid = np.round(np.arange(season_lower, season_upper + 0.001, 0.01), 2)
            for decay in fine_grid:
                for season_carryover in fine_season_grid:
                    score = evaluate_decay_for_metric(spec, float(decay), float(season_carryover))
                    if score < best_score:
                        best_score = score
                        best_decay = float(decay)
                        best_season_carryover = float(season_carryover)

        tuned[spec.source] = {
            "decay": best_decay if best_decay is not None else 0.8,
            "season_carryover": best_season_carryover if best_season_carryover is not None else 0.95,
        }
        print(
            f"Best params for {spec.source}: decay={tuned[spec.source]['decay']:.2f}, "
            f"season_carryover={tuned[spec.source]['season_carryover']:.2f} (mse={best_score:.6f})"
        )
    return tuned


def build_player_metrics(
    pbp: pd.DataFrame,
    decay_rates: dict[str, dict[str, float]] | None = None,
    future: FutureContext | None = None,
) -> pd.DataFrame:
    frames = [apply_metric_spec(spec, decay_rates, future) for spec in collect_player_metric_specs(pbp)]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["player_id", "player_name", "game_id", "source"], keep="last")
    return combined


def build_team_metrics(
    pbp: pd.DataFrame,
    decay_rates: dict[str, dict[str, float]] | None = None,
    future: FutureContext | None = None,
) -> pd.DataFrame:
    frames = [apply_metric_spec(spec, decay_rates, future) for spec in collect_team_metric_specs(pbp, future=future)]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["team", "game_id", "source"], keep="last")
    return combined


def build_wide_tables(
    player_stats: pd.DataFrame,
    team_stats: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot the long feature tables into the wide form the simulation runtime reads."""
    player_stats = player_stats.copy()
    team_stats = team_stats.copy()
    player_stats["average"] = player_stats["bayes_estimate_mean"].combine_first(player_stats["average"])
    team_stats["average"] = team_stats["bayes_estimate_mean"].combine_first(team_stats["average"])

    try:
        ids = nfl.import_ids()
        id_to_name = dict(zip(ids["gsis_id"], ids["name"]))
        player_stats["player_name"] = player_stats["player_id"].map(id_to_name)
    except Exception as error:  # keep the builder usable offline
        print(f"Skipping player name normalization: {error}")

    player_stats_wide = (
        player_stats.pivot_table(
            index=["player_id", "player_name", "game_id", "game_date"],
            columns=["source"],
            values="average",
        )
        .reset_index()
        .sort_values(by=["player_id", "game_date", "game_id"])
    )
    player_stats_wide = player_stats_wide.groupby("player_id", group_keys=False).apply(lambda g: g.ffill())
    player_stats_wide = player_stats_wide.set_index(["player_id", "game_id"], drop=True)

    team_stats_wide = (
        team_stats.pivot_table(index=["team", "game_id", "game_date"], columns=["source"], values="average")
        .reset_index()
        .sort_values(by=["team", "game_date", "game_id"])
    )
    team_stats_wide = team_stats_wide.groupby("team", group_keys=False).apply(lambda g: g.ffill())
    for column in TEAM_DIRECT_LAG_COLUMNS:
        if column in team_stats_wide.columns:
            team_stats_wide[column] = team_stats_wide.groupby("team")[column].shift(1)
    team_stats_wide = team_stats_wide.set_index(["team", "game_id"], drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    player_stats_wide.to_csv(output_dir / "player_stats_wide.csv")
    team_stats_wide.to_csv(output_dir / "team_stats_wide.csv")
    return player_stats_wide, team_stats_wide


def build_feature_tables(
    years: Iterable[int] = DEFAULT_YEARS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cache_path: Path = DEFAULT_CACHE_PATH,
    decay_file: Path = DEFAULT_DECAY_FILE,
    fit_decays: bool = False,
    future_weeks: int = 0,
    write_wide: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pbp = load_pbp_data(years, cache_path=cache_path)

    if fit_decays:
        metric_specs = collect_player_metric_specs(pbp) + collect_team_metric_specs(pbp)
        decay_rates = fit_decay_rates(metric_specs)
        save_decay_rates(decay_rates, decay_file)
    else:
        decay_rates = load_decay_rates(decay_file)

    future = build_future_context(pbp, weeks_ahead=future_weeks) if future_weeks > 0 else None

    player_stats = build_player_metrics(pbp, decay_rates=decay_rates, future=future)
    team_stats = build_team_metrics(pbp, decay_rates=decay_rates, future=future)

    output_dir.mkdir(parents=True, exist_ok=True)
    player_stats.to_csv(output_dir / "player_stats.csv", index=False)
    team_stats.to_csv(output_dir / "Team_Stats.csv", index=False)
    if write_wide:
        build_wide_tables(player_stats, team_stats, output_dir=output_dir)
    return player_stats, team_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Bayesian football feature tables.")
    parser.add_argument(
        "--fit-decays",
        action="store_true",
        help="Backtest historical one-step-ahead error and save the best decay per metric.",
    )
    parser.add_argument(
        "--decay-file",
        type=Path,
        default=DEFAULT_DECAY_FILE,
        help="JSON file used to load or save per-metric decay values.",
    )
    parser.add_argument(
        "--evaluate-metric",
        type=str,
        help="Evaluate every coarse decay/season-carryover combination for one metric and save a CSV grid.",
    )
    parser.add_argument(
        "--grid-output",
        type=Path,
        default=Path("decay_grid.csv"),
        help="CSV path for --evaluate-metric output.",
    )
    parser.add_argument(
        "--future-weeks",
        type=int,
        default=1,
        help="Emit pregame feature rows for this many upcoming scheduled weeks that have no play-by-play yet.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(DEFAULT_YEARS),
        help="Seasons of play-by-play to build features from.",
    )
    parser.add_argument(
        "--write-wide",
        action="store_true",
        help="Also export player_stats_wide.csv and team_stats_wide.csv for the simulation runtime.",
    )
    args = parser.parse_args()

    if args.evaluate_metric:
        pbp = load_pbp_data(args.years)
        metric_specs = collect_player_metric_specs(pbp) + collect_team_metric_specs(pbp)
        matching_specs = [spec for spec in metric_specs if spec.source == args.evaluate_metric]
        if not matching_specs:
            available = ", ".join(sorted(spec.source for spec in metric_specs))
            raise SystemExit(f"Unknown metric {args.evaluate_metric!r}. Available metrics: {available}")

        grid = evaluate_decay_grid_for_metric(matching_specs[0])
        args.grid_output.parent.mkdir(parents=True, exist_ok=True)
        grid.to_csv(args.grid_output, index=False)
        best = grid.iloc[0]
        print(
            f"Saved {len(grid)} rows to {args.grid_output}. "
            f"Best: decay={best['decay']:.2f}, "
            f"season_carryover={best['season_carryover']:.2f}, mse={best['mse']:.6f}"
        )
        return

    build_feature_tables(
        years=args.years,
        decay_file=args.decay_file,
        fit_decays=args.fit_decays,
        future_weeks=args.future_weeks,
        write_wide=args.write_wide,
    )


if __name__ == "__main__":
    main()
