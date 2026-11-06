"""Offline retained metrics with season-frozen Bayesian priors.

Direct team tendencies remain same-game aggregates for wide_tables' single lag.
Massey and QB continuity are already pregame in the shared collectors.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import FunctionType
from typing import Iterable

import numpy as np
import pandas as pd

from bayesian_feature_builder import (
    FutureContext, MetricSpec, _config_for_metric, apply_metric_spec,
    collect_player_metric_specs, collect_team_metric_specs,
    standardize_player_frame, standardize_team_frame,
)
from compare_weighted_rolling_features import _frozen_bayesian_history
from ridge_game_sim.feature_builder import canonicalize_teams, prepare_pbp
from ridge_game_sim.paths import digest, output_dir, write_json


PLAYER_COLUMNS = ["player_id", "player_name", "game_id", "game_date", "source",
                  "average", "bayes_estimate_mean", "bayes_estimate_std"]
TEAM_COLUMNS = ["team", "game_id", "game_date", "source", "average",
                "bayes_estimate_mean", "bayes_estimate_std"]
DIRECT_SOURCES = {
    "qb_team_games_played", "pass_oe_weekly_variation", "target_gini_roll4",
    "rb_carry_split_gini_roll4", "rb_top_carry_share_roll4",
    "passing_massey_rating", "rushing_massey_rating",
    "passing_massey_offense", "passing_massey_defense",
    "rushing_massey_offense", "rushing_massey_defense",
}


def season_week(frame: pd.DataFrame) -> pd.DataFrame:
    parts = frame["game_id"].astype(str).str.extract(r"^(\d{4})_(\d{2})_")
    if parts.isna().any().any():
        raise ValueError("Retained metrics require NFL season/week game IDs")
    return parts.astype(int).rename(columns={0: "season", 1: "week"})


def _slots(spec: MetricSpec, schedule: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    if spec.scope == "team":
        slots = pd.concat([
            schedule.assign(team=schedule.home_team),
            schedule.assign(team=schedule.away_team),
        ], ignore_index=True)[["team", "game_id", "game_date"]]
        return slots.rename(columns={"team": spec.entity_col, "game_date": spec.date_col})
    roles = {
        "passer_player_id": {"QB"},
        "receiver_player_id": {"WR", "TE", "RB", "FB"},
        "rusher_player_id": {"RB", "FB", "QB", "WR"},
        "player_id": {"QB", "WR", "TE", "RB", "FB"},
    }
    if spec.entity_col not in roles:
        raise ValueError(f"Unsupported player entity: {spec.entity_col}")
    slots = roster[roster.position.isin(roles[spec.entity_col])].copy()
    slots = slots.drop(columns=["game_date"], errors="ignore").merge(
        schedule[["game_id", "game_date"]], on="game_id", validate="many_to_one",
    )
    slots["player_name"] = slots.get("player_name", slots.player_id)
    columns = ["player_id", "game_id", "game_date"]
    if spec.name_col:
        columns.append("player_name")
    return slots[columns].rename(columns={
        "player_id": spec.entity_col, "player_name": spec.name_col or "player_name",
        "game_date": spec.date_col,
    })


def retained_from_specs(
    specs: Iterable[MetricSpec], schedule: pd.DataFrame, roster: pd.DataFrame,
    prediction_season: int, excluded_sources: Iterable[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Emit one season; no prior or observation may come from a later season."""
    schedule, roster = [canonicalize_teams(frame) for frame in (schedule, roster)]
    schedule = schedule.loc[season_week(schedule).season <= prediction_season].copy()
    if "game_date" not in schedule:
        schedule["game_date"] = schedule["gameday"]
    schedule["game_date"] = pd.to_datetime(schedule.game_date)
    roster = roster[roster.game_id.isin(schedule.game_id)].copy()
    frames = {"player": [], "team": []}
    diagnostics = []
    seen = set()
    for spec in specs:
        if spec.source in set(excluded_sources):
            continue
        if spec.kind not in {"binary", "normal", "direct"} or spec.scope not in frames:
            raise ValueError(f"Unsupported retained spec: {spec.source}/{spec.kind}/{spec.scope}")
        key = (spec.scope, spec.source)
        if key in seen:
            raise ValueError(f"Duplicate retained spec: {key}")
        seen.add(key)
        if spec.kind == "direct" and spec.source not in DIRECT_SOURCES:
            raise ValueError(f"Unsupported direct timing semantics: {spec.source}")
        frame = canonicalize_teams(spec.frame)
        required = {spec.entity_col, "game_id", spec.date_col, "rate"}
        if spec.kind != "direct":
            required.add(spec.count_col)
        if not required <= set(frame):
            raise ValueError(f"{spec.source}: missing {sorted(required - set(frame))}")
        frame = frame.loc[season_week(frame).season <= prediction_season].copy()
        frame = frame.dropna(subset=[spec.entity_col])
        frame[spec.entity_col] = frame[spec.entity_col].astype(str)
        frame[spec.date_col] = pd.to_datetime(frame[spec.date_col], errors="raise")
        keys = [spec.entity_col, "game_id", spec.date_col]
        if frame.duplicated([spec.entity_col, "game_id"]).any():
            raise ValueError(f"{spec.source}: duplicate entity/game observations")
        if spec.kind == "direct":
            future_slots = _slots(spec, schedule, roster)
            slot_seasons = season_week(future_slots) if not future_slots.empty else pd.DataFrame(columns=["season", "week"])
            if not future_slots.empty:
                future_slots = future_slots.loc[slot_seasons.season.eq(prediction_season)].copy()
                future_slots = future_slots.loc[~future_slots.game_id.astype(str).isin(frame.game_id.astype(str))]
            if not future_slots.empty:
                records = []
                current_periods = season_week(frame)
                frame = frame.copy()
                frame["_season"] = current_periods.season
                frame["_week"] = current_periods.week
                for slot in future_slots.itertuples(index=False):
                    entity = str(getattr(slot, spec.entity_col))
                    slot_period = season_week(pd.DataFrame({"game_id": [slot.game_id]})).iloc[0]
                    history = frame.loc[(frame[spec.entity_col].astype(str).eq(entity)) &
                                         ((frame._season < slot_period.season) |
                                          ((frame._season == slot_period.season) & (frame._week < slot_period.week)))].sort_values([spec.date_col, "game_id"])
                    value = float(history.rate.iloc[-1]) if not history.empty and pd.notna(history.rate.iloc[-1]) else 0.0
                    if spec.source == "qb_team_games_played" and not history.empty:
                        value += 1.0
                    record = {spec.entity_col: entity, "game_id": str(slot.game_id), spec.date_col: getattr(slot, spec.date_col), "rate": value}
                    if spec.name_col:
                        record[spec.name_col] = getattr(slot, spec.name_col, entity)
                    records.append(record)
                frame = frame.drop(columns=["_season", "_week"])
                frame = pd.concat([frame, pd.DataFrame(records)], ignore_index=True)
            result = apply_metric_spec(replace(spec, frame=frame))
            prior_rows = None
            reason = "direct_shared_definition_with_pregame_slots"
        else:
            frame["rate"] = pd.to_numeric(frame["rate"], errors="raise")
            frame[spec.count_col] = pd.to_numeric(frame[spec.count_col], errors="raise")
            valid = np.isfinite(frame.rate) & np.isfinite(frame[spec.count_col]) & frame[spec.count_col].gt(0)
            if spec.kind == "binary" and not frame.loc[valid, "rate"].between(0, 1).all():
                raise ValueError(f"{spec.source}: binary observations outside [0, 1]")
            frame.loc[~valid, "rate"] = np.nan
            frame.loc[~valid, spec.count_col] = np.nan
            training = valid & season_week(frame).season.lt(prediction_season)
            prior_rows = int(training.sum())
            if not prior_rows:
                diagnostics.append({"season": prediction_season, "scope": spec.scope,
                                    "metric": spec.source, "prior_rows": 0, "rows": 0,
                                    "reason": "no_preseason_observations"})
                continue
            slots = _slots(spec, schedule, roster)
            slots[spec.entity_col] = slots[spec.entity_col].astype(str)
            slots = slots.drop_duplicates([spec.entity_col, "game_id"])
            missing = slots.merge(frame[keys], on=keys, how="left", indicator=True)
            missing = missing[missing._merge.eq("left_only")].drop(columns="_merge")
            missing["rate"] = np.nan
            missing[spec.count_col] = np.nan
            history = pd.concat([frame, missing], ignore_index=True)
            periods = season_week(history)
            history["season"] = periods.season
            if history.assign(_season=periods.season, _week=periods.week).duplicated(
                [spec.entity_col, "_season", "_week"]
            ).any():
                raise ValueError(f"{spec.source}: multiple entity games in a league week unsupported")
            config = _config_for_metric(spec.kind, spec.source, None)
            predictions = _frozen_bayesian_history(spec, history, config, prediction_season - 1)
            result = predictions.merge(history.drop(columns=["rate", spec.count_col]), on=keys,
                                       how="left", validate="one_to_one")
            result["average"] = result["bayesian_prediction"]
            result["bayes_estimate_mean"] = result["average"]
            result["bayes_estimate_std"] = np.nan
            result = result.rename(columns={spec.entity_col: "team" if spec.scope == "team" else "player_id",
                                            spec.date_col: "game_date"})
            if spec.name_col:
                result = result.rename(columns={spec.name_col: "player_name"})
            standardize = standardize_team_frame if spec.scope == "team" else standardize_player_frame
            result = standardize(result, spec.source)
            reason = "season_frozen_prior_preweek_update"
        result = result.loc[season_week(result).season.eq(prediction_season)].copy()
        if result.empty:
            reason = "no_metric_observations"
        if np.isinf(result[["average", "bayes_estimate_mean"]].to_numpy()).any():
            raise ValueError(f"{spec.source}: infinite retained estimates")
        frames[spec.scope].append(result)
        diagnostics.append({"season": prediction_season, "scope": spec.scope, "metric": spec.source,
                            "prior_rows": prior_rows, "rows": len(result), "reason": reason})
    def combine(scope, columns):
        return pd.concat(frames[scope], ignore_index=True) if frames[scope] else pd.DataFrame(columns=columns)
    return combine("player", PLAYER_COLUMNS), combine("team", TEAM_COLUMNS), pd.DataFrame(diagnostics)


def future_context_from_schedule(upcoming: pd.DataFrame) -> FutureContext | None:
    if upcoming.empty:
        return None
    future_games = upcoming[["game_id", "game_date", "home_team", "away_team", "season", "week"]].copy()
    future_games["game_date"] = pd.to_datetime(future_games["game_date"], errors="coerce")
    future_games = future_games.dropna(subset=["game_date", "home_team", "away_team"])
    if future_games.empty:
        return None
    future_teams = pd.concat([
        future_games[["home_team", "game_id", "game_date"]].rename(columns={"home_team": "team"}),
        future_games[["away_team", "game_id", "game_date"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True)
    return FutureContext(future_games, future_teams, pd.DataFrame())


def prepare_retained_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    frame = prepare_pbp(pbp)
    if "season_type" in frame:
        frame = frame[frame.season_type.eq("REG")].copy()
    if "game_date" not in frame:
        frame["game_date"] = frame["gameday"]
    frame["game_date"] = pd.to_datetime(frame.game_date)
    derived = {
        "is_rz": lambda: frame.yardline_100.le(20).astype(int),
        "close": lambda: frame.ydstogo.le(2).astype(int),
        "went_for_it": lambda: frame.play_type.isin(["run", "pass"]).astype(int),
        "game_state": lambda: np.select([frame.score_differential.gt(3), frame.score_differential.lt(-3)],
                                        ["winning", "losing"], default="tied"),
    }
    for name, compute in derived.items():
        if name not in frame:
            frame[name] = compute()
    if "is_interception_worthy" not in frame:
        raise ValueError("Explicit FTN is_interception_worthy required; zero substitution forbidden")
    if "route" not in frame or not frame.route.notna().any():
        raise ValueError("Explicit FTN route observations required; route metrics cannot be silently omitted")
    return frame


def _offline_player_specs(pbp: pd.DataFrame, snaps: pd.DataFrame) -> list[MetricSpec]:
    """Rebind only the collector's I/O dependency, without mutating shared globals."""
    pbp = normalize_player_names(pbp)
    required = {"game_type", "offense_snaps", "position", "gsis_id", "name", "team", "game_id"}
    if not required <= set(snaps):
        raise ValueError(f"Local snap counts missing {sorted(required - set(snaps))}")
    namespace = dict(collect_player_metric_specs.__globals__)
    namespace["load_snap_counts"] = lambda years: snaps.loc[
        season_week(snaps).season.isin(years)
    ].copy()
    collector = FunctionType(collect_player_metric_specs.__code__, namespace,
                             collect_player_metric_specs.__name__, collect_player_metric_specs.__defaults__)
    return collector(pbp)


def normalize_player_names(pbp: pd.DataFrame) -> pd.DataFrame:
    frame = pbp.copy()
    for role in ("receiver", "passer", "rusher"):
        identity, name = f"{role}_player_id", f"{role}_player_name"
        if identity not in frame or name not in frame:
            continue
        known = frame.loc[frame[identity].notna() & frame[name].notna(), [identity, name]]
        names = known.drop_duplicates(identity, keep="first").set_index(identity)[name]
        frame[name] = frame[identity].map(names).combine_first(frame[name])
    return frame


def regenerate_retained_features(
    pbp: pd.DataFrame, schedule: pd.DataFrame, roster: pd.DataFrame,
    snap_counts: pd.DataFrame, destination: Path, excluded_sources: Iterable[str] = (),
    seasons: Iterable[int] = range(2017, 2027), input_paths: Iterable[Path] = (),
    prior_bundle: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    destination = output_dir(destination)
    if any((destination / name).exists() for name in ("bundle_manifest.json", "training_manifest.json")):
        raise ValueError("Cannot regenerate retained features beneath trained/sealed models")
    diagnostics = []
    player_frames, team_frames = [], []
    seasons = sorted(set(seasons))
    if not seasons or any(season < 2017 or season > 2100 for season in seasons):
        raise ValueError("Prediction seasons must be within 2017-2100")
    excluded_sources = tuple(excluded_sources)
    try:
        pbp = prepare_retained_pbp(pbp)
        schedule, roster, snap_counts = [canonicalize_teams(frame) for frame in (schedule, roster, snap_counts)]
        for season in seasons:
            prefix = pbp.loc[season_week(pbp).season <= season].copy()
            if prefix.empty:
                diagnostics.append(pd.DataFrame([{"season": season, "reason": "no_pbp", "rows": 0}]))
                continue
            played_ids = set(prefix.game_id.astype(str))
            upcoming = schedule.loc[season_week(schedule).season.eq(season) & ~schedule.game_id.astype(str).isin(played_ids)].copy()
            future = future_context_from_schedule(upcoming)
            specs = _offline_player_specs(prefix, snap_counts) + collect_team_metric_specs(prefix, future=future)
            conditional = {
                "player": {f"{route}_bayes" for route in (
                    "FLAT", "QUICK OUT", "HITCH", "SLANT", "CURL", "DIG", "POST",
                    "GO", "OUT", "CROSS", "WHEEL", "SWING", "SCREEN", "CORNER",
                )},
                "team": {f"{side}{state}_rate" for side in ("", "def_")
                         for state in ("winning", "losing", "tied")} | {
                    "passing_massey_offense", "passing_massey_defense",
                    "rushing_massey_offense", "rushing_massey_defense",
                },
            }
            for scope, sources in conditional.items():
                absent = sources - {spec.source for spec in specs if spec.scope == scope} - set(excluded_sources)
                if absent:
                    diagnostics.append(pd.DataFrame([
                        {"season": season, "scope": scope, "metric": source,
                         "reason": "no_metric_observations", "rows": 0}
                        for source in sorted(absent)
                    ]))
            player, team, diagnostic = retained_from_specs(specs, schedule, roster, season, excluded_sources)
            player_frames.append(player)
            team_frames.append(team)
            diagnostics.append(diagnostic)
        players = pd.concat(player_frames, ignore_index=True) if player_frames else pd.DataFrame(columns=PLAYER_COLUMNS)
        teams = pd.concat(team_frames, ignore_index=True) if team_frames else pd.DataFrame(columns=TEAM_COLUMNS)
        if prior_bundle is not None:
            refresh_from = min(seasons)
            previous_player = pd.read_csv(Path(prior_bundle) / "retained_player_stats.csv", dtype={"player_id": str, "game_id": str})
            previous_team = pd.read_csv(Path(prior_bundle) / "retained_team_stats.csv", dtype={"team": str, "game_id": str})
            previous_player = previous_player.loc[season_week(previous_player).season.lt(refresh_from)]
            previous_team = previous_team.loc[season_week(previous_team).season.lt(refresh_from)]
            players = pd.concat([previous_player, players], ignore_index=True, sort=False)
            teams = pd.concat([previous_team, teams], ignore_index=True, sort=False)
        for name, frame in (("retained_player_stats.csv", players), ("retained_team_stats.csv", teams)):
            frame.to_csv(destination / name, index=False)
        write_json(destination / "retained_manifest.json", {
            "status": "complete", "prediction_seasons": seasons, "prior_cutoff": "prediction_season - 1",
            "estimator": "frozen_bayesian_history", "decay": "fixed shared defaults (0.8, 0.95); no fitted root decay reuse",
            "direct_metrics": "shared definitions; team tendency lag applied once by wide_tables",
            "uncertainty": "smooth posterior std unavailable from frozen helper",
            "excluded_sources": sorted(excluded_sources),
            "carried_forward_from": str(prior_bundle) if prior_bundle is not None else None,
            "input_hashes": {str(path): digest(path) for path in input_paths},
            "implementation_hashes": {str(Path(__file__)): digest(Path(__file__)),
                                      str(Path(collect_team_metric_specs.__code__.co_filename)): digest(Path(collect_team_metric_specs.__code__.co_filename)),
                                      str(Path(_frozen_bayesian_history.__code__.co_filename)): digest(Path(_frozen_bayesian_history.__code__.co_filename))},
            "tables": {name: digest(destination / name) for name in ("retained_player_stats.csv", "retained_team_stats.csv")},
        })
    except Exception as error:
        diagnostics.append(pd.DataFrame([{"reason": "failed", "error": str(error)}]))
        write_json(destination / "retained_manifest.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        diagnostic = pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame(columns=["season", "metric", "reason", "rows"])
        diagnostic.to_csv(destination / "retained_diagnostics.csv", index=False)
    return players, teams, diagnostic