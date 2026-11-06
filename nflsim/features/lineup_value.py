"""Recompute pregame player value for a simulated lineup and shift lineup-dependent ridge features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from nflsim.features.player_value import (
    GROUP_LIMITS, PV_COLUMNS, PV_DEFENSE_COLUMNS, PV_OFFENSE_COLUMNS, ROSTER_GROUP,
    depth_dl_order, lineup_player_value, lineup_requests,
)
from nflsim.names import row_display_name
from nflsim.ridge.metrics import TARGETS
from nflsim.semantics import is_isolated

OPPONENT_ALIASES = {"opp_def_pass_epa_rate": "def_pass_epa_rate", "opp_def_rush_epa_rate": "def_rush_epa_rate",
                    "opp_pass_epa_rate": "epa_rate", "opp_rushing_epa_rate": "rushing_epa_rate"}
BINARY_OUTPUTS = {output for target in TARGETS if target.kind == "binary"
                  for output in (target.player_output, target.defense_output) if output}


class PlayerValueTables:
    def __init__(self, directory: Path):
        directory = Path(directory)
        self.directory = directory
        self.estimates = pd.read_csv(directory / "pm_estimates.csv", dtype={"player_id": str})
        self.history = pd.read_csv(directory / "snap_share_history.csv", dtype={"player_id": str, "game_id": str},
                                   parse_dates=["game_date"])
        self.values = pd.read_csv(directory / "player_value.csv", dtype={"game_id": str, "team": str})
        self.coefficients = pd.read_csv(directory / "ridge_pv_coefficients.csv")
        raw = json.loads((directory / "pv_share_defaults.json").read_text(encoding="utf-8"))
        self.defaults = {(key.split("|")[0], int(key.split("|")[1])): float(value) for key, value in raw.items()}
        self.isolated_mu = is_isolated(directory)

    def bundle_values(self, game_id: str, team: str) -> dict[str, float] | None:
        match = self.values[self.values.game_id.eq(str(game_id)) & self.values.team.eq(str(team))]
        if match.empty:
            return None
        row = match.iloc[0]
        return {column: float(row[column]) for column in PV_COLUMNS if np.isfinite(row[column])}

    def betas(self, season: int, week: int) -> pd.DataFrame:
        frame = self.coefficients
        key = frame.season * 100 + frame.week
        prior = frame[key <= int(season) * 100 + int(week)]
        if prior.empty:
            return prior
        latest = (prior.season * 100 + prior.week).max()
        return prior[(prior.season * 100 + prior.week) == latest]


def player_value_tables(runtime) -> PlayerValueTables | None:
    """Cached bundle PV tables; None for fixtures without them (sealed bundles are validated to have them)."""
    if not hasattr(runtime, "_pv_tables"):
        directory = getattr(runtime, "data_dir", None)
        ready = directory is not None and (Path(directory) / "pm_estimates.csv").exists()
        runtime._pv_tables = PlayerValueTables(directory) if ready else None
    return runtime._pv_tables


def dl_assignments(roster: pd.DataFrame, depth: pd.DataFrame, team: str, week: int, kickoff, tables: PlayerValueTables,
                   roster_for_team_week) -> list[dict[str, str]]:
    roster_week = roster_for_team_week(roster, team, week)
    if roster_week.empty:
        return []
    ordered = depth_dl_order(roster_week, depth, team, kickoff, tables.history)[:GROUP_LIMITS["DL"]]
    names = {str(row.player_id): row_display_name(row) for row in roster_week.itertuples()}
    return [{"player_id": pid, "player_name": names.get(pid, pid), "position": "DL"} for pid in ordered]


def lineup_values(tables: PlayerValueTables, assignments: Mapping[str, Mapping[str, str]], dl: list[Mapping[str, str]] | list[str],
                  season: int, week: int, kickoff) -> tuple[dict[str, float], pd.DataFrame]:
    dl_ids = [item["player_id"] if isinstance(item, Mapping) else str(item) for item in dl]
    requests = lineup_requests(assignments, dl_ids, season, week, kickoff)
    return lineup_player_value(requests, tables.estimates, tables.history, tables.defaults)


def _metric_deltas(betas: pd.DataFrame, old: Mapping[str, float], new: Mapping[str, float], team: str) -> dict[tuple[str, str], float]:
    deltas: dict[tuple[str, str], float] = {}
    for row in betas.itertuples():
        term = str(row.term)
        if term.startswith("offense_qb:"):
            if term.partition(":")[2] != str(team) or "pv_off_qb" not in new or "pv_off_qb" not in old:
                continue
            change = float(new["pv_off_qb"]) - float(old["pv_off_qb"])
        else:
            if term not in new or term not in old:
                continue
            change = float(new[term]) - float(old[term])
        key = (str(row.metric), str(row.side))
        deltas[key] = deltas.get(key, 0.0) + float(row.beta) * change
    return deltas


def _shift(value, delta: float, metric: str) -> float:
    shifted = float(value) + delta
    return float(np.clip(shifted, 0.0, 1.0)) if metric in BINARY_OUTPUTS else shifted


def apply_lineup_value(tables: PlayerValueTables, team_features: pd.DataFrame, players: Mapping[str, list],
                       new_values: Mapping[str, Mapping[str, float]], season: int, week: int,
                       old_values: Mapping[str, Mapping[str, float] | None] | None = None,
                       update_aliases: bool = True) -> pd.DataFrame:
    """Overwrite raw PV inputs; legacy bundles also shift folded estimates by beta * (new PV - bundle PV).

    Isolated-mu bundles never mutate entity mus: a lineup change only updates the PV input columns.
    ``team_features`` is indexed by team; ``players`` maps team to Player objects.
    """
    features = team_features.copy()
    isolated = bool(getattr(tables, "isolated_mu", False))
    betas = tables.betas(season, week) if not isolated else None
    old_values = old_values or {}
    for team in features.index:
        new = dict(new_values[team])
        if not isolated:
            old = old_values.get(team) or {column: float(features.loc[team].get(column, np.nan)) for column in PV_COLUMNS}
            old = {column: value for column, value in old.items() if np.isfinite(value)}
            deltas = _metric_deltas(betas, old, new, str(team))
        else:
            deltas = {}
        for (metric, side), delta in deltas.items():
            if metric in features.columns and np.isfinite(features.loc[team, metric]):
                features.loc[team, metric] = _shift(features.loc[team, metric], delta, metric)
            if side != "subject":
                continue
            for player in players.get(team, []):
                attributes = player.attributes
                if metric in attributes.columns and np.isfinite(float(attributes.iloc[0][metric])):
                    attributes.loc[attributes.index[0], metric] = _shift(attributes.iloc[0][metric], delta, metric)
        for column in PV_COLUMNS:
            if column in features.columns or column in new:
                features.loc[team, column] = float(new[column])
    teams = list(features.index)
    if update_aliases and len(teams) == 2:
        for team, opponent in ((teams[0], teams[1]), (teams[1], teams[0])):
            for alias, source in OPPONENT_ALIASES.items():
                if alias in features.columns and source in features.columns:
                    features.loc[team, alias] = features.loc[opponent, source]
    return features


def neutral_values(tables: PlayerValueTables, season: int, week: int) -> dict[str, float]:
    """League-mean team PV over the latest completed season-to-date window before (season, week)."""
    values = tables.values.copy()
    parsed = values.game_id.str.extract(r"^(\d{4})_(\d{2})_").astype(float)
    key = parsed[0] * 100 + parsed[1]
    earlier = values[key < int(season) * 100 + int(week)]
    if earlier.empty:
        return {column: 0.0 for column in PV_COLUMNS}
    latest_season = (key[earlier.index] // 100).max()
    window = earlier[(key[earlier.index] // 100).eq(latest_season)]
    return {column: float(window[column].mean()) for column in PV_COLUMNS}


__all__ = ["PlayerValueTables", "player_value_tables", "dl_assignments", "lineup_values", "apply_lineup_value",
           "neutral_values", "PV_OFFENSE_COLUMNS", "PV_DEFENSE_COLUMNS", "ROSTER_GROUP"]
