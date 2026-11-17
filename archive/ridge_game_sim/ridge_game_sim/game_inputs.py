from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from ridge_game_sim.paths import ROOT
from ridge_game_sim.runtime import RidgeRuntime
from torch_game_simulation_runtime import Player, ROOF_TYPE, SKILL_SLOT_ORDER

SLOT_LIMITS = {"QB": 1, "WR": 4, "RB": 3, "TE": 3}
SLOT_RANK_STAT = {"WR": "medium", "TE": "medium", "RB": "rush_rate"}
MAX_REST_DAYS = 21.0


def load_runtime(artifact_dir: Path = ROOT / "artifacts/runtime") -> RidgeRuntime:
    from ridge_game_sim.bundle import validate_bundle, verify_bundle_hash

    validate_bundle(artifact_dir, require_ready=True)
    verify_bundle_hash(artifact_dir)
    return RidgeRuntime(artifact_dir, strict_schema=True)


def pregame_team_snapshot(
    runtime: RidgeRuntime,
    team_stats: pd.DataFrame,
    team: str,
    kickoff: pd.Timestamp,
    *,
    home_rest: float | None = None,
    opponent_rest: float | None = None,
) -> pd.Series:
    frame = team_stats[team_stats.team.astype(str).eq(str(team))].copy()
    frame["game_date"] = pd.to_datetime(frame.game_date, errors="coerce")
    frame = frame[frame.game_date.le(pd.Timestamp(kickoff))]
    if frame.empty:
        raise ValueError(f"No pregame feature snapshot for {team} on or before {pd.Timestamp(kickoff)}")
    row = frame.sort_values(["game_date", "game_id"]).iloc[-1]
    row = row.reindex(runtime.required_team_feature_columns)
    if home_rest is not None and "days_rest" in row:
        row["days_rest"] = min(max(float(home_rest), 0.0), MAX_REST_DAYS)
    if opponent_rest is not None and "opp_days_rest" in row:
        row["opp_days_rest"] = min(max(float(opponent_rest), 0.0), MAX_REST_DAYS)
    for model in runtime.context_map:
        runtime.checked_context(model, row)
    return row


def pregame_player_snapshot(
    runtime: RidgeRuntime,
    player_stats: pd.DataFrame,
    player_id: str,
    kickoff: pd.Timestamp,
    position: str,
) -> pd.DataFrame:
    frame = player_stats[player_stats.player_id.astype(str).eq(str(player_id))].copy()
    frame["game_date"] = pd.to_datetime(frame.game_date, errors="coerce")
    frame = frame[frame.game_date.le(pd.Timestamp(kickoff))]
    columns = list(runtime.required_player_feature_columns)
    if frame.empty:
        history = player_stats.copy()
        history["game_date"] = pd.to_datetime(history.game_date, errors="coerce")
        history = history[history.game_date.le(pd.Timestamp(kickoff))]
        position_rows = history[history.get("position", pd.Series(index=history.index, dtype=object)).astype(str).eq(position)]
        source = position_rows if not position_rows.empty else history
        defaults = source.reindex(columns=columns).mean(numeric_only=True).reindex(columns)
        if defaults.isna().all():
            raise ValueError(f"No pregame player priors available for {position}")
        values = defaults.fillna(0.0).to_dict()
    else:
        values = frame.sort_values(["game_date", "game_id"]).iloc[-1].reindex(columns).to_dict()
        if any(pd.isna(value) for value in values.values()):
            history = player_stats.copy()
            history["game_date"] = pd.to_datetime(history.game_date, errors="coerce")
            history = history[history.game_date.le(pd.Timestamp(kickoff))]
            defaults = history.reindex(columns=columns).mean(numeric_only=True)
            values = {name: values[name] if pd.notna(values[name]) else defaults.get(name, 0.0) for name in columns}
    values = {name: float(value) if pd.notna(value) and np.isfinite(float(value)) else 0.0 for name, value in values.items()}
    return pd.DataFrame([{ "player_id": str(player_id), **values }])


def roster_for_team_week(roster: pd.DataFrame, team: str, week: int) -> pd.DataFrame:
    rows = roster[roster.team.astype(str).eq(str(team))].copy()
    if rows.empty:
        return rows
    rows["week"] = pd.to_numeric(rows.week, errors="coerce")
    weeks = sorted(rows.week.dropna().astype(int).unique())
    prior = [value for value in weeks if value <= int(week)]
    if not prior:
        return rows.iloc[0:0]
    selected_week = max(prior)
    rows = rows[rows.week.eq(selected_week)]
    if "status" in rows:
        active = rows[rows.status.eq("ACT")]
        if not active.empty:
            rows = active
    return rows.dropna(subset=["player_id", "position"])


def _roster_player_name(row) -> str:
    for column in ("football_name", "player_name"):
        value = getattr(row, column, None)
        if value is not None and pd.notna(value) and str(value).strip():
            return str(value)
    return str(row.player_id)


def slot_assignments(
    roster: pd.DataFrame,
    depth: pd.DataFrame,
    team: str,
    week: int,
    kickoff: pd.Timestamp,
    player_stats: pd.DataFrame,
    *,
    use_model_rank: bool = True,
) -> dict[str, dict[str, str]]:
    roster_week = roster_for_team_week(roster, team, week)
    if roster_week.empty:
        raise ValueError(f"No roster records for {team} in or before week {week}")
    candidates = {}
    depth_rows = depth.copy()
    if not depth_rows.empty and "dt" in depth_rows:
        depth_rows["dt"] = pd.to_datetime(depth_rows.dt, errors="coerce", utc=True).dt.tz_convert(None)
        depth_rows = depth_rows[(depth_rows.team.astype(str).eq(str(team))) & depth_rows.dt.le(pd.Timestamp(kickoff) + pd.Timedelta(days=1))]
        if not depth_rows.empty:
            depth_rows = depth_rows[depth_rows.dt.eq(depth_rows.dt.max())]
    for position, limit in SLOT_LIMITS.items():
        available = roster_week[roster_week.position.eq(position)]
        ids = available.player_id.astype(str).tolist()
        if "pos_abb" in depth_rows:
            ranked = depth_rows[depth_rows.pos_abb.eq(position)].copy()
            rank_column = next((name for name in ("depth_chart_position", "pos_rank") if name in ranked), None)
            if rank_column:
                ranked = ranked.sort_values(rank_column)
            ordered_ids = [str(value) for value in ranked.get("gsis_id", pd.Series(dtype=str)).dropna()]
            ids = list(dict.fromkeys([pid for pid in ordered_ids if pid in ids] + ids))
        if use_model_rank and position in SLOT_RANK_STAT:
            stat = SLOT_RANK_STAT[position]
            scored = []
            for offset, player_id in enumerate(ids):
                player_rows = player_stats[player_stats.player_id.astype(str).eq(player_id)].copy()
                player_rows["game_date"] = pd.to_datetime(player_rows.game_date, errors="coerce")
                player_rows = player_rows[player_rows.game_date.le(pd.Timestamp(kickoff))]
                score = float(player_rows.sort_values(["game_date", "game_id"]).iloc[-1].get(stat, 0.0)) if not player_rows.empty else 0.0
                scored.append((-score, offset, player_id))
            ids = [item[2] for item in sorted(scored)]
        candidates[position] = ids[:limit]
    assignments = {}
    player_names = {str(row.player_id): _roster_player_name(row) for row in roster_week.itertuples()}
    for position, limit in SLOT_LIMITS.items():
        ids = candidates[position]
        for index in range(limit):
            slot = f"{position}{index + 1}"
            player_id = ids[index] if index < len(ids) else f"ridge_placeholder_{team}_{week}_{slot}"
            assignments[slot] = {"player_id": player_id, "player_name": player_names.get(player_id, player_id), "position": position}
    return assignments


def make_players(runtime: RidgeRuntime, player_stats: pd.DataFrame, slots: dict[str, dict[str, str]], kickoff: pd.Timestamp) -> list[Player]:
    players = []
    for slot in SKILL_SLOT_ORDER:
        if slot not in slots:
            raise ValueError(f"Missing required simulation slot {slot}")
        item = slots[slot]
        attributes = pregame_player_snapshot(runtime, player_stats, item["player_id"], kickoff, slot[:-1])
        attributes["player_name"] = item["player_name"]
        players.append(Player(item["player_name"], slot, item["player_id"], attributes))
    return players


def available_slot_players(roster: pd.DataFrame, team: str, week: int, depth: pd.DataFrame,
                           kickoff: pd.Timestamp, player_stats: pd.DataFrame,
                           use_model_rank: bool = True) -> dict[str, list[dict[str, str]]]:
    roster_week = roster_for_team_week(roster, team, week)
    if roster_week.empty:
        raise ValueError(f"No roster records for {team} in or before week {week}")
    preferred = slot_assignments(roster, depth, team, week, kickoff, player_stats,
                                 use_model_rank=use_model_rank)
    result = {}
    for position, limit in SLOT_LIMITS.items():
        rows = roster_week[roster_week.position.eq(position)]
        lookup = {}
        for row in rows.itertuples():
            player_id = str(row.player_id)
            lookup[player_id] = _roster_player_name(row)
        ids = list(lookup)
        ordered = []
        for index in range(1, limit + 1):
            candidate = preferred.get(f"{position}{index}", {}).get("player_id")
            if candidate in lookup and candidate not in ordered:
                ordered.append(candidate)
        ordered.extend(player_id for player_id in ids if player_id not in ordered)
        result[position] = [{"player_id": player_id, "player_name": lookup[player_id], "position": position}
                            for player_id in ordered]
    return result


def make_players_from_assignments(runtime: RidgeRuntime, player_stats: pd.DataFrame,
                                  assignments: dict[str, dict[str, str]], kickoff: pd.Timestamp,
                                  overrides: dict[str, dict[str, float]] | None = None) -> list[Player]:
    overrides = overrides or {}
    players = make_players(runtime, player_stats, assignments, kickoff)
    for player in players:
        for column, value in overrides.get(player.player_id, {}).items():
            if column not in runtime.required_player_feature_columns:
                raise ValueError(f"Cannot override non-player model input {column}")
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"Nonfinite player override for {player.player_id}/{column}")
            player.attributes.loc[player.attributes.index[0], column] = numeric
    return players


def actual_roof(game_row: pd.Series) -> str:
    value = game_row.get("roof")
    if value is not None and pd.notna(value):
        text = str(value).strip().lower()
        if text in {"closed", "dome", "indoors", "indoor"}:
            return "dome"
        if text in {"open", "outdoors", "outdoor"}:
            return "outdoors"
        raise ValueError(f"Unknown scheduled roof value: {value}")
    team = str(game_row.get("home_team", ""))
    if team not in ROOF_TYPE:
        raise ValueError(f"No roof mapping for home team {team}")
    return ROOF_TYPE[team]


def rest_before(schedule: pd.DataFrame, team: str, kickoff: pd.Timestamp) -> float:
    dates = pd.to_datetime(schedule.gameday, errors="coerce")
    prior = schedule.loc[
        (schedule.home_team.astype(str).eq(str(team)) | schedule.away_team.astype(str).eq(str(team)))
        & dates.lt(pd.Timestamp(kickoff))
    ]
    if prior.empty:
        return 7.0
    previous = pd.to_datetime(prior.gameday, errors="coerce").max()
    days = int((pd.Timestamp(kickoff).normalize() - previous.normalize()).days)
    return float(min(max(days, 0), MAX_REST_DAYS))


def build_team_pair(runtime, team_stats, game_row, schedule):
    kickoff = pd.Timestamp(game_row.gameday)
    home, away = str(game_row.home_team), str(game_row.away_team)
    game_id = str(game_row.game_id)
    current = team_stats[team_stats.game_id.astype(str).eq(game_id)]
    features = {}
    for team, opponent in ((home, away), (away, home)):
        match = current[current.team.astype(str).eq(team)]
        if len(match) != 1:
            raise ValueError(f"Expected exactly one Ridge pregame team row for {game_id}/{team}; got {len(match)}")
        row = match.iloc[0].reindex(runtime.required_team_feature_columns).copy()
        own_rest, other_rest = rest_before(schedule, team, kickoff), rest_before(schedule, opponent, kickoff)
        if "days_rest" in row:
            row["days_rest"] = own_rest
        if "opp_days_rest" in row:
            row["opp_days_rest"] = other_rest
        for model in runtime.context_map:
            runtime.checked_context(model, row)
        features[team] = row
    return pd.DataFrame.from_dict(features, orient="index").reindex([home, away])


def build_real_game(runtime, player_stats, team_stats, roster, depth, schedule, game_row, simulations,
                    use_model_rank=True, assignments_by_team=None, player_overrides=None):
    from ridge_game_sim.execution import prepared_game

    row = game_row if isinstance(game_row, pd.Series) else pd.Series(game_row._asdict())
    kickoff = pd.Timestamp(row.gameday)
    home, away = str(row.home_team), str(row.away_team)
    team_features = build_team_pair(runtime, team_stats, row, schedule)
    assignments_by_team = assignments_by_team or {}
    players = {}
    for team in (away, home):
        assignments = assignments_by_team.get(team)
        if assignments is None:
            assignments = slot_assignments(roster, depth, team, int(row.week), kickoff, player_stats, use_model_rank=use_model_rank)
        players[team] = make_players_from_assignments(runtime, player_stats, assignments, kickoff,
                                                      (player_overrides or {}).get(team, {}))
    roof = actual_roof(pd.Series(row))
    return prepared_game(row, team_features, players[home], players[away], simulations, roof)
