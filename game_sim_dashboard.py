"""Streamlit dashboard for simulating a single upcoming NFL game with editable inputs.

Pick a week and game, adjust each team's depth chart, override any player's model
inputs, then run N simulations and export fantasy-scored results.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import nfl_data_py as nfl
import numpy as np
import pandas as pd
import streamlit as st

from game_sim_fantasy import (
    fantasy_correlation_matrix,
    latest_player_row,
    prior_fill_values,
)
from game_sim_execution import prepare_game, run_prepared_games
from project_paths import DEFAULT_SIMULATION_OUTPUT_DIR
from torch_game_simulation_runtime import Player, ROOF_TYPE, SKILL_SLOT_ORDER, SimulationRuntime

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError:
    get_script_run_ctx = None

WORKSPACE_DIR = DEFAULT_SIMULATION_OUTPUT_DIR
DEFAULT_SEASON = 2026
SLOT_LIMITS = {"QB": 1, "WR": 4, "RB": 3, "TE": 3}
SKILL_POSITIONS = tuple(SLOT_LIMITS)
ACTIVE_STATUSES = {"ACT"}
TEAM_CODE_ALIASES = {"ARI": ["AZ"]}
MAX_REST_DAYS = 21.0
# Torch Modeling.py ranks WR/TE by `medium` and RB by `rush_rate` when it labels slots for training.
SLOT_RANK_STAT = {"WR": "medium", "TE": "medium", "RB": "rush_rate"}


# ---------------------------------------------------------------- data loading


@st.cache_resource(show_spinner="Loading simulation models...")
def load_runtime() -> SimulationRuntime:
    return SimulationRuntime()


@st.cache_data(show_spinner="Loading schedule...")
def load_schedule(season: int) -> pd.DataFrame:
    schedule = nfl.import_schedules([season])
    schedule = schedule[(schedule["game_type"] == "REG") & schedule["week"].between(1, 18)].copy()
    schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce")
    schedule = schedule.dropna(subset=["gameday", "home_team", "away_team"])
    return schedule.sort_values(["week", "gameday", "game_id"]).reset_index(drop=True)


@st.cache_data(show_spinner="Loading rosters...")
def load_rosters(season: int) -> pd.DataFrame:
    rosters = nfl.import_weekly_rosters([season])
    rosters = rosters[rosters["position"].isin(SKILL_POSITIONS)].copy()
    rosters["player_id"] = rosters["player_id"].astype(str)
    rosters["week"] = pd.to_numeric(rosters["week"], errors="coerce")
    return rosters.dropna(subset=["week"])


@st.cache_data(show_spinner="Loading depth charts...")
def load_depth_charts(season: int) -> pd.DataFrame:
    depth = nfl.import_depth_charts([season]).copy()
    depth["dt"] = pd.to_datetime(depth["dt"], utc=True).dt.tz_convert(None)
    depth = depth[depth["pos_abb"].isin(SKILL_POSITIONS)].copy()
    depth["gsis_id"] = depth["gsis_id"].astype(str)
    return depth


# ------------------------------------------------------------ roster helpers


def roster_for_team_week(rosters: pd.DataFrame, team: str, week: int) -> pd.DataFrame:
    candidates = [str(team)] + TEAM_CODE_ALIASES.get(str(team), [])
    team_rows = rosters[rosters["team"].isin(candidates)]
    if team_rows.empty:
        return team_rows

    available = sorted(team_rows["week"].astype(int).unique().tolist())
    if week in available:
        chosen = week
    else:
        prior = [value for value in available if value <= week]
        chosen = prior[-1] if prior else available[-1]
    return team_rows[team_rows["week"].astype(int) == chosen].copy()


def depth_chart_order(
    depth_charts: pd.DataFrame,
    roster_week: pd.DataFrame,
    team: str,
    gameday: pd.Timestamp,
) -> Dict[str, List[str]]:
    """Active player ids per position, in published depth chart order."""
    if roster_week.empty:
        return {position: [] for position in SLOT_LIMITS}

    active_ids = set(roster_week.loc[roster_week["status"].isin(ACTIVE_STATUSES), "player_id"])
    if not active_ids:
        active_ids = set(roster_week["player_id"])

    depth = depth_charts[
        (depth_charts["team"] == team) & (depth_charts["dt"] <= gameday + pd.Timedelta(days=1))
    ]
    if not depth.empty:
        depth = depth[depth["dt"] == depth["dt"].max()]

    order: Dict[str, List[str]] = {}
    for position in SLOT_LIMITS:
        ranked: List[str] = []
        if not depth.empty:
            ranks = depth[depth["pos_abb"] == position]
            sort_col = next((c for c in ("depth_chart_position", "pos_rank") if c in ranks.columns), None)
            if sort_col is not None:
                ranks = ranks.sort_values(sort_col)
            ranked = [str(value).strip() for value in ranks["gsis_id"]]

        # Roster order backfills anyone the depth chart omits.
        ranked += [pid for pid in roster_week.loc[roster_week["position"] == position, "player_id"] if pid not in ranked]
        order[position] = [pid for pid in dict.fromkeys(ranked) if pid in active_ids]
    return order


@st.cache_data(show_spinner=False)
def load_prior_fill(_player_stats: pd.DataFrame, columns: Tuple[str, ...]) -> Dict[str, float]:
    return prior_fill_values(_player_stats, columns)


def latest_stat_lookup(
    player_stats: pd.DataFrame,
    player_ids: Iterable[str],
    before_date: pd.Timestamp,
    column: str,
    default: float,
) -> Dict[str, float]:
    wanted = {str(pid) for pid in player_ids}
    subset = player_stats[player_stats["player_id"].astype(str).isin(wanted)].copy()
    if subset.empty or column not in subset.columns:
        return {}
    subset["_date"] = pd.to_datetime(subset["game_date"], errors="coerce")
    subset = subset[subset["_date"] <= pd.Timestamp(before_date)].sort_values("_date")
    latest = subset.groupby(subset["player_id"].astype(str)).tail(1)
    return {
        str(pid): float(value) if pd.notna(value) else default
        for pid, value in zip(latest["player_id"].astype(str), latest[column])
    }


def build_slot_assignments(
    order: Dict[str, List[str]],
    player_stats: pd.DataFrame,
    gameday: pd.Timestamp,
    use_model_rank: bool,
    fill_values: Dict[str, float],
) -> Dict[str, str]:
    """Slot name -> player_id, either in depth chart order or the model's training order."""
    assignments: Dict[str, str] = {}
    used: set[str] = set()
    for position, limit in SLOT_LIMITS.items():
        candidates = [pid for pid in order.get(position, []) if pid not in used]
        stat = SLOT_RANK_STAT.get(position) if use_model_rank else None
        if stat:
            default = float(fill_values.get(stat, 0.0))
            scores = latest_stat_lookup(player_stats, candidates, gameday, stat, default)
            depth_rank = {pid: index for index, pid in enumerate(candidates)}
            candidates = sorted(candidates, key=lambda pid: (-scores.get(pid, default), depth_rank[pid]))
        for index, player_id in enumerate(candidates[:limit], start=1):
            assignments[f"{position}{index}"] = player_id
            used.add(player_id)
    return assignments


# ------------------------------------------------------------- team features


def derive_rest_days(team_stats: pd.DataFrame, team: str, gameday: pd.Timestamp) -> float:
    # Strictly earlier: a future-week feature row is dated on kickoff and is not a played game.
    prior = team_stats[(team_stats["team"].astype(str) == str(team)) & (team_stats["game_date"] < gameday)]
    if prior.empty:
        return float("nan")
    last_game = pd.to_datetime(prior["game_date"], errors="coerce").max()
    if pd.isna(last_game):
        return float("nan")
    rest = float((pd.Timestamp(gameday).normalize() - pd.Timestamp(last_game).normalize()).days)
    return min(rest, MAX_REST_DAYS) if rest > 0 else float("nan")


def team_feature_snapshot(team_stats: pd.DataFrame, team: str, gameday: pd.Timestamp) -> pd.Series:
    frame = team_stats[team_stats["team"].astype(str) == str(team)].copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame = frame[frame["game_date"] <= gameday]
    if frame.empty:
        raise ValueError(f"No team features for {team} on or before {pd.Timestamp(gameday).date()}")
    return frame.sort_values(["game_date", "game_id"]).iloc[-1].copy()


def build_team_features(
    runtime: SimulationRuntime,
    team_stats: pd.DataFrame,
    game_row: pd.Series,
) -> pd.DataFrame:
    gameday = pd.Timestamp(game_row["gameday"])
    home_team = str(game_row["home_team"])
    away_team = str(game_row["away_team"])

    features = pd.DataFrame(
        [team_feature_snapshot(team_stats, home_team, gameday), team_feature_snapshot(team_stats, away_team, gameday)]
    ).set_index("team")
    features = features.reindex([home_team, away_team]).reindex(columns=runtime.required_team_feature_columns)

    rest: Dict[str, float] = {}
    for team, schedule_key in ((home_team, "home_rest"), (away_team, "away_rest")):
        history = derive_rest_days(team_stats, team, gameday)
        scheduled = pd.to_numeric(pd.Series([game_row.get(schedule_key, np.nan)]), errors="coerce").iloc[0]
        if pd.notna(history):
            rest[team] = float(min(history, MAX_REST_DAYS))
        elif pd.notna(scheduled):
            rest[team] = float(min(float(scheduled), MAX_REST_DAYS))
        else:
            rest[team] = 7.0

    if "days_rest" in features.columns:
        features.loc[home_team, "days_rest"] = rest[home_team]
        features.loc[away_team, "days_rest"] = rest[away_team]
    if "opp_days_rest" in features.columns:
        features.loc[home_team, "opp_days_rest"] = rest[away_team]
        features.loc[away_team, "opp_days_rest"] = rest[home_team]
    return features


# ------------------------------------------------------------------ UI pieces


def stat_groups(runtime: SimulationRuntime) -> Dict[str, List[str]]:
    return {
        "Passer": list(runtime.passer_columns),
        "Receiver": list(runtime.receiver_columns),
        "Rusher": list(runtime.rusher_columns),
        "QB context": list(runtime.qb_context_columns),
    }


def player_option_ids(roster_week: pd.DataFrame, position: str) -> List[str]:
    subset = roster_week[roster_week["position"] == position]
    active = subset[subset["status"].isin(ACTIVE_STATUSES)]
    if active.empty:
        active = subset
    return active.drop_duplicates(subset=["player_id"])["player_id"].tolist()


def render_depth_chart(team: str, roster_week: pd.DataFrame, defaults: Dict[str, str], key_prefix: str) -> Dict[str, str]:
    name_lookup = dict(zip(roster_week["player_id"], roster_week["player_name"]))
    assignments: Dict[str, str] = {}
    for position, limit in SLOT_LIMITS.items():
        options = player_option_ids(roster_week, position)
        for index in range(1, limit + 1):
            slot = f"{position}{index}"
            default_id = defaults.get(slot)
            slot_options = list(options)
            if default_id and default_id not in slot_options:
                slot_options.insert(0, default_id)
            if not slot_options:
                st.warning(f"{team}: no {position} available for {slot}")
                continue
            default_index = slot_options.index(default_id) if default_id in slot_options else 0
            chosen = st.selectbox(
                slot,
                options=slot_options,
                index=default_index,
                format_func=lambda pid: f"{name_lookup.get(pid, pid)}  ({pid})",
                key=f"{key_prefix}_{slot}",
            )
            assignments[slot] = chosen

    duplicates = [pid for pid, count in pd.Series(list(assignments.values())).value_counts().items() if count > 1]
    if duplicates:
        st.error(f"{team}: {', '.join(name_lookup.get(pid, pid) for pid in duplicates)} assigned to multiple slots.")
    return assignments


def render_player_editor(
    runtime: SimulationRuntime,
    player_stats: pd.DataFrame,
    roster_lookup: Dict[str, str],
    selection: List[Tuple[str, str, str]],
    gameday: pd.Timestamp,
    game_id: str,
    fill_values: Dict[str, float],
) -> None:
    """Edit one player's model inputs; edits only persist once locked in."""
    labels = {f"{team} {slot} - {roster_lookup.get(pid, pid)}": pid for team, slot, pid in selection}
    if not labels:
        return

    chosen_label = st.selectbox("Player", list(labels), key=f"editor_{game_id}")
    player_id = labels[chosen_label]
    overrides: Dict[str, Dict[str, float]] = st.session_state.player_overrides
    locked = overrides.get(player_id, {})

    baseline_row = latest_player_row(
        player_stats, player_id, gameday, runtime.required_player_feature_columns, fill_values
    )
    baseline = {
        column: float(baseline_row[column].iloc[0])
        for columns in stat_groups(runtime).values()
        for column in columns
    }
    display = pd.DataFrame(
        [
            {"group": group, "stat": column, "value": float(locked.get(column, baseline[column]))}
            for group, columns in stat_groups(runtime).items()
            for column in columns
        ]
    )

    # Bumped on every commit so the editor reseeds instead of replaying stale cell edits.
    version = st.session_state.editor_version.get(player_id, 0)
    edited = st.data_editor(
        display,
        key=f"stats_{game_id}_{player_id}_{version}",
        width="stretch",
        hide_index=True,
        disabled=["group", "stat"],
        column_config={"value": st.column_config.NumberColumn("value", format="%.4f")},
    )

    lock_column, reset_column, status_column = st.columns([1, 1, 4])
    if lock_column.button("Lock in stats", key=f"lock_{game_id}_{player_id}_{version}"):
        committed = {
            str(stat): float(value)
            for stat, value in zip(edited["stat"], edited["value"])
            if pd.notna(value) and not np.isclose(float(value), baseline[str(stat)])
        }
        if committed:
            overrides[player_id] = committed
        else:
            overrides.pop(player_id, None)
        st.session_state.editor_version[player_id] = version + 1
        st.rerun()

    if reset_column.button("Reset player", key=f"reset_{game_id}_{player_id}_{version}", disabled=not locked):
        overrides.pop(player_id, None)
        st.session_state.editor_version[player_id] = version + 1
        st.rerun()

    if locked:
        status_column.success(f"{len(locked)} locked: {', '.join(sorted(locked))}")
    else:
        status_column.caption("Edits are ignored until you lock them in.")

    if overrides:
        summary = pd.DataFrame(
            [
                {"player": roster_lookup.get(pid, pid), "stat": stat, "value": value}
                for pid, values in overrides.items()
                for stat, value in sorted(values.items())
            ]
        )
        with st.expander(f"Locked overrides this session ({len(overrides)} players)"):
            st.dataframe(summary, width="stretch", hide_index=True)


# ------------------------------------------------------------------ simulation


def build_players(
    runtime: SimulationRuntime,
    player_stats: pd.DataFrame,
    assignments: Dict[str, str],
    name_lookup: Dict[str, str],
    gameday: pd.Timestamp,
    overrides: Dict[str, Dict[str, float]],
    fill_values: Dict[str, float],
) -> List[Player]:
    players: List[Player] = []
    for slot, player_id in assignments.items():
        attributes = latest_player_row(
            player_stats, player_id, gameday, runtime.required_player_feature_columns, fill_values
        )
        for column, value in overrides.get(player_id, {}).items():
            attributes[column] = float(value)
        players.append(
            Player(
                name=str(name_lookup.get(player_id, player_id)),
                position=slot,
                player_id=str(player_id),
                attributes=attributes,
            )
        )
    return players


def run_simulations(
    runtime: SimulationRuntime,
    game_row: pd.Series,
    team_features: pd.DataFrame,
    home_players: List[Player],
    away_players: List[Player],
    simulations: int,
    workers: int = 1,
    backend: str = "Local",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    home_team = str(game_row["home_team"])
    away_team = str(game_row["away_team"])
    roof = ROOF_TYPE.get(home_team, "outdoors")

    progress = st.progress(0.0, text=f"Simulating {away_team} @ {home_team}...")

    prepared = prepare_game(
        runtime,
        game_row,
        team_features,
        home_players,
        away_players,
        simulations,
        roof,
    )

    def note(completed: int, total: int) -> None:
        progress.progress(completed / total, text=f"Simulation {completed} of {total}")

    try:
        results = run_prepared_games(
            [prepared],
            backend,
            workers,
            runtime.data_dir,
            progress=note,
            modal_app_name=os.environ.get("MODAL_GAME_SIM_APP", "nfl-game-sim"),
            modal_environment=os.environ.get("MODAL_ENVIRONMENT"),
        )
        return results[prepared.game_id]
    finally:
        progress.empty()


def summarize_games(game_frame: pd.DataFrame) -> pd.DataFrame:
    summaries = [_summarize_one_game(group) for _, group in game_frame.groupby("game_id", sort=False)]
    return pd.DataFrame(summaries)


def _summarize_one_game(group: pd.DataFrame) -> Dict[str, object]:
    home = group["home_score"].to_numpy(dtype=float)
    away = group["away_score"].to_numpy(dtype=float)
    spreads = home - away
    totals = home + away
    decisive = max(float(np.mean(home != away)), 1e-9)
    summary: Dict[str, object] = {
        "game_id": group["game_id"].iloc[0],
        "week": group["week"].iloc[0],
        "home_team": group["home_team"].iloc[0],
        "away_team": group["away_team"].iloc[0],
        "simulations_run": len(group),
        "sim_home_median": float(np.median(home)),
        "sim_away_median": float(np.median(away)),
        "sim_spread": float(np.median(spreads)),
        "sim_total": float(np.median(totals)),
        "sim_home_win_prob": float(np.mean(home > away) / decisive),
        "sim_spread_stdev": float(np.std(spreads)),
        "sim_total_stdev": float(np.std(totals)),
    }
    for label, values in (("spread", spreads), ("total", totals)):
        for percentile in (10, 25, 75, 90):
            summary[f"sim_{label}_p{percentile}"] = float(np.percentile(values, percentile))
    return summary


def merge_with_existing(
    target_dir: Path,
    stem: str,
    game_frame: pd.DataFrame,
    player_frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Continue simulation numbering from whatever is already on disk."""
    games_path = target_dir / f"{stem}_games.csv"
    players_path = target_dir / f"{stem}_players.csv"
    if not games_path.exists() or not players_path.exists():
        return game_frame, player_frame, 0

    existing_games = pd.read_csv(games_path)
    existing_players = pd.read_csv(players_path)
    offset = int(pd.to_numeric(existing_games["simulation"], errors="coerce").max())
    game_frame = game_frame.assign(simulation=game_frame["simulation"] + offset)
    player_frame = player_frame.assign(simulation=player_frame["simulation"] + offset)
    return (
        pd.concat([existing_games, game_frame], ignore_index=True),
        pd.concat([existing_players, player_frame], ignore_index=True),
        offset,
    )


def sanitize_stem(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", stem.strip())
    return cleaned or "game_simulation"


# ------------------------------------------------------------------- app body


def simulate_week(
    runtime: SimulationRuntime,
    week_games: pd.DataFrame,
    team_stats: pd.DataFrame,
    player_stats: pd.DataFrame,
    rosters: pd.DataFrame,
    depth_charts: pd.DataFrame,
    fill_values: Dict[str, float],
    use_model_rank: bool,
    week: int,
    simulations: int,
    workers: int,
    target_dir: Path,
    backend: str = "Local",
) -> None:
    """Simulate every game on the week's schedule with default depth charts, no player overrides."""
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        st.error(f"Cannot use output folder: {error}")
        return

    total_games = len(week_games)
    overall_progress = st.progress(0.0, text="Starting week simulation...")
    prepared_games = []
    failures: List[str] = []

    for position, (_, game_row) in enumerate(week_games.iterrows(), start=1):
        game_id = str(game_row["game_id"])
        home_team = str(game_row["home_team"])
        away_team = str(game_row["away_team"])
        overall_progress.progress(
            0.0,
            text=f"Preparing game {position}/{total_games}: {away_team} @ {home_team}",
        )

        try:
            team_features = build_team_features(runtime, team_stats, game_row)
        except ValueError as error:
            failures.append(f"{game_id}: {error}")
            continue

        gameday = pd.Timestamp(game_row["gameday"])
        assignments: Dict[str, Dict[str, str]] = {}
        roster_weeks: Dict[str, pd.DataFrame] = {}
        missing_roster = False
        for team in (away_team, home_team):
            roster_week = roster_for_team_week(rosters, team, week)
            if roster_week.empty:
                failures.append(f"{game_id}: no roster rows for {team}")
                missing_roster = True
                break
            roster_weeks[team] = roster_week
            order = depth_chart_order(depth_charts, roster_week, team, gameday)
            assignments[team] = build_slot_assignments(
                order, player_stats, gameday, use_model_rank, fill_values
            )
        if missing_roster:
            continue

        name_lookup: Dict[str, str] = {}
        for roster_week in roster_weeks.values():
            name_lookup.update(dict(zip(roster_week["player_id"], roster_week["player_name"])))

        home_players = build_players(
            runtime, player_stats, assignments[home_team], name_lookup, gameday, {}, fill_values
        )
        away_players = build_players(
            runtime, player_stats, assignments[away_team], name_lookup, gameday, {}, fill_values
        )
        prepared_games.append(
            prepare_game(
                runtime,
                game_row,
                team_features,
                home_players,
                away_players,
                simulations,
                ROOF_TYPE.get(home_team, "outdoors"),
            )
        )

    if failures:
        st.warning("Skipped games:\n" + "\n".join(failures))
    if not prepared_games:
        overall_progress.empty()
        st.error("No games were simulated.")
        return

    def note(completed: int, total: int) -> None:
        overall_progress.progress(
            completed / total,
            text=f"Completed {completed} of {total} simulations across {len(prepared_games)} games",
        )

    try:
        results = run_prepared_games(
            prepared_games,
            backend,
            workers,
            runtime.data_dir,
            progress=note,
            modal_app_name=os.environ.get("MODAL_GAME_SIM_APP", "nfl-game-sim"),
            modal_environment=os.environ.get("MODAL_ENVIRONMENT"),
        )
    except Exception as error:
        st.error(f"{backend} simulation failed: {error}")
        return
    finally:
        overall_progress.empty()

    game_frames: List[pd.DataFrame] = []
    player_frames: List[pd.DataFrame] = []
    for prepared in prepared_games:
        game_frame, player_frame = results[prepared.game_id]
        stem = sanitize_stem(prepared.game_id)
        game_frame.to_csv(target_dir / f"{stem}_games.csv", index=False)
        player_frame.to_csv(target_dir / f"{stem}_players.csv", index=False)
        game_frames.append(game_frame)
        player_frames.append(player_frame)

    combined_games = pd.concat(game_frames, ignore_index=True)
    combined_players = pd.concat(player_frames, ignore_index=True)
    combined_summary = summarize_games(combined_games)

    week_stem = f"week_{week:02d}"
    combined_games.to_csv(target_dir / f"{week_stem}_games.csv", index=False)
    combined_players.to_csv(target_dir / f"{week_stem}_players.csv", index=False)
    combined_summary.to_csv(target_dir / f"{week_stem}_summary.csv", index=False)

    st.success(f"Simulated {len(game_frames)}/{total_games} games. Wrote combined CSVs to {target_dir}")
    st.dataframe(combined_summary, width="stretch")


def build_synthetic_team_features(runtime: SimulationRuntime, rest_days: float = 7.0) -> pd.Series:
    """Team features set to replacement-level values for all required inputs."""
    cols = [c for c in runtime.required_team_feature_columns if c in runtime.team_stats_wide.columns]
    syn_features = runtime.team_stats_wide[cols].mean(numeric_only=True).copy()
    syn_features["days_rest"] = float(rest_days)
    syn_features["opp_days_rest"] = float(rest_days)
    for col in runtime.required_team_feature_columns:
        if col not in syn_features.index or pd.isna(syn_features[col]):
            syn_features[col] = 0.0
    return syn_features.reindex(runtime.required_team_feature_columns)


def build_synthetic_players(runtime: SimulationRuntime) -> List[Player]:
    """Build 11 skill players (QB1, WR1-4, RB1-3, TE1-3) with replacement-level attributes."""
    syn_player_means = runtime.player_stats_wide[runtime.required_player_feature_columns].mean(numeric_only=True)
    syn_player_df = pd.DataFrame([syn_player_means.to_dict()])
    for col in runtime.required_player_feature_columns:
        if col not in syn_player_df.columns or pd.isna(syn_player_df[col].iloc[0]):
            syn_player_df[col] = 0.0

    return [
        Player(
            name=f"Synthetic {slot}",
            position=slot,
            player_id=f"SYN_{slot}",
            attributes=syn_player_df.copy(),
        )
        for slot in SKILL_SLOT_ORDER
    ]


def simulate_synthetic_home_games(
    runtime: SimulationRuntime,
    season: int,
    week: int,
    schedule: pd.DataFrame,
    team_stats: pd.DataFrame,
    player_stats: pd.DataFrame,
    rosters: pd.DataFrame,
    depth_charts: pd.DataFrame,
    fill_values: Dict[str, float],
    use_model_rank: bool,
    simulations: int,
    workers: int,
    target_dir: Path,
    backend: str = "Local",
    teams: Optional[List[str]] = None,
) -> None:
    """Simulate all teams at home against a synthetic opponent with replacement-level inputs."""
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        st.error(f"Cannot use output folder: {error}")
        return

    if teams is None:
        all_teams = sorted(set(schedule["home_team"].dropna().unique().tolist()))
        if not all_teams:
            all_teams = sorted(set(team_stats["team"].dropna().unique().tolist()))
    else:
        all_teams = list(teams)

    total_teams = len(all_teams)
    overall_progress = st.progress(0.0, text="Starting synthetic opponent home simulations...")
    prepared_games = []
    failures: List[str] = []

    syn_team_features = build_synthetic_team_features(runtime, rest_days=7.0)
    syn_players = build_synthetic_players(runtime)

    week_sched = schedule[schedule["week"].astype(int) == int(week)]
    default_gameday = (
        pd.Timestamp(week_sched["gameday"].max())
        if not week_sched.empty
        else pd.Timestamp(f"{season}-09-15")
    )

    for index, home_team in enumerate(all_teams, start=1):
        overall_progress.progress(
            0.0,
            text=f"Preparing team {index}/{total_teams}: SYN @ {home_team}",
        )
        game_id = f"{season}_{week:02d}_SYN_{home_team}"

        matching_games = week_sched[
            (week_sched["home_team"] == home_team) | (week_sched["away_team"] == home_team)
        ]
        gameday = (
            pd.Timestamp(matching_games.iloc[0]["gameday"])
            if not matching_games.empty
            else default_gameday
        )

        try:
            home_feature_row = team_feature_snapshot(team_stats, home_team, gameday).copy()
        except ValueError as error:
            failures.append(f"{home_team}: {error}")
            continue

        history_rest = derive_rest_days(team_stats, home_team, gameday)
        home_rest = min(history_rest, MAX_REST_DAYS) if pd.notna(history_rest) else 7.0
        home_feature_row["days_rest"] = home_rest
        home_feature_row["opp_days_rest"] = 7.0

        team_features = pd.DataFrame(
            [home_feature_row, syn_team_features], index=[home_team, "SYN"]
        ).reindex(columns=runtime.required_team_feature_columns)

        roster_week = roster_for_team_week(rosters, home_team, week)
        if roster_week.empty:
            failures.append(f"{home_team}: no roster rows found for week {week}")
            continue

        name_lookup = dict(zip(roster_week["player_id"], roster_week["player_name"]))
        order = depth_chart_order(depth_charts, roster_week, home_team, gameday)
        assignments = build_slot_assignments(order, player_stats, gameday, use_model_rank, fill_values)
        home_players = build_players(
            runtime, player_stats, assignments, name_lookup, gameday, {}, fill_values
        )

        game_row = pd.Series(
            {
                "game_id": game_id,
                "week": int(week),
                "home_team": home_team,
                "away_team": "SYN",
                "gameday": gameday,
            }
        )
        prepared_games.append(
            prepare_game(
                runtime,
                game_row,
                team_features,
                home_players,
                syn_players,
                simulations,
                ROOF_TYPE.get(home_team, "outdoors"),
            )
        )

    if failures:
        st.warning("Skipped teams:\n" + "\n".join(failures))
    if not prepared_games:
        overall_progress.empty()
        st.error("No games were simulated.")
        return

    def note(completed: int, total: int) -> None:
        overall_progress.progress(
            completed / total,
            text=f"Completed {completed} of {total} simulations across {len(prepared_games)} teams",
        )

    try:
        results = run_prepared_games(
            prepared_games,
            backend,
            workers,
            runtime.data_dir,
            progress=note,
            modal_app_name=os.environ.get("MODAL_GAME_SIM_APP", "nfl-game-sim"),
            modal_environment=os.environ.get("MODAL_ENVIRONMENT"),
        )
    except Exception as error:
        st.error(f"{backend} simulation failed: {error}")
        return
    finally:
        overall_progress.empty()

    game_frames: List[pd.DataFrame] = []
    player_frames: List[pd.DataFrame] = []
    for prepared in prepared_games:
        game_frame, player_frame = results[prepared.game_id]
        stem = sanitize_stem(prepared.game_id)
        game_frame.to_csv(target_dir / f"{stem}_games.csv", index=False)
        player_frame.to_csv(target_dir / f"{stem}_players.csv", index=False)
        game_frames.append(game_frame)
        player_frames.append(player_frame)

    combined_games = pd.concat(game_frames, ignore_index=True)
    combined_players = pd.concat(player_frames, ignore_index=True)
    combined_summary = summarize_games(combined_games).sort_values(
        ["sim_spread", "sim_home_win_prob"], ascending=False
    ).reset_index(drop=True)

    week_stem = f"synthetic_home_week_{week:02d}"
    combined_games.to_csv(target_dir / f"{week_stem}_games.csv", index=False)
    combined_players.to_csv(target_dir / f"{week_stem}_players.csv", index=False)
    combined_summary.to_csv(target_dir / f"{week_stem}_summary.csv", index=False)
    combined_summary.to_csv(target_dir / "synthetic_home_summary.csv", index=False)

    st.success(
        f"Simulated {len(game_frames)}/{total_teams} teams at home vs. synthetic opponent. "
        f"Wrote combined CSVs to {target_dir}"
    )
    st.dataframe(combined_summary, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Weekly Game Simulator", layout="wide")
    st.title("Weekly Game Simulator")

    st.session_state.setdefault("player_overrides", {})
    st.session_state.setdefault("editor_version", {})

    runtime = load_runtime()
    if runtime.schema_issues:
        with st.expander(f"{len(runtime.schema_issues)} model schema warnings"):
            st.write(runtime.schema_issues)

    season = st.sidebar.number_input("Season", min_value=2016, max_value=2100, value=DEFAULT_SEASON, step=1)
    schedule = load_schedule(int(season))
    if schedule.empty:
        st.error(f"No regular season schedule available for {int(season)}.")
        return

    weeks = sorted(schedule["week"].astype(int).unique().tolist())
    week = st.sidebar.selectbox("Week", weeks, key="week_select")
    week_games = schedule[schedule["week"].astype(int) == int(week)].reset_index(drop=True)
    game_labels = {
        f"{row.away_team} @ {row.home_team}  ({pd.Timestamp(row.gameday).date()})": row.game_id
        for row in week_games.itertuples()
    }
    game_label = st.sidebar.selectbox("Game", list(game_labels), key=f"game_select_{week}")
    game_id = game_labels[game_label]
    game_row = week_games[week_games["game_id"] == game_id].iloc[0]

    if st.sidebar.button("Clear stat overrides"):
        st.session_state.player_overrides = {}
        st.session_state.editor_version = {}
        st.rerun()

    ordering = st.sidebar.radio(
        "Slot ordering",
        ["Model rank (matches training)", "Depth chart"],
        help=(
            "The target and rusher models were trained with WR/TE slots ranked by `medium` target share "
            "and RB slots by `rush_rate`. Depth chart order feeds those slots a different meaning."
        ),
    )
    use_model_rank = ordering.startswith("Model rank")

    gameday = pd.Timestamp(game_row["gameday"])
    home_team = str(game_row["home_team"])
    away_team = str(game_row["away_team"])

    player_stats = runtime.player_stats_wide
    team_stats = runtime.team_stats_wide.copy()
    team_stats["game_date"] = pd.to_datetime(team_stats["game_date"], errors="coerce")
    fill_values = load_prior_fill(player_stats, tuple(runtime.required_player_feature_columns))

    if not (team_stats["game_id"] == game_id).any():
        st.warning(
            f"No feature row for {game_id} yet, so inputs fall back to each side's last played game and "
            "exclude it. Rebuild with `python bayesian_feature_builder.py --future-weeks 1 --write-wide`, "
            "then restart the app."
        )

    try:
        team_features = build_team_features(runtime, team_stats, game_row)
    except ValueError as error:
        st.error(str(error))
        return

    rosters = load_rosters(int(season))
    depth_charts = load_depth_charts(int(season))

    st.subheader("Depth charts")
    columns = st.columns(2)
    assignments: Dict[str, Dict[str, str]] = {}
    roster_weeks: Dict[str, pd.DataFrame] = {}
    for column, team in zip(columns, (away_team, home_team)):
        with column:
            st.markdown(f"**{team}** ({'away' if team == away_team else 'home'})")
            roster_week = roster_for_team_week(rosters, team, int(week))
            if roster_week.empty:
                st.error(f"No roster rows for {team}.")
                return
            roster_weeks[team] = roster_week
            order = depth_chart_order(depth_charts, roster_week, team, gameday)
            defaults = build_slot_assignments(order, player_stats, gameday, use_model_rank, fill_values)
            key_prefix = f"slot_{game_id}_{team}_{'rank' if use_model_rank else 'depth'}"
            assignments[team] = render_depth_chart(team, roster_week, defaults, key_prefix)

    name_lookup: Dict[str, str] = {}
    for roster_week in roster_weeks.values():
        name_lookup.update(dict(zip(roster_week["player_id"], roster_week["player_name"])))

    st.subheader("Player inputs")
    selection = [(team, slot, pid) for team, slots in assignments.items() for slot, pid in slots.items()]
    render_player_editor(runtime, player_stats, name_lookup, selection, gameday, str(game_id), fill_values)

    st.subheader("Run")
    run_columns = st.columns([1, 1, 1, 2, 2])
    simulations = run_columns[0].number_input("Simulations", min_value=1, max_value=10000, value=200, step=25)
    backend = run_columns[1].selectbox("Backend", ["Local", "Modal"], key="single_backend")
    parallel_label = "Workers" if backend == "Local" else "Max containers"
    workers = run_columns[2].number_input(
        parallel_label,
        min_value=1,
        max_value=6 if backend == "Local" else 32,
        value=1 if backend == "Local" else 8,
        step=1,
        key=f"single_parallelism_{backend.lower()}",
    )
    output_dir = run_columns[3].text_input("Output folder", value=str(WORKSPACE_DIR))
    stem = run_columns[4].text_input("File name", value=f"{game_id}_sim")
    append = st.checkbox("Append to existing files", value=False)
    st.caption(
        "Local workers and Modal containers each load one model runtime. Modal results return here and are written locally."
    )

    st.subheader("Simulate entire week")
    week_columns = st.columns([1, 1, 1, 2])
    week_simulations = week_columns[0].number_input(
        "Simulations per game", min_value=1, max_value=10000, value=200, step=25, key="week_simulations"
    )
    week_backend = week_columns[1].selectbox("Backend", ["Local", "Modal"], key="week_backend")
    week_parallel_label = "Workers" if week_backend == "Local" else "Max containers"
    week_workers = week_columns[2].number_input(
        week_parallel_label,
        min_value=1,
        max_value=6 if week_backend == "Local" else 32,
        value=1 if week_backend == "Local" else 16,
        step=1,
        key=f"week_parallelism_{week_backend.lower()}",
    )
    week_output_dir = week_columns[3].text_input(
        "Week output folder", value=str(WORKSPACE_DIR / f"week_{int(week):02d}"), key="week_output_dir"
    )
    st.caption(
        f"Runs all {len(week_games)} games on week {int(week)} with default depth charts (no per-player "
        "overrides) and writes per-game plus combined CSVs to the folder above."
    )
    if st.button("Simulate all games this week", type="primary"):
        simulate_week(
            runtime,
            week_games,
            team_stats,
            player_stats,
            rosters,
            depth_charts,
            fill_values,
            use_model_rank,
            int(week),
            int(week_simulations),
            int(week_workers),
            Path(week_output_dir).expanduser(),
            week_backend,
        )

    st.subheader("Simulate all teams vs. synthetic opponent (Home)")
    syn_columns = st.columns([1, 1, 1, 2])
    syn_simulations = syn_columns[0].number_input(
        "Simulations per team", min_value=1, max_value=10000, value=200, step=25, key="syn_simulations"
    )
    syn_backend = syn_columns[1].selectbox("Backend", ["Local", "Modal"], key="syn_backend")
    syn_parallel_label = "Workers" if syn_backend == "Local" else "Max containers"
    syn_workers = syn_columns[2].number_input(
        syn_parallel_label,
        min_value=1,
        max_value=6 if syn_backend == "Local" else 32,
        value=1 if syn_backend == "Local" else 16,
        step=1,
        key=f"syn_parallelism_{syn_backend.lower()}",
    )
    syn_output_dir = syn_columns[3].text_input(
        "Synthetic output folder",
        value=str(WORKSPACE_DIR / f"synthetic_home_week_{int(week):02d}"),
        key="syn_output_dir",
    )
    st.caption(
        "Simulates all 32 NFL teams hosting a synthetic opponent with replacement-level values for all team "
        "and player inputs in their home stadiums, isolating true team strength and player outputs."
    )
    if st.button("Simulate all teams at home vs. synthetic opponent", type="primary", key="btn_syn_all"):
        simulate_synthetic_home_games(
            runtime,
            int(season),
            int(week),
            schedule,
            team_stats,
            player_stats,
            rosters,
            depth_charts,
            fill_values,
            use_model_rank,
            int(syn_simulations),
            int(syn_workers),
            Path(syn_output_dir).expanduser(),
            syn_backend,
        )

    if not st.button("Run simulations", type="primary"):
        return

    target_dir = Path(output_dir).expanduser()
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        st.error(f"Cannot use output folder: {error}")
        return

    overrides = st.session_state.player_overrides
    home_players = build_players(
        runtime, player_stats, assignments[home_team], name_lookup, gameday, overrides, fill_values
    )
    away_players = build_players(
        runtime, player_stats, assignments[away_team], name_lookup, gameday, overrides, fill_values
    )

    try:
        game_frame, player_frame = run_simulations(
            runtime,
            game_row,
            team_features,
            home_players,
            away_players,
            int(simulations),
            int(workers),
            backend,
        )
    except Exception as error:
        st.error(f"{backend} simulation failed: {error}")
        return

    safe_stem = sanitize_stem(stem)
    if append:
        game_frame, player_frame, offset = merge_with_existing(target_dir, safe_stem, game_frame, player_frame)
        if offset:
            st.info(f"Appended {int(simulations)} simulations after {offset} existing ones.")

    summary_frame = summarize_games(game_frame)
    correlation = fantasy_correlation_matrix(player_frame)

    outputs = {
        f"{safe_stem}_players.csv": player_frame,
        f"{safe_stem}_games.csv": game_frame,
        f"{safe_stem}_summary.csv": summary_frame,
        f"{safe_stem}_fantasy_correlation.csv": correlation,
    }
    for filename, frame in outputs.items():
        frame.to_csv(target_dir / filename, index=filename.endswith("_fantasy_correlation.csv"))

    st.success(f"Wrote {len(outputs)} files to {target_dir}")
    st.dataframe(summary_frame, width="stretch")

    st.markdown("**Fantasy points by player**")
    fantasy = (
        player_frame.groupby(["team", "position", "player_name"])["fantasy_points"]
        .agg(mean="mean", median="median", p90=lambda s: s.quantile(0.9))
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    st.dataframe(fantasy, width="stretch")


if __name__ == "__main__":
    if get_script_run_ctx is None or get_script_run_ctx() is None:
        os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())])
    main()
