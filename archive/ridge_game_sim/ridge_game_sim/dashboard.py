from __future__ import annotations

import os
import sys
import json
from datetime import datetime
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd
import streamlit as st

from game_sim_fantasy import fantasy_correlation_matrix
from project_paths import DEFAULT_SIMULATION_OUTPUT_DIR
from ridge_game_sim.execution import execute_ridge_games
from ridge_game_sim.game_inputs import (available_slot_players, build_real_game, load_runtime,
                                        pregame_player_snapshot, SLOT_LIMITS)
from ridge_game_sim.paths import ROOT, digest
from ridge_game_sim.synthetic_games import (build_synthetic_game, _load_inputs,
                                            synthetic_game_row, synthetic_team_options)
from ridge_game_sim.runtime import RidgeRuntime
from ridge_game_sim.bundle import verify_bundle_hash
from ridge_game_sim.weekly_refresh import refresh_weekly_bundle


@st.cache_resource(show_spinner="Validating and loading Ridge model bundle...")
def cached_runtime(artifact_dir: str, bundle_hash: str) -> RidgeRuntime:
    if verify_bundle_hash(Path(artifact_dir)) != bundle_hash:
        raise ValueError("Ridge bundle changed while dashboard was open; restart the app")
    return load_runtime(Path(artifact_dir))


@st.cache_data(show_spinner="Loading Ridge feature tables...")
def cached_tables(artifact_dir: str, feature_hash: str):
    root = Path(artifact_dir)
    players = pd.read_csv(root / "player_stats_wide.csv", dtype={"player_id": str, "game_id": str})
    teams = pd.read_csv(root / "team_stats_wide.csv", dtype={"team": str, "game_id": str})
    return players, teams


def _format_result(frames):
    games = pd.concat([pair[0] for pair in frames.values()], ignore_index=True)
    players = pd.concat([pair[1] for pair in frames.values()], ignore_index=True)
    games = games.assign(home_win=games.home_score.gt(games.away_score).astype(float))
    summary = games.groupby(["game_id", "week", "home_team", "away_team"], as_index=False).agg(
        simulations=("simulation", "count"), home_median=("home_score", "median"),
        away_median=("away_score", "median"), spread_median=("spread", "median"),
        total_median=("total", "median"), home_win_probability=("home_win", "mean"),
    )
    return games, players, summary


def default_save_folder(mode: str, week: int) -> Path:
    if mode == "Synthetic opponents":
        return DEFAULT_SIMULATION_OUTPUT_DIR / f"synthetic_home_week_{int(week):02d}"
    return ROOT / "outputs"


def output_run_directory(save_folder: str | Path, run_id: str) -> Path:
    name = str(run_id).strip()
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Run ID must be a single folder name without path separators")
    destination = Path(save_folder).expanduser().resolve() / name
    if destination.exists():
        raise FileExistsError(f"Output folder already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def lineup_controls(runtime, player_stats, roster, depth, team, week, kickoff, use_model_rank, key_prefix):
    candidates = available_slot_players(roster, team, week, depth, kickoff, player_stats,
                                        use_model_rank=use_model_rank)
    assignments = {}
    st.markdown(f"**{team} lineup**")
    for position, limit in SLOT_LIMITS.items():
        rows = candidates[position]
        used = set()
        columns = st.columns(limit)
        for index in range(limit):
            slot = f"{position}{index + 1}"
            options = [row["player_id"] for row in rows if row["player_id"] not in used]
            if not options:
                options = [f"ridge_placeholder_{team}_{week}_{slot}"]
                labels = {options[0]: f"Replacement {slot}"}
            else:
                labels = {row["player_id"]: f"{row['player_name']} ({row['player_id']})" for row in rows if row["player_id"] in options}
            default = next((row["player_id"] for row in rows if row["player_id"] in options), options[0])
            selected = columns[index].selectbox(
                slot, options, index=options.index(default), format_func=lambda value, names=labels: names.get(value, value),
                key=f"{key_prefix}_{team}_{slot}",
            )
            used.add(selected)
            assignments[slot] = {"player_id": selected, "player_name": labels.get(selected, selected), "position": position}
    return assignments


def player_stat_editor(runtime, player_stats, assignments_by_team, kickoff, editor_key):
    choices = [(team, slot, item["player_id"], item["player_name"])
               for team, slots in assignments_by_team.items() for slot, item in slots.items()]
    labels = {f"{team} {slot}: {name}": (team, slot, player_id)
              for team, slot, player_id, name in choices}
    selected = st.selectbox("Player inputs to edit", list(labels), key=f"{editor_key}_player")
    team, slot, player_id = labels[selected]
    position = slot[:-1]
    baseline = pregame_player_snapshot(runtime, player_stats, player_id, kickoff, position)
    overrides = st.session_state.setdefault("ridge_player_overrides", {})
    identity = f"{editor_key}:{team}:{player_id}"
    locked = overrides.get(identity, {})
    columns = list(runtime.required_player_feature_columns)
    frame = pd.DataFrame([{"stat": column, "value": float(locked.get(column, baseline.iloc[0].get(column, 0.0)))} for column in columns])
    version = st.session_state.setdefault("ridge_editor_version", {}).get(identity, 0)
    edited = st.data_editor(frame, key=f"{editor_key}_{player_id}_{version}", hide_index=True, disabled=["stat"],
                            column_config={"value": st.column_config.NumberColumn("value", format="%.4f")})
    apply_col, reset_col = st.columns(2)
    if apply_col.button("Apply these inputs", key=f"apply_{identity}_{version}"):
        updates = {str(name): float(value) for name, value in zip(edited.stat, edited.value)
                   if pd.notna(value) and not np.isclose(float(value), float(baseline.iloc[0].get(name, 0.0)))}
        if updates:
            overrides[identity] = updates
        else:
            overrides.pop(identity, None)
        st.session_state.ridge_editor_version[identity] = version + 1
        st.rerun()
    if reset_col.button("Reset player inputs", key=f"reset_{identity}_{version}", disabled=not bool(locked)):
        overrides.pop(identity, None)
        st.session_state.ridge_editor_version[identity] = version + 1
        st.rerun()
    return {team: {player_id: locked} } if locked else {}


def available_bundles() -> list[Path]:
    base = ROOT / "artifacts/runtime"
    versions = sorted((ROOT / "artifacts/bundles").glob("*"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [*(path for path in versions if (path / "bundle_manifest.json").is_file()), base]


def bundle_label(path: str | Path) -> str:
    path = Path(path)
    if path.resolve() == (ROOT / "artifacts/runtime").resolve():
        return "Base trained bundle"
    return f"Weekly Ridge bundle {path.name[:12]}"


def apply_pending_bundle(session_state, options=None):
    """Apply a refreshed bundle selection before its selector is instantiated."""
    pending = session_state.pop("ridge_bundle_path_pending", None)
    if pending is not None:
        if options is None or str(pending) in {str(Path(option).resolve()) for option in options}:
            session_state["ridge_bundle_choice"] = str(Path(pending).resolve())
    return pending


def main():
    st.set_page_config(page_title="Ridge Game Simulator", layout="wide")
    st.title("Ridge Game Simulator")
    st.caption("Separate Ridge-trained models, features and outputs. The Bayesian simulator remains unchanged.")
    bundle_options = [str(path.resolve()) for path in available_bundles()]
    apply_pending_bundle(st.session_state, bundle_options)
    current_choice = st.session_state.get("ridge_bundle_choice")
    if current_choice not in bundle_options:
        current_choice = bundle_options[0]
    artifact_text = st.sidebar.selectbox(
        "Ridge bundle", bundle_options, index=bundle_options.index(current_choice),
        format_func=bundle_label, key="ridge_bundle_choice",
    )
    artifact_dir = Path(artifact_text).expanduser().resolve()
    if not artifact_dir.is_dir():
        st.error(f"Ridge bundle directory not found: {artifact_dir}")
        st.stop()
    bundle_hash = verify_bundle_hash(artifact_dir)
    runtime = cached_runtime(str(artifact_dir), bundle_hash)
    st.warning("QB team-start count is retained from the trained feature contract. Its historical training-row player association was selected using same-game pass attempts; a pregame-only corrected fit is a planned validation item.")
    with st.sidebar.expander("Weekly data refresh", expanded=False):
        refresh_season = datetime.now().year
        st.caption(f"Active data season: {refresh_season}. Historical seasons are kept frozen.")
        st.caption("Downloads the active season, preserves earlier frozen PBP/roster history, builds scheduled pregame rows, and publishes a new immutable Ridge bundle. Model weights and the validated 2024 ridge selection are reused.")
        if st.button("Refresh data and Ridge features", key="ridge_refresh_button"):
            try:
                with st.status(f"Refreshing {refresh_season} data and features... this can take several minutes.", expanded=True):
                    new_bundle = refresh_weekly_bundle(refresh_season, artifact_dir)
                st.session_state["ridge_bundle_path_pending"] = str(new_bundle)
                st.cache_resource.clear()
                st.cache_data.clear()
                st.success(f"Published Ridge bundle {new_bundle.name}; reloading dashboard.")
                st.rerun()
            except Exception as error:
                st.error(f"Weekly Ridge refresh failed. Existing bundle remains unchanged. {error}")
    if artifact_dir.parent.name == "bundles":
        with st.sidebar.expander("Modal bundle upload", expanded=False):
            st.caption("Upload only publishes this immutable bundle to the isolated Ridge Modal volume. It does not deploy code or start simulations.")
            confirm_upload = st.checkbox("I authorize upload to the Ridge Modal volume", key="ridge_confirm_upload")
            if st.button("Upload selected Ridge bundle", disabled=not confirm_upload, key="ridge_upload_button"):
                try:
                    from ridge_game_sim.sync_artifacts import upload_bundle
                    uploaded_hash = upload_bundle(artifact_dir, confirmed=confirm_upload)
                    st.success(f"Uploaded Ridge bundle {uploaded_hash} to the isolated Modal volume.")
                except Exception as error:
                    st.error(f"Ridge Modal upload failed: {error}")
    players, teams = cached_tables(str(artifact_dir), digest(artifact_dir / "feature_manifest.json"))
    available_seasons = sorted({int(value[:4]) for value in teams.game_id.astype(str) if value[:4].isdigit()})
    if not available_seasons:
        st.error("No season-tagged Ridge team rows are available.")
        st.stop()
    default_season = available_seasons[-1]
    season = st.sidebar.selectbox("Season with generated Ridge feature rows", available_seasons, index=len(available_seasons)-1)
    schedule, roster, depth = _load_inputs(int(season), artifact_dir / "weekly_inputs")
    synthetic_schedule = schedule.copy()
    if schedule.empty:
        st.error(f"No regular-season schedule for {season}.")
        st.stop()
    ready_ids = set(teams.game_id.astype(str))
    schedule = schedule[schedule.game_id.astype(str).isin(ready_ids)].copy()
    team_counts = teams[teams.game_id.astype(str).isin(schedule.game_id.astype(str))].groupby("game_id").team.nunique()
    schedule = schedule[schedule.game_id.astype(str).map(team_counts).fillna(0).eq(2)]
    if schedule.empty:
        st.warning("No scheduled games have Ridge pregame feature rows for both teams. Build future-week Ridge features before forecasting unplayed games.")
        st.stop()
    week = st.sidebar.selectbox("Week", sorted(schedule.week.astype(int).unique()))
    week_rows = schedule[schedule.week.astype(int).eq(int(week))].sort_values(["gameday", "game_id"])
    game_labels = {f"{row.away_team} @ {row.home_team} ({pd.Timestamp(row.gameday).date()})": row.game_id for row in week_rows.itertuples()}
    selected_label = st.sidebar.selectbox("Game", list(game_labels))
    selected_id = game_labels[selected_label]
    mode = st.sidebar.radio("Run type", ["Single game", "Full week", "Synthetic opponents"])
    simulations = int(st.sidebar.number_input("Simulations per game", min_value=1, max_value=10000, value=20, step=5))
    backend = st.sidebar.selectbox("Backend", ["Local", "Modal"])
    parallelism = int(st.sidebar.number_input("Workers / containers", min_value=1, value=1 if backend == "Local" else 8, step=1))
    use_model_rank = st.sidebar.checkbox("Rank slots using model metrics", value=True)
    if mode == "Synthetic opponents":
        team_options = synthetic_team_options(synthetic_schedule)
        selected_teams = st.multiselect("Teams", team_options, default=team_options)
    else:
        selected_teams = []
    customize_lineup = st.checkbox("Customize depth chart and player inputs", value=mode == "Single game",
                                   disabled=mode == "Full week")
    custom_assignments = {}
    custom_player_overrides = {}
    if customize_lineup and mode in {"Single game", "Synthetic opponents"}:
        if mode == "Single game":
            customize_teams = [str(week_rows.loc[week_rows.game_id.astype(str).eq(str(selected_id)), "home_team"].iloc[0]),
                               str(week_rows.loc[week_rows.game_id.astype(str).eq(str(selected_id)), "away_team"].iloc[0])]
            custom_game_row = week_rows.loc[week_rows.game_id.astype(str).eq(str(selected_id))].iloc[0]
        else:
            if not selected_teams:
                customize_teams = []
            else:
                customize_team = st.selectbox("Customize this synthetic team", selected_teams)
                customize_teams = [customize_team]
            custom_game_row = (synthetic_game_row(synthetic_schedule, customize_teams[0], int(week))
                               if customize_teams else week_rows.iloc[0])
        kickoff = pd.Timestamp(custom_game_row.gameday)
        custom_key = f"{season}_{week}_{selected_id if mode == 'Single game' else 'synthetic'}"
        for team in customize_teams:
            with st.expander(f"{team} depth chart", expanded=False):
                custom_assignments[team] = lineup_controls(
                    runtime, players, roster, depth, team, int(week), kickoff,
                    use_model_rank, custom_key,
                )
        if custom_assignments:
            with st.expander("Edit one player's Ridge model inputs", expanded=False):
                selected_override = player_stat_editor(runtime, players, custom_assignments, kickoff, custom_key)
                custom_player_overrides = selected_override
    run_id = st.sidebar.text_input("Run ID", f"ridge-{season}-w{int(week):02d}-{datetime.now():%Y%m%d-%H%M%S}", key="ridge_run_id")
    save_key = f"ridge_synthetic_save_folder_{int(week):02d}" if mode == "Synthetic opponents" else "ridge_save_folder"
    save_folder = st.sidebar.text_input("Save folder", str(default_save_folder(mode, int(week))), key=save_key)

    if st.button("Run Ridge simulations", type="primary"):
        if backend == "Modal":
            upload_record = ROOT / "config/last_upload.json"
            if not upload_record.exists() or json.loads(upload_record.read_text()).get("bundle_hash") != bundle_hash:
                st.error("This Ridge bundle is not uploaded to the isolated Modal volume. Use the confirmation-gated upload control first.")
                st.stop()
        if mode == "Single game":
            rows = week_rows[week_rows.game_id.astype(str).eq(str(selected_id))]
        else:
            rows = week_rows
        prepared = []
        failures = []
        with st.status("Preparing Ridge game inputs..."):
            if mode == "Synthetic opponents":
                for team in selected_teams:
                    try:
                        game_row = synthetic_game_row(synthetic_schedule, team, int(week))
                        prepared.append(build_synthetic_game(runtime, players, teams, roster, depth,
                                                             synthetic_schedule, team, game_row, simulations,
                                                             use_model_rank,
                                                             assignments=custom_assignments.get(team),
                                                             player_overrides=custom_player_overrides.get(team, {})))
                    except (ValueError, KeyError) as error:
                        failures.append(f"{team}: {error}")
            else:
                for _, row in rows.iterrows():
                    try:
                        prepared.append(build_real_game(runtime, players, teams, roster, depth,
                                                        schedule, row, simulations, use_model_rank,
                                                        assignments_by_team=custom_assignments,
                                                        player_overrides=custom_player_overrides))
                    except (ValueError, KeyError) as error:
                        failures.append(f"{row.game_id}: {error}")
        if failures:
            st.warning("Skipped games:\n" + "\n".join(failures))
        if not prepared:
            st.error("No valid Ridge games were prepared; no Bayesian fallback was attempted.")
            st.stop()
        progress = st.progress(0.0, text="Running Ridge simulations...")
        result = execute_ridge_games(prepared, artifact_dir, bundle_hash, backend=backend,
                                     parallelism=parallelism, run_id=run_id,
                                     progress=lambda done, total: progress.progress(done / total, text=f"Ridge simulations {done}/{total}"))
        progress.empty()
        games, player_rows, summary = _format_result(result)
        try:
            destination = output_run_directory(save_folder, run_id)
        except (ValueError, OSError, FileExistsError) as error:
            st.error(f"Cannot create Ridge output folder: {error}")
            st.stop()
        games.to_csv(destination / "games.csv", index=False)
        player_rows.to_csv(destination / "players.csv", index=False)
        summary.to_csv(destination / "summary.csv", index=False)
        fantasy_correlation_matrix(player_rows).to_csv(destination / "fantasy_correlation.csv")
        manifest = {
            "run_id": run_id, "season": int(season), "week": int(week),
            "mode": mode, "backend": backend, "simulations_per_game": simulations,
            "bundle_hash": bundle_hash, "feature_manifest_sha256": digest(artifact_dir / "feature_manifest.json"),
            "games": games.game_id.astype(str).unique().tolist(),
            "fantasy_scoring": "game_sim_fantasy.add_fantasy_points",
            "synthetic_home_away_swap": (
                "roles reversed after floor(simulations_per_game / 2) simulations"
                if mode == "Synthetic opponents" else "not applicable"
            ),
        }
        (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        st.session_state["ridge_last_run"] = str(destination)
        st.success(f"Wrote Ridge simulation outputs to {destination}")
        st.dataframe(summary, hide_index=True, width="stretch")
        st.dataframe(player_rows.groupby(["team", "position", "player_name"], as_index=False).fantasy_points.mean().sort_values("fantasy_points", ascending=False), hide_index=True, width="stretch")

    last_run = st.session_state.get("ridge_last_run")
    if last_run:
        st.caption(f"Last Ridge output folder: {last_run}")


if __name__ == "__main__":
    if not st.runtime.exists():
        os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())])
    main()
