"""Simulate the rest of a season with a different starting quarterback on one team."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from nflsim import ui
from nflsim.bundle import verify_bundle_hash
from nflsim.bundle_labels import all_bundles, bundle_label
from nflsim.features.game_inputs import slot_assignments
from nflsim.sim import standings
from nflsim.sim.runtime import RidgeRuntime
from nflsim.sim.season import Move, latest_bundle, load_experiment, quarterbacks, run_experiment, saved_experiments
from nflsim.sim.synthetic_games import _load_inputs, snapshot_depth

SEASON = 2026
BUNDLE_HELP = "Which version of the models to use. Each is named for what changed. 'Current' is the newest one that can simulate upcoming games."
LIMITS = ("The new QB's own inputs and player value replace the old QB's. The team's offense ratings do not change, so they still "
          "reflect the old QB's games, and a backup's rate stats can rest on few plays.")


@st.cache_resource(show_spinner="Checking the bundle's files...")
def verified_hash(path: str) -> str:
    return verify_bundle_hash(Path(path))


@st.cache_data(show_spinner=False, ttl=600)
def bundle_choices() -> dict[str, str]:
    current = latest_bundle()
    return {str(info.path.resolve()): bundle_label(info, current) for info in all_bundles() if info.can_simulate_upcoming}


@st.cache_resource(show_spinner="Loading model bundle...")
def cached_runtime(path: str, bundle_hash: str) -> RidgeRuntime:
    return RidgeRuntime(path, strict_schema=True)


@st.cache_data(show_spinner="Loading rosters...")
def season_state(path: str, bundle_hash: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Rostered QBs, unplayed games, and each team's default starter for its next game."""
    runtime = cached_runtime(path, bundle_hash)
    schedule, roster, _ = _load_inputs(SEASON, Path(path) / "weekly_inputs")
    remaining = schedule[schedule.home_score.isna()].sort_values(["week", "gameday"])
    depth = snapshot_depth(roster, schedule)
    depth["dt"] = pd.to_datetime(depth.dt, errors="coerce", utc=True).dt.tz_convert(None)
    current = {}
    for team in standings.TEAMS:
        games = remaining[remaining.home_team.eq(team) | remaining.away_team.eq(team)]
        if len(games):
            game = games.iloc[0]
            current[team] = slot_assignments(roster, depth, team, int(game.week), pd.Timestamp(game.gameday),
                                             runtime.player_stats_wide)["QB1"]["player_id"]
    return quarterbacks(roster), remaining[["game_id", "week", "home_team", "away_team"]], current


def team_games(result, team: str) -> pd.DataFrame:
    """Re-simulated games from `team`'s side."""
    games = result.games[result.games.home_team.eq(team) | result.games.away_team.eq(team)]
    home = games.home_team.eq(team)
    sign = home.map({True: 1, False: -1})
    return pd.DataFrame({
        "week": games.week, "opponent": games.away_team.where(home, "@" + games.home_team),
        "margin_base": sign * games.margin_base, "margin_experiment": sign * games.margin_experiment,
        "margin_change": sign * (games.margin_experiment - games.margin_base),
        "win_base": games.home_win_base.where(home, 1 - games.home_win_base),
        "win_experiment": games.home_win_experiment.where(home, 1 - games.home_win_experiment),
    })


def show(result, names: dict[str, str]) -> None:
    move = result.manifest["moves"][0]
    team = move["team"]
    row = result.comparison[result.comparison.team.eq(team)].iloc[0]
    ui.section(f"{team} with {names.get(move['player'], move['player'])} from week {result.manifest['first_week']}")
    tiles = st.columns(4)
    tiles[0].metric("Wins", f"{row.wins_experiment:.2f}", f"{row.wins_change:+.2f}")
    for tile, label, column in zip(tiles[1:], ("Playoffs", "Division title", "Top seed"), ("playoffs", "division_title", "bye")):
        tile.metric(label, f"{row[f'{column}_experiment']:.1%}", f"{row[f'{column}_change'] * 100:+.1f} pts")
    st.caption(f"{result.manifest['simulations_per_game']:,} simulations per game, {result.manifest['seasons']:,} seasons. {LIMITS}")
    number = st.column_config.NumberColumn
    st.markdown(f"**{team} games**")
    st.dataframe(team_games(result, team), hide_index=True, width="stretch", column_config={
        "week": "Week", "opponent": "Opponent", "margin_base": number("Margin before", format="%+.1f"),
        "margin_experiment": number("Margin after", format="%+.1f"), "margin_change": number("Change", format="%+.1f"),
        "win_base": number("Win before", format="percent"), "win_experiment": number("Win after", format="percent"),
    })
    st.markdown("**Teams whose outlook moved**")
    comparison = result.comparison
    moved = comparison[comparison.team.isin(result.manifest["affected_teams"]) | comparison.playoffs_change.abs().ge(0.01)]
    table = moved[["team", "wins_base", "wins_experiment", "wins_change", "playoffs_base", "playoffs_experiment", "playoffs_change",
                   "playoffs_se", "division_title_change", "bye_change"]]
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "team": "Team", "wins_base": number("Wins before", format="%.2f"), "wins_experiment": number("Wins after", format="%.2f"),
        "wins_change": number("Change", format="%+.2f"), "playoffs_base": number("Playoffs before", format="percent"),
        "playoffs_experiment": number("Playoffs after", format="percent"), "playoffs_change": number("Change", format="percent"),
        "playoffs_se": number("Std. error", format="percent"), "division_title_change": number("Division change", format="percent"),
        "bye_change": number("Top seed change", format="percent"),
    })
    st.caption(f"Saved in {result.output}")


def describe(path: str, names: dict[str, str]) -> str:
    manifest = load_experiment(Path(path)).manifest
    move = manifest["moves"][0]
    return (f"{move['team']}: {names.get(move['player'], move['player'])} from week {manifest['first_week']} "
            f"({manifest['simulations_per_game']:,} sims)")


def main():
    ui.page("Season QB Experiments")
    choices = bundle_choices()
    artifact_dir = Path(st.sidebar.selectbox("Model bundle", list(choices), format_func=choices.get, help=BUNDLE_HELP))
    bundle_hash = verified_hash(str(artifact_dir))
    ui.header("Season QB experiments", "Simulate the rest of the season with a different starter.",
              {"Bundle": choices[str(artifact_dir)].split(" · ")[0], "Season": SEASON})
    qbs, remaining, current = season_state(str(artifact_dir), bundle_hash)
    names = dict(zip(qbs.player_id, qbs.player_name))
    labels = {row.player_id: f"{row.player_name} ({row.team})" for row in qbs.itertuples()}

    teams = sorted(standings.TEAMS)
    team = st.sidebar.selectbox("Team", teams, help='The team getting a new starting quarterback.')
    starter = current.get(team)
    st.sidebar.caption(f"Current starter: {names.get(starter, starter)}")
    backups = qbs[qbs.team.eq(team) & qbs.player_id.ne(starter)].player_id.tolist()
    elsewhere = qbs[qbs.team.ne(team)].player_id.tolist()
    alternate = st.sidebar.selectbox("New starter", backups + elsewhere, format_func=labels.get,
                                     help="This team's backups come first. A QB from another team leaves that team too.")
    weeks = sorted(remaining[remaining.home_team.eq(team) | remaining.away_team.eq(team)].week.astype(int).unique())
    first_week = st.sidebar.selectbox("Starts from week", weeks, disabled=not weeks, help='First game the new starter plays. Earlier games keep the current starter.')
    simulations = st.sidebar.select_slider("Simulations per game", [40, 100, 200, 500, 1000], value=100, help='Simulations of each remaining game. 100 is a good default; use 500 or more for numbers you plan to rely on.')
    seasons = st.sidebar.select_slider("Seasons", [2000, 5000, 10000, 20000], value=10000, help='Full seasons assembled from those game simulations to get standings and playoff odds. This step is quick.')
    backend = st.sidebar.radio("Run on", ["local", "modal"], horizontal=True,
                               help="Modal needs this bundle uploaded first, from Data operations on the simulator page.")
    parallelism = st.sidebar.slider("Parallel workers", 1, 64 if backend == "modal" else 12, 8, help="Simulations running at once. More is faster, up to your PC's limit.")
    st.sidebar.caption("The first run for a bundle and simulation count also builds the baseline season. Later runs reuse it and "
                       "re-simulate only the teams involved. Stay on this page while it runs.")

    if st.sidebar.button("Run experiment", type="primary", disabled=not weeks, help='Simulate the affected games and compare with the baseline. The first run on a bundle or new forecast also simulates every remaining game.'):
        status, bar = st.empty(), st.progress(0.0)
        try:
            result = run_experiment(
                cached_runtime(str(artifact_dir), bundle_hash), artifact_dir, bundle_hash, SEASON,
                [Move(alternate, team, "QB1")], first_week=int(first_week),
                name=f"{team.lower()}-{alternate}-wk{first_week}-sims{simulations}-seasons{seasons}",
                simulations=simulations, seasons=seasons, backend=backend, parallelism=parallelism,
                on_stage=status.info, on_progress=lambda done, total: bar.progress(done / total))
        except Exception as error:
            st.error(f"The experiment failed: {error}")
            st.stop()
        finally:
            status.empty()
            bar.empty()
        st.session_state["season_experiment"] = str(result.output)

    saved = [str(path) for path in saved_experiments(SEASON, bundle_hash)]
    if not saved:
        st.info("Choose a team and a new starter, then run the experiment.")
        return
    chosen = st.session_state.get("season_experiment")
    picked = st.selectbox("Saved experiments", saved, index=saved.index(chosen) if chosen in saved else 0,
                          format_func=lambda path: describe(path, names), help='Finished experiments on this bundle, newest first.')
    show(load_experiment(Path(picked)), names)


main()
