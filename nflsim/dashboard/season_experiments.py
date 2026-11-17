"""Simulate the rest of a season with a different starting quarterback on one team."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nflsim import ui
from nflsim.bundle import verify_bundle_hash
from nflsim.bundle_labels import all_bundles, bundle_label
from nflsim.paths import canonical_hash
from nflsim.features.game_inputs import slot_assignments
from nflsim.sim import standings
from nflsim.sim.runtime import RidgeRuntime
from nflsim.sim.season import (Move, ensure_seasons, experiment_tag, latest_bundle, load_experiment, quarterbacks, run_experiment,
                               saved_experiments)
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
    st.markdown(f"**{team} games**")
    st.dataframe(
        ui.styled_frame(
            team_games(result, team),
            {"margin_base": ui.signed, "margin_experiment": ui.signed, "margin_change": ui.signed, "win_base": ui.percent, "win_experiment": ui.percent},
            {"margin_base": (0.0, ui.SPREAD_SCALE), "margin_experiment": (0.0, ui.SPREAD_SCALE), "margin_change": (0.0, ui.GAP_SCALE),
             "win_base": (0.5, ui.WIN_SCALE), "win_experiment": (0.5, ui.WIN_SCALE)},
        ),
        hide_index=True, width="stretch", column_config={
            "week": "Week", "opponent": "Opponent", "margin_base": "Margin before", "margin_experiment": "Margin after", "margin_change": "Change",
            "win_base": "Win before", "win_experiment": "Win after",
        },
    )
    st.markdown("**Teams whose outlook moved**")
    comparison = result.comparison
    moved = comparison[comparison.team.isin(result.manifest["affected_teams"]) | comparison.playoffs_change.abs().ge(0.01)]
    table = moved[["team", "wins_base", "wins_experiment", "wins_change", "playoffs_base", "playoffs_experiment", "playoffs_change",
                   "playoffs_se", "division_title_change", "bye_change"]]
    st.dataframe(
        ui.styled_frame(
            table,
            {"wins_base": "{:.2f}", "wins_experiment": "{:.2f}", "wins_change": "{:+.2f}", "playoffs_base": ui.percent, "playoffs_experiment": ui.percent,
             "playoffs_change": ui.signed_percent, "playoffs_se": ui.percent, "division_title_change": ui.signed_percent, "bye_change": ui.signed_percent},
            {"wins_base": (8.5, 4.0), "wins_experiment": (8.5, 4.0), "wins_change": (0.0, 1.0), "playoffs_base": (0.5, 0.5), "playoffs_experiment": (0.5, 0.5),
             "playoffs_change": (0.0, 0.2), "division_title_change": (0.0, 0.2), "bye_change": (0.0, 0.2)},
        ),
        hide_index=True, width="stretch", column_config={
            "team": "Team", "wins_base": "Wins before", "wins_experiment": "Wins after", "wins_change": "Change", "playoffs_base": "Playoffs before",
            "playoffs_experiment": "Playoffs after", "playoffs_change": "Change", "playoffs_se": "Std. error", "division_title_change": "Division change",
            "bye_change": "Top seed change",
        },
    )
    st.caption(f"Saved in {result.output}")


# ------------------------------------------------------------------ comparing experiments for one team
GAMES = 17
MISSED = "#d5d9de"


@st.cache_data(show_spinner="Rebuilding season results for older experiments...", max_entries=64)
def team_seasons(output: str, artifact_dir: str, team: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """`team`'s simulated seasons in the baseline and the experiment, plus the league's points per game."""
    base, experiment = ensure_seasons(Path(output), Path(artifact_dir))
    league = float(base.points_for.mean()) / GAMES
    return base[base.team.eq(team)].reset_index(drop=True), experiment[experiment.team.eq(team)].reset_index(drop=True), league


@st.cache_data(show_spinner=False, max_entries=64)
def game_win_rates(path: str, team: str) -> pd.DataFrame:
    """`team`'s win rate in each game of a cached score file: game_id, win."""
    scores = pd.read_parquet(path)
    home_game = scores.game_id.str.endswith("_" + team)
    away_game = scores.game_id.str.contains(f"_{team}_")
    scores = scores[home_game | away_game]
    margin = (scores.home_score - scores.away_score).where(scores.game_id.str.endswith("_" + team), scores.away_score - scores.home_score)
    return scores.assign(win=(margin.clip(-1, 1) + 1) / 2).groupby("game_id", as_index=False).win.mean()


def scenario_label(manifest: dict, names: dict[str, str]) -> str:
    moves = ", ".join(f"{names.get(move['player'], move['player'])}" + (f" to {move['team']}" if move["team"] else " out")
                      for move in manifest["moves"])
    return f"{moves} from week {manifest['first_week']}"


def season_rows(label: str, seasons: pd.DataFrame, league: float) -> dict:
    return {"scenario": label, "wins": seasons.wins.mean(), "playoffs": seasons.seed.gt(0).mean(),
            "division_title": seasons.seed.between(1, 4).mean(), "bye": seasons.seed.eq(1).mean(),
            "points_for": seasons.points_for.mean() / GAMES, "points_against": seasons.points_against.mean() / GAMES,
            "league": league}


def comparison_table(rows: list[dict]) -> None:
    table = pd.DataFrame(rows)
    base = table.iloc[0]
    table["wins_change"] = table.wins - base.wins
    table["playoffs_change"] = table.playoffs - base.playoffs
    league = float(base.league)
    # Levels are shaded against the current starter, so color reads as better or worse than him.
    columns = ["scenario", "wins", "wins_change", "playoffs", "playoffs_change", "division_title", "bye", "points_for", "points_against"]
    st.dataframe(
        ui.styled_frame(
            table[columns],
            {"wins": "{:.2f}", "wins_change": "{:+.2f}", "playoffs": ui.percent, "playoffs_change": ui.signed_percent,
             "division_title": ui.percent, "bye": ui.percent, "points_for": ui.number, "points_against": ui.number},
            {"wins": (base.wins, 1.0), "wins_change": (0.0, 1.0), "playoffs": (base.playoffs, 0.15), "playoffs_change": (0.0, 0.15),
             "division_title": (base.division_title, 0.15), "bye": (base.bye, 0.15), "points_for": (league, ui.POINTS_SCALE),
             "points_against": (league, -ui.POINTS_SCALE)},
        ),
        hide_index=True, width="stretch", column_config={
            "scenario": st.column_config.TextColumn("Starter", width="large"), "wins": "Wins", "wins_change": "Change",
            "playoffs": "Playoffs", "playoffs_change": "Change", "division_title": "Division", "bye": "Top seed",
            "points_for": "Points/game", "points_against": "Allowed/game",
        },
    )


def colors(labels: list[str]) -> dict[str, str]:
    """The baseline in gray, then experiments in the fixed series order."""
    return {label: (ui.MUTED if index == 0 else ui.SERIES[(index - 1) % len(ui.SERIES)]) for index, label in enumerate(labels)}


def wins_chart(seasons: dict[str, pd.DataFrame], palette: dict[str, str]):
    figure = go.Figure()
    for label, frame in seasons.items():
        counts = frame.wins.apply(np.floor).value_counts(normalize=True).reindex(range(GAMES + 1), fill_value=0)
        start = int(counts[counts.gt(0)].index.min())
        counts = counts.loc[start:]
        figure.add_trace(go.Scatter(
            x=counts.index, y=counts.values, name=label, mode="lines+markers",
            line={"color": palette[label], "width": 2, **({"dash": "dash"} if label == next(iter(seasons)) else {})},
            marker={"size": 8, "line": {"color": ui.SURFACE, "width": 2}},
            hovertemplate=f"{label}<br>%{{x}} wins: %{{y:.1%}}<extra></extra>"))
    figure.update_layout(title="Season wins", xaxis_title="Wins (a tie counts as half)", yaxis_title="Share of seasons",
                         yaxis_tickformat=".0%", xaxis_dtick=1, hovermode="closest", legend={"orientation": "h", "y": -0.32},
                         margin={"t": 50, "b": 10})
    return ui.style_plotly(figure, height=420)


def seeds_chart(seasons: dict[str, pd.DataFrame]):
    shares = pd.DataFrame({label: frame.seed.value_counts(normalize=True) for label, frame in seasons.items()}).fillna(0).T
    shades = ["#0d3b73", "#1f5fae", "#3f7fc8", "#6a9fd8", "#93bce6", "#b8d4f0", "#d9e8f8"]
    figure = go.Figure()
    for seed in range(1, standings.SEEDS + 1):
        if seed in shares:
            figure.add_trace(go.Bar(y=shares.index, x=shares[seed], name=f"Seed {seed}", orientation="h",
                                    marker={"color": shades[seed - 1], "line": {"color": ui.SURFACE, "width": 2}},
                                    hovertemplate=f"%{{y}}<br>Seed {seed}: %{{x:.1%}}<extra></extra>"))
    if 0 in shares:
        figure.add_trace(go.Bar(y=shares.index, x=shares[0], name="Missed", orientation="h",
                                marker={"color": MISSED, "line": {"color": ui.SURFACE, "width": 2}},
                                hovertemplate="%{y}<br>Missed the playoffs: %{x:.1%}<extra></extra>"))
    figure.update_layout(title="Playoff seed", barmode="stack", xaxis_tickformat=".0%", xaxis_range=[0, 1],
                         yaxis={"autorange": "reversed", "automargin": True}, legend={"orientation": "h", "y": -0.2, "traceorder": "normal"},
                         margin={"t": 50, "b": 10})
    return ui.style_plotly(figure, height=120 + 46 * len(shares))


def points_chart(seasons: dict[str, pd.DataFrame], palette: dict[str, str], column: str, title: str):
    figure = go.Figure()
    for label, frame in seasons.items():
        figure.add_trace(go.Box(x=frame[column] / GAMES, name=label, orientation="h", boxpoints=False,
                                marker_color=palette[label], line={"width": 2}, fillcolor="rgba(0,0,0,0)",
                                hoverinfo="x", showlegend=False))
    figure.update_layout(title=title, xaxis_title="Per game, over the full season",
                         yaxis={"autorange": "reversed", "automargin": True, "showticklabels": True}, margin={"t": 50, "b": 10})
    return ui.style_plotly(figure, height=120 + 46 * len(seasons))


def weekly_chart(rates: dict[str, pd.DataFrame], palette: dict[str, str], schedule: pd.DataFrame, team: str):
    figure = go.Figure()
    games = schedule.assign(opponent=schedule.away_team.where(schedule.home_team.eq(team), "@" + schedule.home_team))
    for label, frame in rates.items():
        frame = frame.merge(games[["game_id", "week", "opponent"]], on="game_id").sort_values("week")
        figure.add_trace(go.Scatter(
            x=frame.week, y=frame.win, name=label, mode="lines+markers", customdata=frame.opponent,
            line={"color": palette[label], "width": 2, **({"dash": "dash"} if label == next(iter(rates)) else {})},
            marker={"size": 8, "line": {"color": ui.SURFACE, "width": 2}},
            hovertemplate=f"{label}<br>Week %{{x}} vs %{{customdata}}: %{{y:.1%}}<extra></extra>"))
    ticks = games.sort_values("week")
    figure.add_hline(y=0.5, line={"color": ui.BORDER, "dash": "dash", "width": 1})
    figure.update_layout(title="Win chance by game", yaxis_tickformat=".0%", yaxis_range=[0, 1], hovermode="closest",
                         xaxis={"tickvals": ticks.week, "ticktext": [f"{week}<br>{opponent}" for week, opponent in zip(ticks.week, ticks.opponent)]},
                         legend={"orientation": "h", "y": -0.3}, margin={"t": 50, "b": 10})
    return ui.style_plotly(figure, height=420)


def compare_team(saved: list[str], names: dict[str, str], current: dict[str, str], artifact_dir: Path, remaining: pd.DataFrame) -> None:
    manifests = {path: load_experiment(Path(path)).manifest for path in saved}
    moved = pd.Series([move["team"] for manifest in manifests.values() for move in manifest["moves"] if move["team"]])
    teams = sorted(standings.TEAMS)
    default = moved.value_counts().index[0] if len(moved) else teams[0]
    left, middle = st.columns([1, 2])
    team = left.selectbox("Team", teams, index=teams.index(default), key="compare_team",
                          help="Whose seasons to compare. Experiments that move a QB to this team are picked first.")
    groups = sorted({experiment_tag(manifest) + f"-n{manifest['seasons']}" for manifest in manifests.values()})
    def group_label(group):
        sample = next(m for m in manifests.values() if experiment_tag(m) + f"-n{m['seasons']}" == group)
        return f"{sample['simulations_per_game']:,} sims per game, {sample['seasons']:,} seasons, forecast {sample['weather_forecast']['tag'][2:]}"
    group = groups[0]
    if len(groups) > 1:
        counts = {g: sum(experiment_tag(m) + f"-n{m['seasons']}" == g for m in manifests.values()) for g in groups}
        group = st.selectbox("Baseline", groups, index=groups.index(max(counts, key=counts.get)), format_func=group_label,
                             help="Experiments compare only against the baseline they were run with: the same simulation count and weather forecast.")
    pool = [path for path, manifest in manifests.items() if experiment_tag(manifest) + f"-n{manifest['seasons']}" == group]
    own = [path for path in pool if any(move["team"] == team for move in manifests[path]["moves"])]
    picked = middle.multiselect("Experiments", pool, default=(own or pool)[:8], max_selections=8,
                                format_func=lambda path: scenario_label(manifests[path], names), key=f"compare_{team}_{group}",
                                help="Up to eight. Experiments for other teams show how those moves change this team's season.")
    if not picked:
        st.info("Pick at least one experiment.")
        return
    picked = sorted(picked, key=lambda path: scenario_label(manifests[path], names))
    starter = current.get(team)
    base_label = f"{names.get(starter, starter)} (current starter)" if starter else "Baseline"
    first_base, _, league = team_seasons(picked[0], str(artifact_dir), team)
    seasons, rows = {base_label: first_base}, [season_rows(base_label, first_base, league)]
    for path in picked:
        _, experiment, _ = team_seasons(path, str(artifact_dir), team)
        label = scenario_label(manifests[path], names)
        seasons[label] = experiment
        rows.append(season_rows(label, experiment, league))
    palette = colors(list(seasons))

    sample = manifests[picked[0]]
    ui.section(f"{team} seasons by starter")
    comparison_table(rows)
    st.caption(f"Full-season averages over {sample['seasons']:,} seasons, {sample['simulations_per_game']:,} simulations per game, "
               "including games already played. Each experiment uses the same season draws as the baseline, so the gaps come "
               "from the moves. Blue is better and orange worse than the current starter for wins and rates, and than the "
               f"league average of {league:.1f} per game for points. {LIMITS}")
    for chart, name in ((wins_chart(seasons, palette), f"{team} wins by starter"), (seeds_chart(seasons), f"{team} playoff seeds")):
        st.plotly_chart(chart, theme=None, use_container_width=True)
        ui.image_actions(chart, name)
    scored, allowed = st.columns(2)
    for column, stat, title in ((scored, "points_for", "Points scored"), (allowed, "points_against", "Points allowed")):
        chart = points_chart(seasons, palette, stat, title)
        column.plotly_chart(chart, theme=None, use_container_width=True)
        ui.image_actions(chart, f"{team} {title}", container=column)
    st.caption("Boxes cover the middle half of seasons, the line inside is the median, and the whiskers reach the rest "
               "short of outliers.")
    root = Path(picked[0]).parent.parent
    base_rates = game_win_rates(str(root / f"baseline-{experiment_tag(sample)}.parquet"), team)
    rates = {base_label: base_rates}
    for path in picked:
        manifest = manifests[path]
        key = manifest.get("experiment_key") or canonical_hash(
            [[asdict(Move(**move)) for move in manifest["moves"]], manifest["first_week"], experiment_tag(manifest)])[:12]
        rerun = game_win_rates(str(root / f"experiment-{key}-{experiment_tag(manifest)}.parquet"), team)
        rates[scenario_label(manifest, names)] = pd.concat([base_rates[~base_rates.game_id.isin(rerun.game_id)], rerun])
    schedule = remaining[remaining.home_team.eq(team) | remaining.away_team.eq(team)]
    chart = weekly_chart(rates, palette, schedule, team)
    st.plotly_chart(chart, theme=None, use_container_width=True)
    ui.image_actions(chart, f"{team} weekly win chances")


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
    one, several = st.tabs(["One experiment", "Compare for a team"])
    with one:
        chosen = st.session_state.get("season_experiment")
        picked = st.selectbox("Saved experiments", saved, index=saved.index(chosen) if chosen in saved else 0,
                              format_func=lambda path: describe(path, names), help='Finished experiments on this bundle, newest first.')
        show(load_experiment(Path(picked)), names)
    with several:
        compare_team(saved, names, current, artifact_dir, remaining)


main()
