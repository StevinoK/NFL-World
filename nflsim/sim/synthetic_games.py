"""Rate teams by simulating each against the whole league, itself included, at a neutral site.

An averaged opponent plays like no real team, so ratings come from real lineups.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations_with_replacement
from pathlib import Path

import pandas as pd

from nflsim.engine.chunk_execution import serialize_players
from nflsim.sim.execution import RidgePreparedGame, execute_ridge_games
from nflsim.features.game_inputs import (
    make_players_from_assignments, pregame_team_snapshot, rest_before, roster_for_team_week, slot_assignments,
)
from nflsim.features.builder import canonicalize_teams
from nflsim.features.lineup_value import OPPONENT_ALIASES, apply_lineup_value, dl_assignments, lineup_values, player_value_tables
from nflsim.paths import ROOT, output_dir, write_json
from nflsim.sim.runtime import RidgeRuntime
from nflsim.features.forecast_archive import pregame_weather
from nflsim.features.weather import weather_features

# Every pairing is played in the same mild outdoor conditions, so a rating doesn't move with one week's weather.
NEUTRAL_WEATHER = {"roof": "outdoors", "temp": 60.0, "wind": 7.0, "precipitation": 0.0}
DEFAULT_SIMULATIONS = 64


@dataclass(frozen=True)
class LeagueTeam:
    name: str
    season: int
    features: dict
    players: list


def synthetic_team_options(schedule: pd.DataFrame) -> list[str]:
    return sorted(set(schedule.home_team.dropna().astype(str)) | set(schedule.away_team.dropna().astype(str)))


def synthetic_game_row(schedule: pd.DataFrame, team: str, week: int) -> pd.Series:
    weekly = schedule.loc[schedule.week.astype(int).eq(int(week))].copy()
    if weekly.empty:
        raise ValueError(f"No scheduled dates for week {week}")
    candidates = weekly.loc[weekly.home_team.astype(str).eq(team) | weekly.away_team.astype(str).eq(team)]
    if not candidates.empty:
        return candidates.sort_values(["gameday", "game_id"]).iloc[0].copy()
    if team not in synthetic_team_options(schedule):
        raise ValueError(f"Unknown team: {team}")
    first_date = pd.to_datetime(weekly.gameday, errors="raise").min().normalize()
    sunday = first_date + pd.Timedelta(days=(6 - first_date.weekday()) % 7)
    return pd.Series({"season": int(weekly.season.iloc[0]), "week": int(week),
                      "gameday": sunday, "home_team": team, "away_team": "BYE",
                      "roof": "outdoors", "location": "Neutral"})


def prepare_league_team(runtime: RidgeRuntime, player_stats: pd.DataFrame, team_stats: pd.DataFrame,
                        roster: pd.DataFrame, depth: pd.DataFrame, schedule: pd.DataFrame, team: str, week: int,
                        use_model_rank: bool = True, assignments=None, player_overrides=None, dl=None,
                        pv_override=None) -> LeagueTeam:
    """A team's inputs and lineup for the week, ready to meet any opponent at a neutral site."""
    game_row = synthetic_game_row(schedule, team, week)
    kickoff = pd.Timestamp(game_row.gameday)
    season = int(game_row.season)
    row = pregame_team_snapshot(runtime, team_stats, team, kickoff).copy()
    row["days_rest"] = rest_before(schedule, team, kickoff)
    row["opp_days_rest"] = 7.0
    for column in runtime.required_team_feature_columns:
        if column.endswith("_venue_effect"):
            row[column] = 0.0
    for column, value in weather_features(NEUTRAL_WEATHER).items():
        if column in row.index:
            row[column] = value
    slots = assignments or slot_assignments(roster, depth, team, int(week), kickoff, player_stats, use_model_rank=use_model_rank)
    players = make_players_from_assignments(runtime, player_stats, slots, kickoff, player_overrides)
    tables = player_value_tables(runtime)
    if tables is not None:
        if dl is None:
            dl = dl_assignments(roster, depth, team, int(week), kickoff, tables, roster_for_team_week)
        values, _ = lineup_values(tables, slots, dl, season, int(week), kickoff)
        values.update(pv_override or {})
        frame = apply_lineup_value(tables, pd.DataFrame([row], index=[team]), {team: players}, {team: values}, season, int(week),
                                   update_aliases=False)
        row = frame.loc[team]
    for model in runtime.context_map:
        runtime.checked_context(model, row)
    return LeagueTeam(team, season, row.to_dict(), serialize_players(players))


def league_game(first: LeagueTeam, second: LeagueTeam, week: int, simulations: int) -> RidgePreparedGame:
    """A neutral-site pairing. A team against itself gets half the simulations, because both sides count for that team."""
    features = {first.name: dict(first.features), second.name: dict(second.features)}
    for team, opponent in ((first, second), (second, first)):
        for alias, source in OPPONENT_ALIASES.items():
            if alias in features[team.name]:
                features[team.name][alias] = opponent.features[source]
    return RidgePreparedGame(
        game_id=f"{first.season}_{int(week):02d}_PAIR_{first.name}_{second.name}", week=int(week),
        home_team=first.name, away_team=second.name, roof=NEUTRAL_WEATHER["roof"],
        home_players=first.players, away_players=second.players, team_features=features,
        simulations=max(1, int(simulations) // 2) if first.name == second.name else int(simulations),
        neutral_site=True, swap_home_away_halfway=True,
    )


def league_pairs(teams, selected=None) -> list[tuple[str, str]]:
    """Every pairing with a selected team in it, including each selected team against itself, so all face the same field."""
    chosen = set(teams if selected is None else selected)
    return [(first, second) for first, second in combinations_with_replacement(sorted(teams), 2)
            if (first in chosen or second in chosen) and (first != second or first in chosen)]


def build_league_games(runtime: RidgeRuntime, player_stats: pd.DataFrame, team_stats: pd.DataFrame, roster: pd.DataFrame,
                       depth: pd.DataFrame, schedule: pd.DataFrame, week: int, simulations: int, selected=None,
                       use_model_rank: bool = True, custom=None):
    """Returns the pairings and any teams that couldn't be prepared. custom holds a team's prepare_league_team lineup changes."""
    teams, failures = {}, []
    for team in synthetic_team_options(schedule):
        try:
            teams[team] = prepare_league_team(runtime, player_stats, team_stats, roster, depth, schedule, team, week,
                                              use_model_rank, **(custom or {}).get(team, {}))
        except (ValueError, KeyError) as error:
            failures.append({"team": team, "error": str(error)})
    games = [league_game(teams[first], teams[second], week, simulations) for first, second in league_pairs(teams, selected)]
    return games, failures


def team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Game rows seen from each team, with that team as home_team, for the teams that faced the whole field.

    A team's own game gives one row per side. Runs from the old averaged opponent read the same way, minus that opponent.
    """
    swapped = games.rename(columns={"home_team": "away_team", "away_team": "home_team", "home_score": "away_score", "away_score": "home_score"})
    rows = pd.concat([games, swapped.assign(spread=-games.spread)], ignore_index=True)
    rows = rows[rows.home_team.ne("SYN")]
    opponents = rows.groupby("home_team").away_team.nunique()
    return rows[rows.home_team.isin(opponents[opponents.eq(opponents.max())].index)].reset_index(drop=True)


def snapshot_depth(roster: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Depth-chart frame for a weekly-input snapshot, read off each roster row's depth_team."""
    depth = roster[[column for column in ("season", "week", "game_id", "team", "player_id", "position", "depth_team") if column in roster]].copy()
    if "depth_team" not in depth:
        depth["depth_team"] = 999
    depth["gsis_id"] = depth.player_id
    depth["pos_abb"] = depth.position
    depth["depth_chart_position"] = pd.to_numeric(depth.depth_team, errors="coerce").fillna(999)
    depth["dt"] = depth.game_id.astype(str).map(schedule.set_index("game_id").gameday)
    return depth


def _load_inputs(season: int, inputs_dir: Path | None = None):
    from nflsim.features.weekly_refresh import attach_roster_player_names

    inputs = Path(inputs_dir) if inputs_dir is not None else ROOT / "artifacts/inputs"
    schedule_path, roster_path = inputs / "schedule.parquet", inputs / "roster.parquet"
    use_snapshot = False
    if schedule_path.exists() and roster_path.exists():
        schedule = pd.read_parquet(schedule_path)
        roster = pd.read_parquet(roster_path)
        seasons = set(pd.to_numeric(schedule.get("season"), errors="coerce").dropna().astype(int))
        use_snapshot = int(season) in seasons
    else:
        schedule, roster = pd.DataFrame(), pd.DataFrame()
        seasons = set()

    if use_snapshot:
        depth = snapshot_depth(roster, schedule)
    elif int(season) > max(seasons, default=0):
        import nfl_data_py as nfl
        from nflsim.features.weekly_refresh import attach_current_depth_order
        schedule = canonicalize_teams(nfl.import_schedules([season]))
        roster = nfl.import_weekly_rosters([season])
        depth = nfl.import_depth_charts([season])
        schedule = schedule[schedule.game_type.eq("REG")].copy()
        schedule["game_date"] = pd.to_datetime(schedule.gameday, errors="coerce")
        roster = canonicalize_teams(roster)
        team_games = pd.concat([
            schedule[["game_id", "season", "week", "home_team"]].rename(columns={"home_team": "team"}),
            schedule[["game_id", "season", "week", "away_team"]].rename(columns={"away_team": "team"}),
        ], ignore_index=True)
        roster = roster.merge(team_games, on=["season", "week", "team"], how="inner", validate="many_to_one")
        if "report_status" not in roster:
            roster["report_status"] = pd.NA
        roster["player_name"] = roster.get("full_name", roster.player_id)
        roster = attach_current_depth_order(roster, depth, season)
    elif not schedule.empty:
        raise ValueError(f"Input snapshot has no schedule for historical season {season}; prepare that season's inputs first")
    else:
        import nfl_data_py as nfl
        schedule = nfl.import_schedules([season])
        roster = nfl.import_weekly_rosters([season])
        depth = nfl.import_depth_charts([season])
    schedule, roster = canonicalize_teams(schedule), canonicalize_teams(roster)
    try:
        import nfl_data_py as nfl
        ids = nfl.import_ids()
    except Exception:
        ids = pd.DataFrame()
    roster = attach_roster_player_names(roster, ids)
    depth = canonicalize_teams(depth)
    schedule = schedule[(schedule.season.eq(season)) & schedule.game_type.eq("REG") & schedule.week.between(1, 18)].copy()
    schedule["gameday"] = pd.to_datetime(schedule.gameday, errors="coerce")
    schedule = pregame_weather(schedule)
    roster = roster[roster.season.eq(season)].copy()
    if "dt" in depth:
        depth["dt"] = pd.to_datetime(depth.dt, errors="coerce", utc=True).dt.tz_convert(None)
    return schedule, roster, depth


def team_table(games: pd.DataFrame) -> pd.DataFrame:
    rows = team_games(games)
    return rows.assign(win=rows.home_score.gt(rows.away_score).astype(float)).groupby("home_team", as_index=False).agg(
        simulations=("spread", "count"), points_for=("home_score", "mean"), points_against=("away_score", "mean"),
        margin_mean=("spread", "mean"), margin_median=("spread", "median"), total_median=("total", "median"),
        win_probability=("win", "mean")).rename(columns={"home_team": "team"}).sort_values("margin_mean", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Simulate every team against the whole league at a neutral site")
    parser.add_argument("--season", type=int, default=datetime.now().year)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS, help="Per pairing; each team gets about 32 times this")
    parser.add_argument("--teams", nargs="+", help="Rate only these teams, still against the whole league")
    parser.add_argument("--backend", choices=("local", "modal"), required=True)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/runtime")
    parser.add_argument("--depth-chart-order", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.week <= 18 or args.simulations < 2 or args.parallelism < 1:
        parser.error("week, simulations and parallelism are outside valid ranges")
    from nflsim.bundle import validate_bundle, verify_bundle_hash
    validate_bundle(args.artifact_dir)
    bundle_hash = verify_bundle_hash(args.artifact_dir)
    runtime = RidgeRuntime(args.artifact_dir, strict_schema=True)
    player_stats, team_stats = runtime.player_stats_wide, runtime.team_stats_wide
    schedule, roster, depth = _load_inputs(args.season, Path(args.artifact_dir) / "weekly_inputs")
    selected = [team.upper() for team in args.teams] if args.teams else None
    run_id = args.run_id or datetime.now().strftime("ridge-league-%Y%m%d-%H%M%S")
    games, failures = build_league_games(runtime, player_stats, team_stats, roster, depth, schedule, args.week, args.simulations,
                                         selected, use_model_rank=not args.depth_chart_order)
    if not games:
        raise RuntimeError(f"No league games prepared: {failures}")
    output = output_dir(ROOT / "outputs/synthetic" / run_id)
    results = execute_ridge_games(games, args.artifact_dir, bundle_hash, backend=args.backend,
                                  parallelism=args.parallelism, run_id=run_id, seed=args.seed)
    output = output_dir(output)
    combined_games = pd.concat([results[game.game_id][0] for game in games], ignore_index=True)
    combined_players = pd.concat([results[game.game_id][1] for game in games], ignore_index=True)
    combined_games.to_csv(output / "synthetic_games.csv", index=False)
    combined_players.to_csv(output / "synthetic_players.csv", index=False)
    team_table(combined_games).to_csv(output / "synthetic_summary.csv", index=False)
    write_json(output / "run_manifest.json", {"run_id": run_id, "bundle_hash": bundle_hash, "season": args.season,
        "week": args.week, "simulations_per_pairing": args.simulations, "backend": args.backend,
        "teams_requested": selected or sorted({game.home_team for game in games}), "games_completed": len(games), "failures": failures,
        "neutral_site": True, "weather": NEUTRAL_WEATHER, "field": "every team, including itself at half the simulations",
        "venue_effect": "zero for both teams"})
    print(f"Wrote {len(games)} league pairings to {output}")
    if failures:
        print(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
