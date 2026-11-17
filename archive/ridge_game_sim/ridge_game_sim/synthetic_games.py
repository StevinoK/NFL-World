from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ridge_game_sim.execution import execute_ridge_games
from ridge_game_sim.game_inputs import (
    actual_roof, make_players, rest_before, roster_for_team_week, slot_assignments,
)
from ridge_game_sim.feature_builder import canonicalize_teams
from ridge_game_sim.paths import ROOT, digest, output_dir, write_json
from ridge_game_sim.runtime import RidgeRuntime
from torch_game_simulation_runtime import Player, SKILL_SLOT_ORDER


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
                      "gameday": sunday, "home_team": team, "away_team": "SYN",
                      "roof": "outdoors", "location": "Neutral"})


def build_synthetic_game(runtime: RidgeRuntime, player_stats: pd.DataFrame, team_stats: pd.DataFrame,
                         roster: pd.DataFrame, depth: pd.DataFrame, schedule: pd.DataFrame,
                         home_team: str, game_row: pd.Series, simulations: int,
                         use_model_rank: bool = True, assignments=None, player_overrides=None):
    from ridge_game_sim.execution import prepared_game

    schedule = schedule.copy()
    kickoff = pd.Timestamp(game_row.gameday)
    home_team = str(home_team)
    cutoff_team = team_stats.copy()
    cutoff_team["game_date"] = pd.to_datetime(cutoff_team.game_date, errors="coerce")
    history = cutoff_team[cutoff_team.game_date.le(kickoff)]
    home_history = history[history.team.astype(str).eq(home_team)]
    if home_history.empty:
        raise ValueError(f"No pregame Ridge team history for {home_team}")
    home_row = home_history.sort_values(["game_date", "game_id"]).iloc[-1].reindex(runtime.required_team_feature_columns).copy()

    week_schedule = schedule[schedule.week.astype(int).eq(int(game_row.week))].copy()
    if "season" in week_schedule and "season" in game_row.index:
        week_schedule = week_schedule[week_schedule.season.astype(int).eq(int(game_row.season))]
    week_game_ids = set(week_schedule.game_id.astype(str))
    week_teams = set(week_schedule.home_team.dropna().astype(str)) | set(week_schedule.away_team.dropna().astype(str))
    current_team_rows = cutoff_team[
        cutoff_team.game_id.astype(str).isin(week_game_ids)
        & cutoff_team.team.astype(str).isin(week_teams)
    ].sort_values(["game_date", "game_id"]).drop_duplicates("team", keep="last")
    represented_teams = set(current_team_rows.team.astype(str))
    bye_teams = set(synthetic_team_options(schedule)) - represented_teams
    if bye_teams:
        prior_team = cutoff_team[
            pd.to_datetime(cutoff_team.game_date, errors="coerce").lt(kickoff)
            & cutoff_team.team.astype(str).isin(bye_teams)
        ].sort_values(["game_date", "game_id"]).drop_duplicates("team", keep="last")
        current_team_rows = pd.concat([current_team_rows, prior_team], ignore_index=True)
    if current_team_rows.empty:
        raise ValueError(f"No league team inputs available for the synthetic opponent in week {game_row.week}")
    synthetic_team = current_team_rows.reindex(columns=runtime.required_team_feature_columns).mean(numeric_only=True).fillna(0.0)
    for column in runtime.required_team_feature_columns:
        if column.endswith("_venue_effect"):
            synthetic_team[column] = 0.0
            home_row[column] = 0.0
    home_row["days_rest"] = rest_before(schedule, home_team, kickoff)
    home_row["opp_days_rest"] = 7.0
    for model in runtime.context_map:
        runtime.checked_context(model, home_row)
        runtime.checked_context(model, synthetic_team)
    team_features = pd.DataFrame([home_row, synthetic_team], index=[home_team, "SYN"]).reindex(columns=runtime.required_team_feature_columns)

    slots = assignments or slot_assignments(roster, depth, home_team, int(game_row.week), kickoff, player_stats, use_model_rank=use_model_rank)
    from ridge_game_sim.game_inputs import make_players_from_assignments
    home_players = make_players_from_assignments(runtime, player_stats, slots, kickoff, player_overrides)

    player_columns = list(runtime.required_player_feature_columns)
    opposing_lineups = []
    for team in current_team_rows.team.astype(str):
        opponent_row = synthetic_game_row(schedule, team, int(game_row.week))
        opponent_kickoff = pd.Timestamp(opponent_row.gameday)
        try:
            opponent_slots = slot_assignments(
                roster, depth, team, int(game_row.week), opponent_kickoff, player_stats,
                use_model_rank=use_model_rank,
            )
            opposing_lineups.extend(make_players(runtime, player_stats, opponent_slots, opponent_kickoff))
        except ValueError:
            continue
    if opposing_lineups:
        current_players = pd.DataFrame([
            {"position": player.position[:-1], **player.attributes.iloc[0].reindex(player_columns).to_dict()}
            for player in opposing_lineups
        ])
        current_players[player_columns] = current_players[player_columns].apply(pd.to_numeric, errors="coerce")
        position_means = current_players.groupby("position")[player_columns].mean()
        league_player_mean = current_players[player_columns].mean().fillna(0.0)
    else:
        before = player_stats.copy()
        before["game_date"] = pd.to_datetime(before.game_date, errors="coerce")
        before = before[before.game_date.lt(kickoff)]
        before = before.sort_values(["player_id", "game_date", "game_id"]).drop_duplicates("player_id", keep="last")
        values = before.reindex(columns=player_columns).apply(pd.to_numeric, errors="coerce").mean().fillna(0.0)
        position_means = pd.DataFrame([values], index=["OTHER"])
        league_player_mean = values
    synthetic_players = []
    for slot in SKILL_SLOT_ORDER:
        position = slot[:-1]
        means = position_means.loc[position] if position in position_means.index else league_player_mean
        values = means.reindex(player_columns).fillna(0.0).to_dict()
        synthetic_players.append(Player(f"Synthetic {slot}", slot, f"SYN_{slot}", pd.DataFrame([values])))
    synthetic_game = pd.Series({"game_id": f"{game_row.season}_{int(game_row.week):02d}_SYN_{home_team}",
                                "week": int(game_row.week), "home_team": home_team, "away_team": "SYN",
                                "gameday": kickoff, "location": "Neutral"})
    return prepared_game(synthetic_game, team_features, home_players, synthetic_players,
                         simulations, actual_roof(game_row), swap_home_away_halfway=True)


def _load_inputs(season: int, inputs_dir: Path | None = None):
    from ridge_game_sim.weekly_refresh import attach_roster_player_names

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
        depth = roster[[column for column in ("season", "week", "game_id", "team", "player_id", "position", "depth_team") if column in roster]].copy()
        if "depth_team" not in depth:
            depth["depth_team"] = 999
        depth["gsis_id"] = depth.player_id
        depth["pos_abb"] = depth.position
        depth["depth_chart_position"] = pd.to_numeric(depth.depth_team, errors="coerce").fillna(999)
        depth["dt"] = depth.game_id.astype(str).map(schedule.set_index("game_id").gameday)
    elif int(season) > max(seasons, default=0):
        import nfl_data_py as nfl
        from ridge_game_sim.weekly_refresh import attach_current_depth_order
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
    roster = roster[roster.season.eq(season)].copy()
    if "dt" in depth:
        depth["dt"] = pd.to_datetime(depth.dt, errors="coerce", utc=True).dt.tz_convert(None)
    return schedule, roster, depth


def main():
    parser = argparse.ArgumentParser(description="Run Ridge-model synthetic-opponent fantasy simulations")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--teams", nargs="+")
    parser.add_argument("--backend", choices=("local", "modal"), required=True)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/runtime")
    parser.add_argument("--depth-chart-order", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.week <= 18 or args.simulations < 1 or args.parallelism < 1:
        parser.error("week, simulations and parallelism are outside valid ranges")
    from ridge_game_sim.bundle import validate_bundle, verify_bundle_hash
    validate_bundle(args.artifact_dir)
    bundle_hash = verify_bundle_hash(args.artifact_dir)
    runtime = RidgeRuntime(args.artifact_dir, strict_schema=True)
    player_stats, team_stats = runtime.player_stats_wide, runtime.team_stats_wide
    schedule, roster, depth = _load_inputs(args.season, Path(args.artifact_dir) / "weekly_inputs")
    teams = synthetic_team_options(schedule) if not args.teams else [team.upper() for team in args.teams]
    run_id = args.run_id or datetime.now().strftime("ridge-synthetic-%Y%m%d-%H%M%S")
    games = []
    failures = []
    for team in teams:
        try:
            game_row = synthetic_game_row(schedule, team, args.week)
            games.append(build_synthetic_game(runtime, player_stats, team_stats, roster, depth,
                                              schedule, team, game_row, args.simulations,
                                              use_model_rank=not args.depth_chart_order))
        except (ValueError, KeyError) as error:
            failures.append({"team": team, "error": str(error)})
    if not games:
        raise RuntimeError(f"No Ridge synthetic games prepared: {failures}")
    output = output_dir(ROOT / "outputs/synthetic" / run_id)
    results = execute_ridge_games(games, args.artifact_dir, bundle_hash, backend=args.backend,
                                  parallelism=args.parallelism, run_id=run_id, seed=args.seed)
    game_frames, player_frames = [], []
    for game in games:
        game_frame, player_frame = results[game.game_id]
        stem = game.game_id
        game_frame.to_csv(output / f"{stem}_games.csv", index=False)
        player_frame.to_csv(output / f"{stem}_players.csv", index=False)
        game_frames.append(game_frame)
        player_frames.append(player_frame)
    combined_games, combined_players = pd.concat(game_frames, ignore_index=True), pd.concat(player_frames, ignore_index=True)
    combined_games.to_csv(output / "synthetic_games.csv", index=False)
    combined_players.to_csv(output / "synthetic_players.csv", index=False)
    summary = combined_games.groupby(["game_id", "home_team", "away_team"], as_index=False).agg(
        simulations=("simulation", "count"), home_median=("home_score", "median"), away_median=("away_score", "median"),
        spread_median=("spread", "median"), total_median=("total", "median"))
    summary.to_csv(output / "synthetic_summary.csv", index=False)
    write_json(output / "run_manifest.json", {"run_id": run_id, "bundle_hash": bundle_hash, "season": args.season,
        "week": args.week, "simulations_per_game": args.simulations, "backend": args.backend,
        "teams_requested": teams, "games_completed": len(games), "failures": failures,
        "home_away_swap_after_simulation": args.simulations // 2,
        "neutral_site": True, "synthetic_offense_effect": "current league mean", "synthetic_venue_effect": "zero for both teams",
        "synthetic_feature_cutoff": "scheduled-week pregame rows for active teams; latest strictly before kickoff for bye teams"})
    print(f"Wrote {len(games)} Ridge synthetic games to {output}")
    if failures:
        print(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
