"""CLI script to simulate all NFL teams at home against a synthetic opponent.

A synthetic opponent has league-average values for all team features and all
player inputs (QB1, WR1-4, RB1-3, TE1-3). Each NFL team plays in their home stadium
under their actual venue roof condition, using their week's depth chart and latest
Bayesian attributes.

Outputs:
  - Per-game CSVs: <game_id>_games.csv, <game_id>_players.csv
  - Combined CSVs: synthetic_home_games.csv, synthetic_home_players.csv, synthetic_home_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nfl_data_py as nfl
import numpy as np
import pandas as pd

from game_sim_dashboard import (
    DEFAULT_SEASON,
    MAX_REST_DAYS,
    SKILL_POSITIONS,
    SLOT_LIMITS,
    SLOT_RANK_STAT,
    build_players,
    build_slot_assignments,
    depth_chart_order,
    load_depth_charts,
    load_rosters,
    load_schedule,
    prior_fill_values,
    roster_for_team_week,
    sanitize_stem,
    summarize_games,
    team_feature_snapshot,
)
from game_sim_fantasy import add_fantasy_points, fantasy_correlation_matrix, team_unit_rows
from game_sim_worker import init_worker, serialize_players, simulate_chunk_task, simulate_task
from project_paths import DEFAULT_SIMULATION_OUTPUT_DIR
from torch_game_simulation_runtime import Game, Player, ROOF_TYPE, SKILL_SLOT_ORDER, SimulationRuntime, Team

WORKSPACE_DIR = DEFAULT_SIMULATION_OUTPUT_DIR


def build_synthetic_team_features(runtime: SimulationRuntime, rest_days: float = 7.0) -> pd.Series:
    """Build team features set to league-average values for all required inputs."""
    cols = [c for c in runtime.required_team_feature_columns if c in runtime.team_stats_wide.columns]
    syn_features = runtime.team_stats_wide[cols].mean(numeric_only=True).copy()
    syn_features["days_rest"] = float(rest_days)
    syn_features["opp_days_rest"] = float(rest_days)
    for col in runtime.required_team_feature_columns:
        if col not in syn_features.index or pd.isna(syn_features[col]):
            syn_features[col] = 0.0
    return syn_features.reindex(runtime.required_team_feature_columns)


def build_synthetic_players(runtime: SimulationRuntime) -> List[Player]:
    """Build 11 skill players (QB1, WR1-4, RB1-3, TE1-3) with league-average attributes."""
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


def run_single_team_simulation(
    runtime: SimulationRuntime,
    game_row: pd.Series,
    team_features: pd.DataFrame,
    home_players: List[Player],
    syn_players: List[Player],
    simulations: int,
    workers: int = 1,
    executor: Optional[ProcessPoolExecutor] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run simulations for a single game against synthetic opponent."""
    home_team = str(game_row["home_team"])
    away_team = "SYN"
    roof = ROOF_TYPE.get(home_team, "outdoors")

    game_rows: List[Dict[str, object]] = []
    player_rows: List[Dict[str, object]] = []

    if workers > 1 and simulations > 1:
        task_template = {
            "data_dir": str(runtime.data_dir),
            "home_team": home_team,
            "away_team": away_team,
            "roof": roof,
            "home_players": serialize_players(home_players),
            "away_players": serialize_players(syn_players),
            "team_features": {
                home_team: team_features.loc[home_team].to_dict(),
                away_team: team_features.loc[away_team].to_dict(),
            },
        }

        chunk_size = max(1, min(25, int(np.ceil(simulations / (workers * 4)))))
        chunks: List[Tuple[int, int]] = []
        curr = 1
        while curr <= simulations:
            cnt = min(chunk_size, simulations - curr + 1)
            chunks.append((curr, cnt))
            curr += cnt

        def _run_pool(exec_pool: ProcessPoolExecutor) -> None:
            futures = [
                exec_pool.submit(
                    simulate_chunk_task,
                    {**task_template, "start_simulation": start_idx, "count": cnt},
                )
                for start_idx, cnt in chunks
            ]
            for future in as_completed(futures):
                result = future.result()
                game_rows.extend(result["game_rows"])
                player_rows.extend(result["player_rows"])

        if executor is not None:
            _run_pool(executor)
        else:
            with ProcessPoolExecutor(
                max_workers=workers, initializer=init_worker, initargs=(str(runtime.data_dir),)
            ) as local_executor:
                _run_pool(local_executor)

        game_rows.sort(key=lambda row: int(row["simulation"]))
        player_rows.sort(key=lambda row: (int(row["simulation"]), str(row["team"]), str(row["player_name"])))
    else:
        for idx in range(simulations):
            runtime.reset_nan_input_log()
            syn_clones = [Player(p.name, p.position, p.player_id, p.attributes) for p in syn_players]
            home_clones = [Player(p.name, p.position, p.player_id, p.attributes) for p in home_players]
            away = Team(away_team, syn_clones, "away", team_features.loc[away_team], runtime)
            home = Team(home_team, home_clones, "home", team_features.loc[home_team], runtime)
            Game(away, home, roof, runtime).simulate_game()

            game_rows.append(
                {
                    "simulation": idx + 1,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": home.score,
                    "away_score": away.score,
                    "total": home.score + away.score,
                    "spread": home.score - away.score,
                }
            )
            for team in (home, away):
                for player in team.players:
                    row: Dict[str, object] = {
                        "simulation": idx + 1,
                        "team": team.name,
                        "player_name": player.name,
                        "player_id": player.player_id,
                        "position": player.position,
                    }
                    row.update(player.stats)
                    player_rows.append(row)
                player_rows.extend(team_unit_rows(team, idx + 1))

    game_frame = pd.DataFrame(game_rows)
    player_frame = add_fantasy_points(pd.DataFrame(player_rows))
    for frame in (game_frame, player_frame):
        frame.insert(0, "game_id", str(game_row["game_id"]))
        frame.insert(1, "week", int(game_row["week"]))
    return game_frame, player_frame


def run_all_teams_synthetic(
    season: int = DEFAULT_SEASON,
    week: int = 1,
    simulations: int = 200,
    workers: int = 4,
    output_dir: Optional[Path] = None,
    use_model_rank: bool = True,
    teams: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate all teams at home against a synthetic opponent."""
    runtime = SimulationRuntime()
    target_dir = output_dir or (WORKSPACE_DIR / f"synthetic_home_week_{week:02d}")
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data for season {season}, week {week}...")
    schedule = load_schedule(season)
    rosters = load_rosters(season)
    depth_charts = load_depth_charts(season)

    player_stats = runtime.player_stats_wide
    team_stats = runtime.team_stats_wide.copy()
    team_stats["game_date"] = pd.to_datetime(team_stats["game_date"], errors="coerce")
    fill_values = prior_fill_values(player_stats, tuple(runtime.required_player_feature_columns))

    syn_team_features = build_synthetic_team_features(runtime, rest_days=7.0)
    syn_players = build_synthetic_players(runtime)

    if teams:
        all_teams = [t.upper().strip() for t in teams]
    else:
        all_teams = sorted(set(schedule["home_team"].dropna().unique().tolist()))
        if not all_teams:
            all_teams = sorted(set(team_stats["team"].dropna().unique().tolist()))

    week_sched = schedule[schedule["week"].astype(int) == int(week)]
    default_gameday = (
        pd.Timestamp(week_sched["gameday"].max())
        if not week_sched.empty
        else pd.Timestamp(f"{season}-09-15")
    )

    print(f"Simulating {len(all_teams)} teams at home vs synthetic opponent ({simulations} sims each, {workers} workers)...")
    game_frames: List[pd.DataFrame] = []
    player_frames: List[pd.DataFrame] = []
    failures: List[str] = []

    def _execute_syn(executor: Optional[ProcessPoolExecutor]) -> None:
        for index, home_team in enumerate(all_teams, start=1):
            game_id = f"{season}_{week:02d}_SYN_{home_team}"
            matching = week_sched[(week_sched["home_team"] == home_team) | (week_sched["away_team"] == home_team)]
            gameday = pd.Timestamp(matching.iloc[0]["gameday"]) if not matching.empty else default_gameday

            print(f"[{index:02d}/{len(all_teams):02d}] SYN @ {home_team} (Roof: {ROOF_TYPE.get(home_team, 'outdoors')})...", end=" ", flush=True)

            try:
                home_feature_row = team_feature_snapshot(team_stats, home_team, gameday).copy()
            except ValueError as err:
                print(f"FAILED (features): {err}")
                failures.append(f"{home_team}: {err}")
                continue

            home_feature_row["days_rest"] = 7.0
            home_feature_row["opp_days_rest"] = 7.0

            team_features = pd.DataFrame(
                [home_feature_row, syn_team_features], index=[home_team, "SYN"]
            ).reindex(columns=runtime.required_team_feature_columns)

            roster_week = roster_for_team_week(rosters, home_team, week)
            if roster_week.empty:
                print(f"FAILED (no roster for week {week})")
                failures.append(f"{home_team}: no roster for week {week}")
                continue

            name_lookup = dict(zip(roster_week["player_id"], roster_week["player_name"]))
            order = depth_chart_order(depth_charts, roster_week, home_team, gameday)
            assignments = build_slot_assignments(order, player_stats, gameday, use_model_rank, fill_values)
            home_players = build_players(runtime, player_stats, assignments, name_lookup, gameday, {}, fill_values)

            game_row = pd.Series(
                {
                    "game_id": game_id,
                    "week": int(week),
                    "home_team": home_team,
                    "away_team": "SYN",
                    "gameday": gameday,
                }
            )

            game_frame, player_frame = run_single_team_simulation(
                runtime,
                game_row,
                team_features,
                home_players,
                syn_players,
                simulations,
                workers,
                executor=executor,
            )

            stem = sanitize_stem(game_id)
            game_frame.to_csv(target_dir / f"{stem}_games.csv", index=False)
            player_frame.to_csv(target_dir / f"{stem}_players.csv", index=False)
            game_frames.append(game_frame)
            player_frames.append(player_frame)

            median_h = float(np.median(game_frame["home_score"]))
            median_a = float(np.median(game_frame["away_score"]))
            win_p = float(np.mean(game_frame["home_score"] > game_frame["away_score"]))
            print(f"Done -> Median: {home_team} {median_h:.0f} - SYN {median_a:.0f} | Win Prob: {win_p:.1%}")

    if workers > 1 and simulations > 1:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=init_worker, initargs=(str(runtime.data_dir),)
        ) as shared_executor:
            _execute_syn(shared_executor)
    else:
        _execute_syn(None)

    if not game_frames:
        raise RuntimeError("No team simulations succeeded.")

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

    print("\n" + "=" * 80)
    print(f"SYNTHETIC OPPONENT HOME GAME SUMMARY (Season {season}, Week {week})")
    print("=" * 80)
    display_cols = [
        "home_team",
        "sim_home_median",
        "sim_away_median",
        "sim_spread",
        "sim_total",
        "sim_home_win_prob",
        "sim_spread_p25",
        "sim_spread_p75",
    ]
    summary_display = combined_summary[display_cols].copy()
    summary_display.columns = [
        "Team",
        "Team Pts",
        "Syn Pts",
        "Spread",
        "Total",
        "Win Prob",
        "Spread P25",
        "Spread P75",
    ]
    print(summary_display.to_string(index=False))
    print(f"\nSaved combined CSVs to: {target_dir}")

    if failures:
        print(f"\nWarnings/Skipped ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")

    return combined_games, combined_players, combined_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate all NFL teams at home against a synthetic (league-average) opponent."
    )
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON, help="NFL season (default: 2026)")
    parser.add_argument("--week", type=int, default=1, help="Regular season week (1-18, default: 1)")
    parser.add_argument("--simulations", type=int, default=200, help="Simulations per team (default: 200)")
    parser.add_argument("--workers", type=int, default=4, help="Process pool worker count (default: 4)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory (default: outputs/simulations/synthetic_home_week_<week>)",
    )
    parser.add_argument(
        "--ordering",
        choices=["model_rank", "depth_chart"],
        default="model_rank",
        help="Slot assignment method (default: model_rank)",
    )
    parser.add_argument(
        "--teams",
        nargs="+",
        default=None,
        help="Optional subset of team abbreviations to simulate (e.g. --teams KC BUF BAL)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_model_rank = args.ordering == "model_rank"
    run_all_teams_synthetic(
        season=args.season,
        week=args.week,
        simulations=args.simulations,
        workers=args.workers,
        output_dir=args.output_dir,
        use_model_rank=use_model_rank,
        teams=args.teams,
    )


if __name__ == "__main__":
    main()
