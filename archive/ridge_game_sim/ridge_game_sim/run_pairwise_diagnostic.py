from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from ridge_game_sim.bundle import validate_bundle, verify_bundle_hash
from ridge_game_sim.execution import execute_ridge_games, prepared_game
from ridge_game_sim.game_inputs import (
    actual_roof, make_players_from_assignments, pregame_team_snapshot,
    rest_before, slot_assignments,
)
from ridge_game_sim.paths import ROOT, output_dir, write_json
from ridge_game_sim.runtime import RidgeRuntime
from ridge_game_sim.synthetic_games import (
    _load_inputs, build_synthetic_game, synthetic_game_row,
)


def build_real_matchup(runtime, player_stats, team_stats, roster, depth, schedule,
                       home_team: str, away_team: str, week: int, simulations: int):
    home_row = synthetic_game_row(schedule, home_team, week)
    away_row = synthetic_game_row(schedule, away_team, week)
    kickoff = pd.Timestamp(home_row.gameday)
    features = {}
    players = {}

    for team in (home_team, away_team):
        team_row = pregame_team_snapshot(runtime, team_stats, team, kickoff).copy()
        team_row["days_rest"] = rest_before(schedule, team, kickoff)
        team_row["opp_days_rest"] = 7.0
        for column in runtime.required_team_feature_columns:
            if column.endswith("_venue_effect"):
                team_row[column] = 0.0
        for model in runtime.context_map:
            runtime.checked_context(model, team_row)
        features[team] = team_row

        team_kickoff = pd.Timestamp(home_row.gameday if team == home_team else away_row.gameday)
        assignments = slot_assignments(
            roster, depth, team, week, team_kickoff, player_stats, use_model_rank=True,
        )
        players[team] = make_players_from_assignments(
            runtime, player_stats, assignments, team_kickoff,
        )

    game_row = pd.Series({
        "game_id": f"{home_row.season}_{week:02d}_PAIR_{home_team}_{away_team}",
        "season": int(home_row.season), "week": week, "home_team": home_team,
        "away_team": away_team, "gameday": kickoff, "location": "Neutral",
    })
    frame = pd.DataFrame.from_dict(features, orient="index").reindex(
        index=[home_team, away_team], columns=runtime.required_team_feature_columns,
    )
    return prepared_game(
        game_row, frame, players[home_team], players[away_team], simulations,
        actual_roof(home_row), swap_home_away_halfway=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Compare a Ridge team against synthetic and real opponents")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, default=5)
    parser.add_argument("--team", default="ARI")
    parser.add_argument("--opponent", default="MIA")
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--parallelism", type=int, default=32)
    parser.add_argument("--backend", choices=("local", "modal"), default="modal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/runtime")
    args = parser.parse_args()
    if not 1 <= args.week <= 18 or args.simulations < 1 or args.parallelism < 1:
        parser.error("week, simulations and parallelism are outside valid ranges")

    validate_bundle(args.artifact_dir)
    bundle_hash = verify_bundle_hash(args.artifact_dir)
    runtime = RidgeRuntime(args.artifact_dir, strict_schema=True)
    player_stats, team_stats = runtime.player_stats_wide, runtime.team_stats_wide
    schedule, roster, depth = _load_inputs(args.season, args.artifact_dir / "weekly_inputs")
    team, opponent = args.team.upper(), args.opponent.upper()
    team_row = synthetic_game_row(schedule, team, args.week)

    games = [
        build_synthetic_game(
            runtime, player_stats, team_stats, roster, depth, schedule, team,
            team_row, args.simulations, use_model_rank=True,
        ),
        build_real_matchup(
            runtime, player_stats, team_stats, roster, depth, schedule,
            team, opponent, args.week, args.simulations,
        ),
    ]
    run_id = args.run_id or f"pairwise-{team}-vs-synthetic-and-{opponent}-{args.season}-w{args.week:02d}-{datetime.now():%Y%m%d-%H%M%S}"
    results = execute_ridge_games(
        games, args.artifact_dir, bundle_hash, backend=args.backend,
        parallelism=args.parallelism, run_id=run_id, seed=args.seed,
    )

    output = output_dir(ROOT / "outputs/synthetic" / run_id)
    summaries = []
    for game in games:
        games_frame, players_frame = results[game.game_id]
        games_frame.to_csv(output / f"{game.game_id}_games.csv", index=False)
        players_frame.to_csv(output / f"{game.game_id}_players.csv", index=False)
        team_scores = games_frame.home_score
        opponent_scores = games_frame.away_score
        ties = team_scores.eq(opponent_scores)
        wins = team_scores.gt(opponent_scores)
        summaries.append({
            "game_id": game.game_id, "team": game.home_team, "opponent": game.away_team,
            "simulations": len(games_frame), "wins": int(wins.sum()),
            "losses": int((team_scores < opponent_scores).sum()), "ties": int(ties.sum()),
            "win_probability_tie_adjusted": float((wins.sum() + 0.5 * ties.sum()) / len(games_frame)),
            "mean_team_score": float(team_scores.mean()),
            "mean_opponent_score": float(opponent_scores.mean()),
            "mean_spread": float(games_frame.spread.mean()),
            "median_spread": float(games_frame.spread.median()),
        })

    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    write_json(output / "run_manifest.json", {
        "run_id": run_id, "season": args.season, "week": args.week,
        "team": team, "opponent": opponent,
        "simulations_per_matchup": args.simulations, "backend": args.backend,
        "parallelism": args.parallelism, "bundle_hash": bundle_hash,
        "neutral_site": True, "home_away_swap_after_simulation": args.simulations // 2,
        "matchups": [game.game_id for game in games],
    })
    print(f"Wrote pairwise diagnostics to {output}")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()