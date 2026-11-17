from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from ridge_game_sim.paths import ROOT, assert_protected, digest, output_dir, protected_hashes, write_json


def main():
    parser = argparse.ArgumentParser(description="Download explicit ridge experiment inputs into its own cache")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2016, datetime.now().year + 1)))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/inputs")
    args = parser.parse_args()
    if min(args.years) < 2016 or max(args.years) > datetime.now().year:
        parser.error(f"Input years must be between 2016 and the current year ({datetime.now().year})")
    output = output_dir(args.output_dir)
    before = protected_hashes()
    try:
        import nfl_data_py as nfl
        from bayesian_feature_builder import load_pbp_data
        pbp = load_pbp_data(args.years, cache_path=output_dir(ROOT / "artifacts/cache/pbp"))
        schedule = nfl.import_schedules(args.years)
        schedule = schedule[schedule.game_type.eq("REG")].copy()
        schedule["game_date"] = schedule.gameday
        schedule["location"] = schedule.get("location", pd.Series("Home", index=schedule.index))
        venue = next((column for column in ("stadium_id", "game_stadium", "stadium") if column in schedule), None)
        if venue is None:
            raise ValueError("Schedule lacks actual stadium identities; supply enriched schedule CSV explicitly")
        roster = nfl.import_weekly_rosters(args.years)
        team_games = pd.concat([schedule[["game_id", "season", "week", "home_team"]].rename(columns={"home_team": "team"}), schedule[["game_id", "season", "week", "away_team"]].rename(columns={"away_team": "team"})])
        roster = roster.merge(team_games, on=["season", "week", "team"], how="inner", validate="many_to_one")
        injuries = nfl.import_injuries(args.years)
        injuries = injuries.drop_duplicates(["season", "week", "gsis_id"], keep="last").rename(columns={"gsis_id": "player_id"})
        roster = roster.merge(injuries[["season", "week", "player_id", "report_status"]], on=["season", "week", "player_id"], how="left")
        depth = nfl.import_depth_charts(args.years)
        if not {"season", "week", "gsis_id", "depth_team"} <= set(depth):
            raise ValueError("Depth chart schema lacks historical depth_team; supply pregame roster CSV explicitly")
        depth = depth.drop_duplicates(["season", "week", "gsis_id"], keep="first").rename(columns={"gsis_id": "player_id"})
        roster = roster.merge(depth[["season", "week", "player_id", "depth_team"]], on=["season", "week", "player_id"], how="left")
        from bayesian_feature_builder import load_snap_counts
        from ridge_game_sim.feature_builder import canonicalize_teams
        snaps = canonicalize_teams(load_snap_counts(args.years))
        required_snaps = {"game_type", "offense_snaps", "position", "gsis_id", "name", "team", "game_id"}
        if not required_snaps <= set(snaps):
            raise ValueError(f"Joined snap counts missing {sorted(required_snaps-set(snaps))}")
        roster["position"] = roster.position.replace({"FB": "RB"})
        roster["player_name"] = roster.get("full_name", roster.player_id)
        roster = roster.drop_duplicates(["game_id", "team", "player_id"])
        for name, frame in (("pbp.parquet", pbp), ("schedule.parquet", schedule), ("roster.parquet", roster), ("snap_counts.parquet", snaps)):
            frame.to_parquet(output / name, index=False)
        write_json(output / "input_manifest.json", {"years": args.years, "files": {name: digest(output/name) for name in ("pbp.parquet", "schedule.parquet", "roster.parquet", "snap_counts.parquet")}, "source": "nfl_data_py", "note": "Pregame roster/depth and actual stadium identity are required; unsupported upstream schemas fail explicitly"})
    finally:
        assert_protected(before)


if __name__ == "__main__":
    main()