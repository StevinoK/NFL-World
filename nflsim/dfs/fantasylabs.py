"""Finished DraftKings contests from FantasyLabs' contests dashboard: real ownership, duplication and results.

For each slate in a date range, the largest contest is saved under data/fantasylabs/nfl/<date>_<contest_id>/:

- contest.json: name, size, entry fee, prizes, duplicate and unique lineup counts.
- players.csv: one row per player and roster slot (CPT or FLEX) with salary, FantasyLabs' projection, actual
  ownership and actual DraftKings points.
- lineups.parquet: one row per distinct lineup with the captain, five FLEX players, how many entries and users
  played it, its points, rank and payout.
- user_lineups.parquet: which user entered which lineup and how many times.

The files come from the endpoints the dashboard itself reads; FantasyLabs' terms apply, so keep them local.

    python -m nflsim.dfs.fantasylabs --start 2026-09-04 --end 2026-11-07 --showdown-only
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from nflsim.paths import ROOT

SERVICE = "https://service.fantasylabs.com"
FILES = "https://dh5nxc6yx3kwy.cloudfront.net/contests/nfl/{date_id}/{contest_id}/{part}/"
OUTPUT = ROOT.parent / "data/fantasylabs/nfl"
HEADERS = {"User-Agent": "nflsim-research/1.0"}
PAUSE = 1.0
SLOTS = ["CPT1", "FLEX1", "FLEX2", "FLEX3", "FLEX4", "FLEX5"]


def _get(url: str, **params) -> dict:
    response = requests.get(url, params=params, headers=HEADERS, timeout=120)
    response.raise_for_status()
    time.sleep(PAUSE)
    return response.json()


def slates(start: date, end: date, showdown_only: bool = False) -> pd.DataFrame:
    """Every DraftKings slate whose contests start between `start` and `end`."""
    rows = {}
    day = start
    while day <= end:
        for source in _get(f"{SERVICE}/contest-sources/", sport_id=1, date=day.isoformat())["contest-sources"]:
            if not source["source_name"].startswith("draftkings.com"):
                continue
            for group in source["draft_groups"]:
                rows[group["id"]] = {"group_id": group["id"], "start": group["contest_start_date"], "slate": source["display_name"],
                                     "suffix": group["contest_suffix"], "games": group["game_count"],
                                     "showdown": "showdown" in source["source_name"]}
        day += timedelta(days=1)
    table = pd.DataFrame(rows.values(), columns=["group_id", "start", "slate", "suffix", "games", "showdown"])
    table = table[pd.to_datetime(table.start).dt.date.between(start, end)]
    return table[table.showdown] if showdown_only else table


def largest_contest(group_id: int) -> dict | None:
    contests = _get(f"{SERVICE}/live-contests/", sport="NFL", contest_group_id=group_id)["live_contests"]
    return max(contests, key=lambda contest: contest["contest_size"]) if contests else None


def players_table(data: dict) -> pd.DataFrame:
    rows = []
    for key, player in data["players"].items():
        rows.append({"key": key, "player_id": player["playerId"], "name": player["fullName"], "team": player["currentTeam"],
                     "position": player["position"], "slot": "CPT" if player["rosterPosition"] == "CPT" else "FLEX",
                     "salary": player["salary"], "projected_points": player["projPoints"],
                     "ownership": player["ownership"] / 100, "actual_points": player["actualPoints"],
                     "home": player["homeVisitor"] == "Home", "favorite": player["favDog"] == "Favorite"})
    return pd.DataFrame(rows).sort_values(["slot", "salary"], ascending=[True, False]).reset_index(drop=True)


def lineups_table(lineups: dict) -> pd.DataFrame:
    rows = []
    for lineup in lineups["lineups"].values():
        slots = lineup["lineupPlayers"]
        rows.append({"lineup": lineup["lineupHash"], **{slot.lower(): slots.get(slot) for slot in SLOTS},
                     "entries": lineup["lineupCt"], "users": lineup["lineupUserCt"], "points": lineup["points"],
                     "salary": lineup["totalSalary"], "rank": lineup["lineupRank"], "payout": lineup.get("payout", 0.0),
                     "total_ownership": lineup["totalOwn"] / 100})
    return pd.DataFrame(rows).rename(columns={"cpt1": "cpt"}).sort_values("rank").reset_index(drop=True)


def user_lineups_table(data: dict) -> pd.DataFrame:
    rows = [{"user": user, "lineup": lineup["lineupHash"], "entries": lineup["lineupCt"]}
            for user, details in data["users"].items() for lineup in details["lineups"].values()]
    return pd.DataFrame(rows)


def save_contest(contest: dict, output: Path = OUTPUT) -> Path:
    folder = output / f"{contest['date_id']}_{contest['contest_id']}"
    if (folder / "lineups.parquet").is_file():
        return folder
    fetch = lambda part: _get(FILES.format(date_id=contest["date_id"], contest_id=contest["contest_id"], part=part))
    data, lineups = fetch("data"), fetch("lineups")
    folder.mkdir(parents=True, exist_ok=True)
    meta = {**data["contest"], "contest_group_id": contest["contest_group_id"], "date_id": contest["date_id"]}
    (folder / "contest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    players_table(data).to_csv(folder / "players.csv", index=False)
    user_lineups_table(data).to_parquet(folder / "user_lineups.parquet", index=False)
    lineups_table(lineups).to_parquet(folder / "lineups.parquet", index=False)  # Written last: marks the folder complete.
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Save finished DraftKings contests from FantasyLabs")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(days=1))
    parser.add_argument("--showdown-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    table = slates(args.start, args.end, args.showdown_only)
    print(f"{len(table)} slates from {args.start} to {args.end}", flush=True)
    for slate in table.sort_values("start").itertuples():
        contest = largest_contest(slate.group_id)
        if contest is None:
            print(f"  {slate.start} {slate.slate} {slate.suffix}: no contests listed", flush=True)
            continue
        try:
            folder = save_contest(contest, args.output)
        except requests.HTTPError as error:
            print(f"  {slate.start} {slate.suffix}: {contest['contest_name']} failed ({error})", flush=True)
            continue
        print(f"  {slate.start} {slate.slate} {slate.suffix}: {contest['contest_name']}, {contest['contest_size']:,} entries -> {folder.name}",
              flush=True)


if __name__ == "__main__":
    main()
