"""Build the NFLSIM_TIMEOUTS settings from nflverse play-by-play: when teams call timeouts, and how long the
play ran before they did.

After a play whose clock keeps running (a completion, run or sack that neither scores nor changes possession), the
chance that the defense, or else the offense, calls a timeout right away. It depends on the window (the last two
minutes of a half, the rest of the last five minutes of the game, any other time) and on that side's margin, and
only counts plays where that side had a timeout left. ``play_seconds`` are quantiles of the time that ran off
before such timeouts.

    python -m nflsim.sim.timeout_settings --pbp-dir <folder of pbp_<season>.parquet> --seasons 2021-2025 \\
        --output nflsim/config/timeouts.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nflsim.paths import output_dir, write_json

COLUMNS = ["game_id", "season_type", "play_id", "play_type", "posteam", "defteam", "posteam_score", "defteam_score",
           "qtr", "half_seconds_remaining", "game_seconds_remaining", "timeout", "timeout_team", "complete_pass",
           "rush_attempt", "sack", "touchdown", "interception", "fumble_lost", "penalty",
           "posteam_timeouts_remaining", "defteam_timeouts_remaining"]
MARGIN_BUCKETS = ("down9+", "down1-8", "tied", "up1-8", "up9+")


def window(quarter, half_seconds, game_seconds) -> str:
    if quarter in (2, 4) and half_seconds <= 120:
        return "half_last2"
    if quarter == 4 and game_seconds <= 300:
        return "q4_last5"
    return "other"


def margin_bucket(margin: float) -> str:
    return "down9+" if margin <= -9 else "down1-8" if margin < 0 else "tied" if margin == 0 else "up1-8" if margin < 9 else "up9+"


def build_settings(pbp: pd.DataFrame, prior_strength: float = 20.0) -> dict:
    pbp = pbp[pbp.season_type.eq("REG")].sort_values(["game_id", "play_id"]).copy()
    following = pbp.groupby("game_id")
    pbp["next_timeout"] = following.timeout.shift(-1).eq(1)
    pbp["next_team"] = following.timeout_team.shift(-1)
    pbp["runoff"] = pbp.game_seconds_remaining - following.game_seconds_remaining.shift(-1)
    running = (pbp.complete_pass.eq(1) | pbp.rush_attempt.eq(1) | pbp.sack.eq(1)) & pbp.penalty.ne(1) \
        & pbp.touchdown.ne(1) & pbp.interception.ne(1) & pbp.fumble_lost.ne(1) & pbp.qtr.le(4)
    plays = pbp[running].copy()
    plays["window"] = [window(q, h, g) for q, h, g in zip(plays.qtr, plays.half_seconds_remaining, plays.game_seconds_remaining)]
    policy, counts = {}, {}
    for side, team, left, sign in (("defense", "defteam", "defteam_timeouts_remaining", -1),
                                   ("offense", "posteam", "posteam_timeouts_remaining", 1)):
        able = plays[plays[left].gt(0)]
        called = able.next_timeout & able.next_team.eq(able[team])
        margin = sign * (able.posteam_score - able.defteam_score)
        buckets = [margin_bucket(value) for value in margin]
        for name in ("half_last2", "q4_last5", "other"):
            in_window = able.window.eq(name).to_numpy()
            base = float(called[in_window].mean()) if in_window.any() else 0.0
            for bucket in MARGIN_BUCKETS:
                rows = in_window & np.array([b == bucket for b in buckets])
                k, n = float(called[rows].sum()), float(rows.sum())
                policy[f"{name}|{side}|{bucket}"] = round((k + prior_strength * base) / (n + prior_strength), 4)
                counts[f"{name}|{side}|{bucket}"] = int(n)
    stopped = plays[plays.next_timeout & plays.runoff.between(1, 40)].runoff
    timeouts = pbp[pbp.timeout.eq(1) & pbp.qtr.le(4)]
    return {"source": "nflverse play-by-play, regular season", "per_half": 3, "overtime": 2, "policy": policy,
            "plays_seen": counts, "play_seconds": [float(value) for value in np.quantile(stopped, np.linspace(0, 1, 21))],
            "per_game": {"timeouts": round(len(timeouts) / pbp.game_id.nunique(), 2)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pbp-dir", type=Path, required=True)
    parser.add_argument("--seasons", default="2021-2025")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first, _, last = args.seasons.partition("-")
    seasons = range(int(first), int(last or first) + 1)
    pbp = pd.concat([pd.read_parquet(args.pbp_dir / f"pbp_{season}.parquet", columns=COLUMNS) for season in seasons],
                    ignore_index=True)
    settings = {"seasons": list(seasons), **build_settings(pbp)}
    output_dir(args.output.parent)
    write_json(args.output, settings)
    print("timeouts per game", settings["per_game"]["timeouts"], "play seconds quantiles", settings["play_seconds"][::5])
    print(pd.Series(settings["policy"]).to_string())


if __name__ == "__main__":
    main()
