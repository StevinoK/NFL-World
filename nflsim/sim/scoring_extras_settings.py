"""Build the NFLSIM_SCORING_EXTRAS settings (two-point tries, safeties, onside kicks) from nflverse play-by-play.

Rates by score margin are shrunk toward their window's overall rate (``prior_strength`` pseudo-tries), so a margin
seen a handful of times doesn't decide a sure two-point try. Windows: two-point tries split at the 4th quarter
("early", "late"); onside kicks by the kicking team's margin in the last five minutes of the 4th ("last5"), the
rest of the 4th ("q4") and any other time ("other").

    python -m nflsim.sim.scoring_extras_settings --pbp-dir <folder of pbp_<season>.parquet> --seasons 2021-2025 \\
        --output nflsim/config/scoring_extras.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nflsim.paths import output_dir, write_json

COLUMNS = ["game_id", "season_type", "play_id", "play_type", "posteam", "posteam_score", "defteam_score", "qtr",
           "game_seconds_remaining", "two_point_attempt", "two_point_conv_result", "extra_point_attempt", "pass",
           "kickoff_attempt", "own_kickoff_recovery", "desc", "safety", "yardline_100"]
SCRIMMAGE = ["pass", "run", "qb_kneel", "qb_spike", "no_play", "field_goal", "punt"]


def shrunk_rates(flags: pd.Series, margins: pd.Series, max_margin: int, prior_strength: float) -> dict:
    base = float(flags.mean()) if len(flags) else 0.0
    grouped = flags.groupby(margins.clip(-max_margin, max_margin).round().astype(int)).agg(["sum", "size"])
    table = {str(margin): round(float((row["sum"] + prior_strength * base) / (row["size"] + prior_strength)), 4)
             for margin, row in grouped.iterrows()}
    return {**table, "default": round(base, 4)}


def onside_window(quarter, seconds_left) -> str:
    """Seconds left in the game (regulation)."""
    return "last5" if quarter == 4 and seconds_left <= 300 else "q4" if quarter == 4 else "other"


def next_start(pbp: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """The team and own yardline of the first scrimmage snap after each of ``plays``."""
    snaps = pbp[pbp.play_type.isin(SCRIMMAGE)][["game_id", "play_id", "posteam", "yardline_100"]].sort_values(["game_id", "play_id"])
    merged = pd.merge_asof(plays.sort_values("play_id")[["game_id", "play_id"]].assign(after=lambda f: f.play_id + 1e-6),
                           snaps.rename(columns={"play_id": "snap"}).sort_values("snap"),
                           left_on="after", right_on="snap", by="game_id", direction="forward")
    return merged.assign(start=100 - merged.yardline_100)


def build_settings(pbp: pd.DataFrame, max_margin: int = 21, prior_strength: float = 10.0) -> dict:
    pbp = pbp[pbp.season_type.eq("REG")].copy()
    tries = pbp[pbp.two_point_attempt.eq(1) | pbp.extra_point_attempt.eq(1)]
    margin = tries.posteam_score - tries.defteam_score        # after the six points
    two = tries.two_point_attempt.eq(1)
    late = tries.qtr >= 4
    attempts = tries[two]
    passing = attempts["pass"].eq(1)
    made = attempts.two_point_conv_result.eq("success")
    kickoffs = pbp[pbp.kickoff_attempt.eq(1)].copy()
    kickoffs["onside"] = kickoffs.desc.str.contains("onside", case=False, na=False)
    kicker_margin = kickoffs.defteam_score - kickoffs.posteam_score   # nflverse kickoffs: posteam receives
    window = pd.Series([onside_window(q, s) for q, s in zip(kickoffs.qtr, kickoffs.game_seconds_remaining)], index=kickoffs.index)
    onside = kickoffs[kickoffs.onside]
    starts = next_start(pbp, onside).merge(onside[["game_id", "play_id", "posteam"]].rename(columns={"posteam": "receiver"}),
                                           on=["game_id", "play_id"])
    recovered = onside.own_kickoff_recovery.eq(1).to_numpy()
    kept = starts.posteam.ne(starts.receiver)  # the kicking team snapped next (re-kicks after penalties included)
    safeties = next_start(pbp, pbp[pbp.safety.eq(1)])
    return {
        "source": "nflverse play-by-play, regular season",
        "max_margin": max_margin,
        "go_for_two": {"early": shrunk_rates(two[~late].astype(float), margin[~late], max_margin, prior_strength),
                       "late": shrunk_rates(two[late].astype(float), margin[late], max_margin, prior_strength)},
        "pass_share": round(float(passing.mean()), 4),
        "pass_success": round(float(made[passing].mean()), 4),
        "rush_success": round(float(made[~passing].mean()), 4),
        "safety_free_kick_start": int(round(float(safeties.start.mean()))),
        "onside": {
            **{name: shrunk_rates(kickoffs.onside[window.eq(name)].astype(float), kicker_margin[window.eq(name)],
                                  max_margin, prior_strength) for name in ("last5", "q4", "other")},
            "recovery": round(float(recovered.mean()), 4),
            "recovered_start": int(round(float(starts.start[kept.to_numpy()].median()))),
            "lost_start": int(round(float(starts.start[~kept.to_numpy()].median()))),
        },
        "per_game": {"two_point_tries": round(float(two.sum() / pbp.game_id.nunique()), 3),
                     "onside_kicks": round(float(len(onside) / pbp.game_id.nunique()), 3),
                     "safeties": round(float(pbp.safety.eq(1).sum() / pbp.game_id.nunique()), 3)},
    }


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
    print({key: settings[key] for key in ("pass_share", "pass_success", "rush_success", "safety_free_kick_start", "per_game")})
    print("onside recovery", settings["onside"]["recovery"], "starts", settings["onside"]["recovered_start"], settings["onside"]["lost_start"])


if __name__ == "__main__":
    main()
