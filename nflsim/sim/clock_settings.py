"""Build the overtime and kneel-down settings the sim reads (nflsim/config/clock_fixes.json by default).

Kneels: the chance the offense kneels, from 2021-2025 regular-season snaps (runs, passes and kneels) in the last 200 seconds
of a half.
  - Fourth quarter, offense ahead: by clock slack, the seconds left beyond 40 per remaining down after this one
    (slack <= 0 means kneeling can run out the clock unless the defense has timeouts; the sim has no timeouts, so the
    rate is averaged over them).
  - Second quarter, any score: by seconds left and field position.
Elapsed seconds per kneel come from the same snaps (the clock until the next snap, capped at the clock left).
Overtime: ten minutes, with the regular-season rule from 2025 on (both teams get a possession).

    python -m nflsim.sim.clock_settings [output json]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nflsim.features.bayesian_history import load_pbp_data

SEASONS = [2021, 2022, 2023, 2024, 2025]
SLACK_EDGES = [-1000, -20, 0, 10, 20, 40, 1000]
Q2_SECONDS = [0, 10, 20, 30, 45, 60, 200]
Q2_YARDS = [0, 40, 60, 80, 100]  # yards to the end zone


def rate_table(frame, column, edges):
    groups = frame.groupby(pd.cut(frame[column], edges, right=True), observed=False).kneel
    return [{"upper": float(interval.right), "rate": round(float(g.mean()), 4) if len(g) >= 30 else 0.0, "n": int(len(g))}
            for interval, g in groups]


def main():
    pbp = load_pbp_data(SEASONS)
    pbp = pbp[pbp.season_type.eq("REG")].sort_values(["game_id", "play_id"]).copy()
    pbp["next_seconds"] = pbp.groupby("game_id").game_seconds_remaining.shift(-1)
    snaps = pbp[pbp.play_type.isin(["run", "pass", "qb_kneel"]) & pbp.down.notna()].copy()
    snaps["kneel"] = snaps.qb_kneel.eq(1).astype(float)
    snaps["lead"] = snaps.posteam_score - snaps.defteam_score
    late = snaps[snaps.half_seconds_remaining.le(200)]
    q4 = late[late.qtr.eq(4) & late.lead.gt(0)].copy()
    q4["slack"] = q4.half_seconds_remaining - 40 * (4 - q4.down)
    q2 = late[late.qtr.eq(2)]
    q2_table = []
    for (low, high) in zip(Q2_YARDS[:-1], Q2_YARDS[1:]):
        cell = q2[q2.yardline_100.gt(low) & q2.yardline_100.le(high)]
        q2_table.append({"yards_upper": high, "by_seconds": rate_table(cell, "half_seconds_remaining", Q2_SECONDS)})
    kneels = pbp[pbp.qb_kneel.eq(1)]
    elapsed = (kneels.game_seconds_remaining - kneels.next_seconds).dropna().clip(0, 45).round().astype(int)
    counts = elapsed.value_counts().sort_index()
    games = pbp.game_id.nunique()
    settings = {
        "seasons": SEASONS,
        "q4_lead_by_slack": rate_table(q4, "slack", SLACK_EDGES),
        "q2_by_yards_and_seconds": q2_table,
        "kneel_elapsed": {"seconds": counts.index.tolist(), "share": (counts / counts.sum()).round(5).tolist()},
        "kneel_yards": -1,
        "overtime_seconds": 600,
        "observed_per_game": {"kneels": round(len(kneels) / games, 3),
                              "overtime_share": round(float(pbp.groupby("game_id").qtr.max().ge(5).mean()), 4),
                              "final_ties": round(float(pbp.groupby("game_id")[["total_home_score", "total_away_score"]].apply(
                                  lambda g: g.total_home_score.iloc[-1] == g.total_away_score.iloc[-1]).mean()), 4)},
    }
    Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "nflsim" / "config" / "clock_fixes.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in settings.items() if k != "kneel_elapsed"}, indent=1))


if __name__ == "__main__":
    main()
