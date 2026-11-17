from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def rank_disagreements(results: pd.DataFrame, simulations: int = 200, limit: int = 20) -> pd.DataFrame:
    required = {
        "game_id", "week", "home_team", "away_team", "spread_line", "total_line",
        "sim_spread", "sim_total", "simulations_run",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Results missing required columns: {sorted(missing)}")
    frame = results.copy()
    for column in ("spread_line", "total_line", "sim_spread", "sim_total", "simulations_run"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame.simulations_run.eq(simulations)].dropna(
        subset=["spread_line", "total_line", "sim_spread", "sim_total"]
    ).copy()
    if frame.empty:
        raise ValueError(f"No completed games with {simulations} simulations and both Vegas lines")
    frame["spread_disagreement_points"] = (frame.sim_spread - frame.spread_line).abs()
    frame["total_disagreement_points"] = (frame.sim_total - frame.total_line).abs()
    spread_larger = frame.spread_disagreement_points.ge(frame.total_disagreement_points)
    frame["largest_disagreement_market"] = np.where(spread_larger, "Spread", "Total")
    frame["largest_disagreement_points"] = frame[["spread_disagreement_points", "total_disagreement_points"]].max(axis=1)
    columns = [
        "game_id", "week", "home_team", "away_team", "largest_disagreement_market",
        "largest_disagreement_points", "spread_line", "sim_spread", "spread_disagreement_points",
        "total_line", "sim_total", "total_disagreement_points", "simulations_run",
    ]
    return (frame.sort_values(["largest_disagreement_points", "game_id"], ascending=[False, True])
            .head(limit).loc[:, columns].reset_index(drop=True).assign(rank=lambda data: data.index + 1)
            .reindex(columns=["rank", *columns]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank completed backtest games by disagreement with Vegas spread/total lines")
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    results = pd.read_csv(args.results)
    ranked = rank_disagreements(results, args.simulations, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output, index=False)
    print(f"Ranked {len(ranked)} games from {ranked.simulations_run.nunique()} simulation count(s): {args.output}")
    print(ranked.to_string(index=False))


if __name__ == "__main__":
    main()