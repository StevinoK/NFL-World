"""Score a saved backtest's joint outcomes against Kalshi same-game parlays.

Kalshi parlays are user-built combos of single-game markets that pay only if every leg settles in
the chosen direction; most trade once, as a fill at creation. This scans the settled combos whose
legs all come from one backtest game, takes the last trade before kickoff as the market's price, and
prices the same combo as the share of simulations in which every leg hits. Two independence
references show how much of each price is correlation: the model's own leg probabilities multiplied
together, and Kalshi's closing leg prices multiplied together.

    python -m nflsim.sim.kalshi_parlays --benchmark nflsim/benchmarks/2026/rebuild-a04940-2026-w1-7-200
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from nflsim.paths import ROOT, output_dir
from nflsim.sim.kalshi_compare import (SERIES, backtest_games, fee, game_record, get, load_simulations, log_loss,
                                       market_outcomes, name_index, smooth)

COMBO_SERIES = ("KXMVECROSSCATEGORY", "KXMVESPORTSMULTIGAMEEXTENDED")
PRICE_BUCKETS = (0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.5, 1.0)


def scan_combos(games: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Settled combos closing in the backtest window whose legs all belong to one backtest game."""
    if path.exists():
        return pd.read_parquet(path)
    keys = set(games.event_key)
    low = int(games.kickoff.min().timestamp()) - 86400
    high = int(games.kickoff.max().timestamp()) + 12 * 3600
    # Progress is checkpointed so a dropped connection resumes from the last page instead of restarting.
    progress = path.with_suffix(".progress.json")
    state = json.loads(progress.read_text()) if progress.exists() else {"done": [], "series": None, "cursor": None, "rows": [], "scanned": 0}
    rows, scanned = state["rows"], state["scanned"]
    for series in COMBO_SERIES:
        if series in state["done"]:
            continue
        cursor = state["cursor"] if state["series"] == series else None
        while True:
            page = get("/markets", series_ticker=series, status="settled", limit=1000, min_close_ts=low,
                       max_close_ts=high, **({"cursor": cursor} if cursor else {}))
            for market in page.get("markets", []):
                legs = market.get("mve_selected_legs") or []
                events = {leg["event_ticker"].split("-", 1)[1] for leg in legs}
                if legs and len(events) == 1 and events <= keys and all(leg["event_ticker"].startswith("KXNFL") for leg in legs):
                    rows.append({"ticker": market["ticker"], "event_key": events.pop(), "legs": json.dumps(legs),
                                 "n_legs": len(legs), "result": market.get("result"),
                                 "volume": float(market.get("volume_fp") or 0),
                                 "last_price": float(market.get("last_price_dollars") or "nan"),
                                 "created": market.get("created_time")})
            scanned += len(page.get("markets", []))
            cursor = page.get("cursor")
            print(f"  {series}: scanned {scanned:,}, kept {len(rows):,}", end="\r", flush=True)
            if not cursor:
                state["done"].append(series)
            state.update(series=series, cursor=cursor, rows=rows, scanned=scanned)
            if not cursor or scanned % 50000 == 0:
                progress.write_text(json.dumps(state))
            if not cursor:
                break
    print()
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)
    return frame


def pregame_fill(ticker: str, kickoff: pd.Timestamp) -> dict:
    trades = get("/markets/trades", ticker=ticker, limit=1000).get("trades", [])
    before = [trade for trade in trades if pd.Timestamp(trade["created_time"]) < kickoff]
    if not before:
        return {"ticker": ticker, "trades": len(trades), "pregame_trades": 0}
    last = max(before, key=lambda trade: trade["created_time"])
    contracts = sum(float(trade["count_fp"]) for trade in before)
    return {"ticker": ticker, "trades": len(trades), "pregame_trades": len(before), "pregame_contracts": contracts,
            "price": float(last["yes_price_dollars"]), "fill_time": last["created_time"]}


def pregame_fills(combos: pd.DataFrame, games: pd.DataFrame, path: Path, workers: int) -> pd.DataFrame:
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["ticker"])
    todo = combos[~combos.ticker.isin(done.ticker)]
    kickoff = games.set_index("event_key").kickoff
    fresh = []
    with ThreadPoolExecutor(workers) as pool:
        for count, row in enumerate(pool.map(lambda item: pregame_fill(item.ticker, kickoff[item.event_key]),
                                             todo.itertuples()), 1):
            fresh.append(row)
            if count % 2000 == 0 or count == len(todo):
                done = pd.concat([done, pd.DataFrame(fresh)], ignore_index=True)
                done.to_parquet(path, index=False)
                fresh = []
                print(f"  fills {count:,}/{len(todo):,}", end="\r", flush=True)
    print()
    return done


def price_combos(combos: pd.DataFrame, singles: pd.DataFrame, games: pd.DataFrame, benchmark: Path,
                 roster: pd.DataFrame) -> pd.DataFrame:
    names = name_index(roster)
    singles = singles.set_index("ticker")
    known = set(singles.index)
    by_key = games.set_index("event_key").game_id
    out = []
    for key, rows in combos.groupby("event_key"):
        game_id = by_key[key]
        game = game_record(games, game_id)
        scores, players = load_simulations(benchmark, game_id)
        cache: dict[str, np.ndarray | None] = {}
        for combo in rows.itertuples():
            legs = json.loads(combo.legs)
            if not all(leg["market_ticker"] in known for leg in legs):
                continue
            joint = np.ones(len(scores))
            independent, market_independent = 1.0, 1.0
            for leg in legs:
                ticker = leg["market_ticker"]
                if ticker not in cache:
                    market = SimpleNamespace(**singles.loc[ticker].to_dict(), ticker=ticker)
                    cache[ticker] = market_outcomes(market, game, scores, players, names)[0]
                outcome = cache[ticker]
                if outcome is None:
                    break
                outcome = outcome if leg["side"] == "yes" else 1 - outcome
                joint = joint * outcome
                independent *= smooth(outcome.sum(), len(outcome))
                mid = (singles.at[ticker, "close_bid"] + singles.at[ticker, "close_ask"]) / 2
                market_independent *= mid if leg["side"] == "yes" else 1 - mid
            else:
                out.append({"ticker": combo.ticker, "game_id": game_id, "n_legs": combo.n_legs,
                            "price": combo.price, "y": float(combo.result == "yes"),
                            "model": smooth(joint.sum(), len(joint)), "model_hits": joint.sum(),
                            "model_independent": independent, "market_independent": market_independent,
                            "families": ",".join(sorted({SERIES[singles.at[leg['market_ticker'], 'series']][0] for leg in legs}))})
    return pd.DataFrame(out)


def clustered_mean(values: pd.Series, clusters: pd.Series) -> tuple[float, float]:
    mean = values.mean()
    resid = (values - mean).groupby(clusters.values).sum()
    return mean, float(np.sqrt((resid ** 2).sum()) / len(values))


def summarize(priced: pd.DataFrame, edge: float) -> dict:
    priced = priced.copy()
    priced["bucket"] = pd.cut(priced.price, PRICE_BUCKETS, include_lowest=True)
    calibration = priced.groupby("bucket", observed=True).agg(
        combos=("y", "size"), games=("game_id", "nunique"), kalshi=("price", "mean"), model=("model", "mean"),
        model_independent=("model_independent", "mean"), market_independent=("market_independent", "mean"),
        hit_rate=("y", "mean")).reset_index()
    calibration["bucket"] = calibration.bucket.astype(str)
    scores = {}
    for label, column in (("kalshi", "price"), ("model", "model"), ("model_independent", "model_independent"),
                          ("market_independent", "market_independent")):
        scores[f"logloss_{label}"] = float(log_loss(priced[column], priced.y).mean())
        scores[f"brier_{label}"] = float(((priced[column] - priced.y) ** 2).mean())
    diff = log_loss(priced.model, priced.y) - log_loss(priced.price, priced.y)
    scores["logloss_model_minus_kalshi"], scores["logloss_model_minus_kalshi_se"] = clustered_mean(diff, priced.game_id)
    diff = log_loss(priced.model, priced.y) - log_loss(priced.model_independent, priced.y)
    scores["logloss_joint_minus_independent"], scores["logloss_joint_minus_independent_se"] = clustered_mean(diff, priced.game_id)
    diff = log_loss(priced.price, priced.y) - log_loss(priced.market_independent, priced.y)
    scores["logloss_kalshi_minus_leg_product"], scores["logloss_kalshi_minus_leg_product_se"] = clustered_mean(diff, priced.game_id)
    # Correlation lift: how far each joint price sits above its own independence product, in log odds-free ratio.
    lift = np.log(priced.model / priced.model_independent)
    market_lift = np.log(priced.price / priced.market_independent)
    scores["mean_log_lift_model"], scores["mean_log_lift_kalshi"] = float(lift.mean()), float(market_lift.mean())
    scores["lift_corr"] = float(np.corrcoef(lift, market_lift)[0, 1])
    # Buy YES at the fill price (plus fee) when the model's edge clears ``edge``; buy NO likewise.
    yes_cost = priced.price + fee(priced.price)
    no_cost = (1 - priced.price) + fee(1 - priced.price)
    bets = []
    for side, cost, win, take in (("yes", yes_cost, priced.y, priced.model - yes_cost >= edge),
                                  ("no", no_cost, 1 - priced.y, (1 - priced.model) - no_cost >= edge)):
        if take.any():
            profit = (win - cost)[take]
            roi = profit.sum() / cost[take].sum()
            resid = (profit - roi * cost[take]).groupby(priced.game_id[take].values).sum()
            bets.append({"side": side, "bets": int(take.sum()), "roi": float(roi),
                         "roi_se": float(np.sqrt((resid ** 2).sum()) / cost[take].sum())})
    return {"combos": len(priced), "games": int(priced.game_id.nunique()), **scores,
            "calibration": calibration.to_dict("records"), "betting": bets}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=30000, help="combos to fetch pregame fills for")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--edge", type=float, default=0.03)
    parser.add_argument("--roster", type=Path, default=None)
    args = parser.parse_args()
    benchmark = args.benchmark.resolve()
    cache = output_dir(ROOT / "outputs/kalshi/cache" / f"{benchmark.parent.name}_combos")
    output = output_dir(ROOT / "outputs/kalshi" / benchmark.name)
    games = backtest_games(benchmark)
    combos = scan_combos(games, cache / "combos.parquet")
    singles = pd.read_csv(output / "markets.csv")
    # Only combos whose every leg is a single market we have quotes and simulations for can be priced.
    usable = set(singles.ticker[singles.close_bid.notna()])
    combos["priceable"] = combos.legs.map(lambda legs: all(leg["market_ticker"] in usable for leg in json.loads(legs)))
    print(f"{len(combos):,} same-game combos; {combos.priceable.sum():,} with every leg priceable")
    sample = combos[combos.priceable].sample(min(args.sample, int(combos.priceable.sum())), random_state=0)
    fills = pregame_fills(sample, games, cache / "fills.parquet", args.workers)
    sample = sample.merge(fills[fills.pregame_trades > 0][["ticker", "price"]], on="ticker")
    roster_path = args.roster or sorted((ROOT / "artifacts/bundles").glob("*/weekly_inputs/roster.parquet"))[-1]
    priced = price_combos(sample, singles, games, benchmark, pd.read_parquet(roster_path))
    priced = priced[priced.price.between(0.001, 0.999)]
    priced.to_csv(output / "parlays_scored.csv", index=False)
    summary = summarize(priced, args.edge)
    summary["by_legs"] = {legs: {key: value for key, value in summarize(part, args.edge).items() if key != "calibration"}
                          for legs, part in priced.groupby(priced.n_legs.clip(upper=6)) if len(part) >= 200}
    (output / "parlays_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({key: value for key, value in summary.items() if key not in ("calibration", "by_legs")}, indent=1, default=float))
    print(pd.DataFrame(summary["calibration"]).round(4).to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
