"""Score a saved backtest against Kalshi NFL markets (game lines, alt ladders and player props).

Kalshi lists every line as a binary "over strike" market: moneyline, spread and total ladders,
team totals, and player ladders for yards, receptions, completions, attempts and touchdowns.
For each settled market this pulls hourly bid/ask candles, takes the first quote that is at most
``MAX_WIDTH`` wide as the opening price and the last such quote at or before kickoff as the
closing price, and prices the same event from the backtest's simulations. The outcome is
Kalshi's own settlement; markets settled at a fair price (player active without a snap) drop out.

    python -m nflsim.sim.kalshi_compare --benchmark nflsim/benchmarks/2026/rebuild-a04940-2026-w1-7-200
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from nflsim.names import display_name
from nflsim.paths import ROOT, output_dir

API = "https://api.elections.kalshi.com/trade-api/v2"
EASTERN = ZoneInfo("America/New_York")
MAX_WIDTH = 0.10
BATCH = 20
MAX_CANDLES = 5000
LOOKBACK_DAYS = 14

# series -> (family, simulated stat); game-level families price from the game score frame.
SERIES = {
    "KXNFLGAME": ("moneyline", None),
    "KXNFLSPREAD": ("spread", None),
    "KXNFLTOTAL": ("total", None),
    "KXNFLTEAMTOTAL": ("team_total", None),
    "KXNFLPASSYDS": ("passing_yards", ("passing_yards",)),
    "KXNFLRSHYDS": ("rushing_yards", ("rushing_yards",)),
    "KXNFLRECYDS": ("receiving_yards", ("receiving_yards",)),
    "KXNFLRRYDS": ("rush_rec_yards", ("rushing_yards", "receiving_yards")),
    "KXNFLREC": ("receptions", ("receptions",)),
    "KXNFLPASSCOMP": ("completions", ("completions",)),
    "KXNFLPASSATT": ("pass_attempts", ("pass_attempts",)),
    "KXNFLRSHATT": ("rush_attempts", ("rushing_attempts",)),
    "KXNFLPASSTDS": ("passing_tds", ("passing_touchdowns",)),
    "KXNFLPASSINT": ("interceptions", ("interceptions",)),
    "KXNFLTD": ("anytime_td", ("rushing_touchdowns", "receiving_touchdowns")),
    "KXNFLWINMARGIN": ("win_margin", None),
    "KXNFLBOTH": ("both_score", None),
    # Protected props settle on the same stat; a player hurt in the first half settles at the pregame price.
    "KXNFLPASSYDSPROT": ("passing_yards_protected", ("passing_yards",)),
    "KXNFLRSHYDSPROT": ("rushing_yards_protected", ("rushing_yards",)),
    "KXNFLRECYDSPROT": ("receiving_yards_protected", ("receiving_yards",)),
    "KXNFLRECPROT": ("receptions_protected", ("receptions",)),
    "KXNFLPASSTDSPROT": ("passing_tds_protected", ("passing_touchdowns",)),
    "KXNFLTDPROT": ("anytime_td_protected", ("rushing_touchdowns", "receiving_touchdowns")),
}
KALSHI_TEAM = {"JAX": "JAC", "LA": "LAR"}
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def get(path: str, **params) -> dict:
    for attempt in range(10):
        try:
            response = requests.get(API + path, params=params, timeout=60)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
            time.sleep(2 + 2 * attempt)
            continue
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(1 + attempt)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"Kalshi request failed after retries: {path} {params}")


def series_events(series: str) -> list[str]:
    events, cursor = [], None
    while True:
        page = get("/events", series_ticker=series, limit=200, **({"cursor": cursor} if cursor else {}))
        events += [event["event_ticker"] for event in page.get("events", [])]
        cursor = page.get("cursor")
        if not cursor:
            return events


def kalshi_team(team: str) -> str:
    return KALSHI_TEAM.get(team, team)


def backtest_games(benchmark: Path) -> pd.DataFrame:
    games = pd.DataFrame(json.loads((benchmark / "run_manifest.json").read_text())["identity"]["games"])
    local = pd.to_datetime(games.gameday.str[:10] + " " + games.gametime)
    games["kickoff"] = [stamp.tz_localize(EASTERN).tz_convert(timezone.utc) for stamp in local]
    games["event_key"] = (local.dt.strftime("%y%b%d").str.upper()
                          + games.away_team.map(kalshi_team) + games.home_team.map(kalshi_team))
    return games[["game_id", "week", "home_team", "away_team", "kickoff", "event_key"]]


def candles(tickers: list[str], start: int, end: int) -> dict[str, list]:
    # The batch endpoint refuses requests spanning too many candles in total.
    start = max(start, end - LOOKBACK_DAYS * 86400)
    size = max(1, min(BATCH, MAX_CANDLES // max(1, (end - start) // 3600)))
    out = {}
    for offset in range(0, len(tickers), size):
        chunk = tickers[offset:offset + size]
        page = get("/markets/candlesticks", market_tickers=",".join(chunk), start_ts=start, end_ts=end, period_interval=60)
        out.update({row["market_ticker"]: row["candlesticks"] for row in page.get("markets", [])})
    return out


def quote(candle) -> tuple[float, float] | None:
    bid, ask = candle["yes_bid"].get("close_dollars"), candle["yes_ask"].get("close_dollars")
    if bid is None or ask is None:
        return None
    bid, ask = float(bid), float(ask)
    return (bid, ask) if 0 < bid <= ask < 1 and ask - bid <= MAX_WIDTH + 1e-9 else None


def price_points(history: list, kickoff: int) -> dict:
    quoted = [(candle["end_period_ts"], quote(candle)) for candle in history if candle["end_period_ts"] <= kickoff]
    quoted = [(stamp, value) for stamp, value in quoted if value]
    if not quoted:
        return {}
    (open_ts, (open_bid, open_ask)), (close_ts, (close_bid, close_ask)) = quoted[0], quoted[-1]
    return {"open_ts": open_ts, "open_bid": open_bid, "open_ask": open_ask,
            "close_ts": close_ts, "close_bid": close_bid, "close_ask": close_ask}


def fetch(games: pd.DataFrame, cache: Path) -> pd.DataFrame:
    """Settled Kalshi markets for the backtest games with open/close quotes; cached per event."""
    cache = output_dir(cache)
    keys = dict(zip(games.event_key, games.itertuples()))
    rows = []
    for series in SERIES:
        events = [event for event in series_events(series) if event.split("-", 1)[1] in keys]
        print(f"{series}: {len(events)} events")
        for event in events:
            path = cache / f"{event}.json"
            if not path.exists():
                game = keys[event.split("-", 1)[1]]
                markets = get("/markets", event_ticker=event, limit=1000)["markets"]
                kickoff = int(game.kickoff.timestamp())
                start = min(int(pd.Timestamp(market["open_time"]).timestamp()) for market in markets) if markets else kickoff
                history = candles([market["ticker"] for market in markets], start, kickoff)
                path.write_text(json.dumps({"markets": markets, "candles": history, "kickoff": kickoff}))
            payload = json.loads(path.read_text())
            game = keys[event.split("-", 1)[1]]
            for market in payload["markets"]:
                rows.append({"series": series, "event": event, "game_id": game.game_id, "week": game.week,
                             "ticker": market["ticker"], "subtitle": market.get("yes_sub_title") or "",
                             "strike": market.get("floor_strike"), "strike_type": market.get("strike_type"),
                             "settlement": float(market.get("settlement_value_dollars") or "nan"),
                             "volume": float(market.get("volume_fp") or 0),
                             **price_points(payload["candles"].get(market["ticker"], []), payload["kickoff"])})
    return pd.DataFrame(rows)


def normal(name: str) -> str:
    name = re.sub(r"[^a-z ]", "", str(name).lower().replace("-", " "))
    return " ".join(SUFFIX.sub("", name).split())


def name_index(roster: pd.DataFrame) -> dict[str, set[str]]:
    """player_id -> normalized names under nickname, legal first name and full-name spellings."""
    index: dict[str, set[str]] = {}
    for row in roster.dropna(subset=["player_id"]).to_dict("records"):
        names = {display_name(row.get("football_name"), row.get("last_name")),
                 display_name(None, row.get("last_name"), row.get("first_name")), row.get("full_name") or ""}
        index.setdefault(str(row["player_id"]), set()).update(normal(name) for name in names if name)
    return index


def subject(row, game) -> tuple[str, str | None]:
    """(ticker tail, team) for game-level markets: the team a moneyline/spread/margin market is about."""
    tail = row.ticker.split("-")[2] if row.ticker.count("-") >= 2 else ""
    team = next((team for team in (game.home_team, game.away_team) if tail.startswith(kalshi_team(team))), None)
    return tail, team


def exceeds(values: np.ndarray, market) -> np.ndarray:
    return values >= market.strike if market.strike_type == "greater_or_equal" else values > market.strike


def market_outcomes(market, game, scores: pd.DataFrame, players: pd.DataFrame, names: dict) -> tuple[np.ndarray | None, str]:
    """Per-simulation YES outcome (1, 0, or 0.5 for a moneyline tie) and the market's subject."""
    family, stats = SERIES[market.series]
    home, away = scores.home.to_numpy(), scores.away.to_numpy()
    if family == "total":
        return exceeds(home + away, market).astype(float), ""
    if family == "both_score":
        return exceeds(np.minimum(home, away), market).astype(float), ""
    if stats is None:
        tail, team = subject(market, game)
        if family == "win_margin" and tail == "TIE":
            return (home == away).astype(float), "TIE"
        if team is None:
            return None, ""
        margin = home - away if team == game.home_team else away - home
        if family == "moneyline":
            return (margin > 0) + 0.5 * (margin == 0), team
        if family == "spread":
            return exceeds(margin, market).astype(float), team
        if family == "win_margin":
            bucket = tail[len(kalshi_team(team)):]
            low, high = (int(bucket[:-4]), np.inf) if bucket.endswith("PLUS") else map(int, bucket.split("TO"))
            return ((margin >= low) & (margin <= high)).astype(float), team
        return exceeds(home if team == game.home_team else away, market).astype(float), team
    player = normal(market.subtitle.split(":")[0])
    ids = [pid for pid in players.player_id.unique() if player in names.get(pid, ())]
    if len(ids) != 1:
        return None, player
    values = players[players.player_id == ids[0]].set_index("simulation")[list(stats)].sum(axis=1)
    values = values.reindex(scores.simulation, fill_value=0).to_numpy()
    return exceeds(values, market).astype(float), ids[0]


def smooth(hits: float, count: int) -> float:
    return (hits + 0.5) / (count + 1)


def load_simulations(benchmark: Path, game_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(benchmark / "raw_games" / f"{game_id}.csv").sort_values("simulation")
    players = pd.read_csv(benchmark / "raw_games" / f"{game_id}_players.csv")
    players["player_id"] = players.player_id.astype(str)
    return scores, players


def game_record(games: pd.DataFrame, game_id: str) -> pd.Series:
    game = games.set_index("game_id").loc[game_id].copy()
    game["game_id"] = game_id
    return game


def price_markets(markets: pd.DataFrame, games: pd.DataFrame, benchmark: Path, roster: pd.DataFrame) -> pd.DataFrame:
    names = name_index(roster)
    out = []
    for game_id, rows in markets.groupby("game_id"):
        game = game_record(games, game_id)
        scores, players = load_simulations(benchmark, game_id)
        for market in rows.itertuples():
            outcomes, key = market_outcomes(market, game, scores, players, names)
            probability = np.nan if outcomes is None else smooth(outcomes.sum(), len(outcomes))
            out.append({**market._asdict(), "model": probability, "subject": key})
    frame = pd.DataFrame(out).drop(columns="Index")
    frame["family"] = frame.series.map(lambda series: SERIES[series][0])
    return frame


def log_loss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fee(price):
    # Kalshi taker fee per contract: ceil(7% * P * (1 - P)) cents.
    return np.ceil(7 * price * (1 - price) - 1e-9) / 100


def clustered(values: pd.Series, clusters: pd.Series) -> tuple[float, float]:
    sums = values.groupby(clusters.values).sum()
    counts = values.groupby(clusters.values).size()
    mean = sums.sum() / counts.sum()
    resid = (values - mean).groupby(clusters.values).sum()
    return mean, float(np.sqrt((resid ** 2).sum()) / counts.sum())


def summarize(frame: pd.DataFrame, edge: float = 0.05) -> pd.DataFrame:
    scored = frame[frame.settlement.isin([0.0, 1.0]) & frame.model.notna() & frame.close_bid.notna()].copy()
    scored["y"] = scored.settlement
    scored["open_mid"] = (scored.open_bid + scored.open_ask) / 2
    scored["close_mid"] = (scored.close_bid + scored.close_ask) / 2
    # The main line per subject is the rung priced closest to even at close (the book's O/U line).
    scored["main"] = False
    distance = (scored.close_mid - 0.5).abs()
    group = ["game_id", "series", "subject"]
    scored.loc[distance.groupby([scored[column] for column in group]).idxmin().values, "main"] = True
    rows = []
    subsets = [(family, label, part) for family, rungs in scored.groupby("family")
               for label, part in (("all rungs", rungs), ("main line", rungs[rungs.main]))]
    for family, subset, part in subsets:
        record = {"family": family, "subset": subset, "markets": len(part), "games": part.game_id.nunique()}
        for label, column in (("model", "model"), ("open", "open_mid"), ("close", "close_mid")):
            record[f"brier_{label}"] = ((part[column] - part.y) ** 2).mean()
            record[f"logloss_{label}"] = log_loss(part[column], part.y).mean()
        for label, column in (("open", "open_mid"), ("close", "close_mid")):
            diff = (part.model - part.y) ** 2 - (part[column] - part.y) ** 2
            record[f"brier_vs_{label}"], record[f"brier_vs_{label}_se"] = clustered(diff, part.game_id)
        moved = part.close_mid - part.open_mid
        lean = part.model - part.open_mid
        active = moved.abs() >= 0.01
        record["moved_toward_model"] = (np.sign(moved[active]) == np.sign(lean[active])).mean() if active.any() else np.nan
        record["lean_move_corr"] = np.corrcoef(lean, moved)[0, 1] if moved.std() > 0 and lean.std() > 0 else np.nan
        # Take the side the model prefers at the closing ask when its edge after fees exceeds ``edge``.
        yes_cost = part.close_ask + fee(part.close_ask)
        no_cost = (1 - part.close_bid) + fee(1 - part.close_bid)
        buy_yes = part.model - yes_cost >= edge
        buy_no = (1 - part.model) - no_cost >= edge
        profit = pd.concat([(part.y - yes_cost)[buy_yes], ((1 - part.y) - no_cost)[buy_no]])
        cost = pd.concat([yes_cost[buy_yes], no_cost[buy_no]])
        record["bets"] = len(profit)
        record["roi"] = profit.sum() / cost.sum() if len(profit) else np.nan
        if len(profit):
            _, record["roi_se"] = clustered(profit, part.game_id.loc[profit.index])
            record["roi_se"] = record["roi_se"] * len(profit) / cost.sum()
        rows.append(record)
    return pd.DataFrame(rows), scored


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--roster", type=Path, default=None, help="roster parquet with football_name/first/last names")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--edge", type=float, default=0.05)
    args = parser.parse_args()
    benchmark = args.benchmark.resolve()
    season = benchmark.parent.name
    output = output_dir(args.output or ROOT / "outputs/kalshi" / benchmark.name)
    games = backtest_games(benchmark)
    markets = fetch(games, ROOT / "outputs/kalshi/cache" / season)
    roster_path = args.roster or sorted((ROOT / "artifacts/bundles").glob("*/weekly_inputs/roster.parquet"))[-1]
    priced = price_markets(markets, games, benchmark, pd.read_parquet(roster_path))
    summary, scored = summarize(priced, args.edge)
    priced.to_csv(output / "markets.csv", index=False)
    scored.to_csv(output / "scored.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    coverage = priced.assign(priced_ok=priced.model.notna(), settled=priced.settlement.isin([0.0, 1.0]),
                             quoted=priced.close_bid.notna())
    coverage.groupby("family")[["priced_ok", "settled", "quoted"]].agg(["sum", "size"]).to_csv(output / "coverage.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.4f}".format):
        print(summary.to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
