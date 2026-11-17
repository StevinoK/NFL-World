"""Spread and win odds for NFL games from Kalshi and Polymarket, read as the market's own numbers.

Both list a ladder of "team wins by more than X" contracts instead of one line, so the spread is where that ladder
crosses 50%: the home margin the market thinks is as likely to be beaten as not. Spread and win odds are home-team
numbers, matching the schedule's spread_line. Before kickoff the prices are live quotes; once a game starts, prices
move with the score, so the last hourly price before kickoff is used instead. Thin markets, with a bid-ask gap wider
than 10 cents, are ignored.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

from nflsim.sim.kalshi_compare import EASTERN, MAX_WIDTH, candles, get, kalshi_team, price_points

POLYMARKET = "https://gamma-api.polymarket.com"
POLYMARKET_HISTORY = "https://clob.polymarket.com/prices-history"
POLYMARKET_TEAM = {"LAR": "LA"}
HISTORY_SECONDS = 7 * 86400
# A crossing between strikes further apart than this says little about the line, so it's left blank.
MAX_BRACKET = 3.0
EMPTY = {"spread": np.nan, "home_win": np.nan}
Point = Tuple[float, float]


def ladder_spread(points: Iterable[Point]) -> float:
    """Home margin where P(home margin > m) crosses 50%, from (m, probability) points."""
    frame = pd.DataFrame(list(points), columns=["margin", "probability"]).groupby("margin").probability.mean().sort_index()
    if len(frame) < 2:
        return float("nan")
    margins, values = frame.index.to_numpy(dtype=float), frame.to_numpy(dtype=float)
    # Quotes are noisy, but the true curve only falls as the margin rises.
    values = (np.minimum.accumulate(values) + np.maximum.accumulate(values[::-1])[::-1]) / 2
    below = np.flatnonzero(values <= 0.5)
    if len(below) == 0 or below[0] == 0:
        return float("nan")
    high = below[0]
    low = high - 1
    if margins[high] - margins[low] > MAX_BRACKET:
        return float("nan")
    if values[low] == values[high]:
        return float((margins[low] + margins[high]) / 2)
    return float(margins[low] + (values[low] - 0.5) / (values[low] - values[high]) * (margins[high] - margins[low]))


def midpoint(bid, ask) -> Optional[float]:
    if bid is None or ask is None:
        return None
    bid, ask = float(bid), float(ask)
    return (bid + ask) / 2 if 0 < bid <= ask < 1 and ask - bid <= MAX_WIDTH + 1e-9 else None


def _home_win(home: Optional[float], away: Optional[float]) -> float:
    if home is None and away is None:
        return float("nan")
    if home is None:
        return 1 - away
    return home if away is None else home / (home + away)


# ------------------------------------------------------------------------ Kalshi


def _kalshi_markets(series: str, key: str) -> List[dict]:
    return get("/markets", event_ticker=f"{series}-{key}", limit=200).get("markets", [])


def _kalshi_prices(markets: List[dict], kickoff: pd.Timestamp, now: pd.Timestamp) -> Dict[str, float]:
    prices = {}
    if now < kickoff:
        for market in markets:
            value = midpoint(market.get("yes_bid_dollars"), market.get("yes_ask_dollars"))
            if value is not None:
                prices[market["ticker"]] = value
        return prices
    end = int(kickoff.timestamp())
    for ticker, history in candles([market["ticker"] for market in markets], end - HISTORY_SECONDS, end).items():
        close = price_points(history, end)
        if close:
            prices[ticker] = (close["close_bid"] + close["close_ask"]) / 2
    return prices


def kalshi_game(home: str, away: str, day: pd.Timestamp, kickoff: pd.Timestamp, now: pd.Timestamp) -> Dict[str, float]:
    stamp = day.strftime("%y%b%d").upper()
    home_code, away_code = kalshi_team(home), kalshi_team(away)
    # Neutral-site games can list the two teams in either order.
    for key in (f"{stamp}{away_code}{home_code}", f"{stamp}{home_code}{away_code}"):
        moneyline = _kalshi_markets("KXNFLGAME", key)
        if moneyline:
            break
    else:
        return dict(EMPTY)
    spreads = _kalshi_markets("KXNFLSPREAD", key)
    prices = _kalshi_prices(moneyline + spreads, kickoff, now)
    wins = {market["ticker"].rsplit("-", 1)[1]: prices.get(market["ticker"]) for market in moneyline}
    home_win = _home_win(wins.get(home_code), wins.get(away_code))
    points: List[Point] = [(0.0, home_win)] if np.isfinite(home_win) else []
    for market in spreads:
        price = prices.get(market["ticker"])
        strike = market.get("floor_strike")
        if price is None or strike is None:
            continue
        tail = market["ticker"].rsplit("-", 1)[1]
        if re.fullmatch(rf"{home_code}\d+", tail):
            points.append((float(strike), price))
        elif re.fullmatch(rf"{away_code}\d+", tail):
            points.append((-float(strike), 1 - price))
    return {"spread": ladder_spread(points), "home_win": home_win}


# -------------------------------------------------------------------- Polymarket


def _polymarket_get(url: str, **params):
    for attempt in range(4):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code != 429 and response.status_code < 500:
                response.raise_for_status()
                return response.json()
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(1 + 2 * attempt)
    raise requests.ConnectionError(f"Polymarket request failed after retries: {url}")


def _polymarket_event(slugs: Sequence[str]) -> Tuple[Optional[dict], Optional[str]]:
    for slug in slugs:
        events = _polymarket_get(f"{POLYMARKET}/events", slug=slug)
        if events:
            return events[0], slug
    return None, None


def _polymarket_history(token: str, kickoff: pd.Timestamp) -> Optional[float]:
    end = int(kickoff.timestamp())
    payload = _polymarket_get(POLYMARKET_HISTORY, market=token, startTs=end - HISTORY_SECONDS, endTs=end, fidelity=60)
    history = [row for row in payload.get("history", []) if row["t"] <= end]
    return float(history[-1]["p"]) if history else None


def _polymarket_price(market: dict, kickoff: pd.Timestamp, now: pd.Timestamp) -> Optional[float]:
    """Price of the market's first listed outcome."""
    if now < kickoff and not market.get("closed"):
        return midpoint(market.get("bestBid"), market.get("bestAsk"))
    return _polymarket_history(json.loads(market["clobTokenIds"])[0], kickoff)


def polymarket_points(markets: List[dict], home_name: str, away_name: str, price_of) -> Tuple[List[Point], float]:
    """Ladder points and home win probability from an event's moneyline and spread markets."""
    points, home_win = [], float("nan")
    for market in markets:
        kind = market.get("sportsMarketType")
        if kind not in ("moneyline", "spreads"):
            continue
        price = price_of(market)
        if price is None:
            continue
        outcomes = json.loads(market["outcomes"])
        if kind == "moneyline":
            if home_name in outcomes:
                home_win = price if outcomes[0] == home_name else 1 - price
                points.append((0.0, home_win))
            continue
        named = re.match(r"Spread: (.+?) \(", market.get("question", ""))
        line = market.get("line")
        if not named or line is None or named.group(1) not in outcomes:
            continue
        # Price of "named team covers line": P(named margin > -line).
        covers = price if outcomes[0] == named.group(1) else 1 - price
        if named.group(1) == home_name:
            points.append((-float(line), covers))
        elif named.group(1) == away_name:
            points.append((float(line), 1 - covers))
    return points, home_win


def polymarket_game(home: str, away: str, day: pd.Timestamp, kickoff: pd.Timestamp, now: pd.Timestamp) -> Dict[str, float]:
    # Slugs carry the UTC kickoff date, so night games land on the next day.
    dates = list(dict.fromkeys([kickoff.strftime("%Y-%m-%d"), day.strftime("%Y-%m-%d")]))
    first, second = (POLYMARKET_TEAM.get(team, team).lower() for team in (away, home))
    event, slug = _polymarket_event([f"nfl-{a}-{b}-{date}" for a, b in ((first, second), (second, first)) for date in dates])
    if event is None:
        return dict(EMPTY)
    markets = event.get("markets", [])
    moneyline = next((market for market in markets if market.get("sportsMarketType") == "moneyline"), None)
    if moneyline is None:
        return dict(EMPTY)
    names = json.loads(moneyline["outcomes"])
    # Outcomes are listed in slug order, so the names follow the teams in the slug.
    away_name, home_name = names if slug.startswith(f"nfl-{first}-{second}-") else names[::-1]
    points, home_win = polymarket_points(markets, home_name, away_name, lambda market: _polymarket_price(market, kickoff, now))
    return {"spread": ladder_spread(points), "home_win": home_win}


# ------------------------------------------------------------------------- games


def _game_row(row: dict, now: pd.Timestamp) -> Tuple[dict, List[str]]:
    day = pd.Timestamp(str(row["gameday"])[:10])
    kickoff = pd.Timestamp(f"{str(row['gameday'])[:10]} {row['gametime']}").tz_localize(EASTERN).tz_convert("UTC")
    output, failed = {"game_id": row["game_id"]}, []
    for name, read in (("kalshi", kalshi_game), ("polymarket", polymarket_game)):
        try:
            result = read(row["home_team"], row["away_team"], day, kickoff, now)
        except (requests.RequestException, RuntimeError, KeyError, ValueError) as error:
            failed.append(f"{name}: {type(error).__name__}")
            result = EMPTY
        output[f"{name}_spread"], output[f"{name}_home_win"] = result["spread"], result["home_win"]
    return output, failed


def market_lines(games: pd.DataFrame, now: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Per-game Kalshi and Polymarket spread and home win odds, plus any read errors.

    games needs game_id, home_team, away_team, gameday and gametime (Eastern).
    """
    now = now if now is not None else pd.Timestamp(time.time(), unit="s", tz="UTC")
    rows = games.dropna(subset=["gameday", "gametime"]).to_dict("records")
    if not rows:
        return pd.DataFrame(columns=["game_id", "kalshi_spread", "kalshi_home_win", "polymarket_spread", "polymarket_home_win"]), []
    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(lambda row: _game_row(row, now), rows))
    return pd.DataFrame([row for row, _ in results]), sorted({error for _, errors in results for error in errors})
