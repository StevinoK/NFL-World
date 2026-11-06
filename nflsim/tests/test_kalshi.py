from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from nflsim.sim.kalshi_compare import kalshi_team, market_outcomes, normal, price_points

GAME = pd.Series({"game_id": "2026_04_PIT_CLE", "home_team": "CLE", "away_team": "PIT"})
SCORES = pd.DataFrame({"simulation": [1, 2, 3, 4], "home": [24, 17, 20, 10], "away": [17, 17, 35, 28]})
PLAYERS = pd.DataFrame({"simulation": [1, 2, 4], "player_id": "P1", "receiving_yards": [60, 59, 120],
                        "rushing_yards": [0, 5, 0], "receptions": [6, 4, 9]})
NAMES = {"P1": {"dk metcalf"}}


def market(series, tail, strike=None, strike_type="greater", subtitle=""):
    return SimpleNamespace(series=series, ticker=f"{series}-26OCT01PITCLE-{tail}", strike=strike,
                           strike_type=strike_type, subtitle=subtitle)


def outcomes(*args, **kwargs):
    return market_outcomes(market(*args, **kwargs), GAME, SCORES, PLAYERS, NAMES)[0].tolist()


def test_game_markets_follow_kalshi_strike_semantics():
    assert outcomes("KXNFLGAME", "CLE") == [1, 0.5, 0, 0]
    assert outcomes("KXNFLSPREAD", "PIT8", 7.5) == [0, 0, 1, 1]
    assert outcomes("KXNFLTOTAL", "41", 40.5) == [1, 0, 1, 0]
    assert outcomes("KXNFLBOTH", "17", 17, "greater_or_equal") == [1, 1, 1, 0]
    assert outcomes("KXNFLWINMARGIN", "CLE1TO6") == [0, 0, 0, 0]
    assert outcomes("KXNFLWINMARGIN", "CLE7TO14") == [1, 0, 0, 0]
    assert outcomes("KXNFLWINMARGIN", "PIT15PLUS") == [0, 0, 1, 1]
    assert outcomes("KXNFLWINMARGIN", "TIE") == [0, 1, 0, 0]


def test_player_props_match_names_and_count_missing_simulations_as_zero():
    assert outcomes("KXNFLRECYDS", "PITDMETCALF14-60", 59.5, subtitle="DK Metcalf: 60+") == [1, 0, 0, 1]
    assert outcomes("KXNFLRRYDS", "PITDMETCALF14-60", 59.5, subtitle="D.K. Metcalf: 60+") == [1, 1, 0, 1]
    assert market_outcomes(market("KXNFLREC", "X", 5.5, subtitle="Someone Else: 6+"), GAME, SCORES, PLAYERS, NAMES)[0] is None
    assert normal("Kenneth Walker III") == "kenneth walker" and kalshi_team("LA") == "LAR"


def test_parlay_joint_probability_multiplies_legs_within_each_simulation():
    win = np.array(outcomes("KXNFLGAME", "PIT"))
    catches = np.array(outcomes("KXNFLREC", "PITDMETCALF14-6", 5.5, subtitle="DK Metcalf: 6+"))
    assert (win * catches).tolist() == [0, 0, 0, 1]
    assert (win * (1 - catches)).tolist() == [0, 0.5, 1, 0]


def test_price_points_use_first_and_last_tight_quote_before_kickoff():
    candle = lambda ts, bid, ask: {"end_period_ts": ts, "yes_bid": {"close_dollars": bid}, "yes_ask": {"close_dollars": ask}}
    history = [candle(1, "0.0000", "0.9700"), candle(2, "0.4000", "0.4400"), candle(3, "0.5000", "0.5200"),
               candle(4, "0.9900", "1.0000")]
    assert price_points(history, kickoff=3) == {"open_ts": 2, "open_bid": 0.4, "open_ask": 0.44,
                                                "close_ts": 3, "close_bid": 0.5, "close_ask": 0.52}
