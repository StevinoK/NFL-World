from __future__ import annotations

import json
import math

import pandas as pd

from nflsim.dashboard.power_ratings import Source, TEAMS, fetch_ratings, parse_inpredictable, parse_nfelo, parse_sagarin, parse_teamrankings
from nflsim import ui
from nflsim.dashboard.synthetic_report import summary_table_html
from nflsim.sim.market_lines import ladder_spread, midpoint, polymarket_points


def test_ladder_spread_finds_the_fifty_percent_margin():
    assert ladder_spread([(3.5, 0.6), (6.5, 0.4)]) == 5.0
    assert ladder_spread([(-2.5, 0.7), (0.0, 0.5), (2.5, 0.3)]) == 0.0


def test_ladder_spread_handles_noise_and_blanks_what_it_cannot_read():
    assert 4.5 < ladder_spread([(1.5, 0.68), (2.5, 0.665), (3.5, 0.67), (4.5, 0.55), (5.5, 0.53), (6.5, 0.47)]) < 6.5
    assert math.isnan(ladder_spread([(1.5, 0.7), (2.5, 0.6)]))
    assert math.isnan(ladder_spread([(1.5, 0.4), (2.5, 0.3)]))
    assert math.isnan(ladder_spread([(0.0, 0.62), (7.5, 0.4)]))


def test_midpoint_skips_thin_and_empty_quotes():
    assert midpoint("0.70", "0.72") == 0.71
    assert midpoint("0.50", "0.70") is None
    assert midpoint(None, "0.70") is None
    assert midpoint("0", "0.05") is None


def market(kind, question, outcomes, line=None):
    return {"sportsMarketType": kind, "question": question, "outcomes": json.dumps(outcomes), "line": line}


def test_polymarket_points_put_both_teams_on_the_home_margin():
    markets = [
        market("moneyline", "Bucs vs. Bears", ["Buccaneers", "Bears"]),
        market("spreads", "Spread: Bears (-6.5)", ["Bears", "Buccaneers"], -6.5),
        market("spreads", "Spread: Buccaneers (-1.5)", ["Buccaneers", "Bears"], -1.5),
        market("totals", "O/U 44.5", ["Over", "Under"], 44.5),
    ]
    prices = {"Bucs vs. Bears": 0.3, "Spread: Bears (-6.5)": 0.48, "Spread: Buccaneers (-1.5)": 0.26, "O/U 44.5": 0.5}
    points, home_win = polymarket_points(markets, "Bears", "Buccaneers", lambda row: prices[row["question"]])
    assert round(home_win, 2) == 0.7
    assert sorted((margin, round(price, 2)) for margin, price in points) == [(-1.5, 0.74), (0.0, 0.7), (6.5, 0.48)]


def page(*rows):
    return "<table>" + "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows) + "</table>"


def test_rating_pages_are_read_and_team_codes_are_ours():
    nfelo, _ = parse_nfelo(page(["", "Team", "nfelo", "QB Adj", "Value"], ["1", "Raiders logo Raiders OAK 4-4", "1500", "+1", "+6.0"]))
    assert nfelo == {"LV": 6.0}
    gpf, as_of = parse_inpredictable("As of November 06, 2026 " + page(["Rank", "Team"], ["1", "JAC", "", "46,49", "5.3", "1.1,2.2", "5.7"]))
    assert gpf == {"JAX": 5.3} and as_of == "2026-11-06"
    assert parse_teamrankings(page(["Rank", "Team", "Rating"], ["4", "LA Rams (4-4)", "4.6"]))[0] == {"LA": 4.6}
    sagarin, _ = parse_sagarin("""NFL 2026 ratings through games of November 2 Monday - Week 8
   1  San Francisco 49ers     =  24.30    6   1   0   18.70(  30)   2  0  0 |   3  0  0 |   25.56    1 |   25.91    1 |
   2  Washington              =  20.00    6   1   0   18.70(  30)   2  0  0 |   3  0  0 |   19.50    9 |   25.91    1 |
   1  San Francisco 49ers     =  99.00    6   1   0   18.70(  30)   2  0  0 |   3  0  0 |   99.00    1 |   25.91    1 |
""")
    assert sagarin == {"SF": 25.56, "WAS": 19.5}


def test_ratings_are_centered_and_a_bad_source_is_only_noted():
    good = {team: float(index) for index, team in enumerate(sorted(TEAMS))}

    def broken():
        raise ValueError("page changed")

    frame, notes = fetch_ratings((
        Source("Good", "https://a.example", lambda: (good, "2026-11-07")),
        Source("Short", "https://b.example", lambda: ({"KC": 1.0}, "")),
        Source("Broken", "https://c.example", broken),
    ))
    assert list(frame.columns) == ["Good"] and abs(frame.Good.mean()) < 1e-9 and len(frame) == 32
    assert [item["label"] for item in notes["failed"]] == ["Short", "Broken"]


def test_table_shows_ratings_for_a_run_that_spells_the_rams_lar():
    summary = pd.DataFrame([{"team": "LAR", "mean_spread": 3.7, "win_prob": 0.6, "points_for": 24.0, "points_against": 20.0}])
    styles = pd.DataFrame(columns=["team_color", "team_color2", "team_logo_espn"])
    html = summary_table_html(summary, styles, pd.DataFrame({"FPI": {"LA": 4.4, "KC": 1.0}}), None)
    assert "+4.4" in html and ui.BLANK not in html
