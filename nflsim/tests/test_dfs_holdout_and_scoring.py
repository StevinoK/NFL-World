from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from nflsim.dfs.optimizer import draft_our_entries, portfolio_results, split_simulations
from nflsim.engine.fantasy import add_fantasy_points
from nflsim.engine.simulation_runtime import Game


def test_split_simulations_is_disjoint_and_complete():
    pick, judge = split_simulations(101, holdout=0.5, seed=3)
    assert len(np.intersect1d(pick, judge)) == 0
    assert sorted(np.concatenate([pick, judge]).tolist()) == list(range(101))
    assert np.array_equal(split_simulations(101, seed=3)[0], pick)


def test_portfolio_results_win_share_and_profit():
    scores = np.array([[10, 1, 10, 1], [5, 5, 5, 5], [4, 4, 4, 4]], dtype=np.float32)
    result = portfolio_results(scores, [0], [1, 2])
    assert result["win_share"] == 0.5 and result["simulations"] == 4
    payouts = pd.DataFrame({"min_place": [1], "max_place": [1], "cash": [10.0]})
    with_cash = portfolio_results(scores, [0], [1, 2], payouts, entry_fee=1.0, field_size=3)
    assert with_cash["expected_profit"] == 0.5 * 9 + 0.5 * -1


def test_drafting_uses_only_the_picking_simulations():
    # Lineup 0 wins only in columns 0-1, lineup 1 only in columns 2-3; the field is lineup 2.
    scores = np.array([[9, 9, 0, 0], [0, 0, 9, 9], [5, 5, 5, 5]], dtype=np.float32)
    lineups = pd.DataFrame({"Players": [["A (CPT)"], ["B (CPT)"], ["C (CPT)"]], "Salary": [1, 1, 1]})
    player_map = {name: {"Roster Position": "CPT"} for name in ("A (CPT)", "B (CPT)", "C (CPT)")}
    entries = pd.DataFrame({"Entry ID": ["1"], "Contest ID": ["1"], "CPT": [""]})
    for columns, expected in ((np.array([0, 1]), 0), (np.array([2, 3]), 1)):
        _, drafted, _ = draft_our_entries(entries, scores, [2], lineups, player_map, pick_columns=columns)
        assert drafted.lineup_index.tolist() == [expected]


def test_yardage_bonuses_start_at_the_threshold():
    frame = pd.DataFrame({"position": ["QB1", "RB1", "WR1"], "passing_yards": [300, 0, 0], "rushing_yards": [0, 100, 0],
                          "receiving_yards": [0, 0, 99]})
    assert add_fantasy_points(frame).fantasy_points.round(2).tolist() == [15.0, 13.0, 9.9]


def test_points_allowed_removes_the_opponents_return_touchdowns():
    home = SimpleNamespace(score=24, stats={"defensive_touchdown": 1})
    away = SimpleNamespace(score=10, stats={"defensive_touchdown": 0})
    Game.update_points_allowed(SimpleNamespace(team1=home, team2=away))
    # Home allowed the away team's 10; away allowed 24 less the 7 that home's defense scored.
    assert home.stats["pa"] == 10 and away.stats["pa"] == 17


def test_name_keys_ignore_first_name_periods_everywhere():
    from nflsim.dfs.optimizer import name_keys
    from nflsim.dfs.public_projections import _short_name

    keys = name_keys(pd.Series(["D.J. Moore", "DJ Moore", "T.J. Hockenson", "Amon-Ra St. Brown", "Justin Jefferson"]))
    assert keys.tolist() == ["DJ.Moore", "DJ.Moore", "TJ.Hockenson", "Amo.St.", "Jus.Jefferson"]
    assert _short_name("D.J.", "Moore", "WR", "BUF") == "DJ.Moore"
    assert _short_name("Amon-Ra", "St. Brown", "WR", "DET") == "Amo.St."


def test_slate_from_simulations_reads_season_and_week(tmp_path):
    from nflsim.dfs.optimizer import slate_from_simulations

    path = tmp_path / "players.csv"
    pd.DataFrame({"game_id": ["2026_09_BUF_MIN"] * 2, "week": [9, 9], "player_name": ["a", "b"]}).to_csv(path, index=False)
    assert slate_from_simulations(path) == (2026, 9)
    pd.DataFrame({"game_id": ["2026_08_A_B", "2026_09_C_D"], "week": [8, 9]}).to_csv(path, index=False)
    assert slate_from_simulations(path) is None
    pd.DataFrame({"simulation": [1], "Jus.Jefferson": [10.0]}).to_csv(path, index=False)
    assert slate_from_simulations(path) is None
