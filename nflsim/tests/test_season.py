from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflsim.sim import standings
from nflsim.sim.season import Move, apply_moves, game_changes, parse_move, play_seasons, quarterbacks
from nflsim.sim.standings import INDEX, Season, break_tie, seed

RNG = np.random.default_rng(0)


def season(*games):
    """Games as (home, away, result) with result 1 home win, 0 away win, 0.5 tie."""
    home, away, result = zip(*games)
    return Season.from_results(np.array([INDEX[t] for t in home]), np.array([INDEX[t] for t in away]), np.array(result, float))


def test_division_tie_goes_to_head_to_head_then_division_record():
    played = season(("BUF", "MIA", 1), ("MIA", "NYJ", 1), ("NYJ", "BUF", 1), ("NE", "BUF", 0), ("NE", "MIA", 0))
    assert break_tie(played, [INDEX["BUF"], INDEX["MIA"]], standings._division_steps, RNG) == INDEX["BUF"]
    split = season(("BUF", "MIA", 1), ("MIA", "BUF", 1), ("BUF", "NE", 1), ("MIA", "NE", 0), ("NYJ", "MIA", 0))
    # Split head-to-head; BUF is 2-1 in the division and MIA 2-2 (1-1 vs BUF, 1-0 vs NYJ, 0-1 vs NE).
    assert break_tie(split, [INDEX["BUF"], INDEX["MIA"]], standings._division_steps, RNG) == INDEX["BUF"]


def test_three_club_wildcard_uses_head_to_head_only_as_a_sweep():
    swept = season(("KC", "PIT", 1), ("KC", "HOU", 1), ("PIT", "HOU", 1))
    group = [INDEX["KC"], INDEX["PIT"], INDEX["HOU"]]
    assert break_tie(swept, group, standings._wildcard_steps, RNG) == INDEX["KC"]
    # No sweep (HOU never met PIT): conference record decides instead.
    partial = season(("KC", "PIT", 1), ("HOU", "KC", 1), ("HOU", "DEN", 1), ("PIT", "BAL", 1), ("HOU", "BAL", 1))
    assert break_tie(partial, group, standings._wildcard_steps, RNG) == INDEX["HOU"]


def test_seeding_gives_each_conference_four_division_winners_and_three_wildcards():
    teams = list(standings.TEAMS)
    games = [(teams[i], teams[(i + step) % 32], RNG.choice([0, 0.5, 1])) for step in (1, 3, 5, 7, 9) for i in range(32)]
    seeds = seed(season(*games), RNG)
    for conference in ("AFC", "NFC"):
        mask = standings.CONFERENCE == conference
        assert sorted(seeds[mask][seeds[mask] > 0]) == list(range(1, 8))
        champions = np.flatnonzero(mask & (seeds >= 1) & (seeds <= 4))
        assert len(set(standings.DIVISION[champions])) == 4
    records = season(*games).record
    for division in standings.DIVISIONS:
        members = np.flatnonzero(standings.DIVISION == division)
        winner = members[(seeds[members] >= 1) & (seeds[members] <= 4)][0]
        assert records[winner] == records[members].max()


def test_parse_move():
    assert parse_move("Kirk Cousins>min@qb1") == Move("Kirk Cousins", "MIN", "QB1")
    assert parse_move("Patrick Mahomes > OUT") == Move("Patrick Mahomes", None)
    for bad in ("Kirk Cousins", "Kirk Cousins>XYZ", "Kirk Cousins>MIN@QB2", "Kirk Cousins>OUT@QB1"):
        with pytest.raises(ValueError):
            parse_move(bad)


def test_apply_moves_reassigns_upcoming_weeks_only():
    schedule = pd.DataFrame({"game_id": ["G9", "G10", "G11"], "week": [9, 10, 11], "home_team": ["MIN", "LV", "MIN"],
                             "away_team": ["DET", "MIN", "GB"], "gameday": pd.to_datetime(["2026-11-01", "2026-11-08", "2026-11-15"])})
    roster = pd.DataFrame({"player_id": ["00-0029604"] * 3 + ["00-1"], "player_name": ["Kirk Cousins"] * 3 + ["Other QB"],
                           "team": ["LV", "LV", "LV", "MIN"], "week": [8, 9, 10, 9], "game_id": ["a", "b", "c", "G9"],
                           "position": "QB", "status": "ACT", "depth_team": [1, 1, 1, 1]})
    moved, affected = apply_moves(roster, schedule, [Move("Kirk Cousins", "MIN", "QB1")], first_week=10)
    cousins = moved[moved.player_id.eq("00-0029604")].sort_values("week")
    assert cousins[["team", "week"]].values.tolist() == [["LV", 8], ["LV", 9], ["MIN", 10], ["MIN", 11]]
    assert cousins.game_id.tolist()[2:] == ["G10", "G11"] and affected == {"LV", "MIN"}
    out, affected = apply_moves(roster, schedule, [Move("00-0029604", None)], first_week=9)
    assert out.player_id.eq("00-0029604").sum() == 1 and affected == {"LV"}


def test_season_draws_are_shared_so_unchanged_scores_give_identical_seasons():
    teams = list(standings.TEAMS)
    schedule = pd.DataFrame({"game_id": [f"G{i:02d}" for i in range(32)], "week": 1,
                             "home_team": teams, "away_team": teams[1:] + teams[:1],
                             "home_score": [21.0] * 16 + [np.nan] * 16, "away_score": [17.0] * 16 + [np.nan] * 16})
    sims = pd.DataFrame([(f"G{i:02d}", s, 20 + s % 3, 21) for i in range(16, 32) for s in range(1, 5)],
                        columns=["game_id", "simulation", "home_score", "away_score"])
    first, second = play_seasons(schedule, sims, 50, seed=7), play_seasons(schedule, sims, 50, seed=7)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    assert first[0].sum(1).tolist() == [32.0] * 50


def test_quarterbacks_uses_each_players_latest_team():
    roster = pd.DataFrame({"player_id": ["00-1", "00-1", "00-2", "00-3"], "player_name": ["A B", "A B", "C D", "E F"],
                           "team": ["LV", "MIN", "MIN", "MIN"], "week": [8, 9, 9, 9], "position": ["QB", "QB", "QB", "WR"],
                           "depth_team": [1, 2, 1, 1]})
    table = quarterbacks(roster)
    assert table.player_id.tolist() == ["00-2", "00-1"] and table.team.tolist() == ["MIN", "MIN"]


def test_game_changes_compares_only_rerun_games():
    schedule = pd.DataFrame({"game_id": ["G1", "G2"], "week": [9, 10], "home_team": ["MIN", "LV"], "away_team": ["DET", "GB"]})
    base = pd.DataFrame({"game_id": ["G1", "G1", "G2"], "simulation": [1, 2, 1], "home_score": [20, 30, 10], "away_score": [10, 30, 20]})
    rerun = pd.DataFrame({"game_id": ["G1", "G1"], "simulation": [1, 2], "home_score": [10, 30], "away_score": [20, 20]})
    table = game_changes(base, rerun, schedule)
    assert table.game_id.tolist() == ["G1"]
    assert table.margin_base[0] == 5 and table.margin_experiment[0] == 0
    assert table.home_win_base[0] == 0.75 and table.home_win_experiment[0] == 0.5
    assert table.total_base[0] == 45 and table.total_experiment[0] == 40
