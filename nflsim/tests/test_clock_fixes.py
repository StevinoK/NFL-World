from types import SimpleNamespace

from nflsim.engine.simulation_runtime import Game, kneel_chance

FIXES = {
    "q4_lead_by_slack": [{"upper": 0.0, "rate": 0.8}, {"upper": 1000.0, "rate": 0.0}],
    "q2_by_yards_and_seconds": [{"yards_upper": 60, "by_seconds": [{"upper": 200.0, "rate": 0.0}]},
                                {"yards_upper": 100, "by_seconds": [{"upper": 10.0, "rate": 0.9}, {"upper": 200.0, "rate": 0.0}]}],
}


def test_kneel_chance_needs_a_lead_and_a_short_clock():
    assert kneel_chance(FIXES, 4, 70, 2, 3, 50) == 0.8      # slack 70 - 80 <= 0
    assert kneel_chance(FIXES, 4, 130, 1, 3, 50) == 0.0     # slack 10
    assert kneel_chance(FIXES, 4, 30, 1, -3, 50) == 0.0     # behind
    assert kneel_chance(FIXES, 2, 8, 1, -3, 75) == 0.9      # own territory, half ending
    assert kneel_chance(FIXES, 2, 8, 1, -3, 30) == 0.0
    assert kneel_chance(FIXES, 3, 8, 1, 10, 75) == 0.0


def overtime(first_score, second_score, done):
    game = Game.__new__(Game)
    first, second = SimpleNamespace(score=first_score), SimpleNamespace(score=second_score)
    game.team1, game.team2 = first, second
    game.quarter, game.overtime_first = 5, first
    game.possession = game.overtime_holder = first
    game.overtime_done = [team for team, ended in ((first, done[0]), (second, done[1])) if ended]
    return game, first, second


def test_overtime_gives_both_teams_a_possession():
    game, first, second = overtime(7, 0, (False, False))
    assert not game.overtime_over()                         # first team scores: second still gets the ball
    game.possession = second
    assert not game.overtime_over()                         # first possession over, second team has it
    second.score = 3
    assert not game.overtime_over()                         # still behind on its possession
    game.possession = first
    assert game.overtime_over()                             # second possession over, first team ahead


def test_overtime_ends_when_second_team_goes_ahead_or_first_is_scored_on():
    game, first, second = overtime(3, 3, (True, False))
    game.possession = game.overtime_holder = second
    second.score = 9
    assert game.overtime_over()
    game, first, second = overtime(0, 2, (False, False))  # safety on the first possession
    assert game.overtime_over()
    game, first, second = overtime(3, 3, (True, True))
    assert not game.overtime_over()
