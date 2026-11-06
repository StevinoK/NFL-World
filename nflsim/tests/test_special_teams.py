import numpy as np
import pandas as pd

from nflsim.engine.simulation_runtime import Game
from nflsim.features import special_teams


def kickoff_pbp():
    """Three kickoffs to B: returned to the 27, a return TD, and an onside kick A recovers."""
    rows = [
        {"play_id": 1, "posteam": "B", "defteam": "A", "yardline_100": 35.0, "kickoff_attempt": 1},
        {"play_id": 2, "posteam": "B", "defteam": "A", "yardline_100": 73.0, "kickoff_attempt": 0},
        {"play_id": 3, "posteam": "B", "defteam": "A", "yardline_100": 35.0, "kickoff_attempt": 1, "touchdown": 1, "td_team": "B"},
        {"play_id": 4, "posteam": "A", "defteam": "B", "yardline_100": 35.0, "kickoff_attempt": 0},
        {"play_id": 5, "posteam": "B", "defteam": "A", "yardline_100": 35.0, "kickoff_attempt": 1, "own_kickoff_recovery": 1},
        {"play_id": 6, "posteam": "A", "defteam": "B", "yardline_100": 52.0, "kickoff_attempt": 0},
    ]
    frame = pd.DataFrame(rows).assign(game_id="2024_01_A_B", season=2024, week=1)
    return frame.fillna({"touchdown": 0, "td_team": "", "own_kickoff_recovery": 0})


def test_kickoff_starts_reads_the_next_snap_and_marks_return_touchdowns():
    starts = special_teams.kickoff_starts(kickoff_pbp())
    assert starts.start.tolist() == [27.0, 100.0]
    assert (starts.kicking == "A").all() and (starts.receiving == "B").all()


def test_pregame_features_use_only_earlier_weeks_and_shrink():
    events = pd.DataFrame({"team": "A", "season": 2024, "week": [1, 2, 3], "dev": [4.0, 4.0, 100.0]})
    targets = pd.DataFrame({"team": "A", "season": 2024, "week": [1, 3]})
    values = special_teams.pregame(events, targets, decay=1.0, carry=1.0, k=2.0)
    # Week 1 has no history; week 3 sees weeks 1-2 only (the week-3 outlier is ignored), shrunk by k.
    assert values[0] == 0.0
    assert np.isclose(values[1], 8.0 / 4.0)


class _Team:
    def __init__(self, name, home):
        self.name, self.home, self.score = name, home, 0
        self.stats = {"defensive_touchdown": 0}
        self.attributes = pd.Series(dtype=float)
        self.model_latents = {}


def _game(starts):
    game = Game.__new__(Game)
    game.team1, game.team2 = _Team("A", "home"), _Team("B", "away")
    game.quarter, game.clock, game.dome, game.down, game.distance = 1, 900, "outdoors", 1, 10
    game.diagnostics = {"special_teams": {"kickoffs": 0, "kickoff_start_sum": 0.0, "kickoff_return_tds": 0, "punt_return_tds": 0}}
    draws = iter(starts)
    game.special_teams_values = lambda kicking, receiving: {}
    game.sample_special_teams = lambda model, state, values, team: next(draws)
    return game


def test_kickoff_return_touchdown_scores_and_kicks_back():
    game = _game([100.0, 31.0])
    game.kickoff(game.team1)
    # B returns the first kick for a TD, then kicks back; A starts at its 31.
    assert game.team2.score == 7 and game.team2.stats["defensive_touchdown"] == 1
    assert game.possession is game.team1 and game.defense is game.team2 and game.yardline == 31.0
    assert game.diagnostics["special_teams"]["kickoffs"] == 2
