"""Field-position, scoring and play-call rules, checked on plays whose model outcomes are forced."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
import torch

from nflsim.engine import rules
from nflsim.engine.simulation_runtime import Game, Play, Player, SimulationRuntime, Team

SURE, NEVER = 50.0, -50.0
MODELS = ["play", "ay", "target", "sack", "comp", "int", "yac", "fumble", "def_ret", "kick", "rush", "rusher", "punt",
          "time", "penalty_no_play", "penalty_on_play", "penalty_yards", "penalty_first_down"]


def binary(probability_one: bool):
    return [[NEVER, SURE]] if probability_one else [[SURE, NEVER]]


class StubRuntime(SimulationRuntime):
    """Every model returns fixed logits, so each test decides the outcome of every draw."""

    pass_outcomes_given_attempt = True

    def __init__(self, **logits):
        self.logits = {
            "play": [[NEVER, SURE, NEVER, NEVER]],              # kick, pass, punt, run
            "ay": [[SURE]],                                     # always the deeper air-yards bin
            "target": [[NEVER, SURE] + [NEVER] * 9],            # WR1
            "sack": binary(False), "comp": binary(False), "int": binary(False),
            "fumble": binary(False), "kick": binary(True),
            "def_ret": [[NEVER]],                               # no return
        }
        self.logits.update(logits)
        self.model_dict = {model: object() for model in self.logits}
        self.bin_dict = {"ay": {"0": np.array([5, 30])}, "def_ret": {"0": np.array([0, 90])}}
        self.column_dict = {model: ["game_state_cols"] for model in MODELS}
        self.game_effects = None
        self.schema_issues = []

    def forward_model_logits(self, model_key, play_frame, base_values=None, latent_draw=None):
        return torch.tensor(self.logits[model_key], dtype=torch.float32)


def team(name, home, runtime):
    columns = {*runtime.def_cols, *runtime.team_stats_cols, *runtime.team_tendency_columns,
               *runtime.pregame_context_columns, *runtime.qb_context_columns}
    attributes = pd.Series({column: 0.0 for column in columns})
    player_columns = [*runtime.passer_columns, *runtime.receiver_columns, *runtime.rusher_columns,
                      *runtime.qb_context_columns]
    players = [Player(f"{name} {slot}", slot, f"{name}-{slot}", pd.DataFrame([{c: 0.0 for c in player_columns}]))
               for slot in ("QB1", "WR1", "RB1", "P")]
    return Team(name, players, home, attributes, runtime)


def play(runtime, yardline, down=1, distance=10, seed=0):
    np.random.seed(seed)
    offense, defense = team("A", "home", runtime), team("B", "away", runtime)
    game = Game(offense, defense, "outdoors", runtime)
    game.possession, game.defense = offense, defense
    game.yardline, game.down, game.distance = yardline, down, distance
    result = Play(down, distance, yardline, game=game)
    result.simulate(offense, defense)
    qb = next(player for player in offense.players if player.position == "QB1")
    return result, qb, game


def test_field_goal_geometry_and_missed_kick_spot():
    assert rules.field_goal_distance(65) == 52
    assert rules.missed_field_goal_start(60) == 47      # spot of the kick, 7 yards behind the line
    assert rules.missed_field_goal_start(90) == 20      # the spot would be inside the 20
    assert rules.end_zone_turnover_start(5) == 20 and rules.end_zone_turnover_start(35) == 35


def test_without_play_type_renormalizes_the_rest():
    assert rules.without_play_type([0.5, 0.25, 0.0, 0.25], "kick").tolist() == [0.0, 0.5, 0.0, 0.5]
    with pytest.raises(ValueError):
        rules.without_play_type([1.0, 0.0, 0.0, 0.0], "kick")


def test_a_sack_is_not_a_pass_attempt():
    result, qb, _ = play(StubRuntime(sack=binary(True)), yardline=50)
    assert result.type == "sack" and result.yards_gained < 0
    assert qb.stats["sacks"] == 1 and qb.stats["pass_attempts"] == 0


def test_interception_in_the_field_of_play_keeps_its_spot():
    result, qb, _ = play(StubRuntime(int=binary(True)), yardline=50)
    # Caught at the offense's 80, so the defense starts at its own 20 before any return.
    assert result.turnover and result.yards_gained == 30 and qb.stats["air_yards"] == 30


def test_end_zone_interception_is_a_touchback_unless_returned_past_the_20():
    result, qb, _ = play(StubRuntime(int=binary(True)), yardline=90)
    assert result.turnover and not result.defensive_touchdown
    assert result.yards_gained == -10 and qb.stats["air_yards"] == 10       # defense starts at its 20
    returned, _, _ = play(StubRuntime(int=binary(True), def_ret=[[SURE]]), yardline=90)
    # A 90-yard return from the goal line ends at the defense's 90, not a touchdown.
    assert not returned.defensive_touchdown and 100 - (90 + returned.yards_gained) == 90


def test_missed_field_goal_hands_over_at_the_spot_of_the_kick():
    kick = [[SURE, NEVER, NEVER, NEVER]]
    result, _, _ = play(StubRuntime(play=kick, kick=binary(False)), yardline=60, down=4)
    assert result.field_goal_attempt and not result.field_goal_made
    assert 100 - (60 + result.yards_gained) == 47
    made, _, game = play(StubRuntime(play=kick), yardline=60, down=4)
    assert made.field_goal_made and made.yards_gained == 0 and game.possession.score == 3


def test_model_play_calls_experiment_keeps_the_model_choice(monkeypatch):
    run = [[NEVER, NEVER, NEVER, SURE]]
    monkeypatch.delenv(rules.MODEL_PLAY_CALLS_ENV, raising=False)
    forced, _, _ = play(StubRuntime(play=run, rush=[[NEVER]]), yardline=70, down=4, distance=3)
    assert forced.field_goal_attempt                                        # old override: 4th-and-3, 47-yard try
    monkeypatch.setenv(rules.MODEL_PLAY_CALLS_ENV, "1")
    runtime = StubRuntime(play=run, rush=[[NEVER]], rusher=[[NEVER] * 5 + [SURE] + [NEVER] * 5])
    runtime.bin_dict["rush"] = {"0": np.array([4, 10])}
    kept, _, game = play(runtime, yardline=70, down=4, distance=3)
    assert kept.type == "rush" and game.diagnostics["play_calls"]["run"] == 1


def test_model_play_calls_experiment_allows_end_of_half_kicks_but_not_impossible_ones(monkeypatch):
    monkeypatch.setenv(rules.MODEL_PLAY_CALLS_ENV, "1")
    kick = [[SURE, 0.0, NEVER, NEVER]]
    early, _, _ = play(StubRuntime(play=kick), yardline=75, down=2)
    assert early.field_goal_attempt and early.field_goal_made              # a 42-yard try on 2nd down stands
    far, _, game = play(StubRuntime(play=kick), yardline=20, down=2)        # a 97-yard try is redrawn
    assert not far.field_goal_attempt and game.diagnostics["play_calls"]["kick"] == 0


def test_return_touchdowns_kick_their_own_try():
    runtime = StubRuntime()
    offense, defense = team("A", "home", runtime), team("B", "away", runtime)
    game = Game(offense, defense, "outdoors", runtime)
    defense.score = 6
    assert game.try_after_touchdown(defense, offense) == 1 and defense.score == 7
    assert game.plays[-1].pat_attempt


def test_drives_record_their_start_points_and_red_zone_trips():
    runtime = StubRuntime()
    home, away = team("A", "home", runtime), team("B", "away", runtime)
    game = Game(home, away, "outdoors", runtime)
    game.possession, game.defense, game.yardline = home, away, 25.0
    game.start_drive("kickoff")
    game.current_drive["red_zone"] = True
    home.score += 7
    game.switch_possession(35.0, how="punt")                  # B starts at its 65... from A's 35 in A's terms
    away.score += 3
    game.end_drive()
    assert game.drive_summary(home) == {"drives": 1.0, "drive_points": 7.0, "red_zone_trips": 1.0, "drive_start_mean": 25.0}
    assert game.drive_summary(away) == {"drives": 1.0, "drive_points": 3.0, "red_zone_trips": 0.0, "drive_start_mean": 65.0}
    assert [drive["how"] for drive in game.drives[id(away)]] == ["punt"]


def test_actual_drives_need_a_snap_and_start_at_the_own_yardline():
    from nflsim.sim.joint_outcomes import actual_drive_stats

    rows = [  # A: a kickoff to its 25, a run, a touchdown pass and the PAT; then B returns the kickoff for a touchdown
        ("kickoff", "A", 1, "A 25", 0, 0, 0), ("run", "A", 1, "A 25", 0, 0, 0), ("pass", "A", 1, "A 25", 1, 0, 6),
        ("extra_point", "A", 1, "A 25", 1, 6, 7), ("kickoff", "B", 2, "A 0", 0, 0, 6),
        ("run", "A", 3, "MID 50", 0, 7, 7), ("punt", "A", 3, "MID 50", 0, 7, 7), ("run", "B", 4, "A 40", 1, 7, 7)]
    pbp = pd.DataFrame(rows, columns=["play_type", "posteam", "fixed_drive", "drive_start_yard_line", "drive_inside20",
                                      "posteam_score", "posteam_score_post"]).assign(game_id="g", season_type="REG")
    stats = actual_drive_stats(pbp).set_index("team")
    assert stats.loc["A", "team.drives"] == 2 and stats.loc["A", "team.points_per_drive"] == 3.5
    assert stats.loc["A", "team.drive_start_mean"] == 37.5 and stats.loc["A", "team.red_zone_trips"] == 1
    assert stats.loc["B", "team.drives"] == 1 and stats.loc["B", "team.drive_start_mean"] == 60


def full_game_runtime():
    """Mixed, fixed probabilities for every decision, so whole games run on the stub."""
    runtime = StubRuntime(
        play=[[-0.5, 1.0, 0.0, 0.8]], sack=[[2.5, 0.0]], comp=[[0.0, 0.6]], int=[[3.5, 0.0]], fumble=[[4.0, 0.0]],
        kick=[[0.0, 2.0]], ay=[[1.0, -0.5]], yac=[[0.5, -1.0]], rush=[[1.0, -0.5, -2.0]], punt=[[1.0]], time=[[0.0]],
        def_ret=[[-1.0]], rusher=[[NEVER] * 5 + [1.0, 0.0] + [NEVER] * 4], target=[[NEVER, 1.0, NEVER, NEVER, NEVER, 0.0] + [NEVER] * 5])
    runtime.bin_dict.update({"ay": {"0": np.array([0, 8, 25])}, "yac": {"0": np.array([0, 5, 30])},
                             "rush": {"0": np.array([-2, 3, 8, 40])}, "punt": {"0": np.array([38, 50])},
                             "time": {"0": np.array([28, 40])}})
    return runtime


def test_drive_points_add_up_to_the_score_over_whole_games():
    runtime = full_game_runtime()
    for seed in range(20):
        np.random.seed(seed)
        random.seed(seed)
        home, away = team("A", "home", runtime), team("B", "away", runtime)
        game = Game(away, home, "outdoors", runtime)
        game.simulate_game()
        for side, other in ((home, away), (away, home)):
            summary = game.drive_summary(side)
            assert summary["drives"] >= 4
            # Every point is a drive's, a return or defensive touchdown's, or a safety's (worth 2 to the defense).
            safeties = side.score - summary["drive_points"] - game.return_td_points[id(side)]
            assert safeties >= 0 and safeties % 2 == 0
        assert abs(game.drive_summary(home)["drives"] - game.drive_summary(away)["drives"]) <= 2
