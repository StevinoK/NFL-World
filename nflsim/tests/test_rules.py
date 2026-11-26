"""Field-position, scoring and play-call rules, checked on plays whose model outcomes are forced."""
from __future__ import annotations

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
