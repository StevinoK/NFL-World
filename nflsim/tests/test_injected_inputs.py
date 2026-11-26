import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nflsim.engine import injected_inputs
from nflsim.engine.simulation_runtime import Game, PlayRow

sys.path.insert(0, str(Path(__file__).parent))
from test_rules import StubRuntime, team  # noqa: E402


def test_training_columns_match_the_simulator_definitions():
    pbp = pd.DataFrame({"score_differential": [-7, 3], "game_seconds_remaining": [3600, 300], "qtr": [1, 4],
                        "home_opening_kickoff": [1, 1], "posteam": ["AWY", "AWY"], "home_team": "HOM",
                        "away_team": "AWY", "air_yards": [12.0, np.nan], "ydstogo": [10, 5]})
    out = injected_inputs.add_training_columns(pbp)
    assert out.diff_time_ratio.tolist() == [-7.0, pytest.approx(3 * np.exp(4 * 3300 / 3600))]
    assert out.receive_2h_ko.tolist() == [1.0, 0.0]                      # the away team gets the second half
    assert out.distance_to_sticks.iloc[0] == 2.0 and out.air_is_zero.iloc[0] == 0.0 and np.isnan(out.air_is_zero.iloc[1])
    assert injected_inputs.diff_time_ratio(3, 300) == pytest.approx(out.diff_time_ratio.iloc[1])


def test_fill_reads_the_row_the_situation_and_the_offense():
    row = {"air_yards": 0.0, "ydstogo": 4, "qtr": 9}
    state = {"qtr": 2, "half_seconds_remaining": 95.0, "_attributes": pd.Series({"weather_wind_speed": 1.5})}
    values = injected_inputs.fill(["qtr", "half_seconds_remaining", "air_is_zero", "distance_to_sticks",
                                   "weather_wind_speed", "posteam_timeouts_remaining"], row, state)
    assert values == {"half_seconds_remaining": 95.0, "air_is_zero": 1.0, "distance_to_sticks": -4.0,
                      "weather_wind_speed": 1.5}                      # qtr was already in the row; no timeouts known


def runtime_expecting(columns):
    runtime = StubRuntime()
    runtime.processor = {"play": SimpleNamespace(feature_names_in_=np.array(["down", *columns]))}
    return runtime


def test_the_runtime_fills_only_what_the_model_expects():
    runtime = runtime_expecting(["qtr", "weather_wind_speed", "pass_oe"])
    assert runtime.injected_columns("play") == ["qtr", "weather_wind_speed"]
    runtime.play_state = {"qtr": 3, "_attributes": pd.Series({"weather_wind_speed": 0.5})}
    row = runtime.with_injected("play", PlayRow({"down": 1}))
    assert dict(row) == {"down": 1, "qtr": 3, "weather_wind_speed": 0.5}
    assert runtime.with_injected("kick", PlayRow({"down": 1})) == {"down": 1}   # no processor, nothing added


def test_the_game_publishes_the_situation_before_each_snap():
    runtime = StubRuntime()
    home, away = team("A", "home", runtime), team("B", "away", runtime)
    game = Game(away, home, "outdoors", runtime)
    game.opening_receiver = home
    game.possession, game.defense, game.quarter, game.clock = away, home, 2, 100.0
    away.score, home.score = 10, 3
    game.publish_play_state()
    state = runtime.play_state
    assert state["qtr"] == 2 and state["half_seconds_remaining"] == 100.0 and state["receive_2h_ko"] == 1.0
    assert state["diff_time_ratio"] == pytest.approx(injected_inputs.diff_time_ratio(7, game.game_seconds()))
    assert "posteam_timeouts_remaining" not in state and state["_attributes"] is away.attributes
    game.quarter, game.clock = 3, 300.0
    game.publish_play_state()
    assert runtime.play_state["half_seconds_remaining"] == 1200.0 and runtime.play_state["receive_2h_ko"] == 0.0


def test_models_with_timeout_inputs_need_the_timeouts_switch(monkeypatch):
    monkeypatch.delenv("NFLSIM_TIMEOUTS", raising=False)
    runtime = runtime_expecting(["posteam_timeouts_remaining"])
    with pytest.raises(ValueError, match="NFLSIM_TIMEOUTS"):
        Game(team("B", "away", runtime), team("A", "home", runtime), "outdoors", runtime)


def test_added_inputs_must_be_ones_the_simulator_supplies():
    from nflsim.models.fit_minimal import apply_added_inputs
    specs = {"kick": {"columns": ["down", "yardline_100"]}}
    added = apply_added_inputs(specs, {"kick": ["weather_wind_speed", "qtr", "down"]})
    assert added["kick"]["columns"] == ["down", "yardline_100", "weather_wind_speed", "qtr"]
    with pytest.raises(ValueError, match="can't supply"):
        apply_added_inputs(specs, {"kick": ["pass_oe"]})
    with pytest.raises(ValueError, match="unknown models"):
        apply_added_inputs(specs, {"punt": ["qtr"]})
