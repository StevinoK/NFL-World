from __future__ import annotations

import numpy as np
import pandas as pd

from nflsim.features import forecast_archive as fa
from nflsim.features.weather import weather_frame


def test_pregame_weather_replaces_recorded_weather_and_leaves_unknown_games_empty(tmp_path):
    path = tmp_path / "forecasts.csv"
    pd.DataFrame({"game_id": ["known"], "temp": [40.0], "wind": [15.0], "precipitation": [6.0]}).to_csv(path, index=False)
    plays = pd.DataFrame({"game_id": ["known", "known", "old"], "temp": [90, 90, 90], "wind": [1, 1, 1], "roof": ["outdoors"] * 3})
    weather = fa.pregame_weather(plays, path)
    assert weather.temp.tolist()[:2] == [40.0, 40.0] and weather.precipitation.tolist()[:2] == [6.0, 6.0]
    assert np.isnan(weather.temp.iloc[2]) and np.isnan(weather.precipitation.iloc[2])
    inputs = weather_frame(weather)
    assert inputs.weather_missing.tolist() == [0.0, 0.0, 1.0]
    assert inputs.weather_temp_delta.iloc[0] == -1.0 and inputs.weather_temp_delta.iloc[2] == 0.0


def test_precipitation_input_grows_more_slowly_and_is_capped():
    games = pd.DataFrame({"roof": ["outdoors"] * 4 + ["dome"], "temp": [60.0] * 5, "wind": [5.0] * 5, "precipitation": [0.0, 1.0, 8.0, 500.0, 8.0]})
    values = weather_frame(games).weather_precipitation.tolist()
    assert values[0] == 0.0 and values[1] < values[2] < 8 * values[1]
    assert values[3] == 1.5 and values[4] == 0.0


def test_game_rows_read_kickoff_weather_and_game_precipitation():
    hours = pd.date_range("2026-11-08 13:00", periods=7, freq="h")
    hourly = {"time": hours.strftime("%Y-%m-%dT%H:%M").tolist(), "temperature_2m": [40, 50, 50, 50, 50, 50, 50],
              "wind_speed_10m": [10, 20, 20, 20, 20, 20, 20], "precipitation": [1, 1, 1, 1, 9, 9, 9]}
    games = pd.DataFrame({"game_id": ["g", "late"], "gameday": ["2026-11-08", "2026-11-08"], "gametime": ["13:30", "18:00"]})
    table = fa.game_rows(games, hourly)
    assert table.to_dict("records") == [{"game_id": "g", "temp": 45.0, "wind": 15.0, "precipitation": 4.0}]


def test_backtest_team_rows_take_the_forecast_for_scheduled_games_only(tmp_path, monkeypatch):
    from nflsim.sim import benchmark_2025 as benchmark

    path = tmp_path / "forecasts.csv"
    pd.DataFrame({"game_id": ["now"], "temp": [40.0], "wind": [15.0], "precipitation": [0.0]}).to_csv(path, index=False)
    monkeypatch.setattr(benchmark, "pregame_weather", lambda frame: fa.pregame_weather(frame, path))
    teams = pd.DataFrame({"game_id": ["now", "earlier"], "weather_temp_delta": [9.0, 9.0], "weather_outdoor": [0.0, 0.0],
                          "weather_wind_speed": [0.0, 0.0], "weather_missing": [0.0, 0.0]})
    result = benchmark.pregame_team_weather(teams, pd.DataFrame({"game_id": ["now"], "roof": ["outdoors"]}))
    assert result.weather_temp_delta.tolist() == [-1.0, 9.0] and result.weather_precipitation.tolist() == [0.0, 0.0]
