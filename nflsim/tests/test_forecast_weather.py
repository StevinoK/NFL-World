from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
import requests

from nflsim.bundle_labels import BundleInfo, bundle_label
from nflsim.features import forecast_weather as fw
from nflsim.features.weather import weather_frame


def schedule():
    return pd.DataFrame({
        "game_id": ["played", "soon", "dome", "far"], "stadium_id": ["GNB00", "GNB00", "DET00", "GNB00"],
        "roof": ["outdoors", "outdoors", "dome", "outdoors"], "gameday": ["2026-11-01", "2026-11-08", "2026-11-08", "2026-12-20"],
        "gametime": ["13:00", "13:30", "13:00", "13:00"], "home_score": [20.0, np.nan, np.nan, np.nan],
        "temp": [55.0, np.nan, np.nan, np.nan], "wind": [8.0, np.nan, np.nan, np.nan],
    })


def test_only_upcoming_open_air_games_in_the_horizon_need_a_forecast():
    assert fw.games_needing_forecast(schedule(), date(2026, 11, 6)).game_id.tolist() == ["soon"]


def test_every_upcoming_open_air_stadium_has_coordinates_or_is_skipped():
    games = schedule().assign(stadium_id=["XXX", "XXX", "XXX", "XXX"])
    assert fw.fetch_forecast(games, "2026-11-06").empty


def test_kickoff_value_falls_between_the_hours():
    games = fw.games_needing_forecast(schedule(), date(2026, 11, 6))
    hourly = {"time": ["2026-11-08T13:00", "2026-11-08T14:00"], "temperature_2m": [40, 50], "wind_speed_10m": [10, 20], "precipitation": [0, 1]}
    table = fw.parse_forecast(games, [{"hourly": hourly}], ["GNB00"], "2026-11-06")
    assert table.iloc[0].to_dict() == {"game_id": "soon", "temp": 45.0, "wind": 15.0, "precipitation": 0.5, "fetched_on": "2026-11-06"}


def test_game_outside_the_returned_hours_gets_no_forecast():
    games = fw.games_needing_forecast(schedule(), date(2026, 11, 6))
    hourly = {"time": ["2026-11-09T13:00", "2026-11-09T14:00"], "temperature_2m": [40, 50], "wind_speed_10m": [10, 20], "precipitation": [0, 1]}
    assert fw.parse_forecast(games, [{"hourly": hourly}], ["GNB00"], "2026-11-06").empty


def test_forecast_fills_gaps_clears_missing_flag_and_never_replaces_recorded_weather():
    forecast = pd.DataFrame({"game_id": ["soon", "played"], "temp": [40.0, 99.0], "wind": [12.0, 99.0], "precipitation": [0.0, 0.0], "fetched_on": ["d", "d"]})
    filled = fw.apply_forecast(schedule(), forecast).set_index("game_id")
    assert filled.loc["played", "temp"] == 55 and not filled.loc["played", "weather_forecast"]
    assert filled.loc["soon", "wind"] == 12 and filled.loc["soon", "weather_forecast"]
    assert np.isnan(filled.loc["far", "temp"])
    assert weather_frame(filled.reset_index()).set_index(filled.index).loc[["soon", "far"], "weather_missing"].tolist() == [0.0, 1.0]


def test_forecast_is_saved_once_a_day(tmp_path, monkeypatch):
    calls = []

    def fetch(games, fetched_on):
        calls.append(fetched_on)
        return pd.DataFrame({"game_id": ["soon"], "temp": [40.0], "wind": [12.0], "precipitation": [0.0], "fetched_on": [fetched_on]})

    monkeypatch.setattr(fw, "fetch_forecast", fetch)
    day = date(2026, 11, 6)
    first = fw.load_forecast(schedule(), today=day, folder=tmp_path)
    second = fw.load_forecast(schedule(), today=day, folder=tmp_path)
    assert len(calls) == 1 and first.temp.tolist() == second.temp.tolist()
    fw.load_forecast(schedule(), today=day, folder=tmp_path, refresh=True)
    assert len(calls) == 2


def test_failed_fetch_falls_back_to_the_newest_saved_forecast(tmp_path, monkeypatch):
    pd.DataFrame({"game_id": ["soon"], "temp": [40.0], "wind": [12.0], "precipitation": [0.0], "fetched_on": ["old"]}).to_csv(tmp_path / "forecast-20261105.csv", index=False)

    def fail(games, fetched_on):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(fw, "fetch_forecast", fail)
    assert fw.load_forecast(schedule(), today=date(2026, 11, 6), folder=tmp_path).fetched_on.tolist() == ["old"]
    assert fw.load_forecast(schedule(), today=date(2026, 11, 6), folder=tmp_path / "none").empty


def test_tag_changes_with_the_weather():
    base = pd.DataFrame({"game_id": ["a"], "temp": [40.0], "wind": [12.0]})
    assert fw.forecast_tag(base) != fw.forecast_tag(base.assign(wind=13.0))
    assert fw.forecast_tag(base.iloc[:0]) == "wxnone"


@pytest.mark.parametrize("through, expected", [("2026 week 8", "through 2026 week 8"), (None, "training data only")])
def test_bundle_label_says_what_it_is(tmp_path, through, expected):
    from datetime import datetime
    info = BundleInfo(tmp_path, "Seed-averaged models", "", datetime(2026, 11, 5), through)
    assert bundle_label(info, tmp_path) == f"Seed-averaged models · built Nov 5 · {expected} · current"
    assert bundle_label(info) == f"Seed-averaged models · built Nov 5 · {expected}"
