from __future__ import annotations

import numpy as np
import pandas as pd

from nflsim.features import game_precipitation as gp
from nflsim.ridge.expectations import context_features


def test_only_played_open_air_games_not_in_the_table_are_fetched():
    schedule = pd.DataFrame({"game_id": ["new", "known", "dome", "future", "open"], "roof": ["outdoors", "outdoors", "dome", "outdoors", "open"],
                             "stadium_id": ["GNB00", "GNB00", "DET00", "GNB00", "ATL97"], "home_score": [20, 20, 20, np.nan, 20]})
    assert gp.games_to_fetch(schedule, {"known"}).game_id.tolist() == ["new", "open"]


def test_game_totals_add_up_the_game_window_only():
    hours = pd.date_range("2026-11-08 12:00", periods=8, freq="h")
    hourly = {"time": hours.strftime("%Y-%m-%dT%H:%M").tolist(), "rain": [9, 1, 1, 1, 1, 9, 9, 9], "snowfall": [0, 0, 0, 0.5, 0, 0, 0, 0]}
    games = pd.DataFrame({"game_id": ["g"], "gameday": ["2026-11-08"], "gametime": ["13:25"]})
    assert gp.game_totals(games, hourly).iloc[0].to_dict() == {"game_id": "g", "rain_mm": 4.0, "snow_cm": 0.5}


def test_game_past_the_returned_hours_is_left_out():
    hourly = {"time": ["2026-11-08T13:00"], "rain": [1.0], "snowfall": [0.0]}
    assert gp.game_totals(pd.DataFrame({"game_id": ["g"], "gameday": ["2026-11-08"], "gametime": ["13:00"]}), hourly).empty


def test_context_reads_amounts_and_treats_indoor_games_as_dry(tmp_path, monkeypatch):
    path = tmp_path / "rain.csv"
    pd.DataFrame({"game_id": ["wet"], "rain_mm": [6.0], "snow_cm": [1.5]}).to_csv(path, index=False)
    monkeypatch.setattr(gp, "TABLE", path)
    gp.load_table.__defaults__ = (path,)
    pbp = pd.DataFrame({"game_id": ["wet", "unknown", "indoor"], "roof": ["outdoors", "outdoors", "dome"]})
    try:
        context = context_features(pbp)
    finally:
        gp.load_table.__defaults__ = (gp.TABLE,)
    assert context.ctx_rain_mm.tolist()[0] == 6.0 and context.ctx_snow_cm.tolist()[0] == 1.5
    assert np.isnan(context.ctx_rain_mm.iloc[1]) and context.ctx_rain_mm.iloc[2] == 0.0
