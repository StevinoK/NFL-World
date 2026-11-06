import numpy as np
import pandas as pd
import pytest

from nflsim.features.game_inputs import (
    available_slot_players, build_team_pair, make_players, make_players_from_assignments, pregame_player_snapshot,
    pregame_team_snapshot, roster_for_team_week, slot_assignments,
)


class Runtime:
    required_player_feature_columns = ["yac_bayes", "medium", "rush_rate", "ridge_yac_oe_venue_effect"]
    required_team_feature_columns = ["epa_rate", "days_rest", "ridge_yac_oe_venue_effect"]
    context_map = {"yac": ["ridge_yac_oe_venue_effect"]}

    @staticmethod
    def checked_context(model, values):
        value = float(values["ridge_yac_oe_venue_effect"])
        if not np.isfinite(value):
            raise ValueError("invalid context")
        return {"ridge_yac_oe_venue_effect": value}


@pytest.fixture
def runtime():
    return Runtime()


def test_team_snapshot_uses_latest_pregame_game_id_row(runtime):
    features = pd.DataFrame([
        {"team": "A", "game_id": "2024_01_A_B", "game_date": "2024-09-01", "epa_rate": 0.2, "days_rest": 7, "ridge_yac_oe_venue_effect": -0.1},
        {"team": "A", "game_id": "2024_02_A_C", "game_date": "2024-09-08", "epa_rate": 999, "days_rest": 7, "ridge_yac_oe_venue_effect": 999},
    ])
    row = pregame_team_snapshot(runtime, features, "A", pd.Timestamp("2024-09-08"))
    assert row.epa_rate == 999
    before = pregame_team_snapshot(runtime, features, "A", pd.Timestamp("2024-09-07"))
    assert before.epa_rate == 0.2


def test_player_snapshot_is_pregame_and_fills_missing_from_available_history(runtime):
    stats = pd.DataFrame([
        {"player_id": "P", "game_id": "2024_01_A_B", "game_date": "2024-09-01", "yac_bayes": 2, "medium": .4, "rush_rate": .1, "ridge_yac_oe_venue_effect": .2},
        {"player_id": "P", "game_id": "2024_03_A_C", "game_date": "2024-09-15", "yac_bayes": 900, "medium": .4, "rush_rate": .1, "ridge_yac_oe_venue_effect": 900},
    ])
    row = pregame_player_snapshot(runtime, stats, "P", pd.Timestamp("2024-09-08"), "WR")
    assert row.yac_bayes.iloc[0] == 2
    assert row.ridge_yac_oe_venue_effect.iloc[0] == .2
    rookie = pregame_player_snapshot(runtime, stats, "ROOKIE", pd.Timestamp("2024-09-08"), "WR")
    assert np.isfinite(rookie[runtime.required_player_feature_columns].to_numpy(dtype=float)).all()


def test_unavailable_ridge_context_is_fail_fast(runtime):
    features = pd.DataFrame([{"team": "A", "game_id": "2024_01_A_B", "game_date": "2024-09-01", "epa_rate": 0.2, "days_rest": 7, "ridge_yac_oe_venue_effect": np.nan}])
    with pytest.raises(ValueError, match="context"):
        pregame_team_snapshot(runtime, features, "A", pd.Timestamp("2024-09-01"))


def test_build_team_pair_uses_game_forecast_for_both_teams():
    runtime = Runtime()
    runtime.required_team_feature_columns = ["weather_outdoor", "weather_temp_delta", "weather_wind_speed", "weather_missing"]
    runtime.context_map = {"target": runtime.required_team_feature_columns}
    runtime.checked_context = lambda model, values: {column: float(values[column]) for column in runtime.context_map[model]}
    game_id = "2024_01_A_B"
    teams = pd.DataFrame([{"game_id": game_id, "team": team, "game_date": "2024-09-01",
                           "weather_outdoor": 0.0, "weather_temp_delta": 0.0,
                           "weather_wind_speed": 0.0, "weather_missing": 1.0}
                          for team in ("A", "B")])
    game = pd.Series({"game_id": game_id, "gameday": "2024-09-01", "week": 1,
                      "home_team": "A", "away_team": "B", "roof": "open", "temp": 40, "wind": 15})
    features = build_team_pair(runtime, teams, game, pd.DataFrame(columns=["gameday", "home_team", "away_team"]))
    assert features.weather_outdoor.eq(1.0).all()
    assert features.weather_temp_delta.eq(-1.0).all()
    assert features.weather_wind_speed.eq(1.0).all()
    assert features.weather_missing.eq(0.0).all()


def test_roster_week_prefers_active_current_or_latest_prior():
    roster = pd.DataFrame([
        {"team": "A", "week": 1, "player_id": "OLD", "position": "QB", "status": "ACT"},
        {"team": "A", "week": 2, "player_id": "INJ", "position": "QB", "status": "RES"},
        {"team": "A", "week": 2, "player_id": "NEW", "position": "QB", "status": "ACT"},
    ])
    assert roster_for_team_week(roster, "A", 2).player_id.tolist() == ["NEW"]
    assert roster_for_team_week(roster, "A", 3).player_id.tolist() == ["NEW"]


def test_slots_preserve_runtime_order_and_build_all_eleven():
    runtime = Runtime()
    roster = pd.DataFrame([
        {"team": "A", "week": 1, "player_id": f"{pos}{i}", "player_name": f"{pos}{i}",
         "football_name": "QB nickname" if pos == "QB" and i == 1 else None,
         "last_name": "Last" if pos == "QB" and i == 1 else None,
         "position": pos, "status": "ACT"}
        for pos, count in (("QB", 1), ("WR", 4), ("RB", 3), ("TE", 3))
        for i in range(1, count + 1)
    ])
    rows = []
    for item in roster.to_dict("records"):
        rows.append({"player_id": item["player_id"], "game_id": "2024_01_A_B", "game_date": "2024-09-01", "yac_bayes": .1, "medium": float(item["player_id"][-1]), "rush_rate": float(item["player_id"][-1]), "ridge_yac_oe_venue_effect": .3})
    stats = pd.DataFrame(rows)
    slots = slot_assignments(roster, pd.DataFrame(), "A", 1, pd.Timestamp("2024-09-01"), stats)
    candidates = available_slot_players(roster, "A", 1, pd.DataFrame(), pd.Timestamp("2024-09-01"), stats)
    assert slots["QB1"]["player_name"] == "QB nickname Last"
    assert candidates["QB"][0]["player_name"] == "QB nickname Last"
    players = make_players(runtime, stats, slots, pd.Timestamp("2024-09-01"))
    assert [player.position for player in players] == ["QB1", "WR1", "WR2", "WR3", "WR4", "RB1", "RB2", "RB3", "TE1", "TE2", "TE3"]
    assert players[0].attributes.ridge_yac_oe_venue_effect.iloc[0] == .3


def test_locked_player_overrides_reach_serialized_attributes():
    runtime = Runtime()
    stats = pd.DataFrame([{"player_id": "P", "game_id": "2024_01_A_B", "game_date": "2024-09-01",
                           "yac_bayes": 1.0, "medium": .5, "rush_rate": .3,
                           "ridge_yac_oe_venue_effect": .2}])
    assignments = {slot: {"player_id": f"X_{slot}", "player_name": slot, "position": slot[:-1]}
                   for slot in ("QB1", "WR1", "WR2", "WR3", "WR4", "RB1", "RB2", "RB3", "TE1", "TE2", "TE3")}
    assignments["WR1"] = {"player_id": "P", "player_name": "Receiver Name", "position": "WR"}
    players = make_players_from_assignments(runtime, stats, assignments, pd.Timestamp("2024-09-02"),
                                            {"P": {"yac_bayes": 4.5, "medium": .9}})
    wr1 = next(player for player in players if player.position == "WR1")
    assert wr1.name == "Receiver Name"
    assert wr1.attributes.yac_bayes.iloc[0] == 4.5
    assert wr1.attributes.medium.iloc[0] == .9
    assert wr1.attributes.ridge_yac_oe_venue_effect.iloc[0] == .2
    with pytest.raises(ValueError, match="non-player model input"):
        make_players_from_assignments(runtime, stats, assignments, pd.Timestamp("2024-09-02"),
                                      {"P": {"ridge_unknown_context": 2}})
