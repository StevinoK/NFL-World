import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nflsim.features import pregame_tables as tables
from nflsim.remote.worker import game_week

sys.path.insert(0, str(Path(__file__).parent))
from test_rules import StubRuntime, team  # noqa: E402


def test_decayed_sums_use_only_earlier_weeks():
    events = pd.DataFrame({"season": [2024, 2024, 2024], "week": [1, 2, 3], "x": [1.0, 10.0, 100.0]})
    weeks = pd.DataFrame({"season": [2024, 2024, 2025], "week": [1, 3, 1]})
    out = tables.decayed_before(events, [], ["x"], weeks).set_index(["season", "week"]).x
    assert (2024, 1) not in out.index                                   # nothing before the first week
    assert out[(2024, 3)] == pytest.approx(tables.WEEK_DECAY * 1.0 + 10.0)
    gap = tables.week_index(2025, 1) - tables.week_index(2024, 3)
    assert out[(2025, 1)] == pytest.approx(100.0 * tables.WEEK_DECAY ** (gap - 1) + 10.0 * tables.WEEK_DECAY ** gap
                                           + 1.0 * tables.WEEK_DECAY ** (gap + 1))


def pbp():
    rows = []
    for week, kicker in ((1, "K1"), (2, "K1"), (3, "K2")):
        for offense, defense in (("A", "B"), ("B", "A")):
            game = f"2024_{week:02d}_A_B"
            base = dict(game_id=game, season=2024, week=week, season_type="REG", posteam=offense, defteam=defense,
                        down=1, ydstogo=10, yardline_100=60, qtr=1, score_differential=0, qb_scramble=0, qb_dropback=0)
            rows.append({**base, "play_type": "run", "rush": 1, "pass": 0, "run_location": "left",
                         "yards_gained": 10.0 if offense == "A" else 2.0})
            rows.append({**base, "play_type": "field_goal", "field_goal_attempt": 1, "kick_distance": 45,
                         "field_goal_result": "made" if offense == "A" else "missed",
                         "kicker_player_id": kicker if offense == "A" else "KB"})
    return pd.DataFrame(rows)


def test_tables_are_pregame_and_kickers_carry_their_own_history():
    table = tables.build_tables(pbp()).set_index(["season", "week", "team"])
    assert np.isnan(table.loc[(2024, 1, "A"), "pg_off_run_left_ypc"])            # no earlier week
    # Week 3: A gained more running left than B, and B's defense allowed A's yards.
    assert table.loc[(2024, 3, "A"), "pg_off_run_left_ypc"] > table.loc[(2024, 3, "B"), "pg_off_run_left_ypc"]
    assert table.loc[(2024, 3, "B"), "pg_def_run_left_ypc"] == pytest.approx(table.loc[(2024, 3, "A"), "pg_off_run_left_ypc"])
    assert table.loc[(2024, 3, "A"), "pg_off_kicker_fg_oe"] > 0 > table.loc[(2024, 3, "B"), "pg_off_kicker_fg_oe"]
    # Week 4 (the forecast week): A's kicker is now K2, with one make of his own, so his rating is not K1's.
    assert table.loc[(2024, 4, "A"), "pg_off_kicker_fg_oe"] != table.loc[(2024, 3, "A"), "pg_off_kicker_fg_oe"]
    assert np.isnan(table.loc[(2024, 3, "A"), "pg_off_screen_rate"])           # no FTN charting here


def test_training_join_and_simulator_lookup_split_offense_and_defense():
    table = tables.build_tables(pbp())
    plays = pd.DataFrame({"season": [2024], "week": [3], "posteam": ["A"], "defteam": ["B"]})
    joined = tables.attach_pregame_tables(plays, table).iloc[0]
    values = tables.game_state_values(tables.lookup(table), 2024, 3, "A", "B")
    for column in tables.COLUMNS:
        assert (np.isnan(joined[column]) and np.isnan(values[column])) or joined[column] == pytest.approx(values[column])
    with pytest.raises(ValueError, match="no 2024 week 9"):
        tables.game_state_values(tables.lookup(table), 2024, 9, "A", "B")
    assert game_week("2026_07_KC_BUF") == (2026, 7) and game_week("synthetic-1") is None


def test_models_with_table_inputs_need_the_tables(monkeypatch):
    from nflsim.engine.simulation_runtime import Game
    runtime = StubRuntime()
    runtime.processor = {"kick": SimpleNamespace(feature_names_in_=np.array(["down", "pg_off_kicker_fg_oe"]))}
    with pytest.raises(ValueError, match="NFLSIM_PREGAME_TABLES"):
        Game(team("B", "away", runtime), team("A", "home", runtime), "outdoors", runtime)
    table = tables.build_tables(pbp())
    runtime.pregame, runtime.game_week = tables.lookup(table), (2024, 3)
    home, away = team("A", "home", runtime), team("B", "away", runtime)
    game = Game(away, home, "outdoors", runtime)
    game.possession, game.defense = home, away
    game.publish_play_state()
    assert runtime.play_state["pg_off_kicker_fg_oe"] > 0 and "pg_def_run_left_ypc" in runtime.play_state
