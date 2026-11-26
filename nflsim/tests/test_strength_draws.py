import json

import numpy as np
import pandas as pd

from nflsim.ridge.uncertainty import parse_weeks, with_opponent_copies
from nflsim.sim import strength_draws


class Holder:
    def __init__(self, name, attributes, players=()):
        self.name, self.player_id, self.attributes, self.players = name, name, attributes, list(players)


def table(tmp_path):
    rows = pd.DataFrame([
        # A's defense coefficient feeds its own defense rating and B's opponent-defense context.
        ("g", "team", "A", "def_pass_epa_rate", "epa_rate", "defense", "A", 0.04),
        ("g", "team", "A", "ridge_cpoe_defense_mu", "cpoe_bayes", "defense", "A", 0.5),
        ("g", "team", "B", "ridge_cpoe_opponent_defense_effect", "cpoe_bayes", "defense", "A", 0.5),
        ("g", "team", "B", "epa_rate", "epa_rate", "subject", "B", 0.06),
        ("g", "player", "QB", "cpoe_bayes", "cpoe_bayes", "subject", "QB", 1.0),
        ("g", "team", "B", "not_in_attributes", "epa_rate", "subject", "B", 9.0),
    ], columns=["game_id", "kind", "entity", "column", "fit", "block", "label", "sd"])
    path = tmp_path / "sd.csv"
    rows.to_csv(path, index=False)
    strength_draws._TABLES.clear()
    return path


def teams():
    qb = Holder("QB", pd.DataFrame([{"cpoe_bayes": 2.0}]))
    a = Holder("A", pd.Series({"def_pass_epa_rate": 0.0, "ridge_cpoe_defense_mu": 1.0}))
    b = Holder("B", pd.Series({"epa_rate": 0.1, "ridge_cpoe_opponent_defense_effect": 1.0}), [qb])
    return a, b, qb


def test_columns_reading_one_coefficient_move_together(tmp_path):
    config = {"sd_table": str(table(tmp_path)), "scale": 1.0}
    a, b, qb = teams()
    original = a.attributes
    moved = strength_draws.apply("g", (a, b), np.random.default_rng(3), config)
    assert moved == 5 and original["ridge_cpoe_defense_mu"] == 1.0          # the shared input was copied
    own = a.attributes["ridge_cpoe_defense_mu"] - 1.0
    assert own != 0 and np.isclose(b.attributes["ridge_cpoe_opponent_defense_effect"] - 1.0, own)
    # A different fit's defense coefficient (pass EPA, not CPOE) gets its own draw.
    assert a.attributes["def_pass_epa_rate"] != 0.0 and not np.isclose(a.attributes["def_pass_epa_rate"] / 0.04, own / 0.5)
    assert qb.attributes.cpoe_bayes.iat[0] != 2.0 and "not_in_attributes" not in b.attributes


def test_draws_repeat_with_the_seed_and_vanish_at_scale_zero(tmp_path):
    path = str(table(tmp_path))
    first, second, still = teams(), teams(), teams()
    strength_draws.apply("g", first[:2], np.random.default_rng(7), {"sd_table": path})
    strength_draws.apply("g", second[:2], np.random.default_rng(7), {"sd_table": path})
    strength_draws.apply("g", still[:2], np.random.default_rng(7), {"sd_table": path, "scale": 0.0})
    assert first[1].attributes.equals(second[1].attributes)
    assert still[1].attributes["epa_rate"] == 0.1
    assert strength_draws.apply("other", teams()[:2], np.random.default_rng(7), {"sd_table": path}) == 0


def test_settings_come_from_the_environment(tmp_path, monkeypatch):
    path = tmp_path / "draws.json"
    path.write_text(json.dumps({"sd_table": "x.csv", "scale": 0.85}))
    monkeypatch.delenv(strength_draws.STRENGTH_DRAWS_ENV, raising=False)
    assert strength_draws.settings() is None
    monkeypatch.setenv(strength_draws.STRENGTH_DRAWS_ENV, str(path))
    assert strength_draws.settings()["scale"] == 0.85


def test_opponent_copies_read_the_opponents_coefficient():
    sds = pd.DataFrame([("g", "team", "A", "def_pass_epa_rate", "epa_rate", "defense", "A", 0.04)],
                       columns=["game_id", "kind", "entity", "column", "fit", "block", "label", "sd"])
    schedule = pd.DataFrame({"game_id": ["g"], "home_team": ["A"], "away_team": ["B"]})
    full = with_opponent_copies(sds, schedule)
    copy = full[full.column.eq("opp_def_pass_epa_rate")].iloc[0]
    assert copy.entity == "B" and copy.label == "A" and copy.sd == 0.04
    assert parse_weeks("1-3,7") == [1, 2, 3, 7]
