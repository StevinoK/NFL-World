from __future__ import annotations

import pandas as pd

from nflsim.engine.player_names import player_names
from nflsim.features.weekly_refresh import attach_roster_player_names
from nflsim.names import display_name, roster_display_names


def roster():
    # 2026 weekly rosters carry only the first name in player_name.
    return pd.DataFrame({
        "player_id": ["A", "A", "B", "C"], "season": [2025, 2026, 2026, 2026], "week": [18, 8, 8, 8],
        "player_name": ["Old", "Matthew", "Cam", "Solo"], "football_name": ["Old", "Matthew", "Cam", None],
        "first_name": ["John", "John", "Cameron", None], "last_name": ["Stafford", "Stafford", "Ward", None],
    })


def test_display_name_is_nickname_plus_last_name_with_fallbacks():
    assert display_name("Cam", "Ward", "Cameron") == "Cam Ward"
    assert display_name(None, "Ward", "Cameron") == "Cameron Ward"
    assert display_name(None, None, None, "Solo") == "Solo"
    assert display_name(None, None, None, None, fallback="00-1") == "00-1"


def test_every_name_source_uses_nickname_and_last_name():
    assert roster_display_names(roster()) == {"A": "Matthew Stafford", "B": "Cam Ward", "C": "Solo"}
    attached = attach_roster_player_names(roster())
    assert attached.player_name.tolist()[1:] == ["Matthew Stafford", "Cam Ward", "Solo"]
    assert player_names(roster())["B"] == "Cam Ward"
