import pandas as pd
import json

from nflsim.features.build import selection_input_hashes
from nflsim.paths import digest, write_json
from nflsim.features.weekly_refresh import (add_scheduled_roster_slots, attach_current_depth_order,
    attach_roster_player_names,
    check_pm_freshness, normalize_depth_team, update_bundle_feature_lineage)
from nflsim.features.weekly_refresh import refresh_weekly_bundle
from nflsim.dashboard.app import apply_pending_bundle
import pytest


def test_selection_hashes_ignore_new_season_rows():
    pbp = pd.DataFrame({"game_id": ["2024_01_A_B", "2025_01_A_B"], "play_id": [1, 1], "epa": [1.0, 2.0]})
    schedule = pd.DataFrame({"game_id": ["2024_01_A_B", "2025_01_A_B"], "home_team": ["A", "A"]})
    roster = pd.DataFrame({"game_id": ["2024_01_A_B", "2025_01_A_B"], "team": ["A", "A"],
                           "player_id": ["P1", "P2"], "position": ["QB", "QB"]})
    frozen = selection_input_hashes(pbp, schedule, roster)
    changed = pbp.copy()
    changed.loc[1, "epa"] = 999
    grown = pd.concat([changed, pd.DataFrame([{"game_id": "2026_01_A_B", "play_id": 1, "epa": -500}])], ignore_index=True)
    schedule = pd.concat([schedule, pd.DataFrame([{"game_id": "2026_01_A_B", "home_team": "A"}])], ignore_index=True)
    roster = pd.concat([roster, pd.DataFrame([{"game_id": "2026_01_A_B", "team": "A", "player_id": "P3", "position": "QB"}])], ignore_index=True)
    assert selection_input_hashes(grown, schedule, roster) == frozen


def test_selection_hashes_ignore_numeric_dtype_promotion():
    pbp = pd.DataFrame({"game_id": ["2024_01_A_B"], "play_id": [1], "epa": pd.Series([1.25], dtype="float32")})
    schedule = pd.DataFrame({"game_id": ["2024_01_A_B"], "home_team": ["A"]})
    roster = pd.DataFrame({"game_id": ["2024_01_A_B"], "team": ["A"], "player_id": ["P1"], "position": ["QB"]})
    before = selection_input_hashes(pbp, schedule, roster)
    grown = pd.concat([pbp, pd.DataFrame([{"game_id": "2026_01_A_B", "play_id": 2, "epa": 2.5}])], ignore_index=True)
    assert grown.epa.dtype == "float64"
    assert selection_input_hashes(grown, schedule, roster) == before


def test_weekly_bundle_lineage_tracks_new_feature_manifest(tmp_path):
    candidate = tmp_path / "bundle"
    candidate.mkdir()
    feature_path = candidate / "feature_manifest.json"
    write_json(feature_path, {"tables": {"team_stats_wide.csv": "new-table-hash"}})
    original_lineage = {"features": "old", "seed": 42}
    training = {"lineage": dict(original_lineage), "trained": {
        "play": {"lineage": dict(original_lineage), "artifact_hashes": {"play_model.pt": "unchanged"}},
        "yac": {"lineage": dict(original_lineage), "artifact_hashes": {"yac_model.pt": "unchanged"}},
    }}
    write_json(candidate / "training_manifest.json", training)
    update_bundle_feature_lineage(candidate)
    changed = json.loads((candidate / "training_manifest.json").read_text())
    assert changed["lineage"]["features"] == digest(feature_path)
    assert changed["lineage"]["seed"] == 42
    assert all(row["lineage"] == changed["lineage"] for row in changed["trained"].values())
    assert changed["trained"]["play"]["artifact_hashes"] == {"play_model.pt": "unchanged"}


def test_weekly_refresh_refuses_to_rewrite_historical_seasons():
    from datetime import datetime
    historical = datetime.now().year - 1
    with pytest.raises(ValueError, match="historical seasons stay frozen"):
        refresh_weekly_bundle(historical)


def test_pm_freshness_allows_only_explicit_stale_groups():
    schedule = pd.DataFrame([{"season": 2026, "week": 7, "home_score": None}])
    estimates = pd.DataFrame([
        {"group": group, "season": 2026, "week": 6}
        for group in ("QB", "WR", "RB", "TE")
    ] + [{"group": "DL", "season": 2025, "week": 18}])

    with pytest.raises(ValueError, match="DL estimates stop at 2026 week nan"):
        check_pm_freshness(estimates, schedule, 2026)

    result = check_pm_freshness(estimates, schedule, 2026, allow_stale_groups=("DL",))
    assert result["required_pm_through"] == [2026, 6]
    assert result["stale_groups"] == {"DL": {"season": 2025, "week": 18, "cross_season_decay": 0.8}}


def test_current_depth_chart_schema_maps_pos_rank_and_checks_team():
    roster = pd.DataFrame([
        {"season": 2026, "week": 5, "team": "PHI", "player_id": "QB1", "position": "QB"},
        {"season": 2026, "week": 5, "team": "PHI", "player_id": "WR1", "position": "WR"},
        {"season": 2026, "week": 5, "team": "DAL", "player_id": "OTHER", "position": "QB"},
    ])
    depth = pd.DataFrame([
        {"dt": "2026-10-05T15:53:14Z", "team": "PHI", "gsis_id": "QB1", "pos_abb": "QB", "pos_rank": 1},
        {"dt": "2026-10-05T15:53:14Z", "team": "PHI", "gsis_id": "WR1", "pos_abb": "WR", "pos_rank": 2},
        {"dt": "2026-10-05T15:53:14Z", "team": "PHI", "gsis_id": "OTHER", "pos_abb": "QB", "pos_rank": 3},
    ])
    result = attach_current_depth_order(roster, depth, 2026).set_index("player_id")
    assert result.loc["QB1", "depth_team"] == 1
    assert result.loc["WR1", "depth_team"] == 2
    assert pd.isna(result.loc["OTHER", "depth_team"])


def test_unplayed_schedule_games_get_latest_roster_and_week_injuries():
    roster = pd.DataFrame([
        {"game_id": "2026_04_A_B", "season": 2026, "week": 4, "team": "A", "player_id": "QB1",
         "position": "QB", "status": "ACT", "report_status": None},
        {"game_id": "2026_04_A_B", "season": 2026, "week": 4, "team": "B", "player_id": "QB2",
         "position": "QB", "status": "ACT", "report_status": None},
    ])
    schedule = pd.DataFrame([
        {"game_id": "2026_04_A_B", "season": 2026, "week": 4, "game_date": "2026-09-28", "home_team": "A", "away_team": "B", "game_type": "REG"},
        {"game_id": "2026_05_A_C", "season": 2026, "week": 5, "game_date": "2026-10-08", "home_team": "A", "away_team": "C", "game_type": "REG"},
        {"game_id": "2026_06_B_D", "season": 2026, "week": 6, "game_date": "2026-10-15", "home_team": "B", "away_team": "D", "game_type": "REG"},
    ])
    injuries = pd.DataFrame([{"season": 2026, "week": 5, "gsis_id": "QB1", "report_status": "Out"}])
    result = add_scheduled_roster_slots(roster, schedule, 2026, injuries)
    projected = result[result.roster_is_projected]
    assert set(projected.game_id) == {"2026_05_A_C", "2026_06_B_D"}
    assert projected.loc[projected.game_id.eq("2026_05_A_C"), "report_status"].iloc[0] == "Out"
    assert projected.roster_source_week.eq(4).all()
    assert not result.loc[~result.roster_is_projected].game_id.isin(["2026_05_A_C", "2026_06_B_D"]).any()


def test_projected_slots_follow_players_latest_team_after_bye():
    roster = pd.DataFrame([
        {"game_id": "2026_06_A_X", "season": 2026, "week": 6, "team": "A", "player_id": "WR1", "position": "WR", "status": "CUT"},
        {"game_id": "2026_06_A_X", "season": 2026, "week": 6, "team": "A", "player_id": "WR2", "position": "WR", "status": "ACT"},
        {"game_id": "2026_07_B_Y", "season": 2026, "week": 7, "team": "B", "player_id": "WR1", "position": "WR", "status": "ACT"},
    ])
    schedule = pd.DataFrame([
        {"game_id": "2026_06_A_X", "season": 2026, "week": 6, "game_date": "2026-10-11", "home_team": "A", "away_team": "X", "game_type": "REG"},
        {"game_id": "2026_07_B_Y", "season": 2026, "week": 7, "game_date": "2026-10-18", "home_team": "B", "away_team": "Y", "game_type": "REG"},
        {"game_id": "2026_08_A_Z", "season": 2026, "week": 8, "game_date": "2026-10-25", "home_team": "A", "away_team": "Z", "game_type": "REG"},
        {"game_id": "2026_08_B_W", "season": 2026, "week": 8, "game_date": "2026-10-25", "home_team": "B", "away_team": "W", "game_type": "REG"},
    ])
    projected = add_scheduled_roster_slots(roster, schedule, 2026)
    projected = projected[projected.roster_is_projected]
    assert set(projected.loc[projected.player_id.eq("WR1"), "team"]) == {"B"}
    assert set(projected.loc[projected.player_id.eq("WR2"), "team"]) == {"A"}


def test_mixed_historical_and_current_depth_ranks_write_to_parquet(tmp_path):
    historical = pd.DataFrame({"game_id": ["2025_18_A_B", "2025_18_A_B"],
                               "player_id": ["P1", "P2"], "depth_team": ["1", "None"]})
    current = pd.DataFrame({"game_id": ["2026_05_A_B"], "player_id": ["P3"],
                            "depth_team": [2.0]})
    combined = normalize_depth_team(pd.concat([historical, current], ignore_index=True))
    assert combined.depth_team.dtype == "float64"
    path = tmp_path / "roster.parquet"
    combined.to_parquet(path, index=False)
    restored = pd.read_parquet(path)
    assert restored.depth_team.tolist()[0] == 1.0
    assert pd.isna(restored.depth_team.tolist()[1])
    assert restored.depth_team.tolist()[2] == 2.0


def test_roster_names_prefer_nickname_and_last_name_then_fallbacks():
    roster = pd.DataFrame([
        {"player_id": "P1", "football_name": "Nickname", "full_name": "Full Name",
         "first_name": "First", "last_name": "Last"},
        {"player_id": "P2", "football_name": None, "first_name": "First", "last_name": "Last"},
        {"player_id": "P3", "football_name": None, "first_name": None, "last_name": None},
        {"player_id": "P4", "football_name": None, "first_name": None, "last_name": None},
    ])
    ids = pd.DataFrame({"gsis_id": ["P3"], "name": ["Mapped Name"]})
    result = attach_roster_player_names(roster, ids)
    assert result.player_name.tolist() == ["Nickname Last", "First Last", "Mapped Name", "P4"]


def test_refreshed_bundle_path_is_applied_before_widget_creation():
    state = {"ridge_bundle_choice": "old", "ridge_bundle_path_pending": "new"}
    assert apply_pending_bundle(state, ["old", "new"]) == "new"
    assert state["ridge_bundle_choice"] == "new"
    assert "ridge_bundle_path_pending" not in state


def test_no_pending_refresh_leaves_widget_path_untouched():
    state = {"ridge_bundle_choice": "current"}
    assert apply_pending_bundle(state) is None
    assert state == {"ridge_bundle_choice": "current"}


def test_pending_bundle_not_in_selector_options_does_not_change_selection():
    state = {"ridge_bundle_choice": "current", "ridge_bundle_path_pending": "missing"}
    assert apply_pending_bundle(state, ["current", "base"]) == "missing"
    assert state["ridge_bundle_choice"] == "current"
