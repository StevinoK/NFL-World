import pandas as pd
import pytest

from ridge_game_sim.compare_2025 import compare


def fixture():
    return pd.DataFrame([{"game_id": f"G{index}", "week": index+1, "home_team": "A", "away_team": "B", "actual_home_score": 20+index, "actual_away_score": 17, "actual_spread": 3+index, "actual_total": 37+index, "sim_spread": 2+index, "sim_total": 40, "sim_home_win_prob": 0.6, "sim_home_median": 21, "sim_away_median": 19, "simulations_run": 200, "spread_line": 3, "total_line": 39} for index in range(4)])


def test_identical_arms_zero_delta_reproducible():
    data = fixture()
    table, summary = compare(data, data, 25, 7)
    assert table.delta_ridge_minus_baseline.eq(0).all()
    assert table.paired_ci_low.eq(0).all()
    repeated, _ = compare(data, data, 25, 7)
    pd.testing.assert_frame_equal(table, repeated)
    assert summary["common_games"] == 4


def test_missing_unequal_counts_and_mismatched_games():
    baseline = fixture()
    ridge = baseline.iloc[:3].copy()
    ridge.simulations_run = 5
    table, summary = compare(ridge, baseline, 10)
    assert summary["baseline_excluded"] == ["G3"]
    assert summary["unequal_simulation_counts"]
    ridge.loc[0, "away_team"] = "C"
    with pytest.raises(ValueError, match="away_team"):
        compare(ridge, baseline, 0)


def test_provider_aliases_match_without_changing_game_ids():
    baseline = fixture().assign(home_team="LA")
    ridge = baseline.assign(home_team="LAR")
    table, summary = compare(ridge, baseline, 0)
    assert summary["common_games"] == len(baseline)
    assert summary["game_ids"] == sorted(baseline.game_id)
    assert table.delta_ridge_minus_baseline.eq(0).all()
    assert baseline.home_team.eq("LA").all()


def test_wrong_spread_sign_rejected():
    frame = fixture()
    frame.actual_spread *= -1
    with pytest.raises(ValueError, match="sign"):
        compare(frame, fixture(), 0)