from __future__ import annotations

import numpy as np
import pandas as pd

from nflsim.features.player_value import EXTENDED_PV_COLUMNS, PV_COLUMNS
from nflsim.ridge.epa_luck import luck_adjusted_epa
from nflsim.ridge.expectations import EXPECTATIONS, add_expectations, context_features
from nflsim.ridge.metrics import (
    TARGETS_BY_NAME, RidgeConfig, attach_player_value, build_design, regime_multiplier,
)


def _rows():
    return pd.DataFrame({
        "subject": ["p1", "p1", "p1", "p2", "p2"],
        "posteam": ["AAA", "AAA", "BBB", "CCC", "CCC"],
        "offense_coach": ["Coach A", "Coach A", "Coach B", "Coach C", "Coach D"],
        "season": [2023, 2024, 2024, 2024, 2024],
        "week": [5, 1, 2, 1, 2],
    })


def test_regime_multiplier_defaults_to_unchanged_weights():
    np.testing.assert_array_equal(regime_multiplier(_rows(), 2024, 3, RidgeConfig()), np.ones(5))


def test_regime_multiplier_discounts_rows_before_team_and_coach_changes():
    config = RidgeConfig(team_change_carryover=0.5, coach_change_carryover=0.8)
    weights = regime_multiplier(_rows(), 2024, 3, config)
    # p1 now plays for BBB under Coach B: older AAA rows changed team and coach.
    np.testing.assert_allclose(weights[:3], [0.4, 0.4, 1.0])
    # p2 stayed on CCC but the head coach changed.
    np.testing.assert_allclose(weights[3:], [0.8, 1.0])
    # Only games before the prediction week define the current regime.
    np.testing.assert_allclose(regime_multiplier(_rows(), 2024, 2, config)[:3], [1.0, 1.0, 0.4])
    assert "_ct0.5_cc0.8" in config.label() and "_ct" not in RidgeConfig().label()


def test_extended_player_value_columns_enter_design_by_side():
    rows = pd.DataFrame({"game_id": ["g1"], "posteam": ["AAA"], "defteam": ["BBB"], "subject": ["p1"],
                         "venue": ["V"], "is_home": [1.0], "rest_diff": [0.0], "position": ["WR"]})
    table = pd.DataFrame({"game_id": ["g1", "g1"], "team": ["AAA", "BBB"],
                          **{column: [1.0, 2.0] for column in EXTENDED_PV_COLUMNS}})
    attached = attach_player_value(rows, table)
    assert attached["pv_off_ol"].item() == 1.0 and attached["pv_def_lb"].item() == 2.0
    design = build_design(attached, "player")
    assert list(design.labels[design.blocks == "player_value"]) == EXTENDED_PV_COLUMNS
    base = attach_player_value(rows, table[["game_id", "team", *PV_COLUMNS]])
    assert list(build_design(base, "player").labels[build_design(base, "player").blocks == "player_value"]) == PV_COLUMNS


def test_context_features_parse_weather_text_and_neutralize_domes():
    pbp = pd.DataFrame({
        "weather": ["Light Rain Temp: 41° F, Humidity: 90%, Wind: NW 18 mph", "Snow Temp: 28° F, Wind: N 5 mph",
                    "Sunny, no rain Temp: 70° F", "Rain Temp: 40° F"],
        "temp": [np.nan, np.nan, np.nan, np.nan], "wind": [np.nan, 6.0, np.nan, np.nan],
        "roof": ["outdoors", "open", "outdoors", "dome"], "surface": ["grass", "fieldturf", "a_turf", "sportturf"],
    })
    context = context_features(pbp)
    assert context["ctx_temp"].tolist()[:2] == [41.0, 28.0]
    assert context["ctx_wind"].tolist()[:2] == [18.0, 6.0]
    assert context["ctx_rain"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert context["ctx_snow"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert context.loc[3, ["ctx_outdoor", "ctx_temp", "ctx_wind"]].tolist() == [0.0, 70.0, 0.0]


def test_expectations_are_out_of_sample_by_season():
    rng = np.random.default_rng(0)
    n = 3000
    seasons = np.repeat([2022, 2023, 2024], n // 3)
    wind = rng.uniform(0, 25, n)
    pbp = pd.DataFrame({
        "game_id": [f"{season}_01_G{index % 40}" for season, index in zip(seasons, range(n))],
        "season": seasons, "rush_attempt": 1, "down": rng.integers(1, 5, n), "ydstogo": rng.integers(1, 15, n),
        "yardline_100": rng.integers(1, 99, n), "qtr": 1, "half_seconds_remaining": 900, "score_differential": 0,
        "shotgun": 0, "no_huddle": 0, "goal_to_go": 0, "roof": "outdoors", "wind": wind, "temp": 60.0,
    })
    pbp["yards_gained"] = 5.0 - 0.15 * wind + rng.normal(0, 1, n)
    out = add_expectations(pbp, ["rush_yards"])
    assert out["rush_yards__expected"].notna().all()
    # Wind lowers expected rushing yards and the residual is centered.
    assert np.corrcoef(out["ctx_wind"], out["rush_yards__expected"])[0, 1] < -0.8
    assert abs(out.loc[out.season == 2024, "rush_yards__xoe"].mean()) < 0.2
    # Changing 2024 outcomes cannot change the 2023 expectations (trained on earlier seasons only).
    shifted = pbp.copy()
    shifted.loc[shifted.season == 2024, "yards_gained"] += 10
    again = add_expectations(shifted, ["rush_yards"])
    np.testing.assert_allclose(again.loc[again.season == 2023, "rush_yards__expected"],
                               out.loc[out.season == 2023, "rush_yards__expected"])
    assert set(EXPECTATIONS) <= set(TARGETS_BY_NAME)


def test_luck_adjusted_epa_charges_turnovers_at_league_average():
    rng = np.random.default_rng(0)
    n = 4000
    pbp = pd.DataFrame({
        "game_id": np.where(np.arange(n) < n // 2, "2023_01_AAA_BBB", "2024_01_AAA_BBB"),
        "epa": rng.normal(0, 1, n), "pass": 1, "rush": 0, "yardline_100": 75.0,
        "interception": 0, "fumble": 0, "wp": 0.5,
    })
    picks = np.arange(0, n, 50)
    pbp.loc[picks, "interception"] = 1
    pbp.loc[picks, "epa"] = np.where(np.arange(len(picks)) % 2 == 0, -2.0, -9.0)  # pick vs pick-six
    pbp.loc[n - 1, "wp"] = 0.01
    adjusted = luck_adjusted_epa(pbp, garbage_wp=0.05)
    late_picks = [index for index in picks if index >= n // 2]
    # 2024 picks are all charged the 2023 average, whatever their own return was.
    np.testing.assert_allclose(adjusted[late_picks], -5.5)
    assert np.isnan(adjusted[n - 1])
    # Extremes are clipped to quantiles of earlier seasons; changing 2024 cannot move 2023 values.
    assert adjusted[pbp["interception"].eq(0)].max() < pbp["epa"].max()
    shifted = pbp.assign(epa=np.where(np.arange(n) >= n // 2, pbp["epa"] + 5, pbp["epa"]))
    np.testing.assert_allclose(luck_adjusted_epa(shifted)[: n // 2], luck_adjusted_epa(pbp)[: n // 2])


def test_production_targets_read_over_expectation_columns_under_the_same_names():
    from nflsim.ridge.expectations import PRODUCTION_XOE_TARGETS, production_targets
    from nflsim.ridge.metrics import TARGETS

    adopted = {target.name: target for target in production_targets(TARGETS)}
    assert [target.name for target in TARGETS] == list(adopted)
    for name, target in adopted.items():
        original = TARGETS_BY_NAME[name]
        assert (target.player_output, target.defense_output) == (original.player_output, original.defense_output)
        assert target.value_col == (f"{name}__xoe" if name in PRODUCTION_XOE_TARGETS else original.value_col)


def test_with_production_expectations_adds_only_new_columns():
    from nflsim.ridge.expectations import PRODUCTION_XOE_TARGETS, with_production_expectations

    rng = np.random.default_rng(1)
    n = 2000
    seasons = np.repeat([2023, 2024], n // 2)
    pbp = pd.DataFrame({
        "game_id": [f"{season}_01_G{index % 30}" for season, index in zip(seasons, range(n))], "season": seasons,
        "down": rng.integers(1, 5, n), "ydstogo": rng.integers(1, 15, n), "yardline_100": rng.integers(1, 99, n),
        "pass_attempt": 1, "complete_pass": rng.integers(0, 2, n), "target": 1, "rush_attempt": 1, "sack": 0,
        "explosive": 0, "first_down": rng.integers(0, 2, n), "yards_gained": rng.normal(5, 3, n),
        "air_yards": rng.normal(8, 5, n), "roof": "dome", "unrelated": 7,
    })
    out = with_production_expectations(pbp)
    added = {f"{name}__{kind}" for name in PRODUCTION_XOE_TARGETS for kind in ("expected", "xoe")}
    assert set(out.columns) == set(pbp.columns) | added
    assert with_production_expectations(out) is out
