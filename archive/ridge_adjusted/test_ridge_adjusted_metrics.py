from __future__ import annotations

import numpy as np
import pandas as pd

from ridge_adjusted_metrics import (
    RidgeConfig,
    RidgeTarget,
    aggregate_output_predictions,
    build_design,
    build_game_context,
    build_target_rows,
    fit_coefficients,
    recency_weights,
    rolling_predictions,
    solve_ridge,
    weekly_coefficients,
)


TEAMS = ("AAA", "BBB", "CCC", "DDD")
VENUE_EFFECT = {"AAA": -0.4, "BBB": 0.0, "CCC": 0.0, "DDD": 0.0}
DEFENSE_EFFECT = {"AAA": 0.3, "BBB": -0.3, "CCC": 0.1, "DDD": -0.1}


def _synthetic_pbp(seasons=(2022, 2023), weeks=8, plays_per_game=40, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for season in seasons:
        for week in range(1, weeks + 1):
            order = rng.permutation(TEAMS)
            for home, away in ((order[0], order[1]), (order[2], order[3])):
                game_id = f"{season}_{week:02d}_{away}_{home}"
                game_date = pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=7 * week)
                for offense, defense in ((home, away), (away, home)):
                    for play in range(plays_per_game):
                        player = f"{offense}_P{play % 2}"
                        value = (
                            0.5
                            + VENUE_EFFECT[home]
                            + DEFENSE_EFFECT[defense]
                            + (0.2 if offense == home else 0.0)
                            + rng.normal(0.0, 0.05)
                        )
                        records.append(
                            {
                                "game_id": game_id,
                                "game_date": game_date,
                                "home_team": home,
                                "away_team": away,
                                "stadium_id": f"{home}_STADIUM",
                                "location": "Home",
                                "posteam": offense,
                                "defteam": defense,
                                "receiver_player_id": player,
                                "passer_player_id": f"{offense}_QB",
                                "complete_pass": 1,
                                "yac_oe": value,
                            }
                        )
    return pd.DataFrame.from_records(records)


TARGET = RidgeTarget(
    "yac_oe", "normal", "player", "receiver_player_id", "yac_oe",
    lambda pbp: pbp["complete_pass"].eq(1), "yac_bayes", "yac_oe_allowed", "passer_player_id",
)


def _rows(pbp: pd.DataFrame, extra_blocks=()) -> pd.DataFrame:
    return build_target_rows(pbp, TARGET, build_game_context(pbp), {}, extra_blocks=extra_blocks)


def test_player_matrix_includes_both_teams_and_venue_in_one_fit() -> None:
    rows = _rows(_synthetic_pbp(seasons=(2023,), weeks=2, plays_per_game=2))
    design = build_design(rows, "player")
    assert {"subject", "offense_team", "defense", "venue", "home", "rest"}.issubset(set(design.blocks))
    row = rows.index[0]
    active = {(design.blocks[index], design.labels[index]) for index in design.matrix[row].indices}
    assert ("subject", rows.loc[row, "subject"]) in active
    assert ("offense_team", rows.loc[row, "posteam"]) in active
    assert ("defense", rows.loc[row, "defteam"]) in active
    assert ("venue", rows.loc[row, "venue"]) in active
    assert "home" in set(design.blocks) and "rest" in set(design.blocks)


def test_team_matrix_uses_subject_as_offense_team() -> None:
    rows = _rows(_synthetic_pbp(seasons=(2023,), weeks=2, plays_per_game=2))
    team_rows = rows.assign(subject=rows["posteam"])
    design = build_design(team_rows, "team")
    assert "offense_team" not in set(design.blocks)
    assert {"subject", "defense", "venue", "home", "rest"}.issubset(set(design.blocks))
    row = team_rows.index[0]
    active = {(design.blocks[index], design.labels[index]) for index in design.matrix[row].indices}
    assert ("subject", team_rows.loc[row, "posteam"]) in active
    assert ("defense", team_rows.loc[row, "defteam"]) in active


def test_aggregated_rows_match_play_level_fit() -> None:
    pbp = _synthetic_pbp(seasons=(2023,), weeks=4, plays_per_game=6)
    rows = _rows(pbp)
    design = build_design(rows, "player")
    config = RidgeConfig()
    weights = np.ones(len(rows)) * rows["count"].to_numpy()
    beta_aggregated = solve_ridge(design.matrix, rows["rate"].to_numpy(), weights, design.penalty(config))

    keys = ["subject", "posteam", "defteam", "game_id"]
    plays = pbp.rename(columns={"receiver_player_id": "subject"})[keys + ["yac_oe"]].merge(
        rows[keys + ["venue", "is_home", "rest_diff", "position"]], on=keys
    )
    play_design = build_design(plays, "player")
    beta_play = solve_ridge(
        play_design.matrix, plays["yac_oe"].to_numpy(), np.ones(len(plays)), play_design.penalty(config)
    )
    assert np.allclose(beta_aggregated, beta_play, atol=1e-8)


def test_predictions_are_pregame() -> None:
    pbp = _synthetic_pbp()
    rows = _rows(pbp)
    design = build_design(rows, "player")
    baseline = rolling_predictions(rows, design, RidgeConfig(), [2023])
    changed = rows.copy()
    target_week = (changed["season"] == 2023) & (changed["week"] == 4)
    changed.loc[target_week, "rate"] += 50.0
    altered = rolling_predictions(changed, design, RidgeConfig(), [2023])
    week = rows.loc[baseline.index, "week"]
    assert np.allclose(baseline.loc[week <= 4, "pred_full"], altered.loc[week <= 4, "pred_full"])
    assert not np.allclose(baseline.loc[week > 4, "pred_full"], altered.loc[week > 4, "pred_full"])


def test_recovers_venue_defense_and_home_effects() -> None:
    pbp = _synthetic_pbp(seasons=(2021, 2022, 2023), weeks=17)
    rows = _rows(pbp)
    design = build_design(rows, "player")
    config = RidgeConfig(within_decay=1.0, across_decay=1.0, season_carryover=1.0, lookback_seasons=3,
                         lambda_subject=1.0, lambda_defense=1.0, lambda_venue=1.0, lambda_position=1.0)
    coefficients = fit_coefficients(rows, design, config, 2024, 1).set_index(["block", "label"])["coefficient"]
    assert abs(coefficients[("home", "is_home")] - 0.2) < 0.03
    venue = coefficients.xs("venue")
    assert venue["AAA_STADIUM"] - venue.drop("AAA_STADIUM").mean() < -0.3
    defense = coefficients.xs("defense")
    assert defense["AAA"] > defense["CCC"] > defense["DDD"] > defense["BBB"]


def test_recency_weights_follow_season_rules() -> None:
    config = RidgeConfig(within_decay=0.9, across_decay=0.99, season_carryover=0.5, lookback_seasons=1)
    seasons = np.array([2023, 2023, 2024, 2024, 2022])
    weeks = np.array([17, 18, 1, 2, 18])
    last = np.array([18, 18, 18, 18, 18])
    weights = recency_weights(seasons, weeks, last, 2024, 2, config)
    assert np.isclose(weights[2], 0.9)
    assert weights[3] == 0.0
    assert weights[4] == 0.0
    assert weights[2] > weights[1]
    assert np.isclose(weights[0] / weights[1], 0.99)


def test_past_opponent_value_updates_retroactively() -> None:
    pbp = _synthetic_pbp(seasons=(2023,), weeks=10)
    rows = _rows(pbp)
    design = build_design(rows, "player")
    config = RidgeConfig(lambda_defense=30.0)
    early = fit_coefficients(rows, design, config, 2023, 3).set_index(["block", "label"])["coefficient"]
    late = fit_coefficients(rows, design, config, 2023, 10).set_index(["block", "label"])["coefficient"]
    assert abs(late[("defense", "AAA")] - early[("defense", "AAA")]) > 1e-6


def test_weekly_coefficients_are_pregame_and_include_entity_estimates() -> None:
    rows = _rows(_synthetic_pbp(seasons=(2022, 2023), weeks=4))
    design = build_design(rows, "player")
    config = RidgeConfig()
    history = weekly_coefficients(rows, design, config)
    changed = rows.copy()
    changed.loc[(changed["season"] == 2023) & (changed["week"] == 2), "rate"] += 10.0
    altered = weekly_coefficients(changed, design, config)
    early = history[history["season"].eq(2023) & history["week"].le(2)].reset_index(drop=True)
    other = altered[altered["season"].eq(2023) & altered["week"].le(2)].reset_index(drop=True)
    assert np.allclose(early["estimate"], other["estimate"])
    late = history[history["season"].eq(2023) & history["week"].ge(3)]
    other_late = altered[altered["season"].eq(2023) & altered["week"].ge(3)]
    assert not np.allclose(late["estimate"], other_late["estimate"])
    subject = early[early["block"].eq("subject")]
    assert np.isfinite(subject["estimate"]).all()
    assert {"subject", "defense", "venue", "offense_team"}.issubset(set(early["block"]))


def test_teammate_rows_aggregate_to_subject_game() -> None:
    pbp = _synthetic_pbp(seasons=(2022, 2023), weeks=4)
    rows = _rows(pbp, extra_blocks=("teammate",))
    design = build_design(rows, "player", extra_blocks=("teammate",))
    predictions = rolling_predictions(rows, design, RidgeConfig(), [2023])
    aggregated = aggregate_output_predictions(rows, predictions, "subject")
    assert not aggregated.duplicated(["entity", "game_id"]).any()
    assert np.isfinite(aggregated[["pred_full", "pred_neutral"]].to_numpy()).all()
