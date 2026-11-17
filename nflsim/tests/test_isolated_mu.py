from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nflsim.features.build import build_feature_tables
from nflsim.configuration import v2_model_manifest
from nflsim.features.builder import build_estimates
from nflsim.features.game_inputs import pregame_player_snapshot
from nflsim.features.lineup_value import apply_lineup_value
from nflsim.features.player_value import PV_COLUMNS
from nflsim.ridge.metrics import (
    RidgeConfig, RidgeTarget, build_design, build_game_context, build_target_rows, fit_coefficients, rolling_predictions,
    weekly_coefficients,
)
from nflsim.sim.runtime import RidgeRuntime, with_opponent_context
from nflsim.semantics import ISOLATED_MU_CONTEXT_V2, ISOLATED_MU_V1, LEGACY_SEMANTICS, semantics_of
from nflsim.features.training_data import context_eligibility
from nflsim.ridge.metrics import TARGETS
from test_pv_ridge_metrics import TARGET, _synthetic_pbp

BINARY = RidgeTarget("first_down", "binary", "player", "receiver_player_id", "first_down",
                     lambda pbp: pbp["complete_pass"].eq(1), "fd_bayes", None, "passer_player_id")


def _inputs(pbp: pd.DataFrame, extra_player: str | None = None):
    schedule = pbp.drop_duplicates("game_id")[["game_id", "game_date", "home_team", "away_team", "stadium_id", "location"]]
    rows = []
    for game in schedule.itertuples():
        for team in (game.home_team, game.away_team):
            players = [f"{team}_P0", f"{team}_P1"] + ([extra_player] if extra_player and game.game_id == schedule.game_id.iloc[-1] else [])
            rows += [{"game_id": game.game_id, "team": team, "player_id": player, "position": "WR"} for player in players]
    return schedule, pd.DataFrame(rows)


def _estimates(pbp, schedule, roster, target=TARGET):
    names = {metric: {} for metric in (target.player_output, target.defense_output) if metric}
    return build_estimates(pbp, schedule, roster, names, [target], predict_seasons=[2023])


def test_rolling_mu_is_only_the_entity_coefficient():
    pbp = _synthetic_pbp(seasons=(2022, 2023), weeks=4)
    rows = build_target_rows(pbp, TARGET, build_game_context(pbp), {})
    design = build_design(rows, "player")
    config = RidgeConfig()
    isolated = rolling_predictions(rows, design, config, [2023])
    legacy = rolling_predictions(rows, design, config, [2023], isolated=False)
    row = isolated.index[rows.loc[isolated.index, "week"].eq(3)][0]
    beta = fit_coefficients(rows, design, config, 2023, 3).set_index(["block", "label"]).coefficient
    assert np.isclose(isolated.loc[row, "pred_subject_neutral"], beta[("subject", rows.loc[row, "subject"])])
    assert np.isclose(isolated.loc[row, "pred_defense_neutral"], beta[("defense", rows.loc[row, "defteam"])])
    assert not np.isclose(legacy.loc[row, "pred_subject_neutral"], isolated.loc[row, "pred_subject_neutral"])
    assert np.allclose(isolated["pred_full"], legacy["pred_full"])
    weekly = weekly_coefficients(rows, design, config)
    assert np.allclose(weekly["mu"], weekly["coefficient"])


def test_published_mu_is_raw_subject_and_defense_coefficient_from_prior_fit():
    pbp = _synthetic_pbp(seasons=(2022, 2023), weeks=4)
    schedule, roster = _inputs(pbp)
    players, teams, context, diagnostics = _estimates(pbp, schedule, roster)
    game_id = schedule.game_id[schedule.game_id.str.startswith("2023_03")].iloc[0]
    games = build_game_context(schedule)
    window = games[(games.season < 2023) | ((games.season == 2023) & (games.week < 3))]
    rows = build_target_rows(pbp[pbp.game_id.isin(window.game_id)], TARGET, window, {p: "WR" for p in roster.player_id})
    beta = fit_coefficients(rows, build_design(rows, "player"), RidgeConfig(), 2023, 3).set_index(["block", "label"]).coefficient
    subject = players[(players.source == "yac_bayes") & (players.game_id == game_id)]
    assert len(subject) == 4 and subject.estimand.eq(ISOLATED_MU_V1).all()
    for row in subject.itertuples():
        assert np.isclose(row.average, beta[("subject", row.player_id)])
    defense = teams[(teams.source == "yac_oe_allowed") & (teams.game_id == game_id)]
    for row in defense.itertuples():
        assert np.isclose(row.average, beta[("defense", row.team)])
    assert diagnostics.reason.eq("intercept_coefficient").any()
    assert (diagnostics[diagnostics.reason.eq("position_prior")].estimate == 0.0).all()


def test_mu_ignores_prediction_game_venue_opponent_and_outcomes():
    pbp = _synthetic_pbp(seasons=(2022, 2023), weeks=4)
    schedule, roster = _inputs(pbp)
    base = _estimates(pbp, schedule, roster)[0]
    target_game = schedule.game_id[schedule.game_id.str.startswith("2023_03")].iloc[0]
    changed_pbp, changed_schedule = pbp.copy(), schedule.copy()
    changed_pbp.loc[changed_pbp.game_id.eq(target_game), "yac_oe"] += 25.0
    for frame in (changed_pbp, changed_schedule):
        frame.loc[frame.game_id.eq(target_game), "stadium_id"] = "NEW_VENUE"
    altered = _estimates(changed_pbp, changed_schedule, roster)[0]
    keys = ["game_id", "player_id", "source"]
    merged = base.merge(altered, on=keys, suffixes=("", "_new"))
    mask = merged.game_id.eq(target_game) | merged.game_id.str.startswith("2023_03") | merged.game_id.str.startswith("2023_02")
    assert mask.any() and np.allclose(merged.loc[mask, "average"], merged.loc[mask, "average_new"])
    later = merged[merged.game_id.str.startswith("2023_04")]
    assert not np.allclose(later.average, later.average_new)


def test_unseen_entity_has_zero_mu_and_binary_mu_is_signed():
    pbp = _synthetic_pbp(seasons=(2022, 2023), weeks=4)
    rng = np.random.default_rng(1)
    pbp["first_down"] = (rng.random(len(pbp)) < np.where(pbp.receiver_player_id.str.endswith("P0"), 0.1, 0.9)).astype(int)
    schedule, roster = _inputs(pbp, extra_player="ROOKIE")
    players, _, _, diagnostics = _estimates(pbp, schedule, roster, BINARY)
    fd = players[players.source == "fd_bayes"]
    assert fd.average.lt(0).any() and fd.average.gt(0).any()
    assert fd.average.between(-1, 1).all()
    rookie = fd[fd.player_id == "ROOKIE"]
    assert len(rookie) > 0 and rookie.average.eq(0.0).all()
    assert diagnostics.reason.eq("unseen_entity_prior").any()


class _Tables:
    isolated_mu = True

    def betas(self, season, week):
        raise AssertionError("isolated schema must not apply PV betas")


def test_isolated_lineup_changes_pv_inputs_only_and_aliases_copy_opposing_mus():
    base = {"epa_rate": 0.1, "def_pass_epa_rate": -0.05, "opp_pass_epa_rate": 0.0, "opp_def_pass_epa_rate": 0.0,
            **{column: 0.0 for column in PV_COLUMNS}}
    features = pd.DataFrame([base, {**base, "epa_rate": 0.3}], index=["BAL", "PIT"])
    player = type("P", (), {"attributes": pd.DataFrame([{"ypc_bayes": 0.2}])})()
    new = {"BAL": {**{c: 0.0 for c in PV_COLUMNS}, "pv_off_qb": -0.6}, "PIT": {c: 0.0 for c in PV_COLUMNS}}
    result = apply_lineup_value(_Tables(), features, {"BAL": [player]}, new, 2025, 3)
    assert result.loc["BAL", "epa_rate"] == 0.1 and result.loc["PIT", "epa_rate"] == 0.3
    assert player.attributes.loc[0, "ypc_bayes"] == 0.2
    assert result.loc["BAL", "pv_off_qb"] == -0.6
    assert result.loc["BAL", "opp_pass_epa_rate"] == 0.3 and result.loc["PIT", "opp_pass_epa_rate"] == 0.1


def test_runtime_rejects_schema_mismatch_between_features_and_models(tmp_path):
    (tmp_path / "training_manifest.json").write_text(json.dumps({"context_map": {}, "feature_semantics": ISOLATED_MU_V1}))
    (tmp_path / "feature_manifest.json").write_text(json.dumps({"tables": {}}))
    assert semantics_of(tmp_path) == LEGACY_SEMANTICS
    with pytest.raises(ValueError, match="does not match"):
        RidgeRuntime(tmp_path)


def test_refresh_refuses_legacy_prior_bundle(tmp_path):
    (tmp_path / "feature_manifest.json").write_text(json.dumps({"tables": {}}))
    from nflsim.paths import ROOT
    with pytest.raises(ValueError, match="legacy-conditioned"):
        build_feature_tables(None, None, None, None, tmp_path / "p.json", ROOT / "tests" / "_tmp" / "refresh_gate",
                             prior_bundle=tmp_path)


def test_unseen_player_snapshot_uses_zero_mu_for_covered_metrics():
    runtime = type("R", (), {"required_player_feature_columns": ["fd_bayes", "deep"], "mu_player_metrics": {"fd_bayes"}})()
    stats = pd.DataFrame({"player_id": ["A", "B"], "game_date": ["2024-01-01"] * 2, "game_id": ["g1", "g2"],
                          "position": ["WR", "WR"], "fd_bayes": [0.4, 0.6], "deep": [0.2, 0.4]})
    row = pregame_player_snapshot(runtime, stats, "NEW", pd.Timestamp("2024-09-01"), "WR")
    assert row.fd_bayes.iloc[0] == 0.0 and np.isclose(row.deep.iloc[0], 0.3)
    legacy = type("R", (), {"required_player_feature_columns": ["fd_bayes", "deep"]})()
    assert np.isclose(pregame_player_snapshot(legacy, stats, "NEW", pd.Timestamp("2024-09-01"), "WR").fd_bayes.iloc[0], 0.5)


def test_v2_paired_defense_output_uses_same_subject_fit_coefficient():
    pbp = _synthetic_pbp(seasons=(2022, 2023), weeks=4)
    schedule, roster = _inputs(pbp)
    target = TARGETS[0]
    parameters = {target.player_output: {}, target.defense_output: {}}
    players, teams, context, _ = build_estimates(
        pbp, schedule, roster, parameters, [target], predict_seasons=[2023],
        feature_semantics=ISOLATED_MU_CONTEXT_V2,
    )
    game_id = schedule.game_id[schedule.game_id.str.startswith("2023_03")].iloc[0]
    history_games = build_game_context(schedule)
    history_games = history_games[
        (history_games.season < 2023) | ((history_games.season == 2023) & (history_games.week < 3))
    ]
    rows = build_target_rows(
        pbp[pbp.game_id.isin(history_games.game_id)], target, history_games,
        {player: "WR" for player in roster.player_id},
    )
    beta = fit_coefficients(rows, build_design(rows, "player"), RidgeConfig(), 2023, 3).set_index(
        ["block", "label"]
    ).coefficient
    game = schedule[schedule.game_id.eq(game_id)].iloc[0]
    defense_team = str(game.away_team)
    published = teams[(teams.source == target.defense_output) & teams.game_id.eq(game_id)
                      & teams.team.eq(defense_team)].average.iloc[0]
    assert np.isclose(published, beta[("defense", defense_team)])
    offense_mu = players[(players.source == target.player_output) & players.game_id.eq(game_id)]
    assert offense_mu.average.notna().all()
    ctx = context[context.game_id.eq(game_id) & context.team.eq(defense_team)].iloc[0]
    assert np.isclose(ctx[f"ridge_{target.name}_defense_mu"], beta[("defense", defense_team)])


def test_v2_unpaired_defense_and_venue_context_are_separate_and_match_fit():
    pbp = _synthetic_pbp(seasons=(2022, 2023), weeks=4)
    pbp["first_down"] = np.where(pbp.receiver_player_id.str.endswith("P0"), 1, 0)
    schedule, roster = _inputs(pbp)
    target = next(item for item in TARGETS if item.name == "first_down")
    players, _, context, _ = build_estimates(
        pbp, schedule, roster, {target.player_output: {}}, [target], predict_seasons=[2023],
        feature_semantics=ISOLATED_MU_CONTEXT_V2,
    )
    game_id = schedule.game_id[schedule.game_id.str.startswith("2023_03")].iloc[0]
    history_games = build_game_context(schedule)
    history_games = history_games[
        (history_games.season < 2023) | ((history_games.season == 2023) & (history_games.week < 3))
    ]
    rows = build_target_rows(
        pbp[pbp.game_id.isin(history_games.game_id)], target, history_games,
        {player: "WR" for player in roster.player_id},
    )
    beta = fit_coefficients(rows, build_design(rows, "player"), RidgeConfig(), 2023, 3).set_index(
        ["block", "label"]
    ).coefficient
    game = schedule[schedule.game_id.eq(game_id)].iloc[0]
    offense_team, defense_team = str(game.home_team), str(game.away_team)
    row = context[context.game_id.eq(game_id) & context.team.eq(offense_team)].iloc[0]
    assert np.isclose(row[f"ridge_{target.name}_opponent_defense_effect"], beta[("defense", defense_team)])
    assert np.isclose(row[f"ridge_{target.name}_venue_effect"], beta[("venue", row.venue)])
    assert np.isclose(row[f"ridge_{target.name}_offense_effect"], beta[("offense_team", offense_team)])
    assert np.isclose(players[(players.source == target.player_output) & players.game_id.eq(game_id)
                              & players.player_id.eq(f"{offense_team}_P0")].average.iloc[0],
                      beta[("subject", f"{offense_team}_P0")])


def test_v2_runtime_rebinds_matched_defense_context_to_actual_opponent():
    mapping = {"ridge_yac_oe_opponent_defense_effect": "ridge_yac_oe_defense_mu"}
    offense = pd.Series({"ridge_yac_oe_defense_mu": 0.15, "ridge_yac_oe_opponent_defense_effect": -0.2})
    opponent = pd.Series({"ridge_yac_oe_defense_mu": -0.35})
    rebound = with_opponent_context(mapping, offense, opponent)
    assert rebound["ridge_yac_oe_opponent_defense_effect"] == -0.35
    assert rebound["ridge_yac_oe_defense_mu"] == 0.15


def test_v2_adds_context_to_cascade_stage_two_and_stage_three_contracts():
    manifest = {"models": {"target": {"context_columns": ["pv_off_qb"],
                                        "sidecar": {"kind": "cascade", "context_columns": ["days_rest"]}}}}
    updated = v2_model_manifest(manifest)
    target = updated["models"]["target"]
    assert updated["feature_semantics"] == ISOLATED_MU_CONTEXT_V2
    assert "ridge_yards_per_target_opponent_defense_effect" in target["context_columns"]
    assert "ridge_yards_per_target_opponent_defense_effect" in target["sidecar"]["context_columns"]
    assert "ridge_yards_per_target_venue_effect" in target["sidecar"]["context_columns"]


def test_v2_missing_optional_context_does_not_exclude_training_rows():
    from nflsim.configuration import context_columns
    from nflsim.semantics import context_available_column, opponent_defense_column

    columns = context_columns("ay", ISOLATED_MU_CONTEXT_V2)
    frame = pd.DataFrame(0.0, index=range(4), columns=columns)
    opponent = opponent_defense_column("interception_worthy")
    available = context_available_column(opponent)
    frame.loc[:, opponent] = np.nan
    frame.loc[:, available] = 0.0
    frame.loc[:, "ridge_cpoe_venue_effect"] = np.nan
    frame.loc[:, context_available_column("ridge_cpoe_venue_effect")] = 0.0
    eligible, diagnostics = context_eligibility(frame, "ay", ISOLATED_MU_CONTEXT_V2)
    assert eligible.all()
    assert diagnostics["excluded_rows"] == 0
    assert diagnostics["unavailable_by_column"][opponent] == len(frame)
    from nflsim.features.training_data import _neutralize_v2_context
    prepared = _neutralize_v2_context(frame)
    assert prepared[opponent].eq(0.0).all()
    assert prepared[available].eq(0.0).all()
    assert prepared["ridge_cpoe_venue_effect"].eq(0.0).all()


def test_optional_context_normalization_preserves_flags_and_is_idempotent():
    from nflsim.features.training_data import _neutralize_v2_context

    frame = pd.DataFrame({
        "ridge_cpoe_opponent_defense_effect": [np.nan, 0.4],
        "ridge_cpoe_opponent_defense_effect_available": [0.0, 1.0],
        "ridge_cpoe_defense_mu_available": [0.0, np.nan],
    })
    result = _neutralize_v2_context(frame)
    assert result["ridge_cpoe_defense_mu_available"].eq(0.0).all()
    assert result["ridge_cpoe_opponent_defense_effect_available"].tolist() == [0.0, 1.0]
    assert not any(column.endswith("_available_available") for column in result)
    pd.testing.assert_frame_equal(result, _neutralize_v2_context(result))


@pytest.mark.parametrize("model,label", [("ay", "air_yards"), ("comp", "complete_pass"),
                                         ("sack", "sack"), ("scramble", "qb_scramble")])
def test_sparse_optional_context_preserves_all_historical_model_rows(model, label):
    from nflsim.configuration import context_columns
    from nflsim.features.training_data import model_frame

    frame = pd.DataFrame({
        "game_id": [f"{season}_01_A_B" for season in range(2017, 2025)],
        "penalty": 0, "qb_dropback": 1, "pass_attempt": 1,
        "air_yards": 5.0, "complete_pass": 1, "sack": 0, "qb_scramble": 0,
    })
    for column in context_columns(model, ISOLATED_MU_CONTEXT_V2):
        frame[column] = 0.1 if not column.endswith("_available") else 1.0
    sparse = [column for column in frame if column.startswith("ridge_interception_worthy_")]
    for column in sparse:
        frame.loc[:4, column] = 0.0 if column.endswith("_available") else np.nan
    selected, target = model_frame(model, frame, pd.DataFrame(), pd.DataFrame(), ISOLATED_MU_CONTEXT_V2)
    assert selected.game_id.tolist() == frame.game_id.tolist()
    assert target.tolist() == frame[label].tolist()
    assert not selected[context_columns(model, ISOLATED_MU_CONTEXT_V2)].isna().any().any()
