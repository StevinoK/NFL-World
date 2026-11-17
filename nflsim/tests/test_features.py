import json
import numpy as np
import pandas as pd
import pytest

from nflsim.ridge.metrics import TARGETS
from nflsim.features.player_value import PV_COLUMNS
from nflsim.features.builder import build_estimates, overlay, wide_tables, skill_roster


def test_full_provider_roster_filters_unusable_records_with_diagnostics():
    pbp, games, roster = fixture()
    extra = pd.DataFrame([
        {"game_id": games.game_id.iloc[0], "team": "A", "player_id": "DEF", "position": "DB"},
        {"game_id": games.game_id.iloc[0], "team": "A", "player_id": "UNKNOWN", "position": None},
        {"game_id": games.game_id.iloc[0], "team": "A", "player_id": None, "position": "WR"},
        {"game_id": games.game_id.iloc[0], "team": "A", "player_id": " ", "position": "QB"},
    ])
    raw = pd.concat([roster, extra], ignore_index=True)
    cleaned, diagnostics = skill_roster(raw)
    pd.testing.assert_frame_equal(cleaned.reset_index(drop=True), roster.reset_index(drop=True), check_dtype=False)
    assert {row["reason"]: row["count"] for row in diagnostics} == {
        "roster_missing_position": 1, "roster_non_skill_position": 1, "roster_missing_player_id": 2,
    }
    expected = build_estimates(pbp, games, roster, {"yac_bayes": {}, "yac_oe_allowed": {}}, [TARGETS[0]])
    actual = build_estimates(pbp, games, raw, {"yac_bayes": {}, "yac_oe_allowed": {}}, [TARGETS[0]])
    for original, filtered in zip(expected[:3], actual[:3]):
        pd.testing.assert_frame_equal(original, filtered)
    assert actual[3].reason.eq("roster_missing_player_id").any()


def test_missing_roster_game_key_remains_an_error():
    _, _, roster = fixture()
    roster.loc[0, "game_id"] = None
    with pytest.raises(ValueError, match="game/team"):
        skill_roster(roster)
from nflsim.features import training_data
from nflsim.features.training_data import SimulationRuntime, model_frame, prepare_frame


def fixture():
    games, plays, roster = [], [], []
    for week in range(1, 5):
        game = dict(game_id=f"2024_{week:02d}_B_A", game_date=f"2024-09-{week*7:02d}", home_team="A", away_team="B", stadium_id="NEUTRAL" if week == 3 else "STADIUM", location="Neutral" if week == 3 else "Home")
        games.append(game)
        for team, defense in (("A", "B"), ("B", "A")):
            roster.append(dict(game_id=game["game_id"], team=team, player_id=team+"P", position="WR", player_name=None))
            plays.append(dict(**game, posteam=team, defteam=defense, receiver_player_id=team+"P", passer_player_id=team+"Q", complete_pass=1, yac_oe=float(week)+(team == "A")))
    roster.append(dict(game_id=games[2]["game_id"], team="A", player_id="ROOKIE", position="WR", player_name=None))
    return pd.DataFrame(plays), pd.DataFrame(games), pd.DataFrame(roster)


def test_preweek_cutoff_rookies_and_context():
    pbp, games, roster = fixture()
    params = {"yac_bayes": {}, "yac_oe_allowed": {}}
    before = build_estimates(pbp, games, roster, params, [TARGETS[0]])
    changed = pbp.copy()
    changed.loc[changed.game_id >= "2024_03", "yac_oe"] += 1000
    after = build_estimates(changed, games, roster, params, [TARGETS[0]])
    for original, altered in zip(before[:3], after[:3]):
        selected = original.game_id <= "2024_03_B_A"
        pd.testing.assert_frame_equal(original[selected], altered[selected])
    assert "ROOKIE" in set(before[0].player_id)
    assert "2024_01_B_A" not in set(before[0].game_id)
    neutral = before[2][before[2].game_id == "2024_03_B_A"]
    assert neutral.is_home.eq(0).all()
    assert neutral.venue.eq("NEUTRAL").all()
    assert neutral.ridge_yac_oe_venue_effect.eq(0).all()
    assert before[3].reason.eq("unseen_entity_prior").any()


def test_weather_context_is_shared_across_both_team_rows():
    pbp, games, roster = fixture()
    games = games.assign(roof="outdoors", temp=40.0, wind=15.0)
    player, team, context, _ = build_estimates(
        pbp, games, roster, {"yac_bayes": {}, "yac_oe_allowed": {}}, [TARGETS[0]])
    _, team_wide = wide_tables(player, team, context)
    expected = {"weather_outdoor": 1.0, "weather_temp_delta": -1.0,
                "weather_wind_speed": 1.0, "weather_missing": 0.0}
    for column, value in expected.items():
        assert team_wide[column].eq(value).all()


def test_active_season_filter_preserves_prior_season_fitting_history():
    pbp24, games24, roster24 = fixture()
    pbp23, games23, roster23 = [frame.copy() for frame in (pbp24, games24, roster24)]
    for frame in (pbp23, games23, roster23):
        frame["game_id"] = frame.game_id.astype(str).str.replace(r"^2024_", "2023_", regex=True)
        if "game_date" in frame:
            frame["game_date"] = pd.to_datetime(frame.game_date) - pd.Timedelta(days=365)
    pbp, games, roster = (pd.concat([old, current], ignore_index=True)
                          for old, current in zip((pbp23, games23, roster23), (pbp24, games24, roster24)))
    params = {"yac_bayes": {}, "yac_oe_allowed": {}}
    all_seasons = build_estimates(pbp, games, roster, params, [TARGETS[0]])
    refreshed = build_estimates(pbp, games, roster, params, [TARGETS[0]], predict_seasons=[2024])
    expected = all_seasons[0][all_seasons[0].game_id.str.startswith("2024_")].reset_index(drop=True)
    actual = refreshed[0].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)
    assert refreshed[0].game_id.str.startswith("2024_").all()
    assert set(refreshed[3].season.astype(int)) == {2024}


def test_overlay_preserves_uncovered_and_both_means():
    pbp, games, roster = fixture()
    player, team, context, _ = build_estimates(pbp, games, roster, {"yac_bayes": {}, "yac_oe_allowed": {}}, [TARGETS[0]])
    original = player.copy()
    original.average = -999
    original.bayes_estimate_mean = -998
    uncovered = player.iloc[:1].assign(source="usage", average=0.75, bayes_estimate_mean=0.75)
    combined = overlay(pd.concat([original, uncovered]), player, "player_id")
    assert combined[combined.source == "usage"].average.iloc[0] == 0.75
    assert np.allclose(combined[combined.source == "yac_bayes"].average, player.average)
    player_wide, team_wide = wide_tables(combined, team, context)
    assert "ROOKIE" in set(player_wide.player_id)
    assert player_wide.player_name.notna().all()
    assert "ridge_yac_oe_offense_effect" in team_wide


def test_wide_tables_combine_csv_and_timestamp_dates_for_same_game():
    pbp, games, roster = fixture()
    player, team, context, _ = build_estimates(pbp, games, roster, {"yac_bayes": {}, "yac_oe_allowed": {}}, [TARGETS[0]])
    retained = team.copy().assign(source="retained_usage", average=0.5, bayes_estimate_mean=0.5)
    retained["game_date"] = retained.game_date.astype(str)
    mixed = pd.concat([retained, team], ignore_index=True)
    _, wide = wide_tables(player, mixed, context)
    assert not wide.duplicated(["team", "game_id"]).any()
    assert wide.retained_usage.eq(0.5).all()
    assert wide.yac_oe_allowed.notna().all()


def test_prepared_roster_shape_uses_schedule_timing():
    pbp, games, roster = fixture()
    roster = roster.assign(season=2024, week=99, game_date="1999-01-01", depth_team=1, report_status=None)
    player, _, _, _ = build_estimates(pbp, games, roster, {"yac_bayes": {}, "yac_oe_allowed": {}}, [TARGETS[0]])
    assert set(player.game_id) == set(games.game_id.iloc[1:])
    assert player.game_date.min().year == 2024


@pytest.mark.parametrize("alias,canonical", [("LA", "LAR"), ("STL", "LAR"), ("SD", "LAC"), ("OAK", "LV"), ("JAC", "JAX"), ("WSH", "WAS")])
def test_team_aliases_share_fit_and_roster_keys(alias, canonical):
    pbp, games, roster = fixture()
    games = games.replace({"A": canonical})
    roster = roster.replace({"A": alias})
    pbp = pbp.replace({"A": alias})
    result = build_estimates(pbp, games, roster, {"yac_bayes": {}, "yac_oe_allowed": {}}, [TARGETS[0]])
    expected = build_estimates(pbp.replace({alias: canonical}), games, roster.replace({alias: canonical}), {"yac_bayes": {}, "yac_oe_allowed": {}}, [TARGETS[0]])
    for actual, canonical_frame in zip(result, expected):
        pd.testing.assert_frame_equal(actual, canonical_frame)
    assert canonical in set(result[1].team)


@pytest.mark.parametrize("column,value", [("team", "UNKNOWN"), ("game_id", "2024_99_B_A")])
def test_unmatched_roster_keys_are_rejected(column, value):
    pbp, games, roster = fixture()
    roster.loc[0, column] = value
    with pytest.raises(ValueError, match="Unmatched roster keys"):
        build_estimates(pbp, games, roster, {"yac_bayes": {}, "yac_oe_allowed": {}}, [TARGETS[0]])


def test_sparse_ftn_warmup_is_unavailable_and_not_future_filled():
    pbp, games, roster = fixture()
    target = next(target for target in TARGETS if target.name == "interception_worthy")
    pbp[target.value_col] = [np.nan, np.nan, np.nan, np.nan, 1, 0, 0, 1]
    pbp["pass_attempt"] = 1
    params = {metric: {} for metric in (target.player_output, target.defense_output) if metric}
    result = build_estimates(pbp, games, roster, params, [target])
    columns = ["ridge_interception_worthy_venue_effect", "ridge_interception_worthy_offense_effect"]
    context = result[2]
    assert context.loc[context.game_id.le("2024_03_B_A"), columns].isna().all().all()
    assert np.isfinite(context.loc[context.game_id.eq("2024_04_B_A"), columns]).all().all()
    assert result[3].availability.eq("unavailable_warmup").any()
    changed = pbp.copy()
    changed.loc[changed.game_id.ge("2024_03_B_A"), target.value_col] = 0
    altered = build_estimates(changed, games, roster, params, [target])[2]
    pd.testing.assert_frame_equal(context[context.game_id.le("2024_03_B_A")], altered[altered.game_id.le("2024_03_B_A")])


def test_model_specific_context_eligibility_and_label_alignment():
    frame = pd.DataFrame({"penalty": [0, 0, 0, 1], "qb_dropback": [1, 1, 1, 1], "interception": [0, 1, 0, 1], "sack": [1, 0, 0, 0], "qb_scramble": 0, "air_yards": [np.nan, 5.0, 5.0, 5.0], "ridge_interception_worthy_venue_effect": [np.nan, 0, np.inf, 0], "ridge_interception_worthy_offense_effect": [np.nan, 0.2, 0, 0], "ridge_sack_venue_effect": [0, 0, 0, 0], "ridge_sack_offense_effect": [0, 0, 0, 0]}, index=[10, 20, 30, 40])
    frame = frame.assign(**{column: 0.1 for column in PV_COLUMNS})
    eligible, labels = model_frame("int", frame, pd.DataFrame(), pd.DataFrame())
    assert labels.tolist() == [1]
    assert len(eligible) == 1
    diagnostics = eligible.attrs["ridge_context_eligibility"]
    assert diagnostics["candidate_rows"] == 2
    assert diagnostics["excluded_rows"] == 1
    assert diagnostics["unavailable_by_column"]["ridge_interception_worthy_venue_effect"] == 1
    other, labels = model_frame("sack", frame, pd.DataFrame(), pd.DataFrame())
    assert len(other) == 3
    assert labels.tolist() == [1, 0, 0]


def test_completion_and_interception_train_on_throws_only():
    frame = pd.DataFrame({"game_id": "2024_01_A_B", "penalty": 0, "qb_dropback": 1, "sack": [1, 0, 0, 0], "qb_scramble": [0, 1, 0, 0],
                          "air_yards": [np.nan, np.nan, np.nan, 8.0], "complete_pass": [0, 0, 0, 1], "interception": 0})
    for model in ("comp", "int"):
        assert model_frame(model, frame, pd.DataFrame(), pd.DataFrame())[0].air_yards.tolist() == [8.0]
    assert len(model_frame("sack", frame, pd.DataFrame(), pd.DataFrame())[0]) == 4
    assert len(model_frame("scramble", frame, pd.DataFrame(), pd.DataFrame())[0]) == 4


@pytest.mark.parametrize("comp, interception, expected", [("attempts", "attempts", True), (None, None, False)])
def test_runtime_reads_how_pass_outcomes_were_trained(tmp_path, comp, interception, expected):
    runtime = SimulationRuntime.__new__(SimulationRuntime)
    runtime.data_dir = tmp_path
    (tmp_path / "comp_arch.json").write_text(json.dumps({"row_scope": comp}))
    (tmp_path / "int_arch.json").write_text(json.dumps({"row_scope": interception}))
    assert runtime._pass_outcomes_given_attempt() is expected


def test_runtime_refuses_mixed_pass_outcome_training(tmp_path):
    runtime = SimulationRuntime.__new__(SimulationRuntime)
    runtime.data_dir = tmp_path
    (tmp_path / "comp_arch.json").write_text(json.dumps({"row_scope": "attempts"}))
    (tmp_path / "int_arch.json").write_text("{}")
    with pytest.raises(ValueError, match="disagree"):
        runtime._pass_outcomes_given_attempt()


def test_prepare_frame_preserves_sparse_context_and_canonical_team_joins(monkeypatch):
    monkeypatch.setattr(training_data, "_filter_non_meaningful_games", lambda frame: frame)
    frame = pd.DataFrame({"game_id": ["2024_01_LA_B", "2024_02_LA_B"], "season": [2024, 2024], "season_type": ["REG", "REG"], "aborted_play": [0, 0], "penalty": [0, 0], "posteam": ["LA", "LA"], "defteam": ["WSH", "WSH"], "play_id": [1, 1], "game_seconds_remaining": [300, 300], "kick_distance": [np.nan, np.nan], "return_yards": [0, 0], "receiver_player_id": ["WR", "WR"], "passer_player_id": ["QB", "QB"], "qb_dropback": [1, 1], "interception": [0, 1], "sack": [1, 0], "qb_scramble": [0, 0], "air_yards": [np.nan, 5.0]})
    players = pd.DataFrame([{"game_id": game, "player_id": player} for game in frame.game_id for player in ("WR", "QB")])
    for column in set(SimulationRuntime.receiver_columns + SimulationRuntime.passer_columns + SimulationRuntime.qb_context_columns):
        players[column] = 0.5
    teams = pd.DataFrame([{"game_id": game, "team": team} for game in frame.game_id for team in ("LAR", "WAS")])
    for column in set(SimulationRuntime.def_cols + SimulationRuntime.team_stats_cols + SimulationRuntime.pregame_context_columns):
        teams[column] = 0.25
    teams["ridge_interception_worthy_venue_effect"] = [np.nan, np.nan, 0, 0]
    teams["ridge_interception_worthy_offense_effect"] = [np.nan, np.nan, 0.1, 0.1]
    teams["ridge_sack_venue_effect"] = 0
    teams["ridge_sack_offense_effect"] = 0
    for column in PV_COLUMNS:
        teams[column] = 0.1
    prepared = prepare_frame(frame, players, teams)
    assert prepared.posteam.eq("LAR").all()
    assert prepared.defteam.eq("WAS").all()
    assert prepared[SimulationRuntime.def_cols].notna().all().all()
    assert prepared.attrs["ridge_context_eligibility"]["int"]["excluded_rows"] == 1
    assert prepared.attrs["ridge_context_eligibility"]["sack"]["excluded_rows"] == 0
    assert prepared.attrs["ridge_context_eligibility"]["comp"]["excluded_rows"] == 2
    selected, labels = model_frame("int", prepared, players, pd.DataFrame())
    assert selected.game_id.tolist() == ["2024_02_LA_B"]
    assert labels.tolist() == [1]
    assert len(model_frame("sack", prepared, players, pd.DataFrame())[0]) == 2


def test_scramble_rows_carry_the_scrambling_qbs_passer_stats(monkeypatch):
    monkeypatch.setattr(training_data, "_filter_non_meaningful_games", lambda frame: frame)
    frame = pd.DataFrame({"game_id": ["2024_01_LA_B"] * 2, "season": [2024, 2024], "season_type": ["REG", "REG"], "aborted_play": [0, 0], "penalty": [0, 0], "posteam": ["LA", "LA"], "defteam": ["WSH", "WSH"], "play_id": [1, 2], "game_seconds_remaining": [300, 290], "kick_distance": [np.nan, np.nan], "return_yards": [0, 0], "receiver_player_id": ["WR", None], "passer_player_id": ["QB", None], "rusher_player_id": [None, "QB"], "qb_scramble": [0, 1], "qb_dropback": [1, 1]})
    players = pd.DataFrame([{"game_id": "2024_01_LA_B", "player_id": player} for player in ("WR", "QB")])
    for column in set(SimulationRuntime.receiver_columns + SimulationRuntime.passer_columns + SimulationRuntime.qb_context_columns):
        players[column] = 0.5
    teams = pd.DataFrame([{"game_id": "2024_01_LA_B", "team": team} for team in ("LAR", "WAS")])
    for column in set(SimulationRuntime.def_cols + SimulationRuntime.team_stats_cols + SimulationRuntime.pregame_context_columns):
        teams[column] = 0.25
    prepared = prepare_frame(frame, players, teams)
    # nflverse has no passer on scrambles; missing passer stats would identify every scramble
    assert prepared[SimulationRuntime.passer_columns].notna().all().all()


def test_sparse_target_without_any_observations_has_stable_context_schema():
    pbp, games, roster = fixture()
    target = next(target for target in TARGETS if target.name == "interception_worthy")
    pbp[target.value_col] = np.nan
    pbp["pass_attempt"] = 1
    player, team, context, diagnostics = build_estimates(pbp, games, roster, {target.player_output: {}}, iter([target]))
    assert player.empty and team.empty
    assert context.filter(like="ridge_interception_worthy_").isna().all().all()
    assert len(context.filter(like="ridge_interception_worthy_").columns) == 2
    assert diagnostics.availability.eq("unavailable_warmup").all()
    plays = context.assign(penalty=0, qb_dropback=1, interception=0, sack=0, qb_scramble=0, air_yards=5.0)
    selected, labels = model_frame("int", plays, pd.DataFrame(), pd.DataFrame())
    assert selected.empty and labels.empty
    assert selected.attrs["ridge_context_eligibility"]["excluded_rows"] == len(plays)


def test_zero_weight_history_is_unavailable_not_zero_context():
    pbp, games, roster = fixture()
    _, _, context, diagnostics = build_estimates(pbp, games, roster, {"yac_bayes": {"within_decay": 0}, "yac_oe_allowed": {"within_decay": 0}}, [TARGETS[0]])
    assert context.filter(like="ridge_").isna().all().all()
    assert diagnostics.reason.eq("no_positive_history_weights").any()


def test_only_recent_four_seasons_of_ftn_are_eligible_after_first_observation():
    pbp, games, roster = fixture()
    pbp, games, roster = [pd.concat([frame.iloc[:2].assign(game_id=f"{season}_01_B_A", game_date=f"{season}-09-01") for season in range(2017, 2025)], ignore_index=True) for frame in (pbp, games.iloc[:1], roster)]
    target = next(target for target in TARGETS if target.name == "interception_worthy")
    pbp["pass_attempt"] = 1
    pbp[target.value_col] = np.where(pbp.game_id.str[:4].astype(int).ge(2021), 1, np.nan)
    _, _, context, _ = build_estimates(pbp, games, roster, {target.player_output: {}}, [target])
    plays = context.assign(penalty=0, qb_dropback=1, interception=0, sack=0, qb_scramble=0, air_yards=5.0, **{column: 0.1 for column in PV_COLUMNS})
    selected, _ = model_frame("int", plays, pd.DataFrame(), pd.DataFrame())
    assert set(selected.game_id.str[:4]) == {"2022", "2023", "2024"}
    assert plays.loc[plays.game_id.str[:4].le("2021"), "ridge_interception_worthy_venue_effect"].isna().all()