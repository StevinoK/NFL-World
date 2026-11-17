from __future__ import annotations

import json

from nflsim.features.bayesian_history import DECAY_FILE_NOTE, load_decay_rates, save_decay_rates
from nflsim.features.input_sources import FILE_NAME, input_sources, write_input_sources


def _settings(target, side, within_decay):
    return {"target": target, "side": side, "within_decay": within_decay, "across_decay": 1.0, "season_carryover": 0.5,
            "lookback_seasons": 1, "lambda_subject": 30.0, "lambda_defense": 1000.0, "lambda_venue": 100.0,
            "lambda_offense_team": 300.0, "lambda_player_value": 10.0}


def _bundle(path):
    columns = ["down", "ypc_bayes_RB1", "ypc_bayes_RB2", "rush_yards_per_carry", "int_rate", "ridge_rush_yards_venue_effect",
               "rush_rate_RB1", "tied_rate", "rb_top_carry_share_roll4", "pv_off_qb", "kick_st_punt_gross", "days_rest"]
    (path / "model_manifest.json").write_text(json.dumps({"models": {"rush": {"columns": columns, "context_columns": ["weather_outdoor"]}}}))
    (path / "feature_manifest.json").write_text(json.dumps({
        "retained_player_metrics": ["rush_rate"], "retained_team_metrics": ["tied_rate", "rb_top_carry_share_roll4"],
        "usage": {"columns": ["rush_rate"]}, "player_value": {"columns": ["pv_off_qb"]},
        "special_teams": {"columns": ["st_punt_gross"]}, "weather": {"columns": ["weather_outdoor"]}}))
    (path / "ridge_selection_parameters.json").write_text(json.dumps({"max_selection_season": 2024, "parameters": {
        "ypc_bayes": _settings("rush_yards", "subject", 0.92), "rush_yards_per_carry": _settings("rush_yards", "defense", 0.97),
        "int_rate": _settings("interception", "defense", 0.85), "cpoe_deep": _settings("cpoe_deep", "subject", 1.0)}}))
    (path / "retained_diagnostics.csv").write_text(
        "metric,reason\ntied_rate,season_frozen_prior_preweek_update\nrb_top_carry_share_roll4,direct_shared_definition_with_pregame_slots\n")


def test_table_reports_the_fit_that_applies(tmp_path):
    _bundle(tmp_path)
    table = input_sources(tmp_path).set_index("input")
    # A paired defense stat and its context read the offense fit, whatever was tuned under their own names.
    assert table.loc["rush_yards_per_carry", "built_by"] == "ridge defense block"
    assert table.loc["rush_yards_per_carry", "settings_from"] == "ypc_bayes"
    assert table.loc["rush_yards_per_carry", "within_decay"] == 0.92
    assert table.loc["ridge_rush_yards_venue_effect", "settings_from"] == "ypc_bayes"
    assert table.loc["int_rate", "built_by"] == "ridge defense fit"
    assert table.loc["int_rate", "within_decay"] == 0.85
    assert "lambda_defense=max" in table.loc["int_rate", "at_grid_edge"]
    assert table.loc["ypc_bayes", "slots"] == "RB1, RB2"
    assert table.loc["cpoe_deep", "model_count"] == 0
    # The smoother's fixed decay applies, and usage shares win over a retained listing of the same name.
    assert table.loc["tied_rate", "smoother_decay"] == 0.8
    assert table.loc["rb_top_carry_share_roll4", "built_by"] == "retained, not smoothed"
    assert table.loc["rush_rate", "built_by"] == "usage shares"
    assert {table.loc[name, "built_by"] for name in ("pv_off_qb", "kick_st_punt_gross", "weather_outdoor", "days_rest", "down")} == {
        "player value", "special teams", "weather", "schedule", "game state"}


def test_write_skips_bundles_without_flat_models(tmp_path):
    write_input_sources(tmp_path)
    assert not (tmp_path / FILE_NAME).exists()
    _bundle(tmp_path)
    write_input_sources(tmp_path)
    assert (tmp_path / FILE_NAME).is_file()


def test_decay_file_keeps_its_warning_and_loads(tmp_path):
    path = tmp_path / "rates.json"
    save_decay_rates({"tied_rate": {"decay": 0.9, "season_carryover": 0.8}}, path)
    assert json.loads(path.read_text())["note"] == DECAY_FILE_NOTE
    assert load_decay_rates(path) == {"tied_rate": {"decay": 0.9, "season_carryover": 0.8}}
    assert len(load_decay_rates()) == 62
