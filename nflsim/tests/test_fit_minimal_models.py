from __future__ import annotations

import json
from pathlib import Path

from nflsim.engine.simulation_runtime import SimulationRuntime
from nflsim.models.fit_minimal import NOISY_COLUMNS, reduced_spec


REFERENCE = Path(__file__).resolve().parents[2] / "archive/nflsim_bundles/pv-isolated-mu-neutral-pilot-r2-2025"


def test_completion_schema_keeps_features_pv_and_single_paired_defense_input():
    manifest = json.loads((REFERENCE / "model_manifest.json").read_text())
    spec = reduced_spec("comp", manifest["models"]["comp"])
    selected = set(spec["columns"] + spec["context_columns"])

    assert {"air_yards", "cpoe_bayes", "pass_oe", "pv_off_qb", "pv_def_dl"} <= selected
    assert {"medium_cpoe_allowed", "short_cpoe_allowed", "sack_rate"} <= selected
    assert not selected & NOISY_COLUMNS
    assert "ridge_cpoe_opponent_defense_effect" in selected
    assert not any(column.endswith("_available") for column in selected)
    assert not (selected & {"def_pass_epa_rate", "def_rush_epa_rate"})


def test_play_schema_keeps_original_features_and_deduplicates_team_epa_defense():
    manifest = json.loads((REFERENCE / "model_manifest.json").read_text())
    spec = reduced_spec("play", manifest["models"]["play"])
    selected = set(spec["columns"] + spec["context_columns"])

    assert {"down", "score_differential", "pass_oe", "team_4th_rate", "pv_off_qb", "weather_wind_speed"} <= selected
    assert "ridge_cpoe_opponent_defense_effect" in selected
    assert "ridge_pass_epa_opponent_defense_effect" in selected
    assert "ridge_rush_epa_opponent_defense_effect" in selected
    assert "ridge_cpoe_venue_effect" in selected
    assert "ridge_cpoe_offense_effect" in selected
    assert "passer_columns" in spec["groups"]
    assert set(SimulationRuntime.passer_columns) - NOISY_COLUMNS <= set(spec["columns"])
    assert not selected & NOISY_COLUMNS
    assert "def_pass_epa_rate" not in selected
    assert "opp_def_pass_epa_rate" not in selected
    assert "def_rush_epa_rate" not in selected
    assert "opp_def_rush_epa_rate" not in selected
    assert not any(column.endswith("_available") for column in selected)