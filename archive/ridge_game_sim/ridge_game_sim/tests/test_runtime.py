from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ridge_game_sim.runtime import RidgeRuntime
from torch_game_simulation_runtime import SimulationRuntime


def bare():
    runtime = object.__new__(RidgeRuntime)
    runtime.context_map = {"target": ["ridge_first_down_venue_effect"]}
    return runtime


def test_injects_before_selector_and_forward(monkeypatch):
    runtime = bare()
    captured = []
    monkeypatch.setattr(SimulationRuntime, "forward_model_logits", lambda self, key, frame, base, latent: captured.append((frame.copy(), base)))
    for effect in (2.5, -3.0):
        runtime.forward_model_logits("target", pd.DataFrame([{"down": 1}]), {"ridge_first_down_venue_effect": effect})
    assert captured[0][0].ridge_first_down_venue_effect.iloc[0] == 2.5
    assert captured[1][1]["ridge_first_down_venue_effect"] == -3
    with pytest.raises(ValueError):
        runtime.forward_model_logits("target", pd.DataFrame([{"down": 1}]), {})


def test_prior_rejects_missing_and_nan(monkeypatch):
    runtime = bare()
    monkeypatch.setattr(SimulationRuntime, "transform_with_processor", lambda self, key, frame: frame.to_numpy())
    with pytest.raises(ValueError):
        runtime.transform_with_processor("target_prior", pd.DataFrame([{"other": 0}]))
    with pytest.raises(ValueError):
        runtime.transform_with_processor("target_prior", pd.DataFrame([{"ridge_first_down_venue_effect": np.nan}]))
    result = runtime.transform_with_processor("target_prior", pd.DataFrame([{"ridge_first_down_venue_effect": 0}]))
    assert result[0, 0] == 0