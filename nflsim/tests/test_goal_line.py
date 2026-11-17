import numpy as np
import pandas as pd
import pytest
import torch
from coral_pytorch.losses import corn_loss

from nflsim.engine.simulation_runtime import PlayRow, SimulationRuntime
from nflsim.features.goal_line import GOAL_LINE_COLUMNS, goal_line_inputs, with_goal_line_inputs, yards_available
from nflsim.models.fit_minimal import reduced_spec
from nflsim.models.game_effects import Data, ModelRows, local_terms, row_moments
from nflsim.models.train import goal_capped_corn_loss


def test_goal_line_inputs_match_for_scalars_and_arrays():
    many = goal_line_inputs(np.array([-2.0, 0.0, 4.0, 50.0]))
    assert set(many) == set(GOAL_LINE_COLUMNS)
    assert many["goal_5"].tolist() == [0, 0, 4, 5]
    assert many["goal_20"].tolist() == [0, 0, 4, 20]
    one = goal_line_inputs(4.0)
    assert {key: float(value) for key, value in one.items()} == pytest.approx({key: float(value[2]) for key, value in many.items()})


def test_yards_after_catch_start_where_the_throw_ended():
    frame = pd.DataFrame({"yardline_100": [10, 30], "air_yards": [4, 30]})
    assert yards_available("yac", frame).tolist() == [6, 0]
    assert yards_available("rush", frame).tolist() == [10, 30]
    assert with_goal_line_inputs("yac", frame).goal_5.tolist() == [5, 0]


def test_capped_loss_matches_corn_when_nothing_is_capped():
    torch.manual_seed(0)
    logits, labels = torch.randn(32, 5), torch.randint(0, 6, (32,))
    capped = torch.zeros(32, dtype=torch.bool)
    assert goal_capped_corn_loss(logits, labels, capped).item() == pytest.approx(corn_loss(logits, labels, 6).item(), rel=1e-5)


def test_capped_row_says_nothing_about_stopping_at_the_cap():
    logits = torch.zeros(1, 4, requires_grad=True)
    goal_capped_corn_loss(logits, torch.tensor([2]), torch.tensor([True])).backward()
    gradient = logits.grad[0].tolist()
    assert gradient[0] < 0 and gradient[1] < 0  # It did get past the earlier yards.
    assert gradient[2] == 0 and gradient[3] == 0  # Whether it stopped here or went deeper is unknown.


def test_game_effects_skip_the_stop_term_on_capped_rows():
    base, label, delta = torch.zeros(1, 3, dtype=torch.double), torch.tensor([1]), torch.zeros(1, 1, dtype=torch.double)
    exact = local_terms("corn", base, label, delta)
    capped = local_terms("corn", base, label, delta, capped=torch.tensor([True]))
    assert capped[0].item() > exact[0].item()
    assert capped[1].item() == pytest.approx(0.5)  # Only the step it passed pushes the shift.
    assert capped[2].item() == pytest.approx(0.25)


def test_stat_moments_cut_draws_at_the_goal_line():
    logits, label = torch.full((1, 3), 50.0, dtype=torch.double), torch.tensor([1])  # Nearly all mass on the last bin.
    uncut = row_moments("corn", logits, label, None, np.array([0.0, 5.0, 10.0, 40.0]))
    cut = row_moments("corn", logits, label, None, np.array([0.0, 5.0, 10.0, 40.0]), torch.tensor([5.0], dtype=torch.double))
    assert uncut[0].item() == pytest.approx(40)
    assert cut[0].item() == pytest.approx(5)
    assert cut[2].item() == 5


def test_data_marks_capped_rows_and_keeps_them_through_subsets():
    rows = {"ay": ModelRows("ay", "corn", 3, np.zeros((3, 3)), np.array([1, 2, 1]), np.array([0.0, 5.0, 10.0, 40.0]),
                            np.array(["a", "a", "b"]), np.array(["X", "X", "Y"]), np.array([5.0, 40.0, 30.0]))}
    data = Data(rows)
    assert data.items["ay"]["capped"].tolist() == [True, False, False]
    subset = data.subset(["a"])
    assert subset.items["ay"]["capped"].tolist() == [True, False]
    assert subset.items["ay"]["limit"].tolist() == [5.0, 40.0]


def test_only_yardage_models_get_goal_line_training():
    spec = {"columns": ["yardline_100"], "groups": [], "context_columns": [], "sidecar": {"kind": "corn"}}
    assert reduced_spec("ay", spec)["columns"][-len(GOAL_LINE_COLUMNS):] == list(GOAL_LINE_COLUMNS)
    assert reduced_spec("ay", spec)["sidecar"]["goal_capped"] is True
    assert "goal_capped" not in reduced_spec("sack", spec)["sidecar"]
    assert not set(GOAL_LINE_COLUMNS) & set(reduced_spec("sack", spec)["columns"])


def test_runtime_adds_goal_line_inputs_only_where_trained():
    runtime = SimulationRuntime.__new__(SimulationRuntime)
    assert runtime.with_goal_line_inputs("ay", PlayRow(down=1), 4) == {"down": 1}  # Older bundles.
    runtime.goal_line_models = frozenset({"ay"})
    row = runtime.with_goal_line_inputs("ay", PlayRow(down=1), 4)
    assert row["goal_3"] == 3 and row["goal_10"] == 4
    assert runtime.with_goal_line_inputs("yac", PlayRow(down=1), 4) == {"down": 1}
