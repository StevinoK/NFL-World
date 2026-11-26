import json

import numpy as np
import pandas as pd
import pytest
import torch

from nflsim.engine.fast_inputs import CompiledProcessor
from nflsim.engine.model_architecture import MultitaskHead, build_network
from nflsim.models import multitask


def tasks(rows=3000, seed=0):
    """Three models that share a situation signal (x1, x2) and each have an input of their own."""
    rng = np.random.default_rng(seed)
    games = np.repeat([f"2024_{index:03d}_A_B" for index in range(rows // 10)], 10)
    shared = pd.DataFrame({"game_id": games, "x1": rng.normal(size=rows), "x2": rng.normal(size=rows),
                           "roof": rng.choice(["dome", "outdoors"], rows)})
    signal = shared.x1 - 0.5 * shared.x2
    first = shared.assign(own_a=rng.normal(size=rows))
    second = shared.assign(own_b=rng.normal(size=rows), side=rng.choice(["L", "R", None], rows))
    third = shared.assign(own_c=rng.normal(size=rows))
    a = (signal + first.own_a + rng.normal(size=rows) > 0).astype(int)
    b = pd.Series(np.select([signal + second.own_b < -0.5, signal + second.own_b > 0.5], [0, 2], 1))
    yards = np.clip(np.round(2 * signal + third.own_c + rng.normal(size=rows) + 4), 0, 8)
    flat = {"kind": "flat", "dims": {"output_dim": 2}}
    return [multitask.make_task("a", "a", flat, first, a, ["x1", "x2", "roof", "own_a"]),
            multitask.make_task("b", "b", {"kind": "flat", "dims": {"output_dim": 3, "class_labels": [0, 1, 2]}},
                                second, b, ["x1", "x2", "roof", "own_b", "side"]),
            multitask.make_task("c", "c", {"kind": "corn", "dims": {}}, third, pd.Series(yards), ["x2", "x1", "roof", "own_c"])]


def test_heads_share_one_trunk_and_scale_shared_inputs_alike():
    heads, processors, dims, metrics, history, common, arch = multitask.train(
        tasks(), {"trunk_hidden_dims": [16], "embedding_dim": 8, "head_hidden_dims": [8]}, epochs=4, batch_size=256)
    assert common == ["x1", "x2", "roof"]
    assert heads["a"].trunk is heads["b"].trunk is heads["c"].trunk
    scales = {model: dict(zip(processor.named_transformers_["num"].feature_names_in_, processor.named_transformers_["num"].mean_))
              for model, processor in processors.items()}
    assert scales["a"]["x1"] == scales["b"]["x1"] == scales["c"]["x1"]
    assert all(np.isfinite(item["test_log_loss"]) for item in metrics.values()) and history
    assert dims["c"]["output_dim"] == 8                                   # CORN: classes - 1 logits
    # Every head beats guessing its classes evenly.
    assert metrics["a"]["test_log_loss"] < np.log(2) and metrics["b"]["test_log_loss"] < np.log(3)


def test_a_saved_head_reloads_and_runs_the_fast_input_path(tmp_path):
    work = tasks(rows=1000)
    heads, processors, dims, metrics, history, common, arch = multitask.train(
        work, {"trunk_hidden_dims": [8], "embedding_dim": 4, "head_hidden_dims": [4]}, epochs=1, batch_size=256)
    base = tmp_path / "base"
    base.mkdir()
    for task in work:
        (base / f"{task.prefix}_arch.json").write_text(json.dumps(task.sidecar))
    output = tmp_path / "out"
    multitask.write_refit(base, output, work, heads, processors, dims, metrics, history, common, arch)
    sidecar = json.loads((output / "b_arch.json").read_text())
    assert sidecar["kind"] == "multitask" and sidecar["head"] == "flat" and sidecar["dims"]["class_labels"] == [0, 1, 2]
    loaded = build_network("multitask", sidecar["dims"], sidecar["arch"])
    loaded.load_state_dict(torch.load(output / "b_model.pt"))
    loaded.eval()
    frame = work[1].frame.iloc[:5]
    sklearn_inputs = multitask.transform(processors["b"], frame)
    assert torch.allclose(loaded(sklearn_inputs), heads["b"](sklearn_inputs))
    compiled = CompiledProcessor(processors["b"], None)
    assert CompiledProcessor.supports(processors["b"])
    for index, row in enumerate(frame.to_dict("records")):
        values, _ = compiled.transform(row)
        assert np.allclose(np.clip(np.nan_to_num(values), -10, 10), sklearn_inputs[index].numpy(), atol=1e-5)
    assert json.loads((output / "multitask_summary.json").read_text())["common_columns"] == ["x1", "x2", "roof"]
    with pytest.raises(ValueError, match="not empty"):
        multitask.write_refit(base, output, work, heads, processors, dims, metrics, history, common, arch)


def test_the_runtime_checks_a_heads_input_width():
    from nflsim.engine.simulation_runtime import SimulationRuntime
    head = MultitaskHead({"input_dim": 5, "common_index": [0, 1], "own_index": [2, 3, 4], "output_dim": 2},
                         {"trunk_hidden_dims": [4], "embedding_dim": 3, "head_hidden_dims": [3]})
    runtime = SimulationRuntime.__new__(SimulationRuntime)
    runtime.model_dict, runtime.schema_issues, runtime.strict_schema = {"m": head}, [], True
    assert runtime.as_model_input("m", np.zeros((1, 5))).shape == (1, 5)
    with pytest.raises(ValueError, match="below expected width 5"):
        runtime.as_model_input("m", np.zeros((1, 4)))
