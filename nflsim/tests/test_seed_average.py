import numpy as np
import pandas as pd
import pytest
import torch

from nflsim.models.seed_average import fit_copy, network_probabilities, soft_loss, soft_targets


def test_probabilities_sum_to_one_for_both_heads():
    logits = np.random.default_rng(0).normal(size=(20, 5))
    for kind, width in (("corn", 6), ("flat", 5)):
        probabilities = network_probabilities(kind, logits)
        assert probabilities.shape == (20, width)
        assert probabilities.sum(1) == pytest.approx(np.ones(20))
        assert (probabilities >= 0).all()


def test_corn_targets_rebuild_the_probabilities():
    logits = np.random.default_rng(1).normal(size=(10, 4))
    probabilities = network_probabilities("corn", logits)
    target, weight = soft_targets("corn", probabilities)
    assert target.numpy() == pytest.approx(1 / (1 + np.exp(-logits)), abs=1e-5)
    assert weight[:, 0].numpy() == pytest.approx(np.ones(10))  # Every play reaches the first step.


@pytest.mark.parametrize("kind", ["corn", "flat"])
def test_loss_is_lowest_at_the_averaged_probabilities(kind):
    logits = torch.from_numpy(np.random.default_rng(2).normal(size=(50, 4))).float()
    target, weight = soft_targets(kind, network_probabilities(kind, logits.numpy().astype(float)))
    exact = soft_loss(kind, logits, target, weight)
    assert exact < soft_loss(kind, logits + 0.5 * torch.randn(50, 4), target, weight)


def test_copy_learns_the_averaged_probability():
    rng = np.random.default_rng(3)
    games = np.array([f"g{index % 200}" for index in range(4000)])
    frame = pd.DataFrame({"x": rng.normal(size=4000), "game_id": games})
    chance = 1 / (1 + np.exp(-2 * frame.x.to_numpy()))
    probabilities = np.stack([1 - chance, chance], axis=1).astype(np.float32)
    sidecar = {"kind": "flat", "arch": {"hidden_dims": [16], "dropout": [0.0], "activation": "gelu", "norm": "none"}, "hyperparameters": {"batch_size": 256, "learning_rate": 0.01}}
    network, processor, _ = fit_copy(frame, probabilities, ["x"], sidecar, seed=7, epochs=40)
    with torch.no_grad():
        inputs = torch.from_numpy(processor.transform(frame[["x"]]).astype(np.float32))
        learned = torch.softmax(network(inputs), 1)[:, 1].numpy()
    assert np.abs(learned - chance).mean() < 0.03


def test_copy_refuses_a_changing_learning_rate():
    frame = pd.DataFrame({"x": np.arange(30.0), "game_id": [f"g{index % 6}" for index in range(30)]})
    sidecar = {"kind": "flat", "arch": {"hidden_dims": [4]}, "hyperparameters": {"scheduler": "cosine"}}
    with pytest.raises(ValueError):
        fit_copy(frame, np.full((30, 2), 0.5, dtype=np.float32), ["x"], sidecar, seed=1, epochs=1)
