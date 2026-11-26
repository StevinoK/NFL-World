import json

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nflsim.engine.model_architecture import build_network
from nflsim.models.game_effects import out_of_fold_logits
from nflsim.models.train import HOLDOUT_FOLDS, fit_network, game_folds, holdout_games, holdout_split


def frame(games=60, plays=10, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.repeat([f"2024_{index:02d}_A_B" for index in range(games)], plays)
    return pd.DataFrame({"game_id": ids, "game_team_id": ids, "skill": rng.normal(size=len(ids))})


def test_folds_split_games_and_fold_zero_is_the_shared_holdout():
    data = frame()
    folds = game_folds(data.game_id)
    assert set(folds) == set(range(HOLDOUT_FOLDS)) and np.array_equal(folds == 0, holdout_games(data.game_id))
    train, test = holdout_split(data, 3)
    assert set(folds[test]) == {3} and 3 not in set(folds[train])
    everything, none = holdout_split(data, None)
    assert len(everything) == len(data) and len(none) == 0


def test_the_final_network_trains_on_every_game():
    data = frame()
    labels = pd.Series((data.skill > 0).astype(int))
    sidecar = {"kind": "flat", "dims": {"output_dim": 2}, "arch": {"hidden_dims": [4], "dropout": [0.0]},
               "hyperparameters": {"batch_size": 64}, "holdout_fold": None}
    model, processor, probabilities, actual, classes, history, _ = fit_network(data, labels, ["skill"], sidecar, epochs=2)
    assert probabilities.shape == (0, 2) and len(actual) == 0 and history
    with pytest.raises(ValueError, match="flat and CORN"):
        fit_network(data, labels, ["skill"], {**sidecar, "kind": "hierarchical", "prior_columns": ["skill"],
                                              "dims": {"output_dim": 2, "latent_dim": 1}}, epochs=1)


def test_out_of_fold_logits_come_from_the_network_that_held_the_game_out(tmp_path):
    data = frame()
    processor = ColumnTransformer([("num", StandardScaler(), ["skill"]),
                                   ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [])]).fit(data)
    for fold in range(HOLDOUT_FOLDS):
        network = build_network("flat", {"input_dim": 1, "output_dim": 2}, {"hidden_dims": [2], "dropout": [0.0]})
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network.out.bias.copy_(torch.tensor([0.0, float(fold)]))      # every row: logits (0, fold)
        torch.save(network.state_dict(), tmp_path / f"m_fold{fold}_model.pt")
        joblib.dump(processor, tmp_path / f"m_fold{fold}_preprocessor.pkl")
    sidecar = {"kind": "flat", "arch": {"hidden_dims": [2], "dropout": [0.0]}, "cross_fit_folds": HOLDOUT_FOLDS}
    (tmp_path / "m_arch.json").write_text(json.dumps(sidecar))
    logits = out_of_fold_logits(tmp_path, "m", "m", sidecar, data)
    assert np.array_equal(logits[:, 1], game_folds(data.game_id).astype(float))
    with pytest.raises(ValueError, match="cross-fit"):
        out_of_fold_logits(tmp_path, "m", "m", {"kind": "flat"}, data)
