def test_resume_checks_committed_artifact_hashes_and_lineage(tmp_path):
    from nflsim.models.train import verify_completed_training
    from nflsim.paths import digest, write_json
    path = tmp_path / "checkpoint.json"
    write_json(path, {"test": True})
    lineage = {"seed": 42, "features": "frozen"}
    record = {"lineage": lineage, "artifact_hashes": {path.name: digest(path)}}
    verify_completed_training("rusher", record, tmp_path, lineage)
    with pytest.raises(ValueError, match="provenance"):
        verify_completed_training("rusher", record, tmp_path, {"seed": 1})
    write_json(path, {"test": False})
    with pytest.raises(ValueError, match="artifact changed"):
        verify_completed_training("rusher", record, tmp_path, lineage)

def test_undefined_cascade_auroc_is_recorded_without_hiding_invalid_loss(tmp_path):
    import json
    from nflsim.models.train import cascade_metric_record
    from nflsim.paths import write_json
    metrics = {"stage1": {"val_metrics": {"auroc_macro_ovr": float("nan"), "log_loss": 0.4}},
               "stage3": {"val_metrics": {"auroc_macro_ovr": np.float32(np.nan), "accuracy": 0.8}}}
    record = cascade_metric_record(metrics)
    write_json(tmp_path / "training.json", record)
    saved = json.loads((tmp_path / "training.json").read_text())
    assert saved["metrics"]["stage1"]["val_metrics"]["auroc_macro_ovr"] is None
    assert saved["undefined_metrics"] == ["stage1.val_metrics.auroc_macro_ovr", "stage3.val_metrics.auroc_macro_ovr"]
    assert np.isnan(metrics["stage1"]["val_metrics"]["auroc_macro_ovr"])
    with pytest.raises(ValueError, match="log_loss"):
        cascade_metric_record({"log_loss": float("nan")})

def test_ftn_merge_season_columns_are_normalized():
    from nflsim.features.training_data import normalize_training_season
    frame = pd.DataFrame({"game_id": ["2017_01_A_B", "2024_01_A_B", "2025_01_A_B"],
                          "season_x": [2017, 2024, 2025], "season_y": [np.nan, 2024, 2025],
                          "week_x": [1, 1, 1], "week_y": [np.nan, 1, 1]})
    normalized = normalize_training_season(frame)
    assert normalized.season.tolist() == [2017, 2024, 2025]
    assert normalized.week.tolist() == [1, 1, 1]
    assert "season" not in frame
    frame.loc[1, "season_y"] = 2025
    with pytest.raises(ValueError, match="season_y disagrees"):
        normalize_training_season(frame)


def test_missing_source_season_uses_validated_game_id():
    from nflsim.features.training_data import normalize_training_season
    assert normalize_training_season(pd.DataFrame({"game_id": ["2024_01_A_B"]})).season.iloc[0] == 2024
    with pytest.raises(ValueError, match="invalid season"):
        normalize_training_season(pd.DataFrame({"game_id": ["unknown"]}))

import copy
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from nflsim.engine.cascade_trainer import CascadeConfig, fit_cascade_general
from nflsim.models import train as train_models
from nflsim.models.train import fit_network, train_one
from nflsim.features.training_data import SKILL_SLOT_ORDER, slot_frame


def synthetic():
    rng = np.random.default_rng(17)
    return pd.DataFrame({"game_id": np.repeat(np.arange(50).astype(str), 12), "game_team_id": np.repeat(np.arange(50).astype(str), 12), "season": 2024, "skill": rng.normal(size=600), "ridge_cpoe_venue_effect": np.repeat(rng.normal(size=50), 12)})


@pytest.mark.parametrize("kind,classes", [("flat", 2), ("corn", 4)])
def test_reduced_classifier_and_ordinal(kind, classes):
    torch.set_num_threads(1)
    frame = synthetic()
    labels = pd.Series(np.arange(len(frame)) % classes)
    sidecar = {"kind": kind, "dims": {"output_dim": classes}, "arch": {"hidden_dims": [8], "dropout": [0.0]}, "hyperparameters": {"batch_size": 64}}
    model, processor, probs, actual, bins, history, prior = fit_network(frame, labels, ["skill", "ridge_cpoe_venue_effect"], sidecar, epochs=2)
    assert probs.shape == (len(actual), classes)
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(1), 1)
    assert model.fc1.in_features == len(processor.get_feature_names_out())
    assert len(history) == 2
    assert not any("Torch Modeling" in name for name in sys.modules)


def test_reduced_cascade_has_context_all_stages():
    torch.set_num_threads(1)
    frame = synthetic()
    frame["label"] = (np.arange(len(frame)) % 2).astype(str)
    config = CascadeConfig(epochs=1, stage2_epochs=1, hidden_dims=(8,), stage2_hidden_dims=(8,), n_bins=8, batch_size=64)
    result = fit_cascade_general(frame, "label", ["skill", "ridge_cpoe_venue_effect"], ["ridge_cpoe_venue_effect"], class_labels=["0", "1"], config=config, group_column="game_team_id", device="cpu")
    assert "ridge_cpoe_venue_effect" in result.stage1_preprocessor.feature_names_in_
    assert "ridge_cpoe_venue_effect" in result.stage3_preprocessor.feature_names_in_
    assert result.context_columns == ["ridge_cpoe_venue_effect"]
    assert result.stage2_model.net[0].in_features == 1


def test_future_training_rejected(tmp_path):
    frame = synthetic().assign(season=2025)
    with pytest.raises(ValueError, match="2017-2024"):
        train_one("sack", frame, pd.Series(np.zeros(len(frame))), {}, tmp_path, 1, 42)


def test_slot_labels_follow_fixed_order():
    roster = pd.DataFrame([{"game_id": "G", "team": team, "player_id": team+slot, "position": slot[:-1], "depth_team": int(slot[-1]), "report_status": ""} for team in ("A", "B") for slot in SKILL_SLOT_ORDER])
    players = roster[["game_id", "player_id"]].assign(rush_rate=lambda frame: np.arange(len(frame), 0, -1), medium=lambda frame: np.arange(len(frame), 0, -1), ypt=2)
    frame = pd.DataFrame([{"game_id": "G", "posteam": "A", "receiver_player_id": "AWR2"}, {"game_id": "G", "posteam": "B", "receiver_player_id": "BWR2"}])
    result = slot_frame(frame, players, roster, "receiver", ["ypt"])
    assert result.selection_label.eq(2).all()
    assert result.QB1.tolist() == ["AQB1", "BQB1"]


@pytest.mark.parametrize("scheduler", ["none", "cosine", "plateau", "one_cycle"])
@pytest.mark.parametrize("kind", ["flat", "hierarchical"])
def test_scheduler_weighting_and_seed_contract(kind, scheduler):
    frame = synthetic()
    labels = pd.Series((np.arange(len(frame)) % 7 == 0).astype(int))
    sidecar = {"kind": kind, "dims": {"output_dim": 2, "latent_dim": 2}, "prior_columns": ["ridge_cpoe_venue_effect"], "arch": {"hidden_dims": [8], "prior_hidden_dims": [8], "dropout": [0.1]}, "hyperparameters": {"optimizer": "sgd_momentum", "scheduler": scheduler, "scheduler_params": {"t_max": 2, "patience": 0}, "batch_size": 64, "class_weighting": "balanced", "label_smoothing": 0.1, "grad_clip_norm": 0.5, "kl_weight": 0.02, "kl_warmup_epochs": 2}}
    first = fit_network(frame, labels, ["skill"], sidecar, epochs=2, seed=19)
    second = fit_network(frame, labels, ["skill"], sidecar, epochs=2, seed=19)
    assert np.allclose(first[2], second[2])
    assert np.allclose(first[2].sum(1), 1)
    assert first[5] == second[5]
    assert next(first[0].parameters()).device.type == ("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("failure", ["seed", "missing_group", "few_groups", "target"])
def test_invalid_training_contracts(failure):
    frame = synthetic()
    labels = pd.Series(np.zeros(len(frame)))
    seed = 42
    if failure == "seed":
        seed = -1
    elif failure == "missing_group":
        frame.loc[0, "game_id"] = None
    elif failure == "few_groups":
        frame["game_id"] = "one"
    else:
        labels.iloc[0] = np.nan
    with pytest.raises(ValueError):
        fit_network(frame, labels, ["skill"], {"kind": "flat", "dims": {"output_dim": 2}}, epochs=1, seed=seed)


def test_ngmm_real_helper_contract():
    frame = synthetic()
    labels = pd.Series(np.arange(len(frame)) % 3)
    sidecar = {"kind": "ngmm", "dims": {"output_dim": 3, "latent_dim": 1}, "arch": {"hidden_dims": [8], "context_hidden_dims": [4], "dropout": [0.1, 0.0]}, "prior_columns": ["ridge_cpoe_venue_effect"], "hyperparameters": {"optimizer": "adamw", "scheduler": "none", "group_batch_size": 8, "quadrature_samples": 2, "observations_per_group": 4, "weight_decay": 0.02, "grad_clip_norm": 5.0}}
    result = fit_network(frame, labels, ["skill"], sidecar, epochs=2, seed=11)
    assert result[2].shape == (len(result[3]), 3)
    assert np.allclose(result[2].sum(1), 1)
    assert len(result[5]) == 2
    assert result[6] is not None
    assert result[0].architecture.dropout == 0.1
    assert not any("Torch Modeling" in name for name in sys.modules)


@pytest.mark.parametrize("step_mode", ["epoch", "plateau", "batch"])
def test_shared_helpers_receive_source_parameters(monkeypatch, step_mode):
    frame = synthetic()
    labels = pd.Series((np.arange(len(frame)) % 7 == 0).astype(int))
    params = {"batch_size": 73, "optimizer": "adam", "learning_rate": 0.002, "weight_decay": 0.04, "scheduler": "cosine", "class_weighting": "balanced", "label_smoothing": 0.12, "grad_clip_norm": 0.7}
    sidecar = {"kind": "flat", "dims": {"output_dim": 2}, "arch": {"hidden_dims": [8], "dropout": [0.0]}, "hyperparameters": params}
    original = copy.deepcopy(sidecar)
    captured = {"batches": [], "steps": []}
    optimizer_factory = train_models.build_optimizer
    cross_entropy = torch.nn.functional.cross_entropy
    clip_grad = torch.nn.utils.clip_grad_norm_

    def optimizer(parameters, hparams):
        for key, value in params.items():
            assert hparams[key] == value
        return optimizer_factory(parameters, hparams)

    class Scheduler:
        last_epoch = 0
        total_steps = 1000

        def step(self, *args):
            captured["steps"].append(args)
            self.last_epoch += 1

    def scheduler(optimizer, hparams, max_epochs, steps_per_epoch):
        assert hparams["scheduler"] == "cosine"
        assert max_epochs == 2
        captured["steps_per_epoch"] = steps_per_epoch
        return Scheduler(), step_mode

    def loss(logits, targets, **kwargs):
        assert kwargs["label_smoothing"] == 0.12
        assert kwargs["weight"].device == logits.device
        assert not torch.allclose(kwargs["weight"], torch.ones_like(kwargs["weight"]))
        if torch.is_grad_enabled():
            captured["batches"].append(len(targets))
        return cross_entropy(logits, targets, **kwargs)

    def clip(parameters, max_norm):
        assert max_norm == 0.7
        captured["clips"] = captured.get("clips", 0) + 1
        return clip_grad(parameters, max_norm)

    monkeypatch.setattr(train_models, "build_optimizer", optimizer)
    monkeypatch.setattr(train_models, "build_scheduler", scheduler)
    monkeypatch.setattr(torch.nn.functional, "cross_entropy", loss)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    fit_network(frame, labels, ["skill"], sidecar, epochs=2)
    assert max(captured["batches"]) == 73
    assert len(captured["batches"]) == 2 * captured["steps_per_epoch"]
    assert captured["clips"] == len(captured["batches"])
    assert len(captured["steps"]) == (len(captured["batches"]) if step_mode == "batch" else 2)
    assert all(len(args) == (1 if step_mode == "plateau" else 0) for args in captured["steps"])
    assert sidecar == original


def test_ngmm_forwards_only_explicit_supported_arguments(monkeypatch):
    import nflsim.engine.ngmm_training as ngmm_training
    signature = inspect.signature(ngmm_training.train_ngmm_grouped)
    frame = synthetic()
    params = {"group_batch_size": 7, "batch_size": 99, "quadrature_samples": 4, "observations_per_group": 8, "learning_rate": 0.003, "weight_decay": 0.02, "patience": 3}
    sidecar = {"kind": "ngmm", "dims": {"output_dim": 2, "latent_dim": 1}, "arch": {"hidden_dims": [8], "context_hidden_dims": [4], "dropout": [0.2]}, "prior_columns": ["ridge_cpoe_venue_effect"], "hyperparameters": params}

    def helper(**kwargs):
        signature.bind(**kwargs)
        assert set(kwargs) <= set(signature.parameters)
        assert kwargs["batch_size"] == 7
        assert kwargs["seed"] == 31
        assert kwargs["dropout"] == 0.2
        for key in ("quadrature_samples", "observations_per_group", "learning_rate", "weight_decay", "patience"):
            assert kwargs[key] == params[key]
        assert set(kwargs["train_frame"].game_team_id).isdisjoint(kwargs["test_frame"].game_team_id)
        return {}

    monkeypatch.setattr(ngmm_training, "train_ngmm_grouped", helper)
    with pytest.raises(ValueError, match="NGMM result missing keys"):
        fit_network(frame, pd.Series(np.arange(len(frame)) % 2), ["skill"], sidecar, epochs=1, seed=31)


@pytest.mark.parametrize("name", ["ay", "sack"])
def test_actual_snapshot_sidecar_compatibility(name):
    manifest = json.loads((Path(train_models.__file__).parents[1] / "config/model_manifest.json").read_text())
    sidecar = copy.deepcopy(manifest["models"][name]["sidecar"])
    original = copy.deepcopy(sidecar)
    frame = synthetic()
    classes = sidecar["dims"]["output_dim"]
    result = fit_network(frame, pd.Series(np.arange(len(frame)) % classes), ["skill"], sidecar, epochs=1)
    assert result[2].shape == (len(result[3]), classes)
    assert np.allclose(result[2].sum(1), 1)
    assert result[0].arch == sidecar["arch"]
    assert sidecar == original


@pytest.mark.parametrize("key,value", [("optimizer", "adam"), ("scheduler", "cosine"), ("grad_clip_norm", 1.0), ("label_smoothing", 0.1), ("class_weighting", "balanced")])
def test_ngmm_rejects_parameters_the_shared_helper_ignores(key, value):
    sidecar = {"kind": "ngmm", "dims": {"output_dim": 2, "latent_dim": 1}, "arch": {}, "prior_columns": ["skill"], "hyperparameters": {key: value}}
    with pytest.raises(ValueError, match=f"does not support {key}"):
        fit_network(synthetic(), pd.Series(np.arange(600) % 2), ["skill"], sidecar, epochs=1)


def test_train_one_preserves_source_parameters_and_effective_dimensions(tmp_path, monkeypatch):
    frame = synthetic()
    sidecar = {"kind": "hierarchical", "dims": {"output_dim": 2, "latent_dim": 1, "target_mode": "ordinal"}, "arch": {"hidden_dims": [8], "prior_hidden_dims": [4], "dropout": [0.0]}, "hyperparameters": {"latent_dim": 3, "kl_weight": 0.0, "optimizer": "adam", "scheduler": "cosine", "scheduler_params": {"t_max": 2}, "batch_size": 64, "grad_clip_norm": 0.5, "label_smoothing": 0.0, "class_weighting": "none"}}
    spec = {"sidecar": sidecar, "prefix": "example", "columns": ["skill"], "context_columns": [], "prior_columns": ["ridge_cpoe_venue_effect"], "groups": ["test"]}
    original = copy.deepcopy(spec)
    saved = {}
    monkeypatch.setattr(train_models, "output_dir", lambda path: tmp_path)
    monkeypatch.setattr(train_models, "write_json", lambda path, value: saved.update(copy.deepcopy(value)))
    monkeypatch.setattr(train_models.ModelReportBuilder, "write_html", lambda *args, **kwargs: None)
    result = train_one("example", frame, pd.Series(np.arange(len(frame)) % 2), spec, tmp_path, 1, 42)
    assert saved["hyperparameters"] == original["sidecar"]["hyperparameters"]
    assert saved["arch"] == original["sidecar"]["arch"]
    assert saved["dims"]["latent_dim"] == 3
    assert saved["dims"]["ordinal_output"] is True
    assert saved["seed"] == 42
    assert saved["seasons"] == [2024]
    assert result["epochs"] == 1
    assert spec == original
    reloaded = train_models.build_network(saved["kind"], saved["dims"], saved["arch"])
    reloaded.load_state_dict(torch.load(tmp_path / "example_model.pt", map_location="cpu", weights_only=True))


def test_maximum_numpy_seed_is_valid():
    frame = synthetic()
    sidecar = {"kind": "flat", "dims": {"output_dim": 2}, "arch": {"hidden_dims": [8], "dropout": [0.0]}}
    result = fit_network(frame, pd.Series(np.arange(len(frame)) % 2), ["skill"], sidecar, epochs=1, seed=2**32-1)
    assert np.isfinite(result[2]).all()