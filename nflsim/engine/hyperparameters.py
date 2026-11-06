"""Optuna-backed hyperparameter search and best-parameter storage for Torch Modeling.

`--mode tune` runs a study per model and records the winning configuration here; `--mode
train` reloads it. Studies use SQLite storage so they can be resumed across invocations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from nflsim.engine.model_architecture import normalize_arch

TUNING_DIR = Path(__file__).resolve().parents[1] / "outputs" / "hyperparameter_tuning"
SEARCH_SPACE_VERSION = 3
NGMM_SEARCH_SPACE_VERSION = 1
HIDDEN_DIM_CHOICES = [512, 256, 128, 64, 32]
LATENT_DIM_CHOICES = [1, 2]
BATCH_SIZE_CHOICES = [256, 512, 1024, 2048]
GRAD_CLIP_CHOICES = [0.0, 0.5, 1.0, 5.0]
RARE_POSITIVE_MODELS = {"interception", "fumble", "penalty_no_play", "penalty_first_down"}


def tuning_dir() -> Path:
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    return TUNING_DIR


def best_params_path(model_name: str) -> Path:
    return tuning_dir() / f"{model_name}_best_params.json"


def trials_csv_path(model_name: str) -> Path:
    return tuning_dir() / f"{model_name}_trials.csv"


def study_storage_uri(model_name: str) -> str:
    return f"sqlite:///{(tuning_dir() / f'{model_name}_study.db').as_posix()}"


def leaderboard_path() -> Path:
    return tuning_dir() / "tuning_leaderboard.csv"


# --------------------------------------------------------------------------------------
# Defaults / search space
# --------------------------------------------------------------------------------------


def default_hyperparameters(is_hierarchical: bool) -> Dict[str, Any]:
    """Hyperparameters reproducing the pre-tuning behavior of each training path."""
    params: Dict[str, Any] = {
        "arch": normalize_arch(None, "hierarchical" if is_hierarchical else "flat"),
        "learning_rate": 1e-4 if is_hierarchical else 1e-3,
        "weight_decay": 0.01,
        "batch_size": 512,
        "optimizer": "adamw",
        "scheduler": "none",
        "scheduler_params": {},
        "grad_clip_norm": 0.0,
        "label_smoothing": 0.0,
        "class_weighting": "none",
    }
    if is_hierarchical:
        params.update(
            {
                "latent_dim": None,  # falls back to model_feature_config.json
                "kl_weight": None,
                "kl_warmup_epochs": 20,
                "max_groups_per_batch": 4,
                "max_group_fraction": 0.7,
            }
        )
    return params


def default_ngmm_hyperparameters(target_mode: str = "multiclass") -> Dict[str, Any]:
    """Return NGMM defaults without legacy VAE/KL parameters."""
    return {
        "ngmm_search_space_version": NGMM_SEARCH_SPACE_VERSION,
        "arch": {
            "hidden_dims": [128, 64, 32],
            "context_hidden_dims": [64, 32],
            "dropout": 0.1,
        },
        "target_mode": str(target_mode),
        "latent_dim": 1,
        "quadrature_samples": 32,
        "group_batch_size": 32,
        "observations_per_group": 256,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "optimizer": "adam",
        "scheduler": "none",
        "scheduler_params": {},
        "grad_clip_norm": 5.0,
    }


def suggest_ngmm_hyperparameters(trial, target_mode: str = "multiclass") -> Dict[str, Any]:
    """Suggest NGMM parameters; deliberately excludes hierarchical VAE knobs."""
    n_layers = trial.suggest_int("ngmm_n_layers", 1, 3)
    hidden_dims = []
    ceiling = 256
    for index in range(n_layers):
        width = trial.suggest_categorical(f"ngmm_hidden_dim_l{index + 1}", [32, 64, 128, 256])
        width = min(int(width), ceiling)
        hidden_dims.append(width)
        ceiling = width
    context_layers = trial.suggest_int("ngmm_context_n_layers", 1, 2)
    context_hidden_dims = [trial.suggest_categorical(f"ngmm_context_hidden_dim_l{index + 1}", [16, 32, 64]) for index in range(context_layers)]
    params = default_ngmm_hyperparameters(target_mode)
    params.update({
        "arch": {
            "hidden_dims": hidden_dims,
            "context_hidden_dims": context_hidden_dims,
            "dropout": trial.suggest_float("ngmm_dropout", 0.0, 0.4, step=0.05),
        },
        "latent_dim": trial.suggest_categorical("ngmm_latent_dim", [1, 2, 4, 8]),
        "quadrature_samples": trial.suggest_categorical("ngmm_quadrature_samples", [8, 16, 32, 64]),
        "group_batch_size": trial.suggest_categorical("ngmm_group_batch_size", [8, 16, 32, 64]),
        "observations_per_group": trial.suggest_categorical("ngmm_observations_per_group", [128, 256, 512]),
        "learning_rate": trial.suggest_float("ngmm_learning_rate", 1e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("ngmm_weight_decay", 1e-7, 1e-2, log=True),
        "grad_clip_norm": trial.suggest_categorical("ngmm_grad_clip_norm", [1.0, 5.0, 10.0]),
    })
    return params


def suggest_hyperparameters(trial, model_name: str, target_mode: str, is_hierarchical: bool) -> Dict[str, Any]:
    kind = "hierarchical" if is_hierarchical else "flat"

    if is_hierarchical:
        hidden_choices = [256, 128, 64, 32]
        learning_rate_range = (1e-5, 3e-4)
        optimizer_choices = ["adamw", "adam"]
        scheduler_choices = ["none", "cosine", "plateau"]
        dropout_high = 0.3
    else:
        hidden_choices = HIDDEN_DIM_CHOICES
        learning_rate_range = (1e-4, 3e-3)
        optimizer_choices = ["adamw", "adam"]
        scheduler_choices = ["none", "cosine", "plateau"]
        dropout_high = 0.5

    n_layers = trial.suggest_int("n_layers", 1, 4)
    # Widths are drawn from a fixed choice list (so the search distribution stays constant
    # across trials) and then forced non-increasing to keep a funnel shape.
    raw_widths = [
        trial.suggest_categorical(f"hidden_dim_l{index}", hidden_choices)
        for index in range(1, n_layers + 1)
    ]
    hidden_dims: List[int] = []
    ceiling = max(HIDDEN_DIM_CHOICES)
    for width in raw_widths:
        width = min(int(width), ceiling)
        hidden_dims.append(width)
        ceiling = width

    dropout = trial.suggest_float("dropout", 0.0, dropout_high, step=0.05)
    taper = trial.suggest_categorical("dropout_taper", [False, True])
    dropout_per_layer = [dropout / (2 ** index) if taper else dropout for index in range(n_layers)]

    arch = normalize_arch(
        {
            "hidden_dims": hidden_dims,
            "dropout": dropout_per_layer,
            "activation": trial.suggest_categorical("activation", ["leaky_relu", "relu", "gelu", "silu"]),
            "norm": trial.suggest_categorical("norm", ["none", "batchnorm", "layernorm"]),
        },
        kind,
    )

    scheduler = trial.suggest_categorical("scheduler", scheduler_choices)
    scheduler_params: Dict[str, Any] = {}
    if scheduler == "cosine":
        scheduler_params["t_max"] = trial.suggest_int("cosine_t_max", 10, 100, step=10)
    elif scheduler == "plateau":
        scheduler_params["factor"] = trial.suggest_float("plateau_factor", 0.1, 0.7, step=0.1)
        scheduler_params["patience"] = trial.suggest_int("plateau_patience", 3, 12)
    elif scheduler == "one_cycle":
        scheduler_params["max_lr_multiplier"] = trial.suggest_float("onecycle_max_lr_multiplier", 2.0, 20.0, log=True)

    params: Dict[str, Any] = {
        "arch": arch,
        "learning_rate": trial.suggest_float(
            "learning_rate", learning_rate_range[0], learning_rate_range[1], log=True
        ),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES),
        "optimizer": trial.suggest_categorical("optimizer", optimizer_choices),
        "scheduler": scheduler,
        "scheduler_params": scheduler_params,
        "grad_clip_norm": trial.suggest_categorical("grad_clip_norm", GRAD_CLIP_CHOICES),
        "label_smoothing": 0.0,
        "class_weighting": "none",
    }

    if target_mode == "multiclass":
        params["label_smoothing"] = trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.02)
    if target_mode == "binary":
        params["class_weighting"] = trial.suggest_categorical("class_weighting", ["none", "balanced"])

    if is_hierarchical:
        params.update(
            {
                "latent_dim": trial.suggest_categorical("latent_dim", LATENT_DIM_CHOICES),
                "kl_weight": trial.suggest_float("kl_weight", 1e-4, 1.0, log=True),
                "kl_warmup_epochs": trial.suggest_categorical("kl_warmup_epochs", [5, 10, 20, 40]),
                "max_groups_per_batch": trial.suggest_categorical("max_groups_per_batch", [1, 2, 4, 8]),
                "max_group_fraction": trial.suggest_categorical("max_group_fraction", [0.3, 0.5, 0.7, 1.0]),
            }
        )
        params["arch"]["prior_hidden_dims"] = [
            trial.suggest_categorical("prior_hidden_dim", [32, 64, 128])
        ] * trial.suggest_int("prior_n_layers", 1, 3)
        params["arch"] = normalize_arch(params["arch"], kind)

    return params


# --------------------------------------------------------------------------------------
# Optimizer / scheduler factories (shared by the skorch and manual training paths)
# --------------------------------------------------------------------------------------


def build_optimizer(parameters, hparams: Dict[str, Any]) -> torch.optim.Optimizer:
    name = str(hparams.get("optimizer", "adamw")).lower()
    lr = float(hparams.get("learning_rate", 1e-3))
    weight_decay = float(hparams.get("weight_decay", 0.01))
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {name}")


def optimizer_class(hparams: Dict[str, Any]):
    name = str(hparams.get("optimizer", "adamw")).lower()
    return {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
        "sgd_momentum": torch.optim.SGD,
    }[name]


def optimizer_extra_kwargs(hparams: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = {"weight_decay": float(hparams.get("weight_decay", 0.01))}
    if str(hparams.get("optimizer", "adamw")).lower() == "sgd_momentum":
        kwargs.update({"momentum": 0.9, "nesterov": True})
    return kwargs


def build_scheduler(optimizer, hparams: Dict[str, Any], max_epochs: int, steps_per_epoch: int = 1):
    """Return (scheduler, step_mode) where step_mode is 'epoch', 'batch', 'plateau' or None."""
    name = str(hparams.get("scheduler", "none")).lower()
    config = hparams.get("scheduler_params") or {}
    if name == "none":
        return None, None
    if name == "cosine":
        t_max = int(config.get("t_max", max_epochs) or max_epochs)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, t_max)), "epoch"
    if name == "plateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(config.get("factor", 0.5)),
                patience=int(config.get("patience", 5)),
            ),
            "plateau",
        )
    if name == "one_cycle":
        max_lr = float(hparams.get("learning_rate", 1e-3)) * float(config.get("max_lr_multiplier", 10.0))
        # PackedGroupBatchSampler.__len__ is an estimate that can undercount, so allow
        # headroom; the training loop stops stepping once total_steps is exhausted.
        padded_steps = int(math.ceil(max(1, int(steps_per_epoch)) * 1.25)) + 1
        return (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                epochs=max(1, int(max_epochs)),
                steps_per_epoch=padded_steps,
            ),
            "batch",
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def balanced_class_weights(y_train, num_classes: int) -> Optional[torch.Tensor]:
    labels = np.asarray(y_train).astype(np.int64).ravel()
    if labels.size == 0:
        return None
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = labels.size / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------------------
# Evaluation metrics recorded alongside the tuning objective
# --------------------------------------------------------------------------------------


def _safe_probs(probs: np.ndarray) -> np.ndarray:
    values = np.asarray(probs, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    totals = values.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    return values / totals


def ordinal_crps(y_true_class, probs: np.ndarray, bins: Optional[np.ndarray]) -> float:
    """Discrete CRPS over ordered bin values, integrated with the bin spacing."""
    probs = _safe_probs(probs)
    truth = np.asarray(y_true_class, dtype=np.int64)
    n_classes = probs.shape[1]
    if bins is None:
        bins = np.arange(n_classes, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.float64)[:n_classes]

    predicted_cdf = np.cumsum(probs, axis=1)
    indicator = (np.arange(n_classes)[None, :] >= truth[:, None]).astype(np.float64)
    widths = np.diff(bins, append=bins[-1] + (bins[-1] - bins[-2] if n_classes > 1 else 1.0))
    return float(np.mean(np.sum((predicted_cdf - indicator) ** 2 * widths[None, :], axis=1)))


def evaluation_metrics(y_true_class, probs: np.ndarray, bins=None) -> Dict[str, float]:
    probs = _safe_probs(probs)
    truth = np.asarray(y_true_class, dtype=np.int64)
    if truth.size == 0:
        return {}

    row_index = np.arange(truth.size)
    picked = np.clip(truth, 0, probs.shape[1] - 1)
    log_loss = float(-np.mean(np.log(np.clip(probs[row_index, picked], 1e-12, 1.0))))
    accuracy = float(np.mean(np.argmax(probs, axis=1) == truth))

    metrics = {"log_loss": log_loss, "accuracy": accuracy}
    if probs.shape[1] == 2 and np.unique(truth).size > 1:
        metrics["auc"] = float(roc_auc_score(truth, probs[:, 1]))
    if bins is not None:
        metrics["crps"] = ordinal_crps(truth, probs, np.asarray(bins, dtype=np.float64))
    return metrics


# --------------------------------------------------------------------------------------
# Study orchestration and best-parameter persistence
# --------------------------------------------------------------------------------------


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def save_best_hyperparameters(model_name: str, hparams: Dict[str, Any], record: Dict[str, Any]) -> Path:
    payload = {
        "model_name": model_name,
        "search_space_version": SEARCH_SPACE_VERSION,
        "hyperparameters": _json_safe(hparams),
        **_json_safe(record),
    }
    path = best_params_path(model_name)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_best_hyperparameters(model_name: str, is_hierarchical: bool) -> Dict[str, Any]:
    """Return tuned hyperparameters if a study has been run, else the legacy defaults."""
    path = best_params_path(model_name)
    if not path.exists():
        print(f"[tuning] {model_name}: no tuned hyperparameters at {path.name}; using defaults")
        return default_hyperparameters(is_hierarchical)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        print(f"[tuning] {model_name}: unreadable {path.name} ({error}); using defaults")
        return default_hyperparameters(is_hierarchical)

    if payload.get("search_space_version") != SEARCH_SPACE_VERSION:
        print(
            f"[tuning] {model_name}: {path.name} uses an older search space; using stable defaults"
        )
        return default_hyperparameters(is_hierarchical)

    hparams = payload.get("hyperparameters")
    if not isinstance(hparams, dict):
        print(f"[tuning] {model_name}: malformed {path.name}; using defaults")
        return default_hyperparameters(is_hierarchical)

    merged = default_hyperparameters(is_hierarchical)
    merged.update(hparams)
    merged["arch"] = normalize_arch(
        hparams.get("arch"), "hierarchical" if is_hierarchical else "flat"
    )
    print(
        f"[tuning] {model_name}: using tuned hyperparameters from {path.name} "
        f"(val_loss={payload.get('val_loss')}, trial={payload.get('trial_number')})"
    )
    return merged


def load_best_ngmm_hyperparameters(model_name: str, target_mode: str) -> Dict[str, Any]:
    """Load only versioned NGMM parameters; legacy hierarchical files are rejected."""
    defaults = default_ngmm_hyperparameters(target_mode)
    path = best_params_path(model_name)
    if not path.exists():
        return defaults
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return defaults
    hparams = payload.get("hyperparameters")
    if not isinstance(hparams, dict) or hparams.get("ngmm_search_space_version") != NGMM_SEARCH_SPACE_VERSION:
        print(f"[tuning] {model_name}: no compatible NGMM parameters in {path.name}; using NGMM defaults")
        return defaults
    merged = dict(defaults)
    merged.update(hparams)
    if isinstance(hparams.get("arch"), dict):
        merged["arch"] = {**defaults["arch"], **hparams["arch"]}
    return merged


def run_study(
    model_name: str,
    objective,
    n_trials: int,
    timeout: Optional[float] = None,
    seed: int = 42,
    fresh: bool = False,
):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = study_storage_uri(model_name)
    study_name = f"{model_name}_arch_search_v{SEARCH_SPACE_VERSION}"

    if fresh:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f"[tuning] {model_name}: deleted existing study for a fresh search")
        except KeyError:
            pass

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)
    _write_trials_csv(model_name, study)
    return study


def _write_trials_csv(model_name: str, study) -> None:
    try:
        frame = study.trials_dataframe()
    except Exception:
        return
    if frame is None or frame.empty:
        return
    frame.to_csv(trials_csv_path(model_name), index=False)


def update_leaderboard(rows: Sequence[Dict[str, Any]]) -> Optional[Path]:
    if not rows:
        return None
    import pandas as pd

    path = leaderboard_path()
    frame = pd.DataFrame([_json_safe(row) for row in rows])
    if path.exists():
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True, sort=False)
        frame = frame.drop_duplicates(subset=["model_name"], keep="last")
    frame.to_csv(path, index=False)
    return path


def describe_hyperparameters(hparams: Dict[str, Any]) -> str:
    arch = hparams.get("arch", {})
    return (
        f"hidden={arch.get('hidden_dims')} dropout={arch.get('dropout')} "
        f"act={arch.get('activation')} norm={arch.get('norm')} "
        f"opt={hparams.get('optimizer')} lr={hparams.get('learning_rate'):.2e} "
        f"wd={hparams.get('weight_decay'):.2e} bs={hparams.get('batch_size')} "
        f"sched={hparams.get('scheduler')} clip={hparams.get('grad_clip_norm')}"
    )
