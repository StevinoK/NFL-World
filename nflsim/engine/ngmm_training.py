"""Grouped training utilities for the shared NGMM model."""

from __future__ import annotations

import copy
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.nn import functional as F

from nflsim.engine.ngmm_model import NGMMArchitecture, NGMMModel, TargetMode


def _preprocessor(frame: pd.DataFrame, columns: list[str]) -> ColumnTransformer:
    categorical = [column for column in columns if not pd.api.types.is_numeric_dtype(frame[column])]
    numerical = [column for column in columns if column not in categorical]
    transformers = []
    if numerical:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numerical))
    if categorical:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    if not transformers:
        raise ValueError("NGMM requires at least one feature column")
    return ColumnTransformer(transformers)


def _group_context_frame(frame: pd.DataFrame, groups: pd.Series, columns: list[str]) -> pd.DataFrame:
    context = frame[columns].copy()
    for column in columns:
        context[column] = pd.to_numeric(context[column], errors="coerce")
    context["__group__"] = groups.astype(str).to_numpy()
    return context.groupby("__group__", sort=True)[columns].median().fillna(0.0)


def _target_values(
    target: pd.Series,
    target_mode: TargetMode,
    output_dim: int,
    categories: list | np.ndarray | None = None,
) -> tuple[np.ndarray, int, np.ndarray | None]:
    if target_mode == "gaussian":
        return target.to_numpy(dtype=np.float32), 1, None
    values = target.to_numpy()
    if target_mode == "binary":
        return values.astype(np.int64), 2, None
    if target_mode in {"multiclass", "ordinal"}:
        # Callers pre-encode targets as a fixed 0..K-1 index (see prepare_class_targets
        # in Torch Modeling.py). Categories MUST be shared across the fit/validation/test
        # splits passed in by train_ngmm_grouped: deriving them independently per split
        # (the previous behavior) silently remaps class codes whenever a split is missing
        # a rare class (e.g. an infrequent air_yards ordinal bin), scrambling which output
        # column corresponds to which bin and producing a systematically miscalibrated
        # predicted distribution.
        if categories is None:
            categories = list(range(int(output_dim))) if output_dim else sorted(pd.unique(values))
        encoded = pd.Categorical(values, categories=categories).codes.astype(np.int64)
        return encoded, len(categories), np.asarray(categories, dtype=object)
    raise ValueError(f"Unsupported NGMM target mode: {target_mode}")


def train_ngmm_grouped(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    y_test: pd.Series | np.ndarray,
    train_groups: Iterable,
    play_columns: list[str],
    prior_columns: list[str],
    categorical_cols: list[str] | None = None,
    target_mode: TargetMode = "multiclass",
    output_dim: int = 2,
    latent_dim: int = 1,
    max_epochs: int = 100,
    batch_size: int = 32,
    observations_per_group: int = 256,
    quadrature_samples: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    dropout: float = 0.1,
    hidden_dims: tuple[int, ...] = (128, 64, 32),
    context_hidden_dims: tuple[int, ...] = (64, 32),
    validation_size: float = 0.2,
    patience: int = 12,
    seed: int = 42,
    log_progress: bool = True,
    **_: Any,
) -> dict[str, Any]:
    """Fit NGMM using uniform mini-batches of groups.

    The caller supplies a group-disjoint outer test frame. A row split within
    training groups supplies validation data, while group context is fitted
    only from the training portion.
    """
    if not prior_columns:
        raise ValueError("NGMM requires prior/group-context columns")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_frame = train_frame.reset_index(drop=True).copy()
    test_frame = test_frame.reset_index(drop=True).copy()
    groups = pd.Series(np.asarray(list(train_groups)), dtype=str)
    if len(groups) != len(train_frame):
        raise ValueError("train_groups must align with train_frame")
    y_train_series = pd.Series(np.asarray(y_train)).reset_index(drop=True)
    y_test_series = pd.Series(np.asarray(y_test)).reset_index(drop=True)

    validation_mask = np.zeros(len(train_frame), dtype=bool)
    for group in groups.unique():
        indices = np.flatnonzero(groups.to_numpy() == group)
        count = max(1, int(round(len(indices) * validation_size)))
        validation_mask[rng.choice(indices, size=min(count, len(indices)), replace=False)] = True
    fit_frame = train_frame.loc[~validation_mask].reset_index(drop=True)
    validation_frame = train_frame.loc[validation_mask].reset_index(drop=True)
    fit_y = y_train_series.loc[~validation_mask].reset_index(drop=True)
    validation_y = y_train_series.loc[validation_mask].reset_index(drop=True)
    fit_groups = groups.loc[~validation_mask].reset_index(drop=True)
    validation_groups = groups.loc[validation_mask].reset_index(drop=True)

    play_processor = _preprocessor(fit_frame, play_columns)
    X_fit = play_processor.fit_transform(fit_frame[play_columns]).astype(np.float32)
    X_validation = play_processor.transform(validation_frame[play_columns]).astype(np.float32)
    X_test = play_processor.transform(test_frame[play_columns]).astype(np.float32)

    fit_context = _group_context_frame(fit_frame, fit_groups, prior_columns)
    context_processor = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    context_processor.fit(fit_context[prior_columns])
    fit_context_values = context_processor.transform(fit_context[prior_columns]).astype(np.float32)
    validation_context = _group_context_frame(validation_frame, validation_groups, prior_columns)
    test_groups = pd.Series(np.arange(len(test_frame)), dtype=str)
    test_context = _group_context_frame(test_frame, test_groups, prior_columns)
    validation_context_values = context_processor.transform(validation_context.reindex(columns=prior_columns).fillna(0.0)).astype(np.float32)
    test_context_values = context_processor.transform(test_context.reindex(columns=prior_columns).fillna(0.0)).astype(np.float32)

    if target_mode in {"multiclass", "ordinal"} and not output_dim:
        # No fixed output_dim supplied: derive categories from every split up front so
        # fit/validation/test still share one mapping instead of each guessing its own.
        all_values = pd.concat([fit_y, validation_y, y_test_series]).to_numpy()
        shared_categories = sorted(pd.unique(all_values))
    else:
        shared_categories = None
    fit_targets, resolved_output_dim, class_values = _target_values(fit_y, target_mode, output_dim, categories=shared_categories)
    validation_targets, _, _ = _target_values(validation_y, target_mode, resolved_output_dim, categories=class_values)
    test_targets, _, _ = _target_values(y_test_series, target_mode, resolved_output_dim, categories=class_values)
    architecture = NGMMArchitecture(
        hidden_dims=tuple(hidden_dims),
        context_hidden_dims=tuple(context_hidden_dims),
        latent_dim=int(latent_dim),
        dropout=float(dropout),
        target_mode=target_mode,
        output_dim=int(resolved_output_dim),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NGMMModel(X_fit.shape[1], fit_context_values.shape[1], architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    fit_group_names = sorted(fit_groups.unique())
    fit_group_rows = {group: np.flatnonzero(fit_groups.to_numpy() == group) for group in fit_group_names}
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    loss_history: list[dict[str, float]] = []
    started = perf_counter()

    for epoch in range(max_epochs):
        model.train()
        group_order = list(fit_group_names)
        rng.shuffle(group_order)
        train_losses = []
        for start in range(0, len(group_order), max(1, int(batch_size))):
            optimizer.zero_grad(set_to_none=True)
            batch_groups = group_order[start : start + max(1, int(batch_size))]
            batch_loss = torch.zeros((), device=device)
            for group in batch_groups:
                rows = fit_group_rows[group][:observations_per_group]
                context_index = fit_group_names.index(group)
                group_context = torch.from_numpy(fit_context_values[context_index:context_index + 1]).to(device)
                gamma = torch.randn(quadrature_samples, latent_dim, device=device)
                batch_loss = batch_loss - model.group_log_marginal(
                    torch.from_numpy(X_fit[rows]).to(device), group_context,
                    torch.from_numpy(fit_targets[rows]).to(device), gamma,
                )
            batch_loss = batch_loss / max(len(batch_groups), 1)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(batch_loss.detach().cpu()))

        model.eval()
        validation_losses = []
        with torch.no_grad():
            validation_group_names = sorted(validation_groups.unique())
            for group in validation_group_names:
                rows = np.flatnonzero(validation_groups.to_numpy() == group)[:observations_per_group]
                group_context = torch.from_numpy(validation_context_values[validation_group_names.index(group):validation_group_names.index(group) + 1]).to(device)
                gamma = torch.randn(quadrature_samples, latent_dim, device=device)
                validation_losses.append(float(-model.group_log_marginal(torch.from_numpy(X_validation[rows]).to(device), group_context, torch.from_numpy(validation_targets[rows]).to(device), gamma).cpu()))
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        validation_loss = float(np.mean(validation_losses)) if validation_losses else train_loss
        loss_history.append({"train_loss": train_loss, "valid_loss": validation_loss})
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if log_progress and (epoch == 0 or (epoch + 1) % 10 == 0):
            print(f"[ngmm-train] epoch={epoch + 1}/{max_epochs} train_loss={train_loss:.5f} val_loss={validation_loss:.5f}")
        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(X_test).to(device)
        fixed_test = model.fixed_prediction(test_tensor).cpu().numpy()
    if target_mode == "gaussian":
        probabilities = fixed_test
    elif target_mode == "binary":
        probabilities = np.concatenate([1.0 - fixed_test, fixed_test], axis=1)
    else:
        probabilities = fixed_test
    if log_progress:
        print(f"[ngmm-train] finished in {perf_counter() - started:.1f}s best_val={best_validation_loss:.5f}")
    return {
        "model": model,
        "torch_model": model,
        "network_kind": "ngmm",
        "play_preprocessor": play_processor,
        "prior_preprocessor": context_processor,
        "prior_columns": prior_columns,
        "play_columns": play_columns,
        "X_test_play": X_test,
        "X_test_prior": test_context_values,
        "test_probabilities": probabilities,
        "y_test_class": test_targets,
        "bins": None,
        "loss_history": {"train_loss": [row["train_loss"] for row in loss_history], "valid_loss": [row["valid_loss"] for row in loss_history]},
        "device": device,
        "target_mode": target_mode,
        "latent_dim": int(latent_dim),
        "val_loss": float(best_validation_loss),
        "best_epoch": int(np.argmin([row["valid_loss"] for row in loss_history]) + 1) if loss_history else 0,
        "class_values": class_values.tolist() if class_values is not None else None,
    }
