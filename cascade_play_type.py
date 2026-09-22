"""Cascaded residual play-type model.

Three-stage pipeline:
  1. Base MLP classifier trained on 45% of game::team groups.
  2. Context-conditioned heatmap regression on game-averaged, PCA-reduced
     residuals using KL divergence against a discretized, label-smoothed grid.
  3. Final MLP classifier on the remaining 45% of groups, augmented with the
     center-of-mass of the stage-2 residual distribution as an extra feature.

Validation (10% of groups) is held out from stage-1 training and used for
early stopping / evaluation across all three stages.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from armed_play_type import DEFAULT_CLASSES, normal_play_type_features

DEFAULT_DATA_PATH = Path(__file__).resolve().with_name("play_type_training.csv")
DEFAULT_GROUP_CONTEXT_COLUMNS = (
    "pass_oe",
    "pass_oe_weekly_variation",
    "target_gini_roll4",
    "rb_carry_split_gini_roll4",
    "rb_top_carry_share_roll4",
    "pace_of_play_bayes",
    "days_rest",
    "opp_days_rest",
)

# ── configuration ────────────────────────────────────────────────────────────


@dataclass
class CascadeConfig:
    # data splitting
    stage1_frac: float = 0.45
    stage3_frac: float = 0.45
    # stage-1 / stage-3 shared architecture
    hidden_dims: tuple[int, ...] = (64, 32, 16)
    dropout: float = 0.1
    label_smoothing: float = 0.05
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 100
    batch_size: int = 512
    patience: int = 10
    # stage-2 heatmap regression
    n_bins: int = 32
    grid_smoothing_sigma: float = 1.5  # in bin-widths
    grid_label_smoothing: float = 0.05
    stage2_hidden_dims: tuple[int, ...] = (32, 64, 32)
    stage2_dropout: float = 0.1
    stage2_learning_rate: float = 1e-3
    stage2_epochs: int = 150
    stage2_patience: int = 15
    seed: int = 42


# ── models ───────────────────────────────────────────────────────────────────


class PlayTypeMLP(nn.Module):
    """Feed-forward classifier for 4-class play-type prediction."""

    def __init__(self, input_dim: int, n_classes: int, config: CascadeConfig):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in config.hidden_dims:
            layers.extend([nn.Linear(prev, width), nn.ReLU(), nn.Dropout(config.dropout)])
            prev = width
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualGridNet(nn.Module):
    """Predicts a distribution over residual grid bins from group context."""

    def __init__(self, context_dim: int, n_bins: int, config: CascadeConfig):
        super().__init__()
        layers: list[nn.Module] = []
        prev = context_dim
        for width in config.stage2_hidden_dims:
            layers.extend([nn.Linear(prev, width), nn.ReLU(), nn.Dropout(config.stage2_dropout)])
            prev = width
        layers.append(nn.Linear(prev, n_bins))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits over bins."""
        return self.net(x)

    def predict_distribution(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


# ── helpers ──────────────────────────────────────────────────────────────────


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))


def _encode_features(
    fit: pd.DataFrame,
    frames: list[pd.DataFrame],
    columns: list[str],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """One-hot encode categoricals, standardize, return fit + transformed frames."""
    fit_encoded = pd.get_dummies(fit[columns], dtype=float)
    outputs = []
    for frame in frames:
        encoded = pd.get_dummies(frame[columns], dtype=float).reindex(
            columns=fit_encoded.columns, fill_value=0.0
        )
        outputs.append(encoded.to_numpy(dtype=np.float32))
    scaler = StandardScaler().fit(fit_encoded.to_numpy(dtype=np.float32))
    return scaler.transform(fit_encoded.to_numpy(dtype=np.float32)).astype(np.float32), [
        scaler.transform(o).astype(np.float32) for o in outputs
    ]


def _encode_group_context(
    fit: pd.DataFrame,
    frames: list[pd.DataFrame],
    group_column: str,
    columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, list[np.ndarray], StandardScaler]:
    """Aggregate context per group (median), standardize on *fit* groups.

    Returns (fit_group_df, fit_array, [frame_arrays_row_aligned], scaler).
    """
    def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
        vals = frame[columns].apply(pd.to_numeric, errors="coerce")
        return vals.groupby(frame[group_column]).median().fillna(0.0)

    fit_groups = aggregate(fit)
    scaler = StandardScaler().fit(fit_groups.to_numpy(dtype=np.float32))

    def row_aligned(frame: pd.DataFrame) -> np.ndarray:
        groups = aggregate(frame)
        rows = groups.reindex(frame[group_column]).fillna(0.0)
        return scaler.transform(rows.to_numpy(dtype=np.float32)).astype(np.float32)

    fit_arr = scaler.transform(fit_groups.to_numpy(dtype=np.float32)).astype(np.float32)
    return fit_groups, fit_arr, [row_aligned(f) for f in frames], scaler


def _metrics(targets: np.ndarray, probabilities: np.ndarray, classes: tuple[str, ...]) -> dict[str, float]:
    preds = np.argmax(probabilities, axis=1)
    one_hot = np.eye(len(classes), dtype=float)[targets]
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
        "log_loss": float(log_loss(targets, probabilities, labels=list(range(len(classes))))),
        "auroc_macro_ovr": float(roc_auc_score(one_hot, probabilities, multi_class="ovr", average="macro")),
    }


def _confusion_matrix_payload(targets: np.ndarray, probabilities: np.ndarray, classes: tuple[str, ...]) -> dict:
    preds = np.argmax(probabilities, axis=1)
    mat = confusion_matrix(targets, preds, labels=list(range(len(classes))))
    return {"labels": list(classes), "actual_by_predicted": mat.astype(int).tolist()}


def _train_mlp(
    model: PlayTypeMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: CascadeConfig,
    device: str,
) -> tuple[PlayTypeMLP, list[float]]:
    """Train an MLP with early stopping on validation log-loss. Returns (model, history)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    xt = torch.from_numpy(x_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    xv = torch.from_numpy(x_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    history: list[float] = []
    no_improve = 0
    n = xt.shape[0]

    for epoch in range(config.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xt[idx])
            loss = criterion(logits, yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        history.append(avg)

        # validation
        model.eval()
        with torch.no_grad():
            val_probs = torch.softmax(model(xv), dim=1).cpu().numpy()
        val_ll = float(log_loss(y_val, val_probs, labels=[0, 1, 2, 3]))
        if epoch % 5 == 0 or no_improve == 0:
            print(f"  epoch {epoch:3d}  train_loss={avg:.4f}  val_logloss={val_ll:.4f}", flush=True)
        if val_ll < best_loss:
            best_loss = val_ll
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= config.patience:
            print(f"  Early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, history


def _build_soft_grid_targets(
    residuals: np.ndarray,
    bin_edges: np.ndarray,
    sigma_bins: float,
    label_smoothing: float,
) -> np.ndarray:
    """Map scalar residuals to smoothed distributions over bins.

    Each target is a Gaussian-like bump centred on the residual's bin location,
    blended with uniform label smoothing.
    """
    n_bins = len(bin_edges) - 1
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    targets = np.zeros((len(residuals), n_bins), dtype=np.float64)
    for i, r in enumerate(residuals):
        # Gaussian-like weights around the residual value
        dists = (bin_centers - r) / (sigma_bins * bin_width)
        weights = np.exp(-0.5 * dists**2)
        weights /= weights.sum()
        # Label smoothing: blend with uniform
        weights = (1.0 - label_smoothing) * weights + label_smoothing / n_bins
        targets[i] = weights
    return targets


# ── main pipeline ────────────────────────────────────────────────────────────


def fit_cascade(
    frame: pd.DataFrame,
    target_column: str = "play_type",
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    feature_columns: list[str] | None = None,
    group_context_columns: list[str] | None = None,
    config: CascadeConfig | None = None,
    device: str | None = None,
) -> dict[str, object]:
    config = config or CascadeConfig()
    _seed_everything(config.seed)
    feature_columns = list(feature_columns or normal_play_type_features())
    # Filter to columns actually present in the data
    feature_columns = [c for c in feature_columns if c in frame.columns]
    group_context_columns = list(group_context_columns or DEFAULT_GROUP_CONTEXT_COLUMNS)
    group_context_columns = [c for c in group_context_columns if c in frame.columns]
    if not group_context_columns:
        raise ValueError("Need at least one group-context column")

    working = frame.copy()
    if "game_id" not in working.columns or "posteam" not in working.columns:
        raise ValueError("Input must contain game_id and posteam")
    working["ngmm_group"] = working["game_id"].astype(str) + "::" + working["posteam"].fillna("UNKNOWN").astype(str)
    working[target_column] = working[target_column].astype(str)
    working = working.dropna(subset=[target_column, "ngmm_group"] + feature_columns)
    target_map = {label: idx for idx, label in enumerate(classes)}
    unknown = sorted(set(working[target_column]) - set(classes))
    if unknown:
        raise ValueError(f"Unknown target labels: {unknown}")

    # ── 1. Three-way stratified split (45 / 45 / 10) ────────────────────────
    val_frac = 1.0 - config.stage1_frac - config.stage3_frac  # 0.10
    # First split: stage1 (45%) vs remainder (55%)
    splitter1 = GroupShuffleSplit(n_splits=1, test_size=1.0 - config.stage1_frac, random_state=config.seed)
    s1_idx, rem_idx = next(splitter1.split(working, groups=working["ngmm_group"]))
    stage1_train = working.iloc[s1_idx].copy()
    remainder = working.iloc[rem_idx].copy()
    # Second split: stage3 (45/55 of remainder ≈ 81.8%) vs val (10/55 ≈ 18.2%)
    stage3_ratio_of_remainder = config.stage3_frac / (config.stage3_frac + val_frac)
    splitter2 = GroupShuffleSplit(
        n_splits=1, test_size=1.0 - stage3_ratio_of_remainder, random_state=config.seed + 1
    )
    s3_idx, val_idx = next(splitter2.split(remainder, groups=remainder["ngmm_group"]))
    stage3_train = remainder.iloc[s3_idx].copy()
    stage1_val = remainder.iloc[val_idx].copy()

    print(f"Split sizes — stage1_train: {len(stage1_train)}, "
          f"stage3_train: {len(stage3_train)}, stage1_val: {len(stage1_val)}")
    print(f"Groups — stage1_train: {stage1_train['ngmm_group'].nunique()}, "
          f"stage3_train: {stage3_train['ngmm_group'].nunique()}, "
          f"stage1_val: {stage1_val['ngmm_group'].nunique()}")

    # Verify no group overlap
    g1 = set(stage1_train["ngmm_group"].unique())
    g3 = set(stage3_train["ngmm_group"].unique())
    gv = set(stage1_val["ngmm_group"].unique())
    assert not (g1 & g3), "Group overlap between stage1_train and stage3_train"
    assert not (g1 & gv), "Group overlap between stage1_train and stage1_val"
    assert not (g3 & gv), "Group overlap between stage3_train and stage1_val"

    # ── 2. Feature encoding ──────────────────────────────────────────────────
    fit_x, other_x = _encode_features(stage1_train, [stage3_train, stage1_val], feature_columns)
    s3_x, val_x = other_x

    # Group context encoding
    fit_ctx_df, fit_ctx_arr, ctx_frames, ctx_scaler = _encode_group_context(
        stage1_train, [stage3_train, stage1_val], "ngmm_group", group_context_columns
    )
    s3_ctx_rows, val_ctx_rows = ctx_frames

    y_s1 = stage1_train[target_column].map(target_map).to_numpy(np.int64)
    y_s3 = stage3_train[target_column].map(target_map).to_numpy(np.int64)
    y_val = stage1_val[target_column].map(target_map).to_numpy(np.int64)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 3. Stage 1: base MLP classifier ──────────────────────────────────────
    print("\n── Stage 1: training base classifier ──")
    stage1_model = PlayTypeMLP(fit_x.shape[1], len(classes), config)
    stage1_model, s1_history = _train_mlp(stage1_model, fit_x, y_s1, val_x, y_val, config, device)

    with torch.no_grad():
        s1_probs_train = torch.softmax(stage1_model(torch.from_numpy(fit_x).to(device)), dim=1).cpu().numpy()
        s1_probs_val = torch.softmax(stage1_model(torch.from_numpy(val_x).to(device)), dim=1).cpu().numpy()

    s1_val_metrics = _metrics(y_val, s1_probs_val, classes)
    print(f"Stage 1 val metrics: {json.dumps(s1_val_metrics, indent=2)}")

    # ── 4. Residuals → game-averaged → PCA to 1D ────────────────────────────
    print("\n── Stage 2a: computing residuals and PCA ──")
    one_hot_s1 = np.eye(len(classes), dtype=np.float64)[y_s1]
    residuals_4d = one_hot_s1 - s1_probs_train.astype(np.float64)  # (n_plays, 4)

    # Average residuals by game::team group
    resid_df = pd.DataFrame(residuals_4d, columns=[f"resid_{c}" for c in classes])
    resid_df["ngmm_group"] = stage1_train["ngmm_group"].values
    game_residuals = resid_df.groupby("ngmm_group")[[f"resid_{c}" for c in classes]].mean()

    pca = PCA(n_components=1)
    game_resid_1d = pca.fit_transform(game_residuals.to_numpy()).ravel()  # (n_groups,)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_[0]:.4f}")
    print(f"Game-level 1D residuals — mean: {game_resid_1d.mean():.6f}, "
          f"std: {game_resid_1d.std():.6f}, "
          f"range: [{game_resid_1d.min():.4f}, {game_resid_1d.max():.4f}]")

    # ── 5. Discretize into grid with label smoothing ─────────────────────────
    margin = 0.1 * (game_resid_1d.max() - game_resid_1d.min())
    grid_min = game_resid_1d.min() - margin
    grid_max = game_resid_1d.max() + margin
    bin_edges = np.linspace(grid_min, grid_max, config.n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    soft_targets = _build_soft_grid_targets(
        game_resid_1d, bin_edges, config.grid_smoothing_sigma, config.grid_label_smoothing
    )

    # ── 6. Stage 2: context-conditioned heatmap regression ───────────────────
    print("\n── Stage 2b: training residual context model ──")
    # Align context features with the game-level residuals
    ctx_for_groups = fit_ctx_df.reindex(game_residuals.index).fillna(0.0)
    ctx_group_arr = ctx_scaler.transform(ctx_for_groups.to_numpy(dtype=np.float32)).astype(np.float32)

    # Also compute residuals on the validation set for stage-2 early stopping
    one_hot_val = np.eye(len(classes), dtype=np.float64)[y_val]
    val_resid_4d = one_hot_val - s1_probs_val.astype(np.float64)
    val_resid_df = pd.DataFrame(val_resid_4d, columns=[f"resid_{c}" for c in classes])
    val_resid_df["ngmm_group"] = stage1_val["ngmm_group"].values
    val_game_residuals = val_resid_df.groupby("ngmm_group")[[f"resid_{c}" for c in classes]].mean()
    val_game_resid_1d = pca.transform(val_game_residuals.to_numpy()).ravel()
    val_soft_targets = _build_soft_grid_targets(
        val_game_resid_1d, bin_edges, config.grid_smoothing_sigma, config.grid_label_smoothing
    )
    val_ctx_for_groups = stage1_val.groupby("ngmm_group")[group_context_columns].median().fillna(0.0)
    val_ctx_for_groups = val_ctx_for_groups.reindex(val_game_residuals.index).fillna(0.0)
    val_ctx_arr = ctx_scaler.transform(val_ctx_for_groups.to_numpy(dtype=np.float32)).astype(np.float32)

    stage2_model = ResidualGridNet(ctx_group_arr.shape[1], config.n_bins, config).to(device)
    s2_optimizer = torch.optim.Adam(stage2_model.parameters(), lr=config.stage2_learning_rate)

    x_s2 = torch.from_numpy(ctx_group_arr).to(device)
    t_s2 = torch.from_numpy(soft_targets).float().to(device)
    x_s2_val = torch.from_numpy(val_ctx_arr).to(device)
    t_s2_val = torch.from_numpy(val_soft_targets).float().to(device)

    best_s2_state = copy.deepcopy(stage2_model.state_dict())
    best_s2_loss = float("inf")
    s2_history: list[float] = []
    no_improve_s2 = 0
    n_groups = x_s2.shape[0]
    s2_batch = min(64, n_groups)

    for epoch in range(config.stage2_epochs):
        stage2_model.train()
        perm = torch.randperm(n_groups, device=device)
        ep_loss = 0.0
        nb = 0
        for start in range(0, n_groups, s2_batch):
            idx = perm[start : start + s2_batch]
            s2_optimizer.zero_grad(set_to_none=True)
            log_preds = F.log_softmax(stage2_model(x_s2[idx]), dim=-1)
            loss = F.kl_div(log_preds, t_s2[idx], reduction="batchmean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stage2_model.parameters(), 5.0)
            s2_optimizer.step()
            ep_loss += float(loss.detach())
            nb += 1
        avg_s2 = ep_loss / max(nb, 1)
        s2_history.append(avg_s2)

        # Val KL
        stage2_model.eval()
        with torch.no_grad():
            val_log_preds = F.log_softmax(stage2_model(x_s2_val), dim=-1)
            val_kl = float(F.kl_div(val_log_preds, t_s2_val, reduction="batchmean"))
        if epoch % 10 == 0 or no_improve_s2 == 0:
            print(f"  s2 epoch {epoch:3d}  train_kl={avg_s2:.6f}  val_kl={val_kl:.6f}", flush=True)
        if val_kl < best_s2_loss:
            best_s2_loss = val_kl
            best_s2_state = copy.deepcopy(stage2_model.state_dict())
            no_improve_s2 = 0
        else:
            no_improve_s2 += 1
        if no_improve_s2 >= config.stage2_patience:
            print(f"  Stage 2 early stopping at epoch {epoch}", flush=True)
            break

    stage2_model.load_state_dict(best_s2_state)
    stage2_model.eval()
    print(f"Stage 2 best val KL: {best_s2_loss:.6f} (stopped at epoch {len(s2_history)})")

    # ── Predict residual distributions for ALL groups ────────────────────────
    print("\n── Stage 2c: generating center-of-mass predictions for all groups ──")
    all_groups = pd.concat([stage1_train, stage3_train, stage1_val], ignore_index=True)
    all_ctx = all_groups.groupby("ngmm_group")[group_context_columns].median().fillna(0.0)
    all_ctx_arr = ctx_scaler.transform(all_ctx.to_numpy(dtype=np.float32)).astype(np.float32)

    with torch.no_grad():
        all_dist = stage2_model.predict_distribution(torch.from_numpy(all_ctx_arr).to(device)).cpu().numpy()
    # Center of mass
    bin_centers_f32 = bin_centers.astype(np.float64)
    com_values = (all_dist * bin_centers_f32[np.newaxis, :]).sum(axis=1)
    com_map = pd.Series(com_values, index=all_ctx.index)

    # ── 7. Augment stage-3 features with center-of-mass ──────────────────────
    stage3_train_aug = stage3_train.copy()
    stage3_train_aug["residual_com"] = stage3_train_aug["ngmm_group"].map(com_map).fillna(0.0)
    stage1_val_aug = stage1_val.copy()
    stage1_val_aug["residual_com"] = stage1_val_aug["ngmm_group"].map(com_map).fillna(0.0)

    aug_feature_columns = feature_columns + ["residual_com"]
    s3_x_aug, val_x_aug_list = _encode_features(stage3_train_aug, [stage1_val_aug], aug_feature_columns)
    val_x_aug = val_x_aug_list[0]

    # ── 8. Stage 3: final MLP classifier ─────────────────────────────────────
    print("\n── Stage 3: training final classifier with residual_com feature ──")
    stage3_model = PlayTypeMLP(s3_x_aug.shape[1], len(classes), config)
    stage3_model, s3_history = _train_mlp(stage3_model, s3_x_aug, y_s3, val_x_aug, y_val, config, device)

    with torch.no_grad():
        s3_probs_val = torch.softmax(stage3_model(torch.from_numpy(val_x_aug).to(device)), dim=1).cpu().numpy()

    s3_val_metrics = _metrics(y_val, s3_probs_val, classes)
    print(f"Stage 3 val metrics: {json.dumps(s3_val_metrics, indent=2)}")

    # ── 9. Assemble results ──────────────────────────────────────────────────
    return {
        "config": asdict(config),
        "classes": list(classes),
        "split_info": {
            "stage1_train_rows": len(stage1_train),
            "stage3_train_rows": len(stage3_train),
            "stage1_val_rows": len(stage1_val),
            "stage1_train_groups": int(stage1_train["ngmm_group"].nunique()),
            "stage3_train_groups": int(stage3_train["ngmm_group"].nunique()),
            "stage1_val_groups": int(stage1_val["ngmm_group"].nunique()),
        },
        "stage1": {
            "val_metrics": s1_val_metrics,
            "training_history": s1_history,
            "confusion_matrix": _confusion_matrix_payload(y_val, s1_probs_val, classes),
        },
        "stage2": {
            "pca_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
            "grid_range": [float(grid_min), float(grid_max)],
            "n_bins": config.n_bins,
            "best_val_kl": best_s2_loss,
            "training_history": s2_history,
            "n_game_groups": int(len(game_residuals)),
        },
        "stage3": {
            "val_metrics": s3_val_metrics,
            "training_history": s3_history,
            "confusion_matrix": _confusion_matrix_payload(y_val, s3_probs_val, classes),
        },
        "comparison": {
            "stage1_log_loss": s1_val_metrics["log_loss"],
            "stage3_log_loss": s3_val_metrics["log_loss"],
            "log_loss_improvement": s1_val_metrics["log_loss"] - s3_val_metrics["log_loss"],
            "stage1_accuracy": s1_val_metrics["accuracy"],
            "stage3_accuracy": s3_val_metrics["accuracy"],
            "accuracy_improvement": s3_val_metrics["accuracy"] - s1_val_metrics["accuracy"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the cascaded residual play-type model.")
    parser.add_argument("csv", type=Path, nargs="?", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=Path("cascade_play_type_results.json"))
    parser.add_argument("--epochs", type=int, default=CascadeConfig.epochs)
    parser.add_argument("--stage2-epochs", type=int, default=CascadeConfig.stage2_epochs)
    parser.add_argument("--n-bins", type=int, default=CascadeConfig.n_bins)
    parser.add_argument("--batch-size", type=int, default=CascadeConfig.batch_size)
    parser.add_argument("--seed", type=int, default=CascadeConfig.seed)
    args = parser.parse_args()

    config = CascadeConfig(
        epochs=args.epochs,
        stage2_epochs=args.stage2_epochs,
        n_bins=args.n_bins,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    results = fit_cascade(pd.read_csv(args.csv), config=config)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {args.output}")

    # Save confusion matrices as CSVs
    for stage_key in ("stage1", "stage3"):
        cm = results[stage_key]["confusion_matrix"]
        mat_df = pd.DataFrame(
            cm["actual_by_predicted"], index=cm["labels"], columns=cm["labels"]
        )
        mat_df.index.name = "actual"
        mat_df.to_csv(args.output.parent / f"{args.output.stem}_{stage_key}_confusion_matrix.csv")
        print(f"  {stage_key} confusion matrix saved")

    print(json.dumps(results["comparison"], indent=2))


if __name__ == "__main__":
    main()
