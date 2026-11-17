"""Generalized cascaded residual model trainer.

Three-stage pipeline for any play-level classification target:
  1. Base MLP classifier trained on 45% of game::team groups.
  2. Context-conditioned heatmap regression on game-averaged, PCA-reduced
     residuals using KL divergence against a discretized, label-smoothed grid.
  3. Final MLP classifier on the remaining 45% of groups, augmented with a
     blended residual feature (70% true residual / 30% COM during training,
     sampled from the stage-2 distribution at inference).

Stage 2 residual draws replace hierarchical latent sampling in the simulation
runtime; stage 3 models do play-level inference.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.nn import functional as F

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
    # stage-3 residual blending: fraction of true residual vs COM prediction
    stage3_residual_blend: float = 0.7
    seed: int = 42


@dataclass
class CascadeArtifacts:
    """All trained components of a cascade model."""
    stage1_model: nn.Module
    stage2_model: nn.Module
    stage3_model: nn.Module
    stage1_preprocessor: ColumnTransformer
    stage3_preprocessor: ColumnTransformer
    stage2_context_scaler: StandardScaler
    pca: PCA
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    feature_columns: list[str]
    stage3_feature_columns: list[str]
    context_columns: list[str]
    categorical_columns: list[str]
    n_classes: int
    class_labels: list[str]
    config: CascadeConfig
    metrics: dict[str, Any] = field(default_factory=dict)


# ── models ───────────────────────────────────────────────────────────────────


class CascadeMLP(nn.Module):
    """Feed-forward classifier."""

    def __init__(self, input_dim: int, n_classes: int, config: CascadeConfig):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in config.hidden_dims:
            layers.extend([nn.Linear(prev, width), nn.ReLU(), nn.Dropout(config.dropout)])
            prev = width
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)
        # Expose fc1 for runtime width-compatibility checks
        self.fc1 = layers[0]
        self.out = layers[-1]

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
        return self.net(x)

    def predict_distribution(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


# ── helpers ──────────────────────────────────────────────────────────────────


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))


def _build_preprocessor(
    fit: pd.DataFrame,
    columns: list[str],
    categorical_cols: list[str],
) -> tuple[ColumnTransformer, np.ndarray]:
    """Build ColumnTransformer and fit-transform the *fit* frame."""
    cat_cols_present = [c for c in categorical_cols if c in columns]
    num_cols = [c for c in columns if c not in cat_cols_present]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols_present),
    ])
    x_fit = preprocessor.fit_transform(fit[columns]).astype(np.float32)
    x_fit = np.nan_to_num(x_fit, nan=0.0)
    return preprocessor, x_fit


def _transform(preprocessor: ColumnTransformer, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Transform a frame using a fitted preprocessor, auto-filling missing columns."""
    expected = list(preprocessor.feature_names_in_)
    missing = [c for c in expected if c not in frame.columns]
    f = frame.copy()
    for c in missing:
        f[c] = 0.0
    f = f.reindex(columns=expected, fill_value=0.0)
    out = preprocessor.transform(f).astype(np.float32)
    return np.nan_to_num(out, nan=0.0)


def _encode_group_context(
    fit: pd.DataFrame,
    frames: list[pd.DataFrame],
    group_column: str,
    columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, list[np.ndarray], StandardScaler]:
    """Aggregate context per group (median), standardize on *fit* groups."""
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


def _metrics(targets: np.ndarray, probabilities: np.ndarray, n_classes: int) -> dict[str, float]:
    preds = np.argmax(probabilities, axis=1)
    result: dict[str, float] = {
        "accuracy": float(accuracy_score(targets, preds)),
        "log_loss": float(log_loss(targets, probabilities, labels=list(range(n_classes)))),
    }
    if n_classes > 2:
        result["balanced_accuracy"] = float(balanced_accuracy_score(targets, preds))
        one_hot = np.eye(n_classes, dtype=float)[targets]
        result["auroc_macro_ovr"] = float(roc_auc_score(one_hot, probabilities, multi_class="ovr", average="macro"))
    else:
        result["auroc"] = float(roc_auc_score(targets, probabilities[:, 1]))
    return result


def _confusion_matrix_payload(targets: np.ndarray, probabilities: np.ndarray, class_labels: list[str]) -> dict:
    preds = np.argmax(probabilities, axis=1)
    mat = confusion_matrix(targets, preds, labels=list(range(len(class_labels))))
    return {"labels": class_labels, "actual_by_predicted": mat.astype(int).tolist()}


def _train_mlp(
    model: CascadeMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: CascadeConfig,
    device: str,
    label: str = "",
) -> tuple[CascadeMLP, list[float]]:
    """Train an MLP with early stopping on validation log-loss."""
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

        model.eval()
        with torch.no_grad():
            val_probs = torch.softmax(model(xv), dim=1).cpu().numpy()
        val_ll = float(log_loss(y_val, val_probs, labels=list(range(probs_shape(probs := val_probs)))))
        if epoch % 5 == 0 or no_improve == 0:
            prefix = f"[{label}] " if label else ""
            print(f"  {prefix}epoch {epoch:3d}  train_loss={avg:.4f}  val_logloss={val_ll:.4f}", flush=True)
        if val_ll < best_loss:
            best_loss = val_ll
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= config.patience:
            print(f"  [{label}] Early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, history


def probs_shape(probs: np.ndarray) -> int:
    return probs.shape[1]


def _build_soft_grid_targets(
    residuals: np.ndarray,
    bin_edges: np.ndarray,
    sigma_bins: float,
    label_smoothing: float,
) -> np.ndarray:
    """Map scalar residuals to smoothed distributions over bins."""
    n_bins = len(bin_edges) - 1
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    targets = np.zeros((len(residuals), n_bins), dtype=np.float64)
    for i, r in enumerate(residuals):
        dists = (bin_centers - r) / (sigma_bins * bin_width)
        weights = np.exp(-0.5 * dists**2)
        weights /= weights.sum()
        weights = (1.0 - label_smoothing) * weights + label_smoothing / n_bins
        targets[i] = weights
    return targets


def sample_from_grid(
    distribution: np.ndarray,
    bin_edges: np.ndarray,
    rng: np.random.Generator | None = None,
) -> float:
    """Sample a continuous residual value from a discretized bin distribution
    via inverse CDF with linear interpolation within the selected bin."""
    rng = rng or np.random.default_rng()
    n_bins = len(distribution)
    cdf = np.cumsum(distribution)
    cdf = cdf / cdf[-1]  # ensure normalization
    u = rng.uniform()
    bin_idx = int(np.searchsorted(cdf, u, side="right"))
    bin_idx = min(bin_idx, n_bins - 1)
    # Linear interpolation within the bin
    bin_width = bin_edges[bin_idx + 1] - bin_edges[bin_idx]
    # Position within the bin based on how far into the bin's CDF mass we are
    prev_cdf = cdf[bin_idx - 1] if bin_idx > 0 else 0.0
    bin_mass = distribution[bin_idx]
    if bin_mass > 0:
        frac = (u - prev_cdf) / bin_mass
        frac = np.clip(frac, 0.0, 1.0)
    else:
        frac = 0.5
    return float(bin_edges[bin_idx] + frac * bin_width)


# ── save / load ──────────────────────────────────────────────────────────────


def save_cascade_artifacts(
    artifacts: CascadeArtifacts,
    output_dir: Path,
    model_prefix: str,
    feature_groups: list[str] | None = None,
) -> None:
    """Save cascade artifacts in runtime-compatible format.

    Parameters
    ----------
    artifacts : Trained cascade components.
    output_dir : Directory to write artifacts.
    model_prefix : File prefix (e.g., "play", "comp", "target", "rusher").
    feature_groups : Feature group names for runtime column_dict (e.g.,
        ["game_state_cols", "team_stats_columns"]). If None, defaults to
        the raw feature column list.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 3 (inference model)
    torch.save(artifacts.stage3_model.state_dict(), output_dir / f"{model_prefix}_cascade_stage3_model.pt")
    joblib.dump(artifacts.stage3_preprocessor, output_dir / f"{model_prefix}_cascade_stage3_preprocessor.pkl")

    # Stage 2 (residual distribution model)
    torch.save(artifacts.stage2_model.state_dict(), output_dir / f"{model_prefix}_cascade_stage2_model.pt")
    joblib.dump(artifacts.stage2_context_scaler, output_dir / f"{model_prefix}_cascade_stage2_preprocessor.pkl")

    # Stage 1 (training reference only, NOT loaded by runtime)
    torch.save(artifacts.stage1_model.state_dict(), output_dir / f"{model_prefix}_cascade_stage1_model.pt")
    joblib.dump(artifacts.stage1_preprocessor, output_dir / f"{model_prefix}_cascade_stage1_preprocessor.pkl")

    # Metadata: bin edges, PCA state, config
    meta = {
        "kind": "cascade",
        "n_classes": artifacts.n_classes,
        "class_labels": artifacts.class_labels,
        "n_bins": len(artifacts.bin_edges) - 1,
        "bin_edges": artifacts.bin_edges.tolist(),
        "bin_centers": artifacts.bin_centers.tolist(),
        "pca_components": artifacts.pca.components_.tolist(),
        "pca_mean": artifacts.pca.mean_.tolist(),
        "pca_explained_variance_ratio": float(artifacts.pca.explained_variance_ratio_[0]),
        "context_columns": artifacts.context_columns,
        "feature_columns": artifacts.feature_columns,
        "stage3_feature_columns": artifacts.stage3_feature_columns,
        "categorical_columns": artifacts.categorical_columns,
        "latent_dim": 1,
        "config": asdict(artifacts.config),
        "dims": {
            "stage3_input_dim": artifacts.stage3_model.fc1.in_features,
            "stage2_context_dim": artifacts.stage2_model.net[0].in_features,
            "output_dim": artifacts.n_classes,
        },
        "arch": {
            "stage3_hidden_dims": list(artifacts.config.hidden_dims),
            "stage3_dropout": artifacts.config.dropout,
            "stage2_hidden_dims": list(artifacts.config.stage2_hidden_dims),
            "stage2_dropout": artifacts.config.stage2_dropout,
        },
        "metrics": artifacts.metrics,
    }
    with open(output_dir / f"{model_prefix}_cascade_arch.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Column groups for runtime column_dict — must be group names that the
    # runtime resolves via getattr(runtime, group_name).
    joblib.dump(
        feature_groups or artifacts.stage3_feature_columns,
        output_dir / f"{model_prefix}_cascade_columns.pkl",
    )

    print(f"[cascade] Saved artifacts for '{model_prefix}' to {output_dir}")


def load_cascade_artifacts(
    output_dir: Path,
    model_prefix: str,
) -> CascadeArtifacts:
    """Load cascade artifacts from disk."""
    output_dir = Path(output_dir)
    with open(output_dir / f"{model_prefix}_cascade_arch.json") as f:
        meta = json.load(f)

    config = CascadeConfig(**meta["config"])
    n_classes = meta["n_classes"]

    # Reconstruct stage 3 model
    stage3_input_dim = meta["dims"]["stage3_input_dim"]
    stage3_model = CascadeMLP(stage3_input_dim, n_classes, config)
    stage3_model.load_state_dict(torch.load(output_dir / f"{model_prefix}_cascade_stage3_model.pt", map_location="cpu", weights_only=True))
    stage3_model.eval()

    # Reconstruct stage 2 model
    stage2_context_dim = meta["dims"]["stage2_context_dim"]
    n_bins = meta["n_bins"]
    stage2_model = ResidualGridNet(stage2_context_dim, n_bins, config)
    stage2_model.load_state_dict(torch.load(output_dir / f"{model_prefix}_cascade_stage2_model.pt", map_location="cpu", weights_only=True))
    stage2_model.eval()

    # Reconstruct stage 1 model
    stage1_model = CascadeMLP(stage3_input_dim - 1, n_classes, config)  # -1 for residual_com
    stage1_model.load_state_dict(torch.load(output_dir / f"{model_prefix}_cascade_stage1_model.pt", map_location="cpu", weights_only=True))
    stage1_model.eval()

    # Load preprocessors
    stage1_preprocessor = joblib.load(output_dir / f"{model_prefix}_cascade_stage1_preprocessor.pkl")
    stage3_preprocessor = joblib.load(output_dir / f"{model_prefix}_cascade_stage3_preprocessor.pkl")
    stage2_context_scaler = joblib.load(output_dir / f"{model_prefix}_cascade_stage2_preprocessor.pkl")

    # Reconstruct PCA
    pca = PCA(n_components=1)
    pca.components_ = np.array(meta["pca_components"])
    pca.mean_ = np.array(meta["pca_mean"])
    pca.explained_variance_ratio_ = np.array([meta["pca_explained_variance_ratio"]])

    return CascadeArtifacts(
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        stage3_model=stage3_model,
        stage1_preprocessor=stage1_preprocessor,
        stage3_preprocessor=stage3_preprocessor,
        stage2_context_scaler=stage2_context_scaler,
        pca=pca,
        bin_edges=np.array(meta["bin_edges"]),
        bin_centers=np.array(meta["bin_centers"]),
        feature_columns=meta["feature_columns"],
        stage3_feature_columns=meta["stage3_feature_columns"],
        context_columns=meta["context_columns"],
        categorical_columns=meta["categorical_columns"],
        n_classes=n_classes,
        class_labels=meta["class_labels"],
        config=config,
        metrics=meta.get("metrics", {}),
    )


# ── main pipeline ────────────────────────────────────────────────────────────


def fit_cascade_general(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    context_columns: list[str],
    categorical_columns: list[str] | None = None,
    class_labels: list[str] | None = None,
    config: CascadeConfig | None = None,
    device: str | None = None,
    group_column: str | None = None,
    model_label: str = "cascade",
) -> CascadeArtifacts:
    """Fit the full three-stage cascade pipeline on any play-level dataset.

    Parameters
    ----------
    frame : DataFrame with game_id, posteam (or a pre-built group_column), target, features, context.
    target_column : Column name containing the classification target.
    feature_columns : Play-level feature column names.
    context_columns : Game-level context column names (used by stage 2).
    categorical_columns : Categorical feature columns (auto-detected if None).
    class_labels : Ordered class labels (inferred from data if None).
    config : CascadeConfig.
    device : Torch device string.
    group_column : Pre-built group column name; defaults to game_id::posteam.
    model_label : Label for logging.
    """
    config = config or CascadeConfig()
    _seed_everything(config.seed)
    categorical_columns = categorical_columns or []
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── Prepare data ─────────────────────────────────────────────────────────
    working = frame.copy()
    if group_column is None:
        if "game_id" not in working.columns or "posteam" not in working.columns:
            raise ValueError("Need game_id and posteam, or a pre-built group_column")
        working["_cascade_group"] = working["game_id"].astype(str) + "::" + working["posteam"].fillna("UNKNOWN").astype(str)
        group_column = "_cascade_group"
    elif group_column not in working.columns:
        raise ValueError(f"Group column '{group_column}' not in frame")

    # Filter feature columns to those present in data
    feature_columns = [c for c in feature_columns if c in working.columns]
    context_columns = [c for c in context_columns if c in working.columns]
    categorical_columns = [c for c in categorical_columns if c in feature_columns]
    if not context_columns:
        raise ValueError("Need at least one context column")

    working = working.dropna(subset=[target_column, group_column])

    # Build target mapping
    if class_labels is None:
        class_labels = sorted(working[target_column].unique().tolist())
    target_map = {label: idx for idx, label in enumerate(class_labels)}
    unknown = sorted(set(working[target_column].astype(str)) - set(class_labels))
    if unknown:
        # For integer targets, convert to string for comparison
        if working[target_column].dtype in (np.int64, np.float64, int, float):
            working[target_column] = working[target_column].astype(int).astype(str)
            class_labels = sorted(working[target_column].unique().tolist())
            target_map = {label: idx for idx, label in enumerate(class_labels)}
            unknown = sorted(set(working[target_column]) - set(class_labels))
    if unknown:
        raise ValueError(f"Unknown target labels: {unknown}")
    working[target_column] = working[target_column].astype(str)
    n_classes = len(class_labels)

    # ── Three-way stratified split (45 / 45 / 10) ───────────────────────────
    val_frac = 1.0 - config.stage1_frac - config.stage3_frac
    splitter1 = GroupShuffleSplit(n_splits=1, test_size=1.0 - config.stage1_frac, random_state=config.seed)
    s1_idx, rem_idx = next(splitter1.split(working, groups=working[group_column]))
    stage1_train = working.iloc[s1_idx].copy()
    remainder = working.iloc[rem_idx].copy()

    stage3_ratio = config.stage3_frac / (config.stage3_frac + val_frac)
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=1.0 - stage3_ratio, random_state=config.seed + 1)
    s3_idx, val_idx = next(splitter2.split(remainder, groups=remainder[group_column]))
    stage3_train = remainder.iloc[s3_idx].copy()
    stage1_val = remainder.iloc[val_idx].copy()

    print(f"[{model_label}] Split — stage1: {len(stage1_train)} rows/{stage1_train[group_column].nunique()} groups, "
          f"stage3: {len(stage3_train)} rows/{stage3_train[group_column].nunique()} groups, "
          f"val: {len(stage1_val)} rows/{stage1_val[group_column].nunique()} groups")

    # Verify no group overlap
    g1 = set(stage1_train[group_column].unique())
    g3 = set(stage3_train[group_column].unique())
    gv = set(stage1_val[group_column].unique())
    assert not (g1 & g3) and not (g1 & gv) and not (g3 & gv), "Group overlap detected"

    # ── Feature encoding (stage 1 features) ──────────────────────────────────
    stage1_preprocessor, s1_x = _build_preprocessor(stage1_train, feature_columns, categorical_columns)
    val_x = _transform(stage1_preprocessor, stage1_val, feature_columns)

    y_s1 = stage1_train[target_column].map(target_map).to_numpy(np.int64)
    y_s3 = stage3_train[target_column].map(target_map).to_numpy(np.int64)
    y_val = stage1_val[target_column].map(target_map).to_numpy(np.int64)

    # ── Group context encoding ───────────────────────────────────────────────
    fit_ctx_df, fit_ctx_arr, ctx_frames, ctx_scaler = _encode_group_context(
        stage1_train, [stage3_train, stage1_val], group_column, context_columns
    )
    s3_ctx_rows, val_ctx_rows = ctx_frames

    # ── Stage 1: base MLP classifier ─────────────────────────────────────────
    print(f"\n[{model_label}] Stage 1: training base classifier ({n_classes} classes)")
    stage1_model = CascadeMLP(s1_x.shape[1], n_classes, config)
    stage1_model, s1_history = _train_mlp(stage1_model, s1_x, y_s1, val_x, y_val, config, device, label=f"{model_label}/s1")

    with torch.no_grad():
        s1_probs_train = torch.softmax(stage1_model(torch.from_numpy(s1_x).to(device)), dim=1).cpu().numpy()
        s1_probs_val = torch.softmax(stage1_model(torch.from_numpy(val_x).to(device)), dim=1).cpu().numpy()

    s1_val_metrics = _metrics(y_val, s1_probs_val, n_classes)
    print(f"[{model_label}] Stage 1 val: {json.dumps(s1_val_metrics, indent=2)}")

    # ── Stage 2a: residuals → game-averaged → PCA to 1D ────────────────────
    print(f"\n[{model_label}] Stage 2a: computing residuals and PCA")
    one_hot_s1 = np.eye(n_classes, dtype=np.float64)[y_s1]
    residuals_nd = one_hot_s1 - s1_probs_train.astype(np.float64)

    resid_df = pd.DataFrame(residuals_nd, columns=[f"_resid_{i}" for i in range(n_classes)])
    resid_df[group_column] = stage1_train[group_column].values
    game_residuals = resid_df.groupby(group_column)[[f"_resid_{i}" for i in range(n_classes)]].mean()

    pca = PCA(n_components=1)
    game_resid_1d = pca.fit_transform(game_residuals.to_numpy()).ravel()
    print(f"[{model_label}] PCA explained variance: {pca.explained_variance_ratio_[0]:.4f}, "
          f"residual std: {game_resid_1d.std():.6f}, range: [{game_resid_1d.min():.4f}, {game_resid_1d.max():.4f}]")

    # ── Stage 2b: discretize and train context model ─────────────────────────
    margin = 0.1 * (game_resid_1d.max() - game_resid_1d.min())
    grid_min = game_resid_1d.min() - margin
    grid_max = game_resid_1d.max() + margin
    bin_edges = np.linspace(grid_min, grid_max, config.n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    soft_targets = _build_soft_grid_targets(game_resid_1d, bin_edges, config.grid_smoothing_sigma, config.grid_label_smoothing)

    # Build validation targets for stage 2
    one_hot_val = np.eye(n_classes, dtype=np.float64)[y_val]
    val_resid_nd = one_hot_val - s1_probs_val.astype(np.float64)
    val_resid_df = pd.DataFrame(val_resid_nd, columns=[f"_resid_{i}" for i in range(n_classes)])
    val_resid_df[group_column] = stage1_val[group_column].values
    val_game_residuals = val_resid_df.groupby(group_column)[[f"_resid_{i}" for i in range(n_classes)]].mean()
    val_game_resid_1d = pca.transform(val_game_residuals.to_numpy()).ravel()
    val_soft_targets = _build_soft_grid_targets(val_game_resid_1d, bin_edges, config.grid_smoothing_sigma, config.grid_label_smoothing)
    val_ctx_for_groups = stage1_val.groupby(group_column)[context_columns].median().fillna(0.0)
    val_ctx_for_groups = val_ctx_for_groups.reindex(val_game_residuals.index).fillna(0.0)
    val_ctx_arr = ctx_scaler.transform(val_ctx_for_groups.to_numpy(dtype=np.float32)).astype(np.float32)

    # Align context features with game residuals
    ctx_for_groups = fit_ctx_df.reindex(game_residuals.index).fillna(0.0)
    ctx_group_arr = ctx_scaler.transform(ctx_for_groups.to_numpy(dtype=np.float32)).astype(np.float32)

    print(f"\n[{model_label}] Stage 2b: training residual context model")
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

        stage2_model.eval()
        with torch.no_grad():
            val_log_preds = F.log_softmax(stage2_model(x_s2_val), dim=-1)
            val_kl = float(F.kl_div(val_log_preds, t_s2_val, reduction="batchmean"))
        if epoch % 10 == 0 or no_improve_s2 == 0:
            print(f"  [{model_label}/s2] epoch {epoch:3d}  train_kl={avg_s2:.6f}  val_kl={val_kl:.6f}", flush=True)
        if val_kl < best_s2_loss:
            best_s2_loss = val_kl
            best_s2_state = copy.deepcopy(stage2_model.state_dict())
            no_improve_s2 = 0
        else:
            no_improve_s2 += 1
        if no_improve_s2 >= config.stage2_patience:
            print(f"  [{model_label}/s2] Early stopping at epoch {epoch}", flush=True)
            break

    stage2_model.load_state_dict(best_s2_state)
    stage2_model.eval()
    print(f"[{model_label}] Stage 2 best val KL: {best_s2_loss:.6f}")

    # ── Stage 2c: predict residual distributions for ALL groups ──────────────
    all_groups_df = pd.concat([stage1_train, stage3_train, stage1_val], ignore_index=True)
    all_ctx = all_groups_df.groupby(group_column)[context_columns].median().fillna(0.0)
    all_ctx_arr = ctx_scaler.transform(all_ctx.to_numpy(dtype=np.float32)).astype(np.float32)

    with torch.no_grad():
        all_dist = stage2_model.predict_distribution(torch.from_numpy(all_ctx_arr).to(device)).cpu().numpy()
    com_values = (all_dist * bin_centers[np.newaxis, :].astype(np.float64)).sum(axis=1)
    com_map = pd.Series(com_values, index=all_ctx.index)

    # ── Stage 3: augment with blended residual feature ────────────────────────
    print(f"\n[{model_label}] Stage 3: training final classifier with residual feature")
    blend = config.stage3_residual_blend

    # True residuals for stage 3 groups (computed from stage 1 model predictions on stage 3 data)
    s3_x_s1_features = _transform(stage1_preprocessor, stage3_train, feature_columns)
    with torch.no_grad():
        s3_probs = torch.softmax(stage1_model(torch.from_numpy(s3_x_s1_features).to(device)), dim=1).cpu().numpy()
    one_hot_s3 = np.eye(n_classes, dtype=np.float64)[y_s3]
    s3_resid_nd = one_hot_s3 - s3_probs.astype(np.float64)
    s3_resid_df = pd.DataFrame(s3_resid_nd, columns=[f"_resid_{i}" for i in range(n_classes)])
    s3_resid_df[group_column] = stage3_train[group_column].values
    s3_game_residuals = s3_resid_df.groupby(group_column)[[f"_resid_{i}" for i in range(n_classes)]].mean()
    s3_true_resid_1d = pca.transform(s3_game_residuals.to_numpy()).ravel()
    s3_true_resid_map = pd.Series(s3_true_resid_1d, index=s3_game_residuals.index)

    # Blend: 70% true residual + 30% COM prediction
    s3_com = stage3_train[group_column].map(com_map).fillna(0.0).to_numpy()
    s3_true = stage3_train[group_column].map(s3_true_resid_map).fillna(0.0).to_numpy()
    s3_blended = blend * s3_true + (1.0 - blend) * s3_com

    # For validation: use pure COM (inference-like)
    val_com = stage1_val[group_column].map(com_map).fillna(0.0).to_numpy()

    # Build stage 3 feature sets with residual_com
    stage3_train_aug = stage3_train.copy()
    stage3_train_aug["residual_com"] = s3_blended
    stage1_val_aug = stage1_val.copy()
    stage1_val_aug["residual_com"] = val_com

    stage3_feature_columns = feature_columns + ["residual_com"]
    stage3_preprocessor, s3_x_aug = _build_preprocessor(stage3_train_aug, stage3_feature_columns, categorical_columns)
    val_x_aug = _transform(stage3_preprocessor, stage1_val_aug, stage3_feature_columns)

    stage3_model = CascadeMLP(s3_x_aug.shape[1], n_classes, config)
    stage3_model, s3_history = _train_mlp(stage3_model, s3_x_aug, y_s3, val_x_aug, y_val, config, device, label=f"{model_label}/s3")

    with torch.no_grad():
        s3_probs_val = torch.softmax(stage3_model(torch.from_numpy(val_x_aug).to(device)), dim=1).cpu().numpy()

    s3_val_metrics = _metrics(y_val, s3_probs_val, n_classes)
    print(f"[{model_label}] Stage 3 val: {json.dumps(s3_val_metrics, indent=2)}")

    # ── Assemble artifacts ────────────────────────────────────────────────────
    metrics = {
        "split_info": {
            "stage1_train_rows": len(stage1_train),
            "stage3_train_rows": len(stage3_train),
            "stage1_val_rows": len(stage1_val),
            "stage1_train_groups": int(stage1_train[group_column].nunique()),
            "stage3_train_groups": int(stage3_train[group_column].nunique()),
            "stage1_val_groups": int(stage1_val[group_column].nunique()),
        },
        "stage1": {
            "val_metrics": s1_val_metrics,
            "confusion_matrix": _confusion_matrix_payload(y_val, s1_probs_val, class_labels),
        },
        "stage2": {
            "pca_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
            "grid_range": [float(grid_min), float(grid_max)],
            "n_bins": config.n_bins,
            "best_val_kl": best_s2_loss,
        },
        "stage3": {
            "val_metrics": s3_val_metrics,
            "confusion_matrix": _confusion_matrix_payload(y_val, s3_probs_val, class_labels),
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

    return CascadeArtifacts(
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        stage3_model=stage3_model,
        stage1_preprocessor=stage1_preprocessor,
        stage3_preprocessor=stage3_preprocessor,
        stage2_context_scaler=ctx_scaler,
        pca=pca,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        feature_columns=feature_columns,
        stage3_feature_columns=stage3_feature_columns,
        context_columns=context_columns,
        categorical_columns=categorical_columns,
        n_classes=n_classes,
        class_labels=class_labels,
        config=config,
        metrics=metrics,
    )
