"""ARMED play-type classifier for clustered NFL data.

This implements the dense ARMED formulation from Nguyen, Treacher, and
Montillo (IEEE TPAMI, 2023):

* a fixed-effects MLP whose latent representation is adversarially trained
  not to reveal the cluster;
* Bayesian cluster-specific nonlinear slopes and intercepts;
* a mixed-effects prediction head combining fixed and random effects; and
* a Z-predictor that infers a soft mixture of training clusters for unseen
  clusters.

Input CSVs must contain one row per play, a categorical target column, the
normal play-type feature columns from ``model_feature_config.json``, and
``game_id`` plus ``posteam``. The default cluster is deliberately
``game_id::posteam`` for this project. Keep the cluster columns out of the
feature list. The default target labels match this repository's model.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_CLASSES = ("pass", "rush", "punt", "kick")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("model_feature_config.json")
DEFAULT_DATA_PATH = Path(__file__).resolve().with_name("play_type_training.csv")


def normal_play_type_features(config_path: Path = DEFAULT_CONFIG_PATH) -> list[str]:
    """Return the exact expanded feature list configured for play_type."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return list(config["models"]["play_type"]["expanded_columns"])


@dataclass
class ARMEDConfig:
    hidden_dim: int = 128
    latent_dim: int = 32
    random_effect_dim: int = 32
    dropout: float = 0.2
    lambda_g: float = 0.1
    lambda_f: float = 0.5
    lambda_k: float = 1e-4
    z_predictor_weight: float = 1.0
    prior_variance: float = 1.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 512
    patience: int = 12
    seed: int = 42


class ARMEDClassifier(nn.Module):
    """Dense ARMED classifier with nonlinear random slopes and intercepts."""

    def __init__(self, input_dim: int, n_classes: int, n_clusters: int, config: ARMEDConfig):
        super().__init__()
        self.config = config
        self.n_classes = int(n_classes)
        self.n_clusters = int(n_clusters)
        self.fixed = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.LeakyReLU(0.1),
        )
        self.fixed_head = nn.Linear(config.latent_dim, n_classes)
        self.adversary = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(config.hidden_dim // 2, n_clusters),
        )
        # Variational cluster-specific nonlinear slopes and class intercepts.
        self.random_mu = nn.Parameter(torch.zeros(n_clusters, config.latent_dim, n_classes))
        self.random_logvar = nn.Parameter(torch.full((n_clusters, config.latent_dim, n_classes), -4.0))
        self.intercept_mu = nn.Parameter(torch.zeros(n_clusters, n_classes))
        self.intercept_logvar = nn.Parameter(torch.full((n_clusters, n_classes), -4.0))
        self.z_predictor = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(config.hidden_dim // 2, n_clusters),
        )

    def _random_effect(self, latent: torch.Tensor, clusters: torch.Tensor, sample: bool) -> torch.Tensor:
        slope_mu = self.random_mu[clusters]
        intercept_mu = self.intercept_mu[clusters]
        if sample and self.training:
            slope = slope_mu + torch.exp(0.5 * self.random_logvar[clusters]) * torch.randn_like(slope_mu)
            intercept = intercept_mu + torch.exp(0.5 * self.intercept_logvar[clusters]) * torch.randn_like(intercept_mu)
        else:
            slope, intercept = slope_mu, intercept_mu
        return torch.sum(latent.unsqueeze(-1) * slope, dim=1) + intercept

    def forward(self, features: torch.Tensor, clusters: torch.Tensor, sample: bool = True):
        latent = self.fixed(features)
        fixed_logits = self.fixed_head(latent)
        random_logits = self._random_effect(latent, clusters, sample)
        mixed_logits = fixed_logits + random_logits
        cluster_logits = self.adversary(latent)
        return mixed_logits, fixed_logits, cluster_logits, self.kl_divergence()

    def predict_unseen(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Use soft Z-predictor assignments to apply learned random effects."""
        self.eval()
        with torch.no_grad():
            latent = self.fixed(features)
            fixed_logits = self.fixed_head(latent)
            z_weights = torch.softmax(self.z_predictor(features), dim=1)
            random_slope = torch.einsum("bc,ckd->bkd", z_weights, self.random_mu)
            random_intercept = z_weights @ self.intercept_mu
            random_logits = torch.sum(latent.unsqueeze(-1) * random_slope, dim=1) + random_intercept
        return fixed_logits + random_logits, z_weights

    def kl_divergence(self) -> torch.Tensor:
        prior_var = float(self.config.prior_variance)
        slope_var = torch.exp(self.random_logvar)
        intercept_var = torch.exp(self.intercept_logvar)
        slope_kl = 0.5 * torch.sum((slope_var + self.random_mu.square()) / prior_var - 1.0 - self.random_logvar + np.log(prior_var))
        intercept_kl = 0.5 * torch.sum((intercept_var + self.intercept_mu.square()) / prior_var - 1.0 - self.intercept_logvar + np.log(prior_var))
        return (slope_kl + intercept_kl) / self.n_clusters


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _multiclass_metrics(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int) -> dict[str, float]:
    predictions = np.argmax(probabilities, axis=1)
    sensitivities = []
    specificities = []
    aurocs = []
    for class_index in range(n_classes):
        truth = y_true == class_index
        predicted = predictions == class_index
        tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[False, True]).ravel()
        sensitivities.append(tp / max(tp + fn, 1))
        specificities.append(tn / max(tn + fp, 1))
        if truth.min() != truth.max():
            aurocs.append(roc_auc_score(truth, probabilities[:, class_index]))
    return {
        "auroc_macro_ovr": float(np.mean(aurocs)) if aurocs else float("nan"),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "sensitivity_macro_ovr": float(np.mean(sensitivities)),
        "specificity_macro_ovr": float(np.mean(specificities)),
    }


def _fit_transform_features(
    fit: pd.DataFrame,
    other_frames: list[pd.DataFrame],
    feature_columns: list[str],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """One-hot encode configured categorical play features using fit rows only."""
    fit_encoded = pd.get_dummies(fit[feature_columns], dtype=float)
    transformed = []
    for frame in other_frames:
        encoded = pd.get_dummies(frame[feature_columns], dtype=float)
        encoded = encoded.reindex(columns=fit_encoded.columns, fill_value=0.0)
        transformed.append(encoded.to_numpy(dtype=np.float32))
    return fit_encoded.to_numpy(dtype=np.float32), transformed


def evaluate_armed(
    frame: pd.DataFrame,
    target_column: str,
    cluster_column: str,
    feature_columns: Iterable[str],
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    config: ARMEDConfig | None = None,
    test_size: float = 0.2,
    device: str | None = None,
) -> dict[str, object]:
    """Train on complete clusters and evaluate both seen and unseen clusters."""
    config = config or ARMEDConfig()
    _seed_everything(config.seed)
    feature_columns = list(feature_columns)
    working = frame.copy()
    if cluster_column == "game_id::posteam":
        required_cluster_columns = {"game_id", "posteam"}
        missing_cluster_columns = required_cluster_columns - set(working.columns)
        if missing_cluster_columns:
            raise ValueError(f"Cannot construct game_id::posteam; missing columns: {sorted(missing_cluster_columns)}")
        working[cluster_column] = working["game_id"].astype(str) + "::" + working["posteam"].fillna("UNKNOWN").astype(str)
    clean = working.dropna(subset=[target_column, cluster_column] + feature_columns).copy()
    clean[target_column] = clean[target_column].astype(str)
    clean[cluster_column] = clean[cluster_column].astype(str)
    unknown = sorted(set(clean[target_column]) - set(classes))
    if unknown:
        raise ValueError(f"Unknown target labels: {unknown}")

    split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=config.seed)
    train_idx, unseen_idx = next(split.split(clean, groups=clean[cluster_column]))
    train = clean.iloc[train_idx].copy()
    unseen = clean.iloc[unseen_idx].copy()
    # Evaluate seen-cluster performance on rows from clusters that remain in
    # the fit set. This is distinct from the unseen-cluster split above.
    rng = np.random.default_rng(config.seed + 1)
    evaluation_mask = rng.random(len(train)) < test_size
    if evaluation_mask.sum() == 0 or evaluation_mask.sum() == len(train):
        evaluation_mask[: max(1, len(train) // 5)] = True
    seen = train.iloc[np.flatnonzero(evaluation_mask)].copy()
    fit = train.iloc[np.flatnonzero(~evaluation_mask)].copy()

    cluster_names = sorted(fit[cluster_column].unique())
    cluster_to_index = {name: index for index, name in enumerate(cluster_names)}
    fit_encoded, encoded_frames = _fit_transform_features(fit, [seen, unseen], feature_columns)
    scaler = StandardScaler().fit(fit_encoded)
    fit_x = scaler.transform(fit_encoded).astype(np.float32)
    seen_x = scaler.transform(encoded_frames[0]).astype(np.float32)
    unseen_x = scaler.transform(encoded_frames[1]).astype(np.float32)
    fit_y = fit[target_column].map({name: index for index, name in enumerate(classes)}).to_numpy(np.int64)
    seen_y = seen[target_column].map({name: index for index, name in enumerate(classes)}).to_numpy(np.int64)
    unseen_y = unseen[target_column].map({name: index for index, name in enumerate(classes)}).to_numpy(np.int64)
    fit_z = fit[cluster_column].map(cluster_to_index).to_numpy(np.int64)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ARMEDClassifier(fit_x.shape[1], len(classes), len(cluster_names), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(TensorDataset(torch.from_numpy(fit_x), torch.from_numpy(fit_z), torch.from_numpy(fit_y)), batch_size=config.batch_size, shuffle=True)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    no_improvement = 0
    history = []
    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        for batch_x, batch_z, batch_y in loader:
            batch_x, batch_z, batch_y = batch_x.to(device), batch_z.to(device), batch_y.to(device)
            mixed, fixed, cluster_logits, kl = model(batch_x, batch_z, sample=True)
            prediction_loss = F.cross_entropy(mixed, batch_y)
            fixed_loss = F.cross_entropy(fixed, batch_y)
            adversary_loss = F.cross_entropy(cluster_logits, batch_z)
            z_loss = F.cross_entropy(model.z_predictor(batch_x), batch_z)
            loss = prediction_loss + config.lambda_f * fixed_loss - config.lambda_g * adversary_loss + config.lambda_k * kl + config.z_predictor_weight * z_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
        mean_loss = total / max(len(loader), 1)
        history.append(mean_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= config.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        seen_logits, _, _, _ = model(torch.from_numpy(seen_x).to(device), torch.from_numpy(seen[cluster_column].map(cluster_to_index).to_numpy(np.int64)).to(device), sample=False)
        seen_probabilities = torch.softmax(seen_logits, dim=1).cpu().numpy()
        unseen_logits, z_weights = model.predict_unseen(torch.from_numpy(unseen_x).to(device))
        unseen_probabilities = torch.softmax(unseen_logits, dim=1).cpu().numpy()

    return {
        "config": asdict(config),
        "classes": list(classes),
        "n_rows": int(len(clean)),
        "n_training_clusters": len(cluster_names),
        "n_seen_evaluation_clusters": int(seen[cluster_column].nunique()),
        "n_unseen_clusters": int(unseen[cluster_column].nunique()),
        "seen_clusters": _multiclass_metrics(seen_y, seen_probabilities, len(classes)),
        "unseen_clusters": _multiclass_metrics(unseen_y, unseen_probabilities, len(classes)),
        "training_loss": history,
        "z_predictor_entropy_unseen": float((-z_weights * torch.log(torch.clamp(z_weights, min=1e-12))).sum(dim=1).mean().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate an ARMED play-type classifier.")
    parser.add_argument("csv", type=Path, nargs="?", default=DEFAULT_DATA_PATH)
    parser.add_argument("--target-column", default="play_type")
    parser.add_argument("--cluster-column", default="game_id::posteam")
    parser.add_argument("--features", nargs="+", default=None, help="Override normal play_type features")
    parser.add_argument("--output", type=Path, default=Path("armed_play_type_results.json"))
    parser.add_argument("--epochs", type=int, default=ARMEDConfig.epochs)
    parser.add_argument("--lambda-g", type=float, default=ARMEDConfig.lambda_g)
    parser.add_argument("--lambda-f", type=float, default=ARMEDConfig.lambda_f)
    parser.add_argument("--lambda-k", type=float, default=ARMEDConfig.lambda_k)
    args = parser.parse_args()
    config = ARMEDConfig(epochs=args.epochs, lambda_g=args.lambda_g, lambda_f=args.lambda_f, lambda_k=args.lambda_k)
    feature_columns = args.features or normal_play_type_features()
    results = evaluate_armed(pd.read_csv(args.csv), args.target_column, args.cluster_column, feature_columns, config=config)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
