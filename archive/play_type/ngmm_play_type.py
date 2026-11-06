"""Neural generalized mixed-effects model for the play-type data.

This is the NGMM construction of Slavutsky, Salazar, and Blei adapted to a
four-class multinomial-logit response. The paper derives the differentiable
ODE marginal likelihood for one-parameter exponential families; categorical
softmax is not one of those families. Here the same Gaussian random-effect
model is fit with a differentiable Monte Carlo approximation to the group
marginal likelihood:

    p(y_j | X_j) = E_gamma prod_i p(y_ij | x_ij, gamma),
    gamma ~ N(0, I).

The group split is deliberately game_id::posteam. For new groups, the paper's
Bayes-optimal rule uses only f_theta(x), so this runner reports fixed-effects
predictions separately from observed-group mixed predictions.
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


@dataclass
class NGMMConfig:
    hidden_dims: tuple[int, ...] = (16, 8, 4)
    random_effect_dim: int = 1
    group_context_hidden_dim: int = 16
    quadrature_samples: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 100
    group_batch_size: int = 32
    observations_per_group: int = 256
    patience: int = 12
    seed: int = 42


class NGMMClassifier(nn.Module):
    """Multinomial NGMM with context-conditioned Gaussian random effects."""

    def __init__(self, input_dim: int, group_context_dim: int, n_classes: int, config: NGMMConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        previous = input_dim
        for width in config.hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        self.fixed = nn.Sequential(*layers)
        self.fixed_head = nn.Linear(previous, n_classes)
        # g_psi(z) maps group-level team context to random-effect scales.
        self.random_scale = nn.Sequential(
            nn.Linear(group_context_dim, config.group_context_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.group_context_hidden_dim, config.random_effect_dim),
        )
        self.random_basis = nn.Parameter(torch.randn(config.random_effect_dim, n_classes) * 0.01)

    def fixed_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.fixed_head(self.fixed(features))

    def logits_with_random_effect(self, features: torch.Tensor, group_context: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        fixed = self.fixed_logits(features)
        # gamma: [samples, random_effect_dim] or [random_effect_dim]
        scale = F.softplus(self.random_scale(group_context))
        random_logits = gamma @ (scale.unsqueeze(-1) * self.random_basis)
        return fixed + random_logits

    def sampled_group_log_marginal(
        self,
        features: torch.Tensor,
        group_context: torch.Tensor,
        targets: torch.Tensor,
        gamma_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate log E_gamma[prod_i p(y_i | x_i, gamma)] stably."""
        fixed = self.fixed_logits(features)
        scale = F.softplus(self.random_scale(group_context))
        group_logits = fixed.unsqueeze(0) + torch.einsum(
            "sk,kc->sc", gamma_samples, scale.unsqueeze(-1) * self.random_basis
        ).unsqueeze(1)
        log_probabilities = torch.log_softmax(group_logits, dim=-1)
        joint_log_likelihood = log_probabilities[:, torch.arange(targets.numel()), targets].sum(dim=1)
        return torch.logsumexp(joint_log_likelihood, dim=0) - np.log(gamma_samples.shape[0])

    def mixed_probabilities(self, features: torch.Tensor, group_context: torch.Tensor, gamma_samples: torch.Tensor) -> torch.Tensor:
        logits = self.logits_with_random_effect(features, group_context, gamma_samples.mean(dim=0))
        return torch.softmax(logits, dim=1)

    def sample_new_group(self, features: torch.Tensor, group_context: torch.Tensor, n_samples: int = 32) -> torch.Tensor:
        """Sample new-group predictive probabilities from its context."""
        self.eval()
        with torch.no_grad():
            gamma = torch.randn(n_samples, self.config.random_effect_dim, device=features.device)
            fixed = self.fixed_logits(features)
            scale = F.softplus(self.random_scale(group_context))
            random_logits = torch.einsum(
                "sk,bkc->sbc", gamma, scale.unsqueeze(-1) * self.random_basis
            )
            return torch.softmax(fixed.unsqueeze(0) + random_logits, dim=-1)


class GroupBatchSampler:
    def __init__(self, groups: list[np.ndarray], batch_size: int, seed: int):
        self.groups = groups
        self.batch_size = batch_size
        self.seed = seed

    def batches(self, epoch: int) -> Iterable[list[np.ndarray]]:
        rng = np.random.default_rng(self.seed + epoch)
        indices = rng.permutation(len(self.groups))
        for start in range(0, len(indices), self.batch_size):
            yield [self.groups[index] for index in indices[start : start + self.batch_size]]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _encode_features(fit: pd.DataFrame, frames: list[pd.DataFrame], columns: list[str]) -> tuple[np.ndarray, list[np.ndarray]]:
    fit_encoded = pd.get_dummies(fit[columns], dtype=float)
    outputs = []
    for frame in frames:
        encoded = pd.get_dummies(frame[columns], dtype=float).reindex(columns=fit_encoded.columns, fill_value=0.0)
        outputs.append(encoded.to_numpy(dtype=np.float32))
    scaler = StandardScaler().fit(fit_encoded.to_numpy(dtype=np.float32))
    return scaler.transform(fit_encoded.to_numpy(dtype=np.float32)).astype(np.float32), [scaler.transform(output).astype(np.float32) for output in outputs]


def _encode_group_context(
    fit: pd.DataFrame,
    frames: list[pd.DataFrame],
    group_column: str,
    columns: list[str],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Aggregate team context once per group, then standardize using fit groups."""
    def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
        values = frame[columns].apply(pd.to_numeric, errors="coerce")
        return values.groupby(frame[group_column]).median().fillna(0.0)

    fit_groups = aggregate(fit)
    scaler = StandardScaler().fit(fit_groups.to_numpy(dtype=np.float32))
    encoded_frames = []
    def row_aligned(frame: pd.DataFrame) -> np.ndarray:
        groups = aggregate(frame)
        rows = groups.reindex(frame[group_column]).fillna(0.0)
        return scaler.transform(rows.to_numpy(dtype=np.float32)).astype(np.float32)

    return scaler.transform(fit_groups.to_numpy(dtype=np.float32)).astype(np.float32), [row_aligned(frame) for frame in frames]


def _metrics(targets: np.ndarray, probabilities: np.ndarray, classes: tuple[str, ...]) -> dict[str, float]:
    predictions = np.argmax(probabilities, axis=1)
    one_hot = np.eye(len(classes), dtype=float)[targets]
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "log_loss": float(log_loss(targets, probabilities, labels=list(range(len(classes))))),
        "auroc_macro_ovr": float(roc_auc_score(one_hot, probabilities, multi_class="ovr", average="macro")),
    }


def _confusion_matrix(targets: np.ndarray, probabilities: np.ndarray, classes: tuple[str, ...]) -> dict[str, list[list[int]]]:
    """Return an explicitly ordered actual-by-predicted confusion matrix."""
    predictions = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(classes))))
    return {
        "labels": list(classes),
        "actual_by_predicted": matrix.astype(int).tolist(),
    }


def fit_ngmm(
    frame: pd.DataFrame,
    target_column: str = "play_type",
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    feature_columns: Iterable[str] | None = None,
    group_context_columns: Iterable[str] | None = None,
    config: NGMMConfig | None = None,
    device: str | None = None,
) -> dict[str, object]:
    config = config or NGMMConfig()
    _seed_everything(config.seed)
    feature_columns = list(feature_columns or normal_play_type_features())
    group_context_columns = list(group_context_columns or DEFAULT_GROUP_CONTEXT_COLUMNS)
    group_context_columns = [column for column in group_context_columns if column in frame.columns]
    if not group_context_columns:
        raise ValueError("Input must contain at least one configured team-context column")
    working = frame.copy()
    if "game_id" not in working.columns or "posteam" not in working.columns:
        raise ValueError("Input must contain game_id and posteam for game_id::posteam clustering")
    working["ngmm_group"] = working["game_id"].astype(str) + "::" + working["posteam"].fillna("UNKNOWN").astype(str)
    working[target_column] = working[target_column].astype(str)
    working = working.dropna(subset=[target_column, "ngmm_group"] + feature_columns)
    target_map = {label: index for index, label in enumerate(classes)}
    unknown = sorted(set(working[target_column]) - set(classes))
    if unknown:
        raise ValueError(f"Unknown target labels: {unknown}")

    outer_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.seed)
    train_idx, new_idx = next(outer_split.split(working, groups=working["ngmm_group"]))
    train = working.iloc[train_idx].copy()
    new_groups = working.iloc[new_idx].copy()
    rng = np.random.default_rng(config.seed + 1)
    seen_mask = rng.random(len(train)) < 0.2
    if seen_mask.sum() == 0:
        seen_mask[0] = True
    seen_groups = train.iloc[np.flatnonzero(seen_mask)].copy()
    fit = train.iloc[np.flatnonzero(~seen_mask)].copy()

    fit_x, other_x = _encode_features(fit, [seen_groups, new_groups], feature_columns)
    seen_x, new_x = other_x
    fit_context_by_group, context_frames = _encode_group_context(
        fit, [seen_groups, new_groups], "ngmm_group", group_context_columns
    )
    seen_context_rows, new_context_rows = context_frames
    fit_y = fit[target_column].map(target_map).to_numpy(np.int64)
    seen_y = seen_groups[target_column].map(target_map).to_numpy(np.int64)
    new_y = new_groups[target_column].map(target_map).to_numpy(np.int64)
    group_names = sorted(fit["ngmm_group"].unique())
    group_to_index = {name: index for index, name in enumerate(group_names)}
    fit_group_indices = fit["ngmm_group"].map(group_to_index).to_numpy(np.int64)
    grouped_rows = [np.flatnonzero(fit_group_indices == index) for index in range(len(group_names))]
    fit_context = torch.from_numpy(fit_context_by_group)
    fit_context_by_name = {
        name: fit_context_by_group[index] for index, name in enumerate(sorted(fit["ngmm_group"].unique()))
    }
    seen_context = seen_context_rows

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = NGMMClassifier(fit_x.shape[1], fit_context_by_group.shape[1], len(classes), config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    sampler = GroupBatchSampler(grouped_rows, config.group_batch_size, config.seed)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    history: list[float] = []
    no_improvement = 0
    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        batches = 0
        for batch_groups in sampler.batches(epoch):
            optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=device)
            for row_indices in batch_groups:
                row_indices = row_indices[: config.observations_per_group]
                features = torch.from_numpy(fit_x[row_indices]).to(device)
                group_name = fit.iloc[row_indices[0]]["ngmm_group"]
                group_context = torch.from_numpy(fit_context_by_name[group_name]).to(device)
                targets = torch.from_numpy(fit_y[row_indices]).to(device)
                gamma = torch.randn(config.quadrature_samples, config.random_effect_dim, device=device)
                batch_loss = batch_loss - model.sampled_group_log_marginal(features, group_context, targets, gamma)
            batch_loss = batch_loss / max(len(batch_groups), 1)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(batch_loss.detach())
            batches += 1
        epoch_loss = total / max(batches, 1)
        history.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= config.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        seen_features = torch.from_numpy(seen_x).to(device)
        new_features = torch.from_numpy(new_x).to(device)
        seen_context_tensor = torch.from_numpy(seen_context).to(device)
        new_context_tensor = torch.from_numpy(new_context_rows).to(device)
        gamma = torch.zeros(1, config.random_effect_dim, device=device)
        fixed_seen = torch.softmax(model.fixed_logits(seen_features), dim=1).cpu().numpy()
        fixed_new = torch.softmax(model.fixed_logits(new_features), dim=1).cpu().numpy()
        mixed_seen = model.mixed_probabilities(seen_features, seen_context_tensor, gamma).cpu().numpy()
        sampled_new = model.sample_new_group(new_features, new_context_tensor, config.quadrature_samples).mean(dim=0).cpu().numpy()

    return {
        "model": model,
        "config": asdict(config),
        "classes": list(classes),
        "cluster_key": "game_id::posteam",
        "n_rows": int(len(working)),
        "n_training_groups": len(group_names),
        "n_seen_test_groups": int(seen_groups["ngmm_group"].nunique()),
        "n_new_test_groups": int(new_groups["ngmm_group"].nunique()),
        "group_context_columns": group_context_columns,
        "seen_groups_fixed_effects": _metrics(seen_y, fixed_seen, classes),
        "seen_groups_mixed_effects": _metrics(seen_y, mixed_seen, classes),
        "new_groups_fixed_effects": _metrics(new_y, fixed_new, classes),
        "new_groups_context_sampled": _metrics(new_y, sampled_new, classes),
        "confusion_matrices": {
            "seen_groups_fixed_effects": _confusion_matrix(seen_y, fixed_seen, classes),
            "seen_groups_mixed_effects": _confusion_matrix(seen_y, mixed_seen, classes),
            "new_groups_fixed_effects": _confusion_matrix(new_y, fixed_new, classes),
            "new_groups_context_sampled": _confusion_matrix(new_y, sampled_new, classes),
        },
        "training_group_marginal_nll": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the NGMM play-type model.")
    parser.add_argument("csv", type=Path, nargs="?", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=Path("ngmm_play_type_results.json"))
    parser.add_argument("--epochs", type=int, default=NGMMConfig.epochs)
    parser.add_argument("--quadrature-samples", type=int, default=NGMMConfig.quadrature_samples)
    parser.add_argument("--group-batch-size", type=int, default=NGMMConfig.group_batch_size)
    parser.add_argument("--group-context-columns", nargs="+", default=None, help="Override team-context columns used by g_psi(z)")
    args = parser.parse_args()
    config = NGMMConfig(epochs=args.epochs, quadrature_samples=args.quadrature_samples, group_batch_size=args.group_batch_size)
    results = fit_ngmm(pd.read_csv(args.csv), config=config, group_context_columns=args.group_context_columns)
    serializable = {key: value for key, value in results.items() if key != "model"}
    args.output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    matrix_directory = args.output.parent
    for regime, matrix_payload in serializable["confusion_matrices"].items():
        matrix = pd.DataFrame(
            matrix_payload["actual_by_predicted"],
            index=matrix_payload["labels"],
            columns=matrix_payload["labels"],
        )
        matrix.index.name = "actual"
        matrix.to_csv(matrix_directory / f"{args.output.stem}_{regime}_confusion_matrix.csv")
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
