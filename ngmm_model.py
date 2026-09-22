"""Shared Neural Generalized Mixed-Effects Model components.

The model follows Slavutsky, Salazar, and Blei's NGMM form,

    eta_ij = f_theta(x_ij) + g_psi(z_j) gamma_j,
    gamma_j ~ N(0, I),

with a differentiable Monte Carlo approximation to the group marginal
likelihood. ``g_psi`` is group-context dependent and its output is shared by
all observations in a group.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

TargetMode = Literal["binary", "multiclass", "ordinal", "gaussian"]


@dataclass
class NGMMArchitecture:
    hidden_dims: tuple[int, ...] = (128, 64, 32)
    context_hidden_dims: tuple[int, ...] = (64, 32)
    latent_dim: int = 1
    dropout: float = 0.1
    target_mode: TargetMode = "multiclass"
    output_dim: int = 4

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class NGMMModel(nn.Module):
    """NGMM fixed-effects network with context-conditioned random effects."""

    def __init__(self, input_dim: int, context_dim: int, architecture: NGMMArchitecture):
        super().__init__()
        self.architecture = architecture
        self.target_mode = architecture.target_mode
        self.output_dim = int(architecture.output_dim)
        self.latent_dim = int(architecture.latent_dim)

        self.fixed = self._make_mlp(input_dim, architecture.hidden_dims, architecture.dropout)
        fixed_width = architecture.hidden_dims[-1] if architecture.hidden_dims else input_dim
        self.fixed_head = nn.Linear(fixed_width, self._head_dim())
        self.context = self._make_mlp(context_dim, architecture.context_hidden_dims, 0.0)
        context_width = architecture.context_hidden_dims[-1] if architecture.context_hidden_dims else context_dim
        self.random_design = nn.Linear(context_width, self.latent_dim)
        self.random_basis = nn.Parameter(torch.zeros(self.latent_dim, self._head_dim()))
        nn.init.normal_(self.random_basis, mean=0.0, std=0.02)
        if self.target_mode == "gaussian":
            self.log_observation_scale = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _make_mlp(input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, int(width)), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            previous = int(width)
        return nn.Sequential(*layers)

    def _head_dim(self) -> int:
        if self.target_mode in {"binary", "gaussian"}:
            return 1
        if self.target_mode == "ordinal":
            return max(1, self.output_dim - 1)
        return self.output_dim

    def _random_logits(self, context: Tensor, gamma: Tensor) -> Tensor:
        design = F.softplus(self.random_design(self.context(context)))
        effect_basis = design.unsqueeze(-1) * self.random_basis
        return torch.einsum("sk,bko->sbo", gamma, effect_basis)

    def logits(self, features: Tensor, context: Tensor, gamma: Tensor | None = None) -> Tensor:
        fixed_logits = self.fixed_head(self.fixed(features))
        if gamma is None:
            return fixed_logits
        if gamma.ndim == 1:
            gamma = gamma.unsqueeze(0)
        return fixed_logits.unsqueeze(0) + self._random_logits(context, gamma)

    def fixed_output(self, features: Tensor) -> Tensor:
        return self.logits(features, torch.zeros(features.shape[0], self.context[0].in_features, device=features.device))

    def class_probabilities(self, logits: Tensor) -> Tensor:
        if self.target_mode == "binary":
            return torch.sigmoid(logits)
        if self.target_mode == "gaussian":
            return logits
        if self.target_mode == "ordinal":
            cumulative = torch.sigmoid(logits)
            cumulative = torch.cumprod(cumulative, dim=-1)
            first = 1.0 - cumulative[..., :1]
            middle = cumulative[..., :-1] - cumulative[..., 1:]
            return torch.cat([first, middle, cumulative[..., -1:]], dim=-1)
        return torch.softmax(logits, dim=-1)

    def log_likelihood(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self.target_mode == "gaussian":
            scale = torch.exp(self.log_observation_scale).clamp_min(1e-5)
            expanded_targets = targets.float().unsqueeze(0).expand(logits.shape[0], -1)
            residual = (expanded_targets - logits.squeeze(-1)) / scale
            return -0.5 * (residual.square() + 2.0 * self.log_observation_scale + torch.log(torch.tensor(2.0 * torch.pi, device=logits.device)))
        if self.target_mode == "binary":
            expanded_targets = targets.float().unsqueeze(0).expand(logits.shape[0], -1)
            return -F.binary_cross_entropy_with_logits(logits.squeeze(-1), expanded_targets, reduction="none")
        probabilities = self.class_probabilities(logits).clamp_min(1e-8)
        target_indices = targets.long().unsqueeze(0).expand(probabilities.shape[0], -1).unsqueeze(-1)
        return probabilities.gather(dim=-1, index=target_indices).squeeze(-1).log()

    def group_log_marginal(
        self,
        features: Tensor,
        context: Tensor,
        targets: Tensor,
        gamma_samples: Tensor,
    ) -> Tensor:
        """Estimate log E_gamma[prod_i p(y_ij | x_ij, z_j, gamma)]."""
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.shape[0] != 1:
            raise ValueError("A group marginal likelihood requires one context vector per group")
        sampled_logits = self.logits(features, context, gamma_samples)
        log_likelihood = self.log_likelihood(sampled_logits, targets)
        joint = log_likelihood.sum(dim=-1)
        return torch.logsumexp(joint, dim=0) - torch.log(torch.tensor(float(gamma_samples.shape[0]), device=joint.device))

    def sample_new_group(
        self,
        features: Tensor,
        context: Tensor,
        samples: int = 32,
    ) -> Tensor:
        """Return sampled predictive outputs for a new group and its context."""
        self.eval()
        with torch.no_grad():
            gamma = torch.randn(samples, self.latent_dim, device=features.device)
            sampled_logits = self.logits(features, context, gamma)
            return self.class_probabilities(sampled_logits)

    def fixed_prediction(self, features: Tensor) -> Tensor:
        """Return the fixed-effects prediction used for genuinely new groups."""
        self.eval()
        with torch.no_grad():
            return self.class_probabilities(self.fixed_head(self.fixed(features)))
