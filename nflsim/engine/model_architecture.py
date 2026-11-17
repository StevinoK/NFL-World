"""Configurable network architectures shared by training and simulation runtime.

Parameter names (fc1..fcN, drop1..dropN, out, prior_encoder.hidden/mu/logvar) are kept
identical to the original hardcoded 256/128/64 models so pre-existing checkpoints load
unchanged when the default architecture is used.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from nflsim.engine.ngmm_model import NGMMArchitecture, NGMMModel


ARCH_SIDECAR_SCHEMA_VERSION = 1

# Training model name -> artifact filename prefix used on disk and by the runtime.
MODEL_ARTIFACT_PREFIX: Dict[str, str] = {
    "yac": "yac",
    "air_yards": "ay",
    "sack": "sack",
    "scramble": "scramble_classifier",
    "interception": "int",
    "completion": "comp",
    "punt": "punt",
    "time_elapsed": "time",
    "kick": "kick",
    "rush_yards": "rush",
    "scramble_yards": "scramble_yards",
    "play_type": "play",
    "target_selection": "target",
    "rusher_selection": "rusher",
    "fumble": "fumble",
    "defensive_return_yards": "def_ret",
    "penalty_no_play": "penalty_no_play",
    "penalty_on_play": "penalty_on_play",
    "penalty_yards": "penalty_yards",
    "penalty_first_down": "penalty_first_down",
}

DEFAULT_FLAT_ARCH: Dict[str, Any] = {
    "hidden_dims": [256, 128, 64],
    "dropout": [0.3, 0.3, 0.3],
    "activation": "leaky_relu",
    "activation_negative_slope": 0.1,
    "norm": "none",
}

DEFAULT_HIERARCHICAL_ARCH: Dict[str, Any] = {
    "hidden_dims": [256, 128, 64],
    "dropout": [0.1, 0.1, 0.1],
    "activation": "leaky_relu",
    "activation_negative_slope": 0.1,
    "norm": "none",
    "prior_hidden_dims": [64, 64],
    "prior_activation": "relu",
}


def default_arch(kind: str) -> Dict[str, Any]:
    base = DEFAULT_HIERARCHICAL_ARCH if kind == "hierarchical" else DEFAULT_FLAT_ARCH
    return copy.deepcopy(base)


def normalize_arch(arch: Optional[Dict[str, Any]], kind: str) -> Dict[str, Any]:
    """Fill in defaults and broadcast a scalar dropout across all hidden layers."""
    resolved = default_arch(kind)
    if arch:
        resolved.update({key: value for key, value in arch.items() if value is not None})

    hidden_dims = [int(dim) for dim in resolved["hidden_dims"]]
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one layer")
    resolved["hidden_dims"] = hidden_dims

    dropout = resolved["dropout"]
    if isinstance(dropout, (int, float)):
        dropout = [float(dropout)] * len(hidden_dims)
    else:
        dropout = [float(value) for value in dropout]
    if len(dropout) != len(hidden_dims):
        raise ValueError(f"dropout length {len(dropout)} != hidden_dims length {len(hidden_dims)}")
    resolved["dropout"] = dropout

    if kind == "hierarchical":
        resolved["prior_hidden_dims"] = [int(dim) for dim in resolved["prior_hidden_dims"]]
    return resolved


def build_activation(name: str, negative_slope: float = 0.1) -> nn.Module:
    key = str(name).lower()
    if key == "leaky_relu":
        return nn.LeakyReLU(negative_slope=float(negative_slope))
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _build_norm(name: str, width: int) -> nn.Module:
    key = str(name).lower()
    if key == "batchnorm":
        return nn.BatchNorm1d(width)
    if key == "layernorm":
        return nn.LayerNorm(width)
    raise ValueError(f"Unsupported norm: {name}")


def _attach_trunk(module: nn.Module, arch: Dict[str, Any], input_dim: int, output_dim: int) -> None:
    """Assign fc1..fcN / norm1..normN / drop1..dropN / out onto `module`."""
    hidden_dims: List[int] = arch["hidden_dims"]
    dropout: List[float] = arch["dropout"]

    module.n_layers = len(hidden_dims)
    module.act = build_activation(arch["activation"], arch["activation_negative_slope"])
    module.use_norm = str(arch["norm"]).lower() not in {"none", ""}

    previous = int(input_dim)
    for index, (width, rate) in enumerate(zip(hidden_dims, dropout), start=1):
        setattr(module, f"fc{index}", nn.Linear(previous, width))
        if module.use_norm:
            setattr(module, f"norm{index}", _build_norm(arch["norm"], width))
        setattr(module, f"drop{index}", nn.Dropout(rate))
        previous = width
    module.out = nn.Linear(previous, int(output_dim))


def _forward_trunk(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    for index in range(1, module.n_layers + 1):
        x = module.act(getattr(module, f"fc{index}")(x))
        if module.use_norm:
            x = getattr(module, f"norm{index}")(x)
        x = getattr(module, f"drop{index}")(x)
    return module.out(x)


class ConfigurableMLP(nn.Module):
    """Feed-forward stack exposing fc1..fcN / drop1..dropN / out parameter names."""

    def __init__(self, input_dim: int, output_dim: int, arch: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.arch = normalize_arch(arch, "flat")
        _attach_trunk(self, self.arch, input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _forward_trunk(self, x)


class GameModel(ConfigurableMLP):
    def __init__(self, input_dim: int, output_dim: int, arch: Optional[Dict[str, Any]] = None):
        super().__init__(input_dim, output_dim, arch)


class CornModel(ConfigurableMLP):
    """CORN ordinal head: emits (num_classes - 1) cumulative logits."""

    def __init__(self, input_dim: int, output_dim: int, arch: Optional[Dict[str, Any]] = None):
        super().__init__(input_dim, int(output_dim) - 1, arch)


class PriorEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        activation: str = "relu",
    ):
        super().__init__()
        layers: List[nn.Module] = []
        previous = int(input_dim)
        for width in hidden_dims:
            layers.append(nn.Linear(previous, int(width)))
            layers.append(build_activation(activation))
            previous = int(width)
        self.hidden = nn.Sequential(*layers)
        self.mu = nn.Linear(previous, int(latent_dim))
        self.logvar = nn.Linear(previous, int(latent_dim))

    def forward(self, x: torch.Tensor):
        hidden = self.hidden(x)
        return self.mu(hidden), self.logvar(hidden)


class HierarchicalLatentGameModel(nn.Module):
    def __init__(
        self,
        play_input_dim: int,
        prior_input_dim: int,
        latent_dim: int,
        output_dim: int,
        arch: Optional[Dict[str, Any]] = None,
        ordinal_output: bool = False,
    ):
        super().__init__()
        self.arch = normalize_arch(arch, "hierarchical")
        self.latent_dim = int(latent_dim)
        self.ordinal_output = bool(ordinal_output)
        self.prior_encoder = PriorEncoder(
            prior_input_dim,
            self.latent_dim,
            hidden_dims=self.arch["prior_hidden_dims"],
            activation=self.arch["prior_activation"],
        )
        head_output = int(output_dim) - 1 if ordinal_output else int(output_dim)
        _attach_trunk(self, self.arch, int(play_input_dim) + self.latent_dim, head_output)

    def forward_with_z(self, play_x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return _forward_trunk(self, torch.cat([play_x, z], dim=1))

    def sample_z(self, prior_x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.prior_encoder(prior_x)
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, play_x: torch.Tensor, prior_x: torch.Tensor, sample: bool = True):
        mu, logvar = self.prior_encoder(prior_x)
        if sample and self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        logits = self.forward_with_z(play_x, z)
        kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return logits, kl_div


def build_network(kind: str, dims: Dict[str, Any], arch: Optional[Dict[str, Any]] = None) -> nn.Module:
    if kind == "ngmm":
        architecture = NGMMArchitecture(
            hidden_dims=tuple(int(value) for value in (arch or {}).get("hidden_dims", [128, 64, 32])),
            context_hidden_dims=tuple(int(value) for value in (arch or {}).get("context_hidden_dims", [64, 32])),
            latent_dim=int(dims.get("latent_dim", 1)),
            dropout=float((arch or {}).get("dropout", 0.1) if isinstance((arch or {}).get("dropout", 0.1), (int, float)) else (arch or {}).get("dropout", [0.1])[0]),
            target_mode=str(dims.get("target_mode", "multiclass")),
            output_dim=int(dims["output_dim"]),
        )
        return NGMMModel(int(dims["input_dim"]), int(dims["context_dim"]), architecture)
    if kind == "hierarchical":
        return HierarchicalLatentGameModel(
            play_input_dim=int(dims["play_input_dim"]),
            prior_input_dim=int(dims["prior_input_dim"]),
            latent_dim=int(dims["latent_dim"]),
            output_dim=int(dims["output_dim"]),
            arch=arch,
            ordinal_output=bool(dims.get("ordinal_output", False)),
        )
    if kind == "corn":
        return CornModel(int(dims["input_dim"]), int(dims["output_dim"]), arch)
    if kind == "flat":
        return GameModel(int(dims["input_dim"]), int(dims["output_dim"]), arch)
    raise ValueError(f"Unsupported network kind: {kind}")


def infer_hidden_dims_from_state_dict(state_dict: Dict[str, Any]) -> List[int]:
    """Recover hidden widths from an arch-sidecar-less checkpoint."""
    widths: List[int] = []
    index = 1
    while f"fc{index}.weight" in state_dict:
        widths.append(int(state_dict[f"fc{index}.weight"].shape[0]))
        index += 1
    return widths


def infer_arch_from_state_dict(state_dict: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """Best-effort architecture for legacy checkpoints with no sidecar.

    Dropout and activation are unrecoverable from weights; dropout is inert under eval()
    and the defaults match every checkpoint produced before arch sidecars existed.
    """
    arch = default_arch(kind)
    hidden_dims = infer_hidden_dims_from_state_dict(state_dict)
    if hidden_dims:
        arch["hidden_dims"] = hidden_dims
        arch["dropout"] = [arch["dropout"][0]] * len(hidden_dims)
    if "norm1.weight" in state_dict:
        arch["norm"] = "batchnorm" if "norm1.running_mean" in state_dict else "layernorm"
    else:
        arch["norm"] = "none"
    if kind == "hierarchical":
        prior_dims: List[int] = []
        layer_index = 0
        while f"prior_encoder.hidden.{layer_index}.weight" in state_dict:
            prior_dims.append(int(state_dict[f"prior_encoder.hidden.{layer_index}.weight"].shape[0]))
            layer_index += 2
        if prior_dims:
            arch["prior_hidden_dims"] = prior_dims
    return normalize_arch(arch, kind)


def arch_sidecar_path(data_dir, artifact_prefix: str) -> Path:
    return Path(data_dir) / f"{artifact_prefix}_arch.json"


def write_arch_sidecar(path, payload: Dict[str, Any]) -> None:
    body = dict(payload)
    body.setdefault("schema_version", ARCH_SIDECAR_SCHEMA_VERSION)
    Path(path).write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")


def read_arch_sidecar(path) -> Optional[Dict[str, Any]]:
    sidecar = Path(path)
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict) or "kind" not in payload:
        return None
    return payload
