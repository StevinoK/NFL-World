"""Fit and test shared game-level random effects for the play models.

The play models are fit as if plays were independent. Real games are not: a
team-game's plays share an unobserved "day" (execution, game plan, officiating,
weather the features miss). This module estimates that variance as a
generalized linear mixed model on top of the trained networks:

    logit_rm = f_m(x_r) + b_m + G_m u_game + T_m u_team + s_m * v_team,m

with u, v ~ N(0, I) (see nflsim.engine.game_effects). The networks f_m are held
fixed; b, G, T and s are fit by maximizing the Laplace approximation to each
game's marginal likelihood, using only games every network held out
(nflsim.models.train.holdout_games). Integrating the effects out, rather than
plugging in an observed game residual, avoids label leakage, separates
between-game variance from within-game sampling noise, matches what the
simulator draws, and lets one draw move every model together.

    python -m nflsim.models.game_effects --model-dir <refit dir> --output <dir>
"""

from __future__ import annotations

import argparse
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from nflsim.engine.game_effects import FILE_NAME, head_width
from nflsim.engine.model_architecture import build_network
from nflsim.engine.simulation_runtime import SKILL_SLOT_ORDER
from nflsim.features.goal_line import yards_available
from nflsim.features.training_data import prepare_frame
from nflsim.models.train import holdout_games
from nflsim.paths import ROOT, write_json
from nflsim.semantics import ISOLATED_MU_CONTEXT_V2

INPUT_CLIP = 10.0
DTYPE = torch.float64


# ── rows ────────────────────────────────────────────────────────────────────


@dataclass
class ModelRows:
    model: str
    head: str            # binary | corn | multiclass
    width: int           # raw network output width
    base: np.ndarray     # (N, L) base logits: binary eta, CORN cumulative logits, multiclass logits
    label: np.ndarray    # (N,) class index
    values: np.ndarray | None  # CORN bin values (yards/seconds) for dispersion checks
    games: np.ndarray    # (N,) game_id
    teams: np.ndarray    # (N,) posteam
    limit: np.ndarray | None = None  # (N,) yards before the goal line, on models trained with capped rows


def save_rows(rows: dict, path: Path, extra=None) -> None:
    """Plain-dict pickles, so rows load from any entry point (not tied to __main__)."""
    joblib.dump({"rows": {model: vars(item) for model, item in rows.items()}, "extra": extra}, path)


def load_rows(path: Path):
    stored = joblib.load(path)
    return {model: ModelRows(**item) for model, item in stored["rows"].items()}, stored["extra"]


def load_network(directory: Path, model: str, prefix: str):
    sidecar = json.loads((directory / f"{prefix}_arch.json").read_text())
    state = torch.load(directory / f"{prefix}_model.pt", map_location="cpu", weights_only=True)
    # Widths from the checkpoint (older sidecars count CORN outputs differently); CORN heads emit K-1 logits.
    width = int(state["out.weight"].shape[0])
    dims = {"input_dim": int(state["fc1.weight"].shape[1]), "output_dim": width + (1 if sidecar["kind"] == "corn" else 0)}
    network = build_network(sidecar["kind"], dims, sidecar.get("arch"))
    network.load_state_dict(state)
    network.eval().float()
    processor = joblib.load(directory / f"{prefix}_preprocessor.pkl")
    bins = None
    if (directory / f"{prefix}_bins.csv").is_file():
        bins = pd.read_csv(directory / f"{prefix}_bins.csv")["0"].to_numpy(dtype=float)
    return sidecar, network, processor, bins


def transform(processor, frame: pd.DataFrame) -> torch.Tensor:
    values = processor.transform(frame[list(processor.feature_names_in_)])
    values = np.clip(np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0), -INPUT_CLIP, INPUT_CLIP)
    return torch.from_numpy(values)


def forward(network, inputs: torch.Tensor, batch: int = 65536) -> np.ndarray:
    with torch.no_grad():
        return np.concatenate([network(inputs[start:start + batch]).double().numpy() for start in range(0, len(inputs), batch)])


def encode_labels(model: str, sidecar: dict, target: pd.Series, bins: np.ndarray | None) -> np.ndarray:
    if bins is not None:
        values = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
        return np.abs(values[:, None] - bins[None, :]).argmin(axis=1)
    labels = sidecar["dims"].get("class_labels")
    if labels:
        return pd.Categorical(target.astype(str), categories=labels).codes.astype(np.int64)
    return pd.to_numeric(target).to_numpy(dtype=np.int64)


def model_rows(directory: Path, specs: dict, prepared: pd.DataFrame, players, roster, keep=None) -> dict[str, ModelRows]:
    """Base logits from the trained flat/CORN networks for every model's rows (optionally filtered by game)."""
    from nflsim.models.fit_minimal import model_rows as build_rows
    result = {}
    for model, spec in specs.items():
        frame, target, _ = build_rows(model, spec, prepared, players, roster)
        if keep is not None:
            mask = keep(frame.game_id.to_numpy())
            frame, target = frame.loc[mask].reset_index(drop=True), target.loc[mask].reset_index(drop=True)
        sidecar, network, processor, bins = load_network(directory, model, spec["prefix"])
        logits = forward(network, transform(processor, frame))
        if sidecar["kind"] == "corn":
            head, base = "corn", logits
        elif logits.shape[1] == 2:
            head, base = "binary", logits[:, 1:] - logits[:, :1]
        else:
            head, base = "multiclass", logits
        limit = yards_available(model, frame) if sidecar.get("goal_capped") else None
        result[model] = ModelRows(model, head, logits.shape[1], base, encode_labels(model, sidecar, target, bins),
                                  bins, frame.game_id.astype(str).to_numpy(), frame.posteam.astype(str).to_numpy(), limit)
        print(f"[rows] {model}: {len(frame)} rows head={head}", flush=True)
    return result


# ── mixed model ─────────────────────────────────────────────────────────────


def local_terms(head: str, base: torch.Tensor, label: torch.Tensor, delta: torch.Tensor, need_curvature: bool = True, capped: torch.Tensor | None = None):
    """Per-row log-likelihood plus its gradient and negative Hessian in the shift. A capped CORN row only went at least that far."""
    if head == "binary":
        eta = base[:, 0] + delta[:, 0]
        y = label.double()
        ll = y * F.logsigmoid(eta) + (1 - y) * F.logsigmoid(-eta)
        if not need_curvature:
            return ll, None, None
        p = torch.sigmoid(eta)
        return ll, (y - p)[:, None], (p * (1 - p))[:, None, None]
    if head == "corn":
        cumulative = base + delta
        steps = torch.arange(cumulative.shape[1])
        below = (steps[None, :] < label[:, None]).double()
        at = (steps[None, :] == label[:, None]).double()
        if capped is not None:
            at = at * (~capped[:, None]).double()
        ll = (below * F.logsigmoid(cumulative)).sum(1) + (at * F.logsigmoid(-cumulative)).sum(1)
        if not need_curvature:
            return ll, None, None
        p = torch.sigmoid(cumulative)
        gradient = (below * (1 - p)).sum(1) - (at * p).sum(1)
        weight = ((below + at) * p * (1 - p)).sum(1)
        return ll, gradient[:, None], weight[:, None, None]
    logp = torch.log_softmax(base + delta, dim=1)
    ll = logp.gather(1, label[:, None])[:, 0]
    if not need_curvature:
        return ll, None, None
    p = logp.exp()
    gradient = F.one_hot(label, p.shape[1]).double() - p
    weight = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]
    return ll, gradient, weight


class GameEffectModel(torch.nn.Module):
    def __init__(self, heads: dict[str, tuple[str, int]], game_dims: int, team_dims: int, private: bool, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.models = list(heads)
        self.heads = heads
        self.game_dims, self.team_dims, self.private = int(game_dims), int(team_dims), bool(private)
        self.bias = torch.nn.ParameterDict()
        self.game = torch.nn.ParameterDict()
        self.team = torch.nn.ParameterDict()
        self.log_scale = torch.nn.ParameterDict()
        offset = 0
        self.private_offset = {}
        for model, (head, width) in heads.items():
            size = head_width(head, width)
            self.bias[model] = torch.nn.Parameter(torch.zeros(size, dtype=DTYPE))
            self.game[model] = torch.nn.Parameter(0.05 * torch.randn(size, self.game_dims, generator=generator, dtype=DTYPE))
            self.team[model] = torch.nn.Parameter(0.05 * torch.randn(size, self.team_dims, generator=generator, dtype=DTYPE))
            self.log_scale[model] = torch.nn.Parameter(torch.full((size,), np.log(0.1), dtype=DTYPE))
            self.private_offset[model] = offset
            offset += size if self.private else 0
        self.private_dims = offset
        self.side_dims = self.team_dims + self.private_dims
        self.latent_dims = self.game_dims + 2 * self.side_dims

    def columns(self, model: str) -> torch.Tensor:
        """Latent indices each side's rows of a model load on: game block, that side's team block, its private block."""
        head, width = self.heads[model]
        size = head_width(head, width)
        result = []
        for side in (0, 1):
            start = self.game_dims + side * self.side_dims
            columns = list(range(self.game_dims)) + list(range(start, start + self.team_dims))
            if self.private:
                first = start + self.team_dims + self.private_offset[model]
                columns += list(range(first, first + size))
            result.append(columns)
        return torch.tensor(result, dtype=torch.long).reshape(2, -1)

    def loading(self, model: str) -> torch.Tensor:
        blocks = [self.game[model], self.team[model]]
        if self.private:
            blocks.append(torch.diag(self.log_scale[model].exp()))
        return torch.cat(blocks, dim=1)

    def export(self) -> dict:
        models = {}
        for model, (head, width) in self.heads.items():
            size = head_width(head, width)
            models[model] = {
                "head": head, "width": int(width),
                "bias": self.bias[model].detach().tolist(),
                "game_loadings": self.game[model].detach().tolist(),
                "team_loadings": self.team[model].detach().tolist(),
                "private_scale": (self.log_scale[model].exp().detach() if self.private else torch.zeros(size, dtype=DTYPE)).tolist(),
            }
        return {"version": 1, "game_dims": self.game_dims, "team_dims": self.team_dims, "models": models}


class Data:
    """Row tensors indexed by game (latent block) and side (home/away within the game)."""

    def __init__(self, rows: dict[str, ModelRows], games: list[str] | None = None):
        all_games = sorted(set().union(*(set(item.games) for item in rows.values()))) if games is None else games
        self.games = all_games
        lookup = {game: index for index, game in enumerate(all_games)}
        # side 0 = alphabetically first team in the game across every model; loadings do not depend on side
        first_team = pd.concat([pd.Series(item.teams, index=item.games) for item in rows.values()]).groupby(level=0).min()
        self.items = {}
        for model, item in rows.items():
            keep = np.array([game in lookup for game in item.games])
            game_index = np.array([lookup[game] for game in item.games[keep]], dtype=np.int64)
            teams = item.teams[keep]
            side = (teams != first_team.reindex(item.games[keep]).to_numpy()).astype(np.int64)
            self.items[model] = {
                "head": item.head, "width": item.width,
                "base": torch.from_numpy(item.base[keep]).double(),
                "label": torch.from_numpy(item.label[keep]).long(),
                "game": torch.from_numpy(game_index).long(), "side": torch.from_numpy(side).long(),
                "values": item.values,
            }
            if item.limit is not None:
                self.items[model]["limit"] = torch.from_numpy(item.limit[keep]).double()
                self.items[model]["capped"] = torch.from_numpy(item.values[item.label[keep]] >= item.limit[keep])
        self.rows = sum(len(item["label"]) for item in self.items.values())

    def subset(self, games: list[str]) -> "Data":
        result = Data.__new__(Data)
        lookup = {game: index for index, game in enumerate(games)}
        old = np.array([lookup.get(game, -1) for game in self.games])
        result.games, result.items = games, {}
        for model, item in self.items.items():
            mapped = torch.from_numpy(old)[item["game"]]
            keep = mapped >= 0
            result.items[model] = {**item, **{key: item[key][keep] for key in ("base", "label", "side", "limit", "capped") if key in item}, "game": mapped[keep]}
        result.rows = sum(len(item["label"]) for item in result.items.values())
        return result


def shifts(effects: GameEffectModel, item: dict, model: str, z: torch.Tensor) -> torch.Tensor:
    columns = effects.columns(model)
    selected = z[item["game"][:, None], columns[item["side"]]]
    return effects.bias[model] + selected @ effects.loading(model).T


def laplace_terms(effects: GameEffectModel, data: Data, z: torch.Tensor, curvature: bool = True):
    """Joint log-likelihood per game, the latent gradient and the posterior precision at z."""
    games = len(data.games)
    loglik = torch.zeros(games, dtype=DTYPE)
    gradient = -z.clone()
    precision = torch.eye(effects.latent_dims, dtype=DTYPE).repeat(games, 1, 1)
    for model, item in data.items.items():
        ll, local_gradient, weight = local_terms(item["head"], item["base"], item["label"], shifts(effects, item, model, z), curvature, item.get("capped"))
        loglik = loglik.index_add(0, item["game"], ll)
        if not curvature or effects.latent_dims == 0:
            continue
        loading = effects.loading(model)
        columns = effects.columns(model)
        group = item["game"] * 2 + item["side"]
        size = loading.shape[0]
        gradient_sum = torch.zeros(games * 2, size, dtype=DTYPE).index_add(0, group, local_gradient).view(games, 2, size)
        weight_sum = torch.zeros(games * 2, size, size, dtype=DTYPE).index_add(0, group, weight).view(games, 2, size, size)
        batch = torch.arange(games)[:, None, None]
        for side in (0, 1):
            index = columns[side]
            gradient = gradient.index_put((torch.arange(games)[:, None], index[None, :]), gradient_sum[:, side] @ loading, accumulate=True)
            block = loading.T @ weight_sum[:, side] @ loading
            precision = precision.index_put((batch, index[None, :, None], index[None, None, :]), block, accumulate=True)
    return loglik, gradient, precision


def find_modes(effects: GameEffectModel, data: Data, z: torch.Tensor | None = None, steps: int = 8) -> torch.Tensor:
    z = torch.zeros(len(data.games), effects.latent_dims, dtype=DTYPE) if z is None else z
    if effects.latent_dims == 0:
        return z
    with torch.no_grad():
        loglik, gradient, precision = laplace_terms(effects, data, z)
        objective = loglik - 0.5 * (z ** 2).sum(1)
        for _ in range(steps):
            step = torch.linalg.solve(precision, gradient[..., None])[..., 0]
            # Newton with per-game step halving: the posterior is log-concave, but a full step can overshoot.
            size = torch.ones(len(data.games), 1, dtype=DTYPE)
            for _ in range(12):
                candidate = z + size * step
                candidate_objective = laplace_terms(effects, data, candidate, curvature=False)[0] - 0.5 * (candidate ** 2).sum(1)
                worse = candidate_objective < objective - 1e-9
                if not worse.any():
                    break
                size = torch.where(worse[:, None], size / 2, size)
            keep = candidate_objective >= objective - 1e-9
            z = torch.where(keep[:, None], candidate, z)
            loglik, gradient, precision = laplace_terms(effects, data, z)
            objective = loglik - 0.5 * (z ** 2).sum(1)
    return z


def laplace_marginal(effects: GameEffectModel, data: Data, z: torch.Tensor) -> torch.Tensor:
    loglik, _, precision = laplace_terms(effects, data, z)
    if effects.latent_dims == 0:
        return loglik
    logdet = 2 * torch.log(torch.diagonal(torch.linalg.cholesky(precision), dim1=1, dim2=2)).sum(1)
    return loglik - 0.5 * (z ** 2).sum(1) - 0.5 * logdet


def fit(data: Data, game_dims: int, team_dims: int, private: bool, iterations: int = 250, seed: int = 0,
        learning_rate: float = 0.03, penalty: float = 1e-3, log_every: int = 25) -> GameEffectModel:
    heads = {model: (item["head"], item["width"]) for model, item in data.items.items()}
    effects = GameEffectModel(heads, game_dims, team_dims, private, seed)
    optimizer = torch.optim.Adam(effects.parameters(), lr=learning_rate)
    z = None
    for iteration in range(iterations):
        z = find_modes(effects, data, z, steps=8 if iteration == 0 else 2)
        optimizer.zero_grad()
        objective = laplace_marginal(effects, data, z.detach()).sum()
        regularizer = sum((parameter ** 2).sum() for name, parameter in effects.named_parameters() if not name.startswith("log_scale"))
        loss = -(objective / data.rows) + penalty * regularizer / len(data.games)
        loss.backward()
        optimizer.step()
        if log_every and (iteration % log_every == 0 or iteration == iterations - 1):
            print(f"[fit] g{game_dims} t{team_dims} p{int(private)} iter {iteration}: marginal/game {float(objective) / len(data.games):.3f}", flush=True)
    return effects


def importance_marginal(effects: GameEffectModel, data: Data, draws: int = 64, seed: int = 1) -> np.ndarray:
    """Per-game log marginal likelihood by importance sampling from the Laplace posterior (fixed parameters)."""
    with torch.no_grad():
        if effects.latent_dims == 0:
            return laplace_terms(effects, data, torch.zeros(len(data.games), 0, dtype=DTYPE), curvature=False)[0].numpy()
        z = find_modes(effects, data, steps=12)
        _, _, precision = laplace_terms(effects, data, z)
        chol = torch.linalg.cholesky(precision)
        logdet = 2 * torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum(1)
        generator = torch.Generator().manual_seed(seed)
        weights = []
        for _ in range(draws):
            noise = torch.randn(z.shape, generator=generator, dtype=DTYPE)
            # z_s = z + L^{-T} eps has covariance precision^{-1}
            sample = z + torch.linalg.solve_triangular(chol.transpose(1, 2), noise[..., None], upper=True)[..., 0]
            loglik = laplace_terms(effects, data, sample, curvature=False)[0]
            weights.append(loglik - 0.5 * (sample ** 2).sum(1) + 0.5 * (noise ** 2).sum(1) - 0.5 * logdet)
        stacked = torch.stack(weights)
        return (torch.logsumexp(stacked, 0) - np.log(draws)).numpy()


# ── dispersion checks ───────────────────────────────────────────────────────

# (stat, model, class index for multiclass/binary positive, or None for CORN yardage/seconds)
STATS = [
    ("completions", "comp", 1), ("sacks", "sack", 1), ("interceptions", "int", 1), ("scrambles", "scramble", 1),
    ("pass_plays", "play", 1), ("rush_yards", "rush", None), ("air_yards", "ay", None), ("yac", "yac", None),
    ("sack_yards", "sack_yards", None), ("penalties", "penalty_on_play", 1), ("play_seconds", "time", None),
    ("fg_xp_made", "kick", 1), ("wr1_targets", "target", SKILL_SLOT_ORDER.index("WR1")),
    ("rb1_carries", "rusher", SKILL_SLOT_ORDER.index("RB1")),
]


def row_moments(head: str, logits: torch.Tensor, label: torch.Tensor, cls, values, limit=None):
    """Per-row mean and variance of a stat's contribution, and its observed value."""
    if head == "binary":
        p = torch.sigmoid(logits[:, 0])
        return p, p * (1 - p), label.double()
    if head == "multiclass":
        p = torch.softmax(logits, 1)[:, cls]
        return p, p * (1 - p), (label == cls).double()
    survival = torch.cumprod(torch.sigmoid(logits), 1)
    probabilities = torch.cat([1 - survival[:, :1], survival[:, :-1] - survival[:, 1:], survival[:, -1:]], 1)
    support = torch.from_numpy(np.asarray(values, dtype=float))
    observed = support[label]
    if limit is not None:  # A draw past the goal line counts as reaching it, as it does in the sim.
        support = torch.minimum(support[None, :], limit[:, None])
        mean = (probabilities * support).sum(1)
        return mean, (probabilities * support ** 2).sum(1) - mean ** 2, observed
    mean = probabilities @ support
    return mean, probabilities @ support ** 2 - mean ** 2, observed


def stat_draws(data: Data, logit_draws, draws: int):
    """Team-game stat moments under `draws` sampled effect sets.

    logit_draws(model, item, draw) -> raw head logits for that draw. Returns observed totals and
    per-draw conditional means/variances, each (stats, team-games[, draws]).
    """
    groups = len(data.games) * 2
    names, observed, means, variances = [], [], [], []
    for stat, model, cls in STATS:
        item = data.items.get(model)
        if item is None:
            continue
        group = item["game"] * 2 + item["side"]
        mean_draws, variance_draws = torch.zeros(groups, draws, dtype=DTYPE), torch.zeros(groups, draws, dtype=DTYPE)
        for draw in range(draws):
            mean, variance, obs = row_moments(item["head"], logit_draws(model, item, draw), item["label"], cls, item["values"], item.get("limit"))
            mean_draws[:, draw] = torch.zeros(groups, dtype=DTYPE).index_add(0, group, mean)
            variance_draws[:, draw] = torch.zeros(groups, dtype=DTYPE).index_add(0, group, variance)
        names.append(stat)
        observed.append(torch.zeros(groups, dtype=DTYPE).index_add(0, group, obs))
        means.append(mean_draws)
        variances.append(variance_draws)
    present = torch.zeros(groups, dtype=torch.bool)
    for item in data.items.values():
        present[item["game"] * 2 + item["side"]] = True
    return names, torch.stack(observed)[:, present], torch.stack(means)[:, present], torch.stack(variances)[:, present]


def dispersion_report(names, observed, means, variances) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expected = means.mean(2)
    total_variance = variances.mean(2) + means.var(2, unbiased=False)
    standardized = (observed - expected) / total_variance.clamp_min(1e-9).sqrt()
    table = pd.DataFrame({
        "stat": names,
        "variance_ratio": (standardized ** 2).mean(1).numpy(),
        "between_game_share": (means.var(2, unbiased=False) / total_variance.clamp_min(1e-9)).mean(1).numpy(),
        "mean_observed": observed.mean(1).numpy(),
        "mean_expected": expected.mean(1).numpy(),
    })
    centered = means - means.mean(2, keepdim=True)
    implied_cov = torch.einsum("agd,bgd->abg", centered, centered) / (means.shape[2] - 1)
    scale = total_variance.sqrt()
    implied = (implied_cov / (scale[:, None] * scale[None, :])).mean(2).numpy()
    np.fill_diagonal(implied, 1.0)
    actual = np.corrcoef(standardized.numpy())
    return table, pd.DataFrame(implied, index=names, columns=names), pd.DataFrame(actual, index=names, columns=names)


def effect_logits(effects: GameEffectModel | None, draws: int, games: int, seed: int = 2):
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(draws, games, effects.latent_dims, generator=generator, dtype=DTYPE) if effects is not None else None

    def logits(model, item, draw):
        if effects is None:
            return item["base"]
        with torch.no_grad():
            return item["base"] + shifts(effects, item, model, latent[draw])
    return logits


# ── current cascade bundle, for comparison ──────────────────────────────────


def cascade_rows(directory: Path, specs: dict, prepared: pd.DataFrame, players, roster):
    """Rows for a bundle with cascades: flat/CORN rows as usual, cascades with their stage-2 residual sampler."""
    from nflsim.engine.cascade_trainer import load_cascade_artifacts
    from nflsim.models.fit_minimal import model_rows as build_rows
    flat_specs = {model: spec for model, spec in specs.items() if spec["sidecar"]["kind"] != "cascade"}
    rows = model_rows(directory, flat_specs, prepared, players, roster)
    samplers = {}
    for model, spec in specs.items():
        if spec["sidecar"]["kind"] != "cascade":
            continue
        frame, target, _ = build_rows(model, spec, prepared, players, roster)
        artifacts = load_cascade_artifacts(directory, model)
        frame["residual_com"] = 0.0
        processor = artifacts.stage3_preprocessor
        inputs = transform(processor, frame)
        probe = frame.iloc[:1].copy()
        at_zero = transform(processor, probe)[0]
        probe["residual_com"] = 1.0
        at_one = transform(processor, probe)[0]
        column = int(torch.argmax((at_one - at_zero).abs()))
        keys = frame.game_id.astype(str) + "::" + frame.posteam.astype(str)
        context = frame[artifacts.context_columns].apply(pd.to_numeric, errors="coerce").groupby(keys.to_numpy()).median().fillna(0.0)
        scaled = np.clip(np.nan_to_num(artifacts.stage2_context_scaler.transform(context.to_numpy(dtype=np.float32))), -10, 10)
        with torch.no_grad():
            distribution = torch.softmax(artifacts.stage2_model(torch.from_numpy(scaled.astype(np.float32))), -1).double().numpy()
        labels = {"dims": {"class_labels": artifacts.class_labels}} if model == "play" else {"dims": {}}
        com = distribution @ artifacts.bin_centers
        row_group = context.index.get_indexer(keys)
        base_inputs = inputs.clone()
        base_inputs[:, column] = torch.clamp(float(at_zero[column]) + float(at_one[column] - at_zero[column]) * torch.from_numpy(com[row_group]).float(), -INPUT_CLIP, INPUT_CLIP)
        logits = forward(artifacts.stage3_model, base_inputs)
        head = "binary" if logits.shape[1] == 2 else "multiclass"
        base = logits[:, 1:] - logits[:, :1] if head == "binary" else logits
        rows[model] = ModelRows(model, head, logits.shape[1], base, encode_labels(model, labels, target, None), None,
                                frame.game_id.astype(str).to_numpy(), frame.posteam.astype(str).to_numpy())
        samplers[model] = {"inputs": inputs, "column": column, "intercept": float(at_zero[column]),
                           "slope": float(at_one[column] - at_zero[column]), "distribution": distribution,
                           "edges": artifacts.bin_edges, "row_group": row_group, "net": artifacts.stage3_model, "head": head}
        print(f"[rows] {model}: {len(frame)} cascade rows", flush=True)
    return rows, samplers


def cascade_logits(samplers: dict, draws: int, seed: int = 3):
    """Per-draw logits with each cascade's residual sampled per team-game, as the simulator does."""
    rng = np.random.default_rng(seed)
    residuals = {}
    for model, sampler in samplers.items():
        cdf = np.cumsum(sampler["distribution"], 1)
        cdf /= cdf[:, -1:]
        u = rng.uniform(size=(cdf.shape[0], draws))
        index = np.minimum((cdf[:, :, None] <= u[:, None, :]).sum(1), cdf.shape[1] - 1)
        previous = np.where(index > 0, np.take_along_axis(cdf, np.maximum(index - 1, 0), 1), 0.0)
        mass = np.take_along_axis(sampler["distribution"], index, 1)
        fraction = np.clip(np.where(mass > 0, (u - previous) / np.maximum(mass, 1e-12), 0.5), 0, 1)
        edges = sampler["edges"]
        residuals[model] = edges[index] + fraction * (edges[index + 1] - edges[index])

    def logits(model, item, draw):
        sampler = samplers.get(model)
        if sampler is None:
            return item["base"]
        inputs = sampler["inputs"].clone()
        value = sampler["intercept"] + sampler["slope"] * residuals[model][sampler["row_group"], draw]
        inputs[:, sampler["column"]] = torch.clamp(torch.from_numpy(value).float(), -INPUT_CLIP, INPUT_CLIP)
        out = torch.from_numpy(forward(sampler["net"], inputs))
        return out[:, 1:] - out[:, :1] if sampler["head"] == "binary" else out
    return logits


def write_dispersion(output: Path, label: str, data: Data, logit_draws, draws: int) -> pd.DataFrame:
    names, observed, means, variances = stat_draws(data, logit_draws, draws)
    table, implied, actual = dispersion_report(names, observed, means, variances)
    table.to_csv(output / f"dispersion_{label}.csv", index=False)
    implied.to_csv(output / f"correlation_{label}_implied.csv")
    actual.to_csv(output / f"correlation_{label}_observed.csv")
    print(f"[dispersion] {label}\n{table.to_string(index=False)}", flush=True)
    return table


def play_log_loss(data: Data, effects: GameEffectModel | None = None) -> pd.Series:
    """Mean per-play log loss (no game effects; the fitted bias when given)."""
    result = {}
    for model, item in data.items.items():
        delta = effects.bias[model].detach()[None, :] if effects is not None else torch.zeros(1, item["base"].shape[1] if item["head"] == "multiclass" else 1, dtype=DTYPE)
        ll = local_terms(item["head"], item["base"], item["label"], delta.expand(len(item["label"]), -1), False, item.get("capped"))[0]
        result[model] = float(-ll.mean())
    return pd.Series(result)


# ── CLI ─────────────────────────────────────────────────────────────────────

CONFIGS = {
    "independent": (0, 0, False),
    "per_model": (0, 0, True),
    "team1": (0, 1, True),
    "team2": (0, 2, True),
    "team3": (0, 3, True),
    "game1_team2": (1, 2, True),
    "game1_team3": (1, 3, True),
    "game2_team4": (2, 4, True),
}


def fold_of(game: str) -> int:
    return zlib.crc32(f"fold:{game}".encode()) % 2




def load_inputs(tables: Path, seasons, player_table: Path | None = None, team_table: Path | None = None,
                inputs: Path = ROOT / "artifacts/inputs"):
    players, teams = [pd.read_csv(path, dtype={"game_id": str, "player_id": str})
                      for path in (player_table or tables / "player_stats_wide.csv", team_table or tables / "team_stats_wide.csv")]
    pbp = pd.read_parquet(inputs / "pbp.parquet")
    roster = pd.read_parquet(inputs / "roster.parquet")
    return prepare_frame(pbp, players, teams, ISOLATED_MU_CONTEXT_V2, seasons=seasons), players, roster


def refit_specs(model_dir: Path) -> dict:
    from nflsim.models.fit_minimal import REFERENCE, game_effects_specs, reduced_spec
    reference = json.loads((REFERENCE / "model_manifest.json").read_text())
    specs = game_effects_specs({model: reduced_spec(model, spec) for model, spec in reference["models"].items()}, reference)
    trained = json.loads((model_dir / "reduced_model_manifest.json").read_text())["models"]
    return {model: spec for model, spec in specs.items() if model in trained}


def stage_rows(args):
    from nflsim.models.fit_minimal import REFERENCE
    specs = refit_specs(args.model_dir)
    if not (args.output / "holdout_rows.pkl").is_file():
        prepared, players, roster = load_inputs(REFERENCE, tuple(args.seasons), args.player_table, args.team_table, args.inputs)
        save_rows(model_rows(args.model_dir, specs, prepared, players, roster, keep=holdout_games), args.output / "holdout_rows.pkl")
    season = (args.eval_season, args.eval_season)
    if not (args.output / "eval_rows_new.pkl").is_file():
        prepared, players, roster = load_inputs(args.compare_bundle, season, args.player_table, args.team_table, args.inputs)
        save_rows(model_rows(args.model_dir, specs, prepared, players, roster), args.output / "eval_rows_new.pkl")
    prepared, players, roster = load_inputs(args.compare_bundle, season, inputs=args.inputs)
    old_specs = json.loads((args.compare_bundle / "model_manifest.json").read_text())["models"]
    rows, samplers = cascade_rows(args.compare_bundle, old_specs, prepared, players, roster)
    save_rows(rows, args.output / "eval_rows_old.pkl", samplers)


def stage_cv(args):
    data = Data(load_rows(args.output / "holdout_rows.pkl")[0])
    folds = [[game for game in data.games if fold_of(game) == fold] for fold in (0, 1)]
    for name in args.configs:
        game_dims, team_dims, private = CONFIGS[name]
        total = 0.0
        for fold in (0, 1):
            train, test = data.subset(folds[fold]), data.subset(folds[1 - fold])
            effects = fit(train, game_dims, team_dims, private, args.iterations, learning_rate=args.learning_rate, log_every=50)
            total += float(importance_marginal(effects, test).sum())
        record = {"config": name, "game_dims": game_dims, "team_dims": team_dims, "private": private,
                  "games": len(data.games), "heldout_log_marginal": total}
        write_json(args.output / f"cv_{name}.json", record)
        print(f"[cv] {name}: held-out log marginal {total:.1f} ({total / len(data.games):.3f}/game)", flush=True)


def stage_final(args):
    output, draws = args.output, args.draws
    data = Data(load_rows(output / "holdout_rows.pkl")[0])
    records = [json.loads(path.read_text()) for path in sorted(output.glob("cv_*.json"))]
    if records:
        cv = pd.DataFrame(records)
        base = cv.loc[cv.config.eq("independent"), "heldout_log_marginal"]
        if len(base):
            cv["gain_vs_independent"] = cv.heldout_log_marginal - float(base.iloc[0])
            cv["gain_per_game"] = cv.gain_vs_independent / cv.games
        cv.sort_values("heldout_log_marginal", ascending=False).to_csv(output / "cv_marginal_likelihood.csv", index=False)
        print(cv.to_string(index=False), flush=True)
    game_dims, team_dims, private = CONFIGS[args.choose]
    effects = fit(data, game_dims, team_dims, private, args.iterations, learning_rate=args.learning_rate)
    independent = fit(data, 0, 0, False, args.iterations, learning_rate=args.learning_rate, log_every=0)
    spec = effects.export()
    spec["fit"] = {"config": args.choose, "holdout_games": len(data.games), "rows": data.rows,
                   "seasons": list(args.seasons), "holdout": "crc32(game_id) % 5 == 0 (nflsim.models.train.holdout_games)"}
    write_json(output / FILE_NAME, spec)
    write_dispersion(output, "holdout_independent", data, effect_logits(None, draws, len(data.games)), draws)
    write_dispersion(output, f"holdout_{args.choose}", data, effect_logits(effects, draws, len(data.games)), draws)

    new = Data(load_rows(output / "eval_rows_new.pkl")[0])
    old_rows, samplers = load_rows(output / "eval_rows_old.pkl")
    old = Data(old_rows)
    marginal = {"independent": float(importance_marginal(independent, new).sum()),
                args.choose: float(importance_marginal(effects, new).sum())}
    season = args.eval_season
    write_json(output / f"eval_{season}_marginal.json", {**marginal, "games": len(new.games),
                                                          "gain_per_game": (marginal[args.choose] - marginal["independent"]) / len(new.games)})
    print(f"[{season}] log marginal {marginal}", flush=True)
    write_dispersion(output, f"{season}_current_cascade_bundle", old, cascade_logits(samplers, draws), draws)
    write_dispersion(output, f"{season}_new_independent", new, effect_logits(None, draws, len(new.games)), draws)
    write_dispersion(output, f"{season}_new_{args.choose}", new, effect_logits(effects, draws, len(new.games)), draws)
    losses = pd.DataFrame({"current_bundle": play_log_loss(old), "new_networks": play_log_loss(new),
                           "new_with_bias": play_log_loss(new, effects)})
    losses.to_csv(output / f"play_log_loss_{season}.csv")
    print(losses.to_string(), flush=True)


def main():
    from nflsim.paths import ROOT as SIM_ROOT
    parser = argparse.ArgumentParser(description="Fit/test shared game effects on held-out games")
    parser.add_argument("stage", choices=["rows", "cv", "final"])
    parser.add_argument("--model-dir", type=Path, required=True, help="Refit directory with flat/CORN networks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-bundle", type=Path,
                        default=SIM_ROOT / "artifacts/runtime")
    parser.add_argument("--player-table", type=Path,
                        help="rows: player_stats_wide.csv the refit was trained on (default: the reference bundle's)")
    parser.add_argument("--team-table", type=Path,
                        help="rows: team_stats_wide.csv the refit was trained on (default: the reference bundle's)")
    parser.add_argument("--seasons", type=int, nargs=2, default=(2017, 2024), metavar=("FIRST", "LAST"),
                        help="rows: seasons whose held-out fifth the effects are fit on")
    parser.add_argument("--eval-season", type=int, default=2025, help="Out-of-sample season scored against --compare-bundle")
    parser.add_argument("--inputs", type=Path, default=ROOT / "artifacts/inputs", help="Directory with pbp.parquet and roster.parquet")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--choose", default="game1_team2")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    {"rows": stage_rows, "cv": stage_cv, "final": stage_final}[args.stage](args)


if __name__ == "__main__":
    main()
