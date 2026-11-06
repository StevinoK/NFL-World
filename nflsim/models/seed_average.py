"""Fit one network to the averaged predictions of several seeds.

A network's team-level predictions move with its random seed about as much as with a real model change. Averaging seeds
shrinks that noise by the square root of the seed count, and fitting one network to the average keeps the sim's cost at
one network per decision.

    python -m nflsim.models.seed_average --base-dir <refit> --seed-dir <dir> [<dir> ...] --models comp int ... --output <dir>

The output starts as a copy of the base refit, so it packages like any other refit. The first seed directory's seed
names the fit. A seed directory only needs the models being averaged.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nflsim.engine.hyperparameters import build_optimizer, default_hyperparameters
from nflsim.engine.model_architecture import build_network
from nflsim.models.fit_minimal import REFERENCE, metric_record, model_rows
from nflsim.models.game_effects import encode_labels, forward, load_inputs, load_network, refit_specs, transform
from nflsim.models.train import holdout_split
from nflsim.paths import output_dir, write_json

PATIENCE = 12


def network_probabilities(kind: str, logits: np.ndarray) -> np.ndarray:
    """Class probabilities from a network's raw outputs."""
    if kind == "corn":
        onward = np.cumprod(1 / (1 + np.exp(-logits)), axis=1)
        return np.concatenate([1 - onward[:, :1], onward[:, :-1] - onward[:, 1:], onward[:, -1:]], axis=1)
    logits = logits - logits.max(axis=1, keepdims=True)
    return np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)


def soft_targets(kind: str, probabilities: np.ndarray) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Targets for the copy. A CORN step asks whether a play at this yard goes past it, and counts as often as plays get there."""
    if kind != "corn":
        return torch.from_numpy(probabilities.astype(np.float32)), None
    reach = 1 - np.concatenate([np.zeros((len(probabilities), 1)), np.cumsum(probabilities, axis=1)[:, :-1]], axis=1)
    reach = reach.clip(1e-7, 1).astype(np.float32)
    return torch.from_numpy((reach[:, 1:] / reach[:, :-1]).clip(0, 1)), torch.from_numpy(reach[:, :-1])


def soft_loss(kind: str, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    """Mismatch against the averaged probabilities, per row; lowest when the network reproduces them."""
    if kind == "corn":
        return (torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).sum(1).mean()
    return torch.nn.functional.cross_entropy(logits, target)


def fit_copy(frame: pd.DataFrame, probabilities: np.ndarray, columns: list[str], sidecar: dict, seed: int, epochs: int):
    """Fit one network of the same shape to the averaged probabilities. Returns it with its preprocessor and best loss."""
    hparams = default_hyperparameters(False)
    hparams.update({key: value for key, value in (sidecar.get("hyperparameters") or {}).items() if value is not None})
    if str(hparams.get("scheduler", "none")).lower() != "none":
        raise ValueError("Seed averaging only supports a constant learning rate")
    np.random.seed(seed)
    torch.manual_seed(seed)
    train, _ = holdout_split(frame)
    fit, validation = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + 1).split(frame.iloc[train], groups=frame.game_id.iloc[train]))
    fit, validation = train[fit], train[validation]
    categorical = [column for column in columns if not pd.api.types.is_numeric_dtype(frame[column])]
    numeric = [column for column in columns if column not in categorical]
    processor = ColumnTransformer([("num", StandardScaler(), numeric), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)])
    processor.fit(frame.iloc[fit][columns])
    inputs = transform(processor, frame)
    kind = sidecar["kind"]
    target, weight = soft_targets(kind, probabilities)
    network = build_network(kind, {"input_dim": inputs.shape[1], "output_dim": probabilities.shape[1]}, sidecar["arch"])
    optimizer = build_optimizer(network.parameters(), hparams)

    def loss_for(rows):
        return soft_loss(kind, network(inputs[rows]), target[rows], None if weight is None else weight[rows])

    best, state, stale, size = float("inf"), copy.deepcopy(network.state_dict()), 0, int(hparams["batch_size"])
    for epoch in range(epochs):
        network.train()
        order = np.random.default_rng(seed + epoch).permutation(fit)
        for start in range(0, len(order), size):
            optimizer.zero_grad()
            loss_for(order[start:start + size]).backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            value = float(loss_for(validation))
        if value < best:
            best, state, stale = value, copy.deepcopy(network.state_dict()), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    network.load_state_dict(state)
    network.eval().float()
    print(f"[seed average] stopped after {epoch + 1} epochs", flush=True)
    return network, processor, best


def averaged_probabilities(directories: list[Path], model: str, prefix: str, frame: pd.DataFrame) -> np.ndarray:
    total = 0
    for directory in directories:
        sidecar, network, processor, _ = load_network(directory, model, prefix)
        total = total + network_probabilities(sidecar["kind"], forward(network, transform(processor, frame))).astype(np.float32)
    return total / len(directories)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True, help="Refit to start from; it also counts as a seed")
    parser.add_argument("--seed-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--player-table", type=Path, required=True)
    parser.add_argument("--team-table", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--seasons", type=int, nargs=2, default=(2017, 2025))
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    output = output_dir(args.output)
    for path in args.base_dir.iterdir():
        if path.is_file() and not (output / path.name).exists():
            shutil.copy2(path, output / path.name)
    directories = [args.base_dir, *args.seed_dir]
    specs = refit_specs(args.base_dir)
    prepared, players, roster = load_inputs(REFERENCE, tuple(args.seasons), args.player_table, args.team_table, args.inputs)
    for model in args.models:
        spec, prefix = specs[model], specs[model]["prefix"]
        if (output / f"{model}_seed_average.json").is_file():
            print(f"[seed average] {model}: done; skipping", flush=True)
            continue
        sidecar = json.loads((args.base_dir / f"{prefix}_arch.json").read_text())
        seeds = [json.loads((directory / f"{prefix}_arch.json").read_text())["seed"] for directory in directories]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{model}: seed directories repeat a seed: {seeds}")
        frame, target, columns = model_rows(model, spec, prepared, players, roster)
        frame, target = frame.reset_index(drop=True), target.reset_index(drop=True)
        print(f"[seed average] {model}: rows={len(frame)} seeds={seeds}", flush=True)
        probabilities = averaged_probabilities(directories, model, prefix, frame)
        network, processor, best = fit_copy(frame, probabilities, columns, sidecar, seeds[0], args.epochs)
        torch.save(network.state_dict(), output / f"{prefix}_model.pt")
        joblib.dump(processor, output / f"{prefix}_preprocessor.pkl")
        write_json(output / f"{prefix}_arch.json", {**sidecar, "seed": seeds[0], "seed_average": seeds})
        _, held = holdout_split(frame)
        bins = pd.read_csv(args.base_dir / f"{prefix}_bins.csv")["0"].to_numpy(dtype=float) if sidecar["kind"] == "corn" else None
        labels = encode_labels(model, sidecar, target, bins)
        with torch.no_grad():
            copied = network_probabilities(sidecar["kind"], forward(network, transform(processor, frame.iloc[held])))
        classes = list(bins) if bins is not None else list(range(probabilities.shape[1]))
        metrics = json.loads((args.base_dir / f"{model}_fit_metrics.json").read_text())
        write_json(output / f"{model}_fit_metrics.json", {**metrics, "new_metrics": metric_record(labels[held], copied, classes), "best_validation_loss": best, "seed_average": seeds})
        write_json(output / f"{model}_seed_average.json", {"model": model, "seeds": seeds, "teacher_match_loss": best})


if __name__ == "__main__":
    main()
