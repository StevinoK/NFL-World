"""Train the flat and CORN play models together, sharing one trunk over the inputs they all use (an experiment).

Each model normally trains alone, so the network that calls plays and the one that sizes air yards learn the same game
situation and team strengths separately, and their seed noise doesn't average out. Here every chosen model keeps its
own rows, target and inputs (as the base refit built them), but the columns all of them share go through one trunk
whose embedding each model's head reads next to its own columns (``nflsim.engine.model_architecture.MultitaskHead``).
All heads train at once, one model's minibatch at a time, and stop together on their average validation loss.

The output starts as a copy of the base refit, with each chosen model's network replaced by its head (and its
preprocessor by one that scales the shared columns like every other head's), so it fits game effects, packages and
loads like any other refit. Models other than flat and CORN networks keep the base refit's versions.

    python -m nflsim.models.multitask --base-dir <refit> --output <dir> --player-table <csv> --team-table <csv> \\
        --inputs <dir with pbp.parquet and roster.parquet> [--models play ay comp ...]
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from coral_pytorch.losses import corn_loss
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nflsim.engine.model_architecture import ConfigurableMLP, MultitaskHead
from nflsim.features.goal_line import yards_available
from nflsim.models.train import goal_capped_corn_loss, holdout_split
from nflsim.paths import output_dir, write_json

INPUT_CLIP = 10.0
DEFAULT_ARCH = {"trunk_hidden_dims": [256, 128], "embedding_dim": 64, "head_hidden_dims": [64], "dropout": 0.1,
                "activation": "relu"}


@dataclass
class Task:
    model: str
    prefix: str
    kind: str                 # flat | corn
    frame: pd.DataFrame
    columns: list[str]
    classes: list
    labels: np.ndarray        # class index per row
    capped: np.ndarray | None  # CORN rows that only show the play got at least this far (goal line)
    sidecar: dict


def make_task(model: str, prefix: str, sidecar: dict, frame: pd.DataFrame, target: pd.Series, columns: list[str],
              bins: np.ndarray | None = None) -> Task:
    kind = sidecar["kind"]
    if kind == "corn":
        classes = list(bins if bins is not None else sorted(pd.unique(target)))
    else:
        classes = sidecar.get("dims", {}).get("class_labels") or list(range(int(sidecar["dims"]["output_dim"])))
    labels = pd.Categorical(target, categories=classes).codes.astype(np.int64)
    if (labels < 0).any():
        raise ValueError(f"{model}: targets outside the class map")
    capped = None
    if sidecar.get("goal_capped"):
        capped = np.asarray(target, dtype=float) >= yards_available(sidecar.get("model_name", model), frame)
    return Task(model, prefix, kind, frame.reset_index(drop=True), list(columns), classes, labels, capped, sidecar)


def common_columns(tasks: list[Task]) -> list[str]:
    shared = set.intersection(*(set(task.columns) for task in tasks))
    return [column for column in tasks[0].columns if column in shared]


def split_rows(task: Task, seed: int):
    """Fit / validation / test rows: the shared held-out fifth is the test set, a fifth of the rest validates."""
    train, test = holdout_split(task.frame)
    groups = task.frame.game_id.iloc[train]
    fit, validation = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(train, groups=groups))
    return train[fit], train[validation], test


def is_categorical(frame: pd.DataFrame, column: str) -> bool:
    return not pd.api.types.is_numeric_dtype(frame[column])


def fit_processors(tasks: list[Task], common: list[str], fit_rows: dict, sample_rows: int, seed: int) -> dict:
    """One ColumnTransformer per model whose shared columns are scaled and encoded from the same sample of every
    model's rows, so the trunk sees them the same way whichever head it serves."""
    rng = np.random.default_rng(seed)
    per_task = max(1, sample_rows // len(tasks))
    sample = pd.concat([task.frame.iloc[rng.permutation(fit_rows[task.model])[:per_task]][common] for task in tasks],
                       ignore_index=True)
    processors = {}
    for task in tasks:
        own = [column for column in task.columns if column not in common]
        rows = rng.permutation(fit_rows[task.model])[:sample_rows]
        fit_frame = pd.concat([sample, task.frame.iloc[rows][own]], ignore_index=True)
        numeric = [column for column in [*common, *own] if not is_categorical(task.frame, column)]
        categorical = [column for column in [*common, *own] if is_categorical(task.frame, column)]
        for column in categorical:
            # No NaN category: a missing value encodes as all zeros, as the compiled inference path does.
            values = fit_frame[column].astype(object)
            fill = values.dropna().mode()
            fit_frame[column] = values.where(values.notna(), fill.iloc[0] if len(fill) else "missing")
        processor = ColumnTransformer([("num", StandardScaler(), numeric),
                                       ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)])
        processors[task.model] = processor.fit(fit_frame)
    return processors


def column_origin(processor) -> list[str]:
    """The input column behind each transformed output column, in output order."""
    origin = []
    for name, transformer, columns in processor.transformers_:
        if name == "num":
            origin += list(columns)
        elif name == "cat":
            for column, categories in zip(columns, transformer.categories_):
                origin += [column] * len(categories)
    return origin


def transform(processor, frame: pd.DataFrame) -> torch.Tensor:
    values = processor.transform(frame[list(processor.feature_names_in_)])
    return torch.from_numpy(np.clip(np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0), -INPUT_CLIP, INPUT_CLIP))


def head_dims(task: Task, processor, common: list[str]) -> dict:
    origin = column_origin(processor)
    return {"input_dim": len(origin),
            "common_index": [index for index, column in enumerate(origin) if column in common],
            "own_index": [index for index, column in enumerate(origin) if column not in common],
            "output_dim": len(task.classes) - 1 if task.kind == "corn" else len(task.classes)}


def task_loss(task: Task, logits: torch.Tensor, labels: torch.Tensor, capped: torch.Tensor | None) -> torch.Tensor:
    if task.kind == "corn":
        if capped is not None:
            return goal_capped_corn_loss(logits, labels, capped)
        return corn_loss(logits, labels, len(task.classes))
    return torch.nn.functional.cross_entropy(logits, labels)


def probabilities(task: Task, logits: torch.Tensor) -> np.ndarray:
    if task.kind == "corn":
        survival = torch.sigmoid(logits).cumprod(1).numpy()
        return np.concatenate([1 - survival[:, :1], survival[:, :-1] - survival[:, 1:], survival[:, -1:]], axis=1)
    return torch.softmax(logits, 1).numpy()


def train(tasks: list[Task], arch: dict | None = None, epochs: int = 100, seed: int = 42, batch_size: int = 1024,
          learning_rate: float = 1e-3, weight_decay: float = 0.01, patience: int = 8, sample_rows: int = 200_000,
          max_fit_rows: int | None = None):
    """Train every task's head with one shared trunk. Returns heads, processors, sidecar dims, per-task test log
    loss and the history."""
    arch = {**DEFAULT_ARCH, **(arch or {})}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    common = common_columns(tasks)
    if not common:
        raise ValueError("The models share no inputs; nothing for a trunk to read")
    splits = {task.model: split_rows(task, seed) for task in tasks}
    if max_fit_rows:
        splits = {model: (rng.permutation(fit)[:max_fit_rows], validation, test) for model, (fit, validation, test) in splits.items()}
    processors = fit_processors(tasks, common, {model: split[0] for model, split in splits.items()}, sample_rows, seed)
    dims = {task.model: head_dims(task, processors[task.model], common) for task in tasks}
    widths = {len(item["common_index"]) for item in dims.values()}
    if len(widths) != 1:
        raise ValueError(f"Shared columns encode to different widths across models: {sorted(widths)}")
    trunk_arch = {"hidden_dims": arch["trunk_hidden_dims"], "dropout": arch["dropout"], "activation": arch["activation"]}
    trunk = ConfigurableMLP(widths.pop(), int(arch["embedding_dim"]), trunk_arch)
    heads = torch.nn.ModuleDict({task.model: MultitaskHead(dims[task.model], arch, trunk=trunk) for task in tasks})
    optimizer = torch.optim.AdamW(heads.parameters(), lr=learning_rate, weight_decay=weight_decay)
    data = {}
    for task in tasks:
        fit, validation, test = splits[task.model]
        rows = np.concatenate([fit, validation, test])
        inputs = transform(processors[task.model], task.frame.iloc[rows])
        position = {kind: slice(start, start + len(index)) for kind, index, start in
                    (("fit", fit, 0), ("validation", validation, len(fit)), ("test", test, len(fit) + len(validation)))}
        capped = None if task.capped is None else torch.from_numpy(task.capped[rows])
        data[task.model] = (inputs, torch.from_numpy(task.labels[rows]), capped, position)

    def loss_on(task, part):
        inputs, labels, capped, position = data[task.model]
        index = position[part]
        logits = heads[task.model](inputs[index])
        return task_loss(task, logits, labels[index], None if capped is None else capped[index])

    history, best, best_state, stale = [], np.inf, copy.deepcopy(heads.state_dict()), 0
    for epoch in range(epochs):
        heads.train()
        batches = []
        for task in tasks:
            count = data[task.model][3]["fit"].stop
            order = rng.permutation(count)
            batches += [(task, order[start:start + batch_size]) for start in range(0, count, batch_size)]
        for batch in rng.permutation(len(batches)):
            task, index = batches[batch]
            inputs, labels, capped, _ = data[task.model]
            optimizer.zero_grad()
            loss = task_loss(task, heads[task.model](inputs[index]), labels[index], None if capped is None else capped[index])
            if not torch.isfinite(loss):
                raise ValueError(f"{task.model}: nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(heads.parameters(), 5.0)
            optimizer.step()
        heads.eval()
        with torch.no_grad():
            validation = {task.model: float(loss_on(task, "validation")) for task in tasks}
        objective = float(np.mean(list(validation.values())))
        history.append({"epoch": epoch, "valid_loss": objective, **{f"valid_{model}": value for model, value in validation.items()}})
        if objective < best:
            best, best_state, stale = objective, copy.deepcopy(heads.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    heads.load_state_dict(best_state)
    heads.eval()
    metrics = {}
    with torch.no_grad():
        for task in tasks:
            inputs, labels, _, position = data[task.model]
            index = position["test"]
            p = probabilities(task, heads[task.model](inputs[index]))
            chosen = np.clip(p[np.arange(len(p)), labels[index].numpy()], 1e-12, 1)
            metrics[task.model] = {"test_rows": int(len(p)), "test_log_loss": float(-np.log(chosen).mean())}
    return heads, processors, dims, metrics, history, common, arch


def write_refit(base: Path, output: Path, tasks: list[Task], heads, processors, dims, metrics, history, common, arch) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; use a fresh output directory")
    shutil.copytree(base, output, dirs_exist_ok=True)
    for task in tasks:
        torch.save(heads[task.model].state_dict(), output / f"{task.prefix}_model.pt")
        joblib.dump(processors[task.model], output / f"{task.prefix}_preprocessor.pkl")
        sidecar = copy.deepcopy(task.sidecar)
        sidecar.pop("cross_fit_folds", None)  # the base refit's fold networks don't match the heads
        class_labels = sidecar.get("dims", {}).get("class_labels")
        sidecar.update(kind="multitask", head=task.kind, arch=arch,
                       dims={**dims[task.model], **({"class_labels": class_labels} if class_labels else {})},
                       multitask={"models": [item.model for item in tasks], "common_columns": common})
        write_json(output / f"{task.prefix}_arch.json", sidecar)
    base_metrics = {}
    for task in tasks:
        path = base / f"{task.model}_fit_metrics.json"
        if path.is_file():
            base_metrics[task.model] = json.loads(path.read_text()).get("new_metrics", {}).get("log_loss")
    write_json(output / "multitask_summary.json", {
        "base_refit": str(base), "models": [task.model for task in tasks], "common_columns": common, "arch": arch,
        "test_log_loss": {model: item["test_log_loss"] for model, item in metrics.items()},
        "base_test_log_loss": base_metrics, "history": history})


def main() -> None:
    from nflsim.models.fit_minimal import model_rows, with_refit_tables
    from nflsim.models.game_effects import load_inputs, refit_specs
    from nflsim.models.fit_minimal import REFERENCE
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", help="Default: every flat and CORN network in the base refit")
    parser.add_argument("--player-table", type=Path, required=True)
    parser.add_argument("--team-table", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--seasons", type=int, nargs=2, default=(2017, 2025))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-fit-rows", type=int, help="Cap each model's training rows (memory)")
    parser.add_argument("--arch", type=json.loads, default={}, help='JSON overrides, e.g. {"embedding_dim": 96}')
    args = parser.parse_args()
    specs = refit_specs(args.base_dir)
    prepared, players, roster = load_inputs(REFERENCE, tuple(args.seasons), args.player_table, args.team_table, args.inputs)
    prepared = with_refit_tables(prepared, args.base_dir)
    tasks = []
    for model, spec in specs.items():
        if args.models and model not in args.models:
            continue
        sidecar = json.loads((args.base_dir / f"{spec['prefix']}_arch.json").read_text())
        if sidecar["kind"] not in {"flat", "corn"}:
            continue
        frame, target, columns = model_rows(model, spec, prepared, players, roster)
        bins_path = args.base_dir / f"{spec['prefix']}_bins.csv"
        bins = pd.read_csv(bins_path)["0"].to_numpy() if bins_path.is_file() else None
        tasks.append(make_task(model, spec["prefix"], sidecar, frame, target, columns, bins))
        print(f"[multitask] {model}: {len(frame)} rows, {len(columns)} inputs", flush=True)
    heads, processors, dims, metrics, history, common, arch = train(
        tasks, args.arch, args.epochs, args.seed, max_fit_rows=args.max_fit_rows)
    write_refit(args.base_dir, output_dir(args.output), tasks, heads, processors, dims, metrics, history, common, arch)
    print(json.dumps({model: item["test_log_loss"] for model, item in metrics.items()}, indent=1))


if __name__ == "__main__":
    main()
