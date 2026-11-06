from __future__ import annotations

import argparse
import copy
import json
import random
import zlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from coral_pytorch.losses import corn_loss
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nflsim.engine.cascade_trainer import CascadeConfig, fit_cascade_general, save_cascade_artifacts
from nflsim.engine.hyperparameters import balanced_class_weights, build_optimizer, build_scheduler, default_hyperparameters
from nflsim.engine.model_architecture import build_network
from nflsim.engine.model_reporting import ModelReportBuilder
from nflsim.configuration import context_columns
from nflsim.paths import ROOT, assert_protected, canonical_hash, digest, output_dir, protected_hashes, write_json
from nflsim.semantics import ISOLATED_MU_CONTEXT_V2, LEGACY_SEMANTICS
from nflsim.features.goal_line import yards_available
from nflsim.features.training_data import ATTEMPT_MODELS, SKILL_SLOT_ORDER, model_frame, prepare_frame


HOLDOUT_FOLDS = 5


def holdout_games(game_ids) -> np.ndarray:
    """One fixed fifth of games held out from every model, so game effects can be fit on shared unseen games."""
    return np.array([zlib.crc32(str(game).encode()) % HOLDOUT_FOLDS == 0 for game in game_ids], dtype=bool)


def holdout_split(frame):
    if "game_id" not in frame or frame.game_id.isna().any():
        raise ValueError("Training requires nonmissing game_id for the shared holdout")
    held = holdout_games(frame.game_id)
    if held.all() or not held.any():
        raise ValueError("Training requires games on both sides of the shared holdout")
    return np.flatnonzero(~held), np.flatnonzero(held)


def goal_capped_corn_loss(logits, labels, capped):
    """CORN loss where a capped row only says the play got at least this far, so it says nothing about stopping there."""
    tasks = torch.arange(logits.shape[1], device=logits.device)[None, :]
    labels, capped = labels[:, None], capped[:, None]
    used = (tasks <= labels) & ~(capped & (tasks == labels))
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, (labels > tasks).to(logits.dtype), reduction="none")
    return (losses * used).sum() / used.sum().clamp_min(1)


def fit_network(frame, target, columns, sidecar, epochs=100, seed=42):
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if epochs < 1 or frame.empty:
        raise ValueError("Training requires positive epochs and nonempty data")
    if len(target) != len(frame) or pd.isna(np.asarray(target)).any():
        raise ValueError("Target must align with frame and contain no missing labels")
    random.seed(int(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kind = sidecar["kind"]
    if kind not in {"flat", "corn", "ngmm", "hierarchical"}:
        raise ValueError(f"Unsupported active family {kind}")
    group_column = "game_team_id" if kind in {"ngmm", "hierarchical"} else "game_id"
    if group_column not in frame or frame[group_column].isna().any():
        raise ValueError(f"Training requires nonmissing {group_column} groups")
    groups = frame[group_column]
    if groups.nunique() < (2 if kind == "ngmm" else 3):
        raise ValueError("Training requires enough distinct groups for disjoint splits")
    train, test = holdout_split(frame)
    ordinal_target = kind == "corn" or sidecar["dims"].get("ordinal_output", False) or sidecar["dims"].get("target_mode") == "ordinal"
    classes = sidecar["dims"].get("class_labels") or (sorted(pd.unique(target)) if ordinal_target else list(range(int(sidecar["dims"]["output_dim"]))))
    encoded = pd.Categorical(target, categories=classes).codes
    if (encoded < 0).any():
        raise ValueError("Target incompatible with fixed class map")
    hparams = default_hyperparameters(kind == "hierarchical")
    hparams.update({key: value for key, value in (sidecar.get("hyperparameters") or {}).items() if value is not None})
    if hparams.get("kl_weight") is None:
        hparams["kl_weight"] = sidecar["dims"].get("kl_weight", 0.01)
    if kind == "ngmm":
        from nflsim.engine.ngmm_training import train_ngmm_grouped
        prior_columns = sidecar["prior_columns"]
        arch = sidecar["arch"]
        mode = sidecar["dims"].get("target_mode", "multiclass")
        if mode not in {"binary", "multiclass", "ordinal"}:
            raise ValueError("NGMM training requires a classification target mode")
        source_params = sidecar.get("hyperparameters") or {}
        for key, supported in {"optimizer": "adamw", "scheduler": "none", "grad_clip_norm": 5.0, "label_smoothing": 0.0, "class_weighting": "none"}.items():
            if source_params.get(key, supported) != supported:
                raise ValueError(f"Shared NGMM helper does not support {key}={source_params[key]}")
        if groups.iloc[train].value_counts().min() < 2:
            raise ValueError("NGMM training groups need at least two observations for validation")
        dropout = arch.get("dropout", 0.1)
        dropout = float(dropout if isinstance(dropout, (int, float)) else dropout[0])
        result = train_ngmm_grouped(
            train_frame=frame.iloc[train], test_frame=frame.iloc[test],
            y_train=encoded[train], y_test=encoded[test], train_groups=groups.iloc[train],
            play_columns=columns, prior_columns=prior_columns, target_mode=mode,
            output_dim=len(classes), latent_dim=hparams.get("latent_dim") or sidecar["dims"]["latent_dim"],
            max_epochs=epochs, seed=seed, hidden_dims=tuple(arch["hidden_dims"]),
            context_hidden_dims=tuple(arch["context_hidden_dims"]), dropout=dropout,
            learning_rate=source_params.get("learning_rate", 1e-3),
            weight_decay=source_params.get("weight_decay", 0.0),
            batch_size=source_params.get("group_batch_size", source_params.get("batch_size", 32)),
            quadrature_samples=source_params.get("quadrature_samples", 32),
            observations_per_group=source_params.get("observations_per_group", 256),
            validation_size=source_params.get("validation_size", 0.2),
            patience=source_params.get("patience", 12), log_progress=False,
        )
        required = {"model", "play_preprocessor", "prior_preprocessor", "test_probabilities", "y_test_class", "loss_history"}
        if required - result.keys():
            raise ValueError(f"NGMM result missing keys: {sorted(required - result.keys())}")
        history = result["loss_history"]
        if isinstance(history, dict):
            history = [{"epoch": index, "train_loss": train_loss, "valid_loss": valid_loss} for index, (train_loss, valid_loss) in enumerate(zip(history["train_loss"], history["valid_loss"]))]
        probabilities = np.asarray(result["test_probabilities"])
        if probabilities.shape != (len(test), len(classes)) or not np.isfinite(probabilities).all():
            raise ValueError("NGMM returned incompatible test probabilities")
        if not np.array_equal(result["y_test_class"], encoded[test]):
            raise ValueError("NGMM returned incompatible target class codes")
        return result["model"], result["play_preprocessor"], probabilities, encoded[test], classes, history, result["prior_preprocessor"]
    categorical = [column for column in columns if not pd.api.types.is_numeric_dtype(frame[column])]
    numerical = [column for column in columns if column not in categorical]
    processor = ColumnTransformer([("num", StandardScaler(), numerical), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)])
    fit, validation = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=(seed+1) % 2**32).split(frame.iloc[train], groups=groups.iloc[train]))
    fit, validation = train[fit], train[validation]
    processor.fit(frame.iloc[fit][columns])
    transformed = np.nan_to_num(processor.transform(frame[columns]).astype(np.float32), nan=0.0)
    inputs = torch.from_numpy(transformed).to(device)
    labels = torch.from_numpy(encoded.astype(np.int64)).to(device)
    capped = None
    if sidecar.get("goal_capped"):
        capped = torch.from_numpy(np.asarray(target, dtype=float) >= yards_available(sidecar["model_name"], frame)).to(device)
    dims = {"input_dim": inputs.shape[1], "output_dim": len(classes)}
    prior_processor, prior_inputs = None, None
    if kind == "hierarchical":
        prior_columns = sidecar["prior_columns"]
        prior_processor = ColumnTransformer([("num", StandardScaler(), prior_columns)])
        prior_processor.fit(frame.iloc[fit][prior_columns])
        prior_inputs = torch.from_numpy(np.nan_to_num(prior_processor.transform(frame[prior_columns]).astype(np.float32))).to(device)
        dims = {"play_input_dim": inputs.shape[1], "prior_input_dim": prior_inputs.shape[1], "latent_dim": hparams.get("latent_dim") or sidecar["dims"]["latent_dim"], "output_dim": len(classes), "ordinal_output": ordinal_target}
    model = build_network(kind, dims, sidecar["arch"]).to(device)
    optimizer = build_optimizer(model.parameters(), hparams)
    batch_size = int(hparams["batch_size"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    scheduler, step_mode = build_scheduler(optimizer, hparams, epochs, steps_per_epoch=(len(fit)+batch_size-1)//batch_size)
    weighting = hparams["class_weighting"]
    if weighting not in {"none", "balanced"}:
        raise ValueError(f"Unsupported class weighting: {weighting}")
    ordinal = kind == "corn" or dims.get("ordinal_output", False)
    if ordinal and (weighting != "none" or hparams["label_smoothing"] != 0):
        raise ValueError("CORN loss does not support class weighting or label smoothing")
    class_weights = balanced_class_weights(encoded[fit], len(classes)).to(device) if weighting == "balanced" else None
    def loss_for(indices, kl_scale):
        if prior_inputs is None:
            logits, kl = model(inputs[indices]), 0.0
        else:
            logits, kl = model(inputs[indices], prior_inputs[indices])
        if capped is not None:
            return goal_capped_corn_loss(logits, labels[indices], capped[indices]) + hparams["kl_weight"] * kl_scale * kl
        loss = corn_loss(logits, labels[indices], len(classes)) if ordinal else torch.nn.functional.cross_entropy(logits, labels[indices], weight=class_weights, label_smoothing=hparams["label_smoothing"])
        return loss + hparams["kl_weight"] * kl_scale * kl
    history, best_state, best_loss, stale = [], copy.deepcopy(model.state_dict()), float("inf"), 0
    for epoch in range(epochs):
        model.train()
        warmup = hparams.get("kl_warmup_epochs", 0)
        kl_scale = min(1.0, (epoch+1)/warmup) if warmup else 1.0
        permutation = np.random.default_rng(seed+epoch).permutation(fit)
        for start in range(0, len(permutation), batch_size):
            optimizer.zero_grad()
            loss = loss_for(permutation[start:start+batch_size], kl_scale)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            if hparams.get("grad_clip_norm", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hparams["grad_clip_norm"])
            optimizer.step()
            if step_mode == "batch" and scheduler.last_epoch < scheduler.total_steps:
                scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_for(validation, 1.0))
        if not np.isfinite(val_loss):
            raise ValueError("Nonfinite validation loss")
        history.append({"epoch": epoch, "valid_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"]})
        if step_mode == "plateau":
            scheduler.step(val_loss)
        elif step_mode == "epoch":
            scheduler.step()
        if val_loss < best_loss:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= hparams.get("patience", 12):
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(inputs[test]) if prior_inputs is None else model(inputs[test], prior_inputs[test], sample=False)[0]
        if ordinal:
            cumulative = torch.sigmoid(logits).cumprod(1).cpu().numpy()
            probabilities = np.concatenate([1-cumulative[:, :1], cumulative[:, :-1]-cumulative[:, 1:], cumulative[:, -1:]], axis=1)
        else:
            probabilities = torch.softmax(logits, 1).cpu().numpy()
    return model, processor, probabilities, encoded[test], classes, history, prior_processor


def cascade_metric_record(metrics):
    undefined = []

    def convert(value, path):
        if isinstance(value, dict):
            return {key: convert(item, f"{path}.{key}" if path else str(key)) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item, f"{path}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            if np.isnan(value) and path.rsplit(".", 1)[-1] in {"auroc", "auroc_macro_ovr"}:
                undefined.append(path)
                return None
            raise ValueError(f"Nonfinite cascade training metric: {path}")
        return value

    cleaned = convert(metrics, "")
    return {"metrics": cleaned, "undefined_metrics": undefined}


def train_one(model, frame, target, spec, output, epochs, seed, stage2_epochs=None):
    output = output_dir(output)
    if frame.empty or not frame.season.between(2017, 2024).all():
        raise ValueError("Training data must be nonempty and entirely within 2017-2024")
    columns = list(dict.fromkeys(spec["columns"] + spec["context_columns"]))
    missing = set(columns)-set(frame)
    if missing:
        raise ValueError(f"{model}: missing training features {sorted(missing)}")
    for column in spec["context_columns"]:
        if not np.isfinite(pd.to_numeric(frame[column], errors="coerce")).all():
            raise ValueError(f"{model}: unavailable ridge context {column}")
    sidecar = copy.deepcopy(spec["sidecar"])
    prefix = spec["prefix"]
    if sidecar["kind"] == "cascade":
        if model in ATTEMPT_MODELS:
            raise ValueError(f"{model}: cascades cannot record that they were trained on throws only")
        config = CascadeConfig(**sidecar["config"])
        config.epochs, config.seed = epochs, seed
        if stage2_epochs is not None:
            config.stage2_epochs = stage2_epochs
        context = list(dict.fromkeys(sidecar["context_columns"] + spec["context_columns"]))
        if set(context)-set(frame):
            raise ValueError(f"{model}: missing cascade context")
        frame = frame.copy()
        labels = ["kick", "pass", "punt", "rush"] if model == "play" else [str(index) for index in range(11)] if model in {"target", "rusher"} else ["0", "1"]
        frame["__label"] = target.astype(str).to_numpy()
        artifacts = fit_cascade_general(frame, "__label", columns, context, sidecar["categorical_columns"], labels, config, group_column="game_team_id", model_label=model)
        save_cascade_artifacts(artifacts, output, model, feature_groups=spec["groups"])
        joblib.dump(artifacts.stage3_preprocessor, output / f"{model}_preprocessor.pkl")
        return {**cascade_metric_record(artifacts.metrics), "kind": "cascade", "classes": labels}
    sidecar["prior_columns"] = list(dict.fromkeys(spec["prior_columns"] + spec["context_columns"]))
    network, processor, probs, actual, bins, history, prior = fit_network(frame, target, columns, sidecar, epochs, seed)
    torch.save(network.state_dict(), output / f"{prefix}_model.pt")
    joblib.dump(processor, output / f"{prefix}_preprocessor.pkl")
    joblib.dump(spec["groups"], output / f"{prefix}_columns.pkl")
    if prior is not None:
        joblib.dump(prior, output / f"{prefix}_prior_preprocessor.pkl")
    ordinal = sidecar["kind"] == "corn" or sidecar["dims"].get("ordinal_output", False) or sidecar["dims"].get("target_mode") == "ordinal"
    if ordinal:
        pd.DataFrame(bins).to_csv(output / f"{prefix}_bins.csv")
    kind = sidecar["kind"]
    if kind in {"flat", "corn"}:
        sidecar["dims"] = {"input_dim": network.fc1.in_features, "output_dim": len(bins), **({"class_labels": sidecar["dims"]["class_labels"]} if "class_labels" in sidecar["dims"] else {})}
    elif kind == "hierarchical":
        sidecar["dims"].update(play_input_dim=network.fc1.in_features-network.latent_dim, prior_input_dim=network.prior_encoder.hidden[0].in_features, latent_dim=network.latent_dim, output_dim=len(bins), ordinal_output=network.ordinal_output)
    else:
        sidecar["dims"].update(input_dim=network.fixed[0].in_features, context_dim=network.context[0].in_features, latent_dim=network.latent_dim, output_dim=network.output_dim)
    sidecar.update(seasons=sorted(frame.season.unique().astype(int).tolist()), seed=seed, trained_at=pd.Timestamp.utcnow().isoformat())
    if model in ATTEMPT_MODELS:
        sidecar["row_scope"] = "attempts"
    write_json(output / f"{prefix}_arch.json", sidecar)
    report = ModelReportBuilder(title=f"Ridge {model}")
    if len(bins) == 2 and not ordinal:
        report.add_binary_classification(model, actual, probs[:, 1], loss_history=history)
    else:
        report.add_multiclass(model, actual, probs, bins=bins if ordinal else None, loss_history=history)
    report.write_html(output_dir(ROOT / "training" / output.name) / f"{model}_report.html")
    return {"kind": kind, "classes": list(bins), "epochs": len(history)}


def verify_manifest_semantics(manifest, semantics):
    """V2 features need every model manifest (and cascade stage 2) to carry the V2 context schema."""
    if manifest.get("feature_semantics") != semantics:
        raise ValueError(f"Model manifest semantics {manifest.get('feature_semantics')} != features {semantics}")
    for model, spec in manifest["models"].items():
        missing = [column for column in context_columns(model, semantics) if column not in spec["context_columns"]]
        if spec["sidecar"].get("kind") == "cascade":
            missing += [column for column in spec["context_columns"] if column not in spec["sidecar"].get("context_columns", [])]
        if missing:
            raise ValueError(f"{model}: model manifest lacks {semantics} context {sorted(set(missing))}")


def verify_completed_training(model, record, output, lineage):
    if record.get("lineage") != lineage:
        raise ValueError(f"{model}: completed training provenance differs")
    hashes = record.get("artifact_hashes", {})
    if not hashes:
        raise ValueError(f"{model}: completed training has no artifact hashes")
    for name, checksum in hashes.items():
        path = Path(output) / name
        if Path(name).name != name or not path.is_file() or digest(path) != checksum:
            raise ValueError(f"{model}: completed training artifact changed or missing: {name}")


def main():
    parser = argparse.ArgumentParser(description="Standalone isolated training; never imports Torch Modeling.py")
    parser.add_argument("--pbp", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/runtime")
    parser.add_argument("--model-manifest", type=Path, default=ROOT / "config/model_manifest.json")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stage2-epochs", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Skip completed manifest entries after checking provenance and artifact hashes")
    args = parser.parse_args()
    if args.epochs < 1 or (args.stage2_epochs is not None and args.stage2_epochs < 1):
        parser.error("epochs must be positive")
    output = output_dir(args.artifact_dir)
    if (output / "bundle_manifest.json").exists():
        raise ValueError("Sealed bundles are immutable; train into a new experiment-local bundle directory")
    for lock in (ROOT / "benchmarks").rglob("active_bundle.json"):
        if json.loads(lock.read_text()).get("artifact_dir") == str(output):
            raise ValueError("Bundle is in use by a benchmark")
    before = protected_hashes()
    try:
        model_manifest_path = Path(args.model_manifest)
        manifest = json.loads(model_manifest_path.read_text())
        features = json.loads((output / "feature_manifest.json").read_text())
        for name, checksum in features["tables"].items():
            if digest(output / name) != checksum:
                raise ValueError("Frozen feature table changed")
        players, teams = [pd.read_csv(output / name, dtype={"game_id": str, "player_id": str}) for name in ("player_stats_wide.csv", "team_stats_wide.csv")]
        load = lambda path: pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        roster = load(args.roster)
        semantics = features.get("feature_semantics", LEGACY_SEMANTICS)
        frame_semantics = semantics if semantics == ISOLATED_MU_CONTEXT_V2 else None
        if ISOLATED_MU_CONTEXT_V2 in (semantics, manifest.get("feature_semantics")):
            verify_manifest_semantics(manifest, semantics)
        pbp = prepare_frame(load(args.pbp), players, teams, frame_semantics)
        selected = args.models or list(manifest["models"])
        if set(selected)-set(manifest["models"]):
            raise ValueError("Unknown model selection")
        trained = {}
        manifest_path = output / "training_manifest.json"
        lineage = {"features": digest(output / "feature_manifest.json"), "config": digest(model_manifest_path), "pbp": digest(args.pbp), "roster": digest(args.roster), "seed": args.seed, "epochs": args.epochs, "stage2_epochs": args.stage2_epochs}
        if semantics != LEGACY_SEMANTICS:
            lineage["feature_semantics"] = semantics
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old["lineage"] != lineage:
                raise ValueError("Existing training lineage differs; use a new experiment-local bundle directory")
            trained = old["trained"]
        state = {"lineage": lineage, "trained": trained, "context_map": {key: value["context_columns"] for key, value in manifest["models"].items()}, "slot_order": SKILL_SLOT_ORDER, "benchmark_ready": False}
        if semantics != LEGACY_SEMANTICS:
            state["feature_semantics"] = semantics
        write_json(manifest_path, state)
        for model in selected:
            if args.resume and model in trained:
                verify_completed_training(model, trained[model], output, lineage)
                print(f"[train] {model}: verified completed artifacts; skipping", flush=True)
                continue
            frame, target = model_frame(model, pbp, players, roster, frame_semantics)
            spec = manifest["models"][model]
            trained[model] = {**train_one(model, frame, target, spec, output, args.epochs, args.seed, args.stage2_epochs), "seasons": sorted(frame.season.unique().astype(int).tolist()), "seed": args.seed, "trained_at": pd.Timestamp.utcnow().isoformat(), "lineage": lineage, "artifact_hashes": {path.name: digest(path) for path in output.glob(f"{spec['prefix']}_*") if path.is_file()}}
            write_json(manifest_path, state)
        for name, checksum in features["tables"].items():
            if digest(output / name) != checksum:
                raise RuntimeError("Training changed frozen features")
        if set(trained) == set(manifest["models"]):
            from nflsim.bundle import validate_bundle, seal_bundle
            validate_bundle(output, require_ready=False)
            state["benchmark_ready"] = True
            write_json(manifest_path, state)
            seal_bundle(output)
    finally:
        assert_protected(before)


if __name__ == "__main__":
    main()