from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from check_runtime_artifacts import check_artifacts
from cascade_trainer import load_cascade_artifacts
from ridge_game_sim.paths import ROOT, canonical_hash, digest, inside, write_json
from ridge_game_sim.runtime import RidgeRuntime
from torch_game_simulation_runtime import CascadeModelWrapper, SimulationRuntime


def validate_bundle(directory: Path, require_ready=True):
    directory = Path(directory)
    state = json.loads((directory / "training_manifest.json").read_text())
    if require_ready and not state.get("benchmark_ready"):
        raise ValueError("Bundle is not benchmark-ready")
    if require_ready and state.get("synthetic_only"):
        raise ValueError("Synthetic fixtures are never benchmark-ready")
    if set(state["trained"]) != set(SimulationRuntime.models):
        raise ValueError("Every active model must be freshly trained")
    _, missing = check_artifacts(directory)
    if missing:
        raise ValueError(f"Incomplete bundle: {missing}")
    for model, record in state["trained"].items():
        if not state.get("synthetic_only"):
            if not record.get("seasons") or max(record["seasons"]) > 2024 or min(record["seasons"]) < 2017 or record.get("lineage") != state["lineage"]:
                raise ValueError(f"{model}: invalid training provenance")
            for name, checksum in record["artifact_hashes"].items():
                if digest(directory/name) != checksum:
                    raise ValueError(f"{model}: artifact changed {name}")
        if record["kind"] == "cascade":
            artifacts = load_cascade_artifacts(directory, model)
            if artifacts.stage3_model.fc1.in_features != len(artifacts.stage3_preprocessor.get_feature_names_out()):
                raise ValueError(f"{model}: stage 3 width mismatch")
            if artifacts.stage2_model.net[0].in_features != len(artifacts.context_columns):
                raise ValueError(f"{model}: stage 2 width mismatch")
            if (directory / f"{model}_model.pt").exists():
                raise ValueError(f"{model}: superseded checkpoint in fresh bundle")
    features = json.loads((directory / "feature_manifest.json").read_text())
    for name, checksum in features["tables"].items():
        if digest(directory / name) != checksum:
            raise ValueError(f"Frozen feature mismatch {name}")
    runtime = RidgeRuntime(directory, strict_schema=True)
    if set(runtime.model_dict) != set(SimulationRuntime.models):
        raise ValueError("Runtime did not load every active model")
    for key, processor in runtime.processor.items():
        if key.endswith("_cascade_stage2"):
            continue
        columns = list(processor.feature_names_in_)
        values = {column: "outdoors" if column == "roof" else "home" if column == "posteam_type" else "WR" if column == "position" else "rush" if column == "play_type_cat" else "run" if column == "play_outcome" else "no_play" if column == "penalty_context" else 0.1 for column in columns}
        if key.endswith("_prior"):
            runtime.transform_with_processor(key, pd.DataFrame([values]))
            continue
        network = runtime.model_dict[key]
        if isinstance(network, CascadeModelWrapper):
            context = {column: 0.1 for column in network.context_columns}
            latent = np.array([network.sample_residual(context, np.random.default_rng(1))])
        else:
            latent = np.zeros(getattr(network, "latent_dim", 1))
        values.update({column: 0.125 for column in runtime.context_map.get(key, [])})
        result = runtime.forward_model_logits(key, pd.DataFrame([values]), values, latent)
        if not torch.isfinite(result).all():
            raise ValueError(f"Nonfinite inference for {key}")
    if runtime.schema_issues:
        raise ValueError(f"Runtime schema issues: {runtime.schema_issues}")
    return runtime


def seal_bundle(directory: Path):
    directory = inside(directory)
    files = {path.name: digest(path) for path in sorted(directory.iterdir()) if path.is_file() and path.name != "bundle_manifest.json"}
    manifest = {"files": files, "bundle_hash": canonical_hash(files)}
    write_json(directory / "bundle_manifest.json", manifest)
    return manifest["bundle_hash"]


def verify_bundle_hash(directory: Path):
    manifest = json.loads((directory / "bundle_manifest.json").read_text())
    actual = {path.name: digest(path) for path in directory.iterdir() if path.is_file() and path.name != "bundle_manifest.json"}
    if actual != manifest["files"] or canonical_hash(actual) != manifest["bundle_hash"]:
        raise ValueError("Immutable bundle changed")
    return manifest["bundle_hash"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/runtime")
    args = parser.parse_args()
    validate_bundle(args.artifact_dir)
    print(verify_bundle_hash(args.artifact_dir))


if __name__ == "__main__":
    main()