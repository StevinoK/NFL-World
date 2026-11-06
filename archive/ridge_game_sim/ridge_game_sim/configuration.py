from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib

from ridge_game_sim.paths import REPO, ROOT, digest, output_dir, write_json

MODEL_CONTEXT = {
    "play": ["cpoe", "rush_yards"],
    "comp": ["cpoe", "cpoe_short", "cpoe_medium", "cpoe_deep"],
    "target": ["yards_per_target", "first_down"],
    "rusher": ["rush_yards", "explosive_rush"],
    "yac": ["yac_oe", "first_down"],
    "ay": ["cpoe"], "sack": ["sack"], "scramble": ["scramble_yards"],
    "int": ["interception_worthy"], "rush": ["rush_yards", "explosive_rush"],
    "scramble_yards": ["scramble_yards"],
}


def context_columns(model: str) -> list[str]:
    targets = MODEL_CONTEXT.get(model, [])
    return [f"ridge_{target}_{side}_effect" for target in targets for side in ("venue", "offense") if target not in {"pass_epa", "rush_epa"}]


def snapshot(source: Path = REPO / "artifacts/runtime", destination: Path = ROOT / "config") -> dict:
    destination = output_dir(destination)
    models = {}
    for model in ("kick", "ay", "target", "comp", "int", "yac", "sack", "scramble", "rush", "scramble_yards", "punt", "time", "rusher", "play", "fumble", "def_ret", "penalty_no_play", "penalty_on_play", "penalty_yards", "penalty_first_down"):
        prefix = "scramble_classifier" if model == "scramble" and (source / "scramble_classifier_model.pt").exists() else model
        cascade = source / f"{model}_cascade_arch.json"
        arch_path = cascade if cascade.exists() else source / f"{prefix}_arch.json"
        if not arch_path.exists():
            raise ValueError(f"Missing active model sidecar: {arch_path}")
        sidecar = json.loads(arch_path.read_text())
        is_cascade = sidecar["kind"] == "cascade"
        proc = source / f"{model}_cascade_stage3_preprocessor.pkl" if is_cascade else source / f"{prefix}_preprocessor.pkl"
        groups_path = source / f"{model}_cascade_columns.pkl" if is_cascade else source / f"{prefix}_columns.pkl"
        columns = list(map(str, joblib.load(proc).feature_names_in_))
        prior = source / f"{prefix}_prior_preprocessor.pkl"
        models[model] = {"prefix": prefix, "sidecar": sidecar, "columns": [column for column in columns if column != "residual_com"], "groups": list(joblib.load(groups_path)), "prior_columns": list(map(str, joblib.load(prior).feature_names_in_)) if prior.exists() and not is_cascade else [], "context_columns": context_columns(model), "source_hashes": {path.name: digest(path) for path in (arch_path, proc, groups_path)}}
    sources = {}
    for name, path in {
        "model_feature_config.json": REPO / "model_feature_config.json",
        "bayesian_decay_rates.json": REPO / "bayesian_decay_rates.json",
        "ridge_best_params_unverified.json": REPO / "ridge_adjusted_weekly_bakeoff/best_params.json",
    }.items():
        if path.exists():
            shutil.copy2(path, destination / name)
            sources[name] = {"path": str(path), "sha256": digest(path)}
    manifest = {"models": models, "sources": sources, "ridge_selection_verified": False}
    write_json(destination / "model_manifest.json", manifest)
    return manifest


def main():
    import argparse
    from ridge_game_sim.paths import assert_protected, protected_hashes
    parser = argparse.ArgumentParser(description="Read-only source snapshot into the isolated experiment")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "config")
    args = parser.parse_args()
    before = protected_hashes()
    try:
        manifest = snapshot(destination=args.output_dir)
        print({key: value["sidecar"]["kind"] for key, value in manifest["models"].items()})
    finally:
        assert_protected(before)


if __name__ == "__main__":
    main()