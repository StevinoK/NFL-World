from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from pathlib import Path

from nflsim.bundle import seal_bundle, validate_bundle, verify_bundle_hash
from nflsim.models.fit_minimal import REFERENCE, reduced_spec
from nflsim.paths import ROOT, digest, output_dir, write_json
from nflsim.semantics import (
    ISOLATED_MU_CONTEXT_V3, ISOLATED_MU_CONTEXT_V3_DEFINITION,
)


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def package(output: Path) -> Path:
    output = output_dir(output)
    if any(output.iterdir()):
        raise ValueError(f"Candidate output must be empty: {output}")

    shutil.copytree(REFERENCE, output, dirs_exist_ok=True)
    fit_dir = ROOT / "outputs/refits/pv-minimal-player-defense-venue-weather-20261021"
    reference_state = json.loads((REFERENCE / "training_manifest.json").read_text())
    model_manifest = json.loads((REFERENCE / "model_manifest.json").read_text())
    feature_manifest = json.loads((REFERENCE / "feature_manifest.json").read_text())

    for model, spec in model_manifest["models"].items():
        refit_spec = reduced_spec(model, spec)
        prefix = spec["prefix"]
        changed_fit = (fit_dir / f"{model}_fit_metrics.json").is_file()
        if changed_fit:
            artifact_names = (
                [f"{model}_cascade_arch.json", f"{model}_cascade_columns.pkl",
                 *[f"{model}_cascade_stage{stage}_{suffix}" for stage in (1, 2, 3)
                   for suffix in ("model.pt", "preprocessor.pkl")],
                 f"{model}_preprocessor.pkl"]
                if spec["sidecar"]["kind"] == "cascade"
                else [f"{prefix}_model.pt", f"{prefix}_preprocessor.pkl",
                      f"{prefix}_columns.pkl", f"{prefix}_arch.json"]
            )
            if spec["sidecar"]["kind"] != "cascade" and model in {"ay", "punt", "penalty_yards", "rush", "scramble_yards", "time", "def_ret"}:
                artifact_names.append(f"{prefix}_bins.csv")
            for name in artifact_names:
                source = fit_dir / name
                if source.is_file():
                    shutil.copy2(source, output / name)
            if spec["sidecar"]["kind"] == "cascade":
                refit_arch = json.loads((output / f"{model}_cascade_arch.json").read_text())
                spec["sidecar"] = refit_arch
            else:
                spec["sidecar"] = json.loads((output / f"{prefix}_arch.json").read_text())

        spec["columns"] = list(refit_spec["columns"])
        spec["groups"] = list(refit_spec["groups"])
        spec["context_columns"] = list(refit_spec["context_columns"])
        spec["prior_columns"] = []
        if spec["sidecar"].get("kind") == "cascade":
            spec["sidecar"]["context_columns"] = list(refit_spec["context_columns"])

    write_json(output / "model_manifest.json", finite_json(model_manifest))

    feature_manifest["feature_semantics"] = ISOLATED_MU_CONTEXT_V3
    feature_manifest["isolated_mu"] = {
        **ISOLATED_MU_CONTEXT_V3_DEFINITION,
        "metrics": feature_manifest.get("isolated_mu", {}).get("metrics", {}),
    }
    write_json(output / "feature_manifest.json", feature_manifest)

    lineage = copy.deepcopy(reference_state["lineage"])
    lineage["config"] = digest(output / "model_manifest.json")
    lineage["features"] = digest(output / "feature_manifest.json")
    lineage["feature_semantics"] = ISOLATED_MU_CONTEXT_V3
    trained = copy.deepcopy(reference_state["trained"])
    for model, spec in model_manifest["models"].items():
        old_record = trained[model]
        prefix = spec["prefix"]
        names = list(old_record.get("artifact_hashes", {}))
        if model in {"play", "target", "rusher", "comp"}:
            names = [name for name in output.iterdir() if name.name.startswith(f"{model}_cascade_")]
            names = [path.name for path in names]
        for name in names:
            path = output / name
            if path.is_file():
                old_record.setdefault("artifact_hashes", {})[name] = digest(path)
        old_record["kind"] = spec["sidecar"]["kind"]
        old_record["lineage"] = lineage

    state = {
        **reference_state,
        "benchmark_ready": False,
        "feature_semantics": ISOLATED_MU_CONTEXT_V3,
        "lineage": lineage,
        "trained": trained,
        "context_map": {model: spec["context_columns"] for model, spec in model_manifest["models"].items()},
    }
    write_json(output / "training_manifest.json", state)
    validate_bundle(output, require_ready=False)
    state["benchmark_ready"] = True
    write_json(output / "training_manifest.json", state)
    seal_bundle(output)
    validate_bundle(output, require_ready=True)
    verify_bundle_hash(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the isolated simplified-feature refit as a separate validated bundle")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT.parent / "archive/nflsim_bundles/pv-minimal-player-defense-venue-weather-20261021")
    args = parser.parse_args()
    print(package(args.output_dir))


if __name__ == "__main__":
    main()