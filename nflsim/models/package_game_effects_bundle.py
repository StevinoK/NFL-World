"""Package the flat-model refit plus fitted game effects as a sealed bundle.

Feature tables, player-value tables and weekly inputs come from an existing bundle
(so simulations use the same pregame features); the play-model weights come from
`fit_minimal --game-effects` and game_effects.json from `nflsim.models.game_effects`.

    python -m nflsim.models.package_game_effects_bundle --refit-dir <dir> --effects <json> --output <bundle dir>
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import pandas as pd

from nflsim.bundle import seal_bundle, validate_bundle, verify_bundle_hash
from nflsim.features.input_sources import write_input_sources
from nflsim.engine.game_effects import FILE_NAME
from nflsim.models.game_effects import refit_specs
from nflsim.models.package_minimal_bundle import finite_json
from nflsim.paths import ROOT, digest, write_json

BASE = ROOT / "artifacts/runtime"
MODEL_FILES = ("model.pt", "preprocessor.pkl", "columns.pkl", "arch.json", "bins.csv")
USAGE_PARAMS, SPECIAL_TEAMS_PARAMS = "usage_params.json", "special_teams_params.json"


def package(refit_dir: Path, effects: Path, output: Path, base: Path = BASE, player_table: Path | None = None,
            team_table: Path | None = None, features: Path | None = None, weekly_inputs: Path | None = None,
            selection: Path | None = None) -> str:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "bundle_manifest.json").exists():
        raise ValueError("Sealed bundles are immutable; package into a new directory")
    specs = refit_specs(refit_dir)
    base_state = json.loads((base / "training_manifest.json").read_text())
    model_prefixes = set(specs) | {spec["prefix"] for spec in specs.values()}
    for path in base.iterdir():
        stale = any(path.name.startswith(f"{prefix}_") for prefix in model_prefixes | {"scramble_classifier"})
        if path.name in {"training_manifest.json", "bundle_manifest.json", "model_manifest.json"} or stale:
            continue
        if path.is_dir():
            shutil.copytree(path, output / path.name, dirs_exist_ok=True)
        elif not (output / path.name).exists():
            shutil.copy2(path, output / path.name)

    if features is not None:
        # A fresh `nflsim.features.build` output replaces the base bundle's ridge/PV feature files.
        from nflsim.features.weekly_refresh import FEATURE_FILES
        for name in FEATURE_FILES | {"ridge_selection_parameters.json"}:
            if (features / name).is_file():
                shutil.copy2(features / name, output / name)
    if selection is not None:
        shutil.copy2(selection, output / "ridge_selection_parameters.json")
    recorded = json.loads((output / "feature_manifest.json").read_text()).get("selection_sha256")
    if recorded is not None and recorded != digest(output / "ridge_selection_parameters.json"):
        raise ValueError("ridge_selection_parameters.json is not the selection the features were built with; pass --selection")
    if weekly_inputs is not None:
        published = output / "weekly_inputs"
        published.mkdir(exist_ok=True)
        for name in ("schedule.parquet", "roster.parquet"):
            shutil.copy2(weekly_inputs / name, published / name)
        source = json.loads((weekly_inputs / "input_manifest.json").read_text())
        write_json(published / "input_manifest.json", {
            "active_season": source.get("active_season"),
            "files": {name: digest(published / name) for name in ("schedule.parquet", "roster.parquet")},
            "source_manifest_sha256": digest(weekly_inputs / "input_manifest.json")})

    manifest = {"feature_semantics": base_state["feature_semantics"], "models": {},
                "note": "Flat play models with shared game effects (game_effects.json) in place of the cascades; adds sack_yards"}
    for model, spec in specs.items():
        prefix = spec["prefix"]
        for suffix in MODEL_FILES:
            source = refit_dir / f"{prefix}_{suffix}"
            if source.is_file():
                shutil.copy2(source, output / source.name)
        entry = copy.deepcopy(spec)
        entry["sidecar"] = json.loads((output / f"{prefix}_arch.json").read_text())
        manifest["models"][model] = entry
    write_json(output / "model_manifest.json", finite_json(manifest))
    shutil.copy2(effects, output / FILE_NAME)
    if player_table is not None:
        shutil.copy2(player_table, output / "player_stats_wide.csv")
        features = json.loads((output / "feature_manifest.json").read_text())
        features["tables"]["player_stats_wide.csv"] = digest(output / "player_stats_wide.csv")
        usage = player_table.parent / "usage_manifest.json"
        features["usage"] = json.loads(usage.read_text()) if usage.is_file() else {"source": str(player_table)}
        if Path(features["usage"].get("params", "")).is_file():
            # The weekly refresh recomputes usage shares with the parameters the models were trained on.
            shutil.copy2(features["usage"]["params"], output / USAGE_PARAMS)
        write_json(output / "feature_manifest.json", features)
    if team_table is not None:
        shutil.copy2(team_table, output / "team_stats_wide.csv")
        features = json.loads((output / "feature_manifest.json").read_text())
        features["tables"]["team_stats_wide.csv"] = digest(output / "team_stats_wide.csv")
        special = team_table.parent / "special_teams_manifest.json"
        features["special_teams"] = json.loads(special.read_text()) if special.is_file() else {"source": str(team_table)}
        if "params" in features["special_teams"]:
            write_json(output / SPECIAL_TEAMS_PARAMS, features["special_teams"]["params"])
        write_json(output / "feature_manifest.json", features)

    lineage = {**copy.deepcopy(base_state["lineage"]), "config": digest(output / "model_manifest.json"),
               "features": digest(output / "feature_manifest.json")}
    trained = {}
    for model, spec in manifest["models"].items():
        prefix, sidecar = spec["prefix"], spec["sidecar"]
        metrics = json.loads((refit_dir / f"{model}_fit_metrics.json").read_text())
        names = [f"{prefix}_{suffix}" for suffix in MODEL_FILES if (output / f"{prefix}_{suffix}").is_file()]
        trained[model] = {"kind": sidecar["kind"], "seasons": sidecar["seasons"], "seed": sidecar.get("seed"),
                          "trained_at": pd.Timestamp.utcnow().isoformat(), "lineage": lineage,
                          "test_metrics": metrics.get("new_metrics", {}),
                          "artifact_hashes": {name: digest(output / name) for name in names}}
    state = {**{key: value for key, value in base_state.items() if key not in {"trained", "lineage"}},
             "benchmark_ready": False, "lineage": lineage, "trained": trained,
             "context_map": {model: spec["context_columns"] for model, spec in manifest["models"].items()},
             "game_effects": {"file": FILE_NAME, "sha256": digest(output / FILE_NAME)}}
    write_json(output / "training_manifest.json", state)
    validate_bundle(output, require_ready=False)
    state["benchmark_ready"] = True
    write_json(output / "training_manifest.json", state)
    write_input_sources(output)
    bundle_hash = seal_bundle(output)
    validate_bundle(output, require_ready=True)
    verify_bundle_hash(output)
    return bundle_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refit-dir", type=Path, required=True)
    parser.add_argument("--effects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--player-table", type=Path,
                        help="player_stats_wide.csv replacing the base bundle's (e.g. from `nflsim.features.usage write`)")
    parser.add_argument("--team-table", type=Path,
                        help="team_stats_wide.csv replacing the base bundle's (e.g. from `nflsim.features.special_teams write`)")
    parser.add_argument("--features", type=Path,
                        help="`nflsim.features.build` output whose ridge/PV feature files replace the base bundle's")
    parser.add_argument("--weekly-inputs", type=Path, help="Current-season inputs (schedule/roster) to publish with the bundle")
    parser.add_argument("--selection", type=Path,
                        help="Ridge parameter file the --features build used (its `parameters_path`)")
    args = parser.parse_args()
    print(package(args.refit_dir, args.effects, args.output, args.base, args.player_table, args.team_table,
                  args.features, args.weekly_inputs, args.selection))


if __name__ == "__main__":
    main()
