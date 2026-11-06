from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import joblib

from nflsim.paths import REPO, ROOT, digest, output_dir, write_json
from nflsim.features.player_value import PV_COLUMNS
from nflsim.ridge.metrics import TARGETS_BY_NAME
from nflsim.semantics import (
    ISOLATED_MU_CONTEXT_V2, context_available_column, offense_column,
    opponent_defense_column, venue_column,
)
from nflsim.features.weather import WEATHER_COLUMNS

MODEL_CONTEXT = {
    "play": ["cpoe", "rush_yards"],
    "comp": ["cpoe", "cpoe_short", "cpoe_medium", "cpoe_deep"],
    "target": ["yards_per_target", "first_down"],
    "rusher": ["rush_yards", "explosive_rush"],
    "yac": ["yac_oe", "first_down"],
    "ay": ["cpoe"], "sack": ["sack"], "sack_yards": ["sack"], "scramble": ["scramble_yards"],
    "int": ["interception_worthy"], "rush": ["rush_yards", "explosive_rush"],
    "scramble_yards": ["scramble_yards"],
}


_PASSER = ["cpoe", "cpoe_short", "cpoe_medium", "cpoe_deep", "sack", "interception_worthy"]
_RECEIVER = ["yac_oe", "first_down", "yards_per_target"]
_RUSHER = ["rush_yards", "explosive_rush"]
_TEAM = ["pass_epa", "rush_epa"]
# Ridge targets whose mus each model consumes, from the reference manifest feature groups:
# passer_columns, receiver_columns / receiver slot stats, rusher_columns / rusher slot stats, team_stats_columns.
V2_MODEL_CONTEXT = {
    "ay": _PASSER + _TEAM, "comp": _PASSER + _TEAM, "int": _PASSER + _TEAM, "sack": _PASSER + _TEAM, "sack_yards": _PASSER + _TEAM,
    "scramble": _PASSER + _TEAM + ["scramble_yards"],
    "yac": _RECEIVER + _TEAM, "target": _RECEIVER,
    "rush": _RUSHER + _TEAM, "rusher": _RUSHER, "scramble_yards": _RUSHER + _TEAM + ["scramble_yards"],
    "play": ["cpoe", "rush_yards"] + _TEAM, "fumble": _TEAM,
    "penalty_no_play": _TEAM, "penalty_on_play": _TEAM, "penalty_yards": _TEAM, "penalty_first_down": _TEAM,
}


def v2_target_context(target_name: str) -> list[str]:
    """Separate same-fit context columns for one consumed mu; paired defense outputs need no matchup column."""
    target = TARGETS_BY_NAME[target_name]
    columns = [venue_column(target.name)]
    if target.scope == "player":
        columns.append(offense_column(target.name))
    if target.player_output and target.defense_output is None:
        columns.append(opponent_defense_column(target.name))
    elif target.scope == "team" and target.defense_output is not None:
        columns.append(opponent_defense_column(target.name))
    return columns


def v2_optional_context_columns(model: str) -> list[str]:
    targets = list(dict.fromkeys(V2_MODEL_CONTEXT.get(model, [])))
    return [column for target in targets for column in v2_target_context(target)]


def context_columns(model: str, semantics: str | None = None) -> list[str]:
    targets = MODEL_CONTEXT.get(model, [])
    ridge = [f"ridge_{target}_{side}_effect" for target in targets for side in ("venue", "offense") if target not in {"pass_epa", "rush_epa"}]
    if semantics == ISOLATED_MU_CONTEXT_V2:
        extra = v2_optional_context_columns(model)
        replaced = {
            f"ridge_{target}_{suffix}_effect"
            for target in V2_MODEL_CONTEXT.get(model, []) for suffix in ("venue", "offense")
        }
        replaced.update(
            f"ridge_{target}_opponent_defense_effect"
            for target in V2_MODEL_CONTEXT.get(model, [])
            if TARGETS_BY_NAME[target].defense_output is not None
        )
        columns = [column for column in ridge if column not in replaced] + list(PV_COLUMNS) + list(WEATHER_COLUMNS) + extra
        optional = [column for column in columns if column.startswith("ridge_") and not column.endswith("_available")]
        optional.extend(
            f"ridge_{target}_defense_mu"
            for target in V2_MODEL_CONTEXT.get(model, [])
            if TARGETS_BY_NAME[target].player_output
        )
        flags = [context_available_column(column) for column in dict.fromkeys(optional)]
        return list(dict.fromkeys(columns + flags))
    return ridge + list(PV_COLUMNS) + list(WEATHER_COLUMNS)


def v2_model_manifest(manifest: dict) -> dict:
    """Use one V2 same-fit context column per effect in play and cascade inputs."""
    result = copy.deepcopy(manifest)
    for model, spec in result["models"].items():
        required = context_columns(model, ISOLATED_MU_CONTEXT_V2)
        targets = set(V2_MODEL_CONTEXT.get(model, []))
        replaced = {
            f"ridge_{target}_{suffix}_effect"
            for target in targets for suffix in ("venue", "offense")
        }
        retained = [column for column in spec["context_columns"] if column not in replaced]
        spec["context_columns"] = list(dict.fromkeys(retained + required))
        if spec["sidecar"].get("kind") == "cascade":
            sidecar = spec["sidecar"]
            sidecar["context_columns"] = list(dict.fromkeys(
                [column for column in sidecar.get("context_columns", []) if column not in replaced]
                + spec["context_columns"]
            ))
    result["feature_semantics"] = ISOLATED_MU_CONTEXT_V2
    return result


def _regular_completion(sidecar: dict) -> dict:
    if sidecar.get("kind") != "cascade":
        return sidecar
    cascade = sidecar.get("config", {})
    hidden_dims = list(cascade.get("hidden_dims", [64, 32, 16]))
    dropout = float(cascade.get("dropout", 0.1))
    return {
        "kind": "flat",
        "arch": {
            "hidden_dims": hidden_dims,
            "dropout": [dropout] * len(hidden_dims),
            "activation": "relu",
            "activation_negative_slope": 0.1,
            "norm": "none",
        },
        "dims": {"input_dim": 0, "output_dim": 2, "target_mode": "binary"},
        "hyperparameters": {
            "learning_rate": float(cascade.get("learning_rate", 1e-3)),
            "weight_decay": float(cascade.get("weight_decay", 0.0)),
            "batch_size": int(cascade.get("batch_size", 512)),
            "patience": int(cascade.get("patience", 10)),
            "optimizer": "adamw",
            "scheduler": "none",
            "scheduler_params": {},
            "grad_clip_norm": 5.0,
            "label_smoothing": float(cascade.get("label_smoothing", 0.0)),
            "class_weighting": "none",
        },
        "mode": "train",
    }


def snapshot(source: Path = REPO / "archive/bayesian_sim/artifacts/runtime", destination: Path = ROOT / "config",
             regular_completion: bool = False) -> dict:
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
    if regular_completion:
        models["comp"]["sidecar"] = _regular_completion(models["comp"]["sidecar"])
        models["comp"]["prior_columns"] = []
    sources = {}
    for name, path in {
        "model_feature_config.json": REPO / "archive/bayesian_sim/model_feature_config.json",
        "bayesian_decay_rates.json": REPO / "archive/bayesian_sim/bayesian_decay_rates.json",
        "ridge_best_params_unverified.json": REPO / "archive/ridge_adjusted/ridge_adjusted_weekly_bakeoff/best_params.json",
    }.items():
        if path.exists():
            shutil.copy2(path, destination / name)
            sources[name] = {"path": str(path), "sha256": digest(path)}
    manifest = {"models": models, "sources": sources, "ridge_selection_verified": False}
    write_json(destination / "model_manifest.json", manifest)
    return manifest


def main():
    import argparse
    from nflsim.paths import assert_protected, protected_hashes
    parser = argparse.ArgumentParser(description="Read-only source snapshot into the isolated experiment")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "config")
    parser.add_argument("--regular-completion", action="store_true",
                        help="Snapshot the completion classifier as a flat binary model instead of a cascade")
    args = parser.parse_args()
    before = protected_hashes()
    try:
        manifest = snapshot(destination=args.output_dir, regular_completion=args.regular_completion)
        print({key: value["sidecar"]["kind"] for key, value in manifest["models"].items()})
    finally:
        assert_protected(before)


if __name__ == "__main__":
    main()