from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from torch_game_simulation_runtime import HierarchicalLatentGameModel, SimulationRuntime


CONFIG_TO_RUNTIME_MODEL = {
    "air_yards": "ay",
    "completion": "comp",
    "play_type": "play",
    "target_selection": "target",
    "rusher_selection": "rusher",
}


def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def config_hierarchical_map(config: Dict[str, object]) -> Dict[str, Dict[str, int]]:
    models = config.get("models", {}) if isinstance(config, dict) else {}
    mapping: Dict[str, Dict[str, int]] = {}
    if not isinstance(models, dict):
        return mapping

    for config_model, runtime_model in CONFIG_TO_RUNTIME_MODEL.items():
        cfg = models.get(config_model, {})
        if not isinstance(cfg, dict):
            continue
        if not bool(cfg.get("use_prior_game_state", False)):
            continue
        mapping[runtime_model] = {
            "latent_dim": int(cfg.get("latent_dim", 0) or 0),
            "enabled": 1,
        }
    return mapping


def collect_checks(runtime: SimulationRuntime, config_hier: Dict[str, Dict[str, int]]) -> Tuple[List[str], List[str]]:
    findings: List[str] = []
    errors: List[str] = []

    for model_key, cfg in config_hier.items():
        model = runtime.model_dict.get(model_key)
        prior_proc_key = f"{model_key}_prior"
        prior_proc_exists = prior_proc_key in runtime.processor

        is_hier = isinstance(model, HierarchicalLatentGameModel)
        latent_dim = model.latent_dim if is_hier else 0
        config_latent = int(cfg.get("latent_dim", 0))

        if model_key == "target":
            embed_columns = runtime.target_embed_columns
            prior_columns = runtime.target_prior_columns
        elif model_key == "rusher":
            embed_columns = runtime.running_embed_columns
            prior_columns = runtime.rusher_prior_columns
        else:
            embed_columns = []
            prior_proc = runtime.processor.get(prior_proc_key)
            prior_columns = [
                str(column)
                for column in getattr(prior_proc, "feature_names_in_", [])
            ]

        findings.append(
            f"{model_key}: hierarchical={is_hier}, checkpoint_latent_dim={latent_dim}, "
            f"config_latent_dim={config_latent}, embed_width={len(embed_columns)}, "
            f"prior_cols={len(prior_columns)}, prior_proc_exists={prior_proc_exists}"
        )

        if is_hier and not prior_proc_exists:
            errors.append(f"{model_key}: missing prior preprocessor '{prior_proc_key}'")
        if model_key in {"target", "rusher"} and is_hier and len(embed_columns) != latent_dim:
            errors.append(
                f"{model_key}: embed width {len(embed_columns)} does not match checkpoint latent_dim {latent_dim}"
            )
        if config_latent > 0 and is_hier and config_latent != latent_dim:
            errors.append(
                f"{model_key}: config latent_dim {config_latent} does not match checkpoint latent_dim {latent_dim}"
            )

    for issue in runtime.schema_issues:
        errors.append(f"runtime: {issue}")

    # de-dup while preserving order
    deduped_errors: List[str] = []
    seen = set()
    for issue in errors:
        if issue not in seen:
            seen.add(issue)
            deduped_errors.append(issue)

    return findings, deduped_errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate hierarchical runtime schema before backtesting.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace directory containing model_feature_config.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any schema issue is found.",
    )
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    config_path = workspace / "model_feature_config.json"

    config = load_config(config_path)
    hier_models = config_hierarchical_map(config)
    runtime = SimulationRuntime(strict_schema=False)

    findings, errors = collect_checks(runtime, hier_models)

    print("Hierarchical schema diagnostics")
    print(f"workspace: {workspace}")
    print("summary:")
    for line in findings:
        print(f"  - {line}")

    if errors:
        print("issues:")
        for issue in errors:
            print(f"  - {issue}")
    else:
        print("issues:\n  - none")

    if args.strict and errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
