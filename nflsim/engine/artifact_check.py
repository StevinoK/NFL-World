"""Verify that a local game-simulation artifact bundle is complete enough to load."""

from __future__ import annotations

import argparse
from pathlib import Path

from nflsim.engine.defaults import artifact_dir
from nflsim.engine.simulation_runtime import SimulationRuntime


def _first_existing(directory: Path, candidates: list[str]) -> str | None:
    return next((name for name in candidates if (directory / name).is_file()), None)


def check_artifacts(directory: Path) -> tuple[list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []

    for name in ("player_stats_wide.csv", "team_stats_wide.csv"):
        (present if (directory / name).is_file() else missing).append(name)

    optional = [model for model in SimulationRuntime.optional_models if (directory / f"{model}_model.pt").is_file()]
    for model in [*SimulationRuntime.models, *optional]:
        prefix = f"{model}: "
        columns = [f"{model}_columns.pkl", f"{model}_cascade_columns.pkl"]
        processor = [f"{model}_preprocessor.pkl"]
        checkpoint = [f"{model}_model.pt", f"{model}_cascade_arch.json"]
        if model == "scramble":
            columns.insert(0, "scramble_classifier_columns.pkl")
            processor.insert(0, "scramble_classifier_preprocessor.pkl")
            checkpoint.insert(0, "scramble_classifier_model.pt")

        for label, candidates in (("columns", columns), ("preprocessor", processor), ("checkpoint", checkpoint)):
            match = _first_existing(directory, candidates)
            if match is None:
                missing.append(prefix + label)
            else:
                present.append(prefix + match)

    return present, missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Check game-simulation runtime artifacts.")
    parser.add_argument("--artifact-dir", type=Path, default=artifact_dir())
    args = parser.parse_args()
    directory = args.artifact_dir.expanduser()

    if not directory.is_dir():
        print(f"Runtime artifact directory does not exist: {directory}")
        return 1

    present, missing = check_artifacts(directory)
    print(f"Runtime artifact directory: {directory}")
    print(f"Available artifacts: {len(present)}")
    if missing:
        print("Missing artifacts:")
        for item in missing:
            print(f"  - {item}")
        return 1

    print("Runtime artifact check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())