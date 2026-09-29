"""Validate and upload the game simulation runtime bundle to a Modal Volume."""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

from check_runtime_artifacts import check_artifacts
from project_paths import artifact_dir

DEFAULT_VOLUME_NAME = "nfl-game-sim-artifacts"
REMOTE_RUNTIME_DIR = "/runtime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=artifact_dir())
    parser.add_argument("--volume", default=DEFAULT_VOLUME_NAME)
    parser.add_argument("--environment", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_dir = args.artifact_dir.expanduser().resolve()
    if not local_dir.is_dir():
        print(f"Runtime artifact directory does not exist: {local_dir}")
        return 1

    _, missing = check_artifacts(local_dir)
    if missing:
        print("Artifact upload aborted; required runtime files are missing:")
        for item in missing:
            print(f"  - {item}")
        return 1

    volume = modal.Volume.from_name(
        args.volume,
        environment_name=args.environment,
        create_if_missing=True,
    )
    print(f"Uploading {local_dir} to Modal Volume {args.volume}:{REMOTE_RUNTIME_DIR}")
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(local_dir, REMOTE_RUNTIME_DIR)
    print("Artifact upload complete. Redeploy modal_game_sim.py before running simulations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())