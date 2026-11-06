"""Modal deployment for CPU-parallel game simulation chunks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import modal

APP_NAME = os.environ.get("MODAL_GAME_SIM_APP", "nfl-game-sim")
ARTIFACT_VOLUME_NAME = os.environ.get(
    "MODAL_ARTIFACT_VOLUME",
    "nfl-game-sim-artifacts",
)
ARTIFACT_MOUNT = Path("/artifacts")
REMOTE_ARTIFACT_DIR = ARTIFACT_MOUNT / "runtime"

app = modal.App(APP_NAME)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME)

source_files = (
    "game_sim_worker.py",
    "game_sim_fantasy.py",
    "torch_game_simulation_runtime.py",
    "model_architecture.py",
    "ngmm_model.py",
    "project_paths.py",
    "cascade_trainer.py",
)
image = modal.Image.debian_slim(python_version="3.11").pip_install_from_requirements(
    "requirements-modal.txt"
)
image = image.pip_install(
    "torch==2.8.0",
    index_url="https://download.pytorch.org/whl/cpu",
)
for source_file in source_files:
    image = image.add_local_file(source_file, f"/root/{source_file}")


@app.function(
    name="simulate_chunk",
    image=image,
    cpu=1.0,
    memory=4096,
    timeout=30 * 60,
    retries=1,
    max_containers=32,
    scaledown_window=300,
    volumes={ARTIFACT_MOUNT: artifact_volume.with_mount_options(read_only=True)},
)
def simulate_chunk(task: Dict[str, object]) -> Dict[str, object]:
    from game_sim_worker import simulate_chunk_task

    remote_task = {**task, "data_dir": str(REMOTE_ARTIFACT_DIR)}
    return simulate_chunk_task(remote_task)