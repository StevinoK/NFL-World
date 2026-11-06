from pathlib import Path

import modal

from ridge_game_sim.paths import APP_NAME, REPO, ROOT, VOLUME_NAME
from ridge_game_sim.modal_compat import artifact_mount

app = modal.App(APP_NAME)
artifact_volume = modal.Volume.from_name(VOLUME_NAME)
image = modal.Image.debian_slim(python_version="3.11").pip_install_from_requirements(str(REPO / "requirements-modal.txt"))
image = image.pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
for name in ("game_sim_fantasy.py", "torch_game_simulation_runtime.py", "model_architecture.py", "ngmm_model.py", "project_paths.py", "cascade_trainer.py", "check_runtime_artifacts.py"):
    image = image.add_local_file(str(REPO / name), f"/root/{name}")
for name in ("__init__.py", "paths.py", "bundle.py", "runtime.py", "modal_worker.py", "modal_compat.py"):
    image = image.add_local_file(str(ROOT / name), f"/root/ridge_game_sim/{name}")


@app.function(name="simulate_chunk", image=image, cpu=1.0, memory=4096, timeout=1800, retries=1, scaledown_window=300, volumes={"/artifacts": artifact_mount(artifact_volume)})
def simulate_chunk(task):
    from ridge_game_sim.modal_worker import simulate_chunk as worker
    return worker({**task, "data_dir": str(Path("/artifacts/bundles") / task["bundle_hash"] / "runtime")})