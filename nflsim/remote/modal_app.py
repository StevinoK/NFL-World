from pathlib import Path

import modal

from nflsim.paths import APP_NAME, REPO, ROOT, VOLUME_NAME
from nflsim.remote.modal_compat import artifact_mount

# The worker files are added to the image explicitly; mounting the whole package would upload outputs and artifacts.
app = modal.App(APP_NAME, include_source=False)
artifact_volume = modal.Volume.from_name(VOLUME_NAME)
image = modal.Image.debian_slim(python_version="3.11").pip_install_from_requirements(str(REPO / "requirements-modal.txt"))
image = image.pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
# Top-level import closure of nflsim.remote.worker (recompute after changing its imports).
WORKER_FILES = (
    "__init__.py", "paths.py", "bundle.py", "configuration.py", "semantics.py",
    "engine/__init__.py", "engine/artifact_check.py", "engine/batching.py", "engine/cascade_trainer.py", "engine/defaults.py",
    "engine/fast_inputs.py", "engine/game_effects.py",
    "engine/fantasy.py", "engine/model_architecture.py", "engine/ngmm_model.py", "engine/simulation_runtime.py",
    "features/__init__.py", "features/bayesian_history.py", "features/player_value.py", "features/weather.py",
    "remote/__init__.py", "remote/modal_app.py", "remote/modal_compat.py", "remote/worker.py",
    "ridge/__init__.py", "ridge/metrics.py", "sim/__init__.py", "sim/runtime.py",
)
for name in WORKER_FILES:
    image = image.add_local_file(str(ROOT / name), f"/root/nflsim/{name}")


@app.function(name="simulate_chunk", image=image, cpu=1.0, memory=4096, timeout=1800, retries=1, scaledown_window=300, volumes={"/artifacts": artifact_mount(artifact_volume)})
def simulate_chunk(task):
    from nflsim.remote.worker import simulate_chunk as worker
    return worker({**task, "data_dir": str(Path("/artifacts/bundles") / task["bundle_hash"] / "runtime")})