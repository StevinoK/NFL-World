# Football Modeling

Private NFL modeling and simulation workspace. Source, configuration, tests, and documentation are tracked in Git. The game simulator's trained models and feature tables live in `artifacts/runtime/` and are tracked with Git LFS.

## First-Time Setup

Install Git, Git LFS, and Python 3.9 or the Python version documented for the workflow you are running. Then clone the private repository and download the runtime bundle:

```powershell
git clone <your-private-repository-url>
Set-Location Modeling
git lfs install
git lfs pull
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-game-sim.txt
```

Select `.venv\Scripts\python.exe` as the VS Code Python interpreter. The included tasks then use that interpreter instead of a machine-specific path.

## Runtime Artifacts

The default runtime directory is `artifacts/runtime/`. It must contain the model checkpoints, fitted preprocessors/column definitions, model bins/architecture sidecars when applicable, plus `player_stats_wide.csv` and `team_stats_wide.csv`.

Validate the downloaded bundle before starting a dashboard:

```powershell
python check_runtime_artifacts.py
```

Set `FOOTBALL_ARTIFACT_DIR` only when intentionally using an artifact bundle outside the clone:

```powershell
$env:FOOTBALL_ARTIFACT_DIR = 'D:\FootballArtifacts\runtime'
```

Do not commit virtual environments, output folders, caches, raw downloaded data, logs, or experimental artifacts. Keep the LFS bundle to files needed by supported runtime entry points.

## Running Dashboards

```powershell
python run_game_sim_dashboard.py
python run_game_sim_results_dashboard.py
python run_dfs_dashboard.py
python run_feature_dashboard.py
```

Simulation output defaults to `outputs/simulations/` in the repository and is intentionally ignored by Git.

## Modal Game Simulations

The game simulation dashboard can execute simulation chunks locally or on Modal. Input preparation, result validation, fantasy scoring, summaries, and CSV writes remain local. Modal only runs the CPU-bound simulation chunks.

Authenticate the Modal client once:

```powershell
modal setup
```

Validate and upload the current runtime bundle to the named Modal Volume, then deploy the worker app:

```powershell
python check_runtime_artifacts.py
python sync_modal_artifacts.py
modal deploy modal_game_sim.py
```

The defaults are app `nfl-game-sim`, function `simulate_chunk`, and Volume `nfl-game-sim-artifacts`. To use another Modal environment or app name, set these before deploying and launching the dashboard:

```powershell
$env:MODAL_ENVIRONMENT = 'main'
$env:MODAL_GAME_SIM_APP = 'nfl-game-sim'
$env:MODAL_ARTIFACT_VOLUME = 'nfl-game-sim-artifacts'
```

`MODAL_ARTIFACT_VOLUME` is read during deployment and artifact synchronization. Pass matching overrides to the uploader when needed:

```powershell
python sync_modal_artifacts.py --volume nfl-game-sim-artifacts --environment main
modal deploy modal_game_sim.py --env main
```

Whenever files in `artifacts/runtime/` change, run the sync command and then redeploy the app. Redeployment replaces warm containers that may still hold the previous runtime in memory.

In the game simulation dashboard, select `Modal` independently for a single game, full week, or synthetic-opponent slate. `Max containers` limits cloud parallelism and cost; start with 8 for a game and 16 for a slate. Modal failures are reported without automatically rerunning locally. Select `Local` explicitly to use the process pool instead.

Useful operational commands:

```powershell
modal volume ls nfl-game-sim-artifacts /runtime
modal app logs nfl-game-sim
```

The existing results dashboard needs no Modal configuration because all result CSVs are still written to the selected local output folder.

Run orchestration tests without cloud credentials or model loading:

```powershell
python -m unittest tests.test_game_sim_execution
```

## Routine Sync

Before starting work on either computer:

```powershell
git pull --rebase
git lfs pull
```

After a tested change:

```powershell
git add <changed-files>
git commit -m "Describe the change"
git push
```

When runtime artifacts change, stage them normally after Git LFS has been initialized; Git handles their upload automatically:

```powershell
git add artifacts/runtime
git commit -m "Update simulation runtime artifacts"
git push
```