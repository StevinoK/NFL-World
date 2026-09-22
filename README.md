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