# Football Modeling

NFL team-strength and game-simulation workspace built around two packages:

| Package | What it does | Environment |
| --- | --- | --- |
| [`player_value/`](player_value/README.md) | Weekly player plus-minus (box plus-minus from clustered on/off EPA, with in-season updating). Writes `player_value/outputs/plus_minus_weekly_{POS}.csv`. | `NFL_PM.venv` |
| [`nflsim/`](nflsim/README.md) | Ridge-adjusted player/team strengths with player-value and weather inputs, 21 torch play models with shared game-level effects, game/week/synthetic simulation, dashboards, Modal execution. Reads the plus-minus CSVs. | `NFL.venv` |

Supporting folders:

- `data/cache/` is the shared nflverse play-by-play cache (by season).
- `docs/2026-10-model-review.md` records the October 2026 model changes and their out-of-sample results.
- `archive/` holds earlier approaches that are no longer on the main path (see [archive/README.md](archive/README.md)).

## Weekly workflow

```powershell
$pv = '.\NFL_PM.venv\Scripts\python.exe'
$py = '.\NFL.venv\Scripts\python.exe'

# 1. Update player plus-minus through the last completed week
& $pv -m player_value.plus_minus

# 2. Refresh the simulator's weekly bundle and run simulations from the dashboard
& $py -m nflsim.dashboard.run
```

## Tests

```powershell
& $py -m pytest nflsim/tests -q
& $pv -m unittest discover -s player_value/tests -t . -q
```

## Setup

Install Git, Git LFS and Python 3.9 (sim) plus the PyMC environment (player value). Then:

```powershell
git lfs install
git lfs pull
py -3.9 -m venv NFL.venv
.\NFL.venv\Scripts\pip install -r requirements-sim.txt
```

LFS only covers the archived Bayesian runtime (`archive/bayesian_sim/artifacts/runtime/`). Virtual
environments, caches, logs and run outputs are not committed.
