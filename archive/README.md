# Archive

Earlier approaches kept for reference. Nothing here is imported by `nflsim/` or `player_value/`,
except that `nflsim` reads a few frozen files for provenance checks and config snapshots
(`bayesian_sim/artifacts/runtime/`, `bayesian_sim/backtest_2025_*`, `ridge_adjusted/ridge_adjusted_weekly_bakeoff/best_params.json`,
`ridge_game_sim/ridge_game_sim/artifacts/runtime/`).

Shared modules that these scripts imported from the old repository root now live in `nflsim/engine/`
and `nflsim/features/` (for example `torch_game_simulation_runtime.py` is `nflsim/engine/simulation_runtime.py`).
The scripts here still use the old flat imports, so to run one as it was, check out commit `58c084c`
(the last commit before the cleanup).

| Folder | What it was |
| --- | --- |
| `bayesian_sim/` | Original Bayesian-feature torch simulator: `Torch Modeling.py` training, Bayesian game-sim and results dashboards, 2025 backtest and 2026 forward runs, its LFS runtime bundle and Modal app |
| `ridge_game_sim/` | First ridge simulator without player value (superseded by `nflsim/`) |
| `ridge_adjusted/` | Ridge-adjusted metric bakeoffs and explorer that preceded `nflsim/ridge/` |
| `rolling_features/` | Tests for the weighted-rolling and historical feature estimators (the modules themselves moved to `nflsim/features/`) |
| `play_type/` | Play-type classifiers (armed, cascade, NGMM) and hyperparameter tuning experiments |
| `yac_ordinal/` | Ordinal/CORN/CONDOR YAC model sweeps |
| `point_spread/` | Point-spread model and Massey diagnostics |
| `dfs/` | Scratch DFS script (the optimizer itself moved to `nflsim/dfs/`) |
| `nflsim_bundles/` | `nflsim` bundles superseded on 2026-11-02 when `pv-live-20261101` became `nflsim/artifacts/runtime`: the previous live weekly bundle `630d58…`, the old base runtime (`runtime-f1a394`), earlier weekly refreshes and candidate bundles. Benchmarks under `nflsim/benchmarks/` still name their original paths |
| `plus_minus_legacy_outputs/` | Old Bayesian plus-minus plots, iteration exports and QB refit diagnostics |

Removed from the working tree (still in git history): `play_type_training.csv` (193 MB), `backtest_2025_report.html`,
`backtest_2025_nan_input_log.csv`, `smoke_xgb_*_per_play.csv` and the fingerprinted
`NFL PM/box_model_*.joblib` caches, which are regenerated on the next plus-minus run.
