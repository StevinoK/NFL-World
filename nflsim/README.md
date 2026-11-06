# nflsim: Ridge + Player-Value + Weather Game Simulation

The live game simulator. Pregame team/player strengths come from recency-weighted ridge
regressions (`nflsim/ridge/metrics.py`) with a player-value design block. The torch
play models (`nflsim/engine/`) consume those strengths plus weather and game state.

Game-to-game variance comes from shared random effects (`game_effects.json` in a bundle,
`nflsim/engine/game_effects.py`). Each simulated game draws one effect shared by both teams and
one per team that shifts every play model's logits together, plus small model-specific terms.
Bundles from before October 30, 2026 use cascades on play/target/rusher/completion instead.

```powershell
# Retrain flat models + sack_yards, fit the game effects on held-out games, package a bundle
& $py -m nflsim.models.fit_minimal --game-effects --output-dir nflsim/outputs/refits/<name>
& $py -m nflsim.models.game_effects rows  --model-dir nflsim/outputs/refits/<name> --output nflsim/outputs/game_effects
& $py -m nflsim.models.game_effects cv    --model-dir nflsim/outputs/refits/<name> --output nflsim/outputs/game_effects
& $py -m nflsim.models.game_effects final --model-dir nflsim/outputs/refits/<name> --output nflsim/outputs/game_effects --choose <config>
& $py -m nflsim.models.package_game_effects_bundle --refit-dir nflsim/outputs/refits/<name> `
  --effects nflsim/outputs/game_effects/game_effects.json --output nflsim/artifacts/bundles/<bundle>
```

Player value comes from the saved weekly box plus-minus in `player_value/outputs/plus_minus_weekly_*.csv`
(read-only here). It enters as five pregame covariates: `pv_off_qb`, `pv_off_wr`, `pv_off_rb`, `pv_off_te`
(offense) and `pv_def_dl` (defense). Each group value is the sum of strictly-prior PM estimates weighted by trailing
(last 4 games, strictly before kickoff) snap share normalized to 1 within the group. Groups:
QB1 (schedule's announced starter when known), WR1-4, RB1-3, TE1-3, top 4 active DL by depth
then trailing defensive snaps. Cross-season carry uses the plus-minus 0.8 decay; unknown players get
the position's 25th-percentile replacement level.

- Every ridge target gets a `player_value` design block with a tuned `lambda_player_value`.
  Published estimates include `beta * lineup PV`; weekly betas are in `ridge_pv_coefficients.csv`.
- All torch models receive the PV columns as context; `pv_def_dl` comes from the opponent.
- Simulation recomputes PV for the actual (or customized) lineup and shifts folded estimates by
  `beta * (lineup PV - bundle PV)`, so removing a star QB lowers team strength.
- Weekly refresh fails unless plus-minus estimates run through the week before the next game.

## Package layout

| Path | Contents |
| --- | --- |
| `paths.py`, `configuration.py`, `semantics.py`, `storage.py`, `bundle.py` | Experiment root, model manifest, feature semantics, sealed-bundle validation |
| `features/` | Input preparation, weekly refresh, ridge feature tables, PV and weather covariates, retained Bayesian-history features |
| `ridge/` | Ridge metric estimation (`metrics.py`), PV bakeoff, coefficient explorer |
| `models/` | Torch model training and minimal refits |
| `engine/` | Shared torch simulation runtime, model architectures, cascade trainer, fantasy scoring, chunk execution |
| `sim/` | Ridge runtime adapter, local execution, synthetic games, 2025 benchmark and comparison |
| `remote/` | Modal app, worker and artifact upload |
| `dashboard/`, `dfs/` | Streamlit simulation dashboard and DFS optimizer |
| `artifacts/`, `outputs/`, `benchmarks/`, `training/`, `config/` | Data: bundles and inputs, run outputs, benchmark runs, training reports, frozen config snapshot |

```powershell
$py = '.\NFL.venv\Scripts\python.exe'
& $py -m nflsim.dashboard.run                                                   # simulation dashboard
& $py -m nflsim.ridge.bakeoff --quick --targets pass_epa rush_epa               # PV ridge bakeoff
& $py -m nflsim.ridge.bakeoff --quick --targets pass_epa rush_epa --no-player-value
& $py -m nflsim.features.build --pbp nflsim/artifacts/inputs/pbp.parquet `
  --schedule nflsim/artifacts/inputs/schedule.parquet --roster nflsim/artifacts/inputs/roster.parquet `
  --snap-counts nflsim/artifacts/inputs/snap_counts.parquet --select-2024
& $py -m streamlit run nflsim/ridge/explorer.py
& $py -m nflsim.dfs.run                                                         # DFS optimizer
```

The Modal app and volume keep their deployed names (`nfl-ridge-pv-game-sim`). Code moved, so
redeploy `nflsim/remote/modal_app.py` before the next Modal run.

## Ridge variants (experimental, not yet in the trained bundle)

These options exist in `nflsim/ridge/` and are evaluated by `nflsim.ridge.variant_bakeoff`. The
shipped bundle still uses the production targets. Moving to them means rebuilding features and
retraining the torch models.

- **Over-expectation targets** (`ridge/expectations.py`): each target is first predicted from game
  state (down, distance, field position, clock, score, formation, play direction) and weather
  (outdoor/roof, temperature, wind, humidity, rain and snow parsed from the PBP weather text, turf).
  A gradient-boosted model fit on earlier seasons only handles that step. The ridge then solves for
  player/team effects on `outcome - expected`, so a defense is not credited for a windy, rainy game.
- **Team-health player value**: `build_player_value_tables(..., groups=EXTENDED_GROUP_LIMITS)` adds
  `pv_off_ol`, `pv_def_lb` and `pv_def_db` to the ridge player-value block.
- **Regime decay**: `RidgeConfig.team_change_carryover` and `coach_change_carryover` down-weight
  games played before a player changed teams or the offense changed head coach (a proxy for
  play-caller; nflverse has no coordinator data).

```powershell
& $py -m nflsim.ridge.variant_bakeoff --quick        # tunes on 2024, scores 2025; writes nflsim/outputs/variant_bakeoff/
```

The sections below are the original experiment notes, updated for the new module paths.

# Isolated Ridge Game Simulation

This experiment leaves the existing source, runtime artifacts, dashboards and saved
2025 benchmark untouched. All generated files belong under this directory.

## Implemented Workflow

- Preweek ridge neutral player/team estimates replace covered Bayesian metrics.
- Metric-specific venue and offense-team effects enter selected models separately.
- Uncovered Bayesian metrics use priors frozen before each prediction season.
- The current runtime model-family manifest controls cascade, classifier, ordinal
  and latent-model retraining. The executing original training script is not imported.
- A separate runtime adapter propagates context into inference and cascade sampling.
- Historical benchmarks support explicit local or isolated Modal execution,
  deterministic chunk retries, simulation-level resume, diagnostics and reports.
- Comparison reads saved Bayesian results without changing them.

Existing `*_bayes` column names are retained for runtime compatibility. The feature
manifest identifies ridge replacements and retained Bayesian features. Binary
ridge estimates are bounded; ridge posterior uncertainty is not fabricated.

Each bundle's `input_sources.csv` says how every model input is built: ridge or
smoother, which fit's settings apply, and which settings sit at the edge of the tuning
grid. Use it rather than column names. Paired defense stats take their offense stat's
fit. `config/bayesian_decay_rates.json` is read only by research comparisons.

## Ridge Simulation, Fantasy, And Synthetic Games

The Ridge workflow uses the Ridge-trained sealed bundle (`RidgeRuntime` and
`RidgeGame`), not the Bayesian-trained runtime. The existing Bayesian dashboard,
workers, model files, outputs and Modal app remain separate and unchanged. The
parallel Ridge UI is selectable by launching its own app; it writes only under
`nflsim/outputs/`. It offers single-game, full-week and synthetic home
games, with unchanged fantasy scoring and fantasy correlation outputs. The DFS
dashboard continues to load the selected player-simulation CSV as before.
For **Single game**, enable **Customize depth chart and player inputs** to choose
players for each trained slot and edit one selected player's runtime input values;
apply or reset that player's edits before running. **Synthetic opponents** offers
the same controls for the selected home team. **Full week** intentionally uses
model-ranked lineups for each game to avoid an unwieldy page of per-game editors.
The sidebar's **Save folder** is the parent directory; each run is saved to a new
`<Save folder>/<Run ID>/` folder. Existing run folders are never overwritten.

Only games with pregame Ridge feature rows are offered. The shipped feature bundle
contains games through 2025; use the dashboard's weekly refresh to build and validate
the current season's scheduled pregame rows before forecasting an unplayed week. No
silent Bayesian fallback is used.
If a selected bundle contains a season newer than its packaged schedule/roster
snapshot, the dashboard loads that season's current schedule, roster and depth chart
from `nfl_data_py`. The game still must have matching Ridge team feature rows in
the selected bundle before it is offered.
The trained `qb_team_games_played` feature remains a known caveat: historical
training associates it with the quarterback selected by same-game pass attempts.
Its count itself is prior starts, but the player association is not strictly
pregame. The Ridge UI surfaces this caveat; don't describe the complete training
feature set as fully leakage-free until that feature is rebuilt and affected models
are retrained or the feature is removed and models are retrained.

```powershell
& $py -m nflsim.dashboard.run
```

### Run Synthetic Power Rankings

1. Reload the dashboard, choose the latest weekly bundle, and select season/week.
2. Set **Run type** to **Synthetic opponents**. All 32 teams are selected by
  default, including bye teams.
3. Choose **Local** to run without Modal upload or deployment, set simulations
  per team, and give the run a unique **Run ID**.
4. Click **Run Ridge simulations**, not **Upload selected Ridge bundle**.
  Dashboard synthetic output defaults to the repository's
  `outputs/simulations/synthetic_home_week_<week>/<Run ID>/` directory.
5. For **Modal**, first upload the selected bundle with explicit confirmation.
  Repeated uploads verify existing files and upload only missing files; conflicting
  remote contents are rejected without overwrite. Verification can take time.
  Deploy the latest Ridge worker code after runtime code changes, then select
  **Modal** and click **Run Ridge simulations**. Uploading alone does not deploy
  code or run simulations.

Equivalent bounded synthetic CLI run (local, one home team, five simulations):

```powershell
& $py -m nflsim.sim.synthetic_games --backend local --season 2025 `
  --week 1 --teams PHI --simulations 5 --parallelism 1 --run-id ridge-syn-smoke
```

Synthetic rows use player/team history strictly before kickoff, set the
synthetic-offense effect to zero, and play at a neutral site with zero venue
effects for both teams. Home/away model inputs use the midpoint of their trained
encodings; home/away output labels identify teams only. The scheduled roof is retained.
Synthetic selection includes every team in the season schedule, including away
and bye teams. Bye teams use the selected week's Sunday as their feature cutoff,
their latest available pregame features and roster, and an outdoor neutral site.
Outputs (game scores, fantasy-scored player rows, summary and run manifest) are
written under `outputs/synthetic/<run-id>/`. Modal requires the Ridge app and
bundle to have been explicitly deployed/uploaded first.

## Validation Status

The automated tests exercise synthetic historical feature generation, small
classifier/ordinal/latent/cascade training, checkpoint loading, all-model synthetic
inference, the real simulation kernel with synthetic weights, worker retry/resume,
and HTML report generation. Full historical input download, real-data feature
generation, complete training, deployment and paid Modal execution have not been
run. Passing tests do not establish model quality or historical coverage.

FTN interception-worthy and route fields, actual stadium identity, historical
roster/depth information and joined snap counts are required. Unsupported input
schemas fail explicitly. Sparse metrics exclude unavailable warm-up context from
the affected training model; inspect eligibility counts and fallback diagnostics.
The active-season weekly updater accepts both the older `season/week/depth_team`
depth schema and the current `dt/team/pos_abb/pos_rank` schema. A fresh historical
bootstrap still requires historical depth/roster-rank data; a current-only depth
snapshot cannot reconstruct past weekly lineup slots.

## Local Commands

Run from the repository directory using its selected environment. In PowerShell:

```powershell
$py = '.\NFL.venv\Scripts\python.exe'
& $py -m pytest nflsim/tests -q
& $py -m nflsim.configuration
& $py -m nflsim.features.prepare_inputs
```

Initial input preparation downloads 2016 through the current season PBP, schedule, rosters/depth/injuries and
joined snap counts into `artifacts/inputs/`. It uses an experiment PBP cache;
external data-provider libraries may also use their own standard network caches.
These are preparation commands, not model training or Modal execution.

## Weekly Coming-Week Workflow (Dashboard)

After the initial bundle has been trained and validated, the normal weekly refresh
does **not** retrain models. It updates active-season PBP/schedule/roster/snap-count
inputs, generates Ridge pregame rows for played and scheduled games, and publishes
a new immutable bundle that reuses the validated model weights and 2024-selected
Ridge parameters.

1. Start `python -m nflsim.dashboard.run` and open the Ridge PV Game Simulator.
2. Expand **Weekly data refresh**, select the active NFL season, and click **Refresh
  data and Ridge features**. This requires network access to the configured NFL
  data providers and may take several minutes. Existing bundles are not overwritten.
  Current-season PBP is fetched fresh; prior seasons remain frozen from the previous
  input snapshot. Scheduled future rows use only observations from before their week.
3. The dashboard switches to the new hash-addressed bundle after it validates.
  If a refresh stopped after publishing but before switching the UI, select the
  newest **Weekly Ridge bundle** from the bundle selector. Confirm the selected
  season/week appears. Only games with pregame rows for both teams are offered; a
  missing row is not substituted with Bayesian data. The schedule and roster are
  loaded from the bundle's weekly input snapshot, or fetched from the provider if
  the selected feature bundle is newer than its packaged schedule.
4. Select **Single game** or **Full week**, choose simulation count, and select
  **Local** for a smoke check or **Modal** for remote parallel execution. Choose
  the Ridge Modal app only; the existing Bayesian app remains separate.
5. For Modal, expand **Modal bundle upload**, explicitly check the upload
  authorization box, then click **Upload selected Ridge bundle**. This uploads to
  the isolated Ridge volume; it neither deploys code nor starts simulations.
  The Ridge Modal app must have been deployed once. After a bundle upload, select
  Modal and run the desired Ridge game/week.
6. Select **Synthetic opponents** for home-versus-synthetic games. The synthetic
  offense context is neutral, venue context matches the home stadium, and baseline
  player/team values use rows strictly before kickoff.

The same refresh is available programmatically via
`nflsim.features.weekly_refresh.refresh_weekly_bundle(season)`, but dashboard use is
recommended for weekly operation. Refresh downloads only the selected active
season and preserves earlier input seasons. If historical 2016-2024 inputs changed,
the dashboard refresh reproduces the 2024 selection before building features;
otherwise it reuses the selection attested by the current feature manifest.
Selection fingerprints use a fixed feature schema through 2024, so adding current-
season rows or harmless numeric dtype promotion does not trigger retuning. If the
actual 2016-2024 selection inputs change, the refresh explicitly reruns 2024
coordinate selection before publishing the new bundle.

Each dashboard run creates a unique folder under `outputs/<run-id>/` (synthetic
runs from the CLI go to `outputs/synthetic/<run-id>/`) with game rows, fantasy-scored
player rows, summaries, correlation output and a run manifest. Keep the selected
bundle hash with the result folder when comparing runs. To reproduce a run, use the
same immutable bundle, seed, simulation count and roster/order choice.

The dashboard displays the known `qb_team_games_played` historical training
association caveat. Weekly data refresh does not fix or conceal it; correcting that
feature requires a separately validated feature-definition change and retraining
affected models.

The weekly updater has unit coverage for frozen-history hashes, active-season-only
refresh, bundle lineage and both depth schemas. A full live 2026 data refresh and
candidate-bundle publication have not yet been run; inspect the dashboard status
and refresh manifest on the first real refresh before using the new week.

```powershell
& $py -m nflsim.features.build `
  --pbp nflsim/artifacts/inputs/pbp.parquet `
  --schedule nflsim/artifacts/inputs/schedule.parquet `
  --roster nflsim/artifacts/inputs/roster.parquet `
  --snap-counts nflsim/artifacts/inputs/snap_counts.parquet `
  --select-2024

& $py -m nflsim.models.train `
  --pbp nflsim/artifacts/inputs/pbp.parquet `
  --roster nflsim/artifacts/inputs/roster.parquet
```

Selection is reproduced on 2024 rather than trusting an unverified saved bakeoff
configuration. `--quick-selection` is for smoke tests, not a final comparison.
Uncovered metrics currently use fixed shared decay defaults instead of the root
fitted decay file, whose tuning cutoff is unverified. That difference is recorded
in retained-feature provenance and is another saved-baseline comparison confound.

Training defaults to 100 epochs with early stopping; set `--epochs` explicitly
for a larger budget. `--models sack --epochs 2` is a partial smoke run, not a
benchmark-ready bundle. Cascade stage 2 has its own `--stage2-epochs` control.
Training targets are restricted to 2017-2024. A bundle is sealed only after all
active models are trained and validated. Sealed bundles are immutable; rebuilding
requires a new experiment-local artifact directory.

Use `train_models --resume` with the same input paths, seed and epoch settings to
skip models already committed to the training manifest. Artifact hashes and
training provenance are checked before skipping. A model whose artifacts were
written but whose manifest commit failed is retrained. Undefined cascade AUROC
diagnostics are stored as `null` with their paths in `undefined_metrics`; nonfinite
training losses remain errors.

```powershell
& $py -m nflsim.engine.artifact_check --artifact-dir nflsim/artifacts/runtime
& $py -m nflsim.sim.benchmark_2025 --backend local --max-games 1 `
  --simulations 5 --run-id local-smoke
```

The benchmark additionally validates the complete sealed bundle, all cascade
stages, context contracts and artifact hashes. The file-existence checker alone
is not sufficient. Local execution is sequential, never an automatic fallback
when Modal fails.

## Explicit Modal Steps

These commands upload artifacts and deploy a new app. Run them only after reviewing
the trained bundle and authorizing remote execution. They are not automatic.

```powershell
& $py -m nflsim.remote.sync_artifacts --confirm-upload
& $py -m modal deploy nflsim/modal_app.py
& $py -m nflsim.sim.benchmark_2025 --backend modal --max-games 2 `
  --simulations 5 --run-id modal-smoke --max-containers 2
```

App: `nfl-ridge-pv-game-sim`. Volume: `nfl-ridge-pv-game-sim-artifacts`. Uploads use
immutable bundle-hash paths. Workers verify the requested bundle, use a cached
runtime per container and return diagnostics. Production app/volume names are
rejected. Deploy the app once during initial setup. Weekly dashboard uploads add a
new immutable hash path; the existing Ridge worker accepts that hash, so a bundle-
only upload does not require code redeployment. Use matching Modal environments
for upload, deployment and benchmark when applicable.

Older Modal SDKs without `Volume.with_mount_options` use a standard volume mount
with a warning. Workers only read immutable bundle files, but OS/mount-level
read-only enforcement is available only with an SDK supporting that API.

After a successful smoke and one-week validation:

```powershell
& $py -m nflsim.sim.benchmark_2025 --backend modal --simulations 200 `
  --max-containers 32 --run-id ridge-context-2025
& $py -m nflsim.sim.benchmark_2025 --backend modal --simulations 200 `
  --max-containers 32 --run-id ridge-context-2025 --resume
& $py -m nflsim.sim.compare_2025 `
  --run-dir nflsim/benchmarks/2025/ridge-context-2025
```

Resume requires the same bundle, code, schedule, seed and run settings. Inspect
`failures.json` and `run_manifest.json`; only fully simulated games are scored.
The expected regular season is 272 games / 54,400 simulations, not a guaranteed
coverage count. Both sides' scheduled venue identities must match frozen features.

## Output And Interpretation

Training reports are under `training/`; feature/fallback/provenance files and model
artifacts are under `artifacts/runtime/`. Each benchmark has its own directory
under `benchmarks/2025/`, containing chunk checkpoints, raw games/player statistics,
the original-style diagnostic reports, and paired comparison CSV/JSON/HTML.

Comparison uses common game IDs, reports exclusions and simulation counts, and
includes paired bootstrap intervals. Ridge-plus-context changes both estimator
and feature set. Saved-baseline training/runtime provenance is not established;
this is not a causal A/B test.

Player and team stat distributions, and their dependence, are scored by
`nflsim.sim.joint_outcomes` (one `--run label=dir[,dir...]` per bundle; runs over
different weeks are pooled):

```powershell
& $py -m nflsim.sim.joint_outcomes `
  --run old=nflsim/benchmarks/2025/baseline-630d58-2025-w1-2-200 `
  --run new=nflsim/benchmarks/2025/game-effects-13063f-2025-w1-2-200
```

It writes `benchmarks/2025/joint_outcomes/`: `marginal.csv` (bias, CRPS, PIT
dispersion: 12·var(PIT) is 1 when calibrated, above 1 the sims are too narrow,
below 1 too wide), `dependence.csv` (mean within-game rank correlation of the sims
vs the observed correlation of PIT normal scores across team-games, and the log-score
gain of "both over the sim median" events over independent marginals) and
`priorities.csv` (gaps ranked by game-bootstrap standard errors). Each call appends a
row to `benchmarks/2025/joint_outcomes_history.csv` for tracking.

Usage shares (`rush_rate`, `deep`, `medium`, `short`) come from the availability-aware
estimator in `nflsim.features.usage`. It is a softmax over the players on the field, with a
draft-pick prior and a random walk over weeks. `tune` fits hyperparameters on 2017-22, scores
2023-25 and writes `outputs/usage/`. `write --bundle <dir> --output <dir>` replaces the columns
in a `player_stats_wide.csv`. Skill slots for training, backtests and the dashboard all come from
`nflsim.features.slots.rank_slots`; empty slots get zero usage shares.
Home/rest coefficients and optional teammate effects are not added as new inputs
in this initial version. Existing home/rest variables remain in the model inputs.
## Seed-averaged models

A network's team-level predictions move with its random seed about as much as with a real model change.
`python -m nflsim.models.seed_average` averages several seeds' probabilities and fits one network of the same
shape to the average, so the sim still runs one network per decision. The output starts as a copy of the base refit
and packages like any other refit. Each model's `{model}_seed_average.json` records its seeds.

## NFL Models app

`nflsim/dashboard/hub.py` puts every dashboard under one app with tabs across the top; the **NFL Models** desktop shortcut opens it.

- **Simulate:** Game Simulator; Season QB Experiments, which compares the rest of a season under a new starter through
  `nflsim.sim.season.run_experiment` (the same code as `python -m nflsim.sim.season --move`).
- **Results:** Weekly Matchups, for saved full-week runs against the market; Synthetic Report, for saved runs against the synthetic opponent.
- **Tools:** DFS Optimizer, Ridge Graph Builder, Player Estimates.

The individual dashboards still run on their own through `nflsim/dashboard/launch.py`.

Bundles show readable names from `config/bundle_notes.json`; a weekly refresh inherits its parent's name.

Upcoming outdoor games get kickoff temperature and wind from Open-Meteo (`nflsim/features/forecast_weather.py`), saved once a day under
`outputs/weather`. Forecasts reach about 16 days ahead; later games, domes and retractable roofs still count as unknown or indoor weather as
before. Synthetic runs leave the forecast out, so rankings compare fairly from week to week.
