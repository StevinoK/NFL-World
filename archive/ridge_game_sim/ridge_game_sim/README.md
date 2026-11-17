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

## Ridge Simulation, Fantasy, And Synthetic Games

The Ridge workflow uses the Ridge-trained sealed bundle (`RidgeRuntime` and
`RidgeGame`), not the Bayesian-trained runtime. The existing Bayesian dashboard,
workers, model files, outputs and Modal app remain separate and unchanged. The
parallel Ridge UI is selectable by launching its own app; it writes only under
`ridge_game_sim/outputs/`. It offers single-game, full-week and synthetic home
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
& $py -m ridge_game_sim.run_dashboard
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
& $py -m ridge_game_sim.synthetic_games --backend local --season 2025 `
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
& $py -m pytest ridge_game_sim/tests -q
& $py -m ridge_game_sim.configuration
& $py -m ridge_game_sim.prepare_inputs
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

1. Start `python -m ridge_game_sim.run_dashboard` and open the Ridge Game Simulator.
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
`ridge_game_sim.weekly_refresh.refresh_weekly_bundle(season)`, but dashboard use is
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
& $py -m ridge_game_sim.build_features `
  --pbp ridge_game_sim/artifacts/inputs/pbp.parquet `
  --schedule ridge_game_sim/artifacts/inputs/schedule.parquet `
  --roster ridge_game_sim/artifacts/inputs/roster.parquet `
  --snap-counts ridge_game_sim/artifacts/inputs/snap_counts.parquet `
  --select-2024

& $py -m ridge_game_sim.train_models `
  --pbp ridge_game_sim/artifacts/inputs/pbp.parquet `
  --roster ridge_game_sim/artifacts/inputs/roster.parquet
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
& $py check_runtime_artifacts.py --artifact-dir ridge_game_sim/artifacts/runtime
& $py -m ridge_game_sim.benchmark_2025 --backend local --max-games 1 `
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
& $py -m ridge_game_sim.sync_artifacts --confirm-upload
& $py -m modal deploy ridge_game_sim/modal_app.py
& $py -m ridge_game_sim.benchmark_2025 --backend modal --max-games 2 `
  --simulations 5 --run-id modal-smoke --max-containers 2
```

App: `nfl-ridge-game-sim`. Volume: `nfl-ridge-game-sim-artifacts`. Uploads use
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
& $py -m ridge_game_sim.benchmark_2025 --backend modal --simulations 200 `
  --max-containers 32 --run-id ridge-context-2025
& $py -m ridge_game_sim.benchmark_2025 --backend modal --simulations 200 `
  --max-containers 32 --run-id ridge-context-2025 --resume
& $py -m ridge_game_sim.compare_2025 `
  --run-dir ridge_game_sim/benchmarks/2025/ridge-context-2025
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
this is not a causal A/B test. Player distribution comparison is not implemented.
Home/rest coefficients and optional teammate effects are not added as new inputs
in this initial version. Existing home/rest variables remain in the model inputs.