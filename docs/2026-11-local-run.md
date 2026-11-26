# Local run: test branch `ccr-3f8f7ac2-5qtlyb`

Instructions for a Claude Code session (or a person) on the machine with the bundles. The branch adds bug fixes
that are always on, a few experiments that are off unless an environment variable switches them on, and new
scoring outputs. None of it has been backtested yet. The job: run the existing backtest once per configuration,
score the runs against each other, and write the results up.

Background: `docs/2026-11-test-plan.md` (its last section lists what the branch implements) and
"Experiment switches" in `nflsim/README.md`.

## Ground rules

- Work on branch `ccr-3f8f7ac2-5qtlyb`. Don't merge into or push to `main`.
- Don't modify, re-seal or overwrite the live bundle (`nflsim/artifacts/runtime`) or any existing bundle. Don't
  upload anything to Modal or deploy the Modal app.
- Use `--backend local` for every backtest. The experiment switches are environment variables, and Modal workers
  don't see them. Local runs use a process pool, whose workers inherit the variables.
- Give every run the same season, weeks, simulation count and seed, and set `NFLSIM_PAIRED_SEEDS=1` for all of
  them, so differences come from the change and not the draws.
- Clear every `NFLSIM_*` variable between runs (`Remove-Item Env:NFLSIM_* -ErrorAction SilentlyContinue`), and
  check them with `Get-ChildItem Env:NFLSIM_*` before each run.
- If a step fails, stop and report the error with its output. Don't work around a failing check by editing code
  unless it is plainly a path or environment problem on this machine; note anything you change.
- Before anything that will take more than about three hours (refits in step 6), ask the user.

## 0. Setup

```powershell
git fetch origin
git checkout ccr-3f8f7ac2-5qtlyb
git pull
$py = '.\NFL.venv\Scripts\python.exe'
Remove-Item Env:NFLSIM_* -ErrorAction SilentlyContinue
& $py -m pytest nflsim/tests -q
```

All tests should pass here, including the four that need the archived bundles. `test_weekly_refresh`'s widget-state
test also failed in the cloud container; note it if it fails here too, and carry on. Report any other failure and stop.

Pick the window: season 2026, week 1 through the latest completed week (the live bundle trained through 2025, so
all of 2026 is out of sample). Set `$weeks` to that last week and `$cores` to the number of physical cores minus
one. Time one game first (`--max-games 1`); if a full run would take more than about four hours, use
`--simulations 100` for every run instead of 200.

```powershell
$weeks = 12          # latest completed week
$cores = 7
$sims  = 200
function Run-Backtest($id) {
  $env:NFLSIM_PAIRED_SEEDS = "1"
  & $py -m nflsim.sim.benchmark_2025 --backend local --max-containers $cores --season 2026 `
    --week-start 1 --week-end $weeks --simulations $sims --seed 42 `
    --artifact-dir nflsim/artifacts/runtime --run-id "$id-2026-w1-$weeks-$sims"
}
```

Each run writes `nflsim/benchmarks/2026/<run id>/`. Runs resume with `--resume` if interrupted.

## 1. Baseline on this branch (no switches)

```powershell
Remove-Item Env:NFLSIM_* -ErrorAction SilentlyContinue
Run-Backtest "branch-base"
```

Always-on changes in this run: sacks aren't pass attempts; end-zone interceptions are touchbacks unless returned
past the 20; missed field goals go over at the spot of the kick; return and defensive touchdowns kick a try
instead of counting 7; drives are recorded.

## 2. Same backtest on `main`, for the effect of the always-on fixes

Use a separate worktree so the branch checkout stays put, with the branch's virtual environment.

```powershell
$repo = (Get-Location).Path
git worktree add ..\NFL-World-main origin/main
Push-Location ..\NFL-World-main
Remove-Item Env:NFLSIM_* -ErrorAction SilentlyContinue
$env:NFLSIM_PAIRED_SEEDS = "1"
& "$repo\NFL.venv\Scripts\python.exe" -m nflsim.sim.benchmark_2025 --backend local --max-containers $cores `
  --season 2026 --week-start 1 --week-end $weeks --simulations $sims --seed 42 `
  --artifact-dir "$repo\nflsim\artifacts\runtime" --run-id "main-base-2026-w1-$weeks-$sims"
Pop-Location
Copy-Item -Recurse "..\NFL-World-main\nflsim\benchmarks\2026\main-base-2026-w1-$weeks-$sims" nflsim\benchmarks\2026\
```

If the worktree refuses a bundle outside its own tree, copy `nflsim\artifacts\runtime` into the same place in the
worktree (read-only use; don't change it) and point `--artifact-dir` there. If that doesn't work either, skip this
step and say so. The branch's scoring tools read `main`'s runs too (they correct the old sack counting from the
run's diagnostics).

## 3. One switch at a time

Each run: clear the variables, set one switch, run.

```powershell
# a. Play model decides kicks and punts (no forced 4th-down field goals; end-of-half kicks allowed)
Remove-Item Env:NFLSIM_* -ErrorAction SilentlyContinue
$env:NFLSIM_MODEL_PLAY_CALLS = "1"
Run-Backtest "branch-playcalls"

# b. Two-point tries, safeties and onside kicks
Remove-Item Env:NFLSIM_* -ErrorAction SilentlyContinue
$env:NFLSIM_SCORING_EXTRAS = (Resolve-Path nflsim/config/scoring_extras.json).Path
Run-Backtest "branch-extras"
```

**c. Rating uncertainty.** First build the SD table from the inputs the live bundle's features were built with.
Read `nflsim/artifacts/runtime/feature_manifest.json` for the feature semantics and the input files (the latest
weekly refresh folder under `nflsim/artifacts/weekly_refresh/` or `nflsim/artifacts/refresh_inputs/` holds
`pbp.parquet`, `schedule.parquet` and `roster.parquet`; use the one whose hashes match the manifest). It must
include every 2026 week you backtest.

```powershell
& $py -m nflsim.ridge.uncertainty --pbp <inputs>/pbp.parquet --schedule <inputs>/schedule.parquet `
  --roster <inputs>/roster.parquet --player-value nflsim/artifacts/runtime/player_value.csv `
  --feature-semantics <semantics from the manifest> --season 2026 --weeks "1-$weeks" `
  --output nflsim/outputs/strength_sd/2026_w1-$weeks.csv
```

Check the printout: about 280 rows per game; team pass EPA SDs near 0.04-0.06. If a game of the backtest has no
rows, stop and report. Then two runs, scale 0.85 and 1.0:

```powershell
foreach ($run in @(@{ label = "branch-strength085"; scale = 0.85 }, @{ label = "branch-strength100"; scale = 1.0 })) {
  $settings = "nflsim/outputs/strength_sd/$($run.label).json"
  @{ sd_table = (Resolve-Path "nflsim/outputs/strength_sd/2026_w1-$weeks.csv").Path; scale = $run.scale } |
    ConvertTo-Json | Set-Content $settings
  Remove-Item Env:NFLSIM_* -ErrorAction SilentlyContinue
  $env:NFLSIM_STRENGTH_DRAWS = (Resolve-Path $settings).Path
  Run-Backtest $run.label
}
```

## 4. Everything that helped, together

After scoring step 3 (step 5), run once more with every switch that did not hurt (see the decision rule below),
named `branch-combined`. If none helped, skip this.

## 5. Score

```powershell
$runs = "nflsim/benchmarks/2026"
$labels = "branch-base", "main-base", "branch-playcalls", "branch-extras", "branch-strength085", "branch-strength100", "branch-combined"
$runArgs = foreach ($label in $labels) {
  $dir = "$runs/$label-2026-w1-$weeks-$sims"
  if (Test-Path $dir) { "--run"; "$label=$dir" }
}
& $py -m nflsim.sim.forecast_scores --season 2026 @runArgs --output nflsim/outputs/branch_scores
& $py -m nflsim.sim.joint_outcomes --season 2026 @runArgs --output nflsim/benchmarks/2026/joint_outcomes_branch
```

`branch-base` comes first, so it is the reference for the paired differences. Outputs:

- `nflsim/outputs/branch_scores/forecast_scores.csv`: per run, mean Brier, spread and total CRPS, `energy_points`
  (both teams' points together), stat-line energy and variogram scores, each with its difference from
  `branch-base`, standard error and z.
- `nflsim/outputs/branch_scores/forecast_calibration.csv`: spread and total sim SD against RMSE, mean squared z
  (1 is calibrated, above 1 too narrow), simulated vs observed home/away correlation.
- `joint_outcomes_branch/marginal.csv`, `dependence.csv`, `priorities.csv`: player/team dispersion, bias and
  correlation gaps; drive outcomes (`team.drives`, `team.points_per_drive`, `team.red_zone_trips`,
  `team.drive_start_mean`) for branch runs only.
- Per-run `backtest_2025_attempt_diagnostics.csv`: per-game counts, including `two_point_tries`,
  `onside_tries`, `safeties` (extras run) and `play_call_kick` (play-calls run).

Sanity checks to report (real 2021-25 rates per game): two-point tries about 0.5, onside kicks about 0.19,
safeties about 0.05; real drives about 10.5 per team-game, 2.1 points per drive, 3.2 red-zone trips.

## Decision rule

From the test plan: about 100 games resolve only ±0.01 Brier, so read differences with their intervals. Call a
switch helpful if it improves the calibration it targets (mean squared z toward 1, home/away correlation toward
the observed, dispersions toward 1) without making Brier, spread CRPS, total CRPS or `energy_points` worse by more
than one standard error. Call it harmful if any of those four gets worse by two standard errors or more.

## 6. Optional refits (ask the user first; several hours each)

These need new networks, so they can't be judged from backtests alone. Follow the live rebuild pipeline in
`nflsim/outputs/live_rebuild_20261101/pipeline.sh`, with a new output name for every artifact.

- **Forward validation:** the same refit with `fit_minimal --forward-validation`. Compare per-model held-out log
  loss with the live refit's (`summary` files in each refit directory).
- **Timeouts:** refit the clock model with `fit_minimal --models time --time-without-stoppages`, refit the game
  effects and package a new bundle, then backtest it with
  `$env:NFLSIM_TIMEOUTS = (Resolve-Path nflsim/config/timeouts.json).Path`. The simulator refuses `NFLSIM_TIMEOUTS`
  with the live clock model. Look at plays per game and points in the last two minutes of each half.

## 7. Write it up

Create `docs/2026-11-test-results.md`:

1. What ran: weeks, games, simulations per game, run ids, the machine's run time per run, anything skipped and why.
2. A table of every run against `branch-base`: Brier, spread CRPS, total CRPS, `energy_points` (mean, difference,
   z), and from the calibration table spread SD / RMSE / mean squared z, total SD / RMSE / mean squared z,
   simulated vs observed home/away correlation.
3. `main-base` vs `branch-base`: the effect of the always-on fixes, including QB1 pass attempts bias and dispersion.
4. Drive outcomes from `joint_outcomes` for the branch runs: dispersion and bias of drives, points per drive,
   red-zone trips and drive start.
5. The sanity-check counts above for the extras and play-calls runs.
6. A verdict per switch (helpful / neutral / harmful by the decision rule), and which to turn on by default.

Then commit only that file (benchmark and output folders are gitignored) and push to `ccr-3f8f7ac2-5qtlyb`:

```powershell
git add docs/2026-11-test-results.md
git commit -m "Backtest results for the branch's fixes and experiments"
git push origin ccr-3f8f7ac2-5qtlyb
```

Remove the worktree when done: `git worktree remove ..\NFL-World-main`.
