# Local run: test branch `ccr-3f8f7ac2-restructures`

Instructions for a Claude Code session (or a person) on the machine with the bundles. This branch builds on
`ccr-3f8f7ac2-5qtlyb` (run `docs/2026-11-local-run.md` first; its scoring setup and ground rules carry over) and adds
the changes that need new networks: cross-fitting, temperature scaling, seed averaging of the volume models, added
inputs (situation, derived, era, kicker, tendencies), the dropback and throw models, the run-location stage and a
shared-trunk multi-head network. All of it is off unless a flag or switch asks for it; none of it has been trained
on real data yet.

Every item needs a refit, and refits move with their random seed about as much as small real changes. So each one
is compared with a **control refit** trained the same way on the same data and seed, not with the live bundle.

## Ground rules

Those of `docs/2026-11-local-run.md`, plus:

- Work on `ccr-3f8f7ac2-restructures`. Write refits to `nflsim/outputs/refits/<name>`, game effects to
  `nflsim/outputs/game_effects_<name>`, bundles to `nflsim/artifacts/bundles/<name>`, each a new folder. Never
  overwrite the live runtime or an existing refit or bundle.
- Use exactly the arguments of the live rebuild, `nflsim/outputs/live_rebuild_20261101/pipeline.sh` (player and
  team tables, inputs, seasons 2017-2025, epochs, seed, the game-effects config and packaging arguments). Change only
  what each experiment says. Read that script first and write the commands out for this machine.
- Each refit takes hours. Before starting, list the refits you will run, estimate their total time from the live
  rebuild's logs, and ask the user which to run and in what order. Run independent refits in parallel if the
  machine has room.

## The pipeline for one experiment

Shown for the control; each experiment changes the marked steps.

```powershell
$py = '.\NFL.venv\Scripts\python.exe'
$name = "control"
# 1. Refit (add the experiment's flags here)
& $py -m nflsim.models.fit_minimal --game-effects <pipeline.sh arguments> --output-dir "nflsim/outputs/refits/$name"
# 2. Game effects on the held-out games (cv is optional; use the live bundle's config, game_effects.json fit.config)
& $py -m nflsim.models.game_effects rows  --model-dir "nflsim/outputs/refits/$name" --output "nflsim/outputs/game_effects_$name" <pipeline.sh arguments>
& $py -m nflsim.models.game_effects final --model-dir "nflsim/outputs/refits/$name" --output "nflsim/outputs/game_effects_$name" --choose <config> <pipeline.sh arguments>
# 3. Package
& $py -m nflsim.models.package_game_effects_bundle --refit-dir "nflsim/outputs/refits/$name" `
  --effects "nflsim/outputs/game_effects_$name/game_effects.json" --output "nflsim/artifacts/bundles/$name" <pipeline.sh arguments>
# 4. Backtest exactly as in docs/2026-11-local-run.md (same weeks, sims, seed, NFLSIM_PAIRED_SEEDS=1), with
#    --artifact-dir nflsim/artifacts/bundles/$name and run id "$name-2026-w1-$weeks-$sims"
```

Per-model held-out log loss is in each refit's `test_fit_comparison.csv` and `<model>_fit_metrics.json` (same
held-out fifth of games in every refit), and per-play log loss with effects in the game-effects folder's
`play_log_loss_<season>.csv`.

## Experiments

| # | Name | Refit / step changes | Backtest switches |
| --- | --- | --- | --- |
| 0 | `control` | none | none |
| 1 | `temperature` | No refit: rerun step 2 `final` on the **control** refit with `--temperature` (new output folder), then package | none |
| 2 | `seedavg` | No new architecture: train the control's `play target rusher time` with two more seeds (`fit_minimal --models play target rusher time --seed <s>` into their own folders), then `python -m nflsim.models.seed_average --base-dir <control refit> --seed-dir <seed folders> --models play target rusher time --output nflsim/outputs/refits/seedavg <tables and inputs as pipeline.sh>`; then steps 2-4 | none |
| 3 | `forward` | `--forward-validation` | none |
| 4 | `crossfit` | `--cross-fit` in step 1 (about 6x the training time), and `--cross-fit` on step 2 `rows` | none |
| 5 | `dropback` | `--dropback-models` | none |
| 6 | `situation` | `--added-inputs situation.json` (below) | none |
| 7 | `pregame` | build the tables (below), then `--pregame-tables <csv> --added-inputs pregame.json` | `NFLSIM_PREGAME_TABLES=<csv>` |
| 8 | `runlane` | `--pregame-tables <csv> --run-location-model --added-inputs runlane.json` | `NFLSIM_PREGAME_TABLES=<csv>` |
| 9 | `timeouts` | `--time-without-stoppages --added-inputs timeouts.json` | `NFLSIM_TIMEOUTS=nflsim/config/timeouts.json` |
| 10 | `multitask` | No new rows: `python -m nflsim.models.multitask --base-dir <control refit> --output nflsim/outputs/refits/multitask <tables and inputs as pipeline.sh>`, then steps 2-4 | none |

Added-input files (JSON, `{model: [inputs]}`; anything else is refused):

```json
situation.json  {"play": ["qtr", "half_seconds_remaining", "receive_2h_ko", "diff_time_ratio"],
                 "time": ["qtr", "half_seconds_remaining", "diff_time_ratio"],
                 "comp": ["distance_to_sticks", "air_is_zero"], "int": ["distance_to_sticks", "air_is_zero"],
                 "yac": ["distance_to_sticks", "air_is_zero"],
                 "kick": ["weather_wind_speed", "weather_temp_delta", "weather_outdoor"]}
pregame.json    {"play": ["pg_league_pass_rate", "pg_league_go_rate_fg_zone"],
                 "kick": ["pg_off_kicker_fg_oe", "pg_league_fg_make_40_49", "pg_league_fg_make_50_plus"],
                 "kickoff": ["pg_league_ko_touchback"],
                 "ay": ["pg_off_screen_rate", "pg_off_play_action_rate"]}
runlane.json    {"rush": ["run_lane", "pg_run_lane_off_ypc", "pg_run_lane_def_ypc"]}
timeouts.json   {"play": ["posteam_timeouts_remaining", "defteam_timeouts_remaining"],
                 "time": ["posteam_timeouts_remaining", "defteam_timeouts_remaining"]}
```

Drop any input the refit's model already has (check `reduced_model_manifest.json` of the control) or whose model
the refit doesn't train. Weather inputs to `kick` only count if the kick model doesn't already have them.

**Pregame tables** (experiments 7 and 8), from the same play-by-play the refit trains on, covering every
backtested 2026 week (the table adds the week after the last one played):

```powershell
& $py -m nflsim.features.pregame_tables --pbp-dir <folder of pbp_<season>.parquet> --seasons 2016-2026 `
  --output nflsim/outputs/pregame_tables/pregame_tables.csv
```

The bundle inputs hold one `pbp.parquet`; if there is no per-season folder, save that file's seasons as
`pbp_<season>.parquet` in a scratch folder first (only the columns listed in `PBP_COLUMNS` are needed). The
screen and play-action rates need FTN charting columns (`is_screen_pass`, `is_play_action`, merged by
`load_pbp_data` for recent seasons); report if they come out empty and drop those two inputs.

## Checks while running

- Step 1 prints each model's rows and inputs; added inputs must appear in the model's input count. A refit
  that refuses an added input prints why.
- The simulator refuses to run a bundle whose models take timeouts without `NFLSIM_TIMEOUTS`, or pregame inputs
  without `NFLSIM_PREGAME_TABLES`, or `NFLSIM_TIMEOUTS` with a clock model trained without
  `--time-without-stoppages`. Those errors mean a switch is missing, not a bug.
- Experiment 4: `game_effects final` should report `holdout_games` about five times the control's.
- Experiment 1: `temperatures.json`; values far from 1 (below 0.8 or above 1.25) are worth reporting.
- Experiment 8: the backtest's `backtest_2025_attempt_diagnostics.csv` has `run_lane_<lane>` counts per game;
  real 2021-25 non-scramble runs go 38% left, 26% middle, 36% right.
- Experiment 10: `multitask_summary.json` holds each head's held-out log loss next to the control's.

## Score

As in `docs/2026-11-local-run.md` step 5, with `control` first (the reference) and every experiment's run:

```powershell
$labels = "control", "temperature", "seedavg", "forward", "crossfit", "dropback", "situation", "pregame", "runlane", "timeouts", "multitask"
```

Same decision rule as that guide. Also tabulate, per experiment, the per-model held-out log loss against the
control's for the models it changed (from the refit summaries), since per-play gains of 0.5-1% move the backtest
by less than its noise.

## Write it up

`docs/2026-11-test-results-restructures.md`: what ran (and how long each refit took), the backtest table against
`control`, the per-model log-loss table, the checks above, a verdict per experiment, and which to combine into one
refit next. Commit only that file and push to `ccr-3f8f7ac2-restructures`.
