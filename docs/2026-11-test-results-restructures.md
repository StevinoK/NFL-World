# Test results: `ccr-3f8f7ac2-restructures` refit experiments

Run locally 2026-11-26 21:29 to 2026-11-27 15:53, following `docs/2026-11-local-run-restructures.md`. All eleven
experiments ran (0-10). Each experiment is compared with a **control refit** trained the same way on the same data
and seed.

## What ran

- **Data:** the 2026-11-17 weekly refresh (inputs through 2026 week 10, with its features, `ridge_parameters.json`
  and the usage and special-teams tables rebuilt from those features), training seasons 2017-2025.
- **Arguments:** everything else follows `pipeline.sh`: tables, seasons, epochs, seed, the `team3` game-effects
  config (the live bundle's `fit.config`) and the packaging arguments.
- **Backtests:** 2026 weeks 1-11, 163 games, 200 simulations per game, seed 42, `NFLSIM_PAIRED_SEEDS=1`,
  `--backend local` with 5 workers, every `NFLSIM_*` variable cleared before each run.
- **Switches:** `NFLSIM_PREGAME_TABLES` for pregame and runlane, `NFLSIM_TIMEOUTS=nflsim/config/timeouts.json` for
  timeouts, nothing for the rest.
- **Threads:** the control refit ran alone on 6 threads. Every other refit used 3 threads while the previous
  experiment backtested.

| # | Experiment | Refit | Game effects + package | Backtest |
| --- | --- | --- | --- | --- |
| 0 | control | 28 min | 12 min | 80 min |
| 1 | temperature | none (control refit) | 10 min | 80 min |
| 2 | seedavg | 2 seeds × 24 min + 10 min average | 15 min | 80 min |
| 3 | forward | 55 min | 16 min | 80 min |
| 4 | crossfit | 201 min | 47 min | 69 min |
| 5 | dropback | 43 min | 18 min | 71 min |
| 6 | situation | 39 min | 17 min | 69 min |
| 7 | pregame | 43 min | 16 min | 69 min |
| 8 | runlane | 20 min (resumed after a fix, see below) | 13 min | 82 min |
| 9 | timeouts | 36 min | 17 min | 79 min |
| 10 | multitask | 22 min (on the control refit's rows) | 16 min | 78 min |

### Deviations from the guide

1. **Newer inputs.** `pipeline.sh`'s 2026-11-01 inputs stop at 2026 week 8. To backtest weeks 1-11, the inputs,
   features, ridge parameters and tables come from the 2026-11-17 weekly refresh. The user approved this.
2. **Packaging base.** `--base` and `--compare-bundle` use the current live bundle (`b8d92e9c…`) instead of
   `630d58…`.
3. **No cv stage.** The game-effects cv stage was skipped and `final` ran with `--choose team3`. The guide marks cv as
   optional.
4. **Fixed a bug in the runlane experiment** (`nflsim/features/training_data.py`, not committed).
   - **The failure:** the first runlane refit failed in the rush model with `Encoders require their input argument
     must be uniformly strings or numbers. Got ['NAType', 'str']`. Rushes with no charted lane carry pandas `NA` in
     `run_lane`, and the one-hot encoder rejects `NA` mixed with strings.
   - **The fix:** a lane outside the seven named lanes is now `"unknown"` in training frames. `run_location` keeps
     only rows with one of the seven lanes, the same rows as before. The lane ratings for `"unknown"` rows stay NaN,
     as before.
   - **Effect on other experiments:** the simulator always draws a real lane, so `"unknown"` appears only in
     training. No other experiment takes `run_lane` as an input, so the fix doesn't change them. The 19 lane,
     training-data and pregame tests pass.
5. **Old scoring-extras default.** This branch predates `b6c9875` on `ccr-3f8f7ac2-5qtlyb`, so every run here used
   the old default (scoring extras off). Control and experiments share that code.

Checked before trusting the comparisons:

- **Unchanged models match.** Refits that change only some models reproduce the control's other models exactly:
  same held-out log loss to every printed digit. So situation, pregame, runlane and timeouts differ from the control
  only in the models they changed.
- **Full retrains carry refit noise.** Forward, crossfit, seedavg (four models) and multitask retrain everything and
  carry refit-seed noise on top of the change. Forward is a good gauge of that noise: it changes only how
  validation is split, and its Brier moved −1.7 SE.

## Backtest against control

Paired differences over 163 games, `nflsim/outputs/restructures_scores`. Negative is better; z is the difference
divided by its standard error.

| Run | Brier (z) | Spread CRPS (z) | Total CRPS (z) | energy_points (z) | Stat-line energy (z) | Variogram (z) |
| --- | --- | --- | --- | --- | --- | --- |
| control | 0.2367 | 7.322 | 7.788 | 8.363 | 6.203 | 85.67 |
| temperature | −0.0016 (−0.5) | +0.042 (+0.7) | +0.033 (+0.3) | +0.043 (+0.6) | −0.033 (−2.7) | −0.44 (−2.4) |
| seedavg | −0.0050 (−1.6) | −0.039 (−0.8) | +0.029 (+0.4) | −0.012 (−0.2) | −0.074 (**−4.6**) | −1.31 (**−5.2**) |
| forward | −0.0110 (−1.7) | −0.081 (−0.7) | −0.046 (−0.3) | −0.054 (−0.5) | +0.005 (+0.2) | −0.21 (−0.5) |
| crossfit | −0.0128 (**−2.2**) | −0.158 (−1.5) | −0.010 (−0.1) | −0.094 (−0.9) | +0.041 (+1.4) | +0.63 (+1.4) |
| dropback | +0.0005 (+0.1) | +0.112 (+1.1) | +0.091 (+0.7) | +0.106 (+1.1) | +0.019 (+0.9) | +0.77 (+2.4) |
| situation | −0.0109 (**−2.3**) | −0.094 (−1.0) | −0.133 (−1.1) | −0.111 (−1.3) | −0.015 (−0.8) | −0.41 (−1.4) |
| pregame | −0.0027 (−0.8) | −0.000 (−0.0) | +0.001 (+0.0) | −0.005 (−0.1) | −0.026 (−1.7) | −0.54 (−2.1) |
| runlane | −0.0028 (−0.7) | +0.016 (+0.3) | +0.094 (+1.1) | +0.037 (+0.7) | −0.017 (−1.2) | −0.38 (−1.6) |
| timeouts | −0.0073 (**−2.1**) | −0.090 (−1.5) | −0.079 (−1.0) | −0.085 (−1.5) | −0.038 (−2.5) | −0.70 (−2.9) |
| multitask | −0.0180 (**−2.1**) | −0.258 (−1.4) | −0.106 (−0.5) | −0.185 (−1.2) | −0.039 (−1.0) | −0.65 (−1.0) |

Calibration, from `forecast_calibration.csv`:

| Run | Spread SD / RMSE | Spread mean z² | Total SD / RMSE | Total mean z² | Total bias | Sim home/away corr (observed) |
| --- | --- | --- | --- | --- | --- | --- |
| control | 11.49 / 12.92 | 1.278 | 13.30 / 13.74 | 1.072 | −0.38 | 0.141 (0.064) |
| temperature | 11.22 / 12.97 | 1.336 | 12.96 / 13.78 | 1.131 | +1.76 | 0.140 (0.054) |
| seedavg | 11.43 / 12.90 | 1.288 | 13.22 / 13.71 | 1.091 | −0.17 | 0.141 (0.063) |
| forward | 11.55 / 12.82 | 1.252 | 13.06 / 13.64 | 1.094 | +0.15 | 0.119 (0.064) |
| crossfit | 12.70 / 12.66 | **1.006** | 13.64 / 13.66 | **1.018** | −1.20 | **0.070** (0.075) |
| dropback | 13.69 / 13.19 | 0.959 | 14.33 / 13.90 | 0.987 | −2.53 | 0.045 (0.037) |
| situation | 11.68 / 12.86 | 1.210 | 13.29 / 13.48 | 1.050 | −0.59 | 0.125 (0.046) |
| pregame | 11.43 / 12.88 | 1.296 | 13.86 / 13.79 | 0.990 | −1.47 | 0.184 (0.064) |
| runlane | 11.52 / 12.87 | 1.290 | 13.37 / 13.92 | 1.085 | −0.41 | 0.143 (0.081) |
| timeouts | 11.48 / 12.81 | 1.250 | 13.26 / 13.60 | 1.052 | −0.57 | 0.138 (0.061) |
| multitask | 12.76 / 12.48 | 0.988 | 12.88 / 13.57 | 1.131 | +1.27 | 0.011 (0.080) |

The observed home/away correlation differs slightly between runs because it's computed over the games each run
completed with simulations.

Joint outcomes, from `nflsim/benchmarks/2026/joint_outcomes_restructures`:

| Run | Mean abs log dispersion | Mean abs corr gap | Team points bias / dispersion | Drives dispersion | Points/drive dispersion |
| --- | --- | --- | --- | --- | --- |
| control | 0.054 | 0.053 | +0.19 / 1.11 | 1.16 | 1.05 |
| temperature | 0.059 | 0.053 | −0.88 / 1.13 | 1.16 | 1.07 |
| seedavg | 0.055 | 0.056 | +0.08 / 1.11 | 1.17 | 1.06 |
| forward | 0.056 | 0.060 | −0.07 / 1.12 | 1.18 | 1.03 |
| crossfit | **0.048** | 0.055 | +0.60 / 1.04 | 1.18 | 1.00 |
| dropback | 0.052 | 0.058 | +1.26 / 0.99 | 1.12 | 0.92 |
| situation | 0.057 | 0.054 | +0.29 / 1.09 | 1.19 | 1.05 |
| pregame | 0.055 | 0.058 | +0.74 / 1.09 | 1.16 | 1.06 |
| runlane | 0.056 | 0.052 | +0.20 / 1.09 | 1.15 | 1.05 |
| timeouts | 0.055 | 0.058 | +0.28 / 1.11 | 1.12 | 1.06 |
| multitask | 0.063 | 0.067 | −0.63 / 1.05 | 1.17 | 0.99 |

## Per-model held-out log loss against control

From each refit's `<model>_fit_metrics.json`. All runs use the same held-out fifth of games, except crossfit, which
scores out-of-fold on every game. Only the models each experiment changed are listed. Negative is better.

| Experiment | Changes |
| --- | --- |
| seedavg | play −0.44%, rusher −1.10%, target −0.17%, time +0.25% |
| forward (every model) | play −0.67%, kick −1.14%, penalty_yards −1.25%, fumble +1.24%, penalty_on_play +0.88%; the rest within ±0.4% |
| crossfit (every model, out-of-fold) | play −0.70%, int −2.61%, kickoff −3.01%, def_ret −2.34%, scramble −1.22%, kick +1.76%, penalty_no_play +2.20%; the rest within ±0.9% |
| dropback (new models) | dropback 0.4076, throw 0.6905 (no control counterpart) |
| situation | **play −3.28%** (+4 inputs), **time −1.35%** (+3), comp −0.51% (+2), int −0.04%, yac −0.07% |
| pregame | play −0.31% (+2), ay −0.08% (+2), kickoff −0.10% (+1), kick +1.03% (+3) |
| runlane | **rush −1.33%** (+3), new run_location 1.8335 |
| timeouts | **play −1.91%** (+2), time −0.43% (+2) |
| multitask (heads vs control networks) | **play −3.89%**, **rusher −4.16%**, penalty_first_down −1.41%, int −0.82%; kickoff +2.62%, punt_distance +2.76%, punt +2.34%, penalty_yards +1.94%, def_ret +1.53% |

Per-play log loss with game effects on 2026 (`play_log_loss_2026.csv`) agrees:

- play: situation −3.5%, timeouts −2.0%, multitask −4.5%.
- rush: runlane −1.2%.
- time: situation −1.3%.
- multitask is worse on punts and kickoffs by 1-3%.
- dropback is worse on int (+2.6%) and rusher (+2.8%).

## Checks

- **Added inputs.** Every added input shows up in its model's input count, matching the table above. Situation's
  three weather inputs for `kick` were dropped because the control's kick model already has them. No refit refused
  an input.
- **FTN charting.** `is_screen_pass` and `is_play_action` cover 84% of 2022 plays and 96-97% of 2023-2026 plays,
  and nothing before 2022. So `pg_off_screen_rate` and `pg_off_play_action_rate` aren't empty and stayed in, but
  they carry no information for 2016-2021. The ay model they feed moved −0.08%.
- **Temperature.** Most values in `temperatures.json` are within 0.8-1.25. Two are outside: `sack` at 1.50 and
  `scramble` at 1.24. `rusher` is 0.88, the only value well below 1.
- **Crossfit holdout.** `holdout_games` is 2,364, against the control's 455: 5.2×.
- **Run lanes.** Simulated runs go 36.5% left, 28.5% middle, 35.0% right, against the real 38 / 26 / 36. The middle
  is about 2.5 points too often.
- **Multitask.** `multitask_summary.json` heads are in the log-loss table above. It shares a 256-128 trunk and trained
  for 30 epochs.
- **Switches.** None of the switch errors the guide describes appeared.

## Verdict per experiment

The decision rule is the one in `docs/2026-11-local-run.md`:

- **Helpful:** improves the calibration it targets, with no worsening of Brier, spread CRPS, total CRPS or
  energy_points beyond 1 SE.
- **Harmful:** any of those four gets worse by 2 SE or more.

No experiment is harmful by that rule.

| # | Experiment | Verdict | Why |
| --- | --- | --- | --- |
| 1 | temperature | **neutral; don't adopt** | No headline score moves beyond 1 SE. Spread and total mean z² get worse (1.28→1.34, 1.07→1.13) and total bias goes to +1.8. It also rescales trained model outputs after the fit, which conflicts with this project's no-post-hoc-calibration rule. |
| 2 | seedavg | **helpful** | Stat-line energy −4.6 SE and variogram −5.2 SE are the clearest gains in the run. Brier −1.6 SE, nothing worse. One run can't measure what it mainly targets, seed churn. |
| 3 | forward | neutral | Everything is within 1.7 SE. Useful as the refit-noise gauge. |
| 4 | crossfit | **helpful** | It fixes calibration: spread mean z² 1.28→1.01, total 1.07→1.02, home/away correlation 0.141→0.070 against the observed 0.075, and the lowest mean dispersion error. Brier −2.2 SE. Costs about 7× the refit time (201 min on 3 threads). |
| 5 | dropback | neutral; don't adopt | Its wider spreads fix mean z² but overshoot (sim SD 13.7 against RMSE 13.2). Spread CRPS and energy_points get worse by +1.1 SE, variogram by +2.4 SE, and team points bias is +1.3. Its per-play int and rusher losses get worse. |
| 6 | situation | **helpful** | All four headline scores improve, Brier −2.3 SE. Play log loss −3.3%, clock −1.4%. Spread mean z² 1.28→1.21. It changes only five models, so this is a clean comparison. |
| 7 | pregame | neutral | Headline scores are flat. Total mean z² improves to 0.99 and variogram −2.1 SE, but home/away correlation gets worse (0.184 against 0.064) and team points bias is +0.7. The kick inputs make kick log loss worse (+1.0%). |
| 8 | runlane | neutral | Rush log loss −1.3% and run lanes are close to real, but total CRPS +1.1 SE and team rushing-yards bias goes from +3.1 to +4.6. |
| 9 | timeouts | **helpful** | All four headline scores improve, Brier −2.1 SE, stat-line energy −2.5 SE, variogram −2.9 SE. Play log loss −1.9%. Changes only two models. |
| 10 | multitask | promising, mixed | It has the best Brier (−2.1 SE) and spread CRPS, and spread mean z² goes to 0.99. But home/away correlation falls to 0.011 against 0.080, total mean z² gets worse (1.13), total bias is +1.3, dispersion and correlation gaps are the worst of the run, and special-teams heads are 2-3% worse. |

Gains near 2 SE from experiments that retrain everything (crossfit, multitask) are only modestly beyond the refit
noise that forward shows. Situation and timeouts don't have that problem, because their unchanged models are
identical to the control's.

## What to combine next

One refit with **situation + timeouts inputs** (both add to `play` and `time`, and situation's `qtr` and
`half_seconds_remaining` overlap nothing in timeouts), trained with **`--cross-fit`**, then **seed-averaged** over
`play target rusher time`. Backtest it with `NFLSIM_TIMEOUTS` against this control and against crossfit alone.

- **Cost:** about 4 h for the cross-fit refit, plus 2 seeds of the four volume models (about 1 h). The seed-averaging
  step on cross-fit refits is untested.
- **Leave out:** temperature (the post-hoc rule), dropback and pregame.
- **Revisit:**
  - runlane once rushing-yards bias is understood.
  - multitask once the special-teams heads get their own capacity or loss weights, since the play and rusher gains
    are real but the trunk favors the large models.

## Follow-up: combined refit

Run 2026-11-27 16:03-20:32, built as the section above proposed.

- **Inputs:** situation and timeouts together (`play` +6 inputs, `time` +5, `comp`/`int`/`yac` +2 each), and the clock
  model trained without timeout-stopped plays.
- **Training:** `--cross-fit`, then `play target rusher time` seed-averaged over seeds 42, 43 and 44.
- **Game effects:** from cross-fit rows over 2,364 games, `team3`.
- **Backtest:** same settings as every other run, with `NFLSIM_TIMEOUTS`.

**Timing:**
- Cross-fit base refit: 153 min on 4 threads.
- Extra seeds: 18 min each, run in parallel with the base refit on 4 more threads.
- Seed averaging: 8 min.
- Game effects and packaging: 43 min.
- Backtest: 63 min on 7 workers.

**Before looking at the results, the bar for adopting it was:**
- helpful against `control` by the decision rule;
- none of Brier, spread CRPS, total CRPS or energy_points worse than crossfit, situation or timeouts alone by more
  than 1 SE.

It passes both. It has the best mean of every run on all four scores
(`nflsim/outputs/restructures_scores_combined`, control first).

| Run | Brier (z) | Spread CRPS (z) | Total CRPS (z) | energy_points (z) | Stat-line energy (z) | Variogram (z) |
| --- | --- | --- | --- | --- | --- | --- |
| control | 0.2367 | 7.322 | 7.788 | 8.363 | 6.203 | 85.66 |
| crossfit | −0.0128 (−2.2) | −0.158 (−1.5) | −0.010 (−0.1) | −0.094 (−0.9) | +0.041 (+1.4) | +0.63 (+1.4) |
| situation | −0.0109 (−2.3) | −0.094 (−1.0) | −0.133 (−1.1) | −0.111 (−1.3) | −0.015 (−0.8) | −0.41 (−1.4) |
| timeouts | −0.0073 (−2.1) | −0.090 (−1.5) | −0.079 (−1.0) | −0.085 (−1.5) | −0.038 (−2.5) | −0.70 (−2.9) |
| **combined** | **−0.0173 (−2.8)** | **−0.214 (−1.8)** | **−0.154 (−1.0)** | **−0.180 (−1.6)** | −0.031 (−1.1) | −0.83 (−1.9) |

| Run | Spread SD / RMSE | Spread mean z² | Total SD / RMSE | Total mean z² | Total bias | Sim home/away corr (observed) |
| --- | --- | --- | --- | --- | --- | --- |
| control | 11.49 / 12.92 | 1.278 | 13.30 / 13.74 | 1.072 | −0.38 | 0.141 (0.064) |
| crossfit | 12.70 / 12.66 | 1.006 | 13.64 / 13.66 | 1.018 | −1.20 | 0.070 (0.075) |
| combined | 12.76 / 12.68 | **1.002** | 13.74 / 13.48 | 0.963 | −1.79 | 0.073 (0.053) |

**Out-of-fold log loss against the control's held-out log loss**

| Model | Change |
| --- | --- |
| play | −4.36% |
| int | −2.65% |
| time | −1.46% |
| comp | −0.47% |
| target | −0.39% |
| rusher | −0.38% |
| yac | +0.19% |

The seed-averaged networks' own `*_fit_metrics.json` values (play −4.64%, rusher −2.30%) are scored on the held-out
fifth, which the cross-fit full-data networks trained on. So they aren't clean held-out numbers.

**What still needs work:**
- **Team rushing yards:** bias grows from +3.1 to +7.6 per team-game. Crossfit alone was +2.0, situation +2.7,
  timeouts +4.3.
- **Drives and points:** drives bias is +0.54 (control +0.14) and team points bias is +0.89 (control +0.19).
- **Dispersion and correlation:** mean absolute log dispersion is 0.059 (control 0.054) and mean correlation gap is
  0.058 (0.053).
- **Total spread:** total mean z² is 0.96, slightly too wide.
- **Seed averaging's lift is smaller here.** Stat-line energy improves −1.1 SE, against seedavg alone's −4.6 SE.

**Verdict:** adopt as the next candidate refit. It is the only run that improves all four headline scores at once
and fixes spread calibration. Before it goes live, look into the rushing-yards bias, probably from the clock model
trained without timeout stoppages (more plays per game), and check plays per game against real.
