# Test plan for the local pipeline, November 2026

Everything below needs the real bundles, features and simulator, so it waits for the local machine. The
evidence for each item is in `2026-11-accuracy-investigation.md` and `2026-11-forecaster-features.md`; those
tests used public play-by-play with LightGBM stand-ins, not our networks.

## 0. Set up the yardstick first

| Step | Why |
| --- | --- |
| Use a forward split for early stopping and model choice (train ≤ S−1, validate S); keep the shared crc32 fifth only for fitting game effects | The random-by-game holdout never sees drift (rule changes, 4th-down trends) |
| Record the live bundle's simulated spread and total SD | The strength-uncertainty gap below compares SDs from an older bundle with live errors |
| Add per-team drive statistics to `joint_outcomes`: points per drive, drive start EP after punts and after kickoffs, red-zone trips per drive (level and spread across teams) | Calibration targets; drive metrics don't help as per-play inputs but explain points beyond EPA |

Every item below is scored on the same yardstick: per-model held-out log loss (forward split), then the
existing paired backtest (current size; about 100 games resolve only ±0.01 Brier, so read its differences
with their intervals). Backtest metrics: (win Brier and log loss, spread and total CRPS, spread and total SD with mean squared z,
home/away score correlation, `joint_outcomes` dispersions and correlation gaps).

## 1. No retraining (sim and bundle changes only)

| Test | Change | Look for |
| --- | --- | --- |
| Team-strength uncertainty | Draw ridge strengths once per simulated game from their posterior (σ² A⁻¹ from `solve_ridge`), or a learned variance scale | Spread SD toward ~12.4 and total SD toward ~13.8 (mean squared z → 1), lower home/away correlation, player dispersions still near 1 |
| 4th-down override off | Remove the forced FG (≤ 54 yards, 4th-and-2+) and forced punt (≥ 62) in `Play.simulate`; keep only impossible kicks | Go-for-it rate in the old forced-FG zone (real 2025: 17%); score distributions |

## 2. Simulator mechanics (code plus small models)

| Test | Detail | Real rate per game (2021-26) |
| --- | --- | --- |
| Two-point tries | Decision model (score, time) plus success model | 0.44-0.57 tries, ~49% success |
| Safeties | Score 2 and free-kick when a play ends behind the goal line, instead of clipping to the 0 | 0.01-0.06 |
| Onside kicks | Decision (trailing late) plus recovery | 0.15-0.21 |
| Timeouts | Per-team counters per half, a usage policy learned from play-by-play, fed to the clock model | 7.7 used |
| PAT after return/defensive TDs | Replace the fixed +7 | |

Look for: margin key numbers, late-game comeback rates, score SD and home/away correlation.

## 3. One refit with new inputs (run as ablations per model, then together)

| Input | Models | Public-data evidence |
| --- | --- | --- |
| Era / league-environment features (pregame league 4th-down go rate over expected, FG make rate by distance band, pass rate) | All play models | 4th-down decision +10-12%, kick +1%, play call +0.7% (nflfastR uses era dummies in every model) |
| Half seconds remaining, quarter, both timeouts, `receive_2h_ko`, `Diff_Time_Ratio` | `play`, `time`, the new 4th-down/2-pt decisions | Play call +1.25% (with era), clock +1.35% |
| Kicker make rate over expected (shrunk, strictly prior), wind, cold | `kick` | FG +1.15% with era |
| Referee crew penalty rate (shrunk, strictly prior), **only if** the weekly refresh can pull the midweek assignment list (Football Zebras or Pro Football Network); nflverse fills `referee` only after games. Unannounced crews get the league-average value | The four penalty models | Game penalty MSE −3.3% |
| `distance_to_sticks`, `air_is_zero` | `comp`, `int`, `yac` | From nflfastR's CP/xYAC; cheap |
| FTN screen rate and play-action rate (offense) | `ay`, against the existing depth inputs (`adot`, deep/medium/short shares) | FTN family −0.29% on air yards, likely mostly overlapping existing inputs |
| **Run-direction stage**: a new model for run location and gap (offense tendencies, situation), then `rush` with offense/defense rush ratings for the drawn direction | new location model + `rush` | Rush yards −0.24% (z −4.4) with a drawn location; the same information as plain inputs gave −0.04% |
| Opponent-adjusted punt and kickoff field-position ratings (kicking team allowed, receiving team gained) | `punt_distance`, `punt_return`, `kickoff` | Only if step 0 shows simulated teams vary too little in drive start; public test inconclusive (−0.5% to −1.4%, small samples) |
| Team YAC over expected by pass location | `yac` | Optional; −0.14% (z −2.1), weak |

Refit with `fit_minimal --game-effects` (use `--models` for single-model ablations and `--dropped-inputs` to
remove inputs), refit `game_effects` on the held-out games, package, then backtest.

## 4. After the refit

| Test | Detail |
| --- | --- |
| Temperature scaling | One scalar per network on held-out games, with game effects integrated out; check reliability plots |
| Seed averaging | `models/seed_average.py` on the volume models (`play`, `target`, `rusher`, `time`) |

## Tested and dropped (don't rerun)

- Travel, time zones crossed, body-clock kickoff times, divisional games, surface mismatch: no held-out gain on
  game margins.
- Points per drive, points per play, drive start EP, red-zone trips, EDP as per-play inputs: no gain in the
  scrimmage models (use them as calibration targets instead).
- Team splits by pass location, run direction, third down, early down, shotgun, no-huddle: no gain.
- Next Gen Stats player tracking: helps only air yards, likely overlapping existing inputs.
- Red-zone TD rate: week-to-week correlation 0.07.
- Drawn pass-location stage: gives back the oracle's completion gain.

## Decided against

- A larger backtest (training through 2023 to backtest 2024-25): the backtest stays at its current size.
- Blending the sim with the market line.
