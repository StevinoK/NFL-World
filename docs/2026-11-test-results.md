# Test results: branch `ccr-3f8f7ac2-5qtlyb` (local run, 2026-11-26)

Followed `docs/2026-11-local-run.md` steps 0–5 and 7. Step 6 (refits) skipped as asked.

## 1. What ran

- Window: 2026 weeks 1–11, 163 games, 200 simulations per game, seed 42, `--backend local --max-containers 5`,
  `NFLSIM_PAIRED_SEEDS=1`, every other `NFLSIM_*` variable cleared before each run.
- Timing check first: 1 game 52 s, 5 games 189 s (about 34 s per game with 5 workers on 6 physical cores).
- Tests (step 0): 421 passed, 2 skipped, after linking two archived bundles into the worktree (below).

| Run id | Switches | Run time |
|---|---|---|
| `branch-base-2026-w1-11-200` | none | 1 h 34 |
| `main-base-2026-w1-11-200` | none, `main` at 79ea994 | 1 h 35 |
| `branch-playcalls-2026-w1-11-200` | `NFLSIM_MODEL_PLAY_CALLS=1` | 1 h 21 |
| `branch-extras-2026-w1-11-200` | `NFLSIM_SCORING_EXTRAS=nflsim/config/scoring_extras.json` | 1 h 23 |
| `branch-strength085-2026-w1-11-200` | `NFLSIM_STRENGTH_DRAWS`, scale 0.85 | 1 h 18 |
| `branch-strength100-2026-w1-11-200` | `NFLSIM_STRENGTH_DRAWS`, scale 1.0 | 1 h 11 |
| `branch-combined-2026-w1-11-200` | play calls + extras + strength 0.85 | 1 h 04 |

Differences from the procedure, all machine or path issues:

- **Worktrees, not a checkout.** The local branch tracks bundles, runtime artifacts and benchmarks (about 77k files)
  that this code-only branch does not, so `git checkout` would have deleted them. The branch ran in a worktree
  (`..\NFL-World-ccr`) and `main` in another (`..\NFL-World-main`, removed afterwards).
- **Bundle.** `nflsim/artifacts/runtime` on this machine is an older bundle (c4f9e2ae, schedule scored through
  week 8). All runs used the live bundle `b8d92e9c…` (QB-aware passing inputs, scored through week 10), copied
  unchanged into each worktree because `nflsim.paths.inside` rejects paths outside the tree.
- **Archived bundles for tests.** Four tests read `archive/nflsim_bundles/pv-isolated-mu-neutral-pilot-r2-2025`
  and `archive/bayesian_sim/artifacts/runtime`; linked into the worktree as junctions, then all passed.
- **Strength SD table** (`nflsim/outputs/strength_sd/2026_w1-11.csv`): built from the weekly-refresh inputs whose
  hashes match the bundle manifest. 50,460 rows, 259–366 per game (mean 310; the doc expected about 280).
  Team pass EPA SD: offense 0.071, defense 0.047 (the doc expected 0.04–0.06, so offense runs high).
- **First strength 0.85 attempt** failed on its first chunk with `JSONDecodeError: Invalid \escape`: my own driver
  wrote the settings file with unescaped Windows backslashes. Rewritten with `json.dumps`, folder deleted, rerun.

## 2. Every run against `branch-base`

Scores (mean; difference from branch-base; z = difference / paired SE; positive = worse):

| Run | Brier | Spread CRPS | Total CRPS | energy_points |
|---|---|---|---|---|
| branch-base | 0.2201 | 7.125 | 7.534 | 8.152 |
| main-base | +0.0015 (1.05) | +0.023 (0.98) | +0.012 (0.41) | +0.024 (1.20) |
| branch-playcalls | +0.0004 (0.15) | +0.070 (1.59) | −0.063 (−1.01) | −0.023 (−0.54) |
| branch-extras | +0.0000 (0.01) | +0.038 (0.73) | +0.004 (0.06) | +0.020 (0.51) |
| branch-strength085 | +0.0026 (0.63) | +0.007 (0.10) | +0.246 (0.87) | +0.125 (0.76) |
| branch-strength100 | **+0.0119 (2.71)** | +0.100 (1.29) | +0.583 (1.70) | +0.355 (1.81) |
| branch-combined | +0.0052 (1.16) | +0.051 (0.62) | **+0.749 (2.05)** | +0.412 (1.95) |

Stat-line energy / variogram z: main 0.43 / −0.42, playcalls 1.81 / 0.74, extras 0.63 / −0.83,
strength085 4.60 / 4.72, strength100 6.03 / 6.30, combined 4.98 / 5.00.

Calibration (mean squared z: 1 is calibrated, above 1 too narrow):

| Run | Spread SD / RMSE / msz | Total SD / RMSE / msz | Total bias (actual − sim) | Home/away corr sim vs obs |
|---|---|---|---|---|
| branch-base | 11.51 / 12.59 / 1.18 | 12.69 / 13.42 / 1.12 | +1.97 | 0.096 vs 0.055 |
| main-base | 11.57 / 12.68 / 1.19 | 12.72 / 13.40 / 1.11 | +1.70 | 0.093 vs 0.049 |
| branch-playcalls | 10.86 / 12.62 / 1.35 | 12.89 / 13.27 / 1.07 | +0.97 | 0.169 vs 0.049 |
| branch-extras | 11.64 / 12.63 / 1.19 | 12.90 / 13.34 / 1.09 | +1.61 | 0.101 vs 0.049 |
| branch-strength085 | 12.45 / 12.65 / 1.02 | 14.08 / 13.80 / 0.98 | −4.23 | 0.121 vs 0.039 |
| branch-strength100 | 12.53 / 12.75 / 1.04 | 14.42 / 14.38 / 1.01 | −5.67 | 0.138 vs 0.038 |
| branch-combined | 11.68 / 12.70 / 1.18 | 14.38 / 14.59 / 1.03 | −5.98 | 0.205 vs 0.049 |

The observed home/away correlation differs slightly between runs (0.038–0.055) though all score the same 163
games; that comes from the scoring tool, not the runs, and the gaps are small next to the simulated values.

## 3. `main-base` vs `branch-base` (always-on fixes)

The branch is slightly better on all four headline scores, each by about 1 SE (Brier 1.05, spread 0.98,
energy_points 1.20); not conclusive on 163 games but consistently in the branch's favour. Calibration is
unchanged (spread msz 1.19 vs 1.18, total 1.11 vs 1.12). QB1 pass attempts are the same: bias +1.35 on both
(sim 33.2 vs actual 31.8), dispersion 1.115 (branch) vs 1.119 (main), SE 0.055.

## 4. Drive outcomes (branch runs; `team.*` from joint_outcomes, bias = sim − actual)

Actual per team-game: 10.90 drives, 2.10 points per drive, 3.19 red-zone trips, drive start 30.5.

| Run | Drives bias / disp | Pts per drive bias / disp | Red-zone bias / disp | Drive start bias / disp | Points bias |
|---|---|---|---|---|---|
| branch-base | +0.06 / 1.16 | −0.10 / 1.03 | −0.06 / 0.98 | −1.13 / 0.99 | −0.99 |
| branch-playcalls | −0.08 / 1.16 | −0.02 / 1.05 | +0.01 / 0.99 | −0.96 / 0.99 | −0.49 |
| branch-extras | +0.06 / 1.15 | −0.09 / 1.02 | −0.04 / 0.98 | −1.05 / 1.00 | −0.81 |
| branch-strength085 | −0.49 / 1.11 | +0.32 / 0.88 | +0.17 / 0.97 | −1.63 / 1.01 | +2.11 |
| branch-strength100 | −0.55 / 1.10 | +0.41 / 0.84 | +0.23 / 0.96 | −1.72 / 1.03 | +2.84 |
| branch-combined | −0.76 / 1.04 | +0.47 / 0.81 | +0.33 / 0.97 | −1.38 / 1.01 | +2.99 |

Dispersion SEs are about 0.05. Strength draws are not mean-preserving: they cut drives by about half a drive and
raise points per drive by 0.3–0.4, adding 2–3 points per team. Play mix moves too (passes 75 → 68 per game,
runs 47 → 52, punts 8.5 → 7.3). The spread looks calibrated under strength draws partly because both teams
inflate together.

## 5. Sanity counts (per game, from `backtest_2025_attempt_diagnostics.csv`)

| Run | Two-point tries | Onside tries | Safeties | Kick calls | Punts |
|---|---|---|---|---|---|
| real 2021–25 | about 0.5 | about 0.19 | about 0.05 | | |
| branch-base | — | — | — | 4.44 | 8.53 |
| branch-playcalls | — | — | — | 4.33 | 8.16 |
| branch-extras | 0.46 | 0.20 | **0.10** | 4.46 | 8.41 |
| branch-combined | 0.61 | 0.21 | 0.05 | 3.38 | 6.87 |

Two-point tries and onside kicks match real rates. Safeties in the extras run are twice the real rate; in the
combined run they fall back to 0.05 (fewer punts and pinned drives).

## 6. Verdicts

Rule: helpful = improves its target calibration without making Brier, spread CRPS, total CRPS or energy_points
worse by more than 1 SE; harmful = any of those four worse by 2 SE or more.

- **Play calls (`NFLSIM_MODEL_PLAY_CALLS`): neutral.** Spread CRPS worse by 1.6 SE (more than 1, under 2). It
  narrows spreads (msz 1.18 → 1.35) and raises the home/away correlation away from observed (0.10 → 0.17), while
  improving totals (msz 1.07, bias 1.97 → 0.97). Not on by default.
- **Scoring extras (`NFLSIM_SCORING_EXTRAS`): helpful, narrowly.** No score moves by 1 SE; total msz 1.12 → 1.09;
  two-point and onside rates match reality. Safeties run 2x real. Reasonable to turn on by default; worth a look at
  the safety rate first.
- **Strength draws 0.85 (`NFLSIM_STRENGTH_DRAWS`): helpful by the letter of the rule, not recommended.** Spread msz
  1.18 → 1.02 with no headline score worse by 1 SE, but it adds about 2 points per team (total bias +1.97 →
  −4.23) and stat-line energy and variogram scores get worse by 4.6–4.7 SE. Needs mean-preserving draws before
  it is on by default.
- **Strength draws 1.0: harmful.** Brier worse by 2.7 SE.
- **Combined (play calls + extras + strength 0.85): harmful.** Total CRPS worse by 2.05 SE (energy_points 1.95),
  points bias +3.0 per team; the play-calls narrowing cancels the strength widening on spreads (msz 1.18).

Recommendation: turn on scoring extras by default only; keep the others opt-in. Always-on fixes on the branch are
a small, consistent improvement over `main`.

## Follow-up: scoring extras on by default

- `NFLSIM_SCORING_EXTRAS` is now on by default with `nflsim/config/scoring_extras.json`; `NFLSIM_SCORING_EXTRAS=off`
  turns it off, and a path still picks another settings file. Play calls and strength draws stay opt-in.
- Safety rate: penalties were enforced in full up to the goal line (a pre-snap penalty at the 3 put the next snap on
  the goal line, and any no-gain play there was a safety). Penalties now stop at half the distance to either goal
  line. On weeks 1-3 (48 games, 200 sims, paired seeds) safeties fell from 0.102 to 0.084 per game (SE 0.003; real
  0.048); two-point and onside rates unchanged; net penalty yards per game -1.3 to -4.1, since fouls near the
  opponent's goal line are capped too. The remaining excess comes from sacks and runs out of backed-up spots, a
  play-model matter, left as it is.
