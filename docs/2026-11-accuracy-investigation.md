# Simulation accuracy investigation, November 2026

Where the simulator still loses accuracy, and what to try next. The evidence comes from two places: a read of
the live code path (`nflsim/engine/simulation_runtime.py`, `nflsim/models/fit_minimal.py`, `nflsim/models/train.py`,
the committed `config/model_manifest.json`) and new tests on public nflverse play-by-play for 2016-2026 week 11.
Scripts are in `docs/analysis/2026-11-accuracy/` (set `PBP_DIR` to a folder of `pbp_<season>.parquet` files).

**Caveat.** The bundles, features and backtest outputs are not in the repository, so none of this reruns the
torch models or the simulator. The play-level tests use LightGBM (logistic regression for kicks) on public
columns, trained on 2017-24 and scored on 2025-26. They show whether a signal exists out of sample and roughly
how large it is. They do not measure the gain inside the production networks, which already see team features
such as `team_4th_rate`. Every recommendation below still needs the usual refit and paired backtest.

## Starting point

From `2026-10-model-review.md` (live bundle, 2026 weeks 1-7):

- Win Brier is 0.2368 against Vegas 0.2345.
- Simulated scores are too narrow. The latest reported sim SDs are 10.96 for the spread and 12.28 for the total
  (special-teams bundle, 2025 weeks 1-6). The live bundle's errors are larger: spread RMSE 12.44 and total
  RMSE 13.75 (2026 weeks 1-7). The live bundle's own SDs were not reported, so recompute them before relying
  on the gap below.
- The home/away score correlation is +0.09 to +0.11 in the sim. In real games it averages −0.02 over 2016-26
  (table below).
- About 100 games per backtest gives a Brier standard error near ±0.01. Most single changes are below that.

## Recommendations, ranked by expected effect on game-level accuracy

### 1. Propagate team-strength uncertainty into each simulated game

The ridge strengths, player values and usage shares enter every simulated game as point estimates
(README: "ridge posterior uncertainty is not fabricated"). The game effects add within-game variation, but the
simulator never accounts for its own pregame estimates being wrong. A calibrated forecast's SD should match its
own error. The gap implies a missing component of about √(12.44² − 10.96²) ≈ **5.9 points of spread SD** and
√(13.75² − 12.28²) ≈ **6.2 points of total SD**. That is the size of typical in-season uncertainty in NFL
team ratings.

This would also lower the home/away correlation. Draws made separately for each team add independent
variance, which dilutes the shared part.

Implementation options, simplest first:

1. **Ridge posterior draws.** `solve_ridge` (`nflsim/ridge/metrics.py:384`) already forms the normal matrix
   `A = XᵀWX + Λ`. The posterior covariance of the coefficients is `σ² A⁻¹`. Only the two teams' rows are
   needed per game, so take a sparse Cholesky of `A` or solve against the needed unit vectors. Save the per-team
   marginal SDs (and the offense-defense covariance) with the weekly bundle. Draw once per simulated game,
   the same way game effects are drawn.
2. **A learned scale.** Fit one variance multiplier per target (or one shared multiplier) on held-out weeks by
   maximizing the backtest's predictive log score, as was done for the game effects. This corrects for ridge
   σ² being misestimated under recency weighting.

The October review rejected stretching the game-effect scale to match the score spread. This is different:
it adds the parameter uncertainty that was omitted, with its own estimate. Check that the spread and total
dispersion move toward 1 without hurting the player-stat dispersions, most of which are already near 1.

### 2. Remove the hard-coded 4th-down override and give play calling an era signal

`Play.simulate` (`simulation_runtime.py:1177-1189`) overrides the play model on 4th down. It **always kicks**
when the FG distance is ≤ 54 yards and it is 4th-and-2 or longer, unless the offense trails by 4+ in the last
five minutes. It **always punts** when the FG distance is ≥ 62. The model's own prediction, and with it
`team_4th_rate`, is discarded on these plays.

| 2025-26 regular season 4th downs (5,948) | Share | Actual: go for it |
| --- | --- | --- |
| Rule forces a FG | 27.5% | 15.5% |
| Rule forces a punt | 49.4% | 6.1% |

In the forced-FG zone, the go-for-it rate rose from 8-10% (2016-18) to 15-17% (2021-25). That is about 0.6
decisions per game that the sim turns into kicks, and the trailing-late exception is much narrower than real
behavior.

The play-type model has no way to follow that trend either. Its features are game state and team ratings,
with nothing for season or league environment. On held-out 2025-26 plays:

| 4th-down go/punt/FG model | Test log loss | vs base |
| --- | --- | --- |
| Game state (what the sim uses) | 0.2516 | |
| + half seconds, quarter, timeouts | 0.2486 | +1.2% |
| + pregame spread and total | 0.2519 | −0.1% |
| + season | 0.2260 | **+10.2%** |
| + all of the above | 0.2209 | **+12.2%** |
| Recency weights (0.8 per season), no season feature | 0.2437 | +3.2% |

Recommendation: delete the override except for genuinely impossible kicks, add an era input, and let the
model (with `team_4th_rate`) decide. A tree model with a season feature effectively uses the latest season's
behavior. For the networks, use **pregame league-environment features** instead of a raw season index, such
as the trailing league 4th-down go rate over expected and the league FG make rate by distance band, computed
strictly before kickoff. They extrapolate sensibly and also describe the current season.

### 3. Add the missing game mechanics

None of these appear in the simulator:

| Mechanic | Real rate (2021-26 per game) | Sim | Why it matters |
| --- | --- | --- | --- |
| Two-point tries | 0.44-0.57, about 49% success | Always kicks the PAT | Key-number structure of margins, trailing-team comebacks |
| Safeties | 0.01-0.06 | A play ending behind the goal line is clipped to the 0 (`:1932`) and play continues | 2-point swings plus a free kick |
| Onside kicks | 0.15-0.21 | No decision, so the trailing team never regains the ball | Late-game variance, comeback possessions |
| Timeouts | 7.7 used | None; the clock model can't know whether a team can stop the clock | End-of-half and comeback drives |
| PAT after a return/defensive TD | | Fixed +7 (`:1977`, `:2328`) | Small |

Onside kicks and two-point tries are concentrated in close, late games. They should both widen the score
distribution and push the home/away correlation down. Two-point tries need a decision model (score and
time) and a success model. Timeouts need a counter per team per half, plus a usage policy (it can be learned:
P(timeout | state) is in the play-by-play) that feeds the clock model.

### 4. Make the play-calling and clock models aware of the half

The models see `game_seconds_remaining` but not `half_seconds_remaining`, the quarter or timeouts. The end of
the first half sits in the middle of a continuous input, where a scaled MLP has a hard time finding a narrow
window. That window matters: **10.6% of all points (4.8 per game) come in the last two minutes of the first
half**, and there are 0.47 end-of-half FG attempts per game.

| Held-out 2025-26 | Play type (4-class) | Clock runoff (5 s bins) |
| --- | --- | --- |
| Base log loss | 0.5448 | 1.4376 |
| + half seconds, quarter | +0.36% | +1.14% |
| + timeouts | +0.08% | +0.21% |
| + pregame spread and total | +0.21% | |
| + season | +0.68% | |
| + all | **+1.25%** | +1.35% |
| + all + out of bounds, no-huddle | | **+3.56%** |

Out-of-bounds is the largest single clock input, but the sim doesn't produce it. The rush and YAC models would
need an out-of-bounds output (or a small classifier after the yardage draw) for the clock model to use it.

### 5. Validate forward in time, and backtest on more games

- `train.holdout_games` holds out a random fifth of games **within** the training seasons, and the
  validation split is also random by game. Early stopping and model choice therefore never see drift (rule
  changes, 4th-down trends, longer FG range: 50+ yard attempts per team-game went from 0.30 in 2017 to 0.55 in
  2026). Keep the shared fifth for fitting game effects, but choose models and stop training on a later
  season (for example, train ≤ S−1, validate S).
- The backtests cover about 100 games, so most changes land inside ±0.01 Brier. For model selection, also
  train a bundle through 2023 and backtest all of 2024 and 2025 (about 540 games). That roughly halves the
  standard errors, and `bundle.trained_through` already guards against leakage.

### 6. Give the kick model the kicker and the weather

The `kick` model (FGs and PATs) takes game state plus the five team player-value columns. It has no kicker,
wind or temperature; `reduced_spec` adds weather only to models whose reference context already had it.
Results for held-out 2025-26 kicks (3,669, of which 1,715 are FGs), logistic on a distance spline:

| Added | FG log loss | Kicks overall |
| --- | --- | --- |
| Wind, cold | +0.24% | +0.20% |
| Kicker make rate over expected (shrunk, strictly prior) | +0.30% | +0.36% |
| Season | +1.02% | +1.05% |
| All | **+1.15%** | **+1.24%** |

The gain is small but cheap, and it is in a model where each outcome is worth 3 points. LightGBM with
default settings *lost* 1-3% here because about 8k training kicks is too few for it. For the same reason,
consider a smaller network or a GLM for the kick and special-teams heads rather than the shared 256/128/64
trunk.

### 7. Smaller modeling-technique items

- **Temperature scaling per model.** Fit one scalar per network on the held-out fifth and check reliability
  plots. Per-play miscalibration compounds over about 130 plays a game. The game effects already shift
  logits, so fit the temperature with the effects integrated out, using the same Laplace machinery.
- **Seed averaging.** `models/seed_average.py` exists. Extend it to the volume models (play, target, rusher,
  time), where seed noise moves team-level totals.
- **Market blend (optional, product decision).** Sim Brier is 0.0023 behind Vegas. A logistic stack of the
  sim's margin and the market spread, fit on past weeks, would likely beat both. It makes the sim depend on
  the market, so keep it as a separate output if it is wanted at all.

## League environment by season (regular season)

From `drift_and_rules.py`. Home/away correlation is of raw final scores.

| Season | Pts/game | Home/away corr | Spread err SD | 4th-down go rate in forced-FG zone | 50+ FG att / team-game | 2-pt att / game | KO touchback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2016 | 45.6 | 0.04 | 11.8 | 9.6% | 0.29 | 0.41 | 58% |
| 2017 | 43.4 | −0.03 | 13.1 | 8.4% | 0.30 | 0.34 | 57% |
| 2018 | 46.7 | 0.02 | 13.2 | 9.6% | 0.30 | 0.51 | 61% |
| 2019 | 45.6 | −0.07 | 13.0 | 12.6% | 0.28 | 0.44 | 61% |
| 2020 | 49.6 | −0.04 | 12.8 | 13.9% | 0.33 | 0.51 | 61% |
| 2021 | 46.0 | −0.12 | 13.7 | 16.2% | 0.34 | 0.57 | 58% |
| 2022 | 43.8 | 0.11 | 11.4 | 14.4% | 0.41 | 0.44 | 60% |
| 2023 | 43.5 | −0.05 | 13.1 | 14.7% | 0.42 | 0.47 | 73% |
| 2024 | 45.8 | −0.10 | 12.6 | 14.6% | 0.51 | 0.50 | 64% |
| 2025 | 46.0 | −0.03 | 12.3 | 17.3% | 0.49 | 0.47 | 21% |
| 2026 (wk 1-11) | 46.2 | 0.05 | 12.0 | 12.1% | 0.55 | 0.47 | 16% |

## Suggested order of work

1. Strength-uncertainty draws (item 1). This is the largest calibration gap, needs no retraining, and can be
   tested with the existing backtest and `joint_outcomes`.
2. Remove the 4th-down override and add league-environment features (item 2), then add half seconds, quarter
   and timeouts as inputs (item 4). Both need a refit; do them in one refit.
3. Two-point tries, safeties, onside kicks and timeouts (item 3). These are simulator changes plus three small
   models.
4. Forward validation and the larger 2024-25 backtest (item 5), before judging items 2-3.
5. Kick model inputs, temperature scaling and seed averaging (items 6-7).
