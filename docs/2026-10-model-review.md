# Model review, October 2026

What changed in response to the two design notes ("NFL Player Plus/Minus" and "player and team
statistics"), and what the out-of-sample tests showed. All numbers are held-out: plus-minus is
scored against each player's rest-of-season on/off (2019-2025); ridge variants are tuned on 2024
and scored on 2025 with weekly pregame refits.

## Player plus-minus (`player_value/`)

**In-season updating.** The weekly estimate blends the current-season box prediction with a prior.
`player_value.calibration` compares priors: zero; last season decayed (the original method); and a
regression on prior-season estimates, age, draft slot and experience, refit each season on earlier
seasons only, which gives rookies a draft/age prior. It also tunes the prior strength. Real pipeline
refits, MSE change vs the original settings:

| Position | Chosen | Gain |
| --- | --- | --- |
| QB | regression prior, T=200 | +0.5% |
| WR | regression prior, T=400 | +1.4% |
| RB, TE | original (the grid optimum lost 1.1% / 0.1% on refit) | 0 |
| DL | regression prior, T=800 | +0.5% |
| LB, DB | zero prior, T=1600 | +1.3% / +0.2% |
| OL | regression prior, T=400, Madden ratings | +0.35% |

Weighting the box regression's historical rows by plays did not help and stays off.

**Positions without box stats.** OL's only box feature was penalties per game. Madden preseason
ratings (`data/madden/`, joined by GSIS id) now enter the OL box profile. OL plus-minus is the
first version to beat predicting the average, but only barely. Madden added nothing for DL/LB/DB.

**Caveat.** LB and DB plus-minus still do not beat predicting the average rest-of-season on/off,
and OL only just does. Treat those values as weak. OL/DL/LB/DB also cannot update in-season until
nflverse publishes participation data (2026 is not available), so they carry 2025 values forward.

## Ridge metrics (`nflsim/ridge/`)

| Change | Result (2025) |
| --- | --- |
| Targets over a game-state + weather expectation (`expectations.py`) | Rush yards +2.0%, first downs +15% (both significant); CPOE +0.4%, sacks +0.3%, interceptions +0.2%, rush EPA +0.1%; YAC about even |
| OL/LB/DB player value in the design ("team health") | about 0 everywhere |
| Extra decay before team changes / head-coach changes | about 0 (coach carryover 0.4 chosen for team EPA; not significant) |
| Luck-adjusted team EPA (`epa_luck.py`: turnovers at league-average EPA, extremes clipped, optionally no garbage time), test seasons 2022-25 | Pooled pass EPA −0.2%, rush EPA +0.3%; 1 of 16 season comparisons significant. Not adopted |

The first-down gain is large because production estimates absorb a player's role (e.g., deep
targets), and the simulator also adds context, which double-counts it. Weather effects vs a calm
60°F outdoor game: rain, snow or 25°F cut completion by about 2 percentage points and pass EPA by
0.04-0.065/play (`nflsim/outputs/variant_bakeoff/weather_effects.csv`).

**Over-expectation across all targets, test seasons 2022-25** (`nflsim/outputs/variant_bakeoff_xoe`,
quick grid, each season tuned on the one before). Adoption rule: significant pooled gain and no
season significantly worse. Pooled z of the RMSE change (negative = over-expectation better):

| Target | Side | Pooled z | Decision |
| --- | --- | --- | --- |
| First downs | receiver | −22.7 (4/4 seasons significant) | adopt |
| Yards per target | receiver | −10.9 (4/4) | adopt |
| Rush yards | rusher / defense | −4.9 / −2.0 | adopt |
| Sacks | passer / defense | −4.9 / −3.7 | adopt |
| Explosive rushes | rusher | −2.7 | adopt |
| Deep CPOE, rush EPA | | worse in at least one season | keep raw |
| CPOE, short/medium CPOE, interceptions, pass EPA, scramble yards, YAC | | not significant | keep raw |

These five targets are fit over expectation in production (`PRODUCTION_XOE_TARGETS`; same feature
names, recorded in the feature manifest). Interception-worthy throws have no expectation model.

**Pass-rush pressure (`pressure.py`, `nflsim/outputs/variant_bakeoff_pressure`).** Charted pressure
(`was_pressure`) is published only after the season, so two weekly measures were tested instead:
QB hits per dropback (play-by-play, 2016+) and PFR `times_pressured` per dropback (nflverse
`pfr_advstats` weekly pass files, 2018+). Both correlate 0.47-0.48 with charted pressure at the
team-game level. Each measure is a team ridge term (offense allowed, defense rate); its pregame
forecast beats the league-week mean by 3.8-7.5% for hits but not at all for PFR (−1.2% to +0.7%).
As a covariate on passer targets (fit on realized pressure, predicted with the forecast), against
the production targets:

| Target | Hits pooled z | PFR pooled z |
| --- | --- | --- |
| Pass EPA offense / defense | −1.5 / −2.0 | −1.2 / −1.2 |
| Sacks passer / defense | −0.7 / −1.4 | −0.1 / +0.4 |
| Yards per target | −1.4 | −2.0 |
| Interceptions (defense) | +0.5 | −0.7 |
| CPOE | 0.0 | +0.7 |

Gains are at most 0.2% of RMSE; the single |z| ≥ 1.96 (PFR on yards per target, 0.01%) is about
what 14 comparisons yield by chance. Not adopted; the bakeoff variants (`hit`, `pfr`) stay available.

## Game-level variance in the play models (`nflsim/models/game_effects.py`)

The play models treat plays as independent given their features, but plays within a game share an
unobserved "day". The earlier answer was a three-stage cascade on four models (play type, target,
rusher, completion). It averaged a team-game's stage-1 residuals, learned their spread from pregame
context and fed a draw back in as a feature. That had four problems: the game residual included the
play's own outcome (leakage); raw means over 10-35 plays mix true game variance with sampling
noise, which the simulator then adds again; stage 3 trained on a 70/30 blend but saw only draws;
and the four effects were independent of each other and of the other 16 models.

The replacement is a generalized linear mixed model on top of flat networks:

    logit = f_m(x) + b_m + G_m u_game + T_m u_team + s_m v_team,m,   u, v ~ N(0, I)

`u_game` is shared by both teams, `u_team` by every model for one offense in that game, and `v` is
model-specific. The networks are fixed. b, G, T and s are fit by maximizing each game's Laplace
marginal likelihood (Newton steps for the mode), so the effects are integrated out rather than
estimated from outcomes. Every network now holds out the same fifth of games
(`crc32(game_id) % 5 == 0`), and the effects are fit only on those games. The simulator draws u
and v once per game and team (`nflsim/engine/game_effects.py`), exactly as fit.

**Results.** Held-out games 2017-2024: 397 games, 493k plays, 2-fold cross-validated marginal
likelihood by importance sampling (`nflsim/outputs/game_effects/cv_marginal_likelihood.csv`):

| Effects | Gain over independent plays (log-lik/game) |
| --- | --- |
| Model-specific only | +9.88 |
| + 1 shared team factor | +10.16 |
| + 2 / + 3 shared team factors | +10.37 / **+10.60** (chosen) |
| + 1 game factor (3 team) / 2 game + 4 team | +10.50 / +10.38 |

2025, fully out of sample. The variance ratio is the observed team-game variance over the model's
(1 = calibrated, >1 = simulated games too similar, <1 = too spread out):

| Team-game stat | Current bundle (cascades) | New nets, no effects | New nets + effects |
| --- | --- | --- | --- |
| Completions | 0.64 | 1.21 | 1.02 |
| Pass plays | 0.83 | 1.70 | 0.93 |
| WR1 targets / RB1 carries | 0.65 / 0.74 | 2.18 / 4.74 | 0.93 / 0.80 |
| Rush yards / air yards / YAC | 1.41 / 1.30 / 1.31 | 1.33 / 1.31 / 1.30 | 0.97 / 1.01 / 0.99 |
| Interceptions / penalties | 1.20 / 1.34 | 1.13 / 1.48 | 0.96 / 1.03 |
| Play seconds | 1.62 | 1.69 | 0.88 |
| Sacks / scrambles | 1.44 / broken | 1.60 / 2.04 | 1.43 / 1.62 |

Per-play log loss averaged over effect draws is unchanged (the effects move game-level spread, not
the per-play mean); the flat play/target/rusher models are better per play than the cascades'
stage 3 (0.556 vs 0.579, 1.740 vs 1.755, 0.886 vs 0.900). The shared factors load mainly on play
calling and usage (play, target, rusher). Shared loadings on the efficiency models are small: once
pregame features are in, a team's completion, sack and rushing "days" are close to independent.
The observed completion-sack and completion-INT correlations (-0.17) are mostly mechanical (same
dropbacks), and the simulator already draws those outcomes jointly. Remaining gaps: sacks and
scrambles are still under-dispersed (1.4-1.6), and rare rusher slots get large private variances
(logit SD up to 3.4), which leaves RB1 carries slightly over-dispersed.

**Also fixed while testing.** nflverse leaves `passer_player_id` empty on scrambles, so the dropback
models saw missing passer stats on exactly the scramble rows. The scramble model "predicted"
scrambles from that (AUC 0.9997). In simulation the passer is always known, so the last backtest
produced zero scrambles per game (real games average about four). Training now joins the
scrambling QB's stats, and comp, int, sack and scramble were retrained. Sack yardage was a uniform
draw from -10 to -2; it is now a CORN model (`sack_yards`) on the sack model's inputs.

**Game-level backtest.** 2025 weeks 1-6, 93 games, 200 sims each, old bundle 630d58 vs the new
bundle 13063f (paired per game in `nflsim/outputs/game_effects/backtest_w1-6_paired.csv`):

| | Old | New | New − old (± SE) |
| --- | --- | --- | --- |
| Win Brier / log loss | 0.239 / 0.671 | 0.244 / 0.685 | +0.005 ± 0.010 / +0.014 ± 0.024 |
| Spread / total CRPS | 7.08 / 7.45 | 7.12 / 7.53 | +0.03 ± 0.20 / +0.09 ± 0.27 |
| Sim SD of spread / total | 12.0 / 12.3 | 10.9 / 12.0 | |
| Mean squared z, spread / total | 1.20 / 1.13 | 1.38 / 1.22 | |
| Total bias | −3.1 | −0.7 | |

The bundles are tied on game scores. Weeks 3-6 alone are hard for both (win Brier 0.27 against
Vegas's 0.23). Both bundles' score distributions are now a little too narrow; the new bundle's
narrower spread makes that slightly worse.

**Joint player/team outcomes** (`nflsim.sim.joint_outcomes`, same 93 games, outputs in
`nflsim/benchmarks/2025/joint_outcomes/`, tracked in `joint_outcomes_history.csv`). Dispersion is
12·var(PIT): 1 is calibrated, below 1 the sims are too wide.

| Stat | Dispersion old / new | Bias old / new |
| --- | --- | --- |
| QB1 pass attempts / passing yards | 0.81 / 1.13, 0.82 / 1.05 | +6.3 / +3.1, +36 / +23 |
| QB1 rushing yards | 0.83 / 1.02 | −14.2 / +0.2 |
| RB1 carries / rushing yards | 0.64 / 0.53, 0.86 / 0.84 | +0.4 / +3.3, −1.9 / +12.8 |
| WR1 receptions / yards | 0.54 / 0.70, 0.68 / 0.83 | +2.6 / +1.6, +27 / +16 |
| WR2 receptions / yards | 0.56 / 0.67, 0.64 / 0.72 | +1.9 / +1.4, +22 / +16 |
| TE1 receptions / yards | 1.08 / 0.84, 0.94 / 0.76 | +0.8 / +0.2, +7.8 / +1.7 |
| Team rush attempts / rush yards | 0.62 / 0.80, 0.96 / 1.01 | −2.4 / −0.8, −18 / −0.8 |

Averaged over all stats, |log dispersion| falls from 0.20 to 0.15 and the gap between simulated
and observed pair correlations from 0.093 to 0.064. Old-bundle dependence errors that the new
one fixes: WR1 × WR2 yards (sim −0.42 vs observed −0.12, now −0.05 vs −0.09), pass attempts ×
rush attempts (−0.51 vs −0.28, now −0.37 vs −0.26), QB yards × win (0.20 vs 0.04, now 0.08 vs
0.02). Points × opponent points looked under-correlated in weeks 1-2 (observed 0.40) but is about
0 over six weeks; both bundles match it.

What to improve next, ranked by standard errors from calibrated in the new bundle:

1. **RB1 carries** (dispersion 0.53, +3.3 bias, rushing yards +12.8): the rusher effect's large
   private variances on rare slots spread RB1 workloads too far. Cap or pool the slot variances.
2. **Receiver volume is still too spread out** (WR1/WR2/TE1 receptions 0.67-0.84) and receivers
   are over-projected (WR1 +16 yards, WR2 +16), with QB1 still +23 passing yards on +3 attempts.
3. **QB yards × TE1 yards** is under-correlated (0.19 vs 0.42 observed; the old bundle had 0.39).
4. **QB passing TDs × WR1 TD** is over-correlated in both bundles (0.48-0.51 vs 0.30).

## Usage shares and skill slots (`nflsim/features/usage.py`, `nflsim/features/slots.py`)

**Slot bug.** Training assigned skill slots from every roster row for the team-game, including
inactive players. Non-ACT players held 10% of training RB1 slots, 43% of RB2, 65% of RB3 and 21% of
TE1, while the backtest only slotted active players. So the selection models learned from slot
layouts the simulator never builds. `slots.rank_slots` is now the only slot rule for training, backtests and the dashboard:
QB/RB(+FB)/WR/TE that are status ACT (when any row is) and not listed Out; QB by depth chart, RB
by `rush_rate`, WR/TE by `medium`; ties go to the depth chart, then player id. Team-games with an
empty slot (TE3 missing in 516 of 4,222 for 2017-24, RB3 in 67) are now kept. The empty slot gets
zero usage shares and median values for its other stats. The simulator gives its placeholder
player the same zero shares.

**Usage estimator.** The old `rush_rate`/`deep`/`medium`/`short` used a flat ~0.25 prior and skipped
games where a player had no events. It had no draft prior and no availability adjustment, so a
backup's share rose while the starter was out and stayed high when the starter came back. The new
estimator models a team-game's events as a softmax over the players who were on the field
(snap counts), each with a latent strength plus a depth-chart offset:

- Strengths follow a Gaussian random walk (week and offseason variances). A sequential
  diagonal-Laplace filter updates them after each game.
- A new player's prior mean is the position mean plus a draft slope times log(pick / 100), with
  undrafted players counted as pick 300.
- The pregame share is the softmax over the expected active set (status ACT and not Out).

When RB1 is out, RB2's share rises only through the smaller denominator. RB2's strength moves
only as far as his per-snap work is actually better than expected. So RB1 gets his role back as
soon as he is active again. (The `test_usage_slots` injury test checks this.) The emitted value for an
inactive player is his share if he were added back to the expected set.

Hyperparameters were fit on 2017-22 and scored on 2023-25 by pregame log-likelihood per event
(`outputs/usage/usage_scores.csv`):

| Share | Old, 2017-22 | New, 2017-22 | Old, 2023-25 | New, 2023-25 |
|---|---|---|---|---|
| rush_rate (per carry) | −2.30 | −1.22 | −2.24 | −1.20 |
| deep (per target) | −2.08 | −1.74 | −2.02 | −1.68 |
| medium | −2.20 | −1.96 | −2.11 | −1.87 |
| short | −2.41 | −2.20 | −2.35 | −2.16 |

The deep share's week-to-week variance tuned to 0, so deep usage drifts only between seasons.
Under the same slot rule, the new values pick the actual leader more often for RB1 (82% to 84%
on 2023-25) and TE1 (79% to 81%), and slightly less often for WR1 (60% to 59%).
WR1 is ranked on medium share alone, so ranking WRs on total target share is worth trying
(`outputs/usage/slot_accuracy.csv`). Slot ordering is mostly unchanged. The gain is in the share
values the selection models are fed.

**Retrain, refit and backtest.** `python -m nflsim.features.usage write` produced a training
table (`outputs/usage/training_table/`, from the r2 reference, 99.7% of rows covered) and a runtime
table (`outputs/usage/runtime_table/`, from bundle 630d58; 90.1% covered, since 2026 rows mostly
keep old values until the weekly refresh calls the estimator). `fit_minimal`, `game_effects` and
`package_game_effects_bundle` take `--player-table`. The retrain is `outputs/refits/pv-usage-20261031`,
the game-effects refit (same `team3` config) is `outputs/game_effects_usage/`, and the bundle is
`pv-usage-20261031` (hash ab4afd…).

The selection models' held-out scores look worse (rusher log loss 0.92 → 1.10, target 1.72 →
1.89), but they are scored on different rows. The old slot rule dropped every rush whose carrier
was not slotted: 20,596 of 103,216 non-scramble rushes in 2017-24 against 3,773 now. The dropped
rows were the hard ones, and RB1 carried 63% of the kept rows against 58% now. On the same
rows the new shares alone take share-only log loss from 1.75 to 1.15, and on 2025 plays the new
networks beat the old cascade bundle (rusher 1.17 → 1.03, target 2.07 → 1.87). The game
effects now add 6.0 log-likelihood per 2025 game instead of 11.6, and no longer hurt rusher
selection as much. Part of what they had been absorbing was the bad usage inputs.

**Backtest, weeks 1-6** (200 sims per game, 93 games; `joint_outcomes` labels old = 630d58,
effects = 13063f, usage = ab4afd; game-level pairs in `outputs/usage/backtest_w1-6_paired.csv`):

| Stat | Dispersion old / effects / usage | Bias old / effects / usage |
| --- | --- | --- |
| RB1 carries | 0.64 / 0.53 / 0.85 | +0.4 / +3.3 / +0.6 |
| RB1 rushing yards | 0.86 / 0.84 / 1.03 | −1.9 / +12.8 / +1.8 |
| WR1 receptions / yards | 0.54 / 0.70 / 0.94, 0.68 / 0.83 / 1.01 | +2.6 / +1.6 / +0.5, +27 / +16 / +6 |
| WR2 receptions / yards | 0.56 / 0.67 / 0.94, 0.64 / 0.72 / 1.00 | +1.9 / +1.4 / +0.6, +22 / +16 / +5 |
| TE1 receptions / yards | 1.08 / 0.84 / 0.94, 0.94 / 0.76 / 0.85 | +0.8 / +0.2 / −0.1, +7.8 / +1.7 / +0.3 |
| QB1 pass attempts / yards | 0.81 / 1.13 / 1.14, 0.82 / 1.05 / 1.01 | +6.3 / +3.1 / +3.3, +36 / +23 / +26 |
| Team rush attempts | 0.62 / 0.80 / 0.78 | −2.4 / −0.8 / −1.0 |

Mean |log dispersion| over all stats falls to 0.070 (0.204 old, 0.154 effects). The mean
correlation gap is 0.067, level with effects (0.064). Usage fixes QB yards × TE1 yards (sim 0.27 vs
observed 0.33; effects had 0.19 vs 0.42). The game scores do not move: win Brier 0.238 / 0.242 /
0.237, spread CRPS 7.08 / 7.12 / 7.22 and total CRPS 7.45 / 7.53 / 7.45. Every usage − effects
difference is within 1.3 standard errors, and total bias is −0.2.

What to improve next:

1. **Passing volume**: QB1 is still +3 attempts and +26 yards. Receptions are now close, so the
   excess yards come through yards per catch or targets to unslotted players.
2. **Team rush attempts** are too spread out (0.78) and RB1 carries slightly so (0.85).
3. **QB passing TDs × WR1 TD** is still over-correlated (0.44 vs 0.26).
4. **Score distributions are a little narrow** (mean squared z, spread 1.45, total 1.15).
5. `depth_team` is missing for 40-50% of 2017-24 skill rows and all of 2025, and missing rows take
   the depth-4 offset. Filling them from the depth-chart file (`slots.depth_ranks`) or giving them
   a neutral offset should sharpen new players' priors.

## Score spread and special teams (`nflsim/features/special_teams.py`)

**Simulated scores are too narrow.** Against the market over 2016-25 (2,639 games), a team's score
misses its line-implied total by 9.15 points (SD) and the two teams' misses correlate 0.035. The
usage bundle's per-team SD is 8.05 and its home/away correlation 0.09, so its spread SD is 10.9 where
the market's error is 12.7; the old cascade bundle had 8.57, 0.02 and 12.0. Re-simulating 24 games
with the game effects off gives 7.35 per team (on 7.94), so the effects widen scores and the
narrowing came from replacing the cascades with flat models. Controlling for plays and possessions
barely moves the home/away correlation (0.088 to 0.079), so it is not pace; its source is open.
Calibrating the effect scale to the score spread was rejected in favor of fixing models.

Per-team stats (2025 weeks 1-6, usage bundle; `outputs/usage/team_parts_w1-6.csv`): yards
(dispersion 1.02), yards per play (1.08) and offensive TDs (1.04) are calibrated, but points per 100
yards is not (1.32; sim SD 1.8 vs error 2.5). In 2021-25 games, yards explain 44% of team points,
average drive start 2.5% more and return/defensive TDs another 8.5%; the sim produced 0.05 such TDs
per team-game against 0.12.

**Special teams were the gap.** Kickoffs were a fixed rule (68% touchbacks to the 30, otherwise a
normal around the 28), with no team inputs or return TDs; made field goals and defensive TDs
restarted at the 30 and there was no halftime kickoff. The punt model was net yards from game state
only. Real kickoffs changed with the rules: touchback rate 74% in 2023, 65% in 2024 (start 30.2 ±
7.5), 21% in 2025 (start 30.9 ± 9.2), against the rule's ≈ 29.4 ± 3.6.

The new bundle (`pv-special-teams-20261101`, hash a5b87d…) adds three CORN models in
`fit_minimal --game-effects`, built in `training_data.special_teams_rows`:

- `punt_distance`: gross punt yards; a landing past the goal line is a touchback to the 20.
- `punt_return`: return yards given the landing spot and gross distance (fair catches and downed
  punts are 0); a return reaching the end zone is a TD.
- `kickoff`: the receiving team's own-yardline start, 100 for a return TD, with a 2024-rule flag.
  2025 kickoffs are not in the 2017-24 training data, so 2025 is simulated under the 2024 rules
  (its 2025 log loss is 4.3 against 2.0 on 2017-24 test kickoffs; the start-yard SD is still far
  closer than the fixed rule's).

Each takes the game state, weather and team special-teams features: shrunk, recency-weighted
deviations from the same week's league average (punt gross distance, punt-return yards allowed and
gained, kickoff start allowed and gained), tuned out of sample. Team quality carries little signal:
skill against "league average" is 4.8% for punt distance, 1.6% for return yards allowed and about 0
for the kickoff features, so most of the gain must come from the outcome distributions. With the
kickoff model the simulator kicks off at the start, after every score and at halftime (the opening
receiver kicks), and punt-return TDs count toward the defense/special-teams stat. Older bundles keep
the previous behavior. Game effects gain 6.2 per 2025 game (6.0 before).

**Backtest (2025 weeks 1-6, 93 games, 200 sims; `benchmarks/2025/special-teams-a5b87d-2025-w1-6-200`).**
The bundle is neutral on game-level metrics: against the usage bundle, Brier is +0.006 ± 0.010,
log loss +0.010 ± 0.024, spread CRPS −0.02 ± 0.12 and total CRPS +0.05 ± 0.20 (paired 95%
intervals, all including 0). It moves the scoring parts the right way but not far enough:

| | Usage | Special teams | Actual / market |
|---|---|---|---|
| Return/defensive TDs per team-game | 0.05 | 0.08 | 0.11 |
| Per-team sim score SD | 8.05 | 8.23 | 9.15 (market error) |
| Home/away score correlation | 0.088 | 0.113 | 0.035 |
| Spread SD | 10.87 | 10.96 | 12.7 |
| Total SD | 11.88 | 12.28 | 13.2 |
| Points per 100 yards dispersion | 1.32 | 1.24 | 1 |
| Mean points per team | 23.0 | 22.6 | 23.1 |

`joint_outcomes` (appended to the history) is unchanged within noise: team points dispersion 1.11
(1.12), win dispersion 1.16 (1.14), player stats as before. The home/away correlation rose, so
kicking off after every score does not explain it either; it remains the main open item, along with
the 2025 kickoff rule (needs 2025 data in training).

## Live rebuild through 2025 (`nflsim/artifacts/runtime`)

Live since 2026-11-02. The tested bundle `pv-live-20261101` (hash a04940…, now in
`archive/nflsim_bundles/`) became `artifacts/runtime` (hash c4f9e2…). The only change is that it now
carries the ridge selection its features were built with: the packager had kept the base bundle's file.
Weekly refreshes start from the runtime.

One bundle with everything above that tested well:
- over-expectation ridge targets;
- the regression-prior player value (through 2026 week 7);
- availability-aware usage shares and the scramble fix;
- the special-teams features and models (including the 2025 kickoff rule flag);
- `team3` game effects.

Features were rebuilt for 2017-26 from `artifacts/refresh_inputs/2026_20261101`. The 20 play models
were retrained on 2017-25, and the game effects fit on held-out 2017-25 games. The pipeline is in
`nflsim/outputs/live_rebuild_20261101/pipeline.sh`. The bundle check no longer caps training at 2024;
instead a backtest refuses any season the bundle trained on (`bundle.trained_through`).

**2026 play log loss (new vs live `630d58`).**

| Model | Live | New |
| --- | --- | --- |
| Play type | 0.570 | 0.546 |
| Target | 2.04 | 1.86 |
| Rusher | 1.18 | 1.04 |
| Scramble | 2.23 | 0.19 |

Penalty yards, sack, punt, air yards and YAC are slightly better. Fumble, interception and kick are
within 0.002.

**Backtest, 2026 weeks 1-7 (107 games, 200 sims; `benchmarks/2026/rebuild-a04940-2026-w1-7-200` vs
`live-630d58-2026-w1-7-200`).** Paired on game; new − live, with 95% intervals:

| Metric | New − live | New better in |
| --- | --- | --- |
| Win Brier | −0.030 ± 0.019 | 62/107 |
| Win log loss | −0.067 ± 0.043 | 62/107 |
| Spread CRPS | −0.43 ± 0.36 | 67/107 |
| Total CRPS | −0.06 ± 0.45 | 56/107 |

Brier by week:

| Week | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| New | 0.210 | 0.222 | 0.257 | 0.223 | 0.213 | 0.216 | 0.300 |
| Live | 0.240 | 0.260 | 0.271 | 0.226 | 0.252 | 0.292 | 0.313 |
| Vegas | 0.214 | 0.231 | 0.254 | 0.219 | 0.212 | 0.258 | 0.231 |

The new bundle wins every week. Its overall Brier is 0.2368 against Vegas 0.2345 and live 0.2683; spread RMSE is 12.44 vs 13.19, total RMSE 13.75 vs 14.18.

`joint_outcomes`, new vs live:

| Measure | New | Live |
| --- | --- | --- |
| Team points CRPS | 5.29 | 5.47 |
| Mean absolute log dispersion | 0.062 | 0.181 |
| Mean correlation gap | 0.065 | 0.106 |

The live bundle's player distributions were far too narrow, for example:
- RB1 rushing attempts dispersion 0.59;
- WR1 receptions 0.55;
- WR1×WR2 receiving yards correlation −0.42 against an observed +0.07.

The new bundle's largest misses are RB1 attempts (0.83) and a QB-WR1 touchdown correlation that is
too high (0.44 vs 0.26).

## Regularization and Bayesian alternatives

Ridge penalties are tuned per block by out-of-sample coordinate search on 2024, which is the
practical equivalent of choosing prior variances. A Bayesian version treats each block's penalty as
noise variance over prior variance and estimates it by marginal likelihood (empirical Bayes /
type-II ML). That also yields uncertainty for each estimate. Useful references: Sill, *Improved NBA
Adjusted +/- Using Regularization and Out-of-Sample Testing* (MIT Sloan 2010); Engelmann's xRAPM
(box-score prior plus RAPM); and Kalman-filter style daily updating as in DARKO for in-season
priors.
