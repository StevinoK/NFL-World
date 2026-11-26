# What other NFL forecasters feed their models, November 2026

A comparison of public forecasters' inputs and engineered features with ours, with out-of-sample tests of
the game-level ones we lack. Companion to `2026-11-accuracy-investigation.md`. Scripts are in
`docs/analysis/2026-11-accuracy/` (`exp_game_context.py`, `exp_market_ref.py`; set `PBP_DIR` to a folder holding
`pbp_<season>.parquet` files and the nflverse `games.csv`).

## Sources and how far to trust them

The methodology pages for FiveThirtyEight, ESPN FPI, Silver Bulletin (ELWAY), nfeloapp.com and Open Source
Football could not be reached from the analysis environment. What was read directly:

- **nflfastR source** (`nflverse/nflfastR`, `R/helper_add_ep_wp.R`, `helper_add_cp_cpoe.R`, `helper_add_xyac.R`,
  `helper_add_xpass.R`): exact feature lists for the EP, WP, completion, YAC and expected-pass models.
- **nfelo** (`greerreNFL/nfelo`): the source went private in November 2026, but the git history keeps the last
  public code and `config.json` (v4.3.0). Also the `greerreNFL/nfeloqb` QB model config.
- **FiveThirtyEight's published Elo code** (`fivethirtyeight/nfl-elo-game`): the basic 2014-era model only.

ESPN FPI, 538's later QB-adjusted Elo and FTN's DVOA projections are described below from search-result
summaries of their own write-ups, so treat those details as second-hand.

## Play-level models: nflfastR

All four play models include **era indicators**: dummies for 2002-05, 2006-13, 2014-17 and 2018+. They also
share a set of game-clock and context inputs.

| Model | Inputs |
| --- | --- |
| EP | half seconds remaining, yardline, home, roof (retractable / dome / outdoors), yards to go, down dummies, era, both teams' timeouts |
| WP | `receive_2h_ko`, home, half and game seconds remaining, `Diff_Time_Ratio`, score differential, down, yards to go, yardline, both timeouts; the spread version adds `spread_time` |
| Expected pass | down, yards to go, yardline, score differential, half and game seconds, both timeouts, quarter, era |
| Completion | air yards, yardline, yards to go, down, `air_is_zero`, `pass_middle`, era, `qb_hit`, home, roof, `distance_to_sticks` |
| YAC | the completion inputs plus `distance_to_goal` |

The engineered columns are defined as follows:

- `Diff_Time_Ratio` = score differential / exp(−4 · elapsed share of the game). It makes a lead count for more
  as time runs out.
- `spread_time` = pregame spread · exp(−4 · elapsed share). It makes the pregame strength count for less as
  the game goes on.
- `receive_2h_ko` marks the team that receives the second-half kickoff, while still in the first half.
- `distance_to_sticks` = air yards − yards to go.

How these compare with our models:

| Feature | Ours | Recommendation |
| --- | --- | --- |
| Era indicators | None in any model | **Adopt.** This independently supports the earlier finding: a season input improved held-out 4th-down calls by 10%, kicks by 1% and play type by 0.7%. Use league-environment rates for the networks. |
| Half seconds remaining, both teams' timeouts, quarter | Game seconds only | **Adopt** (already item 4 in the investigation; the sim also needs timeout counters). |
| `receive_2h_ko` | None | Add with the half inputs. It shapes end-of-first-half aggression and is free to compute. |
| `Diff_Time_Ratio`, strength × time decay | Separate score, clock and team inputs | Add as explicit inputs to `play`, `time` and the new 4th-down and two-point models. An MLP on standardized inputs learns this interaction slowly. |
| `distance_to_sticks`, `air_is_zero` | `comp`, `int` and `yac` get air yards and yards to go separately | Cheap to add, since the sim knows both values when it draws completion. |
| `pass_middle`, `qb_hit` | The sim draws neither | Both need a new sampled state. The QB-hit rate as a *team covariate* added under 0.2% (pressure bakeoff). Sampling a hit per dropback, as a step before completion, is untested. Lower priority. |

## Team-strength models

| Forecaster | Inputs beyond results (as published) |
| --- | --- |
| 538 Elo (2014 code) | Margin-of-victory multiplier with an autocorrelation correction, home field 65 Elo, K = 20, one-third reversion to the mean between seasons. Later versions (second-hand): about 4 Elo per 1,000 miles traveled, +25 Elo off a bye, and a rolling QB VALUE adjusted for opponent defense. |
| nfelo v4.3.0 | Game outcome = 74% margin, 15% WEPA (weighted EPA), 11% PFF grades. Preseason rating = 53% reversion plus priors from market win totals (weight 0.45), DVOA (0.36) and a units model (0.18). Home-field terms in points: home off bye +0.3, away off bye −0.8, kickoff-time body-clock advantage +0.2, surface difference +0.35, divisional −0.31 vs non-divisional +0.09. The line is regressed toward the market, and the QB is valued by nfeloqb. |
| nfeloqb | EWMA QB value with career and league regression, a draft-slot rookie prior, discounting of the team's prior QB value, a 10% team adjustment on a backup's first start, and wind and temperature discounts. |
| ESPN FPI (second-hand) | EPA-based offense, defense and special-teams components with opponent adjustment and blowout capping. Several seasons weighted toward the latest. Preseason priors (including returning starters and QB) that fade during the season. Game adjustments for site, rest days and travel distance. QB via predictive QBR. |
| FTN DVOA / DAVE (second-hand) | Preseason projection from three seasons of DVOA, a separate QB projection and other variables, blended with in-season DVOA (93% projection in week 1). |

### What we already have

- **Rest:** `days_rest` and `opp_days_rest` in the models, and rest difference in the ridge design.
- **Venue:** a stadium-level ridge block. It covers altitude and stadium-specific home field, which is more
  flexible than one home-field constant.
- **Weather:** in the play models and the over-expectation targets.
- **QB and players:** player value with draft and age priors, the scheduled QB1, and QB-team continuity.
- **Coaching changes:** coach-change decay was tested, with about zero gain.
- **Luck and blowout adjustment:** tested through `epa_luck.py`, with no significant gain.

### Tested here: game context we lack

`exp_game_context.py` tests these features on home margin for 2,735 regular-season games from 2016-25,
excluding neutral sites. It uses leave-one-season-out regression and two controls. The first is a pregame
team rating (recency-weighted ridge on margins, refit each week on earlier games). The second is the closing
spread.

| Feature | Points per unit (t), rating control | Held-out MSE change, z (negative = better) |
| --- | --- | --- |
| Rest difference (days, capped ±7) | +0.23 (2.2) | −0.7 |
| Away team off a bye | −1.8 (−1.7) | +0.5 |
| Travel, per 1,000 miles | −0.28 (−0.7) | +0.9 |
| Time zones crossed eastward | +0.06 (0.3) | +1.9 |
| West team at a 1 p.m. ET kickoff | +0.69 (0.8) | +0.8 |
| East team at a west host at night | +0.74 (0.5) | +1.7 |
| Divisional game | −0.28 (−0.5) | +0.9 |
| Surface differs from the away team's home surface | +0.42 (0.8) | +0.5 |
| Cold game (< 40°F) against a southern away team | +2.3 (1.5) | −0.5 |

None of these improves held-out margins, and with the closing spread as the control none does either. Only
rest difference, which we already have, has a reliable coefficient. Travel, body clock, divisional games and
surface do not earn a place on our current evidence. This agrees with a published 2015-25 study that found
travel effects small and insignificant once team strength (FPI) is controlled. The cold-weather mismatch is
the only one leaning positive. It is a weather × team interaction, so it belongs in the play models, which
already see temperature, rather than as a game-level term.

### Tested here: market information as a prior

nfelo's largest single input besides results is the market: win-total priors and regression toward the
line. `exp_market_ref.py` compares team ratings fit on past results with ratings fit on **earlier weeks'**
closing lines (no same-game information). Margin RMSE, 2016-25:

| Rating | Weeks 1-4 | Weeks 5+ |
| --- | --- | --- |
| From past results | 13.27 | 13.26 |
| From past closing lines (tuned) | 13.22 | 13.12 |
| Both | 13.27 | 13.26 |
| This game's closing line | 12.95 | 12.63 |

Over all weeks, the two ratings together reach 13.10, against 13.21 for results alone.

Earlier weeks' lines carry strength information that results alone miss: about 0.1-0.15 points of RMSE
after week 4, and little in weeks 1-4. Last season's lines are a weak preseason prior. Market win totals,
which nfelo weights most heavily, were not available to test. Most of the market's edge is specific to this
week (injuries, QB status, matchups). A team-additive rating on past lines correlates only about 0.8 with the
current line. So a market-informed preseason prior is worth testing if depending on the market
is acceptable. The bigger target remains this week's lineup information, which the player-value and usage
work already pursues.

## Drive efficiency: points per drive (Silver Bulletin) and expected drive points (Fortgang)

Silver Bulletin's ELWAY projects points per play and plays per game for each team and simulates drives (the
2025 league average was about 2.1 points per drive). Judah Fortgang's *Earned Drive Points* (EDP; "Introducing
Earned Drive Points (and our Drive Quality model)", SportfolioKings, June 2023) rates drives without their
outcomes. It has two layers:

1. **A team-aware score expectation per play.** It is an XGBoost model with the EP model's situation inputs
   plus rolling short- and long-term offense and defense strength (yards per play, success, explosive plays,
   sack rate, among others).
2. **EDP from four drive components:**
   - scoring expectation gained over the drive;
   - the volatility of those gains (big swings get "nicked");
   - the average expectation over the drive relative to its start;
   - a yards-per-play term that removes penalty yardage.

The article reports week-over-week stability more than twice EPA/play's, 3-week vs next-3-week stability of
0.47 (EPA 0.29, success 0.31), and a 0.27 correlation with the next three weeks' points (EPA 0.21). It does not
publish its weights.

The first table below compares simpler drive measures. The rebuild of EDP from the article's description
follows in its own subsection.

**Test (`exp_drive_metrics.py`).** Each metric becomes an opponent-adjusted, recency-weighted ridge rating for
offense and defense, refit each week on earlier games. A team's points are then predicted, leave one season out
(2017-26, 5,066 team-games), from the baseline plus the metric. The baseline is EPA per play and plays per game,
for this team's offense against the opponent's defense and for the reverse matchup. Our team inputs are
EPA-based, and the sim generates pace itself, so this is the fair comparison. "Team-relative" versions subtract
each week's league mean, so the 2024-25 kickoff rule changes can't pose as team skill.

| Added to the baseline (both sides) | Team points RMSE | Margin RMSE | z (negative = better) |
| --- | --- | --- | --- |
| Baseline: EPA/play + plays | 9.595 | 13.450 | |
| + success rate | 9.595 | 13.446 | +0.0 |
| + points per play (ELWAY-style) | **9.524** | **13.278** | **−4.4** |
| + points per drive (Silver Bulletin) | 9.534 | 13.302 | −4.0 |
| + mean EP over the drive, including the start (simple proxy) | 9.533 | 13.285 | −3.9 |
| + EP gained within drives (mean EP − start EP) | 9.594 | 13.443 | −0.3 |
| + points per drive over starting EP | 9.569 | 13.402 | −2.5 |
| + drive start EP (field position) | 9.542 | 13.301 | −3.6 |
| + start EP after punts / after kickoffs | 9.544 / 9.545 | 13.298 / 13.319 | −3.5 / −3.5 |
| + share of drives starting from takeaways or downs | 9.593 | 13.443 | −0.4 |
| + red-zone trips per drive | 9.557 | 13.349 | −3.1 |
| + red-zone TD rate | 9.598 | 13.453 | +0.7 |
| + points scored/allowed | 9.520 | 13.268 | −4.5 |
| + points per drive, team-relative | 9.540 | 13.302 | −3.6 |
| + drive start EP, team-relative | 9.542 | 13.301 | −3.7 |
| + points per drive + start EP + red-zone trips, team-relative | 9.544 | 13.292 | −3.1 |

How stable each rating is (correlation with the next game's value of the same metric): EPA/play 0.34, points per
drive 0.33, EP gained within drives 0.29, drive start EP 0.51, start EP after punts 0.34, red-zone trips per drive
0.29, red-zone TD rate 0.07, takeaway-started drives 0.12.

### Earned Drive Points, rebuilt from the article

`exp_edp.py` follows the article's recipe on nflverse data:

- **Score expectation.** LightGBM on the EP situation plus rolling, strictly prior offense and defense yards
  per play, success, explosive and sack rates over the last 3 and 16 games. The target is the drive's points.
  It is walk-forward: each season is scored by a model fit on earlier seasons.
- **Gain.** The expectation before the drive's last snap minus the expectation at its first snap. The last
  play's own result is left out, which reproduces the article's examples: a fumble at the 5 keeps the credit
  for reaching it, and a one-play 80-yard TD earns little.
- **The other three components.** Volatility is the SD of snap-to-snap changes, average-relative is the mean
  expectation minus the start, and clean YPP excludes penalties.
- **Weights.** Fit by regressing drive points on the components (plus the starting expectation), on earlier
  seasons only.

The rebuilt EDP correlates 0.77 with a drive's actual points.

**Stability (`exp_edp_stability.py`, offense, raw team-game values, 2017-26).** These match the article:

| Metric | Game → next game | 3 games → next 3 | 3 games → next 3 games' PPD | Season → next | Season → next PPD |
| --- | --- | --- | --- | --- | --- |
| EPA/play | 0.17 | 0.35 | 0.34 | 0.41 | 0.41 |
| Success rate | 0.18 | 0.39 | 0.34 | 0.40 | 0.39 |
| Points per drive | 0.16 | 0.35 | 0.35 | 0.42 | 0.42 |
| EDP (rebuilt) | **0.26** | **0.47** | 0.32 | **0.45** | 0.37 |
| EDP's starting expectation alone | 0.31 | 0.50 | 0.26 | 0.49 | 0.30 |

EDP is the most stable metric, as the article says (its 0.47 and our 0.47 agree). It does not predict future
points better, though. Much of the stability is built in: the score expectation takes rolling team strength as
input, so consecutive games share inputs. The starting expectation, which is almost entirely those inputs, is
the most stable piece and the weakest predictor. Stability alone is the wrong yardstick for a feature.

**Held-out points test (`exp_drive_metrics_edp.py`).** This is the same design as above, with test seasons
2018-26 (4,554 team-games) so the rebuilt EDP has a season of history:

| Added to EPA/play + plays (both sides) | Team pts RMSE | Margin RMSE | Total RMSE | z |
| --- | --- | --- | --- | --- |
| Baseline | 9.541 | 13.372 | 13.613 | |
| EDP, rebuilt | 9.501 | 13.310 | 13.563 | −2.7 |
| EDP's components, weighted by the held-out regression | 9.500 | 13.321 | 13.548 | −2.4 |
| EDP's components without the starting expectation | 9.507 | 13.342 | 13.546 | −2.2 |
| Volatility component only | 9.540 | 13.359 | 13.621 | −0.3 |
| Points per drive | 9.481 | 13.232 | 13.582 | −3.7 |
| Points per play | **9.469** | **13.206** | 13.574 | **−4.2** |
| Drive start EP | 9.495 | 13.232 | 13.621 | −3.1 |
| Points per drive + EDP | 9.470 | 13.234 | 13.550 | −3.9 |
| Points per drive + drive start EP + EDP | 9.466 | 13.229 | **13.543** | −4.0 |

Rebuilt EDP adds real information over EPA/play, but less than points per drive or points per play. On top of
points per drive it helps totals (13.58 → 13.55) and not margins. Its team-aware gain and clean-YPP terms carry
something that EPA misses. The volatility penalty carries nothing.

The article weights its components to describe a drive. Fitting the weights to predict *future* points
instead, for example by regressing next-three-game points per drive on rolling component averages, might
narrow the gap; this rebuild doesn't test that. Since the exact weights and model aren't public, the rebuild
may also understate the original.

What this says, across both tables:

- **Drive efficiency carries information EPA per play misses**, worth about 0.15-0.17 points of margin RMSE.
  It survives both teams being in the baseline and removing league-wide trends. Points per play, points per
  drive and the mean-EP proxy carry largely the same signal: combining them adds little over the best one.
- **Where the gain comes from.** Field position (where a team's drives start after punts and kickoffs), and
  reaching the red zone per drive.
- **Where it doesn't come from.**
  - The EP a team gains *within* drives, which is the part closest to an outcome-free drive-quality measure.
    EPA per play already holds that.
  - Red-zone TD rate, which barely repeats (0.07).
  - Takeaway-started drives (0.12).
- This fits an open item in the October review: the sim's yards are calibrated but its **points per 100 yards
  are not** (dispersion 1.24-1.32). Field position and scoring-range trips are exactly how yards turn into points.

### As inputs to the simulator's per-play models

The tables above rate teams; the simulator needs per-play inputs. `exp_play_features.py` asks the question
directly. It mirrors the sim's per-play models on nflverse data: play call, rush yards, sack, completion,
interception, air yards, YAC, punt net yards and kickoff start, with binned targets as the CORN models use.
It compares a LightGBM model on the situation plus EPA-style team ratings with the same model plus offense and
defense ratings of the drive metrics:

- **Situation:** down, distance, yardline, half and game seconds, score, quarter, timeouts, home and weather.
- **Baseline team ratings:** pass EPA, rush EPA, success, sack rate, plays and pass rate, offense and defense.
- **How ratings enter:** opponent-adjusted, refit weekly on earlier games, and attached as the offense's and
  the defense's ratings, as the sim feeds team columns. League mean removed each week.
- **Split:** train 2018-24, test 2025-26. Kickoffs train through 2025 and test 2026, because the 2025 rule has
  no earlier examples.

Change in held-out log loss (negative = better), with z clustered by game:

| Model (test rows) | PPD + points/play | Drive start EP | Red-zone trips | EDP | All drive features |
| --- | --- | --- | --- | --- | --- |
| Play call (55,990) | +0.10% (1.8) | +0.03% | +0.03% | +0.02% | +0.03% |
| Rush yards (20,052) | +0.10% (1.7) | +0.05% | 0.00% | +0.03% | −0.04% |
| Sack (30,348) | −0.04% | +0.16% | −0.06% | −0.01% | −0.08% |
| Completion (26,798) | −0.01% | +0.07% | +0.06% | +0.05% | +0.10% (1.5) |
| Interception (26,798) | +0.12% | +0.25% | +0.33% | −0.04% | +0.41% |
| Air yards (26,798) | +0.12% (**2.4**) | −0.01% | +0.08% | +0.06% | +0.09% |
| YAC (17,270) | +0.04% | −0.09% | +0.03% | −0.05% | 0.00% |
| Punt net yards (3,075) | −0.39% (−1.4) | −0.48% (−1.6) | −0.15% | −0.02% | −0.18% |
| Kickoff start, 2026 (1,537) | −0.67% (−1.4) | −0.71% (−1.2) | −0.55% (−1.4) | −0.29% | −1.44% (**−2.4**) |

**For the scrimmage models these features add nothing.** Every change is within ±0.4%, almost all within noise,
and the one significant change (air yards) is worse. Once a play knows its down, distance and yardline, and
the teams' EPA-style ratings, drive efficiency says nothing more about that play. The special-teams models
lean slightly positive, but the samples are small. On kickoffs, red-zone trips help about as much as
drive-start position, which points to generic team strength rather than field-position skill. Call that
inconclusive.

So the game-level gain above does not come through per-play inputs. It has to come from how the sim's plays
compound into drives: field position after punts, kickoffs and turnovers, and how often drives reach scoring
range. That makes these metrics **calibration targets** for the sim rather than model inputs.

Recommendations (these replace the earlier suggestion to feed drive ratings to the play models):

1. **Add per-team drive statistics to `joint_outcomes`:** points per drive, drive start EP (after punts and
   after kickoffs separately), and red-zone trips per drive, as calibration statistics next to points per 100
   yards. Score both levels and dispersion: do simulated teams *differ* as much as real ones?
2. **If simulated drive starts vary too little between teams, fix it in the special-teams models.** The
   October review found their team inputs carry little signal (4.8% skill for punt distance, about 0 for
   kickoffs). Opponent-adjusted punt and kickoff ratings (kicking team allowed, receiving team gained) are worth
   a refit-and-backtest test, as the slightly positive punt and kickoff results suggest. Test them over more
   seasons than here, for example leave-one-season-out on 2018-25.
3. **Don't add points per drive, points per play, red-zone trips or EDP to the scrimmage models** (`play`,
   `rush`, `comp`, `int`, `ay`, `yac`, `sack`).
4. Red-zone TD rate is noise at the team level (week-to-week correlation 0.07).

## Split, charting and tracking features in the per-play models

The pipeline uses pass location, run location and gap, shotgun and no-huddle only as controls in the ridge
over-expectation baseline (`ridge/expectations.py`). It uses FTN charting only for interception-worthy throws
and routes, and Next Gen Stats not at all.

`exp_split_features.py` tests each family as inputs to the sim's per-play models. The design is the same as
above: LightGBM with the situation plus EPA-style team ratings, train 2018-24, test 2025-26, game-clustered z.
Two families get their own baselines:

- **Next Gen Stats:** both sides of that comparison also get box-score player ratings like the sim's (passer
  EPA, CPOE and sack rate; receiver yards per target, catch rate and YAC; rusher yards per carry).
- **FTN charting:** starts in 2022, so it trains on 2023-24 against a baseline on the same rows.

Every input is available during the season. Play-by-play location fields, FTN charting and NGS weekly files
were all current through week 10 of 2026 on November 19. Participation-only fields (personnel, defenders in
the box, coverage, charted pressure) are excluded.

Change in held-out log loss (negative = better), z in parentheses when |z| > 1.3:

| Family | Play call | Rush yds | Sack | Completion | INT | Air yds | YAC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Team pass EPA by location + middle share | +0.08 | +0.17 (2.6) | +0.08 | +0.06 | −0.04 | −0.08 (−1.4) | −0.04 |
| Team rush EPA inside/outside + by direction | +0.21 (3.6) | +0.11 (1.5) | +0.04 | +0.04 | −0.27 | −0.04 | +0.04 |
| Team 3rd-down conversion, early-down EPA | 0.00 | +0.14 (2.4) | +0.19 (1.4) | +0.03 | −0.03 | +0.10 (1.9) | −0.01 |
| Team shotgun, no-huddle rates | +0.03 | −0.03 | +0.06 | +0.03 | −0.50 (−1.5) | −0.05 | −0.05 |
| Team FTN charting: play action rate and EPA, screen and motion rates, blitz rate, box count, pass rushers | +0.05 | −0.08 | +0.10 | +0.21 (1.6) | −0.80 | **−0.29 (−2.7)** | −0.24 (−1.8) |
| Player NGS: QB time to throw, aggressiveness, intended air yards, CPOE; WR separation, cushion, YAC over expected; RB yards over expected, efficiency, time to LOS, 8+ box share | | +0.02 | +0.04 | −0.10 (−1.4) | −0.22 | **−0.25 (−3.6)** | +0.01 |
| Location stage: the play's actual location | | −0.59 (−8.3) | | −0.74 (−7.8) | −0.14 | | −0.23 (−3.2) |
| Location stage: + team ratings at that location | | **−0.23 (−3.1)** | | −0.01 | −0.37 | | +0.02 |

What this says:

- **Team splits by location, situation and formation add nothing**, and some make the models slightly worse.
  Opponent-adjusted EPA by pass location or run direction is too thin per team to beat the overall ratings.
- **Air yards is the one model that gains**, from NGS (−0.25%, z −3.6) and FTN (−0.29%, z −2.7). Both likely
  overlap inputs the sim already has but this baseline lacks: `adot`, the receivers' deep, medium and short
  shares, and the passer's depth rates. FTN's screen rate is the exception, since the sim has nothing like it
  and screens set air yards at or behind the line. Test FTN screen rate (offense) on the `ay` model; NGS adds
  little elsewhere (completion −0.10%, z −1.4).
- **A location stage.** Knowing the actual pass location or run gap improves completion, YAC and rush yards a
  lot. But that is an oracle: before the snap the sim must draw the location from its own model. In theory a
  correct one-step model already equals the two-step mixture (Σ_l P(l | x) P(y | l, x) = P(y | x)), so a stage
  adds nothing. In practice it does, because the one-step model doesn't learn the interactions the stage writes
  down. `exp_location_stage.py` measures it on held-out 2025-26 rushes (19,909), scoring the two-step model's
  full mixture over drawn locations:

  | Rush yards model | Log loss | vs A | z |
  | --- | --- | --- | --- |
  | A one-step, base inputs | 2.2935 | | |
  | B one-step + direction ratings and offense tendencies as plain inputs | 2.2925 | −0.04% | −0.5 |
  | C two-step: drawn location, then yards with offense/defense ratings at that location | 2.2881 | **−0.24%** | **−4.4** |
  | D two-step: drawn location, no location ratings | 2.2909 | −0.11% | −2.2 |
  | O oracle: actual location | 2.2807 | −0.56% | −6.8 |

  The drawn-location stage keeps about 40% of the oracle's gain, even though the location model barely beats
  league frequencies (log loss 1.851 vs 1.881). It clearly beats giving the same information as plain inputs
  (B). Most of its gain comes with the matchup ratings (C vs D). So build the run-direction stage.

  `exp_pass_location_stage.py` runs the same test for passes. The location (left, middle, right) is drawn after
  air yards and the target, as the sim orders them. The location model sees the offense's middle share, the
  defense's middle share allowed and the receiver's prior location shares. It barely beats league frequencies
  (1.026 vs 1.061). YAC uses location weights updated for the catch. Held-out 2025-26 throws (26,797):

  | Model | Completion | Interception | YAC (given catch) |
  | --- | --- | --- | --- |
  | A one-step (log loss) | 0.5719 | 0.0986 | 1.9905 |
  | B one-step + location ratings and tendencies as plain inputs | +0.03% | +0.08% | −0.06% |
  | C two-step: drawn location + team pass EPA at that location | +0.11% (z 1.9, worse) | −0.31% (z −1.0) | −0.14% (z −2.1) |
  | D two-step: drawn location, no location ratings | −0.05% | +0.24% | −0.04% |
  | O oracle: actual location | −0.46% (z −5.3) | −0.28% | −0.14% (z −1.9) |

  For passes the stage loses the oracle's gain. Completion is exactly the textbook case: knowing the real
  location helps (−0.46%), but a drawn location gives it back (C and D are about 0). The only gain is a small
  one on YAC (−0.14%, z −2.1), from the matchup ratings rather than the location itself, since C equals the
  oracle. With 12 comparisons in this table, one at z −2.1 is weak evidence. **A pass-location stage is not
  worth building**; if anything, test team YAC over expected by location as a YAC input.

Recommendations:

1. **Run direction stage for `rush`:** add a model for run location and gap (team tendencies plus situation)
   and the matching offense and defense rush ratings (inside vs outside, direction) as `rush` inputs given the
   drawn direction. It also lets the sim report run direction in player stats.
2. **FTN screen rate (and play-action rate) for `ay`,** tested against the sim's existing depth inputs.
3. **Skip** team pass-location splits, third-down and early-down splits, shotgun and no-huddle rates, and
   NGS for completion, sack, interception, YAC and rush yards.

## Not from the forecasters, but found while looking: referee crews

The schedule names each game's referee. A crew's shrunk, strictly prior penalty rate over league average cuts
the squared error of a game's penalty count by **3.3%** (2,191 games, 2018-26; crew effects have an SD of 0.68
penalties per game). None of the forecasters above use it. The sim's four penalty models currently see game
state and team features only, so a `referee_penalty_rate` input is cheap and should help the penalty volume
and, slightly, the totals.

**Use it only with a pregame source.** nflverse's `games.csv` fills `referee` only after a game is played: on
November 19, 2026, none of week 11's games, including that night's, had one. The league releases each week's
crew assignments midweek, and Football Zebras and Pro Football Network publish them. So the input works for
forecasts only if the weekly refresh pulls that assignment list. Training can use the after-the-fact column,
since it names the same crews. A game without an announced crew should get the neutral value (league
average), never a guess.

## Summary: what to add

| Priority | Feature | Where |
| --- | --- | --- |
| 1 | Era / league-environment inputs | All play models (nflfastR uses era dummies in every model) |
| 1 | Half seconds, quarter, both timeouts, `receive_2h_ko` | `play`, `time`, 4th-down and two-point decisions |
| 2 | `Diff_Time_Ratio`, strength × time decay | `play`, `time`, decision models |
| 1 | Points per drive, drive start EP, red-zone trips per drive | Calibration checks in `joint_outcomes`, not per-play inputs (no gain in the scrimmage models) |
| 2 | Opponent-adjusted punt and kickoff field-position ratings | `punt_distance`, `punt_return`, `kickoff`, if the calibration check shows too little spread between teams |
| 2 | Referee crew penalty rate, only if the weekly refresh pulls the midweek assignment list | Penalty models |
| 2 | Run-direction stage with offense/defense rush ratings by direction | `rush` (new location model before it) |
| 3 | FTN screen and play-action rates | `ay`, tested against the existing depth inputs |
| 2 | `distance_to_sticks`, `air_is_zero` | `comp`, `int`, `yac` |
| 3 | Market-informed preseason prior (optional) | Ridge preseason start, if market dependence is acceptable |
| — | Travel, time zones, body clock, divisional, surface | Not recommended: no held-out gain |
