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

## Not from the forecasters, but found while looking: referee crews

The schedule names each game's referee. A crew's shrunk, strictly prior penalty rate over league average cuts
the squared error of a game's penalty count by **3.3%** (2,191 games, 2018-26; crew effects have an SD of 0.68
penalties per game). None of the forecasters above use it. The sim's four penalty models currently see game
state and team features only, so a `referee_penalty_rate` input is cheap and should help the penalty volume
and, slightly, the totals.

## Summary: what to add

| Priority | Feature | Where |
| --- | --- | --- |
| 1 | Era / league-environment inputs | All play models (nflfastR uses era dummies in every model) |
| 1 | Half seconds, quarter, both timeouts, `receive_2h_ko` | `play`, `time`, 4th-down and two-point decisions |
| 2 | `Diff_Time_Ratio`, strength × time decay | `play`, `time`, decision models |
| 2 | Referee crew penalty rate | Penalty models |
| 2 | `distance_to_sticks`, `air_is_zero` | `comp`, `int`, `yac` |
| 3 | Market-informed preseason prior (optional) | Ridge preseason start, if market dependence is acceptable |
| — | Travel, time zones, body clock, divisional, surface | Not recommended: no held-out gain |
