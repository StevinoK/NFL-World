# Player Value (Weekly Plus-Minus)

Run from the workspace root using the dedicated PyMC environment:

```powershell
& '.\NFL_PM.venv\Scripts\python.exe' -m player_value.plus_minus
```

The module entry point runs fast `method='box'` for QB, RB, WR, TE, OL, DL, LB,
and DB from 2016 through the current season, using the calibrated in-season settings
described below. Seasons not in `data/cache/` (normally the current one) are downloaded. The constructor still defaults to `method='bayesian'` for
existing callers. Box mode performs no PyMC fitting, sampling, or ensemble fits.
Each estimate uses regular-season plays through the end of its labeled week.
For pregame features, use the last completed week, not the week being predicted.
Clustering features come from play-by-play events. OL and defensive exposure
still requires participation data; missing participation prevents those estimates.

All eight feature bins and on/off outputs require an exact GSIS player ID and
season match to `nfl.import_seasonal_rosters`. Event roles do not establish
positions: a WR jet sweep does not qualify its rusher for RB estimates. Both
the current `player_id` roster schema and the older `gsis_id` schema are accepted.
Roster labels are normalized into QB, RB (including FB/HB), WR, TE, OL
(including C/G/T/OT/OG/LT/RT/LG/RG), DL, LB, and DB. Unrecognized labels and
unmatched players are excluded; missing required roster columns or seasons
raise an error instead of admitting every player. Receiver classification also
uses the play's season, not the career player table. If multiple roster records
explicitly identify different groups for one player-season, each listed group
is eligible, without duplicating rows within a bin. This is seasonal roster
eligibility, not a claim about position on each individual snap.

## In-Season Updating (Calibrated)

`python -m player_value.plus_minus` builds each position with `PlayerPlusMinus.calibrated`. That
constructor reads `player_value/config/in_season_update.json`, which `player_value.calibration`
writes. The calibration scores the weekly estimate `n/(n+T)*b + T/(n+T)*h` against each player's
rest-of-season raw on/off (exposure-weighted) for seasons 2019-2025. It searches:

- the prior `h`: `zero`, `decay` (last season's estimate times `decay**gap`, the original method), or
  `regression` (prior-season estimates, age, draft slot and experience, refit each season on
  earlier seasons only, which gives rookies a draft/age prior);
- the prior strength `T` (`box_prior_snaps`) and the decay.

```powershell
& '.\NFL_PM.venv\Scripts\python.exe' -m player_value.calibration           # report only
& '.\NFL_PM.venv\Scripts\python.exe' -m player_value.calibration --write   # update the config
```

Other box options: `box_weighting='plays'` weights historical training rows by on-field plays
instead of equally, and `madden_features=True` (OL/DL/LB/DB) adds Madden preseason ratings from
`data/madden/` (see `player_value/madden.py` for the source; files are not committed).

## Player Estimate Explorer

Launch the saved weekly estimates from the workspace root (or run the helper
in VS Code with `NFL_PM.venv` selected):

```powershell
& '.\NFL_PM.venv\Scripts\python.exe' -m player_value.run_explorer
```

Requires `streamlit` and `plotly` in that environment. The helper prints its
localhost URL, using the first available port starting at 8510. The explorer
reads the eight weekly splits CSVs beside the app, without importing the model
or running a fit. Search by saved name and select a stable player ID; identical
names include IDs. Source-file position labels are accepted only when the local
authoritative `player_season_positions.parquet` lookup confirms that group for
the player's exact season. Each season/team/position is a separate series;
traded-team snapshots are not averaged, and missing weeks break lines.
Box rows use `box_plus_minus`, Bayesian rows use `bayesian_plus_minus`; no
intervals are displayed. Values are cumulative through the labeled week.
CSV and roster-lookup modification-time/size changes invalidate the load cache on the next rerun
(reload the browser after replacing exports); there is no automatic polling.
The explorer never downloads rosters on a UI rerun. A missing or malformed
lookup prevents display. Production input loading refreshes this lookup;
refresh it independently, without fitting or changing saved estimates, using:

```powershell
& '.\NFL_PM.venv\Scripts\python.exe' -m player_value.positions --years 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
```

Filtering legacy exports fixes displayed membership only. Previously fitted
models may have trained on ineligible players, so retained old estimates are
not corrected numerically by this display filter. Refit the production pipeline
to regenerate those estimates. Model/checkpoint fingerprints cover both the
pipeline and shared position-mapping source; pre-fix bundles and checkpoints
cannot satisfy the new fingerprint and requested seasons are recomputed on resume.

```powershell
& '.\NFL_PM.venv\Scripts\python.exe' -m unittest player_value.tests.test_explorer -v
```

## Fast Box Plus-Minus

```python
from player_value.plus_minus import PlayerPlusMinus

model = PlayerPlusMinus(
    position='QB', years=list(range(2016, 2027)), method='box',
)
model.run_full_pipeline(incremental=True)
results = model.get_results()
```

Include historical years in `years`, not just the target year. For each target
season S, training pools eligible full-season player/team rows from seasons
strictly less than S. A historical season must be older than the current NFL
season (September rollover) and its loaded regular-season data must reach week
17 through 2020 or week 18 thereafter. This is a conservative standard-schedule
check, not proof that every scheduled game or participation record has arrived.
Do not supply partial historical feeds marked as final. There is no automatic
extra historical download; saved score CSVs are not a substitute for training
features. Years provided are both available training years and output years.

Each position/target-season window fits historical feature medians and a shared
`StandardScaler`. All positions default to `box_cluster_iterations=20` sklearn ensemble
members, each with KMeans, a `OneHotEncoder`, and `Ridge`. Seeded random k uses
`cluster_k_min` and `cluster_k_max` (defaults derived from `config['n_clusters']`); bounds are capped
at available historical rows and distinct profiles. Explicit `box_cluster_iterations` accepts integers 1 through
100. Weekly predictions and historical prior predictions average all members;
the exported `cluster` is only the last member's diagnostic label, not a shared
ensemble grouping. Bundle `kmeans`/`encoder`/`ridge` fields retain the first
member for compatibility; `members` holds the fitted ensemble.
There is no current-season fitting of imputation, scaling, clustering, encoding,
or regression, and no MCMC. Missing and nonfinite features use historical
medians; wholly missing historical features use zero. A singleton historical
profile uses one cluster. Settings and source changes invalidate model and
weekly checkpoint fingerprints.

QB ON includes attributed dropbacks, including scrambles: use the passer ID,
falling back to the rusher ID only when `qb_scramble == 1`. The attributed player
must be a roster-eligible QB in that season. Unattributed plays and non-QB
passes are excluded from both sides; arbitrary rusher IDs never fill pass IDs.
The same attribution drives QB features and splits, without mutating input PBP.
QB OFF is the exposure-weighted EPA of all OTHER roster-eligible QBs across ALL
teams in the same season through the cutoff. All of the focal player's team
stints are excluded from OFF. `reference_definition` identifies this league
reference; it is not a single-team backup comparison or a within-cluster target.
Cluster controls capture profile differences without forcing every archetype's
mean to zero. Zero means average-other-QB EPA, not causal replacement value.
No other QB exposure yields a missing raw target, not a fabricated zero reference.

Other positions retain their existing team on/off definitions: offense ON EPA
minus OFF EPA, with the sign reversed for defense. RB/WR/TE ON uses event IDs;
OL and defense use participation exposure. Historical rows with no OFF
observations or nonfinite targets are excluded. Training rows are one
player/team/season, not repeated weekly snapshots, with equal row weights.
Features can include EPA rates, so this is descriptive, not causal player value
or a future-performance validation claim. `box_alpha=100.0` remains fixed.

Reproduce the attribution investigation using a local cached season only:

```powershell
& '.\NFL_PM.venv\Scripts\python.exe' -m player_value.qb_target_diagnostics --season 2024
```

Cached 2024 regular-season pass plays included 1,062 missing-passer scramble
runs at +0.539 EPA/play and 1,167 unidentified non-scramble no-plays at +0.418.
For stints with at least 300 ON plays, mean raw PM was -0.176 under the old
team definition, +0.037 after attribution correction, and +0.021 with the
league reference. These are diagnostic averages, not validation accuracy.

Let b be the current cumulative-profile ridge prediction, h the player prior,
n the current `n_on`, and T=`box_prior_snaps` (default 200). The output is:

```text
w = n / (n + T)
box_plus_minus = w * b + (1 - w) * h
```

For history, the same target-season model predicts completed historical profiles.
Each historical stint prediction is first shrunk toward zero using its own
`n_on / (n_on + T)`. The latest historical season per player is combined across
teams using that season's `n_on` weights, then aged by
`history_decay ** elapsed_years` (default 0.8). History follows player IDs across
trades and is recomputed from historical rows, not recursively from saved weekly
scores. Rookies/players without eligible history have h=0. If no eligible training
rows exist, both prediction and prior are zero, so the first available training
season produces explicitly untrained zero estimates. Sparse weekly profiles
receive stronger shrinkage; the weight increases monotonically with sample size.
These defaults are conservative starting settings, not calibrated parameters.

Fitted bundles are reused once per position/target season within a run and, by
default, persisted as `outputs/models/box_model_{position}_{season}_{fingerprint}.joblib`.
Fingerprints cover historical features/labels, target season,
configuration, script source, and sklearn version. `box_cache=False` disables disk
cache reads/writes but retains in-run reuse. Repeated runs still load inputs and
rebuild historical event summaries to verify the training fingerprint; cached
bundles avoid sklearn refits. Use only trusted locally generated joblib files,
which are pickle-based. Old fingerprinted files may be removed to reclaim space.

Box mode does not estimate Bayesian credible intervals. `pm_lower`, `pm_upper`,
and `pm_uncertainty` are missing, and `interval_kind='not_available'`. No confidence
coverage or calibrated uncertainty is claimed. Rankings/plots label the estimator
and box plots show point estimates without error bars.

## Bayesian Historical Priors

Choose `method='bayesian'` to retain the original cumulative weekly clustering
ensemble and hierarchical PyMC method. The following prior/likelihood settings
describe that method, not box regression.

The latest completed-season player posterior supplies the next season's prior.
History follows player IDs across team changes. Only the final weekly snapshot
from each season is considered, and only the latest available season per player
is used: earlier posteriors already contain their own historical priors.
Multiple team stints are combined using current-season exposure weights, with
between-stint variation retained in the uncertainty.

Default historical mean retention is 0.8 per elapsed year. Prior uncertainty is
the historical interval-derived variance plus 0.06 squared per elapsed year.
Players without history have a zero-centered prior with standard deviation 0.10.
Cluster adjustments use tighter 0.05-scale priors. These are initial modeling
settings, not calibrated hyperparameters. All scales refer to EPA per play.

In the default full mode, only earlier seasons included in the run can supply
historical priors. The first requested season uses the no-history prior; include
earlier years to establish history before a target season. Incremental mode can
also load completed-season history outside the requested years, as described below.
Current-season cumulative data is refit from scratch each week, so a preceding
weekly posterior is not counted again as a prior.

Both methods use these default cumulative weekly eligibility thresholds; box
training also uses the configured threshold on each completed-season row:

| Position | Minimum Sample |
| --- | ---: |
| QB | 10 passing plays |
| RB | 5 rushes plus targets |
| WR / TE | 3 targets |
| OL | 20 eligible participation plays |
| DL / LB / DB | 15 eligible participation plays |

Use `weekly_min_snaps` to override the threshold. An explicitly supplied `config`
retains its own threshold unless `weekly_min_snaps` is provided. Current-season
Bayesian likelihood precision scales with absolute sample size relative to 100 plays,
not relative to the busiest player in a sparse week. More current-season data
therefore gradually outweighs the prior.

Constructor options are `history_decay`, `season_transition_sd`,
`rookie_prior_sd`, and `observation_reference_snaps`.

## Incremental Updates

The option is off by default. Full runs still refit every available week and save
once at the end, replacing previous output rather than merging it.

```python
from player_value.plus_minus import PlayerPlusMinus

# Run from the repository root.
model = PlayerPlusMinus(position='QB', years=[2026], method='bayesian')
model.run_full_pipeline(incremental=True)
results = model.get_results()
```

For Bayesian mode, incremental updates download input for the requested years, load saved history for
the position independently of those years, and fits only missing or invalid
season/week snapshots. Each successfully fitted week is committed immediately,
including its ensemble iterations. Restarting after a failed fit keeps previously
committed weeks and retries the unfinished week. All other saved years/weeks and
columns are retained. Duplicate player/team/season/week identities (plus iteration
for raw fits) keep the last saved row; an invalid week is replaced as a whole only
after its replacement fits successfully. Weeks with no eligible players are not
marked complete and will be reconsidered on a later run.

A saved week is complete only if every averaged player/team has all configured
iteration IDs, both tables contain finite estimates/intervals, their player keys
match exactly, and their means agree. Legacy CSV pairs without a checkpoint can
be imported under the same checks. New checkpoints include a method/configuration/
source manifest. A method or configuration mismatch is not resumed and requested
outputs are recomputed. Legacy Bayesian CSV pairs/checkpoints without a manifest
retain iteration validation; their other settings cannot be verified automatically.
Use full mode to deliberately recompute after changing settings or correcting
historical inputs. Do not run simultaneous writers for the same position.

Saved priors require a validated final regular-season snapshot: week 17 through
2020, week 18 from 2021 onward, from a season older than the current NFL season
(September year rollover). An arbitrary latest saved earlier week is never
promoted to a completed-season prior. If no such snapshot exists, the player
uses the no-history prior or their latest older completed season. This is a
conservative standard-schedule assumption, not a schedule/game-completion service;
nonstandard seasons need explicit handling, and the current NFL season remains
excluded from incremental historical priors until September rollover.

Only run updates after the desired week's games and input feeds are complete.
Completeness here means the ensemble finished, not that every scheduled game or
participation record has arrived. Already saved valid weeks are immutable during
resume, so late data corrections require a full recomputation. Missing weeks still
fit season-to-date inputs, including earlier weeks' plays, not just new plays.

For box mode, completion means a finite, internally consistent scored weekly
table with `estimator='box'`, a training fingerprint, and a cumulative input
fingerprint. No iteration IDs or fabricated ensemble table are required. A new
week is committed immediately; interrupted scoring resumes from prior successful
commits. Valid missing-week resumes reuse the historical cached model. Training
changes invalidate affected target-season fingerprints; cumulative input changes
invalidate affected weeks even when those weeks were previously saved. Unchanged
saved weeks and other years are retained. Box mode never imports legacy CSV-only
results, and neither estimator resumes the other's checkpoint. Changing methods
replaces the shared position output paths; archive outputs first to retain both.
Weeks with no eligible players are still reconsidered on future runs.

## Outputs

Files are saved under `player_value/outputs/` (fitted box models in `outputs/models/`):

- `plus_minus_weekly_{position}.csv`: estimates with `season`,
    `week`, `estimator`, `interval_kind`, `prior_mean`, and `prior_season`. Bayesian
    rows also contain `prior_sd`; box rows contain `box_plus_minus`,
    `box_prediction`, `sample_weight`, `training_seasons` (a JSON list),
    `training_fingerprint`, and `week_fingerprint`.
- `plus_minus_weekly_iterations_{position}.csv`: Bayesian
    individual fits only; removed when committing box results.
- `plus_minus_checkpoint_{position}.zip`: the authoritative
        committed results, containing `manifest.json`, `splits.csv`, and Bayesian
        `iterations.csv` when available.

The `bayesian_plus_minus` column is retained for compatibility.
In box rows that column is an alias of `box_plus_minus`, not a Bayesian posterior.
Consumers must inspect `estimator` and must not interpret missing interval fields
as zero uncertainty. Box output is EPA per play, as in the original method.

Both tables are staged into one archive, flushed, and committed with atomic file
replacement. Resume always reads that pair, never mixes it with CSV exports, and
does not silently fall back to CSVs if the archive is corrupt. CSV exports are
also individually replaced atomically and repaired from the archive at the next
resume if export refresh was interrupted. Two independent CSV paths cannot be
replaced as one filesystem transaction: readers needing a coherent pair during
an active write should read the archive. Use CSVs after the update finishes.

Bayesian weekly intervals average the per-clustering interval endpoints; they
are not quantiles from a pooled ensemble posterior. Printed rankings and plots
show only the latest season/week. RAPM is optional via
`run_full_pipeline(include_rapm=True)`.

## Sampling and Tests

Bayesian sampling explicitly uses PyMC NUTS and the PyTensor Python linker, including
posterior-predictive compilation. This avoids the Windows Numba/LLVM compiler
failure under automatic backend selection, but may run more slowly.

```powershell
& '.\NFL_PM.venv\Scripts\python.exe' -m unittest discover -s player_value/tests -t . -q
& '.\NFL_PM.venv\Scripts\python.exe' -m player_value.sampling_smoke_test
```

The offline regression suite uses real sklearn box fits, mocked Bayesian fits,
blocked downloads/sampling, and temporary outputs. It checks history-only labels,
one fit per target season, future-game isolation, rookie/first-season zero
fallbacks, tiny/all-missing training profiles, monotonic sample shrinkage, model
cache reuse/invalidation, method rejection, and corrected-input/missing-week
resume. Existing Bayesian regressions include
missing-week resume, saved historical priors, deduplication, and interrupted fit,
checkpoint, and export recovery checks. The smoke check runs four real
chains on two synthetic players without downloads or production output writes.
Its short chains check execution only, not statistical convergence.
Bayesian defaults still require 100 ensemble fits per eligible season/week/position;
the executable box example does not. Production downloads and long sampling are
not part of the offline regression command.