"""Out-of-sample calibration of the in-season plus-minus update.

At week w a player's estimate blends the current-season box prediction b with a prior h:

    estimate = n / (n + T) * b + T / (n + T) * h

where n is the player's cumulative on-field plays. This module scores candidate priors and
prior strengths T against what the player does next: the exposure-weighted raw on/off
plus-minus over the rest of that season. Settings that predict the rest of the season best
are written to ``config/in_season_update.json`` and used by ``PlayerPlusMinus.calibrated``.

Priors compared:
- ``zero``: every player starts at league average (0).
- ``decay``: last season's shrunk estimate times ``decay ** gap`` (the original method).
- ``regression``: a weighted ridge on prior-season estimates, age, draft slot and experience,
  refit for each target season on earlier seasons only. Rookies get a draft/age-based prior.

Only saved weekly outputs and seasonal rosters are needed; no plus-minus refit is run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'
CONFIG_PATH = Path(__file__).resolve().parent / 'config' / 'in_season_update.json'
POSITIONS = ('QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB')
DEFENSE = {'DL', 'LB', 'DB'}
CURRENT_PRIOR_SNAPS = 200.0
CURRENT_DECAY = 0.8
PRIOR_SNAPS_GRID = (25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0)
DECAY_GRID = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
UNDRAFTED_PICK = 260.0
PRIOR_FEATURES = ('lag1', 'lag2', 'has_lag1', 'has_lag2', 'lag1_share', 'age', 'age_sq',
                  'log_pick', 'undrafted', 'experience')
PRIOR_RIDGE_ALPHA = 1.0
MIN_EVALUATION_PLAYS = 20


def load_weekly(position: str, directory: Path = OUTPUT_DIR) -> pd.DataFrame:
    frame = pd.read_csv(Path(directory) / f'plus_minus_weekly_{position}.csv', low_memory=False,
                        dtype={'player_id': 'string'})
    frame = frame[frame.get('estimator', 'box').eq('box')] if 'estimator' in frame else frame
    frame = frame.rename(columns={'posteam': 'unit', 'defteam': 'unit', 'possession_team': 'unit'})
    keep = ['player_id', 'unit', 'season', 'week', 'n_on', 'n_off', 'epa_on_sum', 'epa_off_sum',
            'raw_plus_minus', 'box_prediction', 'prior_mean', 'prior_season', 'box_plus_minus']
    return frame[keep].dropna(subset=['player_id', 'season', 'week']).reset_index(drop=True)


def rest_of_season(weekly: pd.DataFrame, defense: bool) -> pd.DataFrame:
    """Attach the rest-of-season raw plus-minus and its play count to every weekly row."""
    keys = ['player_id', 'unit', 'season']
    end = weekly.sort_values('week').groupby(keys, as_index=False).last()
    end = end[keys + ['n_on', 'n_off', 'epa_on_sum', 'epa_off_sum']].add_suffix('_end')
    end = end.rename(columns={f'{key}_end': key for key in keys})
    frame = weekly.merge(end, on=keys, how='left', validate='many_to_one')
    n_on = frame['n_on_end'] - frame['n_on']
    n_off = frame['n_off_end'] - frame['n_off']
    on = (frame['epa_on_sum_end'] - frame['epa_on_sum']) / n_on.where(n_on > 0)
    off = (frame['epa_off_sum_end'] - frame['epa_off_sum']) / n_off.where(n_off > 0)
    frame['ros_plus_minus'] = (off - on) if defense else (on - off)
    frame['ros_plays'] = n_on
    return frame


def player_metadata(seasons) -> pd.DataFrame:
    """Age at September 1, draft slot and experience per player-season from nflverse rosters."""
    import nfl_data_py as nfl
    return metadata_from_rosters(nfl.import_seasonal_rosters(sorted(set(int(season) for season in seasons))))


def metadata_from_rosters(rosters: pd.DataFrame) -> pd.DataFrame:
    rosters = rosters.rename(columns={'gsis_id': 'player_id'}) if 'player_id' not in rosters else rosters
    columns = ['player_id', 'season', 'birth_date', 'draft_number', 'years_exp']
    rosters = rosters.reindex(columns=columns).copy()
    rosters['player_id'] = rosters['player_id'].astype('string')
    birth = pd.to_datetime(rosters['birth_date'], errors='coerce')
    season_start = pd.to_datetime(pd.to_numeric(rosters['season'], errors='coerce').astype('Int64').astype(str) + '-09-01',
                                  errors='coerce')
    rosters['age'] = (season_start - birth).dt.days / 365.25
    return rosters.drop(columns='birth_date').drop_duplicates(['player_id', 'season'], keep='last')


def season_summaries(weekly: pd.DataFrame, prior_snaps: float) -> pd.DataFrame:
    """One shrunk full-season estimate per player-season from saved weekly outputs."""
    end = weekly.sort_values('week').groupby(['player_id', 'unit', 'season'], as_index=False).last()
    end['estimate'] = end['box_prediction'] * end['n_on'] / (end['n_on'] + prior_snaps)
    return summarize_seasons(end)


def summarize_seasons(rows: pd.DataFrame) -> pd.DataFrame:
    """Combine player/team/season rows (``n_on``, shrunk ``estimate``, ``raw_plus_minus``) across team stints."""
    rows = rows.assign(weighted_estimate=rows['estimate'] * rows['n_on'],
                       weighted_raw=rows['raw_plus_minus'] * rows['n_on'])
    grouped = rows.groupby(['player_id', 'season'], as_index=False).agg(
        n_on=('n_on', 'sum'), weighted_estimate=('weighted_estimate', 'sum'), weighted_raw=('weighted_raw', 'sum'))
    grouped['estimate'] = grouped['weighted_estimate'] / grouped['n_on'].where(grouped['n_on'] > 0)
    grouped['raw_plus_minus'] = grouped['weighted_raw'] / grouped['n_on'].where(grouped['n_on'] > 0)
    grouped['share'] = grouped['n_on'] / grouped.groupby('season')['n_on'].transform('max')
    return grouped[['player_id', 'season', 'n_on', 'share', 'estimate', 'raw_plus_minus']]


def prior_features(players: pd.DataFrame, summaries: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Features for each (player_id, season) using only seasons before it."""
    frame = players[['player_id', 'season']].drop_duplicates().astype({'player_id': 'string'})
    summaries = summaries.astype({'player_id': 'string'})
    for lag in (1, 2):
        lagged = summaries[['player_id', 'season', 'estimate', 'share']].copy()
        lagged['season'] = lagged['season'] + lag
        lagged = lagged.rename(columns={'estimate': f'lag{lag}', 'share': f'lag{lag}_share'})
        frame = frame.merge(lagged, on=['player_id', 'season'], how='left')
    frame = frame.drop(columns=['lag2_share'])
    frame['has_lag1'] = frame['lag1'].notna().astype(float)
    frame['has_lag2'] = frame['lag2'].notna().astype(float)
    frame[['lag1', 'lag2', 'lag1_share']] = frame[['lag1', 'lag2', 'lag1_share']].fillna(0.0)
    frame = frame.merge(metadata, on=['player_id', 'season'], how='left')
    frame['age'] = frame['age'].fillna(frame['age'].median() if frame['age'].notna().any() else 26.0)
    frame['age'] = frame['age'] - 26.0
    frame['age_sq'] = frame['age'] ** 2 / 10.0
    pick = pd.to_numeric(frame['draft_number'], errors='coerce')
    frame['undrafted'] = pick.isna().astype(float)
    frame['log_pick'] = np.log(pick.fillna(UNDRAFTED_PICK)) - np.log(UNDRAFTED_PICK)
    frame['experience'] = pd.to_numeric(frame['years_exp'], errors='coerce').fillna(0.0).clip(0, 15) / 5.0
    return frame[['player_id', 'season', *PRIOR_FEATURES]]


def fit_prior_regression(features: pd.DataFrame, targets: pd.DataFrame, alpha: float = PRIOR_RIDGE_ALPHA):
    """Weighted ridge of full-season raw plus-minus on prior features; returns (intercept, coefficients)."""
    data = features.merge(targets[['player_id', 'season', 'raw_plus_minus', 'n_on']], on=['player_id', 'season'])
    data = data[np.isfinite(data['raw_plus_minus']) & (data['n_on'] > 0)]
    if len(data) < len(PRIOR_FEATURES) + 2:
        return 0.0, np.zeros(len(PRIOR_FEATURES))
    x = data[list(PRIOR_FEATURES)].to_numpy(dtype=float)
    y = data['raw_plus_minus'].to_numpy(dtype=float)
    w = data['n_on'].to_numpy(dtype=float)
    w = w / w.mean()
    x_mean = np.average(x, axis=0, weights=w)
    y_mean = np.average(y, weights=w)
    xc, yc = x - x_mean, y - y_mean
    gram = (xc * w[:, None]).T @ xc + alpha * len(x) * np.eye(x.shape[1]) / 100.0
    coefficients = np.linalg.solve(gram, (xc * w[:, None]).T @ yc)
    return float(y_mean - x_mean @ coefficients), coefficients


def regression_priors(weekly: pd.DataFrame, metadata: pd.DataFrame, prior_snaps: float) -> pd.DataFrame:
    """Prior mean per (player_id, season) from a regression fit on strictly earlier seasons."""
    summaries = season_summaries(weekly, prior_snaps)
    players = weekly[['player_id', 'season']].drop_duplicates()
    features = prior_features(pd.concat([players, summaries[['player_id', 'season']]]), summaries, metadata)
    priors = []
    for season in sorted(players['season'].unique()):
        train = features[features['season'] < season]
        intercept, coefficients = fit_prior_regression(train, summaries[summaries['season'] < season])
        current = features[features['season'] == season]
        priors.append(current[['player_id', 'season']].assign(
            regression_prior=intercept + current[list(PRIOR_FEATURES)].to_numpy(dtype=float) @ coefficients))
    return pd.concat(priors, ignore_index=True).drop_duplicates(['player_id', 'season'])


def decayed_prior(frame: pd.DataFrame, decay: float) -> pd.Series:
    """Rescale the saved 0.8-decayed prior to another decay."""
    gap = (frame['season'] - frame['prior_season']).where(frame['prior_season'].notna(), 0)
    base = frame['prior_mean'] / CURRENT_DECAY ** gap
    return (base * decay ** gap).where(frame['prior_season'].notna(), 0.0)


def blended(frame: pd.DataFrame, prior: pd.Series, prior_snaps: float) -> pd.Series:
    weight = frame['n_on'] / (frame['n_on'] + prior_snaps)
    return weight * frame['box_prediction'] + (1 - weight) * prior


def score(frame: pd.DataFrame, estimate: pd.Series) -> float:
    """Exposure-weighted mean squared error against the rest-of-season plus-minus."""
    weights = frame['ros_plays']
    return float(np.average((estimate - frame['ros_plus_minus']) ** 2, weights=weights))


def evaluation_rows(frame: pd.DataFrame, first_season: int) -> pd.DataFrame:
    usable = (frame['season'] >= first_season) & (frame['ros_plays'] >= MIN_EVALUATION_PLAYS) \
        & np.isfinite(frame['ros_plus_minus']) & np.isfinite(frame['box_prediction'])
    return frame[usable]


def calibrate_position(position: str, metadata: pd.DataFrame, directory: Path = OUTPUT_DIR,
                       first_season: int = 2019) -> dict:
    weekly = load_weekly(position, directory)
    frame = evaluation_rows(rest_of_season(weekly, position in DEFENSE), first_season)
    results = []
    baseline_variance = float(np.average((frame['ros_plus_minus'] - np.average(
        frame['ros_plus_minus'], weights=frame['ros_plays'])) ** 2, weights=frame['ros_plays']))
    for prior_snaps in PRIOR_SNAPS_GRID:
        results.append({'prior': 'zero', 'prior_snaps': prior_snaps, 'decay': None,
                        'mse': score(frame, blended(frame, pd.Series(0.0, index=frame.index), prior_snaps))})
        for decay in DECAY_GRID:
            results.append({'prior': 'decay', 'prior_snaps': prior_snaps, 'decay': decay,
                            'mse': score(frame, blended(frame, decayed_prior(frame, decay), prior_snaps))})
        priors = regression_priors(weekly, metadata, prior_snaps)
        merged = frame[['player_id', 'season']].merge(priors, on=['player_id', 'season'], how='left')
        prior = pd.Series(merged['regression_prior'].fillna(0.0).to_numpy(), index=frame.index)
        results.append({'prior': 'regression', 'prior_snaps': prior_snaps, 'decay': None,
                        'mse': score(frame, blended(frame, prior, prior_snaps))})
    table = pd.DataFrame(results)
    current = table[(table['prior'] == 'decay') & (table['prior_snaps'] == CURRENT_PRIOR_SNAPS)
                    & (table['decay'] == CURRENT_DECAY)]['mse'].iloc[0]
    table['skill_vs_current'] = 1 - table['mse'] / current
    table['skill_vs_mean'] = 1 - table['mse'] / baseline_variance
    best = table.sort_values('mse').iloc[0]
    return {
        'position': position, 'rows': int(len(frame)), 'seasons': sorted(int(s) for s in frame['season'].unique()),
        'current_mse': float(current), 'best': {key: (None if pd.isna(value) else value)
                                                for key, value in best.to_dict().items()},
        'table': table,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--positions', nargs='+', default=list(POSITIONS))
    parser.add_argument('--first-season', type=int, default=2019, help='First evaluation season')
    parser.add_argument('--write', action='store_true', help=f'Write chosen settings to {CONFIG_PATH}')
    parser.add_argument('--report', type=Path, default=OUTPUT_DIR / 'in_season_calibration.csv')
    args = parser.parse_args()
    seasons = set()
    for position in args.positions:
        seasons |= set(load_weekly(position)['season'].astype(int))
    metadata = player_metadata(seasons)
    chosen, tables = {}, []
    for position in args.positions:
        result = calibrate_position(position, metadata, first_season=args.first_season)
        best = result['best']
        print(f"{position}: rows={result['rows']} best={best['prior']} T={best['prior_snaps']:g} "
              f"decay={best['decay']} skill_vs_current={best['skill_vs_current']:+.4f} "
              f"skill_vs_mean={best['skill_vs_mean']:+.4f}")
        tables.append(result['table'].assign(position=position))
        chosen[position] = {'prior_model': best['prior'], 'prior_snaps': best['prior_snaps'],
                            'history_decay': best['decay'] if best['decay'] is not None else CURRENT_DECAY,
                            'evaluation_seasons': result['seasons'], 'evaluation_rows': result['rows'],
                            'mse': best['mse'], 'mse_original_method': result['current_mse']}
    pd.concat(tables, ignore_index=True).to_csv(args.report, index=False)
    print(f'Full grid written to {args.report}')
    if args.write:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
        existing.update(chosen)
        CONFIG_PATH.write_text(json.dumps(existing, indent=2, sort_keys=True) + '\n')
        print(f'Wrote {CONFIG_PATH}')


if __name__ == '__main__':
    main()
