"""Compare QB target definitions using one locally cached completed season."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from player_value.positions import eligible_rows, load_lookup


def compare_targets(pbp, lookup):
    plays = pbp[pbp.season_type.eq('REG') & pbp['pass'].eq(1) & pbp.epa.notna()].copy()
    missing = plays[plays.passer_player_id.isna()]
    print('Missing passer attribution:')
    print(missing.groupby(['play_type', 'qb_scramble'], dropna=False).agg(
        plays=('epa', 'size'), epa=('epa', 'mean'), rusher_ids=('rusher_player_id', 'count')).to_string())
    corrected = plays.copy()
    mask = corrected.passer_player_id.isna() & corrected.qb_scramble.eq(1)
    corrected.loc[mask, 'passer_player_id'] = corrected.loc[mask, 'rusher_player_id']
    corrected = eligible_rows(corrected.rename(columns={'passer_player_id': 'player_id'}),
                              lookup, 'QB').rename(columns={'player_id': 'passer_player_id'})
    summaries = []
    for label, frame, league in [('old_team', plays, False),
                                 ('attributed_team', corrected, False),
                                 ('attributed_league', corrected, True)]:
        on = frame.groupby(['passer_player_id', 'posteam', 'season']).epa.agg(['size', 'sum'])
        on = on.reset_index().rename(columns={'passer_player_id': 'player_id',
                                              'size': 'n_on', 'sum': 'on_sum'})
        on = eligible_rows(on, lookup, 'QB')
        keys = ['season'] if league else ['posteam', 'season']
        totals = frame.groupby(keys).epa.agg(['size', 'sum']).rename(
            columns={'size': 'total_n', 'sum': 'total_epa'})
        on = on.merge(totals, on=keys)
        if league:
            own = frame.groupby(['passer_player_id', 'season']).epa.agg(['size', 'sum'])
            own = own.reset_index().rename(columns={'passer_player_id': 'player_id',
                                                    'size': 'own_n', 'sum': 'own_epa'})
            on = on.merge(own, on=['player_id', 'season'])
            on['n_off'] = on.total_n - on.own_n
            on['off_sum'] = on.total_epa - on.own_epa
        else:
            on['n_off'] = on.total_n - on.n_on
            on['off_sum'] = on.total_epa - on.on_sum
        on['epa_on'] = on.on_sum / on.n_on
        on['epa_off'] = on.off_sum / on.n_off.replace(0, np.nan)
        on['raw_plus_minus'] = on.epa_on - on.epa_off
        starters = on[on.n_on.ge(300) & on.n_off.gt(0)]
        summaries.append({'reference': label, 'starters': len(starters),
                          'median_n_off': starters.n_off.median(),
                          **starters[['epa_on', 'epa_off', 'raw_plus_minus']].mean().to_dict()})
    return pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season', type=int, default=2024)
    parser.add_argument('--cache', type=Path, default=Path('artifacts/cache'))
    args = parser.parse_args()
    paths = list((args.cache / f'season={args.season}').glob('*.parquet'))
    if not paths:
        parser.error('No local cached parquet files for the requested season')
    pbp = pd.concat([pd.read_parquet(path).assign(season=args.season) for path in paths],
                    ignore_index=True)
    lookup = load_lookup(Path(__file__).resolve().parent / 'outputs' / 'player_season_positions.parquet')
    print(compare_targets(pbp, lookup).to_string(index=False))


if __name__ == '__main__':
    main()