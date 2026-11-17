"""Authoritative seasonal roster eligibility shared by fitting and display."""

import argparse
import os
from pathlib import Path
import tempfile

import pandas as pd

LOOKUP_NAME = 'player_season_positions.parquet'
NAMES_NAME = 'player_names.parquet'
POSITION_LABELS = {
    'QB': ('QB',), 'RB': ('RB', 'FB', 'HB'), 'WR': ('WR',), 'TE': ('TE',),
    'OL': ('C', 'G', 'T', 'OL', 'OT', 'OG', 'LT', 'RT', 'LG', 'RG'),
    'DL': ('DL', 'DE', 'DT', 'NT'), 'LB': ('LB', 'ILB', 'OLB', 'MLB'),
    'DB': ('DB', 'CB', 'S', 'FS', 'SS'),
}
LABEL_GROUP = {label: group for group, labels in POSITION_LABELS.items() for label in labels}


def normalize_rosters(rosters):
    identity = 'player_id' if 'player_id' in rosters else 'gsis_id'
    required = {identity, 'season', 'position'}
    missing = required.difference(rosters.columns)
    if missing:
        raise ValueError(f"Seasonal roster missing required columns: {', '.join(sorted(missing))}")
    rows = rosters[[identity, 'season', 'position']].rename(
        columns={identity: 'player_id', 'position': 'roster_position'}).copy()
    for column in ('player_id', 'roster_position'):
        rows[column] = rows[column].astype('string').str.strip().replace('', pd.NA)
    rows['roster_position'] = rows['roster_position'].str.upper()
    rows['season'] = pd.to_numeric(rows['season'], errors='coerce')
    rows = rows.dropna(subset=['player_id', 'season', 'roster_position'])
    rows = rows[rows['season'].between(1900, 2200) & rows['season'].mod(1).eq(0)].copy()
    if rows.empty:
        raise ValueError('Seasonal roster has no valid player-season positions.')
    rows['season'] = rows['season'].astype(int)
    rows['position_group'] = rows['roster_position'].map(LABEL_GROUP).astype('string')
    return rows.drop_duplicates().sort_values(
        ['season', 'player_id', 'roster_position']).reset_index(drop=True)


def require_seasons(lookup, years):
    missing = set(years).difference(lookup['season'].unique())
    if missing:
        raise ValueError(f'Seasonal roster missing seasons: {sorted(missing)}')


def eligible_rows(frame, lookup, position):
    if position not in POSITION_LABELS:
        raise ValueError(f'Unsupported position group: {position}')
    keys = ['player_id', 'season']
    eligible = lookup.loc[lookup['position_group'].eq(position), keys].drop_duplicates()
    identities = frame[keys].copy()
    identities['player_id'] = identities['player_id'].astype('string').str.strip()
    identities['season'] = pd.to_numeric(identities['season'], errors='coerce')
    mask = pd.MultiIndex.from_frame(identities).isin(pd.MultiIndex.from_frame(eligible))
    return frame.loc[mask].copy()


def save_lookup(lookup, directory):
    path = Path(directory) / LOOKUP_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = load_lookup(path)
        lookup = pd.concat([
            previous[~previous['season'].isin(lookup['season'].unique())], lookup
        ], ignore_index=True).sort_values(['season', 'player_id', 'roster_position']).reset_index(drop=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix='.parquet')
    os.close(descriptor)
    try:
        lookup.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def display_names(rosters):
    """player_id -> nickname + last name ("Pat Mahomes") from each player's latest roster record."""
    identity = 'player_id' if 'player_id' in rosters else 'gsis_id'
    rows = rosters.dropna(subset=[identity]).sort_values(
        [column for column in ('season', 'week') if column in rosters])
    rows = rows.drop_duplicates(identity, keep='last')

    def text(column):
        values = rows[column] if column in rows else pd.Series(pd.NA, index=rows.index)
        return values.astype('string').str.strip().replace('', pd.NA)

    first = text('football_name').fillna(text('first_name'))
    full = (first + ' ' + text('last_name')).fillna(text('full_name')).fillna(text('player_name'))
    return pd.DataFrame({'player_id': rows[identity].astype('string').str.strip(),
                         'player_name': full}).dropna().reset_index(drop=True)


def save_names(rosters, directory):
    path = Path(directory) / NAMES_NAME
    names = display_names(rosters)
    if path.exists():
        previous = pd.read_parquet(path)
        names = pd.concat([previous[~previous['player_id'].isin(names['player_id'])], names], ignore_index=True)
    names.to_parquet(path, index=False)
    return path


def load_lookup(path):
    raw = pd.read_parquet(path)
    return normalize_rosters(raw.rename(columns={'roster_position': 'position'}))


def refresh_lookup(years, directory):
    import nfl_data_py as nfl

    rosters = nfl.import_seasonal_rosters(list(years))
    lookup = normalize_rosters(rosters)
    require_seasons(lookup, years)
    save_lookup(lookup, directory)
    save_names(rosters, directory)
    return lookup


def main():
    parser = argparse.ArgumentParser(description='Refresh local authoritative seasonal position lookup.')
    parser.add_argument('--years', type=int, nargs='+', default=list(range(2016, 2026)))
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    args = parser.parse_args()
    lookup = refresh_lookup(args.years, args.directory)
    print(f'Saved {len(lookup):,} roster positions to {args.directory / LOOKUP_NAME}')
    print(lookup.groupby('position_group').size().to_string())


if __name__ == '__main__':
    main()