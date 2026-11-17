"""Madden preseason ratings as box plus-minus inputs for positions with few box-score events.

Offensive linemen have almost no individually attributed statistics (the box profile was
penalties only), and defensive box events are sparse. Madden ratings are published before
each season, so they are pregame information. They enter the same clustered ridge as other
box features, which maps ratings to on/off EPA.

Source: per-season files from github.com/theedgepredictor/nfl-madden-data (scraped from
maddenratings.weebly.com, joined to nflverse GSIS ids), stored locally in ``data/madden/``.
The repository has no license, so the files are not committed. A season without a file
uses each player's most recent earlier rating.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MADDEN_DIR = Path(__file__).resolve().parents[1] / 'data' / 'madden'
SOURCE_URL = 'https://raw.githubusercontent.com/theedgepredictor/nfl-madden-data/main/data/madden/dataset/{season}.parquet'

MADDEN_ATTRIBUTES = {
    'OL': ('overallrating', 'passblocking', 'runblocking', 'impactblocking', 'strength', 'awareness'),
    'DL': ('overallrating', 'strength', 'pursuit', 'playrecognition', 'tackle', 'hitpower', 'acceleration'),
    'LB': ('overallrating', 'tackle', 'pursuit', 'playrecognition', 'zonecoverage', 'mancoverage', 'speed'),
    'DB': ('overallrating', 'mancoverage', 'zonecoverage', 'press', 'speed', 'playrecognition', 'tackle'),
}


def feature_names(position: str) -> list[str]:
    return [f'madden_{attribute}' for attribute in MADDEN_ATTRIBUTES.get(position, ())]


def load_ratings(seasons, position: str, directory: Path = MADDEN_DIR) -> pd.DataFrame:
    """Ratings per (player_id, season), carrying a player's latest earlier rating into seasons without a file."""
    attributes = list(MADDEN_ATTRIBUTES[position])
    frames = []
    for path in sorted(Path(directory).glob('*.parquet')):
        frame = pd.read_parquet(path, columns=['player_id', 'season', *attributes])
        frames.append(frame.dropna(subset=['player_id']))
    if not frames:
        raise FileNotFoundError(f'No Madden rating files in {directory}; download them from {SOURCE_URL}')
    ratings = pd.concat(frames, ignore_index=True)
    ratings['player_id'] = ratings['player_id'].astype('string')
    ratings['season'] = pd.to_numeric(ratings['season'], errors='coerce').astype(int)
    ratings = ratings.drop_duplicates(['player_id', 'season'], keep='last').sort_values('season')
    requests = pd.DataFrame({'season': sorted({int(season) for season in seasons})})
    result = []
    for season in requests['season']:
        latest = ratings[ratings['season'] <= season].drop_duplicates('player_id', keep='last')
        result.append(latest.assign(season=season))
    combined = pd.concat(result, ignore_index=True) if result else ratings.iloc[0:0]
    return combined.rename(columns={attribute: f'madden_{attribute}' for attribute in attributes})


def attach_ratings(features: pd.DataFrame, position: str, directory: Path = MADDEN_DIR) -> pd.DataFrame:
    if features.empty:
        return features.assign(**{name: pd.Series(dtype=float) for name in feature_names(position)})
    ratings = load_ratings(features['season'].unique(), position, directory)
    features = features.drop(columns=[name for name in feature_names(position) if name in features])
    keyed = features.assign(player_id=features['player_id'].astype('string'))
    merged = keyed.merge(ratings, on=['player_id', 'season'], how='left', validate='many_to_one')
    merged['player_id'] = features['player_id'].to_numpy()
    return merged
