"""
=============================================================================
NFL Bayesian Plus-Minus with Player Clustering — Full Positional Support
=============================================================================

A Python implementation combining:
  1. Kevin Cole's player clustering methodology for NFL Plus/Minus
  2. Bayesian hierarchical shrinkage toward cluster means  
  3. Snap-level participation data (2016+) for true on-field plus-minus
  4. Free/open-source data via nfl-data-py and nflverse releases

Dependencies:
    pip install nfl-data-py pandas numpy scikit-learn pymc arviz matplotlib pyarrow

Participation data sources:
  - 2016-2022: NGS via nflverse (CC-BY 4.0)
  - 2023+: FTN Data via nflverse (CC-BY-SA 4.0)

Usage:
    model = NFLBayesianPlusMinus(position='DB', years=[2021, 2022, 2023])
    model.run_full_pipeline()
    results = model.get_results()
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import logging
from pathlib import Path
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# The clustering ensemble re-fits the model ~100x; keep pymc's per-fit logging quiet.
logging.getLogger('pymc').setLevel(logging.WARNING)


# =============================================================================
# POSITION CONFIGURATION DICTIONARY
# =============================================================================
# Defines clustering features, play filters, and data requirements per position.
# Features are computed from play-by-play + participation data.
# =============================================================================

POSITION_CONFIG: Dict[str, Dict] = {
    'QB': {
        'side': 'offense',
        'pbp_id_col': 'passer_player_id',
        'play_filter': lambda pbp: pbp['pass'] == 1,
        'min_snaps': 100,
        'n_clusters': 5,
        'use_participation': False,
        'clustering_features': [
            'pass_attempts_per_game', 'completion_pct', 'yards_per_attempt',
            'td_pct', 'int_pct', 'sack_pct', 'epa_per_play', 'cpoe',
            'avg_air_yards', 'scramble_rate'
        ],
        'feature_desc': 'Passing efficiency and volume profile',
    },
    'RB': {
        'side': 'offense',
        'pbp_id_col': 'rusher_player_id',
        'play_filter': lambda pbp: (pbp['rush'] == 1) | 
                                   ((pbp['pass'] == 1) & (pbp['receiver_position'] == 'RB')),
        'min_snaps': 80,
        'n_clusters': 6,
        'use_participation': False,
        'clustering_features': [
            'rush_attempts_per_game', 'yards_per_carry', 'td_rate',
            'target_share', 'catch_rate', 'yards_per_target',
            'epa_per_rush', 'stuff_rate', 'breakaway_rate'
        ],
        'feature_desc': 'Rushing and receiving dual-threat profile',
    },
    'WR': {
        'side': 'offense',
        'pbp_id_col': 'receiver_player_id',
        'play_filter': lambda pbp: (pbp['pass'] == 1) & (pbp['receiver_position'] == 'WR'),
        'min_snaps': 50,
        'n_clusters': 6,
        'use_participation': False,
        'clustering_features': [
            'targets_per_game', 'catch_rate', 'yards_per_target',
            'air_yards_per_target', 'yac_per_reception', 'td_rate',
            'first_down_rate', 'deep_target_rate', 'target_share',
            'epa_per_target'
        ],
        'feature_desc': 'Receiving volume and efficiency profile',
    },
    'TE': {
        'side': 'offense',
        'pbp_id_col': 'receiver_player_id',
        'play_filter': lambda pbp: (pbp['pass'] == 1) & (pbp['receiver_position'] == 'TE'),
        'min_snaps': 40,
        'n_clusters': 5,
        'use_participation': False,
        'clustering_features': [
            'targets_per_game', 'catch_rate', 'yards_per_target',
            'air_yards_per_target', 'yac_per_reception', 'td_rate',
            'first_down_rate', 'target_share', 'epa_per_target',
            'inline_rate'
        ],
        'feature_desc': 'Receiving efficiency with blocking context',
    },
    'OL': {
        'side': 'offense',
        'pbp_id_col': None,
        'play_filter': lambda pbp: (pbp['pass'] == 1) | (pbp['rush'] == 1),
        'min_snaps': 200,
        'n_clusters': 4,
        'use_participation': True,
        'clustering_features': [
            'team_sack_rate_when_on', 'team_epa_per_play_when_on',
            'penalties_per_1000_snaps', 'pass_block_snap_pct'
        ],
        'feature_desc': 'Team offensive line performance profile',
    },
    'DL': {
        'side': 'defense',
        'pbp_id_col': None,
        'play_filter': lambda pbp: (pbp['pass'] == 1) | (pbp['rush'] == 1),
        'min_snaps': 150,
        'n_clusters': 5,
        'use_participation': True,
        'clustering_features': [
            'sacks_per_100_snaps', 'qb_hits_per_100_snaps',
            'tfls_per_100_snaps', 'run_stops_per_100_snaps',
            'pass_rush_snap_pct', 'snaps_per_game'
        ],
        'feature_desc': 'Pass rush and run defense profile',
    },
    'LB': {
        'side': 'defense',
        'pbp_id_col': None,
        'play_filter': lambda pbp: (pbp['pass'] == 1) | (pbp['rush'] == 1),
        'min_snaps': 150,
        'n_clusters': 5,
        'use_participation': True,
        'clustering_features': [
            'tackles_per_100_snaps', 'tfls_per_100_snaps',
            'sacks_per_100_snaps', 'ints_per_100_snaps',
            'pass_deflections_per_100_snaps', 'run_stops_per_100_snaps',
            'pass_rush_snap_pct'
        ],
        'feature_desc': 'Run defense, pass rush, and coverage profile',
    },
    'DB': {
        'side': 'defense',
        'pbp_id_col': None,
        'play_filter': lambda pbp: (pbp['pass'] == 1) | (pbp['rush'] == 1),
        'min_snaps': 150,
        'n_clusters': 5,
        'use_participation': True,
        'clustering_features': [
            'ints_per_100_snaps', 'pass_deflections_per_100_snaps',
            'tackles_per_100_snaps', 'run_stops_per_100_snaps',
            'coverage_snap_pct', 'slot_snap_pct'
        ],
        'feature_desc': 'Coverage and ball skills profile',
    },
}


def load_participation_data(years: List[int]) -> pd.DataFrame:
    """
    Load play-level participation data from nflverse releases.

    Data sources:
      - 2016-2022: NGS participation via nflverse
      - 2023+: FTN Data participation via nflverse (CC-BY-SA 4.0)

    Returns DataFrame with columns including:
      game_id, play_id, possession_team, offense_players, defense_players,
      offense_positions, defense_positions, offense_formation, etc.
    """
    dfs = []
    for year in years:
        if year < 2016:
            print(f"  Skipping {year}: participation data only available from 2016+")
            continue

        urls = [
            f"https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{year}.parquet",
        ]

        loaded = False
        for url in urls:
            try:
                df = pd.read_parquet(url)
                df['season'] = year
                dfs.append(df)
                print(f"  Loaded participation for {year}: {len(df):,} plays")
                loaded = True
                break
            except Exception:
                continue

        if not loaded:
            print(f"  WARNING: Could not load participation for {year}")

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    # Standardize column names
    if 'nflverse_game_id' in result.columns:
        result = result.rename(columns={'nflverse_game_id': 'game_id'})
    return result


def add_receiver_position(pbp: pd.DataFrame) -> pd.DataFrame:
    """Attach receiver_position by merging roster position onto receiver_player_id.

    nflfastR pbp no longer ships a receiver_position column directly, so it must
    be derived from player position data (see compare_yac_ordinal_crps.py pattern).
    """
    players = nfl.import_players()[['gsis_id', 'position']].rename(
        columns={'gsis_id': 'receiver_player_id', 'position': 'receiver_position'}
    )
    players = players.dropna(subset=['receiver_player_id']).drop_duplicates('receiver_player_id')
    pbp = pbp.merge(players, on='receiver_player_id', how='left')
    return pbp


class NFLBayesianPlusMinus:
    """
    NFL Bayesian Plus-Minus with Player Clustering.

    Supports all position groups using snap-level participation data (2016+)
    for true on-field plus-minus, plus event-based on-off for skill positions.
    """

    def __init__(
        self,
        position: str,
        years: List[int],
        config: Optional[Dict] = None,
        random_state: int = 42,
        bayesian_draws: int = 2000,
        bayesian_tune: int = 1000,
        n_cluster_iterations: int = 100,
        cluster_k_min: Optional[int] = None,
        cluster_k_max: Optional[int] = None,
        ensemble_draws: int = 300,
        ensemble_tune: int = 300,
    ):
        position = position.upper()
        if position not in POSITION_CONFIG:
            raise ValueError(f"Position must be one of {list(POSITION_CONFIG.keys())}")

        self.position = position
        self.years = years
        self.config = config or POSITION_CONFIG[position].copy()
        self.random_state = random_state
        self.bayesian_draws = bayesian_draws
        self.bayesian_tune = bayesian_tune

        # Kevin Cole-style clustering ensemble: re-cluster with a random k each iteration,
        # re-fit the hierarchical model each time, and average the estimates at the end.
        self.n_cluster_iterations = n_cluster_iterations
        self.cluster_k_min = cluster_k_min
        self.cluster_k_max = cluster_k_max
        self.ensemble_draws = ensemble_draws
        self.ensemble_tune = ensemble_tune

        self.pbp = None
        self.participation = None
        self.player_features = None
        self.splits_df = None
        self.trace = None
        self.rapm_model = None
        self.kmeans = None
        self.pca = None
        self.scaler = None
        self.cluster_ensemble_results = None
        self._available_features = None
        self._valid_idx = None
        self._X_pca = None

        print(f"Initialized {self.position} Plus-Minus for seasons {self.years}")
        print(f"  Config: {self.config['feature_desc']}")
        print(f"  Requires participation: {self.config['use_participation']}")

    # =====================================================================
    # 1. DATA LOADING
    # =====================================================================

    def load_data(self):
        """Load play-by-play and participation data."""
        print(f"\n{'='*60}")
        print("STEP 1: Loading Data")
        print(f"{'='*60}")

        self.pbp = nfl.import_pbp_data(self.years, downcast=True)
        self.pbp = add_receiver_position(self.pbp)
        print(f"Loaded {len(self.pbp):,} total plays from nfl-data-py")

        needs_participation = self.config['use_participation'] or min(self.years) >= 2016
        if needs_participation:
            print("\nLoading participation data from nflverse releases...")
            self.participation = load_participation_data(self.years)
            if len(self.participation) > 0:
                print(f"Total participation plays: {len(self.participation):,}")
            else:
                print("WARNING: No participation data loaded.")
                if self.config['use_participation']:
                    raise RuntimeError("Participation data required but not loaded.")

        return self

    # =====================================================================
    # 2. PLAYER FEATURE ENGINEERING
    # =====================================================================

    def compute_player_features(self):
        """Compute seasonal stats for clustering."""
        print(f"\n{'='*60}")
        print("STEP 2: Computing Player Features for Clustering")
        print(f"{'='*60}")

        if self.position == 'QB':
            self._compute_qb_features()
        elif self.position == 'RB':
            self._compute_rb_features()
        elif self.position in ('WR', 'TE'):
            self._compute_receiver_features()
        elif self.position == 'OL':
            self._compute_ol_features()
        elif self.position in ('DL', 'LB', 'DB'):
            self._compute_defensive_features()
        else:
            raise ValueError(f"Feature computation not implemented for {self.position}")

        print(f"Computed features for {len(self.player_features)} player-seasons")
        return self

    def _compute_qb_features(self):
        pbp = self.pbp[
            (self.pbp['season_type'] == 'REG') &
            (self.pbp['pass'] == 1) &
            (self.pbp['passer_player_id'].notna()) &
            (self.pbp['epa'].notna())
        ].copy()

        stats = pbp.groupby(['passer_player_id', 'passer_player_name', 'posteam', 'season']).agg(
            games=('game_id', 'nunique'),
            attempts=('play_id', 'count'),
            completions=('complete_pass', 'sum'),
            yards=('passing_yards', lambda x: x.fillna(0).sum()),
            tds=('touchdown', 'sum'),
            ints=('interception', 'sum'),
            sacks=('sack', 'sum'),
            epa_sum=('epa', 'sum'),
            air_yards=('air_yards', lambda x: x.fillna(0).sum()),
            scrambles=('qb_scramble', lambda x: x.fillna(0).sum()),
        ).reset_index()

        stats['pass_attempts_per_game'] = stats['attempts'] / stats['games']
        stats['completion_pct'] = stats['completions'] / stats['attempts']
        stats['yards_per_attempt'] = stats['yards'] / stats['attempts']
        stats['td_pct'] = stats['tds'] / stats['attempts']
        stats['int_pct'] = stats['ints'] / stats['attempts']
        stats['sack_pct'] = stats['sacks'] / stats['attempts']
        stats['epa_per_play'] = stats['epa_sum'] / stats['attempts']
        stats['avg_air_yards'] = stats['air_yards'] / stats['attempts']
        stats['scramble_rate'] = stats['scrambles'] / stats['attempts']

        if 'cpoe' in pbp.columns:
            cpoe_stats = pbp.dropna(subset=['cpoe']).groupby(
                ['passer_player_id', 'posteam', 'season']
            )['cpoe'].mean().reset_index()
            stats = stats.merge(cpoe_stats, on=['passer_player_id', 'posteam', 'season'], how='left')
        else:
            stats['cpoe'] = np.nan

        stats = stats.rename(columns={'passer_player_id': 'player_id', 'passer_player_name': 'player_name'})
        stats = stats[stats['attempts'] >= self.config['min_snaps']]
        self.player_features = stats

    def _compute_rb_features(self):
        pbp = self.pbp[(self.pbp['season_type'] == 'REG') & (self.pbp['epa'].notna())].copy()

        rush = pbp[(pbp['rush'] == 1) & (pbp['rusher_player_id'].notna())].groupby(
            ['rusher_player_id', 'rusher_player_name', 'posteam', 'season']
        ).agg(
            games=('game_id', 'nunique'),
            carries=('play_id', 'count'),
            rush_yards=('rushing_yards', lambda x: x.fillna(0).sum()),
            rush_tds=('touchdown', 'sum'),
            rush_epa=('epa', 'sum'),
            stuffs=('yards_gained', lambda x: (x.fillna(0) <= 0).sum()),
            breakaways=('yards_gained', lambda x: (x.fillna(0) >= 15).sum()),
        ).reset_index()

        rec = pbp[
            (pbp['pass'] == 1) & 
            (pbp['receiver_player_id'].notna()) &
            (pbp['receiver_position'] == 'RB')
        ].groupby(['receiver_player_id', 'posteam', 'season']).agg(
            targets=('play_id', 'count'),
            receptions=('complete_pass', 'sum'),
            rec_yards=('receiving_yards', lambda x: x.fillna(0).sum()),
            rec_epa=('epa', 'sum'),
        ).reset_index()

        stats = rush.rename(columns={'rusher_player_id': 'player_id', 'rusher_player_name': 'player_name'})
        rec = rec.rename(columns={'receiver_player_id': 'player_id'})
        stats = stats.merge(rec, on=['player_id', 'posteam', 'season'], how='left')
        stats = stats.fillna({'targets': 0, 'receptions': 0, 'rec_yards': 0, 'rec_epa': 0})

        team_pass = pbp[pbp['pass'] == 1].groupby(['posteam', 'season']).size().reset_index(name='team_pass_attempts')
        stats = stats.merge(team_pass, on=['posteam', 'season'])

        stats['rush_attempts_per_game'] = stats['carries'] / stats['games']
        stats['yards_per_carry'] = stats['rush_yards'] / stats['carries']
        stats['td_rate'] = stats['rush_tds'] / stats['carries']
        stats['epa_per_rush'] = stats['rush_epa'] / stats['carries']
        stats['stuff_rate'] = stats['stuffs'] / stats['carries']
        stats['breakaway_rate'] = stats['breakaways'] / stats['carries']
        stats['target_share'] = stats['targets'] / stats['team_pass_attempts']
        stats['catch_rate'] = stats['receptions'] / stats['targets'].clip(lower=1)
        stats['yards_per_target'] = stats['rec_yards'] / stats['targets'].clip(lower=1)

        stats = stats[stats['carries'] + stats['targets'] >= self.config['min_snaps']]
        self.player_features = stats

    def _compute_receiver_features(self):
        pbp = self.pbp[
            (self.pbp['season_type'] == 'REG') &
            (self.pbp['pass'] == 1) &
            (self.pbp['receiver_player_id'].notna()) &
            (self.pbp['epa'].notna())
        ].copy()

        if self.position == 'TE':
            pbp = pbp[pbp['receiver_position'] == 'TE']
        else:
            pbp = pbp[pbp['receiver_position'] == 'WR']

        stats = pbp.groupby(['receiver_player_id', 'receiver_player_name', 'posteam', 'season']).agg(
            games=('game_id', 'nunique'),
            targets=('play_id', 'count'),
            receptions=('complete_pass', 'sum'),
            yards=('receiving_yards', lambda x: x.fillna(0).sum()),
            air_yards=('air_yards', lambda x: x.fillna(0).sum()),
            yac=('yards_after_catch', lambda x: x.fillna(0).sum()),
            tds=('touchdown', 'sum'),
            first_downs=('first_down', 'sum'),
            deep_targets=('air_yards', lambda x: (x.fillna(0) >= 20).sum()),
            epa_sum=('epa', 'sum'),
        ).reset_index()

        team_pass = self.pbp[
            (self.pbp['season_type'] == 'REG') & (self.pbp['pass'] == 1)
        ].groupby(['posteam', 'season']).size().reset_index(name='team_pass_attempts')
        stats = stats.merge(team_pass, on=['posteam', 'season'])

        stats['targets_per_game'] = stats['targets'] / stats['games']
        stats['catch_rate'] = stats['receptions'] / stats['targets']
        stats['yards_per_target'] = stats['yards'] / stats['targets']
        stats['air_yards_per_target'] = stats['air_yards'] / stats['targets']
        stats['yac_per_reception'] = stats['yac'] / stats['receptions'].clip(lower=1)
        stats['td_rate'] = stats['tds'] / stats['targets']
        stats['first_down_rate'] = stats['first_downs'] / stats['targets']
        stats['deep_target_rate'] = stats['deep_targets'] / stats['targets']
        stats['target_share'] = stats['targets'] / stats['team_pass_attempts']
        stats['epa_per_target'] = stats['epa_sum'] / stats['targets']

        if self.position == 'TE':
            stats['inline_rate'] = stats['air_yards_per_target'].apply(
                lambda x: max(0, 1 - x/10) if pd.notna(x) else 0.5
            )

        stats = stats.rename(columns={'receiver_player_id': 'player_id', 'receiver_player_name': 'player_name'})
        stats = stats[stats['targets'] >= self.config['min_snaps']]
        self.player_features = stats

    def _compute_ol_features(self):
        """OL features from participation data + team offensive performance."""
        if self.participation is None or len(self.participation) == 0:
            raise RuntimeError("Participation data required for OL features")

        part = self.participation[
            self.participation['offense_players'].notna()
        ][['game_id', 'play_id', 'season', 'possession_team', 'offense_players']].copy()
        part['player_id'] = part['offense_players'].str.split(';')
        part = part.explode('player_id')
        part = part[part['player_id'] != ''].drop_duplicates()

        pbp = self.pbp[
            (self.pbp['season_type'] == 'REG') &
            ((self.pbp['pass'] == 1) | (self.pbp['rush'] == 1)) &
            (self.pbp['epa'].notna())
        ][['game_id', 'play_id', 'season', 'posteam', 'epa', 'pass', 'rush', 'sack', 'penalty']].copy()

        merged = part.merge(pbp, on=['game_id', 'play_id', 'season'])

        rosters = nfl.import_seasonal_rosters(self.years)
        if 'gsis_id' in rosters.columns and 'position' in rosters.columns:
            ol_ids = rosters[rosters['position'].isin(['C', 'G', 'T'])]['gsis_id'].unique()
            merged = merged[merged['player_id'].isin(ol_ids)]

        stats = merged.groupby(['player_id', 'possession_team', 'season']).agg(
            games=('game_id', 'nunique'),
            snaps=('play_id', 'count'),
            team_epa_sum=('epa', 'sum'),
            pass_snaps=('pass', 'sum'),
            rush_snaps=('rush', 'sum'),
            sacks_allowed=('sack', 'sum'),
            penalties=('penalty', lambda x: x.notna().sum()),
        ).reset_index()

        stats['snaps_per_game'] = stats['snaps'] / stats['games']
        stats['team_epa_per_play_when_on'] = stats['team_epa_sum'] / stats['snaps']
        stats['team_sack_rate_when_on'] = stats['sacks_allowed'] / stats['pass_snaps'].clip(lower=1)
        stats['penalties_per_1000_snaps'] = stats['penalties'] / stats['snaps'] * 1000
        stats['pass_block_snap_pct'] = stats['pass_snaps'] / stats['snaps']
        stats['team_pressure_rate_when_on'] = np.nan
        stats['team_rush_success_rate_when_on'] = np.nan

        stats = stats.rename(columns={'possession_team': 'team'})
        stats = stats[stats['snaps'] >= self.config['min_snaps']]
        self.player_features = stats

    def _compute_defensive_features(self):
        """DL/LB/DB features from participation + pbp event data."""
        if self.participation is None or len(self.participation) == 0:
            raise RuntimeError("Participation data required for defensive features")

        part = self.participation[
            self.participation['defense_players'].notna()
        ][['game_id', 'play_id', 'season', 'defense_players']].copy()
        part['player_id'] = part['defense_players'].str.split(';')
        part = part.explode('player_id')
        part = part[part['player_id'] != ''].drop_duplicates()

        pbp = self.pbp[
            (self.pbp['season_type'] == 'REG') &
            ((self.pbp['pass'] == 1) | (self.pbp['rush'] == 1)) &
            (self.pbp['epa'].notna())
        ][['game_id', 'play_id', 'season', 'posteam', 'defteam', 'epa', 'pass', 'rush']].copy()

        merged = part.merge(pbp, on=['game_id', 'play_id', 'season'])

        rosters = nfl.import_seasonal_rosters(self.years)
        if 'gsis_id' in rosters.columns and 'position' in rosters.columns:
            pos_map = {
                'DL': ['DE', 'DT', 'NT', 'DL'],
                'LB': ['LB', 'ILB', 'OLB'],
                'DB': ['CB', 'DB', 'FS', 'SS', 'S']
            }
            pos_ids = rosters[rosters['position'].isin(pos_map.get(self.position, []))]['gsis_id'].unique()
            merged = merged[merged['player_id'].isin(pos_ids)]

        # Snap counts from participation
        snap_stats = merged.groupby(['player_id', 'defteam', 'season']).agg(
            games=('game_id', 'nunique'),
            snaps=('play_id', 'count'),
            pass_snaps=('pass', 'sum'),
            rush_snaps=('rush', 'sum'),
            team_epa_allowed=('epa', 'sum'),
        ).reset_index()

        # Individual box score stats from pbp
        # Sacks
        sack_cols = ['sack_player_id', 'half_sack_1_player_id', 'half_sack_2_player_id']
        sacks = []
        for col in sack_cols:
            if col in self.pbp.columns:
                s = self.pbp[self.pbp[col].notna()].groupby([col, 'defteam', 'season']).size().reset_index(name='sacks')
                s = s.rename(columns={col: 'player_id'})
                sacks.append(s)
        sacks_df = pd.concat(sacks).groupby(['player_id', 'defteam', 'season'])['sacks'].sum().reset_index() if sacks else pd.DataFrame(columns=['player_id', 'defteam', 'season', 'sacks'])

        # QB hits
        hit_cols = ['qb_hit_1_player_id', 'qb_hit_2_player_id']
        hits = []
        for col in hit_cols:
            if col in self.pbp.columns:
                h = self.pbp[self.pbp[col].notna()].groupby([col, 'defteam', 'season']).size().reset_index(name='qb_hits')
                h = h.rename(columns={col: 'player_id'})
                hits.append(h)
        hits_df = pd.concat(hits).groupby(['player_id', 'defteam', 'season'])['qb_hits'].sum().reset_index() if hits else pd.DataFrame(columns=['player_id', 'defteam', 'season', 'qb_hits'])

        # Tackles for loss
        tfl_cols = ['tackle_for_loss_1_player_id', 'tackle_for_loss_2_player_id']
        tfls = []
        for col in tfl_cols:
            if col in self.pbp.columns:
                t = self.pbp[self.pbp[col].notna()].groupby([col, 'defteam', 'season']).size().reset_index(name='tfls')
                t = t.rename(columns={col: 'player_id'})
                tfls.append(t)
        tfls_df = pd.concat(tfls).groupby(['player_id', 'defteam', 'season'])['tfls'].sum().reset_index() if tfls else pd.DataFrame(columns=['player_id', 'defteam', 'season', 'tfls'])

        # Tackles
        tackle_cols = ['solo_tackle_1_player_id', 'solo_tackle_2_player_id']
        tackles = []
        for col in tackle_cols:
            if col in self.pbp.columns:
                t = self.pbp[self.pbp[col].notna()].groupby([col, 'defteam', 'season']).size().reset_index(name='tackles')
                t = t.rename(columns={col: 'player_id'})
                tackles.append(t)
        tackles_df = pd.concat(tackles).groupby(['player_id', 'defteam', 'season'])['tackles'].sum().reset_index() if tackles else pd.DataFrame(columns=['player_id', 'defteam', 'season', 'tackles'])

        # Pass deflections
        pd_cols = ['pass_defense_1_player_id', 'pass_defense_2_player_id']
        pds = []
        for col in pd_cols:
            if col in self.pbp.columns:
                p = self.pbp[self.pbp[col].notna()].groupby([col, 'defteam', 'season']).size().reset_index(name='pass_deflections')
                p = p.rename(columns={col: 'player_id'})
                pds.append(p)
        pds_df = pd.concat(pds).groupby(['player_id', 'defteam', 'season'])['pass_deflections'].sum().reset_index() if pds else pd.DataFrame(columns=['player_id', 'defteam', 'season', 'pass_deflections'])

        # Interceptions
        if 'interception_player_id' in self.pbp.columns:
            ints_df = self.pbp[self.pbp['interception_player_id'].notna()].groupby(
                ['interception_player_id', 'defteam', 'season']
            ).size().reset_index(name='ints').rename(columns={'interception_player_id': 'player_id'})
        else:
            ints_df = pd.DataFrame(columns=['player_id', 'defteam', 'season', 'ints'])

        # Run stops: tackles on run plays where gain <= 3
        run_stops = []
        for col in tackle_cols:
            if col in self.pbp.columns:
                rs = self.pbp[
                    (self.pbp[col].notna()) & 
                    (self.pbp['rush'] == 1) &
                    (self.pbp['yards_gained'].fillna(0) <= 3)
                ].groupby([col, 'defteam', 'season']).size().reset_index(name='run_stops')
                rs = rs.rename(columns={col: 'player_id'})
                run_stops.append(rs)
        run_stops_df = pd.concat(run_stops).groupby(['player_id', 'defteam', 'season'])['run_stops'].sum().reset_index() if run_stops else pd.DataFrame(columns=['player_id', 'defteam', 'season', 'run_stops'])

        # Merge all
        stats = snap_stats
        for df in [sacks_df, hits_df, tfls_df, tackles_df, pds_df, ints_df, run_stops_df]:
            if len(df) > 0:
                stats = stats.merge(df, on=['player_id', 'defteam', 'season'], how='left')

        stats = stats.fillna(0)
        stats['snaps_per_game'] = stats['snaps'] / stats['games']
        stats['sacks_per_100_snaps'] = stats['sacks'] / stats['snaps'] * 100
        stats['qb_hits_per_100_snaps'] = stats['qb_hits'] / stats['snaps'] * 100
        stats['tfls_per_100_snaps'] = stats['tfls'] / stats['snaps'] * 100
        stats['tackles_per_100_snaps'] = stats['tackles'] / stats['snaps'] * 100
        stats['pass_deflections_per_100_snaps'] = stats['pass_deflections'] / stats['snaps'] * 100
        stats['ints_per_100_snaps'] = stats['ints'] / stats['snaps'] * 100
        stats['run_stops_per_100_snaps'] = stats['run_stops'] / stats['snaps'] * 100
        stats['pass_rush_snap_pct'] = stats['pass_snaps'] / stats['snaps']
        stats['coverage_snap_pct'] = stats['pass_snaps'] / stats['snaps']  # Approximation
        stats['slot_snap_pct'] = np.nan
        stats['blitz_rate'] = np.nan
        stats['team_epa_per_play_allowed'] = stats['team_epa_allowed'] / stats['snaps']

        players = nfl.import_players()[['gsis_id', 'short_name']].rename(
            columns={'gsis_id': 'player_id', 'short_name': 'player_name'}
        )
        players = players.dropna(subset=['player_id']).drop_duplicates('player_id')
        stats = stats.merge(players, on='player_id', how='left')
        stats['player_name'] = stats['player_name'].fillna(stats['player_id'])

        stats = stats.rename(columns={'defteam': 'team'})
        stats = stats[stats['snaps'] >= self.config['min_snaps']]
        self.player_features = stats

    # =====================================================================
    # 3. PLAYER CLUSTERING (Kevin Cole's approach)
    # =====================================================================

    def _prepare_clustering_features(self):
        """Fit the scaler + PCA once; only the final K-means step varies per ensemble iteration."""
        print("Preparing clustering features (scaler + PCA fit once, K-means varies per iteration)...")

        features = self.config['clustering_features']
        available_features = [
            f for f in features
            if f in self.player_features.columns and self.player_features[f].notna().any()
        ]
        missing = set(features) - set(available_features)
        if missing:
            print(f"  Note: Missing features (will be imputed): {missing}")

        feature_df = self.player_features[available_features].copy()

        # Impute missing with position median
        for col in feature_df.columns:
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())

        valid_idx = feature_df.dropna().index
        feature_df = feature_df.loc[valid_idx]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(feature_df)

        n_components = min(5, len(available_features))
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        print(f"  PCA explained variance: {self.pca.explained_variance_ratio_.sum():.1%}")

        self._available_features = available_features
        self._valid_idx = valid_idx
        self._X_pca = X_pca
        return self

    def _kmeans_labels(self, n_clusters, random_state):
        """Run K-means on the pre-computed PCA embedding for one ensemble iteration."""
        n_clusters = max(2, min(n_clusters, len(self._valid_idx) // 2 or 2))
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        return kmeans.fit_predict(self._X_pca)

    def cluster_players(self, n_clusters=None):
        """Single-shot clustering (diagnostic use only; run_full_pipeline uses the ensemble)."""
        if self._X_pca is None:
            self._prepare_clustering_features()

        n_clusters = n_clusters or self.config['n_clusters']
        clusters = self._kmeans_labels(n_clusters, self.random_state)

        self.player_features.loc[self._valid_idx, 'cluster'] = clusters
        self.player_features['cluster'] = self.player_features['cluster'].fillna(0).astype(int)

        print(f"\nCluster profiles (mean values, k={n_clusters}):")
        summary = self.player_features.groupby('cluster')[self._available_features].mean()
        print(summary.round(3).to_string())

        return self

    # =====================================================================
    # 4. ON-OFF EPA SPLITS (using participation data for true on-field PM)
    # =====================================================================

    def compute_on_off_splits(self):
        """
        Compute raw on-off EPA splits using actual snap-level participation.

        For offense: ON = plays where player is in offense_players
        For defense: ON = plays where player is in defense_players

        Uses efficient team-total subtraction to avoid per-player set ops.
        """
        print(f"\n{'='*60}")
        print("STEP 3: Computing On-Off EPA Splits")
        print(f"{'='*60}")

        if self.config['use_participation'] and (self.participation is not None):
            return self._compute_on_off_from_participation()
        else:
            return self._compute_on_off_from_pbp_ids()

    def _compute_on_off_from_participation(self):
        """Compute on-off using snap-level participation data."""
        side = self.config['side']

        # Filter pbp to relevant plays
        mask = self.config['play_filter'](self.pbp)
        pbp = self.pbp[
            mask & 
            (self.pbp['season_type'] == 'REG') &
            (self.pbp['epa'].notna())
        ][['game_id', 'play_id', 'season', 'posteam', 'defteam', 'epa']].copy()

        # Prepare participation data
        if side == 'offense':
            part = self.participation[
                self.participation['offense_players'].notna()
            ][['game_id', 'play_id', 'season', 'possession_team', 'offense_players']].copy()
            part['player_id'] = part['offense_players'].str.split(';')
            team_col = 'possession_team'
        else:
            part = self.participation[
                self.participation['defense_players'].notna()
            ][['game_id', 'play_id', 'season', 'defense_players']].copy()
            part['player_id'] = part['defense_players'].str.split(';')
            # Need defteam from pbp merge
            team_col = None

        part = part.explode('player_id')
        part = part[part['player_id'] != ''].drop_duplicates()

        # Merge with pbp to get EPA and team
        merged = part.merge(pbp, on=['game_id', 'play_id', 'season'])

        if side == 'defense':
            team_col = 'defteam'

        # Filter to players in our feature set
        valid_players = set(self.player_features['player_id'].unique())
        merged = merged[merged['player_id'].isin(valid_players)]

        # Compute team-season totals
        team_totals = merged.groupby([team_col, 'season']).agg(
            team_plays=('epa', 'count'),
            team_epa_sum=('epa', 'sum'),
        ).reset_index()

        # Compute player-season ON totals
        player_on = merged.groupby(['player_id', team_col, 'season']).agg(
            n_on=('epa', 'count'),
            epa_on_sum=('epa', 'sum'),
        ).reset_index()

        # Merge team totals
        player_on = player_on.merge(team_totals, on=[team_col, 'season'])

        # Compute OFF = team - player
        player_on['n_off'] = player_on['team_plays'] - player_on['n_on']
        player_on['epa_off_sum'] = player_on['team_epa_sum'] - player_on['epa_on_sum']

        player_on['epa_on'] = player_on['epa_on_sum'] / player_on['n_on']
        player_on['epa_off'] = player_on['epa_off_sum'] / player_on['n_off'].clip(lower=1)

        # For defense, flip sign so positive = good
        if side == 'defense':
            player_on['raw_plus_minus'] = player_on['epa_off'] - player_on['epa_on']
        else:
            player_on['raw_plus_minus'] = player_on['epa_on'] - player_on['epa_off']

        # Standard errors
        # Approximate using pooled SD / sqrt(n)
        player_on['se'] = 0.05  # Placeholder; could compute from play-level SD

        # Add cluster info
        cluster_map = self.player_features.set_index('player_id')['cluster'].to_dict()
        player_on['cluster'] = player_on['player_id'].map(cluster_map)

        # Add player names
        name_map = self.player_features.set_index('player_id')['player_name'].to_dict()
        player_on['player_name'] = player_on['player_id'].map(name_map)
        player_on['team'] = player_on[team_col]

        self.splits_df = player_on[
            player_on['n_on'] >= self.config['min_snaps']
        ].copy()

        print(f"Computed splits for {len(self.splits_df)} player-seasons")
        if len(self.splits_df) > 0:
            print(f"Raw Plus-Minus range: [{self.splits_df['raw_plus_minus'].min():.3f}, "
                  f"{self.splits_df['raw_plus_minus'].max():.3f}]")

        return self

    def _compute_on_off_from_pbp_ids(self):
        """Compute on-off using pbp player ID columns (QB, RB, WR, TE)."""
        pbp = self.pbp[
            (self.pbp['season_type'] == 'REG') &
            self.config['play_filter'](self.pbp) &
            (self.pbp['epa'].notna())
        ].copy()

        id_col = self.config['pbp_id_col']

        # For RB, we need to handle both rush and pass
        if self.position == 'RB':
            # Rush plays
            rush_on = pbp[pbp['rush'] == 1].groupby(['rusher_player_id', 'posteam', 'season']).agg(
                n_on_rush=('epa', 'count'),
                epa_on_rush=('epa', 'sum'),
            ).reset_index().rename(columns={'rusher_player_id': 'player_id'})

            # Pass plays where RB targeted
            rec_on = pbp[
                (pbp['pass'] == 1) & (pbp['receiver_position'] == 'RB')
            ].groupby(['receiver_player_id', 'posteam', 'season']).agg(
                n_on_rec=('epa', 'count'),
                epa_on_rec=('epa', 'sum'),
            ).reset_index().rename(columns={'receiver_player_id': 'player_id'})

            player_on = rush_on.merge(rec_on, on=['player_id', 'posteam', 'season'], how='outer')
            player_on = player_on.fillna(0)
            player_on['n_on'] = player_on['n_on_rush'] + player_on['n_on_rec']
            player_on['epa_on_sum'] = player_on['epa_on_rush'] + player_on['epa_on_rec']

            # Team totals for relevant plays
            team_rush = pbp[pbp['rush'] == 1].groupby(['posteam', 'season']).agg(
                team_rush_plays=('epa', 'count'),
                team_rush_epa=('epa', 'sum'),
            ).reset_index()
            team_rec = pbp[
                (pbp['pass'] == 1) & (pbp['receiver_position'] == 'RB')
            ].groupby(['posteam', 'season']).agg(
                team_rec_plays=('epa', 'count'),
                team_rec_epa=('epa', 'sum'),
            ).reset_index()
            team_totals = team_rush.merge(team_rec, on=['posteam', 'season'], how='outer').fillna(0)
            team_totals['team_plays'] = team_totals['team_rush_plays'] + team_totals['team_rec_plays']
            team_totals['team_epa_sum'] = team_totals['team_rush_epa'] + team_totals['team_rec_epa']

            player_on = player_on.merge(team_totals, on=['posteam', 'season'])
        else:
            player_on = pbp.groupby([id_col, 'posteam', 'season']).agg(
                n_on=('epa', 'count'),
                epa_on_sum=('epa', 'sum'),
            ).reset_index().rename(columns={id_col: 'player_id'})

            team_totals = pbp.groupby(['posteam', 'season']).agg(
                team_plays=('epa', 'count'),
                team_epa_sum=('epa', 'sum'),
            ).reset_index()

            player_on = player_on.merge(team_totals, on=['posteam', 'season'])

        player_on['n_off'] = player_on['team_plays'] - player_on['n_on']
        player_on['epa_off_sum'] = player_on['team_epa_sum'] - player_on['epa_on_sum']

        player_on['epa_on'] = player_on['epa_on_sum'] / player_on['n_on']
        player_on['epa_off'] = player_on['epa_off_sum'] / player_on['n_off'].clip(lower=1)
        player_on['raw_plus_minus'] = player_on['epa_on'] - player_on['epa_off']
        player_on['se'] = 0.05

        cluster_map = self.player_features.set_index('player_id')['cluster'].to_dict()
        player_on['cluster'] = player_on['player_id'].map(cluster_map)

        name_map = self.player_features.set_index('player_id')['player_name'].to_dict()
        player_on['player_name'] = player_on['player_id'].map(name_map)
        player_on['team'] = player_on['posteam']

        self.splits_df = player_on[
            player_on['n_on'] >= self.config['min_snaps']
        ].copy()

        print(f"Computed splits for {len(self.splits_df)} player-seasons")
        if len(self.splits_df) > 0:
            print(f"Raw Plus-Minus range: [{self.splits_df['raw_plus_minus'].min():.3f}, "
                  f"{self.splits_df['raw_plus_minus'].max():.3f}]")

        return self

    # =====================================================================
    # 5. BAYESIAN HIERARCHICAL MODEL
    # =====================================================================

    def _fit_hierarchical_bayesian(self, df, draws, tune, cores=4, sample_posterior_predictive=True):
        """Fit the cluster-informed hierarchical model for one clustering assignment.

        Shared by fit_bayesian_model() (single run) and run_clustering_ensemble()
        (re-fit with a different random clustering each iteration).
        """
        df = df.copy().reset_index(drop=True)
        n_players = len(df)

        unique_clusters = sorted(df['cluster'].unique())
        cluster_map = {c: i for i, c in enumerate(unique_clusters)}
        cluster_idx = df['cluster'].map(cluster_map).values
        n_clusters = len(unique_clusters)

        weights = np.sqrt(df['n_on'] / df['n_on'].max())

        with pm.Model():
            mu_cluster = pm.Normal('mu_cluster', mu=0, sigma=0.2, shape=n_clusters)
            sigma_cluster = pm.HalfNormal('sigma_cluster', sigma=0.15)

            individual = pm.Normal('individual', mu=0, sigma=sigma_cluster, shape=n_players)
            player_effect = mu_cluster[cluster_idx] + individual

            sigma_extra = pm.HalfNormal('sigma_extra', sigma=0.1)
            sigma_obs = df['se'].values + sigma_extra

            pm.Normal(
                'obs',
                mu=player_effect,
                sigma=sigma_obs / weights,
                observed=df['raw_plus_minus'].values
            )

            trace = pm.sample(
                draws=draws,
                tune=tune,
                cores=cores,
                return_inferencedata=True,
                target_accept=0.9,
                progressbar=False,  # rich's Windows console renderer crashes on non-UTF8 stdout (e.g. redirected output)
            )

            if sample_posterior_predictive:
                pm.sample_posterior_predictive(trace, extend_inferencedata=True)

        posterior = trace.posterior
        mu_cluster_post = posterior['mu_cluster'].mean(dim=['chain', 'draw']).values
        individual_post = posterior['individual'].mean(dim=['chain', 'draw']).values

        df['bayesian_plus_minus'] = mu_cluster_post[cluster_idx] + individual_post

        player_effect_samples = (
            posterior['mu_cluster'].values[:, :, cluster_idx] +
            posterior['individual'].values[:, :, :]
        )
        df['pm_lower'] = np.percentile(player_effect_samples, 2.5, axis=(0, 1))
        df['pm_upper'] = np.percentile(player_effect_samples, 97.5, axis=(0, 1))
        df['pm_uncertainty'] = df['pm_upper'] - df['pm_lower']

        return trace, df

    def fit_bayesian_model(self, cores=4):
        """Fit a single hierarchical Bayesian model against the current cluster assignment.

        For Kevin Cole's full ensemble approach (recommended), use run_clustering_ensemble() instead.
        """
        print(f"\n{'='*60}")
        print("Fitting Bayesian Hierarchical Model (single clustering run)")
        print(f"{'='*60}")

        if self.splits_df is None or len(self.splits_df) == 0:
            print("WARNING: No splits data available. Skipping Bayesian model.")
            return self

        self.trace, self.splits_df = self._fit_hierarchical_bayesian(
            self.splits_df, draws=self.bayesian_draws, tune=self.bayesian_tune,
            cores=cores, sample_posterior_predictive=True
        )
        print("Bayesian model sampling complete!")
        return self

    def run_clustering_ensemble(self, cores=4):
        """Kevin Cole-style clustering ensemble.

        Re-clusters players with a randomly chosen number of clusters on each iteration,
        re-fits the hierarchical Bayesian model against that clustering, and averages the
        per-player estimates across all iterations at the end. This avoids the analysis
        being sensitive to any single, arbitrary choice of cluster count.
        """
        print(f"\n{'='*60}")
        print("STEP 4: Clustering Ensemble + Bayesian Hierarchical Model")
        print(f"{'='*60}")

        if self.splits_df is None or len(self.splits_df) == 0:
            print("WARNING: No splits data available. Skipping clustering ensemble.")
            return self

        self._prepare_clustering_features()

        n_valid = len(self._valid_idx)
        k_min = self.cluster_k_min or max(2, self.config['n_clusters'] - 2)
        k_max = self.cluster_k_max or self.config['n_clusters'] + 3
        k_max = max(k_min, min(k_max, max(2, n_valid // 3)))
        print(f"Running {self.n_cluster_iterations} clustering iterations (k sampled from [{k_min}, {k_max}])")

        rng = np.random.RandomState(self.random_state)
        base_df = self.splits_df.drop(columns=['cluster'], errors='ignore').reset_index(drop=True)
        progress_every = max(1, self.n_cluster_iterations // 10)

        iteration_frames = []
        for iteration in range(self.n_cluster_iterations):
            k = int(rng.randint(k_min, k_max + 1))
            seed = int(rng.randint(0, 1_000_000))

            labels = self._kmeans_labels(k, seed)
            cluster_by_id = pd.Series(0, index=self.player_features.index)
            cluster_by_id.loc[self._valid_idx] = labels
            cluster_map = dict(zip(self.player_features['player_id'], cluster_by_id))

            df_iter = base_df.copy()
            df_iter['cluster'] = df_iter['player_id'].map(cluster_map).fillna(0).astype(int)

            _, fitted = self._fit_hierarchical_bayesian(
                df_iter, draws=self.ensemble_draws, tune=self.ensemble_tune,
                cores=cores, sample_posterior_predictive=False
            )
            fitted['iteration'] = iteration
            fitted['n_clusters_used'] = k
            iteration_frames.append(
                fitted[['player_id', 'team', 'season', 'cluster', 'iteration',
                        'n_clusters_used', 'bayesian_plus_minus', 'pm_lower', 'pm_upper']]
            )

            if (iteration + 1) % progress_every == 0 or iteration == self.n_cluster_iterations - 1:
                print(f"  Completed {iteration + 1}/{self.n_cluster_iterations} iterations (last k={k})")

        all_iterations = pd.concat(iteration_frames, ignore_index=True)
        averaged = all_iterations.groupby(['player_id', 'team', 'season'], as_index=False).agg(
            bayesian_plus_minus=('bayesian_plus_minus', 'mean'),
            pm_lower=('pm_lower', 'mean'),
            pm_upper=('pm_upper', 'mean'),
        )
        averaged['pm_uncertainty'] = averaged['pm_upper'] - averaged['pm_lower']

        # Keep the final iteration's cluster labels for illustrative grouping only;
        # the point estimates above are averaged across every clustering in the ensemble.
        last_clusters = all_iterations[all_iterations['iteration'] == self.n_cluster_iterations - 1]
        last_clusters = last_clusters[['player_id', 'team', 'season', 'cluster']]

        self.splits_df = base_df.merge(averaged, on=['player_id', 'team', 'season'], how='left')
        self.splits_df = self.splits_df.merge(last_clusters, on=['player_id', 'team', 'season'], how='left')
        self.splits_df['cluster'] = self.splits_df['cluster'].fillna(0).astype(int)

        self.cluster_ensemble_results = all_iterations
        print(f"Ensemble complete: averaged Bayesian estimates over {self.n_cluster_iterations} cluster iterations")
        print("(cluster labels shown below reflect one representative draw from the ensemble)")

        return self

    # =====================================================================
    # 6. RAPM BASELINE (Ridge Regression)
    # =====================================================================

    def fit_rapm_baseline(self, max_plays=50000):
        """
        Fit Ridge regression (RAPM) baseline for comparison.
        Uses play-level design matrix with player indicators + situational controls.
        """
        print(f"\n{'='*60}")
        print("STEP 5: Fitting Ridge RAPM Baseline (for comparison)")
        print(f"{'='*60}")

        if self.splits_df is None or len(self.splits_df) == 0:
            print("WARNING: No splits data. Skipping RAPM.")
            return self

        valid_players = set(self.splits_df['player_id'].unique())

        # Sample plays
        pbp = self.pbp[
            (self.pbp['season_type'] == 'REG') &
            self.config['play_filter'](self.pbp) &
            (self.pbp['epa'].notna())
        ].copy()

        if len(pbp) > max_plays:
            pbp = pbp.sample(max_plays, random_state=self.random_state)

        n_plays = len(pbp)
        player_ids = sorted(valid_players)
        id_to_idx = {pid: i for i, pid in enumerate(player_ids)}
        n_players = len(player_ids)

        # Build design matrix
        X = np.zeros((n_plays, n_players + 5))

        if self.config['pbp_id_col'] and self.config['pbp_id_col'] in pbp.columns:
            id_col = self.config['pbp_id_col']
            for i, rid in enumerate(pbp[id_col].fillna('')):
                if rid in id_to_idx:
                    X[i, id_to_idx[rid]] = 1
        else:
            # For participation-based positions, we'd need to explode participation
            # This is computationally expensive; skip for now or use approximation
            print("  Note: RAPM for participation-based positions requires full play-player matrix.")
            print("  Skipping RAPM baseline for this position.")
            self.splits_df['rapm'] = np.nan
            return self

        # Situational controls
        X[:, -5] = pbp['down'].fillna(1) / 4.0
        X[:, -4] = pbp['ydstogo'].fillna(10) / 10.0
        X[:, -3] = pbp['yardline_100'].fillna(50) / 100.0
        X[:, -2] = pbp['score_differential'].fillna(0) / 21.0
        X[:, -1] = pbp['qtr'].fillna(1) / 4.0

        y = pbp['epa'].values

        self.rapm_model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
        self.rapm_model.fit(X, y)

        rapm_scores = pd.DataFrame({
            'player_id': player_ids,
            'rapm': self.rapm_model.coef_[:n_players]
        })

        self.splits_df = self.splits_df.merge(rapm_scores, on='player_id', how='left')

        print(f"RAPM fitted on {n_plays:,} plays")
        return self

    # =====================================================================
    # 7. RESULTS & VISUALIZATION
    # =====================================================================

    def get_results(self, top_n=25):
        """Return ranked results with all metrics."""
        if self.splits_df is None or len(self.splits_df) == 0:
            return pd.DataFrame()

        cols = ['player_name', 'team', 'season', 'cluster', 'n_on', 'n_off',
                'raw_plus_minus', 'bayesian_plus_minus', 'pm_lower', 'pm_upper']
        if 'rapm' in self.splits_df.columns:
            cols.append('rapm')

        available_cols = [c for c in cols if c in self.splits_df.columns]
        results = self.splits_df[available_cols].sort_values('bayesian_plus_minus', ascending=False)
        return results.head(top_n)

    def plot_results(self, top_n=20):
        """Visualize top players with uncertainty intervals."""
        if self.splits_df is None or len(self.splits_df) == 0:
            print("No results to plot.")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Top Bayesian estimates with credible intervals
        top = self.splits_df.nlargest(top_n, 'bayesian_plus_minus')
        ax = axes[0, 0]
        y_pos = np.arange(len(top))
        ax.errorbar(
            top['bayesian_plus_minus'], y_pos,
            xerr=[top['bayesian_plus_minus'] - top['pm_lower'],
                  top['pm_upper'] - top['bayesian_plus_minus']],
            fmt='o', capsize=3, color='steelblue'
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{r.player_name} ({r.team})" for _, r in top.iterrows()], fontsize=8)
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Bayesian Plus-Minus (EPA/play)')
        ax.set_title(f'Top {top_n} {self.position}s - Bayesian Estimates with 95% CI')
        ax.invert_yaxis()

        # 2. Raw vs Bayesian (shrinkage)
        ax = axes[0, 1]
        ax.scatter(
            self.splits_df['raw_plus_minus'],
            self.splits_df['bayesian_plus_minus'],
            c=self.splits_df['cluster'], cmap='tab10', alpha=0.6, edgecolors='black'
        )
        lim = max(abs(self.splits_df['raw_plus_minus'].max()), abs(self.splits_df['raw_plus_minus'].min()))
        ax.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5, label='No shrinkage')
        ax.set_xlabel('Raw Plus-Minus')
        ax.set_ylabel('Bayesian Plus-Minus')
        ax.set_title('Shrinkage Effect: Raw vs Bayesian')
        ax.legend()

        # 3. Cluster means
        ax = axes[1, 0]
        cluster_summary = self.splits_df.groupby('cluster').agg({
            'bayesian_plus_minus': 'mean',
            'raw_plus_minus': 'mean',
            'player_name': 'count'
        }).rename(columns={'player_name': 'n_players'})

        x = np.arange(len(cluster_summary))
        width = 0.35
        ax.bar(x - width/2, cluster_summary['raw_plus_minus'], width, label='Raw', alpha=0.7)
        ax.bar(x + width/2, cluster_summary['bayesian_plus_minus'], width, label='Bayesian', alpha=0.7)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Mean Plus-Minus')
        ax.set_title('Cluster-Level Estimates')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i}' for i in cluster_summary.index])
        ax.legend()

        # 4. Bayesian vs RAPM
        ax = axes[1, 1]
        if 'rapm' in self.splits_df.columns and self.splits_df['rapm'].notna().any():
            valid = self.splits_df.dropna(subset=['rapm'])
            ax.scatter(valid['rapm'], valid['bayesian_plus_minus'], alpha=0.5, edgecolors='black')
            lim = max(abs(valid['rapm'].max()), abs(valid['rapm'].min()))
            ax.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5)
            ax.set_xlabel('Ridge RAPM')
            ax.set_ylabel('Bayesian Plus-Minus')
            ax.set_title('Bayesian vs Ridge RAPM')
            corr = valid['rapm'].corr(valid['bayesian_plus_minus'])
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'RAPM not available\nfor this position',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Bayesian vs Ridge RAPM')

        plt.tight_layout()
        output_path = Path(__file__).resolve().parent / f'nfl_bayesian_plusminus_results_{self.position}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\nPlot saved to {output_path}")

        return fig

    def print_cluster_examples(self, n_examples=3):
        """Print example players from each cluster."""
        if self.splits_df is None or len(self.splits_df) == 0:
            return

        print(f"\n{'='*60}")
        print("CLUSTER EXAMPLES")
        print(f"{'='*60}")

        for cluster in sorted(self.splits_df['cluster'].unique()):
            cluster_players = self.splits_df[self.splits_df['cluster'] == cluster]
            top = cluster_players.nlargest(n_examples, 'bayesian_plus_minus')

            print(f"\n--- Cluster {cluster} "
                  f"(n={len(cluster_players)}, "
                  f"mean BPM={cluster_players['bayesian_plus_minus'].mean():.3f}) ---")
            for _, p in top.iterrows():
                print(f"  {p['player_name']} ({p['team']} {p['season']}): "
                      f"BPM={p['bayesian_plus_minus']:+.3f}, "
                      f"raw={p['raw_plus_minus']:+.3f}")

    # =====================================================================
    # 8. FULL PIPELINE
    # =====================================================================

    def save_results(self):
        """Persist the ensemble-averaged splits and raw per-iteration estimates to CSV."""
        out_dir = Path(__file__).resolve().parent

        splits_path = out_dir / f'nfl_bayesian_plusminus_splits_{self.position}.csv'
        self.splits_df.to_csv(splits_path, index=False)
        print(f"Saved final results to {splits_path}")

        if self.cluster_ensemble_results is not None:
            iters_path = out_dir / f'nfl_bayesian_plusminus_ensemble_iterations_{self.position}.csv'
            self.cluster_ensemble_results.to_csv(iters_path, index=False)
            print(f"Saved raw per-iteration ensemble results to {iters_path}")

        return self

    def run_full_pipeline(self):
        """Execute complete analysis pipeline."""
        self.load_data()
        self.compute_player_features()
        self.player_features['cluster'] = 0  # placeholder; real clusters assigned per ensemble iteration
        self.compute_on_off_splits()
        self.run_clustering_ensemble()
        self.fit_rapm_baseline()
        self.save_results()

        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}")

        self.print_cluster_examples()

        print(f"\n{'='*60}")
        print(f"TOP 15 PLAYERS - {self.position} (Bayesian Plus-Minus)")
        print(f"{'='*60}")
        results = self.get_results(15)
        if len(results) > 0:
            print(results.to_string(index=False))
        else:
            print("No results available.")

        return self


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example 1: Wide Receivers (uses pbp event IDs)
    print("="*60)
    print("EXAMPLE 1: Wide Receivers")
    print("="*60)

    wr_model = NFLBayesianPlusMinus(
        position='WR',
        years=[2021, 2022, 2023],
        random_state=42
    )
    wr_model.run_full_pipeline()
    wr_model.plot_results(top_n=20)

    # Example 2: Defensive Backs (uses participation data)
    print("\n" + "="*60)
    print("EXAMPLE 2: Defensive Backs")
    print("="*60)

    db_model = NFLBayesianPlusMinus(
        position='DB',
        years=[2021, 2022, 2023],
        random_state=42
    )
    #db_model.run_full_pipeline()
    #db_model.plot_results(top_n=20)