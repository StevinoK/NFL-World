"""
=============================================================================
NFL Bayesian Plus-Minus with Player Clustering — Full Positional Support
=============================================================================

A Python implementation combining:
  1. Kevin Cole's player clustering methodology for NFL Plus/Minus
  2. Bayesian hierarchical shrinkage toward cluster means  
  3. Snap-level participation data (2016+) for true on-field plus-minus
  4. Free/open-source data via nfl-data-py and nflverse releases

Clustering uses play-by-play event features only. Weekly estimates are
season-to-date through the end of the labeled week, not pregame predictions.
Completed prior seasons supply player priors, with aging and transition uncertainty.
Participation remains required for OL and defensive on/off exposure.

Fast method='box' pools completed seasons strictly before each target season,
fits historical scaler/KMeans/ridge once, and shrinks weekly event-stat predictions
toward aged player history. It does not sample or report credible intervals.
The API retains method='bayesian' by default; executable examples use box mode.

Dependencies:
    pip install nfl-data-py pandas numpy scikit-learn pymc arviz matplotlib pyarrow

Participation data sources:
  - 2016-2022: NGS via nflverse (CC-BY 4.0)
  - 2023+: FTN Data via nflverse (CC-BY-SA 4.0)

Usage:
    model = PlayerPlusMinus(position='DB', years=[2021, 2022, 2023], method='box')
    model.run_full_pipeline()
    results = model.get_results()
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import logging
import io
import os
import tempfile
import zipfile
import hashlib
import json
import joblib
from datetime import date
from pathlib import Path
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV, Ridge
import pymc as pm
from pytensor.compile.mode import Mode
import arviz as az
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import player_value.positions as seasonal_positions
from player_value.calibration import (
    CONFIG_PATH as IN_SEASON_CONFIG_PATH, PRIOR_FEATURES, fit_prior_regression, metadata_from_rosters,
    prior_features, summarize_seasons,
)
from player_value import madden
from player_value.positions import eligible_rows, normalize_rosters, require_seasons, save_lookup

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_CACHE_DIR = REPO_DIR / 'data' / 'cache'
# Weekly estimates, checkpoints, fitted box models and the roster lookup.
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'


def _output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def receiver_position_mask(pbp, position):
    return pbp['receiver_position'].astype('string').str.contains(
        rf'(?:^|\|){position}(?:\||$)', na=False)

# The clustering ensemble re-fits the model ~100x; keep pymc's per-fit logging quiet.
logging.getLogger('pymc').setLevel(logging.WARNING)


# =============================================================================
# POSITION CONFIGURATION DICTIONARY
# =============================================================================
# Defines clustering features, play filters, and data requirements per position.
# Clustering uses pbp events; participation supplies on/off exposure only.
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
                                   ((pbp['pass'] == 1) & receiver_position_mask(pbp, 'RB')),
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
        'play_filter': lambda pbp: (pbp['pass'] == 1) & receiver_position_mask(pbp, 'WR'),
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
        'play_filter': lambda pbp: (pbp['pass'] == 1) & receiver_position_mask(pbp, 'TE'),
        'min_snaps': 40,
        'n_clusters': 5,
        'use_participation': False,
        'clustering_features': [
            'targets_per_game', 'catch_rate', 'yards_per_target',
            'air_yards_per_target', 'yac_per_reception', 'td_rate',
            'first_down_rate', 'target_share', 'epa_per_target'
        ],
        'feature_desc': 'Receiving event volume and efficiency profile',
    },
    'OL': {
        'side': 'offense',
        'pbp_id_col': None,
        'play_filter': lambda pbp: (pbp['pass'] == 1) | (pbp['rush'] == 1),
        'min_snaps': 200,
        'n_clusters': 4,
        'use_participation': True,
        'clustering_features': [
            'penalties_per_game'
        ],
        'feature_desc': 'Individually attributed penalty event profile',
    },
    'DL': {
        'side': 'defense',
        'pbp_id_col': None,
        'play_filter': lambda pbp: (pbp['pass'] == 1) | (pbp['rush'] == 1),
        'min_snaps': 150,
        'n_clusters': 5,
        'use_participation': True,
        'clustering_features': [
            'sacks_per_game', 'qb_hits_per_game',
            'tfls_per_game', 'run_stops_per_game'
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
            'tackles_per_game', 'tfls_per_game',
            'sacks_per_game', 'ints_per_game',
            'pass_deflections_per_game', 'run_stops_per_game'
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
            'ints_per_game', 'pass_deflections_per_game',
            'tackles_per_game', 'run_stops_per_game'
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
    cache_dir = DATA_CACHE_DIR
    for year in years:
        if year < 2016:
            print(f"  Skipping {year}: participation data only available from 2016+")
            continue

        cache_file = cache_dir / f"pbp_participation_{year}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                df['season'] = year
                dfs.append(df)
                print(f"  Loaded cached participation for {year}: {len(df):,} plays")
                continue
            except Exception:
                pass

        urls = [
            f"https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{year}.parquet",
        ]

        loaded = False
        for url in urls:
            try:
                df = pd.read_parquet(url)
                df['season'] = year
                if cache_dir.exists():
                    try:
                        df.to_parquet(cache_file)
                    except Exception:
                        pass
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


def add_receiver_position(pbp: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Use only the receiver's explicitly rostered groups in the play's season."""
    valid = lookup.dropna(subset=['position_group', 'player_id', 'season'])[['player_id', 'season', 'position_group']].drop_duplicates()
    positions = valid.groupby(['player_id', 'season'])['position_group'].agg(lambda s: '|'.join(sorted(s)))
    positions = positions.rename('receiver_position').reset_index().rename(
        columns={'player_id': 'receiver_player_id'})
    rows = pbp.drop(columns=['receiver_position'], errors='ignore').copy()
    rows['receiver_player_id'] = rows['receiver_player_id'].astype('string').str.strip()
    return rows.merge(positions, on=['receiver_player_id', 'season'], how='left', validate='many_to_one')


class PlayerPlusMinus:
    """
    NFL plus-minus with historical box regression or Bayesian player clustering.

    Supports all position groups using snap-level participation data (2016+)
    for true on-field plus-minus, plus event-based on-off for skill positions.
    """

    @classmethod
    def calibrated(cls, position: str, years: List[int], config_path: Path = IN_SEASON_CONFIG_PATH, **overrides):
        """Box model using the out-of-sample in-season settings from ``player_value.calibration``."""
        settings = json.loads(Path(config_path).read_text()).get(position.upper(), {}) if Path(config_path).exists() else {}
        keys = {'prior_model': 'prior_model', 'prior_snaps': 'box_prior_snaps', 'history_decay': 'history_decay',
                'box_weighting': 'box_weighting', 'madden_features': 'madden_features'}
        arguments = {'method': 'box', **{name: settings[key] for key, name in keys.items() if key in settings}}
        return cls(position, years, **{**arguments, **overrides})

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
        weekly_min_snaps: Optional[int] = None,
        history_decay: float = 0.8,
        season_transition_sd: float = 0.06,
        rookie_prior_sd: float = 0.10,
        observation_reference_snaps: float = 100.0,
        method: str = 'bayesian',
        box_alpha: float = 100.0,
        box_prior_snaps: float = 200.0,
        box_cache: bool = True,
        box_cluster_iterations: Optional[int] = None,
        prior_model: str = 'decay',
        box_weighting: str = 'equal',
        madden_features: bool = False,
    ):
        position = position.upper()
        if position not in POSITION_CONFIG:
            raise ValueError(f"Position must be one of {list(POSITION_CONFIG.keys())}")

        self.position = position
        if method not in ('bayesian', 'box'):
            raise ValueError("method must be 'bayesian' or 'box'.")
        if not np.isfinite([box_alpha, box_prior_snaps]).all() or min(box_alpha, box_prior_snaps) <= 0:
            raise ValueError('Box regularization and prior sample size must be positive and finite.')
        if prior_model not in ('decay', 'zero', 'regression'):
            raise ValueError("prior_model must be 'decay', 'zero' or 'regression'.")
        if box_weighting not in ('equal', 'plays'):
            raise ValueError("box_weighting must be 'equal' or 'plays'.")
        self.method = method
        self.prior_model = prior_model
        self.box_weighting = box_weighting
        self.madden_features = madden_features
        self.box_alpha = box_alpha
        self.box_prior_snaps = box_prior_snaps
        self.box_cache = box_cache
        self.box_cluster_iterations = 20 if box_cluster_iterations is None else box_cluster_iterations
        if (not isinstance(self.box_cluster_iterations, int) or isinstance(self.box_cluster_iterations, bool)
                or not 1 <= self.box_cluster_iterations <= 100):
            raise ValueError('Box cluster iterations must be an integer in [1, 100].')
        if ((cluster_k_min is not None and (not isinstance(cluster_k_min, int) or cluster_k_min < 1))
                or (cluster_k_max is not None and (not isinstance(cluster_k_max, int) or cluster_k_max < 1))
                or (cluster_k_min is not None and cluster_k_max is not None and cluster_k_min > cluster_k_max)):
            raise ValueError('Cluster bounds must be positive integers with min <= max.')
        self._box_models = {}
        self._box_history = None
        self.years = years
        self.config = (config if config is not None else POSITION_CONFIG[position]).copy()
        if madden_features:
            if position not in madden.MADDEN_ATTRIBUTES:
                raise ValueError(f'Madden features are only defined for {sorted(madden.MADDEN_ATTRIBUTES)}.')
            self.config['clustering_features'] = list(self.config['clustering_features']) + madden.feature_names(position)
        if weekly_min_snaps is not None:
            self.config['min_snaps'] = weekly_min_snaps
        elif config is None:
            self.config['min_snaps'] = {
                'QB': 10, 'RB': 5, 'WR': 3, 'TE': 3,
                'OL': 20, 'DL': 15, 'LB': 15, 'DB': 15
            }[position]
        if not 0 < history_decay <= 1:
            raise ValueError('history_decay must be in (0, 1].')
        if min(season_transition_sd, rookie_prior_sd, observation_reference_snaps, self.config['min_snaps']) <= 0:
            raise ValueError('Prior scales, reference sample size, and minimum sample must be positive.')
        self.history_decay = history_decay
        self.season_transition_sd = season_transition_sd
        self.rookie_prior_sd = rookie_prior_sd
        self.observation_reference_snaps = observation_reference_snaps
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
        self.rosters = None
        self.players = None
        self._roster_source = None
        self._position_lookup = None
        self._receiver_positions = None

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

        # nfl_data_py's cache mode fails on any uncached season, so download those (e.g. the current one) fresh.
        cached = [year for year in self.years if (DATA_CACHE_DIR / f'season={year}').is_dir()]
        fresh = [year for year in self.years if year not in cached]
        frames = []
        if cached:
            frames.append(nfl.import_pbp_data(cached, downcast=True, cache=True,
                                              alt_path=str(DATA_CACHE_DIR.resolve())))
        if fresh:
            frames.append(nfl.import_pbp_data(fresh, downcast=True))
        self.pbp = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        self.rosters = nfl.import_seasonal_rosters(self.years)
        lookup = self._seasonal_positions()
        require_seasons(lookup, self.years)
        save_lookup(lookup, _output_dir())
        self.pbp = add_receiver_position(self.pbp, lookup)
        print(f"Loaded {len(self.pbp):,} total plays from nfl-data-py")

        needs_participation = self.config['use_participation']
        if needs_participation:
            print("\nLoading participation data from nflverse releases...")
            self.participation = load_participation_data(self.years)
            if len(self.participation) > 0:
                print(f"Total participation plays: {len(self.participation):,}")
            else:
                print("WARNING: No participation data loaded.")
                if self.config['use_participation']:
                    raise RuntimeError("Participation data required but not loaded.")

            self.players = nfl.import_players()

        return self

    # =====================================================================
    # 2. PLAYER FEATURE ENGINEERING
    # =====================================================================

    def _seasonal_positions(self):
        if self.rosters is None:
            self.rosters = nfl.import_seasonal_rosters(self.years)
        if self._roster_source is not self.rosters:
            self._position_lookup = normalize_rosters(self.rosters)
            self._roster_source = self.rosters
            self._receiver_positions = None
        return self._position_lookup

    def _receiver_positions_table(self):
        lookup = self._seasonal_positions()
        if self._receiver_positions is None:
            valid = lookup.dropna(subset=['position_group', 'player_id', 'season'])[['player_id', 'season', 'position_group']].drop_duplicates()
            positions = valid.groupby(['player_id', 'season'])['position_group'].agg(lambda s: '|'.join(sorted(s)))
            self._receiver_positions = positions.rename('receiver_position').reset_index().rename(
                columns={'player_id': 'receiver_player_id'})
        return self._receiver_positions

    def compute_player_features(self):
        """Compute seasonal stats for clustering."""
        lookup = self._seasonal_positions()
        require_seasons(lookup, self.pbp['season'].unique())
        if 'receiver_player_id' in self.pbp:
            positions = self._receiver_positions_table()
            rows = self.pbp.drop(columns=['receiver_position'], errors='ignore').copy()
            rows['receiver_player_id'] = rows['receiver_player_id'].astype('string').str.strip()
            self.pbp = rows.merge(positions, on=['receiver_player_id', 'season'], how='left', validate='many_to_one')
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

        self.player_features = eligible_rows(self.player_features, lookup, self.position)
        if self.madden_features:
            self.player_features = madden.attach_ratings(self.player_features, self.position)
        print(f"Computed features for {len(self.player_features)} player-seasons")
        return self

    def _qb_attributed_plays(self):
        """Attribute dropbacks to seasonal QBs, including identified scramble rushers."""
        pbp = self.pbp[
            self.pbp['season_type'].eq('REG') &
            (self.pbp['pass'].eq(1) | self.pbp['qb_scramble'].eq(1)) &
            self.pbp['epa'].notna()
        ].copy()
        identity = pbp['passer_player_id'].astype('string').str.strip().replace('', pd.NA)
        rusher = pbp.get('rusher_player_id', pd.Series(pd.NA, index=pbp.index)).astype('string')
        rusher = rusher.str.strip().replace('', pd.NA)
        pbp['player_id'] = identity.fillna(rusher.where(pbp['qb_scramble'].eq(1)))
        pbp = eligible_rows(pbp, self._seasonal_positions(), 'QB')
        pbp['passer_player_id'] = pbp['player_id']
        if 'rusher_player_name' in pbp:
            pbp['passer_player_name'] = pbp['passer_player_name'].fillna(
                pbp['rusher_player_name'].where(pbp['qb_scramble'].eq(1)))
        return pbp.drop(columns='player_id')

    def _compute_qb_features(self):
        pbp = self._qb_attributed_plays()

        stats = pbp.groupby(['passer_player_id', 'posteam', 'season']).agg(
            passer_player_name=('passer_player_name', 'first'),
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
            ['rusher_player_id', 'posteam', 'season']
        ).agg(
            rusher_player_name=('rusher_player_name', 'first'),
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
            receiver_position_mask(pbp, 'RB')
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
            pbp = pbp[receiver_position_mask(pbp, 'TE')]
        else:
            pbp = pbp[receiver_position_mask(pbp, 'WR')]

        stats = pbp.groupby(['receiver_player_id', 'posteam', 'season']).agg(
            receiver_player_name=('receiver_player_name', 'first'),
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

        stats = stats.rename(columns={'receiver_player_id': 'player_id', 'receiver_player_name': 'player_name'})
        stats = stats[stats['targets'] >= self.config['min_snaps']]
        self.player_features = stats

    def _compute_ol_features(self):
        """Attributed penalty events, with participation for eligibility and games."""
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

        merged = eligible_rows(merged, self._seasonal_positions(), 'OL')

        stats = merged.groupby(['player_id', 'possession_team', 'season']).agg(
            games=('game_id', 'nunique'),
            snaps=('play_id', 'count'),
        ).reset_index()

        events = self.pbp[
            (self.pbp['season_type'] == 'REG') & (self.pbp['penalty'] == 1)
        ]
        penalties = events.groupby(['penalty_player_id', 'penalty_team', 'season']).size().reset_index(name='penalties')
        penalties = penalties.rename(columns={
            'penalty_player_id': 'player_id', 'penalty_team': 'possession_team'
        })
        stats = stats.merge(penalties, on=['player_id', 'possession_team', 'season'], how='left')
        stats['penalties'] = stats['penalties'].fillna(0)
        stats['penalties_per_game'] = stats['penalties'] / stats['games']
        players = self.players if self.players is not None else nfl.import_players()
        names = players[['gsis_id', 'short_name']].rename(columns={
            'gsis_id': 'player_id', 'short_name': 'player_name'
        }).dropna(subset=['player_id']).drop_duplicates('player_id')
        stats = stats.merge(names, on='player_id', how='left')
        stats['player_name'] = stats['player_name'].fillna(stats['player_id'])

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

        merged = eligible_rows(merged, self._seasonal_positions(), self.position)

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
            stats = stats.merge(df, on=['player_id', 'defteam', 'season'], how='left')

        stats = stats.fillna(0)
        for event in ['sacks', 'qb_hits', 'tfls', 'tackles', 'pass_deflections', 'ints', 'run_stops']:
            stats[f'{event}_per_game'] = stats[event] / stats['games']

        players = self.players if self.players is not None else nfl.import_players()
        players = players[['gsis_id', 'short_name']].rename(
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

        n_components = min(5, len(available_features), len(feature_df))
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        print(f"  PCA explained variance: {self.pca.explained_variance_ratio_.sum():.1%}")

        self._available_features = available_features
        self._valid_idx = valid_idx
        self._X_pca = X_pca
        return self

    def _kmeans_labels(self, n_clusters, random_state):
        """Run K-means on the pre-computed PCA embedding for one ensemble iteration."""
        if len(self._valid_idx) < 2:
            return np.zeros(len(self._valid_idx), dtype=int)
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
        if self.config['use_participation'] and (self.participation is not None):
            self._compute_on_off_from_participation()
        else:
            self._compute_on_off_from_pbp_ids()
        self.splits_df = eligible_rows(self.splits_df, self._seasonal_positions(), self.position)
        return self

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
        covered_plays = self.participation[['game_id', 'play_id', 'season']].drop_duplicates()
        team_pbp = pbp.merge(covered_plays, on=['game_id', 'play_id', 'season'])
        if side == 'offense':
            team_pbp['possession_team'] = team_pbp['posteam']
        team_totals = team_pbp.groupby([team_col, 'season']).agg(
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
        if self.position == 'QB':
            pbp = self._qb_attributed_plays()
            player_on = pbp.groupby(['passer_player_id', 'posteam', 'season']).agg(
                n_on=('epa', 'count'), epa_on_sum=('epa', 'sum'),
            ).reset_index().rename(columns={'passer_player_id': 'player_id'})
            league = pbp.groupby('season').agg(
                league_plays=('epa', 'count'), league_epa_sum=('epa', 'sum'))
            own = pbp.groupby(['passer_player_id', 'season']).agg(
                own_plays=('epa', 'count'), own_epa_sum=('epa', 'sum'),
            ).reset_index().rename(columns={'passer_player_id': 'player_id'})
            player_on = player_on.merge(league, on='season').merge(own, on=['player_id', 'season'])
            team_totals = pbp.groupby(['posteam', 'season']).agg(
                team_plays=('epa', 'count'), team_epa_sum=('epa', 'sum'))
            player_on = player_on.merge(team_totals, on=['posteam', 'season'])
            player_on['reference_definition'] = 'other_roster_qbs_league_season'
        elif self.position == 'RB':
            # Rush plays
            rush_on = pbp[pbp['rush'] == 1].groupby(['rusher_player_id', 'posteam', 'season']).agg(
                n_on_rush=('epa', 'count'),
                epa_on_rush=('epa', 'sum'),
            ).reset_index().rename(columns={'rusher_player_id': 'player_id'})

            # Pass plays where RB targeted
            rec_on = pbp[
                (pbp['pass'] == 1) & receiver_position_mask(pbp, 'RB')
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
                (pbp['pass'] == 1) & receiver_position_mask(pbp, 'RB')
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

        if self.position == 'QB':
            player_on['n_off'] = player_on['league_plays'] - player_on['own_plays']
            player_on['epa_off_sum'] = player_on['league_epa_sum'] - player_on['own_epa_sum']
            denominator = player_on['n_off'].replace(0, np.nan)
        else:
            player_on['n_off'] = player_on['team_plays'] - player_on['n_on']
            player_on['epa_off_sum'] = player_on['team_epa_sum'] - player_on['epa_on_sum']
            denominator = player_on['n_off'].clip(lower=1)

        player_on['epa_on'] = player_on['epa_on_sum'] / player_on['n_on']
        player_on['epa_off'] = player_on['epa_off_sum'] / denominator
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

    def _configuration_fingerprint(self):
        settings = {
            'method': self.method, 'position': self.position,
            'config': {key: value for key, value in self.config.items() if key != 'play_filter'},
            'play_filter': joblib.hash((self.config['play_filter'].__code__.co_code,
                                      self.config['play_filter'].__code__.co_consts)),
            'random_state': self.random_state, 'history_decay': self.history_decay,
            'box_alpha': self.box_alpha, 'box_prior_snaps': self.box_prior_snaps,
            'prior_model': self.prior_model, 'box_weighting': self.box_weighting,
            'madden_features': self.madden_features,
            'box_cluster_iterations': self.box_cluster_iterations,
            'season_transition_sd': self.season_transition_sd,
            'rookie_prior_sd': self.rookie_prior_sd,
            'observation_reference_snaps': self.observation_reference_snaps,
            'iterations': self.n_cluster_iterations, 'k_min': self.cluster_k_min,
            'k_max': self.cluster_k_max, 'draws': self.ensemble_draws, 'tune': self.ensemble_tune,
            'version': 1,
            'position_source': hashlib.sha256(Path(seasonal_positions.__file__).read_bytes()).hexdigest(),
            'source': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                      if Path(__file__).exists() else 'unavailable',
        }
        return joblib.hash(settings)

    def _box_feature_rows(self):
        keys = ['player_id', 'team', 'season']
        features = self.player_features.rename(columns={'posteam': 'team'})
        return self.splits_df.merge(
            features[keys + self.config['clustering_features']], on=keys,
            how='left', validate='one_to_one',
        )

    def _prepare_box_history(self, regular, participation):
        """Build one full-season row per eligible player/team, never weekly duplicates."""
        today = date.today()
        current_season = today.year if today.month >= 9 else today.year - 1
        frames = []
        original = self.pbp, self.participation, self.years
        try:
            for season, plays in regular.groupby('season', sort=True):
                final_week = 18 if season >= 2021 else 17
                if season >= current_season or plays['week'].max() < final_week:
                    continue
                self.pbp = plays.copy()
                self.years = [int(season)]
                if participation is not None:
                    self.participation = participation.merge(
                        plays[['game_id', 'play_id', 'season']].drop_duplicates(),
                        on=['game_id', 'play_id', 'season'],
                    )
                self.compute_player_features()
                if self.player_features.empty:
                    continue
                self.player_features['cluster'] = 0
                self.compute_on_off_splits()
                if not self.splits_df.empty:
                    frame = self._box_feature_rows()
                    frame = frame[np.isfinite(frame['raw_plus_minus']) & (frame['n_off'] > 0)]
                    frames.append(frame)
        finally:
            self.pbp, self.participation, self.years = original
        self._box_history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self._box_models = {}

    def _box_model_for_season(self, season):
        if season in self._box_models:
            return self._box_models[season]
        history = self._box_history
        training = history[history['season'] < season].copy() if not history.empty else history.copy()
        columns = ['player_id', 'team', 'season', 'n_on', 'raw_plus_minus'] + self.config['clustering_features']
        if not training.empty:
            training = training.sort_values(['season', 'player_id', 'team']).reset_index(drop=True)
        fingerprint = joblib.hash((self._configuration_fingerprint(), season,
                                   training.reindex(columns=columns),
                                   sklearn.__version__))
        cache_path = OUTPUT_DIR / 'models' / f'box_model_{self.position}_{season}_{fingerprint}.joblib'
        if self.box_cache and cache_path.exists():
            bundle = joblib.load(cache_path)
            if bundle['fingerprint'] == fingerprint:
                self._box_models[season] = bundle
                return bundle
        bundle = {'fingerprint': fingerprint, 'training_seasons': sorted(training['season'].unique().tolist())
                  if not training.empty else [], 'priors': pd.DataFrame(), 'ridge': None}
        if not training.empty:
            numeric = training[self.config['clustering_features']].replace([np.inf, -np.inf], np.nan)
            medians = numeric.median().fillna(0.0)
            scaler = StandardScaler()
            scaled = scaler.fit_transform(numeric.fillna(medians))
            capacity = min(len(training), len(np.unique(scaled, axis=0)))
            lower = self.cluster_k_min or max(2, self.config['n_clusters'] - 2)
            upper = self.cluster_k_max or max(lower, self.config['n_clusters'] + 3)
            lower, upper = min(lower, upper, capacity), min(upper, capacity)
            generator = np.random.default_rng(self.random_state)
            members = []
            member_predictions = []
            for iteration in range(self.box_cluster_iterations):
                count = (int(generator.integers(lower, upper + 1)) if self.box_cluster_iterations > 1
                         else min(self.config['n_clusters'], capacity))
                if self.box_cluster_iterations == 1 and (self.cluster_k_min is not None or self.cluster_k_max is not None):
                    count = int(generator.integers(lower, upper + 1))
                kmeans = KMeans(n_clusters=max(1, count), random_state=self.random_state + iteration, n_init=10)
                labels = kmeans.fit_predict(scaled)
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                design = np.column_stack([scaled, encoder.fit_transform(labels.reshape(-1, 1))])
                ridge = Ridge(alpha=self.box_alpha)
                weights = training['n_on'].to_numpy(dtype=float) if self.box_weighting == 'plays' else None
                ridge.fit(design, training['raw_plus_minus'].to_numpy(), sample_weight=weights)
                members.append({'kmeans': kmeans, 'encoder': encoder, 'ridge': ridge})
                member_predictions.append(ridge.predict(design))
            predictions = np.mean(member_predictions, axis=0)
            kmeans, encoder, ridge = (members[0][key] for key in ('kmeans', 'encoder', 'ridge'))
            historical = training[['player_id', 'season', 'n_on']].copy()
            historical['estimate'] = predictions * historical['n_on'] / (historical['n_on'] + self.box_prior_snaps)
            priors = []
            for (player_id, prior_season), rows in historical.groupby(['player_id', 'season']):
                priors.append({'player_id': player_id, 'prior_season': prior_season,
                               'prior_mean': np.average(rows['estimate'], weights=rows['n_on'])
                               * self.history_decay ** (season - prior_season)})
            if self.prior_model == 'regression':
                summaries = summarize_seasons(training[['player_id', 'season', 'n_on', 'raw_plus_minus']]
                                              .assign(estimate=historical['estimate'].to_numpy()))
                metadata = self._player_metadata()
                features = prior_features(summaries, summaries, metadata)
                intercept, coefficients = fit_prior_regression(features, summaries)
                bundle['prior_regression'] = {'intercept': intercept, 'coefficients': coefficients,
                                              'summaries': summaries}
            bundle.update(scaler=scaler, medians=medians, kmeans=kmeans, encoder=encoder,
                          ridge=ridge, members=members, priors=pd.DataFrame(priors).sort_values('prior_season')
                          .drop_duplicates('player_id', keep='last'))
        self._box_models[season] = bundle
        if self.box_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(dir=cache_path.parent, suffix='.tmp')
            os.close(descriptor)
            try:
                joblib.dump(bundle, temporary)
                os.replace(temporary, cache_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return bundle

    def _score_box(self, bundle):
        frame = self._box_feature_rows()
        frame['prior_mean'] = 0.0
        frame['prior_season'] = np.nan
        if not bundle['priors'].empty:
            frame = frame.drop(columns=['prior_mean', 'prior_season']).merge(
                bundle['priors'], on='player_id', how='left', validate='many_to_one')
            frame['prior_mean'] = frame['prior_mean'].fillna(0.0)
        if self.prior_model == 'zero':
            frame['prior_mean'] = 0.0
            frame['prior_season'] = np.nan
        elif self.prior_model == 'regression' and 'prior_regression' in bundle:
            frame = self._regression_priors(frame, bundle['prior_regression'])
        frame['cluster'] = 0
        frame['box_prediction'] = 0.0
        if bundle['ridge'] is not None:
            numeric = frame[self.config['clustering_features']].replace([np.inf, -np.inf], np.nan)
            scaled = bundle['scaler'].transform(numeric.fillna(bundle['medians']))
            predictions = []
            for member in bundle.get('members', [bundle]):
                labels = member['kmeans'].predict(scaled)
                design = np.column_stack([scaled, member['encoder'].transform(labels.reshape(-1, 1))])
                predictions.append(member['ridge'].predict(design))
            frame['cluster'] = labels
            frame['box_prediction'] = np.mean(predictions, axis=0)
        frame['sample_weight'] = frame['n_on'] / (frame['n_on'] + self.box_prior_snaps)
        frame['box_plus_minus'] = (frame['sample_weight'] * frame['box_prediction']
                                  + (1 - frame['sample_weight']) * frame['prior_mean'])
        frame['bayesian_plus_minus'] = frame['box_plus_minus']
        frame['estimator'] = 'box'
        frame['interval_kind'] = 'not_available'
        frame['pm_lower'] = np.nan
        frame['pm_upper'] = np.nan
        frame['pm_uncertainty'] = np.nan
        frame['training_fingerprint'] = bundle['fingerprint']
        frame['training_seasons'] = json.dumps(bundle['training_seasons'])
        self.splits_df = frame
        self.cluster_ensemble_results = None
        return self

    def _player_metadata(self):
        rosters = self.rosters if self.rosters is not None else pd.DataFrame(columns=['player_id', 'season'])
        return metadata_from_rosters(rosters)

    def _regression_priors(self, frame, regression):
        """Prior from earlier-season estimates, age, draft slot and experience (fit on earlier seasons)."""
        requests = frame[['player_id', 'season']].drop_duplicates().astype({'player_id': 'string'})
        features = prior_features(requests, regression['summaries'], self._player_metadata())
        features['prior_mean'] = regression['intercept'] + features[list(PRIOR_FEATURES)].to_numpy(dtype=float) \
            @ regression['coefficients']
        latest = regression['summaries'].groupby('player_id')['season'].max().rename('prior_season')
        features = features.merge(latest, left_on='player_id', right_index=True, how='left')
        frame = frame.drop(columns=['prior_mean', 'prior_season']).astype({'player_id': 'string'}).merge(
            features[['player_id', 'season', 'prior_mean', 'prior_season']], on=['player_id', 'season'],
            how='left', validate='many_to_one')
        frame['prior_mean'] = frame['prior_mean'].fillna(0.0)
        return frame

    def _attach_player_priors(self, weekly_frames, season):
        """Carry the latest completed-season posterior forward without reusing weekly evidence."""
        self.splits_df['prior_mean'] = 0.0
        self.splits_df['prior_sd'] = self.rookie_prior_sd
        self.splits_df['prior_season'] = np.nan
        if not weekly_frames:
            return self
        history = pd.concat(weekly_frames, ignore_index=True)
        history = history[history['season'] < season].copy()
        if history.empty:
            return self
        final_week = history.groupby('season')['week'].transform('max')
        history = history[history['week'] == final_week].copy()
        history['posterior_sd'] = (history['pm_upper'] - history['pm_lower']) / 3.92
        records = []
        for (player_id, prior_season), rows in history.groupby(['player_id', 'season']):
            weights = rows['n_on'].to_numpy(dtype=float)
            means = rows['bayesian_plus_minus'].to_numpy(dtype=float)
            mean = np.average(means, weights=weights)
            variance = np.average(rows['posterior_sd'].to_numpy() ** 2 + (means - mean) ** 2, weights=weights)
            gap = season - int(prior_season)
            records.append({
                'player_id': player_id, 'prior_season': int(prior_season),
                'prior_mean': mean * self.history_decay ** gap,
                'prior_sd': np.sqrt(variance + gap * self.season_transition_sd ** 2)
            })
        priors = pd.DataFrame(records).sort_values('prior_season').drop_duplicates('player_id', keep='last')
        self.splits_df = self.splits_df.drop(columns=['prior_mean', 'prior_sd', 'prior_season']).merge(
            priors, on='player_id', how='left'
        )
        self.splits_df['prior_mean'] = self.splits_df['prior_mean'].fillna(0.0)
        self.splits_df['prior_sd'] = self.splits_df['prior_sd'].fillna(self.rookie_prior_sd)
        return self

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

        weights = np.sqrt(df['n_on'] / self.observation_reference_snaps)
        prior_mean = df['prior_mean'].to_numpy() if 'prior_mean' in df else np.zeros(n_players)
        prior_sd = df['prior_sd'].to_numpy() if 'prior_sd' in df else np.full(n_players, self.rookie_prior_sd)

        with pm.Model():
            mu_cluster = pm.Normal('mu_cluster', mu=0, sigma=0.05, shape=n_clusters)
            sigma_cluster = pm.HalfNormal('sigma_cluster', sigma=0.05)

            individual = pm.Normal(
                'individual', mu=prior_mean,
                sigma=pm.math.sqrt(prior_sd ** 2 + sigma_cluster ** 2), shape=n_players
            )
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
                nuts_sampler='pymc',
                compile_kwargs={'mode': Mode(linker='py', optimizer='fast_run')},
                return_inferencedata=True,
                target_accept=0.9,
                progressbar=False,  # rich's Windows console renderer crashes on non-UTF8 stdout (e.g. redirected output)
            )

            if sample_posterior_predictive:
                pm.sample_posterior_predictive(
                    trace, extend_inferencedata=True,
                    compile_kwargs={'mode': Mode(linker='py', optimizer='fast_run')}
                )

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

        cols = ['player_name', 'team', 'season', 'week', 'cluster', 'n_on', 'n_off',
            'raw_plus_minus', 'estimator', 'box_plus_minus', 'bayesian_plus_minus',
            'pm_lower', 'pm_upper', 'interval_kind']
        if 'rapm' in self.splits_df.columns:
            cols.append('rapm')

        snapshot = self._latest_snapshot()
        available_cols = [c for c in cols if c in snapshot.columns]
        results = snapshot[available_cols].sort_values('bayesian_plus_minus', ascending=False)
        return results.head(top_n)

    def _latest_snapshot(self):
        """Use the latest season/week for display, retaining all weeks in saved data."""
        if 'week' not in self.splits_df.columns or self.splits_df.empty:
            return self.splits_df
        snapshot = self.splits_df[self.splits_df['season'] == self.splits_df['season'].max()]
        return snapshot[snapshot['week'] == snapshot['week'].max()]

    def plot_results(self, top_n=20):
        """Visualize top players with uncertainty intervals."""
        if self.splits_df is None or len(self.splits_df) == 0:
            print("No results to plot.")
            return None

        snapshot = self._latest_snapshot()
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Top Bayesian estimates with credible intervals
        top = snapshot.nlargest(top_n, 'bayesian_plus_minus')
        ax = axes[0, 0]
        y_pos = np.arange(len(top))
        label = 'Box' if self.method == 'box' else 'Bayesian'
        if self.method == 'box':
            ax.scatter(top['bayesian_plus_minus'], y_pos, color='steelblue')
        else:
            ax.errorbar(
                top['bayesian_plus_minus'], y_pos,
                xerr=[top['bayesian_plus_minus'] - top['pm_lower'],
                      top['pm_upper'] - top['bayesian_plus_minus']],
                fmt='o', capsize=3, color='steelblue'
            )
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{r.player_name} ({r.team})" for _, r in top.iterrows()], fontsize=8)
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel(f'{label} Plus-Minus (EPA/play)')
        ax.set_title(f'Top {top_n} {self.position}s - {label} Estimates'
                 + (' (no intervals)' if self.method == 'box' else ' with 95% CI'))
        ax.invert_yaxis()

        # 2. Raw vs Bayesian (shrinkage)
        ax = axes[0, 1]
        ax.scatter(
            snapshot['raw_plus_minus'],
            snapshot['bayesian_plus_minus'],
            c=snapshot['cluster'], cmap='tab10', alpha=0.6, edgecolors='black'
        )
        lim = max(abs(snapshot['raw_plus_minus'].max()), abs(snapshot['raw_plus_minus'].min()))
        ax.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5, label='No shrinkage')
        ax.set_xlabel('Raw Plus-Minus')
        ax.set_ylabel(f'{label} Plus-Minus')
        ax.set_title(f'Shrinkage Effect: Raw vs {label}')
        ax.legend()

        # 3. Cluster means
        ax = axes[1, 0]
        cluster_summary = snapshot.groupby('cluster').agg({
            'bayesian_plus_minus': 'mean',
            'raw_plus_minus': 'mean',
            'player_name': 'count'
        }).rename(columns={'player_name': 'n_players'})

        x = np.arange(len(cluster_summary))
        width = 0.35
        ax.bar(x - width/2, cluster_summary['raw_plus_minus'], width, label='Raw', alpha=0.7)
        ax.bar(x + width/2, cluster_summary['bayesian_plus_minus'], width, label=label, alpha=0.7)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Mean Plus-Minus')
        ax.set_title('Cluster-Level Estimates')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i}' for i in cluster_summary.index])
        ax.legend()

        # 4. Bayesian vs RAPM
        ax = axes[1, 1]
        if 'rapm' in snapshot.columns and snapshot['rapm'].notna().any():
            valid = snapshot.dropna(subset=['rapm'])
            ax.scatter(valid['rapm'], valid['bayesian_plus_minus'], alpha=0.5, edgecolors='black')
            lim = max(abs(valid['rapm'].max()), abs(valid['rapm'].min()))
            ax.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5)
            ax.set_xlabel('Ridge RAPM')
            ax.set_ylabel(f'{label} Plus-Minus')
            ax.set_title(f'{label} vs Ridge RAPM')
            corr = valid['rapm'].corr(valid['bayesian_plus_minus'])
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'RAPM not available\nfor this position',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label} vs Ridge RAPM')

        if 'week' in snapshot.columns:
            fig.suptitle(f"{self.position}: {int(snapshot['season'].iloc[0])} through week {int(snapshot['week'].iloc[0])}")
        plt.tight_layout()
        output_path = _output_dir() / f'plus_minus_plot_{self.position}.png'
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

        snapshot = self._latest_snapshot()
        for cluster in sorted(snapshot['cluster'].unique()):
            cluster_players = snapshot[snapshot['cluster'] == cluster]
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
        """Atomically commit both tables, then refresh the public CSV exports."""
        out_dir = _output_dir()
        tables = {'splits.csv': self.splits_df.to_csv(index=False)}
        if self.cluster_ensemble_results is not None:
            tables['iterations.csv'] = self.cluster_ensemble_results.to_csv(index=False)
        checkpoint = out_dir / f'plus_minus_checkpoint_{self.position}.zip'
        descriptor, temporary = tempfile.mkstemp(dir=out_dir, suffix='.tmp')
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr('manifest.json', json.dumps({
                        'method': self.method, 'configuration': self._configuration_fingerprint(),
                        'completion': 'weekly_scores' if self.method == 'box' else 'ensemble_iterations',
                    }))
                    for name, content in tables.items():
                        archive.writestr(name, content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, checkpoint)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self._refresh_result_exports(tables)
        return self

    def _refresh_result_exports(self, tables):
        """Refresh individually atomic CSV views of the authoritative checkpoint."""
        out_dir = _output_dir()
        for name, content in tables.items():
            prefix = 'plus_minus_weekly' if name == 'splits.csv' else 'plus_minus_weekly_iterations'
            destination = out_dir / f'{prefix}_{self.position}.csv'
            descriptor, temporary = tempfile.mkstemp(dir=out_dir, suffix='.tmp')
            try:
                with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        if 'iterations.csv' not in tables:
            stale = out_dir / f'plus_minus_weekly_iterations_{self.position}.csv'
            stale.unlink(missing_ok=True)

    def _load_saved_results(self):
        """Read a committed pair; legacy CSVs are accepted only after validation."""
        out_dir = OUTPUT_DIR
        checkpoint = out_dir / f'plus_minus_checkpoint_{self.position}.zip'
        if checkpoint.exists():
            with zipfile.ZipFile(checkpoint) as archive:
                if 'manifest.json' in archive.namelist():
                    manifest = json.loads(archive.read('manifest.json'))
                    if (manifest.get('method') != self.method
                            or manifest.get('configuration') != self._configuration_fingerprint()):
                        return pd.DataFrame(), pd.DataFrame(), set()
                elif self.method == 'box':
                    return pd.DataFrame(), pd.DataFrame(), set()
                tables = {name: archive.read(name).decode('utf-8')
                          for name in ('splits.csv', 'iterations.csv') if name in archive.namelist()}
                splits = pd.read_csv(io.StringIO(tables['splits.csv']))
                iterations = (pd.read_csv(io.StringIO(tables['iterations.csv']))
                              if 'iterations.csv' in tables else pd.DataFrame())
            self._refresh_result_exports(tables)
        else:
            splits_path = out_dir / f'plus_minus_weekly_{self.position}.csv'
            iterations_path = out_dir / f'plus_minus_weekly_iterations_{self.position}.csv'
            splits = pd.read_csv(splits_path) if splits_path.exists() else pd.DataFrame()
            iterations = pd.read_csv(iterations_path) if iterations_path.exists() else pd.DataFrame()
            if self.method == 'box' or ('estimator' in splits and not splits['estimator'].eq('bayesian').all()):
                return pd.DataFrame(), pd.DataFrame(), set()
        keys = ['player_id', 'team', 'season', 'week']
        for frame, extra in ((splits, []), (iterations, ['iteration'])):
            if not frame.empty:
                if not set(keys + extra).issubset(frame.columns):
                    raise ValueError('Saved weekly results are missing identity columns.')
                frame.drop_duplicates(keys + extra, keep='last', inplace=True)
        return splits, iterations, self._completed_week_keys(splits, iterations)

    def _completed_week_keys(self, splits, iterations):
        """Require all configured fits per player and matching finite averages."""
        keys = ['player_id', 'team', 'season', 'week']
        metrics = ['bayesian_plus_minus', 'pm_lower', 'pm_upper']
        completed = set()
        if self.method == 'box':
            required = ['box_plus_minus', 'box_prediction', 'prior_mean', 'sample_weight',
                        'estimator', 'week_fingerprint', 'training_fingerprint']
            if splits.empty or not set(keys + required).issubset(splits.columns):
                return completed
            for (season, week), rows in splits.groupby(['season', 'week']):
                values = rows[required[:4]].to_numpy(dtype=float)
                expected = rows['sample_weight'] * rows['box_prediction'] + (1 - rows['sample_weight']) * rows['prior_mean']
                if (rows['estimator'].eq('box').all() and np.isfinite(values).all()
                        and not rows.duplicated(keys).any()
                        and rows['week_fingerprint'].notna().all() and rows['week_fingerprint'].nunique() == 1
                        and rows['training_fingerprint'].notna().all() and rows['training_fingerprint'].nunique() == 1
                        and np.allclose(expected, rows['box_plus_minus'])
                        and np.allclose(rows['box_plus_minus'], rows['bayesian_plus_minus'])):
                    completed.add((int(season), int(week)))
            return completed
        if splits.empty or iterations.empty:
            return completed
        if not set(metrics).issubset(splits.columns) or not set(metrics + ['iteration']).issubset(iterations.columns):
            return completed
        for (season, week), averaged in splits.groupby(['season', 'week']):
            raw = iterations[(iterations['season'] == season) & (iterations['week'] == week)]
            if raw.empty or not np.isfinite(averaged[metrics].to_numpy(dtype=float)).all():
                continue
            if not np.isfinite(raw[metrics].to_numpy(dtype=float)).all():
                continue
            grouped = raw.groupby(keys)
            if not grouped['iteration'].apply(
                lambda values: set(values) == set(range(self.n_cluster_iterations))
                and len(values) == self.n_cluster_iterations
            ).all():
                continue
            expected = grouped[metrics].mean().sort_index()
            actual = averaged.set_index(keys)[metrics].sort_index()
            if expected.index.equals(actual.index) and np.allclose(expected, actual, rtol=1e-7, atol=1e-10):
                completed.add((int(season), int(week)))
        return completed

    def _completed_season_history(self, frames):
        """Only known final regular-season snapshots may seed incremental priors."""
        today = date.today()
        current_season = today.year if today.month >= 9 else today.year - 1
        history = []
        for frame in frames:
            if frame.empty:
                continue
            final_week = np.where(frame['season'] >= 2021, 18, 17)
            history.append(frame[(frame['season'] < current_season) & (frame['week'] == final_week)])
        return history

    def run_full_pipeline(self, include_rapm=False, incremental=False):
        """Fit season-to-date estimates with completed-season player priors.

        A week W estimate includes W, so use it only after that week's games.
        Minimum sample thresholds apply cumulatively within each season.
        Box models train only on earlier completed seasons and score without MCMC.
        Incremental mode resumes method-matched weeks and commits each new week.
        """
        self.load_data()
        full_pbp = self.pbp
        full_participation = self.participation
        requested_years = self.years
        regular = full_pbp[full_pbp['season_type'] == 'REG']
        cutoffs = regular[['season', 'week']].dropna().drop_duplicates().sort_values(['season', 'week'])
        season_max_week = {int(s): int(grp['week'].max()) for s, grp in regular.groupby('season')}
        weekly_frames = []
        iteration_frames = []
        completed = set()
        saved_week_fingerprints = {}
        if self.method == 'box':
            self._prepare_box_history(regular, full_participation)
        if incremental:
            saved, saved_iterations, completed = self._load_saved_results()
            if not saved.empty:
                weekly_frames.append(saved)
                if 'week_fingerprint' in saved.columns:
                    for (s, w), grp in saved.groupby(['season', 'week']):
                        fps = grp['week_fingerprint'].dropna().unique()
                        if len(fps) == 1:
                            saved_week_fingerprints[(int(s), int(w))] = fps[0]
            if not saved_iterations.empty:
                iteration_frames.append(saved_iterations)
            cutoffs = cutoffs[cutoffs['season'].isin(requested_years)]
        try:
            for cutoff in cutoffs.itertuples(index=False):
                season, week = int(cutoff.season), int(cutoff.week)
                if self.method == 'bayesian' and (season, week) in completed:
                    continue
                if week == 1 or week % 8 == 0 or week == season_max_week.get(season):
                    print(f"  {self.position}: {season} W{week:02d}...", flush=True)
                self.years = [season]
                self.pbp = regular[(regular['season'] == season) & (regular['week'] <= week)].copy()
                if full_participation is not None:
                    keys = self.pbp[['game_id', 'play_id', 'season']].drop_duplicates()
                    self.participation = full_participation.merge(keys, on=['game_id', 'play_id', 'season'])
                if self.method == 'box':
                    bundle = self._box_model_for_season(season)
                    week_fingerprint = joblib.hash((bundle['fingerprint'], self.pbp, self.participation))
                    if (season, week) in completed and saved_week_fingerprints.get((season, week)) == week_fingerprint:
                        continue
                self._X_pca = None
                self.cluster_ensemble_results = None
                self.compute_player_features()
                if self.player_features.empty:
                    print("  No players meet the cumulative minimum sample; skipping week.")
                    continue
                self.player_features['cluster'] = 0
                self.compute_on_off_splits()
                if self.splits_df.empty:
                    print("  No eligible on/off splits; skipping week.")
                    continue
                if self.method == 'box':
                    self._score_box(bundle)
                    self.splits_df['week_fingerprint'] = week_fingerprint
                elif incremental:
                    valid_history = [
                        frame[frame[['season', 'week']].apply(tuple, axis=1).isin(completed)]
                        for frame in weekly_frames
                    ]
                    self._attach_player_priors(self._completed_season_history(valid_history), season)
                else:
                    self._attach_player_priors(weekly_frames, season)
                if self.method == 'bayesian':
                    self.run_clustering_ensemble()
                    self.splits_df['estimator'] = 'bayesian'
                    self.splits_df['interval_kind'] = 'averaged_posterior_endpoints'
                if include_rapm:
                    self.fit_rapm_baseline()
                self.splits_df['week'] = week
                if incremental:
                    if self.method == 'bayesian' and self.cluster_ensemble_results is None:
                        raise ValueError('Incremental saves require complete ensemble iterations.')
                    candidate = (self.cluster_ensemble_results.assign(week=week)
                                 if self.cluster_ensemble_results is not None else pd.DataFrame())
                    if (season, week) not in self._completed_week_keys(self.splits_df, candidate):
                        raise ValueError('Incomplete or inconsistent weekly ensemble results.')
                    weekly_frames = [frame[~((frame['season'] == season) & (frame['week'] == week))]
                                     for frame in weekly_frames]
                    iteration_frames = [frame[~((frame['season'] == season) & (frame['week'] == week))]
                                        for frame in iteration_frames]
                weekly_frames.append(self.splits_df.copy())
                if self.cluster_ensemble_results is not None:
                    iterations = self.cluster_ensemble_results.copy()
                    iterations['week'] = week
                    iteration_frames.append(iterations)
                if incremental:
                    self.splits_df = pd.concat(weekly_frames, ignore_index=True)
                    self.cluster_ensemble_results = (pd.concat(iteration_frames, ignore_index=True)
                                                     if iteration_frames else None)
                    self.save_results()
                    completed.add((season, week))
                    if self.method == 'box':
                        saved_week_fingerprints[(season, week)] = week_fingerprint
        finally:
            self.pbp = full_pbp
            self.participation = full_participation
            self.years = requested_years
        self.splits_df = pd.concat(weekly_frames, ignore_index=True) if weekly_frames else pd.DataFrame(
            columns=['player_id', 'player_name', 'team', 'season', 'week',
                     'bayesian_plus_minus', 'pm_lower', 'pm_upper', 'estimator', 'interval_kind']
                    + (['box_plus_minus', 'box_prediction', 'sample_weight', 'prior_mean',
                        'prior_season', 'training_fingerprint', 'week_fingerprint']
                       if self.method == 'box' else [])
        )
        self.cluster_ensemble_results = pd.concat(iteration_frames, ignore_index=True) if iteration_frames else None
        if not incremental:
            self.save_results()

        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}")

        self.print_cluster_examples()

        print(f"\n{'='*60}")
        print(f"TOP 15 PLAYERS - {self.position} ({self.method.title()} Plus-Minus)")
        print(f"{'='*60}")
        results = self.get_results(15)
        if len(results) > 0:
            print(results.to_string(index=False))
        else:
            print("No results available.")

        return self


# Former name, kept for existing callers.
NFLBayesianPlusMinus = PlayerPlusMinus


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    for position in POSITION_CONFIG:
        model = PlayerPlusMinus.calibrated(
            position,
            years=list(range(2016, date.today().year + 1)),
            random_state=42,
        )
        model.run_full_pipeline(incremental=True)

