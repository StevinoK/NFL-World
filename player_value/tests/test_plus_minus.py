import json
"""Focused offline regression tests for weekly NFL plus-minus estimates."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import player_value.plus_minus as plus_minus
class NFLPlusMinusTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch('builtins.print'))
        for name in ('import_pbp_data', 'import_players', 'import_seasonal_rosters'):
            self.enterContext(mock.patch.object(
                plus_minus.nfl, name, side_effect=AssertionError('Unexpected data download')
            ))
        self.enterContext(mock.patch.object(
            plus_minus.pm, 'sample', side_effect=AssertionError('Unexpected Bayesian sampling')
        ))
        self.enterContext(mock.patch.object(
            plus_minus, 'load_participation_data',
            side_effect=AssertionError('Unexpected participation download')
        ))

    def make_model(self, position='QB', min_snaps=1):
        config = dict(plus_minus.POSITION_CONFIG[position], min_snaps=min_snaps)
        model = plus_minus.PlayerPlusMinus(
            position, [2024, 2025], config=config, n_cluster_iterations=2,
            cluster_k_min=2, cluster_k_max=2,
        )
        model.rosters = self.skill_rosters(position)
        return model

    def skill_rosters(self, position='QB'):
        return pd.DataFrame([
            (f'qb{number}', season, position)
            for season in (2023, 2024, 2025, 2026) for number in range(4)
        ], columns=['player_id', 'season', 'position'])

    def qb_pbp(self):
        rows = []
        for season in (2024, 2025):
            for week, attempts in ((1, 1), (2, 2)):
                for player_number in range(4):
                    for attempt in range(attempts):
                        rows.append({
                            'season': season, 'week': week, 'season_type': 'REG',
                            'game_id': f'{season}_{week}_AAA_BBB',
                            'play_id': 10 * player_number + attempt,
                            'posteam': 'AAA', 'defteam': 'BBB', 'pass': 1, 'rush': 0,
                            'passer_player_id': f'qb{player_number}',
                            'passer_player_name': f'Quarterback {player_number}',
                            'epa': 0.1 * (player_number + week) + (season - 2024),
                            'passing_yards': 5 + player_number + week,
                            'complete_pass': int(player_number % 2 == 0),
                            'touchdown': int(player_number == 3),
                            'interception': int(player_number == 1),
                            'sack': 0, 'air_yards': player_number + 2,
                            'qb_scramble': 0, 'cpoe': float(player_number),
                        })
        return pd.DataFrame(rows)

    def test_offensive_features_consolidate_name_variants_and_missing_names(self):
        for position in ('QB', 'RB', 'WR', 'TE'):
            for names in ([None, 'Original Name', 'Variant Name'], [None, None, None]):
                with self.subTest(position=position, names=names):
                    model = self.make_model(position, min_snaps=3)
                    pbp = self.qb_pbp().query('season == 2024 and passer_player_id == "qb0"').copy()
                    pbp['passer_player_name'] = names
                    pbp['rusher_player_id'] = pbp['passer_player_id']
                    pbp['rusher_player_name'] = names
                    pbp['receiver_player_id'] = pbp['passer_player_id']
                    pbp['receiver_player_name'] = names
                    pbp['receiver_position'] = position
                    pbp['rush'] = int(position == 'RB')
                    pbp['complete_pass'] = [1, 0, 1]
                    for column in ('passing_yards', 'rushing_yards', 'receiving_yards', 'yards_gained'):
                        pbp[column] = [0, 10, 20]
                    pbp['yards_after_catch'] = [0, 0, 4]
                    pbp['first_down'] = [0, 0, 1]
                    model.pbp = pbp
                    model.compute_player_features()
                    self.assertEqual(len(model.player_features), 1)
                    row = model.player_features.iloc[0]
                    self.assertEqual(row['player_id'], 'qb0')
                    self.assertEqual(row['games'], 2)
                    if names[1] is None:
                        self.assertTrue(pd.isna(row['player_name']))
                    else:
                        self.assertEqual(row['player_name'], 'Original Name')
                    count = 'attempts' if position == 'QB' else 'carries' if position == 'RB' else 'targets'
                    rate = ('yards_per_attempt' if position == 'QB' else
                            'yards_per_carry' if position == 'RB' else 'yards_per_target')
                    self.assertEqual(row[count], 3)
                    self.assertEqual(row[rate], 10)
                    if position == 'QB':
                        self.assertAlmostEqual(row['completion_pct'], 2 / 3)
                        self.assertEqual(row['pass_attempts_per_game'], 1.5)
                    elif position == 'RB':
                        self.assertEqual(row['rush_attempts_per_game'], 1.5)
                        self.assertEqual(row['targets'], 3)
                        self.assertAlmostEqual(row['catch_rate'], 2 / 3)
                        self.assertAlmostEqual(row['stuff_rate'], 1 / 3)
                    else:
                        self.assertEqual(row['targets_per_game'], 1.5)
                        self.assertAlmostEqual(row['catch_rate'], 2 / 3)
                    model.splits_df = model.player_features[['player_id', 'posteam', 'season']].rename(
                        columns={'posteam': 'team'}
                    )
                    joined = model._box_feature_rows()
                    self.assertEqual(len(joined), 1)
                    self.assertFalse(joined.duplicated(['player_id', 'team', 'season']).any())
                    self.assertEqual(joined.iloc[0][rate], 10)

    def test_box_feature_join_still_rejects_duplicate_stable_keys(self):
        model = self.make_model()
        model.pbp = self.qb_pbp()
        model.compute_player_features()
        model.splits_df = model.player_features[['player_id', 'posteam', 'season']].rename(
            columns={'posteam': 'team'}
        )
        model.player_features = pd.concat([model.player_features, model.player_features.iloc[:1]])
        with self.assertRaises(pd.errors.MergeError):
            model._box_feature_rows()

    def receiver_pbp(self):
        pbp = self.qb_pbp()
        pbp['receiver_player_id'] = pbp['passer_player_id'].replace({'qb0': '00-0036322'})
        pbp['receiver_player_name'] = pbp['passer_player_name'].replace({'Quarterback 0': 'J.Jefferson'})
        pbp['rusher_player_id'] = pbp['receiver_player_id']
        pbp['rusher_player_name'] = pbp['receiver_player_name']
        pbp['receiver_position'] = 'RB'
        pbp['rush'] = 1
        for column in ('rushing_yards', 'receiving_yards', 'yards_gained'):
            pbp[column] = pbp['passing_yards']
        pbp['yards_after_catch'] = 2
        pbp['first_down'] = 0
        return pbp

    def test_jefferson_jet_sweeps_never_enter_rb_features_or_splits(self):
        model = self.make_model('RB')
        model.rosters = self.skill_rosters('RB')
        model.rosters['player_id'] = model.rosters['player_id'].replace({'qb0': '00-0036322'})
        model.rosters.loc[model.rosters['player_id'].eq('00-0036322'), 'position'] = 'WR'
        model.pbp = self.receiver_pbp()
        model.compute_player_features()
        self.assertNotIn('00-0036322', model.player_features['player_id'].tolist())
        self.assertEqual(len(model.player_features), 6)
        model.player_features['cluster'] = 0
        model.compute_on_off_splits()
        self.assertNotIn('00-0036322', model.splits_df['player_id'].tolist())
        model.position = 'WR'
        model.config = dict(plus_minus.POSITION_CONFIG['WR'], min_snaps=1)
        model.compute_player_features()
        self.assertEqual(set(model.player_features['player_id']), {'00-0036322'})
        self.assertEqual(set(model.player_features['season']), {2024, 2025})

    def test_receiver_classification_tracks_season_not_career_or_future_roster(self):
        model = self.make_model('RB')
        model.rosters = pd.DataFrame([
            ('00-0036322', 2024, 'WR'), ('00-0036322', 2025, 'HB'),
            ('qb1', 2024, 'RB'), ('qb1', 2025, 'RB'), ('qb2', 2026, 'RB'),
        ], columns=['player_id', 'season', 'position'])
        model.pbp = self.receiver_pbp()
        model.compute_player_features()
        jefferson = model.player_features.query("player_id == '00-0036322'")
        self.assertEqual(jefferson['season'].tolist(), [2025])
        self.assertNotIn('qb2', model.player_features['player_id'].tolist())
        self.assertEqual(set(model.pbp.query("season == 2024 and receiver_player_id == '00-0036322'")['receiver_position']), {'WR'})
        self.assertEqual(set(model.pbp.query("season == 2025 and receiver_player_id == '00-0036322'")['receiver_position']), {'RB'})

    def test_missing_roster_schema_is_not_an_unfiltered_feature_fallback(self):
        model = self.make_model()
        model.pbp = self.qb_pbp()
        model.rosters = pd.DataFrame({'cached_rosters': [True]})
        with self.assertRaisesRegex(ValueError, 'required columns'):
            model.compute_player_features()

    def test_skill_load_downloads_seasonal_rosters_and_persists_lookup(self):
        model = self.make_model('WR')
        pbp = self.receiver_pbp()
        rosters = self.skill_rosters('WR')
        rosters['player_id'] = rosters['player_id'].replace({'qb0': '00-0036322'})
        with mock.patch.object(plus_minus.nfl, 'import_pbp_data', return_value=pbp), \
                mock.patch.object(plus_minus.nfl, 'import_seasonal_rosters', return_value=rosters) as load, \
                mock.patch.object(plus_minus, 'save_lookup') as save:
            model.load_data()
        load.assert_called_once_with([2024, 2025])
        save.assert_called_once()
        self.assertTrue(model.pbp['receiver_position'].eq('WR').all())

    def test_position_source_change_invalidates_box_cache_and_checkpoints(self):
        model = self.make_model()
        before = model._configuration_fingerprint()
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'seasonal_positions.py'
            source.write_text('changed seasonal eligibility', encoding='ascii')
            with mock.patch.object(plus_minus.seasonal_positions, '__file__', str(source)):
                self.assertNotEqual(before, model._configuration_fingerprint())

    def run_weekly(self, pbp, min_snaps=1, position='QB'):
        model = self.make_model(position=position, min_snaps=min_snaps)
        participation = pbp[['game_id', 'play_id', 'season']].drop_duplicates().copy()
        rosters = self.skill_rosters(position)
        players = pd.DataFrame({'cached_players': [True]})
        feature_snapshots = {}
        fit_count = 0

        def load_data():
            model.pbp = pbp
            model.participation = participation
            model.rosters = rosters
            model.players = players
            return model

        def fake_fit(frame, draws, tune, cores, sample_posterior_predictive):
            nonlocal fit_count
            self.assertFalse(sample_posterior_predictive)
            self.assertEqual(draws, model.ensemble_draws)
            self.assertEqual(tune, model.ensemble_tune)
            season = int(model.pbp['season'].iloc[0])
            week = int(model.pbp['week'].max())
            self.assertEqual(
                set(model.participation.itertuples(index=False, name=None)),
                set(model.pbp[['game_id', 'play_id', 'season']].itertuples(index=False, name=None)),
            )
            feature_snapshots[(season, week)] = model.player_features.copy(deep=True)
            fitted = frame.copy(deep=True)
            fitted['bayesian_plus_minus'] = fitted['raw_plus_minus'] / 2 + 0.2 * (fit_count % 2)
            fitted['pm_lower'] = fitted['bayesian_plus_minus'] - 0.3
            fitted['pm_upper'] = fitted['bayesian_plus_minus'] + 0.3
            fit_count += 1
            return None, fitted

        with mock.patch.object(model, 'load_data', side_effect=load_data) as load, \
                mock.patch.object(model, 'save_results', return_value=model) as save, \
                mock.patch.object(model, '_fit_hierarchical_bayesian', side_effect=fake_fit) as fit, \
                mock.patch.object(model, 'fit_rapm_baseline') as rapm:
            self.assertIs(model.run_full_pipeline(), model)
            load.assert_called_once_with()
            save.assert_called_once_with()
            rapm.assert_not_called()
            self.assertEqual(fit.call_count, 2 * len(feature_snapshots))
        self.assertIs(model.pbp, pbp)
        self.assertIs(model.participation, participation)
        self.assertIs(model.rosters, rosters)
        self.assertIs(model.players, players)
        self.assertEqual(model.years, [2024, 2025])
        return model, feature_snapshots

    def posterior_history(self, rows):
        frame = pd.DataFrame(rows, columns=[
            'player_id', 'team', 'season', 'week', 'n_on',
            'bayesian_plus_minus', 'posterior_sd',
        ])
        frame['pm_lower'] = frame['bayesian_plus_minus'] - 1.96 * frame['posterior_sd']
        frame['pm_upper'] = frame['bayesian_plus_minus'] + 1.96 * frame['posterior_sd']
        return frame.drop(columns='posterior_sd')

    def test_priors_exclude_current_and_future_seasons(self):
        model = self.make_model()
        model.splits_df = pd.DataFrame({'player_id': ['veteran', 'current_only', 'future_only']})
        history = self.posterior_history([
            ('veteran', 'AAA', 2024, 18, 100, 0.25, 0.04),
            ('veteran', 'AAA', 2025, 18, 1000, 99.0, 2.0),
            ('veteran', 'AAA', 2026, 18, 1000, -99.0, 2.0),
            ('current_only', 'AAA', 2025, 18, 100, 1.0, 0.04),
            ('future_only', 'AAA', 2026, 18, 100, 1.0, 0.04),
        ])
        original = history.copy(deep=True)
        self.assertIs(model._attach_player_priors([history], 2025), model)
        priors = model.splits_df.set_index('player_id')
        self.assertAlmostEqual(priors.loc['veteran', 'prior_mean'], 0.25 * 0.8)
        self.assertAlmostEqual(priors.loc['veteran', 'prior_sd'], np.sqrt(0.04 ** 2 + 0.06 ** 2))
        self.assertEqual(priors.loc['veteran', 'prior_season'], 2024)
        for player_id in ('current_only', 'future_only'):
            with self.subTest(player_id=player_id):
                self.assertEqual(priors.loc[player_id, 'prior_mean'], 0.0)
                self.assertEqual(priors.loc[player_id, 'prior_sd'], model.rookie_prior_sd)
                self.assertTrue(pd.isna(priors.loc[player_id, 'prior_season']))
        pd.testing.assert_frame_equal(history, original)

    def test_priors_use_only_season_final_week_not_cumulative_weekly_duplicates(self):
        model = self.make_model()
        model.splits_df = pd.DataFrame({'player_id': ['veteran', 'early_only']})
        final = self.posterior_history([('veteran', 'AAA', 2024, 18, 100, 0.5, 0.03)])
        earlier = self.posterior_history([
            ('veteran', 'AAA', 2024, 1, 1000, 10.0, 1.0),
            ('veteran', 'AAA', 2024, 17, 1000, -10.0, 1.0),
            ('early_only', 'AAA', 2024, 17, 100, 1.0, 0.03),
        ])
        model._attach_player_priors([final, earlier, earlier.copy()], 2025)
        priors = model.splits_df.set_index('player_id')
        self.assertAlmostEqual(priors.loc['veteran', 'prior_mean'], 0.5 * 0.8)
        self.assertAlmostEqual(priors.loc['veteran', 'prior_sd'], np.sqrt(0.03 ** 2 + 0.06 ** 2))
        self.assertEqual(priors.loc['veteran', 'prior_season'], 2024)
        self.assertEqual(priors.loc['early_only', 'prior_mean'], 0.0)
        self.assertEqual(priors.loc['early_only', 'prior_sd'], model.rookie_prior_sd)
        self.assertTrue(pd.isna(priors.loc['early_only', 'prior_season']))

    def test_priors_use_latest_completed_posterior_per_player_without_recursive_doublecount(self):
        model = self.make_model()
        model.splits_df = pd.DataFrame({'player_id': ['returning', 'gap_player']})
        latest = self.posterior_history([('returning', 'BBB', 2024, 18, 10, 0.2, 0.02)])
        older = self.posterior_history([
            ('returning', 'AAA', 2023, 18, 10000, 8.0, 1.0),
            ('gap_player', 'AAA', 2023, 18, 100, -0.4, 0.05),
        ])
        model._attach_player_priors([latest, older], 2025)
        priors = model.splits_df.set_index('player_id')
        self.assertAlmostEqual(priors.loc['returning', 'prior_mean'], 0.2 * 0.8)
        self.assertAlmostEqual(priors.loc['returning', 'prior_sd'], np.sqrt(0.02 ** 2 + 0.06 ** 2))
        self.assertEqual(priors.loc['returning', 'prior_season'], 2024)
        self.assertAlmostEqual(priors.loc['gap_player', 'prior_mean'], -0.4 * 0.8 ** 2)
        self.assertAlmostEqual(priors.loc['gap_player', 'prior_sd'], np.sqrt(0.05 ** 2 + 2 * 0.06 ** 2))
        self.assertEqual(priors.loc['gap_player', 'prior_season'], 2023)

    def test_history_decay_shrinks_mean_and_year_gap_increases_transition_variance(self):
        history = self.posterior_history([('veteran', 'AAA', 2024, 18, 100, 0.5, 0.04)])
        deviations = {}
        for decay in (0.5, 0.8, 1.0):
            for gap in (1, 3):
                with self.subTest(decay=decay, gap=gap):
                    model = plus_minus.PlayerPlusMinus(
                        'QB', [2024 + gap], history_decay=decay, season_transition_sd=0.07,
                    )
                    model.splits_df = pd.DataFrame({'player_id': ['veteran']})
                    model._attach_player_priors([history], 2024 + gap)
                    prior = model.splits_df.iloc[0]
                    self.assertAlmostEqual(prior['prior_mean'], 0.5 * decay ** gap)
                    self.assertAlmostEqual(prior['prior_sd'] ** 2, 0.04 ** 2 + gap * 0.07 ** 2)
                    self.assertGreater(prior['prior_sd'], 0.04)
                    deviations[(decay, gap)] = prior['prior_sd']
            self.assertGreater(deviations[(decay, 3)], deviations[(decay, 1)])

    def test_traded_player_prior_merges_by_id_with_snap_weighted_mixture_uncertainty(self):
        model = self.make_model()
        model.splits_df = pd.DataFrame({
            'player_id': ['traded', 'traded'], 'team': ['CCC', 'DDD'], 'n_on': [5, 15],
        })
        history = self.posterior_history([
            ('traded', 'AAA', 2024, 18, 25, 0.2, 0.02),
            ('traded', 'BBB', 2024, 18, 75, 0.6, 0.04),
        ])
        model._attach_player_priors([history], 2025)
        mean = 0.25 * 0.2 + 0.75 * 0.6
        variance = 0.25 * (0.02 ** 2 + (0.2 - mean) ** 2) + 0.75 * (0.04 ** 2 + (0.6 - mean) ** 2)
        self.assertEqual(len(model.splits_df), 2)
        self.assertEqual(model.splits_df['team'].tolist(), ['CCC', 'DDD'])
        np.testing.assert_array_equal(model.splits_df['n_on'], [5, 15])
        np.testing.assert_allclose(model.splits_df['prior_mean'], mean * 0.8)
        np.testing.assert_allclose(model.splits_df['prior_sd'], np.sqrt(variance + 0.06 ** 2))
        np.testing.assert_array_equal(model.splits_df['prior_season'], [2024, 2024])

    def test_missing_players_receive_rookie_prior_with_or_without_completed_history(self):
        completed = self.posterior_history([('other', 'AAA', 2024, 18, 100, 0.5, 0.02)])
        current = completed.assign(season=2025)
        for history in ([], [current], [completed]):
            with self.subTest(history_seasons=[frame['season'].tolist() for frame in history]):
                model = plus_minus.PlayerPlusMinus('QB', [2025], rookie_prior_sd=0.17)
                model.splits_df = pd.DataFrame({
                    'player_id': ['rookie'], 'prior_mean': [99.0],
                    'prior_sd': [99.0], 'prior_season': [2023],
                })
                model._attach_player_priors(history, 2025)
                prior = model.splits_df.iloc[0]
                self.assertEqual(prior['prior_mean'], 0.0)
                self.assertEqual(prior['prior_sd'], 0.17)
                self.assertTrue(pd.isna(prior['prior_season']))

    def test_init_uses_lower_weekly_defaults_without_mutating_position_config(self):
        thresholds = {'QB': 10, 'RB': 5, 'WR': 3, 'TE': 3, 'OL': 20, 'DL': 15, 'LB': 15, 'DB': 15}
        for position, threshold in thresholds.items():
            with self.subTest(position=position):
                original = plus_minus.POSITION_CONFIG[position].copy()
                model = plus_minus.PlayerPlusMinus(position.lower(), [2025])
                self.assertEqual(model.position, position)
                self.assertEqual(model.config['min_snaps'], threshold)
                self.assertLess(threshold, original['min_snaps'])
                self.assertEqual(model.config, dict(original, min_snaps=threshold))
                self.assertIsNot(model.config, plus_minus.POSITION_CONFIG[position])
                self.assertEqual(plus_minus.POSITION_CONFIG[position], original)

    def test_init_preserves_custom_config_and_weekly_override_does_not_mutate_it(self):
        config = dict(plus_minus.POSITION_CONFIG['QB'], min_snaps=37, feature_desc='Custom QB')
        original = config.copy()
        for override in (None, 7):
            with self.subTest(weekly_min_snaps=override):
                model = plus_minus.PlayerPlusMinus('QB', [2025], config=config, weekly_min_snaps=override)
                self.assertIsNot(model.config, config)
                self.assertEqual(model.config, dict(original, min_snaps=37 if override is None else override))
                model.config['min_snaps'] = 99
                self.assertEqual(config, original)
        default_model = plus_minus.PlayerPlusMinus('QB', [2025], weekly_min_snaps=4)
        self.assertEqual(default_model.config['min_snaps'], 4)

    def test_init_rejects_invalid_prior_and_minimum_sample_parameters(self):
        for value in (0.0, -0.1, 1.01):
            with self.subTest(history_decay=value):
                with self.assertRaisesRegex(ValueError, 'history_decay'):
                    plus_minus.PlayerPlusMinus('QB', [2025], history_decay=value)
        for parameter in ('season_transition_sd', 'rookie_prior_sd', 'observation_reference_snaps', 'weekly_min_snaps'):
            for value in (0, -1):
                with self.subTest(parameter=parameter, value=value):
                    with self.assertRaisesRegex(ValueError, 'must be positive'):
                        plus_minus.PlayerPlusMinus('QB', [2025], **{parameter: value})
        for value in (0, -1):
            with self.subTest(custom_min_snaps=value):
                config = dict(plus_minus.POSITION_CONFIG['QB'], min_snaps=value)
                with self.assertRaisesRegex(ValueError, 'must be positive'):
                    plus_minus.PlayerPlusMinus('QB', [2025], config=config)
        with self.assertRaisesRegex(ValueError, 'Position must be one of'):
            plus_minus.PlayerPlusMinus('invalid', [2025])

    def test_weekly_priors_use_completed_final_bayesian_average_without_changing_current_features(self):
        pbp = self.qb_pbp()
        baseline, baseline_features = self.run_weekly(pbp)
        completed = baseline.splits_df.query('season == 2024')
        np.testing.assert_array_equal(completed['prior_mean'], np.zeros(len(completed)))
        np.testing.assert_allclose(completed['prior_sd'], baseline.rookie_prior_sd)
        self.assertTrue(completed['prior_season'].isna().all())
        final = completed.query('week == 2').set_index('player_id')
        for week in (1, 2):
            with self.subTest(week=week):
                current = baseline.splits_df.query('season == 2025 and week == @week').set_index('player_id')
                np.testing.assert_allclose(current['prior_mean'], final.loc[current.index, 'bayesian_plus_minus'] * 0.8)
                np.testing.assert_array_equal(current['prior_season'], np.full(len(current), 2024))
        changed = pbp.copy(deep=True)
        final_plays = (changed['season'] == 2024) & (changed['week'] == 2) & (changed['passer_player_id'] == 'qb0')
        changed.loc[final_plays, 'epa'] += 10.0
        rerun, rerun_features = self.run_weekly(changed)
        changed_final = rerun.splits_df.query('season == 2024 and week == 2').set_index('player_id')
        self.assertNotEqual(final.loc['qb0', 'bayesian_plus_minus'], changed_final.loc['qb0', 'bayesian_plus_minus'])
        before = baseline.splits_df.query('season == 2025 and week == 1').set_index('player_id')
        after = rerun.splits_df.query('season == 2025 and week == 1').set_index('player_id')
        self.assertNotEqual(before.loc['qb0', 'prior_mean'], after.loc['qb0', 'prior_mean'])
        np.testing.assert_allclose(after['prior_mean'], changed_final.loc[after.index, 'bayesian_plus_minus'] * 0.8)
        for week in (1, 2):
            with self.subTest(unchanged_feature_week=week):
                pd.testing.assert_frame_equal(baseline_features[(2025, week)], rerun_features[(2025, week)])
        pd.testing.assert_frame_equal(
            before.drop(columns=['prior_mean', 'prior_sd', 'prior_season']),
            after.drop(columns=['prior_mean', 'prior_sd', 'prior_season']),
        )

    def test_weekly_pipeline_accumulates_within_season_and_averages_iterations(self):
        pbp = self.qb_pbp()
        postseason = pbp.iloc[[0]].assign(season_type='POST', week=3, epa=999.0)
        model, features = self.run_weekly(pd.concat([pbp, postseason], ignore_index=True))
        results = model.splits_df
        expected_keys = {(season, week) for season in (2024, 2025) for week in (1, 2)}
        self.assertEqual(set(zip(results['season'], results['week'])), expected_keys)
        self.assertEqual(set(features), expected_keys)
        self.assertFalse(results.duplicated(['player_id', 'team', 'season', 'week']).any())
        self.assertEqual(len(results), 16)
        for season, week in sorted(expected_keys):
            with self.subTest(season=season, week=week):
                snapshot = results[(results['season'] == season) & (results['week'] == week)]
                attempts = 1 if week == 1 else 3
                np.testing.assert_array_equal(snapshot['n_on'], np.full(4, attempts))
                np.testing.assert_array_equal(snapshot['n_off'], np.full(4, 3 * attempts))
                np.testing.assert_array_equal(features[(season, week)]['attempts'], np.full(4, attempts))
                self.assertEqual(set(features[(season, week)]['season']), {season})
                np.testing.assert_allclose(snapshot['bayesian_plus_minus'], snapshot['raw_plus_minus'] / 2 + 0.1)
                np.testing.assert_allclose(snapshot['pm_lower'], snapshot['bayesian_plus_minus'] - 0.3)
                np.testing.assert_allclose(snapshot['pm_upper'], snapshot['bayesian_plus_minus'] + 0.3)
                np.testing.assert_allclose(snapshot['pm_uncertainty'], 0.6)
                iterations = model.cluster_ensemble_results
                iterations = iterations[(iterations['season'] == season) & (iterations['week'] == week)]
                self.assertEqual(len(iterations), 8)
                self.assertEqual(set(iterations['iteration']), {0, 1})
                self.assertEqual(set(iterations['cluster']), {0, 1})
        self.assertEqual(set(model.get_results()['season']), {2025})
        self.assertEqual(set(model.get_results()['week']), {2})

    def test_future_week_changes_leave_earlier_features_and_results_unchanged(self):
        pbp = self.qb_pbp()
        baseline, baseline_features = self.run_weekly(pbp)
        changed = pbp.copy(deep=True)
        future = (changed['season'] == 2025) & (changed['week'] == 2) & (changed['passer_player_id'] == 'qb0')
        changed.loc[future, 'epa'] += 100
        changed.loc[future, 'passing_yards'] += 1000
        rerun, rerun_features = self.run_weekly(changed)
        for season in (2024, 2025):
            with self.subTest(season=season):
                pd.testing.assert_frame_equal(baseline_features[(season, 1)], rerun_features[(season, 1)])
                for attribute in ('splits_df', 'cluster_ensemble_results'):
                    before = getattr(baseline, attribute)
                    after = getattr(rerun, attribute)
                    mask = (before['season'] == season) & (before['week'] == 1)
                    pd.testing.assert_frame_equal(before.loc[mask], after.loc[mask])
                before = baseline_features[(season, 2)].set_index('player_id')
                after = rerun_features[(season, 2)].set_index('player_id')
                if season == 2025:
                    self.assertGreater(after.loc['qb0', 'yards_per_attempt'], before.loc['qb0', 'yards_per_attempt'])
        before = baseline.splits_df.query('week == 2')['raw_plus_minus'].to_numpy()
        after = rerun.splits_df.query('week == 2')['raw_plus_minus'].to_numpy()
        self.assertFalse(np.allclose(before, after))

    def test_minimum_snaps_applies_cumulatively_and_resets_each_season(self):
        model, features = self.run_weekly(self.qb_pbp(), min_snaps=2)
        self.assertEqual(set(features), {(2024, 2), (2025, 2)})
        self.assertEqual(set(model.splits_df['week']), {2})
        np.testing.assert_array_equal(model.splits_df['n_on'], np.full(8, 3))

    def test_other_skill_positions_produce_cumulative_weekly_estimates(self):
        for position in ('RB', 'WR', 'TE'):
            with self.subTest(position=position):
                pbp = self.qb_pbp()
                pbp['receiver_player_id'] = pbp['passer_player_id']
                pbp['receiver_player_name'] = pbp['passer_player_name']
                pbp['receiver_position'] = position
                pbp['receiving_yards'] = pbp['passing_yards']
                pbp['yards_after_catch'] = 2.0
                pbp['first_down'] = 0
                pbp['yards_gained'] = pbp['passing_yards']
                pbp['rusher_player_id'] = pbp['passer_player_id']
                pbp['rusher_player_name'] = pbp['passer_player_name']
                pbp['rushing_yards'] = pbp['passing_yards']
                if position == 'RB':
                    rushing = pbp.assign(**{'pass': 0, 'rush': 1})
                    rushing['play_id'] += 100
                    pbp = pd.concat([pbp, rushing], ignore_index=True)
                model, features = self.run_weekly(pbp, position=position)
                self.assertEqual(len(model.splits_df), 16)
                multiplier = 2 if position == 'RB' else 1
                np.testing.assert_array_equal(
                    model.splits_df.query('week == 1')['n_on'], np.full(8, multiplier)
                )
                np.testing.assert_array_equal(
                    model.splits_df.query('week == 2')['n_on'], np.full(8, 3 * multiplier)
                )
                for frame in features.values():
                    self.assertTrue(set(model.config['clustering_features']).issubset(frame.columns))

    def defensive_model(self, position='DB', attributed=False):
        model = self.make_model(position)
        model.pbp = pd.DataFrame({
            'game_id': ['game'] * 4, 'play_id': [1, 2, 3, 4],
            'season': [2025] * 4, 'week': [1] * 4, 'season_type': ['REG'] * 4,
            'posteam': ['AAA'] * 4, 'defteam': ['BBB'] * 4,
            'pass': [1, 0, 1, 1], 'rush': [0, 1, 0, 0],
            'epa': [1.0, -2.0, 3.0, 100.0], 'yards_gained': [10, 2, 20, 50],
        })
        model.participation = pd.DataFrame({
            'game_id': ['game'] * 3, 'play_id': [1, 2, 3], 'season': [2025] * 3,
            'possession_team': ['AAA'] * 3,
            'defense_players': ['db1;db2;db1;', 'db1;', 'ineligible'],
        })
        roster_position = {'DB': 'CB', 'LB': 'LB', 'DL': 'DE'}[position]
        model.rosters = pd.DataFrame({
            'gsis_id': ['db1', 'db2', 'ineligible'],
            'position': [roster_position, roster_position, 'QB'], 'season': [2025] * 3,
        })
        model.players = pd.DataFrame({'gsis_id': ['db1', 'db2'], 'short_name': ['D.One', 'D.Two']})
        if attributed:
            for column in ('sack_player_id', 'qb_hit_1_player_id',
                           'tackle_for_loss_1_player_id', 'pass_defense_1_player_id',
                           'interception_player_id'):
                model.pbp[column] = ['db1', None, None, None]
            model.pbp['solo_tackle_1_player_id'] = [None, 'db1', None, None]
        return model

    def test_all_defensive_configured_features_are_generated_from_attributed_events(self):
        for position in ('DL', 'LB', 'DB'):
            with self.subTest(position=position):
                model = self.defensive_model(position, attributed=True)
                model.compute_player_features()
                features = model.player_features.set_index('player_id')
                self.assertEqual(set(features.index), {'db1', 'db2'})
                for column in plus_minus.POSITION_CONFIG[position]['clustering_features']:
                    self.assertIn(column, features.columns)
                    self.assertEqual(features.loc['db1', column], 1.0)
                    self.assertEqual(features.loc['db2', column], 0.0)
                self.assertEqual(features.loc['db1', 'player_name'], 'D.One')
                self.assertEqual(features.loc['db1', 'snaps'], 2)
                self.assertEqual(features.loc['db2', 'snaps'], 1)

    def test_sparse_defensive_events_are_finite_zeros_without_optional_columns(self):
        for position in ('DL', 'LB', 'DB'):
            with self.subTest(position=position):
                model = self.defensive_model(position)
                model.compute_player_features()
                values = model.player_features[model.config['clustering_features']].to_numpy(dtype=float)
                self.assertEqual(values.shape, (2, len(model.config['clustering_features'])))
                self.assertTrue(np.isfinite(values).all())
                np.testing.assert_array_equal(values, np.zeros_like(values))

    def test_participation_filters_actual_schema_and_position_switch_by_season(self):
        for position in ('DB', 'OL'):
            with self.subTest(position=position):
                model = self.defensive_model()
                model.position = position
                model.config = dict(plus_minus.POSITION_CONFIG[position], min_snaps=1)
                model.pbp['sack'] = 0
                model.pbp['penalty'] = 0
                model.pbp['penalty_player_id'] = None
                model.pbp['penalty_team'] = 'AAA'
                model.participation['offense_players'] = model.participation['defense_players']
                model.pbp = pd.concat([model.pbp.assign(season=2024), model.pbp], ignore_index=True)
                model.participation = pd.concat([
                    model.participation.assign(season=2024), model.participation
                ], ignore_index=True)
                model.rosters = pd.DataFrame([
                    ('db1', 2024, 'WR'), ('db1', 2025, position),
                    ('db2', 2024, position), ('db2', 2025, position),
                    ('ineligible', 2024, 'QB'), ('ineligible', 2025, 'QB'),
                ], columns=['player_id', 'season', 'position'])
                model.compute_player_features()
                self.assertEqual(model.player_features.query("player_id == 'db1'")['season'].tolist(), [2025])
                self.assertEqual(model.player_features.query("player_id == 'db2'")['season'].tolist(), [2024, 2025])
                self.assertNotIn('ineligible', model.player_features['player_id'].tolist())
                model.player_features['cluster'] = 0
                model.compute_on_off_splits()
                self.assertEqual(model.splits_df.query("player_id == 'db1'")['season'].tolist(), [2025])

    def test_participation_totals_count_unique_plays_even_without_eligible_defenders(self):
        model = self.defensive_model()
        model.compute_player_features()
        model.player_features['cluster'] = 0
        model.compute_on_off_splits()
        splits = model.splits_df.set_index('player_id')
        self.assertEqual(set(splits.index), {'db1', 'db2'})
        for player_id, n_on, n_off, epa_on_sum, epa_off_sum, raw_pm in (
            ('db1', 2, 1, -1.0, 3.0, 3.5),
            ('db2', 1, 2, 1.0, 1.0, -0.5),
        ):
            with self.subTest(player_id=player_id):
                row = splits.loc[player_id]
                self.assertEqual(row['team_plays'], 3)
                self.assertEqual(row['team_epa_sum'], 2.0)
                self.assertEqual(row['n_on'], n_on)
                self.assertEqual(row['n_off'], n_off)
                self.assertEqual(row['epa_on_sum'], epa_on_sum)
                self.assertEqual(row['epa_off_sum'], epa_off_sum)
                self.assertAlmostEqual(row['raw_plus_minus'], raw_pm)

    def test_ol_penalties_only_charge_culprit_and_include_no_play_penalties(self):
        model = self.make_model('OL')
        model.pbp = pd.DataFrame({
            'game_id': ['game', 'game'], 'play_id': [1, 2], 'season': [2025, 2025],
            'season_type': ['REG', 'REG'], 'posteam': ['AAA', 'AAA'],
            'pass': [1, 0], 'rush': [0, 0], 'sack': [0, 0],
            'epa': [0.25, np.nan], 'penalty': [1, 1],
            'play_type': ['pass', 'no_play'], 'penalty_player_id': ['ol1', 'ol1'],
            'penalty_team': ['AAA', 'AAA'],
        })
        model.participation = pd.DataFrame({
            'game_id': ['game', 'game'], 'play_id': [1, 2], 'season': [2025, 2025],
            'possession_team': ['AAA', 'AAA'], 'offense_players': ['ol1;ol2', 'ol1;ol2'],
        })
        model.rosters = pd.DataFrame({
            'gsis_id': ['ol1', 'ol2'], 'position': ['T', 'G'], 'season': [2025, 2025],
        })
        model.players = pd.DataFrame({'gsis_id': ['ol1', 'ol2'], 'short_name': ['O.One', 'O.Two']})
        model.compute_player_features()
        features = model.player_features.set_index('player_id')
        self.assertEqual(set(features.index), {'ol1', 'ol2'})
        self.assertEqual(features.loc['ol1', 'penalties_per_game'], 2.0)
        self.assertEqual(features.loc['ol2', 'penalties_per_game'], 0.0)
        np.testing.assert_array_equal(features['snaps'], [1, 1])
        np.testing.assert_array_equal(features['games'], [1, 1])

    def test_ol_participation_splits_use_offensive_team_and_unique_plays(self):
        model = self.defensive_model()
        model.position = 'OL'
        model.config = dict(plus_minus.POSITION_CONFIG['OL'], min_snaps=1)
        model.participation['offense_players'] = ['ol1;ol2', 'ol1', 'other']
        model.player_features = pd.DataFrame({
            'player_id': ['ol1', 'ol2'], 'player_name': ['O.One', 'O.Two'], 'cluster': [0, 0]
        })
        model.rosters = pd.DataFrame({'player_id': ['ol1', 'ol2'], 'season': [2025, 2025],
                                     'position': ['OL', 'OL']})
        model.compute_on_off_splits()
        splits = model.splits_df.set_index('player_id')
        self.assertEqual(set(splits['team']), {'AAA'})
        self.assertEqual(splits.loc['ol1', 'n_off'], 1)
        self.assertEqual(splits.loc['ol2', 'n_off'], 2)
        self.assertEqual(splits.loc['ol1', 'raw_plus_minus'], -3.5)

    def test_singleton_clustering_uses_zero_label_without_kmeans(self):
        model = self.make_model()
        model.pbp = self.qb_pbp().iloc[[0]].copy()
        model.compute_player_features()
        with mock.patch.object(plus_minus, 'KMeans', side_effect=AssertionError('Singleton KMeans')) as kmeans:
            self.assertIs(model.cluster_players(), model)
            np.testing.assert_array_equal(model.player_features['cluster'], [0])
            np.testing.assert_array_equal(model._kmeans_labels(99, 42), [0])
            kmeans.assert_not_called()
        self.assertEqual(model._X_pca.shape, (1, 1))

    def test_small_clustering_caps_cluster_count_to_available_samples(self):
        for count in (2, 3):
            with self.subTest(count=count):
                model = self.make_model()
                model.pbp = self.qb_pbp().iloc[:count].copy()
                model.compute_player_features()
                model.cluster_players(n_clusters=99)
                labels = model.player_features['cluster'].to_numpy()
                self.assertEqual(len(labels), count)
                self.assertEqual(set(labels), {0, 1})

    def run_incremental(self, model, pbp, fail_key=None):
        fitted_keys = []
        priors = {}

        def load_data():
            model.pbp = pbp
            model.participation = None
            return model

        def ensemble():
            key = (int(model.pbp['season'].iloc[0]), int(model.pbp['week'].max()))
            if key == fail_key:
                raise RuntimeError('Interrupted fit')
            fitted_keys.append(key)
            priors[key] = model.splits_df.copy(deep=True)
            model.splits_df['bayesian_plus_minus'] = model.splits_df['raw_plus_minus'] / 2
            model.splits_df['pm_lower'] = model.splits_df['bayesian_plus_minus'] - 0.1
            model.splits_df['pm_upper'] = model.splits_df['bayesian_plus_minus'] + 0.1
            model.cluster_ensemble_results = pd.concat([
                model.splits_df.assign(iteration=iteration, n_clusters_used=2)
                for iteration in range(model.n_cluster_iterations)
            ], ignore_index=True)
            return model

        with mock.patch.object(model, 'load_data', side_effect=load_data), \
                mock.patch.object(model, 'run_clustering_ensemble', side_effect=ensemble), \
                mock.patch.object(model, 'save_results', wraps=model.save_results) as save:
            model.run_full_pipeline(incremental=True)
            self.assertEqual(save.call_count, len(fitted_keys))
        return fitted_keys, priors

    def test_incremental_missing_week_restart_preserves_history_and_deduplicates(self):
        pbp = self.qb_pbp()
        initial_pbp = pbp[(pbp['season'] == 2024) & (pbp['week'] == 1)]
        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
            first = self.make_model()
            keys, _ = self.run_incremental(first, initial_pbp)
            self.assertEqual(keys, [(2024, 1)])
            original = first.splits_df.copy(deep=True)
            original_iterations = first.cluster_ensemble_results.copy(deep=True)
            resumed = self.make_model()
            keys, _ = self.run_incremental(resumed, pbp)
            self.assertEqual(keys, [(2024, 2), (2025, 1), (2025, 2)])
            pd.testing.assert_frame_equal(
                resumed.splits_df.query('season == 2024 and week == 1').reset_index(drop=True),
                original, check_dtype=False,
            )
            pd.testing.assert_frame_equal(
                resumed.cluster_ensemble_results.query('season == 2024 and week == 1').reset_index(drop=True),
                original_iterations, check_dtype=False,
            )
            restart = self.make_model()
            keys, _ = self.run_incremental(restart, pbp)
            self.assertEqual(keys, [])
            pd.testing.assert_frame_equal(restart.splits_df, resumed.splits_df, check_dtype=False)
            self.assertFalse(restart.splits_df.duplicated(['player_id', 'team', 'season', 'week']).any())
            self.assertFalse(restart.cluster_ensemble_results.duplicated(
                ['player_id', 'team', 'season', 'week', 'iteration']).any())

    def test_incremental_priors_load_final_season_outside_requested_years(self):
        pbp = self.qb_pbp()
        historical = pbp.query('season == 2024').assign(week=18)
        current = pbp.query('season == 2025')
        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
            first = self.make_model()
            first.years = [2024]
            self.run_incremental(first, historical)
            original = first.splits_df.copy(deep=True)
            resumed = self.make_model()
            resumed.years = [2025]
            keys, priors = self.run_incremental(resumed, current)
            self.assertEqual(keys, [(2025, 1), (2025, 2)])
            expected = original.set_index('player_id')['bayesian_plus_minus'] * resumed.history_decay
            for frame in priors.values():
                actual = frame.set_index('player_id')
                np.testing.assert_allclose(actual.loc[expected.index, 'prior_mean'], expected)
                self.assertEqual(set(actual['prior_season']), {2024})
            pd.testing.assert_frame_equal(resumed.splits_df.query('season == 2024').reset_index(drop=True),
                                          original, check_dtype=False)
            self.assertEqual(resumed.years, [2025])

    def test_incremental_earlier_week_is_not_a_completed_season_prior(self):
        pbp = self.qb_pbp()
        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
            first = self.make_model()
            self.run_incremental(first, pbp.query('season == 2024'))
            resumed = self.make_model()
            resumed.years = [2025]
            _, priors = self.run_incremental(resumed, pbp.query('season == 2025'))
            for frame in priors.values():
                self.assertTrue(frame['prior_season'].isna().all())
                np.testing.assert_array_equal(frame['prior_mean'], 0.0)

    def test_incremental_incomplete_legacy_week_is_refit_without_losing_other_rows(self):
        pbp = self.qb_pbp().query('season == 2024')
        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
            first = self.make_model()
            self.run_incremental(first, pbp)
            original = first.splits_df.query('week == 1').reset_index(drop=True)
            Path(directory, 'plus_minus_checkpoint_QB.zip').unlink()
            iterations_path = Path(directory, 'plus_minus_weekly_iterations_QB.csv')
            damaged = first.cluster_ensemble_results.query('week == 1 or iteration == 0')
            pd.concat([damaged, damaged.iloc[:1]], ignore_index=True).to_csv(iterations_path, index=False)
            splits_path = Path(directory, 'plus_minus_weekly_QB.csv')
            pd.concat([first.splits_df, first.splits_df.iloc[:1]], ignore_index=True).to_csv(splits_path, index=False)
            resumed = self.make_model()
            keys, _ = self.run_incremental(resumed, pbp)
            self.assertEqual(keys, [(2024, 2)])
            pd.testing.assert_frame_equal(
                resumed.splits_df.query('week == 1').sort_values('player_id').reset_index(drop=True),
                original.sort_values('player_id').reset_index(drop=True), check_dtype=False,
            )
            self.assertEqual(len(resumed.splits_df), len(first.splits_df))
            self.assertEqual(len(resumed.cluster_ensemble_results), len(first.cluster_ensemble_results))

    def test_incremental_interrupted_fit_retains_previous_committed_week(self):
        pbp = self.qb_pbp().query('season == 2024')
        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
            interrupted = self.make_model()
            with self.assertRaisesRegex(RuntimeError, 'Interrupted fit'):
                self.run_incremental(interrupted, pbp, fail_key=(2024, 2))
            self.assertIs(interrupted.pbp, pbp)
            self.assertEqual(interrupted.years, [2024, 2025])
            saved, iterations, completed = self.make_model()._load_saved_results()
            self.assertEqual(completed, {(2024, 1)})
            self.assertEqual(set(saved['week']), {1})
            self.assertEqual(set(iterations['week']), {1})
            resumed = self.make_model()
            keys, _ = self.run_incremental(resumed, pbp)
            self.assertEqual(keys, [(2024, 2)])

    def test_atomic_checkpoint_failure_keeps_old_pair_and_cleans_temporary_files(self):
        pbp = self.qb_pbp().query('season == 2024')
        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
            first = self.make_model()
            self.run_incremental(first, pbp.query('week == 1'))
            checkpoint = Path(directory, 'plus_minus_checkpoint_QB.zip')
            original = checkpoint.read_bytes()
            original_replace = plus_minus.os.replace

            def fail_checkpoint(source, destination):
                if Path(destination) == checkpoint:
                    raise OSError('Interrupted commit')
                return original_replace(source, destination)

            with mock.patch.object(plus_minus.os, 'replace', side_effect=fail_checkpoint):
                with self.assertRaisesRegex(OSError, 'Interrupted commit'):
                    self.run_incremental(self.make_model(), pbp)
            self.assertEqual(checkpoint.read_bytes(), original)
            self.assertEqual(self.make_model()._load_saved_results()[2], {(2024, 1)})
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_checkpoint_prevents_mixed_exports_after_interrupted_csv_refresh(self):
        pbp = self.qb_pbp().query('season == 2024')
        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
            self.run_incremental(self.make_model(), pbp.query('week == 1'))
            original_replace = plus_minus.os.replace
            refresh_count = 0

            def fail_iteration_export(source, destination):
                nonlocal refresh_count
                if str(destination).endswith('weekly_iterations_QB.csv'):
                    refresh_count += 1
                    if refresh_count == 2:
                        raise OSError('Interrupted export')
                return original_replace(source, destination)

            with mock.patch.object(plus_minus.os, 'replace', side_effect=fail_iteration_export):
                with self.assertRaisesRegex(OSError, 'Interrupted export'):
                    self.run_incremental(self.make_model(), pbp)
            saved, iterations, completed = self.make_model()._load_saved_results()
            self.assertEqual(completed, {(2024, 1), (2024, 2)})
            self.assertEqual(set(saved['week']), {1, 2})
            self.assertEqual(set(iterations['week']), {1, 2})
            exported = pd.read_csv(Path(directory, 'plus_minus_weekly_iterations_QB.csv'))
            self.assertEqual(set(exported['week']), {1, 2})
            keys, _ = self.run_incremental(self.make_model(), pbp)
            self.assertEqual(keys, [])
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_incremental_mismatched_legacy_averages_are_not_completed(self):
        model, _ = self.run_weekly(self.qb_pbp())
        iterations = model.cluster_ensemble_results.copy(deep=True)
        iterations.loc[iterations['week'] == 2, 'bayesian_plus_minus'] += 1
        self.assertEqual(model._completed_week_keys(model.splits_df, iterations), {(2024, 1), (2025, 1)})

    def test_completed_season_history_uses_standard_final_weeks_and_excludes_current(self):
        model = self.make_model()
        frame = self.posterior_history([
            ('old_final', 'AAA', 2020, 17, 100, 0.1, 0.02),
            ('old_early', 'AAA', 2020, 16, 100, 0.1, 0.02),
            ('new_final', 'AAA', 2021, 18, 100, 0.1, 0.02),
            ('new_early', 'AAA', 2021, 17, 100, 0.1, 0.02),
            ('current', 'AAA', 2026, 18, 100, 0.1, 0.02),
            ('future', 'AAA', 2027, 18, 100, 0.1, 0.02),
        ])
        with mock.patch.object(plus_minus, 'date') as clock:
            clock.today.return_value = plus_minus.pd.Timestamp('2026-10-11').date()
            history = model._completed_season_history([frame])
        self.assertEqual(set(history[0]['player_id']), {'old_final', 'new_final'})

    def test_corrupt_checkpoint_does_not_fall_back_to_legacy_exports(self):
        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
            self.run_incremental(self.make_model(), self.qb_pbp().query('season == 2024 and week == 1'))
            checkpoint = Path(directory, 'plus_minus_checkpoint_QB.zip')
            checkpoint.write_bytes(b'Interrupted or corrupt archive')
            with self.assertRaises(plus_minus.zipfile.BadZipFile):
                self.make_model()._load_saved_results()

    def test_incremental_rejects_incomplete_new_ensemble_without_saving(self):
        model = self.make_model()
        pbp = self.qb_pbp().query('season == 2024 and week == 1')

        def load_data():
            model.pbp = pbp
            model.participation = None

        def incomplete_ensemble():
            model.splits_df['bayesian_plus_minus'] = 0.0
            model.splits_df['pm_lower'] = -0.1
            model.splits_df['pm_upper'] = 0.1
            model.cluster_ensemble_results = model.splits_df.assign(iteration=0)

        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)), \
                mock.patch.object(model, 'load_data', side_effect=load_data), \
                mock.patch.object(model, 'run_clustering_ensemble', side_effect=incomplete_ensemble), \
                mock.patch.object(model, 'save_results') as save:
            with self.assertRaisesRegex(ValueError, 'Incomplete or inconsistent'):
                model.run_full_pipeline(incremental=True)
            save.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_save_results_uses_weekly_filenames_and_preserves_week_keys(self):
        model, _ = self.run_weekly(self.qb_pbp())
        with TemporaryDirectory() as directory:
            with mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
                self.assertIs(model.save_results(), model)
            expected = {
                'plus_minus_weekly_QB.csv': model.splits_df,
                'plus_minus_weekly_iterations_QB.csv': model.cluster_ensemble_results,
            }
            self.assertEqual({path.name for path in Path(directory).iterdir()},
                             set(expected) | {'plus_minus_checkpoint_QB.zip'})
            for filename, frame in expected.items():
                with self.subTest(filename=filename):
                    saved = pd.read_csv(Path(directory) / filename)
                    pd.testing.assert_frame_equal(saved, frame, check_dtype=False)
                    self.assertEqual(set(zip(saved['season'], saved['week'])),
                                     {(2024, 1), (2024, 2), (2025, 1), (2025, 2)})


    def test_qb_scrambles_and_league_reference_exclude_all_own_stints(self):
        model = self.make_model()
        pbp = self.qb_pbp().query('season == 2024 and week == 1').copy()
        pbp['posteam'] = ['AAA', 'BBB', 'CCC', 'DDD']
        pbp['rusher_player_id'] = pd.NA
        pbp['rusher_player_name'] = pd.NA
        scramble = pbp.iloc[[0]].assign(posteam='EEE', passer_player_id=pd.NA,
                                       passer_player_name=pd.NA, rusher_player_id='qb0',
                                       rusher_player_name='Quarterback 0', qb_scramble=1, epa=1.0)
        unidentified = pbp.iloc[[0]].assign(passer_player_id=pd.NA, epa=100.0)
        non_qb = pbp.iloc[[0]].assign(passer_player_id='rb', epa=100.0)
        false_fallback = unidentified.assign(rusher_player_id='qb0')
        model.pbp = pd.concat([pbp, scramble, unidentified, non_qb, false_fallback], ignore_index=True)
        original = model.pbp.copy(deep=True)
        model.compute_player_features()
        model.player_features['cluster'] = 0
        model.compute_on_off_splits()
        rows = model.splits_df.query('player_id == "qb0"')
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows.n_off.eq(3).all())
        np.testing.assert_allclose(rows.epa_off, pbp.iloc[1:].epa.mean())
        features = model.player_features.query('player_id == "qb0" and posteam == "EEE"').iloc[0]
        self.assertEqual(features.attempts, 1)
        self.assertEqual(features.scramble_rate, 1)
        self.assertEqual(features.player_name, 'Quarterback 0')
        self.assertTrue(model.splits_df.reference_definition.eq('other_roster_qbs_league_season').all())
        pd.testing.assert_frame_equal(model.pbp, original)

    def box_pbp(self):
        base = self.qb_pbp()
        historical = []
        for season in (2023, 2024):
            frame = base.query('season == 2024').copy()
            frame['game_id'] = frame['game_id'].str.replace('2024', str(season))
            frame['season'] = season
            frame.loc[frame['week'] == 2, 'week'] = 18
            frame['epa'] += (season - 2023) * 0.1
            historical.append(frame)
        return pd.concat(historical + [base.query('season == 2025')], ignore_index=True)

    def make_box(self, cache=False, **options):
        options.setdefault('box_cluster_iterations', 1)
        model = plus_minus.PlayerPlusMinus(
            'QB', [2023, 2024, 2025], method='box', weekly_min_snaps=1,
            box_cache=cache, **options,
        )
        model.rosters = self.skill_rosters()
        return model

    def test_box_variable_k_is_deterministic_and_scores_average_members(self):
        models = []
        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
            for repeat in range(2):
                model = self.run_box(self.make_box(box_cluster_iterations=8, cluster_k_min=2,
                                                   cluster_k_max=4), self.box_pbp())
                models.append(model)
            bundle = models[0]._box_models[2025]
            counts = [member['kmeans'].n_clusters for member in bundle['members']]
            self.assertEqual(len(counts), 8)
            self.assertGreater(len(set(counts)), 1)
            self.assertTrue(all(2 <= count <= 4 for count in counts))
            frame = models[0].splits_df.query('season == 2025')
            scaled = bundle['scaler'].transform(frame[models[0].config['clustering_features']].fillna(bundle['medians']))
            predictions = []
            for member in bundle['members']:
                labels = member['kmeans'].predict(scaled)
                design = np.column_stack([scaled, member['encoder'].transform(labels.reshape(-1, 1))])
                predictions.append(member['ridge'].predict(design))
            np.testing.assert_allclose(frame.box_prediction, np.mean(predictions, axis=0))
            historical = models[0]._box_history.query('season == 2024')
            historical_scaled = bundle['scaler'].transform(historical[models[0].config['clustering_features']].fillna(bundle['medians']))
            historical_predictions = []
            for member in bundle['members']:
                labels = member['kmeans'].predict(historical_scaled)
                design = np.column_stack([historical_scaled, member['encoder'].transform(labels.reshape(-1, 1))])
                historical_predictions.append(member['ridge'].predict(design))
            expected = np.mean(historical_predictions, axis=0) * historical.n_on / (historical.n_on + models[0].box_prior_snaps)
            expected *= models[0].history_decay
            actual = historical[['player_id']].merge(bundle['priors'], on='player_id').prior_mean
            np.testing.assert_allclose(actual, expected)
            pd.testing.assert_frame_equal(models[0].splits_df, models[1].splits_df)
            changed = self.make_box(box_cluster_iterations=7, cluster_k_min=2, cluster_k_max=4)
            self.assertNotEqual(models[0]._configuration_fingerprint(), changed._configuration_fingerprint())
            capped = self.run_box(self.make_box(box_cluster_iterations=2, cluster_k_max=1), self.box_pbp())
            self.assertTrue(all(member['kmeans'].n_clusters == 1 for member in capped._box_models[2025]['members']))

    def test_box_prior_models_and_play_weighting(self):
        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
            decay = self.run_box(self.make_box(), self.box_pbp())
            zero = self.run_box(self.make_box(prior_model='zero'), self.box_pbp())
            regression = self.run_box(self.make_box(prior_model='regression'), self.box_pbp())
            weighted = self.run_box(self.make_box(box_weighting='plays'), self.box_pbp())
        current = zero.splits_df.query('season == 2025')
        self.assertTrue(current['prior_mean'].eq(0.0).all())
        np.testing.assert_allclose(current['box_plus_minus'], current['sample_weight'] * current['box_prediction'])
        bundle = regression._box_models[2025]
        self.assertEqual(sorted(bundle['prior_regression']['summaries']['season'].unique()), [2023, 2024])
        self.assertTrue(np.isfinite(regression.splits_df['prior_mean']).all())
        self.assertFalse(np.allclose(regression.splits_df.query('season == 2025')['prior_mean'],
                                     decay.splits_df.query('season == 2025')['prior_mean']))
        self.assertNotEqual(decay._configuration_fingerprint(), weighted._configuration_fingerprint())
        for options in ({'prior_model': 'bayes'}, {'box_weighting': 'snaps'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.make_box(**options)

    def test_load_data_downloads_only_uncached_seasons(self):
        model = plus_minus.PlayerPlusMinus('QB', [2024, 2026], method='box')
        frames = {2024: self.qb_pbp().query('season == 2024'),
                  2026: self.qb_pbp().query('season == 2025').assign(season=2026)}
        calls = []

        def import_pbp(years, **kwargs):
            calls.append((list(years), kwargs.get('cache', False)))
            return pd.concat([frames[year] for year in years], ignore_index=True)

        with TemporaryDirectory() as directory:
            Path(directory, 'season=2024').mkdir()
            with mock.patch.object(plus_minus, 'DATA_CACHE_DIR', Path(directory)), \
                    mock.patch.object(plus_minus.nfl, 'import_pbp_data', side_effect=import_pbp), \
                    mock.patch.object(plus_minus.nfl, 'import_seasonal_rosters', side_effect=AssertionError('stop')):
                with self.assertRaisesRegex(AssertionError, 'stop'):
                    model.load_data()
        self.assertEqual(calls, [([2024], True), ([2026], False)])
        self.assertEqual(sorted(model.pbp['season'].unique()), [2024, 2026])

    def test_madden_ratings_join_and_carry_forward(self):
        from player_value import madden
        with TemporaryDirectory() as directory:
            pd.DataFrame({'player_id': ['ol1', 'ol2'], 'season': [2024, 2024], 'overallrating': [80.0, 60.0],
                          'passblocking': [82.0, 55.0], 'runblocking': [75.0, 70.0], 'impactblocking': [70.0, 60.0],
                          'strength': [90.0, 80.0], 'awareness': [85.0, 50.0]}).to_parquet(Path(directory, '2024.parquet'))
            features = pd.DataFrame({'player_id': ['ol1', 'ol2', 'ol3'], 'team': 'AAA', 'season': [2024, 2025, 2025],
                                     'penalties_per_game': [0.1, 0.2, 0.3]})
            joined = madden.attach_ratings(features, 'OL', Path(directory))
        self.assertEqual(joined['madden_overallrating'].tolist()[:2], [80.0, 60.0])
        self.assertTrue(np.isnan(joined['madden_overallrating'].iloc[2]))
        self.assertEqual(joined['player_id'].tolist(), ['ol1', 'ol2', 'ol3'])
        model = plus_minus.PlayerPlusMinus('OL', [2024], method='box', madden_features=True)
        self.assertIn('madden_passblocking', model.config['clustering_features'])
        self.assertNotIn('madden_passblocking', plus_minus.POSITION_CONFIG['OL']['clustering_features'])
        with self.assertRaises(ValueError):
            plus_minus.PlayerPlusMinus('QB', [2024], method='box', madden_features=True)

    def test_calibrated_constructor_reads_position_settings(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'settings.json'
            path.write_text(json.dumps({'WR': {'prior_model': 'regression', 'prior_snaps': 400.0,
                                               'history_decay': 0.9}}))
            model = plus_minus.PlayerPlusMinus.calibrated('wr', [2024], config_path=path)
            self.assertEqual((model.method, model.prior_model, model.box_prior_snaps, model.history_decay),
                             ('box', 'regression', 400.0, 0.9))
            default = plus_minus.PlayerPlusMinus.calibrated('QB', [2024], config_path=path, box_prior_snaps=50.0)
            self.assertEqual((default.prior_model, default.box_prior_snaps), ('decay', 50.0))

    def test_qb_without_other_qbs_has_no_fabricated_reference(self):
        model = self.make_model()
        model.pbp = self.qb_pbp().query('season == 2024 and passer_player_id == "qb0"').copy()
        model.compute_player_features()
        model.player_features['cluster'] = 0
        model.compute_on_off_splits()
        self.assertTrue(model.splits_df.n_off.eq(0).all())
        self.assertTrue(model.splits_df.epa_off.isna().all())
        self.assertTrue(model.splits_df.raw_plus_minus.isna().all())

    def test_box_ensemble_option_validation(self):
        self.assertEqual(plus_minus.PlayerPlusMinus('QB', [2024], method='box').box_cluster_iterations, 20)
        self.assertEqual(plus_minus.PlayerPlusMinus('RB', [2024], method='box').box_cluster_iterations, 20)
        for value in (0, 101, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make_box(box_cluster_iterations=value)
        with self.assertRaises(ValueError):
            self.make_box(cluster_k_min=5, cluster_k_max=2)

    def run_box(self, model, pbp, incremental=False):
        def load():
            model.pbp = pbp
            model.participation = None
            if model.rosters is None:
                model.rosters = self.skill_rosters(model.position)
        with mock.patch.object(model, 'load_data', side_effect=load), \
                mock.patch.object(model, '_fit_hierarchical_bayesian',
                                  side_effect=AssertionError('Box must not fit Bayesian model')), \
                mock.patch.object(model, 'run_clustering_ensemble',
                                  side_effect=AssertionError('Box must not run ensemble')):
            model.run_full_pipeline(incremental=incremental)
        return model

    def test_box_fits_once_per_season_and_uses_only_historical_labels(self):
        original_ridge = plus_minus.Ridge.fit
        original_kmeans = plus_minus.KMeans.fit_predict
        labels = []
        clustering_rows = []

        def ridge_fit(estimator, design, target, **kwargs):
            labels.append(np.asarray(target).copy())
            return original_ridge(estimator, design, target, **kwargs)

        def cluster_fit(estimator, design, **kwargs):
            clustering_rows.append(len(design))
            return original_kmeans(estimator, design, **kwargs)

        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)), \
                mock.patch.object(plus_minus.Ridge, 'fit', new=ridge_fit), \
                mock.patch.object(plus_minus.KMeans, 'fit_predict', new=cluster_fit):
            model = self.run_box(self.make_box(), self.box_pbp())
        self.assertEqual(clustering_rows, [4, 8])
        self.assertEqual(len(labels), 2)
        history = model._box_history.sort_values(['season', 'player_id', 'team'])
        np.testing.assert_allclose(labels[0], history.query('season == 2023')['raw_plus_minus'])
        np.testing.assert_allclose(labels[1], history.query('season < 2025')['raw_plus_minus'])
        self.assertEqual(model._box_models[2023]['training_seasons'], [])
        self.assertEqual(model._box_models[2024]['training_seasons'], [2023])
        self.assertEqual(model._box_models[2025]['training_seasons'], [2023, 2024])
        self.assertIsNone(model.cluster_ensemble_results)
        self.assertTrue(model.splits_df['estimator'].eq('box').all())
        self.assertTrue(model.splits_df[['pm_lower', 'pm_upper']].isna().all().all())
        np.testing.assert_allclose(model.splits_df['bayesian_plus_minus'], model.splits_df['box_plus_minus'])

    def test_box_future_games_do_not_change_earlier_models_or_week_scores(self):
        pbp = self.box_pbp()
        changed = pbp.copy()
        mask = (changed['season'] == 2025) & (changed['week'] == 2)
        changed.loc[mask, ['epa', 'passing_yards', 'air_yards', 'cpoe']] = 9999.0
        future = changed.query('season == 2025').assign(season=2026, epa=-9999.0)
        changed = pd.concat([changed, future], ignore_index=True)
        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
            baseline = self.run_box(self.make_box(box_cluster_iterations=4), pbp)
            modified = self.run_box(self.make_box(box_cluster_iterations=4), changed)
        for season in (2023, 2024, 2025):
            self.assertEqual(baseline._box_models[season]['fingerprint'],
                             modified._box_models[season]['fingerprint'])
        pd.testing.assert_frame_equal(
            baseline.splits_df.query('season < 2025 or week == 1').reset_index(drop=True),
            modified.splits_df.query('season < 2025 or (season == 2025 and week == 1)').reset_index(drop=True),
        )

    def test_box_excludes_incomplete_and_current_seasons_from_training(self):
        pbp = self.box_pbp()
        pbp.loc[pbp['season'] == 2023, 'week'] = 17
        future = pbp.query('season == 2024').assign(season=2026)
        pbp = pd.concat([pbp, future], ignore_index=True)
        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)), \
                mock.patch.object(plus_minus, 'date') as clock:
            clock.today.return_value = pd.Timestamp('2026-10-11').date()
            model = self.run_box(self.make_box(), pbp)
        self.assertEqual(set(model._box_history['season']), {2024})
        self.assertEqual(model._box_models[2025]['training_seasons'], [2024])
        self.assertEqual(model._box_models[2026]['training_seasons'], [2024])

    def test_box_zero_fallback_and_rookie_prior(self):
        pbp = self.box_pbp()
        pbp.loc[(pbp['season'] == 2025) & (pbp['passer_player_id'] == 'qb0'), 'passer_player_id'] = 'rookie'
        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
            model = self.run_box(self.make_box(), pbp)
        first = model.splits_df.query('season == 2023')
        np.testing.assert_array_equal(first['box_plus_minus'], 0.0)
        rookie = model.splits_df.query("season == 2025 and player_id == 'rookie'")
        np.testing.assert_array_equal(rookie['prior_mean'], 0.0)
        self.assertTrue(rookie['prior_season'].isna().all())
        np.testing.assert_allclose(rookie['box_plus_minus'], rookie['sample_weight'] * rookie['box_prediction'])
        veterans = model.splits_df.query("season == 2025 and player_id != 'rookie'")
        self.assertTrue(veterans['prior_season'].eq(2024).all())

    def test_box_sparse_shrinkage_is_monotonic_with_fixed_prediction(self):
        model = self.make_box()
        frame = pd.DataFrame({'player_id': ['rookie'] * 4, 'n_on': [1, 10, 100, 1000]})
        for feature in model.config['clustering_features']:
            frame[feature] = np.nan
        bundle = {'priors': pd.DataFrame(), 'ridge': mock.Mock(), 'scaler': mock.Mock(),
                  'kmeans': mock.Mock(), 'encoder': mock.Mock(), 'fingerprint': 'test',
                  'training_seasons': [2024], 'medians': pd.Series(0.0, index=model.config['clustering_features'])}
        bundle['scaler'].transform.return_value = np.zeros((4, 10))
        bundle['kmeans'].predict.return_value = np.zeros(4, dtype=int)
        bundle['encoder'].transform.return_value = np.ones((4, 1))
        bundle['ridge'].predict.return_value = np.ones(4)
        with mock.patch.object(model, '_box_feature_rows', return_value=frame):
            model._score_box(bundle)
        weights = model.splits_df['sample_weight'].to_numpy()
        self.assertTrue((np.diff(weights) > 0).all())
        self.assertLess(weights[0], 0.01)
        np.testing.assert_allclose(model.splits_df['box_plus_minus'], weights)

    def test_box_tiny_training_and_all_missing_features(self):
        model = self.make_box()
        training = pd.DataFrame({'player_id': ['only'], 'team': ['AAA'], 'season': [2024],
                                 'n_on': [10], 'raw_plus_minus': [0.2]})
        for feature in model.config['clustering_features']:
            training[feature] = np.nan
        model._box_history = training
        bundle = model._box_model_for_season(2025)
        self.assertEqual(bundle['kmeans'].n_clusters, 1)
        self.assertTrue(np.isfinite(bundle['ridge'].coef_).all())

    def test_box_disk_cache_reuses_fit_and_invalidates_changed_history(self):
        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
            baseline = self.run_box(self.make_box(cache=True), self.box_pbp())
            with mock.patch.object(plus_minus.Ridge, 'fit', side_effect=AssertionError('Unexpected refit')), \
                    mock.patch.object(plus_minus.KMeans, 'fit_predict', side_effect=AssertionError('Unexpected refit')):
                cached = self.run_box(self.make_box(cache=True), self.box_pbp())
            pd.testing.assert_frame_equal(baseline.splits_df, cached.splits_df)
            changed = self.box_pbp()
            changed.loc[(changed['season'] == 2024) & (changed['passer_player_id'] == 'qb0'), 'epa'] += 5
            refitted = self.run_box(self.make_box(cache=True), changed)
            self.assertNotEqual(baseline._box_models[2025]['fingerprint'],
                                refitted._box_models[2025]['fingerprint'])

    def test_box_incremental_missing_week_and_changed_settings(self):
        pbp = self.box_pbp()
        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
            first = self.run_box(self.make_box(cache=True), pbp.query('season < 2025 or week == 1'), True)
            with mock.patch.object(plus_minus.Ridge, 'fit', side_effect=AssertionError('Unexpected refit')):
                resumed = self.make_box(cache=True)
                with mock.patch.object(resumed, '_score_box', wraps=resumed._score_box) as score:
                    self.run_box(resumed, pbp, True)
                    self.assertEqual(score.call_count, 1)
                restart = self.make_box(cache=True)
                with mock.patch.object(restart, '_score_box', wraps=restart._score_box) as score:
                    self.run_box(restart, pbp, True)
                    score.assert_not_called()
            self.assertEqual(len(resumed.splits_df), len(first.splits_df) + 4)
            self.assertFalse(resumed.splits_df.duplicated(['player_id', 'team', 'season', 'week']).any())
            self.assertEqual(self.make_model()._load_saved_results()[2], set())
            changed = self.make_box(box_prior_snaps=400)
            with mock.patch.object(changed, '_score_box', wraps=changed._score_box) as score:
                self.run_box(changed, pbp, True)
                self.assertEqual(score.call_count, 6)
            with plus_minus.zipfile.ZipFile(Path(directory, 'plus_minus_checkpoint_QB.zip')) as archive:
                manifest = plus_minus.json.loads(archive.read('manifest.json'))
                self.assertEqual(manifest['method'], 'box')
                self.assertEqual(manifest['completion'], 'weekly_scores')
                self.assertNotIn('iterations.csv', archive.namelist())

    def test_box_interrupted_week_resumes_previous_commit(self):
        pbp = self.box_pbp()
        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
            model = self.make_box(cache=True)
            original = model._score_box
            calls = []

            def interrupted(bundle):
                calls.append(True)
                if len(calls) == 2:
                    raise RuntimeError('Interrupted scoring')
                return original(bundle)

            with mock.patch.object(model, '_score_box', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'Interrupted scoring'):
                    self.run_box(model, pbp, True)
            self.assertEqual(self.make_box()._load_saved_results()[2], {(2023, 1)})
            resumed = self.make_box(cache=True)
            with mock.patch.object(resumed, '_score_box', wraps=resumed._score_box) as score:
                self.run_box(resumed, pbp, True)
                self.assertEqual(score.call_count, 5)

    def test_box_constructor_validation_and_api_default(self):
        self.assertEqual(self.make_model().method, 'bayesian')
        for options in ({'method': 'invalid'}, {'box_alpha': 0}, {'box_prior_snaps': -1},
                        {'box_alpha': np.nan}, {'box_prior_snaps': np.inf}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                plus_minus.PlayerPlusMinus('QB', [2025], **options)

    def test_box_resume_recomputes_only_corrected_week_and_preserves_older_scores(self):
        pbp = self.box_pbp()
        changed = pbp.copy()
        changed.loc[(changed['season'] == 2025) & (changed['week'] == 2), 'passing_yards'] += 100
        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
            baseline = self.run_box(self.make_box(cache=True), pbp, True)
            resumed = self.make_box(cache=True)
            with mock.patch.object(resumed, '_score_box', wraps=resumed._score_box) as score:
                self.run_box(resumed, changed, True)
                self.assertEqual(score.call_count, 1)
            pd.testing.assert_frame_equal(
                baseline.splits_df.query('season < 2025 or week == 1').reset_index(drop=True),
                resumed.splits_df.query('season < 2025 or week == 1').reset_index(drop=True),
                check_dtype=False,
            )

    def test_box_rejects_bayesian_checkpoint_and_legacy_csv_resume(self):
        with TemporaryDirectory() as directory, \
                mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
            self.run_incremental(self.make_model(), self.qb_pbp())
            self.assertEqual(self.make_box()._load_saved_results()[2], set())
            Path(directory, 'plus_minus_checkpoint_QB.zip').unlink()
            self.assertEqual(self.make_box()._load_saved_results()[2], set())
            model = self.make_box()
            with mock.patch.object(model, '_score_box', wraps=model._score_box) as score:
                self.run_box(model, self.box_pbp(), True)
                self.assertEqual(score.call_count, 6)

    def test_box_other_skill_positions_use_historical_models(self):
        for position in ('RB', 'WR', 'TE'):
            with self.subTest(position=position), TemporaryDirectory() as directory, \
                    mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
                pbp = self.box_pbp()
                pbp['receiver_player_id'] = pbp['passer_player_id']
                pbp['receiver_player_name'] = pbp['passer_player_name']
                pbp['receiver_position'] = position
                pbp['receiving_yards'] = pbp['passing_yards']
                pbp['yards_after_catch'] = 2.0
                pbp['first_down'] = 0
                pbp['yards_gained'] = pbp['passing_yards']
                pbp['rusher_player_id'] = pbp['passer_player_id']
                pbp['rusher_player_name'] = pbp['passer_player_name']
                pbp['rushing_yards'] = pbp['passing_yards']
                if position == 'RB':
                    rushing = pbp.assign(**{'pass': 0, 'rush': 1})
                    rushing['play_id'] += 100
                    pbp = pd.concat([pbp, rushing], ignore_index=True)
                model = plus_minus.PlayerPlusMinus(
                    position, [2023, 2024, 2025], method='box', weekly_min_snaps=1, box_cache=False)
                self.run_box(model, pbp)
                self.assertEqual(len(model.splits_df), 24)
                self.assertEqual(model._box_models[2025]['training_seasons'], [2023, 2024])
                self.assertTrue(np.isfinite(model.splits_df['box_plus_minus']).all())

    def test_box_participation_positions_score_historical_and_weekly_profiles(self):
        for position in ('OL', 'DL', 'LB', 'DB'):
            with self.subTest(position=position), TemporaryDirectory() as directory, \
                    mock.patch.object(plus_minus, 'OUTPUT_DIR', Path(directory)):
                fixture = self.defensive_model('DB' if position == 'OL' else position, attributed=True)
                model = plus_minus.PlayerPlusMinus(
                    position, [2024, 2025], method='box', weekly_min_snaps=1, box_cache=False)
                plays, participation = [], []
                for season, week in ((2024, 18), (2025, 1), (2025, 2)):
                    game_id = f'{season}_{week}_AAA_BBB'
                    plays.append(fixture.pbp.assign(season=season, week=week, game_id=game_id))
                    participation.append(fixture.participation.assign(season=season, game_id=game_id))
                model.pbp = pd.concat(plays, ignore_index=True)
                model.participation = pd.concat(participation, ignore_index=True)
                model.rosters = pd.concat([fixture.rosters.assign(season=2024), fixture.rosters], ignore_index=True)
                model.players = fixture.players
                if position == 'OL':
                    model.participation['offense_players'] = model.participation['defense_players']
                    model.rosters.loc[model.rosters['position'] == 'CB', 'position'] = 'T'
                    model.pbp['sack'] = 0
                    model.pbp['penalty'] = 1
                    model.pbp['penalty_player_id'] = 'db1'
                    model.pbp['penalty_team'] = 'AAA'
                with mock.patch.object(model, 'load_data', return_value=model), \
                        mock.patch.object(model, 'run_clustering_ensemble',
                                          side_effect=AssertionError('Unexpected ensemble')):
                    model.run_full_pipeline(incremental=True)
                self.assertEqual(model._box_models[2025]['training_seasons'], [2024])
                self.assertEqual(len(model.splits_df), 6)
                self.assertEqual(model._completed_week_keys(model.splits_df, pd.DataFrame()),
                                 {(2024, 18), (2025, 1), (2025, 2)})
                self.assertTrue(np.isfinite(model.splits_df['box_plus_minus']).all())
                expected_team = 'AAA' if position == 'OL' else 'BBB'
                self.assertEqual(set(model.splits_df['team']), {expected_team})


if __name__ == '__main__':
    unittest.main()