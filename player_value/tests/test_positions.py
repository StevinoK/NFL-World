import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from player_value.positions import (
    LOOKUP_NAME, POSITION_LABELS, eligible_rows, load_lookup, normalize_rosters,
    require_seasons, save_lookup,
)


class SeasonalPositionTests(unittest.TestCase):
    def test_jefferson_and_season_switch_do_not_inherit_other_groups(self):
        lookup = normalize_rosters(pd.DataFrame([
            ('00-0036322', 2024, 'WR'), ('switch', 2024, 'TE'),
            ('switch', 2025, 'OL'), ('future', 2025, 'RB'),
        ], columns=['player_id', 'season', 'position']))
        rows = pd.DataFrame([
            ('00-0036322', 2024), ('switch', 2024), ('switch', 2025), ('future', 2024),
        ], columns=['player_id', 'season'])
        self.assertEqual(eligible_rows(rows, lookup, 'WR')['player_id'].tolist(), ['00-0036322'])
        self.assertEqual(eligible_rows(rows, lookup, 'OL')['season'].tolist(), [2025])
        self.assertEqual(eligible_rows(rows, lookup, 'TE')['season'].tolist(), [2024])
        self.assertTrue(eligible_rows(rows, lookup, 'RB').empty)

    def test_labels_and_explicit_multi_group_season_without_duplicate_rows(self):
        rosters = pd.DataFrame([
            (label, 2024, label) for labels in POSITION_LABELS.values() for label in labels
        ] + [('multi', 2024, 'WR'), ('multi', 2024, 'TE'), ('multi', 2024, 'WR'),
             ('kicker', 2024, 'K'), ('unknown', 2024, 'EDGE')],
            columns=['gsis_id', 'season', 'position'])
        lookup = normalize_rosters(rosters)
        rows = lookup[['player_id', 'season']].drop_duplicates()
        for group, labels in POSITION_LABELS.items():
            eligible = eligible_rows(rows, lookup, group)
            self.assertTrue(set(labels).issubset(eligible['player_id']))
            self.assertFalse(eligible['player_id'].isin(['kicker', 'unknown']).any())
        self.assertEqual(eligible_rows(rows, lookup, 'WR')['player_id'].tolist().count('multi'), 1)
        self.assertIn('multi', eligible_rows(rows, lookup, 'TE')['player_id'].tolist())

    def test_missing_schema_and_missing_year_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'required columns'):
            normalize_rosters(pd.DataFrame({'cached_rosters': [True]}))
        lookup = normalize_rosters(pd.DataFrame({'player_id': ['one'], 'season': [2025], 'position': ['WR']}))
        with self.assertRaisesRegex(ValueError, 'missing seasons'):
            require_seasons(lookup, [2024, 2025])

    def test_lookup_roundtrip(self):
        lookup = normalize_rosters(pd.DataFrame({'player_id': ['one'], 'season': [2025], 'position': [' wr ']}))
        with TemporaryDirectory() as directory:
            save_lookup(lookup, directory)
            pd.testing.assert_frame_equal(load_lookup(Path(directory) / LOOKUP_NAME), lookup)

    def test_refresh_preserves_other_years_but_replaces_updated_season(self):
        first = normalize_rosters(pd.DataFrame([
            ('one', 2024, 'WR'), ('one', 2025, 'WR'),
        ], columns=['player_id', 'season', 'position']))
        update = normalize_rosters(pd.DataFrame([
            ('one', 2025, 'TE'),
        ], columns=['player_id', 'season', 'position']))
        with TemporaryDirectory() as directory:
            save_lookup(first, directory)
            save_lookup(update, directory)
            saved = load_lookup(Path(directory) / LOOKUP_NAME)
            self.assertEqual(saved['season'].tolist(), [2024, 2025])
            self.assertEqual(saved['position_group'].tolist(), ['WR', 'TE'])


if __name__ == '__main__':
    unittest.main()