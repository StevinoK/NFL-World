import unittest
import socket
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from player_value.explorer import (
    estimate_chart, file_fingerprint, load_estimates, player_labels, player_series, prepare_estimates,
)
from player_value.run_explorer import available_port
from player_value.positions import LOOKUP_NAME, normalize_rosters, save_lookup


class EstimatePreparationTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([
            ('id-1', 'A.Smith', 'BBB', 2025, 2, 20, 0.2, 9, 'box'),
            ('id-1', 'A.Smith', 'AAA', 2024, 10, 80, 0.1, 9, 'box'),
            ('id-1', 'A.Smith', 'AAA', 2025, 2, 40, -0.3, 9, 'box'),
            ('id-1', 'A.Smith', 'AAA', 2025, 1, 10, -0.1, 9, 'box'),
            ('id-2', 'A.Smith', 'CCC', 2025, 1, 10, 0.7, 9, 'box'),
        ], columns=['player_id', 'player_name', 'team', 'season', 'week',
                    'n_on', 'box_plus_minus', 'bayesian_plus_minus', 'estimator'])

    def test_chronology_and_traded_rows_are_preserved(self):
        rows = player_series(prepare_estimates(self.fixture(), 'QB'), 'id-1')
        self.assertEqual(rows['period'].tolist(), ['2024 W10', '2025 W01', '2025 W02', '2025 W02'])
        self.assertEqual(rows.loc[rows['period'].eq('2025 W02'), 'estimate'].tolist(), [-0.3, 0.2])
        self.assertEqual(rows['n_on'].tolist(), [80, 10, 40, 20])
        self.assertEqual(len(rows.groupby(['season', 'team', 'position', 'segment'])), 3)

    def test_duplicate_names_use_ids_and_one_option_per_id(self):
        data = prepare_estimates(self.fixture(), 'QB')
        data = pd.concat([data, prepare_estimates(self.fixture().iloc[:1], 'RB')])
        labels = player_labels(data)
        self.assertEqual(len(labels), 2)
        self.assertEqual(labels['id-1'], 'A.Smith (QB, RB) [id-1]')
        self.assertEqual(labels['id-2'], 'A.Smith (QB) [id-2]')
        self.assertEqual(set(player_series(data, 'id-2')['player_id']), {'id-2'})

    def test_box_preference_bayesian_fallback_and_invalid_values(self):
        frame = self.fixture().iloc[:3].copy()
        frame.loc[frame.index[1], 'estimator'] = 'bayesian'
        frame.loc[frame.index[2], 'box_plus_minus'] = np.inf
        data = prepare_estimates(frame, 'WR')
        self.assertEqual(data['estimate'].tolist(), [9, 0.2])
        self.assertEqual(data['estimate_kind'].tolist(), ['Bayesian', 'Box'])
        legacy = frame.iloc[:1].drop(columns=['estimator', 'box_plus_minus'])
        self.assertEqual(prepare_estimates(legacy, 'WR')['estimate'].iloc[0], 9)

    def test_duplicates_keep_last_without_averaging(self):
        frame = self.fixture()
        correction = frame.iloc[:1].assign(box_plus_minus=0.8)
        data = prepare_estimates(pd.concat([frame, correction]), 'QB')
        self.assertEqual(len(data), 5)
        self.assertEqual(data.query("team == 'BBB'")['estimate'].iloc[0], 0.8)

    def test_missing_weeks_and_positions_break_traces(self):
        frame = self.fixture().iloc[:1].copy()
        data = pd.concat([prepare_estimates(frame, 'QB'),
                          prepare_estimates(frame.assign(week=4), 'QB'),
                          prepare_estimates(frame, 'RB')])
        rows = player_series(data, 'id-1')
        self.assertEqual(rows.query("position == 'QB'")['segment'].tolist(), [0, 1])

    def test_chart_trace_boundaries_tooltips_and_category_order(self):
        rows = player_series(prepare_estimates(self.fixture(), 'QB'), 'id-1')
        figure = estimate_chart(rows)
        self.assertEqual(len(figure.data), 3)
        self.assertEqual(list(figure.layout.xaxis.categoryarray), ['2024 W10', '2025 W01', '2025 W02'])
        self.assertEqual(figure.layout.yaxis.title.text, 'Box Plus-Minus (EPA/play)')
        layout = figure.to_plotly_json()['layout']
        self.assertEqual(layout['margin']['l'], 80)
        self.assertEqual(layout['yaxis']['title']['standoff'], 20)
        self.assertTrue(layout['yaxis']['automargin'])
        for trace in figure.data:
            self.assertFalse(trace.connectgaps)
            self.assertEqual(len({record[0] for record in trace.customdata}), 1)
            self.assertEqual(len({record[2] for record in trace.customdata}), 1)
            self.assertIn('n_on', trace.hovertemplate)
            self.assertIsNone(trace.error_y.array)

    def test_launcher_skips_occupied_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(('127.0.0.1', 0))
            port = occupied.getsockname()[1]
            occupied.listen()
            with self.assertRaisesRegex(RuntimeError, 'No available'):
                available_port(port, attempts=1)
            if port < 65535:
                self.assertGreater(available_port(port, attempts=10), port)

    def test_names_missing_identity_and_empty_data(self):
        frame = self.fixture().iloc[:1].assign(player_name=None)
        self.assertEqual(player_labels(prepare_estimates(frame, 'QB'))['id-1'], 'id-1 (QB)')
        self.assertTrue(prepare_estimates(frame.assign(player_id=None), 'QB').empty)
        self.assertTrue(prepare_estimates(pd.DataFrame(), 'QB').empty)
        with self.assertRaisesRegex(ValueError, 'Missing columns'):
            prepare_estimates(frame.drop(columns='week'), 'QB')

    def test_missing_empty_malformed_and_fingerprint_refresh(self):
        with TemporaryDirectory() as directory:
            lookup = normalize_rosters(pd.DataFrame([
                (player_id, season, 'QB') for player_id in ('id-1', 'id-2') for season in (2024, 2025)
            ], columns=['player_id', 'season', 'position']))
            save_lookup(lookup, directory)
            path = Path(directory) / 'plus_minus_weekly_QB.csv'
            initial = file_fingerprint(directory)
            self.assertTrue(load_estimates(initial)[0].empty)
            path.touch()
            empty = file_fingerprint(directory)
            self.assertNotEqual(initial, empty)
            self.assertTrue(load_estimates(empty)[0].empty)
            self.fixture().to_csv(path, index=False)
            populated = file_fingerprint(directory)
            self.assertNotEqual(empty, populated)
            data, issues = load_estimates(populated)
            self.assertEqual(len(data), 5)
            self.assertEqual(len(issues), 7)
            self.fixture().drop(columns='week').to_csv(path, index=False)
            self.assertTrue(load_estimates(file_fingerprint(directory))[0].empty)

    def test_existing_artifacts_filter_jefferson_and_season_switch(self):
        with TemporaryDirectory() as directory:
            frame = self.fixture().iloc[:2].assign(player_id='00-0036322', player_name='J.Jefferson')
            switch = frame.assign(player_id='switch', player_name='Switch')
            raw = pd.concat([frame, switch], ignore_index=True)
            for position in ('WR', 'RB', 'OL'):
                raw.to_csv(Path(directory) / f'plus_minus_weekly_{position}.csv', index=False)
            lookup = normalize_rosters(pd.DataFrame([
                ('00-0036322', 2024, 'WR'), ('00-0036322', 2025, 'WR'),
                ('switch', 2024, 'WR'), ('switch', 2025, 'OL'),
            ], columns=['player_id', 'season', 'position']))
            save_lookup(lookup, directory)
            before = file_fingerprint(directory)
            data, issues = load_estimates(before)
            self.assertEqual(player_labels(data)['00-0036322'], 'J.Jefferson (WR)')
            self.assertEqual(set(data.query("player_id == 'switch' and season == 2024")['position']), {'WR'})
            self.assertEqual(set(data.query("player_id == 'switch' and season == 2025")['position']), {'OL'})
            self.assertTrue(any('roster-ineligible' in issue for issue in issues))
            original_scores = raw.copy(deep=True)
            lookup.loc[lookup['player_id'].eq('switch'), 'roster_position'] = 'TE'
            save_lookup(lookup, directory)
            self.assertNotEqual(before, file_fingerprint(directory))
            refreshed, _ = load_estimates(file_fingerprint(directory))
            self.assertFalse(refreshed['player_id'].eq('switch').any())
            pd.testing.assert_frame_equal(pd.read_csv(Path(directory) / 'plus_minus_weekly_OL.csv'), original_scores)

    def test_missing_or_malformed_roster_lookup_fails_closed(self):
        with TemporaryDirectory() as directory:
            self.fixture().to_csv(Path(directory) / 'plus_minus_weekly_QB.csv', index=False)
            data, issues = load_estimates(file_fingerprint(directory))
            self.assertTrue(data.empty)
            self.assertIn('lookup is missing', issues[0])
            pd.DataFrame({'wrong_schema': [1]}).to_parquet(Path(directory) / LOOKUP_NAME)
            data, issues = load_estimates(file_fingerprint(directory))
            self.assertTrue(data.empty)
            self.assertIn('required columns', issues[0])


if __name__ == '__main__':
    unittest.main()