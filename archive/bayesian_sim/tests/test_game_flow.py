from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import game_flow as gf


def row(seq: int, **kw) -> dict:
    base = {
        "season": 2025, "season_type": "REG", "week": 1, "game_id": "G", "play_id": seq * 10,
        "order_sequence": seq, "home_team": "HOM", "away_team": "AWY", "posteam": "AWY",
        "defteam": "HOM", "qtr": 1, "quarter_seconds_remaining": 900, "game_seconds_remaining": 3600,
        "time": "15:00", "desc": "play", "play_type": "run", "epa": 0.0, "success": 0.0,
        "yards_gained": 0.0, "penalty": 0.0, "home_wp": 0.5, "vegas_home_wp": 0.55,
        "total_home_score": 0, "total_away_score": 0, "fixed_drive": 1, "fixed_drive_result": "Punt",
        "drive_play_count": 3, "drive_time_of_possession": "2:00", "drive_first_downs": 0,
        "drive_start_yard_line": "AWY 25", "drive_game_clock_start": "15:00",
        "touchdown": 0.0, "interception": 0.0, "fumble_lost": 0.0,
    }
    base.update(kw)
    return base


def sample_game() -> pd.DataFrame:
    rows = [
        row(1, play_type="kickoff", desc="HOM kicks 65 yards", epa=0.3, quarter_seconds_remaining=900),
        row(2, epa=0.5, success=1.0, yards_gained=6, quarter_seconds_remaining=880),
        # Listed before seq 2 by play_id: order_sequence must win.
        row(3, play_id=5, play_type="pass", epa=-1.0, yards_gained=0, quarter_seconds_remaining=860),
        row(4, play_type="punt", desc="AWY punts 45 yards", epa=-0.2, quarter_seconds_remaining=850),
        row(5, posteam="HOM", defteam="AWY", fixed_drive=2, fixed_drive_result="Touchdown",
            play_type="pass", epa=2.0, success=1.0, yards_gained=40, qtr=2, quarter_seconds_remaining=300,
            touchdown=1.0, total_home_score=7, home_wp=0.6, drive_game_clock_start="05:10",
            drive_start_yard_line="HOM 40"),
        row(6, posteam="HOM", defteam="AWY", fixed_drive=2, fixed_drive_result="Touchdown",
            play_type="no_play", desc="PENALTY on AWY, Defensive Holding - No Play.", epa=0.4, penalty=1.0,
            qtr=2, quarter_seconds_remaining=290, total_home_score=7, home_wp=0.7),
        row(7, posteam="HOM", defteam="AWY", fixed_drive=2, play_type="extra_point", desc="kicks extra point",
            epa=0.1, qtr=2, quarter_seconds_remaining=290, total_home_score=7, home_wp=0.72),
        # Overtime (regular season, 10 minutes).
        row(8, fixed_drive=3, play_type="run", epa=0.2, qtr=5, quarter_seconds_remaining=600,
            total_home_score=7, home_wp=0.4, drive_game_clock_start="10:00"),
        row(9, fixed_drive=3, play_type="run", epa=0.1, qtr=5, quarter_seconds_remaining=540,
            total_home_score=7, total_away_score=6, home_wp=0.45),
    ]
    return pd.DataFrame(rows)


class GameFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plays = gf.game_plays(sample_game(), "G")

    def test_order_and_elapsed_clock(self) -> None:
        self.assertEqual(self.plays["order_sequence"].tolist(), list(range(1, 10)))
        self.assertEqual(self.plays["elapsed"].tolist(), [0, 20, 40, 50, 1500, 1510, 1510, 3600, 3660])

    def test_offense_and_special_teams_flags(self) -> None:
        flags = self.plays.set_index("order_sequence")
        self.assertEqual(flags.loc[[1, 4, 7], "is_special"].tolist(), [True, True, True])
        self.assertEqual(flags.loc[[2, 3, 5, 6, 8, 9], "is_offense"].all(), True)
        self.assertFalse(flags.loc[[1, 4, 7], "is_offense"].any())

    def test_drive_summary(self) -> None:
        drives = gf.drive_summary(self.plays)
        self.assertEqual(drives["team"].tolist(), ["AWY", "HOM", "AWY"])
        self.assertEqual(drives["seq"].tolist(), [1, 2, 3])
        np.testing.assert_allclose(drives["epa"], [-0.5, 2.4, 0.3])
        self.assertEqual(drives.loc[1, "result_short"], "TD")
        self.assertAlmostEqual(drives.loc[0, "success_rate"], 0.5)
        with_st = gf.drive_summary(self.plays, include_special_teams=True)
        np.testing.assert_allclose(with_st["epa"], [-0.4, 2.5, 0.3])
        self.assertIn("2. HOM Q2 05:10 · TD · +2.4 EPA", drives["label"].tolist())
        table = gf.drive_table(drives)
        self.assertEqual(list(table.index), [1, 2, 3])

    def test_figures(self) -> None:
        colors = {"HOM": "#2a78d6", "AWY": "#eb6834"}
        wp = gf.build_wp_figure(self.plays, colors["HOM"], colors["AWY"])
        line = wp.data[-1]
        self.assertEqual(line.y[-1], 1.0)  # home won 7-6: worm ends at 100%
        self.assertEqual(line.x[-1], 3660)
        worm = gf.build_epa_worm(self.plays, colors)
        finals = {trace.name: trace.y[-1] for trace in worm.data}
        self.assertAlmostEqual(finals["AWY offense"], -0.2)
        self.assertAlmostEqual(finals["HOM offense"], 2.4)
        drive_fig = gf.build_drive_figure(gf.drive_summary(self.plays), colors)
        self.assertEqual({t.name for t in drive_fig.data}, {"HOM", "AWY"})

    def test_catalog_and_colors(self) -> None:
        cat = gf.game_catalog(sample_game())
        self.assertEqual(cat["label"].tolist(), ["2025 W1 · AWY 6 @ HOM 7"])
        colors = gf.distinct_team_colors("HOM", "AWY", {"HOM": "#002244", "AWY": "#002244"},
                                         {"HOM": "#69be28"})
        self.assertEqual(colors, {"HOM": "#69be28", "AWY": "#002244"})


if __name__ == "__main__":
    unittest.main()
