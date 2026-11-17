from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import game_report as gr


def play(play_id: int, **overrides) -> dict:
    row = {
        "season": 2025, "season_type": "REG", "week": 3, "game_id": "2025_03_KC_BUF",
        "play_id": play_id, "posteam": "KC", "defteam": "BUF", "home_team": "BUF",
        "away_team": "KC", "qtr": 1, "time": "10:00", "desc": "", "play_type": "pass",
        "two_point_attempt": 0.0, "epa": 0.0, "air_epa": np.nan, "yac_epa": np.nan,
        "complete_pass": 0.0, "incomplete_pass": 0.0, "interception": 0.0, "penalty": 0.0,
        "fumble_lost": 0.0, "fumbled_1_player_id": None, "fumbled_2_player_id": None,
        "receiver_player_id": "P1", "receiver_player_name": "T.Kelce", "touchdown": 0.0,
        "td_team": None,
    }
    row.update(overrides)
    return row


def sample_pbp() -> pd.DataFrame:
    return pd.DataFrame([
        play(1, desc="15-P.Mahomes pass short middle to 87-T.Kelce to BUF 20 for 12 yards",
             epa=1.5, air_epa=0.5, yac_epa=1.0, complete_pass=1.0),
        play(2, desc="15-P.Mahomes pass deep left to 87-T.Kelce for 30 yards, TOUCHDOWN",
             epa=3.0, air_epa=2.0, yac_epa=1.0, complete_pass=1.0, touchdown=1.0, td_team="KC"),
        play(3, desc="15-P.Mahomes pass incomplete short right to 87-T.Kelce",
             epa=-0.6, air_epa=0.4, yac_epa=-1.0, incomplete_pass=1.0),
        play(4, desc="15-P.Mahomes pass short left intended for 87-T.Kelce INTERCEPTED by 1-X.Y.",
             epa=-4.0, interception=1.0, touchdown=1.0, td_team="BUF"),
        play(5, desc="15-P.Mahomes pass short left to 87-T.Kelce to KC 40 for 5 yards. FUMBLES",
             epa=-3.0, air_epa=0.2, yac_epa=-3.2, complete_pass=1.0, fumble_lost=1.0,
             fumbled_1_player_id="P1"),
        # DPI: nflverse leaves receiver ids empty on nullified plays.
        play(6, desc="15-P.Mahomes pass incomplete deep right to 87-T.Kelce. PENALTY on BUF-24-A.B, "
                     "Defensive Pass Interference, 25 yards - No Play.",
             play_type="no_play", epa=1.8, penalty=1.0, receiver_player_id=None,
             receiver_player_name=None),
        # Two-point tries are excluded.
        play(7, desc="TWO-POINT CONVERSION ATTEMPT. pass to 87-T.Kelce", epa=1.0, two_point_attempt=1.0,
             complete_pass=1.0),
        # A different receiver, never otherwise targeted, drawing a penalty.
        play(8, desc="15-P.Mahomes pass incomplete short left to 10-I.Pacheco. PENALTY on BUF, Holding - No Play.",
             play_type="no_play", epa=0.7, penalty=1.0, receiver_player_id=None, receiver_player_name=None),
    ])


class BuildSegmentsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.segments = gr.build_target_segments(sample_pbp())

    def category_epa(self, play_id: int) -> dict:
        rows = self.segments[self.segments["play_id"] == play_id]
        return dict(zip(rows["category"], rows["epa"].round(6)))

    def test_completion_splits_air_and_yac(self) -> None:
        self.assertEqual(self.category_epa(1), {"Reception": 0.5, "YAC": 1.0})

    def test_outcome_categories(self) -> None:
        self.assertEqual(self.category_epa(3), {"Incomplete": -0.6})
        self.assertEqual(self.category_epa(4), {"Interception": -4.0})
        self.assertEqual(self.category_epa(5), {"Reception": 0.2, "Fumble lost": -3.2})

    def test_nullified_penalty_resolves_target_by_jersey(self) -> None:
        rows = self.segments[self.segments["play_id"] == 6]
        self.assertEqual(rows["category"].tolist(), ["Penalty"])
        self.assertEqual(rows["player_id"].tolist(), ["P1"])

    def test_untracked_penalty_target_gets_synthetic_id(self) -> None:
        rows = self.segments[self.segments["play_id"] == 8]
        self.assertEqual(rows["player_name"].tolist(), ["I.Pacheco"])
        self.assertEqual(rows["player_id"].tolist(), ["KC#10:I.Pacheco"])

    def test_two_point_attempts_excluded(self) -> None:
        self.assertNotIn(7, set(self.segments["play_id"]))

    def test_segments_sum_to_play_epa(self) -> None:
        totals = self.segments.groupby("play_id")["epa"].sum().round(6).to_dict()
        self.assertEqual(totals, {1: 1.5, 2: 3.0, 3: -0.6, 4: -4.0, 5: -3.0, 6: 1.8, 8: 0.7})

    def test_touchdown_side(self) -> None:
        td = self.segments.drop_duplicates("play_id").set_index("play_id")["td"]
        self.assertEqual(td[2], "offense")
        self.assertEqual(td[4], "defense")
        self.assertEqual(td[1], "none")


class StackingTest(unittest.TestCase):
    def test_boxes_stack_away_from_zero_without_overlap(self) -> None:
        segments = gr.build_target_segments(sample_pbp())
        stacked = gr.stack_segments(segments, gr.GROUP_PLAYER_GAME)
        for (_, positive), group in stacked.groupby(["unit", "positive"]):
            group = group.sort_values("base", key=abs)
            tops = (group["base"] + group["epa"]).to_numpy()
            self.assertAlmostEqual(group["base"].iloc[0], 0.0)
            np.testing.assert_allclose(group["base"].to_numpy()[1:], tops[:-1])
            if positive:
                self.assertTrue((group["epa"] > 0).all())
            else:
                self.assertTrue((group["epa"] < 0).all())

    def test_player_game_vs_combined_units(self) -> None:
        week3 = sample_pbp()
        week4 = week3.assign(week=4, game_id="2025_04_KC_LV", defteam="LV")
        segments = gr.build_target_segments(pd.concat([week3, week4], ignore_index=True))
        per_game = gr.stack_segments(segments, gr.GROUP_PLAYER_GAME)
        combined = gr.stack_segments(segments, gr.GROUP_PLAYER)
        kelce_games = per_game[per_game["player_id"] == "P1"]["unit"].nunique()
        self.assertEqual(kelce_games, 2)
        self.assertEqual(combined[combined["player_id"] == "P1"]["unit"].nunique(), 1)

    def test_selection_and_min_targets(self) -> None:
        segments = gr.build_target_segments(sample_pbp())
        picked = gr.select_player_games(segments, [("P1", "2025_03_KC_BUF")])
        self.assertEqual(set(picked["player_id"]), {"P1"})
        team = gr.select_team_games(segments, [("2025_03_KC_BUF", "KC")])
        filtered = gr.filter_min_targets(team, gr.GROUP_PLAYER_GAME, 2)
        self.assertEqual(set(filtered["player_id"]), {"P1"})

    def test_figure_and_table(self) -> None:
        segments = gr.build_target_segments(sample_pbp())
        fig = gr.build_epa_figure(segments)
        names = {trace.name for trace in fig.data}
        self.assertTrue({"Reception", "YAC", "Interception", "Penalty", "Touchdown", "Net EPA"} <= names)
        table = gr.category_table(segments)
        self.assertAlmostEqual(table.loc[table.index.str.startswith("T.Kelce"), "Net EPA"].iloc[0], -1.3)
        self.assertEqual(int(table.loc[table.index.str.startswith("T.Kelce"), "TDs"].iloc[0]), 1)


if __name__ == "__main__":
    unittest.main()
