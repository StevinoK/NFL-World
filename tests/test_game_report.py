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


def rush(play_id: int, **overrides) -> dict:
    row = play(play_id, play_type="run", receiver_player_id=None, receiver_player_name=None,
               rusher_player_id="R1", rusher_player_name="I.Pacheco", qb_kneel=0.0)
    row.update(overrides)
    return row


def rushing_pbp() -> pd.DataFrame:
    return pd.DataFrame([
        rush(101, desc="10-I.Pacheco up the middle to BUF 40 for 6 yards", epa=0.4),
        rush(102, desc="10-I.Pacheco left end for 12 yards, TOUCHDOWN", epa=2.1,
             touchdown=1.0, td_team="KC"),
        rush(103, desc="10-I.Pacheco right tackle to KC 30 for 2 yards. FUMBLES, RECOVERED by BUF",
             epa=-3.5, fumble_lost=1.0, fumbled_1_player_id="R1"),
        rush(104, desc="10-I.Pacheco left guard for 15 yards. PENALTY on KC-70-A.B, Offensive Holding, "
                       "10 yards - No Play.", play_type="no_play", epa=-0.8, penalty=1.0,
             rusher_player_id=None, rusher_player_name=None),
        rush(105, desc="15-P.Mahomes kneels to KC 30 for -1 yards.", play_type="qb_kneel", epa=-0.2,
             qb_kneel=1.0, rusher_player_id="QB", rusher_player_name="P.Mahomes"),
        play(106, desc="15-P.Mahomes pass short right to 10-I.Pacheco to BUF 30 for 4 yards",
             epa=0.3, air_epa=-0.2, yac_epa=0.5, complete_pass=1.0,
             receiver_player_id="R1", receiver_player_name="I.Pacheco"),
    ])


class RushingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.segments = gr.build_player_segments(rushing_pbp())

    def test_rush_categories(self) -> None:
        by_play = {pid: dict(zip(g["category"], g["epa"].round(6))) for pid, g in self.segments.groupby("play_id")}
        self.assertEqual(by_play[101], {"Rush": 0.4})
        self.assertEqual(by_play[102], {"Rush": 2.1})
        self.assertEqual(by_play[103], {"Fumble lost": -3.5})
        self.assertEqual(by_play[104], {"Penalty": -0.8})
        self.assertNotIn(105, by_play)  # kneel-downs excluded

    def test_nullified_run_resolves_carrier_and_kinds(self) -> None:
        rows = self.segments[self.segments["play_id"] == 104]
        self.assertEqual(rows["player_id"].tolist(), ["R1"])
        self.assertEqual(rows["kind"].tolist(), [gr.KIND_RUSH])
        kinds = self.segments.drop_duplicates("play_id").groupby("kind").size().to_dict()
        self.assertEqual(kinds, {gr.KIND_RUSH: 4, gr.KIND_TARGET: 1})

    def test_targets_only_builder_skips_rushes(self) -> None:
        targets = gr.build_target_segments(rushing_pbp())
        self.assertEqual(set(targets["play_id"]), {106})

    def test_table_counts_targets_and_rushes(self) -> None:
        table = gr.category_table(self.segments, gr.GROUP_PLAYER)
        row = table.iloc[0]
        self.assertEqual((int(row["Targets"]), int(row["Rushes"]), int(row["TDs"])), (1, 4, 1))
        self.assertAlmostEqual(row["Net EPA"], 0.4 + 2.1 - 3.5 - 0.8 + 0.3)
        catalog = gr.player_game_catalog(self.segments)
        self.assertIn("1 tgt, 4 rush", catalog["label"].iloc[0])


def qb_pbp() -> pd.DataFrame:
    def qb(play_id: int, **kw) -> dict:
        row = play(play_id, receiver_player_id=None, receiver_player_name=None, passer_player_id="QB1",
                   passer_player_name="P.Mahomes", sack=0.0, qb_scramble=0.0)
        row.update(kw)
        return row

    return pd.DataFrame([
        qb(201, desc="15-P.Mahomes sacked at KC 25 for -8 yards (99-X.Y).", sack=1.0, epa=-1.6),
        qb(202, desc="15-P.Mahomes sacked at KC 20 for -6 yards. FUMBLES, RECOVERED by BUF", sack=1.0,
           epa=-4.0, fumble_lost=1.0, fumbled_1_player_id="QB1"),
        qb(203, desc="15-P.Mahomes scrambles left end to BUF 30 for 9 yards", play_type="run", qb_scramble=1.0,
           rusher_player_id="QB1", rusher_player_name="P.Mahomes", passer_player_id=None, epa=0.9),
        qb(204, desc="15-P.Mahomes sacked at KC 30 for -5 yards. PENALTY on BUF-24-A.B, Defensive Holding, "
                     "5 yards - No Play.", play_type="no_play", penalty=1.0, epa=0.6, passer_player_id=None,
           passer_player_name=None),
        qb(205, desc="15-P.Mahomes scrambles up the middle for 12 yards. PENALTY on KC-70-A.B, Holding - No Play.",
           play_type="no_play", penalty=1.0, epa=-0.7, passer_player_id=None, passer_player_name=None),
        # A plain pass is not a QB segment (its EPA belongs to the target).
        qb(206, desc="15-P.Mahomes pass short left to 87-T.Kelce for 5 yards", complete_pass=1.0, epa=0.4,
           air_epa=0.1, yac_epa=0.3, receiver_player_id="P1", receiver_player_name="T.Kelce"),
    ])


class QuarterbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.segments = gr.build_player_segments(qb_pbp())

    def test_sacks_and_scrambles(self) -> None:
        qb = self.segments[self.segments["player_id"] == "QB1"]
        got = {(r.play_id, r.kind, r.category): round(r.epa, 6) for r in qb.itertuples()}
        self.assertEqual(got, {
            (201, gr.KIND_SACK, "Sack"): -1.6,
            (202, gr.KIND_SACK, "Fumble lost"): -4.0,
            (203, gr.KIND_SCRAMBLE, "Scramble"): 0.9,
            (204, gr.KIND_SACK, "Penalty"): 0.6,
            (205, gr.KIND_SCRAMBLE, "Penalty"): -0.7,
        })
        self.assertEqual(set(qb["position"]), {"QB"})  # inferred without a roster
        self.assertEqual(self.segments.loc[self.segments["play_id"] == 206, "player_id"].tolist(), ["P1", "P1"])

    def test_untargeted_passes_credit_the_passer(self) -> None:
        def qb(play_id: int, **kw) -> dict:
            row = play(play_id, receiver_player_id=None, receiver_player_name=None, passer_player_id="QB1",
                       passer_player_name="P.Mahomes", sack=0.0, qb_scramble=0.0)
            row.update(kw)
            return row

        pbp = pd.concat([qb_pbp(), pd.DataFrame([
            qb(301, desc="15-P.Mahomes pass incomplete short right.", incomplete_pass=1.0, epa=-0.5),
            qb(302, desc="15-P.Mahomes spiked the ball to stop the clock.", play_type="qb_spike",
               incomplete_pass=1.0, epa=-0.3),
            qb(303, desc="15-P.Mahomes pass incomplete short left. PENALTY on KC-15-P.Mahomes, "
                         "Intentional Grounding, 10 yards", incomplete_pass=1.0, penalty=1.0, epa=-1.9),
            qb(304, desc="15-P.Mahomes pass deep middle INTERCEPTED by 21-X.Y at KC 40.", interception=1.0,
               epa=-3.8),
            qb(305, desc="15-P.Mahomes pass incomplete deep right. PENALTY on BUF-24-A.B, Defensive Holding, "
                         "5 yards - No Play.", play_type="no_play", penalty=1.0, epa=0.9, passer_player_id=None,
               passer_player_name=None),
        ])], ignore_index=True)
        seg = gr.build_player_segments(pbp)
        untargeted = seg[seg["kind"] == gr.KIND_UNTARGETED]
        got = {(r.play_id, r.category): round(r.epa, 6) for r in untargeted.itertuples()}
        self.assertEqual(got, {
            (301, "Incomplete"): -0.5, (302, "Incomplete"): -0.3, (303, "Penalty"): -1.9,
            (304, "Interception"): -3.8, (305, "Penalty"): 0.9,
        })
        self.assertEqual(set(untargeted["player_id"]), {"QB1"})
        # Sacks and targeted passes are not double-counted as untargeted.
        self.assertFalse(set(untargeted["play_id"]) & {201, 202, 206})

    def test_kind_filter(self) -> None:
        only_targets = gr.build_player_segments(qb_pbp(), kinds=(gr.KIND_TARGET,))
        self.assertEqual(set(only_targets["play_id"]), {206})


class PositionTest(unittest.TestCase):
    def setUp(self) -> None:
        pbp = pd.concat([sample_pbp(), rushing_pbp(), qb_pbp()], ignore_index=True)
        roster = pd.DataFrame({
            "player_id": ["P1", "R1", "QB1", "Z9"],
            "position": ["TE", "RB", "QB", "WR"],
            "team": ["KC", "KC", "KC", "KC"],
            "jersey_number": [87, 10, 15, 11],
        })
        self.segments = gr.attach_roster(gr.build_player_segments(pbp), roster)

    def test_roster_positions_and_placeholder_resolution(self) -> None:
        positions = self.segments.drop_duplicates("player_id").set_index("player_id")["position"].to_dict()
        self.assertEqual(positions["P1"], "TE")
        self.assertEqual(positions["R1"], "RB")
        self.assertEqual(positions["QB1"], "QB")
        # Penalty-only "I.Pacheco" (#10) from sample_pbp resolves to roster player R1.
        self.assertFalse(self.segments["player_id"].astype(str).str.contains("#").any())

    def test_bars_grouped_by_position(self) -> None:
        stacked = gr.stack_segments(self.segments, gr.GROUP_PLAYER)
        order = gr.unit_order(stacked)
        spans = gr.position_spans(stacked, order)
        self.assertEqual([s[0] for s in spans], ["QB", "RB", "TE"])
        table = gr.category_table(self.segments, gr.GROUP_PLAYER)
        self.assertEqual(table["Pos"].tolist(), ["QB", "RB", "TE"])
        self.assertTrue({"Scrambles", "Sacks", "Sack", "Scramble"} <= set(table.columns))
        fig = gr.build_epa_figure(self.segments, gr.GROUP_PLAYER)
        headers = [a.text for a in fig.layout.annotations]
        self.assertEqual(headers, ["<b>QB</b>", "<b>RB</b>", "<b>TE</b>"])

    def test_position_group_aliases(self) -> None:
        self.assertEqual([gr.position_group(p) for p in ["FB", "hb", "WR", "OL", None]],
                         ["RB", "RB", "WR", "Other", "Other"])


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

    def test_select_drives(self) -> None:
        pbp = sample_pbp().assign(fixed_drive=[1, 1, 2, 2, 3, 3, 4, 4])
        segments = gr.build_target_segments(pbp)
        picked = gr.select_drives(segments, "2025_03_KC_BUF", [2, 3])
        self.assertEqual(sorted(picked["play_id"].unique()), [3, 4, 5, 6])
        self.assertTrue(gr.select_drives(segments, "other_game", [2]).empty)

    def test_figure_and_table(self) -> None:
        segments = gr.build_target_segments(sample_pbp())
        fig = gr.build_epa_figure(segments)
        names = {trace.name for trace in fig.data}
        self.assertTrue({"Reception", "YAC", "Interception", "Penalty", "Touchdown", "Net EPA"} <= names)
        table = gr.category_table(segments)
        self.assertAlmostEqual(table.loc[table.index.str.startswith("T.Kelce"), "Net EPA"].iloc[0], -1.3)
        self.assertEqual(int(table.loc[table.index.str.startswith("T.Kelce"), "TDs"].iloc[0]), 1)


class WeekRangeTest(unittest.TestCase):
    def test_range_spans_seasons_and_totals_per_player(self) -> None:
        frames = [
            sample_pbp().assign(season=2024, week=17, game_id="2024_17_KC_X"),
            sample_pbp().assign(week=2, game_id="2025_02_KC_Y"),
            sample_pbp(),  # 2025 week 3
            sample_pbp().assign(week=5, game_id="2025_05_KC_Z"),
        ]
        segments = gr.build_target_segments(pd.concat(frames, ignore_index=True))
        weeks = gr.week_options(segments)
        self.assertEqual(list(weeks.values()), [(2024, 17), (2025, 2), (2025, 3), (2025, 5)])
        catalog = gr.player_game_catalog(segments)
        in_range = gr.games_in_range(catalog[catalog["player_id"] == "P1"], (2024, 17), (2025, 3))
        self.assertEqual(set(in_range["game_id"]), {"2024_17_KC_X", "2025_02_KC_Y", "2025_03_KC_BUF"})
        picked = gr.select_player_games(segments, in_range[["player_id", "game_id"]].values)
        table = gr.category_table(picked, gr.GROUP_PLAYER)
        self.assertEqual(len(table), 1)
        self.assertAlmostEqual(table["Net EPA"].iloc[0], -1.3 * 3)
        by_season = gr.category_table(picked, gr.GROUP_PLAYER_SEASON)
        self.assertEqual(sorted(by_season.index), ["T.Kelce · KC 2024", "T.Kelce · KC 2025"])
        self.assertAlmostEqual(by_season.loc["T.Kelce · KC 2025", "Net EPA"], -1.3 * 2)
        self.assertEqual(int(by_season.loc["T.Kelce · KC 2025", "Targets"]), 12)


class PackingTest(unittest.TestCase):
    def test_squarify_tiles_exactly(self) -> None:
        areas = [6.0, 6.0, 4.0, 3.0, 2.0, 2.0, 1.0]
        rects = gr.squarify(areas, 0.0, 0.0, 6.0, 4.0)
        for area, (_, _, w, h) in zip(areas, rects):
            self.assertAlmostEqual(w * h, area)
        for x, y, w, h in rects:
            self.assertGreaterEqual(x, -1e-9)
            self.assertGreaterEqual(y, -1e-9)
            self.assertLessEqual(x + w, 6.0 + 1e-9)
            self.assertLessEqual(y + h, 4.0 + 1e-9)

    def test_packed_area_matches_epa_and_stays_in_slot(self) -> None:
        segments = gr.build_target_segments(sample_pbp())
        stacked = gr.stack_segments(segments, gr.GROUP_PLAYER_GAME)
        order = gr.unit_order(stacked)
        packed = gr.pack_segments(stacked, order, px_per_x=100.0, px_per_y=30.0)
        area = (packed["x1"] - packed["x0"]) * (packed["y1"] - packed["y0"])
        np.testing.assert_allclose(area / gr.SLOT_WIDTH, packed["epa"].abs(), rtol=1e-9)
        positions = packed["unit_label"].map({label: i for i, label in enumerate(order)})
        self.assertTrue((packed["x0"] >= positions - gr.SLOT_WIDTH / 2 - 1e-9).all())
        self.assertTrue((packed["x1"] <= positions + gr.SLOT_WIDTH / 2 + 1e-9).all())
        self.assertTrue((packed.loc[packed["positive"], "y0"] >= -1e-9).all())
        self.assertTrue((packed.loc[~packed["positive"], "y1"] <= 1e-9).all())

    def test_packed_bands_follow_fixed_category_order(self) -> None:
        frames = [sample_pbp(), rushing_pbp()]
        segments = gr.build_player_segments(pd.concat(frames, ignore_index=True))
        stacked = gr.stack_segments(segments, gr.GROUP_PLAYER_GAME)
        packed = gr.pack_segments(stacked, gr.unit_order(stacked), px_per_x=100.0, px_per_y=30.0)
        rank = {c: i for i, c in enumerate(gr.CATEGORIES)}
        for (_, positive), strip in packed.groupby(["unit", "positive"]):
            # Distance of each category band from the axis must increase with CATEGORIES order.
            near = strip.assign(d=strip["y0"].abs() if positive else strip["y1"].abs())
            bands = near.groupby("category")["d"].min().sort_values()
            ranks = [rank[c] for c in bands.index]
            self.assertEqual(ranks, sorted(ranks))
            full_width = strip.groupby("category").apply(lambda g: g["x1"].max() - g["x0"].min())
            np.testing.assert_allclose(full_width.to_numpy(), gr.SLOT_WIDTH)

    def test_both_layouts_build(self) -> None:
        segments = gr.build_target_segments(sample_pbp())
        for layout in (gr.LAYOUT_PACKED, gr.LAYOUT_STACKED):
            fig = gr.build_epa_figure(segments, layout=layout)
            self.assertIn("Net EPA", {trace.name for trace in fig.data})


if __name__ == "__main__":
    unittest.main()
