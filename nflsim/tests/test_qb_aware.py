import numpy as np
import pandas as pd

from nflsim.features import qb_aware


def plays(game_id, team, passer, epa, n=20):
    return pd.DataFrame({"game_id": game_id, "posteam": team, "season_type": "REG", "qb_dropback": 1, "epa": epa,
                         "passer_player_id": passer, "pass_oe": 0.0, "pass_attempt": 1, "rush_attempt": 0},
                        index=range(n))


def schedule_pbp():
    # AAA: veteran V for six good games, then backup B for two bad ones; BBB: one starter throughout.
    frames = [plays(f"2020_{w:02d}_AAA_BBB", "AAA", "V" if w <= 6 else "B", 0.3 if w <= 6 else -0.3) for w in range(1, 9)]
    frames += [plays(f"2020_{w:02d}_AAA_BBB", "BBB", "S", 0.0) for w in range(1, 9)]
    return pd.concat(frames, ignore_index=True)


PARAMS = {**qb_aware.DEFAULT, "first_season": 2020, "k": 10.0,
          "offsets": {"pass_epa": {"0": -0.1}, "pass_oe": {}}, "betas": {"epa_rate": 2.0}}


def test_new_starter_shifts_only_his_team_and_only_mapped_columns(tmp_path):
    schedule_pbp().to_parquet(tmp_path / "pbp.parquet")
    table = pd.DataFrame({"game_id": ["2020_07_AAA_BBB", "2020_07_AAA_BBB", "2020_05_AAA_BBB"],
                          "team": ["AAA", "BBB", "AAA"], "epa_rate": [0.1, 0.1, 0.1], "other": [1.0, 1.0, 1.0]})
    out = qb_aware.apply_qb_aware(table, PARAMS, tmp_path)
    assert out.other.tolist() == [1.0, 1.0, 1.0]
    assert out.epa_rate[0] < 0.1 - 0.05  # backup's first start: history barely counts, plus the 0-starts offset
    assert np.isclose(out.epa_rate[1], 0.1)  # same starter all along: no shift
    assert np.isclose(out.epa_rate[2], 0.1, atol=0.02)  # veteran's fifth start (bucket 3-5, no offset)


def test_unplayed_game_uses_depth_chart_starter(tmp_path):
    schedule_pbp().to_parquet(tmp_path / "pbp.parquet")
    roster = pd.DataFrame({"game_id": ["2020_09_AAA_BBB"] * 2, "team": "AAA", "player_id": ["V", "B"],
                           "position": "QB", "depth_team": [1.0, 2.0]})
    roster.to_parquet(tmp_path / "roster.parquet")
    table = pd.DataFrame({"game_id": ["2020_09_AAA_BBB"], "team": ["AAA"], "epa_rate": [0.1]})
    out = qb_aware.apply_qb_aware(table, {**PARAMS, "offsets": {"pass_epa": {}, "pass_oe": {}}}, tmp_path)
    # V back at QB1: his six good games outweigh B's two, tilting the history up from the team blend
    assert out.epa_rate[0] > 0.1 + 0.05
