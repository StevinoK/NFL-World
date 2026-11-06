import numpy as np
import pandas as pd

from nflsim.features import usage
from nflsim.features.slots import rank_slots


def roster_rows(**overrides):
    rows = [
        {"player_id": "QB-A", "position": "QB", "depth_team": 1, "status": "ACT"},
        {"player_id": "QB-B", "position": "QB", "depth_team": 2, "status": "ACT"},
        {"player_id": "RB-IR", "position": "RB", "depth_team": 1, "status": "RES", "rush_rate": 0.9},
        {"player_id": "RB-OUT", "position": "RB", "depth_team": 1, "status": "ACT", "report_status": "Out", "rush_rate": 0.8},
        {"player_id": "RB-A", "position": "RB", "depth_team": 2, "status": "ACT", "rush_rate": 0.3},
        {"player_id": "FB-A", "position": "FB", "depth_team": 3, "status": "ACT", "rush_rate": 0.4},
        {"player_id": "WR-A", "position": "WR", "depth_team": 2, "status": "ACT", "medium": 0.2},
        {"player_id": "WR-B", "position": "WR", "depth_team": 1, "status": "ACT", "medium": 0.2},
    ]
    frame = pd.DataFrame(rows).assign(game_id="G", team="T")
    for column, value in overrides.items():
        frame[column] = value
    return frame


def test_rank_slots_uses_only_active_players_and_breaks_ties_by_depth():
    slots = rank_slots(roster_rows()).set_index("slot").player_id.to_dict()
    assert slots["QB1"] == "QB-A"
    # Injured-reserve and Out backs never take a slot; a fullback competes as an RB.
    assert slots["RB1"] == "FB-A" and slots["RB2"] == "RB-A" and "RB3" not in slots
    assert slots["WR1"] == "WR-B" and slots["WR2"] == "WR-A"


def test_rank_slots_falls_back_to_all_rows_without_status():
    slots = rank_slots(roster_rows(status="DEV")).set_index("slot").player_id.to_dict()
    assert slots["RB1"] == "RB-IR"


def injury_season(games=10, out=(3, 4, 5)):
    """RB1 takes 70% of carries and RB2 20% when both play; RB1 misses `out` and RB2 takes over."""
    roster, counts, played = [], [], set()
    for week in range(1, games+1):
        game_id = f"2020_{week:02d}_T_O"
        for player, depth in (("RB1", 1), ("RB2", 2), ("RB3", 3)):
            hurt = player == "RB1" and week in out
            roster.append({"game_id": game_id, "team": "T", "player_id": player, "position": "RB", "status": "INA" if hurt else "ACT",
                           "report_status": "Out" if hurt else None, "expected": not hurt, "depth": depth, "pick": 100.0})
            if not hurt:
                played.add((game_id, "T", player))
        shares = {"RB2": 21, "RB3": 7} if week in out else {"RB1": 21, "RB2": 6, "RB3": 3}
        counts += [{"event": "rush", "game_id": game_id, "team": "T", "player_id": player, "count": n} for player, n in shares.items()]
    return pd.DataFrame(roster), played, pd.DataFrame(counts)


def test_backup_share_is_not_inflated_by_the_starters_absence():
    roster, played, counts = injury_season()
    params = usage.Params(mean={"QB": 0.0, "RB": 0.0, "WR": 0.0, "TE": 0.0}, draft={p: 0.0 for p in usage.POSITIONS},
                          depth=(0.0, 0.0, 0.0), prior_var=4.0, week_var=0.01, season_var=0.2)
    emitted, _ = usage.run("rush", params, roster, played, counts)
    share = emitted.pivot(index="game_id", columns="player_id", values="share")
    before, after = "2020_03_T_O", "2020_07_T_O"
    # While RB1 is out the backups split his work; when he returns his share is back near
    # where it was and RB2's is no higher than before the injury.
    assert share.loc["2020_04_T_O", "RB2"] > 0.6
    assert abs(share.loc[after, "RB1"] - share.loc[before, "RB1"]) < 0.08
    assert share.loc[after, "RB2"] < share.loc[before, "RB2"] + 0.03
    # The emitted share for an inactive player is his share if he were added back in.
    assert 0.5 < share.loc["2020_04_T_O", "RB1"] < 0.9
    assert np.allclose(share.loc[after, ["RB1", "RB2", "RB3"]].sum(), 1.0)


def test_team_games_with_an_empty_slot_stay_in_training():
    from nflsim.engine.simulation_runtime import SKILL_SLOT_ORDER
    from nflsim.features.training_data import slot_frame
    roster = pd.DataFrame([{"game_id": game, "team": "A", "player_id": "A"+slot, "position": slot[:-1], "depth_team": int(slot[-1]), "report_status": "", "status": "ACT"}
                           for game in ("G1", "G2") for slot in SKILL_SLOT_ORDER])
    roster.loc[roster.player_id.eq("ATE3") & roster.game_id.eq("G1"), "status"] = "INA"
    players = roster[["game_id", "player_id"]].assign(rush_rate=0.2, medium=0.1, ypt=7.0)
    frame = pd.DataFrame([{"game_id": game, "posteam": "A", "receiver_player_id": "AWR2"} for game in ("G1", "G2")])
    result = slot_frame(frame, players, roster, "receiver", ["medium", "ypt"]).set_index("game_id")
    # The team-game without an active third TE keeps its rows; its empty slot has no usage.
    assert pd.isna(result.loc["G1", "TE3"]) and result.selection_label.eq(SKILL_SLOT_ORDER.index("WR2")).all()
    assert result.loc["G1", "medium_TE3"] == 0.0 and result.loc["G2", "medium_TE3"] == 0.1
    assert result.loc["G1", "ypt_TE3"] == 7.0
