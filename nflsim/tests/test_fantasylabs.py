from __future__ import annotations

from nflsim.dfs.fantasylabs import lineups_table, players_table, user_lineups_table

PLAYER = {"playerId": 1, "fullName": "A B", "currentTeam": "BAL", "position": "QB", "salary": 10000, "projPoints": 20.0,
          "ownership": 34.5, "actualPoints": 18.2, "homeVisitor": "Home", "favDog": "Dog"}


def test_players_table_marks_captain_rows_and_converts_ownership():
    table = players_table({"players": {"1:0": {**PLAYER, "rosterPosition": "QB"},
                                       "1:1": {**PLAYER, "rosterPosition": "CPT", "salary": 15000, "ownership": 5.0}}})
    assert table.slot.tolist() == ["CPT", "FLEX"] and table.salary.tolist() == [15000, 10000]
    assert table.ownership.tolist() == [0.05, 0.345] and table.home.all() and not table.favorite.any()


def test_lineups_table_uses_slots_and_counts():
    lineup = {"lineupHash": "1:2:3:4:5:6", "lineupPlayers": {"CPT1": 1, "FLEX1": 2, "FLEX2": 3, "FLEX3": 4, "FLEX4": 5, "FLEX5": 6},
              "lineupCt": 7, "lineupUserCt": 5, "points": 90.5, "totalSalary": 49800, "lineupRank": 3, "payout": 100.0, "totalOwn": 210.0}
    table = lineups_table({"lineups": {"1:2:3:4:5:6": lineup}})
    row = table.iloc[0]
    assert row.cpt == 1 and row.flex5 == 6 and row.entries == 7 and row.users == 5 and row.total_ownership == 2.1


def test_user_lineups_table_lists_each_users_lineups():
    data = {"users": {"u1": {"lineups": {"a": {"lineupHash": "a", "lineupCt": 2}, "b": {"lineupHash": "b", "lineupCt": 1}}}}}
    table = user_lineups_table(data)
    assert table.to_dict("records") == [{"user": "u1", "lineup": "a", "entries": 2}, {"user": "u1", "lineup": "b", "entries": 1}]
