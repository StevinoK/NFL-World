import numpy as np
import pandas as pd

from nflsim.sim.synthetic_games import (LeagueTeam, build_league_games, league_game, league_pairs, prepare_league_team,
                                        synthetic_game_row, synthetic_team_options, team_games, team_table)


def test_cli_recreates_output_directory_removed_during_simulation(monkeypatch, tmp_path):
    import json
    import sys
    from types import SimpleNamespace

    from nflsim import bundle
    from nflsim.sim import synthetic_games

    schedule, teams, roster, depth, players = inputs()
    game_id = "2024_02_PAIR_A_B"
    destination = tmp_path / "outputs/synthetic/recreated"
    runtime = SimpleNamespace(player_stats_wide=players, team_stats_wide=teams)
    game_rows = pd.DataFrame([{
        "game_id": game_id, "home_team": "A", "away_team": "B",
        "simulation": 1, "home_score": 21, "away_score": 14,
        "spread": 7, "total": 35,
    }])

    def make_output(path):
        path.mkdir(parents=True, exist_ok=True)
        return path

    def simulate(*args, **kwargs):
        assert destination.is_dir()
        destination.rmdir()
        return {game_id: (game_rows, pd.DataFrame([{"player_name": "QB1"}]))}

    monkeypatch.setattr(sys, "argv", [
        "synthetic_games", "--backend", "local", "--season", "2024", "--week", "2",
        "--teams", "A", "--simulations", "2", "--run-id", "recreated",
    ])
    monkeypatch.setattr(bundle, "validate_bundle", lambda *args: None)
    monkeypatch.setattr(bundle, "verify_bundle_hash", lambda *args: "test-hash")
    monkeypatch.setattr(synthetic_games, "ROOT", tmp_path)
    monkeypatch.setattr(synthetic_games, "RidgeRuntime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(synthetic_games, "_load_inputs", lambda *args: (schedule, roster, depth))
    monkeypatch.setattr(synthetic_games, "build_league_games", lambda *args, **kwargs: ([SimpleNamespace(game_id=game_id, home_team="A")], []))
    monkeypatch.setattr(synthetic_games, "execute_ridge_games", simulate)
    monkeypatch.setattr(synthetic_games, "output_dir", make_output)
    monkeypatch.setattr(synthetic_games, "write_json", lambda path, value: path.write_text(json.dumps(value)))

    synthetic_games.main()

    assert pd.read_csv(destination / "synthetic_games.csv").simulation.tolist() == [1]
    manifest = json.loads((destination / "run_manifest.json").read_text())
    assert manifest["games_completed"] == 1
    assert manifest["failures"] == []


class Runtime:
    required_player_feature_columns = ["yac_bayes", "medium", "rush_rate", "ridge_yac_oe_venue_effect"]
    required_team_feature_columns = ["epa_rate", "days_rest", "opp_days_rest", "ridge_yac_oe_venue_effect", "ridge_yac_oe_offense_effect"]
    context_map = {"yac": ["ridge_yac_oe_venue_effect", "ridge_yac_oe_offense_effect"]}

    @staticmethod
    def checked_context(model, row):
        values = {key: float(row[key]) for key in Runtime.context_map[model]}
        if not np.isfinite(list(values.values())).all():
            raise ValueError("invalid context")
        return values


def inputs():
    schedule = pd.DataFrame([
        {"game_id": "2024_01_A_C", "season": 2024, "week": 1, "gameday": "2024-09-01", "home_team": "A", "away_team": "C", "stadium_id": "STADIUM-A", "roof": "outdoors", "game_type": "REG"},
        {"game_id": "2024_02_A_B", "season": 2024, "week": 2, "gameday": "2024-09-08", "home_team": "A", "away_team": "B", "stadium_id": "STADIUM-A", "roof": "outdoors", "game_type": "REG"},
    ])
    team_rows = []
    for game_id, date, venue, epa in (("2024_01_A_C", "2024-09-01", .2, 1), ("2024_02_A_B", "2024-09-08", .8, 2)):
        for team in ("A", "B", "C"):
            team_rows.append({"game_id": game_id, "game_date": date, "team": team, "epa_rate": epa,
                              "days_rest": 7, "opp_days_rest": 7,
                              "ridge_yac_oe_venue_effect": venue,
                              "ridge_yac_oe_offense_effect": .1})
    team_stats = pd.DataFrame(team_rows)
    player_rows, roster_rows = [], []
    slot_counts = {"QB": 1, "WR": 4, "RB": 3, "TE": 3}
    for position, count in slot_counts.items():
        for index in range(1, count + 1):
            player_id = f"{position}{index}"
            roster_rows.append({"game_id": "2024_02_A_B", "team": "A", "player_id": player_id,
                                "player_name": player_id, "position": position, "status": "ACT", "week": 2})
            player_rows.append({"game_id": "2024_01_A_C", "player_id": player_id, "game_date": "2024-09-01",
                                "yac_bayes": 1.0, "medium": .5, "rush_rate": .3,
                                "ridge_yac_oe_venue_effect": .2})
    player_stats = pd.DataFrame(player_rows)
    roster, depth = pd.DataFrame(roster_rows), pd.DataFrame()
    return schedule, team_stats, roster, depth, player_stats


def league_team(name, epa):
    return LeagueTeam(name, 2024, {"epa_rate": epa, "opp_pass_epa_rate": 99.0}, [{"player": name}])


def test_league_field_includes_each_team_itself():
    assert league_pairs(["A", "B", "C"]) == [("A", "A"), ("A", "B"), ("A", "C"), ("B", "B"), ("B", "C"), ("C", "C")]


def test_selected_team_still_plays_the_whole_field():
    assert league_pairs(["A", "B", "C"], ["B"]) == [("A", "B"), ("B", "B"), ("B", "C")]


def test_pairing_is_neutral_and_uses_the_real_opponent():
    game = league_game(league_team("A", 1.0), league_team("B", 2.0), 3, 64)
    assert game.game_id == "2024_03_PAIR_A_B" and game.simulations == 64
    assert game.neutral_site and game.swap_home_away_halfway and game.roof == "outdoors"
    assert game.team_features["A"]["opp_pass_epa_rate"] == 2.0 and game.team_features["B"]["opp_pass_epa_rate"] == 1.0


def test_team_against_itself_counts_as_one_opponent():
    game = league_game(league_team("A", 1.0), league_team("A", 1.0), 3, 64)
    assert game.simulations == 32 and game.home_team == game.away_team == "A"
    assert game.team_features["A"]["opp_pass_epa_rate"] == 1.0
    rows = pd.DataFrame({"home_team": ["A"] * game.simulations, "away_team": "A", "home_score": 21, "away_score": 14, "spread": 7, "total": 35})
    seen = team_games(rows)
    assert len(seen) == 64 and seen.spread.mean() == 0


def test_team_games_keeps_only_teams_that_faced_the_whole_field():
    rows = pd.DataFrame([("A", "A", 20, 10), ("A", "B", 30, 20), ("A", "C", 24, 14)], columns=["home_team", "away_team", "home_score", "away_score"])
    rows = rows.assign(spread=rows.home_score - rows.away_score, total=rows.home_score + rows.away_score)
    seen = team_games(rows)
    assert set(seen.home_team) == {"A"} and set(seen.away_team) == {"A", "B", "C"}
    table = team_table(rows).iloc[0]
    assert table.team == "A" and table.simulations == 4 and table.margin_mean == 5


def test_team_games_reads_old_runs_without_the_averaged_opponent():
    rows = pd.DataFrame([("A", "SYN", 24, 17), ("B", "SYN", 10, 20)], columns=["home_team", "away_team", "home_score", "away_score"])
    rows = rows.assign(spread=rows.home_score - rows.away_score, total=rows.home_score + rows.away_score)
    assert sorted(team_games(rows).home_team) == ["A", "B"]


def test_league_team_has_no_venue_effect_and_ignores_later_games():
    schedule, teams, roster, depth, players = inputs()
    later = teams[teams.game_id.eq("2024_02_A_B")].assign(game_id="2024_03_A_B", game_date="2024-09-15", epa_rate=50)
    team = prepare_league_team(Runtime(), players, pd.concat([teams, later]), roster, depth, schedule, "A", 2)
    assert team.features["ridge_yac_oe_venue_effect"] == 0.0
    assert team.features["epa_rate"] == 2
    assert team.features["opp_days_rest"] == 7.0
    assert len(team.players) == 11


def test_bye_team_is_rated_from_its_latest_inputs():
    schedule, teams, roster, depth, players = inputs()
    assert synthetic_team_options(schedule) == ["A", "B", "C"]
    assert synthetic_game_row(schedule, "B", 2).game_id == "2024_02_A_B"
    bye_row = synthetic_game_row(schedule, "C", 2)
    assert bye_row.gameday == pd.Timestamp("2024-09-08")
    assert bye_row.week == 2
    roster["team"] = "C"
    assert prepare_league_team(Runtime(), players, teams, roster, depth, schedule, "C", 2).name == "C"


def test_a_team_that_cannot_be_prepared_is_reported_and_left_out(monkeypatch):
    from nflsim.sim import synthetic_games
    schedule, teams, roster, depth, players = inputs()

    def prepare(runtime, player_stats, team_stats, roster, depth, schedule, team, week, rank):
        if team == "C":
            raise ValueError("no lineup")
        return league_team(team, 1.0)

    monkeypatch.setattr(synthetic_games, "prepare_league_team", prepare)
    games, failures = build_league_games(None, players, teams, roster, depth, schedule, 2, 4)
    assert failures == [{"team": "C", "error": "no lineup"}]
    assert [game.game_id for game in games] == ["2024_02_PAIR_A_A", "2024_02_PAIR_A_B", "2024_02_PAIR_B_B"]


def test_league_game_becomes_neutral_tasks_that_swap_halfway():
    from nflsim.sim.execution import build_ridge_tasks
    game = league_game(league_team("A", 1.0), league_team("B", 2.0), 3, 10)
    tasks = build_ridge_tasks([game], "bundle", "hash", "neutral-run", 42, 2)
    assert all(task["neutral_site"] is True for task in tasks)
    assert all(task["swap_home_away_halfway"] is True for task in tasks)
    assert all(task["swap_after_simulation"] == 5 for task in tasks)
    assert sorted(simulation for task in tasks for simulation in task["simulation_ids"]) == list(range(1, 11))


def test_neutral_runtime_uses_same_home_away_encoding_for_both_sides():
    from nflsim.sim.runtime import RidgeRuntime

    class Processor:
        feature_names_in_ = np.array(["posteam_type", "strength"])

        def transform(self, frame):
            home = frame.posteam_type.eq("home").astype(float).to_numpy()
            return np.column_stack([home, 1.0 - home, frame.strength.to_numpy()])

    runtime = RidgeRuntime.__new__(RidgeRuntime)
    runtime.context_map = {}
    runtime.processor = {"play": Processor()}
    runtime._record_nan_columns = lambda *args: None
    runtime.neutral_site = True
    frame = pd.DataFrame({"posteam_type": ["home", "away"], "strength": [3.0, 4.0]})
    assert np.array_equal(runtime.transform_with_processor("play", frame), [[.5, .5, 3.0], [.5, .5, 4.0]])
    runtime.neutral_site = False
    assert np.array_equal(runtime.transform_with_processor("play", frame), [[1.0, 0.0, 3.0], [0.0, 1.0, 4.0]])


def test_load_inputs_uses_live_current_season_when_snapshot_is_stale(tmp_path, monkeypatch):
    import nfl_data_py as nfl
    from nflsim.sim.synthetic_games import _load_inputs
    snapshot = tmp_path / "weekly_inputs"
    snapshot.mkdir()
    pd.DataFrame([{"game_id": "2025_01_A_B", "season": 2025, "week": 1,
                   "gameday": "2025-09-01", "home_team": "A", "away_team": "B",
                   "game_type": "REG", "stadium_id": "OLD"}]).to_parquet(snapshot / "schedule.parquet")
    pd.DataFrame([{"game_id": "2025_01_A_B", "season": 2025, "week": 1,
                   "team": "A", "player_id": "OLD", "position": "QB", "status": "ACT"}]).to_parquet(snapshot / "roster.parquet")
    monkeypatch.setattr(nfl, "import_schedules", lambda years: pd.DataFrame([{
        "game_id": "2026_06_PHI_DAL", "season": 2026, "week": 6, "gameday": "2026-10-18",
        "home_team": "DAL", "away_team": "PHI", "game_type": "REG", "stadium_id": "DAL-STADIUM",
    }]))
    monkeypatch.setattr(nfl, "import_weekly_rosters", lambda years: pd.DataFrame([{
        "season": 2026, "week": 6, "team": "DAL", "player_id": "QB1", "position": "QB",
        "status": "ACT", "full_name": "Quarterback One",
    }]))
    monkeypatch.setattr(nfl, "import_depth_charts", lambda years: pd.DataFrame([{
        "dt": "2026-10-05T15:53:14Z", "team": "DAL", "gsis_id": "QB1", "pos_abb": "QB", "pos_rank": 1,
    }]))
    schedule, roster, depth = _load_inputs(2026, snapshot)
    assert schedule.game_id.tolist() == ["2026_06_DAL_PHI"] or schedule.game_id.tolist() == ["2026_06_PHI_DAL"]
    assert roster.game_id.eq(schedule.game_id.iloc[0]).all()
    assert roster.depth_team.iloc[0] == 1
    assert {"dt", "team", "gsis_id", "pos_abb", "pos_rank"} <= set(depth.columns)
