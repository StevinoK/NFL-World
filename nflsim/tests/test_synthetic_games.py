import numpy as np
import pandas as pd

from nflsim.sim.synthetic_games import build_synthetic_game, synthetic_game_row, synthetic_team_options


def test_cli_recreates_output_directory_removed_during_simulation(monkeypatch, tmp_path):
    import json
    import sys
    from types import SimpleNamespace

    from nflsim import bundle
    from nflsim.sim import synthetic_games

    schedule, teams, roster, depth, players = inputs()
    game_id = "2024_02_SYN_A"
    destination = tmp_path / "outputs/synthetic/recreated"
    runtime = SimpleNamespace(player_stats_wide=players, team_stats_wide=teams)
    game_rows = pd.DataFrame([{
        "game_id": game_id, "home_team": "A", "away_team": "SYN",
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
        "--teams", "A", "--simulations", "1", "--run-id", "recreated",
    ])
    monkeypatch.setattr(bundle, "validate_bundle", lambda *args: None)
    monkeypatch.setattr(bundle, "verify_bundle_hash", lambda *args: "test-hash")
    monkeypatch.setattr(synthetic_games, "ROOT", tmp_path)
    monkeypatch.setattr(synthetic_games, "RidgeRuntime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(synthetic_games, "_load_inputs", lambda *args: (schedule, roster, depth))
    monkeypatch.setattr(synthetic_games, "build_synthetic_game", lambda *args, **kwargs: SimpleNamespace(game_id=game_id))
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


def test_synthetic_inputs_use_neutral_site_and_zero_venue_effects():
    runtime = Runtime()
    schedule, teams, roster, depth, players = inputs()
    game = build_synthetic_game(runtime, players, teams, roster, depth, schedule, "A", schedule.iloc[1], 2)
    home = game.team_features["A"]
    synthetic = game.team_features["SYN"]
    assert home["ridge_yac_oe_venue_effect"] == 0
    assert synthetic["ridge_yac_oe_venue_effect"] == 0
    assert game.neutral_site is True
    assert np.isclose(synthetic["ridge_yac_oe_offense_effect"], 0.1)
    assert np.isclose(synthetic["epa_rate"], 5.0 / 3.0)
    assert len(game.away_players) == 11


def test_synthetic_baseline_uses_latest_equal_weighted_league_snapshots():
    runtime = Runtime()
    schedule, teams, roster, depth, players = inputs()
    old_team_row = teams.loc[teams.team.eq("B")].iloc[[0]].copy()
    old_team_row["game_id"] = "2024_00_B_C"
    old_team_row["game_date"] = "2024-08-25"
    old_team_row["epa_rate"] = 99.0
    teams = pd.concat([teams, old_team_row], ignore_index=True)

    players.loc[players.player_id.eq("QB1"), "yac_bayes"] = 3.0
    old_qb_row = players.loc[players.player_id.eq("QB1")].iloc[[0]].copy()
    old_qb_row["game_id"] = "2024_00_B_C"
    old_qb_row["game_date"] = "2024-08-25"
    old_qb_row["yac_bayes"] = 99.0
    players = pd.concat([players, old_qb_row], ignore_index=True)

    other_roster = roster.copy()
    other_roster["team"] = "B"
    other_roster["game_id"] = schedule.iloc[1].game_id
    other_roster["player_id"] = "B_" + other_roster.player_id
    roster = pd.concat([roster, other_roster], ignore_index=True)
    other_players = players.sort_values(["game_date", "game_id"]).drop_duplicates("player_id", keep="last").copy()
    other_players["player_id"] = "B_" + other_players.player_id
    players = pd.concat([players, other_players], ignore_index=True)

    game = build_synthetic_game(runtime, players, teams, roster, depth, schedule, "A", schedule.iloc[1], 1)

    assert np.isclose(game.team_features["SYN"]["epa_rate"], 5.0 / 3.0)
    synthetic_by_slot = {player["position"]: player["attributes"] for player in game.away_players}
    assert synthetic_by_slot["QB1"]["yac_bayes"] == 3.0
    assert synthetic_by_slot["WR1"]["yac_bayes"] == 1.0


def test_synthetic_baseline_includes_tested_team_current_week_inputs():
    runtime = Runtime()
    schedule, teams, roster, depth, players = inputs()
    current_game = schedule.iloc[1]

    current_team_rows = teams[teams.game_id.eq(current_game.game_id)].copy()
    current_team_rows["epa_rate"] = current_team_rows.team.map({"A": 100.0, "B": 8.0})
    teams.loc[teams.game_id.eq(current_game.game_id), "epa_rate"] = current_team_rows["epa_rate"].to_numpy()

    other_roster = roster.copy()
    other_roster["team"] = "B"
    other_roster["game_id"] = current_game.game_id
    other_roster["player_id"] = "B_" + other_roster.player_id
    roster = pd.concat([roster, other_roster], ignore_index=True)

    current_player_rows = players.copy()
    current_player_rows["game_id"] = current_game.game_id
    current_player_rows["game_date"] = current_game.gameday
    current_player_rows["player_id"] = "B_" + current_player_rows.player_id
    current_player_rows["yac_bayes"] = 8.0
    players = pd.concat([players, current_player_rows], ignore_index=True)

    game = build_synthetic_game(runtime, players, teams, roster, depth, schedule, "A", current_game, 1,
                                use_model_rank=False)

    assert np.isclose(game.team_features["SYN"]["epa_rate"], (100.0 + 8.0 + 1.0) / 3.0)
    synthetic_by_slot = {player["position"]: player["attributes"] for player in game.away_players}
    assert np.isclose(synthetic_by_slot["QB1"]["yac_bayes"], 4.5)


def test_synthetic_selection_includes_away_and_bye_teams():
    schedule, teams, roster, depth, players = inputs()
    assert synthetic_team_options(schedule) == ["A", "B", "C"]
    assert synthetic_game_row(schedule, "B", 2).game_id == "2024_02_A_B"
    bye_row = synthetic_game_row(schedule, "C", 2)
    assert bye_row.gameday == pd.Timestamp("2024-09-08")
    assert bye_row.week == 2
    roster["team"] = "C"
    game = build_synthetic_game(Runtime(), players, teams, roster, depth, schedule, "C", bye_row, 1)
    assert game.home_team == "C"
    assert game.neutral_site
    assert len(game.home_players) == 11


def test_neutral_synthetic_game_does_not_require_home_stadium_history():
    schedule, teams, roster, depth, players = inputs()
    schedule = schedule.drop(columns="stadium_id")
    game = build_synthetic_game(Runtime(), players, teams, roster, depth, schedule, "A", schedule.iloc[1], 10)
    from nflsim.sim.execution import build_ridge_tasks
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


def test_future_feature_perturbations_do_not_change_synthetic_baseline():
    runtime = Runtime()
    schedule, teams, roster, depth, players = inputs()
    baseline = build_synthetic_game(runtime, players, teams, roster, depth, schedule, "A", schedule.iloc[1], 1)
    changed_teams = teams.copy()
    changed_players = players.copy()
    future_schedule = pd.DataFrame([{"game_id": "2024_03_A_D", "season": 2024, "week": 3,
                                     "gameday": "2024-09-15", "home_team": "A", "away_team": "D",
                                     "stadium_id": "STADIUM-A", "roof": "outdoors", "game_type": "REG"}])
    future_team_rows = teams[teams.game_id.eq("2024_02_A_B")].copy()
    future_team_rows["game_id"], future_team_rows["game_date"] = "2024_03_A_D", "2024-09-15"
    future_team_rows[["epa_rate", "ridge_yac_oe_venue_effect"]] = 999
    changed_teams = pd.concat([changed_teams, future_team_rows], ignore_index=True)
    future_player_rows = players.copy()
    future_player_rows["game_id"], future_player_rows["game_date"] = "2024_03_A_D", "2024-09-15"
    future_player_rows[["yac_bayes", "ridge_yac_oe_venue_effect"]] = 999
    changed_players = pd.concat([changed_players, future_player_rows], ignore_index=True)
    future_schedule = pd.concat([schedule, future_schedule], ignore_index=True)
    perturbed = build_synthetic_game(runtime, changed_players, changed_teams, roster, depth, future_schedule, "A", schedule.iloc[1], 1)
    assert baseline.team_features["SYN"]["epa_rate"] == perturbed.team_features["SYN"]["epa_rate"]
    assert baseline.team_features["SYN"]["ridge_yac_oe_venue_effect"] == perturbed.team_features["SYN"]["ridge_yac_oe_venue_effect"]
    assert baseline.away_players[0]["attributes"]["yac_bayes"] == perturbed.away_players[0]["attributes"]["yac_bayes"]


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
