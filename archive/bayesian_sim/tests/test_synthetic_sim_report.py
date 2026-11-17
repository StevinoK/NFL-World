import pandas as pd

from synthetic_sim_report import default_input_folder, load_games, team_summary


def test_load_games_reads_ridge_dashboard_output_and_summarizes_neutral_results(tmp_path):
    games = pd.DataFrame([
        {"game_id": "2026_05_SYN_A", "week": 5, "simulation": 1, "home_team": "A",
         "away_team": "SYN", "neutral_site": True, "home_score": 24, "away_score": 17,
         "total": 41, "spread": 7, "home_win": 1.0},
        {"game_id": "2026_05_SYN_A", "week": 5, "simulation": 2, "home_team": "A",
         "away_team": "SYN", "neutral_site": True, "home_score": 10, "away_score": 20,
         "total": 30, "spread": -10, "home_win": 0.0},
    ])
    games.to_csv(tmp_path / "games.csv", index=False)
    pd.DataFrame([{"game_id": "2026_05_SYN_A", "simulation": 1, "player_name": "Player A"}]).to_csv(
        tmp_path / "players.csv", index=False
    )
    pd.DataFrame([{"game_id": "2026_05_SYN_A", "home_team": "A", "simulations": 2}]).to_csv(
        tmp_path / "summary.csv", index=False
    )
    loaded = load_games(tmp_path)
    summary = team_summary(loaded).iloc[0]
    assert loaded.neutral_site.all()
    assert summary.team == "A"
    assert summary.sims == 2
    assert summary.mean_spread == -1.5
    assert summary.win_prob == 0.5


def test_default_input_folder_prefers_latest_ridge_run(tmp_path, monkeypatch):
    import synthetic_sim_report

    weekly = tmp_path / "synthetic_home_week_05"
    older = weekly / "ridge-old"
    newer = weekly / "ridge-new"
    older.mkdir(parents=True)
    newer.mkdir()
    (older / "games.csv").write_text("games")
    (newer / "games.csv").write_text("games")
    (older / "games.csv").touch()
    (newer / "games.csv").touch()
    monkeypatch.setattr(synthetic_sim_report, "WORKSPACE_DIR", tmp_path)
    assert default_input_folder() == newer