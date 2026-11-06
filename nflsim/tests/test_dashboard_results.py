from __future__ import annotations

import pandas as pd

from nflsim.dashboard.runs import synthetic_runs, weekly_runs
from nflsim.dashboard.synthetic_report import load_games, team_summary

GAMES = "game_id,home_team,away_team,home_score,away_score,spread,total\n"


def run(folder, game_id, *names):
    folder.mkdir(parents=True)
    for name in names:
        (folder / name).write_text(f"{GAMES}{game_id},A,B,24,17,7,41\n", encoding="utf-8")
    return folder


def test_runs_are_sorted_into_weekly_and_synthetic(tmp_path):
    weekly = run(tmp_path / "week9-2000", "2026_09_B_A", "games.csv", "players.csv")
    from_dashboard = run(tmp_path / "simulations" / "week_09", "2026_09_B_A", "games.csv", "players.csv")
    run(tmp_path / "unfinished", "2026_09_B_A", "games.csv")
    dashboard_synthetic = run(tmp_path / "simulations" / "synthetic_home_week_09" / "ridge-1", "2026_09_SYN_A", "games.csv", "players.csv")
    command_line = run(tmp_path / "synthetic" / "week9-syn-1000", "2026_09_SYN_A", "synthetic_games.csv")
    run(tmp_path / "unlabeled", "2026_09_SYN_A", "games.csv", "players.csv")
    assert set(weekly_runs(tmp_path)) == {weekly, from_dashboard}
    assert set(synthetic_runs(tmp_path)) == {dashboard_synthetic, command_line}


def test_report_reads_command_line_synthetic_runs(tmp_path):
    pd.DataFrame([
        {"game_id": "2026_05_SYN_A", "home_team": "A", "away_team": "SYN", "home_score": 24, "away_score": 17, "total": 41, "spread": 7},
        {"game_id": "2026_05_SYN_A", "home_team": "A", "away_team": "SYN", "home_score": 10, "away_score": 20, "total": 30, "spread": -10},
    ]).to_csv(tmp_path / "synthetic_games.csv", index=False)
    summary = team_summary(load_games(tmp_path)).iloc[0]
    assert summary.team == "A" and summary.sims == 2 and summary.win_prob == 0.5
