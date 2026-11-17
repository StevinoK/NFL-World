import numpy as np
import pandas as pd

from nflsim.sim import joint_outcomes

STATS = sorted({stat for stats in joint_outcomes.SLOT_STATS.values() for stat in stats})


def fake_run(directory, games=8, simulations=60, seed=0):
    rng = np.random.default_rng(seed)
    raw = directory / "raw_games"
    raw.mkdir(parents=True)
    results, plays = [], []
    for g in range(games):
        game_id, home, away = f"2025_01_A{g}_H{g}", f"H{g}", f"A{g}"
        scores, players = [], []
        for simulation in range(1, simulations+1):
            points = {}
            for team in (home, away):
                volume = rng.normal()
                points[team] = 20 + 6*volume
                for slot in joint_outcomes.SLOT_STATS:
                    row = {"game_id": game_id, "team": team, "simulation": simulation, "position": slot, "player_id": f"{team}-{slot}", **{stat: 0.0 for stat in STATS}}
                    row["passing_yards" if slot == "QB1" else "receiving_yards"] = 200 + 50*volume if slot == "QB1" else 60 + 20*volume + 10*rng.normal()
                    players.append(row)
            scores.append({"game_id": game_id, "simulation": simulation, "home_team": home, "away_team": away, "home": points[home], "away": points[away]})
        pd.DataFrame(scores).to_csv(raw / f"{game_id}.csv", index=False)
        pd.DataFrame(players).to_csv(raw / f"{game_id}_players.csv", index=False)
        results.append({"game_id": game_id, "home_team": home, "away_team": away, "actual_home_score": 24.0, "actual_away_score": 17.0})
        for team in (home, away):
            volume = rng.normal()
            for slot in ("WR1", "WR2", "TE1"):
                yards = 60 + 20*volume
                plays.append({"game_id": game_id, "posteam": team, "season_type": "REG", "two_point_attempt": 0, "pass_attempt": 1, "sack": 0, "rush_attempt": 0, "complete_pass": 1, "passer_player_id": f"{team}-QB1", "receiver_player_id": f"{team}-{slot}", "rusher_player_id": None, "passing_yards": yards, "receiving_yards": yards, "rushing_yards": 0.0, "pass_touchdown": 0, "rush_touchdown": 0, "interception": 0})
    pd.DataFrame(results).to_csv(directory / "backtest_2025_results.csv", index=False)
    return pd.DataFrame(plays)


def test_joint_outcomes_recover_simulated_dependence(tmp_path):
    pbp = fake_run(tmp_path)
    scores, margin, pairs, dependence = joint_outcomes.evaluate([tmp_path], pbp)
    margin = margin.set_index("outcome")
    # Every simulated team-game has a matching actual; RB1 never touched the ball and drops out.
    assert margin.loc["QB1.passing_yards", "team_games"] == 16
    assert "RB1.rushing_yards" not in margin.index
    assert not margin.index.str.startswith("opp.").any()
    assert margin.loc["team.win", "team_games"] == 16
    dependence = dependence.set_index("pair")
    assert dependence.loc["QB1.passing_yards x WR1.receiving_yards", "sim_rank_corr"] > 0.8
    assert dependence.loc["WR1.receiving_yards x WR2.receiving_yards", "sim_rank_corr"] > 0.5
    assert dependence.loc["team.points x opp.team.points", "sim_rank_corr"] < 0.3
    assert dependence.loc["QB1.passing_yards x WR1.receiving_yards", "dependence_gain"] > 0
    ranked = joint_outcomes.priorities(margin.reset_index(), dependence.reset_index(), "fake")
    assert set(ranked.kind) == {"dispersion", "dependence"}


def test_crps_matches_definition():
    rng = np.random.default_rng(1)
    x = rng.normal(size=40)
    brute = np.mean(np.abs(x-0.3)) - 0.5*np.mean(np.abs(x[:, None]-x[None, :]))
    assert np.isclose(joint_outcomes.crps(x, 0.3), brute)
