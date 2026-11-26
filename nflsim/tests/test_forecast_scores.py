import numpy as np

from nflsim.sim.forecast_scores import crps, energy_score, variogram_score


def test_one_dimensional_energy_score_is_the_crps():
    draws = np.random.default_rng(0).normal(size=300)
    assert np.isclose(energy_score(draws[:, None], np.array([0.4])), crps(draws, 0.4))


def test_scores_prefer_the_forecast_centred_on_the_outcome():
    rng = np.random.default_rng(1)
    draws = rng.normal(size=(400, 3))
    actual = np.zeros(3)
    assert crps(draws[:, 0], 0.0) < crps(draws[:, 0] + 2, 0.0)
    assert energy_score(draws, actual) < energy_score(draws + 1.5, actual)


def test_variogram_score_sees_the_wrong_correlation():
    rng = np.random.default_rng(2)
    common = rng.normal(size=(2000, 1))
    together = common + 0.1 * rng.normal(size=(2000, 2))
    apart = rng.normal(size=(2000, 2))
    actual = np.array([1.0, 1.0])  # an outcome where the two stats move together
    assert variogram_score(together, actual) < variogram_score(apart, actual)


def test_fair_crps_of_a_point_mass_is_the_absolute_error():
    assert np.isclose(crps(np.full(50, 3.0), 5.0), 2.0)


def test_game_scores_include_the_joint_points_energy_score(tmp_path):
    import pandas as pd

    from nflsim.sim.forecast_scores import game_scores

    rng = np.random.default_rng(3)
    shared = rng.normal(size=2000)
    sims = {"together": np.c_[shared, shared + 0.2 * rng.normal(size=2000)],
            "apart": np.c_[shared, -shared + 0.2 * rng.normal(size=2000)]}
    scored = {}
    for label, draws in sims.items():
        run = tmp_path / label
        (run / "raw_games").mkdir(parents=True)
        pd.DataFrame({"game_id": ["g"], "week": [1], "actual_home_score": [30], "actual_away_score": [30]}).to_csv(
            run / "backtest_2025_results.csv", index=False)
        pd.DataFrame({"home": 24 + 7 * draws[:, 0], "away": 24 + 7 * draws[:, 1]}).to_csv(run / "raw_games" / "g.csv", index=False)
        scored[label] = game_scores(run).iloc[0]
    # Both teams beat their means: only the forecast that has them move together scores well on both at once.
    assert scored["together"].energy_points < scored["apart"].energy_points


def test_calibration_flags_scores_that_are_too_narrow():
    import pandas as pd

    from nflsim.sim.forecast_scores import calibration

    rng = np.random.default_rng(4)
    games = 4000
    home_mean, away_mean = rng.normal(24, 3, games), rng.normal(21, 3, games)
    # Real scores vary by 10 points around the sim means, independently for the two teams.
    frame = pd.DataFrame({"label": "narrow", "actual_home": home_mean + rng.normal(0, 10, games),
                          "actual_away": away_mean + rng.normal(0, 10, games), "sim_home_mean": home_mean,
                          "sim_away_mean": away_mean, "sim_spread_sd": 10.0, "sim_total_sd": 10.0,
                          "sim_home_away_corr": 0.1})
    row = calibration(frame).iloc[0]
    assert np.isclose(row.spread_rmse, np.sqrt(200), rtol=0.05) and np.isclose(row.spread_mean_sq_z, 2.0, rtol=0.1)
    assert np.isclose(row.total_mean_sq_z, 2.0, rtol=0.1) and abs(row.observed_home_away_corr) < 0.05
