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
