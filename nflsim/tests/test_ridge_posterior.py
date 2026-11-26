import numpy as np
import scipy.sparse as sp

from nflsim.ridge.metrics import ridge_posterior, solve_ridge


def simulated_ridge(seed, teams=20, games=400, plays=60, tau=0.1, noise=1.0):
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, tau, teams)
    subject = rng.integers(0, teams, games)
    counts = np.full(games, float(plays))
    rate = truth[subject] + rng.normal(0, noise / np.sqrt(plays), games)
    matrix = sp.csr_matrix((np.ones(games), (np.arange(games), subject)), shape=(games, teams))
    penalty = np.full(teams, noise ** 2 / tau ** 2)  # the prior the data were drawn from
    return matrix, rate, counts, penalty, truth


def test_posterior_mean_matches_the_ridge_solution():
    matrix, rate, counts, penalty, _ = simulated_ridge(0)
    beta, covariance, sigma2 = ridge_posterior(matrix, rate, counts, penalty, counts)
    assert np.allclose(beta, solve_ridge(matrix, rate, counts, penalty))
    assert np.allclose(covariance, covariance.T) and np.isclose(sigma2, 1.0, rtol=0.15)


def test_posterior_sds_cover_the_true_strengths():
    z = []
    for seed in range(40):
        matrix, rate, counts, penalty, truth = simulated_ridge(seed)
        beta, covariance, _ = ridge_posterior(matrix, rate, counts, penalty, counts)
        z.extend((beta - truth) / np.sqrt(np.diag(covariance)))
    assert abs(np.var(z) - 1) < 0.15


def test_selected_sds_match_the_full_covariance():
    from nflsim.ridge.metrics import posterior_sds
    matrix, rate, counts, penalty, _ = simulated_ridge(1)
    beta, covariance, _ = ridge_posterior(matrix, rate, counts * 0.5, penalty, counts)
    selected_beta, sds = posterior_sds(matrix, rate, counts * 0.5, penalty, counts, np.array([3, 0, 7]))
    assert np.allclose(selected_beta, beta) and np.allclose(sds, np.sqrt(np.diag(covariance)[[3, 0, 7]]))
