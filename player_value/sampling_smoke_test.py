"""Small real parallel-sampling check; no downloads or production outputs."""

import numpy as np
import pandas as pd

from player_value.plus_minus import PlayerPlusMinus


def main():
    model = PlayerPlusMinus('QB', [2025])
    frame = pd.DataFrame({
        'player_id': ['returning', 'rookie'],
        'cluster': [0, 0], 'n_on': [10, 10], 'se': [0.05, 0.05],
        'raw_plus_minus': [0.02, -0.02],
        'prior_mean': [0.12, 0.0], 'prior_sd': [0.07, 0.10]
    })
    trace, fitted = model._fit_hierarchical_bayesian(
        frame, draws=20, tune=20, cores=4, sample_posterior_predictive=True
    )
    assert np.isfinite(fitted[['bayesian_plus_minus', 'pm_lower', 'pm_upper']]).all().all()
    print(f"Parallel smoke check passed: {dict(trace.posterior.sizes)}")
    print(fitted[['player_id', 'prior_mean', 'bayesian_plus_minus']].to_string(index=False))


if __name__ == '__main__':
    main()