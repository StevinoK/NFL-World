from __future__ import annotations

import numpy as np
import pandas as pd

from nflsim.ridge.metrics import RidgeConfig, build_design, rolling_predictions
from nflsim.ridge.pressure import add_pressure_columns


def test_pfr_pressures_spread_over_dropbacks_with_canonical_teams(tmp_path):
    pd.DataFrame({
        "game_id": ["2024_01_LA_SF", "2024_01_LA_SF", "2024_01_LA_SF"],
        "team": ["LA", "LA", "SF"], "times_pressured": [3.0, 1.0, 2.0],
    }).to_parquet(tmp_path / "advstats_week_pass_2024.parquet")
    pbp = pd.DataFrame({
        "game_id": ["2024_01_LA_SF"] * 5, "posteam": ["LA", "LA", "LA", "LA", "SF"],
        "qb_dropback": [1, 1, 1, 0, 1], "qb_hit": [1, 0, 0, 0, 0],
    })
    out = add_pressure_columns(pbp, tmp_path)
    # LA (canonical LAR in PFR) had 4 pressures over 3 dropbacks: clipped to 1; SF 2 over 1: clipped to 1.
    assert out["pressure_pfr"].tolist()[:3] == [1.0, 1.0, 1.0]
    assert np.isnan(out["pressure_pfr"].iloc[3])
    assert out["pressure_hit"].tolist()[:3] == [1.0, 0.0, 0.0] and np.isnan(out["pressure_hit"].iloc[3])


def test_pressure_covariate_fits_on_realized_and_predicts_with_forecast():
    rows = pd.DataFrame({
        "subject": ["A", "B", "A", "B"], "defteam": ["X", "Y", "Y", "X"], "venue": ["v"] * 4,
        "is_home": [1.0, 0.0, 1.0, 0.0], "rest_diff": [0.0] * 4, "position": ["TEAM"] * 4,
        "rate": [0.10, 0.30, 0.12, 0.28], "count": [30.0] * 4, "season": [2024] * 4, "week": [1, 1, 2, 2],
        "season_last_week": [18] * 4, "pressure": [0.1, 0.4, 0.2, 0.3], "pressure_forecast": [0.25] * 4,
    })
    design = build_design(rows, "team", pressure=("pressure", "pressure_forecast"))
    assert design.blocks[-1] == "pressure"
    realized = design.matrix[:, -1].toarray().ravel()
    forecast = design.predict_matrix[:, -1].toarray().ravel()
    assert np.allclose(realized, [-0.15, 0.15, -0.05, 0.05]) and np.allclose(forecast, 0.0)
    predictions = rolling_predictions(rows, design, RidgeConfig(), [2024], isolated=False)
    # Week-2 rows are predicted at the (identical) forecast, so only the entity terms separate them.
    assert predictions.loc[[2, 3], "pred_full"].notna().all()
