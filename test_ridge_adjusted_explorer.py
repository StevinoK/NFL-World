import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


def test_metric_change_preserves_team_and_venue(tmp_path, monkeypatch):
    records = []
    for metric, side in (("yac_bayes", "subject"), ("sack_rate", "defense")):
        for block, labels in (("subject", ("player",)), ("venue", ("CHI98", "GB00")), ("defense", ("GB", "CHI"))):
            for label in labels:
                records.append(dict(
                    metric=metric, side=side, block=block, label=label,
                    season=2025, week=2, coefficient=0.1, estimate=0.2,
                    weighted_exposure=10.0,
                ))
    pd.DataFrame(records).to_parquet(tmp_path / "weekly_coefficients.parquet", index=False)
    monkeypatch.setattr(sys, "argv", ["ridge_adjusted_explorer.py", "--output-dir", str(tmp_path)])
    app = AppTest.from_file(str(Path(__file__).with_name("ridge_adjusted_explorer.py"))).run()
    app.selectbox(key="ridge_block").set_value("venue").run()
    app.selectbox(key="ridge_entity").set_value("CHI98").run()
    app.selectbox[0].set_value("yac_bayes").run()
    assert app.selectbox(key="ridge_block").value == "venue"
    assert app.selectbox(key="ridge_entity").value == "CHI98"
    app.selectbox(key="ridge_block").set_value("defense").run()
    app.selectbox(key="ridge_entity").set_value("GB").run()
    app.selectbox[0].set_value("sack_rate").run()
    assert app.selectbox(key="ridge_block").value == "defense"
    assert app.selectbox(key="ridge_entity").value == "GB"
    assert not app.exception