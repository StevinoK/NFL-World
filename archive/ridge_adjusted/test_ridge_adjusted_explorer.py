import sys
import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest
from ridge_graph_builder import DEFAULT_COLORS, artifact_signature, build_figure, current_artifacts, observed_graph_data, read_graph_data, series_color


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
    app.selectbox(key="ridge_metric").set_value("yac_bayes").run()
    assert app.selectbox(key="ridge_block").value == "venue"
    assert app.selectbox(key="ridge_entity").value == "CHI98"
    app.selectbox(key="ridge_block").set_value("defense").run()
    app.selectbox(key="ridge_entity").set_value("GB").run()
    app.selectbox(key="ridge_metric").set_value("sack_rate").run()
    assert app.selectbox(key="ridge_block").value == "defense"
    assert app.selectbox(key="ridge_entity").value == "GB"
    assert not app.exception


def test_production_artifact_loader_uses_active_bundle(tmp_path):
    (tmp_path / "config").mkdir()
    bundle = tmp_path / "artifacts" / "bundles" / "current"
    bundle.mkdir(parents=True)
    (tmp_path / "config" / "last_upload.json").write_text(json.dumps({"bundle_hash": "current"}))
    (bundle / "feature_manifest.json").write_text(json.dumps({"covered_metrics": ["yac_bayes", "sack_rate"]}))
    pd.DataFrame([dict(player_id="p1", player_name="Player One", game_id="2026_05_A_B",
                       game_date="2026-10-01", yac_bayes=2.1)]).to_csv(bundle / "player_stats_wide.csv", index=False)
    pd.DataFrame([dict(team="B", game_id="2026_05_A_B", game_date="2026-10-01", sack_rate=0.07)]).to_csv(
        bundle / "team_stats_wide.csv", index=False)
    pd.DataFrame([dict(game_id="2026_05_A_B", team="A", venue="CHI98", ridge_yac_oe_venue_effect=-0.3,
                       ridge_yac_oe_offense_effect=0.2)]).to_csv(bundle / "ridge_context.csv", index=False)
    assert current_artifacts(tmp_path) == bundle
    data, names = read_graph_data(bundle)
    assert names["p1"] == "Player One"
    assert data.loc[data.metric.eq("yac_bayes"), "estimate"].iloc[0] == 2.1
    assert data.loc[data.block.eq("venue"), "coefficient"].iloc[0] == -0.3
    assert data.loc[data.metric.eq("yac_bayes"), "coefficient"].isna().all()
    assert data.week.eq(5).all() and data.season.eq(2026).all()
    assert data["game_date"].eq("2026-10-01").all()
    assert observed_graph_data(data, today="2026-09-30").empty
    assert len(observed_graph_data(data, today="2026-10-01")) == len(data)
    before = artifact_signature(bundle)
    (bundle / "feature_manifest.json").write_text(json.dumps({"covered_metrics": []}))
    assert artifact_signature(bundle) != before


def test_graph_elements_and_facets():
    frame = pd.DataFrame([
        dict(label="A", entity="A", metric="test", season="2025", week=1, estimate=1.0),
        dict(label="B", entity="B", metric="test", season="2025", week=1, estimate=2.0),
        dict(label="A", entity="A", metric="test", season="2026", week=2, estimate=3.0),
    ])
    for mark in ("Line", "Scatter", "Bar", "Box", "Histogram"):
        figure = build_figure(frame, "week", "estimate", mark, "entity", "season")
        assert len(figure.data) >= 2
    figure = build_figure(frame, "week", "estimate", "Scatter")
    assert figure.data[0].customdata[0][0] == 0


def test_graph_excludes_future_and_unknown_dates():
    frame = pd.DataFrame({"game_date": ["2026-10-07", "2026-10-08", "2026-10-09", None]})
    assert observed_graph_data(frame, today="2026-10-08").index.tolist() == [0, 1]
    historical = pd.DataFrame({"season": [2025], "week": [2]})
    pd.testing.assert_frame_equal(observed_graph_data(historical), historical)


def test_graph_keeps_current_season_with_mixed_date_formats():
    frame = pd.DataFrame({
        "season": [2025, 2026, 2026, 2026],
        "game_date": ["2025-09-07", "2026-09-13 00:00:00", "2026-10-04 00:00:00", "2026-10-11 00:00:00"],
    })
    visible = observed_graph_data(frame, today="2026-10-08")
    assert visible.index.tolist() == [0, 1, 2]
    assert visible["season"].eq(2026).sum() == 2


def test_custom_graph_style():
    frame = pd.DataFrame([dict(label="A", entity="A", season="2025", week=1, estimate=1.0)])
    for mark in ("Line", "Scatter", "Bar", "Box", "Histogram"):
        figure = build_figure(
            frame, "week", "estimate", mark, color_map={"A": "#ff0000"},
            labels={"week": "Game week", "estimate": "Adjusted rating"}, title="Ratings",
            legend_title="Team", height=650, font_size=18, line_width=4, marker_size=10,
            show_grid=False, background="#ffffff", text_color="#123456",
        )
        assert figure.layout.title.text == "Ratings"
        assert figure.layout.xaxis.title.text == "Game week"
        assert figure.layout.yaxis.title.text == "Adjusted rating"
        assert figure.layout.legend.title.text == "Team"
        assert figure.layout.height == 650
        assert figure.layout.font.size == 18
        assert figure.layout.font.color == "#123456"
        assert figure.layout.plot_bgcolor == "#ffffff"
        assert not figure.layout.yaxis.showgrid
        if mark == "Line":
            assert figure.data[0].line.color == "#ff0000"
            assert figure.data[0].line.width == 4
            assert figure.data[0].marker.size == 10
        else:
            assert figure.data[0].marker.color == "#ff0000"
    figure = build_figure(frame, "week", "estimate", "Scatter", color=None, color_map={"Series": "#ff0000"})
    assert figure.data[0].marker.color == "#ff0000"
    frame["weighted_exposure"] = 10.0
    figure = build_figure(frame, "week", "estimate", "Scatter", size="weighted_exposure", marker_size=12)
    assert list(figure.data[0].marker.size) == [10.0]


def test_explorer_filters_future_games_and_applies_style(tmp_path, monkeypatch):
    today = pd.Timestamp.today().normalize()
    records = [
        dict(metric="yac_bayes", side="subject", block="subject", label=label,
             season=today.year, week=week, game_date=date, coefficient=0.1,
             estimate=0.2, weighted_exposure=10.0)
        for label, week, date in (
            ("A", 1, today - pd.Timedelta(days=1)),
            ("A", 2, today + pd.Timedelta(days=7)),
            ("Future only", 2, today + pd.Timedelta(days=7)),
        )
    ]
    pd.DataFrame(records).to_parquet(tmp_path / "weekly_coefficients.parquet", index=False)
    monkeypatch.setattr(sys, "argv", ["ridge_adjusted_explorer.py", "--output-dir", str(tmp_path)])
    app = AppTest.from_file(str(Path(__file__).with_name("ridge_adjusted_explorer.py"))).run()
    assert app.selectbox(key="ridge_entity").options == ["A"]
    app.text_input(key="style_title").set_value("My ratings")
    app.text_input(key="style_x_week_label_").set_value("Game")
    app.text_input(key="style_y_estimate").set_value("Rating")
    app.session_state["style_color_entity_A"] = "#ff0000"
    app.run()
    assert not app.exception
    chart = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert chart["layout"]["title"]["text"] == "My ratings"
    assert chart["layout"]["xaxis"]["title"]["text"] == "Game"
    assert chart["layout"]["yaxis"]["title"]["text"] == "Rating"
    assert chart["data"][0]["line"]["color"] == "#ff0000"
    assert chart["data"][0]["x"] == [f"{today.year} W01"]


def write_full_feature_bundle(directory):
    pd.DataFrame([dict(player_id="p1", player_name="Player One", game_id="2025_02_GB_CHI",
                       game_date="2025-09-14", season=2025, week=2, cpoe_bayes=0.1,
                       adot=9.0, **{"GO_bayes": 0.3, "Unnamed: 0": 1})]).to_csv(
        directory / "player_stats_wide.csv", index=False)
    pd.DataFrame([dict(team="CHI", game_id="2025_02_GB_CHI", game_date="2025-09-14",
                       epa_rate=0.2, pass_oe=0.1, pace_of_play_bayes=29.0,
                       passing_massey_rating=1.0, sack_rate=0.07, def_winning_rate=0.5,
                       short_cpoe_allowed=0.2, ridge_yac_oe_venue_effect=-0.3)]).to_csv(
        directory / "team_stats_wide.csv", index=False)
    pd.DataFrame([dict(venue="CHI98", game_id="2025_02_GB_CHI", game_date="2025-09-14",
                       wind_speed=12.0, roof="outdoors")]).to_csv(
        directory / "venue_stats_wide.csv", index=False)
    pd.DataFrame([dict(game_id="2025_02_GB_CHI", team="CHI", venue="CHI98",
                       ridge_yac_oe_venue_effect=-0.3, ridge_yac_oe_offense_effect=0.2)]).to_csv(
        directory / "ridge_context.csv", index=False)
    (directory / "feature_manifest.json").write_text(json.dumps({"covered_metrics": ["cpoe_bayes"]}))


def test_loader_includes_all_entity_features_not_just_manifest(tmp_path):
    write_full_feature_bundle(tmp_path)
    data, names = read_graph_data(tmp_path)
    assert names == {"p1": "Player One"}
    assert set(data.loc[data.block.eq("subject"), "metric"]) == {"cpoe_bayes", "adot", "GO_bayes"}
    assert set(data.loc[data.block.eq("offense_team"), "metric"]) == {
        "epa_rate", "pass_oe", "pace_of_play_bayes", "passing_massey_rating", "yac_oe",
    }
    assert set(data.loc[data.block.eq("defense"), "metric"]) == {
        "sack_rate", "def_winning_rate", "short_cpoe_allowed",
    }
    assert set(data.loc[data.block.eq("venue"), "metric"]) == {"wind_speed", "yac_oe"}
    assert len(data[data.block.eq("venue") & data.metric.eq("yac_oe")]) == 1
    assert data["game_date"].notna().all()


def test_estimate_type_filters_metrics_and_team_colors(tmp_path, monkeypatch):
    import nfl_data_py as nfl
    import ridge_adjusted_explorer

    write_full_feature_bundle(tmp_path)
    ridge_adjusted_explorer.load_team_colors.clear()
    monkeypatch.setattr(nfl, "import_team_desc", lambda: pd.DataFrame([
        {"team_abbr": "CHI", "team_color": "#0B162A"},
    ]))
    monkeypatch.setattr(sys, "argv", ["ridge_adjusted_explorer.py", "--output-dir", str(tmp_path)])
    app = AppTest.from_file(str(Path(__file__).with_name("ridge_adjusted_explorer.py"))).run()
    sidebar_selectors = app.sidebar.selectbox
    assert sidebar_selectors[0].label == "Estimate type"
    assert set(app.selectbox(key="ridge_metric").options) == {"adot", "cpoe_bayes", "GO_bayes"}
    app.selectbox(key="ridge_block").set_value("offense_team").run()
    assert set(app.selectbox(key="ridge_metric").options) == {
        "epa_rate", "pass_oe", "pace_of_play_bayes", "passing_massey_rating", "yac_oe",
    }
    chart = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert chart["data"][0]["line"]["color"] == "#0B162A"
    app.session_state["style_color_offense_team_entity_CHI"] = "#ff0000"
    app.selectbox(key="ridge_metric").set_value("yac_oe").run()
    app.selectbox(key="graph_y").set_value("coefficient").run()
    app.selectbox(key="ridge_metric").set_value("pass_oe").run()
    assert app.selectbox(key="graph_y").value == "estimate"
    chart = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert chart["data"][0]["line"]["color"] == "#ff0000"
    app.selectbox(key="ridge_block").set_value("venue").run()
    assert set(app.selectbox(key="ridge_metric").options) == {"wind_speed", "yac_oe"}
    app.selectbox(key="ridge_block").set_value("defense").run()
    assert set(app.selectbox(key="ridge_metric").options) == {
        "sack_rate", "def_winning_rate", "short_cpoe_allowed",
    }
    assert not app.exception
    ridge_adjusted_explorer.load_team_colors.clear()


def test_team_color_alias_and_uncolored_series():
    colors = {"LAR": "#003594"}
    assert series_color("LA", 0, "offense_team", "entity", "LA", colors) == "#003594"
    assert series_color("Series", 0, "offense_team", "None", "LA", colors) == "#003594"
    assert series_color("2025", 0, "offense_team", "season", "LA", colors) == DEFAULT_COLORS[0]
    assert series_color("unknown", 1, "offense_team", "entity", "unknown", colors) == DEFAULT_COLORS[1]