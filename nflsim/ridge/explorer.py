"""Interactive weekly ridge coefficients, fitted strictly before each league week."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import nfl_data_py as nfl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nflsim import ui
from nflsim.bundle_labels import all_bundles, bundle_label
from nflsim.sim.season import latest_bundle
from nflsim.ridge.graph import artifact_signature, build_figure, observed_graph_data, read_graph_data, series_color


if __name__ == "__main__" and not st.runtime.exists():
    import subprocess

    raise SystemExit(subprocess.call([
        sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]
    ]))


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "runtime"
BLOCK_NAMES = {
    "subject": "Player / offensive team",
    "defense": "Defensive team",
    "offense_team": "Offensive team context",
    "lineup_value": "Team lineup player value",
    "player_value": "Player-value coefficient",
    "teammate": "Teammate",
    "venue": "Venue",
    "position": "Position",
    "home": "Home advantage",
    "rest": "Rest differential",
    "intercept": "League baseline",
}
TEAM_BLOCKS = {"offense_team", "defense", "lineup_value"}
GROUP_BY_METRIC = {"pv_off_qb": "QB", "pv_off_wr": "WR", "pv_off_rb": "RB", "pv_off_te": "TE", "pv_def_dl": "DL"}


@st.cache_data(show_spinner=False, max_entries=2)
def load_components(path: str, signature: tuple = ()) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"game_id": str, "team": str, "player_id": str})


def lineup_drilldown(directory: Path, teams: list[str], metric: str, seasons, weeks) -> None:
    path = directory / "player_value_components.csv"
    if not path.exists():
        return
    stat = path.stat()
    components = load_components(str(path), (stat.st_mtime_ns, stat.st_size))
    rows = components[components.team.astype(str).isin(teams) & components.group.eq(GROUP_BY_METRIC.get(metric, ""))].copy()
    rows["season"] = rows.game_id.str[:4].astype(int)
    rows["week"] = rows.game_id.str[5:7].astype(int)
    rows = rows[rows.season.isin([int(season) for season in seasons]) & rows.week.between(*weeks)]
    with st.expander("Lineup components (strictly prior PM x normalized trailing snap share)", expanded=False):
        columns = ["game_id", "team", "group", "rank", "player_id", "pm", "pm_source", "share", "share_source", "weight", "contribution"]
        st.dataframe(rows.reindex(columns=columns).sort_values(["game_id", "team", "rank"]), hide_index=True, width="stretch")


@st.cache_resource(show_spinner="Loading ridge estimates...", max_entries=2)
def load_data(directory: str, signature: tuple = ()) -> tuple[pd.DataFrame, dict[str, str]]:
    """Games so far and player names. Shared between reruns, so nothing below may change the table in place."""
    data, names = read_graph_data(Path(directory))
    return observed_graph_data(data), names


@st.cache_data(show_spinner=False, ttl=600)
def bundle_choices() -> dict[str, str]:
    current = latest_bundle()
    return {str(info.path.resolve()): bundle_label(info, current) for info in all_bundles()}


@st.cache_data(show_spinner=False)
def load_team_colors() -> dict[str, str]:
    try:
        teams = nfl.import_team_desc()[["team_abbr", "team_color"]].dropna()
        return dict(zip(teams["team_abbr"], teams["team_color"]))
    except Exception:
        return {}


def main() -> None:
    ui.page("Ridge Graph Builder")
    header_slot = st.container()

    try:
        if "--output-dir" in sys.argv:
            directory = sys.argv[sys.argv.index("--output-dir") + 1]
        else:
            choices = bundle_choices()
            directory = st.sidebar.selectbox("Model bundle", list(choices), format_func=choices.get,
                                             help="Which version of the models to chart. Each is named for what changed. 'Current' is the newest one that can simulate upcoming games.")
        data, names = load_data(directory, artifact_signature(Path(directory)))
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        st.error(f"Ridge artifacts unavailable: {error}")
        st.stop()
    if data.empty:
        st.info("No weekly coefficients available yet.")
        st.stop()

    with st.sidebar:
        ui.section("Data filters")
        present = set(data["block"].unique())
        blocks = [block for block in BLOCK_NAMES if block in present]
        default_block = "subject"
        if default_block not in blocks:
            default_block = blocks[0]
        if st.session_state.get("ridge_block") not in blocks:
            st.session_state["ridge_block"] = default_block
        block = st.selectbox("Estimate type", blocks, key="ridge_block", format_func=BLOCK_NAMES.get, help='Whose effect to look at: a player or offense, a defense, a venue, and so on.')
        block_rows = data[data["block"].eq(block)]
        metric_options = sorted(block_rows["metric"].unique())
        if st.session_state.get("ridge_metric") not in metric_options:
            st.session_state["ridge_metric"] = "cpoe_bayes" if "cpoe_bayes" in metric_options else metric_options[0]
        metric = st.selectbox("Metric", metric_options, key="ridge_metric", help='The stat being estimated, such as completion rate over expectation or EPA per play.')
        group = block_rows[block_rows["metric"].eq(metric)]
        options = sorted(group["label"].astype(str).unique(), key=lambda label: (names.get(label, label), label))

        def display_label(label: str) -> str:
            return f"{names[label]} ({label})" if block in {"subject", "teammate"} and label in names else label

        latest_season = group["season"].max()
        recent = group[group["season"].eq(latest_season)].sort_values("weighted_exposure", ascending=False)
        initial = str(recent.iloc[0]["label"])
        if st.session_state.get("ridge_entity") not in options:
            st.session_state["ridge_entity"] = initial
        label = st.selectbox("Player, team or venue", options, key="ridge_entity", format_func=display_label, help='The main player, team or venue to chart.')
        additional = st.multiselect("Compare entities", [option for option in options if option != label], format_func=display_label, help='Others to chart alongside the main selection.')
        entity = group[group["label"].isin([label, *additional])].sort_values(["season", "week"])
        seasons = sorted(group["season"].unique())
        selected_seasons = st.multiselect("Season", seasons, default=[seasons[-1]], help='Seasons to include.')
        weeks = st.slider("Weeks", 1, 18, (1, 18), help='Range of weeks to include.')

    visible = entity[entity["season"].isin(selected_seasons) & entity["week"].between(*weeks)].copy()
    if visible.empty:
        st.info("No observed pregame estimates match these filters.")
        return
    visible["week_label"] = visible["season"].astype(str) + " W" + visible["week"].astype(str).str.zfill(2)
    visible["entity"] = visible["label"].map(display_label)
    visible["season"] = visible["season"].astype(str)
    if "value_kind" not in visible:
        visible["value_kind"] = "Historical coefficient fit"
    with header_slot:
        ui.header("Ridge Graph Builder", "Weekly ridge coefficients, each fitted strictly before its league week.",
                  {"Bundle": bundle_choices().get(directory, Path(directory).name).split(" · ")[0], "Seasons": f"{data['season'].min()}–{data['season'].max()}",
                   "Estimate": BLOCK_NAMES.get(block, block), "Metric": metric}, eyebrow="Ridge")
    with st.expander("Artifact provenance"):
        st.code(directory, language=None)
        st.write("Games dated after today are excluded. Unavailable coefficients and exposure are left missing.")
    ui.section("Chart")
    toolbar = st.columns([2, 2, 2, 2, 2])
    x = toolbar[0].selectbox("X", ["week_label", "week", "season", "entity", "estimate", "coefficient", "weighted_exposure", "Other metric"], key="graph_x", help='What runs along the bottom of the chart.')
    numeric = [column for column in ("estimate", "coefficient", "weighted_exposure") if visible[column].notna().any()]
    if st.session_state.get("graph_y") not in numeric:
        st.session_state["graph_y"] = numeric[0]
    y = toolbar[1].selectbox("Y", numeric, key="graph_y", help='The value plotted up the side of the chart.')
    color = toolbar[2].selectbox("Color / overlay", ["entity", "season", "None"], key="graph_color", help='Split the chart into colored groups by this field.')
    facet = toolbar[3].selectbox("Wrap", ["None", "season", "entity"], key="graph_facet", help='Draw a separate small chart for each value of this field.')
    mark = toolbar[4].selectbox("Element", ["Line", "Scatter", "Bar", "Box", "Histogram"], key="graph_mark", help='How to draw the data: line, points, bars and so on.')
    if x == "Other metric":
        candidates = sorted(data.loc[data["block"].eq(block), "metric"].unique())
        if st.session_state.get("graph_other_metric") not in candidates:
            st.session_state["graph_other_metric"] = candidates[0]
        other_metric = st.selectbox("X metric", candidates, key="graph_other_metric", help='The metric to put along the bottom of the chart.')
        other = data[data["metric"].eq(other_metric) & data["block"].eq(block)].copy()
        other["season"] = other["season"].astype(str)
        other = other.groupby(["label", "season", "week"], as_index=False)["estimate"].mean()
        visible = visible.merge(other.rename(columns={"estimate": "x_metric"}), on=["label", "season", "week"], how="inner")
        x = "x_metric"
    options_row = st.columns(3)
    smooth = options_row[0].slider("Moving average", 1, 8, 1, help='Smooth the line over this many games. 1 means no smoothing.')
    zero = options_row[1].checkbox("Zero reference", value=block in {"venue", "offense_team"}, help='Draw a dotted line at zero. For most estimates, zero means league average.')
    size = options_row[2].selectbox("Size", ["None"] + [column for column in numeric if column == "weighted_exposure"], help='Scale points by this field.')
    plotted = visible.dropna(subset=[x, y]).copy()
    if plotted.empty:
        st.info("No values available for the selected graph roles.")
        return
    if smooth > 1:
        plotted[y] = plotted.groupby(["label", "season"])[y].transform(lambda values: values.rolling(smooth, min_periods=1).mean())
    with st.expander("Chart style"):
        titles = st.columns(2)
        title = titles[0].text_input("Chart title", key="style_title", help='Title shown above the chart.')
        legend_title = titles[1].text_input("Legend title", value="" if color == "None" else color, key=f"style_legend_{color}", help='Heading shown above the legend.')
        axes = st.columns(2)
        x_label = axes[0].text_input("X-axis label", value=other_metric if x == "x_metric" else x.replace("_", " ").title(), key=f"style_x_{x}_{other_metric if x == 'x_metric' else ''}", help='Label along the bottom axis.')
        y_label = axes[1].text_input("Y-axis label", value=y.replace("_", " ").title(), key=f"style_y_{y}", help='Label along the side axis.')
        series = ["Series"] if color == "None" else list(plotted[color].astype(str).unique())
        color_columns = st.columns(3)
        team_colors = load_team_colors() if block in TEAM_BLOCKS else {}
        color_map = {
            series_name: color_columns[index % 3].color_picker(
                series_name, series_color(series_name, index, block, color, label, team_colors),
                key=f"style_color_{color}_{series_name}" if block not in TEAM_BLOCKS
                else f"style_color_{block}_{color}_{label if color == 'None' else series_name}",
            ) for index, series_name in enumerate(series)
        }
        appearance = st.columns(3)
        background = appearance[0].color_picker("Background", "#ffffff", key="style_background")
        text_color = appearance[1].color_picker("Text color", ui.INK, key="style_text_color")
        show_grid = appearance[2].checkbox("Y-axis gridlines", value=True, key="style_grid", help='Draw horizontal gridlines.')
        dimensions = st.columns(4)
        height = dimensions[0].slider("Chart height", 350, 1000, 550, 50, key="style_height", help='Height of the chart in pixels.')
        font_size = dimensions[1].slider("Font size", 10, 24, 14, key="style_font_size", help="Size of the chart's text.")
        line_width = dimensions[2].slider("Line width", 1, 8, 2, key="style_line_width", help='Thickness of lines.')
        marker_size = dimensions[3].slider("Marker size", 3, 20, 7, key="style_marker_size", help='Size of points.')
    fig = build_figure(plotted, x, y, mark, None if color == "None" else color,
                       None if facet == "None" else facet, None if size == "None" else size,
                       color_map=color_map, labels={x: x_label, y: y_label}, title=title,
                       legend_title=legend_title, height=height, font_size=font_size,
                       line_width=line_width, marker_size=marker_size, show_grid=show_grid,
                       background=background, text_color=text_color)
    fig.update_layout(font_family=ui.FONT, hoverlabel={"bgcolor": ui.SURFACE, "bordercolor": ui.BORDER,
                                                       "font": {"family": ui.FONT, "color": ui.INK}})
    if zero:
        fig.add_hline(y=0, line_dash="dot", line_color=ui.MUTED)
    event = st.plotly_chart(fig, on_select="rerun", selection_mode=("points", "box", "lasso"), key="ridge_graph")
    selected = event.selection.points
    table = plotted
    if selected and mark in {"Line", "Scatter"}:
        row_ids = [point["customdata"][0] for point in selected if point.get("customdata")]
        table = plotted.loc[plotted.index.isin(row_ids)]
    ui.section("Selected rows")
    graph_tab, data_tab = st.tabs(["Summary", "Data"])
    with graph_tab:
        st.dataframe(table.groupby("entity")[y].agg(["count", "mean", "min", "max"]), width="stretch")
    with data_tab:
        st.dataframe(table, hide_index=True, width="stretch")
    if block == "lineup_value":
        lineup_drilldown(Path(directory), [label, *additional], metric, selected_seasons, weeks)
    st.download_button(
        "Download rows", table.to_csv(index=False).encode("utf-8"),
        file_name=f"{metric}_{block}_graph.csv", mime="text/csv",
    help='Save the rows behind this chart as a CSV file.')


if __name__ == "__main__":
    main()