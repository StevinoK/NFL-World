"""Game report: per-player EPA by outcome for played games, with win-probability and EPA worms and a drive filter.

Reads nflverse play-by-play (and season rosters for positions). Logic lives in game_report.py and game_flow.py.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import nfl_data_py as nfl
import pandas as pd
import streamlit as st

from nflsim import ui
from nflsim.dashboard import game_flow, game_report

LEFT_COLOR = "#636EFA"
RIGHT_COLOR = "#EF553B"
# team_stats_wide uses LA/LV; nflverse team_desc historically carries both spellings.
TEAM_DESC_ALIASES = {"LA": "LAR", "LAR": "LA", "LV": "OAK", "OAK": "LV"}


def current_nfl_season(today: Optional[pd.Timestamp] = None) -> int:
    today = today or pd.Timestamp.today()
    return today.year if today.month >= 9 else today.year - 1


REPORT_PBP_COLUMNS = sorted(set(game_report.PBP_COLUMNS) | set(game_flow.GAME_COLUMNS))


@st.cache_data(show_spinner="Loading play-by-play...", ttl=3600)
def load_pbp(season: int) -> pd.DataFrame:
    return nfl.import_pbp_data([season], columns=REPORT_PBP_COLUMNS, downcast=True, cache=False)


@st.cache_data(show_spinner=False, ttl=3600)
def load_player_segments(season: int) -> pd.DataFrame:
    segments = game_report.build_player_segments(load_pbp(season))
    return game_report.attach_roster(segments, load_roster(season))


@st.cache_data(show_spinner=False, ttl=3600)
def load_roster(season: int) -> pd.DataFrame:
    """Season roster (player_id, position, team, jersey_number); empty if unavailable."""
    try:
        roster = nfl.import_seasonal_rosters([season])
    except Exception:
        return pd.DataFrame(columns=["player_id", "position", "team", "jersey_number"])
    keep = [c for c in ("player_id", "position", "team", "jersey_number") if c in roster.columns]
    return roster[keep]


@st.cache_data(show_spinner=False)
def _team_color_tables() -> Tuple[Dict[str, str], Dict[str, str]]:
    try:
        desc = nfl.import_team_desc()[["team_abbr", "team_color", "team_color2"]]
    except Exception:
        return {}, {}
    primary = dict(zip(desc["team_abbr"], desc["team_color"]))
    secondary = dict(zip(desc["team_abbr"], desc["team_color2"]))
    for alias, source in TEAM_DESC_ALIASES.items():
        primary.setdefault(alias, primary.get(source))
        secondary.setdefault(alias, secondary.get(source))
    return primary, secondary


def matchup_colors(home: str, away: str) -> Dict[str, str]:
    primary, secondary = _team_color_tables()
    return game_flow.distinct_team_colors(home, away, primary, secondary, RIGHT_COLOR, LEFT_COLOR)


def render_drive_filter(
    selected: pd.DataFrame,
    game_id: str,
    drives: pd.DataFrame,
    chart_event: Optional[object] = None,
) -> pd.DataFrame:
    """Drive picker for the player chart; clicking/box-selecting drive bars also sets it."""
    drives = drives[drives["team"].isin(selected["team"].unique())]
    if drives.empty:
        return selected
    options = drives["drive"].tolist()
    labels = dict(zip(drives["drive"], drives["label"]))
    key = f"report_drives_{game_id}"
    sig_key = f"{key}_chart_selection"

    if chart_event is not None:
        points = chart_event.get("selection", {}).get("points", []) if hasattr(chart_event, "get") else []
        seqs = {int(round(float(p["x"]))) for p in points if "x" in p}
        from_chart = drives.loc[drives["seq"].isin(seqs), "drive"].tolist()
        # Only a new chart selection overrides the picker, so manual edits stick.
        if st.session_state.get(sig_key) != tuple(from_chart):
            st.session_state[sig_key] = tuple(from_chart)
            st.session_state[key] = from_chart
    if key in st.session_state:
        st.session_state[key] = [d for d in st.session_state[key] if d in options]
    chosen = st.multiselect(
        "Drives", options, format_func=labels.get, key=key, placeholder="All drives",
        help="Filters the player EPA chart; leave empty for all drives. Click or box-select "
             "bars in the drive chart (double-click to clear), or pick here.",
    )
    if not chosen:
        return selected
    return game_report.select_drives(selected, game_id, chosen)


def render_single_game(pbp: pd.DataFrame, segments: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Win-probability and EPA worms plus drive summary; returns the game's player segments."""
    catalog = game_flow.game_catalog(pbp)
    if catalog.empty:
        st.info("No completed games found for the selected seasons.")
        return None
    labels = dict(zip(catalog["label"], catalog["game_id"]))
    game_id = labels[st.selectbox("Game", list(labels), key="report_single_game")]
    plays = game_flow.game_plays(pbp, game_id)
    home, away = plays["home_team"].iloc[0], plays["away_team"].iloc[0]
    colors = matchup_colors(home, away)

    toggles = st.columns(2)
    include_st = toggles[0].checkbox(
        "Include special teams in EPA", value=False, key="report_include_st",
        help="Adds kickoffs, punts, field goals and extra points to the EPA worm and drive EPA.",
    )
    spread = toggles[1].checkbox(
        "Spread-adjusted win probability", value=False, key="report_vegas_wp",
        help="Uses nflverse's win probability that also accounts for the closing point spread.",
    )
    left, right = st.columns(2)
    left.plotly_chart(
        game_flow.build_wp_figure(plays, colors[home], colors[away], spread_adjusted=spread),
        use_container_width=True, theme=None,
    )
    right.plotly_chart(
        game_flow.build_epa_worm(plays, colors, include_special_teams=include_st),
        use_container_width=True, theme=None,
    )

    drives = game_flow.drive_summary(plays, include_special_teams=include_st)
    event = st.plotly_chart(
        game_flow.build_drive_figure(drives, colors), use_container_width=True, theme=None,
        on_select="rerun", selection_mode=("points", "box"), key=f"report_drive_chart_{game_id}",
    )
    with st.expander("Drive table", expanded=False):
        st.dataframe(game_flow.drive_table(drives), use_container_width=True)

    st.subheader("Player EPA")
    selected = segments[segments["game_id"] == game_id]
    return render_drive_filter(selected, game_id, drives, event)


def main() -> None:
    ui.page("Game Report")
    ui.header("Game report", "Player EPA by outcome for games already played, from nflverse play-by-play.", eyebrow="Game results")
    st.caption(
        "Each box is one target, pass, carry, scramble, sack or untargeted pass (completions split into air and YAC); bars are "
        "grouped by position (QB, RB, WR, TE). In the packed layout a box's area is its EPA. Boxes above the line added EPA, boxes below cost EPA, and outcome "
        "colors keep the same order away from the line in every bar. Hatched boxes are touchdowns "
        "(/ = offense scored, x = defensive return TD). The diamond is the player's net EPA. Targeted passes "
        "count for both the QB (Passes) and the receiver, so bars don't add up to team EPA unless one of "
        "Passes or Targets is excluded."
    )
    latest = current_nfl_season()
    controls = st.columns([2, 2, 2])
    seasons = controls[0].multiselect(
        "Seasons", list(range(latest, 1998, -1)), default=[latest], key="report_seasons"
    )
    mode = controls[1].radio(
        "Select by", ["Single game", "Team games", "Players"], horizontal=True, key="report_mode"
    )
    kinds = controls[2].multiselect(
        "Include", game_report.KINDS, default=game_report.KINDS, format_func=game_report.KIND_LABELS.get,
        key="report_kinds",
    )
    controls = st.columns([3, 2, 1])
    # Players are compared on season/range totals by default; team games per game.
    group_by = controls[0].radio(
        "Bars", game_report.GROUP_OPTIONS, horizontal=True,
        index=game_report.GROUP_OPTIONS.index(game_report.GROUP_PLAYER) if mode == "Players" else 0, key=f"report_group_by_{mode}",
    )
    box_layout = controls[1].radio(
        "Boxes", [game_report.LAYOUT_PACKED, game_report.LAYOUT_STACKED], horizontal=True,
        key="report_layout",
    )
    min_plays = controls[2].number_input(
        "Min plays", min_value=1, value=1, step=1, key="report_min_plays",
        help="Minimum plays (passes, targets, carries, scrambles, sacks and untargeted passes) for a bar to be shown.",
    )
    if not seasons or not kinds:
        st.info("Select at least one season and one play type.")
        return

    frames = []
    for season in sorted(seasons):
        try:
            frames.append(load_player_segments(int(season)))
        except Exception as error:  # season not published yet, network issue, ...
            st.warning(f"Could not load {season} play-by-play: {error}")
    frames = [f for f in frames if not f.empty]
    if not frames:
        st.info("No completed games found for the selected seasons.")
        return
    segments = pd.concat(frames, ignore_index=True)
    segments = segments[segments["kind"].isin(kinds)]
    if segments.empty:
        st.info("No plays of the selected types.")
        return

    if mode == "Single game":
        pbp = pd.concat([load_pbp(int(season)) for season in sorted(seasons)
                         if int(season) in set(segments["season"])], ignore_index=True)
        selected = render_single_game(pbp, segments)
        if selected is None:
            return
    elif mode == "Team games":
        catalog = game_report.team_game_catalog(segments)
        labels = catalog["label"].tolist()
        chosen = st.multiselect(
            "Team games (offense)", labels, default=labels[:1], key="report_team_games",
            help="Pick any mix of teams and weeks; each team's targets and carries in that game are shown.",
        )
        picked = catalog[catalog["label"].isin(chosen)]
        selected = game_report.select_team_games(segments, picked[["game_id", "team"]].values)
        if picked["game_id"].nunique() == 1:
            game_id = picked["game_id"].iloc[0]
            plays = game_flow.game_plays(load_pbp(int(picked["season"].iloc[0])), game_id)
            selected = render_drive_filter(selected, game_id, game_flow.drive_summary(plays))
    else:
        catalog = game_report.player_game_catalog(segments)
        players = (
            catalog.groupby("player_id")
            .agg(player_name=("player_name", "first"), teams=("team", lambda s: "/".join(dict.fromkeys(s))),
                 position=("position", lambda s: s.mode().iloc[0]),
                 touches=("targets", "sum"), rushes=("rushes", "sum"),
                 scrambles=("scrambles", "sum"), sacks=("sacks", "sum"), untargeted=("untargeted", "sum"),
                 passes=("passes", "sum"))
            .assign(touches=lambda d: d["touches"] + d["rushes"] + d["scrambles"] + d["sacks"]
                    + d["untargeted"] + d["passes"])
            .sort_values("touches", ascending=False)
        )
        player_labels = {
            pid: f"{r.player_name} ({r.teams} {r.position})" for pid, r in players.iterrows()
        }
        chosen_players = st.multiselect(
            "Players", list(player_labels), format_func=player_labels.get, key="report_players",
        )
        weeks = game_report.week_options(segments)
        week_labels = list(weeks)
        if len(week_labels) > 1:
            start_label, end_label = st.select_slider(
                "Week range", options=week_labels, value=(week_labels[0], week_labels[-1]),
                # Reset to the full range whenever the season selection changes.
                key="report_week_range_" + "_".join(str(season) for season in sorted(seasons)),
                help="Each player's bar totals every target and carry in this range.",
            )
        else:
            start_label = end_label = week_labels[0]
        options = game_report.games_in_range(
            catalog[catalog["player_id"].isin(chosen_players)], weeks[start_label], weeks[end_label]
        )
        with st.expander("Exclude specific games", expanded=False):
            excluded = st.multiselect(
                "Games to leave out", options["label"].tolist(), key="report_excluded_games",
            )
        picked = options[~options["label"].isin(excluded)]
        selected = game_report.select_player_games(segments, picked[["player_id", "game_id"]].values)

    if selected.empty:
        st.info("Pick at least one team game or player game to draw the report.")
        return

    selected = game_report.filter_min_plays(selected, group_by, int(min_plays))
    if selected.empty:
        st.info(f"No players with at least {int(min_plays)} plays in the selection.")
        return
    table = game_report.category_table(selected, group_by)

    fig = game_report.build_epa_figure(selected, group_by, layout=box_layout)
    # Streamlit's chart theme drops plotly fill patterns, which hides the touchdown boxes.
    st.plotly_chart(fig, use_container_width=True, theme=None)
    with st.expander("Table view", expanded=False):
        st.dataframe(table, use_container_width=True)




main()
