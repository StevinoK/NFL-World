"""Post-game EPA report: per-player EPA from targets and rushes, split by outcome.

Builds, from nflverse play-by-play, one segment per (play, outcome component) for
each targeted receiver and each ball carrier, and draws them as boxes above
(positive EPA) or below (negative EPA) a horizontal axis of player names. Used by
the "Game report" view of `game_sim_results_dashboard.py`.

Targets (the receiver on a pass play):

* Completion   -> ``air_epa`` is "Reception", ``yac_epa`` is "YAC".
* Completion followed by a lost fumble by the target -> ``air_epa`` is "Reception",
  the rest of the play's EPA is "Fumble lost".
* Incompletion -> play EPA is "Incomplete".
* Interception -> play EPA is "Interception".

Rushes (the ball carrier on a designed run; kneel-downs excluded):

* Run -> play EPA is "Rush"; if the carrier lost a fumble, it is "Fumble lost".

QB plays (credited to the quarterback):

* Scramble -> play EPA is "Scramble" ("Fumble lost" if he lost a fumble).
* Sack -> play EPA is "Sack" ("Fumble lost" if he lost the fumble).

Any of these wiped out by a penalty (``play_type == "no_play"``: DPI,
offensive holding, ...) -> play EPA is "Penalty". nflverse leaves the player ids
empty on these plays, so the target or ball carrier is parsed from the play
description and resolved by jersey number against the same team's plays in that game.

Positions come from nflverse season rosters (``attach_positions``); bars are grouped
QB, RB, WR, TE, then everyone else.

Two-point attempts are excluded (their EPA is not comparable to scrimmage plays).
Touchdowns are marked with a pattern: "/" when the offense scored, "x" when the
defense returned the play for a touchdown (pick-six, fumble return).
"""

from __future__ import annotations

import re
import textwrap
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PBP_COLUMNS = [
    "season", "season_type", "week", "game_id", "play_id", "posteam", "defteam",
    "home_team", "away_team", "qtr", "time", "desc", "play_type", "two_point_attempt",
    "epa", "air_epa", "yac_epa", "complete_pass", "incomplete_pass", "interception",
    "penalty", "fumble_lost", "fumbled_1_player_id", "fumbled_2_player_id",
    "receiver_player_id", "receiver_player_name", "rusher_player_id", "rusher_player_name",
    "qb_kneel", "qb_scramble", "sack", "passer_player_id", "passer_player_name",
    "touchdown", "td_team", "fixed_drive",
]

KIND_TARGET = "Target"
KIND_RUSH = "Rush"
KIND_SCRAMBLE = "Scramble"
KIND_SACK = "Sack"
KINDS = [KIND_TARGET, KIND_RUSH, KIND_SCRAMBLE, KIND_SACK]
KIND_LABELS = {KIND_TARGET: "Targets", KIND_RUSH: "Rushes", KIND_SCRAMBLE: "Scrambles", KIND_SACK: "Sacks"}

# Fixed band order moving away from the axis (same in every bar), and legend order.
# Colors were checked for adjacent-pair separation (incl. color-vision deficiency) in
# this order; the neutral Incomplete band separates the two runs of hues.
CATEGORIES = [
    "Rush", "Scramble", "Reception", "YAC", "Penalty", "Incomplete", "Sack", "Fumble lost", "Interception",
]
CATEGORY_COLORS: Dict[str, str] = {
    "Rush": "#e87ba4",
    "Scramble": "#008300",
    "Reception": "#2a78d6",
    "YAC": "#1baf7a",
    "Penalty": "#eda100",
    "Incomplete": "#8f8e88",
    "Sack": "#eb6834",
    "Fumble lost": "#6250d6",
    "Interception": "#e34948",
}

POSITION_GROUPS = ["QB", "RB", "WR", "TE", "Other"]
_POSITION_ALIASES = {"HB": "RB", "FB": "RB"}
TD_PATTERNS = {"none": "", "offense": "/", "defense": "x"}
GROUP_PLAYER_GAME = "Player per game"
GROUP_PLAYER_SEASON = "Player per season"
GROUP_PLAYER = "Player (combined)"
GROUP_OPTIONS = [GROUP_PLAYER_GAME, GROUP_PLAYER_SEASON, GROUP_PLAYER]

# "pass short left to 18-M.Harrison" / "pass deep right intended for 22-R.Shaheed"
_TARGET_RE = re.compile(r"(?:\bto|intended for) (\d{1,2})-([^\s(\[]+)")
# "22-R.Davis up the middle" / "8-M.Mariota scrambles right tackle"
_RUSH_RE = re.compile(r"(\d{1,2})-([^\s(\[]+) (?:scrambles )?(?:left|right|up the middle)\b")
# "8-L.Jackson sacked at BUF 28" / "15-P.Mahomes pass short left ..."
_SACK_RE = re.compile(r"(\d{1,2})-([^\s(\[]+) sacked\b")
_PASSER_RE = re.compile(r"(\d{1,2})-([^\s(\[]+) (?:pass|sacked|scrambles)\b")
_PENALTY_SPLIT_RE = re.compile(r"penalty", re.IGNORECASE)


def _flag(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(0).astype(float).eq(1)


def _parse_player(desc: object, pattern: "re.Pattern[str]") -> Optional[tuple]:
    """Return (jersey, short_name) of the last ``pattern`` match before any penalty text."""
    if not isinstance(desc, str):
        return None
    before_penalty = _PENALTY_SPLIT_RE.split(desc, maxsplit=1)[0]
    matches = pattern.findall(before_penalty)
    if not matches:
        return None
    jersey, name = matches[-1]
    return jersey, name.rstrip(".,;")


def parse_target(desc: object) -> Optional[tuple]:
    return _parse_player(desc, _TARGET_RE)


def parse_rusher(desc: object) -> Optional[tuple]:
    return _parse_player(desc, _RUSH_RE)


def parse_sacked(desc: object) -> Optional[tuple]:
    return _parse_player(desc, _SACK_RE)


def parse_passer(desc: object) -> Optional[tuple]:
    return _parse_player(desc, _PASSER_RE)


def _ensure_columns(pbp: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PBP_COLUMNS if c not in pbp.columns]
    if not missing:
        return pbp.copy()
    return pbp.assign(**{c: np.nan for c in missing})


def _jersey_roster(pbp: pd.DataFrame) -> Dict[tuple, tuple]:
    """(game_id, team, jersey) -> (player_id, name) from targets and carries with ids."""
    roster: Dict[tuple, tuple] = {}
    for id_col, name_col, parser in (
        ("receiver_player_id", "receiver_player_name", parse_target),
        ("rusher_player_id", "rusher_player_name", parse_rusher),
        ("passer_player_id", "passer_player_name", parse_passer),
    ):
        rows = pbp[pbp[id_col].notna()].drop_duplicates(["game_id", "posteam", id_col])
        for game_id, team, desc, pid, name in zip(
            rows["game_id"], rows["posteam"], rows["desc"], rows[id_col], rows[name_col]
        ):
            parsed = parser(desc)
            if parsed is not None:
                roster.setdefault((game_id, team, parsed[0]), (pid, name))
    return roster


def _attach_nullified_players(pbp: pd.DataFrame) -> pd.DataFrame:
    """Fill receiver/rusher id+name on penalty-nullified plays from the description."""
    roster = _jersey_roster(pbp)
    # Sacked QB, for live sacks straight from the passer columns.
    live_sack = _flag(pbp, "sack") & pbp["play_type"].ne("no_play")
    pbp["sack_player_id"] = pbp["passer_player_id"].where(live_sack)
    pbp["sack_player_name"] = pbp["passer_player_name"].where(live_sack)
    nullified = (
        pbp["play_type"].eq("no_play") & _flag(pbp, "penalty")
        & pbp["receiver_player_id"].isna() & pbp["rusher_player_id"].isna()
    )
    for idx in pbp.index[nullified]:
        desc = pbp.at[idx, "desc"]
        text = desc if isinstance(desc, str) else ""
        before_penalty = _PENALTY_SPLIT_RE.split(text, maxsplit=1)[0]
        if re.search(r"\bsacked\b", before_penalty):
            prefix, parsed = "sack", parse_sacked(desc)
        elif re.search(r"\bpass\b", before_penalty):
            prefix, parsed = "receiver", parse_target(desc)
        else:
            prefix, parsed = "rusher", parse_rusher(desc)
        if parsed is None:
            continue
        jersey, name = parsed
        team = pbp.at[idx, "posteam"]
        pid, pname = roster.get((pbp.at[idx, "game_id"], team, jersey), (f"{team}#{jersey}:{name}", name))
        pbp.at[idx, f"{prefix}_player_id"] = pid
        pbp.at[idx, f"{prefix}_player_name"] = pname
    return pbp


def build_player_segments(pbp: pd.DataFrame, kinds: Sequence[str] = tuple(KINDS)) -> pd.DataFrame:
    """One row per EPA component of every target and/or carry, tagged with its category."""
    pbp = _attach_nullified_players(_ensure_columns(pbp))
    base = pbp[pbp["epa"].notna() & ~_flag(pbp, "two_point_attempt")]
    nullified_all = base["play_type"].eq("no_play")
    parts: List[pd.DataFrame] = []

    def add(plays: pd.DataFrame, kind: str, mask: pd.Series, category: str, values: pd.Series) -> None:
        if mask.any():
            part = plays.loc[mask].copy()
            part["kind"] = kind
            part["category"] = category
            part["seg_epa"] = values[mask]
            parts.append(part)

    if KIND_TARGET in kinds:
        plays = base[base["receiver_player_id"].notna()].rename(
            columns={"receiver_player_id": "player_id", "receiver_player_name": "player_name"}
        )
        pid = plays["player_id"]
        fumbled = (plays["fumbled_1_player_id"] == pid) | (plays["fumbled_2_player_id"] == pid)
        complete = _flag(plays, "complete_pass")
        interception = _flag(plays, "interception")
        incomplete = _flag(plays, "incomplete_pass")
        fumble_lost = complete & _flag(plays, "fumble_lost") & fumbled
        nullified = nullified_all[plays.index]
        live = ~nullified
        air = plays["air_epa"].fillna(0.0)
        epa = plays["epa"]
        add(plays, KIND_TARGET, nullified, "Penalty", epa)
        add(plays, KIND_TARGET, live & interception, "Interception", epa)
        add(plays, KIND_TARGET, live & ~interception & complete, "Reception", air)
        add(plays, KIND_TARGET, live & ~interception & complete & fumble_lost, "Fumble lost", epa - air)
        add(plays, KIND_TARGET, live & ~interception & complete & ~fumble_lost, "YAC", epa - air)
        add(plays, KIND_TARGET, live & ~interception & ~complete & incomplete, "Incomplete", epa)
        # Live target plays with no pass outcome (penalty enforced on the play itself).
        add(plays, KIND_TARGET, live & ~interception & ~complete & ~incomplete, "Penalty", epa)

    if KIND_RUSH in kinds or KIND_SCRAMBLE in kinds:
        plays = base[base["rusher_player_id"].notna() & ~_flag(base, "qb_kneel")]
        plays = plays[plays["play_type"].isin(["run", "no_play"])].rename(
            columns={"rusher_player_id": "player_id", "rusher_player_name": "player_name"}
        )
        scramble = _flag(plays, "qb_scramble") | plays["desc"].astype(str).str.contains(
            r"\d-\S+ scrambles\b", regex=True
        )
        pid = plays["player_id"]
        fumbled = (plays["fumbled_1_player_id"] == pid) | (plays["fumbled_2_player_id"] == pid)
        fumble_lost = _flag(plays, "fumble_lost") & fumbled
        nullified = plays["play_type"].eq("no_play")
        epa = plays["epa"]
        for kind, is_kind, label in ((KIND_RUSH, ~scramble, "Rush"), (KIND_SCRAMBLE, scramble, "Scramble")):
            if kind in kinds:
                add(plays, kind, is_kind & nullified, "Penalty", epa)
                add(plays, kind, is_kind & ~nullified & fumble_lost, "Fumble lost", epa)
                add(plays, kind, is_kind & ~nullified & ~fumble_lost, label, epa)

    if KIND_SACK in kinds:
        plays = base[base["sack_player_id"].notna()].rename(
            columns={"sack_player_id": "player_id", "sack_player_name": "player_name"}
        )
        pid = plays["player_id"]
        fumbled = (plays["fumbled_1_player_id"] == pid) | (plays["fumbled_2_player_id"] == pid)
        fumble_lost = _flag(plays, "fumble_lost") & fumbled
        nullified = plays["play_type"].eq("no_play")
        epa = plays["epa"]
        add(plays, KIND_SACK, nullified, "Penalty", epa)
        add(plays, KIND_SACK, ~nullified & fumble_lost, "Fumble lost", epa)
        add(plays, KIND_SACK, ~nullified & ~fumble_lost, "Sack", epa)

    if not parts:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)
    seg = pd.concat(parts, ignore_index=True)
    td = _flag(seg, "touchdown")
    seg["td"] = np.where(~td, "none", np.where(seg["td_team"] == seg["posteam"], "offense", "defense"))
    seg = seg.rename(columns={"posteam": "team", "defteam": "opponent"})
    seg["epa"] = seg["seg_epa"].astype(float)
    seg["week"] = seg["week"].astype(int)
    seg["season"] = seg["season"].astype(int)
    seg["drive"] = seg["fixed_drive"]
    # Placeholder until attach_positions(): anyone who was sacked or scrambled is a QB.
    qbs = set(seg.loc[seg["kind"].isin([KIND_SACK, KIND_SCRAMBLE]), "player_id"])
    seg["position"] = np.where(seg["player_id"].isin(qbs), "QB", "Other")
    return seg[SEGMENT_COLUMNS].reset_index(drop=True)


def position_group(position: object) -> str:
    """Map a roster position to QB / RB / WR / TE / Other."""
    if not isinstance(position, str) or not position:
        return "Other"
    position = _POSITION_ALIASES.get(position.upper(), position.upper())
    return position if position in POSITION_GROUPS else "Other"


def attach_roster(segments: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Set position groups from an nflverse season roster and resolve placeholder ids.

    ``roster`` needs ``player_id`` and ``position``; with ``team`` and ``jersey_number``
    it also maps placeholder ids ("TEAM#jersey:Name", from penalty-only plays) to the
    real player. Players missing from the roster keep the inferred placeholder
    position (QB if they were sacked or scrambled, otherwise Other).
    """
    if segments.empty or roster is None or roster.empty:
        return segments
    seg = segments.copy()
    if {"team", "jersey_number"} <= set(roster.columns):
        by_jersey = (
            roster.dropna(subset=["team", "jersey_number"])
            .assign(jersey=lambda r: r["jersey_number"].astype(float).astype(int).astype(str))
            .drop_duplicates(["team", "jersey"])
            .set_index(["team", "jersey"])["player_id"]
        )
        synthetic = seg["player_id"].astype(str).str.extract(r"^([A-Z]+)#(\d+):")
        keys = list(zip(synthetic[0], synthetic[1]))
        resolved = pd.Series([by_jersey.get(k) if isinstance(k[0], str) else None for k in keys], index=seg.index)
        seg["player_id"] = resolved.fillna(seg["player_id"])
    positions = roster.dropna(subset=["player_id"]).drop_duplicates("player_id").set_index("player_id")["position"]
    mapped = seg["player_id"].map(positions).map(lambda p: position_group(p) if isinstance(p, str) else None)
    seg["position"] = mapped.fillna(seg["position"])
    return seg


def attach_positions(segments: pd.DataFrame, roster_positions: Dict[str, str]) -> pd.DataFrame:
    """Shorthand for ``attach_roster`` with only a {player_id: position} mapping."""
    roster = pd.DataFrame({"player_id": list(roster_positions), "position": list(roster_positions.values())})
    return attach_roster(segments, roster)


def build_target_segments(pbp: pd.DataFrame) -> pd.DataFrame:
    """Targets only (receiving EPA)."""
    return build_player_segments(pbp, kinds=(KIND_TARGET,))


SEGMENT_COLUMNS = [
    "season", "season_type", "week", "game_id", "play_id", "team", "opponent", "qtr",
    "time", "desc", "player_id", "player_name", "position", "kind", "category", "epa", "td", "drive",
]


def week_label(season: int, week: int, season_type: str = "REG") -> str:
    suffix = "" if season_type == "REG" else " (P)"
    return f"{season} W{week}{suffix}"


def week_options(segments: pd.DataFrame) -> Dict[str, tuple]:
    """Chronological {label: (season, week)} for every week present in ``segments``."""
    weeks = (
        segments[["season", "week", "season_type"]]
        .drop_duplicates(["season", "week"])
        .sort_values(["season", "week"])
    )
    return {week_label(r.season, r.week, r.season_type): (int(r.season), int(r.week)) for r in weeks.itertuples()}


def games_in_range(catalog: pd.DataFrame, start: tuple, end: tuple) -> pd.DataFrame:
    """Rows of a game catalog whose (season, week) falls within [start, end], inclusive."""
    if catalog.empty:
        return catalog
    keys = list(zip(catalog["season"].astype(int), catalog["week"].astype(int)))
    low, high = min(start, end), max(start, end)
    return catalog[[low <= key <= high for key in keys]]


def team_game_catalog(segments: pd.DataFrame) -> pd.DataFrame:
    """Unique (game, offense) pairs, most recent first, with a display label."""
    cat = (
        segments[["season", "season_type", "week", "game_id", "team", "opponent"]]
        .drop_duplicates(["game_id", "team"])
        .sort_values(["season", "week", "team"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    cat["label"] = [
        f"{week_label(r.season, r.week, r.season_type)} · {r.team} vs {r.opponent}"
        for r in cat.itertuples()
    ]
    return cat


def _usage(targets: int, rushes: int, scrambles: int = 0, sacks: int = 0) -> str:
    parts = [(targets, "tgt"), (rushes, "rush"), (scrambles, "scr"), (sacks, "sk")]
    return ", ".join(f"{n} {unit}" for n, unit in parts if n)


def player_game_catalog(segments: pd.DataFrame) -> pd.DataFrame:
    """Unique (player, game) pairs with target/rush counts and net EPA, most recent first."""
    plays = segments.drop_duplicates(["game_id", "play_id", "player_id"])
    counts = plays.groupby(["player_id", "game_id", "kind"]).size().unstack("kind")
    counts = counts.reindex(columns=KINDS).fillna(0).astype(int)
    counts.columns = ["targets", "rushes", "scrambles", "sacks"]
    net = segments.groupby(["player_id", "game_id"])["epa"].sum().rename("net_epa")
    base = (
        segments[["player_id", "player_name", "position", "season", "season_type", "week", "game_id", "team",
                  "opponent"]]
        .drop_duplicates(["player_id", "game_id"])
        .set_index(["player_id", "game_id"])
        .join(counts)
        .join(net)
        .reset_index()
        .sort_values(["season", "week", "player_name"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    base["label"] = [
        f"{r.player_name} ({r.team}) · {week_label(r.season, r.week, r.season_type)} vs {r.opponent}"
        f" · {_usage(r.targets, r.rushes, r.scrambles, r.sacks)}, {r.net_epa:+.1f} EPA"
        for r in base.itertuples()
    ]
    return base


def _unit_columns(segments: pd.DataFrame, group_by: str) -> pd.DataFrame:
    seg = segments.copy()
    if group_by == GROUP_PLAYER:
        seg["unit"] = seg["player_id"].astype(str)
        teams = seg.groupby("unit")["team"].agg(lambda s: "/".join(dict.fromkeys(s)))
        seg["unit_label"] = seg["player_name"] + "<br>" + seg["unit"].map(teams)
    elif group_by == GROUP_PLAYER_SEASON:
        seg["unit"] = seg["player_id"].astype(str) + "|" + seg["season"].astype(str)
        teams = seg.groupby("unit")["team"].agg(lambda s: "/".join(dict.fromkeys(s)))
        seg["unit_label"] = seg["player_name"] + "<br>" + seg["unit"].map(teams) + " " + seg["season"].astype(str)
    else:
        seg["unit"] = seg["player_id"].astype(str) + "|" + seg["game_id"].astype(str)
        seg["unit_label"] = [
            f"{r.player_name}<br>{r.team} W{r.week}" + ("" if r.season_type == "REG" else "P")
            + f" v {r.opponent}"
            for r in seg.itertuples()
        ]
    # Labels must be unique per unit (two players can share a short name).
    dup = seg.groupby("unit_label")["unit"].transform("nunique") > 1
    seg.loc[dup, "unit_label"] = seg.loc[dup, "unit_label"] + " #" + seg.loc[dup, "player_id"].astype(str).str[-4:]
    if "position" not in seg:
        seg["position"] = "Other"
    # One position group per bar (a player's most frequent group in the selection).
    seg["unit_position"] = seg.groupby("unit")["position"].transform(lambda s: s.mode().iloc[0])
    return seg


def stack_segments(segments: pd.DataFrame, group_by: str = GROUP_PLAYER_GAME) -> pd.DataFrame:
    """Add unit, base and order so each segment is a box stacked away from zero.

    Positive segments stack upward from 0 and negative ones downward; within each
    direction boxes are grouped by category (CATEGORIES order) then chronologically.
    """
    if segments.empty:
        return segments.assign(unit=[], unit_label=[], base=[])
    seg = _unit_columns(segments, group_by)
    seg = seg[seg["epa"] != 0].copy()
    seg["cat_order"] = seg["category"].map({c: i for i, c in enumerate(CATEGORIES)})
    seg["positive"] = seg["epa"] > 0
    seg = seg.sort_values(["unit", "positive", "cat_order", "season", "week", "play_id"])
    seg["top"] = seg.groupby(["unit", "positive"])["epa"].cumsum()
    seg["base"] = seg["top"] - seg["epa"]
    return seg.drop(columns=["top"]).reset_index(drop=True)


def _position_rank(position: str) -> int:
    return POSITION_GROUPS.index(position) if position in POSITION_GROUPS else len(POSITION_GROUPS)


def unit_order(stacked: pd.DataFrame) -> List[str]:
    """Bars grouped by position (QB, RB, WR, TE, Other), best net EPA first within each."""
    units = stacked.groupby("unit_label").agg(net=("epa", "sum"), position=("unit_position", "first"))
    units["rank"] = units["position"].map(_position_rank)
    units = units.sort_values(["rank", "net"], ascending=[True, False])
    return list(units.index)


def position_spans(stacked: pd.DataFrame, order: Sequence[str]) -> List[tuple]:
    """(position, first_index, last_index) for each contiguous position group in ``order``."""
    by_label = stacked.groupby("unit_label")["unit_position"].first()
    spans: List[tuple] = []
    for i, label in enumerate(order):
        position = by_label[label]
        if spans and spans[-1][0] == position:
            spans[-1] = (position, spans[-1][1], i)
        else:
            spans.append((position, i, i))
    return spans


def squarify(areas: Sequence[float], x: float, y: float, w: float, h: float) -> List[tuple]:
    """Squarified treemap (Bruls et al.): tile rectangle (x, y, w, h) with ``areas``.

    ``areas`` must be positive, sorted descending, and sum to ``w * h``. Returns one
    (x, y, w, h) per area in the same order; the first rows sit at (x, y).
    """
    def worst(row: List[float], side: float) -> float:
        total = sum(row)
        return max(side * side * max(row) / (total * total), total * total / (side * side * min(row)))

    rects: List[tuple] = []
    remaining = list(areas)
    while remaining:
        if len(remaining) == 1 or w <= 0 or h <= 0:
            row = remaining
        else:
            side = min(w, h)
            row = [remaining[0]]
            for area in remaining[1:]:
                if worst(row + [area], side) > worst(row, side):
                    break
                row.append(area)
        total = sum(row)
        if w >= h:  # lay the row as a column against the left edge
            col_w = total / h if h > 0 else 0.0
            cy = y
            for area in row:
                rh = area / col_w if col_w > 0 else 0.0
                rects.append((x, cy, col_w, rh))
                cy += rh
            x, w = x + col_w, w - col_w
        else:  # lay the row along the bottom edge
            row_h = total / w if w > 0 else 0.0
            cx = x
            for area in row:
                rw = area / row_h if row_h > 0 else 0.0
                rects.append((cx, y, rw, row_h))
                cx += rw
            y, h = y + row_h, h - row_h
        remaining = remaining[len(row):]
    return rects


LAYOUT_PACKED = "Packed (area = EPA)"
LAYOUT_STACKED = "Stacked (height = EPA)"
SLOT_WIDTH = 0.8  # share of each player's x slot used by its boxes


def pack_segments(
    stacked: pd.DataFrame,
    order: Sequence[str],
    px_per_x: float = 80.0,
    px_per_y: float = 20.0,
    slot_width: float = SLOT_WIDTH,
) -> pd.DataFrame:
    """Place each segment as a rectangle whose area is proportional to |EPA|.

    Every player gets a strip ``slot_width`` wide whose height equals its total
    positive (above the axis) or negative (below) EPA, so a full-width box of height
    h is worth h EPA and the y axis stays in EPA. Inside the strip, each outcome
    category gets a full-width band in the fixed CATEGORIES order moving away from
    the axis (so colors read the same in every bar), and that category's plays are
    tiled inside its band with a squarified treemap computed in pixel space so the
    tiles look as square as the band allows.
    """
    positions = {label: i for i, label in enumerate(order)}
    out = stacked.copy()
    boxes: Dict[object, tuple] = {}
    for (label, positive), strip in out.groupby(["unit_label", "positive"]):
        total = strip["epa"].abs().sum()
        left = positions[label] - slot_width / 2
        w_px, h_px = slot_width * px_per_x, total * px_per_y
        scale = w_px * h_px / total  # pixel area per EPA
        cat_totals = strip.groupby("category")["epa"].apply(lambda s: s.abs().sum())
        cats = [c for c in CATEGORIES if c in cat_totals.index]
        band_y, cat_rects = 0.0, []
        for category in cats:
            band_h = cat_totals[category] * scale / w_px
            cat_rects.append((0.0, band_y, w_px, band_h))
            band_y += band_h
        for category, (cx, cy, cw, ch) in zip(cats, cat_rects):
            plays = strip[strip["category"] == category]
            plays = plays.reindex(plays["epa"].abs().sort_values(ascending=False).index)
            areas = (plays["epa"].abs() * scale).tolist()
            # Rescale so floating error never overflows the category rectangle.
            areas = [a * (cw * ch) / sum(areas) for a in areas]
            for idx, (px, py, pw, ph) in zip(plays.index, squarify(areas, cx, cy, cw, ch)):
                y_lo, y_hi = py / px_per_y, (py + ph) / px_per_y
                if not positive:
                    y_lo, y_hi = -y_hi, -y_lo
                boxes[idx] = (left + px / px_per_x, left + (px + pw) / px_per_x, y_lo, y_hi)
    coords = pd.DataFrame.from_dict(boxes, orient="index", columns=["x0", "x1", "y0", "y1"])
    return out.join(coords)


def _hover_desc(desc: object, width: int = 70) -> str:
    return "<br>".join(textwrap.wrap(str(desc), width)[:5])


def build_epa_figure(
    segments: pd.DataFrame,
    group_by: str = GROUP_PLAYER_GAME,
    title: str = "Player EPA by outcome",
    layout: str = LAYOUT_PACKED,
    plot_width_px: float = 1100.0,
    height: int = 620,
) -> go.Figure:
    stacked = stack_segments(segments, group_by)
    fig = go.Figure()
    if stacked.empty:
        fig.update_layout(title=title)
        return fig
    order = unit_order(stacked)
    positions = {label: i for i, label in enumerate(order)}

    ups = stacked[stacked["positive"]].groupby("unit_label")["epa"].sum()
    downs = stacked[~stacked["positive"]].groupby("unit_label")["epa"].sum()
    top = max(float(ups.max()) if len(ups) else 0.0, 0.5)
    bottom = min(float(downs.min()) if len(downs) else 0.0, -0.5)
    pad = 0.04 * (top - bottom)
    y_range = [bottom - pad, top + 2.5 * pad]  # extra headroom for the position headers

    if layout == LAYOUT_PACKED:
        plot_height_px = height - 110 - 110  # minus top legend/title and bottom tick labels
        rects = pack_segments(
            stacked, order,
            px_per_x=plot_width_px / len(order),
            px_per_y=plot_height_px / (y_range[1] - y_range[0]),
        )
        rects["x_mid"] = (rects["x0"] + rects["x1"]) / 2
        rects["box_w"] = rects["x1"] - rects["x0"]
        rects["box_base"] = rects["y0"]
        rects["box_h"] = rects["y1"] - rects["y0"]
    else:
        rects = stacked.copy()
        rects["x_mid"] = rects["unit_label"].map(positions).astype(float)
        rects["box_w"] = 0.7
        rects["box_base"] = rects["base"]
        rects["box_h"] = rects["epa"]

    # Box outlines separate plays; thin them when bars hold many plays (combined view).
    max_boxes = int(stacked.groupby("unit").size().max())
    outline = 1.5 if max_boxes <= 30 else 0.75 if max_boxes <= 80 else 0.4

    for category in CATEGORIES:
        part = rects[rects["category"] == category]
        if part.empty:
            continue
        # Plotly draws the legend swatch from the first bar; keep that one unpatterned.
        part = part.sort_values("td", key=lambda s: s.ne("none"), kind="stable")
        custom = np.stack([
            part["player_name"],
            [f"{week_label(s, w, t)} {tm} vs {o}" for s, w, t, tm, o in
             zip(part["season"], part["week"], part["season_type"], part["team"], part["opponent"])],
            [f"Q{int(q)} {t}" if pd.notna(q) else "" for q, t in zip(part["qtr"], part["time"])],
            [_hover_desc(d) for d in part["desc"]],
            np.where(part["td"] == "none", "", np.where(part["td"] == "offense", " · TD", " · Defensive TD")),
            [f"{e:+.2f}" for e in part["epa"]],
        ], axis=-1)
        fig.add_trace(go.Bar(
            name=category,
            legendgroup=category,
            x=part["x_mid"],
            y=part["box_h"],
            base=part["box_base"],
            width=part["box_w"],
            marker=dict(
                color=CATEGORY_COLORS[category],
                line=dict(color="rgba(255,255,255,0.85)", width=outline),
                pattern=dict(
                    shape=[TD_PATTERNS[t] for t in part["td"]],
                    fillmode="overlay",
                    fgcolor="rgba(255,255,255,0.9)",
                    size=7,
                    solidity=0.3,
                ),
            ),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b> · %{customdata[1]}<br>%{customdata[2]}<br>"
                f"{category}: " + "%{customdata[5]} EPA%{customdata[4]}<br><br>%{customdata[3]}<extra></extra>"
            ),
        ))

    # Legend keys for the touchdown patterns.
    for td_kind, label in (("offense", "Touchdown"), ("defense", "Defensive TD (return)")):
        if (stacked["td"] == td_kind).any():
            fig.add_trace(go.Bar(
                name=label, x=[0], y=[0], base=[0], width=[0], showlegend=True, hoverinfo="skip",
                marker=dict(color="rgba(0,0,0,0)", line=dict(color="#8f8e88", width=1),
                            pattern=dict(shape=TD_PATTERNS[td_kind], fillmode="replace", fgcolor="#52514e", size=7)),
            ))

    net = stacked.assign(play_key=stacked["game_id"].astype(str) + ":" + stacked["play_id"].astype(str)).groupby(
        "unit_label"
    ).agg(net=("epa", "sum"), player=("player_name", "first"), plays=("play_key", "nunique"),
          position=("unit_position", "first")).reindex(order)
    fig.add_trace(go.Scatter(
        name="Net EPA", x=list(range(len(order))), y=net["net"], mode="markers",
        marker=dict(symbol="diamond", size=10, color="#0b0b0b", line=dict(color="white", width=2)),
        customdata=np.stack([net["player"], net["plays"], net["position"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b> (%{customdata[2]})<br>Net %{y:+.2f} EPA on %{customdata[1]} plays<extra></extra>",
    ))

    # Position groups: alternating shaded bands, separators and a header per group.
    for n, (position, first, last) in enumerate(position_spans(stacked, order)):
        if n % 2 == 0:
            fig.add_shape(type="rect", xref="x", yref="paper", x0=first - 0.5, x1=last + 0.5, y0=0, y1=1,
                          fillcolor="rgba(128,128,128,0.07)", line=dict(width=0), layer="below")
        if n > 0:
            fig.add_shape(type="line", xref="x", yref="paper", x0=first - 0.5, x1=first - 0.5, y0=0, y1=1,
                          line=dict(color="rgba(128,128,128,0.45)", width=1, dash="dot"))
        count = last - first + 1
        name = {"Other": "Other"}.get(position, position + ("s" if count > 1 else ""))
        fig.add_annotation(x=(first + last) / 2, y=1.0, xref="x", yref="paper", yanchor="top",
                           text=f"<b>{name}</b>", showarrow=False, font=dict(size=13, color="#52514e"),
                           bgcolor="rgba(255,255,255,0.75)")

    y_title = "EPA (box area ∝ EPA)" if layout == LAYOUT_PACKED else "EPA"
    fig.add_hline(y=0, line=dict(color="#52514e", width=1.5))
    fig.update_layout(
        title=title,
        barmode="overlay",
        height=height,
        hovermode="closest",
        title_y=0.97,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(t=110, b=40, l=60, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            tickmode="array", tickvals=list(range(len(order))), ticktext=order,
            range=[-0.6, len(order) - 0.4], showgrid=False, zeroline=False,
            tickangle=-45 if len(order) > 12 else 0, automargin=True,
        ),
        yaxis=dict(title=y_title, range=y_range, zeroline=False, gridcolor="rgba(128,128,128,0.18)"),
    )
    return fig


def category_table(segments: pd.DataFrame, group_by: str = GROUP_PLAYER_GAME) -> pd.DataFrame:
    """Per-unit EPA by category (the table view of the chart)."""
    if segments.empty:
        return pd.DataFrame()
    seg = _unit_columns(segments, group_by)
    seg["unit_label"] = seg["unit_label"].str.replace("<br>", " · ", regex=False)
    table = seg.pivot_table(index="unit_label", columns="category", values="epa", aggfunc="sum", fill_value=0.0)
    table = table.reindex(columns=[c for c in CATEGORIES if c in table.columns])
    plays = seg.drop_duplicates(["unit", "game_id", "play_id"])
    counts = plays.groupby(["unit_label", "kind"]).size().unstack("kind").reindex(columns=KINDS).fillna(0)
    tds = plays[plays["td"] == "offense"].groupby("unit_label").size()
    position = seg.groupby("unit_label")["unit_position"].first()
    table.insert(0, "Pos", position.reindex(table.index))
    for n, kind in enumerate(KINDS, start=1):
        table.insert(n, KIND_LABELS[kind], counts[kind].reindex(table.index).fillna(0).astype(int))
    # Hide play-type columns that are all zero (e.g. no QBs in the selection).
    empty = [KIND_LABELS[k] for k in KINDS if k != KIND_TARGET and not table[KIND_LABELS[k]].any()]
    table = table.drop(columns=empty)
    table["TDs"] = tds.reindex(table.index).fillna(0).astype(int)
    table["Net EPA"] = table[[c for c in CATEGORIES if c in table.columns]].sum(axis=1)
    table.index.name = "Player"
    table.columns.name = None
    table["_rank"] = table["Pos"].map(_position_rank)
    table = table.sort_values(["_rank", "Net EPA"], ascending=[True, False]).drop(columns="_rank")
    return table.round(2)


def filter_min_plays(segments: pd.DataFrame, group_by: str, min_plays: int) -> pd.DataFrame:
    """Drop units (player-games, -seasons or players) with fewer than ``min_plays`` touches.

    A touch is a target or a carry.
    """
    if segments.empty or min_plays <= 1:
        return segments
    units = _unit_columns(segments, group_by)["unit"]
    plays = segments.assign(_unit=units.values).drop_duplicates(["_unit", "game_id", "play_id"])
    counts = plays.groupby("_unit").size()
    keep = units.map(counts).fillna(0).values >= min_plays
    return segments[keep]


filter_min_targets = filter_min_plays  # backward-compatible name


def select_drives(segments: pd.DataFrame, game_id: str, drives: Iterable[int]) -> pd.DataFrame:
    """Keep only ``game_id``'s plays from the given ``fixed_drive`` numbers."""
    wanted = {int(d) for d in drives}
    in_game = segments["game_id"] == game_id
    return segments[in_game & segments["drive"].isin(wanted)]


def select_team_games(segments: pd.DataFrame, game_team_pairs: Iterable[Sequence[str]]) -> pd.DataFrame:
    pairs = pd.DataFrame(list(game_team_pairs), columns=["game_id", "team"])
    return segments.merge(pairs, on=["game_id", "team"])


def select_player_games(segments: pd.DataFrame, player_game_pairs: Iterable[Sequence[str]]) -> pd.DataFrame:
    pairs = pd.DataFrame(list(player_game_pairs), columns=["player_id", "game_id"])
    return segments.merge(pairs, on=["player_id", "game_id"])
