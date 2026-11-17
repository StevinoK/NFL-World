"""Pregame team player-value (PV) covariates from saved player_value weekly plus-minus estimates.

Each team-game gets five values: snap-share-weighted plus-minus of the offense QB1,
WR, RB and TE groups and the defense DL group. Estimates and snap shares come only
from games strictly before kickoff; missing players get the position replacement level.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PM_DIR = REPO / "player_value" / "outputs"
PM_FILE = "plus_minus_weekly_{group}.csv"

GROUP_LIMITS = {"QB": 1, "WR": 4, "RB": 3, "TE": 3, "DL": 4}
GROUP_COLUMN = {"QB": "pv_off_qb", "WR": "pv_off_wr", "RB": "pv_off_rb", "TE": "pv_off_te", "DL": "pv_def_dl"}
PV_OFFENSE_COLUMNS = ["pv_off_qb", "pv_off_wr", "pv_off_rb", "pv_off_te"]
PV_DEFENSE_COLUMNS = ["pv_def_dl"]
PV_COLUMNS = PV_OFFENSE_COLUMNS + PV_DEFENSE_COLUMNS
# Team-health groups (offensive line, linebackers, secondary). Not in trained bundles; opt in with
# ``groups=EXTENDED_GROUP_LIMITS`` for ridge experiments.
HEALTH_GROUP_LIMITS = {"OL": 5, "LB": 3, "DB": 4}
EXTENDED_GROUP_LIMITS = {**GROUP_LIMITS, **HEALTH_GROUP_LIMITS}
EXTENDED_GROUP_COLUMN = {**GROUP_COLUMN, "OL": "pv_off_ol", "LB": "pv_def_lb", "DB": "pv_def_db"}
EXTENDED_PV_COLUMNS = PV_OFFENSE_COLUMNS + ["pv_off_ol"] + PV_DEFENSE_COLUMNS + ["pv_def_lb", "pv_def_db"]
DEFENSE_GROUPS = {"DL", "LB", "DB"}
ROSTER_GROUP = {"QB": "QB", "WR": "WR", "RB": "RB", "FB": "RB", "HB": "RB", "TE": "TE",
                "DL": "DL", "DE": "DL", "DT": "DL", "NT": "DL",
                **{position: "OL" for position in ("OL", "C", "G", "T", "OT", "OG", "LT", "RT", "LG", "RG")},
                **{position: "LB" for position in ("LB", "ILB", "OLB", "MLB")},
                **{position: "DB" for position in ("DB", "CB", "S", "SS", "FS", "SAF")}}
DEPTH_CHART_GROUP = {"QB": "QB", "WR": "WR", "RB": "RB", "FB": "RB", "TE": "TE",
                     "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "LDE": "DL", "RDE": "DL",
                     "LDT": "DL", "RDT": "DL", "EDGE": "DL"}
# player_value.plus_minus.PlayerPlusMinus default history_decay.
CROSS_SEASON_DECAY = 0.8
REPLACEMENT_QUANTILE = 0.25
TRAILING_GAMES = 4
DEFAULT_SHARE = 0.5
TEAM_ALIASES = {"LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV", "JAC": "JAX", "WSH": "WAS"}


def _key(season, week):
    return np.asarray(season, dtype=np.int64) * 100 + np.asarray(week, dtype=np.int64)


def pm_input_hashes(pm_dir: Path = DEFAULT_PM_DIR) -> dict[str, str]:
    hashes = {}
    for group in GROUP_LIMITS:
        path = Path(pm_dir) / PM_FILE.format(group=group)
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        hashes[path.name] = hasher.hexdigest()
    return hashes


def prepare_pm_estimates(raw: pd.DataFrame, group: str) -> pd.DataFrame:
    """One n_on-weighted estimate per player and season-week (traded players span teams)."""
    required = {"player_id", "season", "week", "n_on"}
    if not required <= set(raw):
        raise ValueError(f"{group} plus-minus estimates missing {sorted(required - set(raw))}")
    rows = raw.copy()
    box = pd.to_numeric(rows.get("box_plus_minus", pd.Series(np.nan, index=rows.index)), errors="coerce")
    bayes = pd.to_numeric(rows.get("bayesian_plus_minus", pd.Series(np.nan, index=rows.index)), errors="coerce")
    method = rows.get("estimator", pd.Series("box", index=rows.index)).astype("string").str.lower().fillna("box")
    rows["estimate"] = box.where(method.eq("box"), bayes)
    for column in ("season", "week", "n_on"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows["player_id"] = rows["player_id"].astype("string").str.strip()
    rows = rows.dropna(subset=["player_id", "season", "week", "estimate"])
    rows = rows[np.isfinite(rows["estimate"])]
    rows["n_on"] = rows["n_on"].fillna(0).clip(lower=0)
    rows["weight"] = rows["n_on"].where(rows["n_on"] > 0, 1.0)
    rows["weighted"] = rows["estimate"] * rows["weight"]
    grouped = rows.groupby(["player_id", "season", "week"], as_index=False).agg(
        weighted=("weighted", "sum"), weight=("weight", "sum"), n_on=("n_on", "sum"))
    grouped["estimate"] = grouped["weighted"] / grouped["weight"]
    grouped["group"] = group
    grouped[["season", "week"]] = grouped[["season", "week"]].astype(int)
    return grouped[["group", "player_id", "season", "week", "n_on", "estimate"]]


def load_pm_estimates(pm_dir: Path = DEFAULT_PM_DIR, groups: Iterable[str] = GROUP_LIMITS) -> pd.DataFrame:
    wanted = {"player_id", "season", "week", "n_on", "estimator", "box_plus_minus", "bayesian_plus_minus"}
    frames = []
    for group in groups:
        path = Path(pm_dir) / PM_FILE.format(group=group)
        raw = pd.read_csv(path, usecols=lambda column: column in wanted, dtype={"player_id": "string"}, low_memory=False)
        frames.append(prepare_pm_estimates(raw, group))
    return pd.concat(frames, ignore_index=True)


def _asof(left: pd.DataFrame, right: pd.DataFrame, by: list[str], value_columns: list[str]) -> pd.DataFrame:
    """Latest right row with key strictly before each left key, matched within ``by``."""
    left = left.copy()
    left["_order"] = np.arange(len(left))
    left_sorted = left.sort_values("key")
    right_sorted = right.sort_values("key").rename(columns={"key": "source_key"})
    right_sorted["key"] = right_sorted["source_key"]
    merged = pd.merge_asof(left_sorted, right_sorted[by + ["key", "source_key"] + value_columns],
                           on="key", by=by, direction="backward", allow_exact_matches=False)
    return merged.sort_values("_order").drop(columns="_order").set_index(left.index)


def replacement_levels(estimates: pd.DataFrame, quantile: float = REPLACEMENT_QUANTILE) -> pd.DataFrame:
    frame = estimates.assign(key=_key(estimates.season, estimates.week))
    return frame.groupby(["group", "key"], as_index=False).estimate.quantile(quantile).rename(
        columns={"estimate": "replacement"})


def pm_asof(estimates: pd.DataFrame, requests: pd.DataFrame,
            decay: float = CROSS_SEASON_DECAY) -> pd.DataFrame:
    """Strictly prior PM per (group, player_id, season, week) request, filling unknowns with replacement."""
    if requests.empty:
        return requests.assign(pm=pd.Series(dtype=float), pm_source=pd.Series(dtype=str))
    left = requests[["group", "player_id", "season", "week"]].copy()
    left["player_id"] = left["player_id"].astype(str)
    left["key"] = _key(left.season, left.week)
    right = estimates.assign(key=_key(estimates.season, estimates.week), player_id=estimates.player_id.astype(str))
    found = _asof(left, right[["group", "player_id", "key", "estimate"]], ["group", "player_id"], ["estimate"])
    levels = replacement_levels(estimates)
    fill = _asof(left[["group", "key"]], levels, ["group"], ["replacement"])
    target_season = left["season"].to_numpy(dtype=int)
    source_season = (found["source_key"] // 100).to_numpy()
    fill_season = (fill["source_key"] // 100).to_numpy()
    own = found["estimate"].to_numpy(dtype=float) * np.power(decay, np.nan_to_num(target_season - source_season))
    replacement = fill["replacement"].to_numpy(dtype=float) * np.power(decay, np.nan_to_num(target_season - fill_season))
    known = np.isfinite(own)
    result = requests.copy()
    result["pm"] = np.where(known, own, np.nan_to_num(replacement, nan=0.0))
    result["pm_source"] = np.where(known, "estimate", np.where(np.isfinite(replacement), "replacement", "none"))
    result["pm_source_key"] = np.where(known, found["source_key"], fill["source_key"])
    return result


def snap_share_history(snap_counts: pd.DataFrame, schedule: pd.DataFrame, trailing: int = TRAILING_GAMES) -> pd.DataFrame:
    """Per player-game trailing mean offense/defense share including that game (use strictly later)."""
    snaps = snap_counts.copy()
    if "gsis_id" in snaps:
        snaps = snaps.rename(columns={"gsis_id": "player_id"})
    snaps = snaps.dropna(subset=["player_id", "game_id"])
    snaps["player_id"] = snaps.player_id.astype(str)
    if "game_type" in snaps:
        snaps = snaps[snaps.game_type.astype(str).eq("REG")]
    dates = schedule[["game_id", "gameday"]].drop_duplicates("game_id")
    snaps = snaps.merge(dates, on="game_id", how="inner")
    snaps["game_date"] = pd.to_datetime(snaps.gameday, errors="coerce")
    snaps = snaps.dropna(subset=["game_date"]).sort_values(["player_id", "game_date"])
    history = snaps[["player_id", "game_id", "game_date"]].copy()
    history["team"] = snaps["team"].replace(TEAM_ALIASES).astype(str).to_numpy() if "team" in snaps else ""
    for side, column in (("off", "offense_pct"), ("def", "defense_pct")):
        values = pd.to_numeric(snaps[column], errors="coerce").fillna(0.0)
        history[f"{side}_share"] = values.groupby(snaps.player_id).transform(
            lambda series: series.rolling(trailing, min_periods=1).mean()).to_numpy()
    return history.drop_duplicates(["player_id", "game_id"], keep="last").reset_index(drop=True)


def share_asof(history: pd.DataFrame, requests: pd.DataFrame, with_team: bool = False):
    """Trailing share strictly before each request's kickoff; NaN when the player has no history."""
    if requests.empty:
        return pd.DataFrame(columns=["share", "last_team"]) if with_team else pd.Series(dtype=float)
    left = requests[["player_id", "kickoff", "group"]].copy()
    left["player_id"] = left.player_id.astype(str)
    left["_order"] = np.arange(len(left))
    left["kickoff"] = pd.to_datetime(left.kickoff).astype("datetime64[ns]")
    history = history if "team" in history else history.assign(team="")
    right = history[["player_id", "game_date", "off_share", "def_share", "team"]].rename(
        columns={"game_date": "kickoff", "team": "last_team"})
    right = right.assign(kickoff=pd.to_datetime(right.kickoff).astype("datetime64[ns]"),
                         player_id=right.player_id.astype(str)).sort_values("kickoff")
    merged = pd.merge_asof(left.sort_values("kickoff"), right, on="kickoff", by="player_id",
                           direction="backward", allow_exact_matches=False).sort_values("_order")
    share = pd.Series(np.where(merged.group.isin(DEFENSE_GROUPS), merged.def_share, merged.off_share),
                      index=requests.index, dtype=float)
    if with_team:
        return pd.DataFrame({"share": share, "last_team": merged.last_team.to_numpy()}, index=requests.index)
    return share


def _canonical(frame: pd.DataFrame, columns=("team",)) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame:
            frame[column] = frame[column].replace(TEAM_ALIASES)
    return frame


def pregame_lineups(roster: pd.DataFrame, schedule: pd.DataFrame, history: pd.DataFrame,
                    limits: Mapping[str, int] = GROUP_LIMITS) -> pd.DataFrame:
    """Historical pregame group members: active, not ruled Out, ranked by depth then trailing share.

    The schedule's announced starting QB (home_qb_id/away_qb_id) outranks depth when present.
    """
    required = {"game_id", "team", "player_id", "position"}
    if not required <= set(roster):
        raise ValueError(f"Roster must have {sorted(required)}")
    rows = _canonical(roster)
    rows["group"] = rows.position.map(ROSTER_GROUP)
    rows = rows.dropna(subset=["group", "player_id", "game_id"])
    rows["player_id"] = rows.player_id.astype(str)
    if "status" in rows:
        rows = rows[rows.status.astype("string").fillna("ACT").eq("ACT")]
    if "report_status" in rows:
        rows = rows[~rows.report_status.astype("string").eq("Out").fillna(False).astype(bool)]
    games = _canonical(schedule, ("home_team", "away_team"))[["game_id", "season", "week", "gameday"]].drop_duplicates("game_id")
    rows = rows.drop(columns=[column for column in ("season", "week") if column in rows]).merge(games, on="game_id", how="inner")
    rows["kickoff"] = pd.to_datetime(rows.gameday, errors="coerce")
    rows = rows.drop_duplicates(["game_id", "team", "player_id"])
    trailing = share_asof(history, rows, with_team=True)
    rows["share"] = trailing.share
    rows["same_team"] = trailing.last_team.astype(str).eq(rows.team.astype(str))
    depth = pd.to_numeric(rows.get("depth_team", pd.Series(np.nan, index=rows.index)), errors="coerce")
    rows["depth_rank"] = depth.fillna(99.0)
    starters = {}
    sched = _canonical(schedule, ("home_team", "away_team"))
    for side in ("home", "away"):
        if f"{side}_qb_id" in sched:
            known = sched.dropna(subset=[f"{side}_qb_id"])
            starters.update(zip(zip(known.game_id.astype(str), known[f"{side}_team"].astype(str)), known[f"{side}_qb_id"].astype(str)))
    named = [starters.get((game, team)) == player for game, team, player in zip(rows.game_id.astype(str), rows.team.astype(str), rows.player_id)]
    rows.loc[np.array(named) & rows.group.eq("QB").to_numpy(), "depth_rank"] = 0.0
    rows = rows.sort_values(["game_id", "team", "group", "depth_rank", "same_team", "share", "player_id"],
                            ascending=[True, True, True, True, False, False, True], na_position="last")
    rows["rank"] = rows.groupby(["game_id", "team", "group"]).cumcount() + 1
    rows = rows[rows["rank"] <= rows.group.map(limits)]
    return rows[["game_id", "team", "season", "week", "kickoff", "group", "rank", "player_id", "share"]].reset_index(drop=True)


def default_shares(lineups: pd.DataFrame, max_season: int = 2024) -> dict[tuple[str, int], float]:
    known = lineups[lineups.season.le(max_season) & lineups.share.notna()]
    return {(group, int(rank)): float(value) for (group, rank), value in known.groupby(["group", "rank"]).share.median().items()}


def team_values(lineups: pd.DataFrame, estimates: pd.DataFrame,
                defaults: Mapping[tuple[str, int], float] | None = None,
                columns: Mapping[str, str] = GROUP_COLUMN) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weighted group PM per team-game (weights normalized to 1 within each group)."""
    defaults = defaults or {}
    components = pm_asof(estimates, lineups)
    fallback = [defaults.get((group, int(rank)), DEFAULT_SHARE) for group, rank in zip(components.group, components["rank"])]
    components["share_source"] = np.where(components.share.notna(), "trailing", "default")
    components["share"] = components.share.fillna(pd.Series(fallback, index=components.index)).clip(lower=0.0)
    totals = components.groupby(["game_id", "team", "group"]).share.transform("sum")
    sizes = components.groupby(["game_id", "team", "group"]).share.transform("size")
    components["weight"] = np.where(totals > 0, components.share / totals.where(totals > 0, 1.0), 1.0 / sizes)
    components["contribution"] = components.weight * components.pm
    values = components.groupby(["game_id", "team", "group"]).contribution.sum().unstack("group")
    values = values.rename(columns=dict(columns)).reindex(columns=list(columns.values())).reset_index()
    return values, components


def build_player_value_tables(roster: pd.DataFrame, schedule: pd.DataFrame, snap_counts: pd.DataFrame,
                              estimates: pd.DataFrame, max_default_season: int = 2024,
                              groups: Mapping[str, int] = GROUP_LIMITS):
    columns = {group: EXTENDED_GROUP_COLUMN[group] for group in groups}
    history = snap_share_history(snap_counts, schedule)
    lineups = pregame_lineups(roster, schedule, history, groups)
    defaults = default_shares(lineups, max_default_season)
    values, components = team_values(lineups, estimates, defaults, columns)
    missing = values[list(columns.values())].isna()
    if missing.any().any():
        levels = pm_asof(estimates, lineups.drop_duplicates(["game_id", "group"]).assign(player_id="__replacement__"))
        fill = levels.set_index(["game_id", "group"]).pm
        for group, column in columns.items():
            rows = missing[column]
            values.loc[rows, column] = [fill.get((game_id, group), 0.0) for game_id in values.loc[rows, "game_id"]]
    return values, components, history, defaults


def lineup_requests(assignments: Mapping[str, Mapping[str, str]] | Mapping[str, str], dl_ids: Iterable[str],
                    season: int, week: int, kickoff) -> pd.DataFrame:
    """Simulation lineup (QB1, WR1-4, RB1-3, TE1-3 slots plus DL list) as PV requests."""
    rows = []
    for slot, item in assignments.items():
        group = slot.rstrip("0123456789")
        if group not in GROUP_LIMITS:
            continue
        player_id = item["player_id"] if isinstance(item, Mapping) else item
        rows.append({"group": group, "rank": int(slot[len(group):]), "player_id": str(player_id)})
    for rank, player_id in enumerate(list(dl_ids)[:GROUP_LIMITS["DL"]], start=1):
        rows.append({"group": "DL", "rank": rank, "player_id": str(player_id)})
    frame = pd.DataFrame(rows)
    frame = frame[frame["rank"] <= frame.group.map(GROUP_LIMITS)]
    return frame.assign(game_id="lineup", team="lineup", season=int(season), week=int(week), kickoff=pd.Timestamp(kickoff))


def lineup_player_value(requests: pd.DataFrame, estimates: pd.DataFrame, history: pd.DataFrame,
                        defaults: Mapping[tuple[str, int], float] | None = None) -> tuple[dict[str, float], pd.DataFrame]:
    requests = requests.copy()
    requests["share"] = share_asof(history, requests)
    values, components = team_values(requests, estimates, defaults)
    row = values.iloc[0] if not values.empty else pd.Series(dtype=float)
    result = {}
    for group, column in GROUP_COLUMN.items():
        value = row.get(column, np.nan)
        if not np.isfinite(value):
            filler = pm_asof(estimates, pd.DataFrame([{"group": group, "player_id": "__replacement__",
                                                       "season": int(requests.season.iloc[0]), "week": int(requests.week.iloc[0])}]))
            value = float(filler.pm.iloc[0])
        result[column] = float(value)
    return result, components


def depth_dl_order(roster_week: pd.DataFrame, depth: pd.DataFrame, team: str, kickoff, history: pd.DataFrame) -> list[str]:
    """Active DL ids ranked by latest pregame depth chart, then trailing defense share."""
    rows = roster_week[roster_week.position.map(ROSTER_GROUP).eq("DL")].copy()
    if "status" in rows and rows.status.eq("ACT").any():
        rows = rows[rows.status.eq("ACT")]
    if "report_status" in rows:
        rows = rows[~rows.report_status.astype("string").eq("Out").fillna(False).astype(bool)]
    ids = list(dict.fromkeys(rows.player_id.astype(str)))
    if not ids:
        return []
    depth_rank = {}
    if depth is not None and not depth.empty and {"dt", "pos_abb", "gsis_id"} <= set(depth):
        chart = depth.copy()
        chart["dt"] = pd.to_datetime(chart.dt, errors="coerce", utc=True).dt.tz_convert(None)
        chart = chart[chart.team.astype(str).eq(str(team)) & chart.dt.le(pd.Timestamp(kickoff) + pd.Timedelta(days=1))]
        chart = chart[chart.pos_abb.map(DEPTH_CHART_GROUP).eq("DL")]
        if not chart.empty:
            chart = chart[chart.dt.eq(chart.dt.max())]
            rank_column = next((name for name in ("pos_rank", "depth_chart_position") if name in chart), None)
            if rank_column is not None:
                for player_id, rank in zip(chart.gsis_id.astype(str), pd.to_numeric(chart[rank_column], errors="coerce")):
                    depth_rank[player_id] = min(depth_rank.get(player_id, 99.0), float(rank) if np.isfinite(rank) else 99.0)
    elif "depth_team" in rows:
        for player_id, rank in zip(rows.player_id.astype(str), pd.to_numeric(rows.depth_team, errors="coerce")):
            depth_rank[player_id] = float(rank) if np.isfinite(rank) else 99.0
    shares = share_asof(history, pd.DataFrame({"player_id": ids, "kickoff": pd.Timestamp(kickoff), "group": "DL"}))
    share = dict(zip(ids, shares.fillna(-1.0)))
    return sorted(ids, key=lambda pid: (depth_rank.get(pid, 99.0), -share[pid], pid))
