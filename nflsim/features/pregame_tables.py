"""Pregame team and league tables for refits that add era, kicker, tendency and run-direction inputs.

One row per (season, week, team), each value from regular-season plays strictly before that week, with older weeks
discounted (``WEEK_DECAY`` per week; the offseason counts as several weeks) and team rates shrunk toward the league:

- ``pg_league_*``: the league environment (early-down neutral pass rate, go-for-it rate on 4th-and-2+ within 54
  yards of a field goal, field goal make rates 40-49 and 50+, kickoff touchback rate, two-point share of tries);
- ``pg_off_kicker_fg_oe``: the team's current kicker's field goal makes over expected per attempt (expected by
  distance from earlier seasons), following the kicker across teams;
- ``pg_off_screen_rate``, ``pg_off_play_action_rate``: offense tendencies on dropbacks (FTN charting, where present);
- ``pg_off_run_<dir>_ypc`` / ``pg_def_run_<dir>_ypc``: rushing yards per carry gained / allowed to the left, middle
  and right.

Training joins ``pg_off_*`` and ``pg_league_*`` on the offense and ``pg_def_*`` on the defense
(``attach_pregame_tables``); the simulator reads the same table through NFLSIM_PREGAME_TABLES.

    python -m nflsim.features.pregame_tables --pbp-dir <folder of pbp_<season>.parquet> --seasons 2016-2026 \\
        --output nflsim/outputs/pregame_tables/pregame_tables.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nflsim.paths import output_dir

WEEK_DECAY = 0.92
OFFSEASON_WEEKS = 8
TEAM_PRIOR_PLAYS = {"screen": 60.0, "play_action": 60.0, "run": 40.0, "kicker": 20.0}
DIRECTIONS = ("left", "middle", "right")
LEAGUE_COLUMNS = ["pg_league_pass_rate", "pg_league_go_rate_fg_zone", "pg_league_fg_make_40_49",
                  "pg_league_fg_make_50_plus", "pg_league_ko_touchback", "pg_league_two_point_share"]
OFFENSE_COLUMNS = ["pg_off_kicker_fg_oe", "pg_off_screen_rate", "pg_off_play_action_rate",
                   *[f"pg_off_run_{direction}_ypc" for direction in DIRECTIONS]]
DEFENSE_COLUMNS = [f"pg_def_run_{direction}_ypc" for direction in DIRECTIONS]
COLUMNS = [*LEAGUE_COLUMNS, *OFFENSE_COLUMNS, *DEFENSE_COLUMNS]
PBP_COLUMNS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "down", "ydstogo", "yardline_100",
               "qtr", "score_differential", "play_type", "pass", "rush", "qb_scramble", "qb_dropback",
               "field_goal_attempt", "field_goal_result", "kick_distance", "kicker_player_id", "kickoff_attempt",
               "touchback", "two_point_attempt", "extra_point_attempt", "run_location", "yards_gained",
               "is_screen_pass", "is_play_action"]


def week_index(season, week):
    """Weeks on one clock, with the offseason as OFFSEASON_WEEKS extra weeks."""
    return np.asarray(season, dtype=float) * (22 + OFFSEASON_WEEKS) + np.asarray(week, dtype=float)


def target_weeks(pbp: pd.DataFrame) -> pd.DataFrame:
    """Every played (season, week) plus the week after the last one, for forecasts."""
    weeks = pbp[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    last = weeks.iloc[-1]
    return pd.concat([weeks, pd.DataFrame({"season": [last.season], "week": [last.week + 1]})], ignore_index=True)


def decayed_before(events: pd.DataFrame, keys: list[str], values: list[str], weeks: pd.DataFrame) -> pd.DataFrame:
    """For each target week and key, the sum of ``values`` over earlier weeks, each discounted by WEEK_DECAY per
    week between them. ``events`` has season, week, the keys and the values (already summed or not)."""
    if events.empty:
        return pd.DataFrame(columns=["season", "week", *keys, *values])
    sums = events.groupby(["season", "week", *keys], dropna=False)[values].sum().reset_index()
    sums["t"] = week_index(sums.season, sums.week)
    targets = weeks.assign(t=week_index(weeks.season, weeks.week)).sort_values("t")
    out = []
    groups = sums.groupby(keys, dropna=False) if keys else [((), sums)]
    for key, group in groups:
        group = group.sort_values("t")
        t, x = group.t.to_numpy(), group[values].to_numpy(dtype=float)
        for target in targets.itertuples(index=False):
            earlier = t < target.t
            if not earlier.any():
                continue
            weight = WEEK_DECAY ** (target.t - t[earlier] - 1)
            totals = weight @ x[earlier]
            out.append({"season": target.season, "week": target.week,
                        **dict(zip(keys, key if isinstance(key, tuple) else (key,))), **dict(zip(values, totals))})
    return pd.DataFrame(out)


def _flag(frame, column):
    return frame[column].eq(1) if column in frame else pd.Series(False, index=frame.index)


def league_table(pbp: pd.DataFrame, weeks: pd.DataFrame) -> pd.DataFrame:
    plays = pbp
    neutral = plays.down.isin([1, 2]) & plays.qtr.le(3) & plays.score_differential.abs().le(7) & plays.play_type.isin(["pass", "run"])
    fourth = plays.down.eq(4) & plays.play_type.isin(["pass", "run", "punt", "field_goal"]) & plays.qtr.le(4)
    zone = fourth & (plays.yardline_100 + 17).le(54) & plays.ydstogo.ge(2)
    fg = _flag(plays, "field_goal_attempt") & plays.kick_distance.notna()
    made = plays.field_goal_result.eq("made")
    events = pd.DataFrame({
        "season": plays.season, "week": plays.week,
        "pass_n": _flag(plays, "pass") & neutral, "pass_d": neutral,
        "go_n": zone & plays.play_type.isin(["pass", "run"]), "go_d": zone,
        "fg40_n": fg & plays.kick_distance.between(40, 49) & made, "fg40_d": fg & plays.kick_distance.between(40, 49),
        "fg50_n": fg & plays.kick_distance.ge(50) & made, "fg50_d": fg & plays.kick_distance.ge(50),
        "tb_n": _flag(plays, "kickoff_attempt") & _flag(plays, "touchback"), "tb_d": _flag(plays, "kickoff_attempt"),
        "two_n": _flag(plays, "two_point_attempt"), "two_d": _flag(plays, "two_point_attempt") | _flag(plays, "extra_point_attempt"),
    }).astype({column: float for column in ["pass_n", "pass_d", "go_n", "go_d", "fg40_n", "fg40_d", "fg50_n", "fg50_d", "tb_n", "tb_d", "two_n", "two_d"]})
    sums = decayed_before(events, [], [column for column in events if column not in ("season", "week")], weeks)
    table = sums[["season", "week"]].copy()
    for column, stem in zip(LEAGUE_COLUMNS, ["pass", "go", "fg40", "fg50", "tb", "two"]):
        table[column] = sums[f"{stem}_n"] / sums[f"{stem}_d"].where(sums[f"{stem}_d"] > 0)
    return table


def team_rates(events, team_column, numerator, denominator, weeks, prior_plays):
    """Shrunk decayed rate per team-week: (num + k * league) / (den + k)."""
    sums = decayed_before(events, [team_column], [numerator, denominator], weeks)
    league_sums = decayed_before(events, [], [numerator, denominator], weeks)
    league_sums["league"] = league_sums[numerator] / league_sums[denominator].where(league_sums[denominator] > 0)
    merged = sums.merge(league_sums[["season", "week", "league"]], on=["season", "week"], how="left")
    rate = (merged[numerator] + prior_plays * merged.league) / (merged[denominator] + prior_plays)
    return merged[["season", "week", team_column]].assign(rate=rate).rename(columns={team_column: "team"})


def kicker_table(pbp: pd.DataFrame, weeks: pd.DataFrame) -> pd.DataFrame:
    """The current kicker's field goal makes over expected per attempt, shrunk; expected by 5-yard distance band
    from earlier seasons."""
    fg = pbp[_flag(pbp, "field_goal_attempt") & pbp.kick_distance.notna() & pbp.kicker_player_id.notna()].copy()
    if fg.empty:
        return pd.DataFrame(columns=["season", "week", "team", "pg_off_kicker_fg_oe"])
    fg["band"] = (fg.kick_distance // 5).clip(3, 12)
    fg["made"] = fg.field_goal_result.eq("made").astype(float)
    expected = []
    for season in sorted(fg.season.unique()):
        earlier = fg[fg.season < season]
        source = earlier if len(earlier) else fg[fg.season == season]
        rates = source.groupby("band").made.mean()
        rows = fg.season == season
        expected.append(fg.loc[rows, "band"].map(rates).fillna(source.made.mean()))
    fg["over"] = fg.made - pd.concat(expected)
    fg["attempts"] = 1.0
    sums = decayed_before(fg[["season", "week", "kicker_player_id", "over", "attempts"]], ["kicker_player_id"],
                          ["over", "attempts"], weeks)
    sums["oe"] = sums.over / (sums.attempts + TEAM_PRIOR_PLAYS["kicker"])
    kicks = pbp[(_flag(pbp, "field_goal_attempt") | _flag(pbp, "extra_point_attempt")) & pbp.kicker_player_id.notna()]
    kicks = kicks.assign(t=week_index(kicks.season, kicks.week)).sort_values("t")
    targets = weeks.assign(t=week_index(weeks.season, weeks.week))
    rows = []
    for team, team_kicks in kicks.groupby("posteam"):
        t, kicker = team_kicks.t.to_numpy(), team_kicks.kicker_player_id.to_numpy()
        for target in targets.itertuples(index=False):
            earlier = np.flatnonzero(t < target.t)
            if len(earlier):
                rows.append({"season": target.season, "week": target.week, "team": team, "kicker_player_id": kicker[earlier[-1]]})
    current = pd.DataFrame(rows).merge(sums[["season", "week", "kicker_player_id", "oe"]],
                                       on=["season", "week", "kicker_player_id"], how="left")
    return current.assign(pg_off_kicker_fg_oe=current.oe.fillna(0.0))[["season", "week", "team", "pg_off_kicker_fg_oe"]]


def build_tables(pbp: pd.DataFrame) -> pd.DataFrame:
    from nflsim.features.builder import canonicalize_teams
    pbp = canonicalize_teams(pbp[pbp.season_type.eq("REG") & pbp.posteam.notna()])
    weeks = target_weeks(pbp)
    teams = pd.DataFrame({"team": sorted(set(pbp.posteam.dropna()) | set(pbp.defteam.dropna()))})
    table = weeks.merge(teams, how="cross").merge(league_table(pbp, weeks), on=["season", "week"], how="left")
    table = table.merge(kicker_table(pbp, weeks), on=["season", "week", "team"], how="left")
    dropbacks = pbp[_flag(pbp, "qb_dropback")]
    for column, flag, prior in (("pg_off_screen_rate", "is_screen_pass", "screen"), ("pg_off_play_action_rate", "is_play_action", "play_action")):
        if flag not in dropbacks:
            table[column] = np.nan
            continue
        charted = dropbacks[dropbacks[flag].notna()]
        events = pd.DataFrame({"season": charted.season, "week": charted.week, "posteam": charted.posteam,
                               "n": charted[flag].astype(float), "d": 1.0})
        rates = team_rates(events, "posteam", "n", "d", weeks, TEAM_PRIOR_PLAYS[prior]).rename(columns={"rate": column})
        table = table.merge(rates, on=["season", "week", "team"], how="left")
    runs = pbp[_flag(pbp, "rush") & ~_flag(pbp, "qb_scramble") & pbp.run_location.isin(DIRECTIONS) & pbp.yards_gained.notna()]
    for direction in DIRECTIONS:
        side = runs[runs.run_location.eq(direction)]
        for prefix, team_column in (("pg_off", "posteam"), ("pg_def", "defteam")):
            events = pd.DataFrame({"season": side.season, "week": side.week, team_column: side[team_column],
                                   "n": side.yards_gained.astype(float), "d": 1.0})
            rates = team_rates(events, team_column, "n", "d", weeks, TEAM_PRIOR_PLAYS["run"])
            table = table.merge(rates.rename(columns={"rate": f"{prefix}_run_{direction}_ypc"}),
                                on=["season", "week", "team"], how="left")
    return table[["season", "week", "team", *COLUMNS]]


def attach_pregame_tables(frame: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Join the table to play rows: offense and league columns on posteam, defense columns on defteam."""
    keys = ["season", "week"]
    offense = table[[*keys, "team", *LEAGUE_COLUMNS, *OFFENSE_COLUMNS]].rename(columns={"team": "posteam"})
    defense = table[[*keys, "team", *DEFENSE_COLUMNS]].rename(columns={"team": "defteam"})
    frame = frame.drop(columns=[column for column in COLUMNS if column in frame])
    frame = frame.merge(offense, on=[*keys, "posteam"], how="left", validate="many_to_one")
    return frame.merge(defense, on=[*keys, "defteam"], how="left", validate="many_to_one")


PREGAME_TABLES_ENV = "NFLSIM_PREGAME_TABLES"
_LOOKUPS: dict = {}


def simulator_lookup() -> dict | None:
    """The table NFLSIM_PREGAME_TABLES names, as a lookup (read once per process), or None."""
    import os
    path = os.environ.get(PREGAME_TABLES_ENV)
    if not path:
        return None
    if path not in _LOOKUPS:
        _LOOKUPS[path] = lookup(pd.read_csv(path))
    return _LOOKUPS[path]


def game_state_values(table: dict, season: int, week: int, offense: str, defense: str) -> dict:
    """The offense's league and offense columns and the defense's defense columns for one game."""
    own = table.get((int(season), int(week), str(offense)))
    other = table.get((int(season), int(week), str(defense)))
    if own is None or other is None:
        raise ValueError(f"Pregame tables have no {season} week {week} row for {offense if own is None else defense}")
    return {**{column: own[column] for column in (*LEAGUE_COLUMNS, *OFFENSE_COLUMNS)},
            **{column: other[column] for column in DEFENSE_COLUMNS}}


def lookup(table: pd.DataFrame) -> dict:
    """(season, week, team) -> that row's columns, for the simulator."""
    return {(int(row["season"]), int(row["week"]), str(row["team"])): {column: row[column] for column in COLUMNS}
            for row in table.to_dict("records")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pbp-dir", type=Path, required=True)
    parser.add_argument("--seasons", default="2016-2026")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first, _, last = args.seasons.partition("-")
    frames = []
    for season in range(int(first), int(last or first) + 1):
        path = args.pbp_dir / f"pbp_{season}.parquet"
        if path.is_file():
            frame = pd.read_parquet(path)
            frames.append(frame[[column for column in PBP_COLUMNS if column in frame]])
    table = build_tables(pd.concat(frames, ignore_index=True))
    output_dir(args.output.parent)
    table.to_csv(args.output, index=False)
    print(f"{len(table)} team-weeks; means:\n{table[COLUMNS].mean().round(4).to_string()}")


if __name__ == "__main__":
    main()
