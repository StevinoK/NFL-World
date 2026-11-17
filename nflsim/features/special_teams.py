"""Pregame special-teams team features: punt distance, punt and kickoff returns and coverage.

Each event is measured as a deviation from that week's league average, so kickoff rule changes
(2024 dynamic kickoff, 2025 touchback to the 35) and league trends drop out; the rule eras enter the
models as separate flags. Punt distance is also adjusted for field position with the league's
yardline-bucket profile from earlier seasons (a punt from midfield is short for a reason).

A team's feature for a game is its recency-weighted, shrunk mean deviation over earlier games only:

    sum(w_i * dev_i) / (sum(w_i) + k),   w_i = decay ** weeks_back * carry ** seasons_back

with (decay, carry, k) tuned per feature to predict the next game's mean deviation.

    st_punt_gross            kicking team: gross punt distance
    st_punt_return_allowed   kicking team: punt return yards allowed (punts not touchbacks)
    st_punt_return          receiving team: punt return yards gained
    st_ko_start_allowed      kicking team: receiving team's start after its kickoffs
    st_ko_start             receiving team: own start after opponents' kickoffs (return TD = 100)
    st_ko_rule_2024, st_ko_rule_2025   kickoff rule era flags

    python -m nflsim.features.special_teams tune
    python -m nflsim.features.special_teams write --bundle <dir> --output <dir>
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nflsim.features.builder import canonicalize_teams
from nflsim.paths import ROOT, output_dir, write_json

INPUTS = ROOT / "artifacts/inputs"
FEATURES = {
    "st_punt_gross": ("punt", "gross", "kicking"),
    "st_punt_return_allowed": ("punt", "returned", "kicking"),
    "st_punt_return": ("punt", "returned", "receiving"),
    "st_ko_start_allowed": ("kickoff", "start", "kicking"),
    "st_ko_start": ("kickoff", "start", "receiving"),
}
RULE_COLUMNS = {"st_ko_rule_2024": 2024, "st_ko_rule_2025": 2025}
COLUMNS = [*FEATURES, *RULE_COLUMNS]
GRID = {"decay": (0.9, 0.95, 0.98, 1.0), "carry": (0.25, 0.5, 1.0), "k": (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)}
TUNE_SEASONS = (2018, 2024)
DEFAULT = {"decay": 0.95, "carry": 0.5, "k": 20.0}


def _numeric(frame, column):
    return pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(np.nan, index=frame.index)


def load_pbp(inputs: Path = INPUTS) -> pd.DataFrame:
    pbp = canonicalize_teams(pd.read_parquet(inputs / "pbp.parquet"))
    pbp = pbp[pbp.season_type.eq("REG")] if "season_type" in pbp else pbp
    parsed = pbp.game_id.astype(str).str.extract(r"^(\d{4})_(\d{2})_").astype(int)
    pbp = pbp.assign(season=parsed[0].to_numpy(), week=parsed[1].to_numpy())
    return pbp.sort_values(["game_id", "play_id"]).reset_index(drop=True)


def kickoff_starts(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per kickoff: receiving team's own-yardline start (100 = return TD). Onside recoveries and re-kicks dropped."""
    snap = pbp.posteam.notna() & _numeric(pbp, "yardline_100").notna()
    kick = _numeric(pbp, "kickoff_attempt").eq(1)
    rows = pbp.loc[snap, ["game_id", "posteam", "yardline_100"]].assign(is_kick=kick[snap].to_numpy())
    # The next row with a possession after each kickoff tells where the receiving team started.
    following = rows.groupby("game_id")[["posteam", "yardline_100", "is_kick"]].shift(-1)
    kicks = pbp.loc[kick & snap].copy()
    after = following.loc[kicks.index]
    touchdown = _numeric(kicks, "touchdown").eq(1) & kicks.td_team.eq(kicks.posteam)
    recovered = _numeric(kicks, "own_kickoff_recovery").eq(1) | (after.posteam.notna() & after.posteam.ne(kicks.posteam))
    start = 100 - pd.to_numeric(after.yardline_100, errors="coerce")
    kicks = kicks.assign(start=np.where(touchdown, 100.0, start), kicking=kicks.defteam, receiving=kicks.posteam)
    keep = (touchdown | (~recovered & after.is_kick.eq(False) & kicks.start.between(1, 99)))
    return kicks.loc[keep, ["game_id", "season", "week", "kicking", "receiving", "start"]]


def punts(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per unblocked punt: gross distance, whether it was a touchback, return yards (0 when not returned)."""
    mask = _numeric(pbp, "punt_attempt").eq(1) & ~_numeric(pbp, "punt_blocked").eq(1) & _numeric(pbp, "kick_distance").notna()
    rows = pbp.loc[mask & pbp.posteam.notna() & pbp.defteam.notna()]
    return pd.DataFrame({
        "game_id": rows.game_id, "season": rows.season, "week": rows.week,
        "kicking": rows.posteam, "receiving": rows.defteam,
        "yardline_100": _numeric(rows, "yardline_100"), "gross": _numeric(rows, "kick_distance"),
        "touchback": _numeric(rows, "touchback").eq(1), "return_yards": _numeric(rows, "return_yards").fillna(0.0),
    })


def deviations(pbp: pd.DataFrame) -> pd.DataFrame:
    """Long table of (feature, team, game_id, season, week, dev): each event minus its league-week mean."""
    kicks, punt_rows = kickoff_starts(pbp), punts(pbp)
    bucket = (punt_rows.yardline_100.clip(1, 99) // 10).astype(int)
    profile = pd.Series(0.0, index=punt_rows.index)
    for season in sorted(punt_rows.season.unique()):
        current = punt_rows.season.eq(season)
        history = punt_rows.season.lt(season) if punt_rows.season.lt(season).any() else current
        shape = punt_rows.gross[history].groupby(bucket[history]).mean() - punt_rows.gross[history].mean()
        profile[current] = bucket[current].map(shape).fillna(0.0).to_numpy()
    punt_rows = punt_rows.assign(gross_adj=punt_rows.gross - profile)
    returned = punt_rows[~punt_rows.touchback]
    sources = {
        "gross": punt_rows.rename(columns={"gross_adj": "value"}),
        "returned": returned.rename(columns={"return_yards": "value"}),
        "start": kicks.rename(columns={"start": "value"}),
    }
    frames = []
    for name, (_, source, owner) in FEATURES.items():
        events = sources[source]
        dev = events.value - events.groupby(["season", "week"]).value.transform("mean")
        frames.append(pd.DataFrame({"feature": name, "team": events[owner].to_numpy(), "game_id": events.game_id.to_numpy(),
                                    "season": events.season.to_numpy(), "week": events.week.to_numpy(), "dev": dev.to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def pregame(events: pd.DataFrame, targets: pd.DataFrame, decay: float, carry: float, k: float) -> np.ndarray:
    """Shrunk recency-weighted mean deviation for each (team, season, week) target from strictly earlier weeks."""
    result = np.zeros(len(targets))
    by_team = {team: group for team, group in events.groupby("team")}
    positions = pd.Series(np.arange(len(targets)), index=targets.index)
    for team, rows in targets.groupby("team"):
        history = by_team.get(team)
        if history is None:
            continue
        season, week, dev = (history[column].to_numpy(dtype=float) for column in ("season", "week", "dev"))
        target_season, target_week = rows.season.to_numpy(float)[:, None], rows.week.to_numpy(float)[:, None]
        gap = target_season * 20 + target_week - (season * 20 + week)[None, :]
        weights = np.where(gap > 0, decay ** np.maximum(gap, 0) * carry ** (target_season - season[None, :]), 0.0)
        result[positions[rows.index].to_numpy()] = weights @ dev / (weights.sum(axis=1) + k)
    return result


def score(events: pd.DataFrame, feature: str, params: dict, seasons=TUNE_SEASONS) -> dict:
    """Event-weighted MSE of predicting each team-game's mean deviation, against predicting zero."""
    rows = events[events.feature.eq(feature)]
    games = rows.groupby(["team", "game_id", "season", "week"]).dev.agg(["mean", "size"]).reset_index()
    games = games[games.season.between(*seasons)].reset_index(drop=True)
    predicted = pregame(rows, games, **params)
    weights = games["size"].to_numpy(float)
    mse = float(np.average((games["mean"] - predicted) ** 2, weights=weights))
    zero = float(np.average(games["mean"] ** 2, weights=weights))
    return {"mse": mse, "zero_mse": zero, "skill": 1 - mse / zero, "games": int(len(games))}


def tune(events: pd.DataFrame) -> dict:
    chosen = {}
    for feature in FEATURES:
        results = []
        for decay, carry, k in itertools.product(GRID["decay"], GRID["carry"], GRID["k"]):
            params = {"decay": decay, "carry": carry, "k": k}
            results.append({**params, **score(events, feature, params)})
        best = min(results, key=lambda record: record["mse"])
        chosen[feature] = {"params": {key: best[key] for key in ("decay", "carry", "k")},
                           "score": {key: best[key] for key in ("mse", "zero_mse", "skill", "games")}}
        print(feature, chosen[feature], flush=True)
    return chosen


def features_for(table: pd.DataFrame, events: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Special-teams columns for every (team, game_id) row of a team feature table."""
    keys = canonicalize_teams(table[["team", "game_id"]].copy())
    parsed = keys.game_id.astype(str).str.extract(r"^(\d{4})_(\d{2})_").astype(int)
    targets = keys.assign(season=parsed[0].to_numpy(), week=parsed[1].to_numpy()).reset_index(drop=True)
    out = pd.DataFrame(index=table.index)
    for feature in FEATURES:
        rows = events[events.feature.eq(feature)]
        out[feature] = pregame(rows, targets, **params.get(feature, {"params": DEFAULT})["params"])
    for column, season in RULE_COLUMNS.items():
        out[column] = (targets.season.to_numpy() >= season).astype(float)
    return out


def apply_special_teams(table: pd.DataFrame, params: dict, inputs: Path = INPUTS) -> pd.DataFrame:
    """``team_stats_wide`` with the special-teams columns (re)computed from the plays in ``inputs``."""
    values = features_for(table, deviations(load_pbp(inputs)), params)
    return table.drop(columns=[column for column in COLUMNS if column in table]).join(values)


def manifest(params: dict, source) -> dict:
    return {"source": str(source), "params": params, "columns": COLUMNS,
            "definition": __doc__.split("\n\n")[1].replace("\n", " ")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["tune", "write"])
    parser.add_argument("--params", type=Path, default=ROOT / "outputs/special_teams/special_teams_params.json")
    parser.add_argument("--bundle", type=Path, help="write: directory whose team_stats_wide.csv gets the special-teams columns")
    parser.add_argument("--output", type=Path, help="write: destination directory for the updated table")
    parser.add_argument("--inputs", type=Path, default=INPUTS, help="Directory with pbp.parquet")
    args = parser.parse_args()
    if args.stage == "tune":
        output_dir(args.params.parent)
        write_json(args.params, tune(deviations(load_pbp(args.inputs))))
        return
    params = json.loads(args.params.read_text())
    table = pd.read_csv(args.bundle / "team_stats_wide.csv", dtype={"game_id": str})
    table = apply_special_teams(table, params, args.inputs)
    destination = output_dir(args.output)
    table.to_csv(destination / "team_stats_wide.csv", index=False)
    write_json(destination / "special_teams_manifest.json", manifest(params, args.bundle))
    print(destination / "team_stats_wide.csv")


if __name__ == "__main__":
    main()
