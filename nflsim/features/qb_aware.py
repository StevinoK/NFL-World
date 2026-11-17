"""QB-aware team passing inputs: tilt the team-keyed passing columns toward tonight's starter's own games with the team.

For each team-game and raw passing stat s (play-by-play, the team's earlier regular-season games only):
  team_raw = decayed history of the team, every starter alike (what the team inputs follow)
  qb_raw   = the same history with games started by someone else weighted ``w``, plus an offset by how many earlier
             starts tonight's starter has with the team (0, 1, 2, 3-5, 6-10, 11+), fitted as the mean miss in each bucket
             relative to the average game (new starters land below their team's past)
  delta    = qb_raw - team_raw
Each input X built on stat s becomes X + beta * delta, beta the slope of X on team_raw (puts the shift in X's units).
Tonight's starter is the passer with the most dropbacks in played games and depth-chart QB1 in unplayed ones.
2025 holdout (2026-11-17): spread CRPS -0.34 +- 0.13 in quarterback-change games, other games unchanged.

    python -m nflsim.features.qb_aware fit   --bundle <dir with team_stats_wide.csv> [--inputs dir] [--params out.json]
    python -m nflsim.features.qb_aware write --bundle <dir> --output <dir> [--params qb_aware_params.json] [--inputs dir]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nflsim.features.builder import canonicalize_teams
from nflsim.paths import ROOT, output_dir, write_json

INPUTS = ROOT / "artifacts/inputs"
PARAMS_NAME = "qb_aware_params.json"
STATS = {"pass_epa": ("epa_sum", "dropbacks"), "pass_oe": ("pass_oe_sum", "plays")}
INPUT_STATS = {"epa_rate": "pass_epa", "passing_massey_offense": "pass_epa", "passing_massey_rating": "pass_epa",
               "pass_oe": "pass_oe"}
BUCKETS = [-1, 0, 1, 2, 5, 10, 10_000]  # earlier starts with the team: 0, 1, 2, 3-5, 6-10, 11+
DEFAULT = {"decay": 0.9, "k": 150.0, "w": 0.25, "lookback": 40, "first_season": 2016,
           "offset_seasons": [2017, 2021], "beta_seasons": [2017, 2025]}


def team_games(pbp: pd.DataFrame, roster: pd.DataFrame | None, first_season: int) -> pd.DataFrame:
    """One row per (game_id, team): raw passing sums and the starter, for played and (via the depth chart) unplayed games."""
    pbp = canonicalize_teams(pbp)
    pbp = pbp[pbp.season_type.eq("REG")] if "season_type" in pbp else pbp
    pbp = pbp[pbp.posteam.notna() & pbp.game_id.astype(str).str[:4].astype(int).ge(first_season)]
    drop = pbp[pbp.qb_dropback.eq(1) & pbp.epa.notna() & pbp.passer_player_id.notna()]
    plays = pbp[pbp.pass_oe.notna() & (pbp.pass_attempt.eq(1) | pbp.rush_attempt.eq(1))]
    keys = ["game_id", "posteam"]
    games = pd.DataFrame({
        "epa_sum": drop.groupby(keys).epa.sum(), "dropbacks": drop.groupby(keys).size(),
        "starter": drop.groupby(keys).passer_player_id.agg(lambda s: s.value_counts().index[0]),
        "pass_oe_sum": plays.groupby(keys).pass_oe.sum(), "plays": plays.groupby(keys).size(),
    }).reset_index().rename(columns={"posteam": "team"}).dropna(subset=["starter"])
    if roster is not None and {"game_id", "team", "player_id", "position", "depth_team"} <= set(roster):
        qbs = canonicalize_teams(roster)
        qbs = qbs[qbs.position.eq("QB") & qbs.depth_team.notna()].sort_values(["game_id", "team", "depth_team"])
        future = qbs.drop_duplicates(["game_id", "team"])[["game_id", "team", "player_id"]].rename(columns={"player_id": "starter"})
        future = future[~future.set_index(["game_id", "team"]).index.isin(games.set_index(["game_id", "team"]).index)]
        future = future[future.game_id.astype(str).str[:4].astype(int).ge(first_season)]
        games = pd.concat([games, future], ignore_index=True)  # unplayed: sums missing, history only
    games["game_id"] = games.game_id.astype(str)
    games["season"] = games.game_id.str[:4].astype(int)
    games["week"] = games.game_id.str[5:7].astype(int)
    return games.sort_values(["team", "season", "week", "game_id"]).reset_index(drop=True)


def histories(games: pd.DataFrame, params: dict) -> pd.DataFrame:
    decay, k, w, lookback = params["decay"], params["k"], params["w"], int(params["lookback"])
    played = games.dropbacks.notna()
    league = {s: games.loc[played, n].sum() / games.loc[played, d].sum() for s, (n, d) in STATS.items()}
    rows = []
    for team, g in games.groupby("team"):
        g = g.reset_index(drop=True)
        done = g.dropbacks.notna().to_numpy()
        for i in range(len(g)):
            past = g.iloc[:i][done[:i]].iloc[-lookback:]
            weight = decay ** (np.arange(len(past))[::-1] + 1)
            same = (past.starter == g.starter[i]).to_numpy()
            row = {"game_id": g.game_id[i], "team": team, "season": g.season[i], "starts": int(same.sum())}
            for s, (n, d) in STATS.items():
                num, den = past[n].to_numpy(float), past[d].to_numpy(float)
                row[f"{s}_team"] = (np.sum(weight * num) + k * league[s]) / (np.sum(weight * den) + k)
                qw = weight * np.where(same, 1.0, w)
                row[f"{s}_qb"] = (np.sum(qw * num) + k * league[s]) / (np.sum(qw * den) + k)
                row[f"{s}_actual"] = g[n][i] / g[d][i] if done[i] and g[d][i] > 0 else np.nan
                row[f"{s}_weight"] = g[d][i] if done[i] else np.nan
            rows.append(row)
    hist = pd.DataFrame(rows)
    hist["bucket"] = pd.cut(hist.starts, BUCKETS, labels=False)
    return hist


def deltas(hist: pd.DataFrame, offsets: dict) -> pd.DataFrame:
    out = hist[["game_id", "team"]].copy()
    for s in STATS:
        shift = hist.bucket.map({int(b): v for b, v in offsets[s].items()}).fillna(0.0)
        out[f"{s}_team"] = hist[f"{s}_team"]
        out[f"{s}_delta"] = hist[f"{s}_qb"] + shift - hist[f"{s}_team"]
    return out


def fit(table: pd.DataFrame, pbp: pd.DataFrame, roster: pd.DataFrame | None = None, params: dict | None = None) -> dict:
    """Bucket offsets (offset_seasons) and per-input betas (beta_seasons) for ``apply``."""
    params = {**DEFAULT, **(params or {})}
    hist = histories(team_games(pbp, roster, params["first_season"]), params)
    fit_rows = hist[hist.season.between(*params["offset_seasons"])]
    offsets = {}
    for s in STATS:
        ok = fit_rows[f"{s}_actual"].notna()
        weight = fit_rows.loc[ok, f"{s}_weight"]
        miss = (fit_rows[f"{s}_actual"] - fit_rows[f"{s}_qb"])[ok]
        bucket = fit_rows.loc[ok, "bucket"]
        by_bucket = (miss * weight).groupby(bucket).sum() / weight.groupby(bucket).sum()
        offsets[s] = {str(int(b)): float(v) for b, v in (by_bucket - np.average(miss, weights=weight)).items()}
    merged = table.assign(game_id=table.game_id.astype(str)).merge(deltas(hist, offsets), on=["game_id", "team"], how="left")
    seasons = merged.game_id.str[:4].astype(int)
    betas = {}
    for column, s in INPUT_STATS.items():
        if column not in merged:
            continue
        ok = merged[column].notna() & merged[f"{s}_team"].notna() & seasons.between(*params["beta_seasons"])
        x, y = merged.loc[ok, f"{s}_team"], merged.loc[ok, column]
        betas[column] = float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))
    return {**params, "offsets": offsets, "betas": betas}


def apply_qb_aware(table: pd.DataFrame, params: dict, inputs: Path = INPUTS) -> pd.DataFrame:
    """``team_stats_wide`` with the team passing columns shifted toward each game's starter (rows keep their order)."""
    pbp = pd.read_parquet(inputs / "pbp.parquet")
    roster = pd.read_parquet(inputs / "roster.parquet") if (inputs / "roster.parquet").is_file() else None
    hist = histories(team_games(pbp, roster, params["first_season"]), params)
    shifts = deltas(hist, params["offsets"])
    keys = table[["game_id", "team"]].astype({"game_id": str}).merge(shifts, on=["game_id", "team"], how="left")
    out = table.copy()
    for column, beta in params["betas"].items():
        if column in out:
            out[column] = out[column] + (beta * keys[f"{INPUT_STATS[column]}_delta"]).fillna(0.0).to_numpy()
    return out


def manifest(params: dict, source) -> dict:
    return {"source": str(source), "params": params, "columns": sorted(params["betas"]),
            "definition": __doc__.split("\n\n")[0].replace("\n", " ")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["fit", "write"])
    parser.add_argument("--bundle", type=Path, required=True, help="directory with the unshifted team_stats_wide.csv")
    parser.add_argument("--inputs", type=Path, default=INPUTS, help="directory with pbp.parquet (and roster.parquet)")
    parser.add_argument("--params", type=Path, help=f"fit: output (default <bundle>/{PARAMS_NAME}); write: input")
    parser.add_argument("--output", type=Path, help="write: destination directory for the shifted table")
    args = parser.parse_args()
    table = pd.read_csv(args.bundle / "team_stats_wide.csv", dtype={"game_id": str, "team": str}, low_memory=False)
    if args.stage == "fit":
        roster = pd.read_parquet(args.inputs / "roster.parquet") if (args.inputs / "roster.parquet").is_file() else None
        params = fit(table, pd.read_parquet(args.inputs / "pbp.parquet"), roster)
        write_json(args.params or args.bundle / PARAMS_NAME, params)
        print(json.dumps({k: params[k] for k in ("offsets", "betas")}, indent=1))
        return
    params = json.loads((args.params or args.bundle / PARAMS_NAME).read_text())
    destination = output_dir(args.output)
    apply_qb_aware(table, params, args.inputs).to_csv(destination / "team_stats_wide.csv", index=False)
    write_json(destination / "qb_aware_manifest.json", manifest(params, args.bundle))
    print(destination / "team_stats_wide.csv")


if __name__ == "__main__":
    main()
