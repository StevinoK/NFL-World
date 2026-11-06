"""Availability-aware usage shares (carries and targets by depth).

Each player has a latent usage strength per event type. A team-game's carries (or targets in one
depth bucket) are multinomial over the players who were on the field, with probabilities
softmax(strength + depth-chart offset). When a starter is out, his backups' larger shares are
explained by the smaller denominator rather than by higher strengths, so the backups' estimates
are not inflated and the starter's role is intact when he returns.

Strengths are filtered game by game (diagonal Laplace update, Gaussian random walk between weeks
and seasons). New players start from a position-and-draft-slot prior. The pregame share for a
player is the softmax over the team's expected active players (status ACT, not listed Out), with
the player himself included, using only earlier games.

    python -m nflsim.features.usage tune      # fit hyperparameters on 2017-2022, score 2023-2025
    python -m nflsim.features.usage write --bundle <dir> --output <dir>
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from nflsim.paths import ROOT, output_dir, write_json

INPUTS = ROOT / "artifacts/inputs"
POSITIONS = ("QB", "RB", "WR", "TE")
SKILL = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
UNDRAFTED_PICK = 300.0
# Feature column -> event filter. Same definitions as the old shares, minus scrambles and kneels
# for carries (the rusher model never sees them).
EVENTS = {
    "rush_rate": "rush",
    "deep": "deep",
    "medium": "medium",
    "short": "short",
}
TUNE_SEASONS, TEST_SEASONS = (2017, 2022), (2023, 2025)


@dataclass
class Params:
    """Prior mean by position, draft slope by position, depth-chart offsets, variances."""
    mean: dict = field(default_factory=lambda: {"QB": -2.0, "RB": 0.0, "WR": -2.0, "TE": -2.0})
    draft: dict = field(default_factory=lambda: {"QB": 0.0, "RB": -0.3, "WR": -0.3, "TE": -0.3})
    depth: tuple = (0.0, -0.5, -1.0)  # offset for depth_team 2, 3, 4+ relative to 1
    prior_var: float = 1.0
    week_var: float = 0.01
    season_var: float = 0.2

    def vector(self):
        return np.array([*(self.mean[p] for p in POSITIONS), *(self.draft[p] for p in POSITIONS), *self.depth,
                         np.log(self.prior_var), np.log(self.week_var), np.log(self.season_var)])

    @classmethod
    def from_vector(cls, x):
        x = list(map(float, x))
        return cls(dict(zip(POSITIONS, x[0:4])), dict(zip(POSITIONS, x[4:8])), tuple(x[8:11]),
                   float(np.exp(x[11])), float(np.exp(x[12])), float(np.exp(x[13])))

    def to_json(self):
        return {"mean": self.mean, "draft": self.draft, "depth": list(self.depth), "prior_var": self.prior_var,
                "week_var": self.week_var, "season_var": self.season_var}

    @classmethod
    def from_json(cls, data):
        return cls(data["mean"], data["draft"], tuple(data["depth"]), data["prior_var"], data["week_var"], data["season_var"])


def load_inputs(inputs=INPUTS):
    columns = ["game_id", "game_date", "season_type", "posteam", "rush_attempt", "qb_scramble", "qb_kneel",
               "pass_attempt", "sack", "rusher_player_id", "receiver_player_id", "depth_bucket"]
    pbp = pd.read_parquet(inputs / "pbp.parquet", columns=columns)
    roster = pd.read_parquet(inputs / "roster.parquet", columns=["game_id", "team", "player_id", "position", "status", "report_status", "depth_team", "draft_number"])
    snaps = pd.read_parquet(inputs / "snap_counts.parquet", columns=["game_id", "team", "gsis_id", "offense_snaps", "game_type"])
    return pbp, roster, snaps


def event_counts(pbp):
    """(event, game_id, team, player_id) -> count."""
    plays = pbp[pbp.season_type.eq("REG")]
    rush = plays[plays.rush_attempt.eq(1) & plays.qb_scramble.ne(1) & plays.qb_kneel.ne(1) & plays.rusher_player_id.notna()]
    targets = plays[plays.pass_attempt.eq(1) & plays.sack.ne(1) & plays.receiver_player_id.notna()]
    frames = [rush.assign(event="rush", player_id=rush.rusher_player_id)]
    frames += [targets[targets.depth_bucket.eq(bucket)].assign(event=bucket, player_id=lambda f: f.receiver_player_id) for bucket in ("deep", "medium", "short")]
    events = pd.concat(frames)
    return events.groupby(["event", "game_id", "posteam", "player_id"]).size().rename("count").reset_index().rename(columns={"posteam": "team"})


def game_order(roster):
    games = roster[["game_id"]].drop_duplicates()
    parts = games.game_id.str.extract(r"^(\d{4})_(\d{2})_").astype(int)
    games["season"], games["week"] = parts[0], parts[1]
    return games.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def team_games(pbp, roster, snaps):
    """Per team-game: pregame candidates (all skill roster rows) and the players who actually played."""
    roster = roster[roster.position.isin(SKILL)].copy()
    roster["position"] = roster.position.map(SKILL)
    roster["expected"] = roster.status.eq("ACT") & ~roster.report_status.eq("Out")
    roster["depth"] = pd.to_numeric(roster.depth_team, errors="coerce").fillna(4).clip(1, 4).astype(int)
    roster["pick"] = pd.to_numeric(roster.draft_number, errors="coerce").fillna(UNDRAFTED_PICK)
    roster = roster.drop_duplicates(["game_id", "team", "player_id"])
    played = snaps[snaps.game_type.eq("REG") & snaps.offense_snaps.fillna(0).gt(0) & snaps.gsis_id.notna()]
    played = set(zip(played.game_id, played.team, played.gsis_id))
    counts = event_counts(pbp)
    return roster, played, counts


class Filter:
    """Sequential strength filter for one event type."""

    def __init__(self, params: Params):
        self.p = params
        self.mean, self.var, self.last = {}, {}, {}

    def state(self, player, position, pick, period):
        if player not in self.mean:
            self.mean[player] = self.p.mean[position] + self.p.draft[position]*(np.log(pick) - np.log(100.0))
            self.var[player] = self.p.prior_var
            self.last[player] = period
        season, week = period
        last_season, last_week = self.last[player]
        if season > last_season:
            self.var[player] += self.p.season_var + self.p.week_var*week
        else:
            self.var[player] += self.p.week_var*max(week - last_week, 0)
        self.last[player] = period
        return self.mean[player], self.var[player]

    def offsets(self, depth):
        return np.array([0.0, *self.p.depth])[np.asarray(depth) - 1]

    def update(self, players, means, variances, offsets, counts):
        total = counts.sum()
        if total <= 0 or len(players) < 2:
            return
        prior = means.copy()
        m = means.copy()
        for _ in range(2):
            eta = m + offsets
            pi = np.exp(eta - eta.max())
            pi /= pi.sum()
            gradient = (counts - total*pi) - (m - prior)/variances
            curvature = total*pi*(1 - pi) + 1/variances
            m = m + gradient/curvature
        eta = m + offsets
        pi = np.exp(eta - eta.max())
        pi /= pi.sum()
        posterior = 1/(1/variances + total*pi*(1 - pi))
        for player, value, variance in zip(players, m, posterior):
            self.mean[player], self.var[player] = float(value), float(variance)


def run(event, params, roster, played, counts, emit=True):
    """Walk every team-game in order: emit pregame shares, then update on what happened.

    Returns the emitted rows and the pregame log-likelihood of each event, per season.
    """
    flt = Filter(params)
    counts = counts[counts.event.eq(event)]
    by_game = {key: dict(zip(rows.player_id, rows["count"])) for key, rows in counts.groupby(["game_id", "team"])}
    rows_out, scores = [], []
    order = game_order(roster)
    grouped = dict(tuple(roster.groupby("game_id")))
    for game in order.itertuples():
        period = (game.season, game.week)
        for team, rows in grouped[game.game_id].groupby("team"):
            observed = by_game.get((game.game_id, team), {})
            ids = rows.player_id.tolist()
            info = dict(zip(ids, zip(rows.position, rows.pick, rows.depth)))
            for player in observed:
                if player not in info:
                    info[player] = ("WR" if event != "rush" else "RB", UNDRAFTED_PICK, 4)
            states = {player: flt.state(player, info[player][0], info[player][1], period) for player in info}
            expected = [player for player, flag in zip(ids, rows.expected) if flag]
            eta = {player: states[player][0] + flt.offsets([info[player][2]])[0] for player in info}
            if expected:
                pool = np.array([eta[player] for player in expected])
                base = np.exp(pool - pool.max()).sum()*np.exp(pool.max())
                if emit:
                    for player in ids:
                        own = np.exp(eta[player])
                        share = own/(base + (0 if player in expected else own))
                        rows_out.append((player, game.game_id, share))
                if game.season >= TUNE_SEASONS[0]:
                    log_base = np.log(base)
                    for player, n in observed.items():
                        if player in expected:
                            scores.append((game.season, n, n*(eta[player] - log_base)))
            active = [player for player in info if (game.game_id, team, player) in played or player in observed]
            if len(active) >= 2:
                flt.update(active, np.array([states[p][0] for p in active]), np.array([states[p][1] for p in active]),
                           flt.offsets([info[p][2] for p in active]), np.array([observed.get(p, 0) for p in active], float))
    emitted = pd.DataFrame(rows_out, columns=["player_id", "game_id", "share"])
    scored = pd.DataFrame(scores, columns=["season", "events", "log_lik"])
    return emitted, scored


def per_event(scored, seasons):
    frame = scored[scored.season.between(*seasons)]
    return float(frame.log_lik.sum()/frame.events.sum())


def baseline_score(event, column, players, roster, counts):
    """Pregame log-likelihood of the old share, renormalized over the expected active players."""
    roster = roster[roster.expected].merge(players[["player_id", "game_id", column]], on=["player_id", "game_id"], how="left")
    roster[column] = roster[column].fillna(roster[column].median()).clip(1e-4)
    roster["norm"] = roster[column]/roster.groupby(["game_id", "team"])[column].transform("sum")
    observed = counts[counts.event.eq(event)].merge(roster[["game_id", "team", "player_id", "norm"]], on=["game_id", "team", "player_id"])
    observed["season"] = observed.game_id.str[:4].astype(int)
    observed["log_lik"] = observed["count"]*np.log(observed.norm)
    return observed.rename(columns={"count": "events"})[["season", "events", "log_lik"]]


def tune(event, roster, played, counts, start=None, maxiter=300):
    def objective(x):
        _, scored = run(event, Params.from_vector(x), roster, played, counts, emit=False)
        return -per_event(scored, TUNE_SEASONS)
    x0 = (start or Params()).vector()
    result = minimize(objective, x0, method="Powell", options={"maxiter": maxiter, "xtol": 1e-2, "ftol": 1e-5})
    return Params.from_vector(result.x), -result.fun


def apply_usage(table: pd.DataFrame, params: dict, inputs: Path = INPUTS, events=tuple(EVENTS)) -> pd.DataFrame:
    """``player_stats_wide`` with the usage shares replaced by the filter's pregame values (others kept)."""
    pbp, roster, snaps = load_inputs(inputs)
    roster, played, counts = team_games(pbp, roster, snaps)
    table = table.copy()
    keys = pd.MultiIndex.from_frame(table[["player_id", "game_id"]].astype(str))
    for column in events:
        emitted, _ = run(EVENTS[column], Params.from_json(params[column]["params"]), roster, played, counts)
        values = emitted.drop_duplicates(["player_id", "game_id"]).set_index(["player_id", "game_id"]).share
        new = values.reindex(keys).to_numpy()
        print(f"{column}: {np.isfinite(new).mean():.3f} of rows covered", flush=True)
        table[column] = np.where(np.isfinite(new), new, table[column].to_numpy())
    return table


def manifest(params_path, source, events=tuple(EVENTS)) -> dict:
    return {"source": str(source), "params": str(params_path), "columns": list(events),
            "definition": __doc__.split("\n\n")[1].replace("\n", " ")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["tune", "write"])
    parser.add_argument("--events", nargs="+", default=list(EVENTS))
    parser.add_argument("--params", type=Path, default=ROOT / "outputs/usage/usage_params.json")
    parser.add_argument("--baseline", type=Path, default=ROOT.parent / "archive/nflsim_bundles/pv-isolated-mu-neutral-pilot-r2-2025/player_stats_wide.csv")
    parser.add_argument("--bundle", type=Path, help="write: directory whose player_stats_wide.csv gets the new shares")
    parser.add_argument("--output", type=Path, help="write: destination directory for the updated table")
    parser.add_argument("--inputs", type=Path, default=INPUTS, help="Directory with pbp, roster and snap_counts parquet files")
    args = parser.parse_args()
    if args.stage == "tune":
        pbp, roster, snaps = load_inputs(args.inputs)
        roster, played, counts = team_games(pbp, roster, snaps)
        output_dir(args.params.parent)
        saved = json.loads(args.params.read_text()) if args.params.exists() else {}
        old = pd.read_csv(args.baseline, dtype={"player_id": str, "game_id": str}, usecols=["player_id", "game_id", *EVENTS])
        report = []
        for column in args.events:
            event = EVENTS[column]
            start = Params.from_json(saved[column]["params"]) if column in saved else None
            params, tuned = tune(event, roster, played, counts, start)
            _, scored = run(event, params, roster, played, counts, emit=False)
            base = baseline_score(event, column, old, roster, counts)
            record = {"column": column, "tune_new": per_event(scored, TUNE_SEASONS), "tune_old": per_event(base, TUNE_SEASONS),
                      "test_new": per_event(scored, TEST_SEASONS), "test_old": per_event(base, TEST_SEASONS)}
            print(record, flush=True)
            report.append(record)
            saved[column] = {"params": params.to_json(), "scores": record}
            write_json(args.params, saved)
        pd.DataFrame(report).to_csv(args.params.with_name("usage_scores.csv"), index=False)
        return
    table = pd.read_csv(args.bundle / "player_stats_wide.csv", dtype={"player_id": str, "game_id": str})
    table = apply_usage(table, json.loads(args.params.read_text()), args.inputs, args.events)
    destination = output_dir(args.output)
    table.to_csv(destination / "player_stats_wide.csv", index=False)
    write_json(destination / "usage_manifest.json", manifest(args.params, args.bundle, args.events))


if __name__ == "__main__":
    main()
