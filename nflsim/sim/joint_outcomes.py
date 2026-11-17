"""Joint-outcome benchmark for backtest runs.

Scores each team-game's simulated distribution of team and player stats against what happened:
marginal calibration (randomized PIT, CRPS) and dependence (within-game rank correlation of the
sims vs the observed correlation of normal scores across games, and the log score of 2x2 joint
events "A over its sim median and B over its sim median" against the same marginals treated as
independent). Slot players who took no snap-level part in the real game are left out.

    python -m nflsim.sim.joint_outcomes --run old=<run dir>[,<run dir>...] --run new=<...>
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from nflsim.features.bayesian_history import load_pbp_data
from nflsim.paths import ROOT, output_dir, write_json

SLOT_STATS = {
    "QB1": ["passing_yards", "pass_attempts", "completions", "passing_touchdowns", "interceptions", "sacks", "rushing_yards"],
    "RB1": ["rushing_yards", "rushing_attempts", "rushing_touchdowns", "receiving_yards", "receptions"],
    "WR1": ["receiving_yards", "receptions", "receiving_touchdowns"],
    "WR2": ["receiving_yards", "receptions"],
    "TE1": ["receiving_yards", "receptions"],
}
TEAM_STATS = ["points", "win", "rushing_attempts", "rushing_yards"]
PAIRS = [
    ("QB1.passing_yards", "WR1.receiving_yards"),
    ("QB1.passing_yards", "WR2.receiving_yards"),
    ("QB1.passing_yards", "TE1.receiving_yards"),
    ("WR1.receiving_yards", "WR2.receiving_yards"),
    ("WR1.receiving_yards", "TE1.receiving_yards"),
    ("QB1.passing_touchdowns", "WR1.receiving_touchdowns"),
    ("QB1.passing_yards", "RB1.rushing_yards"),
    ("QB1.pass_attempts", "team.rushing_attempts"),
    ("QB1.sacks", "QB1.passing_yards"),
    ("QB1.interceptions", "team.win"),
    ("QB1.passing_yards", "team.points"),
    ("RB1.rushing_yards", "team.points"),
    ("RB1.rushing_touchdowns", "team.points"),
    ("QB1.passing_yards", "team.win"),
    ("QB1.pass_attempts", "team.win"),
    ("team.rushing_attempts", "team.win"),
    ("RB1.rushing_yards", "team.win"),
    ("team.points", "opp.team.points"),
    ("QB1.passing_yards", "opp.QB1.passing_yards"),
    ("QB1.pass_attempts", "opp.team.points"),
    ("team.rushing_attempts", "opp.team.points"),
]
BOOTSTRAP = 500


def with_opponent(frame, keys):
    """Add opp.* copies of every outcome from the other team in the same game (and simulation)."""
    columns = [column for column in frame if column not in [*keys, "team"]]
    other = frame.rename(columns={"team": "opponent", **{column: f"opp.{column}" for column in columns}})
    return frame.merge(other, on=keys).query("team != opponent").drop(columns="opponent").reset_index(drop=True)


def with_win(frame, keys):
    points = with_opponent(frame[[*keys, "team", "team.points"]], keys)
    win = (points["team.points"] > points["opp.team.points"]) + 0.5*(points["team.points"] == points["opp.team.points"])
    return frame.merge(points[[*keys, "team"]].assign(**{"team.win": win.astype(float)}), on=[*keys, "team"])


def simulated(run):
    """Per (game, team, simulation) outcomes and the player id in each slot."""
    frames, slots = [], []
    for path in sorted((run / "raw_games").glob("*_players.csv")):
        players = pd.read_csv(path, usecols=lambda column: column in {"game_id", "team", "simulation", "position", "player_id", *{s for stats in SLOT_STATS.values() for s in stats}})
        scores = pd.read_csv(path.with_name(path.name.replace("_players", "")))
        points = pd.concat([scores[["game_id", "simulation", "home_team", "home"]].set_axis(["game_id", "simulation", "team", "team.points"], axis=1),
                            scores[["game_id", "simulation", "away_team", "away"]].set_axis(["game_id", "simulation", "team", "team.points"], axis=1)])
        offense = players[players.position.str.match(r"^(QB|RB|WR|TE)\d$")]
        totals = offense.groupby(["game_id", "team", "simulation"])[["rushing_attempts", "rushing_yards"]].sum().add_prefix("team.").reset_index()
        tracked = offense[offense.position.isin(list(SLOT_STATS))]
        wide = tracked.pivot_table(index=["game_id", "team", "simulation"], columns="position", values=sorted({s for stats in SLOT_STATS.values() for s in stats}), aggfunc="sum")
        wide.columns = [f"{slot}.{stat}" for stat, slot in wide.columns]
        frame = points.merge(totals, on=["game_id", "team", "simulation"]).merge(wide.reset_index(), on=["game_id", "team", "simulation"])
        frames.append(frame)
        slots.append(tracked.groupby(["game_id", "team", "position"]).player_id.agg(lambda ids: ids.mode().iat[0]).reset_index())
    sims = with_win(pd.concat(frames, ignore_index=True), ["game_id", "simulation"])
    return with_opponent(sims, ["game_id", "simulation"]), pd.concat(slots, ignore_index=True)


def actual_player_stats(pbp):
    plays = pbp[pbp.season_type.eq("REG") & pbp.two_point_attempt.ne(1)]
    dropback, sack = plays.pass_attempt.eq(1), plays.sack.eq(1)
    plays = plays.assign(attempt=(dropback & ~sack).astype(float), rush=plays.rush_attempt.eq(1).astype(float))

    def by(id_column, mask, values):
        rows = plays[mask & plays[id_column].notna()]
        return rows.groupby(["game_id", id_column]).agg(**{name: (column, "sum") for name, column in values.items()}).rename_axis(["game_id", "player_id"])
    passing = by("passer_player_id", dropback, {"passing_yards": "passing_yards", "pass_attempts": "attempt", "completions": "complete_pass", "passing_touchdowns": "pass_touchdown", "interceptions": "interception", "sacks": "sack"})
    rushing = by("rusher_player_id", plays.rush_attempt.eq(1), {"rushing_yards": "rushing_yards", "rushing_attempts": "rush", "rushing_touchdowns": "rush_touchdown"})
    receiving = by("receiver_player_id", dropback & ~sack, {"receiving_yards": "receiving_yards", "receptions": "complete_pass", "receiving_touchdowns": "pass_touchdown"})
    stats = pd.concat([passing, rushing, receiving], axis=1).fillna(0.0).reset_index()
    team = plays[plays.rush_attempt.eq(1)].groupby(["game_id", "posteam"]).agg(**{"team.rushing_attempts": ("rush", "sum"), "team.rushing_yards": ("rushing_yards", "sum")}).rename_axis(["game_id", "team"]).reset_index()
    return stats, team


def observed(run, slots, pbp):
    stats, team = actual_player_stats(pbp)
    results = pd.read_csv(run / "backtest_2025_results.csv")
    rows = pd.concat([results.assign(team=results.home_team, **{"team.points": results.actual_home_score}),
                      results.assign(team=results.away_team, **{"team.points": results.actual_away_score})])
    frame = with_win(rows[["game_id", "team", "team.points"]], ["game_id"]).merge(team, on=["game_id", "team"], how="left")
    for slot, names in SLOT_STATS.items():
        # A slot player with no pass, rush or target in the real game has no stats row and drops out.
        players = slots[slots.position.eq(slot)].merge(stats, on=["game_id", "player_id"], how="left")
        frame = frame.merge(players[["game_id", "team", *names]].rename(columns={name: f"{slot}.{name}" for name in names}), on=["game_id", "team"], how="left")
    return with_opponent(frame, ["game_id"])


def crps(samples, value):
    samples = np.sort(samples)
    n = len(samples)
    return float(np.mean(np.abs(samples-value)) - np.sum((2*np.arange(1, n+1)-n-1)*samples)/n**2)


def score(sims, actual, seed=0):
    """One row per team-game and outcome: PIT, normal score, CRPS, sim median, and the sims."""
    rng = np.random.default_rng(seed)
    groups = dict(tuple(sims.groupby(["game_id", "team"])))
    outcomes = [column for column in sims if "." in column]
    rows = []
    for record in actual.to_dict("records"):
        draws = groups.get((record["game_id"], record["team"]))
        if draws is None:
            continue
        for outcome in outcomes:
            value = record.get(outcome)
            if value is None or pd.isna(value):
                continue
            x = draws[outcome].to_numpy(float)
            pit = float(np.mean(x < value) + rng.uniform()*np.mean(x == value))
            rows.append({"game_id": record["game_id"], "team": record["team"], "outcome": outcome, "actual": float(value), "sim_mean": float(x.mean()), "sim_sd": float(x.std()), "sim_median": float(np.median(x)), "pit": pit, "z": float(norm.ppf(np.clip(pit, 0.5/len(x), 1-0.5/len(x)))), "crps": crps(x, value)})
    return pd.DataFrame(rows)


def marginal(scores):
    scores = scores[~scores.outcome.str.startswith("opp.")]
    summary = scores.groupby("outcome").agg(team_games=("pit", "size"), actual_mean=("actual", "mean"), sim_mean=("sim_mean", "mean"), crps=("crps", "mean"), pit_mean=("pit", "mean"), pit_var=("pit", "var"), coverage_80=("pit", lambda pit: float(pit.between(0.1, 0.9).mean())))
    summary["bias"] = summary.sim_mean-summary.actual_mean
    # 12 var(PIT) is 1 when calibrated; > 1 means the sims are too narrow, < 1 too wide.
    summary["dispersion"] = 12*summary.pit_var
    summary["dispersion_se"] = 12*scores.groupby("outcome").pit.apply(lambda pit: np.sqrt(((pit-pit.mean())**2).var()/len(pit)))
    return summary.drop(columns="pit_var").reset_index()


def pair_scores(sims, scores, pairs):
    groups = dict(tuple(sims.groupby(["game_id", "team"])))
    wide = {name: frame.set_index(["game_id", "team"]) for name, frame in scores.groupby("outcome")}
    rows = []
    for a, b in pairs:
        if a not in wide or b not in wide:
            continue
        both = wide[a][["z", "actual", "sim_median"]].join(wide[b][["z", "actual", "sim_median"]], lsuffix="_a", rsuffix="_b", how="inner")
        for (game_id, team), record in both.iterrows():
            draws = groups[(game_id, team)]
            x, y = draws[a].to_numpy(float), draws[b].to_numpy(float)
            over_x, over_y = x > record.sim_median_a, y > record.sim_median_b
            cells = np.array([[np.sum(~over_x & ~over_y), np.sum(~over_x & over_y)], [np.sum(over_x & ~over_y), np.sum(over_x & over_y)]], float) + 0.25
            joint = cells/cells.sum()
            independent = np.outer(joint.sum(1), joint.sum(0))
            hit = int(record.actual_a > record.sim_median_a), int(record.actual_b > record.sim_median_b)
            rank_corr = pd.Series(x).rank().corr(pd.Series(y).rank()) if x.std() > 0 and y.std() > 0 else np.nan
            rows.append({"pair": f"{a} x {b}", "game_id": game_id, "team": team, "z_a": record.z_a, "z_b": record.z_b, "sim_rank_corr": rank_corr, "p_both_over": joint[1, 1], "both_over": float(hit == (1, 1)), "log_score_joint": -np.log(joint[hit]), "log_score_independent": -np.log(independent[hit])})
    return pd.DataFrame(rows)


def dependence(pairs, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for pair, frame in pairs.groupby("pair", sort=False):
        games = frame.game_id.unique()
        by_game = {game: rows for game, rows in frame.groupby("game_id")}
        gain = frame.log_score_independent-frame.log_score_joint
        boot_corr, boot_gain = [], []
        for _ in range(BOOTSTRAP):
            sample = pd.concat([by_game[game] for game in rng.choice(games, len(games))])
            boot_corr.append(sample.z_a.corr(sample.z_b))
            boot_gain.append((sample.log_score_independent-sample.log_score_joint).mean())
        observed_corr = frame.z_a.corr(frame.z_b)
        out.append({"pair": pair, "team_games": len(frame), "sim_rank_corr": frame.sim_rank_corr.mean(), "observed_corr": observed_corr, "observed_corr_se": float(np.nanstd(boot_corr)), "corr_gap": frame.sim_rank_corr.mean()-observed_corr, "p_both_over": frame.p_both_over.mean(), "both_over_rate": frame.both_over.mean(), "dependence_gain": gain.mean(), "dependence_gain_se": float(np.std(boot_gain))})
    return pd.DataFrame(out)


def evaluate(runs, pbp):
    sims, actual = [], []
    for run in runs:
        run_sims, slots = simulated(run)
        sims.append(run_sims)
        actual.append(observed(run, slots, pbp))
    sims, actual = pd.concat(sims, ignore_index=True), pd.concat(actual, ignore_index=True)
    scores = score(sims, actual)
    pairs = pair_scores(sims, scores, PAIRS)
    return scores, marginal(scores), pairs, dependence(pairs)


def priorities(margins, dependences, label):
    """The largest calibration gaps, in standard errors, for the run labelled `label`."""
    rows = [{"kind": "dispersion", "item": row.outcome, "value": row.dispersion, "target": 1.0, "z": (row.dispersion-1)/row.dispersion_se} for row in margins.itertuples() if row.dispersion_se > 0]
    rows += [{"kind": "dependence", "item": row.pair, "value": row.sim_rank_corr, "target": row.observed_corr, "z": row.corr_gap/row.observed_corr_se} for row in dependences.itertuples() if row.observed_corr_se > 0]
    return pd.DataFrame(rows).assign(label=label).sort_values("z", key=np.abs, ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, help="label=run_dir[,run_dir...]")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--output", type=Path, help="default: benchmarks/<season>/joint_outcomes")
    parser.add_argument("--history", type=Path, help="default: benchmarks/<season>/joint_outcomes_history.csv")
    args = parser.parse_args()
    benchmarks = ROOT / "benchmarks" / str(args.season)
    args.history = args.history or benchmarks / "joint_outcomes_history.csv"
    output = output_dir(args.output or benchmarks / "joint_outcomes")
    pbp = load_pbp_data([args.season])
    margins, dependences, ranked, history = [], [], [], []
    for spec in args.run:
        label, paths = spec.split("=", 1)
        runs = [Path(path) for path in paths.split(",")]
        scores, margin, pairs, depend = evaluate(runs, pbp)
        scores.to_csv(output / f"{label}_scores.csv", index=False)
        pairs.to_csv(output / f"{label}_pairs.csv", index=False)
        margins.append(margin.assign(label=label))
        dependences.append(depend.assign(label=label))
        ranked.append(priorities(margin, depend, label))
        history.append({"date": date.today().isoformat(), "label": label, "runs": ";".join(run.name for run in runs), "games": scores.game_id.nunique(), "mean_crps_points": margin.set_index("outcome").crps.get("team.points"), "mean_abs_log_dispersion": float(np.abs(np.log(margin.dispersion)).mean()), "mean_abs_corr_gap": float(depend.corr_gap.abs().mean()), "mean_dependence_gain": float(depend.dependence_gain.mean())})
    margins, dependences, ranked = pd.concat(margins), pd.concat(dependences), pd.concat(ranked)
    margins.to_csv(output / "marginal.csv", index=False)
    dependences.to_csv(output / "dependence.csv", index=False)
    ranked.to_csv(output / "priorities.csv", index=False)
    history = pd.DataFrame(history)
    history.to_csv(args.history, mode="a", header=not args.history.exists(), index=False)
    write_json(output / "summary.json", {"history": history.to_dict("records")})
    labels = list(margins.label.unique())
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.float_format", "{:.3f}".format):
        print(margins.pivot(index="outcome", columns="label", values=["dispersion", "bias", "crps"])[[(m, l) for m in ("dispersion", "bias", "crps") for l in labels]])
        print(dependences.pivot(index="pair", columns="label", values=["sim_rank_corr", "observed_corr", "dependence_gain"]))
        print(history.to_string(index=False))
        print(ranked.groupby("label").head(12).to_string(index=False))


if __name__ == "__main__":
    main()
