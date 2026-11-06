"""Rest-of-season simulations and roster-move experiments.

Every unplayed game is simulated once per scenario and cached. Seasons are then assembled by drawing one
simulation of each remaining game; under the model the games are independent given their pregame inputs, so
this matches simulating whole seasons. An experiment moves players between rosters (or out of the league),
re-simulates only the games of the teams it touches, and reuses the baseline for the rest. Experiments share
the baseline's run id, game seeds and season draws, so their differences come from the moves, not from noise.

    python -m nflsim.sim.season --simulations 100 --backend local --parallelism 10
    python -m nflsim.sim.season --move "Kirk Cousins>MIN@QB1" --name cousins-to-min
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nflsim.features.game_inputs import build_real_game, slot_assignments
from nflsim.features.slots import SLOT_LIMITS
from nflsim.names import row_display_name
from nflsim.paths import ROOT, canonical_hash
from nflsim.sim import standings
from nflsim.sim.execution import execute_ridge_games
from nflsim.sim.synthetic_games import _load_inputs, snapshot_depth

PLAYER_ID = re.compile(r"^\d{2}-\d{7}$")
SLOT = re.compile(r"^(QB|RB|WR|TE)([1-9])$")


@dataclass(frozen=True)
class Move:
    """Put `player` on `team` from the experiment's first week (team None: out for the season), optionally in `slot`."""

    player: str
    team: str | None
    slot: str | None = None


def parse_move(text: str) -> Move:
    """"Kirk Cousins>MIN@QB1", "Kirk Cousins>MIN" or "Patrick Mahomes>OUT"."""
    if ">" not in text:
        raise ValueError(f"Move must look like 'Player Name>TEAM[@SLOT]' or 'Player Name>OUT': {text}")
    player, target = (part.strip() for part in text.rsplit(">", 1))
    team, _, slot = (part.strip().upper() for part in target.partition("@"))
    if team == "OUT":
        if slot:
            raise ValueError("A player moved OUT cannot take a slot")
        return Move(player, None)
    if team not in standings.INDEX:
        raise ValueError(f"Unknown team {team}")
    if slot:
        match = SLOT.match(slot)
        if not match or int(match.group(2)) > SLOT_LIMITS[match.group(1)]:
            raise ValueError(f"Slot must be one of {', '.join(f'{p}1-{p}{n}' for p, n in SLOT_LIMITS.items())}: {slot}")
    return Move(player, team, slot or None)


def quarterbacks(roster: pd.DataFrame) -> pd.DataFrame:
    """Every rostered QB's latest row: player_id, player_name, team, depth_team."""
    rows = roster[roster.position.astype(str).eq("QB")].assign(player_id=lambda frame: frame.player_id.astype(str))
    rows = rows[rows.player_id.ne("nan")].assign(week=lambda frame: pd.to_numeric(frame.week, errors="coerce"))
    latest = rows.sort_values("week").groupby("player_id", as_index=False).tail(1)
    latest = latest.assign(player_name=latest.apply(row_display_name, axis=1))
    return latest[["player_id", "player_name", "team", "depth_team"]].sort_values(["team", "depth_team", "player_name"]).reset_index(drop=True)


def _normal(name: str) -> str:
    return re.sub(r"[^a-z ]", "", str(name).lower().replace(".", " ")).split(" jr")[0].strip()


def resolve_player(roster: pd.DataFrame, player: str) -> str:
    if PLAYER_ID.match(player):
        return player
    rows = roster[roster.player_name.map(_normal).eq(_normal(player))]
    ids = rows.sort_values("week").player_id.astype(str).unique()
    if len(ids) != 1:
        found = ", ".join(f"{pid} ({rows[rows.player_id.eq(pid)].team.iloc[-1]})" for pid in ids) or "none"
        raise ValueError(f"'{player}' must match exactly one rostered player (found {found}); pass the player id instead")
    return str(ids[0])


def apply_moves(roster: pd.DataFrame, schedule: pd.DataFrame, moves: list[Move], first_week: int):
    """Roster with each move applied from `first_week`, plus the teams whose games it changes."""
    roster = roster.copy()
    weeks = pd.to_numeric(roster.week, errors="coerce")
    affected = set()
    for move in moves:
        player_id = resolve_player(roster, move.player)
        rows = roster[roster.player_id.astype(str).eq(player_id)]
        if rows.empty:
            raise ValueError(f"{move.player} has no 2026 roster rows")
        template = rows.sort_values("week").iloc[-1]
        upcoming = roster.player_id.astype(str).eq(player_id) & weeks.ge(first_week)
        affected |= set(roster.loc[upcoming, "team"].astype(str)) | {str(template.team)}
        roster, weeks = roster[~upcoming], weeks[~upcoming]
        if move.team is None:
            continue
        affected.add(move.team)
        games = schedule[(schedule.home_team.eq(move.team) | schedule.away_team.eq(move.team)) & schedule.week.ge(first_week)]
        added = pd.DataFrame([template] * len(games)).assign(
            team=move.team, week=games.week.to_numpy(), game_id=games.game_id.astype(str).to_numpy(),
            game_date=games.gameday.to_numpy(), status="ACT", roster_is_projected=True)
        if move.slot and "depth_team" in added:
            added["depth_team"] = int(move.slot[2:])
        roster = pd.concat([roster, added], ignore_index=True)
        weeks = pd.to_numeric(roster.week, errors="coerce")
    return roster.reset_index(drop=True), affected


def forced_assignments(roster, depth, player_stats, team, game_row, moves):
    """Default slot assignments for `team`, with each slotted move's player placed in his slot."""
    kickoff = pd.Timestamp(game_row.gameday)
    assignments = slot_assignments(roster, depth, team, int(game_row.week), kickoff, player_stats)
    for move in moves:
        if move.team != team or not move.slot:
            continue
        player_id = resolve_player(roster, move.player)
        position, index = move.slot[:-1], int(move.slot[-1]) - 1
        name = row_display_name(roster[roster.player_id.astype(str).eq(player_id)].iloc[-1])
        group = [assignments[f"{position}{rank + 1}"] for rank in range(SLOT_LIMITS[position])]
        group = [item for item in group if item["player_id"] != player_id]
        group.insert(index, {"player_id": player_id, "player_name": name, "position": position})
        for rank, item in enumerate(group[:SLOT_LIMITS[position]]):
            assignments[f"{position}{rank + 1}"] = item
    return assignments


def simulate(runtime, artifact_dir, bundle_hash, schedule, roster, games, moves, simulations, *, backend, parallelism,
             run_id, seed, on_progress=None) -> pd.DataFrame:
    depth = snapshot_depth(roster, schedule)
    depth["dt"] = pd.to_datetime(depth.dt, errors="coerce", utc=True).dt.tz_convert(None)
    prepared = []
    for row in games.itertuples(index=False):
        row = pd.Series(row._asdict())
        forced = {team: forced_assignments(roster, depth, runtime.player_stats_wide, team, row, moves)
                  for team in (row.home_team, row.away_team) if any(move.team == team and move.slot for move in moves)}
        prepared.append(build_real_game(runtime, runtime.player_stats_wide, runtime.team_stats_wide, roster, depth,
                                        schedule, row, simulations, assignments_by_team=forced))
    done = {"count": 0}

    def progress(completed, total):
        if on_progress:
            on_progress(completed, total)
        if completed * 20 // total > done["count"]:
            done["count"] = completed * 20 // total
            print(f"  simulated {completed:,}/{total:,} game simulations", flush=True)

    results = execute_ridge_games(prepared, artifact_dir, bundle_hash, backend=backend, parallelism=parallelism,
                                  run_id=run_id, seed=seed, progress=progress)
    frames = [frame[["game_id", "simulation", "home_score", "away_score"]] for frame, _ in results.values()]
    return pd.concat(frames, ignore_index=True)


def cached_scores(path: Path, make) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    scores = make()
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(path, index=False)
    return scores


def play_seasons(schedule: pd.DataFrame, scores: pd.DataFrame, seasons: int, seed: int):
    """Wins (+0.5 per tie) and playoff seeds per simulated season, shape (seasons, 32)."""
    played = schedule[schedule.home_score.notna()]
    remaining = schedule[schedule.home_score.isna()].sort_values("game_id")
    results = (np.sign(scores.home_score - scores.away_score) + 1) / 2
    table = scores.assign(result=results).pivot(index="simulation", columns="game_id", values="result")
    table = table.reindex(columns=remaining.game_id.astype(str))
    if table.isna().any().any():
        missing = table.columns[table.isna().any()].tolist()
        raise ValueError(f"No simulations for remaining games {missing[:5]}")
    outcomes = table.to_numpy()
    home = np.concatenate([played.home_team.map(standings.INDEX), remaining.home_team.map(standings.INDEX)]).astype(int)
    away = np.concatenate([played.away_team.map(standings.INDEX), remaining.away_team.map(standings.INDEX)]).astype(int)
    fixed = ((np.sign(played.home_score - played.away_score) + 1) / 2).to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(outcomes), size=(seasons, outcomes.shape[1]))
    wins = np.zeros((seasons, len(standings.TEAMS)))
    seeds = np.zeros((seasons, len(standings.TEAMS)), dtype=int)
    for index in range(seasons):
        result = np.concatenate([fixed, outcomes[draws[index], np.arange(outcomes.shape[1])]])
        season = standings.Season.from_results(home, away, result)
        wins[index] = season.points.sum(1)
        seeds[index] = standings.seed(season, rng)
    return wins, seeds


def summarize(wins: np.ndarray, seeds: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({
        "team": standings.TEAMS, "division": standings.DIVISION, "wins": wins.mean(0),
        "playoffs": (seeds > 0).mean(0), "division_title": standings.division_winners(seeds).mean(0),
        "bye": (seeds == 1).mean(0),
    })
    for position in range(1, standings.SEEDS + 1):
        frame[f"seed_{position}"] = (seeds == position).mean(0)
    return frame.sort_values(["division", "wins"], ascending=[True, False]).reset_index(drop=True)


def compare(base: tuple, experiment: tuple) -> pd.DataFrame:
    """Experiment minus baseline per team, with standard errors from the paired season draws."""
    rows = []
    for team_index, team in enumerate(standings.TEAMS):
        row = {"team": team}
        for name, measure in (("wins", lambda wins, seeds: wins), ("playoffs", lambda wins, seeds: seeds > 0),
                              ("division_title", lambda wins, seeds: standings.division_winners(seeds)),
                              ("bye", lambda wins, seeds: seeds == 1)):
            before = measure(*base)[:, team_index].astype(float)
            after = measure(*experiment)[:, team_index].astype(float)
            difference = after - before
            row.update({f"{name}_base": before.mean(), f"{name}_experiment": after.mean(), f"{name}_change": difference.mean(),
                        f"{name}_se": difference.std(ddof=1) / np.sqrt(len(difference))})
        rows.append(row)
    return pd.DataFrame(rows)


def latest_bundle() -> Path:
    bundles = [path for path in (ROOT / "artifacts/bundles").glob("*") if (path / "weekly_inputs/schedule.parquet").exists()]
    if not bundles:
        raise FileNotFoundError("No weekly-refreshed bundle with weekly_inputs under artifacts/bundles")
    return max(bundles, key=lambda path: path.stat().st_mtime)


def game_changes(baseline: pd.DataFrame, rerun: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Baseline and experiment margin, total and home win rate for each re-simulated game."""

    def per_game(scores):
        margin = scores.home_score - scores.away_score
        return scores.assign(margin=margin, total=scores.home_score + scores.away_score, home_win=(np.sign(margin) + 1) / 2
                             ).groupby("game_id")[["margin", "total", "home_win"]].mean()

    base, experiment = per_game(baseline[baseline.game_id.isin(rerun.game_id)]), per_game(rerun)
    table = base.add_suffix("_base").join(experiment.add_suffix("_experiment")).reset_index()
    games = schedule.assign(game_id=schedule.game_id.astype(str))[["game_id", "week", "home_team", "away_team"]]
    return games.merge(table, on="game_id").sort_values(["week", "game_id"]).reset_index(drop=True)


@dataclass
class Experiment:
    """Files and tables from one finished experiment."""

    output: Path
    comparison: pd.DataFrame
    summary: pd.DataFrame
    games: pd.DataFrame
    manifest: dict


def load_experiment(output: Path) -> Experiment:
    return Experiment(output, pd.read_csv(output / "comparison.csv"), pd.read_csv(output / "summary.csv"),
                      pd.read_csv(output / "games.csv"), json.loads((output / "manifest.json").read_text(encoding="utf-8")))


def season_root(season: int, bundle_hash: str) -> Path:
    return ROOT / "outputs/season" / str(season) / bundle_hash[:12]


def saved_experiments(season: int, bundle_hash: str) -> list[Path]:
    """Finished experiments for this bundle, newest first."""
    folders = (season_root(season, bundle_hash) / "experiments").glob("*")
    finished = [folder for folder in folders if (folder / "manifest.json").is_file() and (folder / "games.csv").is_file()]
    return sorted(finished, key=lambda folder: (folder / "manifest.json").stat().st_mtime, reverse=True)


def run_experiment(runtime, artifact_dir: Path, bundle_hash: str, season: int, moves: list[Move], *, first_week: int | None = None,
                   name: str | None = None, simulations: int = 100, seasons: int = 10000, backend: str = "local",
                   parallelism: int = 8, seed: int = 42, on_stage=None, on_progress=None) -> Experiment:
    """Compare `moves` with the baseline season. The baseline and each move's games are cached, so repeats are quick."""
    stage = on_stage or (lambda text: print(text, flush=True))
    schedule, roster, _ = _load_inputs(season, artifact_dir / "weekly_inputs")
    remaining = schedule[schedule.home_score.isna()].sort_values("game_id")
    if remaining.empty:
        raise ValueError(f"No unplayed games left in {season}")
    first_week = first_week or int(remaining.week.min())
    root = season_root(season, bundle_hash)
    run_id = f"season-{season}-{bundle_hash[:12]}"
    options = dict(backend=backend, parallelism=parallelism, run_id=run_id, seed=seed, on_progress=on_progress)
    tag = f"sims{simulations}-seed{seed}"
    stage(f"Baseline: {len(remaining)} unplayed games from week {int(remaining.week.min())}; bundle {bundle_hash[:12]}")
    baseline = cached_scores(root / f"baseline-{tag}.parquet", lambda: simulate(
        runtime, artifact_dir, bundle_hash, schedule, roster, remaining, [], simulations, **options))
    moved, affected = apply_moves(roster, schedule, moves, first_week)
    changed = remaining[(remaining.home_team.isin(affected) | remaining.away_team.isin(affected))
                        & remaining.week.ge(first_week)]
    key = canonical_hash([[asdict(move) for move in moves], first_week, tag])[:12]
    name = name or "-".join(re.sub(r"[^a-z0-9]+", "", move.player.lower()) + "-" + (move.team or "out").lower()
                            for move in moves)
    output = root / "experiments" / name
    stage(f"Re-simulating {len(changed)} games for {', '.join(sorted(affected))}")
    rerun = cached_scores(root / f"experiment-{key}-{tag}.parquet", lambda: simulate(
        runtime, artifact_dir, bundle_hash, schedule, moved, changed, moves, simulations, **options))
    stage(f"Assembling {seasons:,} seasons")
    base = play_seasons(schedule, baseline, seasons, seed)
    scores = pd.concat([baseline[~baseline.game_id.isin(changed.game_id)], rerun], ignore_index=True)
    experiment = play_seasons(schedule, scores, seasons, seed)
    output.mkdir(parents=True, exist_ok=True)
    summarize(*base).to_csv(root / "baseline_summary.csv", index=False)
    summarize(*experiment).to_csv(output / "summary.csv", index=False)
    compare(base, experiment).sort_values("playoffs_change", key=abs, ascending=False).to_csv(output / "comparison.csv", index=False)
    game_changes(baseline, rerun, schedule).to_csv(output / "games.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "season": season, "bundle_hash": bundle_hash, "moves": [asdict(move) for move in moves],
        "first_week": first_week, "affected_teams": sorted(affected), "games_resimulated": changed.game_id.tolist(),
        "simulations_per_game": simulations, "seasons": seasons, "seed": seed, "run_id": run_id,
        "standard_errors": "paired season draws only; game-simulation noise is shared through common seeds",
    }, indent=2), encoding="utf-8")
    return load_experiment(output)


def main():
    parser = argparse.ArgumentParser(description="Simulate the rest of a season, optionally with roster moves")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--artifact-dir", type=Path, default=None, help="Defaults to the newest weekly-refreshed bundle")
    parser.add_argument("--move", action="append", default=[], help="'Player Name>TEAM[@SLOT]' or 'Player Name>OUT'; repeatable")
    parser.add_argument("--from-week", type=int, default=None, help="First week the moves apply (default: first unplayed week)")
    parser.add_argument("--name", default=None, help="Experiment folder name")
    parser.add_argument("--simulations", type=int, default=100, help="Simulations per remaining game")
    parser.add_argument("--seasons", type=int, default=10000, help="Seasons assembled from the game simulations")
    parser.add_argument("--backend", choices=("local", "modal"), default="local")
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from nflsim.bundle import validate_bundle, verify_bundle_hash
    from nflsim.sim.runtime import RidgeRuntime

    artifact_dir = (args.artifact_dir or latest_bundle()).resolve()
    validate_bundle(artifact_dir)
    bundle_hash = verify_bundle_hash(artifact_dir)
    runtime = RidgeRuntime(artifact_dir, strict_schema=True)
    moves = [parse_move(text) for text in args.move]
    if not moves:
        schedule, roster, _ = _load_inputs(args.season, artifact_dir / "weekly_inputs")
        remaining = schedule[schedule.home_score.isna()].sort_values("game_id")
        root = season_root(args.season, bundle_hash)
        baseline = cached_scores(root / f"baseline-sims{args.simulations}-seed{args.seed}.parquet", lambda: simulate(
            runtime, artifact_dir, bundle_hash, schedule, roster, remaining, [], args.simulations, backend=args.backend,
            parallelism=args.parallelism, run_id=f"season-{args.season}-{bundle_hash[:12]}", seed=args.seed))
        table = summarize(*play_seasons(schedule, baseline, args.seasons, args.seed))
        table.to_csv(root / "baseline_summary.csv", index=False)
        print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        return
    result = run_experiment(runtime, artifact_dir, bundle_hash, args.season, moves, first_week=args.from_week, name=args.name,
                            simulations=args.simulations, seasons=args.seasons, backend=args.backend,
                            parallelism=args.parallelism, seed=args.seed)
    comparison = result.comparison
    shown = comparison[comparison.team.isin(result.manifest["affected_teams"]) | comparison.playoffs_change.abs().ge(0.01)]
    columns = ["team", "wins_base", "wins_experiment", "wins_change", "playoffs_base", "playoffs_experiment",
               "playoffs_change", "playoffs_se", "division_title_change", "bye_change"]
    print(shown[columns].to_string(index=False, float_format=lambda value: f"{value:+.3f}"))
    print(f"Wrote {result.output}")


if __name__ == "__main__":
    main()
