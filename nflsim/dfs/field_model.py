"""A model of the lineups a DraftKings Showdown field enters, fitted to finished contests.

Every valid lineup from a slate's most-projected players gets a probability from a conditional logit on its features:
projected points behind the best lineup, salary left, captain position, the team split, QB stacks and kickers or
defenses at FLEX. The fit maximizes the likelihood of the lineup counts real fields entered (saved from FantasyLabs by
`nflsim.dfs.fantasylabs`). A field drawn from it duplicates lineups the way real fields do, because the popular
lineups carry most of the probability.

Lineups using a player outside the pool can't be drawn; `coverage` reports the share of real entries the pool
explains.

    python -m nflsim.dfs.field_model --folds 5
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from nflsim.dfs.fantasylabs import OUTPUT

POOL = 24
SALARY_CAP = 50_000
CAPTAIN = 1.5
POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
FEATURES = [
    "gap", "gap_past_5", "gap_past_15",  # Projected points behind the slate's best lineup (per 10), piecewise.
    "salary_left", "salary_left_sq", "full_salary",
    "cpt_RB", "cpt_WR", "cpt_TE", "cpt_K", "cpt_DST", "cpt_favorite",
    "split_5", "split_4", "split_2", "split_1",  # Players from the favorite (3 is the base).
    "qbs", "cpt_qb_stacked", "cpt_catcher_with_qb", "flex_K", "flex_DST",
]


@dataclass
class Slate:
    """A slate's player pool: the `POOL` best FLEX projections, with each player's CPT salary."""

    name: str
    players: pd.DataFrame  # player_id, name, team, position, salary, cpt_salary, projection, favorite
    lineups: np.ndarray = field(default=None, repr=False)  # (n, 6) pool indices: captain, then five FLEX
    features: np.ndarray = field(default=None, repr=False)
    counts: np.ndarray | None = field(default=None, repr=False)
    entries: int = 0  # Real entries, including lineups outside the pool.

    @property
    def coverage(self) -> float:
        return float(self.counts.sum() / self.entries) if self.counts is not None and self.entries else float("nan")


def pool_players(players: pd.DataFrame, size: int = POOL) -> pd.DataFrame:
    """FLEX rows of a `players.csv`-shaped table joined with CPT salaries, best projections first."""
    flex = players[players.slot.eq("FLEX")].copy()
    cpt = players[players.slot.eq("CPT")].set_index("player_id").salary.rename("cpt_salary")
    flex = flex.join(cpt, on="player_id")
    flex["cpt_salary"] = flex.cpt_salary.fillna((flex.salary * CAPTAIN).round(-2)).astype(int)
    flex = flex.rename(columns={"projected_points": "projection"}).sort_values("projection", ascending=False).head(size)
    columns = ["player_id", "name", "team", "position", "salary", "cpt_salary", "projection", "favorite"]
    return flex[columns].reset_index(drop=True)


def enumerate_lineups(players: pd.DataFrame, salary_cap: int = SALARY_CAP) -> np.ndarray:
    """Every captain plus five other pool players within the cap and using both teams, as pool indices."""
    count = len(players)
    salary, cpt_salary = players.salary.to_numpy(), players.cpt_salary.to_numpy()
    team = pd.factorize(players.team)[0]
    combos = np.array(list(itertools.combinations(range(count - 1), 5)), dtype=np.int16)
    blocks = []
    for captain in range(count):
        others = np.delete(np.arange(count, dtype=np.int16), captain)
        flex = others[combos]
        total = cpt_salary[captain] + salary[flex].sum(axis=1)
        teams = np.concatenate([team[flex], np.full((len(flex), 1), team[captain])], axis=1)
        keep = (total <= salary_cap) & (teams.min(axis=1) != teams.max(axis=1))
        blocks.append(np.column_stack([np.full(keep.sum(), captain, dtype=np.int16), flex[keep]]))
    return np.concatenate(blocks)


def lineup_features(players: pd.DataFrame, lineups: np.ndarray) -> np.ndarray:
    projection = players.projection.to_numpy(float)
    salary, cpt_salary = players.salary.to_numpy(float), players.cpt_salary.to_numpy(float)
    position = players.position.astype(str).to_numpy()
    favorite = players.favorite.astype(bool).to_numpy()
    team = pd.factorize(players.team)[0]
    captain, flex = lineups[:, 0], lineups[:, 1:]
    points = CAPTAIN * projection[captain] + projection[flex].sum(axis=1)
    gap = (points - points.max()) / 10
    left = (SALARY_CAP - cpt_salary[captain] - salary[flex].sum(axis=1)) / 1000
    favorites = favorite[captain].astype(int) + favorite[flex].sum(axis=1)
    is_qb = position == "QB"
    catcher = np.isin(position, ["WR", "TE"])
    qb_team = np.full(len(lineups), -1)
    for slot in range(6):
        member = lineups[:, slot]
        qb_team = np.where(is_qb[member] & (qb_team < 0), team[member], qb_team)
    captain_qb = is_qb[captain]
    stacked = (catcher[flex] & (team[flex] == team[captain][:, None])).any(axis=1)
    with_qb = (is_qb[flex] & (team[flex] == team[captain][:, None])).any(axis=1)
    columns = [
        gap, np.maximum(gap + 0.5, 0), np.maximum(gap + 1.5, 0),
        left, left ** 2 / 10, (left == 0).astype(float),
        *[(position[captain] == name).astype(float) for name in ["RB", "WR", "TE", "K", "DST"]],
        favorite[captain].astype(float),
        *[(favorites == count).astype(float) for count in (5, 4, 2, 1)],
        is_qb[lineups].sum(axis=1), (captain_qb & stacked).astype(float), (catcher[captain] & with_qb).astype(float),
        (position[flex] == "K").sum(axis=1), (position[flex] == "DST").sum(axis=1),
    ]
    return np.column_stack(columns).astype(np.float32)


def _keys(lineups: np.ndarray) -> np.ndarray:
    """Captain and FLEX set as one integer, so FLEX order doesn't matter."""
    mask = np.zeros(len(lineups), dtype=np.int64)
    for slot in range(1, 6):
        mask |= np.int64(1) << lineups[:, slot].astype(np.int64)
    return (lineups[:, 0].astype(np.int64) << 32) | mask


def observed_counts(slate: Slate, entered: pd.DataFrame) -> np.ndarray:
    """Real entries for each enumerated lineup; `entered` has columns cpt, flex1..flex5 (player ids) and entries."""
    index = {player: position for position, player in enumerate(slate.players.player_id)}
    ids = entered[["cpt", "flex1", "flex2", "flex3", "flex4", "flex5"]].to_numpy()
    mapped = np.vectorize(lambda player: index.get(player, -1))(ids)
    inside = (mapped >= 0).all(axis=1)
    keys = _keys(mapped[inside].astype(np.int16))
    order = np.argsort(_keys(slate.lineups))
    sorted_keys = _keys(slate.lineups)[order]
    position = np.searchsorted(sorted_keys, keys)
    found = (position < len(sorted_keys)) & (sorted_keys[np.minimum(position, len(sorted_keys) - 1)] == keys)
    counts = np.zeros(len(slate.lineups), dtype=np.float64)
    np.add.at(counts, order[position[found]], entered.entries.to_numpy()[inside][found])
    return counts


def load_slate(folder: Path, size: int = POOL) -> Slate:
    players = pool_players(pd.read_csv(folder / "players.csv"), size)
    slate = Slate(folder.name, players)
    slate.lineups = enumerate_lineups(players)
    slate.features = lineup_features(players, slate.lineups)
    entered = pd.read_parquet(folder / "lineups.parquet")
    slate.counts = observed_counts(slate, entered)
    slate.entries = int(entered.entries.sum())
    return slate


def _objective(beta: np.ndarray, slates: list[Slate], penalty: float):
    value, gradient, total = 0.0, np.zeros_like(beta), 0.0
    for slate in slates:
        scores = slate.features.astype(np.float64) @ beta
        top = scores.max()
        weights = np.exp(scores - top)
        normalizer = weights.sum()
        observed = float(slate.counts.sum())
        value -= float(slate.counts @ scores) - observed * (top + np.log(normalizer))
        gradient -= slate.counts @ slate.features - observed * (weights @ slate.features) / normalizer
        total += observed
    value, gradient = value / total, gradient / total
    return value + penalty * beta @ beta, gradient + 2 * penalty * beta


@dataclass
class FieldModel:
    coefficients: np.ndarray
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    pool: int = POOL

    @classmethod
    def fit(cls, slates: list[Slate], penalty: float = 1e-4) -> "FieldModel":
        start = np.zeros(len(FEATURES))
        start[0] = 2.0
        result = minimize(_objective, start, args=(slates, penalty), jac=True, method="L-BFGS-B",
                          options={"maxiter": 500})
        if not result.success:
            raise RuntimeError(f"Field model fit failed: {result.message}")
        return cls(result.x)

    def probabilities(self, slate: Slate) -> np.ndarray:
        scores = slate.features.astype(np.float64) @ self.coefficients
        weights = np.exp(scores - scores.max()).astype(np.float64)
        return weights / weights.sum()

    def sample(self, slate: Slate, entries: int, rng: np.random.Generator) -> np.ndarray:
        """Entries per enumerated lineup in a field of `entries`."""
        return rng.multinomial(entries, self.probabilities(slate))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"pool": self.pool, "coefficients": dict(zip(self.features, map(float, self.coefficients)))},
                                   indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FieldModel":
        saved = json.loads(path.read_text(encoding="utf-8"))
        return cls(np.array([saved["coefficients"][name] for name in FEATURES]), pool=saved["pool"])


# ------------------------------------------------------------------ checks against held-out contests


def ownership(slate: Slate, weights: np.ndarray) -> pd.DataFrame:
    """Share of entries holding each pool player at CPT and at FLEX, given entries (or probabilities) per lineup."""
    weights = weights / weights.sum()
    captain = np.bincount(slate.lineups[:, 0], weights, minlength=len(slate.players))
    flex = sum(np.bincount(slate.lineups[:, slot], weights, minlength=len(slate.players)) for slot in range(1, 6))
    return slate.players[["player_id", "name", "position", "salary", "projection"]].assign(cpt=captain, flex=flex)


def duplicated_share(counts: np.ndarray) -> float:
    """Share of entries whose lineup was entered more than once."""
    return float(counts[counts > 1].sum() / counts.sum())


def evaluate(model: FieldModel, slate: Slate, rng: np.random.Generator) -> dict:
    probabilities = model.probabilities(slate)
    observed = float(slate.counts.sum())
    drawn = model.sample(slate, int(observed), rng)
    actual, predicted = ownership(slate, slate.counts), ownership(slate, probabilities)
    uniform = -np.log(len(slate.lineups))
    left = (SALARY_CAP - slate.players.cpt_salary.to_numpy()[slate.lineups[:, 0]]
            - slate.players.salary.to_numpy()[slate.lineups[:, 1:]].sum(axis=1)) / 1000
    return {
        "slate": slate.name, "lineups": len(slate.lineups), "coverage": slate.coverage,
        "log_lik_per_entry": float(slate.counts @ np.log(probabilities + 1e-300) / observed), "uniform": uniform,
        "flex_own_corr": float(np.corrcoef(actual.flex, predicted.flex)[0, 1]),
        "flex_own_mae": float((actual.flex - predicted.flex).abs().mean()),
        "cpt_own_corr": float(np.corrcoef(actual.cpt, predicted.cpt)[0, 1]),
        "cpt_own_mae": float((actual.cpt - predicted.cpt).abs().mean()),
        "dup_actual": duplicated_share(slate.counts), "dup_model": duplicated_share(drawn),
        "left_actual": float(left @ slate.counts / observed), "left_model": float(left @ probabilities),
    }


def slate_folders(root: Path = OUTPUT) -> list[Path]:
    return sorted(folder for folder in root.iterdir() if (folder / "lineups.parquet").is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and check the Showdown field model")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--pool", type=int, default=POOL)
    parser.add_argument("--save", type=Path, default=Path(__file__).with_name("field_model.json"))
    args = parser.parse_args()
    slates = [load_slate(folder, args.pool) for folder in slate_folders()]
    print(f"{len(slates)} slates, coverage {np.mean([slate.coverage for slate in slates]):.1%} on average", flush=True)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(slates))
    rows = []
    for fold in range(args.folds):
        test = set(order[fold::args.folds])
        model = FieldModel.fit([slate for index, slate in enumerate(slates) if index not in test])
        rows += [evaluate(model, slates[index], rng) for index in sorted(test)]
    table = pd.DataFrame(rows).sort_values("slate")
    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))
    print(table.drop(columns=["slate"]).mean().round(3).to_string())
    model = FieldModel.fit(slates)
    model.save(args.save)
    print(pd.Series(model.coefficients, index=FEATURES).round(3).to_string())
    print(f"Saved {args.save}")


if __name__ == "__main__":
    main()
