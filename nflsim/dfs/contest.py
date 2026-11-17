"""Our Showdown entries against a field drawn from the fitted field model, with DraftKings' tie rule.

The field is the full contest: its lineups are drawn from `field_model`, so popular lineups appear many times. Entries
with equal scores split the prizes for the places they occupy together, which is what makes a heavily duplicated
lineup worth much less than its score suggests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nflsim.dfs.field_model import FieldModel, Slate, enumerate_lineups, lineup_features, pool_players


@dataclass
class Field:
    """Distinct field lineups (rows of six player keys, captain first) and how many entries play each."""

    lineups: list[list[str]]
    counts: np.ndarray

    @property
    def entries(self) -> int:
        return int(self.counts.sum())


def field_players(prices: pd.DataFrame, projections: pd.Series, favorite_team: str) -> pd.DataFrame:
    """A `players.csv`-shaped table for the field model from a DraftKings salary file.

    `projections` maps `short_name` to the public FLEX projection the field sees; `favorite_team` is the team the
    betting line favors."""
    rows = prices.assign(player_id=prices.short_name, name=prices.Name, team=prices.TeamAbbrev, position=prices.Position,
                         slot=np.where(prices["Roster Position"].eq("CPT"), "CPT", "FLEX"), salary=prices.Salary,
                         projected_points=prices.short_name.map(projections).fillna(0.0),
                         favorite=prices.TeamAbbrev.eq(favorite_team))
    return rows[["player_id", "name", "team", "position", "slot", "salary", "projected_points", "favorite"]]


def draw_field(model: FieldModel, players: pd.DataFrame, entries: int, rng: np.random.Generator) -> Field:
    pool = pool_players(players, model.pool)
    slate = Slate("upcoming", pool)
    slate.lineups = enumerate_lineups(pool)
    slate.features = lineup_features(pool, slate.lineups)
    counts = model.sample(slate, entries, rng)
    drawn = np.flatnonzero(counts)
    keys = pool.player_id.to_numpy()
    return Field([list(keys[slate.lineups[index]]) for index in drawn], counts[drawn].astype(np.int64))


def lineup_scores(lineups: list[list[str]], simulations: pd.DataFrame, captain_multiplier: float = 1.5) -> np.ndarray:
    """(lineups, simulations) scores; each lineup lists its captain's key first."""
    column = {name: index for index, name in enumerate(simulations.columns)}
    values = simulations.to_numpy(dtype=np.float32).T
    index = np.array([[column[player] for player in lineup] for lineup in lineups])
    weights = np.array([captain_multiplier, 1, 1, 1, 1, 1], dtype=np.float32)
    return np.einsum("lsn,s->ln", values[index], weights)


def payout_by_place(payouts: pd.DataFrame, places: int) -> np.ndarray:
    """Cash for places 1..places (index 0 unused)."""
    cash = np.zeros(places + 2)
    for row in payouts.itertuples(index=False):
        low, high = max(1, int(row.min_place)), min(places, int(row.max_place))
        if high >= low and np.isfinite(row.cash):
            cash[low:high + 1] = float(row.cash)
    return cash


def rounded(scores: np.ndarray) -> np.ndarray:
    """Scores to the hundredth, so the same lineup scored by different code paths ties exactly."""
    return np.round(np.asarray(scores, dtype=np.float64), 2).astype(np.float32)


class Standings:
    """Sorted competitor scores per simulation, for placing a lineup and splitting ties."""

    def __init__(self, scores: np.ndarray, counts: np.ndarray, cash: np.ndarray):
        scores = rounded(scores)
        rows, columns = scores.shape
        self.sorted = np.empty((rows, columns), dtype=np.float32)
        self.below = np.zeros((rows + 1, columns), dtype=np.int32)  # [k]: entries in the k lowest-scoring rows
        counts = np.asarray(counts, dtype=np.int32)
        for column in range(columns):  # One simulation at a time keeps memory to the two arrays above.
            order = np.argsort(scores[:, column], kind="stable")
            self.sorted[:, column] = scores[order, column]
            np.cumsum(counts[order], out=self.below[1:, column])
        self.total = int(counts.sum())
        self.cumulative_cash = np.concatenate([[0.0], np.cumsum(cash[1:])])  # cash for places 1..k
        self.places = len(cash) - 2

    def cash(self, scores: np.ndarray, extra_above: np.ndarray | None = None, extra_tied: np.ndarray | None = None) -> np.ndarray:
        """Cash each lineup in `scores` (lineups, simulations) wins entered once into these standings, with
        `extra_above` and `extra_tied` more competitors (our other entries) above or level with it."""
        scores = rounded(np.atleast_2d(scores))
        result = np.empty(scores.shape)
        for column in range(scores.shape[1]):
            sorted_column = self.sorted[:, column]
            values = scores[:, column]
            low = np.searchsorted(sorted_column, values, side="left")
            high = np.searchsorted(sorted_column, values, side="right")
            above = self.total - self.below[high, column]
            tied = self.below[high, column] - self.below[low, column]
            if extra_above is not None:
                above = above + extra_above[:, column]
                tied = tied + extra_tied[:, column]
            first = np.minimum(above + 1, self.places + 1)
            last = np.minimum(above + tied + 1, self.places)
            # Tied entries share the prizes of places first..last equally.
            shared = np.where(last >= first, self.cumulative_cash[last] - self.cumulative_cash[first - 1], 0.0)
            result[:, column] = shared / (tied + 1)
        return result


def portfolio_omega(profit: np.ndarray) -> np.ndarray:
    """Omega ratio at zero for each row of per-simulation profit: total gains over total losses."""
    gains = np.maximum(profit, 0.0).sum(axis=-1)
    losses = np.maximum(-profit, 0.0).sum(axis=-1)
    safe = np.where(losses > 0, losses, 1.0)
    return np.where(losses > 0, gains / safe, np.where(gains > 0, np.inf, 0.0))


OBJECTIVES = ("omega", "cash")


@dataclass
class Contest:
    """One contest we enter: its drawn field (scores per simulation and entries per distinct lineup), payout table,
    entry fee and how many entries we put in it."""

    field_scores: np.ndarray
    field_counts: np.ndarray
    payouts: pd.DataFrame
    entry_fee: float
    entries: int
    name: str = ""

    def cash_table(self) -> np.ndarray:
        return payout_by_place(self.payouts, int(self.field_counts.sum()) + self.entries)


class _ContestDraft:
    """Our picks so far in one contest, and what each shortlisted candidate would add to that contest's cash."""

    def __init__(self, candidates: np.ndarray, contest: Contest, shortlist: int):
        self.contest = contest
        self.standings = Standings(contest.field_scores, contest.field_counts, contest.cash_table())
        single = self.standings.cash(candidates).mean(axis=1)
        self.rows = np.argsort(-single)[:shortlist]
        self.pool = rounded(candidates[self.rows])
        self.above = np.zeros(self.pool.shape, dtype=np.int64)  # [c, s]: our picks here outscoring candidate c
        self.tied = np.zeros(self.pool.shape, dtype=np.int64)
        self.picked: list[int] = []  # rows of `pool`
        self.refresh()

    @property
    def open(self) -> int:
        return self.contest.entries - len(self.picked)

    def refresh(self) -> None:
        """`total`: cash our picks here win per simulation. `with_candidate`: that total if each candidate were added,
        counting the cash it takes from our picks it outscores or ties (tied entries split prizes)."""
        own = self.standings.cash(self.pool, self.above, self.tied)
        self.total = np.zeros(self.pool.shape[1])
        taken = np.zeros(self.pool.shape)
        if self.picked:
            scores = self.pool[self.picked]
            pick_above, pick_tied = self.above[self.picked], self.tied[self.picked] - 1  # A pick ties with itself.
            now = self.standings.cash(scores, pick_above, pick_tied)
            self.total = now.sum(axis=0)
            lose_above = now - self.standings.cash(scores, pick_above + 1, pick_tied)
            lose_tied = now - self.standings.cash(scores, pick_above, pick_tied + 1)
            for index in range(len(self.picked)):
                columns = np.flatnonzero((lose_above[index] != 0) | (lose_tied[index] != 0))
                if len(columns):
                    block, target = self.pool[:, columns], scores[index, columns]
                    taken[:, columns] += (block > target) * lose_above[index, columns] + (block == target) * lose_tied[index, columns]
        self.with_candidate = self.total[None, :] + own - taken

    def add(self, index: int) -> None:
        self.picked.append(index)
        self.above += self.pool[index][None, :] > self.pool
        self.tied += self.pool[index][None, :] == self.pool
        self.refresh()


def draft_contests(candidates: np.ndarray, contests: list[Contest], objective: str = "omega",
                   shortlist: int = 2_000) -> tuple[list[list[int]], list[np.ndarray]]:
    """Fill our entries in every contest at once, greedily adding the (lineup, contest) pair that most improves the
    whole portfolio's profit per simulation.

    All contests are scored on the same simulations, so profit in one simulation adds across contests and the
    objective sees how our contests rise and fall together. Within a contest our entries compete with and split
    against each other; across contests they don't. `objective` "omega" maximizes the portfolio's Omega ratio
    (gains over losses at zero profit), with expected profit breaking ties, including the case where nothing ever
    profits; "cash" maximizes expected profit. Each contest's shortlist is its `shortlist` best candidates on their
    own. Returns, per contest, the chosen candidate indices and their per-entry, per-simulation profit."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    states = [_ContestDraft(candidates, contest, shortlist) for contest in contests]
    fees = 0.0
    for _ in range(sum(contest.entries for contest in contests)):
        everything = sum(state.total for state in states)
        best = None  # ((omega, expected), contest, row); the first best contest and row win exact ties.
        for number, state in enumerate(states):
            if state.open <= 0:
                continue
            profit = everything[None, :] - state.total[None, :] + state.with_candidate - (fees + state.contest.entry_fee)
            expected = profit.mean(axis=1)
            if objective == "omega":
                omega = portfolio_omega(profit)
                row = int(np.lexsort((-np.arange(len(expected)), expected, omega))[-1])
                value = (float(omega[row]), float(expected[row]))
            else:
                row = int(np.argmax(expected))
                value = (float(expected[row]),)
            if best is None or value > best[0]:
                best = (value, number, row)
        _, number, row = best
        states[number].add(row)
        fees += states[number].contest.entry_fee
    chosen = [[int(state.rows[index]) for index in state.picked] for state in states]
    profits = [portfolio_cash(candidates[picks], contest.field_scores, contest.field_counts, contest.cash_table()) - contest.entry_fee
               if picks else np.zeros((0, candidates.shape[1])) for picks, contest in zip(chosen, contests)]
    return chosen, profits


def draft(candidates: np.ndarray, field_scores: np.ndarray, field_counts: np.ndarray, payouts: pd.DataFrame, entry_fee: float,
          picks: int, shortlist: int = 2_000, objective: str = "omega") -> tuple[list[int], np.ndarray]:
    """`draft_contests` for one contest: the picks and each pick's per-simulation profit."""
    chosen, profits = draft_contests(candidates, [Contest(field_scores, field_counts, payouts, entry_fee, picks)], objective, shortlist)
    return chosen[0], profits[0]


def portfolio_cash(ours: np.ndarray, field_scores: np.ndarray, field_counts: np.ndarray, cash: np.ndarray) -> np.ndarray:
    """Cash per entry and simulation when all of `ours` are entered together with the field."""
    standings = Standings(field_scores, field_counts, cash)
    ours = rounded(ours)
    above = (ours[None, :, :] > ours[:, None, :]).sum(axis=1)  # [i]: our other entries outscoring entry i
    tied = (ours[None, :, :] == ours[:, None, :]).sum(axis=1) - 1
    return standings.cash(ours, above, tied)
