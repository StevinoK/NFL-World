"""NFL standings, tiebreakers and playoff seeding for one simulated season.

Implements the league's division and wild-card tiebreaking procedures through strength of schedule; the
points-based steps that follow are replaced by a coin toss. Ties within a game count as half a win.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DIVISIONS = {
    "AFC East": ("BUF", "MIA", "NE", "NYJ"), "AFC North": ("BAL", "CIN", "CLE", "PIT"),
    "AFC South": ("HOU", "IND", "JAX", "TEN"), "AFC West": ("DEN", "KC", "LAC", "LV"),
    "NFC East": ("DAL", "NYG", "PHI", "WAS"), "NFC North": ("CHI", "DET", "GB", "MIN"),
    "NFC South": ("ATL", "CAR", "NO", "TB"), "NFC West": ("ARI", "LAR", "SEA", "SF"),
}
TEAMS = tuple(team for teams in DIVISIONS.values() for team in teams)
INDEX = {team: index for index, team in enumerate(TEAMS)}
DIVISION = np.array([next(name for name, teams in DIVISIONS.items() if team in teams) for team in TEAMS])
CONFERENCE = np.array([name[:3] for name in DIVISION])
SEEDS = 7


@dataclass
class Season:
    """Head-to-head tallies for one season. `points[i, j]` is i's wins (+0.5 per tie) over j in `games[i, j]` meetings."""

    games: np.ndarray
    points: np.ndarray
    wins: np.ndarray

    @classmethod
    def from_results(cls, home: np.ndarray, away: np.ndarray, home_result: np.ndarray) -> "Season":
        """`home_result` is 1 for a home win, 0 for an away win and 0.5 for a tie."""
        size = len(TEAMS)
        games, points, wins = (np.zeros((size, size)) for _ in range(3))
        np.add.at(games, (home, away), 1)
        np.add.at(games, (away, home), 1)
        np.add.at(points, (home, away), home_result)
        np.add.at(points, (away, home), 1 - home_result)
        np.add.at(wins, (home, away), home_result == 1)
        np.add.at(wins, (away, home), home_result == 0)
        return cls(games, points, wins)

    def __post_init__(self):
        self.played = self.games.sum(1)
        self.record = _pct(self.points.sum(1), self.played)

    def pct(self, team: int, opponents) -> float:
        return _pct(self.points[team, opponents].sum(), self.games[team, opponents].sum())

    def common(self, group: list[int]) -> list[int]:
        met = np.all(self.games[group] > 0, axis=0)
        return [team for team in np.flatnonzero(met) if team not in group]

    def victory_strength(self, team: int) -> float:
        return _pct((self.wins[team] * self.points.sum(1)).sum(), (self.wins[team] * self.played).sum())

    def schedule_strength(self, team: int) -> float:
        return _pct((self.games[team] * self.points.sum(1)).sum(), (self.games[team] * self.played).sum())


def _pct(points, played):
    return np.divide(points, played, out=np.zeros_like(np.asarray(points, dtype=float)), where=np.asarray(played) > 0)


def _best(values: dict[int, float]) -> list[int]:
    top = max(values.values())
    return [team for team, value in values.items() if np.isclose(value, top)]


def _division_steps(season: Season, group: list[int]):
    division = list(np.flatnonzero(DIVISION == DIVISION[group[0]]))
    conference = list(np.flatnonzero(CONFERENCE == CONFERENCE[group[0]]))
    common = season.common(group)
    yield {team: season.pct(team, [other for other in group if other != team]) for team in group}
    yield {team: season.pct(team, [other for other in division if other != team]) for team in group}
    yield {team: season.pct(team, common) for team in group}
    yield {team: season.pct(team, [other for other in conference if other != team]) for team in group}
    yield {team: season.victory_strength(team) for team in group}
    yield {team: season.schedule_strength(team) for team in group}


def _wildcard_steps(season: Season, group: list[int]):
    conference = list(np.flatnonzero(CONFERENCE == CONFERENCE[group[0]]))
    if len(group) == 2:
        if season.games[group[0], group[1]] > 0:
            yield {team: season.pct(team, [other for other in group if other != team]) for team in group}
    else:
        # Head-to-head applies to three or more clubs only as a sweep: one beat all the others, or lost to all.
        others = {team: [other for other in group if other != team] for team in group}
        met = {team: bool(np.all(season.games[team, others[team]] > 0)) for team in group}
        swept = [team for team in group
                 if met[team] and np.all(season.wins[team, others[team]] == season.games[team, others[team]])]
        if swept:
            yield {team: float(team in swept) for team in group}
        lost = [team for team in group
                if met[team] and np.all(season.wins[others[team], team] == season.games[team, others[team]])]
        if lost:
            yield {team: -float(team in lost) for team in group}
    yield {team: season.pct(team, [other for other in conference if other != team]) for team in group}
    common = season.common(group)
    if all(season.games[team, common].sum() >= 4 for team in group):
        yield {team: season.pct(team, common) for team in group}
    yield {team: season.victory_strength(team) for team in group}
    yield {team: season.schedule_strength(team) for team in group}


def break_tie(season: Season, group: list[int], steps, rng: np.random.Generator) -> int:
    """The club that wins a tie. Each time a step trims the group, the procedure restarts with the survivors."""
    group = sorted(group)
    while len(group) > 1:
        for values in steps(season, group):
            survivors = _best(values)
            if len(survivors) < len(group):
                group = sorted(survivors)
                break
        else:
            return int(rng.choice(group))
    return group[0]


def _rank(season: Season, pool: list[int], steps, rng, eligible=lambda remaining: remaining) -> list[int]:
    order, remaining = [], list(pool)
    while remaining:
        candidates = eligible(remaining)
        leaders = _best({team: float(season.record[team]) for team in candidates})
        winner = leaders[0] if len(leaders) == 1 else break_tie(season, leaders, steps, rng)
        order.append(winner)
        remaining.remove(winner)
    return order


def division_order(season: Season, division: str, rng: np.random.Generator) -> list[int]:
    return _rank(season, list(np.flatnonzero(DIVISION == division)), _division_steps, rng)


def seed(season: Season, rng: np.random.Generator) -> np.ndarray:
    """Playoff seed per team (1-7), 0 for teams that miss the playoffs."""
    seeds = np.zeros(len(TEAMS), dtype=int)
    orders = {division: division_order(season, division, rng) for division in DIVISIONS}
    for conference in ("AFC", "NFC"):
        divisions = [division for division in DIVISIONS if division.startswith(conference)]
        champions = _rank(season, [orders[division][0] for division in divisions], _wildcard_steps, rng)
        rest = [team for division in divisions for team in orders[division][1:]]

        def eligible(remaining):
            # Only the highest-ranked remaining club from each division may take part in a wild-card tie.
            return [next(team for team in orders[division] if team in remaining)
                    for division in divisions if any(team in remaining for team in orders[division])]

        wildcards = _rank(season, rest, _wildcard_steps, rng, eligible)[:SEEDS - len(champions)]
        for position, team in enumerate(champions + wildcards, start=1):
            seeds[team] = position
    return seeds


def division_winners(seeds: np.ndarray) -> np.ndarray:
    return (seeds >= 1) & (seeds <= len(DIVISIONS) // 2)
