from __future__ import annotations

import numpy as np
import pandas as pd

from nflsim.dfs.contest import Standings, draft, payout_by_place, portfolio_cash
from nflsim.dfs.field_model import duplicated_share, enumerate_lineups, lineup_features, observed_counts, ownership, Slate, FEATURES

PAYOUTS = pd.DataFrame({"min_place": [1, 2], "max_place": [1, 3], "cash": [100.0, 10.0]})


def test_payout_by_place():
    assert payout_by_place(PAYOUTS, 5).tolist() == [0, 100, 10, 10, 0, 0, 0]


def test_ties_split_the_places_they_cover():
    cash = payout_by_place(PAYOUTS, 10)
    # One simulation: field lineups scoring 90 (x1), 80 (x2), 50 (x6).
    standings = Standings(np.array([[90.0], [80.0], [50.0]]), np.array([1, 2, 6]), cash)
    result = standings.cash(np.array([[95.0], [90.0], [80.0], [79.0], [10.0]]))[:, 0]
    # 95 wins outright; 90 ties for 1st-2nd; 80 ties two others for 2nd-4th; 79 is 5th; 10 is last.
    assert np.allclose(result, [100, 55, 20 / 3, 0, 0])


def test_drafting_avoids_lineups_the_field_duplicates():
    cash_table = pd.DataFrame({"min_place": [1], "max_place": [1], "cash": [1000.0]})
    # Candidate 0 scores highest but the field plays it 50 times; candidate 1 is slightly lower and unique.
    field_scores = np.array([[100.0, 100.0], [60.0, 60.0]])
    candidates = np.array([[100.0, 100.0], [99.0, 99.0]])
    picks, profit = draft(candidates, field_scores, np.array([50, 10]), cash_table, entry_fee=1.0, picks=1)
    # Against 50 copies of 100 points nothing wins first, so the unique lineup isn't better either; both split or lose.
    assert picks in ([0], [1])
    field_scores = np.array([[100.0, 0.0], [60.0, 60.0]])
    candidates = np.array([[100.0, 0.0], [70.0, 70.0]])
    picks, profit = draft(candidates, field_scores, np.array([50, 10]), cash_table, entry_fee=1.0, picks=1)
    # Copying the chalk wins 1000/51 in one simulation; the unique 70 wins 1000 in the other.
    assert picks == [1] and np.isclose(profit.mean(), 1000 / 2 - 1)


def test_our_own_copies_split_with_each_other():
    cash = payout_by_place(pd.DataFrame({"min_place": [1], "max_place": [1], "cash": [90.0]}), 10)
    result = portfolio_cash(np.array([[50.0], [50.0]]), np.array([[10.0]]), np.array([5]), cash)
    assert np.allclose(result[:, 0], [45, 45])


def small_slate():
    players = pd.DataFrame({"player_id": range(8), "name": list("ABCDEFGH"), "team": ["X"] * 4 + ["Y"] * 4,
                            "position": ["QB", "WR", "RB", "K", "QB", "WR", "TE", "DST"], "salary": [9000, 8000, 7000, 4000] * 2,
                            "cpt_salary": [13500, 12000, 10500, 6000] * 2, "projection": [20, 15, 12, 7, 18, 14, 9, 6],
                            "favorite": [True] * 4 + [False] * 4})
    slate = Slate("test", players)
    slate.lineups = enumerate_lineups(players)
    slate.features = lineup_features(players, slate.lineups)
    return slate


def test_enumeration_respects_cap_and_both_teams():
    slate = small_slate()
    salary = slate.players.cpt_salary.to_numpy()[slate.lineups[:, 0]] + slate.players.salary.to_numpy()[slate.lineups[:, 1:]].sum(1)
    assert (salary <= 50_000).all() and len(slate.lineups) > 0
    teams = slate.players.team.to_numpy()[slate.lineups]
    assert all(len(set(row)) == 2 for row in teams)
    assert slate.features.shape == (len(slate.lineups), len(FEATURES))
    assert np.isclose(slate.features[:, 0].max(), 0)  # The best projected lineup has no gap.


def test_observed_counts_ignore_flex_order_and_outside_players():
    slate = small_slate()
    first = slate.lineups[0]
    entered = pd.DataFrame([[first[0], *first[1:][::-1], 3], [first[0], 99, *first[2:], 2]],
                           columns=["cpt", "flex1", "flex2", "flex3", "flex4", "flex5", "entries"])
    counts = observed_counts(slate, entered)
    assert counts[0] == 3 and counts.sum() == 3
    shares = ownership(slate, counts)
    assert np.isclose(shares.cpt.sum(), 1) and np.isclose(shares.flex.sum(), 5)
    assert duplicated_share(np.array([3, 1, 0])) == 0.75


def brute_force_greedy(candidates, field_scores, field_counts, payouts, entry_fee, picks, objective):
    from nflsim.dfs.contest import portfolio_omega
    cash = payout_by_place(payouts, int(field_counts.sum()) + picks)
    chosen = []
    for step in range(picks):
        values = []
        for index in range(len(candidates)):
            profit = portfolio_cash(candidates[chosen + [index]], field_scores, field_counts, cash).sum(axis=0) - entry_fee * (step + 1)
            values.append((portfolio_omega(profit[None, :])[0], profit.mean()) if objective == "omega" else (profit.mean(),))
        chosen.append(max(range(len(candidates)), key=lambda index: values[index]))
    return chosen


def test_draft_matches_whole_portfolio_greedy():
    rng = np.random.default_rng(3)
    payouts = pd.DataFrame({"min_place": [1, 2, 4], "max_place": [1, 3, 12], "cash": [200.0, 60.0, 15.0]})
    field_scores = np.round(rng.normal(100, 20, (40, 30)), 1)
    field_counts = rng.integers(1, 4, 40)
    # Coarse scores so our candidates tie each other and the field.
    candidates = np.round(rng.normal(105, 20, (25, 30)) / 5) * 5
    candidates[7] = candidates[3]
    for objective in ("omega", "cash"):
        chosen, profit = draft(candidates, field_scores, field_counts, payouts, entry_fee=10.0, picks=5, objective=objective)
        assert chosen == brute_force_greedy(candidates, field_scores, field_counts, payouts, 10.0, 5, objective)
        assert profit.shape == (5, 30)


def test_draft_contests_matches_whole_portfolio_greedy():
    from nflsim.dfs.contest import Contest, draft_contests, portfolio_omega
    rng = np.random.default_rng(11)
    sims = 30
    candidates = np.round(rng.normal(105, 20, (20, sims)) / 5) * 5
    contests = [
        Contest(np.round(rng.normal(100, 20, (40, sims)), 1), rng.integers(1, 4, 40),
                pd.DataFrame({"min_place": [1, 2, 4], "max_place": [1, 3, 12], "cash": [200.0, 60.0, 15.0]}), 10.0, 3, "big"),
        Contest(np.round(rng.normal(100, 20, (6, sims)), 1), np.ones(6, dtype=int),
                pd.DataFrame({"min_place": [1], "max_place": [4], "cash": [18.0]}), 10.0, 2, "double up"),
        Contest(np.round(rng.normal(110, 15, (10, sims)), 1), rng.integers(1, 3, 10),
                pd.DataFrame({"min_place": [1, 2], "max_place": [1, 2], "cash": [300.0, 100.0]}), 50.0, 2, "high stakes"),
    ]
    for objective in ("omega", "cash"):
        chosen, profits = draft_contests(candidates, contests, objective)
        # Reference: re-price the whole portfolio from scratch for every (contest, candidate) choice.
        picks = [[] for _ in contests]
        for _ in range(sum(contest.entries for contest in contests)):
            fees = sum(len(p) * contest.entry_fee for p, contest in zip(picks, contests))
            best = None
            for number, contest in enumerate(contests):
                if len(picks[number]) >= contest.entries:
                    continue
                for index in range(len(candidates)):
                    trial = [p + [index] if k == number else p for k, p in enumerate(picks)]
                    total = sum(portfolio_cash(candidates[p], c.field_scores, c.field_counts, c.cash_table()).sum(axis=0)
                                for p, c in zip(trial, contests) if p)
                    profit = total - fees - contest.entry_fee
                    value = (portfolio_omega(profit[None, :])[0], profit.mean()) if objective == "omega" else (profit.mean(),)
                    if best is None or value > best[0]:
                        best = (value, number, index)
            picks[best[1]].append(best[2])
        assert chosen == picks
        assert [p.shape for p in profits] == [(3, sims), (2, sims), (2, sims)]
