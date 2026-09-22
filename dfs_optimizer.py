"""Core logic for the DK Showdown DFS lineup optimizer.

Pipeline: load prices/simulations -> merge -> generate valid 6-man showdown lineups
(1 CPT + 5 FLEX) -> score each lineup against real simulations and an independent
"field" model -> run a self-play agent equilibrium to estimate field ownership ->
draft our own entries against that field.

No functions here read files/hit the network at import time; callers (the Streamlit
dashboard or the CLI wrapper in `Testing dfs.py`) supply paths and parameters.
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# --------------------------------------------------------------------------- loading


def load_prices(path: str) -> pd.DataFrame:
    """Load a DK salaries CSV and derive the `short_name` key used to join with sims."""
    prices = pd.read_csv(path)

    dst_mask = prices["Position"] == "DST"
    prices.loc[dst_mask, "Name"] = prices.loc[dst_mask, "Name"].str.split().str[0] + " Defense"

    prices["short_name"] = (
        prices["Name"].str.split().str[0].str[0:3] + "." + prices["Name"].str.split().str[1]
    )
    prices.loc[prices["Position"] == "DST", "short_name"] = (
        prices.loc[prices["Position"] == "DST", "TeamAbbrev"].str.split().str[0] + ".Defense"
    )
    prices.loc[prices["Position"] == "K", "short_name"] = (
        prices.loc[prices["Position"] == "K", "TeamAbbrev"].str.split().str[0] + ".Kicker"
    )
    prices["captain"] = prices["Roster Position"] == "CPT"
    return prices


_LONG_SIM_COLUMNS = {"player_name", "fantasy_points", "simulation"}


def load_simulations(path: str) -> pd.DataFrame:
    """Load a simulations CSV, accepting either a wide sheet (one row per joint slate
    simulation, one column per player) or the game_sim_dashboard's long `_players.csv`
    (one row per player per simulation per game), which gets pivoted into the wide shape."""
    raw = pd.read_csv(path)
    if _LONG_SIM_COLUMNS.issubset(raw.columns):
        return pivot_long_simulations(raw)
    return raw.set_index(raw.columns[0])


def pivot_long_simulations(long_simulations: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long `_players.csv` into one row per simulation, one column per player.
    Simulation numbers are assumed to line up across games in the same file (i.e. sim #k
    of every game in the file is one joint slate realization) -- matches how the game sim
    dashboard's "Append to existing files" workflow builds a multi-game slate file."""
    pivoted = long_simulations.pivot_table(
        index="simulation", columns="player_name", values="fantasy_points", aggfunc="mean"
    )
    pivoted.columns.name = None
    return pivoted.fillna(0.0)


def align_simulation_columns(simulations: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Rename simulation columns to the `short_name` convention, keeping mean/cov aligned
    positionally (they're derived from the same column order as the original names)."""
    # Drop any stray non-numeric columns (e.g. a leftover id/label column) before
    # aggregating, since mean()/cov() can't reduce a mix of numbers and strings.
    simulations = simulations.select_dtypes(include="number")
    means = simulations.mean(axis=0)
    covs = simulations.cov()

    short_names = np.where(
        means.index.str.contains("Defense|Kicker", case=False, regex=True),
        means.index.str.split().str[0] + "." + means.index.str.split().str[1],
        means.index.str.split().str[0].str[0:3] + "." + means.index.str.split().str[1],
    )
    means.index = short_names
    covs.index = short_names
    covs.columns = short_names

    simulations.columns = short_names
    return simulations, means, covs


def scale_simulations(simulations: pd.DataFrame, scale_factors: pd.Series) -> pd.DataFrame:
    """Multiply every simulated score of a player by that player's factor, which shifts the
    mean while preserving the distribution's shape (a 0.75 factor turns a 10 mean / 20 max
    into 7.5 / 15). Players missing from `scale_factors` are left untouched."""
    factors = pd.Series(scale_factors, dtype="float64").reindex(simulations.columns).fillna(1.0).to_numpy()
    return pd.DataFrame(
        simulations.to_numpy() * factors, index=simulations.index, columns=simulations.columns
    )


def load_entries(path: str) -> pd.DataFrame:
    if hasattr(path, "seek"):
        path.seek(0)
    export_df = pd.read_csv(path, usecols=range(10))
    return clean_export_ids(export_df)


def clean_export_ids(export_df: pd.DataFrame) -> pd.DataFrame:
    export_df = export_df.dropna(subset=["Entry ID"]).copy()
    export_df["Entry ID"] = export_df["Entry ID"].astype(float).astype("int64").astype(str)
    export_df["Contest ID"] = export_df["Contest ID"].astype(float).astype("int64").astype(str)
    return export_df


_DRAFTKINGS_CONTEST_URL = "https://api.draftkings.com/contests/v1/contests/{contest_id}?format=json"
_CASH_AMOUNT_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d{2})?")
_REQUEST_TIMEOUT = 15


def _parse_cash_amount(description: Any) -> float:
    match = _CASH_AMOUNT_RE.search(str(description))
    if not match:
        return float("nan")
    return float(match.group(0).replace(",", ""))


def parse_draftkings_contest_detail(payload: dict, contest_id: str) -> Tuple[dict, pd.DataFrame]:
    """Extract contest size, entry fee, and payout tiers from DK's contest endpoint."""
    detail = payload["contestDetail"]
    entry_fee = float(detail.get("entryFee", 0.0) or 0.0)
    rows = []
    for tier in detail.get("payoutSummary", []):
        descriptions = tier.get("tierPayoutDescriptions", {})
        cash = _parse_cash_amount(descriptions.get("Cash", ""))
        rows.append(
            {
                "min_place": int(tier.get("minPosition", 0) or 0),
                "max_place": int(tier.get("maxPosition", 0) or 0),
                "cash": cash,
                "profit": cash - entry_fee if np.isfinite(cash) else float("nan"),
            }
        )
    contest = {
        "contest_id": str(contest_id),
        "name": detail.get("name") or detail.get("contestName") or "",
        "entry_fee": entry_fee,
        "maximum_entries": int(detail.get("maximumEntries", 0) or 0),
        "entries": detail.get("entries") or detail.get("entryCount") or detail.get("currentEntries"),
    }
    return contest, pd.DataFrame(rows)


def fetch_draftkings_contest_detail(contest_id: str) -> Tuple[dict, pd.DataFrame]:
    response = requests.get(_DRAFTKINGS_CONTEST_URL.format(contest_id=contest_id), timeout=_REQUEST_TIMEOUT)
    response.raise_for_status()
    return parse_draftkings_contest_detail(response.json(), str(contest_id))


# ------------------------------------------------------------------- kicker ambiguity


def find_ambiguous_kickers(prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Teams with more than one distinct kicker *player* priced (e.g. a practice-squad
    elevation still listed alongside the real starter). Each player normally has multiple
    price-point rows (CPT + FLEX share the same Name but a different Name + ID), so
    ambiguity is judged by distinct `Name`, not row count or `Name + ID`."""
    kickers = prices[prices["Position"] == "K"]
    ambiguous = {}
    for team, group in kickers.groupby("TeamAbbrev"):
        if group["Name"].nunique() > 1:
            ambiguous[team] = group
    return ambiguous


def apply_kicker_selection(prices: pd.DataFrame, keep_names: Dict[str, str]) -> pd.DataFrame:
    """Drop every "K" price-point row for a team except those belonging to the player
    the caller picked as starter (keeps all of that player's price points, e.g. CPT+FLEX)."""
    drop_mask = pd.Series(False, index=prices.index)
    kicker_mask = prices["Position"] == "K"
    for team, keep_name in keep_names.items():
        team_mask = kicker_mask & (prices["TeamAbbrev"] == team) & (prices["Name"] != keep_name)
        drop_mask |= team_mask
    return prices.loc[~drop_mask].reset_index(drop=True)


# ------------------------------------------------------------------------- merging


def merge_prices_and_sims(prices: pd.DataFrame, means: pd.Series, price_floor: int = 600) -> pd.DataFrame:
    merged = prices.merge(means.rename("proj_mean"), left_on="short_name", right_index=True, how="inner")
    return merged[merged["Salary"] > price_floor].reset_index(drop=True)


def build_player_map(prices: pd.DataFrame) -> Dict[str, dict]:
    return prices.set_index("Name + ID")[["short_name", "Roster Position"]].to_dict(orient="index")


def summarize_player_projections(
    merged_prices: pd.DataFrame,
    simulations: pd.DataFrame,
    sleeper: Optional[pd.Series] = None,
    espn: Optional[pd.Series] = None,
    blended: Optional[pd.Series] = None,
    extra_sources: Optional[Dict[str, pd.Series]] = None,
) -> pd.DataFrame:
    """One row per priced player (CPT/FLEX price points collapsed to their short_name):
    our simulation distribution (mean/median/std/p10/p90) plus whichever public
    projection sources were used to build the field model, for a final sanity check
    before exporting."""
    players = merged_prices.drop_duplicates("short_name").set_index("short_name")
    sim_stats = pd.DataFrame(
        {
            "sim_mean": simulations.mean(axis=0),
            "sim_median": simulations.median(axis=0),
            "sim_std": simulations.std(axis=0),
            "sim_p10": simulations.quantile(0.10, axis=0),
            "sim_p90": simulations.quantile(0.90, axis=0),
        }
    )
    summary = players[["Name", "Position", "TeamAbbrev", "Salary"]].join(sim_stats, how="left")
    labelled = [("sleeper_pts", sleeper), ("espn_pts", espn)]
    labelled += [(label, series) for label, series in (extra_sources or {}).items()]
    labelled.append(("field_model_pts", blended))
    for label, series in labelled:
        if series is not None:
            summary[label] = series
    return summary.sort_values("sim_mean", ascending=False)


# ------------------------------------------------------------------------ lineups


def generate_valid_lineups(
    prices: pd.DataFrame,
    salary_min: int = 48000,
    salary_max: int = 50000,
    min_teams: int = 2,
) -> pd.DataFrame:
    """Enumerate valid CPT + 5xFLEX showdown lineups within the salary cap.

    Fixes the captain first, then only combines the remaining FLEX-eligible rows for the
    other 5 slots (excluding that captain's own duplicate row) -- this avoids the
    O(C(n, 6)) dense combination matrix the original script built (which allocated a
    ~790MB uint8 array for a ~50-row price pool).
    """
    prices = prices.reset_index(drop=True)
    salary = prices["Salary"].to_numpy()
    team = prices["TeamAbbrev"].to_numpy()
    short_name = prices["short_name"].to_numpy()
    name_id = prices["Name + ID"].to_numpy()

    flex_mask = (prices["Roster Position"] != "CPT").to_numpy()
    captain_positions = np.flatnonzero((prices["Roster Position"] == "CPT").to_numpy())

    valid_lineups: List[dict] = []
    for cap_idx in captain_positions:
        eligible_flex = np.flatnonzero(flex_mask & (short_name != short_name[cap_idx]))
        m = len(eligible_flex)
        if m < 5:
            continue

        combos = np.fromiter(
            itertools.chain.from_iterable(itertools.combinations(range(m), 5)),
            dtype=np.int64,
        ).reshape(-1, 5)
        combo_positions = eligible_flex[combos]

        combo_salary = salary[combo_positions].sum(axis=1) + salary[cap_idx]
        within_cap = (combo_salary >= salary_min) & (combo_salary <= salary_max)
        if not within_cap.any():
            continue

        for positions, total_salary in zip(combo_positions[within_cap], combo_salary[within_cap]):
            lineup_positions = np.concatenate(([cap_idx], positions))
            names = short_name[lineup_positions]
            if len(set(names)) != len(names):
                continue
            teams = set(team[lineup_positions])
            if len(teams) < min_teams:
                continue
            valid_lineups.append(
                {
                    "Players": list(name_id[lineup_positions]),
                    "Teams": list(teams),
                    "Salary": int(total_salary),
                }
            )

    return pd.DataFrame(valid_lineups)


def compute_lineup_score_distributions(
    lineups_df: pd.DataFrame,
    simulations: pd.DataFrame,
    player_map: Dict[str, dict],
    captain_multiplier: float = 1.5,
    chunk_size: int = 50_000,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Sum per-player simulated scores (with the CPT multiplier) into a per-lineup
    distribution. Call once with the real `simulations` and again with a field/"market"
    simulation set to get comparable score arrays for each.

    Scores are gathered slot-by-slot in float32 chunks rather than one Python loop per
    lineup, which is orders of magnitude faster on the large showdown lineup pools."""
    lineup_records = pd.DataFrame(
        {
            "LineupIndex": lineups_df.index,
            "Players": lineups_df["Players"].to_numpy(),
            "Salary": lineups_df["Salary"].to_numpy(),
        }
    ).reset_index(drop=True)

    players_lists = lineups_df["Players"].tolist()
    num_lineups = len(players_lists)
    num_sims = len(simulations)
    if num_lineups == 0:
        return lineup_records, np.zeros((0, num_sims), dtype=np.float32)

    slots = len(players_lists[0])
    col_pos: Dict[str, int] = {}
    for position, name in enumerate(simulations.columns):
        col_pos.setdefault(name, position)

    lookup = {
        player: (
            col_pos[info["short_name"]],
            captain_multiplier if info["Roster Position"] == "CPT" else 1.0,
        )
        for player, info in player_map.items()
        if info["short_name"] in col_pos
    }
    flat = np.array([lookup[p] for players in players_lists for p in players], dtype=np.float64)
    column_index = flat[:, 0].astype(np.int64).reshape(num_lineups, slots)
    multipliers = flat[:, 1].astype(np.float32).reshape(num_lineups, slots)

    player_scores = np.ascontiguousarray(simulations.to_numpy(dtype=np.float32).T)
    lu_scores_array = np.empty((num_lineups, num_sims), dtype=np.float32)
    for start in range(0, num_lineups, chunk_size):
        stop = min(start + chunk_size, num_lineups)
        block = lu_scores_array[start:stop]
        block[:] = 0.0
        for slot in range(slots):
            block += player_scores[column_index[start:stop, slot], :] * multipliers[start:stop, slot, None]

    return lineup_records, lu_scores_array


# -------------------------------------------------------------------- field model


def build_field_mean(blended_projections: pd.Series, fallback_means: pd.Series) -> pd.Series:
    """Fill any player missing from the blended public projections with our own
    simulation mean, so the field model always has a complete mean vector."""
    return blended_projections.reindex(fallback_means.index).fillna(fallback_means)


def build_field_simulations(covs: pd.DataFrame, mean_vector: pd.Series, n_rows: int) -> pd.DataFrame:
    """Draw the field/opponent score model: an independent mean (public projections when
    available) combined with our own simulation covariance (neither public source exposes
    player covariance)."""
    aligned_mean = mean_vector.reindex(covs.index)
    values = np.random.multivariate_normal(aligned_mean.to_numpy(), covs.to_numpy(), n_rows)
    return pd.DataFrame(values, columns=covs.columns)


def _greedy_best_response(
    lineup_scores: np.ndarray,
    opponent_max: np.ndarray,
    num_picks: int,
    blocked: Optional[np.ndarray] = None,
) -> Tuple[List[int], float]:
    """Greedily pick `num_picks` lineups maximizing the share of simulations where the
    portfolio's best score beats `opponent_max`.

    Only the simulations not already won by earlier picks are re-scanned, so each
    successive pick is cheaper than the last (a portfolio's win set only grows)."""
    num_lineups, num_sims = lineup_scores.shape
    unwon = np.arange(num_sims)
    won = 0
    picks: List[int] = []
    block = blocked if (blocked is not None and blocked.any() and not blocked.all()) else None

    for _ in range(num_picks):
        if unwon.size:
            opponent = np.ascontiguousarray(opponent_max[unwon])
            wins = np.empty(num_lineups, dtype=np.int64)
            # Chunk the row scan so the gathered comparison block stays cache-sized.
            rows_per_chunk = max(1, 4_000_000 // unwon.size)
            for start in range(0, num_lineups, rows_per_chunk):
                stop = min(start + rows_per_chunk, num_lineups)
                candidates = lineup_scores[start:stop]
                if unwon.size != num_sims:
                    candidates = candidates[:, unwon]
                wins[start:stop] = np.count_nonzero(candidates > opponent, axis=1)
            wins += won
        else:
            wins = np.zeros(num_lineups, dtype=np.int64)
        if block is not None:
            wins = np.where(block, -1, wins)
        chosen = int(np.argmax(wins))
        picks.append(chosen)
        if unwon.size:
            newly_won = lineup_scores[chosen, unwon] > opponent_max[unwon]
            won += int(newly_won.sum())
            unwon = unwon[~newly_won]

    return picks, won / num_sims


def _build_payout_by_place(payouts: pd.DataFrame, max_place: int) -> np.ndarray:
    payout_by_place = np.zeros(max_place + 2, dtype=np.float32)
    for row in payouts.itertuples(index=False):
        min_place = max(1, int(row.min_place))
        max_tier_place = min(max_place, int(row.max_place))
        if max_tier_place >= min_place and np.isfinite(row.cash):
            payout_by_place[min_place : max_tier_place + 1] = float(row.cash)
    return payout_by_place


def _places_against_field(
    scores: np.ndarray,
    sorted_field_scores: np.ndarray,
    field_size: int,
    max_place: int,
) -> np.ndarray:
    scores = np.atleast_2d(scores)
    field_entries = sorted_field_scores.shape[0]
    places = np.empty(scores.shape, dtype=np.int32)
    scale = field_size / max(1, field_entries)
    for sim_idx in range(scores.shape[1]):
        greater = field_entries - np.searchsorted(sorted_field_scores[:, sim_idx], scores[:, sim_idx], side="right")
        places[:, sim_idx] = np.minimum(max_place, np.floor(greater * scale).astype(np.int32) + 1)
    return places


def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    upside = np.maximum(returns - threshold, 0.0).sum()
    downside = np.maximum(threshold - returns, 0.0).sum()
    if downside == 0.0:
        return float("inf") if upside > 0.0 else 0.0
    return float(upside / downside)


def _greedy_omega_response(
    lineup_scores: np.ndarray,
    field_indices: Iterable[int],
    payout_tiers: pd.DataFrame,
    entry_fee: float,
    num_picks: int,
    field_size: int,
    pool_size: Optional[int] = 20_000,
    threshold: float = 0.0,
    chunk_size: int = 1_000,
) -> Tuple[List[int], float, np.ndarray]:
    """Greedily draft a portfolio by maximizing Omega ratio of simulated contest profit."""
    num_lineups, num_sims = lineup_scores.shape
    if num_lineups == 0 or num_picks <= 0:
        return [], 0.0, np.zeros(num_sims, dtype=np.float32)

    field_indices_array = np.asarray(list(field_indices), dtype=np.int64)
    if field_indices_array.size == 0:
        return _greedy_best_response(lineup_scores, np.full(num_sims, -np.inf), num_picks)[0], 0.0, np.zeros(num_sims)

    max_place = max(1, int(field_size))
    opponent_field_size = max(1, max_place - num_picks)
    payout_by_place = _build_payout_by_place(payout_tiers, max_place)
    sorted_field_scores = np.sort(lineup_scores[field_indices_array], axis=0)

    candidate_pool = select_candidate_pool(lineup_scores, pool_size)
    selected: List[int] = []
    selected_scores = np.zeros((0, num_sims), dtype=np.float32)
    selected_places = np.zeros((0, num_sims), dtype=np.int32)
    selected_cash = np.zeros((0, num_sims), dtype=np.float32)
    total_cash = np.zeros(num_sims, dtype=np.float32)
    total_entry_fees = float(entry_fee) * num_picks

    for _ in range(num_picks):
        blocked = set(selected)
        best_idx = None
        best_score = -np.inf
        best_expected_profit = -np.inf
        best_places = None
        best_cash = None

        for start in range(0, len(candidate_pool), chunk_size):
            chunk_indices = candidate_pool[start : start + chunk_size]
            if blocked:
                keep = np.array([idx not in blocked for idx in chunk_indices], dtype=bool)
                chunk_indices = chunk_indices[keep]
                if len(chunk_indices) == 0:
                    continue

            chunk_scores = np.ascontiguousarray(lineup_scores[chunk_indices], dtype=np.float32)
            field_places = _places_against_field(chunk_scores, sorted_field_scores, opponent_field_size, max_place)
            own_greater = np.zeros_like(field_places)
            for score in selected_scores:
                own_greater += score[None, :] > chunk_scores
            candidate_places = np.minimum(max_place, field_places + own_greater)
            candidate_cash = payout_by_place[candidate_places]

            portfolio_cash = total_cash[None, :] + candidate_cash
            for place, cash, score in zip(selected_places, selected_cash, selected_scores):
                demoted_places = np.minimum(max_place, place + 1)
                demotion = payout_by_place[demoted_places] - cash
                portfolio_cash += np.where(chunk_scores > score[None, :], demotion[None, :], 0.0)

            returns = portfolio_cash - total_entry_fees
            upside = np.maximum(returns - threshold, 0.0).sum(axis=1)
            downside = np.maximum(threshold - returns, 0.0).sum(axis=1)
            scores = np.divide(upside, downside, out=np.full_like(upside, np.inf), where=downside > 0.0)
            expected_profit = returns.mean(axis=1)
            scores = np.where((downside == 0.0) & (upside == 0.0), 0.0, scores)

            order = np.lexsort((expected_profit, scores))
            chunk_best_pos = int(order[-1])
            chunk_best_score = float(scores[chunk_best_pos])
            chunk_best_profit = float(expected_profit[chunk_best_pos])
            if (chunk_best_score > best_score) or (
                chunk_best_score == best_score and chunk_best_profit > best_expected_profit
            ):
                best_idx = int(chunk_indices[chunk_best_pos])
                best_score = chunk_best_score
                best_expected_profit = chunk_best_profit
                best_places = candidate_places[chunk_best_pos]
                best_cash = candidate_cash[chunk_best_pos]

        if best_idx is None or best_places is None or best_cash is None:
            break

        chosen_scores = lineup_scores[best_idx].astype(np.float32)
        for row_idx in range(len(selected)):
            demoted = chosen_scores > selected_scores[row_idx]
            selected_places[row_idx, demoted] = np.minimum(max_place, selected_places[row_idx, demoted] + 1)
            selected_cash[row_idx] = payout_by_place[selected_places[row_idx]]
        selected.append(best_idx)
        selected_scores = np.vstack([selected_scores, chosen_scores[None, :]])
        selected_places = np.vstack([selected_places, best_places[None, :]])
        selected_cash = np.vstack([selected_cash, best_cash[None, :]])
        total_cash = selected_cash.sum(axis=0)

    returns = total_cash - total_entry_fees
    return selected, omega_ratio(returns, threshold), returns


def select_candidate_pool(scores: np.ndarray, pool_size: Optional[int], upside_z: float = 1.0) -> np.ndarray:
    """Shortlist the lineups the field could plausibly roster, ranked by mean + z*std of
    their simulated scores. Greedy best-response never picks a lineup outside the top tier,
    so pruning the pool cuts the agent loop's cost roughly linearly with negligible drift."""
    if pool_size is None or pool_size >= len(scores):
        return np.arange(len(scores))
    upside = scores.mean(axis=1) + upside_z * scores.std(axis=1)
    top = np.argpartition(-upside, pool_size - 1)[:pool_size]
    return np.sort(top)


class Agent:
    """One portfolio of `team_size` lineups in the simulated GPP field. Iteratively swaps
    lineups via `try_improve` to raise its win probability against the rest of the field."""

    def __init__(self, name, lineup_indices: Iterable[int], num_sims: int, subsample: float = 1.0):
        self.name = name
        self.lineup_indices = list(lineup_indices)
        self.win_prob = 0.0
        self.subsample = (
            None if subsample >= 1.0 else np.random.choice(num_sims, int(num_sims * subsample), replace=False)
        )

    def try_improve(self, lineup_scores: np.ndarray, opponent_indices: Iterable[int], use_all: bool = False) -> None:
        opponent_indices = np.asarray(list(opponent_indices), dtype=np.int64)
        if self.subsample is not None:
            lineup_scores = lineup_scores[:, self.subsample]

        opponent_max = lineup_scores[opponent_indices, :].max(axis=0)
        blocked = None
        if not use_all:
            blocked = np.zeros(lineup_scores.shape[0], dtype=bool)
            blocked[opponent_indices] = True

        self.lineup_indices, self.win_prob = _greedy_best_response(
            lineup_scores, opponent_max, len(self.lineup_indices), blocked
        )


def run_field_simulation(
    lu_scores_array: np.ndarray,
    market_scores_array: np.ndarray,
    lineup_sims_df: pd.DataFrame,
    num_agents: int = 500,
    team_size: int = 5,
    iterations: int = 3,
    subsample: float = 0.5,
    pool_size: Optional[int] = 20_000,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Tuple[List[Agent], pd.Series, List[int]]:
    """Run agent self-play to equilibrium and return the resulting field ownership.

    Speed-ups over the naive loop: the candidate pool is shortlisted once (`pool_size`),
    the simulation subsample is drawn once per iteration and materialized as one compact
    float32 matrix instead of re-slicing the full score array for every agent, and each
    agent's opponent max comes from a running per-agent max matrix (num_agents x sims)
    rather than re-gathering every other agent's lineups.
    """
    pool = select_candidate_pool(market_scores_array, pool_size)
    pool_scores = np.ascontiguousarray(market_scores_array[pool], dtype=np.float32)
    num_pool, num_sims = pool_scores.shape
    team_size = min(int(team_size), num_pool)
    sub_size = max(1, min(num_sims, int(num_sims * subsample)))

    agent_lineups = np.array(
        [np.random.choice(num_pool, team_size, replace=False) for _ in range(num_agents)], dtype=np.int64
    )
    agent_max = pool_scores[agent_lineups].max(axis=1)
    agent_win_probs = np.zeros(num_agents, dtype=np.float64)
    owner_counts = np.zeros(num_pool, dtype=np.int32)
    np.add.at(owner_counts, agent_lineups.ravel(), 1)

    def ownership_counts(indices: np.ndarray) -> pd.Series:
        players = [p for lst in lineup_sims_df.loc[pool[indices], "Players"] for p in lst]
        return pd.Series(players).value_counts() / num_agents / team_size

    counts = ownership_counts(agent_lineups.ravel())

    for iteration in range(iterations):
        sub_cols = (
            np.arange(num_sims) if sub_size == num_sims else np.random.choice(num_sims, sub_size, replace=False)
        )
        sub_scores = np.ascontiguousarray(pool_scores[:, sub_cols])
        sub_agent_max = np.ascontiguousarray(agent_max[:, sub_cols])

        for agent_idx in range(num_agents):
            own = agent_lineups[agent_idx]
            np.subtract.at(owner_counts, own, 1)
            sub_agent_max[agent_idx] = -np.inf
            opponent_max = sub_agent_max.max(axis=0)

            picks, win_prob = _greedy_best_response(sub_scores, opponent_max, team_size, owner_counts > 0)
            picks_array = np.asarray(picks, dtype=np.int64)

            agent_lineups[agent_idx] = picks_array
            np.add.at(owner_counts, picks_array, 1)
            sub_agent_max[agent_idx] = sub_scores[picks_array].max(axis=0)
            agent_max[agent_idx] = pool_scores[picks_array].max(axis=0)
            agent_win_probs[agent_idx] = win_prob

        new_counts = ownership_counts(agent_lineups.ravel())
        drift = float(np.sqrt(np.nanmean((counts.reindex(new_counts.index).fillna(0) - new_counts) ** 2)))
        counts = new_counts
        if progress_callback:
            progress_callback(iteration + 1, iterations, drift)

    agents = []
    for agent_idx in range(num_agents):
        agent = Agent(agent_idx, pool[agent_lineups[agent_idx]], num_sims, subsample=1.0)
        agent.win_prob = float(agent_win_probs[agent_idx])
        agents.append(agent)
    all_indices = [int(i) for i in pool[agent_lineups.ravel()]]

    return agents, counts, all_indices


def draft_our_entries(
    export_df: pd.DataFrame,
    lu_scores_array: np.ndarray,
    field_indices: Iterable[int],
    lineup_sims_df: pd.DataFrame,
    player_map: Dict[str, dict],
    payout_tiers: Optional[pd.DataFrame] = None,
    entry_fee: Optional[float] = None,
    field_size: Optional[int] = None,
    objective: str = "winner",
    omega_pool_size: Optional[int] = 20_000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Draft our own `len(export_df)` entries against the field and fill the DK entries
    template's CPT/FLEX columns (looked up by name, not hardcoded column positions)."""
    if objective == "omega" and payout_tiers is not None and entry_fee is not None and field_size is not None:
        lineup_indices, _, _ = _greedy_omega_response(
            lu_scores_array,
            field_indices,
            payout_tiers,
            float(entry_fee),
            len(export_df),
            int(field_size),
            pool_size=omega_pool_size,
        )
    else:
        our_team = Agent(
            "Us", np.random.choice(len(lu_scores_array), len(export_df), replace=False), lu_scores_array.shape[1]
        )
        our_team.try_improve(lu_scores_array, field_indices, use_all=True)
        lineup_indices = our_team.lineup_indices

    selected_lineups = [
        {"Players": lineup_sims_df.iloc[idx]["Players"], "Salary": lineup_sims_df.iloc[idx]["Salary"]}
        for idx in lineup_indices
    ]
    drafted_lineups_df = pd.DataFrame(selected_lineups)
    our_counts = pd.Series([p for lineup in selected_lineups for p in lineup["Players"]]).value_counts() / len(export_df)

    export_df = export_df.copy()
    cpt_col = next(c for c in export_df.columns if c.startswith("CPT"))
    flex_cols = [c for c in export_df.columns if c.startswith("FLEX")]
    export_df[[cpt_col, *flex_cols]] = export_df[[cpt_col, *flex_cols]].astype(object)
    for i, lineup in drafted_lineups_df.iterrows():
        flex_iter = iter(flex_cols)
        for player in lineup["Players"]:
            if player_map[player]["Roster Position"] == "CPT":
                export_df.loc[i, cpt_col] = player
            else:
                export_df.loc[i, next(flex_iter)] = player

    export_df = export_df.rename(columns={c: "FLEX" for c in flex_cols})
    export_df = export_df.rename(columns={cpt_col: "CPT"})
    return export_df, drafted_lineups_df, our_counts
