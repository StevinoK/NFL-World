import numpy as np
import pandas as pd
import pytest

from nflsim.features.builder import build_estimates
from nflsim.features.lineup_value import apply_lineup_value
from nflsim.features.player_value import (
    CROSS_SEASON_DECAY, PV_COLUMNS, build_player_value_tables, lineup_player_value, lineup_requests, pm_asof,
    prepare_pm_estimates, snap_share_history,
)
from nflsim.ridge.metrics import (
    RidgeConfig, RidgeTarget, build_design, build_game_context, build_target_rows, fit_coefficients,
)


def estimates(rows):
    raw = pd.DataFrame(rows, columns=["player_id", "season", "week", "n_on", "box_plus_minus"]).assign(estimator="box")
    return prepare_pm_estimates(raw, "QB")


def test_pm_asof_is_strictly_prior_with_cross_season_decay_and_replacement():
    table = estimates([("A", 2023, 18, 500, 0.2), ("A", 2024, 1, 30, 0.1), ("A", 2024, 2, 60, 0.3),
                       ("B", 2024, 1, 30, -0.1), ("C", 2024, 1, 30, 0.0), ("D", 2024, 1, 30, 0.05)])
    requests = pd.DataFrame([("QB", "A", 2024, 1), ("QB", "A", 2024, 2), ("QB", "A", 2024, 3), ("QB", "Z", 2024, 2)],
                            columns=["group", "player_id", "season", "week"])
    result = pm_asof(table, requests)
    assert np.isclose(result.pm.iloc[0], 0.2 * CROSS_SEASON_DECAY)
    assert np.isclose(result.pm.iloc[1], 0.1)
    assert np.isclose(result.pm.iloc[2], 0.3)
    week_one = table[table.week.eq(1) & table.season.eq(2024)].estimate
    assert result.pm_source.iloc[3] == "replacement"
    assert np.isclose(result.pm.iloc[3], week_one.quantile(0.25))


def test_traded_player_estimate_is_n_on_weighted():
    raw = pd.DataFrame([("A", 2024, 3, 30, 0.1, "AAA"), ("A", 2024, 3, 90, 0.3, "BBB")],
                       columns=["player_id", "season", "week", "n_on", "box_plus_minus", "team"]).assign(estimator="box")
    assert np.isclose(prepare_pm_estimates(raw, "WR").estimate.iloc[0], 0.25)


def league_inputs():
    schedule = pd.DataFrame({"game_id": ["2024_01_B_A", "2024_02_A_B", "2024_03_B_A"], "season": 2024, "week": [1, 2, 3],
                             "gameday": ["2024-09-08", "2024-09-15", "2024-09-22"], "home_team": ["A", "B", "A"],
                             "away_team": ["B", "A", "B"]})
    roster, snaps = [], []
    for game in schedule.itertuples():
        for team in ("A", "B"):
            for group, ids in (("QB", ["QB1", "QB2"]), ("WR", ["WR1", "WR2"]), ("RB", ["RB1"]), ("TE", ["TE1"]), ("DL", ["DL1", "DL2"])):
                for rank, base in enumerate(ids, start=1):
                    player = f"{team}_{base}"
                    roster.append({"game_id": game.game_id, "team": team, "player_id": player, "position": group,
                                   "status": "ACT", "report_status": None, "depth_team": rank})
                    pct = 1.0 if rank == 1 else 0.5 if group != "QB" else 0.0
                    snaps.append({"game_id": game.game_id, "gsis_id": player, "game_type": "REG",
                                  "offense_pct": 0.0 if group == "DL" else pct, "defense_pct": pct if group == "DL" else 0.0})
    return schedule, pd.DataFrame(roster), pd.DataFrame(snaps)


def test_team_values_weights_normalize_and_use_only_prior_games():
    schedule, roster, snaps = league_inputs()
    pm = []
    for group in ("QB", "WR", "RB", "TE", "DL"):
        for team in ("A", "B"):
            for week in (1, 2):
                for player in roster[roster.position.eq(group) & roster.team.eq(team)].player_id.unique():
                    pm.append({"group": group, "player_id": player, "season": 2024, "week": week, "n_on": 10,
                               "estimate": 0.1 * week if player.endswith("1") else -0.1})
    pm = pd.DataFrame(pm)
    values, components, _, _ = build_player_value_tables(roster, schedule, snaps, pm)
    weights = components.groupby(["game_id", "team", "group"]).weight.sum()
    assert np.allclose(weights, 1.0)
    week3 = components[components.game_id.eq("2024_03_B_A") & components.team.eq("A") & components.group.eq("WR")]
    # Trailing shares 1.0 and 0.5 normalize to 2/3 and 1/3.
    assert np.allclose(sorted(week3.weight), [1 / 3, 2 / 3])
    changed = pm.copy()
    changed.loc[changed.week.eq(2), "estimate"] += 5.0
    altered, *_ = build_player_value_tables(roster, schedule, snaps, changed)
    merged = values.merge(altered, on=["game_id", "team"], suffixes=("", "_new"))
    early = merged[merged.game_id.isin(["2024_01_B_A", "2024_02_A_B"])]
    assert np.allclose(early[PV_COLUMNS].to_numpy(), early[[f"{c}_new" for c in PV_COLUMNS]].to_numpy())
    late = merged[merged.game_id.eq("2024_03_B_A")]
    assert not np.allclose(late.pv_off_qb, late.pv_off_qb_new)


def test_lineup_value_drops_when_star_qb_is_replaced():
    schedule, roster, snaps = league_inputs()
    history = snap_share_history(snaps, schedule)
    pm = pd.DataFrame([{"group": "QB", "player_id": "A_QB1", "season": 2024, "week": 2, "n_on": 300, "estimate": 0.25},
                       {"group": "QB", "player_id": "A_QB2", "season": 2024, "week": 2, "n_on": 5, "estimate": -0.15}])
    starter = lineup_requests({"QB1": "A_QB1"}, [], 2024, 3, "2024-09-22")
    backup = lineup_requests({"QB1": "A_QB2"}, [], 2024, 3, "2024-09-22")
    assert lineup_player_value(starter, pm, history)[0]["pv_off_qb"] > lineup_player_value(backup, pm, history)[0]["pv_off_qb"]


TEAM_EPA = RidgeTarget("pass_epa", "normal", "team", "posteam", "epa", lambda pbp: pbp["pass"].eq(1), "epa_rate", "def_pass_epa_rate")


def synthetic_team_pbp(seasons=(2022, 2023), weeks=10, beta=2.0, seed=3):
    rng = np.random.default_rng(seed)
    teams = ["T1", "T2", "T3", "T4"]
    plays, values = [], []
    for season in seasons:
        for week in range(1, weeks + 1):
            order = rng.permutation(teams)
            for home, away in ((order[0], order[1]), (order[2], order[3])):
                game_id = f"{season}_{week:02d}_{away}_{home}"
                for team in (home, away):
                    qb = rng.normal(0.0, 0.1)
                    values.append({"game_id": game_id, "team": team, "pv_off_qb": qb, "pv_off_wr": 0.0,
                                   "pv_off_rb": 0.0, "pv_off_te": 0.0, "pv_def_dl": 0.0})
                for offense, defense in ((home, away), (away, home)):
                    qb = values[-2 if offense == home else -1]["pv_off_qb"]
                    for _ in range(40):
                        plays.append({"game_id": game_id, "game_date": pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=7 * week),
                                      "home_team": home, "away_team": away, "stadium_id": f"{home}_S", "location": "Home",
                                      "posteam": offense, "defteam": defense, "pass": 1,
                                      "epa": beta * qb + rng.normal(0, 0.05)})
    return pd.DataFrame(plays), pd.DataFrame(values)


def test_ridge_recovers_player_value_coefficient():
    pbp, values = synthetic_team_pbp()
    rows = build_target_rows(pbp, TEAM_EPA, build_game_context(pbp), player_value=values)
    rows = rows.assign(subject=rows.posteam)
    design = build_design(rows, "team", include_qb_offense=True)
    assert "player_value" in set(design.blocks)
    assert "offense_qb" not in set(design.blocks)
    config = RidgeConfig(within_decay=1.0, across_decay=1.0, season_carryover=1.0, lookback_seasons=2, lambda_player_value=0.01)
    coefficients = fit_coefficients(rows, design, config, 2024, 1).set_index(["block", "label"]).coefficient
    assert abs(coefficients.loc["player_value", "pv_off_qb"] - 2.0) < 0.4


def test_published_team_mu_is_invariant_to_current_qb_value():
    pbp, values = synthetic_team_pbp()
    schedule = pbp.drop_duplicates("game_id")[["game_id", "game_date", "home_team", "away_team", "stadium_id", "location"]]
    roster = pd.DataFrame([{"game_id": game, "team": team, "player_id": f"{team}_QB", "position": "QB"}
                           for game in schedule.game_id for team in game.split("_")[2:]])
    params = {"epa_rate": {"lambda_player_value": 0.01}, "def_pass_epa_rate": {"lambda_player_value": 0.01}}
    _, team, _, diagnostics = build_estimates(pbp, schedule, roster, params, [TEAM_EPA], predict_seasons=[2023], player_value=values)
    assert diagnostics.reason.eq("player_value_coefficient").any()
    game = team[team.source.eq("epa_rate")].iloc[-1]
    lowered = values.copy()
    key = lowered.game_id.eq(game.game_id) & lowered.team.eq(game.team)
    lowered.loc[key, "pv_off_qb"] -= 0.3
    _, team_low, _, _ = build_estimates(pbp, schedule, roster, params, [TEAM_EPA], predict_seasons=[2023], player_value=lowered)
    low = team_low[team_low.source.eq("epa_rate") & team_low.game_id.eq(game.game_id) & team_low.team.eq(game.team)].average.iloc[0]
    assert np.isclose(low, game.average)


class Tables:
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def betas(self, season, week):
        return self.coefficients


def test_apply_lineup_value_shifts_folded_metrics_and_opponent_aliases():
    coefficients = pd.DataFrame([{"season": 2025, "week": 3, "metric": "epa_rate", "side": "subject", "term": "pv_off_qb", "beta": 2.0},
                                 {"season": 2025, "week": 3, "metric": "def_pass_epa_rate", "side": "defense", "term": "pv_def_dl", "beta": -1.0}])
    base = {"epa_rate": 0.1, "def_pass_epa_rate": 0.0, "opp_pass_epa_rate": 0.0, "opp_def_pass_epa_rate": 0.0,
            **{column: 0.0 for column in PV_COLUMNS}}
    features = pd.DataFrame([base, base], index=["BAL", "PIT"])
    new = {"BAL": {**{column: 0.0 for column in PV_COLUMNS}, "pv_off_qb": -0.2},
           "PIT": {**{column: 0.0 for column in PV_COLUMNS}, "pv_def_dl": 0.1}}
    result = apply_lineup_value(Tables(coefficients), features, {}, new, 2025, 3)
    assert np.isclose(result.loc["BAL", "epa_rate"], 0.1 - 0.4)
    assert np.isclose(result.loc["PIT", "def_pass_epa_rate"], -0.1)
    assert np.isclose(result.loc["PIT", "opp_pass_epa_rate"], result.loc["BAL", "epa_rate"])
    assert np.isclose(result.loc["BAL", "pv_off_qb"], -0.2)


def test_team_specific_qb_offense_coefficient_adjusts_only_its_team():
    coefficients = pd.DataFrame([
        {"season": 2025, "week": 3, "metric": "epa_rate", "side": "subject", "term": "offense_qb:BAL", "beta": 2.0},
        {"season": 2025, "week": 3, "metric": "epa_rate", "side": "subject", "term": "offense_qb:PIT", "beta": -1.0},
    ])
    base = {"epa_rate": 0.1, **{column: 0.0 for column in PV_COLUMNS}}
    features = pd.DataFrame([base, base], index=["BAL", "PIT"])
    new = {team: {**{column: 0.0 for column in PV_COLUMNS}, "pv_off_qb": -0.2} for team in features.index}
    old = {team: {**{column: 0.0 for column in PV_COLUMNS}, "pv_off_qb": 0.2} for team in features.index}
    result = apply_lineup_value(Tables(coefficients), features, {}, new, 2025, 3, old_values=old,
                                update_aliases=False)
    assert np.isclose(result.loc["BAL", "epa_rate"], -0.7)
    assert np.isclose(result.loc["PIT", "epa_rate"], 0.5)
