from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from nflsim.features.bayesian_history import MetricSpec
from nflsim.features.bayesian_history import FutureContext, collect_team_metric_specs
from nflsim.paths import REPO, digest
from nflsim.features.builder import wide_tables
from nflsim.features.retained import (retained_from_specs, regenerate_retained_features,
    normalize_player_names, future_context_from_schedule, prepare_retained_pbp)


def test_name_variants_aggregate_by_stable_player_identity():
    raw = pd.DataFrame({
        "receiver_player_id": ["00-0027874", "00-0027874", "OTHER", None],
        "receiver_player_name": ["D. Thomas", "De.Thomas", "D. Thomas", None],
        "game_id": ["2017_01_LAC_DEN"] * 4,
        "target": [5, 3, 2, 0],
        "rz_targets": [1, 2, 0, 0],
    })
    fixed = normalize_player_names(raw)
    grouped = fixed.groupby(["receiver_player_id", "receiver_player_name", "game_id"]).agg(
        count=("target", "sum"), successes=("rz_targets", "sum"))
    assert len(grouped) == 2
    player = grouped.loc[("00-0027874", "D. Thomas", "2017_01_LAC_DEN")]
    assert player["count"] == 8
    assert player["successes"] / player["count"] == pytest.approx(3 / 8)
    assert raw.receiver_player_name.iloc[1] == "De.Thomas"


@pytest.fixture
def fixture():
    games = ["2016_01_A_B", "2016_02_A_B", "2017_01_A_B", "2017_02_A_B", "2018_01_A_B"]
    schedule = pd.DataFrame({"game_id": games, "game_date": pd.to_datetime([
        "2016-09-01", "2016-09-08", "2017-09-01", "2017-09-08", "2018-09-01"]),
        "home_team": "A", "away_team": "B"})
    roster = schedule[["game_id"]].assign(player_id="qb", player_name="QB", position="QB", team="A")
    frame = schedule[["game_id", "game_date"]].assign(passer_player_id="qb", passer_player_name="QB",
                                                       rate=[0.2, 0.8, 1.0, 0.0, 0.0], count=10.0)
    return MetricSpec("usage", "binary", frame, "passer_player_id", "passer_player_name"), schedule, roster


@pytest.mark.parametrize("kind", ["binary", "normal"])
def test_current_future_outcomes_do_not_change_earlier_snapshots(fixture, kind):
    spec, schedule, roster = fixture
    spec = replace(spec, kind=kind)
    original, _, _ = retained_from_specs([spec], schedule, roster, 2017)
    perturbed = spec.frame.copy()
    perturbed.loc[perturbed.game_id.isin(["2017_02_A_B", "2018_01_A_B"]), "rate"] = 0.9
    perturbed.loc[perturbed.game_id.isin(["2017_02_A_B", "2018_01_A_B"]), "count"] = 999
    changed, _, _ = retained_from_specs([replace(spec, frame=perturbed)], schedule, roster, 2017)
    pd.testing.assert_frame_equal(original, changed)
    assert set(original.game_id) == {"2017_01_A_B", "2017_02_A_B"}


def test_same_season_updates_only_next_week(fixture):
    spec, schedule, roster = fixture
    original, _, _ = retained_from_specs([spec], schedule, roster, 2017)
    changed_frame = spec.frame.copy()
    changed_frame.loc[changed_frame.game_id.eq("2017_01_A_B"), "rate"] = 0.0
    changed, _, _ = retained_from_specs([replace(spec, frame=changed_frame)], schedule, roster, 2017)
    assert original.average.iloc[0] == changed.average.iloc[0]
    assert original.average.iloc[1] != changed.average.iloc[1]
    assert original.bayes_estimate_std.isna().all()


def test_game_ids_control_cutoff_not_stale_season_column(fixture):
    spec, schedule, roster = fixture
    expected, _, _ = retained_from_specs([spec], schedule, roster, 2017)
    forged = replace(spec, frame=spec.frame.assign(season=2016))
    actual, _, _ = retained_from_specs([forged], schedule, roster, 2017)
    pd.testing.assert_frame_equal(expected, actual)


def test_rookie_and_missing_game_receive_frozen_prior(fixture):
    spec, schedule, roster = fixture
    roster = pd.concat([roster, roster.assign(player_id="rookie")], ignore_index=True)
    frame = spec.frame[~spec.frame.game_id.eq("2017_02_A_B")]
    player, _, _ = retained_from_specs([replace(spec, frame=frame)], schedule, roster, 2017)
    assert len(player) == 4
    assert player.loc[player.player_id.eq("rookie"), "average"].eq(0.5).all()
    assert player.average.notna().all()


def test_no_preseason_data_is_diagnostic_not_substitute(fixture):
    spec, schedule, roster = fixture
    spec = replace(spec, frame=spec.frame[spec.frame.game_id.str.startswith("2017")])
    player, _, diagnostic = retained_from_specs([spec], schedule, roster, 2017)
    assert player.empty
    assert diagnostic.reason.tolist() == ["no_preseason_observations"]


def test_unknown_roof_zero_count_does_not_enter_prior(fixture):
    spec, schedule, roster = fixture
    frame = spec.frame.copy()
    frame.loc[frame.game_id.eq("2016_01_A_B"), ["rate", "count"]] = [0, 0]
    player, _, diagnostic = retained_from_specs([replace(spec, source="qb_dome_rate_bayes", frame=frame)], schedule, roster, 2017)
    assert diagnostic.prior_rows.iloc[0] == 1
    assert player.average.iloc[0] == pytest.approx(0.8)


def test_direct_bypasses_frozen_helper(fixture, monkeypatch):
    spec, schedule, roster = fixture
    monkeypatch.setattr("nflsim.features.retained._frozen_bayesian_history",
                        lambda *args: pytest.fail("Direct metric smoothed"))
    direct = replace(spec, source="qb_team_games_played", kind="direct")
    player, _, _ = retained_from_specs([direct], schedule, roster, 2017)
    assert player.average.tolist() == [1, 0]
    with pytest.raises(ValueError, match="Unsupported direct timing"):
        retained_from_specs([replace(direct, source="unknown")], schedule, roster, 2017)


def test_direct_future_slots_carry_only_preweek_values(fixture):
    spec, schedule, roster = fixture
    schedule = pd.concat([schedule, pd.DataFrame([{
        "game_id": "2017_03_A_B", "game_date": pd.Timestamp("2017-09-22"),
        "home_team": "A", "away_team": "B",
    }])], ignore_index=True)
    roster = pd.concat([roster, pd.DataFrame([{
        "game_id": "2017_03_A_B", "team": "A", "player_id": "qb",
        "player_name": "QB", "position": "QB",
    }])], ignore_index=True)
    direct = replace(spec, source="qb_team_games_played", kind="direct")
    player, _, _ = retained_from_specs([direct], schedule, roster, 2017)
    week3 = player[player.game_id.eq("2017_03_A_B")]
    assert len(week3) == 1
    assert week3.average.iloc[0] == 1.0
    changed = direct.frame.copy()
    changed.loc[changed.game_id.eq("2017_02_A_B"), "rate"] = 999
    altered, _, _ = retained_from_specs([replace(direct, frame=changed)], schedule, roster, 2017)
    assert altered.loc[altered.game_id.eq("2017_03_A_B"), "average"].iloc[0] == 1000


def test_duplicate_week_rejected(fixture):
    spec, schedule, roster = fixture
    duplicate = spec.frame.iloc[[2]].assign(game_id="2017_01_A_C")
    with pytest.raises(ValueError, match="multiple entity games"):
        retained_from_specs([replace(spec, frame=pd.concat([spec.frame, duplicate]))], schedule, roster, 2017)


def test_root_write_rejected_before_collection(fixture, tmp_path, monkeypatch):
    spec, schedule, roster = fixture
    monkeypatch.setattr("nflsim.features.retained.prepare_retained_pbp", lambda frame: pytest.fail("Started unsafe regeneration"))
    destination = REPO / "unsafe_retained_test_output"
    with pytest.raises(ValueError, match="outside experiment"):
        regenerate_retained_features(spec.frame, schedule, roster, pd.DataFrame(), destination)
    assert not destination.exists()


@pytest.fixture
def collector_fixture():
    games, plays, rosters, snaps = [], [], [], []
    for season in (2016, 2017, 2018):
        for week in (1, 2):
            game = {"game_id": f"{season}_{week:02d}_A_B", "game_date": f"{season}-09-{week * 7:02d}",
                    "home_team": "A", "away_team": "B"}
            games.append(game)
            for team, opponent in (("A", "B"), ("B", "A")):
                for position in ("QB", "WR", "RB"):
                    player = team + position
                    rosters.append({"game_id": game["game_id"], "team": team, "player_id": player,
                                    "player_name": player, "position": position})
                    snaps.append({"game_id": game["game_id"], "game_type": "REG", "offense_snaps": 10,
                                  "position": position, "gsis_id": player, "name": player, "team": team})
                for play_type in ("pass", "run"):
                    is_pass = int(play_type == "pass")
                    plays.append({**game, "posteam": team, "defteam": opponent, "play_type": play_type,
                                  "play_id": len(plays), "pass_attempt": is_pass, "complete_pass": is_pass,
                                  "rush_attempt": 1 - is_pass, "pass": is_pass, "rush": 1 - is_pass,
                                  "receiver_player_id": team + "WR" if is_pass else None,
                                  "receiver_player_name": team + "WR" if is_pass else None,
                                  "passer_player_id": team + "QB" if is_pass else None,
                                  "passer_player_name": team + "QB" if is_pass else None,
                                  "rusher_player_id": team + "RB" if not is_pass else None,
                                  "rusher_player_name": team + "RB" if not is_pass else None,
                                  "first_down": 1, "yac_oe": float(week), "air_yards": 8,
                                  "yards_gained": week + 4, "rushing_yards": week + 4,
                                  "qb_scramble": 0, "cpoe": float(week), "sack": 0, "interception": 0,
                                  "is_interception_worthy": 0.2, "route": "SCREEN" if is_pass else None,
                                  "roof": "closed" if week == 1 else "outdoors", "down": 4,
                                  "ydstogo": 2, "score_differential": 0, "yardline_100": 40,
                                  "pass_oe": float(week), "epa": float(week),
                                  "game_seconds_remaining": 3600 - len(plays) * 30})
    return tuple(pd.DataFrame(rows) for rows in (plays, games, rosters, snaps))


def test_real_collectors_offline_persist_and_cutoff(collector_fixture, tmp_path, monkeypatch):
    import nflsim.features.bayesian_history as builder
    pbp, schedule, roster, snaps = collector_fixture
    loader = builder.load_snap_counts
    originals = {path: digest(path) for path in (REPO / "nflsim/features/bayesian_history.py", REPO / "nflsim/config/bayesian_decay_rates.json")}
    def forbidden(*args, **kwargs):
        pytest.fail("Network or unfrozen smoother used")
    monkeypatch.setattr(builder.nfl, "import_snap_counts", forbidden)
    monkeypatch.setattr(builder.nfl, "import_ids", forbidden)
    monkeypatch.setattr(builder, "smooth_binary_history", forbidden)
    monkeypatch.setattr(builder, "smooth_normal_history", forbidden)
    player, team, diagnostic = regenerate_retained_features(
        pbp, schedule, roster, snaps, tmp_path / "first", seasons=(2017, 2018),
        excluded_sources=("yac_bayes",), input_paths=originals,
    )
    assert builder.load_snap_counts is loader
    assert all(digest(path) == checksum for path, checksum in originals.items())
    assert "yac_bayes" not in set(player.source)
    assert {"medium", "qb_dome_rate_bayes", "qb_team_games_played", "SCREEN_bayes"} <= set(player.source)
    assert {"passing_massey_rating", "pass_oe_weekly_variation", "pace_of_play_bayes"} <= set(team.source)
    assert diagnostic.loc[diagnostic.metric.eq("deep"), "reason"].eq("no_preseason_observations").all()
    assert player.loc[player.source.eq("GO_bayes"), "average"].eq(0).all()
    manifest = json.loads((tmp_path / "first/retained_manifest.json").read_text())
    assert manifest["prior_cutoff"] == "prediction_season - 1"
    assert manifest["input_hashes"] == {str(path): value for path, value in originals.items()}
    assert (tmp_path / "first/retained_diagnostics.csv").exists()
    changed = pbp.copy()
    mask = changed.game_id.ge("2017_02")
    changed.loc[mask, ["epa", "pass_oe", "yards_gained", "yac_oe", "cpoe"]] += 20
    changed.loc[mask, "roof"] = "closed"
    altered_player, altered_team, _ = regenerate_retained_features(
        changed, schedule, roster, snaps, tmp_path / "second", seasons=(2017, 2018),
        excluded_sources=("yac_bayes",),
    )
    context = pd.concat([schedule.assign(team="A"), schedule.assign(team="B")])[["game_id", "team"]]
    original_wide = wide_tables(player, team, context)
    altered_wide = wide_tables(altered_player, altered_team, context)
    for original, altered in zip(original_wide, altered_wide):
        pd.testing.assert_frame_equal(original[original.game_id.le("2017_02_A_B")],
                                      altered[altered.game_id.le("2017_02_A_B")])
    prior_roof = player[player.source.eq("qb_dome_rate_bayes") & player.game_id.eq("2017_01_A_B")]
    assert prior_roof.average.between(0, 1).all()


def test_generation_failure_persists_diagnostic(fixture, tmp_path):
    spec, schedule, roster = fixture
    with pytest.raises(AttributeError):
        regenerate_retained_features(spec.frame, schedule, roster, pd.DataFrame(), tmp_path)
    assert json.loads((tmp_path / "retained_manifest.json").read_text())["status"] == "failed"
    assert pd.read_csv(tmp_path / "retained_diagnostics.csv").reason.eq("failed").any()


def test_active_season_retained_refresh_carries_prior_rows(collector_fixture, tmp_path):
    from nflsim.features.retained import regenerate_retained_features
    pbp, schedule, roster, snaps = collector_fixture
    base = tmp_path / "base"
    base.mkdir()
    prior_player = pd.DataFrame([{"player_id": "P", "player_name": "Player", "game_id": "2017_01_A_B",
                                 "game_date": "2017-09-07", "source": "legacy", "average": 1.0,
                                 "bayes_estimate_mean": 1.0, "bayes_estimate_std": float("nan")}])
    prior_team = pd.DataFrame([{"team": "A", "game_id": "2017_01_A_B", "game_date": "2017-09-07",
                                "source": "legacy", "average": .5, "bayes_estimate_mean": .5,
                                "bayes_estimate_std": float("nan")}])
    prior_player.to_csv(base / "retained_player_stats.csv", index=False)
    prior_team.to_csv(base / "retained_team_stats.csv", index=False)
    players, teams, _ = regenerate_retained_features(
        pbp, schedule, roster, snaps, tmp_path / "active", seasons=(2018,),
        excluded_sources=("yac_bayes",), prior_bundle=base,
    )
    assert ((players.game_id == "2017_01_A_B") & (players.source == "legacy")).any()
    assert ((teams.game_id == "2017_01_A_B") & (teams.source == "legacy")).any()
    assert set(players.loc[players.source.ne("legacy"), "game_id"].str[:4].astype(int)) <= {2018}
    manifest = json.loads((tmp_path / "active/retained_manifest.json").read_text())
    assert manifest["prediction_seasons"] == [2018]
    assert manifest["carried_forward_from"] == str(base)


def test_future_massey_week_handles_string_schedule_dates(collector_fixture):
    pbp, schedule, _, _ = collector_fixture
    history = prepare_retained_pbp(pbp)
    upcoming = pd.DataFrame([{"game_id": "2018_03_A_B", "game_date": "2018-09-22",
                              "home_team": "A", "away_team": "B", "season": 2018, "week": 3}])
    future = future_context_from_schedule(upcoming)
    specs = collect_team_metric_specs(history, future=future)
    massey = next(spec for spec in specs if spec.source == "passing_massey_rating")
    assert set(massey.frame.game_id) >= {"2018_03_A_B"}