"""Train cascade residual models for play_type, target_selection, rusher_selection, and completion.

Uses the same data loading pipeline as Torch Modeling.py, then trains each model
using the three-stage cascade from cascade_trainer.py:
  1. Base MLP classifier on 45% of game::team groups
  2. Context-conditioned residual grid model (KL divergence on 32-bin discretized residuals)
  3. Final MLP with blended residual feature (70% true / 30% COM in training)

At simulation time, stage 2 provides stochastic residual draws via inverse CDF sampling
(replacing hierarchical latent z), and stage 3 does play-level inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -- Paths --------------------------------------------------------------------
BASE_DIR = Path("C:/Users/steve/OneDrive/Desktop")
MODELING_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "Writing/WR Clustering"
DEFAULT_SEASON_START = 2017
DEFAULT_SEASON_END = 2024

CASCADE_MODEL_NAMES = ["play_type", "target_selection", "rusher_selection", "completion"]

# Opposing-defense columns (merged on defteam+game_id) — also used as cascade context
DEF_COLUMNS = [
    "deep_cpoe_allowed", "int_rate", "medium_cpoe_allowed", "short_cpoe_allowed",
    "yac_oe_allowed", "sack_rate", "rush_yards_per_carry", "def_rush_epa_rate",
    "def_pass_epa_rate", "def_winning_rate", "def_losing_rate", "def_tied_rate",
    "passing_massey_defense", "rushing_massey_defense",
]

# Offensive team stat columns (merged on posteam+game_id) — also used as cascade context
TEAM_STATS_COLUMNS = [
    "losing_rate", "tied_rate", "team_4th_rate", "winning_rate",
    "rushing_epa_rate", "epa_rate", "passing_massey_rating", "passing_massey_offense",
    "rushing_massey_rating", "rushing_massey_offense",
    "opp_def_pass_epa_rate", "opp_def_rush_epa_rate", "opp_pass_epa_rate", "opp_rushing_epa_rate",
]

# Context columns shared by all cascade models (stage 2 input)
DEFAULT_CONTEXT_COLUMNS = [
    "pass_oe",
    "pass_oe_weekly_variation",
    "target_gini_roll4",
    "rb_carry_split_gini_roll4",
    "rb_top_carry_share_roll4",
    "pace_of_play_bayes",
    "days_rest",
    "opp_days_rest",
] + TEAM_STATS_COLUMNS + DEF_COLUMNS

CATEGORICAL_COLS = ["roof", "posteam_type"]


def load_data(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Load PBP data and merge player/team stats, replicating Torch Modeling.py's pipeline.

    Returns (pbp, player_stats_wide, team_stats_wide, feature_catalog_dict).
    """
    import nfl_data_py as nfl

    player_stats = pd.read_csv(BASE_DIR / "player_stats.csv")
    player_stats["average"] = player_stats["bayes_estimate_mean"].combine_first(player_stats["average"])
    team_stats = pd.read_csv(BASE_DIR / "Team_Stats.csv")
    team_stats["average"] = team_stats["bayes_estimate_mean"].combine_first(team_stats["average"])

    game_state_cols = [
        "defteam_score", "down", "game_seconds_remaining", "posteam_score",
        "posteam_type", "yardline_100", "ydstogo", "roof", "score_differential",
    ]
    pass_columns = ["air_yards"]

    pbp = nfl.import_pbp_data(seasons, cache=True, alt_path=str(CACHE_DIR))
    pbp = pbp[pbp["aborted_play"] == 0]
    pbp = _filter_non_meaningful_games(pbp)

    rosters = nfl.import_weekly_rosters(seasons)
    ids = nfl.import_ids()
    id_to_name = dict(zip(ids["gsis_id"], ids["name"]))
    player_stats["player_name"] = player_stats["player_id"].map(id_to_name)

    rusher_columns = ["close_bayes", "explosive_bayes", "ypc_bayes", "rush_rate", "scramble_rate"]
    player_positions = nfl.import_players()[["gsis_id", "position"]].set_index("gsis_id", drop=True)
    player_positions["position"] = player_positions["position"].replace({"FB": "RB"})
    player_positions = player_positions[player_positions["position"].isin(["RB", "WR", "TE", "QB"])]

    # -- Pivot player/team stats to wide format --------------------------------
    player_stats_wide = (
        player_stats
        .pivot_table(index=["player_id", "player_name", "game_id", "game_date"], columns=["source"], values="average")
        .reset_index()
    )
    player_stats_wide = player_stats_wide.sort_values(by=["player_id", "game_date", "game_id"])
    player_stats_wide = player_stats_wide.groupby("player_id", group_keys=False).apply(lambda g: g.ffill())
    player_stats_wide.set_index(["player_id", "game_id"], inplace=True, drop=True)

    team_stats_wide = (
        team_stats
        .pivot_table(index=["team", "game_id", "game_date"], columns=["source"], values="average")
        .reset_index()
    )
    team_stats_wide = team_stats_wide.sort_values(by=["team", "game_date", "game_id"])
    team_stats_wide = team_stats_wide.groupby("team", group_keys=False).apply(lambda g: g.ffill())

    # Lag direct team tendency features by one game
    team_direct_lag_cols = [
        "pass_oe_weekly_variation", "target_gini_roll4",
        "rb_carry_split_gini_roll4", "rb_top_carry_share_roll4",
    ]
    for col in team_direct_lag_cols:
        if col in team_stats_wide.columns:
            team_stats_wide[col] = team_stats_wide.groupby("team")[col].shift(1)
    team_stats_wide.set_index(["team", "game_id"], inplace=True, drop=True)

    # -- Define feature column groups ------------------------------------------
    def_cols = list(DEF_COLUMNS)
    team_stats_columns = list(TEAM_STATS_COLUMNS)
    team_tendency_columns = [
        "pass_oe", "pass_oe_weekly_variation", "target_gini_roll4",
        "rb_carry_split_gini_roll4", "rb_top_carry_share_roll4", "pace_of_play_bayes",
    ]
    qb_context_columns = ["qb_team_games_played", "qb_dome_rate_bayes"]
    receiver_columns = ["deep", "medium", "short", "fd_bayes", "rz_bayes", "yac_bayes", "adot", "ypt"]
    passer_columns = [
        "cpoe_bayes", "turnover_bayes", "sack_bayes", "cpoe_short", "cpoe_medium",
        "cpoe_deep", "deep_pass_rate", "medium_pass_rate", "short_pass_rate",
    ]

    # -- Merge stats into PBP --------------------------------------------------
    pbp = pbp.merge(player_stats_wide.reindex(columns=receiver_columns), left_on=["receiver_player_id", "game_id"], right_index=True, how="left")
    pbp = pbp.merge(team_stats_wide.reindex(columns=def_cols), left_on=["defteam", "game_id"], right_index=True, how="left")
    pbp = pbp.merge(team_stats_wide.reindex(columns=team_stats_columns), left_on=["posteam", "game_id"], right_index=True, how="left")
    pbp = pbp.merge(player_stats_wide.reindex(columns=passer_columns), left_on=["passer_player_id", "game_id"], right_index=True, how="left")
    pbp = pbp.merge(team_stats_wide.reindex(columns=team_tendency_columns), left_on=["posteam", "game_id"], right_index=True, how="left")

    # Harmonize pass_oe after merge collisions
    if "pass_oe" not in pbp.columns:
        if "pass_oe_x" in pbp.columns and "pass_oe_y" in pbp.columns:
            pbp["pass_oe"] = pbp["pass_oe_y"].combine_first(pbp["pass_oe_x"])
        elif "pass_oe_y" in pbp.columns:
            pbp["pass_oe"] = pbp["pass_oe_y"]
        elif "pass_oe_x" in pbp.columns:
            pbp["pass_oe"] = pbp["pass_oe_x"]

    pbp = pbp.merge(player_stats_wide.reindex(columns=qb_context_columns), left_on=["passer_player_id", "game_id"], right_index=True, how="left")

    # Days rest
    _team_schedule = (
        pd.concat([
            pbp.groupby(["posteam", "game_id"])["game_date"].first().reset_index().rename(columns={"posteam": "team"}),
            pbp.groupby(["defteam", "game_id"])["game_date"].first().reset_index().rename(columns={"defteam": "team"}),
        ])
        .drop_duplicates(subset=["team", "game_id"])
    )
    _team_schedule["game_date"] = pd.to_datetime(_team_schedule["game_date"])
    _team_schedule = _team_schedule.sort_values(["team", "game_date"])
    _team_schedule["days_rest"] = _team_schedule.groupby("team")["game_date"].diff().dt.days
    offseason_baseline = _team_schedule.loc[_team_schedule["days_rest"] >= 30, "days_rest"]
    default_first_rest = float(21) if not offseason_baseline.empty else 7.0
    _team_schedule["days_rest"] = _team_schedule["days_rest"].fillna(default_first_rest)
    _team_schedule = _team_schedule.set_index(["team", "game_id"])
    pbp = pbp.merge(_team_schedule[["days_rest"]], left_on=["posteam", "game_id"], right_index=True, how="left")
    pbp = pbp.merge(
        _team_schedule[["days_rest"]].rename(columns={"days_rest": "opp_days_rest"}),
        left_on=["defteam", "game_id"], right_index=True, how="left",
    )

    # Explicit game::team group key used for cascade splitting/context aggregation
    pbp["game_team_id"] = pbp["game_id"].astype(str) + "::" + pbp["posteam"].astype(str)

    # Filter to regular season and non-penalty plays for the cascade models
    pbp = pbp[pbp["season_type"] == "REG"]

    catalog = {
        "game_state_cols": game_state_cols,
        "pass_columns": pass_columns,
        "rusher_columns": rusher_columns,
        "def_cols": def_cols,
        "team_stats_columns": team_stats_columns,
        "team_tendency_columns": team_tendency_columns,
        "receiver_columns": receiver_columns,
        "passer_columns": passer_columns,
        "qb_context_columns": qb_context_columns,
    }
    return pbp, player_stats_wide, team_stats_wide, catalog


def _filter_non_meaningful_games(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop week-18 games with large spreads (teams resting starters)."""
    if frame is None or len(frame) == 0 or "game_id" not in frame.columns:
        return frame
    game_rows = (
        frame[["game_id", "week", "season_type", "spread_line", "score_differential"]]
        .drop_duplicates(subset=["game_id"])
        .copy()
    )
    if game_rows.empty:
        return frame
    season_type = game_rows.get("season_type", pd.Series("REG", index=game_rows.index)).astype(str).str.upper()
    week = pd.to_numeric(game_rows.get("week", pd.Series(np.nan, index=game_rows.index)), errors="coerce")
    spread = pd.to_numeric(game_rows.get("spread_line", pd.Series(np.nan, index=game_rows.index)), errors="coerce")
    weird_game_ids = game_rows.loc[
        (season_type == "REG") & (week == 18) & (spread.abs() >= 10.0)
    ]["game_id"]
    if len(weird_game_ids):
        frame = frame[~frame["game_id"].isin(weird_game_ids)]
    return frame


def _build_skill_slot_features(
    pbp: pd.DataFrame,
    player_stats_wide: pd.DataFrame,
    rosters: pd.DataFrame,
    catalog: dict[str, list[str]],
    seasons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Build skill-position slot features for target/rusher selection models.

    Returns (pbp_pass, pbp_run, target_slot_cols, rusher_slot_cols).
    """
    import nfl_data_py as nfl

    # -- Replicate roster preprocessing from Torch Modeling.py --
    positions = ["WR", "RB", "TE", "QB"]
    rosters = rosters[rosters["position"].isin(positions)].copy()
    rosters["game_id"] = rosters["team"] + rosters["season"].astype(str) + rosters["week"].astype(str)

    depth_chart = nfl.import_depth_charts(seasons)
    depth_chart = depth_chart[np.isnan(depth_chart["week"]) == 0]
    depth_chart["game_id"] = depth_chart["club_code"] + depth_chart["season"].astype(str) + depth_chart["week"].astype(int).astype(str)

    injuries = nfl.import_injuries(seasons)
    injuries = injuries[injuries["report_status"] == "Out"]
    rosters = rosters.merge(
        injuries.set_index(["week", "season", "gsis_id"])["report_status"],
        left_on=["week", "season", "player_id"], right_index=True, how="left",
    )
    rosters = rosters[rosters["report_status"] != "Out"]

    # Map constructed game_id to actual PBP game_id
    pbp["merge_key"] = pbp["posteam"] + pbp["season"].astype(str) + pbp["week"].astype(str)
    merge_mapping = pbp[["merge_key", "game_id"]].drop_duplicates().dropna()
    rosters["game_id"] = rosters["game_id"].map(merge_mapping.set_index("merge_key")["game_id"])

    # Merge player stats into rosters (player_stats_wide has MultiIndex player_id+game_id)
    rosters = rosters.merge(player_stats_wide, left_on=["player_id", "game_id"], right_index=True, how="left")
    rosters["alt_game_id"] = rosters["team"] + rosters["season"].astype(str) + rosters["week"].astype(str)

    # QB depth chart
    depth_chart_qb = depth_chart[depth_chart["position"] == "QB"]
    qb_roster = rosters[rosters["position"] == "QB"]
    qb_with_depth = qb_roster.merge(
        depth_chart_qb.set_index(["game_id", "gsis_id"])["depth_team"],
        left_on=["alt_game_id", "player_id"],
        right_index=True,
        how="left",
    )
    qb_with_depth_sorted = qb_with_depth.sort_values(by=["alt_game_id", "depth_team"])
    qb_with_depth_sorted.dropna(subset="depth_team", inplace=True)
    top_qb_per_game = (
        qb_with_depth.sort_values(["game_id", "team", "depth_team"])
        .groupby("alt_game_id")
        .first()
        .reset_index()
    )

    wr_te_sorted = (
        rosters[rosters["position"].isin(["WR", "TE"])]
        .sort_values(by="medium", ascending=False)
        .reset_index(drop=True)
    )
    rb_sorted = (
        rosters[rosters["position"] == "RB"]
        .sort_values(by="rush_rate", ascending=False)
        .reset_index(drop=True)
    )
    sorted_players = pd.concat([top_qb_per_game, wr_te_sorted, rb_sorted], ignore_index=True)
    # Ensure game_id is available for grouping
    if "game_id" not in sorted_players.columns:
        sorted_players["game_id"] = sorted_players.get("alt_game_id", "")
    position_limits = {"QB": 1, "WR": 4, "RB": 3, "TE": 3}
    sorted_players["rank"] = sorted_players.groupby(["game_id", "team", "position"]).cumcount()
    sorted_players = sorted_players.sort_values(by=["game_id", "team"]).reset_index(drop=True)
    position_filtered = pd.concat([
        sorted_players[sorted_players["position"] == pos].groupby(["game_id", "team"]).head(n)
        for pos, n in position_limits.items()
    ])
    position_filtered["pos_rank"] = position_filtered.groupby(["game_id", "team", "position"]).cumcount() + 1
    position_filtered["label"] = position_filtered["position"] + position_filtered["pos_rank"].astype(str)

    top_skill_df = position_filtered.pivot(index=["team", "game_id"], columns="label", values="player_id").reset_index()
    column_order = ["game_id"] + ["QB1"] + [f"WR{i}" for i in range(1, 5)] + [f"RB{i}" for i in range(1, 4)] + [f"TE{i}" for i in range(1, 4)]
    top_skill_df = top_skill_df[[c for c in column_order if c in top_skill_df.columns] + ["team"]]
    nan_counts = top_skill_df.isna().sum(axis=1)
    top_skill_df = top_skill_df[nan_counts <= 0]
    top_skill_df = top_skill_df.merge(pbp.set_index("game_id")["game_date"], left_on="game_id", right_index=True, how="left").drop_duplicates()

    # -- Target selection data ------------------------------------------------
    pbp_pass = pbp[pbp["pass_attempt"] == 1].copy()
    pbp_pass = pbp_pass.merge(
        top_skill_df, left_on=["game_id", "posteam"], right_on=["game_id", "team"], how="left"
    )
    skill_cols = [c for c in top_skill_df.columns if c not in ("game_id", "team", "game_date")]
    pbp_pass = pbp_pass[pbp_pass["receiver_player_id"].isin(pbp_pass[skill_cols].stack().unique())]

    def get_target_index(row):
        for i, col in enumerate(skill_cols):
            if row["receiver_player_id"] == row[col]:
                return i
        return -1
    pbp_pass["target_index"] = pbp_pass.apply(get_target_index, axis=1)
    pbp_pass = pbp_pass[pbp_pass["target_index"] >= 0]

    # Slot stats for receivers — keep team in the key so cross-team slot stats never mix
    receiver_columns = catalog["receiver_columns"]
    skill_long = top_skill_df.melt(id_vars=["game_id", "team", "game_date"], value_name="gsis_id", var_name="slot").drop_duplicates()
    skill_long["game_date"] = pd.to_datetime(skill_long["game_date"])
    player_stats_wide_reset = player_stats_wide.reset_index()
    player_stats_wide_reset["game_date"] = pd.to_datetime(player_stats_wide_reset["game_date"])
    skill_long_sorted = skill_long.sort_values(["game_date", "gsis_id"])
    player_stats_sorted = player_stats_wide_reset.sort_values(["game_date", "player_id"])
    skill_long_sorted["player_id"] = skill_long_sorted["gsis_id"]
    # Ensure player_id is in merge_cols (needed by merge_asof 'by' parameter)
    merge_cols = [c for c in player_stats_sorted.columns if c not in skill_long_sorted.columns] + ["game_date"]
    if "player_id" not in merge_cols:
        merge_cols.append("player_id")
    skill_long = pd.merge_asof(
        skill_long_sorted, player_stats_sorted[merge_cols],
        by="player_id", on="game_date", direction="backward",
    )
    receiver_stats_long = skill_long.pivot_table(
        index=["game_id", "team"], columns="slot",
        values=[c for c in receiver_columns if c in skill_long.columns],
    )
    receiver_stats_long.columns = ["_".join(col).strip() for col in receiver_stats_long.columns]
    receiver_stats_long = receiver_stats_long.reset_index()
    pbp_pass = pbp_pass.merge(receiver_stats_long, on=["game_id", "team"], how="left")
    target_slot_cols = [c for c in receiver_stats_long.columns if c not in ("game_id", "team")]

    # -- Rusher selection data ------------------------------------------------
    pbp_run = pbp[(pbp["rush_attempt"] == 1) & (pbp["qb_scramble"] == 0)].copy()
    pbp_run = pbp_run.merge(
        top_skill_df, left_on=["game_id", "posteam"], right_on=["game_id", "team"], how="left"
    )
    pbp_run = pbp_run[pbp_run["rusher_player_id"].isin(pbp_run[skill_cols].stack().unique())]

    def get_rush_index(row):
        for i, col in enumerate(skill_cols):
            if row["rusher_player_id"] == row[col]:
                return i
        return -1
    pbp_run["rush_index"] = pbp_run.apply(get_rush_index, axis=1)
    pbp_run = pbp_run[pbp_run["rush_index"] >= 0]

    rusher_columns = catalog["rusher_columns"]
    rusher_stats_long = skill_long.pivot_table(
        index=["game_id", "team"], columns="slot",
        values=[c for c in rusher_columns if c in skill_long.columns],
    )
    rusher_stats_long.columns = ["_".join(col).strip() for col in rusher_stats_long.columns]
    rusher_stats_long = rusher_stats_long.reset_index()
    pbp_run = pbp_run.merge(rusher_stats_long, on=["game_id", "team"], how="left")
    rusher_slot_cols = [c for c in rusher_stats_long.columns if c[:-4] in rusher_columns]

    return pbp_pass, pbp_run, target_slot_cols, rusher_slot_cols


# -- Model-specific training --------------------------------------------------


def train_play_type(pbp: pd.DataFrame, catalog: dict[str, list[str]], config) -> None:
    from cascade_trainer import fit_cascade_general, save_cascade_artifacts

    play = pbp.copy()
    play["play_type"] = np.select(
        [play["qb_dropback"] == 1, play["rush"] == 1, play["punt_attempt"] == 1, play["field_goal_attempt"] == 1],
        ["pass", "rush", "punt", "kick"],
        default="other",
    )
    play = play[play["play_type"].isin(["pass", "rush", "punt", "kick"])]

    feature_cols = (
        catalog["game_state_cols"]
        + catalog["team_stats_columns"]
        + catalog["def_cols"]
        + catalog["team_tendency_columns"]
        + ["days_rest", "opp_days_rest"]
    )
    feature_cols = [c for c in feature_cols if c in play.columns]

    artifacts = fit_cascade_general(
        frame=play,
        target_column="play_type",
        feature_columns=feature_cols,
        context_columns=[c for c in DEFAULT_CONTEXT_COLUMNS if c in play.columns],
        categorical_columns=CATEGORICAL_COLS,
        class_labels=["kick", "pass", "punt", "rush"],
        config=config,
        group_column="game_team_id",
        model_label="play_type",
    )
    save_cascade_artifacts(artifacts, BASE_DIR, "play", feature_groups=[
        "game_state_cols", "team_stats_columns", "def_cols", "pregame_context_columns",
    ])
    return artifacts


def train_completion(pbp: pd.DataFrame, catalog: dict[str, list[str]], config) -> None:
    from cascade_trainer import fit_cascade_general, save_cascade_artifacts

    pbp_comp = pbp[pbp["qb_dropback"] == 1].copy()
    pbp_comp["completion"] = pbp_comp["complete_pass"].fillna(0).astype(int).astype(str)

    feature_cols = (
        catalog["game_state_cols"]
        + catalog["def_cols"]
        + catalog["passer_columns"]
        + catalog["pass_columns"]
        + catalog["team_stats_columns"]
        + catalog["team_tendency_columns"]
        + catalog["qb_context_columns"]
        + ["days_rest", "opp_days_rest"]
    )
    feature_cols = [c for c in feature_cols if c in pbp_comp.columns]

    artifacts = fit_cascade_general(
        frame=pbp_comp,
        target_column="completion",
        feature_columns=feature_cols,
        context_columns=[c for c in DEFAULT_CONTEXT_COLUMNS if c in pbp_comp.columns],
        categorical_columns=CATEGORICAL_COLS,
        class_labels=["0", "1"],
        config=config,
        group_column="game_team_id",
        model_label="completion",
    )
    save_cascade_artifacts(artifacts, BASE_DIR, "comp", feature_groups=[
        "game_state_cols", "def_cols", "passer_columns", "pass_columns",
        "team_stats_columns", "pregame_context_columns", "qb_context_columns",
    ])
    return artifacts


def train_target_selection(
    pbp_pass: pd.DataFrame,
    target_slot_cols: list[str],
    catalog: dict[str, list[str]],
    config,
) -> None:
    from cascade_trainer import fit_cascade_general, save_cascade_artifacts

    feature_cols = (
        catalog["game_state_cols"]
        + target_slot_cols
        + catalog["pass_columns"]
        + catalog["team_tendency_columns"]
        + catalog["qb_context_columns"]
        + ["days_rest", "opp_days_rest"]
    )
    feature_cols = [c for c in feature_cols if c in pbp_pass.columns]

    pbp_target = pbp_pass.copy()
    pbp_target["target_index"] = pbp_target["target_index"].astype(str)

    n_classes = int(pbp_pass["target_index"].astype(int).max()) + 1
    class_labels = [str(i) for i in range(n_classes)]

    artifacts = fit_cascade_general(
        frame=pbp_target,
        target_column="target_index",
        feature_columns=feature_cols,
        context_columns=[c for c in DEFAULT_CONTEXT_COLUMNS if c in pbp_target.columns],
        categorical_columns=CATEGORICAL_COLS,
        class_labels=class_labels,
        config=config,
        group_column="game_team_id",
        model_label="target_selection",
    )
    save_cascade_artifacts(artifacts, BASE_DIR, "target", feature_groups=[
        "game_state_cols", "slot_stat_cols", "pass_columns",
        "pregame_context_columns", "qb_context_columns",
    ])
    return artifacts


def train_rusher_selection(
    pbp_run: pd.DataFrame,
    rusher_slot_cols: list[str],
    catalog: dict[str, list[str]],
    config,
) -> None:
    from cascade_trainer import fit_cascade_general, save_cascade_artifacts

    feature_cols = (
        catalog["game_state_cols"]
        + rusher_slot_cols
        + catalog["team_tendency_columns"]
        + ["days_rest", "opp_days_rest"]
    )
    feature_cols = [c for c in feature_cols if c in pbp_run.columns]

    pbp_rusher = pbp_run.copy()
    pbp_rusher["rush_index"] = pbp_rusher["rush_index"].astype(str)

    n_classes = int(pbp_run["rush_index"].astype(int).max()) + 1
    class_labels = [str(i) for i in range(n_classes)]

    artifacts = fit_cascade_general(
        frame=pbp_rusher,
        target_column="rush_index",
        feature_columns=feature_cols,
        context_columns=[c for c in DEFAULT_CONTEXT_COLUMNS if c in pbp_rusher.columns],
        categorical_columns=CATEGORICAL_COLS,
        class_labels=class_labels,
        config=config,
        group_column="game_team_id",
        model_label="rusher_selection",
    )
    save_cascade_artifacts(artifacts, BASE_DIR, "rusher", feature_groups=[
        "game_state_cols", "slot_stat_cols", "pregame_context_columns",
    ])
    return artifacts


# -- CLI ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cascade residual models.")
    parser.add_argument("--models", nargs="+", default=CASCADE_MODEL_NAMES, choices=CASCADE_MODEL_NAMES)
    parser.add_argument("--seasons", nargs="+", default=[f"{DEFAULT_SEASON_START}-{DEFAULT_SEASON_END}"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stage2-epochs", type=int, default=150)
    parser.add_argument("--n-bins", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blend", type=float, default=0.7, help="Stage 3 residual blend (0=pure COM, 1=pure true)")
    args = parser.parse_args()

    # Parse seasons
    seasons = []
    for token in args.seasons:
        if "-" in token:
            start, end = token.split("-", 1)
            seasons.extend(range(int(start), int(end) + 1))
        else:
            seasons.append(int(token))
    seasons = sorted(set(seasons))

    print(f"Training cascade models: {args.models}")
    print(f"Seasons: {seasons}")

    from cascade_trainer import CascadeConfig
    config = CascadeConfig(
        epochs=args.epochs,
        stage2_epochs=args.stage2_epochs,
        n_bins=args.n_bins,
        batch_size=args.batch_size,
        seed=args.seed,
        stage3_residual_blend=args.blend,
    )

    # Load data
    print("\n-- Loading data --")
    pbp, player_stats_wide, team_stats_wide, catalog = load_data(seasons)
    print(f"PBP: {len(pbp):,} rows")

    # Check if we need slot features
    need_slots = any(m in args.models for m in ("target_selection", "rusher_selection"))
    if need_slots:
        import nfl_data_py as nfl
        rosters = nfl.import_weekly_rosters(seasons)
        pbp_pass, pbp_run, target_slot_cols, rusher_slot_cols = _build_skill_slot_features(
            pbp, player_stats_wide, rosters, catalog, seasons
        )
        print(f"Target selection: {len(pbp_pass):,} rows, {len(target_slot_cols)} slot cols")
        print(f"Rusher selection: {len(pbp_run):,} rows, {len(rusher_slot_cols)} slot cols")

    # Train selected models
    results = {}
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        if model_name == "play_type":
            results[model_name] = train_play_type(pbp, catalog, config)
        elif model_name == "completion":
            results[model_name] = train_completion(pbp, catalog, config)
        elif model_name == "target_selection":
            results[model_name] = train_target_selection(pbp_pass, target_slot_cols, catalog, config)
        elif model_name == "rusher_selection":
            results[model_name] = train_rusher_selection(pbp_run, rusher_slot_cols, catalog, config)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for model_name, artifacts in results.items():
        m = artifacts.metrics
        print(f"\n{model_name}:")
        print(f"  Stage 1 log_loss: {m['stage1']['val_metrics']['log_loss']:.4f}")
        print(f"  Stage 3 log_loss: {m['stage3']['val_metrics']['log_loss']:.4f}")
        print(f"  Improvement: {m['comparison']['log_loss_improvement']:.4f}")
        print(f"  PCA variance: {m['stage2']['pca_explained_variance_ratio']:.4f}")


if __name__ == "__main__":
    main()
