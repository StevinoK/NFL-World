"""Recency-weighted ridge (generalized Massey) estimates for skill metrics.

Each target is a play-level rate regressed on sparse blocks: subject player (or
offense team), offense team for player fits, opposing defense, venue, home flag,
rest differential and a position effect. Covariates are constant within a subject-game, so play rows
collapse exactly to subject-game rows weighted by play count. The model is
refit once per league week on strictly earlier data, which re-prices past
opponents as their strength is learned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from bayesian_feature_builder import _season_week_from_game_id


REST_DAYS_CAP = 21.0  # matches point_spread_model.REST_DAYS_CAP
DEFAULT_REST_DAYS = 7.0
REGULAR_SEASON_WEEKS = 18
SUBJECT_NEUTRAL_BLOCKS = ("intercept", "position", "subject")
DEFENSE_NEUTRAL_BLOCKS = ("intercept", "defense")
EXTRA_BLOCKS = ("teammate",)
DIAGONAL_JITTER = 1e-8


def _numeric(pbp: pd.DataFrame, column: str) -> pd.Series:
    if column not in pbp.columns:
        return pd.Series(np.nan, index=pbp.index)
    return pd.to_numeric(pbp[column], errors="coerce")


def _flag(column: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda pbp: _numeric(pbp, column).eq(1)


def _depth_flag(bucket: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda pbp: _numeric(pbp, "pass_attempt").eq(1) & pbp["depth_bucket"].eq(bucket)


@dataclass(frozen=True)
class RidgeTarget:
    name: str
    kind: Literal["binary", "normal"]
    scope: Literal["player", "team"]
    subject_col: str
    value_col: str
    mask: Callable[[pd.DataFrame], pd.Series]
    player_output: str | None
    defense_output: str | None
    teammate_col: str | None = None


TARGETS: tuple[RidgeTarget, ...] = (
    RidgeTarget("yac_oe", "normal", "player", "receiver_player_id", "yac_oe", _flag("complete_pass"),
                "yac_bayes", "yac_oe_allowed", "passer_player_id"),
    RidgeTarget("first_down", "binary", "player", "receiver_player_id", "first_down", _flag("complete_pass"),
                "fd_bayes", None, "passer_player_id"),
    RidgeTarget("yards_per_target", "normal", "player", "receiver_player_id", "yards_gained", _flag("target"),
                "ypt", None, "passer_player_id"),
    RidgeTarget("cpoe", "normal", "player", "passer_player_id", "cpoe", _flag("pass_attempt"),
                "cpoe_bayes", None, "receiver_player_id"),
    *(
        RidgeTarget(f"cpoe_{bucket}", "normal", "player", "passer_player_id", "cpoe", _depth_flag(bucket),
                    f"cpoe_{bucket}", f"{bucket}_cpoe_allowed", "receiver_player_id")
        for bucket in ("short", "medium", "deep")
    ),
    RidgeTarget("sack", "binary", "player", "passer_player_id", "sack", _flag("pass_attempt"),
                "sack_bayes", "sack_rate"),
    RidgeTarget("interception_worthy", "binary", "player", "passer_player_id", "is_interception_worthy",
                _flag("pass_attempt"), "turnover_bayes", None),
    RidgeTarget("interception", "binary", "player", "passer_player_id", "interception", _flag("pass"),
                None, "int_rate"),
    RidgeTarget("rush_yards", "normal", "player", "rusher_player_id", "yards_gained", _flag("rush_attempt"),
                "ypc_bayes", "rush_yards_per_carry"),
    RidgeTarget("explosive_rush", "binary", "player", "rusher_player_id", "explosive", _flag("rush_attempt"),
                "explosive_bayes", None),
    RidgeTarget("scramble_yards", "normal", "player", "rusher_player_id", "yards_gained", _flag("qb_scramble"),
                "qb_scramble_ypc_bayes", None),
    RidgeTarget("pass_epa", "normal", "team", "posteam", "epa", _flag("pass"),
                "epa_rate", "def_pass_epa_rate"),
    RidgeTarget("rush_epa", "normal", "team", "posteam", "epa", _flag("rush"),
                "rushing_epa_rate", "def_rush_epa_rate"),
)
TARGETS_BY_NAME = {target.name: target for target in TARGETS}


@dataclass(frozen=True)
class RidgeConfig:
    within_decay: float = 0.95
    across_decay: float = 0.99
    season_carryover: float = 0.5
    lookback_seasons: int = 2
    # Penalties are in units of recency-weighted plays.
    lambda_subject: float = 100.0
    lambda_defense: float = 300.0
    lambda_venue: float = 1000.0
    lambda_position: float = 50.0
    lambda_teammate: float = 300.0
    lambda_offense_team: float = 300.0
    lambda_context: float = 1.0

    def label(self) -> str:
        return (
            f"din{self.within_decay:g}_dout{self.across_decay:g}_c{self.season_carryover:g}"
            f"_L{self.lookback_seasons}_ls{self.lambda_subject:g}_ld{self.lambda_defense:g}"
            f"_lv{self.lambda_venue:g}_lt{self.lambda_teammate:g}_lo{self.lambda_offense_team:g}"
        )

    def block_penalty(self, block: str) -> float:
        return {
            "intercept": 0.0,
            "position": self.lambda_position,
            "subject": self.lambda_subject,
            "defense": self.lambda_defense,
            "venue": self.lambda_venue,
            "home": self.lambda_context,
            "rest": self.lambda_context,
            "teammate": self.lambda_teammate,
            "offense_team": self.lambda_offense_team,
        }[block]


@dataclass(frozen=True)
class DesignMatrix:
    matrix: sp.csr_matrix
    blocks: np.ndarray
    labels: np.ndarray

    def penalty(self, config: RidgeConfig) -> np.ndarray:
        lookup = {block: config.block_penalty(block) for block in np.unique(self.blocks)}
        return np.array([lookup[block] for block in self.blocks], dtype=float) + DIAGONAL_JITTER

    def block_mask(self, blocks: Iterable[str]) -> np.ndarray:
        return np.isin(self.blocks, list(blocks))


def build_game_context(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per game with venue, neutral flag and per-team rest days."""
    venue_source = next(
        (column for column in ("stadium_id", "game_stadium", "stadium") if column in pbp.columns),
        None,
    )
    columns = ["game_id", "game_date", "home_team", "away_team"]
    extra = [column for column in (venue_source, "location") if column is not None and column in pbp.columns]
    games = (
        pbp[columns + extra]
        .dropna(subset=["home_team", "away_team"])
        .drop_duplicates("game_id")
        .copy()
    )
    games["game_id"] = games["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    venue = games[venue_source] if venue_source is not None else pd.Series(np.nan, index=games.index)
    games["venue"] = venue.astype("string").fillna(games["home_team"].astype("string")).astype(str)
    location = games["location"] if "location" in games.columns else pd.Series("", index=games.index)
    games["neutral"] = location.astype("string").str.lower().eq("neutral").fillna(False).astype(bool)
    parsed = [_season_week_from_game_id(game_id) for game_id in games["game_id"]]
    games["season"] = [value[0] for value in parsed]
    games["week"] = [value[1] for value in parsed]

    team_games = pd.concat(
        [
            games[["game_id", "game_date", "home_team"]].rename(columns={"home_team": "team"}),
            games[["game_id", "game_date", "away_team"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    ).sort_values(["team", "game_date", "game_id"])
    gap = team_games.groupby("team")["game_date"].diff().dt.days
    team_games["rest"] = gap.fillna(DEFAULT_REST_DAYS).clip(0.0, REST_DAYS_CAP)
    rest = team_games.set_index(["game_id", "team"])["rest"]
    games["home_rest"] = rest.reindex(list(zip(games["game_id"], games["home_team"]))).to_numpy()
    games["away_rest"] = rest.reindex(list(zip(games["game_id"], games["away_team"]))).to_numpy()
    return games[
        ["game_id", "game_date", "season", "week", "home_team", "away_team", "venue", "neutral", "home_rest", "away_rest"]
    ].reset_index(drop=True)


def build_target_rows(
    pbp: pd.DataFrame,
    target: RidgeTarget,
    games: pd.DataFrame,
    positions: dict[str, str] | None = None,
    extra_blocks: Iterable[str] = (),
) -> pd.DataFrame:
    """Collapse plays to one weighted row per unique design row."""
    extra_blocks = tuple(extra_blocks)
    if target.value_col not in pbp.columns or target.subject_col not in pbp.columns:
        return pd.DataFrame()
    mask = (
        target.mask(pbp).fillna(False).astype(bool)
        & pbp[target.subject_col].notna()
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
    )
    selected = pbp.loc[mask]
    plays = pd.DataFrame(
        {
            "subject": selected[target.subject_col].astype(str),
            "posteam": selected["posteam"].astype(str),
            "defteam": selected["defteam"].astype(str),
            "game_id": selected["game_id"].astype(str),
            "value": pd.to_numeric(selected[target.value_col], errors="coerce"),
        }
    )
    keys = ["subject", "posteam", "defteam", "game_id"]
    if "teammate" in extra_blocks:
        if target.teammate_col is None or target.teammate_col not in selected.columns:
            raise ValueError(f"Target {target.name} has no teammate column")
        plays["teammate"] = selected[target.teammate_col].astype("string").fillna("NONE").astype(str)
        keys.append("teammate")
    plays = plays.dropna(subset=["value"])
    if plays.empty:
        return pd.DataFrame()

    rows = (
        plays.groupby(keys, sort=False)
        .agg(value_sum=("value", "sum"), count=("value", "size"))
        .reset_index()
    )
    rows["count"] = rows["count"].astype(float)
    rows["rate"] = rows["value_sum"] / rows["count"]
    rows = rows.merge(games, on="game_id", how="inner").dropna(subset=["season", "week"])
    rows["season"] = rows["season"].astype(int)
    rows["week"] = rows["week"].astype(int)
    is_home = rows["posteam"].eq(rows["home_team"])
    rows["is_home"] = (is_home & ~rows["neutral"]).astype(float)
    own_rest = np.where(is_home, rows["home_rest"], rows["away_rest"])
    opponent_rest = np.where(is_home, rows["away_rest"], rows["home_rest"])
    rows["rest_diff"] = np.nan_to_num(own_rest - opponent_rest)
    if target.scope == "player":
        lookup = positions or {}
        rows["position"] = rows["subject"].map(lookup).fillna("OTHER").astype(str)
    else:
        rows["position"] = "TEAM"
    rows["season_last_week"] = rows.groupby("season")["week"].transform("max").astype(int)
    return rows.sort_values(["season", "week", "game_id"] + keys[:1]).reset_index(drop=True)


def _one_hot(values: pd.Series, block: str) -> tuple[sp.csr_matrix, list[str], list[str]]:
    codes, uniques = pd.factorize(values.astype(str), sort=True)
    matrix = sp.csr_matrix(
        (np.ones(len(codes)), (np.arange(len(codes)), codes)),
        shape=(len(codes), len(uniques)),
    )
    return matrix, [block] * len(uniques), [str(value) for value in uniques]


def build_design(
    rows: pd.DataFrame,
    scope: Literal["player", "team"],
    extra_blocks: Iterable[str] = (),
) -> DesignMatrix:
    n_rows = len(rows)
    parts: list[sp.csr_matrix] = [sp.csr_matrix(np.ones((n_rows, 1)))]
    blocks: list[str] = ["intercept"]
    labels: list[str] = ["intercept"]

    categorical = []
    if scope == "player":
        categorical.append(("position", rows["position"]))
    categorical.extend([("subject", rows["subject"]), ("defense", rows["defteam"]), ("venue", rows["venue"])])
    if scope == "player":
        categorical.append(("offense_team", rows["posteam"]))
    for block in extra_blocks:
        if block == "teammate":
            categorical.append(("teammate", rows["teammate"]))
        else:
            raise ValueError(f"Unknown extra block {block}")
    for block, values in categorical:
        matrix, block_names, block_labels = _one_hot(values, block)
        parts.append(matrix)
        blocks.extend(block_names)
        labels.extend(block_labels)

    for block, column in (("home", "is_home"), ("rest", "rest_diff")):
        parts.append(sp.csr_matrix(rows[column].to_numpy(dtype=float).reshape(-1, 1)))
        blocks.append(block)
        labels.append(column)
    return DesignMatrix(sp.hstack(parts, format="csr"), np.array(blocks), np.array(labels))


def recency_weights(
    obs_season: np.ndarray,
    obs_week: np.ndarray,
    obs_season_last_week: np.ndarray,
    pred_season: int,
    pred_week: int,
    config: RidgeConfig,
) -> np.ndarray:
    """Pregame weights: same-season decay by week, earlier seasons by carryover plus a slow within-season decay."""
    same_season = obs_season == pred_season
    before = (obs_season < pred_season) | (same_season & (obs_week < pred_week))
    in_window = obs_season >= pred_season - config.lookback_seasons
    season_gap = np.maximum(pred_season - obs_season, 0)
    current = np.power(config.within_decay, np.maximum(pred_week - obs_week, 0))
    weeks_back = (obs_season_last_week - obs_week) + REGULAR_SEASON_WEEKS * np.maximum(season_gap - 1, 0)
    prior = (
        np.power(config.within_decay, max(pred_week - 1, 0))
        * np.power(config.season_carryover, season_gap)
        * np.power(config.across_decay, np.maximum(weeks_back, 0))
    )
    weights = np.where(same_season, current, prior)
    return np.where(before & in_window, weights, 0.0)


def solve_ridge(
    matrix: sp.csr_matrix,
    target: np.ndarray,
    weights: np.ndarray,
    penalty: np.ndarray,
) -> np.ndarray:
    active = np.flatnonzero(weights > 0)
    design = matrix[active]
    weighted = sp.diags(weights[active]) @ design
    normal = (design.T @ weighted + sp.diags(penalty)).tocsc()
    rhs = weighted.T @ target[active]
    return np.asarray(spsolve(normal, rhs), dtype=float)


def _row_weights(rows: pd.DataFrame, pred_season: int, pred_week: int, config: RidgeConfig) -> np.ndarray:
    return recency_weights(
        rows["season"].to_numpy(dtype=int),
        rows["week"].to_numpy(dtype=int),
        rows["season_last_week"].to_numpy(dtype=int),
        pred_season,
        pred_week,
        config,
    ) * rows["count"].to_numpy(dtype=float)


def rolling_predictions(
    rows: pd.DataFrame,
    design: DesignMatrix,
    config: RidgeConfig,
    predict_seasons: Iterable[int],
) -> pd.DataFrame:
    """Pregame predictions for every row in ``predict_seasons``, indexed like ``rows``."""
    predict_seasons = list(predict_seasons)
    target = rows["rate"].to_numpy(dtype=float)
    penalty = design.penalty(config)
    subject_mask = design.block_mask(SUBJECT_NEUTRAL_BLOCKS)
    defense_mask = design.block_mask(DEFENSE_NEUTRAL_BLOCKS)
    season = rows["season"].to_numpy(dtype=int)
    week = rows["week"].to_numpy(dtype=int)
    in_scope = np.isin(season, predict_seasons)
    full = np.full(len(rows), np.nan)
    subject_neutral = np.full(len(rows), np.nan)
    defense_neutral = np.full(len(rows), np.nan)

    for pred_season, pred_week in sorted(set(zip(season[in_scope], week[in_scope]))):
        weights = _row_weights(rows, pred_season, pred_week, config)
        if not (weights > 0).any():
            continue
        beta = solve_ridge(design.matrix, target, weights, penalty)
        index = np.flatnonzero((season == pred_season) & (week == pred_week))
        current = design.matrix[index]
        full[index] = current @ beta
        subject_neutral[index] = current @ np.where(subject_mask, beta, 0.0)
        defense_neutral[index] = current @ np.where(defense_mask, beta, 0.0)

    return pd.DataFrame(
        {
            "pred_full": full[in_scope],
            "pred_subject_neutral": subject_neutral[in_scope],
            "pred_defense_neutral": defense_neutral[in_scope],
        },
        index=rows.index[in_scope],
    )


def fit_coefficients(
    rows: pd.DataFrame,
    design: DesignMatrix,
    config: RidgeConfig,
    pred_season: int,
    pred_week: int,
) -> pd.DataFrame:
    weights = _row_weights(rows, pred_season, pred_week, config)
    beta = solve_ridge(design.matrix, rows["rate"].to_numpy(dtype=float), weights, design.penalty(config))
    exposure = np.asarray(abs(design.matrix).T @ weights).ravel()
    return pd.DataFrame(
        {"block": design.blocks, "label": design.labels, "coefficient": beta, "weighted_exposure": exposure}
    )


def weekly_coefficients(rows: pd.DataFrame, design: DesignMatrix, config: RidgeConfig) -> pd.DataFrame:
    """Estimate every observed league week using strictly earlier games."""
    snapshots = []
    intercept_index = np.flatnonzero(design.blocks == "intercept")[0]
    positions = rows.drop_duplicates("subject").set_index("subject")["position"]
    for season, week in sorted(set(zip(rows["season"], rows["week"]))):
        weights = _row_weights(rows, int(season), int(week), config)
        if not (weights > 0).any():
            continue
        beta = solve_ridge(design.matrix, rows["rate"].to_numpy(dtype=float), weights, design.penalty(config))
        exposure = np.asarray(abs(design.matrix).T @ weights).ravel()
        snapshot = pd.DataFrame({
            "block": design.blocks,
            "label": design.labels,
            "coefficient": beta,
            "weighted_exposure": exposure,
        })
        snapshot = snapshot[(snapshot["weighted_exposure"] > 0) | (snapshot["block"] == "intercept")].copy()
        snapshot["estimate"] = snapshot["coefficient"]
        entity = snapshot["block"].isin(["subject", "defense", "offense_team", "teammate"])
        snapshot.loc[entity, "estimate"] += beta[intercept_index]
        position_effect = dict(zip(design.labels[design.blocks == "position"], beta[design.blocks == "position"]))
        player = snapshot["block"] == "subject"
        if "position" in set(design.blocks):
            snapshot.loc[player, "estimate"] += snapshot.loc[player, "label"].map(positions).map(position_effect).fillna(0)
        snapshot["season"] = int(season)
        snapshot["week"] = int(week)
        snapshots.append(snapshot)
    return pd.concat(snapshots, ignore_index=True) if snapshots else pd.DataFrame(
        columns=["block", "label", "coefficient", "weighted_exposure", "estimate", "season", "week"]
    )


def actual_output_frame(rows: pd.DataFrame, side: Literal["subject", "defense"]) -> pd.DataFrame:
    """Observed per-entity-game rate for the player (subject) or defense output."""
    key = "subject" if side == "subject" else "defteam"
    grouped = (
        rows.groupby([key, "game_id"], sort=False)
        .agg(
            game_date=("game_date", "first"),
            season=("season", "first"),
            week=("week", "first"),
            value_sum=("value_sum", "sum"),
            count=("count", "sum"),
        )
        .reset_index()
        .rename(columns={key: "entity"})
    )
    grouped["rate"] = grouped["value_sum"] / grouped["count"]
    return grouped.sort_values(["entity", "game_date", "game_id"]).reset_index(drop=True)


def aggregate_output_predictions(
    rows: pd.DataFrame,
    predictions: pd.DataFrame,
    side: Literal["subject", "defense"],
) -> pd.DataFrame:
    """Count-weighted roll-up of row predictions to entity-game predictions."""
    key = "subject" if side == "subject" else "defteam"
    neutral_col = "pred_subject_neutral" if side == "subject" else "pred_defense_neutral"
    frame = rows.loc[predictions.index, [key, "game_id", "count"]].join(predictions[["pred_full", neutral_col]])
    frame = frame.dropna(subset=["pred_full", neutral_col])
    frame["full_weighted"] = frame["pred_full"] * frame["count"]
    frame["neutral_weighted"] = frame[neutral_col] * frame["count"]
    grouped = (
        frame.groupby([key, "game_id"], sort=False)
        .agg(count=("count", "sum"), full_weighted=("full_weighted", "sum"), neutral_weighted=("neutral_weighted", "sum"))
        .reset_index()
        .rename(columns={key: "entity"})
    )
    grouped["pred_full"] = grouped["full_weighted"] / grouped["count"]
    grouped["pred_neutral"] = grouped["neutral_weighted"] / grouped["count"]
    return grouped[["entity", "game_id", "pred_full", "pred_neutral"]]
