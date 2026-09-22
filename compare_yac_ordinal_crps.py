import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import nfl_data_py as nfl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss, corn_loss
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping

from feature_config_utils import (
    CONFIG_FILENAME,
    build_feature_catalog,
    ensure_expanded_columns,
    load_feature_config,
    resolve_model_categorical_cols,
    resolve_model_features,
)

BASE_DIR = Path("C:/Users/steve/OneDrive/Desktop")
CACHE_DIR = BASE_DIR / "Writing/WR Clustering"
FEATURE_CONFIG_PATH = Path(__file__).resolve().parent / CONFIG_FILENAME

GAME_STATE_COLS = [
    "defteam_score",
    "down",
    "game_seconds_remaining",
    "posteam_score",
    "posteam_type",
    "yardline_100",
    "ydstogo",
    "roof",
    "score_differential",
]
PASS_COLUMNS = ["air_yards"]
DEF_COLS = [
    "deep_cpoe_allowed",
    "int_rate",
    "medium_cpoe_allowed",
    "short_cpoe_allowed",
    "yac_oe_allowed",
    "sack_rate",
    "rush_yards_per_carry",
    "def_rush_epa_rate",
    "def_pass_epa_rate",
    "def_winning_rate",
    "def_losing_rate",
    "def_tied_rate",
    "passing_massey_defense",
    "rushing_massey_defense",
]
TEAM_STATS_COLUMNS = [
    "losing_rate",
    "tied_rate",
    "team_4th_rate",
    "winning_rate",
    "rushing_epa_rate",
    "epa_rate",
    "passing_massey_rating",
    "passing_massey_offense",
    "rushing_massey_rating",
    "rushing_massey_offense",
    "opp_def_pass_epa_rate",
    "opp_def_rush_epa_rate",
    "opp_pass_epa_rate",
    "opp_rushing_epa_rate",
]
TEAM_TENDENCY_COLUMNS = [
    "pass_oe",
    "pass_oe_weekly_variation",
    "target_gini_roll4",
    "rb_carry_split_gini_roll4",
    "rb_top_carry_share_roll4",
    "pace_of_play_bayes",
]
RECEIVER_COLUMNS = ["deep", "medium", "short", "fd_bayes", "rz_bayes", "yac_bayes", "adot", "ypt"]
PASSER_COLUMNS = [
    "cpoe_bayes",
    "turnover_bayes",
    "sack_bayes",
    "cpoe_short",
    "cpoe_medium",
    "cpoe_deep",
    "deep_pass_rate",
    "medium_pass_rate",
    "short_pass_rate",
    "scramble_rate",
]
QB_CONTEXT_COLUMNS = ["qb_team_games_played", "qb_dome_rate_bayes"]
SUPPORTED_MODELS = ("corn", "condor", "ce", "crps", "soft", "xgb")
MODEL_ALIASES = {"xgboost": "xgb", "xgb": "xgb"}


class CornModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        dropout_rate = float(dropout_rate)
        self.fc1 = nn.Linear(input_dim, 256)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 64)
        self.drop3 = nn.Dropout(dropout_rate)
        self.out = nn.Linear(64, output_dim - 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.drop1(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.1)
        x = self.drop2(x)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.1)
        x = self.drop3(x)
        return self.out(x)


class OrdinalSoftmaxModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        dropout_rate = float(dropout_rate)
        self.fc1 = nn.Linear(input_dim, 256)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 64)
        self.drop3 = nn.Dropout(dropout_rate)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.drop1(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.1)
        x = self.drop2(x)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.1)
        x = self.drop3(x)
        return self.out(x)


class MyCornLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = int(num_classes)

    def forward(self, y_pred, y_true):
        return corn_loss(y_pred, y_true, self.num_classes)


class CondorCoralLoss(nn.Module):
    """CORAL cumulative-link loss used as a CONDOR-style ordinal comparator."""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = int(num_classes)

    def forward(self, logits, y_true):
        levels = levels_from_labelbatch(y_true, num_classes=self.num_classes).to(logits.device)
        return coral_loss(logits, levels)


class CondorModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        dropout_rate = float(dropout_rate)
        self.fc1 = nn.Linear(input_dim, 256)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 64)
        self.drop3 = nn.Dropout(dropout_rate)
        self.out = CoralLayer(size_in=64, num_classes=output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.drop1(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.1)
        x = self.drop2(x)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.1)
        x = self.drop3(x)
        return self.out(x)


class CRPSOrdinalLoss(nn.Module):
    """Discrete CRPS (RPS form) for ordered classes indexed [0, K-1]."""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = int(num_classes)

    def forward(self, logits, y_true):
        probs = torch.softmax(logits, dim=1)
        pred_cdf = torch.cumsum(probs[:, :-1], dim=1)
        thresholds = torch.arange(self.num_classes - 1, device=logits.device).unsqueeze(0)
        true_cdf = (y_true.unsqueeze(1) <= thresholds).float()
        return torch.mean((pred_cdf - true_cdf) ** 2)


class SoftLabelOrdinalCrossEntropyLoss(nn.Module):
    """Ordinal-aware soft-label CE using Gaussian-smoothed targets across class index."""

    def __init__(self, num_classes, sigma=1.5):
        super().__init__()
        self.num_classes = int(num_classes)
        self.sigma = float(sigma)
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0 for soft-label ordinal loss.")

    def forward(self, logits, y_true):
        class_idx = torch.arange(self.num_classes, device=logits.device, dtype=torch.float32).unsqueeze(0)
        y_float = y_true.unsqueeze(1).to(dtype=torch.float32)
        dist2 = (class_idx - y_float) ** 2
        soft_target = torch.exp(-dist2 / (2.0 * (self.sigma ** 2)))
        soft_target = soft_target / soft_target.sum(dim=1, keepdim=True).clamp_min(1e-12)
        log_probs = F.log_softmax(logits, dim=1)
        return -(soft_target * log_probs).sum(dim=1).mean()


def corn_logits_to_class_probs(logits):
    probas = torch.sigmoid(torch.from_numpy(logits))
    probas = torch.cumprod(probas, dim=1).float().detach().numpy()
    num_classes = probas.shape[1] + 1

    class_probs = [1.0 - probas[:, 0]]
    if num_classes > 2:
        for val in range(1, num_classes - 1):
            class_probs.append(probas[:, val - 1] - probas[:, val])
    class_probs.append(probas[:, num_classes - 2])
    return np.array(class_probs).T


def condor_logits_to_class_probs(logits):
    survival = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    num_classes = survival.shape[1] + 1

    class_probs = [1.0 - survival[:, 0]]
    if num_classes > 2:
        for val in range(1, num_classes - 1):
            class_probs.append(survival[:, val - 1] - survival[:, val])
    class_probs.append(survival[:, num_classes - 2])

    probs = np.array(class_probs, dtype=np.float64).T
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def to_dense_float32(matrix):
    if issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)
    matrix[np.isnan(matrix)] = 0.0
    return matrix


def stable_softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def ordinal_crps_score(y_true_values, class_values, probs):
    """
    CRPS for a discrete ordered support using CDF errors across class cutpoints.

    This is the Ranked Probability Score form of CRPS for ordinal classes.
    """
    y_true = np.asarray(y_true_values, dtype=np.float64)
    classes = np.asarray(class_values, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)

    if probs.shape[1] != len(classes):
        raise ValueError("Probability width and class_values length mismatch.")

    if len(classes) < 2:
        return np.zeros(len(y_true), dtype=np.float64)

    thresholds = classes[:-1]
    pred_cdf = np.cumsum(probs[:, :-1], axis=1)
    true_cdf = (y_true[:, None] <= thresholds[None, :]).astype(np.float64)
    return np.mean((pred_cdf - true_cdf) ** 2, axis=1)


def discrete_pit_values(y_true_values, class_values, probs):
    """PIT-like values for a discrete support, using F(y) = P(Y <= y)."""
    y_true = np.asarray(y_true_values, dtype=np.float64)
    classes = np.asarray(class_values, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)

    indicator = (classes[None, :] <= y_true[:, None]).astype(np.float64)
    return np.sum(probs * indicator, axis=1)


def save_calibration_chart(pit_by_model, output_path, bins=10):
    model_names = list(pit_by_model.keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(4 * len(model_names), 3.5), sharey=True)
    if len(model_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        vals = np.asarray(pit_by_model[name], dtype=np.float64)
        vals = np.clip(vals, 0.0, 1.0)
        ax.hist(vals, bins=bins, range=(0.0, 1.0), density=True, alpha=0.7, color="#3366AA", edgecolor="white")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("F(y)")
        ax.set_xlim(0.0, 1.0)

    axes[0].set_ylabel("Density")
    fig.suptitle("Ordinal Calibration (PIT-style histogram)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_model_selection(raw_models):
    selected = []
    for item in str(raw_models).split(","):
        candidate = item.strip().lower()
        if not candidate:
            continue
        normalized = MODEL_ALIASES.get(candidate, candidate)
        if normalized not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{candidate}'. Supported models: {list(SUPPORTED_MODELS)}"
            )
        selected.append(normalized)

    if not selected:
        raise ValueError("Model selection cannot be empty.")

    return tuple(dict.fromkeys(selected))


def net_forward_numpy(net, X):
    outputs = net.forward(X)
    if torch.is_tensor(outputs):
        return outputs.detach().cpu().numpy()
    return np.asarray(outputs)


def build_preprocessor(feature_columns, categorical_cols):
    categorical_cols = [col for col in categorical_cols if col in feature_columns]
    numerical_cols = [col for col in feature_columns if col not in categorical_cols]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])
    return preprocessor


def _build_next_game_wide_tables(player_stats, team_stats):
    player_stats_wide = (
        player_stats
        .pivot_table(index=["player_id", "player_name", "game_id", "game_date"], columns=["source"], values="average")
        .reset_index()
    )
    player_stats_wide = player_stats_wide.sort_values(by=["player_id", "game_date"])
    player_stats_wide = (
        player_stats_wide
        .groupby("player_id", group_keys=False)
        .apply(lambda g: g.ffill())
    )
    player_stats_wide["next_game_id"] = (
        player_stats_wide
        .groupby("player_id")["game_id"]
        .shift(-1)
    )
    player_stats_wide.set_index(["player_id", "next_game_id"], inplace=True, drop=True)

    team_stats_wide = (
        team_stats
        .pivot_table(index=["team", "game_id", "game_date"], columns=["source"], values="average")
        .reset_index()
    )
    team_stats_wide = team_stats_wide.sort_values(by=["team", "game_date"])
    team_stats_wide = team_stats_wide.ffill()
    team_stats_wide["next_game_id"] = (
        team_stats_wide
        .groupby("team")["game_id"]
        .shift(-1)
    )
    team_stats_wide.set_index(["team", "next_game_id"], inplace=True, drop=True)

    return player_stats_wide, team_stats_wide


def _merge_rest_days(pbp):
    team_schedule = (
        pd.concat([
            pbp.groupby(["posteam", "game_id"])["game_date"].first().reset_index().rename(columns={"posteam": "team"}),
            pbp.groupby(["defteam", "game_id"])["game_date"].first().reset_index().rename(columns={"defteam": "team"}),
        ])
        .drop_duplicates(subset=["team", "game_id"])
    )
    team_schedule["game_date"] = pd.to_datetime(team_schedule["game_date"])
    team_schedule = team_schedule.sort_values(["team", "game_date"])
    team_schedule["days_rest"] = (
        team_schedule.groupby("team")["game_date"]
        .diff()
        .dt.days
        .fillna(7)
    )
    team_schedule = team_schedule.set_index(["team", "game_id"])

    pbp = pbp.merge(team_schedule[["days_rest"]], left_on=["posteam", "game_id"], right_index=True, how="left")
    pbp = pbp.merge(
        team_schedule[["days_rest"]].rename(columns={"days_rest": "opp_days_rest"}),
        left_on=["defteam", "game_id"],
        right_index=True,
        how="left",
    )
    return pbp


def load_yac_dataset(seasons):
    player_stats = pd.read_csv(str(BASE_DIR / "player_stats.csv"))
    team_stats = pd.read_csv(str(BASE_DIR / "Team_Stats.csv"))
    player_stats["average"] = player_stats["bayes_estimate_mean"].combine_first(player_stats["average"])
    team_stats["average"] = team_stats["bayes_estimate_mean"].combine_first(team_stats["average"])

    ids = nfl.import_ids()
    id_to_name = dict(zip(ids["gsis_id"], ids["name"]))
    player_stats["player_name"] = player_stats["player_id"].map(id_to_name)

    player_stats["game_date"] = pd.to_datetime(player_stats["game_date"])
    team_stats["game_date"] = pd.to_datetime(team_stats["game_date"])

    player_positions = nfl.import_players()[["gsis_id", "position"]].set_index("gsis_id", drop=True)
    player_positions["position"] = player_positions["position"].replace({"FB": "RB"})
    player_positions = player_positions[player_positions["position"].isin(["RB", "WR", "TE", "QB"])]

    player_stats_wide, team_stats_wide = _build_next_game_wide_tables(player_stats, team_stats)

    pbp = nfl.import_pbp_data(seasons, cache=True, alt_path=str(CACHE_DIR))
    pbp = pbp[pbp["aborted_play"] == 0].copy()

    pbp = pbp.merge(
        player_stats_wide.reindex(columns=RECEIVER_COLUMNS),
        left_on=["receiver_player_id", "game_id"],
        right_index=True,
        how="left",
    )
    pbp = pbp.merge(
        team_stats_wide.reindex(columns=DEF_COLS),
        left_on=["defteam", "game_id"],
        right_index=True,
        how="left",
    )
    pbp = pbp.merge(
        team_stats_wide.reindex(columns=TEAM_STATS_COLUMNS),
        left_on=["posteam", "game_id"],
        right_index=True,
        how="left",
    )
    pbp = pbp.merge(
        player_stats_wide.reindex(columns=PASSER_COLUMNS),
        left_on=["passer_player_id", "game_id"],
        right_index=True,
        how="left",
    )
    pbp = pbp.merge(
        team_stats_wide.reindex(columns=TEAM_TENDENCY_COLUMNS),
        left_on=["posteam", "game_id"],
        right_index=True,
        how="left",
    )

    # Harmonize pass_oe name after merge collisions.
    if "pass_oe" not in pbp.columns:
        if "pass_oe_x" in pbp.columns and "pass_oe_y" in pbp.columns:
            pbp["pass_oe"] = pbp["pass_oe_y"].combine_first(pbp["pass_oe_x"])
        elif "pass_oe_y" in pbp.columns:
            pbp["pass_oe"] = pbp["pass_oe_y"]
        elif "pass_oe_x" in pbp.columns:
            pbp["pass_oe"] = pbp["pass_oe_x"]

    pbp = pbp.merge(
        player_stats_wide.reindex(columns=QB_CONTEXT_COLUMNS),
        left_on=["passer_player_id", "game_id"],
        right_index=True,
        how="left",
    )

    pbp = _merge_rest_days(pbp)

    pbp_yac = pbp[pbp["season_type"] == "REG"].copy()
    pbp_yac = pbp_yac[pbp_yac["yards_after_catch"].notnull() & pbp_yac["air_yards"].notnull()]
    pbp_yac = pbp_yac[pbp_yac["pass_attempt"] == 1].copy()
    pbp_yac = pbp_yac.merge(player_positions, left_on="receiver_player_id", right_index=True, how="left")

    pbp_yac["game_date"] = pd.to_datetime(pbp_yac["game_date"])
    return pbp_yac


def fit_status_quo_corn(X_train, y_train_idx, num_classes, device, max_epochs, batch_size, seed, dropout_rate):
    net = NeuralNetClassifier(
        module=CornModel,
        module__input_dim=X_train.shape[1],
        module__output_dim=num_classes,
        module__dropout_rate=dropout_rate,
        max_epochs=max_epochs,
        batch_size=batch_size,
        optimizer=torch.optim.AdamW,
        criterion=MyCornLoss(num_classes),
        callbacks=[EarlyStopping(patience=20, load_best=True)],
        iterator_train__shuffle=True,
        device=device,
        verbose=1,
    )
    net.set_params(train_split=net.train_split)
    torch.manual_seed(seed)
    net.fit(X_train, y_train_idx)
    return net


def fit_condor(X_train, y_train_idx, num_classes, device, max_epochs, batch_size, seed, dropout_rate):
    net = NeuralNetClassifier(
        module=CondorModel,
        module__input_dim=X_train.shape[1],
        module__output_dim=num_classes,
        module__dropout_rate=dropout_rate,
        max_epochs=max_epochs,
        batch_size=batch_size,
        optimizer=torch.optim.AdamW,
        criterion=CondorCoralLoss(num_classes),
        callbacks=[EarlyStopping(patience=20, load_best=True)],
        iterator_train__shuffle=True,
        device=device,
        verbose=1,
    )
    torch.manual_seed(seed)
    net.fit(X_train, y_train_idx)
    return net


def fit_crps_ordinal(X_train, y_train_idx, num_classes, device, max_epochs, batch_size, seed, dropout_rate):
    net = NeuralNetClassifier(
        module=OrdinalSoftmaxModel,
        module__input_dim=X_train.shape[1],
        module__output_dim=num_classes,
        module__dropout_rate=dropout_rate,
        max_epochs=max_epochs,
        batch_size=batch_size,
        optimizer=torch.optim.AdamW,
        criterion=CRPSOrdinalLoss(num_classes),
        callbacks=[EarlyStopping(patience=20, load_best=True)],
        iterator_train__shuffle=True,
        device=device,
        verbose=1,
    )
    torch.manual_seed(seed)
    net.fit(X_train, y_train_idx)
    return net


def fit_ce_ordinal(X_train, y_train_idx, num_classes, device, max_epochs, batch_size, seed, dropout_rate):
    net = NeuralNetClassifier(
        module=OrdinalSoftmaxModel,
        module__input_dim=X_train.shape[1],
        module__output_dim=num_classes,
        module__dropout_rate=dropout_rate,
        max_epochs=max_epochs,
        batch_size=batch_size,
        optimizer=torch.optim.AdamW,
        criterion=nn.CrossEntropyLoss,
        callbacks=[EarlyStopping(patience=20, load_best=True)],
        iterator_train__shuffle=True,
        device=device,
        verbose=1,
    )
    torch.manual_seed(seed)
    net.fit(X_train, y_train_idx)
    return net


def fit_soft_label_ordinal(X_train, y_train_idx, num_classes, device, max_epochs, batch_size, seed, sigma, dropout_rate):
    net = NeuralNetClassifier(
        module=OrdinalSoftmaxModel,
        module__input_dim=X_train.shape[1],
        module__output_dim=num_classes,
        module__dropout_rate=dropout_rate,
        max_epochs=max_epochs,
        batch_size=batch_size,
        optimizer=torch.optim.AdamW,
        criterion=SoftLabelOrdinalCrossEntropyLoss(num_classes, sigma=sigma),
        callbacks=[EarlyStopping(patience=20, load_best=True)],
        iterator_train__shuffle=True,
        device=device,
        verbose=1,
    )
    torch.manual_seed(seed)
    net.fit(X_train, y_train_idx)
    return net


def fit_xgboost_ordinal(X_train, y_train_idx, num_classes, seed):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for the XGBoost comparison model. Install it with: pip install xgboost"
        ) from exc

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        learning_rate=0.05,
        max_depth=8,
        n_estimators=600,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        eval_metric="merror",
    )
    model.fit(X_train, y_train_idx)
    return model


def main():
    parser = argparse.ArgumentParser(description="Compare YAC ordinal modeling approaches on 2025 CRPS.")
    parser.add_argument("--train-start", type=int, default=2016)
    parser.add_argument("--train-end", type=int, default=2024)
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--soft-label-sigma", type=float, default=1.5)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--output-prefix", type=str, default="yac_ordinal_crps_2025")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(SUPPORTED_MODELS),
        help="Comma-separated model subset: corn,condor,ce,crps,soft,xgb (xgboost also accepted)",
    )
    args = parser.parse_args()
    selected_models = parse_model_selection(args.models)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    seasons = list(range(args.train_start, args.test_season + 1))
    print(f"Loading PBP + stats for seasons {seasons}...")
    yac_frame = load_yac_dataset(seasons)

    feature_catalog = build_feature_catalog(
        {
            "game_state_cols": GAME_STATE_COLS,
            "pass_columns": PASS_COLUMNS,
            "def_cols": DEF_COLS,
            "team_stats_columns": TEAM_STATS_COLUMNS,
            "team_tendency_columns": TEAM_TENDENCY_COLUMNS,
            "receiver_columns": RECEIVER_COLUMNS,
            "passer_columns": PASSER_COLUMNS,
            "qb_context_columns": QB_CONTEXT_COLUMNS,
        }
    )
    feature_config = load_feature_config(FEATURE_CONFIG_PATH, feature_catalog)
    feature_config = ensure_expanded_columns(feature_config, feature_catalog)

    model_name = "yac"
    feature_columns = resolve_model_features(feature_config, model_name, feature_catalog)
    categorical_cols = resolve_model_categorical_cols(feature_config, model_name)

    train_mask = (yac_frame["season"] >= args.train_start) & (yac_frame["season"] <= args.train_end)
    test_mask = yac_frame["season"] == args.test_season

    train_frame = yac_frame.loc[train_mask].copy()
    test_frame = yac_frame.loc[test_mask].copy()

    if train_frame.empty:
        raise ValueError("No training rows found for the requested train season range.")
    if test_frame.empty:
        raise ValueError("No test rows found for the requested test season.")

    y_train_values = pd.to_numeric(train_frame["yards_after_catch"], errors="coerce").to_numpy(dtype=np.float64)
    y_test_values = pd.to_numeric(test_frame["yards_after_catch"], errors="coerce").to_numpy(dtype=np.float64)

    class_values = np.array(sorted(pd.unique(y_train_values)), dtype=np.float64)
    class_to_index = {value: idx for idx, value in enumerate(class_values)}

    missing_train = np.sum(~np.isin(y_train_values, class_values))
    if int(missing_train) > 0:
        raise ValueError("Training target contains unexpected values after class extraction.")

    y_train_idx = np.array([class_to_index[val] for val in y_train_values], dtype=np.int64)

    X_train = train_frame.reindex(columns=feature_columns).copy()
    X_test = test_frame.reindex(columns=feature_columns).copy()

    preprocessor = build_preprocessor(feature_columns, categorical_cols)
    X_train_processed = to_dense_float32(preprocessor.fit_transform(X_train))
    X_test_processed = to_dense_float32(preprocessor.transform(X_test))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    print(f"Train rows: {len(train_frame):,}; test rows ({args.test_season}): {len(test_frame):,}; classes: {len(class_values)}")

    model_probs = {}

    if "corn" in selected_models:
        corn_net = fit_status_quo_corn(
            X_train_processed,
            y_train_idx,
            num_classes=len(class_values),
            device=device,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            dropout_rate=args.dropout_rate,
        )
        corn_logits = net_forward_numpy(corn_net, X_test_processed)
        model_probs["corn"] = corn_logits_to_class_probs(corn_logits)

    if "condor" in selected_models:
        condor_net = fit_condor(
            X_train_processed,
            y_train_idx,
            num_classes=len(class_values),
            device=device,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            dropout_rate=args.dropout_rate,
        )
        condor_logits = net_forward_numpy(condor_net, X_test_processed)
        model_probs["condor"] = condor_logits_to_class_probs(condor_logits)

    if "ce" in selected_models:
        ce_net = fit_ce_ordinal(
            X_train_processed,
            y_train_idx,
            num_classes=len(class_values),
            device=device,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            dropout_rate=args.dropout_rate,
        )
        ce_logits = net_forward_numpy(ce_net, X_test_processed)
        model_probs["ce"] = stable_softmax(ce_logits)

    if "crps" in selected_models:
        crps_net = fit_crps_ordinal(
            X_train_processed,
            y_train_idx,
            num_classes=len(class_values),
            device=device,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            dropout_rate=args.dropout_rate,
        )
        crps_logits = net_forward_numpy(crps_net, X_test_processed)
        model_probs["crps"] = stable_softmax(crps_logits)

    if "soft" in selected_models:
        soft_label_net = fit_soft_label_ordinal(
            X_train_processed,
            y_train_idx,
            num_classes=len(class_values),
            device=device,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            sigma=args.soft_label_sigma,
            dropout_rate=args.dropout_rate,
        )
        soft_label_logits = net_forward_numpy(soft_label_net, X_test_processed)
        model_probs["soft"] = stable_softmax(soft_label_logits)

    if "xgb" in selected_models:
        xgb_model = fit_xgboost_ordinal(
            X_train_processed,
            y_train_idx,
            num_classes=len(class_values),
            seed=args.seed,
        )
        model_probs["xgb"] = np.asarray(xgb_model.predict_proba(X_test_processed), dtype=np.float64)

    for key in list(model_probs.keys()):
        probs = np.clip(model_probs[key], 1e-12, 1.0)
        model_probs[key] = probs / probs.sum(axis=1, keepdims=True)

    crps_per_play_by_model = {
        key: ordinal_crps_score(y_test_values, class_values, probs)
        for key, probs in model_probs.items()
    }

    summary = {
        "train_start": int(args.train_start),
        "train_end": int(args.train_end),
        "test_season": int(args.test_season),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "num_classes": int(len(class_values)),
        "models": list(selected_models),
        "soft_label_sigma": float(args.soft_label_sigma),
        "dropout_rate": float(args.dropout_rate),
    }

    metric_name_map = {
        "corn": "corn_status_quo_crps",
        "condor": "condor_coral_crps",
        "ce": "ce_softmax_crps",
        "crps": "crps_trained_model_crps",
        "soft": "soft_label_ordinal_crps",
        "xgb": "xgb_ordinal_crps",
    }
    for model_key, metric_name in metric_name_map.items():
        if model_key in crps_per_play_by_model:
            summary[metric_name] = float(np.mean(crps_per_play_by_model[model_key]))

    if "corn_status_quo_crps" in summary and "ce_softmax_crps" in summary:
        summary["ce_delta_vs_corn"] = summary["ce_softmax_crps"] - summary["corn_status_quo_crps"]
    if "corn_status_quo_crps" in summary and "condor_coral_crps" in summary:
        summary["condor_delta_vs_corn"] = summary["condor_coral_crps"] - summary["corn_status_quo_crps"]
    if "corn_status_quo_crps" in summary and "crps_trained_model_crps" in summary:
        summary["crps_delta_crps_minus_corn"] = (
            summary["crps_trained_model_crps"] - summary["corn_status_quo_crps"]
        )
    if "corn_status_quo_crps" in summary and "soft_label_ordinal_crps" in summary:
        summary["soft_label_delta_vs_corn"] = (
            summary["soft_label_ordinal_crps"] - summary["corn_status_quo_crps"]
        )

    compare_frame = pd.DataFrame(
        {
            "game_id": test_frame["game_id"].to_numpy(),
            "week": test_frame["week"].to_numpy(),
            "posteam": test_frame["posteam"].to_numpy(),
            "defteam": test_frame["defteam"].to_numpy(),
            "yards_after_catch": y_test_values,
        }
    )

    per_play_col_map = {
        "corn": "corn_crps",
        "condor": "condor_crps",
        "ce": "ce_crps",
        "crps": "crps_model_crps",
        "soft": "soft_label_crps",
        "xgb": "xgb_crps",
    }
    for model_key, col_name in per_play_col_map.items():
        if model_key in crps_per_play_by_model:
            compare_frame[col_name] = crps_per_play_by_model[model_key]

    if "corn_crps" in compare_frame.columns and "ce_crps" in compare_frame.columns:
        compare_frame["ce_gain_vs_corn"] = compare_frame["corn_crps"] - compare_frame["ce_crps"]
    if "corn_crps" in compare_frame.columns and "condor_crps" in compare_frame.columns:
        compare_frame["condor_gain_vs_corn"] = compare_frame["corn_crps"] - compare_frame["condor_crps"]
    if "corn_crps" in compare_frame.columns and "crps_model_crps" in compare_frame.columns:
        compare_frame["crps_gain"] = compare_frame["corn_crps"] - compare_frame["crps_model_crps"]
    if "corn_crps" in compare_frame.columns and "soft_label_crps" in compare_frame.columns:
        compare_frame["soft_label_gain_vs_corn"] = compare_frame["corn_crps"] - compare_frame["soft_label_crps"]
    if "corn_crps" in compare_frame.columns and "xgb_crps" in compare_frame.columns:
        compare_frame["xgb_gain_vs_corn"] = compare_frame["corn_crps"] - compare_frame["xgb_crps"]

    agg_map = {"plays": ("yards_after_catch", "size")}
    for col in [
        "ce_crps",
        "corn_crps",
        "condor_crps",
        "crps_model_crps",
        "soft_label_crps",
        "xgb_crps",
        "condor_gain_vs_corn",
        "ce_gain_vs_corn",
        "crps_gain",
        "soft_label_gain_vs_corn",
        "xgb_gain_vs_corn",
    ]:
        if col in compare_frame.columns:
            agg_map[col] = (col, "mean")

    by_week = (
        compare_frame
        .groupby("week", as_index=False)
        .agg(**agg_map)
        .sort_values("week")
    )

    pit_name_map = {
        "corn": "CORN",
        "condor": "CONDOR (CORAL)",
        "ce": "CE Softmax",
        "crps": "CRPS Loss",
        "soft": "Soft Labels",
        "xgb": "XGBoost",
    }
    pit_by_model = {
        pit_name_map[key]: discrete_pit_values(y_test_values, class_values, probs)
        for key, probs in model_probs.items()
    }

    summary_path = Path(f"{args.output_prefix}_summary.json")
    per_play_path = Path(f"{args.output_prefix}_per_play.csv")
    weekly_path = Path(f"{args.output_prefix}_by_week.csv")
    preprocessor_path = Path(f"{args.output_prefix}_preprocessor.pkl")
    calibration_chart_path = Path(f"{args.output_prefix}_calibration.png")

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    compare_frame.to_csv(per_play_path, index=False)
    by_week.to_csv(weekly_path, index=False)
    joblib.dump(preprocessor, preprocessor_path)
    save_calibration_chart(pit_by_model, calibration_chart_path)

    print("\nCRPS comparison complete")
    print(json.dumps(summary, indent=2))
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {per_play_path}")
    print(f"Wrote: {weekly_path}")
    print(f"Wrote: {calibration_chart_path}")


if __name__ == "__main__":
    main()
