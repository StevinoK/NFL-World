import argparse
import builtins
import copy
import json
import math
import random
from time import perf_counter
from collections import deque
from pathlib import Path

import joblib
import matplotlib

# plt.show() blocks on a GUI backend until the window is closed, which hangs unattended runs.
RUNNING_INTERACTIVE = hasattr(builtins, "get_ipython")
if not RUNNING_INTERACTIVE:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nfl_data_py as nfl
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from project_paths import DEFAULT_ARTIFACT_DIR, REPOSITORY_DIR
from coral_pytorch.losses import corn_loss
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from skorch import NeuralNetClassifier
import skorch.callbacks
from skorch.callbacks import Checkpoint, EarlyStopping, GradientNormClipping, LRScheduler
from model_reporting import ModelReportBuilder
from ngmm_training import train_ngmm_grouped

from model_architecture import (
    MODEL_ARTIFACT_PREFIX,
    CornModel,
    GameModel,
    HierarchicalLatentGameModel,
    arch_sidecar_path,
    normalize_arch,
    write_arch_sidecar,
)
from hyperparameter_tuning import (
    balanced_class_weights,
    build_optimizer,
    build_scheduler,
    default_hyperparameters,
    describe_hyperparameters,
    evaluation_metrics,
    load_best_hyperparameters,
    load_best_ngmm_hyperparameters,
    optimizer_class,
    optimizer_extra_kwargs,
    run_study,
    save_best_hyperparameters,
    suggest_hyperparameters,
    default_ngmm_hyperparameters,
    suggest_ngmm_hyperparameters,
    update_leaderboard,
)

from feature_config_utils import (
    CONFIG_FILENAME,
    build_feature_catalog,
    ensure_expanded_columns,
    expand_feature_groups,
    resolve_model_kl_weight,
    load_feature_config,
    resolve_model_latent_dim,
    resolve_model_categorical_cols,
    resolve_model_prior_enabled,
    resolve_model_prior_features,
    resolve_model_prior_groups,
    resolve_model_features,
    save_feature_config,
)

# Shared paths and defaults
BASE_DIR = DEFAULT_ARTIFACT_DIR
CACHE_DIR = REPOSITORY_DIR / "artifacts" / "cache"
DEFAULT_SEASON_START = 2017
DEFAULT_SEASON_END = 2024
SEASONS = list(range(DEFAULT_SEASON_START, DEFAULT_SEASON_END + 1))
FEATURE_CONFIG_PATH = Path(__file__).resolve().parent / CONFIG_FILENAME
PLAY_TYPE_BUCKET_CALIBRATION_CSV = BASE_DIR / "play_type_bucket_calibration.csv"
PLAY_TYPE_BUCKET_CALIBRATION_SUMMARY_JSON = BASE_DIR / "play_type_bucket_calibration_summary.json"
PLAY_TYPE_BUCKET_MIN_SUPPORT = 5
TRAINING_SEED = 42

TRAINABLE_MODEL_NAMES = [
    "yac",
    "air_yards",
    "sack",
    "scramble",
    "interception",
    "completion",
    "punt",
    "time_elapsed",
    "kick",
    "rush_yards",
    "scramble_yards",
    "play_type",
    "target_selection",
    "rusher_selection",
    "fumble",
    "defensive_return_yards",
    "penalty_no_play",
    "penalty_on_play",
    "penalty_yards",
    "penalty_first_down",
]
def parse_seasons(tokens):
    """Accept either explicit years ("2021 2022") or an inclusive range ("2021-2024")."""
    seasons = []
    for token in tokens:
        text = str(token).strip()
        if "-" in text:
            start_text, _, end_text = text.partition("-")
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid season range: {text}")
            seasons.extend(range(start, end + 1))
        else:
            seasons.append(int(text))
    seasons = sorted(dict.fromkeys(seasons))
    if not seasons:
        raise ValueError("--seasons must resolve to at least one season")
    return seasons


def parse_cli_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help=(
            "Models to retrain or tune. Use one or more names from: "
            + ", ".join(TRAINABLE_MODEL_NAMES)
            + ". Defaults to every model present in the feature config."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model names and exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=TRAINING_SEED,
        help="Random seed used for every model training run (default: 42).",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "tune"],
        default="train",
        help=(
            "train: retrain using the best hyperparameters from the last tuning run. "
            "tune: run an Optuna search and keep the best-validation-loss model (default: train)."
        ),
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=[f"{DEFAULT_SEASON_START}-{DEFAULT_SEASON_END}"],
        metavar="SEASON",
        help=(
            "Seasons to load play-by-play for, as years and/or inclusive ranges "
            f"(default: {DEFAULT_SEASON_START}-{DEFAULT_SEASON_END}). Narrow this when tuning."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of Optuna trials per model when --mode tune (default: 30).",
    )
    parser.add_argument(
        "--tuning-timeout",
        type=float,
        default=None,
        help="Optional per-model wall-clock cap in seconds for --mode tune.",
    )
    parser.add_argument(
        "--tuning-max-epochs",
        type=int,
        default=150,
        help="Epoch cap applied to every trial when --mode tune (default: 150).",
    )
    parser.add_argument(
        "--fresh-study",
        action="store_true",
        help="Delete any existing Optuna study for the selected models before tuning.",
    )
    parser.add_argument(
        "--export-play-type-data",
        action="store_true",
        help="Export the prepared play_type rows and configured features for ARMED training.",
    )
    parser.add_argument(
        "--export-play-type-data-only",
        action="store_true",
        help="Export play_type data and stop before model training.",
    )
    parser.add_argument(
        "--export-team-stats-wide-only",
        action="store_true",
        help="Regenerate team_stats_wide.csv and stop before model training.",
    )
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for model_name in TRAINABLE_MODEL_NAMES:
            print(f" - {model_name}")
        raise SystemExit(0)

    args.seasons = parse_seasons(args.seasons)
    if args.models is not None:
        args.models = list(dict.fromkeys(args.models))
        invalid = [model for model in args.models if model not in TRAINABLE_MODEL_NAMES]
        if invalid:
            invalid_text = ", ".join(invalid)
            valid_text = ", ".join(TRAINABLE_MODEL_NAMES)
            raise ValueError(f"Unknown model(s): {invalid_text}. Valid models: {valid_text}")
    return args


def resolve_selected_models(args, default_models):
    if args.models is not None:
        return list(args.models)
    return list(default_models)


def should_train_model(model_name, selected_models):
    return model_name in selected_models


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def should_use_hierarchical_model(feature_config, model_name):
    return resolve_model_prior_enabled(feature_config, model_name)


def validate_hierarchical_configuration(feature_config, selected_models):
    for model_name in selected_models:
        should_use_hierarchical_model(feature_config, model_name)


def out_path(filename):
    return str(BASE_DIR / filename)


def finish_plot():
    if RUNNING_INTERACTIVE:
        plt.show()
    else:
        plt.close("all")


def fill_nans_with_zero(train_matrix, test_matrix):
    train_matrix[np.isnan(train_matrix)] = 0
    test_matrix[np.isnan(test_matrix)] = 0
    return train_matrix, test_matrix


def blocked_train_test_split(frame, target, groups, test_size=0.2, random_state=42):
    groups_array = np.asarray(groups)
    if groups_array.size == 0 or len(np.unique(groups_array)) < 2:
        return train_test_split(frame, target, test_size=test_size, random_state=random_state)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(frame, target, groups=groups_array))

    frame_train = frame.iloc[train_idx]
    frame_test = frame.iloc[test_idx]
    if hasattr(target, "iloc"):
        target_train = target.iloc[train_idx]
        target_test = target.iloc[test_idx]
    else:
        target_train = target[train_idx]
        target_test = target[test_idx]
    return frame_train, frame_test, target_train, target_test


def unique_columns(columns):
    return list(dict.fromkeys(columns))


def filter_non_meaningful_games(frame):
    """Drop games where teams are not playing meaningful football.

    The common case is the final week of the regular season when large favorites can
    rest starters or otherwise meaningfully reduce effort. We exclude those game_ids
    before any model-specific training subset is built.
    """
    if frame is None or len(frame) == 0:
        return frame
    if "game_id" not in frame.columns:
        return frame

    game_rows = frame[["game_id", "week", "season_type", "spread_line", "score_differential"]].drop_duplicates(subset=["game_id"]).copy()
    if game_rows.empty:
        return frame

    season_type = game_rows.get("season_type", pd.Series("REG", index=game_rows.index)).astype(str).str.upper()
    week = pd.to_numeric(game_rows.get("week", pd.Series(np.nan, index=game_rows.index)), errors="coerce")
    spread = pd.to_numeric(game_rows.get("spread_line", pd.Series(np.nan, index=game_rows.index)), errors="coerce")
    score_diff = pd.to_numeric(game_rows.get("score_differential", pd.Series(np.nan, index=game_rows.index)), errors="coerce")

    weird_game_ids = game_rows.loc[
        (season_type == "REG")
        & (week == 18)
        & (
            (spread.abs() >= 10.0)
            | (score_diff.abs() >= 17.0)
        )
    , "game_id"].astype(str).tolist()

    if not weird_game_ids:
        return frame
    return frame[~frame["game_id"].astype(str).isin(weird_game_ids)].copy()


def _timing_checkpoint(label, start_time):
    now = perf_counter()
    print(f"[timing] {label}: {now - start_time:.3f}s")
    return now


def hierarchical_group_labels(frame, fallback_groups):
    fallback_series = pd.Series(np.asarray(fallback_groups), index=frame.index).astype(str)
    if "posteam" in frame.columns:
        game_series = frame["game_id"].astype(str) if "game_id" in frame.columns else fallback_series
        return game_series + "::" + frame["posteam"].fillna("UNKNOWN").astype(str)
    if "team" in frame.columns:
        game_series = frame["game_id"].astype(str) if "game_id" in frame.columns else fallback_series
        return game_series + "::" + frame["team"].fillna("UNKNOWN").astype(str)
    return fallback_series


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


class GroupBatchSampler(torch.utils.data.Sampler):
    def __init__(self, group_labels, batch_size, shuffle=True, drop_last=False, seed=TRAINING_SEED):
        self.group_labels = np.asarray(group_labels)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.group_to_indices = {}
        for idx, group in enumerate(self.group_labels):
            self.group_to_indices.setdefault(group, []).append(idx)

    def __iter__(self):
        group_keys = list(self.group_to_indices.keys())
        if self.shuffle:
            self.rng.shuffle(group_keys)

        batches = []
        for group_key in group_keys:
            group_indices = list(self.group_to_indices[group_key])
            if self.shuffle:
                self.rng.shuffle(group_indices)

            for start in range(0, len(group_indices), self.batch_size):
                batch = group_indices[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        if self.shuffle:
            self.rng.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return sum(len(indices) // self.batch_size for indices in self.group_to_indices.values())
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.group_to_indices.values())


class PackedGroupBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        group_labels,
        batch_size,
        shuffle=True,
        drop_last=False,
        max_groups_per_batch=4,
        max_group_fraction=0.7,
        seed=TRAINING_SEED,
    ):
        self.group_labels = np.asarray(group_labels)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.max_groups_per_batch = int(max_groups_per_batch)
        self.max_group_fraction = float(max_group_fraction)
        self.rng = np.random.default_rng(seed)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.max_groups_per_batch <= 0:
            raise ValueError("max_groups_per_batch must be > 0")
        if not (0.0 < self.max_group_fraction <= 1.0):
            raise ValueError("max_group_fraction must be in (0, 1]")

        self.group_to_indices = {}
        for idx, group in enumerate(self.group_labels):
            self.group_to_indices.setdefault(group, []).append(idx)
        self.total_samples = int(len(self.group_labels))

    def __iter__(self):
        group_queues = {}
        for group, indices in self.group_to_indices.items():
            buffered = list(indices)
            if self.shuffle:
                self.rng.shuffle(buffered)
            group_queues[group] = deque(buffered)

        active_groups = [group for group, queue in group_queues.items() if len(queue) > 0]
        per_group_cap = max(1, int(self.batch_size * self.max_group_fraction))

        while active_groups:
            if self.shuffle:
                self.rng.shuffle(active_groups)
            selected_groups = active_groups[: self.max_groups_per_batch]

            batch = []
            per_group_counts = {group: 0 for group in selected_groups}
            progressed = True

            # Round-robin fill across a small number of groups to preserve grouped structure
            # without creating tiny one-group-only batches.
            while len(batch) < self.batch_size and progressed:
                progressed = False
                for group in selected_groups:
                    if len(batch) >= self.batch_size:
                        break
                    if per_group_counts[group] >= per_group_cap:
                        continue
                    queue = group_queues[group]
                    if len(queue) == 0:
                        continue
                    batch.append(queue.popleft())
                    per_group_counts[group] += 1
                    progressed = True

            if len(batch) == self.batch_size:
                yield batch
            elif len(batch) > 0 and not self.drop_last:
                yield batch

            active_groups = [group for group in active_groups if len(group_queues[group]) > 0]

    def __len__(self):
        if self.drop_last:
            return self.total_samples // self.batch_size
        return math.ceil(self.total_samples / self.batch_size)
#%%
class MyCornLoss(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, y_pred_proba, y_true):
        return corn_loss(y_pred_proba, y_true,self.num_classes)
class HierarchicalGameModel(nn.Module):
    def __init__(self, input_dim, output_dim, n_games, game_embedding_dim=4):
        super().__init__()
        self.game_embedding_dim = game_embedding_dim

        # Game-level variation (random effect)
        self.game_embed = nn.Embedding(
            num_embeddings=n_games,
            embedding_dim=game_embedding_dim
        )

        combined_dim = input_dim + game_embedding_dim

        # Play-level network
        self.fc1 = nn.Linear(combined_dim, 256)
        self.drop1 = nn.Dropout(0.3)  # Match dropout rate
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.3)  # Match dropout rate
        self.fc3 = nn.Linear(128, 64)
        self.drop3 = nn.Dropout(0.3)  # Match dropout rate
        self.out = nn.Linear(64, output_dim) # Added a final layer
        # Buffer to store empirical covariance for simulation
        self.register_buffer('game_effect_covariance', torch.eye(game_embedding_dim))

    # ✅ Single-argument forward for Skorch
    def forward(self, X):
        X_features, X_game_idx = X

        # Lookup the game-level random effect
        game_effect = self.game_embed(X_game_idx)

        # Concatenate play-level and game-level info
        combined_input = torch.cat([X_features, game_effect], dim=1)

        # Forward pass
        x = F.leaky_relu(self.fc1(combined_input), negative_slope=0.1)
        x = self.drop1(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.1)
        x = self.drop2(x)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.1)
        x = self.drop3(x)
        return self.out(x)


def _prepare_latent_features(train_frame, test_frame, play_columns, prior_columns, categorical_cols):
    play_numerical_cols = [col for col in play_columns if col not in categorical_cols]
    play_preprocessor = ColumnTransformer([
        ('num', StandardScaler(), play_numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ])

    prior_preprocessor = ColumnTransformer([
        ('num', StandardScaler(), prior_columns),
    ])

    X_train_play = play_preprocessor.fit_transform(train_frame[play_columns])
    X_test_play = play_preprocessor.transform(test_frame[play_columns])
    X_train_prior = prior_preprocessor.fit_transform(train_frame[prior_columns])
    X_test_prior = prior_preprocessor.transform(test_frame[prior_columns])

    X_train_play[np.isnan(X_train_play)] = 0
    X_test_play[np.isnan(X_test_play)] = 0
    X_train_prior[np.isnan(X_train_prior)] = 0
    X_test_prior[np.isnan(X_test_prior)] = 0

    return play_preprocessor, prior_preprocessor, X_train_play.astype(np.float32), X_test_play.astype(np.float32), X_train_prior.astype(np.float32), X_test_prior.astype(np.float32)


def prepare_class_targets(target):
    y_cat = pd.get_dummies(target)
    y_class = np.argmax(y_cat.to_numpy(dtype=np.int64), axis=1)
    return y_cat, y_class


def distance_bucket(distance):
    if distance <= 3:
        return "short"
    if distance <= 7:
        return "medium"
    return "long"


def score_bucket(score_diff):
    if score_diff <= -9:
        return "trailing_big"
    if score_diff <= -1:
        return "trailing"
    if score_diff >= 9:
        return "leading_big"
    if score_diff >= 1:
        return "leading"
    return "tied"


def clock_bucket(quarter, clock):
    if quarter >= 4 and clock <= 300:
        return "q4_late"
    if quarter >= 3:
        return "second_half"
    if quarter == 2 and clock <= 300:
        return "q2_late"
    return "first_half"


def playcall_bucket_key(down, distance, score_diff, quarter, clock):
    down_int = int(np.clip(int(down), 1, 4))
    return (
        f"down_{down_int}|dist_{distance_bucket(int(distance))}"
        f"|score_{score_bucket(float(score_diff))}|clock_{clock_bucket(int(quarter), int(clock))}"
    )


def _derive_quarter_and_clock_from_frame(frame):
    quarter_raw = frame["qtr"] if "qtr" in frame.columns else pd.Series(np.nan, index=frame.index)
    seconds_raw = (
        frame["game_seconds_remaining"]
        if "game_seconds_remaining" in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )
    quarter = pd.to_numeric(quarter_raw, errors="coerce")
    game_seconds_remaining = pd.to_numeric(seconds_raw, errors="coerce")

    if quarter.isna().all() and game_seconds_remaining.notna().any():
        seconds = game_seconds_remaining.to_numpy(dtype=float)
        derived_quarter = 4 - np.floor((np.clip(seconds, 1, 3600) - 1) / 900.0)
        quarter = pd.Series(np.clip(derived_quarter, 1, 4), index=frame.index)

    quarter = quarter.fillna(1).astype(int)
    if game_seconds_remaining.notna().any():
        qclock = game_seconds_remaining - (4 - quarter) * 900
        qclock = qclock.fillna(900).clip(lower=0, upper=900).astype(int)
    else:
        qclock = pd.Series(np.full(len(frame), 900, dtype=int), index=frame.index)
    return quarter, qclock


def build_playcall_bucket_series(frame):
    required_cols = ["down", "ydstogo", "score_differential"]
    if not all(col in frame.columns for col in required_cols):
        return pd.Series(["unknown"] * len(frame), index=frame.index)

    down = pd.to_numeric(frame["down"], errors="coerce").fillna(1).astype(int)
    ydstogo = pd.to_numeric(frame["ydstogo"], errors="coerce").fillna(10).astype(int)
    score_diff = pd.to_numeric(frame["score_differential"], errors="coerce").fillna(0.0)
    quarter, qclock = _derive_quarter_and_clock_from_frame(frame)

    keys = []
    for idx in frame.index:
        keys.append(
            playcall_bucket_key(
                down=down.loc[idx],
                distance=ydstogo.loc[idx],
                score_diff=score_diff.loc[idx],
                quarter=quarter.loc[idx],
                clock=qclock.loc[idx],
            )
        )
    return pd.Series(keys, index=frame.index)


def write_play_type_bucket_calibration(bucket_keys, y_true_class, y_pred_probs, class_names):
    class_names = [str(c) for c in class_names]
    if "pass" not in class_names or "rush" not in class_names:
        return

    pass_idx = class_names.index("pass")
    rush_idx = class_names.index("rush")
    true_class = np.asarray(y_true_class, dtype=np.int64)
    probs = np.asarray(y_pred_probs, dtype=np.float64)
    bucket_values = pd.Series(bucket_keys).astype(str).to_numpy()

    frame = pd.DataFrame(
        {
            "bucket_key": bucket_values,
            "is_pass": (true_class == pass_idx).astype(np.int64),
            "is_rush": (true_class == rush_idx).astype(np.int64),
            "pass_prob": probs[:, pass_idx],
            "rush_prob": probs[:, rush_idx],
        }
    )
    frame = frame[(frame["is_pass"] + frame["is_rush"]) > 0].copy()
    if frame.empty:
        return

    denom = frame["pass_prob"] + frame["rush_prob"]
    frame["normalized_pass_prob"] = np.divide(
        frame["pass_prob"],
        denom,
        out=np.full(len(frame), np.nan),
        where=denom.to_numpy(dtype=float) > 0,
    )

    grouped = (
        frame.groupby("bucket_key", as_index=False)
        .agg(
            off_snaps=("is_pass", "size"),
            actual_pass_calls=("is_pass", "sum"),
            actual_run_calls=("is_rush", "sum"),
            predicted_pass_prob_mean=("normalized_pass_prob", "mean"),
        )
    )
    grouped["actual_pass_rate"] = grouped["actual_pass_calls"] / grouped["off_snaps"]
    grouped["pass_rate_error"] = grouped["predicted_pass_prob_mean"] - grouped["actual_pass_rate"]
    grouped["pass_rate_abs_error"] = grouped["pass_rate_error"].abs()
    grouped = grouped.sort_values("off_snaps", ascending=False).reset_index(drop=True)
    grouped.to_csv(str(PLAY_TYPE_BUCKET_CALIBRATION_CSV), index=False)

    filtered = grouped[grouped["off_snaps"] >= int(PLAY_TYPE_BUCKET_MIN_SUPPORT)].copy()
    metric_frame = filtered if not filtered.empty else grouped
    weight = metric_frame["off_snaps"].to_numpy(dtype=float)
    weight_sum = float(np.nansum(weight))
    weighted_mae = None
    if weight_sum > 0:
        weighted_mae = float(np.nansum(metric_frame["pass_rate_abs_error"].to_numpy(dtype=float) * weight) / weight_sum)

    top_errors = []
    for row in metric_frame.sort_values("pass_rate_abs_error", ascending=False).head(15).itertuples(index=False):
        top_errors.append(
            {
                "bucket_key": str(row.bucket_key),
                "off_snaps": int(row.off_snaps),
                "actual_pass_rate": float(row.actual_pass_rate),
                "predicted_pass_prob_mean": float(row.predicted_pass_prob_mean),
                "pass_rate_error": float(row.pass_rate_error),
            }
        )

    summary = {
        "rows": int(len(grouped)),
        "min_support_off_snaps": int(PLAY_TYPE_BUCKET_MIN_SUPPORT),
        "rows_after_min_support": int(len(metric_frame)),
        "weighted_pass_rate_mae": weighted_mae,
        "mean_pass_rate_error": float(metric_frame["pass_rate_error"].mean()),
        "mean_predicted_pass_rate": float(metric_frame["predicted_pass_prob_mean"].mean()),
        "mean_actual_pass_rate": float(metric_frame["actual_pass_rate"].mean()),
        "top_bucket_errors": top_errors,
    }
    PLAY_TYPE_BUCKET_CALIBRATION_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_preprocessor(feature_columns, categorical_cols):
    categorical_cols = [col for col in categorical_cols if col in feature_columns]
    numerical_cols = [col for col in feature_columns if col not in categorical_cols]
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ])
    return preprocessor, numerical_cols, categorical_cols


class SkorchOptunaPruning(skorch.callbacks.Callback):
    """Report validation loss to Optuna each epoch so weak trials can be stopped early."""

    def __init__(self, trial):
        self.trial = trial

    def on_epoch_end(self, net, **kwargs):
        import optuna

        valid_loss = net.history[-1].get("valid_loss")
        if valid_loss is None:
            return
        epoch = len(net.history)
        self.trial.report(float(valid_loss), epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def _skorch_scheduler_callback(hparams, max_epochs):
    name = str(hparams.get("scheduler", "none")).lower()
    config = hparams.get("scheduler_params") or {}
    if name == "none":
        return None
    if name == "cosine":
        return LRScheduler(
            policy=torch.optim.lr_scheduler.CosineAnnealingLR,
            T_max=max(1, int(config.get("t_max", max_epochs) or max_epochs)),
        )
    if name == "plateau":
        return LRScheduler(
            policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
            monitor="valid_loss",
            mode="min",
            factor=float(config.get("factor", 0.5)),
            patience=int(config.get("patience", 5)),
        )
    raise ValueError(f"Unsupported scheduler for the flat training path: {name}")


def fit_standard_classifier(
    X_train,
    X_test,
    y_train_class,
    output_dim,
    ordinal_output=False,
    max_epochs=1500,
    batch_size=512,
    hparams=None,
    optuna_trial=None,
    use_checkpoint=True,
):
    hparams = hparams or default_hyperparameters(False)
    batch_size = int(hparams.get("batch_size", batch_size))
    X_train_np = X_train.astype(np.float32)
    X_test_np = X_test.astype(np.float32)

    if ordinal_output:
        criterion = MyCornLoss(output_dim)
        criterion_kwargs = {}
    else:
        criterion = nn.CrossEntropyLoss
        criterion_kwargs = {"criterion__label_smoothing": float(hparams.get("label_smoothing", 0.0))}
        if str(hparams.get("class_weighting", "none")).lower() == "balanced":
            weights = balanced_class_weights(y_train_class, output_dim)
            if weights is not None:
                criterion_kwargs["criterion__weight"] = weights

    callbacks = [EarlyStopping(patience=20, load_best=True)]
    if use_checkpoint:
        callbacks.append(Checkpoint(load_best=True))
    scheduler_callback = _skorch_scheduler_callback(hparams, max_epochs)
    if scheduler_callback is not None:
        callbacks.append(scheduler_callback)
    grad_clip = float(hparams.get("grad_clip_norm", 0.0) or 0.0)
    if grad_clip > 0:
        callbacks.append(GradientNormClipping(gradient_clip_value=grad_clip))
    if optuna_trial is not None:
        callbacks.append(SkorchOptunaPruning(optuna_trial))

    model_play = NeuralNetClassifier(
        module=CornModel if ordinal_output else GameModel,
        module__input_dim=X_train_np.shape[1],
        module__output_dim=output_dim,
        module__arch=hparams["arch"],
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=float(hparams.get("learning_rate", 0.01)),
        optimizer=optimizer_class(hparams),
        **{f"optimizer__{key}": value for key, value in optimizer_extra_kwargs(hparams).items()},
        criterion=criterion,
        **criterion_kwargs,
        callbacks=callbacks,
        iterator_train__shuffle=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    model_play.fit(X_train_np, y_train_class)
    test_logits = model_play.predict_proba(X_test_np)
    probabilities = corn_logits_to_class_probs(test_logits) if ordinal_output else test_logits
    valid_losses = [row.get("valid_loss") for row in model_play.history if row.get("valid_loss") is not None]
    return {
        'model': model_play,
        'X_test': X_test_np,
        'test_logits': test_logits,
        'test_probabilities': probabilities,
        'val_loss': float(min(valid_losses)) if valid_losses else float('inf'),
        'network_kind': 'corn' if ordinal_output else 'flat',
        'hyperparameters': hparams,
    }


def _as_target_series(target, index):
    if isinstance(target, pd.Series):
        return target
    return pd.Series(target, index=index)


def apply_starting_qb_prior_features(frame, player_stats_wide, qb_prior_columns):
    """Overwrite qb/passer prior fields with the starting QB values per game/team."""
    if frame.empty:
        return frame

    required_cols = {"game_id", "posteam", "passer_player_id"}
    if not required_cols.issubset(frame.columns):
        return frame

    qb_prior_columns = [str(col) for col in qb_prior_columns]
    available_qb_cols = [col for col in qb_prior_columns if col in player_stats_wide.columns]
    if not available_qb_cols:
        return frame

    ordered = frame.sort_values([col for col in ["game_id", "posteam", "play_id"] if col in frame.columns])
    starters = (
        ordered.loc[ordered["passer_player_id"].notna(), ["game_id", "posteam", "passer_player_id"]]
        .drop_duplicates(subset=["game_id", "posteam"], keep="first")
        .rename(columns={"passer_player_id": "starting_qb_player_id"})
    )
    if starters.empty:
        return frame

    qb_features = player_stats_wide.reindex(columns=available_qb_cols).reset_index()
    qb_features["player_id"] = qb_features["player_id"].astype(str)
    starters["starting_qb_player_id"] = starters["starting_qb_player_id"].astype(str)
    starters = starters.merge(
        qb_features,
        left_on=["starting_qb_player_id", "game_id"],
        right_on=["player_id", "game_id"],
        how="left",
    )

    prefixed = {col: f"starting_qb_{col}" for col in available_qb_cols}
    starters = starters.rename(columns=prefixed)
    merged = frame.merge(
        starters[["game_id", "posteam", *prefixed.values()]],
        on=["game_id", "posteam"],
        how="left",
    )

    for col in available_qb_cols:
        source_col = prefixed[col]
        src = pd.to_numeric(merged[source_col], errors="coerce")
        if col in merged.columns:
            current = pd.to_numeric(merged[col], errors="coerce")
            merged[col] = src.combine_first(current)
        else:
            merged[col] = src
    merged = merged.drop(columns=list(prefixed.values()), errors="ignore")
    return merged


def max_epochs_for(optuna_trial, default_epochs):
    """Trials get a tighter epoch cap so a 30-trial search stays affordable."""
    if optuna_trial is None:
        return default_epochs
    return min(int(default_epochs), int(CLI_ARGS.tuning_max_epochs))


def _result_metrics(result):
    probabilities = result.get('test_probabilities')
    truth = result.get('y_test_class')
    if probabilities is None or truth is None:
        return {}
    return evaluation_metrics(truth, probabilities, bins=result.get('bins'))


def tune_model(model_name, target_mode, is_hierarchical, train_fn, framework="legacy"):
    """Run an Optuna study, keeping the fitted model from the best-validation-loss trial."""
    import optuna

    best = {'result': None, 'hparams': None, 'val_loss': float('inf'), 'trial_number': None}

    def objective(trial):
        hparams = (
            suggest_ngmm_hyperparameters(trial, target_mode)
            if framework == "ngmm"
            else suggest_hyperparameters(trial, model_name, target_mode, is_hierarchical)
        )
        print(f"[tune] {model_name} trial {trial.number}: {describe_hyperparameters(hparams)}")
        try:
            result = train_fn(hparams, trial)
        except optuna.TrialPruned:
            raise
        except (RuntimeError, ValueError) as error:
            # Infeasible configurations (e.g. batchnorm on a size-1 batch) are pruned, not fatal.
            print(f"[tune] {model_name} trial {trial.number} failed: {error}")
            raise optuna.TrialPruned()

        val_loss = float(result.get('val_loss', float('inf')))
        if not np.isfinite(val_loss):
            raise optuna.TrialPruned()

        for key, value in _result_metrics(result).items():
            trial.set_user_attr(key, float(value))
        trial.set_user_attr('val_loss', val_loss)

        if val_loss < best['val_loss']:
            best.update({
                'result': result,
                'hparams': hparams,
                'val_loss': val_loss,
                'trial_number': trial.number,
            })
            print(f"[tune] {model_name} trial {trial.number} is the new best (val_loss={val_loss:.5f})")
        return val_loss

    run_study(
        model_name,
        objective,
        n_trials=int(CLI_ARGS.trials),
        timeout=CLI_ARGS.tuning_timeout,
        seed=TRAINING_SEED,
        fresh=bool(CLI_ARGS.fresh_study),
    )

    if best['result'] is None:
        raise RuntimeError(f"{model_name}: every tuning trial failed or was pruned")

    metrics = _result_metrics(best['result'])
    record = {
        'val_loss': best['val_loss'],
        'trial_number': best['trial_number'],
        'trials_requested': int(CLI_ARGS.trials),
        'tuning_max_epochs': int(CLI_ARGS.tuning_max_epochs),
        'seasons': list(SEASONS),
        'seed': int(TRAINING_SEED),
        'target_mode': target_mode,
        'hierarchical': bool(is_hierarchical),
        'test_metrics': metrics,
    }
    save_best_hyperparameters(model_name, best['hparams'], record)
    tuning_leaderboard_rows.append({'model_name': model_name, **record, **metrics})
    print(
        f"[tune] {model_name} best trial {best['trial_number']} val_loss={best['val_loss']:.5f} "
        f"test_metrics={metrics}"
    )
    return best['result']


def run_or_tune(model_name, target_mode, is_hierarchical, train_fn, framework="legacy"):
    """Dispatch a single fit (train mode) or a full Optuna search (tune mode)."""
    if TUNING_MODE:
        return tune_model(model_name, target_mode, is_hierarchical, train_fn, framework=framework)
    hparams = (
        load_best_ngmm_hyperparameters(model_name, target_mode)
        if framework == "ngmm"
        else load_best_hyperparameters(model_name, is_hierarchical)
    )
    print(f"[train] {model_name}: {describe_hyperparameters(hparams)}")
    return train_fn(hparams, None)


def _arch_sidecar_payload(model_name, result):
    network = result['torch_model']
    kind = result.get('network_kind', 'flat')
    hparams = result.get('hyperparameters') or {}
    if kind == 'ngmm':
        architecture = getattr(network, 'architecture', None)
        return {
            'model_name': model_name,
            'artifact_prefix': MODEL_ARTIFACT_PREFIX[model_name],
            'kind': 'ngmm',
            'dims': {
                'input_dim': int(network.fixed[0].in_features),
                'context_dim': int(network.context[0].in_features),
                'latent_dim': int(network.latent_dim),
                'output_dim': int(network.output_dim),
                'target_mode': result.get('target_mode', getattr(network, 'target_mode', 'multiclass')),
            },
            'arch': architecture.to_dict() if architecture is not None else {},
            'hyperparameters': {key: value for key, value in hparams.items()},
            'val_loss': result.get('val_loss'),
            'mode': CLI_ARGS.mode,
            'seasons': list(SEASONS),
            'seed': int(TRAINING_SEED),
            'trained_at': pd.Timestamp.utcnow().isoformat(),
        }
    dims = {'output_dim': int(network.out.out_features)}
    if kind == 'hierarchical':
        dims['latent_dim'] = int(network.latent_dim)
        dims['play_input_dim'] = int(network.fc1.in_features) - int(network.latent_dim)
        dims['prior_input_dim'] = int(network.prior_encoder.hidden[0].in_features)
        dims['ordinal_output'] = bool(network.ordinal_output)
        if network.ordinal_output:
            dims['output_dim'] += 1
    else:
        dims['input_dim'] = int(network.fc1.in_features)
        if kind == 'corn':
            dims['output_dim'] += 1
    return {
        'model_name': model_name,
        'artifact_prefix': MODEL_ARTIFACT_PREFIX[model_name],
        'kind': kind,
        'dims': dims,
        'arch': normalize_arch(
            hparams.get('arch'), 'hierarchical' if kind == 'hierarchical' else 'flat'
        ),
        'hyperparameters': {key: value for key, value in hparams.items() if key != 'arch'},
        'val_loss': result.get('val_loss'),
        'mode': CLI_ARGS.mode,
        'seasons': list(SEASONS),
        'seed': int(TRAINING_SEED),
        'trained_at': pd.Timestamp.utcnow().isoformat(),
    }


def sync_hierarchical_config(model_name, result, feature_config):
    """Keep model_feature_config.json's latent_dim/kl_weight aligned with the saved checkpoint.

    The runtime derives prior-embedding column counts from the config, so a tuned latent_dim
    that only lives in the checkpoint would surface as a schema mismatch downstream.
    """
    if result.get('network_kind') != 'hierarchical':
        return
    hparams = result.get('hyperparameters') or {}
    model_config = feature_config.get("models", {}).get(model_name)
    if model_config is None:
        return

    changed = False
    latent_dim = int(result['torch_model'].latent_dim)
    if int(model_config.get("latent_dim", 0)) != latent_dim:
        model_config["latent_dim"] = latent_dim
        changed = True
    kl_weight = hparams.get("kl_weight")
    if kl_weight is not None and float(model_config.get("kl_weight", -1)) != float(kl_weight):
        model_config["kl_weight"] = float(kl_weight)
        changed = True

    if changed:
        save_feature_config(feature_config, FEATURE_CONFIG_PATH)
        print(f"[config] {model_name}: synced latent_dim={latent_dim} kl_weight={model_config.get('kl_weight')}")


def save_model_artifacts(model_name, result, feature_config):
    """Write the checkpoint, preprocessors, bins, feature groups and arch sidecar."""
    prefix = MODEL_ARTIFACT_PREFIX[model_name]
    torch.save(result['torch_model'].state_dict(), out_path(f"{prefix}_model.pt"))
    joblib.dump(result['play_preprocessor'], out_path(f"{prefix}_preprocessor.pkl"))
    if result.get('prior_preprocessor') is not None:
        joblib.dump(result['prior_preprocessor'], out_path(f"{prefix}_prior_preprocessor.pkl"))
    if result.get('bins') is not None:
        pd.DataFrame(result['bins']).to_csv(out_path(f"{prefix}_bins.csv"))
    joblib.dump(feature_config["models"][model_name]["feature_groups"], out_path(f"{prefix}_columns.pkl"))
    write_arch_sidecar(arch_sidecar_path(BASE_DIR, prefix), _arch_sidecar_payload(model_name, result))
    sync_hierarchical_config(model_name, result, feature_config)


def train_model_from_config(
    feature_config,
    model_name,
    frame,
    target,
    groups,
    play_catalog,
    prior_catalog=None,
    target_mode="multiclass",
    flat_max_epochs=1500,
    hierarchical_max_epochs=200,
    batch_size=512,
):
    play_columns = resolve_model_features(feature_config, model_name, play_catalog)
    categorical_cols = resolve_model_categorical_cols(feature_config, model_name)
    use_hierarchical = should_use_hierarchical_model(feature_config, model_name)

    bins = None
    if target_mode in {"multiclass", "ordinal"}:
        y_cat, y_class = prepare_class_targets(target)
        target_series = pd.Series(y_class, index=frame.index)
        output_dim = y_cat.shape[1]
        if target_mode == "ordinal":
            bins = np.array([float(i) for i in y_cat.columns])
    elif target_mode == "binary":
        y_cat = None
        target_series = _as_target_series(target, frame.index).astype(np.int64)
        output_dim = 2
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    ordinal_output = target_mode == "ordinal"
    groups_series = pd.Series(np.asarray(groups), index=frame.index)
    hierarchical_groups_series = hierarchical_group_labels(frame, groups)

    if use_hierarchical:
        hierarchical_total_start = perf_counter()
        prior_catalog = prior_catalog or play_catalog
        step_start = perf_counter()
        prior_columns = resolve_model_prior_features(feature_config, model_name, prior_catalog)
        combined_frame = frame.reindex(columns=unique_columns(play_columns + prior_columns)).copy()
        _timing_checkpoint(f"{model_name} hierarchical feature assembly", step_start)
        step_start = perf_counter()
        train_frame, test_frame, y_train, y_test = blocked_train_test_split(
            combined_frame,
            target_series,
            groups=hierarchical_groups_series.loc[combined_frame.index],
            test_size=0.2,
            random_state=42,
        )
        _timing_checkpoint(f"{model_name} hierarchical train/test split", step_start)
        y_test_class = y_test.values if hasattr(y_test, 'values') else np.asarray(y_test)

        def train_fn(hparams, optuna_trial):
            ngmm_arch = hparams.get('ngmm_arch') or hparams.get('arch') or {}
            result = train_ngmm_grouped(
                train_frame=train_frame,
                test_frame=test_frame,
                y_train=y_train,
                y_test=y_test,
                train_groups=hierarchical_groups_series.loc[train_frame.index].to_numpy(),
                play_columns=play_columns,
                prior_columns=prior_columns,
                categorical_cols=categorical_cols,
                target_mode=target_mode,
                output_dim=output_dim,
                max_epochs=max_epochs_for(optuna_trial, hierarchical_max_epochs),
                batch_size=batch_size,
                latent_dim=int(hparams.get('latent_dim') or resolve_model_latent_dim(feature_config, model_name, default=1)),
                quadrature_samples=int(hparams.get('quadrature_samples', 32)),
                observations_per_group=int(hparams.get('observations_per_group', 256)),
                learning_rate=float(hparams.get('learning_rate', 1e-3)),
                weight_decay=float(hparams.get('weight_decay', 0.0)),
                dropout=float(ngmm_arch.get('dropout', 0.1) if isinstance(ngmm_arch.get('dropout', 0.1), (int, float)) else ngmm_arch.get('dropout', [0.1])[0]),
                hidden_dims=tuple(ngmm_arch.get('hidden_dims', [128, 64, 32])),
                context_hidden_dims=tuple(ngmm_arch.get('context_hidden_dims', [64, 32])),
                log_progress=optuna_trial is None,
            )
            result.update({
                'play_columns': play_columns,
                'prior_columns': prior_columns,
                'y_test_class': y_test_class,
                'bins': bins,
                'torch_model': result['model'],
                'loss_history': result.get('loss_history'),
                'hyperparameters': hparams,
            })
            return result

        result = run_or_tune(model_name, target_mode, True, train_fn, framework="ngmm")
        _timing_checkpoint(f"{model_name} hierarchical total", hierarchical_total_start)
        return result

    X = frame.reindex(columns=play_columns).copy()
    X_train, X_test, y_train, y_test = blocked_train_test_split(
        X,
        target_series,
        groups=groups_series.loc[X.index],
        test_size=0.2,
        random_state=42,
    )
    preprocessor, _, _ = build_preprocessor(play_columns, categorical_cols)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    X_train_processed, X_test_processed = fill_nans_with_zero(X_train_processed, X_test_processed)
    y_train_class = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
    y_test_class = y_test.values if hasattr(y_test, 'values') else np.asarray(y_test)

    def train_fn(hparams, optuna_trial):
        fitted = fit_standard_classifier(
            X_train_processed,
            X_test_processed,
            y_train_class,
            output_dim=output_dim,
            ordinal_output=ordinal_output,
            max_epochs=max_epochs_for(optuna_trial, flat_max_epochs),
            batch_size=batch_size,
            hparams=hparams,
            optuna_trial=optuna_trial,
            use_checkpoint=optuna_trial is None,
        )
        fitted.update({
            'play_preprocessor': preprocessor,
            'prior_preprocessor': None,
            'play_columns': play_columns,
            'prior_columns': [],
            'y_test_class': y_test_class,
            'bins': bins,
            'torch_model': fitted['model'].module_,
            'loss_history': fitted['model'].history,
        })
        return fitted

    return run_or_tune(model_name, target_mode, False, train_fn)


def train_hierarchical_latent_classifier(
    train_frame,
    test_frame,
    y_train,
    y_test,
    train_groups,
    play_columns,
    prior_columns,
    categorical_cols,
    latent_dim,
    kl_weight,
    output_dim,
    max_epochs=200,
    batch_size=512,
    learning_rate=1e-3,
    validation_size=0.2,
    patience=20,
    ordinal_output=False,
    grouped_batches=False,
    grouped_batch_mode="packed",
    max_groups_per_batch=4,
    max_group_fraction=0.7,
    grad_accum_steps=1,
    dataloader_num_workers=0,
    pin_memory=None,
    use_mixed_precision=True,
    kl_warmup_epochs=20,
    run_label="hierarchical",
    log_progress=True,
    log_every=5,
    hparams=None,
    optuna_trial=None,
):
    hparams = hparams or default_hyperparameters(True)
    arch = normalize_arch(hparams.get("arch"), "hierarchical")
    latent_dim = int(hparams.get("latent_dim") or latent_dim)
    kl_weight = float(hparams["kl_weight"]) if hparams.get("kl_weight") is not None else float(kl_weight)
    kl_warmup_epochs = int(hparams.get("kl_warmup_epochs", kl_warmup_epochs))
    batch_size = int(hparams.get("batch_size", batch_size))
    learning_rate = float(hparams.get("learning_rate", learning_rate))
    max_groups_per_batch = int(hparams.get("max_groups_per_batch", max_groups_per_batch))
    max_group_fraction = float(hparams.get("max_group_fraction", max_group_fraction))
    grad_clip_norm = float(hparams.get("grad_clip_norm", 0.0) or 0.0)

    overall_start = perf_counter()
    train_group_labels_fit = None
    val_group_labels = None
    step_start = perf_counter()
    if train_groups is not None:
        train_groups_series = pd.Series(np.asarray(train_groups), index=train_frame.index)
        train_frame_fit, val_frame, y_train_fit, y_val = blocked_train_test_split(
            train_frame,
            y_train,
            train_groups,
            test_size=validation_size,
            random_state=42,
        )
        train_group_labels_fit = train_groups_series.loc[train_frame_fit.index].to_numpy()
        val_group_labels = train_groups_series.loc[val_frame.index].to_numpy()
    else:
        train_frame_fit, val_frame, y_train_fit, y_val = train_test_split(
            train_frame,
            y_train,
            test_size=validation_size,
            random_state=42,
        )
    _timing_checkpoint("hierarchical train/val split", step_start)

    step_start = perf_counter()
    train_frame_fit = train_frame_fit.reset_index(drop=True)
    val_frame = val_frame.reset_index(drop=True)
    if hasattr(y_train_fit, "reset_index"):
        y_train_fit = y_train_fit.reset_index(drop=True)
    if hasattr(y_val, "reset_index"):
        y_val = y_val.reset_index(drop=True)
    _timing_checkpoint("hierarchical index reset", step_start)

    step_start = perf_counter()
    play_preprocessor, prior_preprocessor, X_train_play, X_test_play, X_train_prior, X_test_prior = _prepare_latent_features(
        train_frame_fit,
        test_frame,
        play_columns,
        prior_columns,
        categorical_cols,
    )
    _timing_checkpoint("hierarchical train/test feature prep", step_start)

    step_start = perf_counter()
    _, _, X_val_play, _, X_val_prior, _ = _prepare_latent_features(
        train_frame_fit,
        val_frame,
        play_columns,
        prior_columns,
        categorical_cols,
    )
    _timing_checkpoint("hierarchical validation feature prep", step_start)

    step_start = perf_counter()
    train_len = min(len(X_train_play), len(X_train_prior), len(y_train_fit))
    val_len = min(len(X_val_play), len(X_val_prior), len(y_val))
    X_train_play = X_train_play[:train_len]
    X_train_prior = X_train_prior[:train_len]
    y_train_fit = y_train_fit.iloc[:train_len] if hasattr(y_train_fit, "iloc") else y_train_fit[:train_len]
    X_val_play = X_val_play[:val_len]
    X_val_prior = X_val_prior[:val_len]
    y_val = y_val.iloc[:val_len] if hasattr(y_val, "iloc") else y_val[:val_len]
    _timing_checkpoint("hierarchical length alignment", step_start)

    step_start = perf_counter()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HierarchicalLatentGameModel(
        play_input_dim=X_train_play.shape[1],
        prior_input_dim=X_train_prior.shape[1],
        latent_dim=latent_dim,
        output_dim=output_dim,
        arch=arch,
        ordinal_output=ordinal_output,
    ).to(device)

    optimizer = build_optimizer(model.parameters(), hparams)
    if ordinal_output:
        criterion = MyCornLoss(output_dim).to(device)
    else:
        class_weights = None
        if str(hparams.get("class_weighting", "none")).lower() == "balanced":
            class_weights = balanced_class_weights(
                y_train_fit.values if hasattr(y_train_fit, 'values') else y_train_fit,
                output_dim,
            )
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None,
            label_smoothing=float(hparams.get("label_smoothing", 0.0)),
        )
    _timing_checkpoint("hierarchical model/optimizer setup", step_start)

    step_start = perf_counter()
    amp_enabled = bool(use_mixed_precision and torch.cuda.is_available())
    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())
    grad_accum_steps = max(1, int(grad_accum_steps))
    dataloader_num_workers = max(0, int(dataloader_num_workers))
    loader_kwargs = {
        "pin_memory": bool(pin_memory),
        "num_workers": dataloader_num_workers,
    }
    if dataloader_num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train_play, dtype=torch.float32),
        torch.tensor(X_train_prior, dtype=torch.float32),
        torch.tensor(y_train_fit.values if hasattr(y_train_fit, 'values') else y_train_fit, dtype=torch.long),
    )
    if grouped_batches and train_group_labels_fit is not None:
        if grouped_batch_mode == "strict":
            train_batch_sampler = GroupBatchSampler(
                train_group_labels_fit,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                seed=TRAINING_SEED,
            )
        elif grouped_batch_mode == "packed":
            train_batch_sampler = PackedGroupBatchSampler(
                train_group_labels_fit,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                max_groups_per_batch=max_groups_per_batch,
                max_group_fraction=max_group_fraction,
                seed=TRAINING_SEED,
            )
        else:
            raise ValueError(f"Unsupported grouped_batch_mode: {grouped_batch_mode}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **loader_kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    _timing_checkpoint("hierarchical train dataset/loader setup", step_start)

    step_start = perf_counter()
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val_play, dtype=torch.float32),
        torch.tensor(X_val_prior, dtype=torch.float32),
        torch.tensor(y_val.values if hasattr(y_val, 'values') else y_val, dtype=torch.long),
    )
    if grouped_batches and val_group_labels is not None:
        if grouped_batch_mode == "strict":
            val_batch_sampler = GroupBatchSampler(
                val_group_labels,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                seed=TRAINING_SEED,
            )
        elif grouped_batch_mode == "packed":
            val_batch_sampler = PackedGroupBatchSampler(
                val_group_labels,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                max_groups_per_batch=max_groups_per_batch,
                max_group_fraction=max_group_fraction,
                seed=TRAINING_SEED,
            )
        else:
            raise ValueError(f"Unsupported grouped_batch_mode: {grouped_batch_mode}")
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_batch_sampler, **loader_kwargs)
    else:
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    _timing_checkpoint("hierarchical val dataset/loader setup", step_start)

    if log_progress:
        print(
            f"[hier-train] {run_label} grouped_batches={grouped_batches} mode={grouped_batch_mode} "
            f"train_batches={len(train_loader)} val_batches={len(val_loader)} "
            f"batch_size={batch_size} grad_accum_steps={grad_accum_steps} amp={amp_enabled}"
        )

    scheduler, scheduler_step_mode = build_scheduler(
        optimizer, hparams, max_epochs=max_epochs, steps_per_epoch=len(train_loader)
    )
    scheduler_total_steps = getattr(scheduler, "total_steps", None)
    scheduler_batch_steps = 0

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    best_epoch_idx = -1
    epochs_without_improvement = 0
    train_loss_history = []
    valid_loss_history = []
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    training_loop_start = perf_counter()
    for epoch_idx in range(max_epochs):
        epoch_start = perf_counter()
        kl_scale = min(1.0, float(epoch_idx + 1) / max(1, int(kl_warmup_epochs)))
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (batch_play, batch_prior, batch_y) in enumerate(train_loader):
            batch_play = batch_play.to(device, non_blocking=bool(pin_memory))
            batch_prior = batch_prior.to(device, non_blocking=bool(pin_memory))
            batch_y = batch_y.to(device, non_blocking=bool(pin_memory))

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits, kl_div = model(batch_play, batch_prior, sample=True)
                loss = criterion(logits, batch_y) + (kl_weight * kl_scale) * kl_div

            scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()

            should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(train_loader))
            if should_step:
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and scheduler_step_mode == "batch":
                    if scheduler_total_steps is None or scheduler_batch_steps < scheduler_total_steps:
                        scheduler.step()
                        scheduler_batch_steps += 1

            train_loss_total += float(loss.detach().cpu())
            train_batches += 1

        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_play, batch_prior, batch_y in val_loader:
                batch_play = batch_play.to(device, non_blocking=bool(pin_memory))
                batch_prior = batch_prior.to(device, non_blocking=bool(pin_memory))
                batch_y = batch_y.to(device, non_blocking=bool(pin_memory))
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    logits, kl_div = model(batch_play, batch_prior, sample=True)
                    loss = criterion(logits, batch_y)
                val_loss_total += float(loss.detach().cpu())
                val_batches += 1

        mean_val_loss = val_loss_total / max(val_batches, 1)
        mean_train_loss = train_loss_total / max(train_batches, 1)
        train_loss_history.append(mean_train_loss)
        valid_loss_history.append(mean_val_loss)

        if scheduler is not None and scheduler_step_mode == "epoch":
            scheduler.step()
        elif scheduler is not None and scheduler_step_mode == "plateau":
            scheduler.step(mean_val_loss)

        if mean_val_loss < best_val_loss - 1e-6:
            best_val_loss = mean_val_loss
            best_epoch_idx = epoch_idx
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if optuna_trial is not None:
            import optuna

            optuna_trial.report(mean_val_loss, epoch_idx)
            if optuna_trial.should_prune():
                raise optuna.TrialPruned()

        if log_progress and (epoch_idx == 0 or ((epoch_idx + 1) % max(1, int(log_every)) == 0)):
            print(
                f"[hier-train] {run_label} epoch={epoch_idx + 1}/{max_epochs} "
                f"train_loss={mean_train_loss:.5f} val_loss={mean_val_loss:.5f} "
                f"best_val={best_val_loss:.5f} no_improve={epochs_without_improvement}/{patience}"
            )

        if epochs_without_improvement >= patience:
            if log_progress:
                print(
                    f"[hier-train] {run_label} early stopping at epoch {epoch_idx + 1}; "
                    f"best epoch={best_epoch_idx + 1} best_val={best_val_loss:.5f}"
                )
            break
        if epoch_idx == 0:
            _timing_checkpoint("hierarchical epoch_1", epoch_start)

    if log_progress and epochs_without_improvement < patience:
        print(
            f"[hier-train] {run_label} finished {len(train_loss_history)} epochs; "
            f"best epoch={best_epoch_idx + 1} best_val={best_val_loss:.5f}"
        )

    _timing_checkpoint("hierarchical training loop total", training_loop_start)

    step_start = perf_counter()
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_play_tensor = torch.tensor(X_test_play, dtype=torch.float32, device=device)
        test_prior_tensor = torch.tensor(X_test_prior, dtype=torch.float32, device=device)
        logits, kl_div = model(test_play_tensor, test_prior_tensor, sample=True)
        probabilities = corn_logits_to_class_probs(logits.cpu().numpy()) if ordinal_output else torch.softmax(logits, dim=1).cpu().numpy()
    _timing_checkpoint("hierarchical test inference", step_start)
    _timing_checkpoint("hierarchical classifier total", overall_start)

    return {
        'model': model,
        'play_preprocessor': play_preprocessor,
        'prior_preprocessor': prior_preprocessor,
        'X_test_play': X_test_play,
        'X_test_prior': X_test_prior,
        'test_probabilities': probabilities,
        'loss_history': {
            'train_loss': train_loss_history,
            'valid_loss': valid_loss_history,
        },
        'device': device,
        'ordinal_output': ordinal_output,
        'val_loss': float(best_val_loss),
        'best_epoch': int(best_epoch_idx + 1),
        'network_kind': 'hierarchical',
        'hyperparameters': hparams,
    }

#%%
CLI_ARGS = parse_cli_args()
SEASONS = CLI_ARGS.seasons
TRAINING_SEED = CLI_ARGS.seed
TUNING_MODE = CLI_ARGS.mode == "tune"
seed_training(TRAINING_SEED)
print(f"Mode: {CLI_ARGS.mode}")
print(f"Seasons: {', '.join(str(season) for season in SEASONS)}")
print(f"Training random seed: {TRAINING_SEED}")
if TUNING_MODE:
    print(
        f"Tuning budget: {CLI_ARGS.trials} trials/model, max {CLI_ARGS.tuning_max_epochs} epochs/trial"
        + (f", timeout {CLI_ARGS.tuning_timeout}s" if CLI_ARGS.tuning_timeout else "")
    )

player_stats = pd.read_csv(out_path("player_stats.csv"))
player_stats['average'] = player_stats['bayes_estimate_mean'].combine_first(player_stats['average'])
team_stats = pd.read_csv(out_path("Team_Stats.csv"))
team_stats['average'] = team_stats['bayes_estimate_mean'].combine_first(team_stats['average'])
game_state_cols = ["defteam_score","down","game_seconds_remaining","posteam_score","posteam_type","yardline_100","ydstogo","roof","score_differential"]
position_cols = ['position']
pbp = nfl.import_pbp_data(SEASONS, cache=True, alt_path=str(CACHE_DIR))
pbp = pbp[pbp['aborted_play']==0]
pbp = filter_non_meaningful_games(pbp)
pass_columns = ["air_yards"]
depth_chart = nfl.import_depth_charts(SEASONS)
rosters = nfl.import_weekly_rosters(SEASONS)
ids = nfl.import_ids()
id_to_name = dict(zip(ids["gsis_id"], ids["name"]))

# Replace player_name with the mapped names
player_stats["player_name"] = player_stats["player_id"].map(id_to_name)
rusher_columns = ['close_bayes','explosive_bayes','ypc_bayes','rush_rate','scramble_rate']
player_positions = nfl.import_players()[['gsis_id','position']].set_index('gsis_id',drop=True)
player_positions['position'] = player_positions['position'].replace({'FB': 'RB'})
player_positions = player_positions[player_positions['position'].isin(['RB','WR','TE','QB'])]
#%%Previous Game
player_stats_wide = (
    player_stats
    .pivot_table(index=['player_id','player_name','game_id','game_date'], columns=['source'], values='average')
    .reset_index()
)
player_stats_wide = player_stats_wide.sort_values(by=['player_id', 'game_date', 'game_id'])
#player_stats_wide.fillna(method='ffill',inplace=True)
player_stats_wide = (
    player_stats_wide
    .groupby('player_id', group_keys=False)
    .apply(lambda g: g.ffill())
)
player_stats_wide.set_index(['player_id','game_id'], inplace=True, drop=True)
team_stats_wide = (
    team_stats
    .pivot_table(index=['team','game_id','game_date'], columns=['source'], values='average')
    .reset_index()
)
team_stats_wide = team_stats_wide.sort_values(by=['team', 'game_date', 'game_id'])
team_stats_wide = (
    team_stats_wide
    .groupby('team', group_keys=False)
    .apply(lambda g: g.ffill())
)

# These direct team tendency features are computed using same-game aggregates,
# so we lag them one game to preserve pregame-only information.
team_direct_lag_cols = [
    'pass_oe_weekly_variation',
    'target_gini_roll4',
    'rb_carry_split_gini_roll4',
    'rb_top_carry_share_roll4',
]
for col in team_direct_lag_cols:
    if col in team_stats_wide.columns:
        team_stats_wide[col] = team_stats_wide.groupby('team')[col].shift(1)

team_stats_wide.set_index(['team','game_id'], inplace=True, drop=True)
def_cols = ['deep_cpoe_allowed','int_rate','medium_cpoe_allowed','short_cpoe_allowed','yac_oe_allowed','sack_rate','rush_yards_per_carry','def_rush_epa_rate','def_pass_epa_rate', 'def_winning_rate','def_losing_rate','def_tied_rate','passing_massey_defense','rushing_massey_defense']
team_stats_columns = ['losing_rate','tied_rate','team_4th_rate','winning_rate','rushing_epa_rate', 'epa_rate', 'passing_massey_rating', 'passing_massey_offense', 'rushing_massey_rating', 'rushing_massey_offense', 'opp_def_pass_epa_rate', 'opp_def_rush_epa_rate', 'opp_pass_epa_rate','opp_rushing_epa_rate']
team_tendency_columns = ['pass_oe', 'pass_oe_weekly_variation', 'target_gini_roll4', 'rb_carry_split_gini_roll4', 'rb_top_carry_share_roll4', 'pace_of_play_bayes']
qb_context_columns = ['qb_team_games_played', 'qb_dome_rate_bayes']

player_stats_wide.to_csv(out_path("player_stats_wide.csv"))
team_stats_wide.to_csv(out_path("team_stats_wide.csv"))
if CLI_ARGS.export_team_stats_wide_only:
    print(f"[export] team stats wide: {len(team_stats_wide):,} rows")
    raise SystemExit(0)
receiver_columns = ['deep','medium','short','fd_bayes','rz_bayes','yac_bayes','adot','ypt']
passer_columns = ["cpoe_bayes", "turnover_bayes","sack_bayes", "cpoe_short", "cpoe_medium", "cpoe_deep", "deep_pass_rate","medium_pass_rate", "short_pass_rate"]
final_routes = ['FLAT','QUICK OUT','HITCH','SLANT','CURL','DIG','POST','GO','OUT','CROSS','WHEEL','SWING','SCREEN']

feature_catalog = build_feature_catalog(
    {
        "game_state_cols": game_state_cols,
        "position_cols": position_cols,
        "pass_columns": pass_columns,
        "rusher_columns": rusher_columns,
        "def_cols": def_cols,
        "team_stats_columns": team_stats_columns,
        "team_tendency_columns": team_tendency_columns,
        "receiver_columns": receiver_columns,
        "passer_columns": passer_columns,
        "qb_context_columns": qb_context_columns,
    }
)
feature_config = load_feature_config(FEATURE_CONFIG_PATH, feature_catalog)
feature_config = ensure_expanded_columns(feature_config, feature_catalog)
report_builder = ModelReportBuilder("Torch Modeling Performance Report")
default_models = [name for name in TRAINABLE_MODEL_NAMES if name in feature_config.get("models", {})]
selected_models = resolve_selected_models(CLI_ARGS, default_models)
validate_hierarchical_configuration(feature_config, selected_models)
print(f"Selected models for {CLI_ARGS.mode}: {', '.join(selected_models)}")
tuning_leaderboard_rows = []

pbp = pbp.merge(player_stats_wide.reindex(columns=receiver_columns), left_on=['receiver_player_id', 'game_id'], right_index=True, how='left')
pbp = pbp.merge(team_stats_wide.reindex(columns=def_cols), left_on=['defteam', 'game_id'], right_index=True, how='left')
pbp = pbp.merge(team_stats_wide.reindex(columns=team_stats_columns), left_on=['posteam', 'game_id'], right_index=True, how='left')
pbp = pbp.merge(player_stats_wide.reindex(columns=passer_columns), left_on=['passer_player_id', 'game_id'], right_index=True, how='left')
pbp = pbp.merge(team_stats_wide.reindex(columns=team_tendency_columns), left_on=['posteam', 'game_id'], right_index=True, how='left')

# Harmonize pass_oe name after merge collisions (native PBP pass_oe vs team tendency pass_oe).
if 'pass_oe' not in pbp.columns:
    if 'pass_oe_x' in pbp.columns and 'pass_oe_y' in pbp.columns:
        pbp['pass_oe'] = pbp['pass_oe_y'].combine_first(pbp['pass_oe_x'])
    elif 'pass_oe_y' in pbp.columns:
        pbp['pass_oe'] = pbp['pass_oe_y']
    elif 'pass_oe_x' in pbp.columns:
        pbp['pass_oe'] = pbp['pass_oe_x']

pbp = pbp.merge(player_stats_wide.reindex(columns=qb_context_columns), left_on=['passer_player_id', 'game_id'], right_index=True, how='left')
qb_prior_columns = unique_columns(passer_columns + qb_context_columns)
pbp = apply_starting_qb_prior_features(pbp, player_stats_wide, qb_prior_columns)

# Days rest: computed directly from the game schedule (no shift/no Bayesian smoothing).
# Each team's days since their previous game, defaulting to 7 for their first game.
_team_schedule = (
    pd.concat([
        pbp.groupby(['posteam', 'game_id'])['game_date'].first().reset_index().rename(columns={'posteam': 'team'}),
        pbp.groupby(['defteam', 'game_id'])['game_date'].first().reset_index().rename(columns={'defteam': 'team'}),
    ])
    .drop_duplicates(subset=['team', 'game_id'])
)
_team_schedule['game_date'] = pd.to_datetime(_team_schedule['game_date'])
_team_schedule = _team_schedule.sort_values(['team', 'game_date'])
_team_schedule['days_rest'] = (
    _team_schedule.groupby('team')['game_date']
    .diff()
    .dt.days
)
offseason_baseline = _team_schedule.loc[_team_schedule['days_rest'] >= 30, 'days_rest']
default_first_rest = float(21) if not offseason_baseline.empty else 7.0
_team_schedule['days_rest'] = _team_schedule['days_rest'].fillna(default_first_rest)
_team_schedule = _team_schedule.set_index(['team', 'game_id'])
pbp = pbp.merge(_team_schedule[['days_rest']], left_on=['posteam', 'game_id'], right_index=True, how='left')
pbp = pbp.merge(
    _team_schedule[['days_rest']].rename(columns={'days_rest': 'opp_days_rest'}),
    left_on=['defteam', 'game_id'],
    right_index=True,
    how='left',
)

pbp['game_team_id'] = pbp['game_id']+pbp['posteam']

# Penalty-inclusive snapshot, captured after all the feature merges above (so it carries the same
# def_cols/team_stats_columns/pregame_context_columns/etc. as `pbp`) but before the penalty==0
# filter below, so penalty modeling (see the Penalty Modeling section) has access to penalty and
# no_play rows that the rest of the pipeline intentionally excludes. elapsed_time is computed once
# here on the full per-game play sequence (before any no_play/kickoff/penalty rows are dropped) so
# that time deltas are not silently inflated by skipping over removed rows; the Time Elapsed
# section below reuses these values via a merge instead of recomputing them on the reduced row set.
pbp_with_penalties = pbp.sort_values(['game_id', 'play_id']).copy()
pbp_with_penalties['elapsed_time'] = pbp_with_penalties.groupby('game_id')['game_seconds_remaining'].diff(-1)

pbp = pbp[pbp['penalty']==0]

#%%YAC Model
pbp_yac = pbp.copy()[pbp['season_type'] == 'REG']  # Only regular season
pbp_yac = pbp_yac[pbp_yac['yards_after_catch'].notnull() & pbp_yac['air_yards'].notnull()]  # Filter relevant plays
pbp_yac = pbp_yac[pbp_yac['pass_attempt'] == 1].copy()
pbp_yac = pbp_yac.merge(player_positions,left_on = 'receiver_player_id',right_index=True, how='left')
pbp_yac['game_date'] = pd.to_datetime(pbp_yac['game_date'])
player_stats['game_date'] = pd.to_datetime(player_stats['game_date'])
team_stats['game_date'] = pd.to_datetime(team_stats['game_date'])
player_stats_sorted = player_stats.sort_values(['player_id', 'game_date']).rename(columns={'player_id': 'receiver_player_id'})
pbp_yac = pbp_yac.sort_values(['receiver_player_id', 'game_date'])
model_name = "yac"
if should_train_model(model_name, selected_models):
    y = pbp_yac['yards_after_catch'].copy()
    groups = pbp_yac['game_id'].to_numpy()
    yac_block_start = perf_counter()
    training = train_model_from_config(
        feature_config,
        model_name,
        pbp_yac,
        y,
        groups,
        feature_catalog,
        target_mode="ordinal",
    )
    _timing_checkpoint("yac training call total", yac_block_start)

    yac_step_start = perf_counter()
    bins = training['bins']
    probs = training['test_probabilities']
    acts = training['y_test_class'].tolist()
    avg = (probs*bins).sum(axis=1)
    act = bins[training['y_test_class']]
    _timing_checkpoint("yac metrics prep", yac_step_start)

    yac_step_start = perf_counter()
    plt.plot(avg,act,'k.')
    finish_plot()
    _timing_checkpoint("yac scatter plot show", yac_step_start)

    yac_step_start = perf_counter()
    cdfs = []
    for i in range(len(acts)):
        cdfs.append(np.sum(probs[i,0:acts[i]]))
    plt.hist(cdfs, density=True)
    finish_plot()
    _timing_checkpoint("yac cdf histogram", yac_step_start)

    yac_step_start = perf_counter()
    save_model_artifacts(model_name, training, feature_config)
    report_builder.add_multiclass("yac", acts, probs, bins=bins, cdfs=cdfs, loss_history=training['loss_history'])
    _timing_checkpoint("yac artifacts and report", yac_step_start)
    del pbp_yac
#%%Air Yards
air_prep_start = perf_counter()
pbp_air = pbp.copy()
pbp_air = pbp_air[pbp_air['season_type'] == 'REG']
pbp_air = pbp_air[pbp_air['pass_attempt'] == 1]
pbp_air = pbp_air[pbp_air['air_yards'].notnull()]
pbp_air = pbp_air.sort_values(['passer_player_id', 'game_date'])
_timing_checkpoint("air_yards dataset prep", air_prep_start)

# Select features and target
model_name = "air_yards"
if should_train_model(model_name, selected_models):
    y = pbp_air['air_yards'].copy()
    groups = pbp_air['game_id'].to_numpy()
    training = train_model_from_config(
        feature_config,
        model_name,
        pbp_air,
        y,
        groups,
        feature_catalog,
        target_mode="ordinal",
    )
    save_model_artifacts(model_name, training, feature_config)
    y_bins = training['bins']
    probs_air = training['test_probabilities']
    report_builder.add_multiclass("air_yards", training['y_test_class'], probs_air, bins=y_bins, loss_history=training['loss_history'])
    del pbp_air
#%%sack
pbp_sack = pbp.copy()
pbp_sack = pbp_sack[pbp_sack['season_type'] == 'REG']
pbp_sack = pbp_sack[(pbp_sack['qb_dropback'] == 1)]
pbp_sack = pbp_sack.sort_values(['passer_player_id', 'game_date'])

# Target: sack (binary classification)
pbp_sack['sack'] = pbp_sack['sack'].fillna(0).astype(int)  # fill nulls if needed

model_name = "sack"
if should_train_model(model_name, selected_models):
    y = pbp_sack['sack'].copy()
    groups = pbp_sack['game_id'].to_numpy()
    training = train_model_from_config(
        feature_config,
        model_name,
        pbp_sack,
        y,
        groups,
        feature_catalog,
        target_mode="binary",
    )
    save_model_artifacts(model_name, training, feature_config)
    report_builder.add_binary_classification("sack", training['y_test_class'], training['test_probabilities'][:, 1], loss_history=training['loss_history'])
    del pbp_sack

#%%scramble
pbp_scramble = pbp.copy()
pbp_scramble = pbp_scramble[pbp_scramble['season_type'] == 'REG']
pbp_scramble = pbp_scramble[pbp_scramble['qb_dropback'] == 1]
pbp_scramble = pbp_scramble.sort_values(['passer_player_id', 'game_date'])

# Target: scramble on QB dropback (binary classification)
pbp_scramble['scramble'] = pbp_scramble['qb_scramble'].fillna(0).astype(int)

model_name = "scramble"
if should_train_model(model_name, selected_models):
    y = pbp_scramble['scramble'].copy()
    groups = pbp_scramble['game_id'].to_numpy()
    training = train_model_from_config(
        feature_config,
        model_name,
        pbp_scramble,
        y,
        groups,
        feature_catalog,
        target_mode="binary",
    )
    save_model_artifacts(model_name, training, feature_config)
    report_builder.add_binary_classification("scramble", training['y_test_class'], training['test_probabilities'][:, 1], loss_history=training['loss_history'])
    del pbp_scramble
#%%interception
pbp_int = pbp.copy()
pbp_int = pbp_int[pbp_int['season_type'] == 'REG']
pbp_int = pbp_int[(pbp_int['qb_dropback'] == 1)]
pbp_int = pbp_int.sort_values(['passer_player_id', 'game_date'])

# Target: sack (binary classification)
pbp_int['interception'] = pbp_int['interception'].fillna(0).astype(int)  # fill nulls if needed

model_name = "interception"
if should_train_model(model_name, selected_models):
    y = pbp_int['interception'].copy()
    groups = pbp_int['game_id'].to_numpy()
    training = train_model_from_config(
        feature_config,
        model_name,
        pbp_int,
        y,
        groups,
        feature_catalog,
        target_mode="binary",
    )
    save_model_artifacts(model_name, training, feature_config)
    report_builder.add_binary_classification("interception", training['y_test_class'], training['test_probabilities'][:, 1], loss_history=training['loss_history'])
    del pbp_int
#%%completion
pbp_comp = pbp.copy()
pbp_comp = pbp_comp[pbp_comp['season_type'] == 'REG']
pbp_comp = pbp_comp[(pbp_comp['qb_dropback'] == 1)]
pbp_comp = pbp_comp.sort_values(['passer_player_id', 'game_date'])
# Target: sack (binary classification)
pbp_comp['completion'] = pbp_comp['complete_pass'].fillna(0).astype(int)  # fill nulls if needed

model_name = "completion"
if should_train_model(model_name, selected_models):
    y = pbp_comp['completion'].copy()
    groups = pbp_comp['game_id'].to_numpy()
    training = train_model_from_config(
        feature_config,
        model_name,
        pbp_comp,
        y,
        groups,
        feature_catalog,
        target_mode="binary",
    )
    save_model_artifacts(model_name, training, feature_config)
    report_builder.add_binary_classification("completion", training['y_test_class'], training['test_probabilities'][:, 1], loss_history=training['loss_history'])
    del pbp_comp
#%%Punt
pbp['net_yards'] = pbp['kick_distance']-pbp['return_yards']
pbp_punt = pbp.copy()
pbp_punt = pbp_punt[pbp_punt['season_type'] == 'REG']
pbp_punt = pbp_punt[pbp_punt['punt_attempt'] == 1]
pbp_punt = pbp_punt[pbp_punt['net_yards'].notnull()]

# Define modeling columns
model_name = "punt"
if should_train_model(model_name, selected_models):
    y = pbp_punt['net_yards'].copy()
    groups = pbp_punt['game_id'].to_numpy()
    training = train_model_from_config(
        feature_config,
        model_name,
        pbp_punt,
        y,
        groups,
        feature_catalog,
        target_mode="ordinal",
    )
    save_model_artifacts(model_name, training, feature_config)
    bins = training['bins']
    probs_punt = training['test_probabilities']
    report_builder.add_multiclass("punt", training['y_test_class'], probs_punt, bins=bins, loss_history=training['loss_history'])
    del pbp_punt
#%%Time Elapsed
pbp = pbp.sort_values(['game_id', 'play_id']).copy()
pbp = pbp[(pbp['play_type']!='no_play') & (pbp['play_type']!='kickoff')]
# Reuse the full-sequence elapsed_time computed earlier (before no_play/kickoff/penalty rows were
# dropped) instead of re-deriving it here: diffing after those rows are removed would silently
# absorb their time into the surrounding gap.
pbp = pbp.merge(pbp_with_penalties[['game_id', 'play_id', 'elapsed_time']], on=['game_id', 'play_id'], how='left')

# Drop first play of each game (diff will be NaN)
pbp = pbp[pbp['elapsed_time'].notnull()]
# Clip extreme values or filter (e.g., if time elapsed is > 120s probably a drive change)
pbp = pbp[(pbp['elapsed_time'] >= 1) & (pbp['elapsed_time'] <= 120)]  # ~play clock max
plt.hist(pbp['elapsed_time'])
finish_plot()

# Categorize play types for modeling
def classify_play(row):
    if row.get("incomplete_pass") == 1:
        return "incomplete_pass"
    elif row.get("complete_pass") == 1:
        return "complete_pass"
    elif row.get("rush_attempt") == 1:
        return "rush"
    elif row.get("sack") == 1:
        return "sack"
    elif row.get("punt_attempt") == 1:
        return "punt"
    elif row.get("interception") == 1:
        return "interception"
    elif row.get("field_goal_attempt") == 1:
        return "field_goal_attempt"
    else:
        return "other"

pbp['play_type_cat'] = pbp.apply(classify_play, axis=1)

# Standalone ("instead of a play") penalties negate the down and it gets replayed, with no actual
# play run (nflverse marks these rows play_type == 'no_play'). These never reach this point today
# because the top-level penalty==0 filter removes them, so give them their own play_type_cat
# category here, sourced from the penalty-inclusive snapshot, and fold them into a local training
# frame used only for the time_elapsed model below (this does not touch the `pbp` used by the
# sections that follow).
pbp_no_play_penalty = pbp_with_penalties[
    (pbp_with_penalties['penalty'] == 1) & (pbp_with_penalties['play_type'] == 'no_play')
].copy()
pbp_no_play_penalty = pbp_no_play_penalty[pbp_no_play_penalty['elapsed_time'].notnull()]
pbp_no_play_penalty = pbp_no_play_penalty[
    (pbp_no_play_penalty['elapsed_time'] >= 1) & (pbp_no_play_penalty['elapsed_time'] <= 120)
]
pbp_no_play_penalty['play_type_cat'] = 'no_play_penalty'

pbp_time_train = pd.concat([pbp, pbp_no_play_penalty], ignore_index=True, sort=False)
y_cat = pd.get_dummies(pbp_time_train['elapsed_time'])

# Modeling columns
play_cols = ['play_type_cat']
time_catalog = build_feature_catalog({
    "game_state_cols": game_state_cols,
    "play_cols": play_cols,
    "pregame_context_columns": feature_catalog.get("pregame_context_columns", []),
    "qb_context_columns": feature_catalog.get("qb_context_columns", []),
    "team_stats_columns": feature_catalog.get("team_stats_columns", []),
    "def_cols": feature_catalog.get("def_cols", []),
})
model_name = "time_elapsed"
if should_train_model(model_name, selected_models):
    groups = pbp_time_train['game_id'].to_numpy()
    training = train_model_from_config(
        feature_config,
        model_name,
        pbp_time_train,
        pbp_time_train['elapsed_time'],
        groups,
        time_catalog,
        prior_catalog=feature_catalog,
        target_mode="ordinal",
    )
    save_model_artifacts(model_name, training, feature_config)
    probs_time = training['test_probabilities']
    report_builder.add_multiclass(
        "time_elapsed",
        training['y_test_class'],
        probs_time,
        bins=training['bins'],
        loss_history=training['loss_history'],
    )
    bins = training['bins']
#%%Penalty Modeling
# Builds four lightweight penalty models from the penalty-inclusive snapshot captured near the
# top of the script (pbp_with_penalties), which the rest of the pipeline intentionally excludes
# via the penalty==0 filter:
#   - penalty_no_play: probability that a snap opportunity results in a standalone penalty
#     (down replayed, no play run) instead of a normal play.
#   - penalty_on_play: probability that an accepted penalty occurs in addition to a normal play
#     that was run (conditioned on the play's outcome type via play_type_cat).
#   - penalty_yards: signed, offense-relative yardage assigned by an accepted penalty (shared
#     across both cases above via the penalty_context categorical feature).
#   - penalty_first_down: whether an accepted penalty grants the offense an automatic first down
#     (shared across both cases above).
penalty_catalog = build_feature_catalog({
    "game_state_cols": game_state_cols,
    "team_stats_columns": feature_catalog.get("team_stats_columns", []),
    "def_cols": feature_catalog.get("def_cols", []),
    "pregame_context_columns": feature_catalog.get("pregame_context_columns", []),
    "play_cols": play_cols,
    "penalty_context_cols": ["penalty_context"],
})

# Same "is this a real play" test used by the play_type model (qb_dropback/rush/punt_attempt/
# field_goal_attempt); every classify_play() category other than "other" implies one of these.
is_real_play_attempt = (
    (pbp_with_penalties['qb_dropback'] == 1)
    | (pbp_with_penalties['rush'] == 1)
    | (pbp_with_penalties['punt_attempt'] == 1)
    | (pbp_with_penalties['field_goal_attempt'] == 1)
)
is_no_play_penalty = (pbp_with_penalties['play_type'] == 'no_play') & (pbp_with_penalties['penalty'] == 1)

model_name = "penalty_no_play"
if should_train_model(model_name, selected_models):
    pbp_penalty_no_play = pbp_with_penalties[is_real_play_attempt | is_no_play_penalty].copy()
    pbp_penalty_no_play['penalty_no_play'] = (
        (pbp_penalty_no_play['play_type'] == 'no_play') & (pbp_penalty_no_play['penalty'] == 1)
    ).astype(int)
    y = pbp_penalty_no_play['penalty_no_play'].copy()
    groups = pbp_penalty_no_play['game_id'].to_numpy()
    training = train_model_from_config(
        feature_config,
        model_name,
        pbp_penalty_no_play,
        y,
        groups,
        penalty_catalog,
        target_mode="binary",
    )
    save_model_artifacts(model_name, training, feature_config)
    report_builder.add_binary_classification(
        "penalty_no_play",
        training['y_test_class'],
        training['test_probabilities'][:, 1],
        loss_history=training['loss_history'],
    )
    del pbp_penalty_no_play

model_name = "penalty_on_play"
if should_train_model(model_name, selected_models):
    pbp_penalty_on_play = pbp_with_penalties[is_real_play_attempt].copy()
    pbp_penalty_on_play['play_type_cat'] = pbp_penalty_on_play.apply(classify_play, axis=1)
    pbp_penalty_on_play['penalty_on_play'] = pbp_penalty_on_play['penalty'].fillna(0).astype(int)
    y = pbp_penalty_on_play['penalty_on_play'].copy()
    groups = pbp_penalty_on_play['game_id'].to_numpy()
    training = train_model_from_config(
        feature_config,
        model_name,
        pbp_penalty_on_play,
        y,
        groups,
        penalty_catalog,
        target_mode="binary",
    )
    save_model_artifacts(model_name, training, feature_config)
    report_builder.add_binary_classification(
        "penalty_on_play",
        training['y_test_class'],
        training['test_probabilities'][:, 1],
        loss_history=training['loss_history'],
    )
    del pbp_penalty_on_play

pbp_penalty_outcomes = pbp_with_penalties[pbp_with_penalties['penalty'] == 1].copy()
pbp_penalty_outcomes = pbp_penalty_outcomes[
    pbp_penalty_outcomes['penalty_yards'].notnull() & pbp_penalty_outcomes['penalty_team'].notnull()
].copy()
pbp_penalty_outcomes['penalty_context'] = np.where(
    pbp_penalty_outcomes['play_type'] == 'no_play', 'no_play', 'on_play'
)
pbp_penalty_outcomes['play_type_cat'] = np.where(
    pbp_penalty_outcomes['penalty_context'] == 'no_play',
    'not_applicable',
    pbp_penalty_outcomes.apply(classify_play, axis=1),
)
pbp_penalty_outcomes['penalty_signed_yards'] = np.where(
    pbp_penalty_outcomes['penalty_team'] == pbp_penalty_outcomes['posteam'],
    -pbp_penalty_outcomes['penalty_yards'],
    pbp_penalty_outcomes['penalty_yards'],
)

model_name = "penalty_yards"
if should_train_model(model_name, selected_models):
    y = pbp_penalty_outcomes['penalty_signed_yards'].copy()
    groups = pbp_penalty_outcomes['game_id'].to_numpy()
    training = train_model_from_config(
        feature_config,
        model_name,
        pbp_penalty_outcomes,
        y,
        groups,
        penalty_catalog,
        target_mode="ordinal",
    )
    y_bins = training['bins']
    probs = training['test_probabilities']
    avg = (probs * y_bins).sum(axis=1)
    act = y_bins[training['y_test_class']]
    plt.plot(avg, act, 'k.')
    finish_plot()
    acts = training['y_test_class'].tolist()
    cdfs = []
    for i in range(len(acts)):
        cdfs.append(np.sum(probs[i, 0:acts[i]]))
    plt.hist(cdfs, density=True)
    finish_plot()

    save_model_artifacts(model_name, training, feature_config)
    report_builder.add_multiclass("penalty_yards", acts, probs, bins=y_bins, cdfs=cdfs, loss_history=training['loss_history'])

model_name = "penalty_first_down"
if should_train_model(model_name, selected_models):
    y = pbp_penalty_outcomes['first_down_penalty'].fillna(0).astype(int).copy()
    groups = pbp_penalty_outcomes['game_id'].to_numpy()
    training = train_model_from_config(
        feature_config,
        model_name,
        pbp_penalty_outcomes,
        y,
        groups,
        penalty_catalog,
        target_mode="binary",
    )
    save_model_artifacts(model_name, training, feature_config)
    report_builder.add_binary_classification(
        "penalty_first_down",
        training['y_test_class'],
        training['test_probabilities'][:, 1],
        loss_history=training['loss_history'],
    )
del pbp_penalty_outcomes
#%%Kicks
# Filter for kicks (you can refine to FG or PAT as needed)
pbp_kick = pbp.copy()
pbp_kick = pbp_kick[pbp_kick['season_type'] == 'REG']
pbp_kick = pbp_kick[(pbp_kick['field_goal_attempt'] == 1) | (pbp_kick['extra_point_attempt'] == 1)]
pbp_kick = pbp_kick[pbp_kick['kick_attempt'] == 1] if 'kick_attempt' in pbp_kick.columns else pbp_kick

# Target: 1 if made, 0 if missed
pbp_kick['kick_made'] = (
    (pbp_kick['field_goal_result'] == 'made') | 
    (pbp_kick['extra_point_result'] == 'good')
).astype(int)

# Drop if result is NA (bad snap, blocked, etc., optional)
pbp_kick = pbp_kick[pbp_kick['kick_made'].notnull()]
y = pbp_kick['kick_made']
model_name = "kick"
if should_train_model(model_name, selected_models):
    groups = pbp_kick['game_id'].to_numpy()
    training = train_model_from_config(
        feature_config,
        model_name,
        pbp_kick,
        y,
        groups,
        feature_catalog,
        target_mode="binary",
    )
    save_model_artifacts(model_name, training, feature_config)
    report_builder.add_binary_classification("kick", training['y_test_class'], training['test_probabilities'][:, 1], loss_history=training['loss_history'])
    del pbp_kick
#%%Rush Yards

pbp_rush = pbp.copy()
pbp_rush = pbp_rush[pbp_rush['season_type'] == 'REG']
pbp_rush = pbp_rush[
    (pbp_rush['rush_attempt'] == 1)
    & (pbp_rush['qb_scramble'].fillna(0) == 0)
    & (pbp_rush['rushing_yards'].notnull())
]
pbp_rush = pbp_rush.merge(player_positions,left_on='rusher_player_id',right_index=True,how='left')
# Join rusher stats
missing_rusher_columns = [c for c in rusher_columns if c not in player_stats_wide.columns]
if missing_rusher_columns:
    print(f"Missing rusher columns in player_stats_wide: {missing_rusher_columns}")

pbp_rush = pbp_rush.merge(player_stats_wide.reindex(columns=rusher_columns),
                          left_on=['rusher_player_id', 'game_id'],
                          right_index=True,
                          how='left')

# Final model features
model_name = "rush_yards"
if should_train_model(model_name, selected_models):
    y = pbp_rush['rushing_yards'].copy()
    groups = pbp_rush['game_id'].to_numpy()
    training = train_model_from_config(
        feature_config,
        model_name,
        pbp_rush,
        y,
        groups,
        feature_catalog,
        target_mode="ordinal",
    )
    y_bins = training['bins']
    probs = training['test_probabilities']
    avg = (probs*y_bins).sum(axis=1)
    act = y_bins[training['y_test_class']]
    plt.plot(avg,act,'k.')
    finish_plot()
    acts = training['y_test_class'].tolist()
    cdfs = []
    for i in range(len(acts)):
        cdfs.append(np.sum(probs[i,0:acts[i]]))
    plt.hist(cdfs, density=True)
    finish_plot()

    save_model_artifacts(model_name, training, feature_config)
    report_builder.add_multiclass("rush_yards", acts, probs, bins=y_bins, cdfs=cdfs, loss_history=training['loss_history'])
    del pbp_rush

#%%Scramble Yards
pbp_scramble = pbp.copy()
pbp_scramble = pbp_scramble[pbp_scramble['season_type'] == 'REG']
pbp_scramble = pbp_scramble[
    (pbp_scramble['rush_attempt'] == 1)
    & (pbp_scramble['qb_scramble'].fillna(0) == 1)
    & (pbp_scramble['rushing_yards'].notnull())
]
pbp_scramble = pbp_scramble.merge(player_positions, left_on='rusher_player_id', right_index=True, how='left')
pbp_scramble = pbp_scramble.merge(
    player_stats_wide.reindex(columns=rusher_columns),
    left_on=['rusher_player_id', 'game_id'],
    right_index=True,
    how='left',
)

model_name = "scramble_yards"
if should_train_model(model_name, selected_models):
    y = pbp_scramble['rushing_yards'].copy()
    groups = pbp_scramble['game_id'].to_numpy()
    training = train_model_from_config(
        feature_config,
        model_name,
        pbp_scramble,
        y,
        groups,
        feature_catalog,
        target_mode="ordinal",
    )
    y_bins = training['bins']
    probs = training['test_probabilities']
    avg = (probs * y_bins).sum(axis=1)
    act = y_bins[training['y_test_class']]
    plt.plot(avg, act, 'k.')
    finish_plot()
    acts = training['y_test_class'].tolist()
    cdfs = []
    for i in range(len(acts)):
        cdfs.append(np.sum(probs[i, 0:acts[i]]))
    plt.hist(cdfs, density=True)
    finish_plot()

    save_model_artifacts(model_name, training, feature_config)
    report_builder.add_multiclass("scramble_yards", acts, probs, bins=y_bins, cdfs=cdfs, loss_history=training['loss_history'])
    del pbp_scramble
#%%Torch Play
play = pbp.copy()

play['play_type'] = np.select(
    [
        play['qb_dropback'] == 1,
        play['rush'] == 1,
        play['punt_attempt'] == 1,
        play['field_goal_attempt'] == 1
    ],
    ['pass', 'rush', 'punt', 'kick'],
    default='other'
)

# Filter to just valid offensive play types
play = play[play['play_type'].isin(['pass', 'rush', 'punt', 'kick'])]

# Preserve the exact prepared feature table consumed by the standalone ARMED
# research runner. This happens after all production feature merges and target
# definitions, so the export cannot drift from the normal play_type model.
if CLI_ARGS.export_play_type_data or CLI_ARGS.export_play_type_data_only:
    export_columns = unique_columns(
        ['game_id', 'posteam', 'play_type']
        + resolve_model_features(feature_config, 'play_type', feature_catalog)
    )
    missing_export_columns = [column for column in export_columns if column not in play.columns]
    if missing_export_columns:
        raise KeyError(f"Cannot export play_type data; missing columns: {missing_export_columns}")
    play[export_columns].to_csv(
        Path(__file__).resolve().parent / 'play_type_training.csv',
        index=False,
    )
    print(f"[export] play_type ARMED data: {len(play):,} rows")
    if CLI_ARGS.export_play_type_data_only:
        raise SystemExit(0)

# Final model features
model_name = "play_type"
if should_train_model(model_name, selected_models):
    modeling_columns = resolve_model_features(feature_config, model_name, feature_catalog)
    X = play[modeling_columns].copy()
    y = play['play_type'].copy()
    groups = pd.Series(play['game_id'].to_numpy(), index=play.index)

    # Diagnose and remove unusable features before preprocessing.
    missing_rate = X.isna().mean().sort_values(ascending=False)
    highly_missing = missing_rate[missing_rate >= 0.95]
    if not highly_missing.empty:
        print("play_type high-missing features (>=95% NaN):")
        print(highly_missing)

    all_nan_cols = missing_rate[missing_rate >= 1.0].index.tolist()
    if all_nan_cols:
        print(f"Dropping all-NaN play_type columns: {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)

    categorical_cols = [
        col for col in resolve_model_categorical_cols(feature_config, model_name)
        if col in X.columns
    ]
    y_cat, y_class = prepare_class_targets(y)
    is_hierarchical = should_use_hierarchical_model(feature_config, model_name)
    if is_hierarchical:
        prior_columns = resolve_model_prior_features(feature_config, model_name, feature_catalog)
        combined_frame = X.reindex(columns=unique_columns(modeling_columns + prior_columns)).copy()
        combined_groups = hierarchical_group_labels(play, groups).loc[combined_frame.index]
        y_series = pd.Series(y_class, index=combined_frame.index)
        train_frame, test_frame, y_train, y_test = blocked_train_test_split(
            combined_frame,
            y_series,
            groups=combined_groups,
            test_size=0.2,
            random_state=42,
        )
        play_type_test_index = test_frame.index
        y_test_class = y_test.values
        train_group_values = combined_groups.loc[train_frame.index].to_numpy()

        def train_fn(hparams, optuna_trial):
            ngmm_arch = hparams.get('ngmm_arch') or hparams.get('arch') or {}
            result = train_ngmm_grouped(
                train_frame=train_frame,
                test_frame=test_frame,
                y_train=y_train,
                y_test=y_test,
                train_groups=train_group_values,
                play_columns=modeling_columns,
                prior_columns=prior_columns,
                categorical_cols=categorical_cols,
                target_mode="multiclass",
                output_dim=y_cat.shape[1],
                max_epochs=max_epochs_for(optuna_trial, 200),
                batch_size=512,
                latent_dim=int(hparams.get('latent_dim') or resolve_model_latent_dim(feature_config, model_name, default=1)),
                quadrature_samples=int(hparams.get('quadrature_samples', 32)),
                observations_per_group=int(hparams.get('observations_per_group', 256)),
                learning_rate=float(hparams.get('learning_rate', 1e-3)),
                weight_decay=float(hparams.get('weight_decay', 0.0)),
                dropout=float(ngmm_arch.get('dropout', 0.1) if isinstance(ngmm_arch.get('dropout', 0.1), (int, float)) else ngmm_arch.get('dropout', [0.1])[0]),
                hidden_dims=tuple(ngmm_arch.get('hidden_dims', [128, 64, 32])),
                context_hidden_dims=tuple(ngmm_arch.get('context_hidden_dims', [64, 32])),
                log_progress=optuna_trial is None,
            )
            result.update({'y_test_class': y_test_class, 'bins': None, 'torch_model': result['model'], 'hyperparameters': hparams})
            return result

        # The custom play_type block uses the same NGMM contract as generic targets.
    else:
        numerical_cols = [col for col in X.columns if col not in categorical_cols]
        X_train, X_test, y_train_df, y_test_df = blocked_train_test_split(
            X,
            y_cat,
            groups=groups.loc[X.index],
            test_size=0.2,
            random_state=42,
        )
        preprocessor_play = ColumnTransformer([
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
        X_train_processed = preprocessor_play.fit_transform(X_train)
        X_test_processed = preprocessor_play.transform(X_test)

        X_train_processed[np.isnan(X_train_processed)] = 0
        X_test_processed[np.isnan(X_test_processed)] = 0
        play_type_test_index = X_test.index
        y_test_class = np.argmax(y_test_df.values.astype(np.int64), axis=1)
        y_train_class_play = np.argmax(y_train_df.values.astype(np.int64), axis=1)

        def train_fn(hparams, optuna_trial):
            fitted = fit_standard_classifier(
                X_train_processed,
                X_test_processed,
                y_train_class_play,
                output_dim=y_cat.shape[1],
                ordinal_output=False,
                max_epochs=max_epochs_for(optuna_trial, 1500),
                hparams=hparams,
                optuna_trial=optuna_trial,
                use_checkpoint=optuna_trial is None,
            )
            fitted.update({
                'play_preprocessor': preprocessor_play,
                'prior_preprocessor': None,
                'y_test_class': y_test_class,
                'bins': None,
                'torch_model': fitted['model'].module_,
                'loss_history': fitted['model'].history,
            })
            return fitted

    training = run_or_tune(
        model_name,
        "multiclass",
        is_hierarchical,
        train_fn,
        framework="ngmm" if is_hierarchical else "legacy",
    )
    y_pred_logits = training['test_probabilities']
    loss_history = training['loss_history']
    save_model_artifacts(model_name, training, feature_config)

    y_pred_classes = np.argmax(y_pred_logits, axis=1)
    report_builder.add_multiclass("play_type", y_test_class, y_pred_logits, loss_history=loss_history)
    if play_type_test_index is not None:
        bucket_series = build_playcall_bucket_series(play.loc[play_type_test_index])
        write_play_type_bucket_calibration(
            bucket_keys=bucket_series.to_numpy(),
            y_true_class=y_test_class,
            y_pred_probs=y_pred_logits,
            class_names=y_cat.columns.tolist(),
        )
    
    # Confusion matrix
    cm = confusion_matrix(y_test_class, y_pred_classes)
    
    # Cleanup
    del play
    
#%%Depth Chart Fixing
rosters = rosters[rosters['status']=='ACT']
positions = ['WR','RB','TE','QB']
rosters = rosters[rosters['position'].isin(['WR','RB','TE','QB'])]
rosters['game_id'] = rosters['team']+rosters['season'].astype(str)+rosters['week'].astype(str)
depth_chart = depth_chart[np.isnan(depth_chart['week'])==0]
depth_chart['game_id'] = depth_chart['club_code']+depth_chart['season'].astype(str)+depth_chart['week'].astype(int).astype(str)
injuries = nfl.import_injuries(SEASONS)
injuries = injuries[injuries['report_status']=='Out']
rosters = rosters.merge(injuries.set_index(['week','season','gsis_id'])['report_status'], left_on = ['week','season','player_id'], right_index=True, how = 'left')
rosters = rosters[rosters['report_status']!='Out']
pbp['merge_key'] = pbp['posteam'] + pbp['season'].astype(str) + pbp['week'].astype(str)
merge_mapping = pbp[['merge_key', 'game_id']].drop_duplicates()
merge_mapping.dropna(inplace=True)
# Now map
rosters['game_id'] = rosters['game_id'].map(merge_mapping.set_index('merge_key')['game_id'])
player_stats_wide.drop(columns=['game_id'], inplace=True, errors='ignore')
rosters = rosters.merge(player_stats_wide, left_on=['player_id', 'game_id'], right_index=True, how='left')
rosters['alt_game_id'] = rosters['team']+rosters['season'].astype(str)+rosters['week'].astype(str)
depth_chart_qb = depth_chart[depth_chart['position'] == 'QB']

qb_roster = rosters[rosters['position'] == 'QB']

# Merge on game_id and player_id to get depth info for each QB in roster
qb_with_depth = qb_roster.merge(
    depth_chart_qb.set_index(['game_id', 'gsis_id'])['depth_team'], 
    left_on=['alt_game_id', 'player_id'],
    right_index=True,
    how='left'
)
qb_with_depth_sorted = qb_with_depth.sort_values(by=['alt_game_id', 'depth_team'])
qb_with_depth_sorted.dropna(subset='depth_team',inplace=True)
top_qb_per_game = (
    qb_with_depth_sorted
    .groupby('alt_game_id')
    .first()
    .reset_index()
)
#%%
wr_te_sorted = (
    rosters[rosters['position'].isin(['WR', 'TE'])]
    .sort_values(by='medium', ascending=False)
).reset_index(drop=True)

# Sort RB by 'rush_rate'
rb_sorted = (
    rosters[rosters['position'] == 'RB']
    .sort_values(by='rush_rate', ascending=False)
).reset_index(drop=True)
sorted_players = pd.concat([top_qb_per_game,wr_te_sorted, rb_sorted], ignore_index=True)
#%%
position_limits = {'QB': 1, 'WR': 4, 'RB': 3, 'TE': 3}
# Keep only relevant positions
sorted_players['rank'] = sorted_players.groupby(['game_id','team', 'position']).cumcount()
sorted_players = sorted_players.sort_values(by=['game_id','team']).reset_index(drop=True)

# Filter to top N by position
position_filtered = pd.concat([
    sorted_players[sorted_players['position'] == pos].groupby(['game_id','team']).head(n)
    for pos, n in position_limits.items()
])

# Assign a label like QB1, WR1, WR2, ..., TE3
position_filtered['pos_rank'] = position_filtered.groupby(['game_id', 'team', 'position']).cumcount() + 1
position_filtered['label'] = position_filtered['position'] + position_filtered['pos_rank'].astype(str)

# Pivot to get GSIS IDs per label
top_skill_df = position_filtered.pivot(index=['team','game_id'], columns='label', values='player_id').reset_index()

# Optional: sort columns in desired order
column_order = ['game_id'] + [f'QB1'] + [f'WR{i}' for i in range(1, 5)] + \
               [f'RB{i}' for i in range(1, 4)] + [f'TE{i}' for i in range(1, 4)]
top_skill_df = top_skill_df[[col for col in column_order if col in top_skill_df.columns]+['team']]
nan_counts = top_skill_df.isna().sum(axis=1)
top_skill_df = top_skill_df[nan_counts<=0]
top_skill_df = top_skill_df.merge(pbp.set_index('game_id')['game_date'],left_on='game_id',right_index=True,how='left').drop_duplicates()
#top_skill_df.to_csv(BASE_DIR / 'depth_charts.csv')
#%%
pbp_pass = pbp[pbp['pass_attempt'] == 1].copy()
pbp_pass = pbp_pass.merge(top_skill_df, on='game_id', how='left')
skill_cols = [col for col in top_skill_df.columns if (col != 'game_id' and col != 'team' and col != 'game_date')]

# Keep plays where the target is in the skill player set
pbp_pass = pbp_pass[pbp_pass['receiver_player_id'].isin(pbp_pass[skill_cols].stack().unique())]

# Label the target as an index (0 to N-1) corresponding to which player in the list was targeted
def get_target_index(row):
    for i, col in enumerate(skill_cols):
        if row['receiver_player_id'] == row[col]:
            return i
    return -1

pbp_pass['target_index'] = pbp_pass.apply(get_target_index, axis=1)
pbp_pass = pbp_pass[pbp_pass['target_index']>=0]
y = pbp_pass['target_index']
#%%
skill_long = top_skill_df.melt(id_vars=['game_id','game_date'], value_name='gsis_id', var_name='slot').drop_duplicates()
# First, make sure the dates are datetime and both DataFrames are sorted properly
skill_long['game_date'] = pd.to_datetime(skill_long['game_date'])
player_stats_wide['game_date'] = pd.to_datetime(player_stats_wide['game_date'])

# Sort both DataFrames by player_id and game_date
skill_long_sorted = skill_long.sort_values(['game_date', 'gsis_id'])
player_stats_sorted = player_stats_wide.sort_values(['game_date','player_id'])
skill_long_sorted['player_id'] = skill_long_sorted['gsis_id']
# Select only the columns you want to merge in
cols = [col for col in player_stats_sorted.columns if col not in skill_long_sorted.columns]+['game_date']
# Merge using merge_asof — this matches on the closest previous game_date per player
skill_long = pd.merge_asof(
    skill_long_sorted,
    player_stats_sorted[cols],
    by='player_id',
    on ='game_date',
    direction='backward'
)

# Pivot back to wide format: one row per game, one column per stat per slot
# `values` is restricted to the numeric stat columns so pandas doesn't try to
# mean-aggregate string id columns (e.g. gsis_id/player_id).
receiver_stats_long = skill_long.pivot_table(
    index='game_id', columns='slot', values=[col for col in receiver_columns if col in skill_long.columns]
)
receiver_stats_long.columns = ['_'.join(col).strip() for col in receiver_stats_long.columns]
receiver_stats_long = receiver_stats_long.reset_index()

# Merge these expanded receiver stats into pbp_pass
pbp_pass = pbp_pass.merge(receiver_stats_long, on='game_id', how='left')
unique_game_ids = pbp_pass['game_team_id'].unique()
game_id_to_idx = {game_id: idx for idx, game_id in enumerate(unique_game_ids)}
pbp_pass['game_idx'] = pbp_pass['game_team_id'].map(game_id_to_idx)
n_games = len(unique_game_ids)

# Separate the 'game_idx' from the main feature matrix X
slot_stat_cols = [col for col in receiver_stats_long.columns if 'game_id' not in col]
X_main = pbp_pass[game_state_cols + slot_stat_cols].copy()
game_indices = pbp_pass['game_idx']
groups = pbp_pass['game_id'].to_numpy()

# Split all components together
X_train_main, X_test_main, y_train, y_test = blocked_train_test_split(
    X_main,
    y,
    groups=groups,
    test_size=0.2,
    random_state=42,
)
game_idx_train = game_indices.loc[X_train_main.index]
game_idx_test = game_indices.loc[X_test_main.index]

# Preprocessing setup (Only for play-level features)
categorical_cols = ['roof', 'posteam_type']
numerical_cols = list(set(X_main.columns) - set(categorical_cols))

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Process the main feature matrix
X_train_processed = preprocessor.fit_transform(X_train_main)
X_test_processed = preprocessor.transform(X_test_main)

# Handle NaNs
X_train_processed[np.isnan(X_train_processed)] = 0
X_test_processed[np.isnan(X_test_processed)] = 0

# Convert to numpy and then to torch tensors
X_train_np = X_train_processed.astype(np.float32)
X_test_np = X_test_processed.astype(np.float32)
y_train_class = y_train.values.astype(np.int64)
y_test_class = y_test.values.astype(np.int64)

# Game indices must also be converted to tensors (long for embedding lookup)
game_idx_train_torch = torch.tensor(game_idx_train.values, dtype=torch.long)
game_idx_test_torch = torch.tensor(game_idx_test.values, dtype=torch.long)
input_dim = X_train_np.shape[1]

# Combine feature tensor and game_idx tensor into a tuple for skorch
X_train_combined = (torch.tensor(X_train_np), game_idx_train_torch)
X_test_combined = (torch.tensor(X_test_np), game_idx_test_torch)
n_classes = y_test_class.max()+1
# Initialize skorch model
model_play = NeuralNetClassifier(
    module=HierarchicalGameModel,
    module__input_dim=input_dim,
    module__output_dim=n_classes,
    module__n_games=n_games,
    module__game_embedding_dim=2,
    max_epochs=1500,  # shorten for demo
    batch_size=512,
    optimizer=torch.optim.AdamW,
    criterion=nn.CrossEntropyLoss,
    callbacks=[EarlyStopping(patience=20, load_best=True)],
    iterator_train__shuffle=True,
    device='cuda' if torch.cuda.is_available() else 'cpu',
)


#%%
# Include game state columns and per-slot stats
slot_stat_cols = [col for col in receiver_stats_long.columns if col[:-4] in receiver_columns]
model_name = "target_selection"
if should_train_model(model_name, selected_models):
    play_catalog = build_feature_catalog(
        {
            "game_state_cols": game_state_cols,
            "pass_columns": pass_columns,
            "slot_stat_cols": slot_stat_cols,
            "pregame_context_columns": feature_catalog.get("pregame_context_columns", []),
        }
    )
    prior_catalog = build_feature_catalog(
        {
            "team_stats_columns": feature_catalog.get("team_stats_columns", team_stats_columns),
            "team_tendency_columns": feature_catalog.get("team_tendency_columns", team_tendency_columns),
            "qb_context_columns": feature_catalog.get("qb_context_columns", qb_context_columns),
            "pregame_context_columns": feature_catalog.get("pregame_context_columns", []),
        }
    )
    play_columns = resolve_model_features(feature_config, model_name, play_catalog)
    prior_columns = resolve_model_prior_features(feature_config, model_name, prior_catalog)
    combined_frame = pbp_pass[unique_columns(play_columns + prior_columns)].copy()
    combined_groups = hierarchical_group_labels(pbp_pass, pbp_pass['game_team_id'].to_numpy()).loc[combined_frame.index]
    output_dim = int(y.max()) + 1
    latent_dim = resolve_model_latent_dim(feature_config, model_name, default=8)
    kl_weight = resolve_model_kl_weight(feature_config, model_name, default=0.05)
    
    train_frame, test_frame, y_train, y_test = blocked_train_test_split(
        combined_frame,
        y,
        test_size=0.2,
        random_state=42,
        groups=combined_groups,
    )
    
    if should_use_hierarchical_model(feature_config, model_name):
        y_test_class = y_test.values
        train_group_values = combined_groups.loc[train_frame.index].to_numpy()

        def train_fn(hparams, optuna_trial):
            ngmm_arch = hparams.get('ngmm_arch') or hparams.get('arch') or {}
            result = train_ngmm_grouped(
                train_frame=train_frame,
                test_frame=test_frame,
                y_train=y_train,
                y_test=y_test,
                train_groups=train_group_values,
                play_columns=play_columns,
                prior_columns=prior_columns,
                categorical_cols=resolve_model_categorical_cols(feature_config, model_name),
                target_mode="multiclass",
                output_dim=output_dim,
                max_epochs=max_epochs_for(optuna_trial, 200),
                batch_size=512,
                latent_dim=int(hparams.get('latent_dim') or latent_dim),
                quadrature_samples=int(hparams.get('quadrature_samples', 32)),
                observations_per_group=int(hparams.get('observations_per_group', 256)),
                learning_rate=float(hparams.get('learning_rate', 1e-3)),
                weight_decay=float(hparams.get('weight_decay', 0.0)),
                dropout=float(ngmm_arch.get('dropout', 0.1) if isinstance(ngmm_arch.get('dropout', 0.1), (int, float)) else ngmm_arch.get('dropout', [0.1])[0]),
                hidden_dims=tuple(ngmm_arch.get('hidden_dims', [128, 64, 32])),
                context_hidden_dims=tuple(ngmm_arch.get('context_hidden_dims', [64, 32])),
                log_progress=optuna_trial is None,
            )
            result.update({'y_test_class': y_test_class, 'bins': None, 'torch_model': result['model'], 'hyperparameters': hparams})
            return result

        result = run_or_tune(model_name, "multiclass", True, train_fn, framework="ngmm")
        save_model_artifacts(model_name, result, feature_config)
        report_builder.add_multiclass(
            "target_selection",
            y_test_class,
            result["test_probabilities"],
            loss_history=result.get("loss_history"),
        )
    else:
        X = pbp_pass[play_columns].copy()
        categorical_cols = resolve_model_categorical_cols(feature_config, model_name)
        numerical_cols = list(set(X.columns) - set(categorical_cols))
        groups = pd.Series(pbp_pass['game_id'].to_numpy(), index=X.index)

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

        X_train, X_test, y_train, y_test = blocked_train_test_split(X, y, groups=groups, test_size=0.2, random_state=42)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        X_train_processed[np.isnan(X_train_processed)] = 0
        X_test_processed[np.isnan(X_test_processed)] = 0

        y_train_class = y_train.values.astype(np.int64)
        y_test_class = y_test.values.astype(np.int64)

        def train_fn(hparams, optuna_trial):
            fitted = fit_standard_classifier(
                X_train_processed,
                X_test_processed,
                y_train_class,
                output_dim=output_dim,
                ordinal_output=False,
                max_epochs=max_epochs_for(optuna_trial, 1500),
                hparams=hparams,
                optuna_trial=optuna_trial,
                use_checkpoint=optuna_trial is None,
            )
            fitted.update({
                'play_preprocessor': preprocessor,
                'prior_preprocessor': None,
                'y_test_class': y_test_class,
                'bins': None,
                'torch_model': fitted['model'].module_,
                'loss_history': fitted['model'].history,
            })
            return fitted

        result = run_or_tune(model_name, "multiclass", False, train_fn)
        save_model_artifacts(model_name, result, feature_config)
        report_builder.add_multiclass(
            "target_selection",
            y_test_class,
            result["test_probabilities"],
            loss_history=result.get("loss_history"),
        )
    
    # =============================================================================
    # #%%
    # pbp_pass = pbp[pbp['pass_attempt'] == 1].copy()
    # pbp_pass = pbp_pass.merge(top_skill_df, on='game_id', how='left')
    # skill_cols = [col for col in top_skill_df.columns if (col != 'game_id' and col != 'team' and col != 'game_date')]
    # 
    # # Keep plays where the target is in the skill player set
    # pbp_pass = pbp_pass[pbp_pass['receiver_player_id'].isin(pbp_pass[skill_cols].stack().unique())]
    # 
    # # Label the target as an index (0 to N-1) corresponding to which player in the list was targeted
    # def get_target_index(row):
    #     for i, col in enumerate(skill_cols):
    #         if row['receiver_player_id'] == row[col]:
    #             return i
    #     return -1
    # 
    # pbp_pass['target_index'] = pbp_pass.apply(get_target_index, axis=1)
    # pbp_pass = pbp_pass[pbp_pass['target_index']>=0]
    # y = pbp_pass['target_index']
    # #%%
    # skill_long = top_skill_df.melt(id_vars=['game_id','game_date'], value_name='gsis_id', var_name='slot').drop_duplicates()
    # # First, make sure the dates are datetime and both DataFrames are sorted properly
    # skill_long['game_date'] = pd.to_datetime(skill_long['game_date'])
    # player_stats_wide['game_date'] = pd.to_datetime(player_stats_wide['game_date'])
    # 
    # # Sort both DataFrames by player_id and game_date
    # skill_long_sorted = skill_long.sort_values(['game_date', 'gsis_id'])
    # player_stats_sorted = player_stats_wide.sort_values(['game_date','player_id'])
    # skill_long_sorted['player_id'] = skill_long_sorted['gsis_id']
    # # Select only the columns you want to merge in
    # cols = [col for col in player_stats_sorted.columns if col not in skill_long_sorted.columns]+['game_date']
    # # Merge using merge_asof — this matches on the closest previous game_date per player
    # skill_long = pd.merge_asof(
    #     skill_long_sorted,
    #     player_stats_sorted[cols],
    #     by='player_id',
    #     on ='game_date',
    #     direction='backward'
    # )
    # 
    # # Pivot back to wide format: one row per game, one column per stat per slot
    # receiver_stats_long = skill_long.pivot_table(index='game_id', columns='slot')
    # receiver_stats_long.columns = ['_'.join(col).strip() for col in receiver_stats_long.columns]
    # receiver_stats_long = receiver_stats_long.reset_index()
    # 
    # # Merge these expanded receiver stats into pbp_pass
    # pbp_pass = pbp_pass.merge(receiver_stats_long, on='game_id', how='left')
    # #%%
    # # Include game state columns and per-slot stats
    # slot_stat_cols = [col for col in receiver_stats_long.columns if col[:-4] in receiver_columns]
    # modeling_columns = game_state_cols+slot_stat_cols+pass_columns
    # 
    # X = pbp_pass[modeling_columns].copy()
    # 
    # categorical_cols = ['roof', 'posteam_type']
    # numerical_cols = list(set(X.columns) - set(categorical_cols))
    # 
    # # Preprocessing
    # preprocessor = ColumnTransformer([
    #     ('num', StandardScaler(), numerical_cols),
    #     ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    # ])
    # 
    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    # X_train_processed = preprocessor.fit_transform(X_train)
    # X_test_processed = preprocessor.transform(X_test)
    # 
    # X_train_processed[np.isnan(X_train_processed)] = 0
    # X_test_processed[np.isnan(X_test_processed)] = 0
    # #%%Target Model
    # input_dim = X_train_processed.shape[1]
    # n_classes = len(skill_cols)
    # 
    # # Convert to numpy arrays
    # X_train_np = X_train_processed.astype(np.float32)
    # X_test_np = X_test_processed.astype(np.float32)
    # y_train_np = y_train.values.astype(np.int64)
    # y_test_np = y_test.values.astype(np.int64)
    # y_train_class = y_train_np
    # y_test_class = y_test_np
    # 
    # # Create skorch model
    # input_dim = X_train_np.shape[1]
    # output_dim = n_classes
    # 
    # model_play = NeuralNetClassifier(
    #     module=GameModel,
    #     module__input_dim=input_dim,
    #     module__output_dim=output_dim,
    #     max_epochs=1500,
    #     batch_size=512,
    #     optimizer=torch.optim.AdamW,
    #     criterion=nn.CrossEntropyLoss,
    #     callbacks=[EarlyStopping(patience=20, load_best=True), Checkpoint(load_best=True)],
    #     iterator_train__shuffle=True,
    #     device='cuda' if torch.cuda.is_available() else 'cpu',
    # )
    # 
    # # Fit model
    # model_play.fit(X_train_np, y_train_class)
    # 
    # # Predict
    # y_pred_logits = model_play.predict_proba(X_test_np)
    # y_pred_class = (y_pred_logits > 0.5).astype(int)
    # y_pred_classes = np.argmax(y_pred_class, axis=1)
    # model_play.predict(X_test_np)
    # # Save
    # torch.save(model_play.module_.state_dict(), BASE_DIR / 'target_model.pt')
    # 
    # joblib.dump(preprocessor, BASE_DIR / 'target_preprocessor.pkl')
    # modeling_columns =['game_state_cols','pass_columns','receiver_columns']
    # joblib.dump(modeling_columns, BASE_DIR / 'target_columns.pkl')
    # #%%
    # =============================================================================
#%%
pbp_run = pbp[pbp['rush_attempt'] == 1].copy()
pbp_run = pbp_run[pbp_run['qb_scramble'] == 0]
pbp_run = pbp_run.merge(top_skill_df, on='game_id', how='left')
skill_cols = [col for col in top_skill_df.columns if (col != 'game_id' and col != 'team' and col != 'game_date')]
# Keep plays where the target is in the skill player set
pbp_run = pbp_run[pbp_run['rusher_player_id'].isin(pbp_run[skill_cols].stack().unique())]

# Label the target as an index (0 to N-1) corresponding to which player in the list was targeted
def get_rush_index(row):
    for i, col in enumerate(skill_cols):
        if row['rusher_player_id'] == row[col]:
            return i
    return -1

pbp_run['rush_index'] = pbp_run.apply(get_rush_index, axis=1)
pbp_run = pbp_run[pbp_run['rush_index']>=0]
y = pbp_run['rush_index']
#%%
#skill_long = top_skill_df.melt(id_vars='game_id', value_name='gsis_id', var_name='slot')

# Merge on receiver stats (must contain stats for each player by game)
#skill_long = skill_long.merge(player_stats_wide[rusher_columns], left_on=['gsis_id', 'game_id'],right_index=True, how='left')

# Pivot back to wide format: one row per game, one column per stat per slot
# `values` is restricted to the numeric stat columns so pandas doesn't try to
# mean-aggregate string id columns (e.g. gsis_id/player_id).
rusher_stats_long = skill_long.pivot_table(
    index='game_id', columns='slot', values=[col for col in rusher_columns if col in skill_long.columns]
)
rusher_stats_long.columns = ['_'.join(col).strip() for col in rusher_stats_long.columns]
rusher_stats_long = rusher_stats_long.reset_index()

# Merge these expanded receiver stats into pbp_pass
pbp_run = pbp_run.merge(rusher_stats_long, on='game_id', how='left')
#%%
slot_stat_cols = [col for col in rusher_stats_long.columns if col[:-4] in rusher_columns]
model_name = "rusher_selection"
if should_train_model(model_name, selected_models):
    play_catalog = build_feature_catalog({
        "game_state_cols": game_state_cols,
        "slot_stat_cols": slot_stat_cols,
        "pregame_context_columns": feature_catalog.get("pregame_context_columns", []),
    })
    prior_catalog = build_feature_catalog(
        {
            "team_stats_columns": feature_catalog.get("team_stats_columns", team_stats_columns),
            "team_tendency_columns": feature_catalog.get("team_tendency_columns", team_tendency_columns),
            "qb_context_columns": feature_catalog.get("qb_context_columns", qb_context_columns),
            "pregame_context_columns": feature_catalog.get("pregame_context_columns", []),
        }
    )
    play_columns = resolve_model_features(feature_config, model_name, play_catalog)
    prior_columns = resolve_model_prior_features(feature_config, model_name, prior_catalog)
    combined_frame = pbp_run[unique_columns(play_columns + prior_columns)].copy()
    combined_groups = hierarchical_group_labels(pbp_run, pbp_run['game_team_id'].to_numpy()).loc[combined_frame.index]
    output_dim = int(y.max()) + 1
    latent_dim = resolve_model_latent_dim(feature_config, model_name, default=8)
    kl_weight = resolve_model_kl_weight(feature_config, model_name, default=0.05)
    
    train_frame, test_frame, y_train, y_test = blocked_train_test_split(
        combined_frame,
        y,
        test_size=0.2,
        random_state=42,
        groups=combined_groups,
    )
    
    if should_use_hierarchical_model(feature_config, model_name):
        y_test_class = y_test.values
        train_group_values = combined_groups.loc[train_frame.index].to_numpy()

        def train_fn(hparams, optuna_trial):
            ngmm_arch = hparams.get('ngmm_arch') or hparams.get('arch') or {}
            result = train_ngmm_grouped(
                train_frame=train_frame,
                test_frame=test_frame,
                y_train=y_train,
                y_test=y_test,
                train_groups=train_group_values,
                play_columns=play_columns,
                prior_columns=prior_columns,
                categorical_cols=resolve_model_categorical_cols(feature_config, model_name),
                target_mode="multiclass",
                output_dim=output_dim,
                max_epochs=max_epochs_for(optuna_trial, 200),
                batch_size=512,
                latent_dim=int(hparams.get('latent_dim') or latent_dim),
                quadrature_samples=int(hparams.get('quadrature_samples', 32)),
                observations_per_group=int(hparams.get('observations_per_group', 256)),
                learning_rate=float(hparams.get('learning_rate', 1e-3)),
                weight_decay=float(hparams.get('weight_decay', 0.0)),
                dropout=float(ngmm_arch.get('dropout', 0.1) if isinstance(ngmm_arch.get('dropout', 0.1), (int, float)) else ngmm_arch.get('dropout', [0.1])[0]),
                hidden_dims=tuple(ngmm_arch.get('hidden_dims', [128, 64, 32])),
                context_hidden_dims=tuple(ngmm_arch.get('context_hidden_dims', [64, 32])),
                log_progress=optuna_trial is None,
            )
            result.update({'y_test_class': y_test_class, 'bins': None, 'torch_model': result['model'], 'hyperparameters': hparams})
            return result

        result = run_or_tune(model_name, "multiclass", True, train_fn, framework="ngmm")
        save_model_artifacts(model_name, result, feature_config)
        report_builder.add_multiclass(
            "rusher_selection",
            y_test_class,
            result["test_probabilities"],
            loss_history=result.get("loss_history"),
        )
    else:
        X = pbp_run[play_columns].copy()
        categorical_cols = resolve_model_categorical_cols(feature_config, model_name)
        numerical_cols = list(set(X.columns) - set(categorical_cols))
        groups = pd.Series(pbp_run['game_id'].to_numpy(), index=X.index)

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

        X_train, X_test, y_train, y_test = blocked_train_test_split(X, y, groups=groups, test_size=0.2, random_state=42)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        X_train_processed[np.isnan(X_train_processed)] = 0
        X_test_processed[np.isnan(X_test_processed)] = 0

        y_train_class = y_train.values.astype(np.int64)
        y_test_class = y_test.values.astype(np.int64)

        def train_fn(hparams, optuna_trial):
            fitted = fit_standard_classifier(
                X_train_processed,
                X_test_processed,
                y_train_class,
                output_dim=len(skill_cols),
                ordinal_output=False,
                max_epochs=max_epochs_for(optuna_trial, 1500),
                hparams=hparams,
                optuna_trial=optuna_trial,
                use_checkpoint=optuna_trial is None,
            )
            fitted.update({
                'play_preprocessor': preprocessor,
                'prior_preprocessor': None,
                'y_test_class': y_test_class,
                'bins': None,
                'torch_model': fitted['model'].module_,
                'loss_history': fitted['model'].history,
            })
            return fitted

        result = run_or_tune(model_name, "multiclass", False, train_fn)
        save_model_artifacts(model_name, result, feature_config)
        report_builder.add_multiclass(
            "rusher_selection",
            y_test_class,
            result["test_probabilities"],
            loss_history=result.get("loss_history"),
        )
    
    # #%%Running Selection
    # pbp_run = pbp[pbp['rush_attempt'] == 1].copy()
    # pbp_run = pbp_run.merge(top_skill_df, on='game_id', how='left')
    # skill_cols = [col for col in top_skill_df.columns if (col != 'game_id' and col != 'team' and col != 'game_date')]
    # # Keep plays where the target is in the skill player set
    # pbp_run = pbp_run[pbp_run['rusher_player_id'].isin(pbp_run[skill_cols].stack().unique())]
    
    # # Label the target as an index (0 to N-1) corresponding to which player in the list was targeted
    # def get_rush_index(row):
    #     for i, col in enumerate(skill_cols):
    #         if row['rusher_player_id'] == row[col]:
    #             return i
    #     return -1
    
    # pbp_run['rush_index'] = pbp_run.apply(get_rush_index, axis=1)
    # pbp_run = pbp_run[pbp_run['rush_index']>=0]
    # y = pbp_run['rush_index']
    # #%%
    # #skill_long = top_skill_df.melt(id_vars='game_id', value_name='gsis_id', var_name='slot')
    
    # # Merge on receiver stats (must contain stats for each player by game)
    # #skill_long = skill_long.merge(player_stats_wide[rusher_columns], left_on=['gsis_id', 'game_id'],right_index=True, how='left')
    
    # # Pivot back to wide format: one row per game, one column per stat per slot
    # rusher_stats_long = skill_long.pivot_table(index='game_id', columns='slot')
    # rusher_stats_long.columns = ['_'.join(col).strip() for col in rusher_stats_long.columns]
    # rusher_stats_long = rusher_stats_long.reset_index()
    
    # # Merge these expanded receiver stats into pbp_pass
    # pbp_run = pbp_run.merge(rusher_stats_long, on='game_id', how='left')
    # #%%
    # # Include game state columns and per-slot stats
    # slot_stat_cols = [col for col in rusher_stats_long.columns if col[:-4] in rusher_columns]
    # modeling_columns = game_state_cols+slot_stat_cols
    
    # X = pbp_run[game_state_cols + slot_stat_cols].copy()
    
    # categorical_cols = ['roof', 'posteam_type']
    # numerical_cols = list(set(X.columns) - set(categorical_cols))
    
    # # Preprocessing
    # preprocessor = ColumnTransformer([
    #     ('num', StandardScaler(), numerical_cols),
    #     ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    # ])
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    # X_train_processed = preprocessor.fit_transform(X_train)
    # X_test_processed = preprocessor.transform(X_test)
    
    # X_train_processed[np.isnan(X_train_processed)] = 0
    # X_test_processed[np.isnan(X_test_processed)] = 0
    # #%%
    # input_dim = X_train_processed.shape[1]
    # n_classes = len(skill_cols)
    
    # # Convert to numpy arrays
    # X_train_np = X_train_processed.astype(np.float32)
    # X_test_np = X_test_processed.astype(np.float32)
    # y_train_np = y_train.values.astype(np.int64)
    # y_test_np = y_test.values.astype(np.int64)
    # y_train_class = y_train_np
    # y_test_class = y_test_np
    
    # # Create skorch model
    # input_dim = X_train_np.shape[1]
    # output_dim = n_classes
    
    # model_play = NeuralNetClassifier(
    #     module=GameModel,
    #     module__input_dim=input_dim,
    #     module__output_dim=output_dim,
    #     max_epochs=1500,
    #     batch_size=512,
    #     optimizer=torch.optim.AdamW,
    #     criterion=nn.CrossEntropyLoss,
    #     callbacks=[EarlyStopping(patience=20, load_best=True), Checkpoint(load_best=True)],
    #     iterator_train__shuffle=True,
    #     device='cuda' if torch.cuda.is_available() else 'cpu',
    # )
    
    # # Fit model
    # model_play.fit(X_train_np, y_train_class)
    
    # # Predict
    # y_pred_logits = model_play.predict_proba(X_test_np)
    # y_pred_class = (y_pred_logits > 0.5).astype(int)
    # y_pred_classes = np.argmax(y_pred_class, axis=1)
    # model_play.predict(X_test_np)
    # # Save
    # torch.save(model_play.module_.state_dict(), BASE_DIR / 'rusher_model.pt')
    
    # joblib.dump(preprocessor, BASE_DIR / 'rusher_preprocessor.pkl')
    # modeling_columns =['game_state_cols','rusher_columns']
    # joblib.dump(modeling_columns, BASE_DIR / 'rusher_columns.pkl')
#%%Testing Fumbling
play = pbp.copy()
play['is_fumble'] = (play['fumble'] == 1).astype(int)
play['play_outcome'] = np.select(
    [
        (play['pass_attempt'] == 1) & (play['sack'] == 1),
        (play['pass_attempt'] == 1) & (play['complete_pass'] == 1),
        (play['rush_attempt'] == 1)
    ],
    ['sack', 'completion', 'run'],
    default='other'
)
play = play[play['play_outcome'].isin(['sack', 'completion', 'run'])]
fumble_catalog = build_feature_catalog(
    {
        "game_state_cols": game_state_cols,
        "team_stats_columns": team_stats_columns,
        "pregame_context_columns": feature_catalog.get("pregame_context_columns", []),
        "def_cols": def_cols,
        "play_outcome_cols": ["play_outcome"],
    }
)
model_name = "fumble"
if should_train_model(model_name, selected_models):
    y = play['is_fumble'].copy()
    groups = play['game_id'].to_numpy()
    training = train_model_from_config(
        feature_config,
        model_name,
        play,
        y,
        groups,
        fumble_catalog,
        prior_catalog=feature_catalog,
        target_mode="binary",
    )
    y_pred_proba = training['test_probabilities'][:, 1]

    save_model_artifacts(model_name, training, feature_config)
    report_builder.add_binary_classification("fumble", training['y_test_class'], y_pred_proba, loss_history=training['loss_history'])

    cm = confusion_matrix(training['y_test_class'], np.argmax(training['test_probabilities'], axis=1))
    
    # Cleanup
    del play
#%%Defensive Return Yards
pbp['defensive_return_yards'] = pbp['return_yards'].combine_first(pbp['fumble_recovery_1_yards'])
pbp_ret = pbp.copy()[(pbp['interception']==1) | (pbp['fumble_lost']==1)]
pbp_ret = pbp_ret.copy()[pbp_ret['season_type'] == 'REG']  # Only regular season
player_stats['game_date'] = pd.to_datetime(player_stats['game_date'])
team_stats['game_date'] = pd.to_datetime(team_stats['game_date'])
player_stats_sorted = player_stats.sort_values(['player_id', 'game_date']).rename(columns={'player_id': 'receiver_player_id'})
model_name = "defensive_return_yards"
if should_train_model(model_name, selected_models):
    y = pbp_ret['return_yards'].copy()
    groups = pbp_ret['game_id'].to_numpy()
    training = train_model_from_config(
        feature_config,
        model_name,
        pbp_ret,
        y,
        groups,
        feature_catalog,
        target_mode="ordinal",
    )
    bins = training['bins']
    probs = training['test_probabilities']
    avg = (probs*bins).sum(axis=1)
    act = bins[training['y_test_class']]
    plt.plot(avg,act,'k.')
    finish_plot()
    acts = training['y_test_class'].tolist()
    cdfs = []
    for i in range(len(acts)):
        cdfs.append(np.sum(probs[i,0:acts[i]]))
    plt.hist(cdfs, density=True)
    finish_plot()

    save_model_artifacts(model_name, training, feature_config)
    report_builder.add_multiclass(
        "defensive_return_yards",
        acts,
        probs,
        bins=bins,
        cdfs=cdfs,
        loss_history=training['loss_history'],
    )
    del pbp_ret
report_builder.write_html(out_path("torch_model_report.html"))
if tuning_leaderboard_rows:
    leaderboard = update_leaderboard(tuning_leaderboard_rows)
    print(f"Tuning leaderboard written to {leaderboard}")
