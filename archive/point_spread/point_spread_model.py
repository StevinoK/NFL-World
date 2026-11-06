"""Standalone point-spread (home_score - away_score) model.

Trains on 2017-2024 REG season games and evaluates on 2025, using:
  - team offense tendency/quality columns (team_stats_wide.csv) for home & away
  - opponent defense columns (team_stats_wide.csv) faced by each offense
  - starting QB passer columns (player_stats_wide.csv) for home & away starters
  - rest days (capped at 21, from the schedule's home_rest/away_rest) and a
    neutral-site indicator for home-field advantage, sourced the same way
    torch_game_simulations_2025_backtest.py derives them

Reuses the precomputed `team_stats_wide.csv` / `player_stats_wide.csv` outputs
written by Torch Modeling.py, so it does not need to recompute the Bayesian
feature sources itself.

Usage:
  python point_spread_model.py                        # train with saved/default hyperparameters
  python point_spread_model.py --mode tune --trials 50 # Optuna search over season-grouped CV, then save best params
"""
import argparse
import json
from pathlib import Path

import joblib
import nfl_data_py as nfl
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from project_paths import DEFAULT_ARTIFACT_DIR

BASE_DIR = DEFAULT_ARTIFACT_DIR
MODEL_DIR = Path(__file__).resolve().parent
BEST_PARAMS_PATH = MODEL_DIR / "point_spread_best_hyperparameters.json"
TRAIN_SEASONS = list(range(2017, 2025))
TEST_SEASON = 2025
ALL_SEASONS = list(range(TRAIN_SEASONS[0], TEST_SEASON + 1))

TEAM_STATS_COLUMNS = [
    'losing_rate', 'tied_rate', 'team_4th_rate', 'winning_rate', 'rushing_epa_rate',
    'epa_rate', 'passing_massey_rating', 'passing_massey_offense', 'rushing_massey_rating',
    'rushing_massey_offense', 'opp_def_pass_epa_rate', 'opp_def_rush_epa_rate',
    'opp_pass_epa_rate', 'opp_rushing_epa_rate',
]
DEF_COLS = [
    'deep_cpoe_allowed', 'int_rate', 'medium_cpoe_allowed', 'short_cpoe_allowed',
    'yac_oe_allowed', 'sack_rate', 'rush_yards_per_carry', 'def_rush_epa_rate',
    'def_pass_epa_rate', 'def_winning_rate', 'def_losing_rate', 'def_tied_rate',
    'passing_massey_defense', 'rushing_massey_defense',
]
PASSER_COLUMNS = [
    'cpoe_bayes', 'turnover_bayes', 'sack_bayes', 'cpoe_short', 'cpoe_medium',
    'cpoe_deep', 'deep_pass_rate', 'medium_pass_rate', 'short_pass_rate',
]
# Same cap used for days_rest in torch_game_simulations_2025_backtest.py.
REST_DAYS_CAP = 21.0
DEFAULT_REST_DAYS = 7.0


def cap_rest_days(value):
    return float(np.clip(value, 0.0, REST_DAYS_CAP))


def load_games():
    schedule = nfl.import_schedules(ALL_SEASONS)
    games = schedule[schedule['game_type'] == 'REG'].copy()
    games = games[games['home_score'].notna() & games['away_score'].notna()].copy()

    # Drop meaningless week-18 blowouts/rest games, mirroring Torch Modeling.py.
    weird_mask = (
        (games['week'] == 18)
        & ((games['spread_line'].abs() >= 10.0) | (games['result'].abs() >= 17.0))
    )
    games = games[~weird_mask].copy()
    games = games.reset_index(drop=True)
    return games


def _reindexed(source_frame, key_series, game_id_series, columns, prefix):
    idx = pd.MultiIndex.from_arrays([key_series.astype(str).to_numpy(), game_id_series.astype(str).to_numpy()])
    sub = source_frame.reindex(index=idx, columns=columns)
    sub.index = key_series.index
    return sub.add_prefix(prefix)


def build_context_features(games):
    home_rest = pd.to_numeric(games['home_rest'], errors='coerce').fillna(DEFAULT_REST_DAYS).map(cap_rest_days)
    away_rest = pd.to_numeric(games['away_rest'], errors='coerce').fillna(DEFAULT_REST_DAYS).map(cap_rest_days)
    # International/Super Bowl games have no true home-field advantage.
    neutral_site = (games['location'].astype(str).str.upper() == 'NEUTRAL').astype(int)

    return pd.DataFrame(
        {
            'home_rest_days': home_rest,
            'away_rest_days': away_rest,
            'rest_days_diff': home_rest - away_rest,
            'neutral_site': neutral_site,
        },
        index=games.index,
    )


def build_features(games, team_stats_wide, player_stats_wide):
    home_off = _reindexed(team_stats_wide, games['home_team'], games['game_id'], TEAM_STATS_COLUMNS, 'home_')
    away_off = _reindexed(team_stats_wide, games['away_team'], games['game_id'], TEAM_STATS_COLUMNS, 'away_')
    # The defense each offense faces belongs to the *other* team.
    home_def_faced = _reindexed(team_stats_wide, games['away_team'], games['game_id'], DEF_COLS, 'home_opp_def_')
    away_def_faced = _reindexed(team_stats_wide, games['home_team'], games['game_id'], DEF_COLS, 'away_opp_def_')
    home_qb = _reindexed(player_stats_wide, games['home_qb_id'], games['game_id'], PASSER_COLUMNS, 'home_qb_')
    away_qb = _reindexed(player_stats_wide, games['away_qb_id'], games['game_id'], PASSER_COLUMNS, 'away_qb_')
    context = build_context_features(games)

    features = pd.concat([home_off, away_off, home_def_faced, away_def_faced, home_qb, away_qb, context], axis=1)
    return features


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"[{label}] MAE={mae:.3f}  RMSE={rmse:.3f}  n={len(y_true)}")
    return {"mae": float(mae), "rmse": float(rmse), "n": int(len(y_true))}


def suggest_hyperparameters(trial):
    return {
        "max_iter": trial.suggest_int("max_iter", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 64),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
    }


def cross_val_mae(params, X_train, y_train, season_groups, n_splits):
    # Group by season so a fold never trains and validates on the same season.
    splitter = GroupKFold(n_splits=n_splits)
    fold_scores = []
    for fold_train_idx, fold_val_idx in splitter.split(X_train, y_train, groups=season_groups):
        model = HistGradientBoostingRegressor(random_state=42, **params)
        model.fit(X_train.iloc[fold_train_idx], y_train.iloc[fold_train_idx])
        preds = model.predict(X_train.iloc[fold_val_idx])
        fold_scores.append(mean_absolute_error(y_train.iloc[fold_val_idx], preds))
    return float(np.mean(fold_scores))


def tune_hyperparameters(X_train, y_train, season_groups, n_trials, n_splits, seed):
    n_splits = min(n_splits, len(np.unique(season_groups)))

    def objective(trial):
        return cross_val_mae(suggest_hyperparameters(trial), X_train, y_train, season_groups, n_splits)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    print(f"[tune] best CV MAE={study.best_value:.3f} params={study.best_params}")
    BEST_PARAMS_PATH.write_text(
        json.dumps({"params": study.best_params, "cv_mae": study.best_value, "n_splits": n_splits}, indent=2),
        encoding="utf-8",
    )
    return study.best_params


def load_best_hyperparameters():
    if not BEST_PARAMS_PATH.exists():
        return {}
    payload = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    return payload.get("params", {})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "tune"], default="train")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials when --mode tune (default: 50).")
    parser.add_argument("--cv-folds", type=int, default=5, help="Season-grouped CV folds when tuning (default: 5).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    games = load_games()

    team_stats_wide = pd.read_csv(BASE_DIR / "team_stats_wide.csv").set_index(['team', 'game_id'])
    player_stats_wide = pd.read_csv(BASE_DIR / "player_stats_wide.csv").set_index(['player_id', 'game_id'])

    features = build_features(games, team_stats_wide, player_stats_wide)
    target = games['result'].astype(float)

    train_mask = games['season'].isin(TRAIN_SEASONS).to_numpy()
    test_mask = (games['season'] == TEST_SEASON).to_numpy()

    X_train, y_train = features.loc[train_mask], target.loc[train_mask]
    X_test, y_test = features.loc[test_mask], target.loc[test_mask]
    print(f"Train games: {len(X_train)} ({TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]}); Test games: {len(X_test)} ({TEST_SEASON})")

    if args.mode == "tune":
        season_groups = games.loc[train_mask, 'season'].to_numpy()
        params = tune_hyperparameters(X_train, y_train, season_groups, args.trials, args.cv_folds, args.seed)
    else:
        params = load_best_hyperparameters()
        print(f"[train] using hyperparameters: {params or 'defaults'}")

    model = HistGradientBoostingRegressor(random_state=42, **params)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {"model": evaluate(y_test, predictions, "model")}

    market_spread = games.loc[test_mask, 'spread_line']
    market_valid = market_spread.notna()
    if market_valid.any():
        metrics["market"] = evaluate(y_test[market_valid], market_spread[market_valid], "market spread_line")

    joblib.dump(model, MODEL_DIR / "point_spread_model.pkl")
    joblib.dump(list(features.columns), MODEL_DIR / "point_spread_model_columns.pkl")

    output = games.loc[test_mask, ['game_id', 'season', 'week', 'home_team', 'away_team', 'spread_line']].copy()
    output['actual_result'] = y_test.to_numpy()
    output['predicted_result'] = predictions
    output.to_csv(MODEL_DIR / "point_spread_predictions_2025.csv", index=False)

    (MODEL_DIR / "point_spread_model_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved predictions to {MODEL_DIR / 'point_spread_predictions_2025.csv'}")
    print(f"Saved summary to {MODEL_DIR / 'point_spread_model_summary.json'}")


if __name__ == "__main__":
    main()
