"""Compare point_spread_model.py predictions against the 2025 sim backtest,
and check whether averaging the two improves on either alone.

Reads:
  - point_spread_predictions_2025.csv (point_spread_model.py output)
  - backtest_2025_results.csv (torch_game_simulations_2025_backtest.py output)
Both use the home_score - away_score sign convention, so they merge directly.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_DIR = Path(__file__).resolve().parent


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"[{label}] MAE={mae:.3f}  RMSE={rmse:.3f}  n={len(y_true)}")
    return {"mae": float(mae), "rmse": float(rmse), "n": int(len(y_true))}


def main():
    point_spread = pd.read_csv(MODEL_DIR / "point_spread_predictions_2025.csv")
    backtest = pd.read_csv(MODEL_DIR / "backtest_2025_results.csv")

    merged = point_spread.merge(
        backtest[["game_id", "actual_spread", "sim_spread", "spread_line"]],
        on="game_id",
        how="inner",
        suffixes=("", "_backtest"),
    )
    merged = merged[merged["actual_result"].notna() & merged["sim_spread"].notna()].copy()
    merged["ensemble_spread"] = (merged["predicted_result"] + merged["sim_spread"]) / 2.0

    y_true = merged["actual_result"]
    print(f"Games with both model and sim predictions: {len(merged)}")
    evaluate(y_true, merged["predicted_result"], "point_spread_model")
    evaluate(y_true, merged["sim_spread"], "2025 sim backtest")
    evaluate(y_true, merged["ensemble_spread"], "average(model, sim)")
    evaluate(y_true, merged["spread_line"], "market spread_line")

    merged.to_csv(MODEL_DIR / "point_spread_ensemble_comparison.csv", index=False)
    print(f"Saved comparison to {MODEL_DIR / 'point_spread_ensemble_comparison.csv'}")


if __name__ == "__main__":
    main()
