"""
Calculate naive baseline metrics for AQI forecasting.

Baselines computed for each forecast horizon (+1h to +72h):
  1. Mean baseline:       always predict the training set mean AQI
  2. Persistence:         predict AQI(t+h) = AQI(t)  ("it stays the same")
  3. Same-hour-yesterday: predict AQI(t+h) = AQI(t - 24 + h)

These give a floor — any useful model MUST beat these.

Run:
    venv\Scripts\python.exe notebooks\calculate_baselines.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HORIZONS = [1, 6, 12, 24, 48, 72]
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def evaluate(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(mask.sum()),
    }


def main():
    csv_path = PROJECT_ROOT / "data" / "raw" / "open_meteo_karachi_2024-08-08_2026-08-08.csv"
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    aqi = df["us_aqi"].values
    n = len(aqi)

    # Chronological split (same as training pipeline)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    train_aqi = aqi[:train_end]
    train_mean = float(np.nanmean(train_aqi))

    print(f"Dataset: {n} rows, train mean AQI = {train_mean:.1f}")
    print(f"Split: train={train_end}, val={val_end - train_end}, test={n - val_end}")
    print()

    # We evaluate baselines on the TEST set (same as model evaluation)
    test_start = val_end

    print("=" * 85)
    print(f"  {'Horizon':>8s}  |  {'Baseline':>22s}  {'RMSE':>8s}  {'MAE':>8s}  {'R2':>8s}")
    print("=" * 85)

    for h in HORIZONS:
        # Ground truth: AQI at time t+h (for each t in the test set)
        # We need indices where both the current and future values exist
        test_indices = list(range(test_start, n - h))
        y_true = np.array([aqi[t + h] for t in test_indices])

        # --- Baseline 1: Mean ---
        y_mean = np.full_like(y_true, train_mean)
        m_mean = evaluate(y_true, y_mean)

        # --- Baseline 2: Persistence (AQI stays the same) ---
        y_persist = np.array([aqi[t] for t in test_indices])
        m_persist = evaluate(y_true, y_persist)

        # --- Baseline 3: Same-hour-yesterday ---
        # For +h horizon: use AQI from 24 hours before the target time
        # target time = t + h, so reference = t + h - 24
        y_yesterday = []
        y_true_yest = []
        for t in test_indices:
            ref = t + h - 24
            if 0 <= ref < n:
                y_yesterday.append(aqi[ref])
                y_true_yest.append(aqi[t + h])
        y_yesterday = np.array(y_yesterday)
        y_true_yest = np.array(y_true_yest)
        m_yesterday = evaluate(y_true_yest, y_yesterday)

        tag = f"+{h}h"
        print(f"  {tag:>8s}  |  {'Mean':>22s}  {m_mean['rmse']:8.2f}  {m_mean['mae']:8.2f}  {m_mean['r2']:8.4f}")
        print(f"  {'':>8s}  |  {'Persistence (no change)':>22s}  {m_persist['rmse']:8.2f}  {m_persist['mae']:8.2f}  {m_persist['r2']:8.4f}")
        print(f"  {'':>8s}  |  {'Same-hour-yesterday':>22s}  {m_yesterday['rmse']:8.2f}  {m_yesterday['mae']:8.2f}  {m_yesterday['r2']:8.4f}")
        print(f"  {'-' * 80}")

    print()
    print("How to read this:")
    print("  - Mean baseline R2 = 0.00 by definition (it IS the mean)")
    print("  - Any model with R2 < 0 is WORSE than predicting the mean")
    print("  - Any model with RMSE > Persistence RMSE is worse than 'no change'")
    print("  - A useful model should beat ALL baselines at its horizon")


if __name__ == "__main__":
    main()
