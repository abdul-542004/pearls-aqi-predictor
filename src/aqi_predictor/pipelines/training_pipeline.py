"""
Training pipeline -- fetches features from Hopsworks, trains three models
(Random Forest, XGBoost, LSTM) for +1h AQI prediction, evaluates them,
and saves the best one.

For multi-day forecasting, the best +1h model is used recursively
via models/predict.py (recursive_forecast).

Run:
    python src/aqi_predictor/pipelines/training_pipeline.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

# Allow running as a script from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

from aqi_predictor.features.hopsworks_utils import (
    get_feature_store,
    get_or_create_feature_group,
)
from aqi_predictor.models import evaluate as eval_mod
from aqi_predictor.models import lstm_model, random_forest, xgboost_model
from aqi_predictor.models.registry import save_model_local, upload_to_hopsworks

# -- Constants ---------------------------------------------------------------

TARGET = "us_aqi"
TARGET_NEXT = "us_aqi_next_1h"

# Columns to exclude from features
DROP_COLS = {"time", TARGET, TARGET_NEXT, "is_hazardous"}

# Chronological split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "models"


# -- Helpers -----------------------------------------------------------------


def fetch_training_data():
    """Read the full feature group from Hopsworks into a DataFrame."""
    print("Connecting to Hopsworks...")
    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)
    df = fg.read()
    print(f"  Fetched {len(df)} rows, {len(df.columns)} columns")
    return df


def prepare_target(df):
    """Create the next-hour AQI target and drop the last row (NaN)."""
    df = df.sort_values("time").reset_index(drop=True)
    df[TARGET_NEXT] = df[TARGET].shift(-1)
    df = df.dropna(subset=[TARGET_NEXT]).reset_index(drop=True)
    return df


def split_data(df):
    """Chronological 70 / 15 / 15 split."""
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    print(f"  Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def extract_xy(df, feature_cols):
    """Extract X (features) and y (target) arrays from a DataFrame."""
    X = df[feature_cols].values.astype(np.float64)
    y = df[TARGET_NEXT].values.astype(np.float64)
    return X, y


# -- Per-model training ------------------------------------------------------


def _train_rf(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate Random Forest."""
    print("\n--- Random Forest --------------------------------")
    model = random_forest.train(X_train, y_train, X_val, y_val)
    preds = model.predict(X_test)
    metrics = eval_mod.evaluate_model(y_test, preds)
    eval_mod.print_metrics(metrics, "Random Forest")
    save_model_local(model, "random_forest", metrics, ARTIFACTS_DIR)
    return metrics


def _train_xgb(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate XGBoost."""
    print("\n--- XGBoost --------------------------------------")
    model = xgboost_model.train(X_train, y_train, X_val, y_val)
    preds = model.predict(X_test)
    metrics = eval_mod.evaluate_model(y_test, preds)
    eval_mod.print_metrics(metrics, "XGBoost")
    save_model_local(model, "xgboost", metrics, ARTIFACTS_DIR)
    return metrics


def _train_lstm(X_train, y_train, X_val, y_val, X_test, y_test):
    """Scale, train, and evaluate LSTM."""
    print("\n--- LSTM -----------------------------------------")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = lstm_model.train(X_train_s, y_train, X_val_s, y_val)

    seq_len = lstm_model.SEQUENCE_LENGTH
    preds = lstm_model.predict(model, X_test_s, seq_len)
    y_test_aligned = y_test[seq_len:]

    metrics = eval_mod.evaluate_model(y_test_aligned, preds)
    eval_mod.print_metrics(metrics, "LSTM")
    save_model_local(model, "lstm", metrics, ARTIFACTS_DIR, extra_files={"scaler": scaler})
    return metrics


# -- Recursive evaluation ----------------------------------------------------


def evaluate_recursive(model, test_df, feature_cols, horizons=(1, 6, 12, 24, 48, 72)):
    """
    Evaluate the recursive prediction strategy on the test set.

    Picks sample points across the test set, runs recursive_forecast from
    each point, and measures accuracy at each horizon.

    Future **weather** data from the test set is provided to the recursive
    loop (simulating the Open-Meteo forecast available in production).
    Future **pollutant** data is NOT provided — those are not available
    at inference time.
    """
    from aqi_predictor.models.predict import (
        _WEATHER_FORECAST_COLS,
        recursive_forecast,
    )

    print("\n--- Recursive Forecast Evaluation -----------------")
    print(f"  Horizons: {horizons}")

    max_h = max(horizons)
    # Sample every 72 rows to get independent forecast origins
    origins = range(24, len(test_df) - max_h, 72)
    print(f"  Evaluating from {len(list(origins))} forecast origins...")

    # Weather columns present in the test set
    available_weather = [c for c in _WEATHER_FORECAST_COLS if c in test_df.columns]

    # Collect actual vs predicted per horizon
    results_by_h = {h: {"actual": [], "predicted": []} for h in horizons}

    for origin_idx in origins:
        # Use 24 rows of history before the origin
        history = test_df.iloc[origin_idx - 24 : origin_idx + 1].copy()

        # Build a weather-only forecast DataFrame from the test set's future
        # rows.  This simulates having a real weather forecast API available
        # in production (e.g. Open-Meteo 3-day forecast).
        future = test_df.iloc[origin_idx + 1 : origin_idx + max_h + 1]
        forecast_weather = future.set_index("time")[available_weather]

        # Run recursive forecast with future weather
        forecast_df = recursive_forecast(
            model, history, feature_cols, steps=max_h,
            forecast_weather_df=forecast_weather,
        )

        # Compare at each horizon
        for h in horizons:
            actual_idx = origin_idx + h
            if actual_idx < len(test_df):
                actual_aqi = test_df.iloc[actual_idx]["us_aqi"]
                pred_aqi = forecast_df.iloc[h - 1]["predicted_aqi"]
                results_by_h[h]["actual"].append(actual_aqi)
                results_by_h[h]["predicted"].append(pred_aqi)

    # Compute metrics per horizon
    recursive_metrics = {}
    for h in horizons:
        if results_by_h[h]["actual"]:
            m = eval_mod.evaluate_model(results_by_h[h]["actual"], results_by_h[h]["predicted"])
            recursive_metrics[f"+{h}h"] = m
            print(f"    +{h:2d}h  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R2={m['r2']:.4f}")

    return recursive_metrics


# -- Main --------------------------------------------------------------------


def run():
    """Execute the training pipeline."""
    # 1. Fetch data
    df = fetch_training_data()

    # 2. Prepare +1h target
    print("\nPreparing target (next-hour AQI)...")
    df = prepare_target(df)

    # 3. Split
    print("Splitting data...")
    train_df, val_df, test_df = split_data(df)

    # 4. Feature columns
    feature_cols = sorted(set(df.columns) - DROP_COLS)
    print(f"  Using {len(feature_cols)} features")

    X_train, y_train = extract_xy(train_df, feature_cols)
    X_val, y_val = extract_xy(val_df, feature_cols)
    X_test, y_test = extract_xy(test_df, feature_cols)

    # 5. Train all 3 models
    results = {}
    results["Random Forest"] = _train_rf(X_train, y_train, X_val, y_val, X_test, y_test)
    results["XGBoost"] = _train_xgb(X_train, y_train, X_val, y_val, X_test, y_test)
    results["LSTM"] = _train_lstm(X_train, y_train, X_val, y_val, X_test, y_test)

    # 6. Compare
    comparison_df, best_name = eval_mod.compare_models(results)

    # 7. Evaluate recursive forecasting with the best tree-based model
    # (LSTM uses different interface, so we use the best of RF/XGBoost)
    tree_models = {k: v for k, v in results.items() if k != "LSTM"}
    best_tree = min(tree_models, key=lambda k: tree_models[k]["rmse"])
    best_tree_path = ARTIFACTS_DIR / best_tree.lower().replace(" ", "_") / "model.pkl"

    import joblib
    best_model = joblib.load(best_tree_path)
    recursive_metrics = evaluate_recursive(best_model, test_df, feature_cols)

    # 8. Save report
    report = {
        "best_model_1h": best_name,
        "direct_1h_metrics": results,
        "recursive_metrics": recursive_metrics,
        "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "features": feature_cols,
    }
    report_path = ARTIFACTS_DIR / "training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Training report saved -> {report_path}")

    # 9. Upload best model to Hopsworks
    best_dir = ARTIFACTS_DIR / best_name.lower().replace(" ", "_")
    print(f"\nUploading best model ({best_name}) to Hopsworks...")
    try:
        upload_to_hopsworks(best_dir, "aqi_forecaster", results[best_name])
    except Exception as e:
        print(f"  Warning: Hopsworks upload failed: {e}")
        print("  Models are still saved locally in artifacts/models/")

    print("\nTraining pipeline complete.")


if __name__ == "__main__":
    run()
