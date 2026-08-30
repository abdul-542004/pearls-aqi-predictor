"""
Training pipeline -- fetches features from Hopsworks, trains three models
(Random Forest, XGBoost, LSTM) for +1h AQI prediction, evaluates them,
and saves the best one.

For multi-horizon forecasting, use the direct_training pipeline instead,
which trains separate models per horizon for better accuracy.

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

# Columns to exclude from features.
# Raw pollutant columns are now dropped by build_features() itself
# when drop_raw_pollutants=True (the default).
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

    # Scale targets for better LSTM convergence
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_s = (y_train - y_mean) / y_std
    y_val_s = (y_val - y_mean) / y_std

    model = lstm_model.train(X_train_s, y_train_s, X_val_s, y_val_s)

    seq_len = lstm_model.SEQUENCE_LENGTH
    preds_s = lstm_model.predict(model, X_test_s, seq_len)
    # Inverse transform
    preds = preds_s * y_std + y_mean
    y_test_aligned = y_test[seq_len:]

    metrics = eval_mod.evaluate_model(y_test_aligned, preds)
    eval_mod.print_metrics(metrics, "LSTM")
    save_model_local(model, "lstm", metrics, ARTIFACTS_DIR, extra_files={
        "scaler": scaler,
        "target_stats": {"mean": float(y_mean), "std": float(y_std)},
    })
    return metrics


# -- Main --------------------------------------------------------------------


def run():
    """Execute the training pipeline."""
    # 1. Fetch data
    df = fetch_training_data()

    # 2. Re-engineer features if raw pollutant columns are present
    raw_pollutant_cols = {"pm2_5", "pm10", "carbon_monoxide",
                          "nitrogen_dioxide", "sulphur_dioxide", "ozone"}
    has_raw_pollutants = bool(raw_pollutant_cols & set(df.columns))

    if has_raw_pollutants:
        print("\nDetected raw pollutant columns in feature store.")
        print("Re-engineering features with leakage fix...")
        from aqi_predictor.features.build_features import build_features
        df = build_features(df, drop_raw_pollutants=True)
        print(f"  After re-engineering: {len(df)} rows, {len(df.columns)} columns")

    # 3. Prepare +1h target
    print("\nPreparing target (next-hour AQI)...")
    df = prepare_target(df)

    # 4. Split
    print("Splitting data...")
    train_df, val_df, test_df = split_data(df)

    # 5. Feature columns
    feature_cols = sorted(set(df.columns) - DROP_COLS)
    print(f"  Using {len(feature_cols)} features")

    X_train, y_train = extract_xy(train_df, feature_cols)
    X_val, y_val = extract_xy(val_df, feature_cols)
    X_test, y_test = extract_xy(test_df, feature_cols)

    # 6. Train all 3 models
    results = {}
    results["Random Forest"] = _train_rf(X_train, y_train, X_val, y_val, X_test, y_test)
    results["XGBoost"] = _train_xgb(X_train, y_train, X_val, y_val, X_test, y_test)
    results["LSTM"] = _train_lstm(X_train, y_train, X_val, y_val, X_test, y_test)

    # 7. Compare
    comparison_df, best_name = eval_mod.compare_models(results)

    # 8. Save report
    report = {
        "best_model_1h": best_name,
        "direct_1h_metrics": results,
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
    print("\nTip: For multi-horizon forecasting, run the direct_training pipeline:")
    print("  python src/aqi_predictor/pipelines/direct_training.py")


if __name__ == "__main__":
    run()
