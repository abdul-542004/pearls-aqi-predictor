"""
Training pipeline -- fetches features from Hopsworks, trains direct multi-horizon
models (Random Forest, XGBoost, LSTM) for +1h, +6h, +12h, +24h, +48h, and +72h AQI
forecasting, evaluates them, and saves the best model per horizon.

Each horizon gets its own XGBoost, Random Forest, and LSTM model. The pipeline
incorporates future weather forecasts/deltas and prunes short-term decaying lag
features for long horizons to avoid error accumulation.

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

# Horizons to train direct models for (in hours)
HORIZONS = [1, 6, 12, 24, 48, 72]

# Columns to always exclude from features
# Note: raw pollutant columns are already dropped by build_features()
# when drop_raw_pollutants=True (the default).
DROP_COLS = {"time", TARGET, "is_hazardous"}

# Weather columns available from Open-Meteo forecast API.
# These are legitimately available for future timestamps in production
# because Open-Meteo provides 3-day (72h) weather forecasts.
WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "pressure_msl", "wind_speed_10m", "wind_direction_10m",
]

# Chronological split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "models"

# -- XGBoost hyperparameters (tuned) ----------------------------------------

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

# -- Random Forest hyperparameters (tuned) -----------------------------------

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_leaf": 5,
    "min_samples_split": 10,
    "max_features": 0.7,
    "n_jobs": -1,
    "random_state": 42,
}


# -- Helpers -----------------------------------------------------------------


def fetch_training_data():
    """Read the full feature group from Hopsworks into a DataFrame."""
    print("Connecting to Hopsworks...")
    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)
    print(f"Reading from feature group '{fg.name}' v{fg.version}...")
    df = fg.read()
    print(f"  Fetched {len(df)} rows, {len(df.columns)} columns")
    return df


def prepare_targets(df, horizons=HORIZONS):
    """
    Create direct targets for each horizon using shift(-h).

    Returns the DataFrame with new target columns and the list of
    target column names.
    """
    df = df.sort_values("time").reset_index(drop=True)

    target_cols = {}
    for h in horizons:
        col = f"us_aqi_next_{h}h"
        df[col] = df[TARGET].shift(-h)
        target_cols[h] = col

    # Drop rows where ANY target is NaN (tail rows)
    all_target_cols = list(target_cols.values())
    df = df.dropna(subset=all_target_cols).reset_index(drop=True)

    return df, target_cols


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


def extract_xy(df, feature_cols, target_col):
    """Extract X (features) and y (target) arrays from a DataFrame."""
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)
    return X, y


def add_forecast_weather_features(df, horizon):
    """
    Add future weather features and weather deltas for a specific forecast horizon.

    For horizon h, this shifts weather columns by -h so the model
    sees "what will the weather be at time t+h?" as features, as well as
    the physical delta (forecast_weather - current_weather).

    In production, these come from the Open-Meteo 3-day forecast API.
    """
    forecast_cols = []
    out = df.copy()

    for col in WEATHER_COLS:
        if col in out.columns:
            fc_col = f"forecast_{col}_{horizon}h"
            out[fc_col] = out[col].shift(-horizon)
            forecast_cols.append(fc_col)

            # Weather delta (meteorological anomaly / trend)
            delta_col = f"delta_{col}_{horizon}h"
            out[delta_col] = out[fc_col] - out[col]
            forecast_cols.append(delta_col)

    # Derived forecast features (wind vectors and temperature-humidity index)
    if "wind_speed_10m" in out.columns and "wind_direction_10m" in out.columns:
        future_ws = out["wind_speed_10m"].shift(-horizon)
        future_wd = np.deg2rad(out["wind_direction_10m"].shift(-horizon))
        u_col = f"forecast_wind_u_{horizon}h"
        v_col = f"forecast_wind_v_{horizon}h"
        out[u_col] = future_ws * np.sin(future_wd)
        out[v_col] = future_ws * np.cos(future_wd)
        forecast_cols.extend([u_col, v_col])

        if "wind_u" in out.columns and "wind_v" in out.columns:
            out[f"delta_wind_u_{horizon}h"] = out[u_col] - out["wind_u"]
            out[f"delta_wind_v_{horizon}h"] = out[v_col] - out["wind_v"]
            forecast_cols.extend([f"delta_wind_u_{horizon}h", f"delta_wind_v_{horizon}h"])

    if "temperature_2m" in out.columns and "relative_humidity_2m" in out.columns:
        th_col = f"forecast_temp_humidity_{horizon}h"
        out[th_col] = (
            out["temperature_2m"].shift(-horizon)
            * out["relative_humidity_2m"].shift(-horizon) / 100.0
        )
        forecast_cols.append(th_col)
        if "temp_humidity" in out.columns:
            out[f"delta_temp_humidity_{horizon}h"] = out[th_col] - out["temp_humidity"]
            forecast_cols.append(f"delta_temp_humidity_{horizon}h")

    # Drop rows where forecast features are NaN (tail rows)
    out = out.dropna(subset=forecast_cols).reset_index(drop=True)

    return out, forecast_cols


def get_horizon_feature_cols(all_cols, horizon):
    """
    Select horizon-specific features to prevent short-term lag overfitting.

    For longer horizons (24h, 48h, 72h), prune high-frequency decaying 1h/3h lags
    and anchor the model on 24h diurnal lags and forecast weather anomalies.
    """
    short_term_noise = {
        "us_aqi_lag_1h", "us_aqi_lag_3h", "aqi_change_1h", "aqi_change_3h",
        "pm2_5_lag_1h", "pm2_5_lag_3h", "pm10_lag_1h", "pm10_lag_3h",
        "ozone_lag_1h", "ozone_lag_3h", "carbon_monoxide_lag_1h",
        "nitrogen_dioxide_lag_1h", "sulphur_dioxide_lag_1h",
        "aqi_rolling_3h", "aqi_std_6h", "aqi_trend_6h"
    }

    if horizon <= 6:
        return [c for c in all_cols if c not in DROP_COLS and not c.startswith("us_aqi_next_")]
    elif horizon == 12:
        drop = {
            "carbon_monoxide_lag_1h", "nitrogen_dioxide_lag_1h",
            "sulphur_dioxide_lag_1h", "ozone_lag_1h", "aqi_change_1h"
        }
        return [c for c in all_cols if c not in drop and c not in DROP_COLS and not c.startswith("us_aqi_next_")]
    else:  # 24h, 48h, 72h
        return [c for c in all_cols if c not in short_term_noise and c not in DROP_COLS and not c.startswith("us_aqi_next_")]


# -- Per-horizon training ----------------------------------------------------


def train_horizon(horizon, target_col, feature_cols,
                  train_df, val_df, test_df, train_lstm=True):
    """
    Train all model types for a single forecast horizon.

    Returns a dict of {model_name: metrics} and the name of the best model.
    """
    tag = f"+{horizon}h"
    print(f"\n{'='*60}")
    print(f"  Training models for {tag} forecast ({len(feature_cols)} features)")
    print(f"{'='*60}")

    X_train, y_train = extract_xy(train_df, feature_cols, target_col)
    X_val, y_val = extract_xy(val_df, feature_cols, target_col)
    X_test, y_test = extract_xy(test_df, feature_cols, target_col)

    results = {}

    # --- XGBoost ---
    print(f"\n--- XGBoost {tag} ---")
    xgb_model = xgboost_model.train(X_train, y_train, X_val, y_val, **XGBOOST_PARAMS)
    xgb_preds = xgb_model.predict(X_test)
    xgb_metrics = eval_mod.evaluate_model(y_test, xgb_preds)
    eval_mod.print_metrics(xgb_metrics, f"XGBoost {tag}")
    save_model_local(
        xgb_model, f"xgboost_{horizon}h", xgb_metrics, ARTIFACTS_DIR,
        extra_files={"feature_names": feature_cols},
    )
    results["XGBoost"] = xgb_metrics

    # --- Random Forest ---
    print(f"\n--- Random Forest {tag} ---")
    rf_model = random_forest.train(X_train, y_train, X_val, y_val, **RF_PARAMS)
    rf_preds = rf_model.predict(X_test)
    rf_metrics = eval_mod.evaluate_model(y_test, rf_preds)
    eval_mod.print_metrics(rf_metrics, f"Random Forest {tag}")
    save_model_local(
        rf_model, f"random_forest_{horizon}h", rf_metrics, ARTIFACTS_DIR,
        extra_files={"feature_names": feature_cols},
    )
    results["Random Forest"] = rf_metrics

    # --- LSTM ---
    if train_lstm:
        print(f"\n--- LSTM {tag} ---")
        scaler_X = StandardScaler()
        X_train_s = scaler_X.fit_transform(X_train)
        X_val_s = scaler_X.transform(X_val)
        X_test_s = scaler_X.transform(X_test)

        # Scale targets for LSTM (helps convergence)
        y_mean, y_std = y_train.mean(), y_train.std()
        y_train_s = (y_train - y_mean) / y_std
        y_val_s = (y_val - y_mean) / y_std

        lstm = lstm_model.train(
            X_train_s, y_train_s, X_val_s, y_val_s,
            hidden_size=128, num_layers=2, dropout=0.3,
            lr=5e-4, patience=15,
        )

        seq_len = lstm_model.SEQUENCE_LENGTH
        preds_s = lstm_model.predict(lstm, X_test_s, seq_len)
        # Inverse-transform predictions
        preds = preds_s * y_std + y_mean
        y_test_aligned = y_test[seq_len:]

        lstm_metrics = eval_mod.evaluate_model(y_test_aligned, preds)
        eval_mod.print_metrics(lstm_metrics, f"LSTM {tag}")
        save_model_local(
            lstm, f"lstm_{horizon}h", lstm_metrics, ARTIFACTS_DIR,
            extra_files={
                "scaler": scaler_X,
                "target_stats": {"mean": float(y_mean), "std": float(y_std)},
                "feature_names": feature_cols,
            },
        )
        results["LSTM"] = lstm_metrics

    # Find best model for this horizon
    best_name = min(results, key=lambda k: results[k]["rmse"])
    print(f"\n  Best model for {tag}: {best_name} (RMSE={results[best_name]['rmse']:.4f})")

    return results, best_name


# -- Main --------------------------------------------------------------------


def run():
    """Execute the direct multi-horizon training pipeline."""
    # 1. Fetch data
    df = fetch_training_data()

    # 2. Defensive check: re-engineer features if raw pollutant columns are detected (legacy v1 compatibility)
    raw_pollutant_cols = {"pm2_5", "pm10", "carbon_monoxide",
                          "nitrogen_dioxide", "sulphur_dioxide", "ozone"}
    has_raw_pollutants = bool(raw_pollutant_cols & set(df.columns))

    if has_raw_pollutants:
        print("\nDetected raw pollutant columns (legacy v1 feature store).")
        print("Re-engineering features with leakage fix...")
        from aqi_predictor.features.build_features import build_features
        df = build_features(df, drop_raw_pollutants=True)
        print(f"  After re-engineering: {len(df)} rows, {len(df.columns)} columns")
    else:
        print("\nUsing clean Version 2 feature store (leakage-free).")

    # 3. Prepare direct targets
    print("\nPreparing direct targets...")
    df, target_cols = prepare_targets(df, HORIZONS)
    print(f"  Created targets: {list(target_cols.values())}")
    print(f"  Remaining rows: {len(df)}")

    # 4. Split (on the base DataFrame before adding forecast features)
    print("\nSplitting data...")
    train_df, val_df, test_df = split_data(df)

    # 5. Base feature columns (exclude targets + metadata)
    all_target_cols = set(target_cols.values())
    base_feature_cols = sorted(set(df.columns) - DROP_COLS - all_target_cols)

    # 6. Train all horizons
    all_results = {}
    best_models = {}
    horizon_features = {}

    for h in HORIZONS:
        target_col = target_cols[h]

        # Add forecast weather features for this horizon.
        # This gives the model "what will the weather be at t+h?" —
        # available in production from Open-Meteo's 3-day forecast API.
        h_train, forecast_cols = add_forecast_weather_features(train_df, h)
        h_val, _ = add_forecast_weather_features(val_df, h)
        h_test, _ = add_forecast_weather_features(test_df, h)

        # Apply horizon-specific feature selection (pruning stale lags for long horizons)
        all_cols = sorted(base_feature_cols + forecast_cols)
        feature_cols = get_horizon_feature_cols(all_cols, h)
        horizon_features[f"+{h}h"] = feature_cols

        if h == HORIZONS[0]:  # Print features once for the first horizon
            print(f"\n  Base features: {len(base_feature_cols)}")
            print(f"  Forecast + Delta features per horizon: {len(forecast_cols)}")
            print(f"  +1h features: {len(feature_cols)}")
            print(f"  Forecast columns: {forecast_cols}")

        results, best_name = train_horizon(
            h, target_col, feature_cols,
            h_train, h_val, h_test,
            train_lstm=True,  # Train LSTM across all horizons
        )
        all_results[f"+{h}h"] = results
        best_models[f"+{h}h"] = best_name

    # 7. Summary
    print("\n" + "=" * 70)
    print("       DIRECT MULTI-HORIZON TRAINING SUMMARY")
    print("=" * 70)
    print(f"  {'Horizon':<10} {'Best Model':<18} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
    print("  " + "-" * 54)
    for h in HORIZONS:
        tag = f"+{h}h"
        best = best_models[tag]
        m = all_results[tag][best]
        print(f"  {tag:<10} {best:<18} {m['rmse']:8.4f} {m['mae']:8.4f} {m['r2']:8.4f}")
    print("=" * 70)

    # 8. Save report
    report = {
        "strategy": "direct_multi_horizon_with_forecast_weather_deltas_and_horizon_pruning",
        "horizons": HORIZONS,
        "results_per_horizon": all_results,
        "best_model_per_horizon": best_models,
        "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "base_features": base_feature_cols,
        "forecast_weather_cols": WEATHER_COLS,
        "horizon_features": horizon_features,
    }
    report_path = ARTIFACTS_DIR / "training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Training report saved -> {report_path}")

    # 9. Upload best model for +1h to Hopsworks (primary model)
    best_1h = best_models["+1h"].lower().replace(" ", "_")
    best_1h_dir = ARTIFACTS_DIR / f"{best_1h}_1h"
    print(f"\nUploading best +1h model ({best_models['+1h']}) to Hopsworks...")
    try:
        upload_to_hopsworks(best_1h_dir, "aqi_forecaster", all_results["+1h"][best_models["+1h"]])
    except Exception as e:
        print(f"  Warning: Hopsworks upload failed: {e}")
        print("  Models are still saved locally in artifacts/models/")

    print("\nDirect multi-horizon training pipeline complete.")


if __name__ == "__main__":
    run()
