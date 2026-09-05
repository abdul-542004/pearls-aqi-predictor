"""
Multi-horizon AQI predictor using direct models.

Instead of recursively feeding +1h predictions back into the model,
this module loads a separate pre-trained model for each forecast
horizon (+1h, +6h, +12h, +24h, +48h, +72h) and predicts directly.

For intermediate hours (e.g., +2h through +5h), predictions are
linearly interpolated between the nearest direct-model outputs.

Falls back to recursive prediction if direct models are not available.

Requires:
  - Trained direct models in artifacts/models/<model_type>_<horizon>h/
  - A feature DataFrame with at least 24 rows of recent history
"""

import collections
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# Direct model horizons (in hours)
DIRECT_HORIZONS = [1, 6, 12, 24, 48, 72]

# Weather columns available from forecast APIs (Open-Meteo)
_WEATHER_FORECAST_COLS = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "pressure_msl", "wind_speed_10m", "wind_direction_10m",
]

# Derived AQI feature columns that we recompute at each step (for recursive fallback)
_AQI_LAG_COLS = {"us_aqi_lag_1h": 1, "us_aqi_lag_3h": 3, "us_aqi_lag_6h": 6,
                 "us_aqi_lag_12h": 12, "us_aqi_lag_24h": 24}
_AQI_ROLLING_COLS = {"aqi_rolling_3h": 3, "aqi_rolling_6h": 6,
                     "aqi_rolling_12h": 12, "aqi_rolling_24h": 24}


def load_model(model_path):
    """Load a scikit-learn or XGBoost model from a .pkl file."""
    return joblib.load(model_path)


def load_direct_models(artifacts_dir, model_type="xgboost"):
    """
    Load pre-trained direct models for each horizon.

    Parameters
    ----------
    artifacts_dir : str | Path
        Root artifacts directory (e.g. artifacts/models/).
    model_type : str
        Model type prefix (e.g. "xgboost", "random_forest").

    Returns
    -------
    dict[int, object]
        Mapping of {horizon_hours: fitted_model}.
    """
    import json
    artifacts_dir = Path(artifacts_dir)
    models = {}

    for h in DIRECT_HORIZONS:
        model_dir = artifacts_dir / f"{model_type}_{h}h"
        model_path = model_dir / "model.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            # Load feature names if saved alongside the model
            feat_path = model_dir / "feature_names.json"
            if feat_path.exists():
                try:
                    with open(feat_path, "r") as f:
                        model._expected_features = json.load(f)
                except Exception:
                    pass
            models[h] = model
            print(f"  Loaded {model_type} +{h}h model from {model_path}")
        else:
            print(f"  Warning: No model found at {model_path}")

    return models


def direct_forecast(models, current_features, feature_cols, max_hours=72,
                    forecast_weather_df=None):
    """
    Generate a multi-step AQI forecast using direct per-horizon models.

    Parameters
    ----------
    models : dict[int, object]
        Mapping of {horizon_hours: fitted_model} from load_direct_models().
    current_features : pd.Series or dict
        The current feature values (latest row from feature-engineered data).
    feature_cols : list[str]
        Base feature column names the models expect, in order.
    max_hours : int
        Maximum forecast horizon (default 72).
    forecast_weather_df : pd.DataFrame, optional
        Future weather data indexed by datetime.  If provided, forecast
        weather features (forecast_temperature_2m_Xh, etc.) are built
        for each horizon from this data.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["hours_ahead", "predicted_aqi"] for each
        hour from 1 to max_hours.
    """
    current_time = pd.to_datetime(current_features.get("time", None))

    direct_preds = {}
    for h, model in sorted(models.items()):
        if h <= max_hours:
            # Build the feature vector including forecast weather for this horizon
            row = dict(current_features) if not isinstance(current_features, dict) else current_features.copy()

            # Add forecast weather features and deltas if available
            if forecast_weather_df is not None and current_time is not None:
                target_time = current_time + pd.Timedelta(hours=h)
                if target_time in forecast_weather_df.index:
                    fw = forecast_weather_df.loc[target_time]
                    for col in _WEATHER_FORECAST_COLS:
                        if col in fw.index:
                            fc_val = fw[col]
                            row[f"forecast_{col}_{h}h"] = fc_val
                            if col in row:
                                row[f"delta_{col}_{h}h"] = fc_val - row[col]

                    # Derived forecast features
                    if "wind_speed_10m" in fw.index and "wind_direction_10m" in fw.index:
                        wd_rad = np.deg2rad(fw["wind_direction_10m"])
                        u_val = fw["wind_speed_10m"] * np.sin(wd_rad)
                        v_val = fw["wind_speed_10m"] * np.cos(wd_rad)
                        row[f"forecast_wind_u_{h}h"] = u_val
                        row[f"forecast_wind_v_{h}h"] = v_val
                        if "wind_u" in row:
                            row[f"delta_wind_u_{h}h"] = u_val - row["wind_u"]
                        if "wind_v" in row:
                            row[f"delta_wind_v_{h}h"] = v_val - row["wind_v"]

                    if "temperature_2m" in fw.index and "relative_humidity_2m" in fw.index:
                        th_val = (
                            fw["temperature_2m"] * fw["relative_humidity_2m"] / 100.0
                        )
                        row[f"forecast_temp_humidity_{h}h"] = th_val
                        if "temp_humidity" in row:
                            row[f"delta_temp_humidity_{h}h"] = th_val - row["temp_humidity"]

            # Get the exact feature columns this specific horizon model expects
            if hasattr(model, "_expected_features") and model._expected_features:
                model_cols = model._expected_features
            else:
                model_feature_cols = [c for c in feature_cols if not c.startswith("forecast_") and not c.startswith("delta_")]
                horizon_extra_cols = [c for c in row.keys() if (c.startswith("forecast_") or c.startswith("delta_")) and c.endswith(f"_{h}h")]
                model_cols = sorted(model_feature_cols + horizon_extra_cols)

            X = np.array([row.get(col, 0.0) for col in model_cols],
                         dtype=np.float64).reshape(1, -1)
            pred = float(model.predict(X)[0])
            direct_preds[h] = max(0.0, min(500.0, pred))  # Clamp to AQI range

    if not direct_preds:
        raise ValueError("No direct models available for forecasting")

    # Interpolate for intermediate hours
    sorted_horizons = sorted(direct_preds.keys())
    predictions = []

    # For hour 0, use current AQI
    current_aqi = float(current_features.get("us_aqi", direct_preds[sorted_horizons[0]]))

    for hour in range(1, max_hours + 1):
        if hour in direct_preds:
            # Direct prediction available
            pred_aqi = direct_preds[hour]
        elif hour < sorted_horizons[0]:
            # Before first direct horizon -- interpolate from current
            h_next = sorted_horizons[0]
            frac = hour / h_next
            pred_aqi = current_aqi + frac * (direct_preds[h_next] - current_aqi)
        elif hour > sorted_horizons[-1]:
            # Beyond last direct horizon -- hold constant
            pred_aqi = direct_preds[sorted_horizons[-1]]
        else:
            # Between two direct horizons -- linear interpolation
            h_prev = max(h for h in sorted_horizons if h <= hour)
            h_next = min(h for h in sorted_horizons if h > hour)
            frac = (hour - h_prev) / (h_next - h_prev)
            pred_aqi = direct_preds[h_prev] + frac * (direct_preds[h_next] - direct_preds[h_prev])

        pred_aqi = max(0.0, min(500.0, pred_aqi))
        predictions.append({"hours_ahead": hour, "predicted_aqi": pred_aqi})

    return pd.DataFrame(predictions)


def direct_forecast_with_time(models, history_df, feature_cols, max_hours=72,
                              forecast_weather_df=None):
    """
    Generate a time-indexed forecast using direct models.

    Like direct_forecast() but adds proper timestamps based on the
    last row in the history DataFrame.

    Parameters
    ----------
    models : dict[int, object]
        Per-horizon models from load_direct_models().
    history_df : pd.DataFrame
        Recent feature-engineered data. The last row is "now".
    feature_cols : list[str]
        Feature column names the models expect.
    max_hours : int
        Maximum forecast horizon.
    forecast_weather_df : pd.DataFrame, optional
        Future weather data indexed by datetime.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["time", "hours_ahead", "predicted_aqi"].
    """
    last_row = history_df.iloc[-1]
    last_time = pd.to_datetime(last_row["time"])

    forecast_df = direct_forecast(
        models, last_row, feature_cols, max_hours=max_hours,
        forecast_weather_df=forecast_weather_df
    )
    forecast_df["time"] = [last_time + pd.Timedelta(hours=h)
                           for h in forecast_df["hours_ahead"]]
    return forecast_df[["time", "hours_ahead", "predicted_aqi"]]


# -- Recursive fallback (kept for backward compatibility) --------------------


def recursive_forecast(
    model, history_df, feature_cols, steps=72,
    forecast_weather_df=None, include_forecast_pollutants=False,
):
    """
    Generate a multi-step AQI forecast using recursive +1h predictions.

    This is the legacy approach kept as a fallback. Prefer direct_forecast()
    for better accuracy at longer horizons.

    Parameters
    ----------
    model : fitted estimator
        A trained +1h model (e.g., XGBRegressor).
    history_df : pd.DataFrame
        Recent feature-engineered data (at least 24 rows).
    feature_cols : list[str]
        Feature column names the model was trained on, in order.
    steps : int
        Number of hours to forecast (default 72 = 3 days).
    forecast_weather_df : pd.DataFrame, optional
        Future weather data indexed by datetime.
    include_forecast_pollutants : bool, optional
        If True, update pollutant columns from forecast data.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["time", "predicted_aqi"].
    """
    aqi_history = collections.deque(
        history_df["us_aqi"].tolist(), maxlen=max(24, steps) + 1
    )

    last_row = history_df.iloc[-1].copy()
    last_time = pd.to_datetime(last_row["time"])

    predictions = []

    for step in range(1, steps + 1):
        next_time = last_time + pd.Timedelta(hours=step)
        row = last_row.copy()
        row["time"] = next_time

        # Update time features
        row["hour_sin"] = np.sin(2 * np.pi * next_time.hour / 24)
        row["hour_cos"] = np.cos(2 * np.pi * next_time.hour / 24)
        row["month_sin"] = np.sin(2 * np.pi * (next_time.month - 1) / 12)
        row["month_cos"] = np.cos(2 * np.pi * (next_time.month - 1) / 12)
        if "dow_sin" in row.index:
            row["dow_sin"] = np.sin(2 * np.pi * next_time.dayofweek / 7)
            row["dow_cos"] = np.cos(2 * np.pi * next_time.dayofweek / 7)
            row["is_weekend"] = int(next_time.dayofweek >= 5)
        if "day_of_week" in row.index:
            row["day_of_week"] = next_time.dayofweek

        # Pull future weather from forecast if available
        if forecast_weather_df is not None and next_time in forecast_weather_df.index:
            forecast_row = forecast_weather_df.loc[next_time]
            for col in _WEATHER_FORECAST_COLS:
                if col in forecast_row.index:
                    row[col] = forecast_row[col]

        # Update AQI lag features
        aqi_list = list(aqi_history)
        for col, lag in _AQI_LAG_COLS.items():
            if col in row.index and lag <= len(aqi_list):
                row[col] = aqi_list[-lag]

        # Update AQI rolling features
        for col, window in _AQI_ROLLING_COLS.items():
            if col in row.index:
                recent = aqi_list[-window:] if len(aqi_list) >= window else aqi_list
                row[col] = np.mean(recent)

        # Update AQI change/trend features
        if "aqi_change_1h" in row.index and len(aqi_list) >= 2:
            row["aqi_change_1h"] = aqi_list[-1] - aqi_list[-2]
        if "aqi_change_3h" in row.index and len(aqi_list) >= 4:
            row["aqi_change_3h"] = aqi_list[-1] - aqi_list[-4]

        # Predict
        X = np.array([row[col] for col in feature_cols],
                     dtype=np.float64).reshape(1, -1)
        predicted_aqi = float(model.predict(X)[0])
        predicted_aqi = max(0.0, min(500.0, predicted_aqi))

        predictions.append({"time": next_time, "predicted_aqi": predicted_aqi})

        # Update buffers
        aqi_history.append(predicted_aqi)
        row["us_aqi"] = predicted_aqi

    return pd.DataFrame(predictions)
