"""
Recursive multi-step predictor for AQI forecasting.

Uses a trained +1h model to predict up to 72 hours ahead by iteratively:
  1. Predicting the next hour's AQI
  2. Updating lag/rolling/derived features with the prediction
  3. Advancing time features by 1 hour
  4. Repeating

Requires:
  - A trained +1h model (e.g., XGBoost from artifacts/models/xgboost_1h/)
  - A feature DataFrame with at least 24 rows of recent history (for rolling/lag)
  - Optionally, future weather/pollutant forecast rows from Open-Meteo
"""

import collections

import joblib
import numpy as np
import pandas as pd


# Derived AQI feature columns that we recompute at each step
_AQI_LAG_COLS = {"us_aqi_lag_1h": 1, "us_aqi_lag_3h": 3, "us_aqi_lag_24h": 24}
_AQI_ROLLING_COLS = {"aqi_rolling_3h": 3, "aqi_rolling_6h": 6, "aqi_rolling_24h": 24}
_POLLUTANT_ROLLING = {
    "pm2_5": "pm2_5_rolling_6h",
    "pm10": "pm10_rolling_6h",
    "carbon_monoxide": "co_rolling_6h",
}


def load_model(model_path):
    """Load a scikit-learn or XGBoost model from a .pkl file."""
    return joblib.load(model_path)


def recursive_forecast(model, history_df, feature_cols, steps=72, forecast_weather_df=None):
    """
    Generate a multi-step AQI forecast using recursive +1h predictions.

    Parameters
    ----------
    model : fitted estimator
        A trained +1h model (e.g., XGBRegressor).
    history_df : pd.DataFrame
        Recent feature-engineered data (at least 24 rows) with all feature
        columns present. The last row is the "current" state.
    feature_cols : list[str]
        The feature column names the model was trained on, in order.
    steps : int
        Number of hours to forecast (default 72 = 3 days).
    forecast_weather_df : pd.DataFrame, optional
        Future weather/pollutant data from Open-Meteo forecast API.
        If provided, columns like temperature_2m, pm2_5, etc. are
        pulled from here instead of being held constant.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``["time", "predicted_aqi"]`` for each
        forecast step.
    """
    # Keep a rolling buffer of recent AQI values for lag/rolling computation
    aqi_history = collections.deque(
        history_df["us_aqi"].tolist(), maxlen=max(24, steps) + 1
    )

    # Keep rolling buffers for pollutants too
    pollutant_histories = {}
    for raw_col in _POLLUTANT_ROLLING:
        if raw_col in history_df.columns:
            pollutant_histories[raw_col] = collections.deque(
                history_df[raw_col].tolist(), maxlen=7
            )

    # Start from the last known row
    last_row = history_df.iloc[-1].copy()
    last_time = pd.to_datetime(last_row["time"])

    predictions = []

    for step in range(1, steps + 1):
        # Advance time by 1 hour
        next_time = last_time + pd.Timedelta(hours=step)
        row = last_row.copy()
        row["time"] = next_time

        # -- Update time features --
        row["hour_sin"] = np.sin(2 * np.pi * next_time.hour / 24)
        row["hour_cos"] = np.cos(2 * np.pi * next_time.hour / 24)
        row["month_sin"] = np.sin(2 * np.pi * (next_time.month - 1) / 12)
        row["month_cos"] = np.cos(2 * np.pi * (next_time.month - 1) / 12)
        row["day_of_week"] = next_time.dayofweek

        # -- Pull future weather/pollutants from forecast if available --
        if forecast_weather_df is not None and next_time in forecast_weather_df.index:
            forecast_row = forecast_weather_df.loc[next_time]
            weather_cols = [
                "temperature_2m", "relative_humidity_2m", "precipitation",
                "pressure_msl", "wind_speed_10m", "wind_direction_10m",
                "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                "sulphur_dioxide", "ozone",
            ]
            for col in weather_cols:
                if col in forecast_row.index:
                    row[col] = forecast_row[col]

        # -- Update AQI lag features --
        aqi_list = list(aqi_history)
        for col, lag in _AQI_LAG_COLS.items():
            if col in row.index and lag <= len(aqi_list):
                row[col] = aqi_list[-lag]

        # -- Update AQI rolling features --
        for col, window in _AQI_ROLLING_COLS.items():
            if col in row.index:
                recent = list(aqi_history)[-window:] if len(aqi_history) >= window else list(aqi_history)
                row[col] = np.mean(recent)

        # -- Update pm2_5 lag --
        if "pm2_5_lag_1h" in row.index and "pm2_5" in pollutant_histories:
            pm_hist = list(pollutant_histories["pm2_5"])
            if pm_hist:
                row["pm2_5_lag_1h"] = pm_hist[-1]

        # -- Update pollutant rolling features --
        for raw_col, feat_col in _POLLUTANT_ROLLING.items():
            if feat_col in row.index and raw_col in pollutant_histories:
                recent = list(pollutant_histories[raw_col])[-6:]
                row[feat_col] = np.mean(recent) if recent else row[feat_col]

        # -- Update AQI change --
        if "aqi_change_1h" in row.index and len(aqi_history) >= 2:
            aqi_list = list(aqi_history)
            row["aqi_change_1h"] = aqi_list[-1] - aqi_list[-2]

        # -- Predict --
        X = np.array([row[col] for col in feature_cols], dtype=np.float64).reshape(1, -1)
        predicted_aqi = float(model.predict(X)[0])

        # Clamp to reasonable range (AQI 0-500)
        predicted_aqi = max(0.0, min(500.0, predicted_aqi))

        predictions.append({"time": next_time, "predicted_aqi": predicted_aqi})

        # -- Update buffers for next iteration --
        aqi_history.append(predicted_aqi)
        row["us_aqi"] = predicted_aqi

        # Update pollutant histories with current row values
        for raw_col in _POLLUTANT_ROLLING:
            if raw_col in pollutant_histories and raw_col in row.index:
                pollutant_histories[raw_col].append(row[raw_col])

    return pd.DataFrame(predictions)
