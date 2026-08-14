"""
Feature engineering for AQI prediction.

Takes raw Open-Meteo data and produces a feature-enriched DataFrame
ready for insertion into the Hopsworks feature store.
"""

import numpy as np
import pandas as pd


# Columns to drop — constants with no predictive value
_DROP_COLUMNS = ["latitude", "longitude", "location_name"]

# Pollutants to compute rolling averages for
_ROLLING_POLLUTANTS = {
    "pm2_5": "pm2_5_rolling_6h",
    "pm10": "pm10_rolling_6h",
    "carbon_monoxide": "co_rolling_6h",
}


def build_features(df, hazardous_threshold=300):
    """
    Engineer features from raw Open-Meteo weather + air-quality data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with columns from ``sources.py`` (weather + AQ merged).
    hazardous_threshold : int
        AQI value at or above which the ``is_hazardous`` flag is set.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame with NaN warm-up rows dropped.
    """
    out = df.copy()

    # Ensure time is datetime and sorted
    out["time"] = pd.to_datetime(out["time"])
    out = out.sort_values("time").reset_index(drop=True)

    # --- Drop constant columns ---
    out = out.drop(columns=[c for c in _DROP_COLUMNS if c in out.columns])

    # --- Time-based features ---
    out = _add_time_features(out)

    # --- Lag features ---
    out = _add_lag_features(out)

    # --- Rolling features ---
    out = _add_rolling_features(out)

    # --- AQI change ---
    out["aqi_change_1h"] = out["us_aqi"].diff()

    # --- Hazardous flag ---
    out["is_hazardous"] = (out["us_aqi"] >= hazardous_threshold).astype(int)

    # --- Drop warm-up NaN rows ---
    out = out.dropna().reset_index(drop=True)

    return out


# ── private helpers ──────────────────────────────────────────────────


def _add_time_features(df):
    """Add cyclical hour/month encoding and day-of-week."""
    hour = df["time"].dt.hour
    month = df["time"].dt.month

    # Cyclical encoding — sin/cos so hour 23 is close to hour 0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    # Day of week as integer (0=Monday … 6=Sunday)
    df["day_of_week"] = df["time"].dt.dayofweek

    return df


def _add_lag_features(df):
    """Add lagged values for the target and PM2.5."""
    df["us_aqi_lag_1h"] = df["us_aqi"].shift(1)
    df["us_aqi_lag_3h"] = df["us_aqi"].shift(3)
    df["us_aqi_lag_24h"] = df["us_aqi"].shift(24)
    df["pm2_5_lag_1h"] = df["pm2_5"].shift(1)

    return df


def _add_rolling_features(df):
    """Add shifted rolling means for AQI and key pollutants."""
    # AQI rolling windows — shift(1) avoids leaking the current value
    shifted_aqi = df["us_aqi"].shift(1)
    df["aqi_rolling_3h"] = shifted_aqi.rolling(3).mean()
    df["aqi_rolling_6h"] = shifted_aqi.rolling(6).mean()
    df["aqi_rolling_24h"] = shifted_aqi.rolling(24).mean()

    # Pollutant rolling windows
    for raw_col, feat_col in _ROLLING_POLLUTANTS.items():
        df[feat_col] = df[raw_col].shift(1).rolling(6).mean()

    return df
