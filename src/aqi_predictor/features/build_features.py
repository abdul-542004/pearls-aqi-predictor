"""
Feature engineering for AQI prediction.

Takes raw Open-Meteo data and produces a feature-enriched DataFrame
ready for insertion into the Hopsworks feature store.
"""

import numpy as np
import pandas as pd


# Columns to drop — constants with no predictive value
_DROP_COLUMNS = ["latitude", "longitude", "location_name"]

# Raw pollutant columns that directly compose the US AQI formula.
# Including these as features causes target leakage since AQI is
# a deterministic function of these values.  We keep only their
# *lagged* and *rolling* derivatives, which represent past values
# legitimately available at inference time.
_RAW_POLLUTANT_COLS = [
    "pm2_5", "pm10", "carbon_monoxide",
    "nitrogen_dioxide", "sulphur_dioxide", "ozone",
]

# Pollutants for which we compute lag and rolling features
_LAG_POLLUTANTS = {
    "pm2_5": [1, 3],
    "pm10": [1, 3],
    "carbon_monoxide": [1],
    "nitrogen_dioxide": [1],
    "sulphur_dioxide": [1],
    "ozone": [1, 3],
}

# Rolling window sizes for each pollutant
_ROLLING_POLLUTANTS = {
    "pm2_5": [6, 12, 24],
    "pm10": [6, 12, 24],
    "carbon_monoxide": [6],
    "nitrogen_dioxide": [6],
    "sulphur_dioxide": [6],
    "ozone": [6, 12],
}


def build_features(df, hazardous_threshold=300, drop_raw_pollutants=True):
    """
    Engineer features from raw Open-Meteo weather + air-quality data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with columns from ``sources.py`` (weather + AQ merged).
    hazardous_threshold : int
        AQI value at or above which the ``is_hazardous`` flag is set.
    drop_raw_pollutants : bool
        If True (default), remove current-timestep pollutant values to
        prevent target leakage.  Lagged/rolling derivatives are kept.

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

    # --- AQI lag features ---
    out = _add_aqi_lag_features(out)

    # --- AQI rolling features ---
    out = _add_aqi_rolling_features(out)

    # --- Pollutant lag features ---
    out = _add_pollutant_lag_features(out)

    # --- Pollutant rolling features ---
    out = _add_pollutant_rolling_features(out)

    # --- AQI change & trend features ---
    out = _add_trend_features(out)

    # --- Weather interaction features ---
    out = _add_weather_interactions(out)

    # --- Hazardous flag ---
    out["is_hazardous"] = (out["us_aqi"] >= hazardous_threshold).astype(int)

    # --- Drop raw pollutant columns to prevent target leakage ---
    if drop_raw_pollutants:
        cols_to_drop = [c for c in _RAW_POLLUTANT_COLS if c in out.columns]
        out = out.drop(columns=cols_to_drop)

    # --- Drop warm-up NaN rows ---
    out = out.dropna().reset_index(drop=True)

    return out


# ── private helpers ──────────────────────────────────────────────────


def _add_time_features(df):
    """Add cyclical hour/month/day-of-week encoding and weekend flag."""
    hour = df["time"].dt.hour
    month = df["time"].dt.month
    dow = df["time"].dt.dayofweek

    # Cyclical encoding — sin/cos so hour 23 is close to hour 0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    # Day of week — cyclical encoding only (no raw integer;
    # the sin/cos pair fully encodes day-of-week information)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Weekend flag (Saturday=5, Sunday=6)
    df["is_weekend"] = (dow >= 5).astype(int)

    return df


def _add_aqi_lag_features(df):
    """Add lagged values of the AQI target.

    ACF analysis confirms significant autocorrelation at all lags
    up to 72h (ACF=0.440 at 72h), with PACF showing direct influence
    at 24h, 48h, and 72h.  Longer lags give the model context for
    multi-day forecasting horizons.
    """
    df["us_aqi_lag_1h"] = df["us_aqi"].shift(1)
    df["us_aqi_lag_3h"] = df["us_aqi"].shift(3)
    df["us_aqi_lag_6h"] = df["us_aqi"].shift(6)
    df["us_aqi_lag_12h"] = df["us_aqi"].shift(12)
    df["us_aqi_lag_24h"] = df["us_aqi"].shift(24)
    df["us_aqi_lag_48h"] = df["us_aqi"].shift(48)   # ACF=0.547
    df["us_aqi_lag_72h"] = df["us_aqi"].shift(72)   # ACF=0.440

    return df


def _add_aqi_rolling_features(df):
    """
    Add shifted rolling means for AQI.
    shift(1) avoids leaking the current value into the rolling window.

    The 48h rolling window provides a smoother multi-day context that
    is more useful for longer forecast horizons (24h–72h) where the
    short-window features become stale.
    """
    shifted_aqi = df["us_aqi"].shift(1)
    df["aqi_rolling_3h"] = shifted_aqi.rolling(3).mean()
    df["aqi_rolling_6h"] = shifted_aqi.rolling(6).mean()
    df["aqi_rolling_12h"] = shifted_aqi.rolling(12).mean()
    df["aqi_rolling_24h"] = shifted_aqi.rolling(24).mean()
    df["aqi_rolling_48h"] = shifted_aqi.rolling(48).mean()

    # Rolling standard deviation — captures volatility
    df["aqi_std_6h"] = shifted_aqi.rolling(6).std()
    df["aqi_std_24h"] = shifted_aqi.rolling(24).std()

    # Rolling min/max — captures range
    df["aqi_min_24h"] = shifted_aqi.rolling(24).min()
    df["aqi_max_24h"] = shifted_aqi.rolling(24).max()

    # Exponential weighted mean — gives more weight to recent values
    # while still capturing long-term context.  Half-life of 12h means
    # data from 24h ago contributes ~25%, from 48h ago ~6%.
    df["aqi_ewm_12h"] = shifted_aqi.ewm(halflife=12).mean()

    return df


def _add_pollutant_lag_features(df):
    """Add lagged values for pollutants (past values, no leakage)."""
    for pollutant, lags in _LAG_POLLUTANTS.items():
        if pollutant in df.columns:
            for lag in lags:
                df[f"{pollutant}_lag_{lag}h"] = df[pollutant].shift(lag)

    return df


def _add_pollutant_rolling_features(df):
    """Add shifted rolling means for pollutants."""
    for pollutant, windows in _ROLLING_POLLUTANTS.items():
        if pollutant in df.columns:
            shifted = df[pollutant].shift(1)
            for window in windows:
                df[f"{pollutant}_rolling_{window}h"] = shifted.rolling(window).mean()

    return df


def _add_trend_features(df):
    """Add rate-of-change and trend features."""
    # AQI change (1-hour diff)
    df["aqi_change_1h"] = df["us_aqi"].diff()

    # AQI change over longer windows
    df["aqi_change_3h"] = df["us_aqi"].diff(3)
    df["aqi_change_6h"] = df["us_aqi"].diff(6)
    df["aqi_change_24h"] = df["us_aqi"].diff(24)

    # Trend: current vs rolling average (positive = above average = worsening)
    if "aqi_rolling_6h" in df.columns:
        df["aqi_trend_6h"] = df["us_aqi"] - df["aqi_rolling_6h"]
    if "aqi_rolling_24h" in df.columns:
        df["aqi_trend_24h"] = df["us_aqi"] - df["aqi_rolling_24h"]

    return df


def _add_weather_interactions(df):
    """Add weather-derived interaction features for pollutant dispersion."""
    # Wind dispersion — a proxy for how effectively wind clears pollutants.
    # Higher wind speed + direction alignment = better dispersion.
    if "wind_speed_10m" in df.columns and "wind_direction_10m" in df.columns:
        wind_dir_rad = np.deg2rad(df["wind_direction_10m"])
        df["wind_u"] = df["wind_speed_10m"] * np.sin(wind_dir_rad)
        df["wind_v"] = df["wind_speed_10m"] * np.cos(wind_dir_rad)

    # Temperature-humidity interaction (affects particulate formation)
    if "temperature_2m" in df.columns and "relative_humidity_2m" in df.columns:
        df["temp_humidity"] = df["temperature_2m"] * df["relative_humidity_2m"] / 100.0

    # Recent precipitation flag — rain washes out pollutants
    if "precipitation" in df.columns:
        df["precip_last_6h"] = df["precipitation"].shift(1).rolling(6).sum()
        df["had_rain_24h"] = (df["precipitation"].shift(1).rolling(24).sum() > 0.1).astype(int)

    return df
