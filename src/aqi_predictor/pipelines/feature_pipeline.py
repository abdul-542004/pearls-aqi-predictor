"""
Feature pipeline — fetches recent data and upserts features into Hopsworks.

Designed to run on a schedule (e.g., hourly via GitHub Actions).
It fetches the last 14 days of actual data plus 3 days of forecast data
from Open-Meteo, engineers features, and inserts them into the feature store.

The 14-day lookback ensures lag/rolling features are properly computed
even for the forecast period.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow running as a script from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

from aqi_predictor.data.sources import Location, fetch_open_meteo_data
from aqi_predictor.features.build_features import build_features
from aqi_predictor.features.hopsworks_utils import (
    get_feature_store,
    get_or_create_feature_group,
    insert_features,
)


def run():
    """Execute the feature pipeline."""
    city = os.getenv("AQI_CITY", "Karachi")
    latitude = float(os.getenv("AQI_LATITUDE", "24.8608"))
    longitude = float(os.getenv("AQI_LONGITUDE", "67.0104"))

    location = Location(latitude=latitude, longitude=longitude, name=city)

    # Fetch last 14 days of actuals + 3 days of forecasts
    # The 14-day lookback gives us enough history to compute
    # rolling-24h and lag-24h features for the forecast window
    print(f"Fetching recent + forecast data for {city}...")
    raw = fetch_open_meteo_data(location, past_days=14, forecast_days=3)
    print(f"  Fetched {len(raw)} rows ({raw['time'].min()} -> {raw['time'].max()})")

    # Engineer features
    print("Building features...")
    features = build_features(raw)
    print(f"  Produced {len(features)} feature rows, {len(features.columns)} columns")

    # Insert into Hopsworks
    print("Connecting to Hopsworks...")
    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)
    insert_features(fg, features)
    print("Feature pipeline complete.")


if __name__ == "__main__":
    run()
