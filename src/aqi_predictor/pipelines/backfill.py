"""
Historical backfill — populates the Hopsworks feature store with 3 years
of engineered features from Open-Meteo historical data.

Run this once to create the training dataset before model training.

Open-Meteo's historical API has a limit on how much data can be
fetched per request, so we chunk the range into 90-day windows.
"""

import os
import sys
from datetime import date, timedelta
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

# Days per chunk — keeps individual API requests manageable
CHUNK_DAYS = 90


def run(years=3):
    """Execute the historical backfill pipeline."""
    city = os.getenv("AQI_CITY", "Karachi")
    latitude = float(os.getenv("AQI_LATITUDE", "24.8608"))
    longitude = float(os.getenv("AQI_LONGITUDE", "67.0104"))

    location = Location(latitude=latitude, longitude=longitude, name=city)

    end = date.today() - timedelta(days=5)
    start = end - timedelta(days=365 * years)

    print(f"Backfilling {city} from {start} to {end} ({years} years)")

    # Connect to Hopsworks once
    print("Connecting to Hopsworks...")
    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)

    # Process in chunks
    chunk_start = start
    chunk_number = 0

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end)
        chunk_number += 1

        print(f"\n--- Chunk {chunk_number}: {chunk_start} → {chunk_end} ---")

        # Fetch raw data for this chunk
        print("  Fetching data from Open-Meteo...")
        raw = fetch_open_meteo_data(
            location,
            start_date=chunk_start,
            end_date=chunk_end,
        )
        print(f"  Fetched {len(raw)} rows")

        # Engineer features
        print("  Building features...")
        features = build_features(raw)
        print(f"  Produced {len(features)} feature rows")

        # Insert into Hopsworks
        if len(features) > 0:
            insert_features(fg, features)
        else:
            print("  No rows after dropping warm-up NaNs, skipping insert")

        # Move to next chunk
        chunk_start = chunk_end + timedelta(days=1)

    print(f"\nBackfill complete. Processed {chunk_number} chunks.")


if __name__ == "__main__":
    run()
