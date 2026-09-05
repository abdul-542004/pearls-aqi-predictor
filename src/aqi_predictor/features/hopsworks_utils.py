"""
Hopsworks feature store utilities.

Handles connection, feature group creation, and data insertion.
"""

import os
import tempfile
import time

import hopsworks


def get_feature_store():
    """Login to Hopsworks and return the feature store handle."""
    # Ensure /tmp exists on Windows for Hopsworks internal cert/PEM files
    try:
        os.makedirs("/tmp", exist_ok=True)
    except Exception:
        pass

    cert_folder = os.environ.get(
        "HOPSWORKS_CERT_FOLDER",
        os.path.join(tempfile.gettempdir(), "hopsworks_certs"),
    )
    os.makedirs(cert_folder, exist_ok=True)

    project = hopsworks.login(
        api_key_value=os.environ.get("HOPSWORKS_API_KEY"),
        project=os.environ.get("HOPSWORKS_PROJECT"),
        cert_folder=cert_folder,
    )
    return project.get_feature_store()


def get_or_create_feature_group(
    fs,
    name="aqi_features",
    version=None,
    description="Hourly AQI features for Karachi (v2: leakage-free direct multi-horizon features)",
    primary_key=None,
    event_time="time",
    time_travel_format="HUDI", # someone on discord suggested this 
):
    """Get an existing feature group or create a new one."""
    if version is None:
        version = int(os.environ.get("HOPSWORKS_FEATURE_GROUP_VERSION", "2"))

    if primary_key is None:
        primary_key = ["time"]

    fg = fs.get_or_create_feature_group(
        name=name,
        version=version,
        description=description,
        primary_key=primary_key,
        event_time=event_time,
        time_travel_format=time_travel_format,
    )
    return fg


MAX_RETRIES = 3
RETRY_BACKOFF = 30  # seconds


def insert_features(fg, df, wait_for_job=False):
    """Insert a DataFrame into the feature group with retry logic.

    By default, does NOT wait for the server-side materialization job
    to finish, which avoids connection timeouts during long backfills.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            fg.insert(df, write_options={"wait_for_job": wait_for_job})
            print(f"Inserted {len(df)} rows into feature group '{fg.name}' v{fg.version}")
            return
        except Exception as exc:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                print(f"  ⚠ Insert attempt {attempt} failed: {exc}")
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ✗ Insert failed after {MAX_RETRIES} attempts")
                raise
