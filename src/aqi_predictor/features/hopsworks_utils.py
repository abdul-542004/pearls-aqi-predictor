"""
Hopsworks feature store utilities.

Handles connection, feature group creation, and data insertion.
"""

import os
import tempfile

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
    version=1,
    description="Hourly AQI features for Karachi",
    primary_key=None,
    event_time="time",
    time_travel_format="HUDI", # someone on discord suggested this 
):
    """Get an existing feature group or create a new one."""
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


def insert_features(fg, df):
    """Insert a DataFrame into the feature group."""
    fg.insert(df)
    print(f"Inserted {len(df)} rows into feature group '{fg.name}' v{fg.version}")
