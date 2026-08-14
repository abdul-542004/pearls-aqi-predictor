"""
Hopsworks feature store utilities.

Handles connection, feature group creation, and data insertion.
"""

import os

import hopsworks


def get_feature_store():
    """Login to Hopsworks and return the feature store handle."""
    project = hopsworks.login(
        api_key_value=os.environ.get("HOPSWORKS_API_KEY"),
        project=os.environ.get("HOPSWORKS_PROJECT"),
    )
    return project.get_feature_store()


def get_or_create_feature_group(
    fs,
    name="aqi_features",
    version=1,
    description="Hourly AQI features for Karachi",
    primary_key=None,
    event_time="time",
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
    )
    return fg


def insert_features(fg, df):
    """Insert a DataFrame into the feature group."""
    fg.insert(df)
    print(f"Inserted {len(df)} rows into feature group '{fg.name}' v{fg.version}")
