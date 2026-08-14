from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import requests

# Free Open-Meteo API endpoints
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"



# Two types of data are fetched from the Open-Meteo API:
# 1. Weather variables (used as input features)
# 2. Air quality and pollutant variables (used as input features and prediction target)

WEATHER_HOURLY_FIELDS = [
    "temperature_2m",         # Air temperature at 2 meters above ground (°C)
    "relative_humidity_2m",   # Relative humidity at 2 meters above ground (%)
    "precipitation",          # Amount of precipitation (rain/snow) during the hour (mm)
    "pressure_msl",           # Atmospheric pressure reduced to mean sea level (hPa)
    "wind_speed_10m",         # Wind speed at 10 meters above ground (km/h)
    "wind_direction_10m",     # Wind direction at 10 meters above ground (degrees: 0°=North, 90°=East)
]

AIR_QUALITY_HOURLY_FIELDS = [
    "pm10",                  # Particulate Matter ≤10 µm (µg/m³)
    "pm2_5",                 # Fine Particulate Matter ≤2.5 µm (µg/m³)
    "carbon_monoxide",       # Carbon Monoxide (µg/m³)
    "nitrogen_dioxide",      # Nitrogen Dioxide (µg/m³)
    "sulphur_dioxide",       # Sulphur Dioxide (µg/m³)
    "ozone",                 # Ground-level Ozone (µg/m³)
    "us_aqi",                # U.S. Air Quality Index calculated from pollutant concentrations
]


@dataclass(frozen=True)
class Location:
    latitude: float
    longitude: float
    name : str | None = None


def fetch_weather(
    location,
    *,
    start_date=None,
    end_date=None,
    past_days=14,
    forecast_days=3,
):
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "hourly": ",".join(WEATHER_HOURLY_FIELDS),
        "timezone": "auto",
    }

    params.update(_time_params(start_date, end_date, past_days, forecast_days))

    url = HISTORICAL_WEATHER_URL if start_date and end_date else FORECAST_URL
    payload = _get_json(url, params)

    return _hourly_payload_to_frame(payload, location)


def fetch_air_quality(
    location,
    *,
    start_date=None,
    end_date=None,
    past_days=14,
    forecast_days=3,
):
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "hourly": ",".join(AIR_QUALITY_HOURLY_FIELDS),
        "timezone": "auto",
    }

    params.update(_time_params(start_date, end_date, past_days, forecast_days))

    payload = _get_json(AIR_QUALITY_URL, params)

    return _hourly_payload_to_frame(payload, location)


def fetch_open_meteo_data(
    location,
    *,
    start_date=None,
    end_date=None,
    past_days=14,
    forecast_days=3,
):
    """
    Fetches weather and air quality data from Open-Meteo API for a given location.
    """
    weather = fetch_weather(
        location,
        start_date=start_date,
        end_date=end_date,
        past_days=past_days,
        forecast_days=forecast_days,
    )

    air_quality = fetch_air_quality(
        location,
        start_date=start_date,
        end_date=end_date,
        past_days=past_days,
        forecast_days=forecast_days,
    )

    join_columns = [
        "time",
        "latitude",
        "longitude",
        "location_name",
    ]

    return (
        weather
        # merge the two dataframes on time, latitude, longitude, and location_name
        .merge(air_quality, on=join_columns, how="outer")
        .sort_values("time")
    )


def _get_json(url, params):
    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def _time_params(
    start_date,
    end_date,
    past_days,
    forecast_days,
):
    if start_date is None and end_date is None:
        return {
            "past_days": past_days,
            "forecast_days": forecast_days,
        }

    if start_date is None or end_date is None:
        raise ValueError(
            "start_date and end_date must be provided together."
        )

    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }


def _hourly_payload_to_frame(payload, location):
    hourly = payload.get("hourly")

    if not hourly or "time" not in hourly:
        raise ValueError(
            "Open-Meteo response did not include hourly data."
        )

    frame = pd.DataFrame(hourly)

    frame["time"] = pd.to_datetime(frame["time"])
    frame["latitude"] = location.latitude
    frame["longitude"] = location.longitude
    frame["location_name"] = location.name

    return frame
