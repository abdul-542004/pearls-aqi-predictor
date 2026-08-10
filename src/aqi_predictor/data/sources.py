from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import requests


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_HOURLY_FIELDS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
]

AIR_QUALITY_HOURLY_FIELDS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
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
    forecast_days=3,
):
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "hourly": ",".join(WEATHER_HOURLY_FIELDS),
        "timezone": "auto",
    }

    params.update(_time_params(start_date, end_date, forecast_days))

    payload = _get_json(FORECAST_URL, params)

    return _hourly_payload_to_frame(payload, location)


def fetch_air_quality(
    location,
    *,
    start_date=None,
    end_date=None,
    forecast_days=3,
):
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "hourly": ",".join(AIR_QUALITY_HOURLY_FIELDS),
        "timezone": "auto",
    }

    params.update(_time_params(start_date, end_date, forecast_days))

    payload = _get_json(AIR_QUALITY_URL, params)

    return _hourly_payload_to_frame(payload, location)


def fetch_open_meteo_data(
    location,
    *,
    start_date=None,
    end_date=None,
    forecast_days=3,
):
    weather = fetch_weather(
        location,
        start_date=start_date,
        end_date=end_date,
        forecast_days=forecast_days,
    )

    air_quality = fetch_air_quality(
        location,
        start_date=start_date,
        end_date=end_date,
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
    forecast_days,
):
    if start_date is None and end_date is None:
        return {
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


