"""
FastAPI application for the Pearls AQI Predictor.

Provides real-time air quality assessments, 72-hour direct multi-horizon forecasts,
historical trends, and model explainability/analytics.

Run locally:
    uvicorn aqi_predictor.app.api:app --app-dir src --reload --port 8000
or:
    python src/aqi_predictor/app/api.py
"""

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup root path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

from aqi_predictor.data.sources import Location, fetch_open_meteo_data
from aqi_predictor.features.build_features import build_features
from aqi_predictor.models.predict import direct_forecast_with_time, load_direct_models

# -- Constants & Configuration -----------------------------------------------

DEFAULT_CITY = os.getenv("AQI_CITY", "Karachi")
DEFAULT_LATITUDE = float(os.getenv("AQI_LATITUDE", "24.8608"))
DEFAULT_LONGITUDE = float(os.getenv("AQI_LONGITUDE", "67.0104"))
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# In-memory application state
state: Dict[str, Any] = {
    "models": {},
    "feature_cols": [],
    "training_report": {},
    "shap_importance_1h": {},
    "shap_importance_24h": {},
    "cache": {
        "raw_data": None,
        "features_df": None,
        "last_fetched": 0,
        "ttl_seconds": 300,  # 5-minute cache
    },
}


# -- EPA AQI Helper Functions ------------------------------------------------


def get_aqi_category(aqi: float) -> Dict[str, Any]:
    """Return EPA AQI category, color code, and advisory."""
    val = float(aqi)
    if val <= 50:
        return {
            "level": "Good",
            "code": "good",
            "color": "#10B981",  # Emerald Green
            "bg_color": "#D1FAE5",
            "severity": 1,
            "advisory": "Air quality is satisfactory and poses little or no risk.",
        }
    elif val <= 100:
        return {
            "level": "Moderate",
            "code": "moderate",
            "color": "#FBBF24",  # Amber/Yellow
            "bg_color": "#FEF3C7",
            "severity": 2,
            "advisory": "Air quality is acceptable. Sensitive individuals should consider limiting prolonged outdoor exertion.",
        }
    elif val <= 150:
        return {
            "level": "Unhealthy for Sensitive Groups",
            "code": "sensitive",
            "color": "#F97316",  # Orange
            "bg_color": "#FFEDD5",
            "severity": 3,
            "advisory": "Members of sensitive groups (asthma, heart/lung disease, children, elderly) should reduce outdoor exertion.",
        }
    elif val <= 200:
        return {
            "level": "Unhealthy",
            "code": "unhealthy",
            "color": "#EF4444",  # Red
            "bg_color": "#FEE2E2",
            "severity": 4,
            "advisory": "Everyone may begin to experience health effects. Avoid prolonged outdoor activities and wear a mask.",
        }
    elif val <= 300:
        return {
            "level": "Very Unhealthy",
            "code": "very_unhealthy",
            "color": "#8B5CF6",  # Purple
            "bg_color": "#EDE9FE",
            "severity": 5,
            "advisory": "Health alert: The risk of health effects is increased for everyone. Stay indoors and run air purifiers.",
        }
    else:
        return {
            "level": "Hazardous",
            "code": "hazardous",
            "color": "#881337",  # Maroon
            "bg_color": "#FFE4E6",
            "severity": 6,
            "advisory": "Health warning of emergency conditions. Everyone is likely to be affected. Keep windows strictly closed.",
        }


def get_dominant_pollutant(row: pd.Series) -> str:
    """Identify which pollutant is most elevated relative to standard thresholds."""
    # Approximate relative severity weights based on typical hazardous levels
    pollutants = {
        "PM2.5": row.get("pm2_5", 0.0) / 35.4,
        "PM10": row.get("pm10", 0.0) / 154.0,
        "Ozone": row.get("ozone", 0.0) / 100.0,
        "NO₂": row.get("nitrogen_dioxide", 0.0) / 100.0,
        "SO₂": row.get("sulphur_dioxide", 0.0) / 75.0,
        "CO": row.get("carbon_monoxide", 0.0) / 1000.0,
    }
    valid = {k: v for k, v in pollutants.items() if not pd.isna(v)}
    return max(valid, key=valid.get) if valid else "PM2.5"


def fetch_cached_weather_and_features(
    location: Location, force_refresh: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch recent 5-day actuals + 3-day forecasts with in-memory TTL caching.
    """
    now = time.time()
    cache = state["cache"]

    if (
        not force_refresh
        and cache["raw_data"] is not None
        and cache["features_df"] is not None
        and (now - cache["last_fetched"]) < cache["ttl_seconds"]
    ):
        return cache["raw_data"], cache["features_df"]

    try:
        # Fetch 5 days history + 3 days forecast from Open-Meteo
        raw = fetch_open_meteo_data(location, past_days=5, forecast_days=3)
        features = build_features(raw, drop_raw_pollutants=True)

        cache["raw_data"] = raw
        cache["features_df"] = features
        cache["last_fetched"] = now
        return raw, features
    except Exception as e:
        if cache["raw_data"] is not None and cache["features_df"] is not None:
            print(f"Warning: Fetch failed, using cached data: {e}")
            return cache["raw_data"], cache["features_df"]
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch real-time air quality data: {str(e)}",
        )


# -- Lifespan Management ----------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models, reports, and feature schema on server startup."""
    print("Initializing Pearls AQI Predictor API...")

    # 1. Load direct multi-horizon models
    try:
        models = load_direct_models(ARTIFACTS_DIR, model_type="xgboost")
        state["models"] = models
        print(f"  Loaded {len(models)} direct forecast models: {list(models.keys())}h")
    except Exception as e:
        print(f"  Warning: Could not load direct models: {e}")

    # 2. Load training report if available
    report_file = ARTIFACTS_DIR / "training_report.json"
    if report_file.exists():
        try:
            with open(report_file, "r") as f:
                state["training_report"] = json.load(f)
                state["feature_cols"] = state["training_report"].get("base_features", [])
            print(f"  Loaded training report from {report_file}")
        except Exception as e:
            print(f"  Warning: Could not load training report: {e}")

    # 3. Load SHAP importance reports if available
    shap_1h = REPORTS_DIR / "shap_importance_1h.json"
    if shap_1h.exists():
        try:
            with open(shap_1h, "r") as f:
                state["shap_importance_1h"] = json.load(f)
        except Exception:
            pass

    shap_24h = REPORTS_DIR / "shap_importance_24h.json"
    if shap_24h.exists():
        try:
            with open(shap_24h, "r") as f:
                state["shap_importance_24h"] = json.load(f)
        except Exception:
            pass

    # 4. Pre-fetch initial data
    try:
        loc = Location(latitude=DEFAULT_LATITUDE, longitude=DEFAULT_LONGITUDE, name=DEFAULT_CITY)
        fetch_cached_weather_and_features(loc)
        print("  Pre-fetched real-time weather & feature store cache")
    except Exception as e:
        print(f"  Initial data pre-fetch notice: {e}")

    print("Ready to serve predictions!\n")
    yield
    print("Shutting down AQI Predictor API...")


# -- FastAPI Application Definition ------------------------------------------

app = FastAPI(
    title="Pearls AQI Predictor API",
    description="Direct Multi-Horizon Air Quality Forecasting & Analytics API for Karachi",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for React frontend (Vite default :5173, Next.js, and production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for easy development and cloud deploy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Response Schemas --------------------------------------------------------


class LocationInfo(BaseModel):
    name: str
    latitude: float
    longitude: float


class AQICategory(BaseModel):
    level: str
    code: str
    color: str
    bg_color: str
    severity: int
    advisory: str


class WeatherMetrics(BaseModel):
    temperature_2m: Optional[float] = None
    relative_humidity_2m: Optional[float] = None
    precipitation: Optional[float] = None
    pressure_msl: Optional[float] = None
    wind_speed_10m: Optional[float] = None
    wind_direction_10m: Optional[float] = None
    wind_u: Optional[float] = None
    wind_v: Optional[float] = None


class PollutantMetrics(BaseModel):
    pm2_5: Optional[float] = None
    pm10: Optional[float] = None
    carbon_monoxide: Optional[float] = None
    nitrogen_dioxide: Optional[float] = None
    sulphur_dioxide: Optional[float] = None
    ozone: Optional[float] = None


class CurrentAQIResponse(BaseModel):
    time: str
    location: LocationInfo
    us_aqi: float
    category: AQICategory
    dominant_pollutant: str
    weather: WeatherMetrics
    pollutants: PollutantMetrics
    is_hazardous: bool


class ForecastPoint(BaseModel):
    time: str
    hours_ahead: int
    predicted_aqi: float
    category: AQICategory
    temperature_2m: Optional[float] = None
    relative_humidity_2m: Optional[float] = None
    wind_speed_10m: Optional[float] = None
    precipitation: Optional[float] = None


class ForecastResponse(BaseModel):
    location: LocationInfo
    generated_at: str
    model_strategy: str
    forecast_hours: int
    summary: Dict[str, Any]
    forecast: List[ForecastPoint]


class HistoryPoint(BaseModel):
    time: str
    us_aqi: float
    category: AQICategory
    temperature_2m: Optional[float] = None
    relative_humidity_2m: Optional[float] = None
    wind_speed_10m: Optional[float] = None
    pm2_5: Optional[float] = None
    pm10: Optional[float] = None


class HistoryResponse(BaseModel):
    location: LocationInfo
    history_hours: int
    history: List[HistoryPoint]


class HealthResponse(BaseModel):
    status: str
    version: str
    location: LocationInfo
    models_loaded: List[int]
    total_models: int
    cache_status: str


# -- API Routes --------------------------------------------------------------


@app.get("/", tags=["System"])
def root():
    """Root endpoint providing metadata and documentation link."""
    return {
        "service": "Pearls AQI Predictor API",
        "version": "2.0.0",
        "docs_url": "/docs",
        "location": DEFAULT_CITY,
        "status": "online",
    }


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def get_health():
    """Health check endpoint displaying model status and server info."""
    models = state.get("models", {})
    cache = state.get("cache", {})
    cache_age = int(time.time() - cache["last_fetched"]) if cache["last_fetched"] else -1

    return HealthResponse(
        status="healthy" if models else "degraded",
        version="2.0.0",
        location=LocationInfo(
            name=DEFAULT_CITY, latitude=DEFAULT_LATITUDE, longitude=DEFAULT_LONGITUDE
        ),
        models_loaded=sorted(list(models.keys())),
        total_models=len(models),
        cache_status=f"Active (age: {cache_age}s, ttl: {cache.get('ttl_seconds')}s)",
    )


def get_now_local() -> pd.Timestamp:
    """Return the current timestamp in Karachi local time (matching Open-Meteo local timezone)."""
    return pd.Timestamp.now(tz="Asia/Karachi").tz_localize(None)


def get_latest_observation(raw_df: pd.DataFrame) -> pd.Series:
    """Filter for the most recent observation at or before now."""
    now_local = get_now_local()
    valid = raw_df.dropna(subset=["us_aqi"]).sort_values("time")
    past_valid = valid[valid["time"] <= now_local]
    if not past_valid.empty:
        return past_valid.iloc[-1]
    if not valid.empty:
        return valid.iloc[-1]
    raise HTTPException(status_code=404, detail="No recent air quality observations found.")


@app.get("/api/current", response_model=CurrentAQIResponse, tags=["Air Quality"])
def get_current_aqi(
    city: Optional[str] = Query(DEFAULT_CITY, description="Target city name"),
    force_refresh: bool = Query(False, description="Bypass cache and fetch fresh data"),
):
    """
    Get the real-time current AQI, EPA category, dominant pollutant,
    and weather/pollutant conditions.
    """
    location = Location(
        latitude=DEFAULT_LATITUDE, longitude=DEFAULT_LONGITUDE, name=city
    )
    raw_df, _ = fetch_cached_weather_and_features(location, force_refresh=force_refresh)

    latest = get_latest_observation(raw_df)
    aqi_val = round(float(latest["us_aqi"]), 1)
    category = get_aqi_category(aqi_val)
    dominant = get_dominant_pollutant(latest)

    # Compute wind vectors if missing
    ws = latest.get("wind_speed_10m", np.nan)
    wd = latest.get("wind_direction_10m", np.nan)
    wind_u = float(ws * np.sin(np.deg2rad(wd))) if not pd.isna(ws) and not pd.isna(wd) else None
    wind_v = float(ws * np.cos(np.deg2rad(wd))) if not pd.isna(ws) and not pd.isna(wd) else None

    return CurrentAQIResponse(
        time=pd.to_datetime(latest["time"]).isoformat(),
        location=LocationInfo(
            name=city or DEFAULT_CITY,
            latitude=location.latitude,
            longitude=location.longitude,
        ),
        us_aqi=aqi_val,
        category=AQICategory(**category),
        dominant_pollutant=dominant,
        weather=WeatherMetrics(
            temperature_2m=float(latest["temperature_2m"]) if not pd.isna(latest.get("temperature_2m")) else None,
            relative_humidity_2m=float(latest["relative_humidity_2m"]) if not pd.isna(latest.get("relative_humidity_2m")) else None,
            precipitation=float(latest["precipitation"]) if not pd.isna(latest.get("precipitation")) else None,
            pressure_msl=float(latest["pressure_msl"]) if not pd.isna(latest.get("pressure_msl")) else None,
            wind_speed_10m=float(latest["wind_speed_10m"]) if not pd.isna(latest.get("wind_speed_10m")) else None,
            wind_direction_10m=float(latest["wind_direction_10m"]) if not pd.isna(latest.get("wind_direction_10m")) else None,
            wind_u=wind_u,
            wind_v=wind_v,
        ),
        pollutants=PollutantMetrics(
            pm2_5=float(latest["pm2_5"]) if not pd.isna(latest.get("pm2_5")) else None,
            pm10=float(latest["pm10"]) if not pd.isna(latest.get("pm10")) else None,
            carbon_monoxide=float(latest["carbon_monoxide"]) if not pd.isna(latest.get("carbon_monoxide")) else None,
            nitrogen_dioxide=float(latest["nitrogen_dioxide"]) if not pd.isna(latest.get("nitrogen_dioxide")) else None,
            sulphur_dioxide=float(latest["sulphur_dioxide"]) if not pd.isna(latest.get("sulphur_dioxide")) else None,
            ozone=float(latest["ozone"]) if not pd.isna(latest.get("ozone")) else None,
        ),
        is_hazardous=bool(aqi_val >= 300),
    )


@app.get("/api/forecast", response_model=ForecastResponse, tags=["Air Quality"])
def get_forecast(
    hours: int = Query(72, ge=1, le=72, description="Forecast horizon in hours (1-72)"),
    force_refresh: bool = Query(False, description="Bypass cache and fetch fresh data"),
):
    """
    Generate a 72-hour direct multi-horizon AQI forecast.

    Uses dedicated models for +1h, +6h, +12h, +24h, +48h, +72h with weather
    forecast integration and linear interpolation for intermediate hours.
    """
    models = state.get("models", {})
    if not models:
        raise HTTPException(
            status_code=503,
            detail="Direct forecasting models are not loaded. Run training_pipeline.py first.",
        )

    location = Location(
        latitude=DEFAULT_LATITUDE, longitude=DEFAULT_LONGITUDE, name=DEFAULT_CITY
    )
    raw_df, features_df = fetch_cached_weather_and_features(location, force_refresh=force_refresh)

    if features_df.empty:
        raise HTTPException(status_code=500, detail="Feature DataFrame is empty.")

    latest_obs = get_latest_observation(raw_df)
    current_time = latest_obs["time"]

    # History up to the current observation time
    history_features = features_df[features_df["time"] <= current_time]
    if history_features.empty:
        history_features = features_df.head(24)

    # Prepare future weather forecast lookup
    forecast_weather = raw_df.set_index("time").copy()

    # Determine base feature columns
    feature_cols = state.get("feature_cols", [])
    if not feature_cols:
        drop = {"time", "us_aqi", "is_hazardous"}
        feature_cols = sorted(set(features_df.columns) - drop)

    # Generate multi-horizon direct forecast
    forecast_df = direct_forecast_with_time(
        models=models,
        history_df=history_features,
        feature_cols=feature_cols,
        max_hours=hours,
        forecast_weather_df=forecast_weather,
    )

    points: List[ForecastPoint] = []
    aqi_values: List[float] = []

    for _, r in forecast_df.iterrows():
        t_str = pd.to_datetime(r["time"]).isoformat()
        h_ahead = int(r["hours_ahead"])
        aqi = round(float(r["predicted_aqi"]), 1)
        aqi_values.append(aqi)

        # Lookup weather at forecast time if available
        fw_row = forecast_weather.loc[r["time"]] if r["time"] in forecast_weather.index else None
        if isinstance(fw_row, pd.DataFrame):
            fw_row = fw_row.iloc[0]

        temp = float(fw_row["temperature_2m"]) if fw_row is not None and not pd.isna(fw_row.get("temperature_2m")) else None
        hum = float(fw_row["relative_humidity_2m"]) if fw_row is not None and not pd.isna(fw_row.get("relative_humidity_2m")) else None
        ws = float(fw_row["wind_speed_10m"]) if fw_row is not None and not pd.isna(fw_row.get("wind_speed_10m")) else None
        prec = float(fw_row["precipitation"]) if fw_row is not None and not pd.isna(fw_row.get("precipitation")) else None

        points.append(
            ForecastPoint(
                time=t_str,
                hours_ahead=h_ahead,
                predicted_aqi=aqi,
                category=AQICategory(**get_aqi_category(aqi)),
                temperature_2m=temp,
                relative_humidity_2m=hum,
                wind_speed_10m=ws,
                precipitation=prec,
            )
        )

    # Summary analytics for the forecast period
    summary = {
        "min_aqi": min(aqi_values) if aqi_values else 0,
        "max_aqi": max(aqi_values) if aqi_values else 0,
        "avg_aqi": round(float(np.mean(aqi_values)), 1) if aqi_values else 0,
        "peak_hour": points[int(np.argmax(aqi_values))].time if aqi_values else "",
        "cleanest_hour": points[int(np.argmin(aqi_values))].time if aqi_values else "",
        "hazardous_hours_count": sum(1 for a in aqi_values if a >= 300),
    }

    return ForecastResponse(
        location=LocationInfo(
            name=DEFAULT_CITY, latitude=DEFAULT_LATITUDE, longitude=DEFAULT_LONGITUDE
        ),
        generated_at=datetime.now(timezone.utc).isoformat(),
        model_strategy="direct_multi_horizon_xgboost_with_weather_forecast_deltas",
        forecast_hours=hours,
        summary=summary,
        forecast=points,
    )


@app.get("/api/history", response_model=HistoryResponse, tags=["Air Quality"])
def get_history(
    hours: int = Query(48, ge=6, le=120, description="Number of past hours to return"),
):
    """
    Get recent historical hourly air quality and weather observations.
    """
    location = Location(
        latitude=DEFAULT_LATITUDE, longitude=DEFAULT_LONGITUDE, name=DEFAULT_CITY
    )
    raw_df, _ = fetch_cached_weather_and_features(location)

    latest_obs = get_latest_observation(raw_df)
    current_time = latest_obs["time"]

    # Get past observations up to now
    valid_obs = raw_df.dropna(subset=["us_aqi"]).sort_values("time")
    past_obs = valid_obs[valid_obs["time"] <= current_time]
    recent = past_obs.tail(hours)

    points: List[HistoryPoint] = []
    for _, r in recent.iterrows():
        aqi = round(float(r["us_aqi"]), 1)
        points.append(
            HistoryPoint(
                time=pd.to_datetime(r["time"]).isoformat(),
                us_aqi=aqi,
                category=AQICategory(**get_aqi_category(aqi)),
                temperature_2m=float(r["temperature_2m"]) if not pd.isna(r.get("temperature_2m")) else None,
                relative_humidity_2m=float(r["relative_humidity_2m"]) if not pd.isna(r.get("relative_humidity_2m")) else None,
                wind_speed_10m=float(r["wind_speed_10m"]) if not pd.isna(r.get("wind_speed_10m")) else None,
                pm2_5=float(r["pm2_5"]) if not pd.isna(r.get("pm2_5")) else None,
                pm10=float(r["pm10"]) if not pd.isna(r.get("pm10")) else None,
            )
        )

    return HistoryResponse(
        location=LocationInfo(
            name=DEFAULT_CITY, latitude=DEFAULT_LATITUDE, longitude=DEFAULT_LONGITUDE
        ),
        history_hours=len(points),
        history=points,
    )



@app.get("/api/analytics", tags=["Model Analytics & Explainability"])
def get_model_analytics():
    """
    Get model evaluation metrics (RMSE, MAE, R²) and SHAP feature importance.
    """
    report = state.get("training_report", {})
    shap_1h = state.get("shap_importance_1h", {})
    shap_24h = state.get("shap_importance_24h", {})

    return {
        "status": "available" if report else "not_found",
        "training_report": report,
        "shap_importance": {
            "1h_model": shap_1h,
            "24h_model": shap_24h,
        },
        "model_architecture": {
            "strategy": "Direct Multi-Horizon Forecasting",
            "horizons": [1, 6, 12, 24, 48, 72],
            "models_evaluated": ["XGBoost", "Random Forest", "LSTM"],
            "features_used": len(state.get("feature_cols", [])),
            "key_features": [
                "us_aqi_lag_24h (Diurnal autocorrelation)",
                "delta_wind_u (Wind dispersion change)",
                "delta_temp_humidity (Particulate formation index)",
                "forecast_wind_u_24h",
                "aqi_ewm_12h (Decaying historical context)",
            ],
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "aqi_predictor.app.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        app_dir=str(PROJECT_ROOT / "src"),
    )

