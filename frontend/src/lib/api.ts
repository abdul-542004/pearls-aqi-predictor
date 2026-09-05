// API client for the FastAPI backend

const BASE_URL = "/api"

export interface LocationInfo {
  name: string
  latitude: number
  longitude: number
}

export interface AQICategory {
  level: string
  code: string
  color: string
  bg_color: string
  severity: number
  advisory: string
}

export interface WeatherMetrics {
  temperature_2m: number | null
  relative_humidity_2m: number | null
  precipitation: number | null
  pressure_msl: number | null
  wind_speed_10m: number | null
  wind_direction_10m: number | null
  wind_u: number | null
  wind_v: number | null
}

export interface PollutantMetrics {
  pm2_5: number | null
  pm10: number | null
  carbon_monoxide: number | null
  nitrogen_dioxide: number | null
  sulphur_dioxide: number | null
  ozone: number | null
}

export interface CurrentAQIResponse {
  time: string
  location: LocationInfo
  us_aqi: number
  category: AQICategory
  dominant_pollutant: string
  weather: WeatherMetrics
  pollutants: PollutantMetrics
  is_hazardous: boolean
}

export interface ForecastPoint {
  time: string
  hours_ahead: number
  predicted_aqi: number
  category: AQICategory
  temperature_2m: number | null
  relative_humidity_2m: number | null
  wind_speed_10m: number | null
  precipitation: number | null
}

export interface ForecastResponse {
  location: LocationInfo
  generated_at: string
  model_strategy: string
  forecast_hours: number
  summary: {
    min_aqi: number
    max_aqi: number
    avg_aqi: number
    peak_hour: string
    cleanest_hour: string
    hazardous_hours_count: number
  }
  forecast: ForecastPoint[]
}

export interface HistoryPoint {
  time: string
  us_aqi: number
  category: AQICategory
  temperature_2m: number | null
  relative_humidity_2m: number | null
  wind_speed_10m: number | null
  pm2_5: number | null
  pm10: number | null
}

export interface HistoryResponse {
  location: LocationInfo
  history_hours: number
  history: HistoryPoint[]
}

export interface ShapFeature {
  feature: string
  mean_abs_shap: number
}

export interface AnalyticsResponse {
  status: string
  training_report: {
    strategy?: string
    horizons?: number[]
    results_per_horizon?: Record<string, Record<string, { rmse: number; mae: number; r2: number }>>
    best_model_per_horizon?: Record<string, string>
    split?: { train: number; val: number; test: number }
  }
  shap_importance: {
    "1h_model": ShapFeature[]
    "24h_model": ShapFeature[]
  }
  model_architecture: {
    strategy: string
    horizons: number[]
    models_evaluated: string[]
    features_used: number
    key_features: string[]
  }
}

export interface HealthResponse {
  status: string
  version: string
  location: LocationInfo
  models_loaded: number[]
  total_models: number
  cache_status: string
}

async function fetchJSON<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`API Error ${res.status}: ${detail}`)
  }
  return res.json()
}

export const api = {
  getCurrentAQI: () => fetchJSON<CurrentAQIResponse>(`${BASE_URL}/current`),
  getForecast: (hours = 72) => fetchJSON<ForecastResponse>(`${BASE_URL}/forecast?hours=${hours}`),
  getHistory: (hours = 48) => fetchJSON<HistoryResponse>(`${BASE_URL}/history?hours=${hours}`),
  getAnalytics: () => fetchJSON<AnalyticsResponse>(`${BASE_URL}/analytics`),
  getHealth: () => fetchJSON<HealthResponse>(`${BASE_URL}/health`),
}
