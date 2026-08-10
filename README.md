# Pearls AQI Predictor

Minimal scaffold for an AQI forecasting project that predicts the next 3 days of air quality using Open-Meteo for weather and pollutant data.

## Structure

- `src/aqi_predictor/data/` - API clients and raw data access.
- `src/aqi_predictor/features/` - feature and target generation.
- `src/aqi_predictor/pipelines/` - feature, backfill, and training pipeline entry points.
- `src/aqi_predictor/models/` - model evaluation, prediction, and registry helpers.
- `src/aqi_predictor/app/` - Streamlit dashboard and API entry points.
- `config/` - local configuration templates.
- `.github/workflows/` - scheduled automation scaffold.
- `data/`, `artifacts/`, `reports/` - local working outputs kept out of version control where appropriate.

Python files are intentionally empty placeholders.
