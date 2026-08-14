# Karachi AQI EDA — Full Analysis

This document presents a complete review of the [01_eda_open_meteo_aqi.ipynb](file:///c:/MyWorks/aqi-predictor/notebooks/01_eda_open_meteo_aqi.ipynb) notebook, covering every textual output **and** visual chart produced.

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| **City** | Karachi (24.8608°N, 67.0104°E) |
| **Source** | Open-Meteo API (weather + air quality) |
| **Time range** | 2024-08-08 → 2026-08-08 (~2 years) |
| **Granularity** | Hourly |
| **Total rows** | 17,544 |
| **Missing values** | **Zero** across all 17 columns |

### Columns (17 total)

| Column | Dtype | Role |
|---|---|---|
| `time` | datetime | Timestamp |
| `temperature_2m` | float64 | Weather |
| `relative_humidity_2m` | int64 | Weather |
| `precipitation` | float64 | Weather |
| `pressure_msl` | float64 | Weather |
| `wind_speed_10m` | float64 | Weather |
| `wind_direction_10m` | int64 | Weather |
| `latitude` / `longitude` | float64 | Constant (dropped later) |
| `location_name` | str | Constant ("Karachi") |
| `pm10`, `pm2_5` | float64 | Pollutant |
| `carbon_monoxide` | float64 | Pollutant |
| `nitrogen_dioxide` | float64 | Pollutant |
| `sulphur_dioxide` | float64 | Pollutant |
| `ozone` | float64 | Pollutant |
| `us_aqi` | int64 | **Target** |

### Target Variable (`us_aqi`) Summary

| Stat | Value |
|---|---|
| Mean | 87.99 |
| Std | 21.95 |
| Min | 41 |
| 25 % | 73 |
| Median | 82 |
| 75 % | 98 |
| Max | 173 |

> [!IMPORTANT]
> The AQI distribution is **right-skewed**: the bulk of readings sit in the 60–100 "Moderate" band, but there is a long tail reaching 173 ("Unhealthy"). This skew has implications for model choice (e.g., log-transform target, or use tree-based models that handle skew naturally).

---

## 2. Train / Test Split

An **80/20 time-based** split was used (no random shuffle — correct for time series):

| Set | Rows | Start | End |
|---|---|---|---|
| Train | 14,035 | 2024-08-08 00:00 | 2026-03-15 18:00 |
| Test | 3,509 | 2026-03-15 19:00 | 2026-08-08 23:00 |

> [!TIP]
> Good practice: the split avoids data leakage from future observations into training.

---

## 3. Chart-by-Chart Interpretation

### Chart 1 — Feature Histograms (all numeric columns)

![Feature histograms for all numeric columns](C:/Users/abdul/.gemini/antigravity-ide/brain/5dc352e1-e755-41d9-94a1-b2184742aec0/images/1.png)

**Key observations:**

- **`temperature_2m`**: Roughly bimodal, peak around 28–32 °C. Karachi's warm climate is evident; cooler winter months create a secondary mode near 18–22 °C.
- **`relative_humidity_2m`**: Wide spread (5–100 %). Very roughly bimodal — dry winter vs. humid monsoon season.
- **`precipitation`**: Extremely right-skewed. The vast majority of hours have **zero** rainfall (mean = 0.03 mm). Rare monsoon bursts create a long tail up to 16.5 mm.
- **`pressure_msl`**: Approximately normal, centered ~1008 hPa with a range of ~32 hPa. Seasonal pressure variation visible.
- **`wind_speed_10m`**: Right-skewed, mode around 7–10 km/h, tail up to ~47 km/h. Calm hours are common.
- **`wind_direction_10m`**: Multi-modal with a dominant mode around 240–270° (WSW–W), reflecting Karachi's prevailing sea-breeze direction.
- **`pm10`**: Heavily right-skewed; most values < 100 µg/m³ but extreme spikes reach 504.
- **`pm2_5`**: Also right-skewed; majority < 40 µg/m³, outliers to ~109.
- **`carbon_monoxide`**: **Very heavily right-skewed** (mean 510, max 4,515). This distribution suggests occasional severe pollution events.
- **`nitrogen_dioxide`**: Right-skewed, mode near 5–15 µg/m³.
- **`sulphur_dioxide`**: Right-skewed, concentrated around 5–20 µg/m³.
- **`ozone`**: Broader spread, roughly uniform in 30–150 µg/m³, then tails off.
- **`us_aqi`** (target): Right-skewed, mode ~75–80, long tail to 173.

> [!NOTE]
> Most pollutant features are heavily right-skewed with occasional extreme spikes. **Log transforms** or **robust scaling** should be considered during feature engineering to handle outliers.

---

### Chart 2 — Hourly US AQI Time Series (2024-08 to 2026-03)

![Hourly US AQI time series for Karachi](C:/Users/abdul/.gemini/antigravity-ide/brain/5dc352e1-e755-41d9-94a1-b2184742aec0/images/2.png)

**Key observations:**

- **Clear seasonality**: AQI tends to be higher and more volatile during the **winter months** (Oct–Feb) and lower / calmer during the **monsoon season** (Jun–Sep). This makes sense — monsoon rains wash out particulates, while winter inversions trap pollution.
- **Typical range**: Most readings oscillate between 60–120, with spikes reaching 150–173.
- **No obvious long-term trend** (upward or downward) across the 2-year window.

> [!IMPORTANT]
> The strong seasonal cycle justifies including `month` as a feature. The high-frequency oscillations also suggest **hour-of-day** effects, validated in the next chart.

---

### Chart 3 — Individual Pollutant Time Series (sub-plots)

![Individual pollutant time series](C:/Users/abdul/.gemini/antigravity-ide/brain/5dc352e1-e755-41d9-94a1-b2184742aec0/images/3.png)

**Key observations (per pollutant):**

- **`pm10`**: Shows several dramatic spikes (up to ~300+), especially in winter months. Baseline is relatively low in summer.
- **`pm2_5`**: Mirrors PM10 pattern but at a smaller scale. Winter peaks are clear.
- **`carbon_monoxide`**: Highly episodic with extreme bursts up to ~2,500+ µg/m³. These events are concentrated in the **winter season** and appear correlated with PM spikes — likely from heating, industrial activity, or temperature inversions.
- **`nitrogen_dioxide`**: Moderate day-to-day variation with winter peaks.
- **`sulphur_dioxide`**: Relatively stable around 5–20, occasional spikes to ~50–65.
- **`ozone`**: Fairly stable with a slight summer increase (photochemical production is enhanced by sunlight).

> [!TIP]
> The co-occurrence of PM10, PM2.5, CO, and NO₂ spikes in winter suggests **multicollinearity** among pollutants. Consider PCA or feature selection to avoid redundancy.

---

### Chart 4 — Average AQI by Hour of Day

![Average AQI by hour of day](C:/Users/abdul/.gemini/antigravity-ide/brain/5dc352e1-e755-41d9-94a1-b2184742aec0/images/4.png)

**Key observations:**

- AQI is **relatively flat** from midnight to ~4 PM (~89).
- A **sharp spike** begins at **4–5 PM**, peaking around **7–8 PM** at ~95.7.
- Rapid drop-off from 8 PM to midnight.

**Interpretation:** The evening peak (5–8 PM) aligns with:
1. **Rush-hour traffic** (vehicular emissions peak)
2. **Atmospheric boundary layer collapse** at sunset — pollutants get trapped near the surface as the mixing layer shrinks.

> [!IMPORTANT]
> The `hour` feature is **highly informative** and should be included in the model. This distinct diurnal pattern is a strong signal for AQI prediction.

---

### Chart 5 — Average AQI by Day of Week

![Average AQI by day of week](C:/Users/abdul/.gemini/antigravity-ide/brain/5dc352e1-e755-41d9-94a1-b2184742aec0/images/5.png)

**Key observations:**

- All days cluster tightly between ~88–91 AQI. The difference between the highest (Tuesday ~91) and lowest (Thursday ~88) is only ~3 points.
- **No meaningful day-of-week effect** is visible.

> [!NOTE]
> `day_of_week` appears to be a **weak feature**. It may add noise rather than signal if included in the model. Consider dropping it or using it only as a secondary feature.

---

### Chart 6 — Numeric Feature Correlation Heatmap

![Numeric feature correlation heatmap](C:/Users/abdul/.gemini/antigravity-ide/brain/5dc352e1-e755-41d9-94a1-b2184742aec0/images/6.png)

**Correlations with `us_aqi` (actual values from notebook):**

| Feature | Correlation with `us_aqi` | Strength |
|---|---|---|
| `aqi_rolling_6h` | **0.960** (very strong positive) | 🟢 Top feature |
| `pm2_5` | **0.722** (strong positive) | 🟢 Key pollutant |
| `sulphur_dioxide` | **0.508** (moderate-strong positive) | 🟢 Useful |
| `pressure_msl` | **0.443** (moderate positive) | 🟢 Useful |
| `carbon_monoxide` | **0.439** (moderate positive) | 🟢 Useful |
| `nitrogen_dioxide` | **0.342** (moderate positive) | 🟡 Moderate |
| `pm10` | **0.226** (weak positive) | 🟡 Moderate |
| `ozone` | **0.142** (weak positive) | 🟡 Weak |
| `aqi_change_1h` | **0.068** (near zero) | 🔴 Very weak |
| `hour` | **0.043** (near zero) | 🔴 Very weak (linear) |
| `month` | **0.006** (near zero) | 🔴 Near zero (linear) |
| `precipitation` | **-0.044** (near zero) | 🔴 Near zero |
| `wind_direction_10m` | **-0.271** (weak negative) | 🟡 Moderate |
| `relative_humidity_2m` | **-0.332** (moderate negative) | 🟡 Moderate |
| `temperature_2m` | **-0.349** (moderate negative) | 🟡 Moderate |
| `wind_speed_10m` | **-0.405** (moderate negative) | 🟢 Useful (inverse) |

> [!WARNING]
> **`hour` and `month` show near-zero *linear* correlations** despite visually obvious patterns in Charts 4 and the time series. This is because their effects are **cyclical/non-linear** (e.g., hour 19 is high, hour 12 is low — a linear correlation misses this). They remain valuable features, especially with cyclical encoding (sin/cos transforms) or in tree-based models.

**Inter-feature correlations of note:**
- **`pm2_5` ↔ `pm10`** and **`carbon_monoxide` ↔ `nitrogen_dioxide`**: Strong positive (pollutant cluster)
- **`temperature_2m` ↔ `pressure_msl`**: Moderate negative (seasonal)
- **`aqi_rolling_6h` ↔ `us_aqi`**: Dominant correlation at **0.96** — the 6h rolling mean captures most of the AQI signal.
- **`wind_speed_10m`**: Notable **negative** correlation (-0.40) — higher wind disperses pollutants, lowering AQI.
- **`temperature_2m`** and **`relative_humidity_2m`**: Both **negatively** correlated (~-0.35, -0.33) — warmer, more humid conditions (monsoon) correspond to cleaner air.

> [!IMPORTANT]
> The **6-hour rolling AQI** (0.96) dominates all other features. While this is excellent for short-term forecasting, be careful: for 3-day-ahead predictions (the project goal), this lag feature won't be available. The model will need to rely more heavily on **PM2.5** (0.72), **sulphur dioxide** (0.51), **weather variables** (pressure, wind, temperature, humidity), and **forecast weather data** from Open-Meteo.

---

### Chart 7 — Scatter Matrix (key features vs us_aqi)

![Scatter matrix of key features](C:/Users/abdul/.gemini/antigravity-ide/brain/5dc352e1-e755-41d9-94a1-b2184742aec0/images/7.png)

**Key observations:**

- **`us_aqi` vs `pm2_5`**: Clear positive linear-ish relationship, strongest visual signal. Higher PM2.5 → higher AQI (expected, as PM2.5 is a primary AQI driver).
- **`us_aqi` vs `pm10`**: Also positive but noisier, with a "wedge" shape (high PM10 values can correspond to moderate or high AQI).
- **`us_aqi` vs `ozone`**: Cloud-like with slight negative trend; no strong visual signal.
- **`us_aqi` vs `temperature_2m`**: No clear linear relationship. However, extreme heat + low AQI OR extreme cold + high AQI patterns are faintly visible (seasonal effect).
- **`us_aqi` vs `relative_humidity_2m`**: Very scattered, near zero correlation.
- **`us_aqi` vs `wind_speed_10m`**: Near zero correlation; wind speed alone doesn't linearly predict AQI.

> [!NOTE]
> Non-linear relationships (e.g., temperature and wind speed interactions) may still be captured by tree-based models even if linear correlations are weak.

---

### Chart 8 — AQI vs PM2.5, colored by Hour

![AQI vs PM2.5 colored by hour](C:/Users/abdul/.gemini/antigravity-ide/brain/5dc352e1-e755-41d9-94a1-b2184742aec0/images/8.png)

**Key observations:**

- Clear **positive trend** between PM2.5 and AQI, confirming PM2.5 is a dominant driver.
- The **color gradient** (hour) reveals a pattern: the **higher-AQI, higher-PM2.5** region (upper-right) is populated more by **evening hours (16–20, shown in green/yellow)**, consistent with the evening AQI peak seen in Chart 4.
- **Morning/night hours (0–8, shown in purple/dark blue)** tend to cluster in the **lower-left** (lower AQI, lower PM2.5).

> [!TIP]
> This confirms that `hour` interacts with `pm2_5` in predicting AQI. Feature interactions or time-aware models could exploit this.

---

## 4. Feature Engineering Recommendations

The notebook concludes with these recommendations for [build_features.py](file:///c:/MyWorks/aqi-predictor/src/aqi_predictor/features/build_features.py):

| Feature | Rationale |
|---|---|
| `hour`, `month` | Strong diurnal & seasonal patterns |
| `day_of_week` | Weak effect — include cautiously |
| `aqi_change_1h` | Captures momentum / trend direction |
| `aqi_rolling_6h` (shifted by 1h) | Strongest correlated feature; avoids leakage |
| 3h, 24h rolling pollutant averages | Capture multi-scale temporal patterns |
| Lagged `us_aqi`, `pm2_5`, `pm10`, weather vars | Time-series features for autoregressive models |
| Hazardous AQI alert flag | Binary indicator for downstream alerting |

> [!WARNING]
> The rolling mean is correctly shifted by 1 hour to avoid data leakage. **Ensure all lag/rolling features follow this practice** in the production pipeline.

---

## 5. Overall Assessment & Next Steps

### Strengths of this EDA
- Clean, well-structured notebook following Geron's methodology
- Proper time-based train/test split (no data leakage from random shuffle)
- Good coverage of distribution analysis, time-series visualization, correlation analysis, and feature interaction plots
- Zero missing values simplifies preprocessing
- Feature engineering ideas are grounded in the visual patterns observed

### Areas to Consider for Next Steps
1. **Outlier handling**: Carbon monoxide has extreme spikes (max 4,515 vs. mean 510). Decide whether to clip, log-transform, or flag these.
2. **Feature selection**: Given multicollinearity among pollutants, consider PCA or variance inflation factor analysis.
3. **Stationarity testing**: Run ADF/KPSS tests on `us_aqi` to guide model selection (ARIMA-family vs. ML).
4. **Cross-validation strategy**: Use time-series CV (e.g., `TimeSeriesSplit`) rather than k-fold during training.
5. **Explore non-linear interactions**: The scatter matrix shows patterns that linear models may miss.
