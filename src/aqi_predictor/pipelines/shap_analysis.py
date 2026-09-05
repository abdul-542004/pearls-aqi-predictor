"""
SHAP feature importance analysis for the direct multi-horizon XGBoost models.

Runs SHAP on the +1h and +24h XGBoost models to understand what drives
short-term vs long-term AQI predictions with the improved feature set.

Run:
    python -m aqi_predictor.pipelines.shap_analysis
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(PROJECT_ROOT / ".env")

from aqi_predictor.features.build_features import build_features
from aqi_predictor.features.hopsworks_utils import (
    get_feature_store,
    get_or_create_feature_group,
)
from aqi_predictor.pipelines.training_pipeline import (
    HORIZONS, TARGET, DROP_COLS, WEATHER_COLS,
    prepare_targets, split_data, add_forecast_weather_features,
    get_horizon_feature_cols,
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "models"
OUTPUT_DIR = PROJECT_ROOT / "reports"

# Horizons to analyze with SHAP (short-term vs long-term comparison)
SHAP_HORIZONS = [1, 24]


def load_xgboost_model(horizon):
    """Load the trained XGBoost model for a specific horizon."""
    model_dir = ARTIFACTS_DIR / f"xgboost_{horizon}h"
    model_path = model_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}")
    model = joblib.load(model_path)
    feat_path = model_dir / "feature_names.json"
    if feat_path.exists():
        try:
            with open(feat_path, "r") as f:
                model._expected_features = json.load(f)
        except Exception:
            pass
    return model


def run_shap_for_horizon(model, X_test, feature_cols, horizon, output_dir):
    """Run SHAP analysis for a single horizon model."""
    import shap

    tag = f"+{horizon}h"
    print(f"\n{'='*60}")
    print(f"  SHAP Analysis for {tag} XGBoost Model")
    print(f"{'='*60}")

    explainer = shap.TreeExplainer(model)

    # Use a sample for speed (500 rows is plenty for importance)
    sample_size = min(500, len(X_test))
    X_sample = X_test[:sample_size]

    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP values (feature importance)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print(f"\n  Top 20 features by SHAP importance ({tag}):")
    print(f"  {'Feature':>35s}  {'|SHAP|':>8s}")
    print(f"  {'-'*50}")
    for _, row in importance_df.head(20).iterrows():
        bar_len = int(row["mean_abs_shap"] / importance_df["mean_abs_shap"].max() * 30)
        bar = "#" * bar_len
        print(f"  {row['feature']:>35s}  {row['mean_abs_shap']:8.4f}  {bar}")

    # Check for pollutant leakage (should be clean now)
    raw_pollutants = {"pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"}
    pollutant_feats = importance_df[importance_df["feature"].isin(raw_pollutants)]
    if len(pollutant_feats) > 0:
        total = importance_df["mean_abs_shap"].sum()
        pollutant_pct = pollutant_feats["mean_abs_shap"].sum() / total * 100
        print(f"\n  Raw pollutant features: {pollutant_pct:.1f}% of total importance")
        if pollutant_pct > 30:
            print("  [WARNING] Raw pollutants still dominate -- check for leakage")
    else:
        print(f"\n  [OK] No raw pollutant features in model -- leakage fix confirmed")

    # Category breakdown: what types of features matter?
    categories = {
        "AQI Lags": [f for f in feature_cols if f.startswith("us_aqi_lag")],
        "AQI Rolling/Stats": [f for f in feature_cols if any(f.startswith(p) for p in ["aqi_rolling", "aqi_std", "aqi_min", "aqi_max", "aqi_ewm"])],
        "AQI Trend/Change": [f for f in feature_cols if any(f.startswith(p) for p in ["aqi_change", "aqi_trend"])],
        "Pollutant Lags/Rolling": [f for f in feature_cols if any(p in f for p in ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "co_rolling"])],
        "Weather (current)": [f for f in feature_cols if f in {"temperature_2m", "relative_humidity_2m", "precipitation", "pressure_msl", "wind_speed_10m", "wind_direction_10m", "wind_u", "wind_v", "temp_humidity", "precip_last_6h", "had_rain_24h"}],
        "Weather (forecast/delta)": [f for f in feature_cols if f.startswith("forecast_") or f.startswith("delta_")],
        "Time Features": [f for f in feature_cols if any(f.startswith(p) for p in ["hour_", "month_", "dow_", "is_weekend"])],
    }

    total_importance = importance_df["mean_abs_shap"].sum()
    print(f"\n  Feature category importance breakdown:")
    for cat_name, cat_feats in categories.items():
        cat_importance = importance_df[importance_df["feature"].isin(cat_feats)]["mean_abs_shap"].sum()
        pct = cat_importance / total_importance * 100
        bar = "#" * int(pct / 2)
        print(f"    {cat_name:>25s}: {pct:5.1f}%  {bar}")

    # Save bar plot
    output_dir.mkdir(parents=True, exist_ok=True)

    # Color by category
    def get_color(feat):
        if feat.startswith("us_aqi_lag"): return "#2196F3"  # Blue - AQI lags
        if any(feat.startswith(p) for p in ["aqi_rolling", "aqi_std", "aqi_min", "aqi_max", "aqi_ewm"]): return "#4CAF50"  # Green - AQI stats
        if any(feat.startswith(p) for p in ["aqi_change", "aqi_trend"]): return "#FF9800"  # Orange - trends
        if feat.startswith("forecast_") or feat.startswith("delta_"): return "#9C27B0"  # Purple - forecast weather & deltas
        if any(p in feat for p in ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]): return "#F44336"  # Red - pollutants
        return "#607D8B"  # Grey - everything else

    top_n = min(30, len(importance_df))
    top_df = importance_df.head(top_n)
    colors = [get_color(f) for f in top_df["feature"]]

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
    ax.barh(range(top_n), top_df["mean_abs_shap"].values, color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_df["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature Importance ({tag}) — Blue=AQI lags, Green=AQI stats, Orange=Trends, Purple=Forecast weather & deltas")
    plt.tight_layout()
    bar_path = output_dir / f"shap_importance_{horizon}h.png"
    plt.savefig(bar_path, dpi=150)
    print(f"\n  Bar plot saved -> {bar_path}")

    # SHAP beeswarm plot
    fig2, ax2 = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False, max_display=20)
    summary_path = output_dir / f"shap_summary_{horizon}h.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150)
    print(f"  Summary plot saved -> {summary_path}")

    # Save raw importance data
    importance_path = output_dir / f"shap_importance_{horizon}h.json"
    importance_df.to_json(importance_path, orient="records", indent=2)
    print(f"  Importance data saved -> {importance_path}")

    plt.close("all")
    return importance_df


def main():
    # 1. Fetch and prepare data (same as training pipeline)
    print("Connecting to Hopsworks...")
    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)
    df = fg.read()
    print(f"  Fetched {len(df)} rows")

    # Re-engineer features
    raw_pollutant_cols = {"pm2_5", "pm10", "carbon_monoxide",
                          "nitrogen_dioxide", "sulphur_dioxide", "ozone"}
    if bool(raw_pollutant_cols & set(df.columns)):
        print("Re-engineering features with improved feature set...")
        df = build_features(df, drop_raw_pollutants=True)
        print(f"  After re-engineering: {len(df)} rows, {len(df.columns)} columns")

    # Prepare targets and split
    df, target_cols = prepare_targets(df, HORIZONS)
    train_df, val_df, test_df = split_data(df)

    # Base feature columns
    all_target_cols = set(target_cols.values())
    base_feature_cols = sorted(set(df.columns) - DROP_COLS - all_target_cols)

    # 2. Run SHAP for each selected horizon
    all_importance = {}

    for h in SHAP_HORIZONS:
        print(f"\n--- Loading XGBoost +{h}h model ---")
        try:
            model = load_xgboost_model(h)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        # Add forecast weather features for this horizon
        h_test, forecast_cols = add_forecast_weather_features(test_df, h)
        all_cols = sorted(base_feature_cols + forecast_cols)
        feature_cols = getattr(model, "_expected_features", get_horizon_feature_cols(all_cols, h))

        X_test = h_test[feature_cols].values.astype(np.float64)
        print(f"  Test set: {len(h_test)} rows, {len(feature_cols)} features")

        importance_df = run_shap_for_horizon(
            model, X_test, feature_cols, h, OUTPUT_DIR,
        )
        all_importance[h] = importance_df

    # 3. Compare short-term vs long-term if both are available
    if 1 in all_importance and 24 in all_importance:
        print(f"\n{'='*60}")
        print(f"  Short-term vs Long-term Feature Importance Comparison")
        print(f"{'='*60}")

        imp_1h = all_importance[1].set_index("feature")["mean_abs_shap"]
        imp_24h = all_importance[24].set_index("feature")["mean_abs_shap"]

        # Normalize to percentages for fair comparison
        imp_1h_pct = (imp_1h / imp_1h.sum() * 100)
        imp_24h_pct = (imp_24h / imp_24h.sum() * 100)

        # Features that matter MORE for 24h vs 1h
        common_feats = imp_1h_pct.index.intersection(imp_24h_pct.index)
        comparison = pd.DataFrame({
            "+1h (%)": imp_1h_pct.reindex(common_feats).fillna(0),
            "+24h (%)": imp_24h_pct.reindex(common_feats).fillna(0),
        })
        comparison["shift"] = comparison["+24h (%)"] - comparison["+1h (%)"]
        comparison = comparison.sort_values("shift", ascending=False)

        print("\n  Features gaining importance at longer horizons:")
        for feat, row in comparison.head(10).iterrows():
            if row["shift"] > 0.1:
                print(f"    {feat:>35s}:  +1h={row['+1h (%)']:5.1f}%  +24h={row['+24h (%)']:5.1f}%  (shift: +{row['shift']:.1f}%)")

        print("\n  Features losing importance at longer horizons:")
        for feat, row in comparison.tail(10).iterrows():
            if row["shift"] < -0.1:
                print(f"    {feat:>35s}:  +1h={row['+1h (%)']:5.1f}%  +24h={row['+24h (%)']:5.1f}%  (shift: {row['shift']:.1f}%)")

    plt.close("all")
    print(f"\nSHAP analysis complete. All outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
