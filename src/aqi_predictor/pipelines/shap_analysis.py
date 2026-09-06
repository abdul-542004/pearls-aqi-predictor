"""
SHAP feature importance analysis for the direct multi-horizon XGBoost models.

Runs SHAP on direct multi-horizon XGBoost models (+1h, +6h, +12h, +24h, +48h, +72h)
to understand what drives short-term, diurnal, and multi-day AQI predictions.
Automatically checks for staleness against trained model timestamps and refreshes
any outdated or missing SHAP reports.

Run:
    python -m aqi_predictor.pipelines.shap_analysis
    python -m aqi_predictor.pipelines.shap_analysis --force
    python -m aqi_predictor.pipelines.shap_analysis --horizons 1 6 12 24 48 72
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

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

# Default to all direct forecasting horizons
DEFAULT_SHAP_HORIZONS = HORIZONS  # [1, 6, 12, 24, 48, 72]


def is_horizon_stale(horizon: int, output_dir: Path = OUTPUT_DIR, artifacts_dir: Path = ARTIFACTS_DIR) -> bool:
    """
    Check whether SHAP outputs for a horizon are missing or older than the trained model.

    Returns True if:
      - JSON importance report or plots are missing.
      - Model exists and its modification time is newer than the JSON report.
    """
    json_path = output_dir / f"shap_importance_{horizon}h.json"
    bar_path = output_dir / f"shap_importance_{horizon}h.png"
    summary_path = output_dir / f"shap_summary_{horizon}h.png"

    if not (json_path.exists() and bar_path.exists() and summary_path.exists()):
        return True

    model_path = artifacts_dir / f"xgboost_{horizon}h" / "model.pkl"
    if not model_path.exists():
        return False

    model_mtime = model_path.stat().st_mtime
    json_mtime = json_path.stat().st_mtime
    return model_mtime > json_mtime


def load_xgboost_model(horizon: int, artifacts_dir: Path = ARTIFACTS_DIR):
    """Load the trained XGBoost model for a specific horizon."""
    model_dir = artifacts_dir / f"xgboost_{horizon}h"
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


def get_feature_color(feat: str) -> str:
    """Return color palette code based on feature category."""
    if feat.startswith("us_aqi_lag"):
        return "#2196F3"  # Blue - AQI lags
    if any(feat.startswith(p) for p in ["aqi_rolling", "aqi_std", "aqi_min", "aqi_max", "aqi_ewm"]):
        return "#4CAF50"  # Green - AQI stats
    if any(feat.startswith(p) for p in ["aqi_change", "aqi_trend"]):
        return "#FF9800"  # Orange - trends
    if feat.startswith("forecast_") or feat.startswith("delta_"):
        return "#9C27B0"  # Purple - forecast weather & deltas
    if any(p in feat for p in ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]):
        return "#F44336"  # Red - pollutants
    return "#607D8B"  # Slate Grey - others


def run_shap_for_horizon(model, X_test, feature_cols, horizon: int, output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Run SHAP analysis for a single horizon model and generate plots/artifacts."""
    import shap

    tag = f"+{horizon}h"
    print(f"\n{'='*65}")
    print(f"  SHAP Analysis for {tag} XGBoost Model")
    print(f"{'='*65}")

    explainer = shap.TreeExplainer(model)

    # Use a sample for speed (500 rows is representative for global feature importance)
    sample_size = min(500, len(X_test))
    X_sample = X_test[:sample_size]

    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP values (feature importance)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print(f"\n  Top 15 features by SHAP importance ({tag}):")
    print(f"  {'Feature':>35s}  {'|SHAP|':>8s}")
    print(f"  {'-'*50}")
    for _, row in importance_df.head(15).iterrows():
        bar_len = int(row["mean_abs_shap"] / max(importance_df["mean_abs_shap"].max(), 1e-6) * 30)
        bar = "#" * bar_len
        print(f"  {row['feature']:>35s}  {row['mean_abs_shap']:8.4f}  {bar}")

    # Check for raw pollutant leakage
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

    total_importance = max(importance_df["mean_abs_shap"].sum(), 1e-6)
    print(f"\n  Feature category breakdown ({tag}):")
    for cat_name, cat_feats in categories.items():
        cat_importance = importance_df[importance_df["feature"].isin(cat_feats)]["mean_abs_shap"].sum()
        pct = cat_importance / total_importance * 100
        bar = "#" * int(pct / 2)
        print(f"    {cat_name:>25s}: {pct:5.1f}%  {bar}")

    # Output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save bar plot
    top_n = min(25, len(importance_df))
    top_df = importance_df.head(top_n)
    colors = [get_feature_color(f) for f in top_df["feature"]]

    fig, ax = plt.subplots(figsize=(10, max(7, top_n * 0.32)))
    ax.barh(range(top_n), top_df["mean_abs_shap"].values, color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_df["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (AQI Points)")
    ax.set_title(f"XGBoost Feature Importance ({tag} Horizon)\nBlue=AQI Lags, Green=AQI Stats, Orange=Trends, Purple=Forecast Weather/Deltas")
    plt.tight_layout()
    bar_path = output_dir / f"shap_importance_{horizon}h.png"
    plt.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"  Bar plot saved -> {bar_path}")

    # 2. Save SHAP beeswarm summary plot
    fig2 = plt.figure(figsize=(10, max(7, top_n * 0.32)))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False, max_display=20)
    summary_path = output_dir / f"shap_summary_{horizon}h.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150)
    plt.close("all")
    print(f"  Summary plot saved -> {summary_path}")

    # 3. Save raw importance data
    importance_path = output_dir / f"shap_importance_{horizon}h.json"
    importance_df.to_json(importance_path, orient="records", indent=2)
    print(f"  Importance data saved -> {importance_path}")

    # Backward compatibility aliases for +1h primary horizon
    if horizon == 1:
        shutil.copyfile(importance_path, output_dir / "shap_importance.json")
        shutil.copyfile(bar_path, output_dir / "shap_feature_importance.png")
        shutil.copyfile(summary_path, output_dir / "shap_summary.png")

    return importance_df


def generate_cross_horizon_evolution(all_importance: Dict[int, pd.DataFrame], output_dir: Path = OUTPUT_DIR):
    """
    Generate cross-horizon feature importance progression chart showing how
    drivers transition from short-term lag persistence to diurnal cycles and
    long-term meteorological dynamics across all horizons.
    """
    if len(all_importance) < 2:
        return

    horizons = sorted(all_importance.keys())
    tag_list = [f"+{h}h" for h in horizons]

    # Calculate normalized percentage importance per horizon
    norm_dfs = {}
    for h in horizons:
        df_h = all_importance[h].copy()
        tot = max(df_h["mean_abs_shap"].sum(), 1e-6)
        df_h["pct"] = df_h["mean_abs_shap"] / tot * 100
        norm_dfs[h] = df_h.set_index("feature")["pct"]

    combined_df = pd.DataFrame(norm_dfs).fillna(0)
    combined_df.columns = tag_list

    # Select top 10 features with highest average importance across all horizons
    combined_df["mean_pct"] = combined_df.mean(axis=1)
    top_features = combined_df.sort_values("mean_pct", ascending=False).head(10).drop(columns=["mean_pct"])

    fig, ax = plt.subplots(figsize=(11, 6))
    markers = ["o", "s", "^", "D", "v", "p", "*", "h", "x", "d"]
    for idx, (feat, row) in enumerate(top_features.iterrows()):
        color = get_feature_color(feat)
        marker = markers[idx % len(markers)]
        ax.plot(tag_list, row.values, label=feat, marker=marker, linewidth=2, color=color)

    ax.set_title("Cross-Horizon SHAP Feature Importance Evolution (+1h to +72h)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Forecast Horizon", fontsize=11)
    ax.set_ylabel("Share of Total SHAP Importance (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8.5, framealpha=0.9)
    plt.tight_layout()

    evolution_path = output_dir / "shap_cross_horizon_evolution.png"
    plt.savefig(evolution_path, dpi=150)
    plt.close(fig)
    print(f"\n  Cross-horizon evolution plot saved -> {evolution_path}")


def run_shap_analysis(
    horizons: Optional[List[int]] = None,
    force: bool = False,
    output_dir: Path = OUTPUT_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Dict[int, pd.DataFrame]:
    """
    Run SHAP analysis for given horizons, refreshing stale or missing reports.
    """
    if horizons is None:
        horizons = DEFAULT_SHAP_HORIZONS

    print("Checking staleness for horizons:", horizons)
    horizons_to_run = []
    for h in horizons:
        stale = is_horizon_stale(h, output_dir, artifacts_dir)
        if force or stale:
            reason = "forced" if force else ("report missing or older than model" if stale else "up to date")
            print(f"  Horizon +{h}h: NEEDS RUN ({reason})")
            horizons_to_run.append(h)
        else:
            print(f"  Horizon +{h}h: UP-TO-DATE (skipping, use --force to refresh)")

    all_importance: Dict[int, pd.DataFrame] = {}

    # Load existing up-to-date reports into all_importance so we can still do comparison
    for h in horizons:
        if h not in horizons_to_run:
            json_file = output_dir / f"shap_importance_{h}h.json"
            if json_file.exists():
                try:
                    all_importance[h] = pd.read_json(json_file)
                except Exception:
                    horizons_to_run.append(h)

    if not horizons_to_run:
        print("\nAll requested SHAP reports are already up-to-date! Nothing to recalculate.")
        if len(all_importance) >= 2:
            generate_cross_horizon_evolution(all_importance, output_dir)
        return all_importance

    # 1. Fetch and prepare data from Hopsworks
    print("\nConnecting to Hopsworks Feature Store...")
    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)
    df = fg.read()
    print(f"  Fetched {len(df)} rows from Hopsworks")

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

    # 2. Run SHAP for each horizon that needs computation
    for h in horizons_to_run:
        print(f"\n--- Loading XGBoost +{h}h model ---")
        try:
            model = load_xgboost_model(h, artifacts_dir)
        except FileNotFoundError as e:
            print(f"  Skipping horizon +{h}h: {e}")
            continue

        # Add forecast weather features for this horizon
        h_test, forecast_cols = add_forecast_weather_features(test_df, h)
        all_cols = sorted(base_feature_cols + forecast_cols)
        feature_cols = getattr(model, "_expected_features", get_horizon_feature_cols(all_cols, h))

        X_test = h_test[feature_cols].values.astype(np.float64)
        print(f"  Test set: {len(h_test)} rows, {len(feature_cols)} features")

        importance_df = run_shap_for_horizon(
            model, X_test, feature_cols, h, output_dir,
        )
        all_importance[h] = importance_df

    # 3. Cross-horizon evolution chart and comparisons
    if len(all_importance) >= 2:
        generate_cross_horizon_evolution(all_importance, output_dir)

    print(f"\nSHAP analysis complete. Outputs available in {output_dir}")
    return all_importance


def main():
    parser = argparse.ArgumentParser(description="Run SHAP feature importance for XGBoost models.")
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=DEFAULT_SHAP_HORIZONS,
        help=f"List of forecast horizons in hours (default: {DEFAULT_SHAP_HORIZONS})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of SHAP analysis even if reports are up-to-date",
    )
    args = parser.parse_args()

    run_shap_analysis(horizons=args.horizons, force=args.force)


if __name__ == "__main__":
    main()
