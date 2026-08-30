"""
Advanced EDA for AQI Prediction — Supplementary Analysis

Fills gaps identified in the project review:
  1. ADF stationarity test on us_aqi (raw + differenced)
  2. ACF / PACF plots — reveals useful lag horizons
  3. STL seasonal decomposition — trend + daily cycle + residual
  4. Outlier analysis — characterize AQI spikes

All plots saved to reports/.

Run:
    python notebooks/eda_advanced.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import STL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def load_data():
    """Load the raw Karachi AQI dataset."""
    csv_path = PROJECT_ROOT / "data" / "raw" / "open_meteo_karachi_2024-08-08_2026-08-08.csv"
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    print(f"Loaded {len(df)} rows  ({df['time'].min()} -> {df['time'].max()})")
    return df


# -- 1. Stationarity Test ---------------------------------------------------


def stationarity_test(series, name="us_aqi"):
    """Run Augmented Dickey-Fuller test and print results."""
    print(f"\n{'='*60}")
    print(f"  ADF Stationarity Test -- {name}")
    print(f"{'='*60}")

    result = adfuller(series.dropna(), autolag="AIC")
    labels = ["ADF Statistic", "p-value", "Lags Used", "Observations"]

    for label, val in zip(labels, result[:4]):
        print(f"  {label:>20s}: {val}")
    for key, val in result[4].items():
        print(f"  Critical Value ({key:>3s}): {val:.4f}")

    if result[1] < 0.05:
        print(f"\n  PASS: {name} IS stationary (p={result[1]:.6f} < 0.05)")
    else:
        print(f"\n  FAIL: {name} is NOT stationary (p={result[1]:.6f} >= 0.05)")

    return result


# -- 2. ACF / PACF Analysis -------------------------------------------------


def plot_acf_pacf(series, max_lags=168, save_path=None):
    """
    Plot ACF and PACF up to `max_lags` hours (default 168 = 7 days).

    This directly answers: "How many hours of lag are useful?"
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # ACF
    acf_vals = acf(series.dropna(), nlags=max_lags, fft=True)
    axes[0].bar(range(len(acf_vals)), acf_vals, width=0.6, color="#3498db", alpha=0.8)
    # 95% confidence interval
    ci = 1.96 / np.sqrt(len(series))
    axes[0].axhline(ci, color="red", linestyle="--", alpha=0.5, label="95% CI")
    axes[0].axhline(-ci, color="red", linestyle="--", alpha=0.5)
    axes[0].axhline(0, color="black", linewidth=0.5)
    # Mark key horizons
    for h, label in [(1, "+1h"), (6, "+6h"), (12, "+12h"), (24, "+24h"), (48, "+48h"), (72, "+72h")]:
        if h <= max_lags:
            axes[0].axvline(h, color="green", linestyle=":", alpha=0.5)
            axes[0].text(h, acf_vals[h] + 0.02, f"{label}\n{acf_vals[h]:.3f}",
                        ha="center", fontsize=8, color="green")
    axes[0].set_title("Autocorrelation Function (ACF) -- How correlated is AQI with its past?")
    axes[0].set_xlabel("Lag (hours)")
    axes[0].set_ylabel("ACF")
    axes[0].legend()

    # PACF
    pacf_vals = pacf(series.dropna(), nlags=min(max_lags, len(series) // 2 - 1))
    axes[1].bar(range(len(pacf_vals)), pacf_vals, width=0.6, color="#e74c3c", alpha=0.8)
    axes[1].axhline(ci, color="blue", linestyle="--", alpha=0.5, label="95% CI")
    axes[1].axhline(-ci, color="blue", linestyle="--", alpha=0.5)
    axes[1].axhline(0, color="black", linewidth=0.5)
    for h, label in [(1, "+1h"), (6, "+6h"), (12, "+12h"), (24, "+24h"), (48, "+48h"), (72, "+72h")]:
        if h < len(pacf_vals):
            axes[1].axvline(h, color="green", linestyle=":", alpha=0.5)
            axes[1].text(h, pacf_vals[h] + 0.02, f"{label}\n{pacf_vals[h]:.3f}",
                        ha="center", fontsize=8, color="green")
    axes[1].set_title("Partial Autocorrelation Function (PACF) -- Direct influence of each lag")
    axes[1].set_xlabel("Lag (hours)")
    axes[1].set_ylabel("PACF")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved ACF/PACF plot -> {save_path}")
    plt.close()

    return acf_vals, pacf_vals


def analyze_acf_for_lags(acf_vals, pacf_vals, series_len):
    """Analyze ACF/PACF values and recommend whether longer lags are useful."""
    ci = 1.96 / np.sqrt(series_len)

    print(f"\n{'='*60}")
    print(f"  ACF/PACF Analysis -- Lag Usefulness")
    print(f"{'='*60}")
    print(f"  95% CI threshold: +/-{ci:.4f}")
    print()

    horizons = [1, 3, 6, 12, 24, 48, 72]
    print(f"  {'Lag':>6s}  {'ACF':>8s}  {'PACF':>8s}  {'ACF sig?':>10s}  {'PACF sig?':>10s}")
    print(f"  {'-'*50}")
    for h in horizons:
        acf_val = acf_vals[h] if h < len(acf_vals) else float("nan")
        pacf_val = pacf_vals[h] if h < len(pacf_vals) else float("nan")
        acf_sig = "YES" if abs(acf_val) > ci else "no"
        pacf_sig = "YES" if abs(pacf_val) > ci else "no"
        print(f"  {h:>4d}h  {acf_val:>8.4f}  {pacf_val:>8.4f}  {acf_sig:>10s}  {pacf_sig:>10s}")

    # Recommendations
    print(f"\n  Recommendations:")
    if len(acf_vals) > 48 and abs(acf_vals[48]) > ci:
        print(f"  + 48h lag shows significant ACF ({acf_vals[48]:.4f}) -- ADD us_aqi_lag_48h")
    else:
        print(f"  - 48h lag ACF is weak -- adding lag_48h may not help much")

    if len(acf_vals) > 72 and abs(acf_vals[72]) > ci:
        print(f"  + 72h lag shows significant ACF ({acf_vals[72]:.4f}) -- ADD us_aqi_lag_72h")
    else:
        print(f"  - 72h lag ACF is weak -- adding lag_72h may not help much")

    # Check for 24h periodicity (daily cycle)
    if len(acf_vals) > 24 and acf_vals[24] > acf_vals[23] and acf_vals[24] > acf_vals[25]:
        print(f"  + 24h lag shows a local ACF peak ({acf_vals[24]:.4f}) -- confirms daily cycle")


# -- 3. Seasonal Decomposition ----------------------------------------------


def seasonal_decomposition(series, period=24, save_path=None):
    """
    STL decomposition with 24-hour period (daily cycle).
    Shows: trend + seasonal (daily pattern) + residual.
    """
    print(f"\n{'='*60}")
    print(f"  STL Seasonal Decomposition (period={period}h)")
    print(f"{'='*60}")

    stl = STL(series.dropna(), period=period, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(series.values, linewidth=0.3, color="#3498db")
    axes[0].set_ylabel("Observed")
    axes[0].set_title("STL Decomposition of Hourly US AQI (24h period)")

    axes[1].plot(result.trend.values, linewidth=0.8, color="#e74c3c")
    axes[1].set_ylabel("Trend")

    axes[2].plot(result.seasonal.values, linewidth=0.3, color="#2ecc71")
    axes[2].set_ylabel("Seasonal (Daily)")

    axes[3].plot(result.resid.values, linewidth=0.3, color="#9b59b6", alpha=0.7)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Hour index")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved STL plot -> {save_path}")
    plt.close()

    # Variance decomposition
    total_var = series.dropna().var()
    trend_var = result.trend.var()
    seasonal_var = result.seasonal.var()
    resid_var = result.resid.dropna().var()

    print(f"  Total variance:    {total_var:.2f}")
    print(f"  Trend variance:    {trend_var:.2f} ({trend_var/total_var*100:.1f}%)")
    print(f"  Seasonal variance: {seasonal_var:.2f} ({seasonal_var/total_var*100:.1f}%)")
    print(f"  Residual variance: {resid_var:.2f} ({resid_var/total_var*100:.1f}%)")

    # Daily seasonal pattern
    daily_pattern = result.seasonal.values[:24]
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(range(24), daily_pattern, marker="o", color="#2ecc71", linewidth=2)
    ax2.set_title("Average Daily AQI Seasonal Pattern")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Seasonal Component (AQI deviation)")
    ax2.set_xticks(range(24))
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    daily_path = save_path.parent / "eda_daily_pattern.png" if save_path else None
    if daily_path:
        plt.savefig(daily_path, dpi=150)
        print(f"  Saved daily pattern plot -> {daily_path}")
    plt.close()

    return result


# -- 4. Outlier Analysis -----------------------------------------------------


def outlier_analysis(df, save_path=None):
    """Identify and characterize AQI outlier spikes."""
    aqi = df["us_aqi"]

    print(f"\n{'='*60}")
    print(f"  Outlier Analysis")
    print(f"{'='*60}")

    q1 = aqi.quantile(0.25)
    q3 = aqi.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[aqi > upper].copy()
    print(f"  IQR: {iqr:.1f}  (Q1={q1:.1f}, Q3={q3:.1f})")
    print(f"  Upper bound (Q3 + 1.5*IQR): {upper:.1f}")
    print(f"  Outlier rows: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    print(f"  Outlier AQI range: {outliers['us_aqi'].min():.0f} - {outliers['us_aqi'].max():.0f}")

    # Distribution of AQI categories
    categories = pd.cut(aqi, bins=[0, 50, 100, 150, 200, 300, 500],
                        labels=["Good", "Moderate", "Unhealthy-SG", "Unhealthy", "Very Unhealthy", "Hazardous"])
    cat_counts = categories.value_counts().sort_index()
    print(f"\n  AQI Category Distribution:")
    for cat, count in cat_counts.items():
        pct = count / len(df) * 100
        bar = "#" * int(pct)
        print(f"    {cat:>16s}: {count:>5d} ({pct:>5.1f}%) {bar}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot by hour
    df_plot = df.copy()
    df_plot["hour"] = df_plot["time"].dt.hour
    sns.boxplot(data=df_plot, x="hour", y="us_aqi", ax=axes[0],
                flierprops={"marker": ".", "markersize": 2, "alpha": 0.3})
    axes[0].set_title("AQI Distribution by Hour (outliers visible)")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("US AQI")

    # AQI category pie chart
    colors = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]
    cat_counts.plot.pie(ax=axes[1], colors=colors[:len(cat_counts)],
                        autopct="%1.1f%%", startangle=90)
    axes[1].set_title("AQI Category Distribution")
    axes[1].set_ylabel("")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved outlier plot -> {save_path}")
    plt.close()

    return outliers


# -- Main --------------------------------------------------------------------


def main():
    df = load_data()
    aqi = df["us_aqi"]

    print("\n" + "=" * 70)
    print("     ADVANCED EDA -- AQI TIME SERIES ANALYSIS")
    print("=" * 70)

    # 1. Stationarity
    stationarity_test(aqi, "us_aqi (raw)")
    stationarity_test(aqi.diff().dropna(), "us_aqi (1h differenced)")

    # 2. ACF / PACF
    acf_vals, pacf_vals = plot_acf_pacf(
        aqi, max_lags=168,
        save_path=REPORTS_DIR / "eda_acf_pacf.png",
    )
    analyze_acf_for_lags(acf_vals, pacf_vals, len(aqi))

    # 3. Seasonal Decomposition
    seasonal_decomposition(
        aqi, period=24,
        save_path=REPORTS_DIR / "eda_stl_decomposition.png",
    )

    # 4. Outlier Analysis
    outlier_analysis(df, save_path=REPORTS_DIR / "eda_outliers.png")

    print("\n" + "=" * 70)
    print("     ADVANCED EDA COMPLETE -- all plots saved to reports/")
    print("=" * 70)


if __name__ == "__main__":
    main()
