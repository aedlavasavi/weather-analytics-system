"""Weather analytics with pandas / numpy.

Observed data (past hours up to the current time) is kept separate from
forecast data (hours after the current time), as the system prompt requires.
"""
import math

import numpy as np
import pandas as pd


def _clean(value):
    """Make numpy/pandas values JSON-safe (NaN -> None)."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if (math.isnan(value) or math.isinf(value)) else round(float(value), 2)
    return value


def build_hourly_df(payload):
    """Turn the API 'hourly' block into a DataFrame and tag each row observed/forecast."""
    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(
        columns={
            "temperature_2m": "temp_c",
            "relative_humidity_2m": "humidity_pct",
            "precipitation": "precip_mm",
            "wind_speed_10m": "wind_kmh",
            "pressure_msl": "pressure_hpa",
        }
    )
    cutoff = pd.to_datetime(payload["current"]["time"])
    df["kind"] = np.where(df["time"] <= cutoff, "observed", "forecast")
    df["date"] = df["time"].dt.date.astype(str)
    return df


def daily_summary(df):
    """Aggregate hourly rows to daily values."""
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby("date").agg(
        temp_mean=("temp_c", "mean"),
        temp_min=("temp_c", "min"),
        temp_max=("temp_c", "max"),
        humidity_mean=("humidity_pct", "mean"),
        precip_total=("precip_mm", "sum"),
        wind_mean=("wind_kmh", "mean"),
        wind_max=("wind_kmh", "max"),
        pressure_mean=("pressure_hpa", "mean"),
    )
    return grouped.reset_index()


def trend_per_day(series):
    """Linear trend (units per day) using least squares. None if too little data."""
    s = series.dropna()
    if len(s) < 3:
        return None
    slope, _ = np.polyfit(np.arange(len(s)), s.values, 1)
    return float(slope)


def detect_anomalies(daily, column="temp_mean", threshold=2.0):
    """Flag days whose value is more than `threshold` std-devs from the period mean."""
    s = daily[column].dropna()
    if len(s) < 5 or s.std(ddof=0) == 0:
        return []
    z = (daily[column] - s.mean()) / s.std(ddof=0)
    flagged = daily[z.abs() > threshold]
    return [
        {"date": row["date"], "value": row[column], "z_score": z.loc[idx]}
        for idx, row in flagged.iterrows()
    ]


def baseline_forecast(daily_observed, horizon=3, holdout=5):
    """Simple linear-trend baseline for mean temperature with a hold-out MAE.

    This is an illustrative statistical baseline, NOT a replacement for the
    provider's forecast or official meteorological forecasts.
    """
    s = daily_observed["temp_mean"].dropna().reset_index(drop=True)
    if len(s) < holdout + 5:
        return None
    train, test = s.iloc[:-holdout], s.iloc[-holdout:]
    slope, intercept = np.polyfit(np.arange(len(train)), train.values, 1)
    pred_test = intercept + slope * np.arange(len(train), len(train) + holdout)
    mae = float(np.mean(np.abs(pred_test - test.values)))

    slope, intercept = np.polyfit(np.arange(len(s)), s.values, 1)
    future_x = np.arange(len(s), len(s) + horizon)
    preds = intercept + slope * future_x
    return {"method": "linear trend on daily mean temperature", "holdout_mae_c": mae,
            "predictions_c": [float(p) for p in preds]}


def run_analytics(payload):
    """Build the complete analytics report for one location."""
    df = build_hourly_df(payload)
    observed = df[df["kind"] == "observed"]
    forecast = df[df["kind"] == "forecast"]
    daily_obs = daily_summary(observed)
    daily_fc = daily_summary(forecast)

    # Data quality checks
    quality = {
        "observed_hours": int(len(observed)),
        "forecast_hours": int(len(forecast)),
        "missing_values": int(observed[["temp_c", "humidity_pct", "precip_mm",
                                        "wind_kmh", "pressure_hpa"]].isna().sum().sum()),
    }

    stats = {}
    if not observed.empty:
        for col, label in [("temp_c", "temperature_c"), ("humidity_pct", "humidity_pct"),
                           ("wind_kmh", "wind_kmh"), ("pressure_hpa", "pressure_hpa")]:
            stats[label] = {
                "mean": observed[col].mean(), "min": observed[col].min(),
                "max": observed[col].max(), "std": observed[col].std(),
            }
        stats["rainfall_mm"] = {
            "total": observed["precip_mm"].sum(),
            "rainy_days": int((daily_obs["precip_total"] >= 1.0).sum()) if not daily_obs.empty else 0,
        }

    corr = {}
    num_cols = ["temp_c", "humidity_pct", "wind_kmh", "pressure_hpa", "precip_mm"]
    if len(observed) >= 10:
        corr_df = observed[num_cols].corr()
        corr = {a: {b: corr_df.loc[a, b] for b in num_cols} for a in num_cols}

    report = {
        "source": "Open-Meteo (open-meteo.com)",
        "timezone": payload.get("timezone"),
        "observed_until": payload["current"]["time"],
        "quality": quality,
        "stats": stats,
        "temperature_trend_c_per_day": trend_per_day(daily_obs["temp_mean"]) if not daily_obs.empty else None,
        "correlations": corr,
        "anomalies": {
            "temperature": detect_anomalies(daily_obs, "temp_mean") if not daily_obs.empty else [],
            "rainfall": detect_anomalies(daily_obs, "precip_total") if not daily_obs.empty else [],
        },
        "daily_observed": daily_obs.to_dict(orient="records"),
        "daily_forecast": daily_fc.to_dict(orient="records"),
        "baseline_forecast": baseline_forecast(daily_obs) if not daily_obs.empty else None,
        "scatter": observed[["temp_c", "humidity_pct"]].dropna().iloc[::3].to_dict(orient="records"),
        "notes": [
            "Observed values are model-assimilated data for the location, not a station reading.",
            "Forecasts are estimates; accuracy falls as the horizon grows.",
            "This tool does not replace official weather warnings.",
        ],
    }
    return _clean(report)
