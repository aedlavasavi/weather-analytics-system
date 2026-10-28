"""Stage 2 (DEFAULT engine) - pandas batch job. Needs only pandas + numpy.

raw CSV --clean--> curated Parquet (hourly, daily) --aggregations / window functions / ML--> results/summary.json

Why this is the default: it has NO Java, NO Hadoop, NO winutils and NO special native database library.
Some Windows PCs have an "Application Control" / Smart App Control policy that blocks unsigned
compiled modules (error: "DLL load failed ... An Application Control policy has blocked this file").
pandas and numpy are normally allowed, so this engine runs where DuckDB or Spark cannot.

Optional parts degrade instead of crashing, and the result says exactly what was used:
  * Parquet files need pyarrow; if it cannot be loaded the curated zone is written as gzip CSV.
  * scikit-learn is used for the models if it loads; otherwise a numpy least-squares model is used.

Produces exactly the same results/summary.json structure as the DuckDB and Spark jobs.
"""
import argparse
import datetime as dt
import json
import logging
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger("bigdata.pandas")

RAW_COLUMNS = ["city", "country", "latitude", "longitude", "time",
               "temp_c", "humidity_pct", "precip_mm", "wind_kmh", "pressure_hpa", "cloud_pct"]
TEXT_COLUMNS = ("city", "country", "time")
NUMERIC = [c for c in RAW_COLUMNS if c not in TEXT_COLUMNS]

# Physically plausible ranges; values outside become missing (counted in the quality report).
BOUNDS = {"temp_c": (-60, 60), "humidity_pct": (0, 100), "precip_mm": (0, 500),
          "wind_kmh": (0, 400), "pressure_hpa": (850, 1100), "cloud_pct": (0, 100)}

ANOMALY_Z = 3.0
TEST_DAYS = 60
FEATURES = ["temp_mean", "lag1", "lag2", "roll7", "humidity_mean", "pressure_mean",
            "precip_total", "cloud_mean", "wind_mean", "doy_sin", "doy_cos"]


# ----------------------------------------------------------------------------- helpers
def _clean(v, nd=2):
    """Make a value JSON-safe: NaN/inf -> None, floats rounded, dates -> text."""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, nd)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, pd.Timestamp):
        return str(v.date())
    if isinstance(v, dt.datetime):
        return str(v.date())
    if isinstance(v, dt.date):
        return str(v)
    if v is pd.NaT:
        return None
    return v


def _rows(df, columns, nd=2):
    return [{c: _clean(v, nd) for c, v in zip(columns, r)} for r in df[columns].itertuples(index=False, name=None)]


def _clear_dir(path):
    """Delete an old curated folder; stop if files are locked (e.g. by OneDrive) rather than
    silently writing duplicate rows next to old files."""
    if not path.exists():
        return
    import gc
    gc.collect()
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        time.sleep(0.05)
    leftovers = [p for p in path.rglob("*") if p.is_file()] if path.exists() else []
    if leftovers:
        time.sleep(0.1)
        shutil.rmtree(path, ignore_errors=True)
        leftovers = [p for p in path.rglob("*") if p.is_file()] if path.exists() else []
    if leftovers:
        raise RuntimeError(f"Could not remove old data files in '{path}' (for example "
                           f"'{leftovers[0].name}'). Close programs using them or pause OneDrive "
                           "sync for this folder, then run again.")


def _check_headers(files):
    """Clear error if a CSV lacks a required column."""
    for f in files:
        with open(f, "r", encoding="utf-8-sig", newline="") as fh:
            header = [h.strip() for h in fh.readline().strip().split(",")]
        missing = [c for c in RAW_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"{Path(f).name} is missing required column(s): {', '.join(missing)}")


def _versions():
    try:
        import sklearn
        ml = f"scikit-learn {sklearn.__version__}"
    except Exception:  # noqa: BLE001  (ImportError, or a DLL blocked by Windows policy)
        ml = "numpy least-squares model (scikit-learn unavailable)"
    return f"pandas {pd.__version__} + {ml}"


# ----------------------------------------------------------------------------- public API
def run_job(raw_dir=None, curated_dir=None, results_dir=None):
    raw_dir = Path(raw_dir or config.raw_dir())
    curated_dir = Path(curated_dir or config.curated_dir())
    results_dir = Path(results_dir or config.results_dir())
    files = sorted(str(p) for p in raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {raw_dir}. Run the ingestion step first.")
    manifest_path = raw_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    _check_headers(files)

    started = time.time()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    summary = _pipeline(files, curated_dir, manifest)
    summary["run"].update({
        "engine": _versions(),
        "master": "in-process (single machine)",
        "started_at": started_at,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "duration_seconds": round(time.time() - started, 1),
    })
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote %s", results_dir / "summary.json")
    return summary


def _read_raw(files):
    """Read every CSV as text, then convert: bad numbers/timestamps become missing (never crash)."""
    frames = []
    for f in files:
        frames.append(pd.read_csv(f, dtype=str, encoding="utf-8-sig", keep_default_na=False,
                                  usecols=lambda c: c.strip() in RAW_COLUMNS, skipinitialspace=True))
    raw = pd.concat(frames, ignore_index=True)
    raw.columns = [c.strip() for c in raw.columns]
    for c in ("city", "country"):
        raw[c] = raw[c].str.strip().replace("", np.nan)
    raw["ts"] = pd.to_datetime(raw["time"].str.strip(), format="%Y-%m-%dT%H:%M", errors="coerce")
    for c in NUMERIC:
        v = pd.to_numeric(raw[c].str.strip(), errors="coerce")
        raw[c] = v.where(np.isfinite(v))
    return raw.drop(columns=["time"])


def _pipeline(files, curated_dir, manifest):
    # ---------- 1. READ raw zone ----------
    raw = _read_raw(files)
    raw_rows = len(raw)

    # ---------- 2. CLEAN ----------
    valid = raw[raw["city"].notna() & raw["ts"].notna()]
    valid_rows = len(valid)
    invalid_rows = raw_rows - valid_rows
    dedup = valid.drop_duplicates(subset=["city", "ts"], keep="first")
    dedup_rows = len(dedup)

    missing, out_of_range = {}, {}
    hourly = dedup.copy()
    for c, (lo, hi) in BOUNDS.items():
        col = hourly[c]
        missing[c] = int(col.isna().sum())
        ok = col.between(lo, hi)
        out_of_range[c] = int((col.notna() & ~ok).sum())
        hourly[c] = col.where(ok)
    hourly = hourly[hourly["temp_c"].notna()].copy()
    clean_rows = len(hourly)

    hourly["date"] = hourly["ts"].dt.normalize()
    hourly["year"] = hourly["ts"].dt.year.astype("int32")
    hourly["month"] = hourly["ts"].dt.month.astype("int32")
    hourly["hour"] = hourly["ts"].dt.hour.astype("int32")

    # ---------- 3. WRITE curated zone ----------
    writer = _write_table(hourly, curated_dir / "hourly", ["city", "year"])

    # ---------- 4. DAILY aggregates ----------
    g = hourly.groupby(["city", "date"], sort=True)
    daily = g.agg(hours=("temp_c", "size"),
                  temp_mean=("temp_c", "mean"), temp_min=("temp_c", "min"), temp_max=("temp_c", "max"),
                  humidity_mean=("humidity_pct", "mean"),
                  precip_total=("precip_mm", lambda s: s.sum(min_count=1)),
                  wind_mean=("wind_kmh", "mean"), wind_max=("wind_kmh", "max"),
                  pressure_mean=("pressure_hpa", "mean"), cloud_mean=("cloud_pct", "mean")).reset_index()
    daily = daily[daily["hours"] >= 20].reset_index(drop=True)  # ignore partial days
    daily["year"] = daily["date"].dt.year.astype("int32")
    daily["month"] = daily["date"].dt.month.astype("int32")
    n_daily = len(daily)
    _write_table(daily, curated_dir / "daily", ["city"])

    # ---------- 5. Summaries ----------
    cs = daily.groupby("city").agg(
        avg_temp=("temp_mean", "mean"), max_temp=("temp_max", "max"), min_temp=("temp_min", "min"),
        avg_humidity=("humidity_mean", "mean"), total_rain_mm=("precip_total", "sum"),
        rainy_days=("precip_total", lambda s: int((s >= 1).sum())), days=("temp_mean", "size")).reset_index()
    cs = cs.sort_values("city")
    for c, nd in (("avg_temp", 2), ("max_temp", 1), ("min_temp", 1), ("avg_humidity", 1), ("total_rain_mm", 1)):
        cs[c] = cs[c].round(nd)
    city_stats = _rows(cs, ["city", "avg_temp", "max_temp", "min_temp", "avg_humidity",
                            "total_rain_mm", "rainy_days", "days"], nd=2)

    mo = daily.groupby(["city", "year", "month"]).agg(
        avg_temp=("temp_mean", "mean"), max_temp=("temp_max", "max"), min_temp=("temp_min", "min"),
        avg_humidity=("humidity_mean", "mean"), rain_mm=("precip_total", "sum"),
        avg_wind=("wind_mean", "mean")).reset_index().sort_values(["city", "year", "month"])
    for c, nd in (("avg_temp", 2), ("max_temp", 1), ("min_temp", 1), ("avg_humidity", 1), ("rain_mm", 1), ("avg_wind", 1)):
        mo[c] = mo[c].round(nd)
    monthly = _rows(mo, ["city", "year", "month", "avg_temp", "max_temp", "min_temp",
                         "avg_humidity", "rain_mm", "avg_wind"])

    # ---------- 6. Window-style calculations: rolling mean + seasonal anomalies ----------
    daily = daily.sort_values(["city", "date"]).reset_index(drop=True)
    daily["roll7"] = daily.groupby("city")["temp_mean"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    recent = daily.groupby("city", group_keys=False).tail(60).sort_values(["city", "date"])
    recent_daily = _rows(recent, ["city", "date", "temp_mean", "roll7"])

    grp = daily.groupby(["city", "month"])["temp_mean"]
    daily["clim_mean"] = grp.transform("mean")
    sd = grp.transform("std")  # sample standard deviation, like SQL stddev_samp
    daily["z"] = (daily["temp_mean"] - daily["clim_mean"]) / sd.where(sd > 0)
    an = daily[daily["z"].abs() >= ANOMALY_Z].copy()
    an["abs_z"] = an["z"].abs()
    an = an.sort_values(["abs_z", "city", "date"], ascending=[False, True, True]).head(15)
    anomalies = _rows(an, ["city", "date", "temp_mean", "clim_mean", "z"])

    rain = daily[daily["precip_total"].notna()].sort_values(
        ["precip_total", "city", "date"], ascending=[False, True, True]).head(10)
    extreme_rain = _rows(rain, ["city", "date", "precip_total"])

    # ---------- 7. Correlations ----------
    cols = ["temp_mean", "humidity_mean", "precip_total", "wind_mean", "pressure_mean", "cloud_mean"]
    corr = daily[cols].corr()
    correlations = [{"x": a, "y": b, "r": _clean(corr.loc[a, b])}
                    for i, a in enumerate(cols) for b in cols[i + 1:]]

    # ---------- 8. Machine learning ----------
    ml = _forecast_model(daily)

    return {
        "run": {
            "data_mode": manifest.get("mode", "unknown"),
            "data_source": manifest.get("source", "unknown"),
            "data_start": str(hourly["date"].min().date()), "data_end": str(hourly["date"].max().date()),
            "cities": sorted(r["city"] for r in city_stats),
            "raw_rows": raw_rows, "clean_rows": clean_rows, "daily_rows": n_daily,
            "parquet_writer": writer,
            "raw_reader": "pandas CSV reader",
        },
        "quality": {
            "raw_rows": raw_rows, "invalid_timestamp_or_city_rows": invalid_rows,
            "duplicates_removed": valid_rows - dedup_rows,
            "rows_after_dedup": dedup_rows,
            "missing_values": missing,
            "out_of_range_set_to_null": out_of_range,
            "rows_dropped_no_temperature": dedup_rows - clean_rows,
            "clean_rows": clean_rows,
        },
        "city_stats": city_stats, "monthly": monthly, "recent_daily": recent_daily,
        "correlations": correlations,
        "anomalies": anomalies,
        "anomaly_rule": f"|z| >= {ANOMALY_Z} versus the same city and calendar month",
        "extreme_rain_days": extreme_rain,
        "ml": ml,
        "notes": [
            "Aggregations ran with pandas (group-by and rolling-window operations) over the cleaned data in the data lake.",
            "Model estimates are statistical, not official forecasts; follow your national weather service for warnings.",
        ] + (["DATA IS SYNTHETIC: generated for demonstration, not real observations."]
             if manifest.get("mode") == "synthetic" else [])
          + (["Parquet support (pyarrow) could not be loaded on this PC, so the curated zone was written as gzip CSV."]
             if "CSV" in writer else []),
    }


def _write_table(df, path, partition_cols):
    """Partitioned Parquet via pyarrow. Only if pyarrow itself cannot be loaded (for example a Windows
    Application Control policy blocks its DLL) the same partitions are written as gzip CSV. Any other
    error is raised, not hidden."""
    _clear_dir(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        log.warning("pyarrow unavailable (%s); writing gzip CSV instead of Parquet", exc)
        for keys, part in df.groupby(partition_cols, sort=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            folder = path.joinpath(*[f"{c}={k}" for c, k in zip(partition_cols, keys)])
            folder.mkdir(parents=True, exist_ok=True)
            part.drop(columns=partition_cols).to_csv(folder / "part-0.csv.gz", index=False, compression="gzip")
        return "pandas gzip-CSV writer (pyarrow unavailable)"
    df.to_parquet(path, engine="pyarrow", partition_cols=partition_cols, index=False)
    return "pandas + PyArrow partitioned Parquet writer"


# ----------------------------------------------------------------------------- machine learning
def _metrics(y_true, y_pred):
    err = np.asarray(y_pred, float) - np.asarray(y_true, float)
    y = np.asarray(y_true, float)
    ss_res, ss_tot = float((err ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return {"rmse": _clean(math.sqrt(float((err ** 2).mean())), 3),
            "mae": _clean(float(np.abs(err).mean()), 3),
            "r2": _clean(1 - ss_res / ss_tot if ss_tot else float("nan"), 3)}


class _NumpyLinear:
    """Least-squares linear regression on standardised features (used if scikit-learn cannot load)."""

    def fit(self, x, y):
        self.mu, self.sd = x.mean(axis=0), x.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        a = np.column_stack([np.ones(len(x)), (x - self.mu) / self.sd])
        self.coef = np.linalg.lstsq(a, y, rcond=None)[0]
        return self

    def predict(self, x):
        return self.coef[0] + ((x - self.mu) / self.sd) @ self.coef[1:]

    def importance(self):
        w = np.abs(self.coef[1:])
        return w / w.sum() if w.sum() else w


def _forecast_model(daily):
    d = daily.sort_values(["city", "date"]).copy()
    g = d.groupby("city")
    d["lag1"], d["lag2"] = g["temp_mean"].shift(1), g["temp_mean"].shift(2)
    next_d, next_t = g["date"].shift(-1), g["temp_mean"].shift(-1)
    d["target"] = next_t.where((next_d - d["date"]) == pd.Timedelta(days=1))
    doy = (d["date"] + pd.Timedelta(days=1)).dt.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    labeled = d.dropna(subset=FEATURES + ["target"])
    if labeled.empty:
        return {"status": "skipped", "reason": "not enough data"}
    cutoff = labeled["date"].max() - pd.Timedelta(days=TEST_DAYS)
    train, test = labeled[labeled["date"] <= cutoff], labeled[labeled["date"] > cutoff]
    n_train, n_test = len(train), len(test)
    if n_train < 100 or n_test < 10:
        return {"status": "skipped", "reason": f"not enough data (train={n_train}, test={n_test})"}

    x_train, y_train = train[FEATURES].to_numpy(float), train["target"].to_numpy(float)
    x_test, y_test = test[FEATURES].to_numpy(float), test["target"].to_numpy(float)
    metrics = {"persistence_baseline": _metrics(y_test, test["temp_mean"].to_numpy(float))}  # "same as today"

    models = {}
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        models["linear_regression"] = LinearRegression().fit(x_train, y_train)
        models["random_forest"] = RandomForestRegressor(n_estimators=40, max_depth=8, random_state=42).fit(x_train, y_train)
        importance = models["random_forest"].feature_importances_
    except ImportError as exc:  # scikit-learn (or its DLLs) cannot be loaded: use the numpy model
        log.warning("scikit-learn unavailable (%s); using the numpy linear model only", exc)
        lin = _NumpyLinear().fit(x_train, y_train)
        models = {"linear_regression": lin}
        importance = lin.importance()
    for name, m in models.items():
        metrics[name] = _metrics(y_test, m.predict(x_test))

    best_name = min(models, key=lambda k: metrics[k]["rmse"])
    best = models[best_name]
    latest = d.groupby("city").tail(1).dropna(subset=FEATURES).sort_values("city")
    preds = best.predict(latest[FEATURES].to_numpy(float)) if len(latest) else []
    estimates = [{"city": r.city, "as_of_date": str(r.date.date()),
                  "last_mean_temp_c": _clean(r.temp_mean), "predicted_next_day_mean_temp_c": _clean(p)}
                 for r, p in zip(latest.itertuples(), preds)]

    ranked = sorted(zip(FEATURES, importance), key=lambda t: -t[1])
    return {
        "status": "ok", "target": "next-day mean temperature (deg C)",
        "train_rows": n_train, "test_rows": n_test, "test_days": TEST_DAYS,
        "features": FEATURES, "metrics": metrics, "best_model": best_name,
        "feature_importance": [{"feature": f, "importance": _clean(float(i), 3)} for f, i in ranked],
        "next_day_estimates": estimates,
        "note": "Time-based split: the last 60 days are held out for testing. "
                "Estimates are for the day after 'as_of_date' (the last day in the dataset).",
    }


def main(argv=None):
    argparse.ArgumentParser(description="Run the pandas batch job on the data lake").parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = run_job()
    print(json.dumps(s["run"], indent=2))


if __name__ == "__main__":
    main()
