"""Stage 2 (DEFAULT engine) - DuckDB + scikit-learn batch job.

raw CSV --clean--> curated Parquet (hourly, daily) --SQL / window functions / ML--> results/summary.json

Why DuckDB: it is a columnar analytical database engine that runs inside Python (pip install duckdb).
It reads CSV and Parquet directly, runs full SQL with window functions on data larger than RAM, and
needs NO Java, NO Hadoop and NO winutils.exe, so it behaves the same on Windows, Linux and macOS.

Produces exactly the same results/summary.json structure as the Apache Spark job
(bigdata/spark_job.py), so the dashboard works with either engine.
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

from . import config

log = logging.getLogger("bigdata.duckdb")

RAW_COLUMNS = ["city", "country", "latitude", "longitude", "time",
               "temp_c", "humidity_pct", "precip_mm", "wind_kmh", "pressure_hpa", "cloud_pct"]
TEXT_COLUMNS = ("city", "country", "time")

# Physically plausible ranges; values outside become NULL (counted in the quality report).
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
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (dt.date, dt.datetime)):
        return str(v)
    return v


def _sql_path(path):
    """Path as a SQL string literal (single quotes doubled)."""
    return "'" + str(path).replace("'", "''") + "'"


def _clear_dir(path):
    """Delete an old curated folder; stop if files are locked (e.g. by OneDrive) rather than
    silently writing duplicate rows next to old files."""
    if not path.exists():
        return
    shutil.rmtree(path, ignore_errors=True)
    leftovers = list(path.rglob("*.parquet")) if path.exists() else []
    if leftovers:
        raise RuntimeError(f"Could not remove old Parquet files in '{path}' (for example "
                           f"'{leftovers[0].name}'). Close programs using them or pause OneDrive "
                           "sync for this folder, then run again.")


class _Db:
    """Tiny wrapper around a DuckDB connection."""

    def __init__(self):
        import duckdb
        self.con = duckdb.connect()  # in-memory database; data lives in Parquet files

    def one(self, sql, params=None):
        return self.con.execute(sql, params or []).fetchone()[0]

    def rows(self, sql, nd=2):
        cur = self.con.execute(sql)
        names = [d[0] for d in cur.description]
        return [{n: _clean(v, nd) for n, v in zip(names, r)} for r in cur.fetchall()]

    def df(self, sql):
        return self.con.execute(sql).df()

    def run(self, sql, params=None):
        self.con.execute(sql, params or [])


def _check_headers(files):
    """Clear error if a CSV lacks a required column (instead of a confusing SQL error)."""
    for f in files:
        with open(f, "r", encoding="utf-8-sig", newline="") as fh:
            header = [h.strip() for h in fh.readline().strip().split(",")]
        missing = [c for c in RAW_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"{Path(f).name} is missing required column(s): {', '.join(missing)}")


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

    import duckdb
    import sklearn

    started = time.time()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    summary = _pipeline(files, curated_dir, manifest)
    summary["run"].update({
        "engine": f"DuckDB {duckdb.__version__} + scikit-learn {sklearn.__version__}",
        "master": "in-process (single machine)",
        "started_at": started_at,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "duration_seconds": round(time.time() - started, 1),
    })
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote %s", results_dir / "summary.json")
    return summary


def _pipeline(files, curated_dir, manifest):
    db = _Db()

    # ---------- 1. READ raw zone ----------
    # Everything is read as text first; bad numbers become NULL (like Spark's PERMISSIVE mode).
    db.run("CREATE MACRO num(x) AS CASE WHEN isfinite(TRY_CAST(x AS DOUBLE)) THEN TRY_CAST(x AS DOUBLE) END")
    numeric = [c for c in RAW_COLUMNS if c not in TEXT_COLUMNS]
    db.run(f"""
        CREATE TABLE raw AS
        SELECT NULLIF(TRIM(city), '') AS city, NULLIF(TRIM(country), '') AS country,
               {', '.join(f'num({c}) AS {c}' for c in numeric)},
               TRY_STRPTIME(TRIM("time"), '%Y-%m-%dT%H:%M') AS ts
        FROM read_csv(?, header = true, all_varchar = true, delim = ',', quote = '"', escape = '"',
                      strict_mode = false)""", [files])  # explicit dialect: tolerates BOM / mixed line endings
    raw_rows = db.one("SELECT count(*) FROM raw")

    # ---------- 2. CLEAN ----------
    valid_rows = db.one("SELECT count(*) FROM raw WHERE city IS NOT NULL AND ts IS NOT NULL")
    invalid_rows = raw_rows - valid_rows
    db.run("""CREATE TABLE dedup AS
              SELECT * FROM raw WHERE city IS NOT NULL AND ts IS NOT NULL
              QUALIFY row_number() OVER (PARTITION BY city, ts) = 1""")
    parts = ["count(*) AS rows"]
    for c, (lo, hi) in BOUNDS.items():
        parts.append(f"count(*) FILTER (WHERE {c} IS NULL) AS {c}__missing")
        parts.append(f"count(*) FILTER (WHERE {c} IS NOT NULL AND NOT ({c} BETWEEN {lo} AND {hi})) AS {c}__out_of_range")
    names = ["rows"] + [f"{c}__{k}" for c in BOUNDS for k in ("missing", "out_of_range")]
    q = dict(zip(names, db.con.execute(f"SELECT {', '.join(parts)} FROM dedup").fetchone()))
    dedup_rows = q["rows"]

    cleaned_cols = ", ".join(f'CASE WHEN {c} BETWEEN {lo} AND {hi} THEN {c} END AS {c}'
                             for c, (lo, hi) in BOUNDS.items())
    db.run(f"""
        CREATE TABLE hourly AS
        SELECT * FROM (
            SELECT city, country, latitude, longitude, ts, {cleaned_cols},
                   CAST(ts AS DATE) AS "date", year(ts) AS year, month(ts) AS month, hour(ts) AS hour
            FROM dedup)
        WHERE temp_c IS NOT NULL""")
    clean_rows = db.one("SELECT count(*) FROM hourly")

    # ---------- 3. WRITE curated zone (columnar Parquet, partitioned) ----------
    _write_parquet(db, "hourly", curated_dir / "hourly", ["city", "year"])

    # ---------- 4. DAILY aggregates ----------
    db.run("""
        CREATE TABLE daily AS
        SELECT city, "date", count(*) AS hours,
               avg(temp_c) AS temp_mean, min(temp_c) AS temp_min, max(temp_c) AS temp_max,
               avg(humidity_pct) AS humidity_mean, sum(precip_mm) AS precip_total,
               avg(wind_kmh) AS wind_mean, max(wind_kmh) AS wind_max,
               avg(pressure_hpa) AS pressure_mean, avg(cloud_pct) AS cloud_mean,
               year("date") AS year, month("date") AS month
        FROM hourly GROUP BY city, "date" HAVING count(*) >= 20""")  # ignore partial days
    n_daily = db.one("SELECT count(*) FROM daily")
    _write_parquet(db, "daily", curated_dir / "daily", ["city"])

    # ---------- 5. SQL summaries ----------
    city_stats = db.rows("""
        SELECT city,
               ROUND(AVG(temp_mean), 2) AS avg_temp, ROUND(MAX(temp_max), 1) AS max_temp,
               ROUND(MIN(temp_min), 1) AS min_temp, ROUND(AVG(humidity_mean), 1) AS avg_humidity,
               ROUND(SUM(precip_total), 1) AS total_rain_mm,
               SUM(CASE WHEN precip_total >= 1 THEN 1 ELSE 0 END) AS rainy_days,
               COUNT(*) AS days
        FROM daily GROUP BY city ORDER BY city""")
    monthly = db.rows("""
        SELECT city, year, month,
               ROUND(AVG(temp_mean), 2) AS avg_temp, ROUND(MAX(temp_max), 1) AS max_temp,
               ROUND(MIN(temp_min), 1) AS min_temp, ROUND(AVG(humidity_mean), 1) AS avg_humidity,
               ROUND(SUM(precip_total), 1) AS rain_mm, ROUND(AVG(wind_mean), 1) AS avg_wind
        FROM daily GROUP BY city, year, month ORDER BY city, year, month""")

    # ---------- 6. WINDOW functions: rolling mean + seasonal anomalies ----------
    recent_daily = db.rows("""
        SELECT city, "date", temp_mean, roll7 FROM (
            SELECT city, "date", temp_mean,
                   avg(temp_mean) OVER (PARTITION BY city ORDER BY "date" ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS roll7,
                   row_number() OVER (PARTITION BY city ORDER BY "date" DESC) AS rn
            FROM daily) WHERE rn <= 60
        ORDER BY city, "date" """)
    anomalies = db.rows(f"""
        SELECT city, "date", temp_mean, clim_mean, z FROM (
            SELECT city, "date", temp_mean,
                   avg(temp_mean) OVER w AS clim_mean,
                   (temp_mean - avg(temp_mean) OVER w) / stddev_samp(temp_mean) OVER w AS z
            FROM daily WINDOW w AS (PARTITION BY city, month))
        WHERE abs(z) >= {ANOMALY_Z} ORDER BY abs(z) DESC, city, "date" LIMIT 15""")
    extreme_rain = db.rows("""
        SELECT city, "date", precip_total FROM daily
        WHERE precip_total IS NOT NULL ORDER BY precip_total DESC, city, "date" LIMIT 10""")

    # ---------- 7. CORRELATIONS across all cities ----------
    cols = ["temp_mean", "humidity_mean", "precip_total", "wind_mean", "pressure_mean", "cloud_mean"]
    pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]]
    corr_vals = db.con.execute("SELECT " + ", ".join(f"corr({a}, {b})" for a, b in pairs)
                               + " FROM daily").fetchone()
    correlations = [{"x": a, "y": b, "r": _clean(v)} for (a, b), v in zip(pairs, corr_vals)]

    # ---------- 8. Machine learning: next-day mean temperature ----------
    ml = _forecast_model(db)

    first, last = db.con.execute('SELECT min("date"), max("date") FROM hourly').fetchone()
    return {
        "run": {
            "data_mode": manifest.get("mode", "unknown"),
            "data_source": manifest.get("source", "unknown"),
            "data_start": str(first), "data_end": str(last),
            "cities": sorted(r["city"] for r in city_stats),
            "raw_rows": raw_rows, "clean_rows": clean_rows, "daily_rows": n_daily,
            "parquet_writer": "DuckDB partitioned Parquet writer",
            "raw_reader": "DuckDB CSV reader",
        },
        "quality": {
            "raw_rows": raw_rows, "invalid_timestamp_or_city_rows": invalid_rows,
            "duplicates_removed": valid_rows - dedup_rows,
            "rows_after_dedup": dedup_rows,
            "missing_values": {c: q[f"{c}__missing"] for c in BOUNDS},
            "out_of_range_set_to_null": {c: q[f"{c}__out_of_range"] for c in BOUNDS},
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
            "Aggregations ran as DuckDB SQL queries over the cleaned data in the data lake.",
            "Model estimates are statistical, not official forecasts; follow your national weather service for warnings.",
        ] + (["DATA IS SYNTHETIC: generated for demonstration, not real observations."]
             if manifest.get("mode") == "synthetic" else []),
    }


def _write_parquet(db, table, path, partition_cols):
    _clear_dir(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db.run(f"COPY {table} TO {_sql_path(path)} "
           f"(FORMAT PARQUET, PARTITION_BY ({', '.join(partition_cols)}), OVERWRITE_OR_IGNORE)")


def _score(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {"rmse": _clean(math.sqrt(mean_squared_error(y_true, y_pred)), 3),
            "mae": _clean(mean_absolute_error(y_true, y_pred), 3),
            "r2": _clean(r2_score(y_true, y_pred), 3)}


def _forecast_model(db):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    df = db.df("""
        WITH w AS (
            SELECT city, "date", temp_mean, humidity_mean, pressure_mean, precip_total, cloud_mean, wind_mean,
                   lag(temp_mean, 1) OVER o AS lag1, lag(temp_mean, 2) OVER o AS lag2,
                   avg(temp_mean) OVER (PARTITION BY city ORDER BY "date" ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS roll7,
                   lead("date", 1) OVER o AS next_d, lead(temp_mean, 1) OVER o AS next_t
            FROM daily WINDOW o AS (PARTITION BY city ORDER BY "date"))
        SELECT *,
               CASE WHEN date_diff('day', "date", next_d) = 1 THEN next_t END AS target,
               sin(2 * pi() * dayofyear("date" + INTERVAL 1 DAY) / 365.25) AS doy_sin,
               cos(2 * pi() * dayofyear("date" + INTERVAL 1 DAY) / 365.25) AS doy_cos
        FROM w""")
    labeled = df.dropna(subset=FEATURES + ["target"])
    if labeled.empty:
        return {"status": "skipped", "reason": "not enough data"}
    cutoff = labeled["date"].max() - np.timedelta64(TEST_DAYS, "D")
    train, test = labeled[labeled["date"] <= cutoff], labeled[labeled["date"] > cutoff]
    n_train, n_test = len(train), len(test)
    if n_train < 100 or n_test < 10:
        return {"status": "skipped", "reason": f"not enough data (train={n_train}, test={n_test})"}

    x_train, y_train = train[FEATURES].to_numpy(float), train["target"].to_numpy(float)
    x_test, y_test = test[FEATURES].to_numpy(float), test["target"].to_numpy(float)
    metrics = {"persistence_baseline": _score(y_test, test["temp_mean"].to_numpy(float))}  # "same as today"
    lr = LinearRegression().fit(x_train, y_train)
    metrics["linear_regression"] = _score(y_test, lr.predict(x_test))
    rf = RandomForestRegressor(n_estimators=40, max_depth=8, random_state=42).fit(x_train, y_train)
    metrics["random_forest"] = _score(y_test, rf.predict(x_test))

    best_name = min(("linear_regression", "random_forest"), key=lambda k: metrics[k]["rmse"])
    best = lr if best_name == "linear_regression" else rf

    latest = df.sort_values("date").groupby("city").tail(1).dropna(subset=FEATURES).sort_values("city")
    preds = best.predict(latest[FEATURES].to_numpy(float)) if len(latest) else []
    estimates = [{"city": r.city, "as_of_date": str(r.date.date()),
                  "last_mean_temp_c": _clean(r.temp_mean), "predicted_next_day_mean_temp_c": _clean(p)}
                 for r, p in zip(latest.itertuples(), preds)]

    importance = sorted(zip(FEATURES, rf.feature_importances_), key=lambda t: -t[1])
    return {
        "status": "ok", "target": "next-day mean temperature (deg C)",
        "train_rows": n_train, "test_rows": n_test, "test_days": TEST_DAYS,
        "features": FEATURES, "metrics": metrics, "best_model": best_name,
        "feature_importance": [{"feature": f, "importance": _clean(float(i), 3)} for f, i in importance],
        "next_day_estimates": estimates,
        "note": "Time-based split: the last 60 days are held out for testing. "
                "Estimates are for the day after 'as_of_date' (the last day in the dataset).",
    }


def main(argv=None):
    argparse.ArgumentParser(description="Run the DuckDB batch job on the data lake").parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = run_job()
    print(json.dumps(s["run"], indent=2))


if __name__ == "__main__":
    main()
