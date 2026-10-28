"""Stage 2 - Apache Spark batch job (PySpark).

raw CSV  --clean-->  curated Parquet (hourly, daily)  --Spark SQL / window functions /
MLlib-->  results/summary.json  (read by the Flask dashboard)

Spark concepts used: DataFrames, explicit schema, Spark SQL, window functions,
partitioned Parquet, aggregations, correlation, MLlib pipelines (regression).

WINDOWS NOTE (why there are two I/O modes)
------------------------------------------
Whenever Spark touches a LOCAL file path (glob a folder, list files, read or write Parquet) it goes
through Hadoop's local file system. On Windows that layer calls native code from
``hadoop.dll`` / ``winutils.exe`` (``NativeIO$Windows.access0``). If those files are missing, even
the very first ``spark.read.csv("folder/*.csv")`` dies with ``UnsatisfiedLinkError``. Config
switches such as ``spark.hadoop.io.native.lib.available=false`` do not help, because
``FileUtil.canRead`` calls the native method directly.

So this job has two I/O modes (environment variable ``SPARK_IO_MODE``):

  spark   Spark reads the CSV files and writes Parquet itself (default on Linux/macOS/Docker, and on
          Windows when HADOOP_HOME\\bin contains BOTH winutils.exe and hadoop.dll).
  driver  "Windows-safe" mode, used automatically on Windows without those native files. Spark never
          touches the local file system: raw CSV files are read with pandas and handed to Spark as an
          Arrow table using the explicit RAW_SCHEMA, and Parquet files are written with pyarrow.
          Cleaning, deduplication, Spark SQL, window functions, correlations and MLlib all still
          run in Spark. Limitation: raw files and curated tables pass through the driver's memory,
          which is fine for millions of rows but not for data larger than the driver's RAM.

  auto    (default) picks the mode as described above.

A fallback to driver mode also happens if Spark's own I/O fails with a KNOWN Windows Hadoop native
error. Any other error is raised unchanged.
"""
import argparse
import datetime as dt
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path

from pyspark import StorageLevel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from . import config
from .storage import get_store

log = logging.getLogger("bigdata.spark")

RAW_SCHEMA = StructType([
    StructField("city", StringType()), StructField("country", StringType()),
    StructField("latitude", DoubleType()), StructField("longitude", DoubleType()),
    StructField("time", StringType()),
    StructField("temp_c", DoubleType()), StructField("humidity_pct", DoubleType()),
    StructField("precip_mm", DoubleType()), StructField("wind_kmh", DoubleType()),
    StructField("pressure_hpa", DoubleType()), StructField("cloud_pct", DoubleType()),
])

# Physically plausible ranges; values outside become NULL (counted in the quality report).
BOUNDS = {"temp_c": (-60, 60), "humidity_pct": (0, 100), "precip_mm": (0, 500),
          "wind_kmh": (0, 400), "pressure_hpa": (850, 1100), "cloud_pct": (0, 100)}

ANOMALY_Z = 3.0
TEST_DAYS = 60
FEATURES = ["temp_mean", "lag1", "lag2", "roll7", "humidity_mean", "pressure_mean",
            "precip_total", "cloud_mean", "wind_mean", "doy_sin", "doy_cos"]


def build_spark(app_name="WeatherBigData"):
    # Make sure Spark starts the SAME Python as this script (on Windows the bare "python"
    # alias can point at the Microsoft Store stub).
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    builder = (SparkSession.builder
             .master(os.getenv("SPARK_MASTER", "local[*]"))
             .appName(app_name)
             .config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "8"))
             .config("spark.sql.session.timeZone", "UTC"))

    # S3 / Hadoop AWS credentials if provided (for environments using hadoop-aws / s3a)
    ak = (os.getenv("AWS_ACCESS_KEY_ID") or "").strip()
    sk = (os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip()
    st = (os.getenv("AWS_SESSION_TOKEN") or "").strip()
    ep = (os.getenv("S3_ENDPOINT_URL") or "").strip()
    if ak and sk:
        builder = builder.config("spark.hadoop.fs.s3a.access.key", ak)
        builder = builder.config("spark.hadoop.fs.s3a.secret.key", sk)
        if st:
            builder = builder.config("spark.hadoop.fs.s3a.session.token", st)
    if ep:
        builder = builder.config("spark.hadoop.fs.s3a.endpoint", ep)
        builder = builder.config("spark.hadoop.fs.s3a.path.style.access", "true")

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _clean(v, nd=2):
    if v is None:
        return None
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else round(v, nd)
    if isinstance(v, (dt.date, dt.datetime)):
        return str(v)
    return v


def _rows(rows, nd=2):
    return [{k: _clean(v, nd) for k, v in r.asDict().items()} for r in rows]


def _uri(path):
    """file:/// URI that Spark/Hadoop understand. Spaces must stay UNencoded: Hadoop encodes
    them itself, so Path.as_uri() (which gives %20) ends up double-encoded and not found."""
    p = Path(path).resolve().as_posix()
    return "file://" + ("" if p.startswith("/") else "/") + p


# ---------------------------------------------------------------------------
# I/O mode selection and Windows error detection
# ---------------------------------------------------------------------------
def _hadoop_native_ready():
    """True if Hadoop's Windows native helpers (winutils.exe AND hadoop.dll) are installed."""
    home = os.getenv("HADOOP_HOME") or os.getenv("hadoop.home.dir")
    if not home:
        return False
    b = Path(home) / "bin"
    return (b / "winutils.exe").exists() and (b / "hadoop.dll").exists()


def _resolve_io_mode():
    """'spark' = Spark does file I/O. 'driver' = pandas/pyarrow do it. Override: SPARK_IO_MODE."""
    pref = os.getenv("SPARK_IO_MODE", "auto").strip().lower()
    if pref in ("spark", "driver"):
        return pref
    if sys.platform.startswith("win") and not _hadoop_native_ready():
        return "driver"
    return "spark"


def _is_windows_native_error(exc):
    """True only for the KNOWN Windows Hadoop native-library failures, never for other errors."""
    parts, seen, e = [], set(), exc
    while e is not None and id(e) not in seen:  # include the whole cause chain
        seen.add(id(e))
        parts.append(str(e))
        java = getattr(e, "java_exception", None)
        if java is not None:
            parts.append(str(java))
        e = e.__cause__ or e.__context__
    text = "\n".join(parts)
    return (
        "NativeIO$Windows" in text
        or "nativeio.NativeIO" in text
        or ("UnsatisfiedLinkError" in text and "org.apache.hadoop" in text)
        or ("winutils.exe" in text and ("Could not locate" in text or "not found" in text.lower()))
    )


def _clear_dir(path):
    """Delete an old curated folder; refuse to continue if files are locked (e.g. by OneDrive),
    because writing on top of old files would silently duplicate rows."""
    if not path.exists():
        return
    import gc
    gc.collect()
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        time.sleep(0.05)
    leftovers = list(path.rglob("*.parquet")) if path.exists() else []
    if leftovers:
        time.sleep(0.1)
        shutil.rmtree(path, ignore_errors=True)
        leftovers = list(path.rglob("*.parquet")) if path.exists() else []
    if leftovers:
        raise RuntimeError(f"Could not remove old Parquet files in '{path}' (for example "
                           f"'{leftovers[0].name}'). Close programs using them or pause OneDrive "
                           "sync for this folder, then run again.")


class LakeWriter:
    """All local file I/O of the job: reads the raw zone, writes the curated zone.

    ``read_raw`` returns a Spark DataFrame in RAW_SCHEMA. ``save`` writes partitioned Parquet and
    returns a DataFrame to continue with. Both honour the I/O mode (see module docstring).
    """

    LABELS = {"spark": "Spark distributed Parquet writer",
              "driver": "driver-side pyarrow writer (Windows-safe mode)"}
    READERS = {"spark": "Spark CSV reader",
               "driver": "pandas CSV reader + Arrow hand-off to Spark (Windows-safe mode)"}

    def __init__(self, spark):
        self.spark = spark
        self.mode = _resolve_io_mode()
        if self.mode == "driver":
            log.warning("Windows-safe I/O mode: raw CSVs are read with pandas and Parquet is written "
                        "with pyarrow; Spark does not touch the local file system "
                        "(set SPARK_IO_MODE=spark to force Spark's own readers/writers).")

    @property
    def label(self):  # Parquet writer
        return self.LABELS[self.mode]

    @property
    def reader_label(self):
        return self.READERS[self.mode]

    # ----- reading the raw zone -------------------------------------------
    def read_raw(self, raw_dir):
        raw_dir = Path(raw_dir)
        if self.mode == "spark":
            try:
                return self._read_raw_spark(raw_dir)
            except Exception as exc:  # noqa: BLE001
                if not _is_windows_native_error(exc):
                    raise
                self._switch_to_driver("reading the raw CSV files")
        return self._read_raw_driver(raw_dir)

    def _read_raw_spark(self, raw_dir):
        return self.spark.read.csv(_uri(raw_dir) + "/*.csv", header=True, schema=RAW_SCHEMA)

    def _read_raw_driver(self, raw_dir):
        import pandas as pd
        import pyarrow as pa

        files = sorted(raw_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files in {raw_dir}. Run the ingestion step first.")
        names = [f.name for f in RAW_SCHEMA.fields]
        text_cols = [f.name for f in RAW_SCHEMA.fields if isinstance(f.dataType, StringType)]
        frames = []
        for f in files:
            # Everything is read as text first so that bad numbers can be converted safely below.
            # utf-8-sig removes a byte-order mark if Excel/Notepad added one.
            part = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
            missing = [c for c in names if c not in part.columns]
            if missing:
                raise ValueError(f"{f.name} is missing required column(s): {', '.join(missing)}")
            frames.append(part[names])
        pdf = pd.concat(frames, ignore_index=True)
        for c in names:
            if c not in text_cols:
                # Unparseable text becomes missing (like Spark's PERMISSIVE CSV mode).
                pdf[c] = pd.to_numeric(pdf[c], errors="coerce")

        arrow_schema = pa.schema([(f.name, pa.string() if f.name in text_cols else pa.float64())
                                  for f in RAW_SCHEMA.fields])
        # from_pandas turns NaN into real nulls, which is what the quality report counts.
        table = pa.Table.from_pandas(pdf, schema=arrow_schema, preserve_index=False)
        try:
            return self.spark.createDataFrame(table, schema=RAW_SCHEMA)
        except TypeError:  # PySpark build without pyarrow.Table support: slower Python rows
            rows = list(zip(*[table.column(i).to_pylist() for i in range(table.num_columns)]))
            return self.spark.createDataFrame(rows, schema=RAW_SCHEMA)

    # ----- writing the curated zone ---------------------------------------
    def save(self, df, path, partition_cols):
        path = Path(path)
        if self.mode == "spark":
            try:
                return self._save_spark(df, path, partition_cols)
            except Exception as exc:  # noqa: BLE001
                if not _is_windows_native_error(exc):
                    raise
                self._switch_to_driver("writing Parquet files")
        return self._save_driver(df, path, partition_cols)

    def _save_spark(self, df, path, partition_cols):
        uri = _uri(path)
        df.write.mode("overwrite").partitionBy(*partition_cols).parquet(uri)
        return self.spark.read.parquet(uri)  # later steps read back from the lake

    def _save_driver(self, df, path, partition_cols):
        import pyarrow.parquet as pq

        df = df.persist(StorageLevel.MEMORY_AND_DISK)  # computed once, reused afterwards
        if hasattr(df, "toArrow"):
            table = df.toArrow()
        else:  # PySpark < 4.0
            import pyarrow as pa
            table = pa.Table.from_pandas(df.toPandas(), preserve_index=False)
        _clear_dir(path)  # also removes half-written Spark output
        path.mkdir(parents=True, exist_ok=True)
        pq.write_to_dataset(table, root_path=str(path), partition_cols=list(partition_cols))
        return df

    def _switch_to_driver(self, what):
        log.warning("Spark failed while %s because Hadoop's Windows native files "
                    "(winutils.exe / hadoop.dll) are missing. Switching to Windows-safe driver "
                    "mode (pandas + pyarrow) and continuing.", what)
        self.mode = "driver"


def run_job(raw_dir=None, curated_dir=None, results_dir=None):
    raw_dir = Path(raw_dir or config.raw_dir())
    curated_dir = Path(curated_dir or config.curated_dir())
    results_dir = Path(results_dir or config.results_dir())

    store = get_store()
    # If no local CSV files and cloud storage is active, try downloading raw zone from object storage
    if not list(raw_dir.glob("*.csv")) and store.backend in ("s3", "supabase"):
        log.info("No CSV files found in local %s; attempting download from %s ...", raw_dir, store.describe())
        try:
            downloaded = store.download_zone("raw", raw_dir)
            log.info("Downloaded %d raw files from %s.", downloaded, store.backend)
        except Exception as exc:
            log.warning("Could not download raw files from %s: %s", store.backend, exc)

    if not list(raw_dir.glob("*.csv")):
        raise FileNotFoundError(f"No CSV files in {raw_dir}. Run the ingestion step first or upload raw files to cloud storage.")
    manifest_path = raw_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    started = time.time()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    spark = build_spark()
    try:
        summary = _pipeline(spark, raw_dir, curated_dir, manifest)
    finally:
        spark_version = spark.version
        master = spark.sparkContext.master
        spark.stop()

    summary["run"].update({
        "engine": f"Apache Spark {spark_version}", "master": master,
        "started_at": started_at,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "duration_seconds": round(time.time() - started, 1),
    })
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote %s", summary_path)

    # Auto-publish curated and results to cloud storage when S3 or Supabase backend is enabled
    if store.backend in ("s3", "supabase"):
        try:
            log.info("Auto-publishing batch results to %s storage ...", store.backend)
            store.publish(zones=("curated", "results"))
        except Exception as exc:
            log.warning("Could not auto-publish to %s: %s", store.backend, exc)

    return summary


def _pipeline(spark, raw_dir, curated_dir, manifest):
    lake = LakeWriter(spark)

    # ---------- 1. READ raw zone ----------
    raw = lake.read_raw(raw_dir)
    typed = (raw.withColumn("ts", F.to_timestamp("time", "yyyy-MM-dd'T'HH:mm"))
                .persist(StorageLevel.MEMORY_AND_DISK))
    raw_rows = typed.count()

    # ---------- 2. CLEAN ----------
    valid = typed.filter(F.col("city").isNotNull() & F.col("ts").isNotNull())
    invalid_rows = raw_rows - valid.count()
    dedup = valid.dropDuplicates(["city", "ts"])

    agg_exprs = [F.count("*").alias("rows")]
    for c, (lo, hi) in BOUNDS.items():
        agg_exprs.append(F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"{c}__missing"))
        agg_exprs.append(F.sum(F.when(F.col(c).isNotNull() & ~F.col(c).between(lo, hi), 1)
                               .otherwise(0)).alias(f"{c}__out_of_range"))
    q = dedup.agg(*agg_exprs).collect()[0].asDict()
    dedup_rows = q["rows"]

    def in_range(c):
        lo, hi = BOUNDS[c]
        return F.when(F.col(c).between(lo, hi), F.col(c)).alias(c)  # NULL when out of range

    cleaned = (dedup.select("city", "country", "latitude", "longitude", "ts",
                            *[in_range(c) for c in BOUNDS])
                    .filter(F.col("temp_c").isNotNull())
                    .withColumn("date", F.to_date("ts"))
                    .withColumn("year", F.year("ts"))
                    .withColumn("month", F.month("ts"))
                    .withColumn("hour", F.hour("ts")))

    # ---------- 3. WRITE curated zone (columnar Parquet, partitioned) ----------
    hourly = lake.save(cleaned, curated_dir / "hourly", ["city", "year"])
    clean_rows = hourly.count()
    typed.unpersist()

    # ---------- 4. DAILY aggregates ----------
    daily = (hourly.groupBy("city", "date")
             .agg(F.count("*").alias("hours"),
                  F.avg("temp_c").alias("temp_mean"), F.min("temp_c").alias("temp_min"),
                  F.max("temp_c").alias("temp_max"), F.avg("humidity_pct").alias("humidity_mean"),
                  F.sum("precip_mm").alias("precip_total"), F.avg("wind_kmh").alias("wind_mean"),
                  F.max("wind_kmh").alias("wind_max"), F.avg("pressure_hpa").alias("pressure_mean"),
                  F.avg("cloud_pct").alias("cloud_mean"))
             .filter(F.col("hours") >= 20)  # ignore partial days
             .withColumn("year", F.year("date")).withColumn("month", F.month("date")))
    daily = lake.save(daily, curated_dir / "daily", ["city"]).persist(StorageLevel.MEMORY_AND_DISK)
    daily.createOrReplaceTempView("daily")
    n_daily = daily.count()

    # ---------- 5. SPARK SQL summaries ----------
    city_stats = _rows(spark.sql("""
        SELECT city,
               ROUND(AVG(temp_mean), 2)                      AS avg_temp,
               ROUND(MAX(temp_max), 1)                       AS max_temp,
               ROUND(MIN(temp_min), 1)                       AS min_temp,
               ROUND(AVG(humidity_mean), 1)                  AS avg_humidity,
               ROUND(SUM(precip_total), 1)                   AS total_rain_mm,
               SUM(CASE WHEN precip_total >= 1 THEN 1 ELSE 0 END) AS rainy_days,
               COUNT(*)                                      AS days
        FROM daily GROUP BY city ORDER BY city""").collect())

    monthly = _rows(spark.sql("""
        SELECT city, year, month,
               ROUND(AVG(temp_mean), 2) AS avg_temp, ROUND(MAX(temp_max), 1) AS max_temp,
               ROUND(MIN(temp_min), 1) AS min_temp, ROUND(AVG(humidity_mean), 1) AS avg_humidity,
               ROUND(SUM(precip_total), 1) AS rain_mm, ROUND(AVG(wind_mean), 1) AS avg_wind
        FROM daily GROUP BY city, year, month ORDER BY city, year, month""").collect())

    # ---------- 6. WINDOW functions: rolling mean + seasonal anomalies ----------
    w_city = Window.partitionBy("city").orderBy("date")
    recent = (daily.withColumn("roll7", F.avg("temp_mean").over(w_city.rowsBetween(-6, 0)))
                   .withColumn("rn", F.row_number().over(Window.partitionBy("city").orderBy(F.desc("date"))))
                   .filter(F.col("rn") <= 60)
                   .select("city", "date", "temp_mean", "roll7").orderBy("city", "date"))
    recent_daily = _rows(recent.collect())

    w_month = Window.partitionBy("city", "month")  # compare a day with the SAME calendar month
    anomalies = (daily.withColumn("clim_mean", F.avg("temp_mean").over(w_month))
                      .withColumn("clim_std", F.stddev("temp_mean").over(w_month))
                      .withColumn("z", (F.col("temp_mean") - F.col("clim_mean")) / F.col("clim_std"))
                      .filter(F.abs("z") >= ANOMALY_Z)
                      .orderBy(F.abs("z").desc()).limit(15)
                      .select("city", "date", "temp_mean", "clim_mean", "z").collect())
    extreme_rain = daily.orderBy(F.desc("precip_total")).limit(10).select(
        "city", "date", "precip_total").collect()

    # ---------- 7. CORRELATIONS across all cities ----------
    cols = ["temp_mean", "humidity_mean", "precip_total", "wind_mean", "pressure_mean", "cloud_mean"]
    pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]]
    corr_row = daily.agg(*[F.corr(a, b).alias(f"{a}|{b}") for a, b in pairs]).collect()[0].asDict()
    correlations = [{"x": k.split("|")[0], "y": k.split("|")[1], "r": _clean(v)} for k, v in corr_row.items()]

    # ---------- 8. MLlib: next-day mean temperature ----------
    ml = _forecast_model(daily, w_city)
    daily.unpersist()

    first_last = hourly.agg(F.min("date"), F.max("date")).collect()[0]
    return {
        "run": {
            "data_mode": manifest.get("mode", "unknown"),
            "data_source": manifest.get("source", "unknown"),
            "data_start": str(first_last[0]), "data_end": str(first_last[1]),
            "cities": sorted(r["city"] for r in city_stats),
            "raw_rows": raw_rows, "clean_rows": clean_rows, "daily_rows": n_daily,
            "parquet_writer": lake.label,
            "raw_reader": lake.reader_label,
        },
        "quality": {
            "raw_rows": raw_rows, "invalid_timestamp_or_city_rows": invalid_rows,
            "duplicates_removed": (raw_rows - invalid_rows) - dedup_rows,
            "rows_after_dedup": dedup_rows,
            "missing_values": {c: q[f"{c}__missing"] for c in BOUNDS},
            "out_of_range_set_to_null": {c: q[f"{c}__out_of_range"] for c in BOUNDS},
            "rows_dropped_no_temperature": dedup_rows - clean_rows,
            "clean_rows": clean_rows,
        },
        "city_stats": city_stats, "monthly": monthly, "recent_daily": recent_daily,
        "correlations": correlations,
        "anomalies": _rows(anomalies), "anomaly_rule": f"|z| >= {ANOMALY_Z} versus the same city and calendar month",
        "extreme_rain_days": _rows(extreme_rain),
        "ml": ml,
        "notes": [
            "Aggregations ran as Apache Spark jobs over the cleaned data in the data lake.",
            "Model estimates are statistical, not official forecasts; follow your national weather service for warnings.",
        ] + (["DATA IS SYNTHETIC: generated for demonstration, not real observations."]
             if manifest.get("mode") == "synthetic" else [])
          + (["Windows-safe mode: raw CSV files were read with pandas and curated Parquet files were "
              "written with pyarrow because Hadoop's Windows native files are not installed; all "
              "cleaning, SQL, window functions and MLlib still ran in Spark."]
             if lake.mode == "driver" else []),
    }


def _forecast_model(daily, w_city):
    feat = (daily
            .withColumn("lag1", F.lag("temp_mean", 1).over(w_city))
            .withColumn("lag2", F.lag("temp_mean", 2).over(w_city))
            .withColumn("roll7", F.avg("temp_mean").over(w_city.rowsBetween(-6, 0)))
            .withColumn("next_date", F.date_add("date", 1))
            .withColumn("target", F.when(F.datediff(F.lead("date", 1).over(w_city), F.col("date")) == 1,
                                         F.lead("temp_mean", 1).over(w_city)))
            .withColumn("doy_sin", F.sin(2 * math.pi * F.dayofyear("next_date") / 365.25))
            .withColumn("doy_cos", F.cos(2 * math.pi * F.dayofyear("next_date") / 365.25)))

    labeled = feat.dropna(subset=FEATURES + ["target"])
    max_date = labeled.agg(F.max("date")).collect()[0][0]
    if max_date is None:
        return {"status": "skipped", "reason": "not enough data"}
    cutoff = max_date - dt.timedelta(days=TEST_DAYS)
    train = labeled.filter(F.col("date") <= F.lit(cutoff))
    test = labeled.filter(F.col("date") > F.lit(cutoff))
    n_train, n_test = train.count(), test.count()
    if n_train < 100 or n_test < 10:
        return {"status": "skipped", "reason": f"not enough data (train={n_train}, test={n_test})"}

    assembler = VectorAssembler(inputCols=FEATURES, outputCol="features", handleInvalid="skip")
    train_v, test_v = assembler.transform(train), assembler.transform(test)

    def score(pred_df):
        out = {}
        for m in ("rmse", "mae", "r2"):
            out[m] = _clean(RegressionEvaluator(labelCol="target", predictionCol="prediction",
                                                metricName=m).evaluate(pred_df), 3)
        return out

    metrics = {"persistence_baseline": score(test_v.withColumn("prediction", F.col("temp_mean")))}
    lr = LinearRegression(featuresCol="features", labelCol="target").fit(train_v)
    metrics["linear_regression"] = score(lr.transform(test_v))
    rf = RandomForestRegressor(featuresCol="features", labelCol="target", numTrees=40,
                               maxDepth=8, seed=42).fit(train_v)
    metrics["random_forest"] = score(rf.transform(test_v))

    best_name = min(("linear_regression", "random_forest"), key=lambda k: metrics[k]["rmse"])
    best = lr if best_name == "linear_regression" else rf

    latest = (feat.withColumn("rn", F.row_number().over(Window.partitionBy("city").orderBy(F.desc("date"))))
                  .filter(F.col("rn") == 1).dropna(subset=FEATURES))
    est = best.transform(assembler.transform(latest)).select(
        "city", F.col("date").alias("as_of_date"), F.col("temp_mean").alias("last_mean_temp_c"),
        F.col("prediction").alias("predicted_next_day_mean_temp_c")).orderBy("city")

    importance = sorted(zip(FEATURES, rf.featureImportances.toArray()), key=lambda t: -t[1])
    return {
        "status": "ok", "target": "next-day mean temperature (deg C)",
        "train_rows": n_train, "test_rows": n_test, "test_days": TEST_DAYS,
        "features": FEATURES, "metrics": metrics, "best_model": best_name,
        "feature_importance": [{"feature": f, "importance": _clean(float(i), 3)} for f, i in importance],
        "next_day_estimates": _rows(est.collect()),
        "note": "Time-based split: the last 60 days are held out for testing. "
                "Estimates are for the day after 'as_of_date' (the last day in the dataset).",
    }


def main(argv=None):
    argparse.ArgumentParser(description="Run the Spark batch job on the data lake").parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = run_job()
    print(json.dumps(s["run"], indent=2))


if __name__ == "__main__":
    main()
