"""Tests for the big data + cloud storage layer.

* The Spark test runs a REAL local Spark job on a small synthetic dataset.
* S3 tests use `moto` (an AWS emulator), so no AWS account or internet is needed.
  They prove the code paths work, not that your real AWS account is configured.
Run:  python -m unittest -v
"""
import json
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(), "bd_test.db"))

from bigdata import config, ingest as ingest_mod  # noqa: E402

import importlib.util  # noqa: E402
import shutil  # noqa: E402

# Apache Spark is an OPTIONAL engine: its tests are skipped when pyspark or Java is missing.
HAS_SPARK = importlib.util.find_spec("pyspark") is not None and shutil.which("java") is not None
needs_spark = unittest.skipUnless(HAS_SPARK, "optional Spark engine: pyspark and Java are not installed")
from bigdata.storage import LocalStore, S3Store, SupabaseStore  # noqa: E402


class FakeResp:
    def __init__(self, data, status=200):
        self._d, self.status_code = data, status

    def json(self):
        return self._d

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.exceptions.HTTPError(str(self.status_code))


class SyntheticDataTests(unittest.TestCase):
    def setUp(self):
        self.city = config.CITIES[0]
        self.df = ingest_mod.synthetic_city_frame(self.city, date(2025, 1, 1), date(2025, 3, 31), seed=1)

    def test_shape_and_columns(self):
        self.assertEqual(list(self.df.columns), ingest_mod.COLUMNS)
        self.assertGreaterEqual(len(self.df), 90 * 24)  # plus a few duplicates

    def test_contains_dirty_rows_for_cleaning_demo(self):
        self.assertGreater(self.df["time"].duplicated().sum(), 0)

    def test_deterministic_for_same_seed(self):
        again = ingest_mod.synthetic_city_frame(self.city, date(2025, 1, 1), date(2025, 3, 31), seed=1)
        pd.testing.assert_frame_equal(self.df, again)

    def test_plausible_values(self):
        good = self.df["temp_c"].dropna()
        self.assertTrue(good.between(0, 55).all())
        self.assertTrue(self.df["cloud_pct"].between(0, 100).all())


class IngestTests(unittest.TestCase):
    def test_synthetic_writes_csv_and_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            m = ingest_mod.ingest("synthetic", years=0.1, n_cities=2, raw_dir=d, end_date=date(2026, 1, 31))
            self.assertEqual(m["mode"], "synthetic")
            self.assertIn("SYNTHETIC", m["source"])
            self.assertEqual(len(list(Path(d).glob("*.csv"))), 2)
            self.assertTrue((Path(d) / "_manifest.json").exists())

    def test_live_mode_parses_api_response(self):
        n = 48
        payload = {"hourly": {
            "time": [f"2026-01-{1 + i // 24:02d}T{i % 24:02d}:00" for i in range(n)],
            "temperature_2m": [20.0] * n, "relative_humidity_2m": [60] * n, "precipitation": [0.0] * n,
            "wind_speed_10m": [10.0] * n, "pressure_msl": [1010.0] * n, "cloud_cover": [40] * n}}
        with tempfile.TemporaryDirectory() as d, \
                mock.patch("bigdata.ingest.requests.get", return_value=FakeResp(payload)):
            m = ingest_mod.ingest("live", years=0.01, n_cities=2, raw_dir=d, end_date=date(2026, 1, 31))
            self.assertEqual(m["mode"], "live")
            self.assertEqual(m["rows"], 2 * n)
            self.assertEqual(m["source"], "Open-Meteo Archive API")

    def test_live_mode_total_failure_gives_helpful_error(self):
        import requests
        with tempfile.TemporaryDirectory() as d, \
                mock.patch("bigdata.ingest.requests.get", side_effect=requests.ConnectionError("down")), \
                mock.patch("bigdata.ingest._time.sleep"):
            with self.assertRaises(ingest_mod.IngestError) as ctx:
                ingest_mod.ingest("live", years=0.01, n_cities=2, raw_dir=d, end_date=date(2026, 1, 31))
            self.assertIn("--mode synthetic", str(ctx.exception))

    def test_bad_mode_rejected(self):
        with self.assertRaises(ValueError):
            ingest_mod.ingest("nope")


class S3StorageTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {
            "AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
            "AWS_DEFAULT_REGION": "us-east-1", "DATA_LAKE_DIR": tempfile.mkdtemp()})
        self.env.start()
        lake = config.lake_dir()
        (lake / "results").mkdir(parents=True)
        (lake / "curated" / "daily").mkdir(parents=True)
        (lake / "results" / "summary.json").write_text(json.dumps({"run": {"engine": "x"}}))
        (lake / "curated" / "daily" / "part-0.parquet").write_bytes(b"PAR1")
        (lake / "curated" / "daily" / ".part-0.parquet.crc").write_bytes(b"crc")

    def tearDown(self):
        self.env.stop()

    def test_publish_and_read_back(self):
        from moto import mock_aws
        with mock_aws():
            store = S3Store("test-bucket", region="us-east-1")
            res = store.publish()
            self.assertEqual(res["uploaded_files"], 2)  # .crc file skipped
            self.assertEqual(store.get_json("results/summary.json")["run"]["engine"], "x")
            self.assertIsNone(store.get_json("results/missing.json"))

    def test_publish_replaces_stale_objects(self):
        from moto import mock_aws
        with mock_aws():
            store = S3Store("test-bucket", region="us-east-1")
            store.ensure_bucket()
            store.client.put_object(Bucket="test-bucket", Key="curated/old/stale.parquet", Body=b"x")
            store.publish()
            keys = [o["Key"] for o in store.client.list_objects_v2(Bucket="test-bucket")["Contents"]]
            self.assertNotIn("curated/old/stale.parquet", keys)

    def test_local_store_reads_lake(self):
        self.assertEqual(LocalStore().get_json("results/summary.json")["run"]["engine"], "x")
        self.assertIsNone(LocalStore().get_json("results/nope.json"))

    def test_local_store_file_ops(self):
        store = LocalStore()
        tmp_f = Path(tempfile.mktemp(suffix=".txt"))
        tmp_f.write_text("hello local")
        store.upload_file(str(tmp_f), "raw/hello.txt")
        self.assertTrue(store.exists("raw/hello.txt"))
        dest_f = Path(tempfile.mktemp(suffix=".txt"))
        store.download_file("raw/hello.txt", str(dest_f))
        self.assertEqual(dest_f.read_text(), "hello local")

    def test_s3_upload_download_file(self):
        from moto import mock_aws
        with mock_aws():
            store = S3Store("test-bucket", region="us-east-1")
            store.ensure_bucket()
            tmp_f = Path(tempfile.mktemp(suffix=".txt"))
            tmp_f.write_text("content on disk")
            s3_uri = store.upload_file(str(tmp_f), "raw/sample.txt")
            self.assertEqual(s3_uri, "s3://test-bucket/raw/sample.txt")
            self.assertTrue(store.exists("raw/sample.txt"))
            self.assertIn("raw/sample.txt", store.list_keys("raw/"))

            dest_f = Path(tempfile.mktemp(suffix=".txt"))
            store.download_file("raw/sample.txt", str(dest_f))
            self.assertEqual(dest_f.read_text(), "content on disk")

    def test_s3_put_and_get_json(self):
        from moto import mock_aws
        with mock_aws():
            store = S3Store("test-bucket", region="us-east-1")
            payload = {"status": "success", "count": 42}
            store.put_json("results/test.json", payload)
            data = store.get_json("results/test.json")
            self.assertEqual(data, payload)

    def test_s3_download_zone_and_lake(self):
        from moto import mock_aws
        with mock_aws():
            store = S3Store("test-bucket", region="us-east-1")
            store.publish()  # uploads initial files from setUp
            with tempfile.TemporaryDirectory() as target_dir:
                count = store.download_zone("results", Path(target_dir) / "results")
                self.assertGreaterEqual(count, 1)
                self.assertTrue((Path(target_dir) / "results" / "summary.json").exists())


class MockBucketProxy:
    def __init__(self, bucket_name="weather-data-lake", files=None):
        self.bucket_name = bucket_name
        self.files = files if files is not None else {}

    def upload(self, path, file, file_options=None):
        clean_path = path.replace("\\", "/").lstrip("/")
        if isinstance(file, bytes):
            data = file
        elif isinstance(file, str):
            data = file.encode("utf-8")
        elif hasattr(file, "read"):
            data = file.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
        else:
            data = bytes(file)
        self.files[clean_path] = data
        return {"Key": clean_path}

    def download(self, path, options=None, query_params=None):
        clean_path = path.replace("\\", "/").lstrip("/")
        if clean_path not in self.files:
            raise RuntimeError(f"Object not found: 404 for {path}")
        return self.files[clean_path]

    def exists(self, path):
        return path.replace("\\", "/").lstrip("/") in self.files

    def list(self, path=None, options=None):
        clean_path = (path or "").replace("\\", "/").strip("/")
        results = []
        seen_dirs = set()
        for k in self.files:
            if clean_path and not k.startswith(clean_path + "/"):
                continue
            rel = k[len(clean_path) + 1:] if clean_path else k
            parts = rel.split("/")
            if len(parts) > 1:
                dir_name = parts[0]
                if dir_name not in seen_dirs:
                    seen_dirs.add(dir_name)
                    results.append({"name": dir_name, "id": None, "metadata": None})
            else:
                results.append({"name": parts[0], "id": "file-id", "metadata": {"size": len(self.files[k])}})
        return results

    def remove(self, paths):
        for p in paths:
            self.files.pop(p.replace("\\", "/").lstrip("/"), None)
        return [{"name": p} for p in paths]


class MockStorageClient:
    def __init__(self, files=None):
        self._files = files if files is not None else {}
        self.buckets = {"weather-data-lake": {}}

    def from_(self, bucket_name):
        return MockBucketProxy(bucket_name, self._files)

    def get_bucket(self, bucket_name):
        return {"id": bucket_name, "name": bucket_name}

    def create_bucket(self, bucket_name, options=None):
        self.buckets[bucket_name] = options or {}
        return {"name": bucket_name}


class MockSupabaseClient:
    def __init__(self, files=None):
        self.storage = MockStorageClient(files)


class SupabaseStorageTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {
            "SUPABASE_URL": "https://testproject.supabase.co",
            "SUPABASE_KEY": "test-anon-key-123",
            "SUPABASE_BUCKET": "weather-data-lake",
            "DATA_LAKE_DIR": tempfile.mkdtemp(),
        })
        self.env.start()
        self.mock_client = MockSupabaseClient()

    def tearDown(self):
        self.env.stop()

    def test_missing_url_or_key_raises_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            SupabaseStore(url="", key="valid-key")
        self.assertIn("SUPABASE_URL is not set", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            SupabaseStore(url="https://valid.supabase.co", key="")
        self.assertIn("SUPABASE_KEY is not set", str(ctx.exception))

    def test_upload_and_download_file(self):
        store = SupabaseStore(client=self.mock_client)
        tmp_src = Path(tempfile.mktemp(suffix=".csv"))
        tmp_src.write_text("city,temp\nHyderabad,28.5\n")

        uri = store.upload_file(str(tmp_src), "raw/hyderabad.csv")
        self.assertEqual(uri, "supabase://weather-data-lake/raw/hyderabad.csv")
        self.assertTrue(store.exists("raw/hyderabad.csv"))

        tmp_dst = Path(tempfile.mktemp(suffix=".csv"))
        store.download_file("raw/hyderabad.csv", str(tmp_dst))
        self.assertEqual(tmp_dst.read_text(), tmp_src.read_text())

    def test_put_and_get_json(self):
        store = SupabaseStore(client=self.mock_client)
        data = {"engine": "Apache Spark", "clean_rows": 1000}
        uri = store.put_json("results/summary.json", data)
        self.assertEqual(uri, "supabase://weather-data-lake/results/summary.json")

        loaded = store.get_json("results/summary.json")
        self.assertEqual(loaded, data)
        self.assertIsNone(store.get_json("results/missing.json"))

    def test_exists_and_list_keys(self):
        store = SupabaseStore(client=self.mock_client)
        store.put_json("curated/hourly/city=Delhi/part.parquet", {"d": 1})
        store.put_json("curated/hourly/city=Pune/part.parquet", {"d": 2})
        store.put_json("results/summary.json", {"d": 3})

        self.assertTrue(store.exists("results/summary.json"))
        self.assertFalse(store.exists("results/nope.json"))

        curated_keys = store.list_keys("curated")
        self.assertIn("curated/hourly/city=Delhi/part.parquet", curated_keys)
        self.assertIn("curated/hourly/city=Pune/part.parquet", curated_keys)
        self.assertNotIn("results/summary.json", curated_keys)

    def test_publish_and_download_zone(self):
        lake = config.lake_dir()
        (lake / "results").mkdir(parents=True)
        (lake / "results" / "summary.json").write_text(json.dumps({"test": True}))
        (lake / "curated" / "daily").mkdir(parents=True)
        (lake / "curated" / "daily" / "day.parquet").write_bytes(b"PAR1")

        store = SupabaseStore(client=self.mock_client)
        res = store.publish(["results", "curated"])
        self.assertEqual(res["uploaded_files"], 2)
        self.assertEqual(res["backend"], "supabase")

        with tempfile.TemporaryDirectory() as target_dir:
            count = store.download_zone("results", Path(target_dir) / "results")
            self.assertEqual(count, 1)
            self.assertTrue((Path(target_dir) / "results" / "summary.json").exists())

    def test_describe_never_exposes_secret_key(self):
        store = SupabaseStore(client=self.mock_client)
        desc = store.describe()
        self.assertEqual(desc["backend"], "supabase")
        self.assertEqual(desc["bucket"], "weather-data-lake")
        self.assertEqual(desc["endpoint"], "https://testproject.supabase.co")
        self.assertNotIn("key", desc)
        self.assertNotIn("test-anon-key-123", str(desc))


@needs_spark
class SparkPipelineTests(unittest.TestCase):
    """Runs a real Spark job (takes ~30-60 s)."""

    @classmethod
    def setUpClass(cls):
        # Folder name with spaces on purpose (regression test: "bda cc project" broke Spark paths)
        cls.tmp = tempfile.mkdtemp(prefix="lake with spaces ")
        cls.env = mock.patch.dict(os.environ, {"DATA_LAKE_DIR": cls.tmp, "STORAGE_BACKEND": "local",
                                               "SPARK_IO_MODE": "auto"})
        cls.env.start()
        ingest_mod.ingest("synthetic", years=1.5, n_cities=2, end_date=date(2026, 10, 18))
        from bigdata.spark_job import run_job
        cls.summary = run_job()

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()

    def test_cleaning(self):
        q = self.summary["quality"]
        self.assertGreater(q["duplicates_removed"], 0)
        self.assertLess(q["clean_rows"], q["raw_rows"])
        self.assertEqual(q["raw_rows"] - q["duplicates_removed"] - q["rows_dropped_no_temperature"], q["clean_rows"])

    def test_parquet_lake_is_partitioned(self):
        hourly = Path(self.tmp) / "curated" / "hourly"
        self.assertTrue((hourly / "city=Hyderabad").is_dir())
        self.assertTrue(list(hourly.rglob("*.parquet")))
        self.assertTrue(list((Path(self.tmp) / "curated" / "daily").rglob("*.parquet")))

    def test_sql_aggregates(self):
        self.assertEqual(len(self.summary["city_stats"]), 2)
        self.assertGreater(len(self.summary["monthly"]), 20)
        for c in self.summary["city_stats"]:
            self.assertTrue(10 < c["avg_temp"] < 40)

    def test_anomalies_and_correlations(self):
        self.assertGreaterEqual(len(self.summary["anomalies"]), 1)
        self.assertTrue(all(abs(a["z"]) >= 3.0 for a in self.summary["anomalies"]))
        self.assertEqual(len(self.summary["correlations"]), 15)

    def test_ml_model(self):
        ml = self.summary["ml"]
        self.assertEqual(ml["status"], "ok")
        for k in ("persistence_baseline", "linear_regression", "random_forest"):
            self.assertGreater(ml["metrics"][k]["rmse"], 0)
        self.assertIn(ml["best_model"], ("linear_regression", "random_forest"))
        self.assertEqual(len(ml["next_day_estimates"]), 2)

    def test_run_metadata_and_labels(self):
        run = self.summary["run"]
        self.assertEqual(run["data_mode"], "synthetic")
        self.assertIn("Spark", run["engine"])
        self.assertTrue(any("SYNTHETIC" in n for n in self.summary["notes"]))
        self.assertTrue((Path(self.tmp) / "results" / "summary.json").exists())

    def test_job_without_raw_data_fails_clearly(self):
        from bigdata.spark_job import run_job
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError):
                run_job(raw_dir=empty)


WINDOWS_ERROR = ("py4j.protocol.Py4JJavaError: An error occurred while calling o570.parquet.\n"
                 "Caused by: java.lang.UnsatisfiedLinkError: 'boolean org.apache.hadoop.io.nativeio."
                 "NativeIO$Windows.access0(java.lang.String, int)'")


@needs_spark
class WindowsSupportTests(unittest.TestCase):
    """Fast tests (no Spark session) for the Windows-safe logic."""

    def test_path_uri_keeps_spaces_unencoded(self):
        from bigdata.spark_job import _uri
        self.assertIn("/tmp/a b/c", _uri("/tmp/a b/c"))
        self.assertNotIn("%20", _uri("/tmp/bda cc project/raw"))

    def test_error_detection(self):
        from bigdata.spark_job import _is_windows_native_error
        self.assertTrue(_is_windows_native_error(RuntimeError(WINDOWS_ERROR)))
        self.assertTrue(_is_windows_native_error(RuntimeError(
            "Could not locate executable null\\bin\\winutils.exe in the Hadoop binaries.")))
        # error hidden in the cause chain still counts
        try:
            try:
                raise RuntimeError(WINDOWS_ERROR)
            except RuntimeError as inner:
                raise ValueError("wrapped") from inner
        except ValueError as outer:
            self.assertTrue(_is_windows_native_error(outer))
        # genuine application errors and harmless warnings must NOT trigger the fallback
        self.assertFalse(_is_windows_native_error(ValueError("column not found")))
        self.assertFalse(_is_windows_native_error(RuntimeError("HADOOP_HOME and hadoop.home.dir are unset")))
        self.assertFalse(_is_windows_native_error(RuntimeError(
            "java.lang.UnsatisfiedLinkError: no snappyjava in java.library.path")))

    def test_locked_old_files_stop_the_write_instead_of_duplicating_rows(self):
        from bigdata.spark_job import _clear_dir
        d = Path(tempfile.mkdtemp()) / "hourly" / "city=X"
        d.mkdir(parents=True)
        (d / "old.parquet").write_bytes(b"PAR1")
        with mock.patch("bigdata.spark_job.shutil.rmtree"):  # simulate a file locked by OneDrive
            with self.assertRaises(RuntimeError) as ctx:
                _clear_dir(d.parent)
        self.assertIn("OneDrive", str(ctx.exception))

    def test_windows_without_native_files_uses_driver_mode(self):
        import sys
        from bigdata import spark_job
        env = {k: v for k, v in os.environ.items() if k not in ("HADOOP_HOME", "SPARK_IO_MODE")}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(sys, "platform", "win32"):
            self.assertEqual(spark_job._resolve_io_mode(), "driver")

    def test_windows_with_native_files_uses_spark_mode(self):
        import sys
        from bigdata import spark_job
        home = Path(tempfile.mkdtemp())
        (home / "bin").mkdir()
        (home / "bin" / "winutils.exe").write_bytes(b"x")
        (home / "bin" / "hadoop.dll").write_bytes(b"x")
        with mock.patch.dict(os.environ, {"HADOOP_HOME": str(home)}), \
                mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(sys, "platform", "win32"):
            os.environ.pop("SPARK_IO_MODE", None)
            self.assertEqual(spark_job._resolve_io_mode(), "spark")

    def test_manual_override_and_linux_default(self):
        import sys
        from bigdata import spark_job
        with mock.patch.dict(os.environ, {"SPARK_IO_MODE": "driver"}):
            self.assertEqual(spark_job._resolve_io_mode(), "driver")
        with mock.patch.dict(os.environ, {"SPARK_IO_MODE": "auto"}), mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(spark_job._resolve_io_mode(), "spark")

    def test_unrelated_errors_are_not_swallowed(self):
        from bigdata.spark_job import LakeWriter
        with mock.patch.dict(os.environ, {"SPARK_IO_MODE": "spark"}):
            writer = LakeWriter(spark=None)
        with mock.patch.object(LakeWriter, "_save_spark", side_effect=ValueError("boom")):
            with self.assertRaises(ValueError):
                writer.save(None, "/tmp/x", ["city"])


@needs_spark
class RawReaderTests(unittest.TestCase):
    """The pandas -> Arrow -> Spark raw-CSV reader used in Windows-safe mode (real Spark session)."""

    HEADER = "city,country,latitude,longitude,time,temp_c,humidity_pct,precip_mm,wind_kmh,pressure_hpa,cloud_pct\n"
    ROWS = ("Pune,India,18.5,73.8,2026-01-01T00:00,25.5,60,0,10,1010,40\n"
            "Pune,India,18.5,73.8,2026-01-01T01:00,,61,0,11,1010,41\n"      # empty temp -> null
            "Pune,India,18.5,73.8,2026-01-01T02:00,26.0,abc,0,12,1010,42\n"  # bad number -> null
            "Pune,India,18.5,73.8,2026-01-01T03:00,26.5,62,0.5,13,NaN,43\n")  # literal NaN -> null

    @classmethod
    def setUpClass(cls):
        from bigdata.spark_job import build_spark
        cls.spark = build_spark("raw-reader-tests")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def _dir(self, text=None, encoding="utf-8", header=None):
        d = Path(tempfile.mkdtemp(prefix="raw dir "))
        (d / "pune.csv").write_text((header or self.HEADER) + (text or self.ROWS), encoding=encoding)
        return d

    def _lake(self, mode):
        from bigdata.spark_job import LakeWriter
        with mock.patch.dict(os.environ, {"SPARK_IO_MODE": mode}):
            return LakeWriter(self.spark)

    def test_driver_reader_schema_and_real_nulls(self):
        from pyspark.sql import functions as F
        from bigdata.spark_job import RAW_SCHEMA
        df = self._lake("driver").read_raw(self._dir())
        self.assertEqual([(f.name, f.dataType) for f in df.schema.fields],
                         [(f.name, f.dataType) for f in RAW_SCHEMA.fields])
        r = df.agg(F.count("*").alias("n"),
                   F.sum(F.col("temp_c").isNull().cast("int")).alias("null_temp"),
                   F.sum(F.col("humidity_pct").isNull().cast("int")).alias("null_hum"),
                   F.sum(F.col("pressure_hpa").isNull().cast("int")).alias("null_pres"),
                   F.sum(F.isnan("temp_c").cast("int")).alias("nan_temp")).collect()[0]
        self.assertEqual((r["n"], r["null_temp"], r["null_hum"], r["null_pres"], r["nan_temp"]), (4, 1, 1, 1, 0))

    def test_driver_reader_equals_spark_reader(self):
        from bigdata.spark_job import _hadoop_native_ready
        import sys
        if sys.platform.startswith("win") and not _hadoop_native_ready():
            self.skipTest("Spark CSV reader cannot glob on Windows without hadoop.dll / winutils.exe")
        # Same rows through both readers (empty value and unparseable text included). The literal
        # text "NaN" is excluded on purpose: Spark keeps it as NaN, the pandas path makes it null.
        common = "".join(self.ROWS.splitlines(keepends=True)[:3])
        d = self._dir(text=common)
        a = sorted(map(tuple, self._lake("driver").read_raw(d).collect()), key=str)
        b = sorted(map(tuple, self._lake("spark").read_raw(d).collect()), key=str)
        self.assertEqual(len(a), 3)
        self.assertEqual(a, b)

    def test_missing_column_is_a_clear_error(self):
        d = self._dir(text="Pune,India,1,2,2026-01-01T00:00,25\n", header="city,country,latitude,longitude,time,temp_c\n")
        with self.assertRaises(ValueError) as ctx:
            self._lake("driver").read_raw(d)
        self.assertIn("humidity_pct", str(ctx.exception))

    def test_byte_order_mark_is_ignored(self):
        df = self._lake("driver").read_raw(self._dir(encoding="utf-8-sig"))
        self.assertEqual(df.count(), 4)
        self.assertIn("city", df.columns)

    def test_native_error_while_reading_falls_back_to_pandas(self):
        from bigdata.spark_job import LakeWriter
        lake = self._lake("spark")
        with mock.patch.object(LakeWriter, "_read_raw_spark", side_effect=RuntimeError(WINDOWS_ERROR)):
            df = lake.read_raw(self._dir())
        self.assertEqual(lake.mode, "driver")
        self.assertEqual(df.count(), 4)

    def test_unrelated_read_error_is_raised_not_hidden(self):
        from bigdata.spark_job import LakeWriter
        lake = self._lake("spark")
        with mock.patch.object(LakeWriter, "_read_raw_spark", side_effect=ValueError("boom")):
            with self.assertRaises(ValueError):
                lake.read_raw(self._dir())
        self.assertEqual(lake.mode, "spark")


NO_FS_JAVA = """
package sim;
import java.io.IOException;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.util.Progressable;
/** Local Hadoop file system that fails like Windows without hadoop.dll. */
public class NoLocalFs extends RawLocalFileSystem {
    private static UnsatisfiedLinkError boom(String op, Path p) {
        System.err.println("[NoLocalFs] BLOCKED " + op + " " + p);
        return new UnsatisfiedLinkError("'boolean org.apache.hadoop.io.nativeio.NativeIO$Windows.access0(java.lang.String, int)' (simulated)");
    }
    @Override public FileStatus[] listStatus(Path f) throws IOException { throw boom("listStatus", f); }
    @Override public FileStatus getFileStatus(Path f) throws IOException { throw boom("getFileStatus", f); }
    @Override public FSDataInputStream open(Path f, int b) throws IOException { throw boom("open", f); }
    @Override public FSDataOutputStream create(Path f, FsPermission p, boolean o, int b, short r, long bs, Progressable pr) throws IOException { throw boom("create", f); }
    @Override public FSDataOutputStream append(Path f, int b, Progressable pr) throws IOException { throw boom("append", f); }
    @Override public boolean mkdirs(Path f, FsPermission p) throws IOException { throw boom("mkdirs", f); }
    @Override public boolean rename(Path a, Path b) throws IOException { throw boom("rename", a); }
    @Override public boolean delete(Path f, boolean r) throws IOException { throw boom("delete", f); }
}
"""


@needs_spark
@unittest.skipIf(__import__("sys").platform.startswith("win"), "Simulation only needed on non-Windows; Windows tests native behavior directly")
@unittest.skipUnless(__import__("shutil").which("javac"), "needs a JDK (javac) to build the simulated broken file system")
class NoHadoopFileSystemTests(unittest.TestCase):
    """Runs the pipeline in a FRESH Python/JVM process whose local Hadoop file system throws the
    Windows NativeIO UnsatisfiedLinkError on every operation. Passing means that Windows-safe mode
    never needs Hadoop's local file system (so no winutils.exe / hadoop.dll). Simulation only: it is
    not a real Windows run."""

    @classmethod
    def setUpClass(cls):
        import subprocess
        import sys
        import pyspark
        cls.root = str(Path(__file__).resolve().parent)
        cls.sim = Path(tempfile.mkdtemp(prefix="nofs_"))  # no spaces: it goes into PYSPARK_SUBMIT_ARGS
        (cls.sim / "src" / "sim").mkdir(parents=True)
        (cls.sim / "classes").mkdir()
        src = cls.sim / "src" / "sim" / "NoLocalFs.java"
        src.write_text(NO_FS_JAVA)
        jars = str(Path(pyspark.__file__).parent / "jars" / "*")
        subprocess.run(["javac", "-nowarn", "-cp", jars, "-d", str(cls.sim / "classes"), str(src)],
                       check=True, capture_output=True)
        cls.submit = ("--conf spark.hadoop.fs.file.impl=sim.NoLocalFs "
                      "--conf spark.hadoop.fs.file.impl.disable.cache=true "
                      f"--driver-class-path {cls.sim / 'classes'} pyspark-shell")

    def _run(self, code, io_mode, lake):
        import subprocess
        import sys
        env = dict(os.environ, PYSPARK_SUBMIT_ARGS=self.submit, SPARK_IO_MODE=io_mode,
                   DATA_LAKE_DIR=str(lake), PYTHONPATH=self.root)
        return subprocess.run([sys.executable, "-W", "ignore", "-c", code], env=env,
                              capture_output=True, text=True, timeout=400)

    def test_control_spark_csv_glob_fails_like_on_windows(self):
        raw = Path(tempfile.mkdtemp(prefix="bda cc project "))
        (raw / "a.csv").write_text("city,time\nX,1\n")
        code = (f"from pyspark.sql import SparkSession\n"
                f"s = SparkSession.builder.master('local[1]').config('spark.ui.enabled','false').getOrCreate()\n"
                f"try:\n    s.read.csv('file://{raw.as_posix()}/*.csv', header=True).count(); print('NO ERROR')\n"
                f"except Exception as e:\n    print('FAILED', 'UnsatisfiedLinkError' in str(e))\n")
        res = self._run(code, "spark", raw)
        self.assertIn("FAILED True", res.stdout, res.stdout + res.stderr[-1500:])
        self.assertIn("BLOCKED", res.stderr)  # proves the trap is really installed

    def test_driver_mode_pipeline_never_touches_hadoop_local_fs(self):
        lake = Path(tempfile.mkdtemp(prefix="bda cc project "))
        code = ("from datetime import date\n"
                "from bigdata import ingest\n"
                "from bigdata.spark_job import run_job\n"
                "ingest.ingest('synthetic', years=1.2, n_cities=2, end_date=date(2026, 10, 18))\n"
                "s = run_job()\n"
                "print('RESULT', s['run']['raw_rows'], s['ml']['status'], s['run']['raw_reader'])\n")
        res = self._run(code, "driver", lake)
        self.assertEqual(res.returncode, 0, res.stderr[-3000:])
        self.assertIn("RESULT 21046 ok pandas CSV reader", res.stdout)
        self.assertNotIn("BLOCKED", res.stderr)
        self.assertTrue((lake / "results" / "summary.json").exists())
        self.assertTrue(list((lake / "curated" / "hourly").rglob("*.parquet")))


@needs_spark
class WindowsFallbackPipelineTests(unittest.TestCase):
    """Simulates the exact Windows failure (Spark's Parquet writer raising UnsatisfiedLinkError)
    and checks that the job recovers by itself and still produces full results. Real Spark run."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bda cc project ")
        cls.env = mock.patch.dict(os.environ, {"DATA_LAKE_DIR": cls.tmp, "STORAGE_BACKEND": "local",
                                               "SPARK_IO_MODE": "auto"})
        cls.env.start()
        ingest_mod.ingest("synthetic", years=1.2, n_cities=2, end_date=date(2026, 10, 18))
        from bigdata.spark_job import LakeWriter, run_job
        with mock.patch.object(LakeWriter, "_read_raw_spark", side_effect=RuntimeError(WINDOWS_ERROR)), \
             mock.patch.object(LakeWriter, "_save_spark", side_effect=RuntimeError(WINDOWS_ERROR)):
            cls.summary = run_job()

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()

    def test_job_recovered_and_used_driver_writer(self):
        self.assertIn("driver-side", self.summary["run"]["parquet_writer"])
        self.assertTrue(any("Windows-safe" in n for n in self.summary["notes"]))

    def test_parquet_lake_written_and_partitioned(self):
        hourly = Path(self.tmp) / "curated" / "hourly"
        self.assertTrue((hourly / "city=Hyderabad").is_dir())
        self.assertTrue(list(hourly.rglob("*.parquet")))
        self.assertTrue(list((Path(self.tmp) / "curated" / "daily" / "city=Mumbai").glob("*.parquet")))

    def test_parquet_files_are_readable_and_complete(self):
        import pyarrow.dataset as ds
        table = ds.dataset(Path(self.tmp) / "curated" / "hourly", partitioning="hive").to_table()
        self.assertEqual(table.num_rows, self.summary["run"]["clean_rows"])

    def test_full_analytics_still_ran(self):
        self.assertEqual(self.summary["ml"]["status"], "ok")
        self.assertEqual(len(self.summary["city_stats"]), 2)
        self.assertGreaterEqual(len(self.summary["anomalies"]), 1)
        self.assertTrue((Path(self.tmp) / "results" / "summary.json").exists())


class BigDataApiTests(unittest.TestCase):
    def setUp(self):
        import app as app_module
        app_module.app.testing = True
        self.client = app_module.app.test_client()
        self.summary = {"run": {"engine": "Apache Spark test", "data_mode": "synthetic",
                                "data_source": "SYNTHETIC", "cities": ["A"], "clean_rows": 1,
                                "data_start": "2025-01-01", "data_end": "2025-12-31"},
                        "city_stats": [{"city": "A", "avg_temp": 25}], "anomalies": [],
                        "ml": {"metrics": {}, "best_model": "linear_regression", "next_day_estimates": []}}

    def test_404_with_hint_when_pipeline_not_run(self):
        with mock.patch.dict(os.environ, {"STORAGE_BACKEND": "local", "DATA_LAKE_DIR": tempfile.mkdtemp()}):
            r = self.client.get("/api/bigdata/summary")
            self.assertEqual(r.status_code, 404)
            self.assertIn("run_pipeline", r.get_json()["hint"])
            self.assertFalse(self.client.get("/api/bigdata/status").get_json()["results_available"])

    def test_summary_from_local_lake(self):
        lake = tempfile.mkdtemp()
        (Path(lake) / "results").mkdir()
        (Path(lake) / "results" / "summary.json").write_text(json.dumps(self.summary))
        with mock.patch.dict(os.environ, {"STORAGE_BACKEND": "local", "DATA_LAKE_DIR": lake}):
            r = self.client.get("/api/bigdata/summary")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.get_json()["storage"]["backend"], "local")
            self.assertTrue(self.client.get("/api/bigdata/status").get_json()["results_available"])

    def test_web_app_reads_results_from_s3(self):
        from moto import mock_aws
        lake = tempfile.mkdtemp()
        (Path(lake) / "results").mkdir()
        (Path(lake) / "results" / "summary.json").write_text(json.dumps(self.summary))
        env = {"STORAGE_BACKEND": "s3", "S3_BUCKET": "weather-lake", "S3_ENDPOINT_URL": "",
               "AWS_REGION": "us-east-1", "AWS_ACCESS_KEY_ID": "testing",
               "AWS_SECRET_ACCESS_KEY": "testing", "DATA_LAKE_DIR": lake}
        with mock.patch.dict(os.environ, env), mock_aws():
            from bigdata.storage import get_store
            get_store().publish()
            r = self.client.get("/api/bigdata/summary")
            self.assertEqual(r.status_code, 200)
            body = r.get_json()
            self.assertEqual(body["storage"]["backend"], "s3")
            self.assertEqual(body["run"]["engine"], "Apache Spark test")

    def test_unreachable_storage_is_502_not_crash(self):
        env = {"STORAGE_BACKEND": "s3", "S3_BUCKET": "b", "S3_ENDPOINT_URL": "http://127.0.0.1:9",
               "AWS_ACCESS_KEY_ID": "x", "AWS_SECRET_ACCESS_KEY": "y", "AWS_REGION": "us-east-1"}
        with mock.patch.dict(os.environ, env):
            r = self.client.get("/api/bigdata/summary")
            self.assertEqual(r.status_code, 502)
            self.assertIn("not reachable", r.get_json()["error"])

    def test_web_app_reads_results_from_supabase(self):
        env = {
            "STORAGE_BACKEND": "supabase",
            "SUPABASE_URL": "https://testproject.supabase.co",
            "SUPABASE_KEY": "test-anon-key-123",
            "SUPABASE_BUCKET": "weather-data-lake",
        }
        mock_client = MockSupabaseClient()
        mock_client.storage.from_("weather-data-lake").upload(
            "results/summary.json",
            json.dumps(self.summary).encode("utf-8")
        )
        with mock.patch.dict(os.environ, env), mock.patch("supabase.create_client", return_value=mock_client):
            r = self.client.get("/api/bigdata/summary")
            self.assertEqual(r.status_code, 200)
            body = r.get_json()
            self.assertEqual(body["storage"]["backend"], "supabase")
            self.assertEqual(body["storage"]["bucket"], "weather-data-lake")
            self.assertEqual(body["run"]["engine"], "Apache Spark test")

            status_res = self.client.get("/api/bigdata/status")
            self.assertEqual(status_res.status_code, 200)
            status_body = status_res.get_json()
            self.assertTrue(status_body["results_available"])
            self.assertEqual(status_body["storage"]["backend"], "supabase")

    def test_unreachable_supabase_is_502_not_crash(self):
        env = {
            "STORAGE_BACKEND": "supabase",
            "SUPABASE_URL": "https://testproject.supabase.co",
            "SUPABASE_KEY": "test-anon-key-123",
            "SUPABASE_BUCKET": "weather-data-lake",
        }
        class ErrorBucketProxy:
            def download(self, *args, **kwargs):
                raise ConnectionError("Network error: failed to reach Supabase storage")
            def exists(self, *args, **kwargs):
                raise ConnectionError("Network error: failed to reach Supabase storage")

        class ErrorClient:
            def __init__(self):
                self.storage = self
            def from_(self, bucket_name):
                return ErrorBucketProxy()

        with mock.patch.dict(os.environ, env), mock.patch("supabase.create_client", return_value=ErrorClient()):
            r = self.client.get("/api/bigdata/summary")
            self.assertEqual(r.status_code, 502)
            self.assertIn("not reachable", r.get_json()["error"])
            self.assertIn("SUPABASE_URL", r.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
