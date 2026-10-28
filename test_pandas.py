"""Tests for the default pandas pipeline (needs only pandas + numpy; no DuckDB, Java or Spark).
Run:  python -m unittest -v
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from bigdata import ingest as ingest_mod
from bigdata.pandas_job import run_job

HEADER = "city,country,latitude,longitude,time,temp_c,humidity_pct,precip_mm,wind_kmh,pressure_hpa,cloud_pct\n"
ROOT = Path(__file__).resolve().parent


class PandasPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bda cc project ")
        cls.env = mock.patch.dict(os.environ, {"DATA_LAKE_DIR": cls.tmp, "STORAGE_BACKEND": "local"})
        cls.env.start()
        cls.manifest = ingest_mod.ingest("synthetic", years=1.5, n_cities=3, end_date=date(2026, 10, 18))
        cls.summary = run_job()

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()

    def test_row_accounting(self):
        q = self.summary["quality"]
        self.assertEqual(q["raw_rows"], self.manifest["rows"])
        self.assertGreater(q["duplicates_removed"], 0)
        self.assertGreater(q["missing_values"]["temp_c"], 0)
        self.assertGreater(q["out_of_range_set_to_null"]["humidity_pct"], 0)
        self.assertEqual(q["raw_rows"] - q["invalid_timestamp_or_city_rows"] - q["duplicates_removed"], q["rows_after_dedup"])
        self.assertEqual(q["rows_after_dedup"] - q["rows_dropped_no_temperature"], q["clean_rows"])

    def test_parquet_lake(self):
        import pyarrow.dataset as ds
        import gc
        table = ds.dataset(Path(self.tmp) / "curated" / "hourly", partitioning="hive").to_table()
        self.assertEqual(table.num_rows, self.summary["run"]["clean_rows"])
        table = None
        gc.collect()

    def test_results(self):
        s = self.summary
        self.assertEqual(len(s["city_stats"]), 3)
        self.assertEqual(len(s["correlations"]), 15)
        self.assertEqual(len(s["recent_daily"]), 180)
        self.assertTrue(all(abs(a["z"]) >= 3 for a in s["anomalies"]))
        self.assertTrue(all(len(a["date"]) == 10 for a in s["anomalies"]))
        self.assertEqual(s["ml"]["status"], "ok")
        self.assertEqual(len(s["ml"]["next_day_estimates"]), 3)
        self.assertNotIn("DuckDB", s["run"]["engine"])
        json.dumps(s)

    def test_rerun_does_not_duplicate_rows(self):
        import gc
        gc.collect()
        again = run_job()
        import pyarrow.dataset as ds
        table = ds.dataset(Path(self.tmp) / "curated" / "hourly", partitioning="hive").to_table()
        self.assertEqual(table.num_rows, again["run"]["clean_rows"])
        table = None
        gc.collect()


class BlockedLibraryTests(unittest.TestCase):
    """Windows Application Control can block compiled modules: the job must still finish, and say so."""

    def test_without_pyarrow_and_sklearn(self):
        tmp = tempfile.mkdtemp(prefix="blocked ")
        with mock.patch.dict(os.environ, {"DATA_LAKE_DIR": tmp}):
            ingest_mod.ingest("synthetic", years=1.2, n_cities=2, end_date=date(2026, 10, 18))
            blocked = {m: None for m in ("pyarrow", "sklearn", "sklearn.ensemble", "sklearn.linear_model")}
            with mock.patch.dict(sys.modules, blocked):
                s = run_job()
        self.assertEqual(s["ml"]["status"], "ok")
        self.assertEqual(s["ml"]["best_model"], "linear_regression")
        self.assertIn("gzip", s["run"]["parquet_writer"])
        self.assertTrue(list(Path(tmp, "curated").rglob("*.csv.gz")))
        self.assertTrue(any("gzip CSV" in n for n in s["notes"]))


class MessyCsvTests(unittest.TestCase):
    def test_bom_crlf_blank_text_nan_and_duplicates(self):
        d = Path(tempfile.mkdtemp(prefix="messy lake "))
        (d / "raw").mkdir()
        rows = [f"Pune,India,18.5,73.8,2026-01-01T{h:02d}:00,25.{h % 10},60,0,10,1010,40\r\n" for h in range(24)]
        rows[1] = "Pune,India,18.5,73.8,2026-01-01T01:00,,61,0,11,1010,41\r\n"
        rows[2] = "Pune,India,18.5,73.8,2026-01-01T02:00,26.0,abc,0,12,1010,42\r\n"
        rows[3] = "Pune,India,18.5,73.8,2026-01-01T03:00,26.5,62,0.5,13,NaN,43\r\n"
        rows.append(rows[5])
        rows.append("Pune,India,18.5,73.8,not-a-time,25,60,0,10,1010,40\r\n")
        rows.append(",India,18.5,73.8,2026-01-02T00:00,25,60,0,10,1010,40\r\n")
        rows.append("Pune,India,18.5,73.8,2026-01-02T01:00,999,60,0,10,1010,40\r\n")
        with open(d / "raw" / "a.csv", "w", encoding="utf-8-sig", newline="") as fh:
            fh.write(HEADER + "".join(rows))
        with mock.patch.dict(os.environ, {"DATA_LAKE_DIR": str(d)}):
            s = run_job()
        q = s["quality"]
        self.assertEqual((q["raw_rows"], q["invalid_timestamp_or_city_rows"], q["duplicates_removed"]), (28, 2, 1))
        self.assertEqual(q["missing_values"]["temp_c"], 1)
        self.assertEqual(q["out_of_range_set_to_null"]["temp_c"], 1)
        self.assertEqual(q["missing_values"]["humidity_pct"], 1)
        self.assertEqual(q["missing_values"]["pressure_hpa"], 1)
        self.assertEqual(q["rows_dropped_no_temperature"], 2)
        self.assertEqual(q["clean_rows"], 23)
        self.assertEqual(s["ml"]["status"], "skipped")

    def test_clear_errors(self):
        d = Path(tempfile.mkdtemp())
        (d / "raw").mkdir()
        with mock.patch.dict(os.environ, {"DATA_LAKE_DIR": str(d)}):
            with self.assertRaises(FileNotFoundError):
                run_job()
            (d / "raw" / "a.csv").write_text("city,temp_c\nA,1\n")
            with self.assertRaises(ValueError) as ctx:
                run_job()
        self.assertIn("humidity_pct", str(ctx.exception))


class CommandLineTests(unittest.TestCase):
    def test_default_command_and_verify(self):
        lake = tempfile.mkdtemp(prefix="bda cc project ")
        env = dict(os.environ, DATA_LAKE_DIR=lake, STORAGE_BACKEND="local", PYTHONPATH=str(ROOT),
                   DB_PATH=os.path.join(lake, "t.db"))
        env.pop("PIPELINE_ENGINE", None)
        run = subprocess.run([sys.executable, "-W", "ignore", "-m", "bigdata.run_pipeline", "--engine", "pandas", "--mode", "synthetic",
                              "--years", "1.2", "--cities", "2"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr[-2000:])
        self.assertIn("pandas batch job", run.stdout)
        v = subprocess.run([sys.executable, "-W", "ignore", "verify_results.py"], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr[-2000:])
        self.assertIn("ALL CHECKS PASSED", v.stdout)


class SparkFallbackRuleTests(unittest.TestCase):
    def test_only_startup_problems_trigger_fallback(self):
        from bigdata.run_pipeline import _spark_cannot_start
        self.assertTrue(_spark_cannot_start(ImportError("No module named pyspark")))
        self.assertTrue(_spark_cannot_start(RuntimeError("[JAVA_GATEWAY_EXITED] Java gateway process exited")))
        self.assertTrue(_spark_cannot_start(RuntimeError("JAVA_HOME is not set")))
        self.assertFalse(_spark_cannot_start(ValueError("humidity_pct column missing")))
        self.assertFalse(_spark_cannot_start(RuntimeError("Could not remove old Parquet files")))


if __name__ == "__main__":
    unittest.main()
