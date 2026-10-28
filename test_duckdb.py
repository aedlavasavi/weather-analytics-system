"""Tests for the default DuckDB + scikit-learn pipeline (no Java, no Spark needed).
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
from bigdata.duckdb_job import _check_headers, _clear_dir, _sql_path, run_job

try:
    import duckdb  # noqa: F401
    HAVE_DUCKDB = True
except Exception:  # noqa: BLE001  (not installed, or blocked by a Windows policy)
    HAVE_DUCKDB = False

HEADER = "city,country,latitude,longitude,time,temp_c,humidity_pct,precip_mm,wind_kmh,pressure_hpa,cloud_pct\n"
ROOT = Path(__file__).resolve().parent


@unittest.skipUnless(HAVE_DUCKDB, "duckdb not available")
class DuckDbPipelineTests(unittest.TestCase):
    """Full pipeline on synthetic data inside a folder whose name contains spaces (like 'bda cc project')."""

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
        self.assertEqual(q["raw_rows"] - q["invalid_timestamp_or_city_rows"] - q["duplicates_removed"],
                         q["rows_after_dedup"])
        self.assertEqual(q["rows_after_dedup"] - q["rows_dropped_no_temperature"], q["clean_rows"])

    def test_parquet_lake_partitioned_and_complete(self):
        import pyarrow.dataset as ds
        hourly = Path(self.tmp) / "curated" / "hourly"
        self.assertTrue((hourly / "city=Hyderabad").is_dir())
        table = ds.dataset(hourly, partitioning="hive").to_table()
        self.assertEqual(table.num_rows, self.summary["run"]["clean_rows"])
        self.assertTrue(list((Path(self.tmp) / "curated" / "daily").rglob("*.parquet")))

    def test_sql_results(self):
        self.assertEqual(len(self.summary["city_stats"]), 3)
        for c in self.summary["city_stats"]:
            self.assertTrue(10 < c["avg_temp"] < 40)
        self.assertGreater(len(self.summary["monthly"]), 30)
        self.assertEqual(len(self.summary["correlations"]), 15)
        self.assertEqual(len(self.summary["extreme_rain_days"]), 10)

    def test_window_function_results(self):
        recent = self.summary["recent_daily"]
        self.assertEqual(len(recent), 3 * 60)
        self.assertTrue(all(r["roll7"] is not None for r in recent))
        self.assertGreaterEqual(len(self.summary["anomalies"]), 1)
        self.assertTrue(all(abs(a["z"]) >= 3.0 for a in self.summary["anomalies"]))

    def test_machine_learning(self):
        ml = self.summary["ml"]
        self.assertEqual(ml["status"], "ok")
        for k in ("persistence_baseline", "linear_regression", "random_forest"):
            self.assertTrue(0 < ml["metrics"][k]["rmse"] < 5)
        self.assertIn(ml["best_model"], ("linear_regression", "random_forest"))
        self.assertEqual(len(ml["next_day_estimates"]), 3)
        self.assertEqual(len(ml["feature_importance"]), len(ml["features"]))

    def test_summary_json_matches_dashboard_contract(self):
        on_disk = json.loads((Path(self.tmp) / "results" / "summary.json").read_text(encoding="utf-8"))
        for key in ("run", "quality", "city_stats", "monthly", "recent_daily", "correlations",
                    "anomalies", "anomaly_rule", "extreme_rain_days", "ml", "notes"):
            self.assertIn(key, on_disk)
        for key in ("engine", "data_mode", "data_source", "data_start", "data_end", "cities",
                    "raw_rows", "clean_rows", "daily_rows", "duration_seconds", "finished_at"):
            self.assertIn(key, on_disk["run"])
        self.assertIn("DuckDB", on_disk["run"]["engine"])
        self.assertTrue(any("SYNTHETIC" in n for n in on_disk["notes"]))

    def test_rerun_does_not_duplicate_rows(self):
        again = run_job()
        import pyarrow.dataset as ds
        table = ds.dataset(Path(self.tmp) / "curated" / "hourly", partitioning="hive").to_table()
        self.assertEqual(table.num_rows, again["run"]["clean_rows"])


@unittest.skipUnless(HAVE_DUCKDB, "duckdb not available")
class MessyCsvTests(unittest.TestCase):
    """Real-world dirty input: BOM, Windows line endings, blanks, text in number columns, NaN."""

    def _lake(self, text, header=HEADER, encoding="utf-8", newline=""):
        d = Path(tempfile.mkdtemp(prefix="messy lake "))
        (d / "raw").mkdir()
        with open(d / "raw" / "a.csv", "w", encoding=encoding, newline=newline) as fh:
            fh.write(header + text)
        return d

    def _run(self, lake):
        with mock.patch.dict(os.environ, {"DATA_LAKE_DIR": str(lake)}):
            return run_job()

    def test_bom_crlf_blank_text_nan_and_duplicates(self):
        rows = []
        for h in range(24):  # one full day so it survives the >=20 hours rule
            rows.append(f"Pune,India,18.5,73.8,2026-01-01T{h:02d}:00,25.{h % 10},60,0,10,1010,40\r\n")
        rows[1] = "Pune,India,18.5,73.8,2026-01-01T01:00,,61,0,11,1010,41\r\n"        # blank temperature
        rows[2] = "Pune,India,18.5,73.8,2026-01-01T02:00,26.0,abc,0,12,1010,42\r\n"   # text in a number column
        rows[3] = "Pune,India,18.5,73.8,2026-01-01T03:00,26.5,62,0.5,13,NaN,43\r\n"   # NaN literal
        rows.append(rows[5])                                                           # exact duplicate
        rows.append("Pune,India,18.5,73.8,not-a-time,25,60,0,10,1010,40\r\n")          # bad timestamp
        rows.append(",India,18.5,73.8,2026-01-02T00:00,25,60,0,10,1010,40\r\n")        # missing city
        rows.append("Pune,India,18.5,73.8,2026-01-02T01:00,999,60,0,10,1010,40\r\n")   # impossible temperature
        s = self._run(self._lake("".join(rows), encoding="utf-8-sig"))
        q = s["quality"]
        self.assertEqual(q["raw_rows"], 28)
        self.assertEqual(q["invalid_timestamp_or_city_rows"], 2)
        self.assertEqual(q["duplicates_removed"], 1)
        self.assertEqual(q["missing_values"]["temp_c"], 1)
        self.assertEqual(q["out_of_range_set_to_null"]["temp_c"], 1)
        self.assertEqual(q["missing_values"]["humidity_pct"], 1)      # "abc" became NULL
        self.assertEqual(q["missing_values"]["pressure_hpa"], 1)      # "NaN" became NULL
        self.assertEqual(q["rows_dropped_no_temperature"], 2)        # blank + 999
        self.assertEqual(q["clean_rows"], 23)                        # 24 hourly rows - 1 blank temperature
        self.assertEqual(s["ml"]["status"], "skipped")               # too little data: reported, not crashed

    def test_missing_column_is_a_clear_error(self):
        lake = self._lake("Pune,India,1,2,2026-01-01T00:00,25\n", header="city,country,latitude,longitude,time,temp_c\n")
        with self.assertRaises(ValueError) as ctx:
            self._run(lake)
        self.assertIn("humidity_pct", str(ctx.exception))

    def test_no_csv_files_is_a_clear_error(self):
        lake = Path(tempfile.mkdtemp())
        (lake / "raw").mkdir()
        with self.assertRaises(FileNotFoundError):
            self._run(lake)

    def test_helpers(self):
        self.assertEqual(_sql_path("C:\\Users\\O'Brien\\lake"), "'C:\\Users\\O''Brien\\lake'")
        d = Path(tempfile.mkdtemp()) / "x" / "city=A"
        d.mkdir(parents=True)
        (d / "old.parquet").write_bytes(b"PAR1")
        with mock.patch("bigdata.duckdb_job.shutil.rmtree"):  # simulate a file locked by OneDrive
            with self.assertRaises(RuntimeError) as ctx:
                _clear_dir(d.parent)
        self.assertIn("OneDrive", str(ctx.exception))
        _check_headers([])  # no files: nothing to check, no error


@unittest.skipUnless(HAVE_DUCKDB, "duckdb not available")
class CommandLineTests(unittest.TestCase):
    """The exact commands users type."""

    def test_pipeline_command_and_verify_script(self):
        lake = tempfile.mkdtemp(prefix="bda cc project ")
        env = dict(os.environ, DATA_LAKE_DIR=lake, STORAGE_BACKEND="local", PYTHONPATH=str(ROOT),
                   DB_PATH=os.path.join(lake, "t.db"))
        env.pop("PIPELINE_ENGINE", None)
        run = subprocess.run([sys.executable, "-W", "ignore", "-m", "bigdata.run_pipeline", "--engine", "duckdb",
                              "--mode", "synthetic", "--years", "1.2", "--cities", "2"],
                             cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr[-2000:])
        self.assertIn("raw rows from 2 cities", run.stdout)
        self.assertIn("DuckDB", run.stdout)
        verify = subprocess.run([sys.executable, "-W", "ignore", "verify_results.py"], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(verify.returncode, 0, verify.stdout + verify.stderr[-2000:])
        self.assertIn("ALL CHECKS PASSED", verify.stdout)
        self.assertNotIn("FAIL", verify.stdout)

    def test_verify_without_results_fails_clearly(self):
        lake = tempfile.mkdtemp()
        env = dict(os.environ, DATA_LAKE_DIR=lake, STORAGE_BACKEND="local", PYTHONPATH=str(ROOT))
        res = subprocess.run([sys.executable, "-W", "ignore", "verify_results.py"], cwd=ROOT, env=env,
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(res.returncode, 1)
        self.assertIn("run_pipeline", res.stdout)


if __name__ == "__main__":
    unittest.main()
