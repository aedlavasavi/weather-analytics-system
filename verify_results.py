"""Checks the pipeline output and the dashboard API in one go.

    python verify_results.py

Run it after:  python -m bigdata.run_pipeline
Prints PASS / FAIL for every check and exits with code 0 only if everything passed.
"""
import sys

from dotenv import load_dotenv

load_dotenv()

from bigdata import config  # noqa: E402
from bigdata.storage import get_store  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def main():
    try:
        summary = get_store().get_json(config.RESULTS_KEY)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  could not read results: {exc}")
        return 1
    if summary is None:
        print("FAIL  no results found. Run first:  python -m bigdata.run_pipeline")
        return 1

    run, q, ml = summary["run"], summary["quality"], summary["ml"]
    print(f"Engine: {run['engine']} | data: {run['data_mode']} | {run['data_start']} to {run['data_end']}\n")

    check("raw rows were ingested", run["raw_rows"] > 0, f"{run['raw_rows']:,} raw rows")
    check("cleaning kept most rows", 0.9 * run["raw_rows"] <= run["clean_rows"] <= run["raw_rows"],
          f"{run['clean_rows']:,} clean rows")
    check("row accounting is consistent",
          q["raw_rows"] - q["invalid_timestamp_or_city_rows"] - q["duplicates_removed"] == q["rows_after_dedup"]
          and q["rows_after_dedup"] - q["rows_dropped_no_temperature"] == q["clean_rows"],
          f"{q['duplicates_removed']} duplicates removed, {q['rows_dropped_no_temperature']} rows without temperature dropped")
    check("city statistics exist for every city", len(summary["city_stats"]) == len(run["cities"]) > 0,
          f"{len(run['cities'])} cities")
    check("monthly statistics exist", len(summary["monthly"]) > 0, f"{len(summary['monthly'])} rows")
    check("recent daily trend exists", len(summary["recent_daily"]) > 0)
    check("correlation matrix has 15 pairs", len(summary["correlations"]) == 15)
    check("anomaly detection ran", isinstance(summary["anomalies"], list), f"{len(summary['anomalies'])} unusual days")
    check("extreme rain analysis ran", len(summary["extreme_rain_days"]) > 0)
    check("forecast model trained", ml.get("status") == "ok", ml.get("reason", ""))
    if ml.get("status") == "ok":
        m = ml["metrics"]
        check("forecast metrics are sane", all(0 < m[k]["rmse"] < 10 for k in m),
              ", ".join(f"{k} RMSE {m[k]['rmse']}" for k in m))
        check("next-day estimates exist for every city", len(ml["next_day_estimates"]) == len(run["cities"]))
        check("feature importance exists", len(ml["feature_importance"]) > 0)

    # Dashboard API (no browser or internet needed: uses Flask's built-in test client)
    try:
        import app as app_module
        client = app_module.app.test_client()
        r = client.get("/api/health")
        check("API /api/health", r.status_code == 200 and r.get_json()["status"] == "ok")
        r = client.get("/api/bigdata/summary")
        check("API /api/bigdata/summary", r.status_code == 200 and r.get_json()["run"]["engine"] == run["engine"])
        check("dashboard page loads", client.get("/").status_code == 200)
        check("Chart.js is served locally (works offline)", client.get("/static/chart.umd.js").status_code == 200)
    except Exception as exc:  # noqa: BLE001
        check("dashboard API", False, str(exc))

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    if all(results):
        print("ALL CHECKS PASSED. Start the dashboard:  python app.py   (open http://127.0.0.1:5000)")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
