"""One command for the whole pipeline:

    python -m bigdata.run_pipeline                       # synthetic demo data, Apache Spark engine (default)
    python -m bigdata.run_pipeline --mode live --years 3 # real Open-Meteo history (internet needed)
    python -m bigdata.run_pipeline --engine duckdb       # optional: DuckDB SQL engine (pip install duckdb)
    python -m bigdata.run_pipeline --engine pandas       # no-Java alternative (also used automatically if Spark cannot start)
"""
import argparse
import logging
import os
import sys

from dotenv import load_dotenv


def _spark_cannot_start(exc):
    """True only for 'Spark itself is not usable here' (not installed, Java missing/wrong), never for job bugs."""
    if isinstance(exc, ImportError):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(m in text for m in ("java_gateway_exited", "java gateway process exited", "java_home",
                                   "java not found", "no such file or directory: 'java'",
                                   "unsupportedclassversion", "unsupported class file major version"))


def main(argv=None):
    load_dotenv()
    from . import config  # imported after load_dotenv so .env values are visible
    from .ingest import IngestError, ingest

    p = argparse.ArgumentParser(description="Ingest -> big data batch job -> publish to object storage")
    p.add_argument("--mode", choices=["live", "synthetic"], default="synthetic",
                   help="live = real Open-Meteo history (internet needed); synthetic = fake demo data")
    p.add_argument("--engine", choices=["pandas", "duckdb", "spark"],
                   default=os.getenv("PIPELINE_ENGINE", "spark").strip().lower(),
                   help="spark = default, Apache Spark (needs Java 17/21); pandas = no Java needed; duckdb = DuckDB SQL")
    p.add_argument("--no-fallback", action="store_true",
                   help="do not switch to the pandas engine if Spark cannot start (Java missing etc.)")
    p.add_argument("--years", type=float, default=2, help="years of hourly history per city")
    p.add_argument("--cities", type=int, default=10, help="number of cities (max 10)")
    p.add_argument("--publish", action="store_true",
                   help="upload the data lake to object storage (done automatically if STORAGE_BACKEND=s3)")
    p.add_argument("--download", action="store_true",
                   help="download the data lake from object storage before running the job")
    p.add_argument("--skip-ingest", action="store_true",
                   help="skip ingestion and process existing raw data in the data lake")
    args = p.parse_args(argv)
    if args.engine not in ("pandas", "duckdb", "spark"):
        p.error("PIPELINE_ENGINE must be 'pandas', 'duckdb' or 'spark'")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from .storage import get_store
    store = get_store()

    if args.download:
        if config.storage_backend() == "local":
            print("Note: STORAGE_BACKEND is 'local'; nothing to download from cloud.")
        else:
            print(f"Downloading data lake from object storage ({store.describe()}) ...")
            try:
                count = store.download_lake()
                print(f"      Downloaded {count} file(s) to {config.lake_dir()}")
            except Exception as exc:  # noqa: BLE001
                print(f"Download failed: {exc}", file=sys.stderr)
                return 4

    if not args.skip_ingest:
        print(f"[1/3] Ingesting {args.cities} cities, {args.years} year(s), mode={args.mode} ...")
        try:
            manifest = ingest(args.mode, args.years, args.cities)
        except IngestError as exc:
            print(f"Ingestion failed: {exc}", file=sys.stderr)
            return 1
        print(f"      {manifest['rows']:,} raw rows from {len(manifest['cities'])} cities")
        if manifest["failed"]:
            print(f"      WARNING: failed cities: {manifest['failed']}")
    else:
        print("[1/3] Skipping ingestion (--skip-ingest); using existing data in the lake.")

    summary = None
    if args.engine == "spark":
        print("[2/3] Running Apache Spark batch job (clean -> Parquet -> Spark SQL -> MLlib) ...")
        try:
            from .spark_job import run_job
            summary = run_job()
        except Exception as exc:  # noqa: BLE001
            if args.no_fallback or not _spark_cannot_start(exc):
                raise  # a real error in the job: show it, do not hide it
            print(f"\n      !! Apache Spark could not start on this PC: {str(exc).splitlines()[0][:200]}\n"
                  "      !! Install Java 17 (Temurin) and reopen CMD, or ignore this: switching to the pandas engine.\n",
                  file=sys.stderr)
            args.engine = "pandas"
    if summary is None:
        if args.engine == "duckdb":
            print("[2/3] Running DuckDB batch job (clean -> Parquet -> SQL -> machine learning) ...")
            from .duckdb_job import run_job
        else:
            print("[2/3] Running pandas batch job (clean -> Parquet -> analysis -> machine learning) ...")
            from .pandas_job import run_job
        try:
            summary = run_job()
        except ImportError as exc:
            print(f"Could not load the '{args.engine}' engine on this PC: {exc}\n"
                  "Use the pandas engine instead:  python -m bigdata.run_pipeline --engine pandas", file=sys.stderr)
            return 3
    run = summary["run"]
    print(f"      {run['clean_rows']:,} clean rows in {run['duration_seconds']} s using {run['engine']}")

    from .storage import get_store
    store = get_store()
    if args.publish or config.storage_backend() in ("s3", "supabase"):
        print(f"[3/3] Publishing data lake to object storage ({store.describe()}) ...")
        try:
            print("      ", store.publish())
        except Exception as exc:  # noqa: BLE001
            print(f"Publishing failed: {exc}", file=sys.stderr)
            return 2
    else:
        print("[3/3] Storage backend is 'local': results stay in", config.lake_dir())
    print("Done. Check the results with:  python verify_results.py   then start the dashboard:  python app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
