"""Flask blueprint that serves the results of the Spark batch job.

The dashboard never talks to Spark directly: the batch job writes
results/summary.json to the data lake (local folder or S3/MinIO) and this
blueprint reads it from there. That decoupling is what lets the web app and the
big-data job run as separate containers/services in the cloud.
"""
import logging

from flask import Blueprint, jsonify

from bigdata import config
from bigdata.storage import get_store

log = logging.getLogger("weather-app.bigdata")
bp = Blueprint("bigdata", __name__, url_prefix="/api/bigdata")

HINT = "Run the pipeline first:  python -m bigdata.run_pipeline --mode synthetic"


def _storage_error_message(store):
    backend = getattr(store, "backend", "storage")
    if backend == "supabase":
        return "Object storage is not reachable. Check SUPABASE_URL and SUPABASE_KEY."
    if backend == "s3":
        return "Object storage is not reachable. Check S3_ENDPOINT_URL and credentials."
    return "Object storage is not reachable. Check storage configuration and permissions."


def load_summary():
    """Return the Spark results dict, or None if the pipeline has not been run yet."""
    return get_store().get_json(config.RESULTS_KEY)


@bp.route("/status")
def status():
    store = get_store()
    info = store.describe()
    try:
        summary = store.get_json(config.RESULTS_KEY)
    except Exception as exc:  # noqa: BLE001
        log.warning("Storage not reachable: %s", exc)
        return jsonify({"storage": info, "results_available": False,
                        "error": _storage_error_message(store)}), 502
    return jsonify({"storage": info, "results_available": summary is not None,
                    "run": summary["run"] if summary else None})


@bp.route("/summary")
def summary():
    store = get_store()
    try:
        data = load_summary()
    except Exception as exc:  # noqa: BLE001
        log.warning("Storage not reachable: %s", exc)
        return jsonify({"error": _storage_error_message(store)}), 502
    if data is None:
        return jsonify({"error": "No big data results found yet.", "hint": HINT}), 404
    data["storage"] = store.describe()
    return jsonify(data)
