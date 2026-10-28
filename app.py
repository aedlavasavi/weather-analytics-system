"""Cloud-Based Weather Data Analytics System - Flask backend.

Run locally:  python app.py   ->  http://127.0.0.1:5000
"""
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

import weather_service as ws
from analytics import run_analytics
from bigdata_api import bp as bigdata_bp, load_summary

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "weather.db"))
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "600"))  # 10 minutes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("weather-app")

app = Flask(__name__)
app.register_blueprint(bigdata_bp)  # /api/bigdata/* (Spark batch results)

_cache = {}  # key -> (timestamp, value)


# ---------- storage (SQLite stands in for a cloud database) ----------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS observations (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   city TEXT, country TEXT, observed_at TEXT, collected_at TEXT,
                   temp_c REAL, humidity_pct REAL, wind_kmh REAL,
                   pressure_hpa REAL, precip_mm REAL)"""
        )


def store_observation(place, current):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO observations (city, country, observed_at, collected_at, temp_c,"
                " humidity_pct, wind_kmh, pressure_hpa, precip_mm) VALUES (?,?,?,?,?,?,?,?,?)",
                (place["name"], place["country"], current.get("time"),
                 datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 current.get("temperature_2m"), current.get("relative_humidity_2m"),
                 current.get("wind_speed_10m"), current.get("pressure_msl"),
                 current.get("precipitation")),
            )
    except sqlite3.Error:
        log.exception("Could not store observation")  # storage is best-effort


# ---------- helpers ----------
def get_weather_bundle(city):
    """Geocode + fetch, with a short in-memory cache to respect API limits."""
    key = city.strip().lower()
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    place = ws.geocode(city)
    payload = ws.fetch_weather(place["latitude"], place["longitude"])
    bundle = {"place": place, "payload": payload}
    _cache[key] = (time.time(), bundle)
    store_observation(place, payload["current"])
    return bundle


def current_summary(bundle):
    cur = bundle["payload"]["current"]
    units = bundle["payload"].get("current_units", {})
    return {
        "time": cur.get("time"),
        "timezone": bundle["payload"].get("timezone"),
        "temperature_c": cur.get("temperature_2m"),
        "feels_like_c": cur.get("apparent_temperature"),
        "humidity_pct": cur.get("relative_humidity_2m"),
        "wind_kmh": cur.get("wind_speed_10m"),
        "wind_direction_deg": cur.get("wind_direction_10m"),
        "pressure_hpa": cur.get("pressure_msl"),
        "cloud_cover_pct": cur.get("cloud_cover"),
        "precipitation_mm": cur.get("precipitation"),
        "condition": ws.describe_code(cur.get("weather_code")),
        "sunrise": (bundle["payload"].get("daily", {}).get("sunrise") or [None])[0],
        "sunset": (bundle["payload"].get("daily", {}).get("sunset") or [None])[0],
        "units": units,
    }


def error(message, status):
    return jsonify({"error": message}), status


# ---------- routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/weather")
def weather():
    city = request.args.get("city", "").strip()
    if not city:
        return error("Please provide a city, e.g. /api/weather?city=Hyderabad", 400)
    try:
        bundle = get_weather_bundle(city)
    except ws.WeatherServiceError as exc:
        status = 404 if "not found" in str(exc) else 502
        return error(str(exc), status)
    return jsonify({
        "location": bundle["place"],
        "source": "Open-Meteo (open-meteo.com)",
        "current": current_summary(bundle),
    })


@app.route("/api/analytics")
def analytics():
    city = request.args.get("city", "").strip()
    if not city:
        return error("Please provide a city, e.g. /api/analytics?city=Hyderabad", 400)
    try:
        bundle = get_weather_bundle(city)
        report = run_analytics(bundle["payload"])
    except ws.WeatherServiceError as exc:
        status = 404 if "not found" in str(exc) else 502
        return error(str(exc), status)
    except Exception:  # noqa: BLE001 - keep the API alive, log details
        log.exception("Analytics failed")
        return error("Analytics failed unexpectedly. Check the server log.", 500)
    return jsonify({"location": bundle["place"], "report": report})


@app.route("/api/history")
def history():
    city = request.args.get("city", "").strip()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM observations WHERE lower(city)=lower(?) ORDER BY id DESC LIMIT 100",
            (city,),
        ).fetchall()
    return jsonify({"city": city, "stored_observations": [dict(r) for r in rows]})


init_db()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
