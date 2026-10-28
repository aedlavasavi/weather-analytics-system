"""Stage 1 - Data ingestion into the RAW zone of the data lake.

Two modes:
  live       real hourly history from the Open-Meteo Archive API (needs internet)
  synthetic  generated hourly data that imitates Indian city climates. It is
             FAKE data for demos/tests and is labelled as such in every result.
"""
import argparse
import json
import logging
import re
import time as _time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config

log = logging.getLogger("bigdata.ingest")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,pressure_msl,cloud_cover"
COLUMNS = ["city", "country", "latitude", "longitude", "time",
           "temp_c", "humidity_pct", "precip_mm", "wind_kmh", "pressure_hpa", "cloud_pct"]


class IngestError(Exception):
    pass


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _ar1(n, phi, sigma, rng):
    noise = rng.normal(0, sigma, n)
    out = np.empty(n)
    out[0] = noise[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + noise[i]
    return out


def synthetic_city_frame(city, start, end, seed):
    """Hourly synthetic weather: seasons + day/night cycle + monsoon + heatwaves + dirty rows."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    n = len(times)
    n_days = n // 24
    doy = times.dayofyear.to_numpy()
    hour = times.hour.to_numpy()

    seasonal = city["amp_t"] * np.sin(2 * np.pi * (doy - 110) / 365.0)
    diurnal = 4.0 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    bump = np.exp(-(((doy - 200) / 45.0) ** 2))  # monsoon peaks around mid-July

    daily_noise = _ar1(n_days, 0.7, 0.8, rng)
    heat = np.zeros(n_days)  # a few 3-day heatwaves per year => real anomalies to detect
    for s in rng.integers(0, max(1, n_days - 3), size=max(2, 2 * round(n_days / 365))):
        heat[s:s + 3] += 6.0
    temp = (city["mean_t"] + seasonal + diurnal
            + np.repeat(daily_noise + heat, 24) + rng.normal(0, 0.4, n))

    wet_prob = 0.05 + 0.6 * city["monsoon"] * np.exp(-(((np.arange(n_days) % 365 - 200) / 45.0) ** 2))
    rainy_day = rng.random(n_days) < wet_prob
    rain = np.repeat(rainy_day, 24) * (rng.random(n) < 0.25) * rng.exponential(2.0, n)

    humidity = np.clip(city["mean_h"] - 1.2 * (temp - city["mean_t"])
                       + 20 * city["monsoon"] * bump + 15 * (rain > 0) + rng.normal(0, 3, n), 15, 100)
    wind = np.abs(rng.normal(12 + 6 * city["monsoon"] * bump, 4, n))
    pressure = (city["base_p"] - 4 * city["monsoon"] * bump - 0.15 * seasonal
                + np.repeat(_ar1(n_days, 0.8, 0.8, rng), 24) + rng.normal(0, 0.3, n))
    cloud = np.clip(30 + 50 * city["monsoon"] * bump + 30 * (rain > 0) + rng.normal(0, 15, n), 0, 100)

    df = pd.DataFrame({
        "city": city["name"], "country": city["country"],
        "latitude": city["lat"], "longitude": city["lon"],
        "time": times.strftime("%Y-%m-%dT%H:%M"),
        "temp_c": temp.round(1), "humidity_pct": humidity.round(1), "precip_mm": rain.round(1),
        "wind_kmh": wind.round(1), "pressure_hpa": pressure.round(1), "cloud_pct": cloud.round(0),
    })
    # Deliberately messy rows so the Spark cleaning stage has real work to do.
    df.loc[rng.random(n) < 0.003, "temp_c"] = np.nan          # missing readings
    df.loc[rng.random(n) < 0.0005, "humidity_pct"] = -999.0   # sensor error codes
    df = pd.concat([df, df.sample(frac=0.001, random_state=seed)])  # duplicate rows
    return df.sort_values("time", kind="stable").reset_index(drop=True)[COLUMNS]


def fetch_live_city(city, start, end, retries=3):
    params = {"latitude": city["lat"], "longitude": city["lon"],
              "start_date": start.isoformat(), "end_date": end.isoformat(),
              "hourly": HOURLY, "timezone": "auto"}
    last = None
    for attempt in range(retries):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=90)
            resp.raise_for_status()
            hourly = resp.json()["hourly"]
            break
        except (requests.RequestException, KeyError, ValueError) as exc:
            last = exc
            log.warning("%s: attempt %d failed: %s", city["name"], attempt + 1, exc)
            _time.sleep(2 ** attempt)
    else:
        raise IngestError(f"{city['name']}: {last}")
    return pd.DataFrame({
        "city": city["name"], "country": city["country"],
        "latitude": city["lat"], "longitude": city["lon"],
        "time": hourly["time"], "temp_c": hourly["temperature_2m"],
        "humidity_pct": hourly["relative_humidity_2m"], "precip_mm": hourly["precipitation"],
        "wind_kmh": hourly["wind_speed_10m"], "pressure_hpa": hourly["pressure_msl"],
        "cloud_pct": hourly["cloud_cover"],
    })[COLUMNS]


def ingest(mode="synthetic", years=2, n_cities=10, raw_dir=None, end_date=None, seed=42):
    """Write one CSV per city into the raw zone and return a manifest dict."""
    if mode not in ("live", "synthetic"):
        raise ValueError("mode must be 'live' or 'synthetic'")
    raw_dir = Path(raw_dir or config.raw_dir())
    end = end_date or (date.today() - timedelta(days=7))  # archive data lags a few days
    start = end - timedelta(days=int(365 * years) - 1)
    cities = config.CITIES[:max(1, min(n_cities, len(config.CITIES)))]

    raw_dir.mkdir(parents=True, exist_ok=True)
    for old in list(raw_dir.glob("*.csv")) + list(raw_dir.glob("_manifest.json")):
        old.unlink()  # never mix synthetic and live files

    done, failed, rows = [], [], 0
    for i, city in enumerate(cities):
        try:
            df = (synthetic_city_frame(city, start, end, seed + i) if mode == "synthetic"
                  else fetch_live_city(city, start, end))
        except IngestError as exc:
            failed.append(str(exc))
            continue
        df.to_csv(raw_dir / f"{slugify(city['name'])}.csv", index=False)
        done.append(city["name"])
        rows += len(df)
        log.info("Ingested %s (%d rows)", city["name"], len(df))

    if not done:
        hint = " Check your internet connection, or run with --mode synthetic." if mode == "live" else ""
        raise IngestError("No city could be ingested." + hint + " Details: " + "; ".join(failed))

    manifest = {
        "mode": mode, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "cities": done, "failed": failed, "rows": rows,
        "source": "Open-Meteo Archive API" if mode == "live" else "SYNTHETIC demo data (not real observations)",
    }
    (raw_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description="Ingest weather data into the raw zone")
    p.add_argument("--mode", choices=["live", "synthetic"], default="synthetic")
    p.add_argument("--years", type=float, default=2)
    p.add_argument("--cities", type=int, default=10)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(ingest(args.mode, args.years, args.cities), indent=2))


if __name__ == "__main__":
    main()
