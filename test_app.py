"""Offline tests. The Open-Meteo calls are mocked with SYNTHETIC data
shaped like the real API responses, so these tests prove the code paths work,
not that live APIs are reachable. Run:  python -m unittest -v
"""
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest import mock

import numpy as np

# Configure environment BEFORE importing the app.
_tmp = tempfile.mkdtemp()
os.environ["DB_PATH"] = os.path.join(_tmp, "test.db")

import app as app_module  # noqa: E402
import weather_service as ws  # noqa: E402
from analytics import run_analytics  # noqa: E402


def fake_forecast_payload(past_days=30, forecast_days=7):
    """Synthetic API-shaped payload with one injected hot-day anomaly."""
    rng = np.random.default_rng(42)
    now = datetime(2026, 10, 12, 14, 0)
    start = datetime(2026, 10, 12) - timedelta(days=past_days)
    hours = int((past_days + forecast_days) * 24)
    times = [start + timedelta(hours=i) for i in range(hours)]
    temp, hum, rain, wind, pres = [], [], [], [], []
    for i, t in enumerate(times):
        base = 27 + 5 * np.sin((t.hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.4)
        if t.date() == (now - timedelta(days=10)).date():
            base += 12  # anomaly
        temp.append(round(float(base), 1))
        hum.append(round(float(70 - (base - 27) * 2 + rng.normal(0, 2)), 1))
        rain.append(round(float(rng.random() < 0.05) * rng.uniform(0.5, 6), 1))
        wind.append(round(float(abs(rng.normal(12, 4))), 1))
        pres.append(round(float(1008 + rng.normal(0, 1.5)), 1))
    return {
        "timezone": "Asia/Kolkata",
        "current": {
            "time": now.strftime("%Y-%m-%dT%H:%M"), "temperature_2m": 29.4,
            "relative_humidity_2m": 62, "apparent_temperature": 31.0, "precipitation": 0.0,
            "weather_code": 2, "cloud_cover": 40, "pressure_msl": 1008.2,
            "wind_speed_10m": 11.5, "wind_direction_10m": 250,
        },
        "current_units": {"temperature_2m": "°C"},
        "hourly": {
            "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
            "temperature_2m": temp, "relative_humidity_2m": hum, "precipitation": rain,
            "wind_speed_10m": wind, "pressure_msl": pres,
        },
        "daily": {"sunrise": ["2026-10-12T06:03"], "sunset": ["2026-10-12T17:57"]},
    }


GEOCODE_OK = {"results": [{"name": "Hyderabad", "country": "India", "admin1": "Telangana",
                           "latitude": 17.38, "longitude": 78.47, "timezone": "Asia/Kolkata"}]}


class FakeResp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status
        self.text = str(data)

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.exceptions.HTTPError(f"{self.status_code}")


def fake_get(url, params=None, timeout=None):
    if "geocoding" in url:
        if params and params.get("name", "").lower() == "nowhereville":
            return FakeResp({})  # API returns no 'results' key for unknown cities
        return FakeResp(GEOCODE_OK)
    return FakeResp(fake_forecast_payload())


class WeatherAppTests(unittest.TestCase):
    def setUp(self):
        app_module._cache.clear()
        app_module.app.testing = True
        self.client = app_module.app.test_client()

    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")

    def test_index_page_renders(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Weather Data Analytics", r.data)

    def test_weather_requires_city(self):
        self.assertEqual(self.client.get("/api/weather").status_code, 400)

    @mock.patch("weather_service.requests.get", side_effect=fake_get)
    def test_weather_ok(self, _):
        r = self.client.get("/api/weather?city=Hyderabad")
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["location"]["name"], "Hyderabad")
        self.assertEqual(body["current"]["condition"], "Partly cloudy")
        self.assertEqual(body["current"]["temperature_c"], 29.4)

    @mock.patch("weather_service.requests.get", side_effect=fake_get)
    def test_unknown_city_is_404(self, _):
        r = self.client.get("/api/weather?city=Nowhereville")
        self.assertEqual(r.status_code, 404)
        self.assertIn("not found", r.get_json()["error"])

    @mock.patch("weather_service.requests.get", side_effect=fake_get)
    def test_analytics_report(self, _):
        r = self.client.get("/api/analytics?city=Hyderabad")
        self.assertEqual(r.status_code, 200)
        rep = r.get_json()["report"]
        self.assertEqual(rep["quality"]["missing_values"], 0)
        # observed and forecast are kept separate
        self.assertTrue(rep["daily_observed"])
        self.assertTrue(rep["daily_forecast"])
        self.assertLess(rep["daily_observed"][-1]["date"], "2026-10-14")
        self.assertGreater(rep["daily_forecast"][-1]["date"], "2026-10-12")
        # injected anomaly day (10 days before 2026-10-12) must be flagged
        flagged = [a["date"] for a in rep["anomalies"]["temperature"]]
        self.assertIn("2026-10-02", flagged)
        self.assertIsNotNone(rep["baseline_forecast"])
        self.assertEqual(len(rep["baseline_forecast"]["predictions_c"]), 3)
        self.assertIn("temp_c", rep["correlations"])

    def test_analytics_json_safe_with_missing_values(self):
        payload = fake_forecast_payload()
        payload["hourly"]["temperature_2m"][5] = None
        rep = run_analytics(payload)
        self.assertEqual(rep["quality"]["missing_values"], 1)

    @mock.patch("weather_service.requests.get", side_effect=fake_get)
    def test_observation_is_stored(self, _):
        self.client.get("/api/weather?city=Hyderabad")
        r = self.client.get("/api/history?city=Hyderabad")
        self.assertGreaterEqual(len(r.get_json()["stored_observations"]), 1)

    def test_chat_endpoint_removed(self):
        r = self.client.post("/api/chat", json={"message": "hello"})
        self.assertEqual(r.status_code, 404)

    @mock.patch("weather_service.requests.get")
    def test_network_failure_is_handled(self, mock_get):
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("boom")
        r = self.client.get("/api/weather?city=Hyderabad")
        self.assertEqual(r.status_code, 502)
        self.assertIn("Could not reach", r.get_json()["error"])

    def test_geocode_empty_input(self):
        with self.assertRaises(ws.WeatherServiceError):
            ws.geocode("  ")


if __name__ == "__main__":
    unittest.main()
