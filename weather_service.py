"""Weather data collection using the free Open-Meteo APIs (no API key needed).

Data source: https://open-meteo.com  (geocoding + forecast/historical endpoints)
"""
import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 15  # seconds

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "pressure_msl",
]
CURRENT_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "weather_code",
    "cloud_cover",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
]

# WMO weather interpretation codes -> text
WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


class WeatherServiceError(Exception):
    """Raised for any problem collecting weather data (network, API, unknown city)."""


def describe_code(code):
    try:
        return WMO_CODES.get(int(code), "Unknown")
    except (TypeError, ValueError):
        return "Unknown"


def _get_json(url, params):
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout as exc:
        raise WeatherServiceError("The weather service timed out. Please try again.") from exc
    except requests.exceptions.RequestException as exc:
        raise WeatherServiceError(f"Could not reach the weather service: {exc}") from exc
    except ValueError as exc:
        raise WeatherServiceError("The weather service returned invalid data.") from exc


def geocode(city):
    """Convert a city name into coordinates."""
    city = (city or "").strip()
    if not city:
        raise WeatherServiceError("Please provide a city name.")
    data = _get_json(
        GEOCODE_URL,
        {"name": city, "count": 1, "language": "en", "format": "json"},
    )
    results = data.get("results") or []
    if not results:
        raise WeatherServiceError(f"City '{city}' was not found. Check the spelling.")
    top = results[0]
    return {
        "name": top.get("name", city),
        "country": top.get("country", ""),
        "admin1": top.get("admin1", ""),
        "latitude": top["latitude"],
        "longitude": top["longitude"],
        "timezone": top.get("timezone", "auto"),
    }


def fetch_weather(latitude, longitude, past_days=30, forecast_days=7):
    """Fetch current conditions, hourly series (past + forecast) and daily sun times."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join(CURRENT_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "daily": "sunrise,sunset",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "auto",
    }
    data = _get_json(FORECAST_URL, params)
    if "hourly" not in data or "current" not in data:
        raise WeatherServiceError("The weather service response was incomplete.")
    return data
