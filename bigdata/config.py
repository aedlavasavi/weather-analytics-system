"""Central settings. Everything is read lazily from environment variables so that
the same code runs on a laptop, in Docker, or on a cloud VM without changes."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_KEY = "results/summary.json"

# Cities with coordinates (used for live API calls) and rough climate
# parameters (used ONLY by the synthetic data generator).
CITIES = [
    {"name": "Hyderabad", "country": "India", "lat": 17.385, "lon": 78.4867, "mean_t": 27.0, "amp_t": 5.0, "mean_h": 60, "monsoon": 0.8, "base_p": 1009},
    {"name": "Mumbai", "country": "India", "lat": 19.076, "lon": 72.8777, "mean_t": 28.0, "amp_t": 2.5, "mean_h": 72, "monsoon": 1.0, "base_p": 1009},
    {"name": "Delhi", "country": "India", "lat": 28.6139, "lon": 77.209, "mean_t": 25.0, "amp_t": 9.0, "mean_h": 55, "monsoon": 0.6, "base_p": 1005},
    {"name": "Chennai", "country": "India", "lat": 13.0827, "lon": 80.2707, "mean_t": 29.0, "amp_t": 3.5, "mean_h": 70, "monsoon": 0.5, "base_p": 1009},
    {"name": "Bengaluru", "country": "India", "lat": 12.9716, "lon": 77.5946, "mean_t": 24.0, "amp_t": 3.0, "mean_h": 65, "monsoon": 0.6, "base_p": 1011},
    {"name": "Kolkata", "country": "India", "lat": 22.5726, "lon": 88.3639, "mean_t": 27.0, "amp_t": 5.5, "mean_h": 75, "monsoon": 0.9, "base_p": 1007},
    {"name": "Jaipur", "country": "India", "lat": 26.9124, "lon": 75.7873, "mean_t": 26.0, "amp_t": 8.5, "mean_h": 48, "monsoon": 0.5, "base_p": 1006},
    {"name": "Visakhapatnam", "country": "India", "lat": 17.6868, "lon": 83.2185, "mean_t": 28.0, "amp_t": 4.0, "mean_h": 72, "monsoon": 0.7, "base_p": 1009},
    {"name": "Pune", "country": "India", "lat": 18.5204, "lon": 73.8567, "mean_t": 25.5, "amp_t": 4.5, "mean_h": 58, "monsoon": 0.8, "base_p": 1010},
    {"name": "Ahmedabad", "country": "India", "lat": 23.0225, "lon": 72.5714, "mean_t": 28.0, "amp_t": 7.0, "mean_h": 50, "monsoon": 0.6, "base_p": 1007},
]


def lake_dir() -> Path:
    """Local working copy of the data lake (raw / curated / results zones)."""
    return Path(os.getenv("DATA_LAKE_DIR", str(BASE_DIR / "datalake")))


def raw_dir() -> Path:
    return lake_dir() / "raw"


def curated_dir() -> Path:
    return lake_dir() / "curated"


def results_dir() -> Path:
    return lake_dir() / "results"


def storage_backend() -> str:
    return os.getenv("STORAGE_BACKEND", "local").strip().lower()


def s3_settings() -> dict:
    return {
        "bucket": os.getenv("S3_BUCKET", "weather-data-lake"),
        "endpoint_url": os.getenv("S3_ENDPOINT_URL") or None,  # set for MinIO, empty for AWS
        "region": os.getenv("AWS_REGION", "ap-south-1"),
    }


def supabase_settings() -> dict:
    return {
        "url": os.getenv("SUPABASE_URL", "").strip(),
        "key": os.getenv("SUPABASE_KEY", "").strip(),
        "bucket": os.getenv("SUPABASE_BUCKET", "weather-data-lake").strip(),
    }
