# Cloud-Based Weather Data Analytics System

Cloud-based weather analytics with an **Apache Spark big data batch-processing pipeline**, a cloud-ready **data lake (supporting local storage, Amazon S3, and Supabase Storage)**, and an interactive **Flask web dashboard**.

* **Big Data Engine: Apache Spark** (PySpark: DataFrames, explicit schema, Spark SQL, window functions, partitioned Parquet, MLlib Linear Regression & Random Forest). Requires **Java 17 or 21**. On Windows, automatically uses a Windows-safe I/O mode to avoid Hadoop native `winutils.exe` / `hadoop.dll` issues.
* **Storage Backends:**
  * `STORAGE_BACKEND=local`: Data lake resides on the local filesystem (`./datalake`).
  * `STORAGE_BACKEND=s3`: Data lake syncs with **Amazon S3** (or S3-compatible object storage such as MinIO). Supports upload, download, and Spark batch access. AWS credentials handled securely via environment variables or IAM roles.
  * `STORAGE_BACKEND=supabase`: Data lake syncs with **Supabase Storage** bucket (e.g. `weather-data-lake`). Uses official `supabase` Python SDK. Uploads, downloads, and serves batch summaries directly to/from Supabase bucket.
* **Weather Dashboard:** Real-time conditions via Open-Meteo, city search, historical observations via SQLite, statistical anomaly detection (z-scores), 7-day forecasts, and Chart.js visualizations.

---

## Architecture & Project Structure

```
weather-analytics/
├── app.py                     Flask web application (dashboard & REST API)
├── analytics.py               Live weather analytics, statistics, and anomaly detection
├── weather_service.py         Open-Meteo API client (live weather & geocoding)
├── bigdata_api.py             /api/bigdata/* endpoints (serves Spark data lake results)
├── verify_results.py          Verification script for pipeline output & dashboard API
├── run_all.bat                Windows CMD shortcut: setup, run pipeline, verify, and start
├── requirements.txt           Core dependencies (Flask, PySpark, boto3, pandas, etc.)
├── requirements-dev.txt       Development & testing dependencies (moto[s3])
├── requirements-duckdb.txt    Optional DuckDB dependencies
├── .env.example               Environment template (copy to .env)
├── bigdata/
│   ├── config.py              Central configuration (cities, lake paths, S3 settings)
│   ├── ingest.py              STAGE 1: Ingestion -> Raw CSV (live or synthetic)
│   ├── spark_job.py           STAGE 2: Apache Spark batch job (clean, Parquet, SQL, MLlib)
│   ├── pandas_job.py          STAGE 2 (fallback): Lightweight pandas batch job
│   ├── duckdb_job.py          STAGE 2 (optional): DuckDB SQL batch job
│   ├── storage.py             STAGE 3: Storage layer (LocalStore, S3Store & SupabaseStore)
│   └── run_pipeline.py        One-command runner for stages 1-3
├── templates/
│   └── index.html             Weather dashboard HTML UI & Chart.js client
├── static/
│   └── chart.umd.js           Bundled Chart.js library (works fully offline)
├── Dockerfile, docker-compose.yml, DEPLOY_AWS.md   Cloud & container deployment
├── test_app.py, test_bigdata.py, test_pandas.py    Automated test suites
└── datalake/                  Data lake directory (local or mirrored to S3)
    ├── raw/                   Raw ingested hourly weather CSV files
    ├── curated/               Cleaned Parquet (hourly, daily) partitioned by city/year
    └── results/summary.json   Batch analytics summary read by the dashboard
```

---

## Technologies Used

| Purpose | Tool |
|---|---|
| Big Data Processing | **Apache Spark (PySpark 4.x)**: Spark SQL, window functions, aggregations |
| Machine Learning | **Spark MLlib**: Linear Regression & Random Forest Regressor |
| Cloud Object Storage | **Amazon S3** (`boto3`) and **Supabase Storage** (`supabase`) |
| Data Lake Storage Format | **Apache Parquet** (columnar, Hive-partitioned by `city` and `year`) |
| Web Dashboard & Backend | **Python Flask** with responsive UI and Chart.js visualizations |
| Weather Data Source | **Open-Meteo API** (live & historical archive) + synthetic climate generator |

---

## Windows Installation & Execution (Command Prompt)

### Prerequisites
* **Python 3.10+** (Python 3.11 recommended): `python --version`
* **Java 17 or 21** (Eclipse Temurin 17 recommended): `java -version`

### Step-by-Step Setup

Open Windows **Command Prompt (`cmd.exe`)** and run:

```cmd
:: 1. Navigate to the project directory
cd /d "C:\Users\ACER\OneDrive\Apps\Desktop\bda cc project\weather-analytics"

:: 2. Create and activate a virtual environment
python -m venv venv
call venv\Scripts\activate

:: 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

:: 4. Configure environment
if not exist .env copy .env.example .env

:: 5. Run the Apache Spark batch pipeline (synthetic demo data, 10 cities, 1.5 years)
python -m bigdata.run_pipeline --engine spark --mode synthetic

:: 6. Verify pipeline and API output
python verify_results.py

:: 7. Start the dashboard
python app.py
```

Open your browser to: **http://127.0.0.1:5000**

> **Shortcut:** You can also simply run `run_all.bat`.

---

## Storage Backend Configuration (Local, Amazon S3 & Supabase)

The data lake supports local filesystem storage, Amazon S3, and Supabase Storage through `.env`:

### 1. Local Storage (Default)
```ini
STORAGE_BACKEND=local
DATA_LAKE_DIR=./datalake
```
No cloud credentials required. Data lake files remain in the `./datalake` folder.

### 2. Amazon S3 Cloud Storage
```ini
STORAGE_BACKEND=s3
S3_BUCKET=your-weather-data-lake-bucket
AWS_REGION=ap-south-1

# Leave S3_ENDPOINT_URL empty for AWS S3 (or set to http://localhost:9000 for local MinIO)
S3_ENDPOINT_URL=

# AWS Credentials (handled securely - never hardcoded in code):
# Option A: Explicit access keys in .env
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

# Option B: Leave keys empty to automatically use IAM roles (EC2/ECS) or ~/.aws/credentials
```

### 3. Supabase Cloud Storage
```ini
STORAGE_BACKEND=supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-service-role-or-anon-key
SUPABASE_BUCKET=weather-data-lake
```

* **Bucket Setup**: The application automatically checks and creates the bucket `weather-data-lake` if your key has bucket creation permissions. Alternatively, create a bucket named `weather-data-lake` in your Supabase Dashboard under **Storage**.
* **Security**: Credentials are read strictly from environment variables and are never hardcoded or printed in logs or UI endpoints.

### Cloud Pipeline Operations (S3 & Supabase)
* **Publish/Upload to Cloud Storage**:
  ```cmd
  :: For S3 (when STORAGE_BACKEND=s3) or Supabase (when STORAGE_BACKEND=supabase)
  python -m bigdata.run_pipeline --engine spark --mode synthetic --publish
  ```
  Automatically uploads `raw/`, `curated/` (partitioned Parquet), and `results/summary.json` to the configured bucket.
* **Download from Cloud Storage**:
  ```cmd
  python -m bigdata.run_pipeline --download
  ```
  Downloads existing data lake files from the cloud bucket to the local working directory.
* **Dashboard Cloud Reading**:
  When `STORAGE_BACKEND=s3` or `STORAGE_BACKEND=supabase`, the Flask dashboard queries cloud storage directly for `results/summary.json` and displays the active cloud storage badge in the UI.

---

## Running Automated Tests

Run the offline and mocked test suites (no live AWS or Supabase credentials required):

```cmd
:: Web application & analytics tests
python -m unittest -v test_app.py

:: Big Data, Spark pipeline, S3, and Supabase storage tests (uses moto & mocked Supabase client)
python -m unittest -v test_bigdata.py

:: Pandas fallback tests
python -m unittest -v test_pandas.py
```

---

## REST API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Interactive Weather Analytics Dashboard |
| `/api/health` | `GET` | Health status check |
| `/api/weather?city=<name>` | `GET` | Current weather observations from Open-Meteo |
| `/api/analytics?city=<name>` | `GET` | Single-city pandas analytics, trends, anomalies, and forecasts |
| `/api/history?city=<name>` | `GET` | Historical stored observations from local database |
| `/api/bigdata/status` | `GET` | Data lake storage backend info and last Spark run status |
| `/api/bigdata/summary` | `GET` | Full Spark batch job analytics, SQL summaries, and ML predictions |

---

## Troubleshooting

| Issue | Resolution |
|---|---|
| `'java' is not recognized` / `JAVA_GATEWAY_EXITED` | Install OpenJDK 17 (Eclipse Temurin 17) and add `bin` to `PATH`. |
| `UnsatisfiedLinkError ... NativeIO$Windows` | Handled automatically by `spark_job.py` Windows-safe mode (`SPARK_IO_MODE=auto`). |
| `Object storage is not reachable` | For S3: check `S3_BUCKET`, `AWS_REGION`, and AWS keys in `.env`. For Supabase: verify `SUPABASE_URL` and `SUPABASE_KEY` in `.env`. |
| `Could not remove old Parquet files ... OneDrive` | Pause OneDrive sync for the folder or run with `SPARK_IO_MODE=driver`. |
| Port 5000 in use | Set `set PORT=5001` and run `python app.py`. |
