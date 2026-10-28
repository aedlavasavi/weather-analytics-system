# One image for both the web app and the Spark pipeline.
FROM python:3.12-slim-bookworm

# Spark needs a Java runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-spark.txt ./
RUN pip install --no-cache-dir -r requirements-spark.txt gunicorn

COPY . .
ENV PORT=5000 \
    DATA_LAKE_DIR=/data/datalake \
    DB_PATH=/data/weather.db
RUN mkdir -p /data
EXPOSE 5000

# Web app (the pipeline service overrides this command in docker-compose.yml)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "2", "--timeout", "120", "app:app"]
