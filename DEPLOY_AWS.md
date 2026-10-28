# Deploying to AWS (optional)

> **Not tested by me.** I could not reach AWS from my sandbox. The S3 code is tested against an AWS
> emulator (moto), and the Docker files could not be built here either. Expect to adjust details.
> AWS charges for usage; check the free-tier limits and set a billing alarm first.

## Option A: real S3 bucket, run everything else on your laptop

1. AWS Console → S3 → create a bucket (e.g. `my-weather-data-lake`) in your region.
2. IAM → create a user with an access key and a policy allowing `s3:ListBucket`, `s3:GetObject`,
   `s3:PutObject`, `s3:DeleteObject`, `s3:CreateBucket` on that bucket.
3. Edit `.env`:
   ```
   STORAGE_BACKEND=s3
   S3_BUCKET=my-weather-data-lake
   S3_ENDPOINT_URL=            # leave empty for AWS
   AWS_REGION=ap-south-1
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...
   ```
4. `python -m bigdata.run_pipeline --mode live` (uploads to S3), then `python app.py`.

## Option B: run the Docker image on an EC2 server

1. Launch an Ubuntu EC2 instance (t3.medium or larger; Spark needs RAM), open port 5000 in its security group.
2. Install Docker, copy the project, create `.env` as above (prefer an **IAM role** on the instance
   instead of access keys).
3. `docker build -t weather-analytics .`
4. Pipeline: `docker run --rm --env-file .env weather-analytics python -m bigdata.run_pipeline --mode live --publish`
5. Web: `docker run -d -p 5000:5000 --env-file .env weather-analytics`

## Option C: bigger data

Package `bigdata/spark_job.py` as a step on **Amazon EMR** or a **Databricks** job and read/write
`s3a://` paths. The transformation logic stays the same; only the paths and `SPARK_MASTER` change.
