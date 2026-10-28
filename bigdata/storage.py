"""Object storage layer.

* LocalStore - the data lake is stored locally in DATA_LAKE_DIR (default: ./datalake).
* S3Store    - integrates with Amazon S3 (or S3-compatible stores such as MinIO).

Credentials are NEVER written in code. boto3 reads AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY, and AWS_SESSION_TOKEN from the environment,
~/.aws/credentials, or IAM roles when running on AWS (EC2/ECS/EKS).
"""
import json
import logging
import os
import shutil
import time
from pathlib import Path

from . import config

log = logging.getLogger("bigdata.storage")
ZONES = ("raw", "curated", "results")


class LocalStore:
    backend = "local"

    def publish(self, zones=ZONES):
        """No-op upload: files already reside on the local filesystem."""
        return {"uploaded_files": 0, "backend": "local", "note": "local backend, nothing to upload"}

    def upload_file(self, local_path, key):
        """Copy a file to the specified key in the local data lake."""
        dest = config.lake_dir() / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = Path(local_path)
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        return str(dest)

    def download_file(self, key, local_path):
        """Retrieve a file from the local data lake."""
        src = config.lake_dir() / key
        dest = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(f"Key '{key}' not found in local data lake ({src})")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        return dest

    def download_zone(self, zone, target_dir=None):
        """Ensure zone directory exists locally."""
        src = config.lake_dir() / zone
        if not src.exists():
            return 0
        target = Path(target_dir or src)
        if target.resolve() != src.resolve():
            shutil.copytree(src, target, dirs_exist_ok=True)
        return len([p for p in target.rglob("*") if p.is_file()])

    def download_lake(self, zones=ZONES, target_dir=None):
        total = 0
        for zone in zones:
            total += self.download_zone(zone, (Path(target_dir) / zone) if target_dir else None)
        return total

    def get_json(self, key):
        path = config.lake_dir() / key
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put_json(self, key, data):
        path = config.lake_dir() / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return str(path)

    def list_keys(self, prefix=""):
        lake = config.lake_dir()
        prefix_clean = prefix.replace("\\", "/").strip("/")
        base = (lake / prefix_clean) if prefix_clean else lake
        if not base.exists():
            return []
        if base.is_file():
            return [base.relative_to(lake).as_posix()]
        return sorted([p.relative_to(lake).as_posix() for p in base.rglob("*") if p.is_file()])

    def exists(self, key):
        return (config.lake_dir() / key).exists()

    def describe(self):
        return {"backend": "local", "location": str(config.lake_dir())}


class S3Store:
    backend = "s3"

    def __init__(self, bucket, endpoint_url=None, region="ap-south-1", client=None):
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.region = region or "ap-south-1"
        if client is None:
            import boto3
            from botocore.config import Config

            boto_kwargs = {
                "service_name": "s3",
                "endpoint_url": endpoint_url or None,
                "region_name": self.region,
                "config": Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    retries={"max_attempts": 3},
                    connect_timeout=5,
                    read_timeout=30,
                ),
            }
            # Only provide credentials explicitly if non-empty; otherwise boto3 uses IAM / env / ~/.aws
            ak = (os.getenv("AWS_ACCESS_KEY_ID") or "").strip()
            sk = (os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip()
            st = (os.getenv("AWS_SESSION_TOKEN") or "").strip()
            if ak and sk:
                boto_kwargs["aws_access_key_id"] = ak
                boto_kwargs["aws_secret_access_key"] = sk
                if st:
                    boto_kwargs["aws_session_token"] = st
            client = boto3.client(**boto_kwargs)
        self.client = client

    # -- bucket management ------------------------------------------------
    def ensure_bucket(self, attempts=6):
        """Create the bucket if it does not already exist. Retries for slow MinIO startup."""
        from botocore.exceptions import BotoCoreError, ClientError

        last = None
        for i in range(attempts):
            try:
                try:
                    self.client.head_bucket(Bucket=self.bucket)
                    return
                except ClientError as exc:
                    code = str(exc.response.get("Error", {}).get("Code", ""))
                    if code not in ("404", "NoSuchBucket", "NotFound"):
                        raise
                kwargs = {"Bucket": self.bucket}
                if not self.endpoint_url and self.region != "us-east-1":
                    kwargs["CreateBucketConfiguration"] = {"LocationConstraint": self.region}
                self.client.create_bucket(**kwargs)
                log.info("Created S3 bucket %s (region: %s)", self.bucket, self.region)
                return
            except (BotoCoreError, ClientError) as exc:
                last = exc
                log.warning("Bucket check failed (%s); retry %d/%d", exc, i + 1, attempts)
                time.sleep(2)
        raise RuntimeError(f"Could not reach object storage bucket '{self.bucket}': {last}")

    def _delete_prefix(self, prefix):
        prefix_clean = prefix.replace("\\", "/").lstrip("/")
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix_clean):
            keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if keys:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": keys})

    # -- uploads & publishing ---------------------------------------------
    def upload_file(self, local_path, key):
        """Upload a single file to S3."""
        self.ensure_bucket()
        s3_key = str(key).replace("\\", "/").lstrip("/")
        self.client.upload_file(str(local_path), self.bucket, s3_key)
        log.info("Uploaded %s to s3://%s/%s", local_path, self.bucket, s3_key)
        return f"s3://{self.bucket}/{s3_key}"

    def publish(self, zones=ZONES):
        """Mirror local lake zones into the S3 bucket (replaces old objects in the zone)."""
        self.ensure_bucket()
        lake = config.lake_dir()
        uploaded = 0
        for zone in zones:
            zone_dir = lake / zone
            if not zone_dir.exists():
                continue
            self._delete_prefix(f"{zone}/")
            for path in sorted(zone_dir.rglob("*")):
                if path.is_file() and not path.name.endswith(".crc") and not path.name.startswith("."):
                    key = path.relative_to(lake).as_posix()
                    self.client.upload_file(str(path), self.bucket, key)
                    uploaded += 1
        log.info("Uploaded %d files to s3://%s", uploaded, self.bucket)
        return {"uploaded_files": uploaded, "bucket": self.bucket, "backend": "s3"}

    def put_json(self, key, data):
        """Serialize and upload a Python dictionary as a JSON object to S3."""
        self.ensure_bucket()
        s3_key = str(key).replace("\\", "/").lstrip("/")
        body = json.dumps(data, indent=2).encode("utf-8")
        self.client.put_object(Bucket=self.bucket, Key=s3_key, Body=body, ContentType="application/json")
        log.info("Wrote JSON to s3://%s/%s", self.bucket, s3_key)
        return f"s3://{self.bucket}/{s3_key}"

    # -- downloads & queries ----------------------------------------------
    def download_file(self, key, local_path):
        """Download an S3 object to a local file path."""
        s3_key = str(key).replace("\\", "/").lstrip("/")
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, s3_key, str(target))
        log.info("Downloaded s3://%s/%s to %s", self.bucket, s3_key, target)
        return target

    def download_zone(self, zone, target_dir=None):
        """Download all files under an S3 zone prefix to local disk."""
        prefix = f"{zone.strip('/')}/"
        target_dir = Path(target_dir or (config.lake_dir() / zone))
        target_dir.mkdir(parents=True, exist_ok=True)
        paginator = self.client.get_paginator("list_objects_v2")
        downloaded = 0
        from botocore.exceptions import ClientError
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    k = obj["Key"]
                    if k.endswith("/"):
                        continue
                    rel_path = k[len(prefix):]
                    dest = target_dir / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    self.client.download_file(self.bucket, k, str(dest))
                    downloaded += 1
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchBucket", "404"):
                return 0
            raise
        log.info("Downloaded %d files from s3://%s/%s to %s", downloaded, self.bucket, prefix, target_dir)
        return downloaded

    def download_lake(self, zones=ZONES, target_dir=None):
        lake = Path(target_dir or config.lake_dir())
        return sum(self.download_zone(z, lake / z) for z in zones)

    def get_json(self, key):
        from botocore.exceptions import ClientError

        s3_key = str(key).replace("\\", "/").lstrip("/")
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=s3_key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NoSuchBucket"):
                return None
            raise
        return json.loads(obj["Body"].read().decode("utf-8"))

    def list_keys(self, prefix=""):
        prefix_clean = prefix.replace("\\", "/").lstrip("/")
        paginator = self.client.get_paginator("list_objects_v2")
        keys = []
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix_clean):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
        except Exception:
            return []
        return sorted(keys)

    def exists(self, key):
        from botocore.exceptions import ClientError

        s3_key = str(key).replace("\\", "/").lstrip("/")
        try:
            self.client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError:
            return False

    def describe(self):
        return {
            "backend": "s3",
            "bucket": self.bucket,
            "endpoint": self.endpoint_url or "AWS S3",
            "region": self.region,
        }


class SupabaseStore:
    backend = "supabase"

    def __init__(self, url=None, key=None, bucket=None, client=None):
        self.url = (url if url is not None else os.getenv("SUPABASE_URL", "")).strip()
        self.key = (key if key is not None else os.getenv("SUPABASE_KEY", "")).strip()
        self.bucket_name = (bucket if bucket is not None else os.getenv("SUPABASE_BUCKET", "weather-data-lake")).strip()

        if client is None:
            if not self.url:
                raise ValueError("Supabase configuration missing: SUPABASE_URL is not set. Please set it in your .env file.")
            if not self.key:
                raise ValueError("Supabase configuration missing: SUPABASE_KEY is not set. Please set it in your .env file.")
            try:
                from supabase import create_client
                self.client = create_client(self.url, self.key)
            except Exception as exc:
                log.error("Failed to initialize Supabase client for %s: %s", self.url, exc)
                raise RuntimeError(f"Could not connect to Supabase at '{self.url}': {exc}") from exc
        else:
            self.client = client

    @property
    def bucket(self):
        return self.client.storage.from_(self.bucket_name)

    def ensure_bucket(self):
        """Ensure the target bucket exists in Supabase Storage."""
        try:
            self.client.storage.get_bucket(self.bucket_name)
        except Exception:
            try:
                self.client.storage.create_bucket(self.bucket_name, options={"public": False})
                log.info("Created Supabase storage bucket '%s'", self.bucket_name)
            except Exception as exc:
                log.debug("Bucket '%s' check/create note: %s", self.bucket_name, exc)

    def _format_storage_error(self, exc, action, key):
        msg = str(exc)
        hint = ""
        if "row-level security" in msg.lower() or "403" in msg or "unauthorized" in msg.lower():
            hint = (" (Permission denied: Row-Level Security violation. For backend data pipelines, "
                    "use the 'service_role' secret key from Supabase Dashboard -> Project Settings -> API, "
                    f"or add an INSERT/UPDATE RLS policy on bucket '{self.bucket_name}' for role 'anon')")
        return RuntimeError(f"Failed to {action} '{key}' in Supabase bucket '{self.bucket_name}': {msg}{hint}")

    def upload_file(self, local_path, key):
        """Upload a local file to Supabase storage."""
        self.ensure_bucket()
        clean_key = str(key).replace("\\", "/").lstrip("/")
        src = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(f"Local file '{local_path}' does not exist")
        try:
            with open(src, "rb") as f:
                content = f.read()
            self.bucket.upload(path=clean_key, file=content, file_options={"upsert": "true"})
            log.info("Uploaded %s to supabase://%s/%s", local_path, self.bucket_name, clean_key)
            return f"supabase://{self.bucket_name}/{clean_key}"
        except Exception as exc:
            err = self._format_storage_error(exc, "upload", clean_key)
            log.error("Supabase upload failed: %s", err)
            raise err from exc

    def download_file(self, key, local_path):
        """Download a file from Supabase storage to a local path."""
        clean_key = str(key).replace("\\", "/").lstrip("/")
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            content = self.bucket.download(clean_key)
            dest.write_bytes(content)
            log.info("Downloaded supabase://%s/%s to %s", self.bucket_name, clean_key, dest)
            return dest
        except Exception as exc:
            log.error("Supabase download failed for key '%s': %s", clean_key, exc)
            raise FileNotFoundError(f"File '{clean_key}' not found in Supabase bucket '{self.bucket_name}': {exc}") from exc

    def put_json(self, key, data):
        """Serialize and upload a Python dictionary as a JSON object to Supabase storage."""
        self.ensure_bucket()
        clean_key = str(key).replace("\\", "/").lstrip("/")
        body = json.dumps(data, indent=2).encode("utf-8")
        try:
            self.bucket.upload(
                path=clean_key,
                file=body,
                file_options={"upsert": "true", "content-type": "application/json"},
            )
            log.info("Wrote JSON to supabase://%s/%s", self.bucket_name, clean_key)
            return f"supabase://{self.bucket_name}/{clean_key}"
        except Exception as exc:
            err = self._format_storage_error(exc, "put JSON", clean_key)
            log.error("Supabase put_json failed: %s", err)
            raise err from exc

    def get_json(self, key):
        """Download and parse a JSON object from Supabase storage."""
        clean_key = str(key).replace("\\", "/").lstrip("/")
        try:
            content = self.bucket.download(clean_key)
            return json.loads(content.decode("utf-8"))
        except Exception as exc:
            err_text = str(exc).lower()
            if "not found" in err_text or "404" in err_text or "nosuchkey" in err_text:
                return None
            log.warning("Could not retrieve JSON from supabase://%s/%s: %s", self.bucket_name, clean_key, exc)
            raise

    def list_keys(self, prefix=""):
        """Recursively list all file keys under a prefix in Supabase."""
        prefix_clean = prefix.replace("\\", "/").strip("/")
        files = []

        def _traverse(dir_path):
            try:
                items = self.bucket.list(dir_path)
            except Exception as exc:
                log.warning("Supabase list error for path '%s': %s", dir_path, exc)
                return
            for item in items:
                name = item.get("name", "")
                if not name or name.startswith("."):
                    continue
                full_path = f"{dir_path}/{name}".strip("/") if dir_path else name
                if item.get("id") is None and item.get("metadata") is None:
                    _traverse(full_path)
                else:
                    files.append(full_path)

        _traverse(prefix_clean)
        return sorted(files)

    def exists(self, key):
        """Check if a file exists in the Supabase bucket."""
        clean_key = str(key).replace("\\", "/").lstrip("/")
        try:
            return self.bucket.exists(clean_key)
        except Exception:
            return False

    def publish(self, zones=ZONES):
        """Mirror local data lake zones to Supabase storage."""
        self.ensure_bucket()
        lake = config.lake_dir()
        uploaded = 0
        for zone in zones:
            zone_dir = lake / zone
            if not zone_dir.exists():
                continue
            try:
                old_files = self.list_keys(zone)
                if old_files:
                    self.bucket.remove(old_files)
            except Exception as exc:
                log.warning("Note: could not clear old files in zone '%s': %s", zone, exc)

            for path in sorted(zone_dir.rglob("*")):
                if path.is_file() and not path.name.endswith(".crc") and not path.name.startswith("."):
                    key = path.relative_to(lake).as_posix()
                    with open(path, "rb") as f:
                        self.bucket.upload(
                            path=key,
                            file=f.read(),
                            file_options={"upsert": "true"},
                        )
                    uploaded += 1
        log.info("Uploaded %d files to supabase://%s", uploaded, self.bucket_name)
        return {"uploaded_files": uploaded, "bucket": self.bucket_name, "backend": "supabase"}

    def download_zone(self, zone, target_dir=None):
        """Download all files under a zone prefix from Supabase to local disk."""
        prefix = zone.strip("/")
        target_dir = Path(target_dir or (config.lake_dir() / zone))
        target_dir.mkdir(parents=True, exist_ok=True)
        keys = self.list_keys(prefix)
        downloaded = 0
        for k in keys:
            rel_path = k[len(prefix):].lstrip("/")
            dest = target_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            content = self.bucket.download(k)
            dest.write_bytes(content)
            downloaded += 1
        log.info("Downloaded %d files from supabase://%s/%s to %s", downloaded, self.bucket_name, prefix, target_dir)
        return downloaded

    def download_lake(self, zones=ZONES, target_dir=None):
        lake = Path(target_dir or config.lake_dir())
        return sum(self.download_zone(z, lake / z) for z in zones)

    def describe(self):
        return {
            "backend": "supabase",
            "bucket": self.bucket_name,
            "endpoint": self.url or "Supabase Storage",
        }


def get_store():
    backend = config.storage_backend()
    if backend == "supabase":
        return SupabaseStore(**config.supabase_settings())
    if backend == "s3":
        return S3Store(**config.s3_settings())
    return LocalStore()
