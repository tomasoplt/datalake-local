"""
ingest_to_minio.py
------------------
Ingests files from the bind-mounted /data directory into MinIO.

Strategy per file type:
  .csv     → Spark (parse, infer schema, write as CSV to MinIO)
  .json    → Spark (parse, infer schema, write as JSON to MinIO)
  .parquet → Spark (read and re-write to MinIO)
  .txt     → boto3 direct upload (plain text — no parsing, no transformation)

Docker config assumed:
  bind-mount  →  D:\\docs  →  /data  (inside jupyter container)
  MinIO       →  http://minio:9000   bucket: datalake
"""

import os
import sys
import subprocess

# Auto-install boto3 if not available in the container
try:
    import boto3
    from botocore.client import Config
except ImportError:
    print("[INFO] boto3 not found — installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3", "-q"])
    import boto3
    from botocore.client import Config
from pyspark.sql import SparkSession

# ── Configuration ─────────────────────────────────────────────────────────────

MINIO_ENDPOINT   = "http://minio:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "admin123"
MINIO_BUCKET     = "datalake"

LOCAL_DATA_DIR   = "/data"   # bind-mounted D:\docs inside the container

# ── Sanity check ──────────────────────────────────────────────────────────────

if not os.path.exists(LOCAL_DATA_DIR):
    sys.exit(
        f"[ERROR] '{LOCAL_DATA_DIR}' does not exist inside the container.\n"
        "Make sure the bind-mount is configured in docker-compose.yml:\n"
        "  volumes:\n"
        "    - d:/docs:/data"
    )

all_entries = os.listdir(LOCAL_DATA_DIR)
files = [f for f in all_entries if os.path.isfile(os.path.join(LOCAL_DATA_DIR, f))]

if not files:
    sys.exit(f"[ERROR] '{LOCAL_DATA_DIR}' contains no files — nothing to ingest.")

print(f"[INFO] Files found in {LOCAL_DATA_DIR}:")
for f in files:
    print(f"       • {f}")

# ── boto3 client (for direct TXT uploads) ─────────────────────────────────────

s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)

# ── Spark session (for structured formats) ────────────────────────────────────

def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("ingest_to_minio")
        .config("spark.hadoop.fs.s3a.impl",              "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint",          MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key",        MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key",        MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .getOrCreate()
    )

spark = None  # lazy init — only started if structured files exist

def s3a(subfolder: str) -> str:
    return f"s3a://{MINIO_BUCKET}/{subfolder}"

# ── Ingest functions ──────────────────────────────────────────────────────────

def upload_txt(filename: str) -> None:
    """Upload a plain-text file as-is to MinIO using boto3."""
    local_path = os.path.join(LOCAL_DATA_DIR, filename)
    object_key = f"raw/txt/{filename}"
    size_kb    = os.path.getsize(local_path) / 1024

    print(f"\n[TXT] Uploading {filename} ({size_kb:.1f} KB) → s3://{MINIO_BUCKET}/{object_key}")
    s3.upload_file(local_path, MINIO_BUCKET, object_key)
    print(f"[TXT] ✓ Done")


def ingest_csv(paths: list) -> None:
    print(f"\n[CSV] Reading {len(paths)} file(s) → {s3a('raw/csv')}")
    df = spark.read.option("header", True).option("inferSchema", True).csv(paths)
    df.printSchema()
    print(f"[CSV] Row count: {df.count()}")
    df.write.mode("overwrite").option("header", True).csv(s3a("raw/csv"))
    print(f"[CSV] ✓ Written to {s3a('raw/csv')}")


def ingest_json(paths: list) -> None:
    print(f"\n[JSON] Reading {len(paths)} file(s) → {s3a('raw/json')}")
    df = spark.read.option("multiLine", True).json(paths)
    df.printSchema()
    print(f"[JSON] Row count: {df.count()}")
    df.write.mode("overwrite").json(s3a("raw/json"))
    print(f"[JSON] ✓ Written to {s3a('raw/json')}")


def ingest_parquet(paths: list) -> None:
    print(f"\n[PARQUET] Reading {len(paths)} file(s) → {s3a('raw/parquet')}")
    df = spark.read.parquet(*paths)
    df.printSchema()
    print(f"[PARQUET] Row count: {df.count()}")
    df.write.mode("overwrite").parquet(s3a("raw/parquet"))
    print(f"[PARQUET] ✓ Written to {s3a('raw/parquet')}")

# ── Classify files ────────────────────────────────────────────────────────────

def fp(filename: str) -> str:
    return os.path.join(LOCAL_DATA_DIR, filename)

csv_files     = [fp(f) for f in files if f.lower().endswith(".csv")]
json_files    = [fp(f) for f in files if f.lower().endswith(".json")]
parquet_files = [fp(f) for f in files if f.lower().endswith(".parquet")]
txt_files     = [f     for f in files if f.lower().endswith(".txt")]
skipped       = [f     for f in files if not f.lower().endswith((".csv", ".json", ".parquet", ".txt"))]

# ── TXT: direct upload via boto3 (no Spark needed) ───────────────────────────

txt_ok, txt_fail = 0, []
for filename in txt_files:
    try:
        upload_txt(filename)
        txt_ok += 1
    except Exception as e:
        print(f"[TXT] ✗ Failed to upload {filename}: {e}")
        txt_fail.append(filename)

# ── Structured formats: process via Spark ────────────────────────────────────

if csv_files or json_files or parquet_files:
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    print("\n[INFO] Spark session started.")

    if csv_files:
        ingest_csv(csv_files)
    if json_files:
        ingest_json(json_files)
    if parquet_files:
        ingest_parquet(parquet_files)

    spark.stop()

# ── Summary ───────────────────────────────────────────────────────────────────

total = len(csv_files) + len(json_files) + len(parquet_files) + txt_ok

print(f"\n{'='*55}")
print(f"  Ingestion complete.")
print(f"  Files processed : {total}")
print(f"  Files skipped   : {len(skipped)}")
if txt_fail:
    print(f"  TXT failures    : {len(txt_fail)} — {txt_fail}")
print(f"  MinIO bucket    : {MINIO_BUCKET}")
print(f"  Targets:")
if csv_files:     print(f"    CSV     → {s3a('raw/csv')}")
if json_files:    print(f"    JSON    → {s3a('raw/json')}")
if parquet_files: print(f"    Parquet → {s3a('raw/parquet')}")
if txt_files:     print(f"    TXT     → s3://{MINIO_BUCKET}/raw/txt/  (přímý upload)")
if skipped:       print(f"  Skipped   : {skipped}")
print(f"{'='*55}\n")