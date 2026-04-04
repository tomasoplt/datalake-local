"""
query_data.py
-------------
Interaktivní dotazování nad daty v MinIO.

  .txt soubory  → stažení přes boto3 + dotaz přes Ollama (gemma3:1b)
  .csv / .json / .parquet → Spark SQL dotaz

Použití:
  1. Spusť skript: %run query_data.py
  2. Zadej název souboru a svůj dotaz.
"""

import os
import sys
import subprocess

# ── Auto-install závislostí ───────────────────────────────────────────────────

for pkg in ["boto3"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"[INFO] Instaluji {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import boto3
import urllib.request
import json
from botocore.client import Config
from pyspark.sql import SparkSession

# ── Konfigurace ───────────────────────────────────────────────────────────────

MINIO_ENDPOINT   = "http://minio:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "admin123"
MINIO_BUCKET     = "datalake"

# ── Klienti ───────────────────────────────────────────────────────────────────

s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)

OLLAMA_URL   = "http://host.docker.internal:11434"
OLLAMA_MODEL = "gemma3:1b"

# ── Pomocné funkce ────────────────────────────────────────────────────────────

def list_files() -> list[dict]:
    """Vypíše všechny soubory v bucketu."""
    response = s3.list_objects_v2(Bucket=MINIO_BUCKET)
    return response.get("Contents", [])


def download_txt(object_key: str) -> str:
    """Stáhne textový soubor z MinIO a vrátí jeho obsah."""
    obj = s3.get_object(Bucket=MINIO_BUCKET, Key=object_key)
    return obj["Body"].read().decode("utf-8", errors="replace")


def ask_ollama(context: str, question: str) -> str:
    """Pošle kontext + dotaz do lokální Ollamy a vrátí odpověď."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": (
            f"Zde je obsah dokumentu:\n\n"
            f"---\n{context}\n---\n\n"
            f"Otázka: {question}\n\n"
            f"Odpověz stručně a přesně pouze na základě textu výše."
        ),
        "stream": False
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result["response"]


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("query_minio")
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


def query_structured(object_key: str, sql: str) -> None:
    """Načte strukturovaný soubor z MinIO a spustí Spark SQL dotaz."""
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    path  = f"s3a://{MINIO_BUCKET}/{object_key}"
    ext   = object_key.rsplit(".", 1)[-1].lower()

    if ext == "csv":
        df = spark.read.option("header", True).option("inferSchema", True).csv(path)
    elif ext == "json":
        df = spark.read.option("multiLine", True).json(path)
    elif ext == "parquet":
        df = spark.read.parquet(path)
    else:
        print(f"[WARN] Nepodporovaný formát: {ext}")
        spark.stop()
        return

    df.createOrReplaceTempView("data")
    print(f"\n[INFO] Schéma souboru {object_key}:")
    df.printSchema()
    print(f"\n[SQL] {sql}\n")
    spark.sql(sql).show(truncate=False)
    spark.stop()

# ── Hlavní smyčka ─────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("  Dotazování nad daty v MinIO")
print("="*55)

# Zobraz dostupné soubory
objects = list_files()
if not objects:
    print("[WARN] Bucket je prázdný. Nejprve spusť ingest_to_minio.py")
    sys.exit(0)

print("\nDostupné soubory v MinIO:\n")
for i, obj in enumerate(objects):
    print(f"  [{i+1}] {obj['Key']}  ({obj['Size']/1024:.1f} KB)")

# Výběr souboru
print()
choice = input("Zadej číslo souboru: ").strip()
try:
    selected = objects[int(choice) - 1]["Key"]
except (ValueError, IndexError):
    print("[ERROR] Neplatná volba.")
    sys.exit(1)

print(f"\nVybrán: {selected}")
ext = selected.rsplit(".", 1)[-1].lower()

# Dotaz
question = input("Zadej dotaz / SQL: ").strip()

print("\n" + "-"*55)

if ext == "txt":
    # Textový soubor → Claude API
    print(f"[INFO] Stahuji {selected} z MinIO...")
    content = download_txt(selected)
    print(f"[INFO] Délka dokumentu: {len(content)} znaků")
    print(f"[INFO] Odesílám dotaz do Ollama ({OLLAMA_MODEL})...\n")
    answer = ask_ollama(content, question)
    print("Odpověď:\n")
    print(answer)

else:
    # Strukturovaný soubor → Spark SQL
    print(f"[INFO] Spouštím Spark SQL dotaz na {selected}...")
    query_structured(selected, question)

print("\n" + "="*55 + "\n")