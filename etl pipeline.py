

import sys
import subprocess

try:
    import boto3
    from botocore.client import Config
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3", "-q"])
    import boto3
    from botocore.client import Config

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType

# ── Konfigurace ───────────────────────────────────────────────────────────────

MINIO_ENDPOINT   = "http://minio:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "admin123"
MINIO_BUCKET     = "datalake"

HADOOP_AWS_VERSION = "3.3.4"
AWS_SDK_VERSION    = "1.12.367"

RAW_PATH    = f"s3a://{MINIO_BUCKET}/raw/csv"
SILVER_PATH = f"s3a://{MINIO_BUCKET}/processed/silver"
GOLD_PATH   = f"s3a://{MINIO_BUCKET}/processed/gold"

# ── Spark session ─────────────────────────────────────────────────────────────

def get_spark() -> SparkSession:
    # Zastav starou session pokud existuje
    existing = SparkSession.getActiveSession()
    if existing:
        print("[INFO] Stopping existing Spark session...")
        existing.stop()

    return (
        SparkSession.builder
        .appName("etl_pipeline")
        .config("spark.jars.packages",
                f"org.apache.hadoop:hadoop-aws:{HADOOP_AWS_VERSION},"
                f"com.amazonaws:aws-java-sdk-bundle:{AWS_SDK_VERSION}")
        .config("spark.hadoop.fs.s3a.impl",
                "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint",          MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key",        MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key",        MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )

# ── RAW → SILVER ──────────────────────────────────────────────────────────────

def silver_orders(spark: SparkSession) -> None:
    """
    Načte orders.csv a aplikuje:
      - správné typy sloupců
      - parsování data
      - odvozené sloupce: year, month
      - deduplikaci a filtrování neplatných řádků
    """
    print("\n[SILVER] orders...")
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)   # typy definujeme ručně níže
        .csv(f"{RAW_PATH}/orders")
    )

    df = (
        df
        .withColumn("order_id",    F.col("order_id").cast(IntegerType()))
        .withColumn("order_date",  F.to_date(F.col("order_date"), "yyyy-MM-dd"))
        .withColumn("customer_id", F.col("customer_id").cast(IntegerType()))
        .withColumn("product_id",  F.col("product_id").cast(IntegerType()))
        .withColumn("quantity",    F.col("quantity").cast(IntegerType()))
        .withColumn("unit_price",  F.col("unit_price").cast(DoubleType()))
        .withColumn("total_price", F.col("total_price").cast(DoubleType()))
        # odvozené sloupce pro particionování a trend
        .withColumn("year",        F.year("order_date"))
        .withColumn("month",       F.month("order_date"))
        # audit
        .withColumn("_etl_ts",     F.current_timestamp())
    )

    # Odstraň řádky s chybějícím klíčem nebo datem
    df = df.dropna(subset=["order_id", "order_date", "customer_id", "product_id"])
    df = df.dropDuplicates(["order_id"])

    print(f"  Řádků: {df.count()}")
    df.printSchema()

    (
        df.write
        .mode("overwrite")
        .partitionBy("year", "month")
        .parquet(f"{SILVER_PATH}/orders")
    )
    print(f"  ✓ Uloženo → {SILVER_PATH}/orders  (particionováno rok/měsíc)")


def silver_customers(spark: SparkSession) -> None:
    """Načte customers.csv a standardizuje segment na uppercase."""
    print("\n[SILVER] customers...")
    df = (
        spark.read
        .option("header", True)
        .csv(f"{RAW_PATH}/customers")
    )

    df = (
        df
        .withColumn("customer_id", F.col("customer_id").cast(IntegerType()))
        .withColumn("segment",     F.upper(F.trim(F.col("segment"))))
        .withColumn("city",        F.trim(F.col("city")))
        .withColumn("name",        F.trim(F.col("name")))
        .withColumn("_etl_ts",     F.current_timestamp())
        .dropDuplicates(["customer_id"])
        .dropna(subset=["customer_id"])
    )

    print(f"  Řádků: {df.count()}")
    df.write.mode("overwrite").parquet(f"{SILVER_PATH}/customers")
    print(f"  ✓ Uloženo → {SILVER_PATH}/customers")


def silver_products(spark: SparkSession) -> None:
    """Načte products.csv a standardizuje kategorii."""
    print("\n[SILVER] products...")
    df = (
        spark.read
        .option("header", True)
        .csv(f"{RAW_PATH}/products")
    )

    df = (
        df
        .withColumn("product_id", F.col("product_id").cast(IntegerType()))
        .withColumn("unit_price", F.col("unit_price").cast(DoubleType()))
        .withColumn("category",   F.trim(F.col("category")))
        .withColumn("name",       F.trim(F.col("name")))
        .withColumn("_etl_ts",    F.current_timestamp())
        .dropDuplicates(["product_id"])
        .dropna(subset=["product_id"])
    )

    print(f"  Řádků: {df.count()}")
    df.write.mode("overwrite").parquet(f"{SILVER_PATH}/products")
    print(f"  ✓ Uloženo → {SILVER_PATH}/products")


# ── SILVER → GOLD ─────────────────────────────────────────────────────────────

def build_gold(spark: SparkSession) -> None:
    """Načte silver Parquet tabulky a vytvoří 4 analytické gold pohledy."""

    print("\n[GOLD] Načítám silver tabulky...")
    orders    = spark.read.parquet(f"{SILVER_PATH}/orders")
    customers = spark.read.parquet(f"{SILVER_PATH}/customers")
    products  = spark.read.parquet(f"{SILVER_PATH}/products")

    # Kompletní denormalizovaná tabulka — základ pro všechny gold pohledy
    joined = (
        orders
        .join(customers, on="customer_id", how="left")
        .join(
            products.select(
                "product_id",
                "category",
                F.col("name").alias("product_name")
            ),
            on="product_id", how="left"
        )
    )

    # ── Gold 1: Tržby podle kategorie produktu ────────────────────────────────
    print("\n[GOLD] 1/4 — tržby podle kategorie...")
    gold_kategorie = (
        joined
        .filter(F.col("status") == "completed")
        .groupBy("category")
        .agg(
            F.count("order_id")              .alias("pocet_objednavek"),
            F.sum("total_price")             .alias("trzby_celkem"),
            F.round(F.avg("total_price"), 2) .alias("prumerna_objednavka"),
            F.sum("quantity")                .alias("prodano_kusu"),
        )
        .orderBy(F.desc("trzby_celkem"))
    )
    gold_kategorie.show(truncate=False)
    gold_kategorie.write.mode("overwrite").parquet(f"{GOLD_PATH}/trzby_kategorie")
    print(f"  ✓ → {GOLD_PATH}/trzby_kategorie")

    # ── Gold 2: Top zákazníci ─────────────────────────────────────────────────
    print("\n[GOLD] 2/4 — top zákazníci...")
    gold_zakaznici = (
        joined
        .filter(F.col("status") == "completed")
        .groupBy("customer_id", "name", "city", "segment")
        .agg(
            F.count("order_id")              .alias("pocet_objednavek"),
            F.sum("total_price")             .alias("trzby_celkem"),
            F.round(F.avg("total_price"), 2) .alias("prumerna_objednavka"),
            F.min("order_date")              .alias("prvni_objednavka"),
            F.max("order_date")              .alias("posledni_objednavka"),
        )
        .orderBy(F.desc("trzby_celkem"))
    )
    gold_zakaznici.show(truncate=False)
    gold_zakaznici.write.mode("overwrite").parquet(f"{GOLD_PATH}/top_zakaznici")
    print(f"  ✓ → {GOLD_PATH}/top_zakaznici")

    # ── Gold 3: Měsíční trend tržeb ───────────────────────────────────────────
    print("\n[GOLD] 3/4 — měsíční trend...")
    gold_trend = (
        joined
        .filter(F.col("status") == "completed")
        .groupBy("year", "month")
        .agg(
            F.count("order_id")              .alias("pocet_objednavek"),
            F.sum("total_price")             .alias("trzby_celkem"),
            F.round(F.avg("total_price"), 2) .alias("prumerna_objednavka"),
        )
        .orderBy("year", "month")
    )
    gold_trend.show(24, truncate=False)
    gold_trend.write.mode("overwrite").parquet(f"{GOLD_PATH}/mesicni_trend")
    print(f"  ✓ → {GOLD_PATH}/mesicni_trend")

    # ── Gold 4: Kompletní detail objednávek (pro BI / ad-hoc dotazy) ─────────
    print("\n[GOLD] 4/4 — detail objednávek...")
    gold_detail = (
        joined.select(
            "order_id", "order_date", "year", "month", "status",
            "customer_id",
            F.col("name").alias("customer_name"),
            "city", "segment",
            "product_id", "product_name", "category",
            "quantity", "unit_price", "total_price",
        )
        .orderBy("order_date")
    )
    gold_detail.show(5, truncate=False)
    print(f"  (zobrazeno 5 z {gold_detail.count()} řádků)")
    gold_detail.write.mode("overwrite").parquet(f"{GOLD_PATH}/objednavky_detail")
    print(f"  ✓ → {GOLD_PATH}/objednavky_detail")


# ── Validace ──────────────────────────────────────────────────────────────────

def validate(spark: SparkSession) -> None:
    print("\n[VALIDATE] Kontrola výstupů...")
    checks = [
        ("silver/orders",          "orders (silver)"),
        ("silver/customers",       "customers (silver)"),
        ("silver/products",        "products (silver)"),
        ("gold/trzby_kategorie",   "tržby podle kategorie"),
        ("gold/top_zakaznici",     "top zákazníci"),
        ("gold/mesicni_trend",     "měsíční trend"),
        ("gold/objednavky_detail", "detail objednávek"),
    ]
    all_ok = True
    for path, label in checks:
        try:
            df = spark.read.parquet(f"s3a://{MINIO_BUCKET}/processed/{path}")
            print(f"  ✓ {label:<30} {df.count():>5} řádků")
        except Exception as e:
            print(f"  ✗ {label:<30} CHYBA: {e}")
            all_ok = False
    print("[VALIDATE] " + ("Vše OK ✓" if all_ok else "Nalezeny chyby ✗"))


# ── Hlavní pipeline ───────────────────────────────────────────────────────────

def run_pipeline() -> None:
    print("=" * 60)
    print("  ETL Pipeline: orders / customers / products")
    print("  raw CSV → silver Parquet → gold analytické pohledy")
    print("=" * 60)

    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    try:
        # RAW → SILVER
        silver_orders(spark)
        silver_customers(spark)
        silver_products(spark)

        # SILVER → GOLD
        build_gold(spark)

        # Validace
        validate(spark)

        print("\n" + "=" * 60)
        print("  Pipeline dokončena ✓")
        print(f"  Silver → {SILVER_PATH}/")
        print(f"  Gold   → {GOLD_PATH}/")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Pipeline selhala: {e}")
        raise
    finally:
        spark.stop()


run_pipeline()