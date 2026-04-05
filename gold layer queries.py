"""
query_gold.py
─────────────
Ad-hoc SQL dotazy nad gold vrstvou v MinIO.

Načte gold Parquet tabulky jako Spark dočasné views a umožňuje
dotazovat je přes spark.sql() — stejná syntaxe jako běžný SQL.

Gold tabulky:
  trzby_kategorie   → tržby podle kategorie produktu
  top_zakaznici     → zákazníci seřazení podle tržeb
  mesicni_trend     → měsíční vývoj tržeb
  objednavky_detail → kompletní denormalizovaná tabulka

MinIO: http://minio:9000  |  bucket: datalake
"""

from pyspark.sql import SparkSession

# ── Konfigurace ───────────────────────────────────────────────────────────────

MINIO_ENDPOINT   = "http://minio:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "admin123"
MINIO_BUCKET     = "datalake"

HADOOP_AWS_VERSION = "3.3.4"
AWS_SDK_VERSION    = "1.12.367"

GOLD_PATH = f"s3a://{MINIO_BUCKET}/processed/gold"

# ── Spark session ─────────────────────────────────────────────────────────────

def get_spark() -> SparkSession:
    existing = SparkSession.getActiveSession()
    if existing:
        existing.stop()

    return (
        SparkSession.builder
        .appName("query_gold")
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
        .getOrCreate()
    )

# ── Načtení gold tabulek jako SQL views ───────────────────────────────────────

def load_gold_views(spark: SparkSession) -> None:
    """Načte všechny gold Parquet tabulky jako dočasné SQL views."""
    tables = [
        "trzby_kategorie",
        "top_zakaznici",
        "mesicni_trend",
        "objednavky_detail",
    ]
    print("[INFO] Načítám gold tabulky jako SQL views...")
    for table in tables:
        (
            spark.read
            .parquet(f"{GOLD_PATH}/{table}")
            .createOrReplaceTempView(table)
        )
        print(f"  ✓ {table}")
    print()

# ── Pomocná funkce pro spuštění dotazu ────────────────────────────────────────

def q(spark: SparkSession, title: str, sql: str, n: int = 20) -> None:
    """Spustí SQL dotaz, vypíše nadpis a výsledek."""
    print(f"{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    spark.sql(sql).show(n, truncate=False)


# ── Dotazy ────────────────────────────────────────────────────────────────────

def run_queries(spark: SparkSession) -> None:

    # ── 1. Celkové tržby a objednávky podle kategorie ─────────────────────────
    q(spark, "1. Tržby podle kategorie produktu", """
        SELECT
            category,
            pocet_objednavek,
            ROUND(trzby_celkem, 0)        AS trzby_celkem_czk,
            ROUND(prumerna_objednavka, 0) AS prumerna_objednavka_czk,
            prodano_kusu
        FROM trzby_kategorie
        ORDER BY trzby_celkem_czk DESC
    """)

    # ── 2. Top 5 zákazníků ────────────────────────────────────────────────────
    q(spark, "2. Top 5 zákazníků podle tržeb", """
        SELECT
            name,
            city,
            segment,
            pocet_objednavek,
            ROUND(trzby_celkem, 0) AS trzby_celkem_czk
        FROM top_zakaznici
        ORDER BY trzby_celkem DESC
        LIMIT 5
    """)

    # ── 3. B2B vs B2C srovnání ────────────────────────────────────────────────
    q(spark, "3. B2B vs B2C — srovnání segmentů", """
        SELECT
            segment,
            COUNT(*)                              AS pocet_zakazniku,
            SUM(pocet_objednavek)                 AS objednavky_celkem,
            ROUND(SUM(trzby_celkem), 0)           AS trzby_celkem_czk,
            ROUND(AVG(prumerna_objednavka), 0)    AS prumerna_objednavka_czk
        FROM top_zakaznici
        GROUP BY segment
        ORDER BY trzby_celkem_czk DESC
    """)

    # ── 4. Měsíční trend s kumulativním součtem ───────────────────────────────
    q(spark, "4. Měsíční trend tržeb + kumulativní součet", """
        SELECT
            year,
            month,
            pocet_objednavek,
            ROUND(trzby_celkem, 0) AS trzby_czk,
            ROUND(
                SUM(trzby_celkem) OVER (
                    PARTITION BY year ORDER BY month
                ), 0
            ) AS kumulativni_trzby_czk
        FROM mesicni_trend
        ORDER BY year, month
    """)

    # ── 5. Nejlepší měsíc v roce ──────────────────────────────────────────────
    q(spark, "5. Nejlepší a nejhorší měsíc roku", """
        SELECT
            month,
            ROUND(trzby_celkem, 0) AS trzby_czk,
            pocet_objednavek,
            RANK() OVER (ORDER BY trzby_celkem DESC) AS rank_desc
        FROM mesicni_trend
        ORDER BY trzby_czk DESC
    """)

    # ── 6. Průměrná hodnota objednávky podle města ────────────────────────────
    q(spark, "6. Průměrná hodnota objednávky podle města", """
        SELECT
            city,
            COUNT(order_id)                    AS pocet_objednavek,
            ROUND(SUM(total_price), 0)         AS trzby_celkem_czk,
            ROUND(AVG(total_price), 0)         AS prumerna_objednavka_czk,
            ROUND(MAX(total_price), 0)         AS nejvyssi_objednavka_czk
        FROM objednavky_detail
        WHERE status = 'completed'
        GROUP BY city
        ORDER BY trzby_celkem_czk DESC
    """)

    # ── 7. Kategorie × měsíc (pivot-style) ───────────────────────────────────
    q(spark, "7. Tržby podle kategorie a měsíce", """
        SELECT
            month,
            ROUND(SUM(CASE WHEN category = 'Electronics' THEN total_price ELSE 0 END), 0) AS electronics_czk,
            ROUND(SUM(CASE WHEN category = 'Furniture'   THEN total_price ELSE 0 END), 0) AS furniture_czk,
            ROUND(SUM(CASE WHEN category = 'Stationery'  THEN total_price ELSE 0 END), 0) AS stationery_czk,
            ROUND(SUM(total_price), 0)                                                     AS celkem_czk
        FROM objednavky_detail
        WHERE status = 'completed'
        GROUP BY month
        ORDER BY month
    """)

    # ── 8. Zákazníci bez objednávky v posledním kvartálu ─────────────────────
    q(spark, "8. Zákazníci neaktivní v Q4 (říjen–prosinec)", """
        SELECT
            name,
            city,
            segment,
            MAX(order_date) AS posledni_objednavka
        FROM objednavky_detail
        WHERE status = 'completed'
        GROUP BY name, city, segment
        HAVING MAX(month) < 10
        ORDER BY posledni_objednavka
    """)

    # ── 9. Vlastní dotaz — šablona ────────────────────────────────────────────
    print(f"{'─' * 60}")
    print("  9. Vlastní dotaz — upravte dle potřeby")
    print(f"{'─' * 60}")
    spark.sql("""
        SELECT *
        FROM objednavky_detail
        WHERE category = 'Electronics'
          AND status   = 'completed'
          AND quantity >= 5
        ORDER BY total_price DESC
        LIMIT 10
    """).show(truncate=False)


# ── Hlavní blok ───────────────────────────────────────────────────────────────

spark = get_spark()
spark.sparkContext.setLogLevel("WARN")

load_gold_views(spark)
run_queries(spark)

# spark.stop()   # odkomentuj pokud chceš session ukončit po dotazech
