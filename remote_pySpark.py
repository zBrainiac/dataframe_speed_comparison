"""
Fraud Risk Scoring — remote_pySpark: Classic PySpark (local Spark, data from Snowflake stage)

Replaces df.toPandas().to_csv() with Spark's native CSV writer
to avoid collectToPython OOM at write-back for large datasets (500K+ rows).
"""

VERSION = "remote_pySpark"

import os
os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 12g --conf spark.driver.extraJavaOptions=--add-opens=java.base/javax.security.auth=ALL-UNNAMED pyspark-shell"
import sys
import platform
import time
from datetime import datetime, timezone

import snowflake.connector
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, BooleanType,
)

SF_DB = os.getenv("SF_DATABASE", "MY_DATABASE")
SF_SCHEMA = os.getenv("SF_SCHEMA", "RISK_SCORING_MODEL")
SF_STAGE = os.getenv("SF_STAGE", "DATA")
FQN = f"@{SF_DB}.{SF_SCHEMA}.{SF_STAGE}"

DATA_MODE = {
    "small":  f"{FQN}/synthetic_fraud_data_small.csv",
    "500k":   f"{FQN}/synthetic_fraud_data_500k.csv",
    "1mio":   f"{FQN}/synthetic_fraud_data_1mio.csv",
    "medium": f"{FQN}/synthetic_fraud_data_medium.csv",
    "big":    f"{FQN}/synthetic_fraud_data.csv",
}
DATA_SIZE = os.getenv("DATA_SIZE", "small").lower()
DATA_IN = DATA_MODE[DATA_SIZE]
DATA_OUT = f"{SF_DB}.{SF_SCHEMA}.ENGINEERED_FEATURES"

STAGE_SCHEMA = StructType([
    StructField("transaction_id", StringType()),
    StructField("customer_id", StringType()),
    StructField("card_number", StringType()),
    StructField("timestamp", StringType()),
    StructField("merchant_category", StringType()),
    StructField("merchant_type", StringType()),
    StructField("merchant", StringType()),
    StructField("amount", DoubleType()),
    StructField("currency", StringType()),
    StructField("country", StringType()),
    StructField("city", StringType()),
    StructField("city_size", StringType()),
    StructField("card_type", StringType()),
    StructField("card_present", BooleanType()),
    StructField("device", StringType()),
    StructField("channel", StringType()),
    StructField("device_fingerprint", StringType()),
    StructField("ip_address", StringType()),
    StructField("distance_from_home", IntegerType()),
    StructField("high_risk_merchant", BooleanType()),
    StructField("transaction_hour", IntegerType()),
    StructField("weekend_transaction", BooleanType()),
    StructField("velocity_last_hour", StringType()),
    StructField("is_fraud", BooleanType()),
])

DROP_COLS = [
    "transaction_id", "customer_id", "card_number",
    "merchant", "device_fingerprint", "ip_address", "timestamp", "city",
]
LABEL_COL = "is_fraud"


def _parse_bool(v):
    return v.strip().lower() == "true" if v else None

def _parse_int(v):
    try: return int(v) if v else None
    except ValueError: return None

def _parse_float(v):
    try: return float(v) if v else None
    except ValueError: return None


# ─── LINE 1 of 2 that differs ───────────────────────────────────────
def get_spark():
    return SparkSession.builder.appName("FraudRiskScoring").master("local[2]").config("spark.driver.memory", "16g").getOrCreate()


# ─── LINE 2 of 2 that differs ───────────────────────────────────────
def load_data(spark):
    conn_name = os.getenv("SNOWFLAKE_CONNECTION_NAME", "default")
    print(f"    [load_data] Connecting to Snowflake (connection={conn_name}) ...")
    conn = snowflake.connector.connect(connection_name=conn_name)
    cur = conn.cursor()
    try:
        print(f"    [load_data] USE DATABASE {SF_DB}")
        cur.execute(f"USE DATABASE {SF_DB}")
        print(f"    [load_data] USE SCHEMA {SF_SCHEMA}")
        cur.execute(f"USE SCHEMA {SF_SCHEMA}")
        print(f"    [load_data] Creating temporary file format CSV_FF ...")
        cur.execute(
            "CREATE TEMPORARY FILE FORMAT IF NOT EXISTS CSV_FF "
            "TYPE='CSV' SKIP_HEADER=1 FIELD_OPTIONALLY_ENCLOSED_BY='\"'"
        )
        sql = (
            "SELECT $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,"
            "$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,"
            "$21,$22,$23,$24 "
            f"FROM {DATA_IN} (FILE_FORMAT => 'CSV_FF')"
        )
        print(f"    [load_data] Executing stage query: {DATA_IN}")
        cur.execute(sql)
        print(f"    [load_data] Fetching all rows into memory ...")
        rows = cur.fetchall()
        print(f"    [load_data] Fetched {len(rows)} rows from Snowflake stage")
    finally:
        cur.close()
        conn.close()
        print(f"    [load_data] Snowflake connection closed")

    print(f"    [load_data] Parsing & type-casting {len(rows)} rows ...")
    typed_rows = []
    for r in rows:
        typed_rows.append((
            r[0], r[1], r[2], r[3], r[4], r[5], r[6],
            _parse_float(r[7]), r[8], r[9], r[10], r[11], r[12],
            _parse_bool(r[13]), r[14], r[15], r[16], r[17],
            _parse_int(r[18]), _parse_bool(r[19]),
            _parse_int(r[20]), _parse_bool(r[21]),
            r[22], _parse_bool(r[23]),
        ))
    print(f"    [load_data] Creating DataFrame ({len(STAGE_SCHEMA.fields)} fields) ...")
    return spark.createDataFrame(typed_rows, schema=STAGE_SCHEMA)


# ═══════════════════════════════════════════════════════════════════════
#  Everything below is IDENTICAL in remote_pySpark.py and remote_SnowparkConnect.py
# ═══════════════════════════════════════════════════════════════════════

def engineer_features(df):
    for field in ["num_transactions", "total_amount", "unique_merchants",
                  "unique_countries", "max_single_amount"]:
        cast = IntegerType() if field in ("num_transactions", "unique_merchants", "unique_countries") else DoubleType()
        df = df.withColumn(
            f"velocity_{field}",
            F.get_json_object(
                F.regexp_replace("velocity_last_hour", "'", '"'),
                f"$.{field}",
            ).cast(cast),
        )
        print(f"    [feature_eng] Extracted velocity_{field}")
    df = df.drop("velocity_last_hour")
    print(f"    [feature_eng] Dropped velocity_last_hour (replaced by 5 numeric columns)")

    existing = set(df.columns)
    dropped = [c for c in DROP_COLS if c in existing]
    df = df.drop(*dropped)
    print(f"    [feature_eng] Dropped {len(dropped)} ID/PII columns: {dropped}")

    for col_name in ["card_present", "high_risk_merchant", "weekend_transaction"]:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(IntegerType()))
            print(f"    [feature_eng] Cast {col_name} -> IntegerType")
    df = df.withColumn("is_fraud", F.col("is_fraud").cast(IntegerType()))
    print(f"    [feature_eng] Cast is_fraud -> IntegerType")
    return df


def run_pipeline():
    telemetry = {}
    t_start = time.monotonic()
    start_ts = datetime.now(timezone.utc)

    print("=" * 60)
    print("  FRAUD RISK SCORING PIPELINE")
    print(f"  Version: {VERSION}")
    print("=" * 60)
    print(f"  Started at  : {start_ts:%Y-%m-%d %H:%M:%S %Z}")
    print(f"  Data size   : {DATA_SIZE}")
    print(f"  Data In     : {DATA_IN}")
    print(f"  Output      : {DATA_OUT}")
    print(f"  Python      : {sys.version}")
    print(f"  Platform    : {platform.platform()}")
    print(f"  PID         : {os.getpid()}")
    print(f"  Connection  : {os.getenv('SNOWFLAKE_CONNECTION_NAME', '(default)')}")
    print("=" * 60)

    print(f"\n[1/6] Initializing {VERSION} session ...")
    t0 = time.monotonic()
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    telemetry["session_init"] = time.monotonic() - t0
    print(f"       Spark version  : {spark.version}")
    print(f"       Master         : {spark.sparkContext.master}")
    print(f"       App name       : {spark.sparkContext.appName}")
    print(f"       Parallelism    : {spark.sparkContext.defaultParallelism}")
    print(f"       Done in {telemetry['session_init']:.2f}s")

    print(f"\n[2/6] Loading data from Snowflake ...")
    print(f"       Source: {DATA_IN}")
    t0 = time.monotonic()
    raw_df = load_data(spark)
    total_rows = raw_df.count()
    telemetry["data_load"] = time.monotonic() - t0
    print(f"       Rows loaded    : {total_rows:,}")
    print(f"       Columns        : {len(raw_df.columns)}")
    print(f"       Column names   : {raw_df.columns}")
    print(f"       Schema:")
    for field in raw_df.schema.fields:
        print(f"         {field.name:<30} {str(field.dataType):<20} nullable={field.nullable}")
    print(f"       Done in {telemetry['data_load']:.2f}s")

    print(f"\n[3/6] Engineering features ...")
    t0 = time.monotonic()
    df = engineer_features(raw_df)
    telemetry["feature_eng"] = time.monotonic() - t0
    print(f"       Columns before : {len(raw_df.columns)}")
    print(f"       Columns after  : {len(df.columns)}")
    print(f"       Final columns  : {df.columns}")
    print(f"       Done in {telemetry['feature_eng']:.2f}s")

    print(f"\n[4/6] Splitting data (75/25, seed=42) ...")
    t0 = time.monotonic()
    train_df, test_df = df.randomSplit([0.75, 0.25], seed=42)
    train_rows = train_df.count()
    test_rows = test_df.count()
    telemetry["split"] = time.monotonic() - t0
    print(f"       Train rows     : {train_rows:,} ({train_rows/total_rows*100:.1f}%)")
    print(f"       Test rows      : {test_rows:,} ({test_rows/total_rows*100:.1f}%)")
    print(f"       Done in {telemetry['split']:.2f}s")

    print(f"\n[5/6] Computing aggregation statistics ...")
    t0 = time.monotonic()

    print("\n--- Fraud Rate by Merchant Category ---")
    df.groupBy("merchant_category").agg(
        F.count("*").alias("total_txns"),
        F.sum(LABEL_COL).alias("fraud_count"),
        F.round(F.avg(F.col(LABEL_COL).cast("double")) * 100, 2).alias("fraud_rate_pct"),
        F.round(F.avg("amount"), 2).alias("avg_amount"),
        F.round(F.sum("amount"), 2).alias("total_amount"),
    ).orderBy(F.desc("fraud_rate_pct")).show(20, truncate=False)

    print("--- Fraud Rate by Country ---")
    df.groupBy("country").agg(
        F.count("*").alias("total_txns"),
        F.sum(LABEL_COL).alias("fraud_count"),
        F.round(F.avg(F.col(LABEL_COL).cast("double")) * 100, 2).alias("fraud_rate_pct"),
        F.round(F.avg("amount"), 2).alias("avg_amount"),
    ).orderBy(F.desc("fraud_rate_pct")).show(20, truncate=False)

    print("--- Fraud Rate by Channel & Device ---")
    df.groupBy("channel", "device").agg(
        F.count("*").alias("total_txns"),
        F.sum(LABEL_COL).alias("fraud_count"),
        F.round(F.avg(F.col(LABEL_COL).cast("double")) * 100, 2).alias("fraud_rate_pct"),
    ).orderBy(F.desc("fraud_rate_pct")).show(20, truncate=False)

    print("--- Velocity Stats by Fraud Flag ---")
    df.groupBy(LABEL_COL).agg(
        F.round(F.avg("velocity_num_transactions"), 2).alias("avg_velocity_txns"),
        F.round(F.avg("velocity_total_amount"), 2).alias("avg_velocity_amount"),
        F.round(F.avg("velocity_unique_merchants"), 2).alias("avg_velocity_merchants"),
        F.round(F.avg("velocity_max_single_amount"), 2).alias("avg_velocity_max_amount"),
    ).show(truncate=False)

    telemetry["aggregations"] = time.monotonic() - t0
    print(f"       Done in {telemetry['aggregations']:.2f}s")

    print(f"\n[6/6] Writing engineered features ...")
    t0 = time.monotonic()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "engineered_features_v1")
    df.repartition(4).write.mode("overwrite").option("header", "true").csv(out_path)
    telemetry["write_back"] = time.monotonic() - t0
    print(f"       Data Output    : {out_path}/")
    print(f"       Mode           : overwrite")
    print(f"       Format         : CSV (Spark native writer, no collectToPython)")
    print(f"       Done in {telemetry['write_back']:.2f}s")

    print("\n       Stopping Spark session ...")
    spark.stop()
    print("       Spark session stopped.")

    duration = time.monotonic() - t_start
    end_ts = datetime.now(timezone.utc)

    print("\n" + "=" * 60)
    print(f"  TELEMETRY: {VERSION}")
    print("=" * 60)
    print(f"  Started       : {start_ts:%Y-%m-%d %H:%M:%S %Z}")
    print(f"  Finished      : {end_ts:%Y-%m-%d %H:%M:%S %Z}")
    print(f"  Total duration : {duration:.2f}s")
    print(f"  Total rows     : {total_rows:,}")
    print(f"  Data size      : {DATA_SIZE}")
    print("-" * 60)
    for step, elapsed in telemetry.items():
        pct = (elapsed / duration) * 100
        bar = "#" * int(pct / 2)
        print(f"  {step:<16}: {elapsed:>8.2f}s  ({pct:>5.1f}%)  {bar}")
    print("-" * 60)
    print(f"  {'TOTAL':<16}: {duration:>8.2f}s  (100.0%)")
    print("=" * 60)
    print(">>> Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
