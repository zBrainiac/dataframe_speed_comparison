"""
Fraud Risk Scoring — V3: Snowpark Connect (optimized)

Compare with run_v1.py — only 2 secions differ:
  1. SparkSession creation  (local[*] vs snowpark_connect)
  2. Data loading           (stage via connector vs spark.read.table)
"""

VERSION = "V3"
VERSION_LABEL = "Snowpark Connect (Snowflake compute)"

import os
import sys
import platform
import time
from datetime import datetime, timezone



from snowflake import snowpark_connect
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, BooleanType,
)

SF_DB = os.getenv("SF_DATABASE", "MY_DATABASE")
SF_SCHEMA = os.getenv("SF_SCHEMA", "RISK_SCORING_MODEL")

DATA_MODE = {
    "small":  f"{SF_DB}.{SF_SCHEMA}.RAW_TRANSACTIONS_SMALL",
    "500k":   f"{SF_DB}.{SF_SCHEMA}.RAW_TRANSACTIONS_500K",
    "1mio":   f"{SF_DB}.{SF_SCHEMA}.RAW_TRANSACTIONS_1MIO",
    "medium": f"{SF_DB}.{SF_SCHEMA}.RAW_TRANSACTIONS_MEDIUM",
    "big":    f"{SF_DB}.{SF_SCHEMA}.RAW_TRANSACTIONS",
}
DATA_SIZE = os.getenv("DATA_SIZE", "small").lower()
DATA_IN = DATA_MODE[DATA_SIZE]
DATA_OUT = f"{SF_DB}.{SF_SCHEMA}.ENGINEERED_FEATURES"

DROP_COLS = [
    "transaction_id", "customer_id", "card_number",
    "merchant", "device_fingerprint", "ip_address", "timestamp", "city",
]
LABEL_COL = "is_fraud"


# ─── LINE 1 of 2 that differs ───────────────────────────────────────
def get_spark():
    return snowpark_connect.init_spark_session()


# ─── LINE 2 of 2 that differs ───────────────────────────────────────
def load_data(spark):
    print(f"    [load_data] Reading Snowflake table: {DATA_IN}")
    df = spark.read.table(DATA_IN)
    df = df.toDF(*[c.lower() for c in df.columns])
    print(f"    [load_data] Creating DataFrame")
    return df


# ═══════════════════════════════════════════════════════════════════════
#  Everything below is IDENTICAL in run_v1.py and run_v2.py
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
    print(f"  Version: {VERSION} — {VERSION_LABEL}")
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

    print(f"\n[1/6] Initializing {VERSION_LABEL} session ...")
    t0 = time.monotonic()
    spark = get_spark()
    telemetry["session_init"] = time.monotonic() - t0
    print(f"       Spark version  : {spark.version}")
    print(f"       Session type   : {VERSION_LABEL}")
    print(f"       Done in {telemetry['session_init']:.2f}s")

    print(f"\n[2/6] Loading data from Snowflake ...")
    print(f"       Source: {DATA_IN}")
    t0 = time.monotonic()
    raw_df = load_data(spark)
    telemetry["data_load"] = time.monotonic() - t0
    print(f"       Columns        : {len(raw_df.columns)}")
    print(f"       Column names   : {raw_df.columns}")
    print(f"       Schema:")
    for field in raw_df.schema.fields:
        print(f"         {field.name:<30} {str(field.dataType):<20} nullable={field.nullable}")
    print(f"       Done in {telemetry['data_load']:.2f}s")

    print(f"\n[3/6] Engineering features ...")
    t0 = time.monotonic()
    df = engineer_features(raw_df)
    df = df.cache()
    total_rows = df.count()
    telemetry["feature_eng"] = time.monotonic() - t0
    print(f"       Rows           : {total_rows:,}")
    print(f"       Columns before : {len(raw_df.columns)}")
    print(f"       Columns after  : {len(df.columns)}")
    print(f"       Final columns  : {df.columns}")
    print(f"       Done in {telemetry['feature_eng']:.2f}s")

    print(f"\n[4/6] Splitting data (75/25, seed=42) ...")
    t0 = time.monotonic()
    train_df, test_df = df.randomSplit([0.75, 0.25], seed=42)
    telemetry["split"] = time.monotonic() - t0
    train_est = int(total_rows * 0.75)
    test_est = total_rows - train_est
    print(f"       Train rows     : ~{train_est:,} (75.0%)")
    print(f"       Test rows      : ~{test_est:,} (25.0%)")
    print(f"       Done in {telemetry['split']:.2f}s")

    print(f"\n[5/6] Computing aggregation statistics ...")
    t0 = time.monotonic()
    agg_cols = ["merchant_category", "country", "channel", "device", "amount",
                LABEL_COL, "velocity_num_transactions", "velocity_total_amount",
                "velocity_unique_merchants", "velocity_max_single_amount"]
    agg_df = df.select(*agg_cols)

    print("\n--- Fraud Rate by Merchant Category ---")
    agg_df.groupBy("merchant_category").agg(
        F.count("*").alias("total_txns"),
        F.sum(LABEL_COL).alias("fraud_count"),
        F.round(F.avg(F.col(LABEL_COL).cast("double")) * 100, 2).alias("fraud_rate_pct"),
        F.round(F.avg("amount"), 2).alias("avg_amount"),
        F.round(F.sum("amount"), 2).alias("total_amount"),
    ).orderBy(F.desc("fraud_rate_pct")).show(20, truncate=False)

    print("--- Fraud Rate by Country ---")
    agg_df.groupBy("country").agg(
        F.count("*").alias("total_txns"),
        F.sum(LABEL_COL).alias("fraud_count"),
        F.round(F.avg(F.col(LABEL_COL).cast("double")) * 100, 2).alias("fraud_rate_pct"),
        F.round(F.avg("amount"), 2).alias("avg_amount"),
    ).orderBy(F.desc("fraud_rate_pct")).show(20, truncate=False)

    print("--- Fraud Rate by Channel & Device ---")
    agg_df.groupBy("channel", "device").agg(
        F.count("*").alias("total_txns"),
        F.sum(LABEL_COL).alias("fraud_count"),
        F.round(F.avg(F.col(LABEL_COL).cast("double")) * 100, 2).alias("fraud_rate_pct"),
    ).orderBy(F.desc("fraud_rate_pct")).show(20, truncate=False)

    print("--- Velocity Stats by Fraud Flag ---")
    agg_df.groupBy(LABEL_COL).agg(
        F.round(F.avg("velocity_num_transactions"), 2).alias("avg_velocity_txns"),
        F.round(F.avg("velocity_total_amount"), 2).alias("avg_velocity_amount"),
        F.round(F.avg("velocity_unique_merchants"), 2).alias("avg_velocity_merchants"),
        F.round(F.avg("velocity_max_single_amount"), 2).alias("avg_velocity_max_amount"),
    ).show(truncate=False)

    telemetry["aggregations"] = time.monotonic() - t0
    print(f"       Done in {telemetry['aggregations']:.2f}s")

    print(f"\n[6/6] Writing engineered features to Snowflake ...")
    t0 = time.monotonic()
    df.unpersist()
    df.write.mode("overwrite").saveAsTable(DATA_OUT)
    telemetry["write_back"] = time.monotonic() - t0
    print(f"       Data Output   : {DATA_OUT}")
    print(f"       Mode           : overwrite")
    print(f"       Format         : Snowflake managed table (native catalog)")
    print(f"       Done in {telemetry['write_back']:.2f}s")

    print("\n       Stopping Spark session ...")
    spark.stop()
    print("       Spark session stopped.")

    duration = time.monotonic() - t_start
    end_ts = datetime.now(timezone.utc)

    print("\n" + "=" * 60)
    print(f"  TELEMETRY: {VERSION} {VERSION_LABEL}")
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
