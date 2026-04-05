"""
Fraud Risk Scoring — local_polars: Polars DataFrame (local CSV files, no PySpark)

Baseline version without any Spark/JVM dependency. Reads data from local CSV
files in ./data/ and processes everything with Polars' lazy/eager API.
Same pipeline steps as local_pySpark for a direct apples-to-apples comparison.
"""

VERSION = "local_polars"

import os
import sys
import platform
import time
from datetime import datetime, timezone

import polars as pl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_MODE = {
    "small":  os.path.join(DATA_DIR, "synthetic_fraud_data_small.csv"),
    "500k":   os.path.join(DATA_DIR, "synthetic_fraud_data_500k.csv"),
    "1mio":   os.path.join(DATA_DIR, "synthetic_fraud_data_1mio.csv"),
    "medium": os.path.join(DATA_DIR, "synthetic_fraud_data_medium.csv"),
    "big":    os.path.join(DATA_DIR, "synthetic_fraud_data.csv"),
}
DATA_SIZE = os.getenv("DATA_SIZE", "small").lower()
DATA_IN = DATA_MODE[DATA_SIZE]
DATA_OUT = os.getenv("SF_TABLE_OUT", "MY_DATABASE.RISK_SCORING_MODEL.ENGINEERED_FEATURES")

DROP_COLS = [
    "transaction_id", "customer_id", "card_number",
    "merchant", "device_fingerprint", "ip_address", "timestamp", "city",
]
LABEL_COL = "is_fraud"

SCHEMA_OVERRIDES = {
    "transaction_id": pl.Utf8,
    "customer_id": pl.Utf8,
    "card_number": pl.Utf8,
    "timestamp": pl.Utf8,
    "merchant_category": pl.Utf8,
    "merchant_type": pl.Utf8,
    "merchant": pl.Utf8,
    "amount": pl.Float64,
    "currency": pl.Utf8,
    "country": pl.Utf8,
    "city": pl.Utf8,
    "city_size": pl.Utf8,
    "card_type": pl.Utf8,
    "card_present": pl.Utf8,
    "device": pl.Utf8,
    "channel": pl.Utf8,
    "device_fingerprint": pl.Utf8,
    "ip_address": pl.Utf8,
    "distance_from_home": pl.Int64,
    "high_risk_merchant": pl.Utf8,
    "transaction_hour": pl.Int64,
    "weekend_transaction": pl.Utf8,
    "velocity_last_hour": pl.Utf8,
    "is_fraud": pl.Utf8,
}

BOOL_MAP = {"true": True, "false": False, "True": True, "False": False}


def load_data():
    print(f"    [load_data] Reading local CSV: {DATA_IN}")
    df = pl.read_csv(DATA_IN, schema_overrides=SCHEMA_OVERRIDES)
    for col in ["card_present", "high_risk_merchant", "weekend_transaction", "is_fraud"]:
        df = df.with_columns(
            pl.col(col).map_elements(lambda v: BOOL_MAP.get(v), return_dtype=pl.Boolean).alias(col)
        )
    print(f"    [load_data] CSV loaded successfully")
    return df


def engineer_features(df):
    velocity_fields = {
        "num_transactions": pl.Int64,
        "total_amount": pl.Float64,
        "unique_merchants": pl.Int64,
        "unique_countries": pl.Int64,
        "max_single_amount": pl.Float64,
    }
    for field, dtype in velocity_fields.items():
        df = df.with_columns(
            pl.col("velocity_last_hour")
            .str.replace_all("'", '"')
            .str.json_path_match(f"$.{field}")
            .cast(dtype)
            .alias(f"velocity_{field}")
        )
        print(f"    [feature_eng] Extracted velocity_{field}")

    df = df.drop("velocity_last_hour")
    print(f"    [feature_eng] Dropped velocity_last_hour (replaced by 5 numeric columns)")

    existing = set(df.columns)
    dropped = [c for c in DROP_COLS if c in existing]
    df = df.drop(dropped)
    print(f"    [feature_eng] Dropped {len(dropped)} ID/PII columns: {dropped}")

    for col_name in ["card_present", "high_risk_merchant", "weekend_transaction", "is_fraud"]:
        if col_name in df.columns:
            df = df.with_columns(pl.col(col_name).cast(pl.Int32).alias(col_name))
            print(f"    [feature_eng] Cast {col_name} -> Int32")
    return df


def show_df(df, title, n=20):
    print(f"\n{title}")
    print(df.head(n))
    print()


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
    print(f"  Polars      : {pl.__version__}")
    print("=" * 60)

    print(f"\n[1/6] Initializing {VERSION} session ...")
    t0 = time.monotonic()
    telemetry["session_init"] = time.monotonic() - t0
    print(f"       Engine         : Polars {pl.__version__}")
    print(f"       No JVM needed")
    print(f"       Done in {telemetry['session_init']:.2f}s")

    print(f"\n[2/6] Loading data ...")
    print(f"       Source: {DATA_IN}")
    t0 = time.monotonic()
    raw_df = load_data()
    total_rows = len(raw_df)
    telemetry["data_load"] = time.monotonic() - t0
    print(f"       Rows loaded    : {total_rows:,}")
    print(f"       Columns        : {len(raw_df.columns)}")
    print(f"       Column names   : {raw_df.columns}")
    print(f"       Schema:")
    for col_name, col_dtype in zip(raw_df.columns, raw_df.dtypes):
        null_count = raw_df[col_name].null_count()
        print(f"         {col_name:<30} {str(col_dtype):<20} nulls={null_count}")
    mem_bytes = raw_df.estimated_size()
    print(f"       Memory usage   : {mem_bytes / 1024**2:.1f} MB")
    print(f"       Done in {telemetry['data_load']:.2f}s")

    print(f"\n[3/6] Engineering features ...")
    t0 = time.monotonic()
    cols_before = len(raw_df.columns)
    df = engineer_features(raw_df)
    telemetry["feature_eng"] = time.monotonic() - t0
    print(f"       Columns before : {cols_before}")
    print(f"       Columns after  : {len(df.columns)}")
    print(f"       Final columns  : {df.columns}")
    print(f"       Done in {telemetry['feature_eng']:.2f}s")

    print(f"\n[4/6] Splitting data (75/25, seed=42) ...")
    t0 = time.monotonic()
    df = df.with_row_index("_idx")
    shuffled = df.sample(fraction=1.0, seed=42, shuffle=True)
    split_point = int(total_rows * 0.75)
    train_df = shuffled.head(split_point).drop("_idx")
    test_df = shuffled.tail(total_rows - split_point).drop("_idx")
    df = df.drop("_idx")
    train_rows = len(train_df)
    test_rows = len(test_df)
    telemetry["split"] = time.monotonic() - t0
    print(f"       Train rows     : {train_rows:,} ({train_rows/total_rows*100:.1f}%)")
    print(f"       Test rows      : {test_rows:,} ({test_rows/total_rows*100:.1f}%)")
    print(f"       Done in {telemetry['split']:.2f}s")

    print(f"\n[5/6] Computing aggregation statistics ...")
    t0 = time.monotonic()

    agg1 = df.group_by("merchant_category").agg(
        pl.len().alias("total_txns"),
        pl.col(LABEL_COL).sum().alias("fraud_count"),
        (pl.col(LABEL_COL).cast(pl.Float64).mean() * 100).round(2).alias("fraud_rate_pct"),
        pl.col("amount").mean().round(2).alias("avg_amount"),
        pl.col("amount").sum().round(2).alias("total_amount"),
    ).sort("fraud_rate_pct", descending=True)
    show_df(agg1, "--- Fraud Rate by Merchant Category ---")

    agg2 = df.group_by("country").agg(
        pl.len().alias("total_txns"),
        pl.col(LABEL_COL).sum().alias("fraud_count"),
        (pl.col(LABEL_COL).cast(pl.Float64).mean() * 100).round(2).alias("fraud_rate_pct"),
        pl.col("amount").mean().round(2).alias("avg_amount"),
    ).sort("fraud_rate_pct", descending=True)
    show_df(agg2, "--- Fraud Rate by Country ---")

    agg3 = df.group_by("channel", "device").agg(
        pl.len().alias("total_txns"),
        pl.col(LABEL_COL).sum().alias("fraud_count"),
        (pl.col(LABEL_COL).cast(pl.Float64).mean() * 100).round(2).alias("fraud_rate_pct"),
    ).sort("fraud_rate_pct", descending=True)
    show_df(agg3, "--- Fraud Rate by Channel & Device ---")

    agg4 = df.group_by(LABEL_COL).agg(
        pl.col("velocity_num_transactions").mean().round(2).alias("avg_velocity_txns"),
        pl.col("velocity_total_amount").mean().round(2).alias("avg_velocity_amount"),
        pl.col("velocity_unique_merchants").mean().round(2).alias("avg_velocity_merchants"),
        pl.col("velocity_max_single_amount").mean().round(2).alias("avg_velocity_max_amount"),
    )
    show_df(agg4, "--- Velocity Stats by Fraud Flag ---")

    telemetry["aggregations"] = time.monotonic() - t0
    print(f"       Done in {telemetry['aggregations']:.2f}s")

    print(f"\n[6/6] Writing engineered features ...")
    t0 = time.monotonic()
    out_path = os.path.join(OUTPUT_DIR, "engineered_features_polars.csv")
    df.write_csv(out_path)
    telemetry["write_back"] = time.monotonic() - t0
    print(f"       Data Output    : {out_path}")
    print(f"       Mode           : overwrite")
    print(f"       Format         : CSV (Polars)")
    print(f"       Done in {telemetry['write_back']:.2f}s")

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
