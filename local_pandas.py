"""
Fraud Risk Scoring — local_pandas: Pure Pandas (local CSV files, no PySpark)

Baseline version without any Spark dependency. Reads data from local CSV
files in ./data/ and processes everything with pandas. Same pipeline steps
as local_pySpark for a direct apples-to-apples comparison.
"""

VERSION = "local_pandas"

import os
import sys
import json
import platform
import time
from datetime import datetime, timezone

import pandas as pd
import numpy as np

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

DTYPE_MAP = {
    "transaction_id": str,
    "customer_id": str,
    "card_number": str,
    "timestamp": str,
    "merchant_category": str,
    "merchant_type": str,
    "merchant": str,
    "amount": float,
    "currency": str,
    "country": str,
    "city": str,
    "city_size": str,
    "card_type": str,
    "card_present": str,
    "device": str,
    "channel": str,
    "device_fingerprint": str,
    "ip_address": str,
    "distance_from_home": "Int64",
    "high_risk_merchant": str,
    "transaction_hour": "Int64",
    "weekend_transaction": str,
    "velocity_last_hour": str,
    "is_fraud": str,
}

DROP_COLS = [
    "transaction_id", "customer_id", "card_number",
    "merchant", "device_fingerprint", "ip_address", "timestamp", "city",
]
LABEL_COL = "is_fraud"

BOOL_MAP = {"true": True, "false": False, "True": True, "False": False}


def load_data():
    print(f"    [load_data] Reading local CSV: {DATA_IN}")
    df = pd.read_csv(DATA_IN, dtype=DTYPE_MAP)
    for col in ["card_present", "high_risk_merchant", "weekend_transaction", "is_fraud"]:
        df[col] = df[col].map(BOOL_MAP)
    print(f"    [load_data] CSV loaded successfully")
    return df


def engineer_features(df):
    velocity_fields = {
        "num_transactions": "Int64",
        "total_amount": float,
        "unique_merchants": "Int64",
        "unique_countries": "Int64",
        "max_single_amount": float,
    }
    for field, dtype in velocity_fields.items():
        df[f"velocity_{field}"] = df["velocity_last_hour"].apply(
            lambda v: json.loads(v.replace("'", '"')).get(field) if pd.notna(v) else None
        ).astype(dtype)
        print(f"    [feature_eng] Extracted velocity_{field}")

    df = df.drop(columns=["velocity_last_hour"])
    print(f"    [feature_eng] Dropped velocity_last_hour (replaced by 5 numeric columns)")

    existing = set(df.columns)
    dropped = [c for c in DROP_COLS if c in existing]
    df = df.drop(columns=dropped)
    print(f"    [feature_eng] Dropped {len(dropped)} ID/PII columns: {dropped}")

    for col_name in ["card_present", "high_risk_merchant", "weekend_transaction"]:
        if col_name in df.columns:
            df[col_name] = df[col_name].astype("Int64")
            print(f"    [feature_eng] Cast {col_name} -> Int64")
    df["is_fraud"] = df["is_fraud"].astype("Int64")
    print(f"    [feature_eng] Cast is_fraud -> Int64")
    return df


def show_df(df, title, n=20):
    print(f"\n{title}")
    print(df.head(n).to_string(index=False))
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
    print(f"  Pandas      : {pd.__version__}")
    print(f"  NumPy       : {np.__version__}")
    print("=" * 60)

    print(f"\n[1/6] Initializing {VERSION} session ...")
    t0 = time.monotonic()
    telemetry["session_init"] = time.monotonic() - t0
    print(f"       Engine         : pandas {pd.__version__}")
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
    print(f"       Column names   : {list(raw_df.columns)}")
    print(f"       Schema:")
    for col in raw_df.columns:
        print(f"         {col:<30} {str(raw_df[col].dtype):<20} nulls={raw_df[col].isna().sum()}")
    print(f"       Memory usage   : {raw_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"       Done in {telemetry['data_load']:.2f}s")

    print(f"\n[3/6] Engineering features ...")
    t0 = time.monotonic()
    cols_before = len(raw_df.columns)
    df = engineer_features(raw_df)
    telemetry["feature_eng"] = time.monotonic() - t0
    print(f"       Columns before : {cols_before}")
    print(f"       Columns after  : {len(df.columns)}")
    print(f"       Final columns  : {list(df.columns)}")
    print(f"       Done in {telemetry['feature_eng']:.2f}s")

    print(f"\n[4/6] Splitting data (75/25, seed=42) ...")
    t0 = time.monotonic()
    np.random.seed(42)
    mask = np.random.rand(len(df)) < 0.75
    train_df = df[mask].copy()
    test_df = df[~mask].copy()
    train_rows = len(train_df)
    test_rows = len(test_df)
    telemetry["split"] = time.monotonic() - t0
    print(f"       Train rows     : {train_rows:,} ({train_rows/total_rows*100:.1f}%)")
    print(f"       Test rows      : {test_rows:,} ({test_rows/total_rows*100:.1f}%)")
    print(f"       Done in {telemetry['split']:.2f}s")

    print(f"\n[5/6] Computing aggregation statistics ...")
    t0 = time.monotonic()

    agg1 = df.groupby("merchant_category").agg(
        total_txns=("is_fraud", "count"),
        fraud_count=("is_fraud", "sum"),
        fraud_rate_pct=("is_fraud", lambda x: round(x.mean() * 100, 2)),
        avg_amount=("amount", lambda x: round(x.mean(), 2)),
        total_amount=("amount", lambda x: round(x.sum(), 2)),
    ).sort_values("fraud_rate_pct", ascending=False)
    show_df(agg1.reset_index(), "--- Fraud Rate by Merchant Category ---")

    agg2 = df.groupby("country").agg(
        total_txns=("is_fraud", "count"),
        fraud_count=("is_fraud", "sum"),
        fraud_rate_pct=("is_fraud", lambda x: round(x.mean() * 100, 2)),
        avg_amount=("amount", lambda x: round(x.mean(), 2)),
    ).sort_values("fraud_rate_pct", ascending=False)
    show_df(agg2.reset_index(), "--- Fraud Rate by Country ---")

    agg3 = df.groupby(["channel", "device"]).agg(
        total_txns=("is_fraud", "count"),
        fraud_count=("is_fraud", "sum"),
        fraud_rate_pct=("is_fraud", lambda x: round(x.mean() * 100, 2)),
    ).sort_values("fraud_rate_pct", ascending=False)
    show_df(agg3.reset_index(), "--- Fraud Rate by Channel & Device ---")

    agg4 = df.groupby(LABEL_COL).agg(
        avg_velocity_txns=("velocity_num_transactions", lambda x: round(x.mean(), 2)),
        avg_velocity_amount=("velocity_total_amount", lambda x: round(x.mean(), 2)),
        avg_velocity_merchants=("velocity_unique_merchants", lambda x: round(x.mean(), 2)),
        avg_velocity_max_amount=("velocity_max_single_amount", lambda x: round(x.mean(), 2)),
    )
    show_df(agg4.reset_index(), "--- Velocity Stats by Fraud Flag ---")

    telemetry["aggregations"] = time.monotonic() - t0
    print(f"       Done in {telemetry['aggregations']:.2f}s")

    print(f"\n[6/6] Writing engineered features ...")
    t0 = time.monotonic()
    out_path = os.path.join(OUTPUT_DIR, "engineered_features_pandas.csv")
    df.to_csv(out_path, index=False)
    telemetry["write_back"] = time.monotonic() - t0
    print(f"       Data Output    : {out_path}")
    print(f"       Mode           : overwrite")
    print(f"       Format         : CSV (pandas)")
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
