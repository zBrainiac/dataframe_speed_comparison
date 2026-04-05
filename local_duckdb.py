"""
Fraud Risk Scoring — local_duckdb: Pure DuckDB SQL (local CSV files, no PySpark)

Baseline version without any Spark/JVM dependency. Reads data from local CSV
files in ./data/ and processes everything with DuckDB's SQL engine.
Same pipeline steps as local_pySpark for a direct apples-to-apples comparison.
"""

VERSION = "local_duckdb"

import os
import sys
import platform
import time
from datetime import datetime, timezone

import duckdb

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
    print(f"  DuckDB      : {duckdb.__version__}")
    print("=" * 60)

    # ─── [1/6] Session init ─────────────────────────────────────────
    print(f"\n[1/6] Initializing {VERSION} session ...")
    t0 = time.monotonic()
    con = duckdb.connect()
    telemetry["session_init"] = time.monotonic() - t0
    print(f"       Engine         : DuckDB {duckdb.__version__}")
    print(f"       No JVM needed")
    print(f"       Done in {telemetry['session_init']:.2f}s")

    # ─── [2/6] Data load ────────────────────────────────────────────
    print(f"\n[2/6] Loading data ...")
    print(f"       Source: {DATA_IN}")
    t0 = time.monotonic()
    con.execute(f"""
        CREATE TABLE raw AS
        SELECT * FROM read_csv('{DATA_IN}',
            header=true,
            auto_detect=true,
            all_varchar=false
        )
    """)
    total_rows = con.execute("SELECT count(*) FROM raw").fetchone()[0]
    telemetry["data_load"] = time.monotonic() - t0

    schema = con.execute("DESCRIBE raw").fetchall()
    print(f"       Rows loaded    : {total_rows:,}")
    print(f"       Columns        : {len(schema)}")
    print(f"       Column names   : {[r[0] for r in schema]}")
    print(f"       Schema:")
    for col_name, col_type, *_ in schema:
        print(f"         {col_name:<30} {col_type:<20}")
    print(f"       Done in {telemetry['data_load']:.2f}s")

    # ─── [3/6] Feature engineering ──────────────────────────────────
    print(f"\n[3/6] Engineering features ...")
    t0 = time.monotonic()
    cols_before = len(schema)

    velocity_fields = {
        "num_transactions": "INTEGER",
        "total_amount": "DOUBLE",
        "unique_merchants": "INTEGER",
        "unique_countries": "INTEGER",
        "max_single_amount": "DOUBLE",
    }

    velocity_exprs = []
    for field, dtype in velocity_fields.items():
        velocity_exprs.append(
            f"CAST(json_extract_string(replace(velocity_last_hour, '''', '\"'), '$.{field}') AS {dtype}) AS velocity_{field}"
        )
        print(f"    [feature_eng] Extracted velocity_{field}")

    drop_set = set(DROP_COLS + ["velocity_last_hour"])
    keep_cols = [r[0] for r in schema if r[0] not in drop_set]
    dropped = [c for c in DROP_COLS if c in [r[0] for r in schema]]
    print(f"    [feature_eng] Dropped velocity_last_hour (replaced by 5 numeric columns)")
    print(f"    [feature_eng] Dropped {len(dropped)} ID/PII columns: {dropped}")

    bool_cast_cols = ["card_present", "high_risk_merchant", "weekend_transaction", "is_fraud"]
    select_parts = []
    for c in keep_cols:
        if c in bool_cast_cols:
            select_parts.append(f"CAST({c} AS INTEGER) AS {c}")
            print(f"    [feature_eng] Cast {c} -> INTEGER")
        else:
            select_parts.append(c)

    select_sql = ", ".join(select_parts + velocity_exprs)
    con.execute(f"CREATE TABLE engineered AS SELECT {select_sql} FROM raw")

    eng_schema = con.execute("DESCRIBE engineered").fetchall()
    telemetry["feature_eng"] = time.monotonic() - t0
    print(f"       Columns before : {cols_before}")
    print(f"       Columns after  : {len(eng_schema)}")
    print(f"       Final columns  : {[r[0] for r in eng_schema]}")
    print(f"       Done in {telemetry['feature_eng']:.2f}s")

    # ─── [4/6] Split ────────────────────────────────────────────────
    print(f"\n[4/6] Splitting data (75/25, seed=42) ...")
    t0 = time.monotonic()
    con.execute("SELECT setseed(0.42)")
    con.execute("CREATE TABLE train AS SELECT * FROM engineered WHERE random() < 0.75")
    con.execute("CREATE TABLE test AS SELECT * FROM engineered WHERE rowid NOT IN (SELECT rowid FROM train)")

    train_rows = con.execute("SELECT count(*) FROM train").fetchone()[0]
    test_rows = con.execute("SELECT count(*) FROM test").fetchone()[0]
    telemetry["split"] = time.monotonic() - t0
    print(f"       Train rows     : {train_rows:,} ({train_rows/total_rows*100:.1f}%)")
    print(f"       Test rows      : {test_rows:,} ({test_rows/total_rows*100:.1f}%)")
    print(f"       Done in {telemetry['split']:.2f}s")

    # ─── [5/6] Aggregations ─────────────────────────────────────────
    print(f"\n[5/6] Computing aggregation statistics ...")
    t0 = time.monotonic()

    print("\n--- Fraud Rate by Merchant Category ---")
    print(con.sql("""
        SELECT merchant_category,
               count(*) AS total_txns,
               sum(is_fraud) AS fraud_count,
               round(avg(CAST(is_fraud AS DOUBLE)) * 100, 2) AS fraud_rate_pct,
               round(avg(amount), 2) AS avg_amount,
               round(sum(amount), 2) AS total_amount
        FROM engineered
        GROUP BY merchant_category
        ORDER BY fraud_rate_pct DESC
        LIMIT 20
    """))
    print()

    print("--- Fraud Rate by Country ---")
    print(con.sql("""
        SELECT country,
               count(*) AS total_txns,
               sum(is_fraud) AS fraud_count,
               round(avg(CAST(is_fraud AS DOUBLE)) * 100, 2) AS fraud_rate_pct,
               round(avg(amount), 2) AS avg_amount
        FROM engineered
        GROUP BY country
        ORDER BY fraud_rate_pct DESC
        LIMIT 20
    """))
    print()

    print("--- Fraud Rate by Channel & Device ---")
    print(con.sql("""
        SELECT channel, device,
               count(*) AS total_txns,
               sum(is_fraud) AS fraud_count,
               round(avg(CAST(is_fraud AS DOUBLE)) * 100, 2) AS fraud_rate_pct
        FROM engineered
        GROUP BY channel, device
        ORDER BY fraud_rate_pct DESC
        LIMIT 20
    """))
    print()

    print("--- Velocity Stats by Fraud Flag ---")
    print(con.sql("""
        SELECT is_fraud,
               round(avg(velocity_num_transactions), 2) AS avg_velocity_txns,
               round(avg(velocity_total_amount), 2) AS avg_velocity_amount,
               round(avg(velocity_unique_merchants), 2) AS avg_velocity_merchants,
               round(avg(velocity_max_single_amount), 2) AS avg_velocity_max_amount
        FROM engineered
        GROUP BY is_fraud
    """))
    print()

    telemetry["aggregations"] = time.monotonic() - t0
    print(f"       Done in {telemetry['aggregations']:.2f}s")

    # ─── [6/6] Write back ───────────────────────────────────────────
    print(f"\n[6/6] Writing engineered features ...")
    t0 = time.monotonic()
    out_path = os.path.join(OUTPUT_DIR, "engineered_features_duckdb.csv")
    con.execute(f"COPY engineered TO '{out_path}' (HEADER, DELIMITER ',')")
    telemetry["write_back"] = time.monotonic() - t0
    print(f"       Data Output    : {out_path}")
    print(f"       Mode           : overwrite")
    print(f"       Format         : CSV (DuckDB COPY)")
    print(f"       Done in {telemetry['write_back']:.2f}s")

    con.close()

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
