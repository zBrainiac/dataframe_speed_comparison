"""
Run V0 (local CSV), V1 (Classic PySpark), V2 (Snowpark Connect) and
V3 (Snowpark Connect optimized) side-by-side and compare.

Usage:
    python run_comparison.py              # small dataset (~3K rows)
    python run_comparison.py --big        # big dataset (~7.5M rows)
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

PYTHON = os.path.join(os.path.dirname(__file__), ".venv", "bin", "python")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

V0_SCRIPT = os.path.join(SCRIPT_DIR, "run_v0.py")
V1_SCRIPT = os.path.join(SCRIPT_DIR, "run_v1.py")
V2_SCRIPT = os.path.join(SCRIPT_DIR, "run_v2.py")
V3_SCRIPT = os.path.join(SCRIPT_DIR, "run_v3.py")

CONN_NAME = os.getenv("SNOWFLAKE_CONNECTION_NAME", "default")

V0_CONN = CONN_NAME
V1_CONN = CONN_NAME
V2_CONN = CONN_NAME
V3_CONN = CONN_NAME

SF_DB = os.getenv("SF_DATABASE", "MY_DATABASE")
SF_SCHEMA = os.getenv("SF_SCHEMA", "RISK_SCORING_MODEL")
SF_STAGE = os.getenv("SF_STAGE", "DATA")
FQN_STAGE = f"@{SF_DB}.{SF_SCHEMA}.{SF_STAGE}"
FQN_TABLE = f"{SF_DB}.{SF_SCHEMA}"

DATA_SIZES = {
    "small":  {"v1_stage": f"{FQN_STAGE}/synthetic_fraud_data_small.csv",
               "v2_table": f"{FQN_TABLE}.RAW_TRANSACTIONS_SMALL",
               "approx_rows": "~3,000"},
    "500k":   {"v1_stage": f"{FQN_STAGE}/synthetic_fraud_data_500k.csv",
               "v2_table": f"{FQN_TABLE}.RAW_TRANSACTIONS_500K",
               "approx_rows": "~500,000"},
    "1mio":   {"v1_stage": f"{FQN_STAGE}/synthetic_fraud_data_1mio.csv",
               "v2_table": f"{FQN_TABLE}.RAW_TRANSACTIONS_1MIO",
               "approx_rows": "~1,000,000"},
    "medium": {"v1_stage": f"{FQN_STAGE}/synthetic_fraud_data_medium.csv",
               "v2_table": f"{FQN_TABLE}.RAW_TRANSACTIONS_MEDIUM",
               "approx_rows": "~2,500,000"},
    "big":    {"v1_stage": f"{FQN_STAGE}/synthetic_fraud_data.csv",
               "v2_table": f"{FQN_TABLE}.RAW_TRANSACTIONS",
               "approx_rows": "~7,500,000"},
}

TELEMETRY_RE = re.compile(r"^\s+(\w+)\s*:\s+([\d.]+)s\s")
DURATION_RE = re.compile(r"^\s+Total duration\s*:\s+([\d.]+)s$")
ROWS_RE = re.compile(r"^\s+Total rows\s*:\s+([\d,]+)$")


def parse_telemetry(output: str) -> dict:
    result = {"steps": {}, "total_duration": None, "total_rows": None}
    for line in output.splitlines():
        m = DURATION_RE.match(line)
        if m:
            result["total_duration"] = float(m.group(1))
            continue
        m = ROWS_RE.match(line)
        if m:
            result["total_rows"] = int(m.group(1).replace(",", ""))
            continue
        m = TELEMETRY_RE.match(line)
        if m:
            result["steps"][m.group(1)] = float(m.group(2))
    return result


def run_version(label: str, script: str, conn_name: str, data_size: str, info: dict) -> dict:
    env = os.environ.copy()
    env["SNOWFLAKE_CONNECTION_NAME"] = conn_name
    env["DATA_SIZE"] = data_size

    print(f"\n{'='*70}")
    print(f"  RUNNING: {label}")
    print(f"{'='*70}")
    print(f"  Script          : {os.path.basename(script)}")
    print(f"  Connection      : {conn_name}")
    print(f"  Data size       : {data_size} ({info['approx_rows']} rows)")
    if "V1" in label:
        print(f"  Data source     : {info['v1_stage']}")
    elif "V0" in label:
        print(f"  Data source     : local CSV")
    else:
        print(f"  Data source     : {info['v2_table']}")
    print(f"  Python          : {PYTHON}")
    print(f"{'='*70}")
    print(f"  Launching subprocess ... ", end="", flush=True)

    t0 = time.monotonic()
    proc = subprocess.run(
        [PYTHON, script],
        env=env,
        capture_output=True,
        text=True,
        cwd=SCRIPT_DIR,
    )
    wall_time = time.monotonic() - t0
    print(f"done ({wall_time:.1f}s wall time)")

    print(f"\n  --- {label} stdout ---")
    print(proc.stdout)

    if proc.stderr:
        err_lines = [l for l in proc.stderr.splitlines()
                     if "WARN" not in l and "WARNING" not in l and "INFO" not in l]
        if err_lines:
            print(f"  --- {label} stderr (filtered, last 20 lines) ---")
            print("\n".join(err_lines[-20:]))

    if proc.returncode != 0:
        print(f"\n  *** {label} FAILED (exit code {proc.returncode}) ***")
        if proc.stderr:
            print(f"  --- Full stderr ---")
            print(proc.stderr[-2000:])
        return {"label": label, "success": False, "wall_time": wall_time}

    print(f"  --- {label} completed successfully ---")

    telemetry = parse_telemetry(proc.stdout)
    telemetry["label"] = label
    telemetry["success"] = True
    telemetry["wall_time"] = wall_time
    return telemetry


def print_comparison(results: list[dict], data_size: str, info: dict):
    print("\n\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    title = f"COMPARISON: V0 vs V1 vs V2 vs V3  [{data_size.upper()}]"
    print(f"*  {title:<75}*")
    print("*" + " " * 78 + "*")
    print("*" * 80)

    failed = [r for r in results if not r.get("success")]
    if failed:
        print("\n  *** One or more pipelines FAILED — comparison incomplete ***")
        for r in failed:
            print(f"  {r['label']:<25}: FAILED")
        succeeded = [r for r in results if r.get("success")]
        if len(succeeded) < 2:
            return
        results = succeeded

    print(f"\n  Data size: {data_size} ({info['approx_rows']} rows)")
    for r in results:
        rows = r.get('total_rows', 'N/A')
        rows_fmt = f"{rows:,}" if isinstance(rows, int) else rows
        print(f"  {r['label']:<25} rows: {rows_fmt}")
    print()

    labels = [r["label"] for r in results]
    short = [l.split()[0] for l in labels]
    hdr = f"  {'Step':<18}"
    for s in short:
        hdr += f" {s:>12}"
    print(hdr)
    print("  " + "-" * (18 + 13 * len(results)))

    all_steps = list(dict.fromkeys(s for r in results for s in r.get("steps", {}).keys()))
    for step in all_steps:
        line = f"  {step:<18}"
        for r in results:
            t = r.get("steps", {}).get(step)
            line += f" {t:>10.2f}s" if t is not None else f" {'N/A':>12}"
        print(line)

    print("  " + "-" * (18 + 13 * len(results)))
    durations = [r.get("total_duration") or r["wall_time"] for r in results]
    line = f"  {'TOTAL':<18}"
    for d in durations:
        line += f" {d:>10.2f}s"
    print(line)

    print()
    fastest_idx = durations.index(min(durations))
    slowest_idx = durations.index(max(durations))
    if durations[fastest_idx] > 0:
        ratio_val = durations[slowest_idx] / durations[fastest_idx]
        print(f"  >>> {labels[fastest_idx]} is {ratio_val:.1f}x FASTER than {labels[slowest_idx]} <<<")
        print(f"  >>> Time saved: {durations[slowest_idx] - durations[fastest_idx]:.2f}s <<<")

    print()
    print("  " + "=" * 70)
    print("  KEY TAKEAWAY: Only 2 secions of code differ between all versions!")
    print()
    print("  Line 1: SparkSession creation")
    print("    V0: spark.read.csv('local_file.csv')")
    print("    V1: SparkSession.builder.master('local[*]').getOrCreate()")
    print("    V2/V3: snowpark_connect.init_spark_session()")
    print()
    print("  Line 2: Data loading")
    print("    V0: spark.read.csv (local file)")
    print("    V1: snowflake.connector -> fetchall -> createDataFrame")
    print("    V2/V3: spark.read.table('TABLE_NAME')")
    print("  " + "=" * 70)
    print()


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    parser = argparse.ArgumentParser(description="Compare Classic PySpark vs Snowpark Connect")
    parser.add_argument("--big", action="store_true", help="Use big dataset (~7.5M rows)")
    parser.add_argument("--medium", action="store_true", help="Use medium dataset (~2.5M rows)")
    parser.add_argument("--500k", dest="fivehundredk", action="store_true", help="Use 500K row dataset")
    parser.add_argument("--1mio", dest="onemio", action="store_true", help="Use 1M row dataset")
    args = parser.parse_args()
    data_size = "big" if args.big else "medium" if args.medium else "1mio" if args.onemio else "500k" if args.fivehundredk else "small"
    info = DATA_SIZES[data_size]

    log_file = os.path.join(SCRIPT_DIR, "comparison_output.log")
    log_fh = open(log_file, "w")
    sys.stdout = Tee(sys.__stdout__, log_fh)

    ts = datetime.now(timezone.utc)

    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*  PySpark Migration Showcase: Classic PySpark -> Snowpark Connect  *")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()
    print(f"  Timestamp       : {ts:%Y-%m-%d %H:%M:%S %Z}")
    print(f"  Data size       : {data_size} ({info['approx_rows']} rows)")
    print(f"  V0 data source  : local CSV")
    print(f"  V1 data source  : {info['v1_stage']}")
    print(f"  V2 data source  : {info['v2_table']}")
    print(f"  V3 data source  : {info['v2_table']}")
    print(f"  V0 connection   : {V0_CONN}")
    print(f"  V1 connection   : {V1_CONN}")
    print(f"  V2 connection   : {V2_CONN}")
    print(f"  V3 connection   : {V3_CONN}")
    print(f"  Python          : {sys.version}")
    print(f"  Platform        : {platform.platform()}")
    print(f"  Working dir     : {SCRIPT_DIR}")
    print(f"  Log file        : {log_file}")
    print()
    print("  This showcase demonstrates that migrating from classic PySpark")
    print("  to Snowpark Connect requires only 2 sections of code changes.")
    print("  The entire pipeline logic remains 100% identical.")
    print()

    total_t0 = time.monotonic()

    v0 = run_version("V0 Local CSV PySpark", V0_SCRIPT, V0_CONN, data_size, info)
    v1 = run_version("V1 Classic PySpark", V1_SCRIPT, V1_CONN, data_size, info)
    v2 = run_version("V2 Snowpark Connect", V2_SCRIPT, V2_CONN, data_size, info)
    v3 = run_version("V3 Snowpark Connect", V3_SCRIPT, V3_CONN, data_size, info)

    total_wall = time.monotonic() - total_t0

    print_comparison([v0, v1, v2, v3], data_size, info)

    results = {
        "timestamp": ts.isoformat(),
        "data_size": data_size,
        "total_wall_time": round(total_wall, 2),
        "v0": v0,
        "v1": v1,
        "v2": v2,
        "v3": v3,
    }
    out_file = os.path.join(SCRIPT_DIR, "comparison_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Results JSON    : {out_file}")
    print(f"  Full log        : {log_file}")
    print(f"  Total wall time : {total_wall:.2f}s (all pipelines)")
    print()

    log_fh.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
