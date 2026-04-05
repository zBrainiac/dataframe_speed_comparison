"""
Run local_duckdb, local_polars, local_pandas, local_pySpark,
remote_pySpark, remote_SnowparkConnect, and remote_SnowparkConnect_Optimized
side-by-side and compare.

Usage:
    python run_comparison.py              # small dataset (~3K rows)
    python run_comparison.py --big        # big dataset (~7.5M rows)
    python run_comparison.py --all        # ALL data sizes, generates chart
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
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

V0D_SCRIPT = os.path.join(SCRIPT_DIR, "local_duckdb.py")
V0L_SCRIPT = os.path.join(SCRIPT_DIR, "local_polars.py")
V0P_SCRIPT = os.path.join(SCRIPT_DIR, "local_pandas.py")
V0_SCRIPT = os.path.join(SCRIPT_DIR, "local_pySpark.py")
V1_SCRIPT = os.path.join(SCRIPT_DIR, "remote_pySpark.py")
V2_SCRIPT = os.path.join(SCRIPT_DIR, "remote_SnowparkConnect.py")
V3_SCRIPT = os.path.join(SCRIPT_DIR, "remote_SnowparkConnect_Optimized.py")

CONN_NAME = os.getenv("SNOWFLAKE_CONNECTION_NAME", "default")

V0D_CONN = CONN_NAME
V0L_CONN = CONN_NAME
V0P_CONN = CONN_NAME
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
    if "remote_pySpark" == label.split()[0]:
        print(f"  Data source     : {info['v1_stage']}")
    elif label.startswith("local_"):
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
    title = f"COMPARISON: local_duckdb vs local_polars vs local_pandas vs local_pySpark vs remote_pySpark vs remote_SnowparkConnect vs remote_SnowparkConnect_Optimized  [{data_size.upper()}]"
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
    print("    local_*: spark.read.csv('local_file.csv')")
    print("    remote_pySpark: SparkSession.builder.master('local[*]').getOrCreate()")
    print("    remote_SnowparkConnect*: snowpark_connect.init_spark_session()")
    print()
    print("  Line 2: Data loading")
    print("    local_*: spark.read.csv (local file)")
    print("    remote_pySpark: snowflake.connector -> fetchall -> createDataFrame")
    print("    remote_SnowparkConnect*: spark.read.table('TABLE_NAME')")
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


VERSION_DEFS = [
    ("local_duckdb",                      "DuckDB SQL (local CSV)",      V0D_SCRIPT, "V0D_CONN"),
    ("local_polars",                      "Polars DataFrame (local CSV)", V0L_SCRIPT, "V0L_CONN"),
    ("local_pandas",                      "Pure Pandas",                 V0P_SCRIPT, "V0P_CONN"),
    ("local_pySpark",                     "Classic PySpark (local CSV)",  V0_SCRIPT, "V0_CONN"),
    ("remote_pySpark",                    "Classic PySpark (SF stage)",   V1_SCRIPT, "V1_CONN"),
    ("remote_SnowparkConnect",            "Snowpark Connect",             V2_SCRIPT, "V2_CONN"),
    ("remote_SnowparkConnect_Optimized",  "Snowpark Connect (optimized)", V3_SCRIPT, "V3_CONN"),
]

SIZE_LABELS = {
    "small": "3K",
    "500k":  "500K",
    "1mio":  "1M",
    "medium":"2.5M",
    "big":   "7.5M",
}

VERSION_COLORS = {
    "local_duckdb":                      "#FF6F00",
    "local_polars":                      "#E91E63",
    "local_pandas":                      "#9467bd",
    "local_pySpark":                     "#4285F4",
    "remote_pySpark":                    "#EA4335",
    "remote_SnowparkConnect":            "#34A853",
    "remote_SnowparkConnect_Optimized":  "#FBBC05",
}


def generate_chart(all_results: dict, chart_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    size_order = [s for s in ["small", "500k", "1mio", "medium", "big"] if s in all_results]
    version_keys = [vd[0] for vd in VERSION_DEFS]

    x_labels = [SIZE_LABELS.get(s, s) for s in size_order]
    x = np.arange(len(size_order))
    n_versions = len(version_keys)
    bar_width = 0.11
    offsets = np.arange(n_versions) - (n_versions - 1) / 2

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, vkey in enumerate(version_keys):
        vals = []
        for s in size_order:
            size_data = all_results[s]
            vresult = size_data.get(vkey.lower().replace("-", "_"), size_data.get(vkey.lower()))
            if vresult and vresult.get("success"):
                vals.append(vresult.get("total_duration") or vresult.get("wall_time", 0))
            else:
                vals.append(0)

        color = VERSION_COLORS.get(vkey, "#999999")
        bars = ax.bar(x + offsets[i] * bar_width, vals, bar_width,
                      label=f"{vkey} {VERSION_DEFS[i][1]}", color=color)

        for j, (bar, val) in enumerate(zip(bars, vals)):
            s = size_order[j]
            vresult = all_results[s].get(vkey.lower().replace("-", "_"), all_results[s].get(vkey.lower()))
            if vresult and not vresult.get("success"):
                ax.text(bar.get_x() + bar.get_width() / 2, 2, "OOM\ncrash",
                        ha="center", va="bottom", fontsize=7, fontweight="bold", color="red")
            elif val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.0f}s",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Dataset Size (rows)", fontsize=12)
    ax.set_ylabel("Total Time (seconds)", fontsize=12)
    ax.set_title("PySpark vs Snowpark Connect \u2014 Total Pipeline Time", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"\n  Chart saved to: {chart_path}")


def run_single_size(data_size: str, ts: datetime) -> dict:
    info = DATA_SIZES[data_size]

    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*  PySpark Migration Showcase: Classic PySpark -> Snowpark Connect  *")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()
    print(f"  Timestamp       : {ts:%Y-%m-%d %H:%M:%S %Z}")
    print(f"  Data size       : {data_size} ({info['approx_rows']} rows)")
    print(f"  local_duckdb source : local CSV (DuckDB SQL)")
    print(f"  local_polars source : local CSV (Polars DataFrame)")
    print(f"  local_pandas source : local CSV (pure pandas)")
    print(f"  local_pySpark source: local CSV")
    print(f"  remote_pySpark src  : {info['v1_stage']}")
    print(f"  remote_SC src       : {info['v2_table']}")
    print(f"  remote_SC_Opt src   : {info['v2_table']}")
    print(f"  local_duckdb conn   : {V0D_CONN}")
    print(f"  local_polars conn   : {V0L_CONN}")
    print(f"  local_pandas conn   : {V0P_CONN}")
    print(f"  local_pySpark conn  : {V0_CONN}")
    print(f"  remote_pySpark conn : {V1_CONN}")
    print(f"  remote_SC conn      : {V2_CONN}")
    print(f"  remote_SC_Opt conn  : {V3_CONN}")
    print(f"  Python          : {sys.version}")
    print(f"  Platform        : {platform.platform()}")
    print(f"  Working dir     : {SCRIPT_DIR}")
    print()

    total_t0 = time.monotonic()

    v0d = run_version("local_duckdb", V0D_SCRIPT, V0D_CONN, data_size, info)
    v0l = run_version("local_polars", V0L_SCRIPT, V0L_CONN, data_size, info)
    v0p = run_version("local_pandas", V0P_SCRIPT, V0P_CONN, data_size, info)
    v0 = run_version("local_pySpark", V0_SCRIPT, V0_CONN, data_size, info)
    v1 = run_version("remote_pySpark", V1_SCRIPT, V1_CONN, data_size, info)
    v2 = run_version("remote_SnowparkConnect", V2_SCRIPT, V2_CONN, data_size, info)
    v3 = run_version("remote_SnowparkConnect_Optimized", V3_SCRIPT, V3_CONN, data_size, info)

    total_wall = time.monotonic() - total_t0

    print_comparison([v0d, v0l, v0p, v0, v1, v2, v3], data_size, info)

    return {
        "timestamp": ts.isoformat(),
        "data_size": data_size,
        "total_wall_time": round(total_wall, 2),
        "local_duckdb": v0d,
        "local_polars": v0l,
        "local_pandas": v0p,
        "local_pyspark": v0,
        "remote_pyspark": v1,
        "remote_snowparkconnect": v2,
        "remote_snowparkconnect_optimized": v3,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Classic PySpark vs Snowpark Connect")
    parser.add_argument("--big", action="store_true", help="Use big dataset (~7.5M rows)")
    parser.add_argument("--medium", action="store_true", help="Use medium dataset (~2.5M rows)")
    parser.add_argument("--500k", dest="fivehundredk", action="store_true", help="Use 500K row dataset")
    parser.add_argument("--1mio", dest="onemio", action="store_true", help="Use 1M row dataset")
    parser.add_argument("--all", dest="run_all", action="store_true", help="Run ALL data sizes and generate chart")
    args = parser.parse_args()

    log_file = os.path.join(OUTPUT_DIR, "comparison_output.log")
    log_fh = open(log_file, "w")
    sys.stdout = Tee(sys.__stdout__, log_fh)

    ts = datetime.now(timezone.utc)

    if args.run_all:
        all_results = {}
        for data_size in ["small", "500k", "1mio", "medium", "big"]:
            print(f"\n{'#'*70}")
            print(f"#  DATA SIZE: {data_size.upper()} ({DATA_SIZES[data_size]['approx_rows']} rows)")
            print(f"{'#'*70}\n")
            result = run_single_size(data_size, ts)
            all_results[data_size] = result

        out_file = os.path.join(OUTPUT_DIR, "comparison_results_all.json")
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)

        chart_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        chart_path = os.path.join(OUTPUT_DIR, f"benchmark_chart_{chart_ts}.png")
        generate_chart(all_results, chart_path)

        print(f"\n  All-sizes JSON  : {out_file}")
        print(f"  Chart           : {chart_path}")
        print(f"  Full log        : {log_file}")
        print()
    else:
        data_size = "big" if args.big else "medium" if args.medium else "1mio" if args.onemio else "500k" if args.fivehundredk else "small"

        print(f"  Log file        : {log_file}")
        print()
        print("  This showcase demonstrates that migrating from classic PySpark")
        print("  to Snowpark Connect requires only 2 sections of code changes.")
        print("  The entire pipeline logic remains 100% identical.")
        print()

        result = run_single_size(data_size, ts)

        out_file = os.path.join(OUTPUT_DIR, "comparison_results.json")
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        chart_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        chart_path = os.path.join(OUTPUT_DIR, f"benchmark_chart_{chart_ts}.png")
        generate_chart({data_size: result}, chart_path)

        print(f"  Results JSON    : {out_file}")
        print(f"  Chart           : {chart_path}")
        print(f"  Full log        : {log_file}")
        print(f"  Total wall time : {result['total_wall_time']:.2f}s (all pipelines)")
        print()

    log_fh.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
