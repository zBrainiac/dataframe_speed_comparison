# PySpark to Snowpark Connect Migration Showcase

## Why

Migrating from classic PySpark to **Snowpark Connect** is simpler than most teams expect.
This project proves it: as a showcase an entire fraud-risk-scoring pipeline runs on four different execution
backends, and **only two functions change** between versions — `get_spark()` and `load_data()`.
Everything else — feature engineering, splitting, aggregations, write-back — stays 100% identical.

The goal is to give data engineers a concrete, runnable comparison so they can evaluate
Snowpark Connect with confidence.

## What

A fraud-risk-scoring pipeline implemented in **seven versions**:

| Script | Engine | Data Source | Description |
|--------|--------|-------------|-------------|
| **local_duckdb** | DuckDB SQL | Local CSV files in `./data/` | No JVM — embedded SQL analytics engine |
| **local_polars** | Polars DataFrame | Local CSV files in `./data/` | No JVM — Rust-based columnar engine |
| **local_pandas** | Pure Pandas + NumPy | Local CSV files in `./data/` | No JVM — pure Python baseline |
| **local_pySpark** | Classic PySpark (`local[2]`) | Local CSV files in `./data/` | Baseline — no Snowflake dependency for data loading |
| **remote_pySpark** | Classic PySpark (`local[2]`) | Snowflake stage via `snowflake.connector` | Typical pattern: fetch from stage, build DataFrame locally |
| **remote_SnowparkConnect** | Snowpark Connect | Snowflake table via `spark.read.table()` | Drop-in replacement — compute runs on Snowflake |
| **remote_SnowparkConnect_Optimized** | Snowpark Connect (optimized) | Snowflake table via `spark.read.table()` | remote_SnowparkConnect + performance tuning (cache, projection, deferred counts) |

### System Architecture

```
                        +---------------------------------------------+
                        |              SNOWFLAKE CLOUD                 |
                        |                                              |
                        |  +--------+   +---------------------------+  |
                        |  | Stage  |   | Warehouse (compute)       |  |
                        |  | @DATA/ |   |                           |  |
                        |  | *.csv  |   |  spark.read.table()       |  |
                        |  +---+----+   |  feature engineering      |  |
                        |      |        |  split / aggregations     |  |
                        |      |        |  write-back               |  |
                        |      |        +---------------------------+  |
                        |      |              ^             ^          |
                        +------|--------------|-------------|----------+
                               |              |             |
            +------------------+---------+    |             |
            |   remote_pySpark path      |    |             |
            |  snowflake.connector       |    |             |
            |  fetchall -> local memory  |    |             |
            +------------------+---------+    |             |
                               |              |             |
  +----------------------------+--------------+-------------+--------+
  |                    LOCAL MACHINE (Apple M1 Pro, 16 GB)           |
  |                                                                  |
  |  +-------+    +-----------+     +-----------+   +-------------+  |
  |  | local |    | remote    |     | remote    |   | remote SC   |  |
  |  | Py    |    | PySpark + |     | Snowpark  |   | Optimized   |  |
  |  | Spark |    | SF stage  |     | Connect   |   |             |  |
  |  |       |    |           |     |           |   |             |  |
  |  +---+---+    +-----+-----+     +-----+-----+   +------+------+  |
  |      |              |                |                 |         |
  |      v              v                v                 v         |
  |  ./data/*.csv   collectToPython   Snowflake          Snowflake   |
  |  (read local)   (read local)      compute            compute     |
  |                                   (pushdown)         + cache     |
  |                                                      + project   |
  +------------------------------------------------------------------+

  Data flow:
    local_*:                    local CSV  -->  local engine  -->  local output
    remote_pySpark:             SF stage   -->  local memory  -->  local Spark
    remote_SnowparkConnect*:    SF table   -->  Snowflake compute (via Snowpark Connect)  -->  SF table
```
## Comparison Results

![Benchmark Chart](benchmark_chart.png)

### Pipeline Steps (identical across all versions)

1. **Session init** — create a SparkSession
2. **Data load** — read raw transactions
3. **Feature engineering** — parse velocity JSON, drop PII, cast types
4. **Train/test split** — 75/25, seed=42
5. **Aggregation statistics** — fraud rate by category, country, channel/device, velocity
6. **Write-back** — save engineered features

### remote_SnowparkConnect_Optimized — Optimizations

1. `.cache()` after feature engineering to avoid recomputation
2. Deferred `.count()` — piggybacks on cache materialization
3. Estimated train/test counts instead of expensive `.count()` calls
4. Column projection (`df.select(...)`) before aggregations (10 columns instead of 20)
5. `.unpersist()` before write to free memory

### Key Takeaway

Only **2 sections of code** differ between all versions:

```python
# 1. Session creation
# local_*:                   SparkSession.builder.master("local[2]").getOrCreate()
# remote_SnowparkConnect*:   snowpark_connect.init_spark_session()

# 2. Data loading
# local_*:                   spark.read.csv("local_file.csv")
# remote_pySpark:            snowflake.connector -> fetchall -> createDataFrame
# remote_SnowparkConnect*:   spark.read.table("TABLE_NAME")
```

The entire pipeline logic — feature engineering, splitting, aggregations, write-back —
remains **100% identical** across all seven versions.

## How

### Prerequisites

**Required:**

- Python 3.12+
- PySpark 3.5+
- `snowpark-connect[jdk]` (for remote_SnowparkConnect*)
- A Snowflake account with a named connection configured (for remote_*)
- JDK 17+ (handled automatically via `--add-opens` flags)

**Test environment (not required — listed for reproducibility):**

- Apple M1 Pro, 16 GB RAM
- Snowflake XS Warehouse

### Environment Variables

All Snowflake-specific references are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SNOWFLAKE_CONNECTION_NAME` | `default` | Named connection from `~/.snowflake/config.toml` |
| `SF_DATABASE` | `MY_DATABASE` | Snowflake database containing the schema |
| `SF_SCHEMA` | `RISK_SCORING_MODEL` | Schema with raw transaction tables |
| `SF_STAGE` | `DATA` | Internal stage name where CSVs are uploaded |
| `DATA_SIZE` | `small` | Dataset size: `small`, `500k`, `1mio`, `medium`, `big` |

### Project Structure

```
pyspark/
├── README.md               # This file
├── config.toml.example     # Sample Snowflake CLI connection config
├── local_duckdb.py                      # DuckDB SQL, local CSV
├── local_polars.py                      # Polars DataFrame, local CSV
├── local_pandas.py                      # Pure Pandas, local CSV
├── local_pySpark.py                     # Classic PySpark, local CSV
├── remote_pySpark.py                    # Classic PySpark, Snowflake stage
├── remote_SnowparkConnect.py            # Snowpark Connect
├── remote_SnowparkConnect_Optimized.py  # Snowpark Connect (optimized)
├── run_comparison.py        # Orchestrator: runs all versions, prints comparison
├── setup.sql               # DDL to create Snowflake tables from staged CSVs
├── output/
│   ├── comparison_results.json          # Latest comparison results (JSON)
│   ├── comparison_output.log            # Full comparison output log
│   ├── benchmark_chart_<timestamp>.png  # Benchmark chart
│   └── engineered_features_*.csv        # Pipeline output files
└── data/
    ├── synthetic_fraud_data_small.csv    # ~3K rows
    ├── synthetic_fraud_data_500k.csv     # ~500K rows
    ├── synthetic_fraud_data_1mio.csv     # ~1M rows
    ├── synthetic_fraud_data_medium.csv   # ~2.5M rows
    └── synthetic_fraud_data.csv          # ~7.5M rows
```

### Input Data

Source: [Kaggle — Risk Scoring Model](https://www.kaggle.com/code/lyneshiacorrea/risk-scoring-model)

Synthetic fraud transaction data with 24 columns including transaction details,
merchant info, geographic data, device/channel info, and a binary `is_fraud` label.

### Data Modes

| Mode | Rows | CSV Size | Description |
|------|------|----------|-------------|
| `small` | ~3,000 | ~1.1 MB | Quick smoke test |
| `500k` | ~500,000 | ~188 MB | Medium benchmark |
| `1mio` | ~1,000,000 | ~375 MB | Larger benchmark |
| `medium` | ~2,500,000 | ~937 MB | Stress test |
| `big` | ~7,500,000 | ~2.7 GB | Full dataset |

CSV files live in `./data/`. Snowflake tables are pre-loaded via `setup.sql`.

### Snowflake CLI Connection

Copy the sample config and fill in your credentials:

```bash
cp config.toml.example ~/.snowflake/config.toml
chmod 0600 ~/.snowflake/config.toml
# Edit ~/.snowflake/config.toml with your account, user, and password
```

See `config.toml.example` for the expected format.

The config file defines **named connections** (e.g. `default`). When running remote_* scripts,
pass the connection name via the `SNOWFLAKE_CONNECTION_NAME` environment variable so the scripts
know which Snowflake account to target:

```bash
SNOWFLAKE_CONNECTION_NAME=default python remote_SnowparkConnect.py --500k
```

### Setup (Snowflake objects)

```bash
snow sql -f setup.sql -c default
```

### Running a Single Version

```bash
# local variants — no JVM needed
DATA_SIZE=500k .venv/bin/python local_duckdb.py
DATA_SIZE=500k .venv/bin/python local_polars.py
DATA_SIZE=500k .venv/bin/python local_pandas.py

# local PySpark
DATA_SIZE=500k .venv/bin/python local_pySpark.py

# remote — Snowflake stage
SNOWFLAKE_CONNECTION_NAME=NEWS_CREW_SVC SF_DATABASE=MD_TEST DATA_SIZE=500k .venv/bin/python remote_pySpark.py

# remote — Snowpark Connect
SNOWFLAKE_CONNECTION_NAME=NEWS_CREW_SVC SF_DATABASE=MD_TEST DATA_SIZE=500k .venv/bin/python remote_SnowparkConnect.py

# remote — Snowpark Connect (optimized)
SNOWFLAKE_CONNECTION_NAME=NEWS_CREW_SVC SF_DATABASE=MD_TEST DATA_SIZE=500k .venv/bin/python remote_SnowparkConnect_Optimized.py
```

### Running the Full Comparison

Set the environment variables for your Snowflake connection first:

```bash
export SNOWFLAKE_CONNECTION_NAME=NEWS_CREW_SVC
export SF_DATABASE=MD_TEST
.venv/bin/python run_comparison.py
```

Available data-size flags:

```bash
.venv/bin/python run_comparison.py              # small (~3,000 rows)
.venv/bin/python run_comparison.py --500k       # 500K rows
.venv/bin/python run_comparison.py --1mio       # 1M rows
.venv/bin/python run_comparison.py --medium     # 2.5M rows
.venv/bin/python run_comparison.py --big        # 7.5M rows
```

Results are written to `output/comparison_results.json` and `output/comparison_output.log`.

### Small (~3,000 rows)

| Step | local_pySpark | remote_pySpark | remote_SnowparkConnect | remote_SnowparkConnect_Optimized |
|------|---------------:|---------------:|---------------:|---------------:|
| session_init | 11.66s | 28.85s | 12.84s | 12.07s |
| data_load | 5.29s | 13.84s | 0.50s | 0.14s |
| feature_eng | 1.10s | 0.60s | 0.69s | 2.81s |
| split | 2.61s | 3.34s | 5.19s | 0.00s |
| aggregations | 5.78s | 4.16s | 2.74s | 2.96s |
| write_back | 1.19s | 0.38s | 1.44s | 2.51s |
| **TOTAL** | **37.44s** | **53.05s** | **23.51s** | **20.64s** |

> remote_SnowparkConnect_Optimized is **2.6x faster** than remote_pySpark and **1.1x faster** than remote_SnowparkConnect.

### 500K rows

| Step | local_pySpark | remote_pySpark | remote_SnowparkConnect | remote_SnowparkConnect_Optimized |
|------|---------------:|---------------:|---------------:|---------------:|
| session_init | 7.27s | 7.10s | 10.99s | 10.21s |
| data_load | 2.03s | 21.80s | 0.36s | 0.10s |
| feature_eng | 0.16s | 0.17s | 0.78s | 5.28s |
| split | 8.40s | 9.20s | 13.68s | 0.00s |
| aggregations | 5.28s | 15.37s | 1.33s | 2.48s |
| write_back | 8.79s | crashed | 4.48s | 3.98s |
| **TOTAL** | **32.44s** | **crashed** | **31.72s** | **22.50s** |

> remote_SnowparkConnect_Optimized is **1.4x faster** than remote_SnowparkConnect and local_pySpark — saving ~10 seconds.
> remote_pySpark completes steps 1–5 (53.64s) but crashes at write-back (`collectToPython` OOM via Py4J/Netty).

### 1M rows

| Step | local_pySpark | remote_pySpark | remote_SnowparkConnect | remote_SnowparkConnect_Optimized |
|------|---------------:|---------------:|---------------:|---------------:|
| session_init | 7.17s | 7.05s | 10.32s | 8.08s |
| data_load | 2.58s | 37.10s | 0.44s | 0.15s |
| feature_eng | 0.15s | 0.18s | 0.43s | 3.76s |
| split | 20.10s | 19.75s | 14.34s | 0.00s |
| aggregations | 11.58s | 12.00s | 1.41s | 2.57s |
| write_back | 16.62s | crashed | 8.72s | 2.56s |
| **TOTAL** | **58.88s** | **crashed** | **35.75s** | **17.30s** |

> remote_SnowparkConnect_Optimized is **3.4x faster** than local_pySpark — saving 41.6 seconds.
> remote_pySpark completes steps 1–5 (76.08s) but crashes at write-back (`collectToPython` OOM via Py4J/Netty).

### 2.5M rows

| Step | local_pySpark | remote_pySpark | remote_SnowparkConnect | remote_SnowparkConnect_Optimized |
|------|---------------:|---------------:|---------------:|---------------:|
| session_init | 7.27s | 8.06s | 10.73s | 8.77s |
| data_load | 3.31s | 80.91s | 0.94s | 0.11s |
| feature_eng | 0.15s | 0.27s | 0.56s | 7.16s |
| split | 43.80s | 48.61s | 16.51s | 0.00s |
| aggregations | 23.27s | 61.08s | 4.43s | 1.96s |
| write_back | 44.62s | crashed | 27.46s | 3.32s |
| **TOTAL** | **123.15s** | **crashed** | **60.73s** | **21.47s** |

> remote_SnowparkConnect_Optimized is **5.7x faster** than local_pySpark and **2.8x faster** than remote_SnowparkConnect — saving over 100 seconds.
> remote_pySpark completes steps 1–5 (198.93s) but crashes at write-back (`collectToPython` OOM via Py4J/Netty).

### 7.5M rows

| Step | local_pySpark | remote_pySpark | remote_SnowparkConnect | remote_SnowparkConnect_Optimized |
|------|---------------:|---------------:|---------------:|---------------:|
| session_init | 7.84s | 7.61s | 10.74s | 10.65s |
| data_load | 6.17s | 334.92s | 0.89s | 0.17s |
| feature_eng | 0.28s | 0.32s | 0.47s | 11.18s |
| split | 129.00s | 246.76s | 27.04s | 0.00s |
| aggregations | 65.93s | 256.63s | 4.92s | 2.38s |
| write_back | 119.87s | crashed | 64.29s | 4.46s |
| **TOTAL** | **329.70s** | **crashed** | **108.44s** | **28.98s** |

> remote_SnowparkConnect_Optimized is **11.4x faster** than local_pySpark and **3.7x faster** than remote_SnowparkConnect — saving over 300 seconds.
> remote_pySpark completes steps 1–5 (846.24s) but crashes at write-back (`collectToPython` OOM / Netty connection reset).

## Limitations and Notes
- **Single-machine benchmarks:** All results were collected on a single Apple M1 Pro (16 GB RAM). Production performance on larger Snowflake warehouses will differ.


