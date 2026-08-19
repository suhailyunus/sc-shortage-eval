"""
One-time (or repeatable) loader: reads the M5 CSVs from data/raw/, reshapes
sales into long format using the same, already-tested reshape_sales_long
logic from src/preprocess.py, and pushes three tables into Snowflake's
RAW schema.

This is deliberately a Python step, not a dbt model: unpivoting 1,913
day columns (d_1..d_1913) into long format is awkward and fragile to do
in raw SQL (Snowflake's UNPIVOT needs an explicit column list), while
pandas already does this correctly and it's already covered by
tests/test_leakage.py's assumptions. dbt picks up from here and owns
everything downstream (joins, the stress target, feature engineering).

Usage:
    python scripts/load_raw_to_snowflake.py
    python scripts/load_raw_to_snowflake.py --max-items 100   # dev-scale load

Requires a filled-in .env (see .env.example) and the snowflake-connector-python
package with pandas support.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.preprocess import reshape_sales_long  # noqa: E402


def get_connection():
    """Open a Snowflake connection using credentials from .env."""
    import snowflake.connector

    load_dotenv(REPO_ROOT / ".env")

    required = [
        "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_ROLE", "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA",
    ]
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise SystemExit(
            f"Missing required .env values: {missing}. "
            f"Copy .env.example to .env and fill these in first."
        )

    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        role=os.environ["SNOWFLAKE_ROLE"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )


def load_table(conn, df: pd.DataFrame, table_name: str) -> None:
    """Overwrite a RAW-schema table with the given DataFrame."""
    from snowflake.connector.pandas_tools import write_pandas

    # Snowflake convention: unquoted identifiers are upper-cased. Match
    # that here so column names line up cleanly with dbt sources later,
    # rather than ending up with mixed-case columns that need quoting
    # everywhere downstream.
    df = df.copy()
    df.columns = [c.upper() for c in df.columns]

    success, num_chunks, num_rows, _ = write_pandas(
        conn,
        df,
        table_name=table_name,
        auto_create_table=True,
        overwrite=True,
    )
    status = "OK" if success else "FAILED"
    print(f"  [{status}] {table_name}: {num_rows:,} rows in {num_chunks} chunk(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default=str(REPO_ROOT / "data" / "raw"),
        help="Directory containing the M5 CSVs",
    )
    parser.add_argument(
        "--max-items", type=int, default=None,
        help="Optionally restrict to the first N items (dev-scale load, faster/cheaper)",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print("Reading CSVs...")
    sales = pd.read_csv(data_dir / "sales_train_validation.csv")
    calendar = pd.read_csv(data_dir / "calendar.csv")
    prices = pd.read_csv(data_dir / "sell_prices.csv")

    if args.max_items is not None:
        items = sales["item_id"].drop_duplicates().head(args.max_items)
        sales = sales[sales["item_id"].isin(items)]
        print(f"Restricted to {args.max_items} items -> {len(sales):,} series rows")

    print("Reshaping sales to long format (reusing src/preprocess.reshape_sales_long)...")
    sales_long = reshape_sales_long(sales)
    print(f"  {len(sales_long):,} long-format rows")

    print("\nConnecting to Snowflake...")
    conn = get_connection()

    try:
        print("\nLoading tables into RAW schema:")
        load_table(conn, sales_long, "SALES_LONG")
        load_table(conn, calendar, "CALENDAR")
        load_table(conn, prices, "SELL_PRICES")
    finally:
        conn.close()

    print("\nDone. Verify in Snowsight with: SELECT COUNT(*) FROM RAW.SALES_LONG;")


if __name__ == "__main__":
    main()
