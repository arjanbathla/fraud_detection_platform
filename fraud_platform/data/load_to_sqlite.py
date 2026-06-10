"""One-off import: load the Kaggle creditcard.csv into a SQLite database.

    python -m fraud_platform.data.load_to_sqlite

Download creditcard.csv from Kaggle ("Credit Card Fraud Detection") into data/, then run this
once. It reads the CSV in chunks (the file is ~144MB) and writes a single `transactions` table
that the loader and the EDA module query from. After this the platform never touches the CSV.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from fraud_platform.config import (
    CREDITCARD_CSV,
    CREDITCARD_DB,
    DB_TABLE,
    NUMERIC_FEATURES,
    TARGET,
)


def load_csv_to_sqlite(csv_path: str | Path = CREDITCARD_CSV,
                       db_path: str | Path = CREDITCARD_DB,
                       table: str = DB_TABLE,
                       chunksize: int = 50_000) -> int:
    """Read creditcard.csv and write it to `table` in a SQLite db. Returns the row count."""
    csv_path = Path(csv_path)
    db_path = Path(db_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"creditcard.csv not found at {csv_path}.\n"
            "Download the Kaggle 'Credit Card Fraud Detection' dataset and put creditcard.csv "
            "in the data/ folder first."
        )

    conn = sqlite3.connect(db_path)
    rows = 0
    try:
        first = True
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            if first:
                missing = [c for c in NUMERIC_FEATURES + [TARGET] if c not in chunk.columns]
                if missing:
                    raise ValueError(f"creditcard.csv is missing expected columns: {missing}")
            chunk.to_sql(table, conn, if_exists="replace" if first else "append", index=False)
            rows += len(chunk)
            first = False
        # an index on Class makes the EDA group-bys quick
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_class ON {table}({TARGET})")
        conn.commit()
    finally:
        conn.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Load creditcard.csv into SQLite")
    ap.add_argument("--csv", type=str, default=str(CREDITCARD_CSV))
    ap.add_argument("--db", type=str, default=str(CREDITCARD_DB))
    args = ap.parse_args()
    n = load_csv_to_sqlite(args.csv, args.db)
    print(f"loaded {n:,} rows into {args.db} (table '{DB_TABLE}')")


if __name__ == "__main__":
    main()
