"""Exploratory data analysis as plain SQL straight against the SQLite db.

Six profiling queries that answer the questions you'd put on a fraud-detection writeup:
  * how (im)balanced are the classes,
  * how does the fraud rate move with time of day / across the two-day window,
  * how do transaction amounts differ between fraud and legit.

    python -m fraud_platform.data.eda          # prints every query's result

Time is seconds since the first transaction, so hour-of-day = (Time / 3600) % 24 and the dataset
spans two days (Time < 86400 = day 1). Class is 0/1, so AVG(Class) is just the fraud rate.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from fraud_platform.config import CREDITCARD_DB, DB_TABLE

EDA_QUERIES: dict[str, str] = {
    # class balance — how rare is fraud
    "class_balance": """
        SELECT Class,
               COUNT(*) AS n,
               100.0 * COUNT(*) / (SELECT COUNT(*) FROM {table}) AS pct
        FROM {table}
        GROUP BY Class
        ORDER BY Class
    """,
    # fraud rate by hour of day
    "fraud_rate_by_hour": """
        SELECT CAST(Time / 3600 AS INTEGER) % 24 AS hour,
               COUNT(*) AS n,
               SUM(Class) AS n_fraud,
               AVG(Class) AS fraud_rate
        FROM {table}
        GROUP BY hour
        ORDER BY hour
    """,
    # fraud rate across the two-day window
    "fraud_rate_by_day": """
        SELECT CASE WHEN Time < 86400 THEN 1 ELSE 2 END AS day,
               COUNT(*) AS n,
               SUM(Class) AS n_fraud,
               AVG(Class) AS fraud_rate
        FROM {table}
        GROUP BY day
        ORDER BY day
    """,
    # amount summary stats, fraud vs legit
    "amount_stats_by_class": """
        SELECT Class,
               COUNT(*) AS n,
               MIN(Amount) AS min_amount,
               AVG(Amount) AS avg_amount,
               MAX(Amount) AS max_amount,
               SUM(Amount) AS total_amount
        FROM {table}
        GROUP BY Class
        ORDER BY Class
    """,
    # amount distribution by bucket, fraud vs legit
    "amount_distribution_by_class": """
        SELECT Class,
               CASE
                   WHEN Amount = 0 THEN '0'
                   WHEN Amount < 10 THEN '0-10'
                   WHEN Amount < 50 THEN '10-50'
                   WHEN Amount < 100 THEN '50-100'
                   WHEN Amount < 500 THEN '100-500'
                   ELSE '500+'
               END AS amount_bucket,
               COUNT(*) AS n
        FROM {table}
        GROUP BY Class, amount_bucket
        ORDER BY Class, MIN(Amount)
    """,
    # how much of each class is high-value (>= 200)
    "high_value_share_by_class": """
        SELECT Class,
               COUNT(*) AS n,
               SUM(CASE WHEN Amount >= 200 THEN 1 ELSE 0 END) AS n_high_value,
               AVG(CASE WHEN Amount >= 200 THEN 1.0 ELSE 0.0 END) AS high_value_share
        FROM {table}
        GROUP BY Class
        ORDER BY Class
    """,
}


def run_eda(db_path: str | Path = CREDITCARD_DB, table: str = DB_TABLE) -> dict[str, pd.DataFrame]:
    """Run every EDA query against the db and return name -> result dataframe."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite db not found at {db_path}. Build it with "
            "python -m fraud_platform.data.load_to_sqlite first."
        )
    conn = sqlite3.connect(db_path)
    try:
        return {name: pd.read_sql_query(sql.format(table=table), conn)
                for name, sql in EDA_QUERIES.items()}
    finally:
        conn.close()


def main() -> None:
    results = run_eda()
    for name, frame in results.items():
        print(f"\n=== {name} ===")
        print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
