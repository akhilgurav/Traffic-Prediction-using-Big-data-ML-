"""
Exports the full traffic_snapshots table to a single CSV file.
Good for quickly eyeballing data in Excel or sharing a snapshot - the
Parquet export (export_to_parquet.py) is the one to use for actual
downstream analytics/Spark work.

Run manually whenever you want a fresh CSV:
    python export_to_csv.py

Or, from your host machine, via the collector's Docker image (no local
Python needed):
    docker compose run --rm collector python export_to_csv.py

Output:
    data/csv/traffic_snapshots.csv   (overwritten each run, full table)
"""

import logging
import os
import pandas as pd
from sqlalchemy import create_engine

from config.settings import POSTGRES_DSN, RAW_JSON_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Reuse the same base data dir as raw JSON / parquet ("./data" on host)
CSV_DIR = os.path.join(os.path.dirname(RAW_JSON_DIR.rstrip("/")), "csv")


def export_csv():
    engine = create_engine(POSTGRES_DSN)
    query = "SELECT * FROM traffic_snapshots ORDER BY collected_at"
    df = pd.read_sql(query, engine)

    if df.empty:
        logger.info("No rows yet - nothing to export.")
        return

    os.makedirs(CSV_DIR, exist_ok=True)
    out_path = os.path.join(CSV_DIR, "traffic_snapshots.csv")
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_path)


if __name__ == "__main__":
    export_csv()
