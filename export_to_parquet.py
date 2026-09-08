"""
Exports traffic_snapshots from PostgreSQL to a partitioned Parquet dataset.

Run manually or as a daily cron job:
    python export_to_parquet.py

Output layout (partitioned by date, good for Spark later):
    data/parquet/date=2026-07-30/part.parquet
"""

import logging
import pandas as pd
from sqlalchemy import create_engine

from config.settings import POSTGRES_DSN, PARQUET_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_all():
    engine = create_engine(POSTGRES_DSN)
    query = """
        SELECT * FROM traffic_snapshots
        ORDER BY collected_at
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        logger.info("No rows to export yet.")
        return

    df["date"] = pd.to_datetime(df["collected_at"]).dt.date

    for date, group in df.groupby("date"):
        out_dir = f"{PARQUET_DIR}/date={date}"
        import os
        os.makedirs(out_dir, exist_ok=True)
        group.drop(columns=["date"]).to_parquet(f"{out_dir}/part.parquet", index=False)
        logger.info("Wrote %d rows for %s", len(group), date)


if __name__ == "__main__":
    export_all()
