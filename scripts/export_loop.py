"""
Runs export_to_parquet.export_all() on a schedule, inside its own container.
Controlled by EXPORT_INTERVAL_SECONDS (default: once every 24h).
"""

import logging
import os
import time

from export_to_parquet import export_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INTERVAL = int(os.environ.get("EXPORT_INTERVAL_SECONDS", str(24 * 60 * 60)))


def main():
    logger.info("Parquet exporter started, running every %ss", INTERVAL)
    while True:
        try:
            export_all()
        except Exception as e:
            logger.exception("Export failed: %s", e)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
