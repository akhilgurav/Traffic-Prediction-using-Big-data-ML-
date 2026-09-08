"""
Runs hdfs_sync.sync_all() on a schedule, inside its own container.
Controlled by HDFS_SYNC_INTERVAL_SECONDS (default: once every hour).
"""

import logging
import os
import time

from hdfs_sync import sync_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INTERVAL = int(os.environ.get("HDFS_SYNC_INTERVAL_SECONDS", str(60 * 60)))
RETRY_INTERVAL = int(os.environ.get("HDFS_SYNC_RETRY_SECONDS", "30"))


def main():
    logger.info("HDFS sync loop started, running every %ss (retry every %ss on failure)", INTERVAL, RETRY_INTERVAL)
    while True:
        try:
            ok = sync_all()
        except Exception as e:
            logger.exception("HDFS sync loop iteration failed: %s", e)
            ok = False
        # e.g. namenode still starting up on first run - don't make the user
        # wait a full hour to find out the retry worked.
        time.sleep(INTERVAL if ok else RETRY_INTERVAL)


if __name__ == "__main__":
    main()
