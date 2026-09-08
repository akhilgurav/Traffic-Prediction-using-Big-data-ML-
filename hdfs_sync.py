"""
HDFS backup sync.

Mirrors the local data/raw_json and data/parquet directories into HDFS over
WebHDFS. This is the "HDFS (Raw Backup)" branch of the architecture diagram
-  it runs ALONGSIDE Kafka -> Spark Streaming, not instead of it. Local
files on the ./data volume remain the source of truth for every other
service (feature engineering, training, the dashboard); this only keeps a
second, durable copy for long-term / historical storage.

Safe by design: if HDFS is unreachable or HDFS_ENABLED=false, sync_all()
logs and returns without raising  -  exactly like kafka_producer.py's
"never take down the pipeline" philosophy.

Run once:
    python hdfs_sync.py
Run on a schedule (see docker-compose.yml `hdfs-sync` service):
    python -m scripts.hdfs_sync_loop
"""

import logging
import os

from config.settings import (
    HDFS_ENABLED,
    HDFS_RAW_PATH,
    HDFS_URL,
    HDFS_USER,
    PARQUET_DIR,
    RAW_JSON_DIR,
)

logger = logging.getLogger(__name__)

HDFS_PARQUET_PATH = os.environ.get("HDFS_PARQUET_PATH", "/traffic/parquet")


def get_client():
    """
    Lazy import so `hdfs` (the WebHDFS client library) is only required when
    HDFS is actually used -  every other service in this project runs fine
    without it installed.
    """
    from hdfs import InsecureClient

    return InsecureClient(HDFS_URL, user=HDFS_USER)


def _upload_tree(client, local_root: str, hdfs_root: str) -> int:
    """
    Walk local_root and upload every file to the matching path under
    hdfs_root, preserving the local directory structure (so
    data/raw_json/<location_id>/<date>.jsonl becomes
    <hdfs_root>/<location_id>/<date>.jsonl, and likewise for the
    date=YYYY-MM-DD/ partitions under data/parquet).

    Skips files whose HDFS copy already has the same byte size, so a
    growing raw_json/*.jsonl file that was already uploaded gets re-sent
    on the next run (its size changed) while untouched historical parquet
    partitions don't get needlessly re-uploaded every hour.
    """
    if not os.path.isdir(local_root):
        logger.info("Nothing to sync yet at %s", local_root)
        return 0

    uploaded = 0
    for dirpath, _dirnames, filenames in os.walk(local_root):
        rel = os.path.relpath(dirpath, local_root)
        hdfs_dir = hdfs_root if rel == "." else f"{hdfs_root}/{rel.replace(os.sep, '/')}"

        for fname in filenames:
            local_path = os.path.join(dirpath, fname)
            hdfs_path = f"{hdfs_dir}/{fname}"

            try:
                status = client.status(hdfs_path, strict=False)
            except Exception:
                status = None
            if status is not None and status.get("length") == os.path.getsize(local_path):
                continue  # already up to date

            client.upload(hdfs_path, local_path, overwrite=True)
            uploaded += 1

    return uploaded


def sync_all() -> bool:
    if not HDFS_ENABLED:
        logger.info("HDFS_ENABLED=false, skipping HDFS sync.")
        return True

    try:
        client = get_client()
        client.makedirs(HDFS_RAW_PATH)
        client.makedirs(HDFS_PARQUET_PATH)

        n_raw = _upload_tree(client, RAW_JSON_DIR, HDFS_RAW_PATH)
        n_parquet = _upload_tree(client, PARQUET_DIR, HDFS_PARQUET_PATH)

        logger.info(
            "HDFS sync complete: %d raw JSON file(s) -> %s, %d parquet file(s) -> %s",
            n_raw, HDFS_RAW_PATH, n_parquet, HDFS_PARQUET_PATH,
        )
        return True
    except Exception as e:
        # Same philosophy as kafka_producer.py: HDFS being down should never
        # take down the collector/exporter/training pipeline.
        logger.warning("HDFS sync failed (%s) - local data is unaffected.", e)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sync_all()
