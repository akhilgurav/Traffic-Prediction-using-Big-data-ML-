"""
One-off backfill: recovers TomTom road segment geometry from your EXISTING
raw JSON backups (data/raw_json/<location_id>/*.jsonl) and writes it into
traffic_snapshots.segment_geometry for rows collected before that column
existed.

Why this works: the collector always wrote the full TomTom response
(record["_raw"]["flow"]) to the raw JSON files - it was only stripped right
before the Postgres insert. So the geometry for your whole history is
already sitting on disk, just never made it into the database until now.

Matching strategy: Postgres' collected_at is set by DEFAULT now() at INSERT
time, not copied from the record's own "collected_at" field, so the two
timestamps are close but not always byte-identical. For each location, this
does a nearest-timestamp match (within --tolerance-seconds) between raw
JSON lines and DB rows that don't have geometry yet, then bulk-updates.

Safe to re-run:
  - Raw JSON files are only ever read, never modified.
  - Only rows where segment_geometry IS NULL are touched.
  - Re-running after a partial run just picks up whatever's still NULL.

Usage:
    python backfill_segment_geometry.py                  # do it
    python backfill_segment_geometry.py --dry-run         # report only
    python backfill_segment_geometry.py --tolerance-seconds 5
"""

import argparse
import bisect
import json
import logging
from datetime import datetime
from pathlib import Path

import psycopg2
import psycopg2.extras

from config.settings import POSTGRES_DSN, RAW_JSON_DIR, LOCATIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_geometries(location_dir: Path):
    """
    Yield (collected_at, geometry) for every raw JSON line under a
    location's folder that has road segment coordinates. Malformed lines
    or lines without geometry are skipped, not fatal.
    """
    for path in sorted(location_dir.glob("*.jsonl")):
        with open(path, "r") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON at %s:%d", path, line_no)
                    continue

                flow_raw = (rec.get("_raw") or {}).get("flow") or {}
                coords = (flow_raw.get("coordinates") or {}).get("coordinate", [])
                geometry = [
                    [c["latitude"], c["longitude"]]
                    for c in coords
                    if "latitude" in c and "longitude" in c
                ]
                if len(geometry) < 2:
                    continue

                try:
                    collected_at = datetime.fromisoformat(rec["collected_at"])
                except (KeyError, ValueError, TypeError):
                    continue

                yield collected_at, geometry


def backfill_location(conn, location_id: str, raw_dir: Path, tolerance: float, dry_run: bool):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, collected_at FROM traffic_snapshots
            WHERE location_id = %s AND segment_geometry IS NULL
            ORDER BY collected_at
            """,
            (location_id,),
        )
        rows = cur.fetchall()

    if not rows:
        return 0, 0

    ids = [r[0] for r in rows]
    timestamps = [r[1] for r in rows]

    # row_id -> (best_diff_seconds, geometry_json_str)
    best_match = {}

    for collected_at, geometry in load_raw_geometries(raw_dir):
        idx = bisect.bisect_left(timestamps, collected_at)
        candidates = [i for i in (idx - 1, idx) if 0 <= i < len(timestamps)]
        if not candidates:
            continue
        best_i = min(candidates, key=lambda i: abs((timestamps[i] - collected_at).total_seconds()))
        diff = abs((timestamps[best_i] - collected_at).total_seconds())
        if diff > tolerance:
            continue
        row_id = ids[best_i]
        prev = best_match.get(row_id)
        if prev is None or diff < prev[0]:
            best_match[row_id] = (diff, json.dumps(geometry))

    pairs = [(row_id, geom) for row_id, (_, geom) in best_match.items()]

    if pairs and not dry_run:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                UPDATE traffic_snapshots AS t
                SET segment_geometry = v.geom::jsonb
                FROM (VALUES %s) AS v(id, geom)
                WHERE t.id = v.id
                """,
                pairs,
                template="(%s, %s)",
            )
        conn.commit()

    return len(pairs), len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Report match counts without writing anything")
    parser.add_argument("--tolerance-seconds", type=float, default=10.0,
                         help="Max gap between a raw JSON line's timestamp and a DB row's "
                              "collected_at for them to be considered the same reading (default: 10s)")
    args = parser.parse_args()

    raw_root = Path(RAW_JSON_DIR)
    conn = psycopg2.connect(POSTGRES_DSN)

    total_matched, total_candidates = 0, 0
    try:
        for loc in LOCATIONS:
            location_id = loc["location_id"]
            raw_dir = raw_root / location_id
            if not raw_dir.exists():
                logger.info("[%s] no raw_json folder, skipping", location_id)
                continue

            matched, candidates = backfill_location(conn, location_id, raw_dir, args.tolerance_seconds, args.dry_run)
            total_matched += matched
            total_candidates += candidates
            logger.info(
                "[%s] %s%d / %d rows without geometry matched",
                location_id, "(dry-run) " if args.dry_run else "", matched, candidates,
            )
    finally:
        conn.close()

    logger.info(
        "Done. %s%d / %d total rows matched.",
        "(dry-run, nothing written) " if args.dry_run else "",
        total_matched, total_candidates,
    )
    if total_candidates - total_matched:
        logger.info(
            "%d rows stayed without geometry - likely predate the collector reaching this "
            "location, or fell outside the --tolerance-seconds window (try increasing it).",
            total_candidates - total_matched,
        )


if __name__ == "__main__":
    main()
