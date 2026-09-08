"""
Phase 1  -  Python Data Collector.

Polls TomTom Flow, TomTom Incidents, and OpenWeather for every configured
location, merges them into one unified JSON record, and stores it two ways:

  1. Raw JSON  -> data/raw_json/<location_id>/<timestamp>.json   (backup, and
                  later gets moved/synced into HDFS)
  2. Structured -> PostgreSQL traffic_snapshots table

Run:
    python collector_main.py

Run continuously in the background (Linux/Mac):
    nohup python collector_main.py > collector.log 2>&1 &

No Spark, no Kafka at this stage  -  deliberately simple and easy to debug.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

from config.settings import LOCATIONS, POLL_INTERVAL_SECONDS, RAW_JSON_DIR
from collectors.tomtom_traffic import fetch_flow
from collectors.tomtom_incidents import fetch_incidents
from collectors.openweather import fetch_weather
from db.db_utils import ensure_locations_seeded, insert_snapshot
from kafka_producer import publish_record

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def collect_one_location(loc: dict) -> dict:
    """Fetch all three APIs for one location and merge into one record."""
    now = datetime.now(timezone.utc)
    lat, lon = loc["lat"], loc["lon"]

    flow = fetch_flow(lat, lon) or {}
    incidents = fetch_incidents(lat, lon) or {}
    weather = fetch_weather(lat, lon) or {}

    unified = {
        "location_id": loc["location_id"],
        "location_name": loc["name"],
        "area": loc.get("area"),
        "lat": lat,
        "lon": lon,
        "collected_at": now.isoformat(),

        "current_speed": flow.get("current_speed"),
        "free_flow_speed": flow.get("free_flow_speed"),
        "current_travel_time": flow.get("current_travel_time"),
        "free_flow_travel_time": flow.get("free_flow_travel_time"),
        "confidence": flow.get("confidence"),
        "road_closure": flow.get("road_closure"),
        # Polyline of the actual road segment this reading covers - [[lat, lon], ...]
        "segment_geometry": flow.get("segment_geometry"),

        "incident_count": incidents.get("incident_count"),
        "incident_severity_max": incidents.get("incident_severity_max"),
        "incident_severity_avg": incidents.get("incident_severity_avg"),

        "temperature_c": weather.get("temperature_c"),
        "feels_like_c": weather.get("feels_like_c"),
        "humidity_pct": weather.get("humidity_pct"),
        "wind_speed_ms": weather.get("wind_speed_ms"),
        "rain_1h_mm": weather.get("rain_1h_mm"),
        "weather_main": weather.get("weather_main"),
        "weather_description": weather.get("weather_description"),
        "visibility_m": weather.get("visibility_m"),

        # keep raw API payloads too, for debugging / future feature ideas
        "_raw": {
            "flow": flow.get("_raw"),
            "incidents": incidents.get("_raw"),
            "weather": weather.get("_raw"),
        },
    }
    return unified


def write_raw_json(record: dict) -> str:
    """
    Append the unified record as one JSON line to
    data/raw_json/<location_id>/<date>.jsonl
    Returns the file path (all records for that location+day share one file).

    Using one growing file per location per day instead of one file per poll
    keeps the file count sane over a multi-week run (5 files/day instead of
    thousands) and maps cleanly onto daily HDFS backup later.
    """
    loc_dir = os.path.join(RAW_JSON_DIR, record["location_id"])
    os.makedirs(loc_dir, exist_ok=True)
    date_str = record["collected_at"][:10]  # YYYY-MM-DD
    path = os.path.join(loc_dir, f"{date_str}.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return path


def poll_once():
    for loc in LOCATIONS:
        try:
            record = collect_one_location(loc)
            raw_path = write_raw_json(record)
            publish_record(record)  # Phase 5: feed Spark Structured Streaming (no-op if disabled)

            db_record = dict(record)
            db_record["raw_json_path"] = raw_path
            db_record.pop("_raw", None)
            db_record.pop("location_name", None)
            db_record.pop("lat", None)
            db_record.pop("lon", None)
            db_record.pop("collected_at", None)  # DB default now() handles this

            insert_snapshot(db_record)
            logger.info(
                "OK  %-16s speed=%s free_flow=%s incidents=%s temp=%s",
                loc["location_id"],
                record.get("current_speed"),
                record.get("free_flow_speed"),
                record.get("incident_count"),
                record.get("temperature_c"),
            )
        except Exception as e:
            # Never let one location's failure kill the whole poll cycle
            logger.exception("Failed to collect for %s: %s", loc["location_id"], e)


def main():
    logger.info("Starting Pune Traffic Collector  -  polling every %ss", POLL_INTERVAL_SECONDS)
    ensure_locations_seeded()

    while True:
        start = time.time()
        poll_once()
        elapsed = time.time() - start
        sleep_for = max(0, POLL_INTERVAL_SECONDS - elapsed)
        logger.info("Cycle took %.1fs, sleeping %.1fs", elapsed, sleep_for)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
