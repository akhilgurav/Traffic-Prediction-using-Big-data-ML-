"""
Thin PostgreSQL access layer used by the collector.
Uses psycopg2 with a simple connection-per-call pattern  -  fine at
30-60s polling frequency. Swap for a connection pool later if needed.
"""

import psycopg2
import psycopg2.extras
import logging

from config.settings import POSTGRES_DSN, LOCATIONS

logger = logging.getLogger(__name__)


def get_connection():
    return psycopg2.connect(POSTGRES_DSN)


def ensure_locations_seeded():
    """Insert configured locations into the locations table if missing."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            for loc in LOCATIONS:
                cur.execute(
                    """
                    INSERT INTO locations (location_id, name, area, lat, lon)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (location_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        area = EXCLUDED.area,
                        lat = EXCLUDED.lat,
                        lon = EXCLUDED.lon
                    """,
                    (loc["location_id"], loc["name"], loc.get("area"), loc["lat"], loc["lon"]),
                )
        conn.commit()
    logger.info("Locations table seeded (%d locations).", len(LOCATIONS))


def insert_snapshot(record: dict):
    """
    Insert one unified record (dict) into traffic_snapshots.
    Missing keys default to NULL  -  collectors should still populate the
    dict with None for any field they couldn't fetch, so partial API
    failures don't crash the whole poll cycle.
    """
    columns = [
        "location_id", "area", "current_speed", "free_flow_speed",
        "current_travel_time", "free_flow_travel_time", "confidence",
        "road_closure", "segment_geometry", "incident_count", "incident_severity_max",
        "incident_severity_avg",
        "temperature_c", "feels_like_c", "humidity_pct", "wind_speed_ms",
        "rain_1h_mm", "weather_main", "weather_description", "visibility_m",
        "raw_json_path",
    ]
    values = [
        psycopg2.extras.Json(record["segment_geometry"])
        if c == "segment_geometry" and record.get("segment_geometry") is not None
        else record.get(c)
        for c in columns
    ]
    placeholders = ", ".join(["%s"] * len(columns))
    col_list = ", ".join(columns)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO traffic_snapshots ({col_list}) VALUES ({placeholders})",
                values,
            )
        conn.commit()
