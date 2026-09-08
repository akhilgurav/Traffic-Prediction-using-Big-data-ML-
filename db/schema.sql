-- Pune Traffic Pipeline — PostgreSQL schema
-- Run with: psql -U traffic_user -d traffic_db -f db/schema.sql

CREATE TABLE IF NOT EXISTS locations (
    location_id     TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    -- Neighborhood this road/junction belongs to (e.g. "kothrud"). Several
    -- locations now share one area so predictions can be made per-road while
    -- still rolling back up to a neighborhood view on the dashboard.
    area             TEXT,
    lat              DOUBLE PRECISION NOT NULL,
    lon              DOUBLE PRECISION NOT NULL
);

-- One row per (location, poll timestamp). This is the "unified record"
-- that Phase 1 is responsible for producing.
CREATE TABLE IF NOT EXISTS traffic_snapshots (
    id                      BIGSERIAL PRIMARY KEY,
    location_id             TEXT NOT NULL REFERENCES locations(location_id),
    -- Denormalized copy of locations.area - export_to_parquet.py does a
    -- straight `SELECT * FROM traffic_snapshots`, so keeping area here
    -- (rather than only in `locations`) means feature_engineering.py and
    -- train_model.py get it for free with no join.
    area                     TEXT,
    collected_at            TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- TomTom Flow API fields
    current_speed           DOUBLE PRECISION,
    free_flow_speed         DOUBLE PRECISION,
    current_travel_time     INTEGER,          -- seconds
    free_flow_travel_time   INTEGER,          -- seconds
    confidence               DOUBLE PRECISION,
    road_closure             BOOLEAN,
    -- Polyline of the actual road segment this reading covers, as returned
    -- by TomTom Flow: [[lat, lon], [lat, lon], ...]. Used to draw real roads
    -- on the map instead of a single point per location.
    segment_geometry          JSONB,

    -- TomTom Incidents API (aggregated for the bbox)
    incident_count           INTEGER,
    incident_severity_max     INTEGER,        -- 0-4 magnitude of delay, TomTom scale
    incident_severity_avg     DOUBLE PRECISION, -- mean magnitude across the bbox's incidents - unlike max, this doesn't pin at 4 in dense areas

    -- OpenWeather fields
    temperature_c             DOUBLE PRECISION,
    feels_like_c               DOUBLE PRECISION,
    humidity_pct                DOUBLE PRECISION,
    wind_speed_ms                DOUBLE PRECISION,
    rain_1h_mm                    DOUBLE PRECISION,
    weather_main                   TEXT,
    weather_description             TEXT,
    visibility_m                     INTEGER,

    -- path to the exact raw JSON backup this row was derived from
    raw_json_path                     TEXT
);

CREATE INDEX IF NOT EXISTS idx_snapshots_location_time
    ON traffic_snapshots (location_id, collected_at);

CREATE INDEX IF NOT EXISTS idx_snapshots_time
    ON traffic_snapshots (collected_at);

-- Migration for databases created before segment_geometry existed - schema.sql
-- only auto-runs on first container creation (empty data dir), so if you have
-- an existing Postgres volume, apply this manually:
--   docker compose exec postgres psql -U traffic_user -d traffic_db -f db/schema.sql
-- (safe to re-run - every statement here is idempotent)
ALTER TABLE traffic_snapshots ADD COLUMN IF NOT EXISTS segment_geometry JSONB;

-- Migration for databases created before area existed (road-level location
-- granularity) - same manual-apply caveat as segment_geometry above.
ALTER TABLE locations ADD COLUMN IF NOT EXISTS area TEXT;
ALTER TABLE traffic_snapshots ADD COLUMN IF NOT EXISTS area TEXT;
ALTER TABLE traffic_snapshots ADD COLUMN IF NOT EXISTS incident_severity_avg DOUBLE PRECISION;

-- Phase 5: one row per (location, poll timestamp) that the streaming job
-- actually scored. IF NOT EXISTS is safe to re-run against your existing
-- database - schema.sql only auto-runs on first container creation, so
-- you'll need to apply this manually (see Phase 5 setup instructions).
CREATE TABLE IF NOT EXISTS live_predictions (
    id                              BIGSERIAL PRIMARY KEY,
    location_id                     TEXT NOT NULL,
    collected_at                    TIMESTAMPTZ NOT NULL,
    predicted_congestion_level      INTEGER NOT NULL,
    predicted_congestion_label      TEXT NOT NULL,
    prob_low                        DOUBLE PRECISION,
    prob_moderate                   DOUBLE PRECISION,
    prob_high                       DOUBLE PRECISION,
    speed_ratio                     DOUBLE PRECISION,
    delay                           INTEGER,
    scored_at                       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_predictions_location_time
    ON live_predictions (location_id, collected_at);

CREATE INDEX IF NOT EXISTS idx_predictions_scored_at
    ON live_predictions (scored_at);
