"""
Central configuration for the Pune Traffic + Weather pipeline.
All secrets are read from environment variables  -  never hardcode API keys.

Set these before running (e.g. in a .env file loaded by python-dotenv, or
exported in your shell):

    export TOMTOM_API_KEY="your_tomtom_key"
    export OPENWEATHER_API_KEY="your_openweather_key"
    export POSTGRES_DSN="postgresql://user:password@localhost:5432/traffic_db"
"""

import os

# ---- API keys -------------------------------------------------------------
TOMTOM_API_KEY = os.environ.get("TOMTOM_API_KEY", "")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

# ---- Database ---------------------------------------------------------------
POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://traffic_user:traffic_pass@localhost:5432/traffic_db"
)

# ---- Polling ----------------------------------------------------------------
POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "45"))  # 30-60s range

# ---- Storage paths ------------------------------------------------------------
RAW_JSON_DIR = os.environ.get("RAW_JSON_DIR", "./data/raw_json")
PARQUET_DIR = os.environ.get("PARQUET_DIR", "./data/parquet")

# ---- HDFS (optional, used once you have a Hadoop cluster available) --------
HDFS_ENABLED = os.environ.get("HDFS_ENABLED", "false").lower() == "true"
HDFS_URL = os.environ.get("HDFS_URL", "http://localhost:9870")  # WebHDFS endpoint
HDFS_USER = os.environ.get("HDFS_USER", "hadoop")
HDFS_RAW_PATH = os.environ.get("HDFS_RAW_PATH", "/traffic/raw_json")

# ---- Kafka (Phase 5: feeds Spark Structured Streaming) ----------------------
KAFKA_ENABLED = os.environ.get("KAFKA_ENABLED", "false").lower() == "true"
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "traffic-snapshots")

# ---- Locations to monitor in Pune --------------------------------------------
# Add/edit freely. lat/lon drive both TomTom Flow queries and OpenWeather.
# For TomTom Incidents we use a bounding box (bbox) around the same point.
LOCATIONS = [
    {
        "location_id": "shivajinagar",
        "name": "Shivajinagar",
        "lat": 18.5308,
        "lon": 73.8478,
    },
    {
        "location_id": "hinjewadi",
        "name": "Hinjewadi IT Park",
        "lat": 18.5912,
        "lon": 73.7389,
    },
    {
        "location_id": "hadapsar",
        "name": "Hadapsar",
        "lat": 18.5089,
        "lon": 73.9260,
    },
    {
        "location_id": "kothrud",
        "name": "Kothrud",
        "lat": 18.5074,
        "lon": 73.8077,
    },
    {
        "location_id": "viman_nagar",
        "name": "Viman Nagar",
        "lat": 18.5679,
        "lon": 73.9143,
    },
]

# Half-width of the bounding box (in degrees) used for TomTom Incident API
# ~0.02 degrees ~ 2.2 km  -  a small area around each point.
INCIDENT_BBOX_DELTA = 0.02
