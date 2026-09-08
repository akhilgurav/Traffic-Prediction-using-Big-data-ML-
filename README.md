# Pune Traffic & Congestion Prediction Pipeline

Step-by-step build guide, matching the 5 phases in your plan.

---

## Phase 1 — Data Collection (build this first, run for 2–4 weeks)

**Goal:** continuously collect traffic + weather data for Pune locations,
store it as raw JSON, structured rows in PostgreSQL, and Parquet for
analytics. **No Spark or Kafka yet.**

Everything below runs in Docker — you only need Docker Desktop (or
Docker Engine + Compose plugin on Linux) installed on your machine.
Nothing else to install locally.

### 1. Get API keys
- TomTom: https://developer.tomtom.com/ → create an app → get an API key
  (free tier: 2,500 requests/day, which is plenty at 45s polling for 5 locations)
- OpenWeather: https://openweathermap.org/api → free tier current-weather API key

### 2. Configure secrets
```bash
cp .env.example .env
# edit .env and fill in TOMTOM_API_KEY, OPENWEATHER_API_KEY
# (Postgres user/password/db already have sane defaults in .env.example)
```
`docker-compose.yml` reads `.env` automatically — nothing to `export` manually.

### 3. Edit your locations (optional)
Open `config/settings.py` and adjust the `LOCATIONS` list — 5 Pune spots
are pre-filled (Shivajinagar, Hinjewadi, Hadapsar, Kothrud, Viman Nagar).
Add more or change coordinates as you like, then rebuild (step 4 already
covers this — compose rebuilds automatically on `up --build`).

### 4. Build and start everything
```bash
docker compose up --build -d
```
This starts three containers:
- **postgres** — the database, schema auto-loaded from `db/schema.sql` on first run
- **collector** — polls the 3 APIs every `POLL_INTERVAL_SECONDS` (default 45s),
  writes raw JSON + inserts into Postgres
- **exporter** — runs `export_to_parquet.py` on a loop (default every 24h)

Check it's working:
```bash
docker compose logs -f collector
```
You should see log lines like:
```
OK  shivajinagar    speed=32.1 free_flow=45.0 incidents=1 temp=28.4
```

Let this stack run continuously for **2–4 weeks** — this is your training
data window. `restart: unless-stopped` means it survives reboots and Docker
daemon restarts automatically.

### 5. Inspect the data anytime
Raw JSON, Parquet, and CSV all land in `./data/` on your host machine
(mounted as a volume, so they persist and are browsable outside the
container):
```
data/raw_json/<location_id>/<date>.jsonl   (one line per poll, appended)
data/parquet/date=<date>/part.parquet
data/csv/traffic_snapshots.csv             (full table, one flat file)
```
To get a fresh CSV of everything collected so far - handy for a quick
look in Excel or sharing a snapshot of the data:
```bash
docker compose run --rm collector python export_to_csv.py
```
This overwrites `data/csv/traffic_snapshots.csv` with the current full
table each time you run it. Re-run it anytime you want an up-to-date copy.

To query Postgres directly:
```bash
docker compose exec postgres psql -U traffic_user -d traffic_db \
  -c "SELECT count(*) FROM traffic_snapshots;"
```
Or browse it visually with pgAdmin (optional, off by default):
```bash
docker compose --profile tools up -d pgadmin
# open http://localhost:5050, login with PGADMIN_EMAIL / PGADMIN_PASSWORD from .env
```

### 6. Common operations
```bash
docker compose stop              # pause everything, keep data
docker compose start             # resume
docker compose down              # stop + remove containers (data volume persists)
docker compose down -v           # ⚠ also deletes the Postgres volume — full reset
docker compose logs -f exporter  # watch the parquet export cycle
docker compose ps                # see what's running
```

### 7. HDFS backup (raw JSON + Parquet, mirrored automatically)
A small single-node HDFS cluster (namenode + datanode) plus an `hdfs-sync`
container that periodically mirrors `data/raw_json/` and `data/parquet/`
into it — the "HDFS (Raw Backup)" branch of the architecture diagram. It
runs alongside everything else, not instead of it: local files stay the
source of truth for feature engineering, training, and the dashboard.

```bash
docker compose --profile hdfs up -d namenode datanode hdfs-sync
docker compose logs -f hdfs-sync
```
- HDFS web UI: http://localhost:9870 (browse `/traffic/raw_json` and `/traffic/parquet`)
- Sync runs every `HDFS_SYNC_INTERVAL_SECONDS` (default 3600s / 1h) — set it lower for a demo:
  ```bash
  HDFS_SYNC_INTERVAL_SECONDS=60 docker compose --profile hdfs up -d hdfs-sync
  ```
- Run a one-off sync manually instead of waiting for the schedule:
  ```bash
  docker compose --profile hdfs run --rm hdfs-sync python hdfs_sync.py
  ```
If HDFS isn't running (or the `hdfs` profile is never started), nothing else in
the pipeline notices or is affected — same "fail quiet, never take down the
collector" philosophy as the Kafka publish path.

**Checkpoint:** after 2-4 weeks you should have thousands of rows in
`traffic_snapshots`, matching raw JSON files, daily Parquet partitions, and
(if you started the `hdfs` profile) a mirrored copy of both in HDFS.
That's your dataset for Phases 2-4.

---

## Phase 2 — Feature Engineering (build once you have ~1-2 weeks of data)

Compute, per location per timestamp:
- `speed_ratio` = current_speed / free_flow_speed
- `delay` = current_travel_time − free_flow_travel_time
- `hour`, `weekday` — from `collected_at`
- `peak_hour` — boolean flag for e.g. 8-11am / 5-8pm
- `rolling_average_speed` — mean current_speed over the trailing 15 minutes,
  per location (needs a window function ordered by time)
- `congestion_change_rate` — change in speed_ratio vs. the previous snapshot

This can be done two ways:
- **Pandas**, working off the Parquet files — good for exploration/notebooks
- **Spark**, working off HDFS/Parquet — good once your dataset is large and
  you want this logic to be reusable in Phase 5's streaming job

I'd suggest we build the Pandas version first (fast to iterate on, easy to
sanity-check the features), then port the same logic into a Spark batch job
that becomes the seed for the streaming feature pipeline in Phase 5.

---

## Phase 3 — Label Generation

Turn `speed_ratio` / `delay` into a categorical `congestion_level` label,
e.g.:
- `speed_ratio > 0.8` → **free_flow**
- `0.5 < speed_ratio <= 0.8` → **moderate**
- `speed_ratio <= 0.5` → **congested**

Thresholds should be tuned by looking at the actual distribution of
`speed_ratio` in your collected data (a histogram will tell you where the
natural breakpoints are, rather than guessing).

---

## Phase 4 — Model Training (Spark MLlib RandomForest)

- Load the labeled, feature-engineered historical Parquet dataset into Spark
- Assemble features with `VectorAssembler`
- Train/test split (e.g. time-based split, not random — don't leak future
  into past for a time series problem)
- Train `RandomForestClassifier` (Spark MLlib)
- Evaluate (accuracy, F1, confusion matrix across the 3 congestion classes)
- Save the trained model (`model.write().save(...)`)

---

## Dashboard — 5-page interactive Command Center

`dashboard/app.py` (Streamlit + Plotly) has five pages, navigable from the sidebar:

1. **Real-Time Monitoring** — current speed, travel time, incidents, weather (live speed map + per-location cards)
2. **Traffic Prediction** — forecast congestion class per location, with an estimated future speed/travel time and a probability trend chart (needs `live_predictions` populated, i.e. Phase 5 streaming running)
3. **Risk & Incident Forecast** — incident probability, road-closure risk, and a composite 0–100 risk score per location
4. **Traffic Analytics** — historical speed trends, hotspot ranking, weather-impact scatter, hour-of-day heatmap
5. **Model Performance** — MAE, RMSE, Accuracy, Precision, Recall, F1, confusion matrix and feature importance, read straight from `data/models/congestion_rf/metrics.json`

Start it with:
```bash
docker compose --profile dashboard up -d dashboard
# open http://localhost:8501
```
Each page degrades gracefully (with a clear "run this command" message) if the upstream service it depends on (collector / streaming / trained model) hasn't been started yet.

`spark_jobs/train_model.py` now writes `metrics.json` alongside the saved model every time it runs — that's what page 5 reads, so re-running training automatically refreshes the dashboard.

---

## Phase 5 — Real-Time Prediction

- Spark Structured Streaming consumes from Kafka (live traffic API → Kafka,
  as in your architecture diagram)
- Apply the **same feature engineering logic** from Phase 2 (this is why
  building it once, reusably, matters)
- Load the trained model from Phase 4 and run `.transform()` for live inference
- Write predictions to PostgreSQL for the Streamlit/Grafana dashboard

**Docker note:** `docker-compose.yml` already has `kafka` + `zookeeper`
services defined, just kept behind a profile so they don't start until
you need them:
```bash
docker compose --profile kafka up -d
```
When we get here, the collector will be pointed at a Kafka producer
instead of writing straight to Postgres, and a new `spark-streaming`
service will be added to the compose file for the consumer/inference job.

---

## Real road segments on the map + Grafana geomap & alerting

The collector now captures the actual road polyline TomTom's Flow API returns
for each reading (`segment_geometry`, stored as JSONB), instead of throwing
it away. This powers:

- **Streamlit → Real-Time Monitoring**: the "Live Road Congestion Map" now
  draws the real road segments, colored by live speed ratio, instead of a
  dot per location. Rows collected *before* this update won't have geometry
  yet and fall back to a marker.
- **Grafana**: a new "Live Congestion Map" geomap panel (point-per-location,
  colored by speed ratio — Grafana core doesn't have a fully reliable way to
  render dynamic per-row road polylines without a plugin, so true road
  segments stay on the Streamlit dashboard), plus a bar gauge of current
  speed by location, two stat panels (roads congested now, avg speed ratio),
  and a table of active incidents/closures.
- **Grafana alerting**: two provisioned rules — *high congestion* (any
  location's speed ratio < 0.4 for 5 consecutive minutes) and *road closure
  detected* (any monitored road reports closed). Both route to a
  `traffic-alerts` contact point that ships as a placeholder webhook — edit
  `grafana/provisioning/alerting/contactpoints.yml` with a real Slack/
  Discord/PagerDuty/webhook URL (or swap in an `email` receiver + SMTP env
  vars) before you rely on it. Rules show up under Alerting → Alert rules →
  "Pune Traffic" folder.

**If you already have a running Postgres volume**, `schema.sql` only
auto-runs on first container creation, so apply the new column manually:
```bash
docker compose exec postgres psql -U traffic_user -d traffic_db -f db/schema.sql
```
(every statement in it is safe to re-run). New snapshots collected after
that will start carrying road geometry; older rows stay point-only —
**unless** you run the backfill script below.

**Recovering geometry for data you already collected:** your raw JSON
backups (`data/raw_json/`) already contain the full TomTom response,
coordinates included — it was only ever stripped right before the Postgres
insert. So your history isn't stuck as point-only:
```bash
python backfill_segment_geometry.py --dry-run   # see match counts first
python backfill_segment_geometry.py              # then actually write
```
It matches raw JSON lines to DB rows by location + nearest timestamp
(10s tolerance by default — widen with `--tolerance-seconds` if a lot of
rows go unmatched) and only touches rows where `segment_geometry` is still
NULL, so it's safe to re-run.

---

## Suggested order of work with me

1. ✅ Phase 1 scaffolding (done — code above)
2. You run Phase 1 for a couple weeks while we build Phase 2/3 against a
   small sample of real data you've collected
3. Once you have a few thousand rows, we build the feature engineering +
   labeling scripts against real data (not synthetic) so thresholds are grounded
4. Then Spark MLlib training
5. Then Kafka + Streaming inference

Want me to also generate a `docker-compose.yml` for Postgres (and later
Kafka) so setup is one command instead of manual installs?
