"""
Phase 5b - Real-time congestion prediction (Spark Structured Streaming)

    Kafka (traffic-snapshots)
      -> parse JSON
      -> [foreachBatch] pull trailing 15-min context per location from
         Postgres (traffic_snapshots), union with the new Kafka rows
      -> build_features()            <- THE SAME function Phase 2/3 use
      -> load Phase 4's trained PipelineModel, .transform() to predict
      -> write predictions to Postgres (live_predictions table)

WHY foreachBatch + a Postgres round-trip instead of pure streaming
aggregation: rolling_average_speed / congestion_change_rate in
build_features() use Window.partitionBy(...).orderBy(...).rangeBetween(...),
which Structured Streaming does not support directly on a streaming
DataFrame. foreachBatch hands each micro-batch to us as a plain static
DataFrame, where arbitrary window functions work fine - but a micro-batch
by itself is only 1-5 rows (however many locations polled in this ~30s
window), nowhere near enough for a meaningful 15-minute rolling average.
So each batch pulls its own trailing context from Postgres (the collector
has already been writing there directly since Phase 1), unions it with the
new rows, runs build_features() over the combined set, then keeps only the
rows that belong to THIS batch to actually score.

Run as a long-running service (see docker-compose.yml `spark-streaming`):
    spark-submit \\
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.postgresql:postgresql:42.7.3 \\
        spark_jobs/streaming_predict.py
"""

import os
import sys
from datetime import timedelta

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# feature_engineering.py lives right next to this file - spark-submit puts
# this script's own directory on sys.path, so a plain import works without
# needing spark_jobs to be an installed package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import ROLLING_WINDOW_SECONDS, build_features  # noqa: E402

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "traffic-snapshots")
MODEL_DIR = os.environ.get("MODEL_DIR", "data/models/congestion_rf")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "data/checkpoints/streaming_predict")

POSTGRES_JDBC_URL = os.environ.get("POSTGRES_JDBC_URL", "jdbc:postgresql://postgres:5432/traffic_db")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "traffic_user")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "traffic_pass")
JDBC_PROPS = {"user": POSTGRES_USER, "password": POSTGRES_PASSWORD, "driver": "org.postgresql.Driver"}

LABELS = {0: "low", 1: "moderate", 2: "high"}

# Same shape the Kafka producer sends (kafka_producer.py), minus _raw.
RECORD_SCHEMA = StructType(
    [
        StructField("location_id", StringType()),
        StructField("location_name", StringType()),
        StructField("lat", DoubleType()),
        StructField("lon", DoubleType()),
        StructField("collected_at", StringType()),
        StructField("current_speed", DoubleType()),
        StructField("free_flow_speed", DoubleType()),
        StructField("current_travel_time", IntegerType()),
        StructField("free_flow_travel_time", IntegerType()),
        StructField("confidence", DoubleType()),
        StructField("road_closure", BooleanType()),
        StructField("incident_count", IntegerType()),
        StructField("incident_severity_max", IntegerType()),
        StructField("temperature_c", DoubleType()),
        StructField("feels_like_c", DoubleType()),
        StructField("humidity_pct", DoubleType()),
        StructField("wind_speed_ms", DoubleType()),
        StructField("rain_1h_mm", DoubleType()),
        StructField("weather_main", StringType()),
        StructField("weather_description", StringType()),
        StructField("visibility_m", IntegerType()),
    ]
)

# Columns that exist both in the Kafka payload and in traffic_snapshots -
# this is exactly what build_features() + the model need, plus join keys.
CONTEXT_COLUMNS = [f.name for f in RECORD_SCHEMA.fields if f.name not in ("location_name", "lat", "lon")]


def fill_defaults_for_scoring(df):
    """
    Same intent as train_model.py's fill_defaults(), but simplified: a live
    micro-batch is too small to compute a meaningful column mean the way
    training does, so weather nulls fall back to fixed, roughly-typical
    Pune values instead. In practice these only ever get hit if OpenWeather
    itself failed (TomTom is the one we've actually seen 403 on) - rare
    enough that a rough fallback beats dropping the row entirely.
    """
    return df.fillna(
        {
            "incident_count": 0,
            "incident_severity_max": 0,
            "confidence": 1.0,
            "rain_1h_mm": 0.0,
            "temperature_c": 27.0,
            "humidity_pct": 60.0,
            "wind_speed_ms": 3.0,
            "visibility_m": 10000,
        }
    )


def main():
    spark = (
        SparkSession.builder.appName("PuneTrafficStreamingPrediction")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    model = PipelineModel.load(MODEL_DIR)
    print(f"Loaded trained model from {MODEL_DIR}", flush=True)

    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = raw_stream.select(
        F.from_json(F.col("value").cast("string"), RECORD_SCHEMA).alias("data")
    ).select("data.*")

    # datetime.isoformat() in Python omits the fractional seconds entirely
    # when microsecond==0, so timestamps aren't 100% uniformly formatted -
    # try the with-microseconds pattern first, fall back to without.
    parsed = parsed.withColumn(
        "collected_at",
        F.coalesce(
            F.to_timestamp("collected_at", "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX"),
            F.to_timestamp("collected_at", "yyyy-MM-dd'T'HH:mm:ssXXX"),
        ),
    )

    def process_batch(batch_df, batch_id):
        batch_df = batch_df.persist()
        count = batch_df.count()
        print(f"[batch {batch_id}] {count} new record(s) from Kafka", flush=True)
        if count == 0:
            batch_df.unpersist()
            return

        locations = [r["location_id"] for r in batch_df.select("location_id").distinct().collect()]
        min_ts = batch_df.agg(F.min("collected_at")).first()[0]
        context_start = min_ts - timedelta(seconds=ROLLING_WINDOW_SECONDS)

        # Push the filter down to Postgres via a subquery, rather than
        # pulling the whole (unboundedly growing) traffic_snapshots table
        # into Spark on every ~30s trigger.
        safe_locations = ", ".join("'" + loc.replace("'", "''") + "'" for loc in locations)
        context_query = (
            "(SELECT " + ", ".join(CONTEXT_COLUMNS) + " FROM traffic_snapshots "
            f"WHERE location_id IN ({safe_locations}) "
            f"AND collected_at >= '{context_start.isoformat()}') AS ctx"
        )
        history = spark.read.jdbc(url=POSTGRES_JDBC_URL, table=context_query, properties=JDBC_PROPS)

        combined = history.unionByName(batch_df.select(*CONTEXT_COLUMNS)).dropDuplicates(
            ["location_id", "collected_at"]
        )

        featured = build_features(combined)
        featured = fill_defaults_for_scoring(featured)

        # Only score rows that belong to THIS batch - history rows were
        # only along for the ride to make rolling/lag features correct.
        new_keys = batch_df.select("location_id", "collected_at").distinct()
        to_score = featured.join(new_keys, on=["location_id", "collected_at"], how="inner")

        if to_score.rdd.isEmpty():
            print(f"[batch {batch_id}] nothing left to score after feature engineering "
                  f"(likely null speed - see the Phase 2 data-quality guard)", flush=True)
            batch_df.unpersist()
            return

        predictions = model.transform(to_score)
        predictions = predictions.withColumn("prob_array", vector_to_array("probability"))

        label_map = F.create_map([F.lit(x) for pair in LABELS.items() for x in pair])
        out = predictions.select(
            "location_id",
            "collected_at",
            F.col("prediction").cast("int").alias("predicted_congestion_level"),
            "speed_ratio",
            "delay",
            F.col("prob_array")[0].alias("prob_low"),
            F.col("prob_array")[1].alias("prob_moderate"),
            F.col("prob_array")[2].alias("prob_high"),
        ).withColumn(
            "predicted_congestion_label", label_map[F.col("predicted_congestion_level")]
        ).withColumn("scored_at", F.current_timestamp())

        out.write.mode("append").jdbc(url=POSTGRES_JDBC_URL, table="live_predictions", properties=JDBC_PROPS)
        print(f"[batch {batch_id}] wrote {out.count()} prediction(s) to live_predictions", flush=True)
        batch_df.unpersist()

    query = (
        parsed.writeStream.foreachBatch(process_batch)
        .option("checkpointLocation", CHECKPOINT_DIR)
        .trigger(processingTime="30 seconds")
        .start()
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()
