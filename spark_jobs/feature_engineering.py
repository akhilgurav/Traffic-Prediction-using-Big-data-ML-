"""
Phase 2 - Feature Engineering (PySpark)

Reads accumulated raw snapshots from Parquet (data/parquet/date=.../part.parquet,
same files export_to_parquet.py has been writing since Phase 1) and computes the
ML features:

    speed_ratio, delay, hour, weekday, peak_hour,
    rolling_average_speed (trailing 15 min), congestion_change_rate

IMPORTANT DESIGN CHOICE: build_features() below is a pure DataFrame -> DataFrame
transform with no Spark session creation or I/O inside it. That's deliberate -
Phase 5's Structured Streaming job will import this exact function and call it
on each micro-batch, so the batch (historical) and streaming (live) feature
logic never drift apart.

Run (batch, historical data):
    spark-submit spark_jobs/feature_engineering.py
"""

import os

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

PARQUET_DIR = os.environ.get("PARQUET_DIR", "data/parquet")
FEATURES_OUT_DIR = os.environ.get("FEATURES_OUT_DIR", "data/features")

# Trailing window for rolling_average_speed, in seconds
ROLLING_WINDOW_SECONDS = 15 * 60

# Hours considered "peak" (24h, local collector time = UTC unless you change it)
MORNING_PEAK = (8, 11)   # 8:00-11:59
EVENING_PEAK = (17, 20)  # 17:00-20:59


def build_features(df):
    """
    Pure transform: raw traffic_snapshots-shaped DataFrame -> feature DataFrame.

    Expects columns: location_id, collected_at (timestamp), current_speed,
    free_flow_speed, current_travel_time, free_flow_travel_time.
    Extra columns (weather, incidents, etc.) just pass through untouched.
    """
    # --- data quality guard --------------------------------------------
    # Rows where the collector/API failed have current_speed / free_flow_speed
    # as NULL (this happens, e.g. transient API errors) - speed_ratio and
    # delay are undefined for them, so drop rather than silently propagate NaN.
    df = df.filter(
        F.col("current_speed").isNotNull()
        & F.col("free_flow_speed").isNotNull()
        & (F.col("free_flow_speed") > 0)
    )

    # --- simple derived fields ------------------------------------------
    df = df.withColumn("speed_ratio", F.col("current_speed") / F.col("free_flow_speed"))
    df = df.withColumn(
        "delay",
        F.col("current_travel_time") - F.col("free_flow_travel_time"),
    )
    df = df.withColumn("hour", F.hour("collected_at"))
    # Spark's dayofweek: 1=Sunday ... 7=Saturday
    df = df.withColumn("weekday", F.dayofweek("collected_at"))
    df = df.withColumn(
        "peak_hour",
        F.when(
            F.col("hour").between(*MORNING_PEAK) | F.col("hour").between(*EVENING_PEAK),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )

    # --- window functions (per location, ordered by time) ---------------
    # rangeBetween needs a numeric ordering column, hence the unix-seconds cast.
    df = df.withColumn("_ts", F.col("collected_at").cast("long"))

    trailing_15min = (
        Window.partitionBy("location_id")
        .orderBy("_ts")
        .rangeBetween(-ROLLING_WINDOW_SECONDS, 0)
    )
    df = df.withColumn("rolling_average_speed", F.avg("current_speed").over(trailing_15min))

    prev_row = Window.partitionBy("location_id").orderBy("_ts")
    df = df.withColumn("_prev_speed_ratio", F.lag("speed_ratio", 1).over(prev_row))
    df = df.withColumn(
        "congestion_change_rate",
        F.when(
            F.col("_prev_speed_ratio").isNotNull(),
            F.col("speed_ratio") - F.col("_prev_speed_ratio"),
        ).otherwise(F.lit(0.0)),
    )

    return df.drop("_ts", "_prev_speed_ratio")


def main():
    spark = SparkSession.builder.appName("PuneTrafficFeatureEngineering").getOrCreate()

    # Spark auto-discovers the `date=YYYY-MM-DD` Hive-style partitions and
    # adds `date` as a column - same directory export_to_parquet.py has been
    # appending to since Phase 1, so no path changes needed as more days land.
    raw = spark.read.parquet(PARQUET_DIR)
    print(f"Read {raw.count()} raw snapshot rows from {PARQUET_DIR}")

    features = build_features(raw)

    (
        features.write.mode("overwrite")
        .partitionBy("location_id")
        .parquet(FEATURES_OUT_DIR)
    )
    print(f"Wrote {features.count()} feature rows to {FEATURES_OUT_DIR}")

    features.select(
        "location_id", "collected_at", "speed_ratio", "delay", "hour",
        "weekday", "peak_hour", "rolling_average_speed", "congestion_change_rate",
    ).orderBy("location_id", "collected_at").show(10, truncate=False)

    spark.stop()



if __name__ == "__main__":
    main()
