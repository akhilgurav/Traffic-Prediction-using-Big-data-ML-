"""
Phase 3 - Label Generation (PySpark)

Reads the feature-engineered dataset from Phase 2 (data/features/, partitioned
by location_id) and adds a congestion label derived from speed_ratio (primary
signal) with delay as a secondary tiebreaker for borderline cases.

IMPORTANT: this is a BATCH-ONLY step. Labels are ground truth used to train
the Phase 4 model - at Phase 5 inference time we PREDICT the label, we don't
compute it. So, unlike build_features() in feature_engineering.py, add_labels()
is never imported by the streaming job. It's kept as a pure DataFrame ->
DataFrame transform anyway, purely so it's easy to unit-test and re-run as
more historical data accumulates.

Run (batch):
    spark-submit spark_jobs/label_generation.py
"""

import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

FEATURES_DIR = os.environ.get("FEATURES_OUT_DIR", "data/features")
LABELED_OUT_DIR = os.environ.get("LABELED_OUT_DIR", "data/labeled")

# --- Thresholds --------------------------------------------------------
# speed_ratio = current_speed / free_flow_speed, so 1.0 = free-flowing,
# closer to 0 = jammed. These are starting points - re-tune once you've
# looked at the actual class distribution printed at the end of this job.
FREE_FLOW_MIN_RATIO = float(os.environ.get("FREE_FLOW_MIN_RATIO", "0.7"))
MODERATE_MIN_RATIO = float(os.environ.get("MODERATE_MIN_RATIO", "0.4"))

# Secondary signal: if speed_ratio sits right at a boundary, a large delay
# (seconds lost vs free-flow travel time) still counts as worse congestion.
# Set to a very large number to effectively disable this tiebreaker.
DELAY_ESCALATION_SECONDS = float(os.environ.get("DELAY_ESCALATION_SECONDS", "180"))

LABELS = {0: "low", 1: "moderate", 2: "high"}


def add_labels(df):
    """
    Pure transform: feature DataFrame (must have speed_ratio, delay) ->
    same DataFrame + congestion_level (int 0/1/2) + congestion_label (string).
    """
    level = (
        F.when(F.col("speed_ratio") >= FREE_FLOW_MIN_RATIO, F.lit(0))
        .when(F.col("speed_ratio") >= MODERATE_MIN_RATIO, F.lit(1))
        .otherwise(F.lit(2))
    )

    # Escalate by one level (capped at 2) if delay is unusually high, even
    # when speed_ratio alone would have called it "low"/"moderate" - e.g. a
    # short road segment can have a decent speed_ratio but still cost you
    # several minutes versus free-flow.
    level = F.when(
        (F.col("delay") >= DELAY_ESCALATION_SECONDS) & (level < 2),
        level + 1,
    ).otherwise(level)

    df = df.withColumn("congestion_level", level.cast("int"))

    mapping = F.create_map([F.lit(x) for pair in LABELS.items() for x in pair])
    df = df.withColumn("congestion_label", mapping[F.col("congestion_level")])

    return df


def main():
    spark = SparkSession.builder.appName("PuneTrafficLabelGeneration").getOrCreate()

    features = spark.read.parquet(FEATURES_DIR)
    print(f"Read {features.count()} feature rows from {FEATURES_DIR}")

    labeled = add_labels(features)

    (
        labeled.write.mode("overwrite")
        .partitionBy("location_id")
        .parquet(LABELED_OUT_DIR)
    )
    print(f"Wrote {labeled.count()} labeled rows to {LABELED_OUT_DIR}")

    print("\nClass distribution (watch for a class with very few rows - ")
    print("that will bite you when training the RandomForest in Phase 4):")
    labeled.groupBy("congestion_level", "congestion_label").count().orderBy(
        "congestion_level"
    ).show()

    print("Per-location breakdown:")
    labeled.groupBy("location_id", "congestion_label").count().orderBy(
        "location_id", "congestion_label"
    ).show(50, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
