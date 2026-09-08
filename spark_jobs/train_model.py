"""
Phase 4 - Model Training (Spark MLlib RandomForest)

Reads the labeled dataset from Phase 3 (data/labeled/) and trains a
RandomForestClassifier to predict congestion_level (0=low, 1=moderate,
2=high) from the engineered features.

NOTE on current data: with only a few hours of history, each location has
mostly sat in ONE congestion regime (e.g. Shivajinagar = always "moderate"
so far). That means location_id is currently a very strong - almost too
strong - predictor on its own. This is fine to train and wire up Phase 5
with now, but treat accuracy numbers from this run as a "pipeline works"
checkpoint, not a final model quality number. Re-run this job periodically
as the collector accumulates more hours (ideally spanning peak + off-peak)
and watch whether accuracy holds up on genuinely varied data.

Run (batch):
    spark-submit spark_jobs/train_model.py
"""

import json
import os
from datetime import datetime, timezone

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

LABELED_DIR = os.environ.get("LABELED_OUT_DIR", "data/labeled")
MODEL_DIR = os.environ.get("MODEL_DIR", "data/models/congestion_rf")
# The dashboard's "Model Performance" page reads this file directly - keep
# it a plain JSON, no Spark/Postgres dependency needed to display it.
METRICS_PATH = os.environ.get("METRICS_PATH", os.path.join(MODEL_DIR, "metrics.json"))

LABELS = {0: "low", 1: "moderate", 2: "high"}

# Fraction of the time range (most recent) held out as the test set.
TEST_TIME_FRACTION = float(os.environ.get("TEST_TIME_FRACTION", "0.2"))

# Numeric features fed straight into the model. Kept as a single list so
# Phase 5's streaming job can import it and know exactly what the model
# expects, rather than guessing at column names.
NUMERIC_FEATURES = [
    # NOTE: speed_ratio and delay are intentionally NOT included here.
    # label_generation.py computes congestion_level as a fixed threshold
    # function of exactly speed_ratio and delay - including them as model
    # inputs would let the model just re-learn that threshold rule instead
    # of actually predicting congestion from independent signals, and
    # would show up as inflated (often ~100%) accuracy that doesn't
    # reflect real predictive power. rolling_average_speed and
    # congestion_change_rate are kept: they summarize the recent TREND
    # rather than being algebraically identical to the label.
    "hour",
    "weekday",
    "peak_hour",
    "rolling_average_speed",
    "congestion_change_rate",
    "incident_count",
    "incident_severity_max",
    "confidence",
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "rain_1h_mm",
    "visibility_m",
]
CATEGORICAL_FEATURE = "location_id"
LABEL_COL = "congestion_level"


def fill_defaults(df):
    """
    RandomForest's VectorAssembler errors on nulls, and weather/incident
    fields can be null on transient API failures. Impute conservative
    defaults rather than dropping rows - dropping would bias the training
    set toward "everything worked" polling cycles.
    """
    return df.fillna(
        {
            "incident_count": 0,
            "incident_severity_max": 0,
            "confidence": 1.0,
            "rain_1h_mm": 0.0,
            # Weather nulls: fill with the column's own average rather than
            # a hardcoded guess, so it's at least self-consistent per run.
        }
    )


def build_pipeline():
    indexer = StringIndexer(
        inputCol=CATEGORICAL_FEATURE, outputCol="location_index", handleInvalid="keep"
    )
    encoder = OneHotEncoder(inputCol="location_index", outputCol="location_ohe")

    assembler = VectorAssembler(
        inputCols=NUMERIC_FEATURES + ["location_ohe"],
        outputCol="features",
        handleInvalid="skip",  # belt-and-suspenders alongside fill_defaults
    )

    rf = RandomForestClassifier(
        labelCol=LABEL_COL,
        featuresCol="features",
        numTrees=100,
        maxDepth=8,
        seed=42,
    )

    return Pipeline(stages=[indexer, encoder, assembler, rf])


def main():
    spark = SparkSession.builder.appName("PuneTrafficModelTraining").getOrCreate()

    df = spark.read.parquet(LABELED_DIR)
    print(f"Read {df.count()} labeled rows from {LABELED_DIR}")

    for col in ["temperature_c", "humidity_pct", "wind_speed_ms", "visibility_m"]:
        mean_val = df.select(F.avg(col)).first()[0] or 0.0
        df = df.fillna({col: mean_val})
    df = fill_defaults(df)

    # --- time-based split -------------------------------------------------
    # row_number() over a global time ordering, then split by row count.
    # coalesce(1) first: at this data volume (thousands of rows) the shuffle
    # cost is irrelevant, and it keeps the ordering simple to reason about.
    total = df.count()
    split_at = int(total * (1 - TEST_TIME_FRACTION))

    df_indexed = df.coalesce(1).withColumn(
        "_row", F.row_number().over(Window.orderBy("collected_at"))
    )
    train_df = df_indexed.filter(F.col("_row") <= split_at).drop("_row")
    test_df = df_indexed.filter(F.col("_row") > split_at).drop("_row")

    print(f"Train rows: {train_df.count()}, Test rows: {test_df.count()} "
          f"(time-based split, last {int(TEST_TIME_FRACTION * 100)}% by time held out)")

    pipeline = build_pipeline()
    model = pipeline.fit(train_df)

    predictions = model.transform(test_df).cache()

    # --- classification metrics -------------------------------------------
    clf_evaluator = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction"
    )
    accuracy = clf_evaluator.evaluate(predictions, {clf_evaluator.metricName: "accuracy"})
    f1 = clf_evaluator.evaluate(predictions, {clf_evaluator.metricName: "f1"})
    precision = clf_evaluator.evaluate(
        predictions, {clf_evaluator.metricName: "weightedPrecision"}
    )
    recall = clf_evaluator.evaluate(
        predictions, {clf_evaluator.metricName: "weightedRecall"}
    )

    # --- MAE / RMSE ----------------------------------------------------------
    # congestion_level (0=low/1=moderate/2=high) is ordinal, so treating the
    # predicted class index as a numeric estimate and running a regression
    # evaluator against it gives a meaningful "how far off" number on top of
    # the classification metrics above (e.g. predicting "moderate" for a
    # "high" row is a smaller error than predicting "low").
    reg_evaluator = RegressionEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction"
    )
    mae = reg_evaluator.evaluate(predictions, {reg_evaluator.metricName: "mae"})
    rmse = reg_evaluator.evaluate(predictions, {reg_evaluator.metricName: "rmse"})

    print(f"Test accuracy:  {accuracy:.4f}")
    print(f"Precision (wt): {precision:.4f}")
    print(f"Recall (wt):    {recall:.4f}")
    print(f"F1 (wt):        {f1:.4f}")
    print(f"MAE:            {mae:.4f}")
    print(f"RMSE:           {rmse:.4f}")

    # --- confusion matrix ----------------------------------------------------
    print("\nConfusion matrix (rows=actual congestion_level, cols=predicted):")

    # groupBy(LABEL_COL) only yields rows for actual values seen in this test
    # split. With limited history a class can be entirely absent from the
    # time-based test window (see NOTE above) - if we don't backfill it, the
    # matrix comes back with fewer rows than len(LABELS), which mismatches
    # metrics["labels"] and breaks any fixed-size (labels x labels) rendering
    # downstream (e.g. the dashboard's px.imshow). Cross-join against every
    # known label first so every row (and column) is always present, even if
    # all-zero.
    all_labels_df = spark.createDataFrame(
        [(k,) for k in LABELS.keys()], [LABEL_COL]
    )
    cm_rows = (
        all_labels_df.join(
            predictions.groupBy(LABEL_COL).pivot("prediction", list(LABELS.keys())).count(),
            on=LABEL_COL,
            how="left",
        )
        .orderBy(LABEL_COL)
        .fillna(0)
        .collect()
    )
    confusion_matrix = [
        [int(row[str(pred)] or 0) for pred in LABELS.keys()] for row in cm_rows
    ]
    for row in cm_rows:
        print(row)

    # # --- confusion matrix ----------------------------------------------------
    # print("\nConfusion matrix (rows=actual congestion_level, cols=predicted):")
    # cm_rows = (
    #     predictions.groupBy(LABEL_COL)
    #     .pivot("prediction", list(LABELS.keys()))
    #     .count()
    #     .orderBy(LABEL_COL)
    #     .fillna(0)
    #     .collect()
    # )
    # confusion_matrix = [
    #     [int(row[str(pred)] or 0) for pred in LABELS.keys()] for row in cm_rows
    # ]
    # for row in cm_rows:
    #     print(row)

    # --- feature importance ---------------------------------------------------
    rf_model = model.stages[-1]
    location_labels = model.stages[0].labels  # StringIndexer's fitted vocab
    importances = rf_model.featureImportances.toArray().tolist()
    # OneHotEncoder drops the last category by default, so the one-hot block
    # is (num_locations - 1) wide - build names to match the actual vector
    # length rather than assuming.
    n_ohe = len(importances) - len(NUMERIC_FEATURES)
    ohe_names = [f"location={loc}" for loc in location_labels[:n_ohe]]
    all_feature_names = NUMERIC_FEATURES + ohe_names
    feature_importance = sorted(
        (
            {"feature": name, "importance": round(float(score), 6)}
            for name, score in zip(all_feature_names, importances)
        ),
        key=lambda x: x["importance"],
        reverse=True,
    )
    print("\nTop features by importance:")
    for fi in feature_importance[:10]:
        print(f"  {fi['feature']:<30} {fi['importance']:.4f}")

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": train_df.count(),
        "test_rows": test_df.count(),
        "accuracy": round(accuracy, 4),
        "precision_weighted": round(precision, 4),
        "recall_weighted": round(recall, 4),
        "f1_weighted": round(f1, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "labels": list(LABELS.values()),
        "confusion_matrix": confusion_matrix,
        "feature_importance": feature_importance,
        "model_dir": MODEL_DIR,
        "num_trees": rf_model.getNumTrees,
        "max_depth": rf_model.getOrDefault("maxDepth"),
    }

    model.write().overwrite().save(MODEL_DIR)
    print(f"Saved trained PipelineModel to {MODEL_DIR}")

    # IMPORTANT: metrics.json must be written AFTER model.write().overwrite()
    # above - .overwrite().save() wipes and recreates MODEL_DIR, so writing
    # metrics.json before it (as an earlier version of this script did)
    # silently deletes the metrics file the moment the model save runs.
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
