"""
Phase 5a - Kafka producer for the live collector.

Publishing here is OPTIONAL and ADDITIVE. The collector already writes to
Postgres + raw JSON directly (Phase 1) and that path is completely
unaffected by anything in this module. Kafka is only used to feed Phase
5b's Spark Structured Streaming job.

Design choice: if Kafka is unreachable (e.g. you haven't started the
`kafka` compose profile) or KAFKA_ENABLED=false, publish_record() logs a
warning once and silently no-ops from then on - it never raises into the
collector's poll loop. Your collector has been running unattended for
hours; nothing here should be able to take it down.
"""

import json
import logging
import threading

from config.settings import KAFKA_BOOTSTRAP_SERVERS, KAFKA_ENABLED, KAFKA_TOPIC

logger = logging.getLogger(__name__)

_producer = None
_lock = threading.Lock()
# Once a connection attempt fails, stay disabled for the rest of this
# process's life rather than retrying every poll cycle. Restart the
# collector (or fix Kafka and restart) to try again.
_disabled_after_failure = not KAFKA_ENABLED


def _get_producer():
    global _producer, _disabled_after_failure
    if _disabled_after_failure:
        return None
    if _producer is not None:
        return _producer
    with _lock:
        if _producer is not None:
            return _producer
        try:
            from kafka import KafkaProducer

            _producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                request_timeout_ms=5000,
                # Skip broker version auto-probing entirely (that's what
                # api_version_auto_timeout_ms used to tune) - just pin a
                # version. Confluent's Kafka images speak a protocol
                # compatible with this for our purposes (simple produce).
                api_version=(2, 5, 0),
            )
            logger.info(
                "Kafka producer connected to %s (topic=%s)",
                KAFKA_BOOTSTRAP_SERVERS,
                KAFKA_TOPIC,
            )
        except Exception as e:
            logger.warning(
                "Kafka producer unavailable (%s) - continuing without streaming "
                "publish for this run. Postgres/raw JSON writes are unaffected.",
                e,
            )
            _disabled_after_failure = True
            return None
    return _producer


def publish_record(record: dict):
    """Publish one unified traffic record to Kafka. Never raises."""
    producer = _get_producer()
    if producer is None:
        return
    try:
        payload = {k: v for k, v in record.items() if k != "_raw"}
        producer.send(KAFKA_TOPIC, key=record.get("location_id"), value=payload)
    except Exception as e:
        logger.warning(
            "Failed to publish record for %s to Kafka: %s", record.get("location_id"), e
        )
