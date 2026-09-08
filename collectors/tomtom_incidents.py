"""
TomTom Traffic Incident API  -  aggregated count + max severity for a bbox
around each monitored point.
Docs: https://developer.tomtom.com/traffic-api/documentation/traffic-incidents/incident-details
"""

import requests
import logging

from config.settings import TOMTOM_API_KEY, INCIDENT_BBOX_DELTA

logger = logging.getLogger(__name__)

INCIDENTS_URL = "https://api.tomtom.com/traffic/services/5/incidentDetails"


def fetch_incidents(lat: float, lon: float) -> dict | None:
    """
    Returns a dict with incident_count, incident_severity_max, incident_severity_avg
    -  or None on failure. TomTom's "magnitudeOfDelay" ranges 0 (unknown) to 4 (major).

    NOTE: incident_severity_max is the max across everything in the bbox, so in a
    dense area with many incidents it pins at 4 almost every poll (there's nearly
    always at least one severity-4 incident somewhere in a ~2.2km box). Keep it
    around for "is anything severe happening", but use incident_severity_avg for
    anything that needs to actually vary meaningfully across locations/time.
    """
    d = INCIDENT_BBOX_DELTA
    bbox = f"{lon - d},{lat - d},{lon + d},{lat + d}"

    params = {
        "bbox": bbox,
        "fields": "{incidents{properties{magnitudeOfDelay,iconCategory}}}",
        "key": TOMTOM_API_KEY,
    }
    try:
        resp = requests.get(INCIDENTS_URL, params=params, timeout=10)
        resp.raise_for_status()
        incidents = resp.json().get("incidents", [])
        severities = [
            i.get("properties", {}).get("magnitudeOfDelay", 0) for i in incidents
        ]
        return {
            "incident_count": len(incidents),
            "incident_severity_max": max(severities) if severities else 0,
            "incident_severity_avg": (sum(severities) / len(severities)) if severities else 0.0,
            "_raw": incidents,
        }
    except requests.RequestException as e:
        logger.error("TomTom Incidents API failed for (%s, %s): %s", lat, lon, e)
        return None
