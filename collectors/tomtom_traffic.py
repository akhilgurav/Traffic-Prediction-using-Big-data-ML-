"""
TomTom Traffic Flow API  -  Segment Data.
Docs: https://developer.tomtom.com/traffic-api/documentation/traffic-flow/flow-segment-data
"""

import requests
import logging

from config.settings import TOMTOM_API_KEY

logger = logging.getLogger(__name__)

FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"


def fetch_flow(lat: float, lon: float) -> dict | None:
    """
    Returns a dict with current_speed, free_flow_speed, current_travel_time,
    free_flow_travel_time, confidence, road_closure  -  or None on failure.
    """
    params = {
        "point": f"{lat},{lon}",
        "key": TOMTOM_API_KEY,
        "unit": "KMPH",
    }
    try:
        resp = requests.get(FLOW_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("flowSegmentData", {})

        # The API also returns the actual polyline of the road segment this
        # reading applies to (data.coordinates.coordinate = [{latitude, longitude}, ...]).
        # We used to throw this away - it's what lets the dashboard draw real
        # roads instead of a single point per monitored location.
        coords = data.get("coordinates", {}).get("coordinate", [])
        segment_geometry = [
            [c["latitude"], c["longitude"]] for c in coords if "latitude" in c and "longitude" in c
        ] or None

        return {
            "current_speed": data.get("currentSpeed"),
            "free_flow_speed": data.get("freeFlowSpeed"),
            "current_travel_time": data.get("currentTravelTime"),
            "free_flow_travel_time": data.get("freeFlowTravelTime"),
            "confidence": data.get("confidence"),
            "road_closure": data.get("roadClosure", False),
            "segment_geometry": segment_geometry,
            "_raw": data,
        }
    except requests.RequestException as e:
        logger.error("TomTom Flow API failed for (%s, %s): %s", lat, lon, e)
        return None
