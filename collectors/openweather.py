"""
OpenWeather Current Weather API.
Docs: https://openweathermap.org/current
"""

import requests
import logging

from config.settings import OPENWEATHER_API_KEY

logger = logging.getLogger(__name__)

WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"


def fetch_weather(lat: float, lon: float) -> dict | None:
    """
    Returns a dict with temperature_c, feels_like_c, humidity_pct,
    wind_speed_ms, rain_1h_mm, weather_main, weather_description,
    visibility_m  -  or None on failure.
    """
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }
    try:
        resp = requests.get(WEATHER_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        main = data.get("main", {})
        weather = (data.get("weather") or [{}])[0]
        rain = data.get("rain", {})
        return {
            "temperature_c": main.get("temp"),
            "feels_like_c": main.get("feels_like"),
            "humidity_pct": main.get("humidity"),
            "wind_speed_ms": data.get("wind", {}).get("speed"),
            "rain_1h_mm": rain.get("1h", 0.0),
            "weather_main": weather.get("main"),
            "weather_description": weather.get("description"),
            "visibility_m": data.get("visibility"),
            "_raw": data,
        }
    except requests.RequestException as e:
        logger.error("OpenWeather API failed for (%s, %s): %s", lat, lon, e)
        return None
