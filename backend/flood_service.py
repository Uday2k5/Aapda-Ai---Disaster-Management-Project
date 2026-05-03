from __future__ import annotations

import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from backend.config import FLOOD_CHECKPOINT, FLOOD_DATA_ROOT


INDIA_BOUNDS = {
    "min_lat": 6.0,
    "max_lat": 38.8,
    "min_lon": 68.0,
    "max_lon": 97.5,
}

FLOOD_PRONE_REGIONS = [
    ("Assam / Brahmaputra basin", 26.2, 92.8, 0.56),
    ("Bihar / Ganga basin", 25.7, 85.3, 0.52),
    ("West Bengal delta", 22.6, 88.4, 0.5),
    ("Kerala coast", 10.3, 76.4, 0.46),
    ("Mumbai / Konkan coast", 19.1, 72.9, 0.42),
    ("Odisha coast", 20.3, 85.8, 0.42),
    ("Uttarakhand foothills", 30.1, 79.0, 0.38),
    ("Bangladesh delta", 23.8, 90.4, 0.58),
    ("Jakarta coastal basin", -6.2, 106.8, 0.5),
]


@dataclass
class FloodModelStatus:
    checkpoint_exists: bool
    checkpoint_path: str
    trained_best_iou: float | None
    data_root_exists: bool


def get_model_status() -> FloodModelStatus:
    best_iou = None
    if FLOOD_CHECKPOINT.exists():
        try:
            checkpoint = torch.load(FLOOD_CHECKPOINT, map_location="cpu")
            best_iou = checkpoint.get("best_iou")
        except Exception:
            best_iou = None

    return FloodModelStatus(
        checkpoint_exists=FLOOD_CHECKPOINT.exists(),
        checkpoint_path=str(FLOOD_CHECKPOINT),
        trained_best_iou=float(best_iou) if best_iou is not None else None,
        data_root_exists=FLOOD_DATA_ROOT.exists(),
    )


def flood_risk_for_location(latitude: float, longitude: float) -> dict[str, Any]:
    weather = _fetch_weather(latitude, longitude)
    regional_factor, nearest_region = _regional_flood_factor(latitude, longitude)

    rain_now = float(weather.get("rain_now_mm", 0.0))
    rain_24h = float(weather.get("rain_24h_mm", 0.0))
    rain_72h = float(weather.get("rain_72h_mm", 0.0))

    rainfall_score = min(0.62, rain_now / 35 * 0.18 + rain_24h / 120 * 0.26 + rain_72h / 240 * 0.18)
    in_india = _inside_india(latitude, longitude)
    location_penalty = 0.0 if in_india else -0.18
    risk_score = _clamp(0.08 + rainfall_score + regional_factor + location_penalty, 0.02, 0.98)

    if risk_score >= 0.55:
        level = "High"
        advice = "Avoid low-lying routes, monitor local authority alerts, and keep an evacuation path ready."
    elif risk_score >= 0.35:
        level = "Moderate"
        advice = "Stay alert for waterlogging, check nearby river/drain updates, and avoid unnecessary travel."
    else:
        level = "Low"
        advice = "No major flood signal from the available live rainfall indicators."

    return {
        "latitude": latitude,
        "longitude": longitude,
        "inside_india": in_india,
        "risk_score": round(risk_score, 3),
        "risk_percent": round(risk_score * 100, 1),
        "risk_level": level,
        "nearest_sensitive_region": nearest_region,
        "weather": weather,
        "advice": advice,
        "model_status": get_model_status().__dict__,
        "method": "Live rainfall + regional flood-proneness. The trained Sentinel-1 model is available for satellite image masks when a region TIFF is supplied.",
    }


def _fetch_weather(latitude: float, longitude: float) -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "precipitation,rain,showers,weather_code",
            "hourly": "precipitation",
            "forecast_days": 3,
            "timezone": "auto",
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{params}"

    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            payload = response.read().decode("utf-8")
        import json

        data = json.loads(payload)
        hourly = data.get("hourly", {}).get("precipitation", []) or []
        current = data.get("current", {}) or {}
        rain_24h = sum(float(v or 0) for v in hourly[:24])
        rain_72h = sum(float(v or 0) for v in hourly[:72])
        return {
            "source": "Open-Meteo",
            "live": True,
            "rain_now_mm": float(current.get("rain") or current.get("precipitation") or 0),
            "rain_24h_mm": round(rain_24h, 2),
            "rain_72h_mm": round(rain_72h, 2),
            "weather_code": current.get("weather_code"),
        }
    except Exception as exc:
        return {
            "source": "offline fallback",
            "live": False,
            "rain_now_mm": 0.0,
            "rain_24h_mm": 0.0,
            "rain_72h_mm": 0.0,
            "error": str(exc),
        }


def _regional_flood_factor(latitude: float, longitude: float) -> tuple[float, str]:
    best_name = "General India region"
    best_factor = 0.04
    for name, lat, lon, weight in FLOOD_PRONE_REGIONS:
        distance = _haversine_km(latitude, longitude, lat, lon)
        influence = weight * math.exp(-distance / 330)
        if influence > best_factor:
            best_factor = influence
            best_name = name
    return round(best_factor, 3), best_name


def _inside_india(latitude: float, longitude: float) -> bool:
    return (
        INDIA_BOUNDS["min_lat"] <= latitude <= INDIA_BOUNDS["max_lat"]
        and INDIA_BOUNDS["min_lon"] <= longitude <= INDIA_BOUNDS["max_lon"]
    )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
