from __future__ import annotations

from backend.earthquake_service import predict_earthquake
from backend.flood_service import flood_risk_for_location


FLOOD_CITIES = [
    {"name": "Guwahati", "country": "India", "latitude": 26.1445, "longitude": 91.7362},
    {"name": "Patna", "country": "India", "latitude": 25.5941, "longitude": 85.1376},
    {"name": "Kolkata", "country": "India", "latitude": 22.5726, "longitude": 88.3639},
    {"name": "Kochi", "country": "India", "latitude": 9.9312, "longitude": 76.2673},
    {"name": "Mumbai", "country": "India", "latitude": 19.076, "longitude": 72.8777},
    {"name": "Bhubaneswar", "country": "India", "latitude": 20.2961, "longitude": 85.8245},
    {"name": "Dhaka", "country": "Bangladesh", "latitude": 23.8103, "longitude": 90.4125},
    {"name": "Jakarta", "country": "Indonesia", "latitude": -6.2088, "longitude": 106.8456},
]

EARTHQUAKE_CITIES = [
    {"name": "Srinagar", "country": "India", "latitude": 34.0837, "longitude": 74.7973, "depth": 15},
    {"name": "Dehradun", "country": "India", "latitude": 30.3165, "longitude": 78.0322, "depth": 12},
    {"name": "Gangtok", "country": "India", "latitude": 27.3314, "longitude": 88.6138, "depth": 18},
    {"name": "Guwahati", "country": "India", "latitude": 26.1445, "longitude": 91.7362, "depth": 18},
    {"name": "Imphal", "country": "India", "latitude": 24.817, "longitude": 93.9368, "depth": 25},
    {"name": "Kathmandu", "country": "Nepal", "latitude": 27.7172, "longitude": 85.324, "depth": 12},
    {"name": "Tokyo", "country": "Japan", "latitude": 35.6762, "longitude": 139.6503, "depth": 30},
    {"name": "Istanbul", "country": "Turkey", "latitude": 41.0082, "longitude": 28.9784, "depth": 15},
]


def get_hotspots() -> dict[str, list[dict[str, object]]]:
    flood = []
    for city in FLOOD_CITIES:
        result = flood_risk_for_location(city["latitude"], city["longitude"])
        flood.append(_city_payload(city, "flood", result))

    earthquake = []
    for city in EARTHQUAKE_CITIES:
        result = predict_earthquake(city["latitude"], city["longitude"], city["depth"])
        earthquake.append(_city_payload(city, "earthquake", result))

    flood.sort(key=lambda item: item["risk_score"], reverse=True)
    earthquake.sort(key=lambda item: item["risk_score"], reverse=True)
    return {"flood": flood, "earthquake": earthquake}


def _city_payload(city: dict[str, object], disaster: str, result: dict[str, object]) -> dict[str, object]:
    return {
        "name": city["name"],
        "country": city["country"],
        "disaster": disaster,
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "depth": city.get("depth", 18),
        "risk_level": result["risk_level"],
        "risk_score": result["risk_score"],
        "risk_percent": result["risk_percent"],
        "metric": result.get("predicted_magnitude", result.get("risk_percent")),
    }
