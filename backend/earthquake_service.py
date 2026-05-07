from __future__ import annotations

import json
from functools import lru_cache
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from backend.config import EARTHQUAKE_DATA, EARTHQUAKE_MODEL, USE_EARTHQUAKE_MODEL


FEATURES = [
    "time",
    "latitude",
    "longitude",
    "depth",
    "mag",
    "days_since",
    "roll_mean",
    "roll_max",
    "mag_gradient",
]

USGS_EVENT_QUERY = "https://earthquake.usgs.gov/fdsnws/event/1/query"
LIVE_WINDOW_DAYS = 50
LIVE_RADIUS_CANDIDATES_KM = (300, 600, 1000, 1500)
MIN_LIVE_EVENTS = 50


@lru_cache(maxsize=1)
def _load_model():
    from tensorflow.keras.models import load_model

    return load_model(EARTHQUAKE_MODEL, compile=False)


@lru_cache(maxsize=1)
def _load_sequence_data() -> pd.DataFrame:
    df = pd.read_csv(EARTHQUAKE_DATA)
    for column in ["time", "latitude", "longitude", "depth", "mag"]:
        if column not in df.columns:
            raise ValueError(f"Missing earthquake data column: {column}")

    return _prepare_sequence_frame(df[["time", "latitude", "longitude", "depth", "mag"]].copy())


def _prepare_sequence_frame(base: pd.DataFrame) -> pd.DataFrame:
    enriched = base.copy()
    enriched["days_since"] = enriched["time"].diff().fillna(0).clip(lower=0)
    enriched["roll_mean"] = enriched["mag"].rolling(window=5).mean().fillna(enriched["mag"].mean())
    enriched["roll_max"] = enriched["mag"].rolling(window=5).max().fillna(enriched["mag"].max())
    enriched["mag_gradient"] = enriched["mag"].diff().fillna(0)

    for column in FEATURES:
        col = enriched[column].astype(float)
        lo, hi = col.min(), col.max()
        if hi > lo:
            enriched[column] = (col - lo) / (hi - lo)
        else:
            enriched[column] = 0.0
    return enriched[FEATURES]


def predict_earthquake(latitude: float, longitude: float, depth: float) -> dict[str, Any]:
    sequence, source = _sequence_for_request(latitude, longitude)

    sequence.iloc[-1, sequence.columns.get_loc("latitude")] = _scale(latitude, 25.0, 40.0)
    sequence.iloc[-1, sequence.columns.get_loc("longitude")] = _scale(longitude, 70.0, 95.0)
    sequence.iloc[-1, sequence.columns.get_loc("depth")] = _scale(depth, 0.0, 700.0)

    x = sequence.to_numpy(dtype=np.float32)[None, :, :]
    model_status = "fallback: TensorFlow disabled for this deployment"
    if USE_EARTHQUAKE_MODEL:
        try:
            model = _load_model()
            predicted_scaled = float(model.predict(x, verbose=0)[0][0])
            model_status = "loaded"
        except Exception as exc:
            model_status = f"fallback: {exc}"
            predicted_scaled = float(sequence["mag"].tail(10).mean())
    else:
        predicted_scaled = float(sequence["mag"].tail(10).mean())

    magnitude = _inverse_magnitude(predicted_scaled)
    seismic_prior, seismic_zone = _seismic_prior(latitude, longitude)
    risk_score = min(0.98, _risk_from_magnitude(magnitude, depth) + seismic_prior)
    level = _risk_level(risk_score)

    return {
        "latitude": latitude,
        "longitude": longitude,
        "depth_km": depth,
        "predicted_magnitude": round(magnitude, 2),
        "risk_score": round(risk_score, 3),
        "risk_percent": round(risk_score * 100, 1),
        "risk_level": level,
        "model_status": model_status,
        "model_path": str(EARTHQUAKE_MODEL),
        "data_points": int(len(sequence)),
        "data_source": source,
        "seismic_zone": seismic_zone,
        "note": "The LSTM estimates the next magnitude pattern from the latest 50 events used in the sequence; it is not an official earthquake warning system.",
    }


def _sequence_for_request(latitude: float, longitude: float) -> tuple[pd.DataFrame, str]:
    live = _load_live_sequence(latitude, longitude)
    if live is not None:
        return live, f"USGS live regional feed ({LIVE_WINDOW_DAYS}-day window)"
    local = _load_sequence_data().tail(50).copy()
    return local, "Local historical earthquake dataset"


def _load_live_sequence(latitude: float, longitude: float) -> pd.DataFrame | None:
    for radius_km in LIVE_RADIUS_CANDIDATES_KM:
        raw = _fetch_usgs_events(latitude, longitude, radius_km)
        if raw is None or len(raw) < MIN_LIVE_EVENTS:
            continue
        prepared = _prepare_sequence_frame(raw)
        return prepared.tail(50).copy()
    return None


def _fetch_usgs_events(latitude: float, longitude: float, radius_km: int) -> pd.DataFrame | None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LIVE_WINDOW_DAYS)
    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%d"),
        "endtime": end.strftime("%Y-%m-%d"),
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "maxradiuskm": radius_km,
        "orderby": "time",
        "limit": 250,
    }
    url = f"{USGS_EVENT_QUERY}?{urlencode(params)}"
    try:
        with urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    rows: list[dict[str, float]] = []
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if len(coords) < 3:
            continue
        timestamp_ms = props.get("time")
        magnitude = props.get("mag")
        if timestamp_ms is None or magnitude is None:
            continue
        rows.append(
            {
                "time": float(timestamp_ms) / 86_400_000.0,
                "latitude": float(coords[1]),
                "longitude": float(coords[0]),
                "depth": float(coords[2]),
                "mag": float(magnitude),
            }
        )

    if len(rows) < MIN_LIVE_EVENTS:
        return None

    frame = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    # Convert time to elapsed days within the fetched live window so the sequence stays compact.
    frame["time"] = frame["time"] - float(frame["time"].iloc[0])
    return frame


def _scale(value: float, low: float, high: float) -> float:
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _inverse_magnitude(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return 2.5 + value * (7.7 - 2.5)


def _risk_from_magnitude(magnitude: float, depth: float) -> float:
    mag_component = max(0.0, min(1.0, (magnitude - 3.5) / 3.0))
    depth_component = max(0.0, min(1.0, (80.0 - min(depth, 80.0)) / 80.0))
    return max(0.03, min(0.98, mag_component * 0.72 + depth_component * 0.18))


def _risk_level(score: float) -> str:
    if score >= 0.62:
        return "High"
    if score >= 0.38:
        return "Moderate"
    return "Low"


def _seismic_prior(latitude: float, longitude: float) -> tuple[float, str]:
    zones = [
        ("Himalayan seismic belt", 31.5, 78.5, 0.38),
        ("North East India seismic belt", 26.0, 92.5, 0.36),
        ("Nepal Himalayan belt", 27.7, 85.3, 0.42),
        ("Japan subduction belt", 35.7, 139.7, 0.44),
        ("North Anatolian fault zone", 41.0, 29.0, 0.4),
    ]
    best_name = "General seismic background"
    best_prior = 0.0
    for name, lat, lon, weight in zones:
        distance = ((latitude - lat) ** 2 + (longitude - lon) ** 2) ** 0.5
        influence = weight * np.exp(-distance / 5.5)
        if influence > best_prior:
            best_prior = float(influence)
            best_name = name
    return round(best_prior, 3), best_name
