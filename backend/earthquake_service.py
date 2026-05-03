from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from backend.config import EARTHQUAKE_DATA, EARTHQUAKE_MODEL


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

    enriched = df[["time", "latitude", "longitude", "depth", "mag"]].copy()
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
    data = _load_sequence_data()
    sequence = data.tail(50).copy()

    sequence.iloc[-1, sequence.columns.get_loc("latitude")] = _scale(latitude, 25.0, 40.0)
    sequence.iloc[-1, sequence.columns.get_loc("longitude")] = _scale(longitude, 70.0, 95.0)
    sequence.iloc[-1, sequence.columns.get_loc("depth")] = _scale(depth, 0.0, 700.0)

    x = sequence.to_numpy(dtype=np.float32)[None, :, :]
    model_status = "loaded"
    try:
        model = _load_model()
        predicted_scaled = float(model.predict(x, verbose=0)[0][0])
    except Exception as exc:
        model_status = f"fallback: {exc}"
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
        "data_points": int(len(data)),
        "seismic_zone": seismic_zone,
        "note": "The LSTM estimates the next magnitude pattern from the latest 50 historical events; it is not an official earthquake warning system.",
    }


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
