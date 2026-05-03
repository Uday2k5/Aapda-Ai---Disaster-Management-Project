from __future__ import annotations

import csv
import io
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from backend.config import (
    FIRMS_API_KEY,
    FIRMS_DATASET,
    WILDFIRE_DATA,
    WILDFIRE_FEATURES,
    WILDFIRE_METADATA,
    WILDFIRE_MODEL,
    WILDFIRE_REPORT,
)


WILDFIRE_REGIONS = [
    ("Uttarakhand forest belt", 30.3165, 78.0322, 0.22),
    ("Himachal hill forests", 31.1048, 77.1734, 0.21),
    ("Jammu and Kashmir forest belt", 34.0837, 74.7973, 0.18),
    ("Odisha dry forest zone", 20.9517, 85.0985, 0.16),
    ("Chhattisgarh forest corridor", 21.2787, 81.8661, 0.15),
    ("Madhya Pradesh dry deciduous belt", 23.2599, 77.4126, 0.14),
    ("California chaparral belt", 34.0522, -118.2437, 0.24),
    ("New South Wales bushfire belt", -33.8688, 151.2093, 0.22),
]


@lru_cache(maxsize=1)
def _load_model():
    import joblib

    return joblib.load(WILDFIRE_MODEL)


@lru_cache(maxsize=1)
def _load_feature_names() -> list[str]:
    if not WILDFIRE_FEATURES.exists():
        return ["lat_bin", "lon_bin", "frp", "fire_last_1d", "fire_last_3d", "fire_last_7d"]
    payload = json.loads(WILDFIRE_FEATURES.read_text())
    return list(payload.get("features", []))


@lru_cache(maxsize=1)
def _load_metadata() -> dict[str, Any]:
    if not WILDFIRE_METADATA.exists():
        return {}
    return json.loads(WILDFIRE_METADATA.read_text())


@lru_cache(maxsize=1)
def _load_model_report() -> str:
    if not WILDFIRE_REPORT.exists():
        return ""
    return WILDFIRE_REPORT.read_text().strip()


@lru_cache(maxsize=1)
def _load_history() -> pd.DataFrame:
    df = pd.read_csv(WILDFIRE_DATA, usecols=["latitude", "longitude", "acq_date", "frp"])
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["lat_bin"] = df["latitude"].round(1)
    df["lon_bin"] = df["longitude"].round(1)
    fires = (
        df.groupby(["lat_bin", "lon_bin", "acq_date"], as_index=False)
        .agg({"frp": "mean"})
        .sort_values(["lat_bin", "lon_bin", "acq_date"])
        .reset_index(drop=True)
    )
    fires["fire_today"] = 1
    return fires


@lru_cache(maxsize=1)
def _all_fire_dates() -> pd.DatetimeIndex:
    history = _load_history()
    return pd.date_range(history["acq_date"].min(), history["acq_date"].max(), freq="D")


@lru_cache(maxsize=1)
def _cell_statistics() -> tuple[pd.DataFrame, pd.DataFrame]:
    fires = _load_history()
    monthly = (
        fires.assign(month=fires["acq_date"].dt.month)
        .groupby(["lat_bin", "lon_bin", "month"], as_index=False)
        .agg(fire_days=("fire_today", "sum"), mean_frp=("frp", "mean"))
    )
    cell_totals = (
        fires.groupby(["lat_bin", "lon_bin"], as_index=False)
        .agg(total_fire_days=("fire_today", "sum"), mean_frp=("frp", "mean"), last_seen=("acq_date", "max"))
    )
    return monthly, cell_totals


def predict_wildfire(latitude: float, longitude: float) -> dict[str, Any]:
    lat_bin = round(latitude, 1)
    lon_bin = round(longitude, 1)
    monthly, cell_totals = _cell_statistics()
    current_month = datetime.now().month
    live_features = _fetch_firms_live_features(latitude, longitude, lat_bin, lon_bin)
    cell_history = _cell_time_series(lat_bin, lon_bin)
    month_history = monthly[(monthly["lat_bin"] == lat_bin) & (monthly["lon_bin"] == lon_bin) & (monthly["month"] == current_month)]
    cell_info = cell_totals[(cell_totals["lat_bin"] == lat_bin) & (cell_totals["lon_bin"] == lon_bin)]

    latest = cell_history.iloc[-1] if not cell_history.empty else None
    fire_last_1d = float(live_features["fire_last_1d"]) if live_features["live"] else float(latest["fire_last_1d"]) if latest is not None else 0.0
    fire_last_3d = float(live_features["fire_last_3d"]) if live_features["live"] else float(latest["fire_last_3d"]) if latest is not None else 0.0
    fire_last_7d = float(live_features["fire_last_7d"]) if live_features["live"] else float(latest["fire_last_7d"]) if latest is not None else 0.0
    frp = float(live_features["frp"]) if live_features["live"] else float(latest["frp"]) if latest is not None else 0.0
    monthly_rate = (
        float(min(1.0, month_history["fire_days"].sum() / max(1, len(cell_history[cell_history["acq_date"].dt.month == current_month]))))
        if not month_history.empty and not cell_history.empty
        else 0.0
    )
    region_prior, region_name = _wildfire_prior(latitude, longitude)

    model_status = "loaded"
    model_probability = None
    try:
        feature_names = _load_feature_names()
        feature_values = {
            "lat_bin": lat_bin,
            "lon_bin": lon_bin,
            "frp": frp,
            "fire_last_1d": fire_last_1d,
            "fire_last_3d": fire_last_3d,
            "fire_last_7d": fire_last_7d,
        }
        x = pd.DataFrame([[feature_values[name] for name in feature_names]], columns=feature_names)
        model = _load_model()
        model_probability = float(model.predict_proba(x)[0][1])
    except Exception as exc:
        model_status = f"fallback: {exc}"

    history_score = min(0.55, frp / 70.0 * 0.24 + fire_last_1d * 0.18 + fire_last_3d / 3.0 * 0.18 + fire_last_7d / 7.0 * 0.16 + monthly_rate * 0.16)
    fallback_probability = _clamp(0.06 + history_score + region_prior, 0.02, 0.98)
    recency_signal = min(1.0, fire_last_1d + fire_last_3d / 3.0 + fire_last_7d / 7.0 + monthly_rate)
    model_weight = 0.35 + 0.3 * recency_signal
    risk_score = _clamp((model_probability * model_weight if model_probability is not None else 0.0) + fallback_probability * (1.0 - model_weight if model_probability is not None else 1.0), 0.02, 0.98)
    risk_level = _risk_level(risk_score)

    recent_count = int(cell_info["total_fire_days"].iloc[0]) if not cell_info.empty else 0
    mean_frp = float(cell_info["mean_frp"].iloc[0]) if not cell_info.empty else 0.0
    last_seen = (
        str(live_features["latest_detection_date"])
        if live_features["live"] and live_features["latest_detection_date"]
        else str(cell_info["last_seen"].iloc[0].date()) if not cell_info.empty else "No recent fire record"
    )

    if risk_score >= 0.58:
        advice = "Avoid forest-edge roads, move toward urban or irrigated corridors, and stay updated on official fire control advisories."
    elif risk_score >= 0.34:
        advice = "Conditions are elevated. Limit activity near dry vegetation and keep an exit route ready."
    else:
        advice = "No major wildfire escalation signal is indicated from the loaded historical model and regional priors."

    return {
        "latitude": latitude,
        "longitude": longitude,
        "risk_score": round(risk_score, 3),
        "risk_percent": round(risk_score * 100, 1),
        "risk_level": risk_level,
        "predicted_probability": round(risk_score, 3),
        "lat_bin": lat_bin,
        "lon_bin": lon_bin,
        "recent_fire_days": recent_count,
        "fire_last_1d": round(fire_last_1d, 3),
        "fire_last_3d": round(fire_last_3d, 3),
        "fire_last_7d": round(fire_last_7d, 3),
        "mean_frp": round(mean_frp, 2),
        "latest_cell_observation": last_seen,
        "seasonal_factor": round(monthly_rate, 3),
        "nearest_fire_region": region_name,
        "feature_source": live_features["source"],
        "model_status": model_status,
        "model_path": str(WILDFIRE_MODEL),
        "metadata": _load_metadata(),
        "model_report": _load_model_report(),
        "note": "This wildfire module uses a historical next-day fire classifier over grid cells and regional priors; it is an academic risk signal, not an official wildfire alert.",
        "advice": advice,
    }


def _wildfire_prior(latitude: float, longitude: float) -> tuple[float, str]:
    best_name = "General background"
    best_score = 0.03
    for name, lat, lon, weight in WILDFIRE_REGIONS:
        distance = _haversine_km(latitude, longitude, lat, lon)
        influence = weight * math.exp(-distance / 260.0)
        if influence > best_score:
            best_score = influence
            best_name = name
    return round(best_score, 3), best_name


def _cell_time_series(lat_bin: float, lon_bin: float) -> pd.DataFrame:
    history = _load_history()
    observed = history[(history["lat_bin"] == lat_bin) & (history["lon_bin"] == lon_bin)][["acq_date", "frp"]].copy()
    timeline = pd.DataFrame({"acq_date": _all_fire_dates()})
    timeline = timeline.merge(observed, on="acq_date", how="left")
    timeline["fire_today"] = timeline["frp"].notna().astype(int)
    timeline["frp"] = timeline["frp"].fillna(0.0)
    timeline["fire_last_1d"] = timeline["fire_today"].shift(1).fillna(0.0)
    timeline["fire_last_3d"] = timeline["fire_today"].shift(1).rolling(3, min_periods=1).sum().fillna(0.0)
    timeline["fire_last_7d"] = timeline["fire_today"].shift(1).rolling(7, min_periods=1).sum().fillna(0.0)
    return timeline


def _fetch_firms_live_features(latitude: float, longitude: float, lat_bin: float, lon_bin: float) -> dict[str, Any]:
    min_lon, min_lat, max_lon, max_lat = _bbox_for_cell(lat_bin, lon_bin)
    area = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_API_KEY}/{FIRMS_DATASET}/{area}/5"

    try:
        with urllib.request.urlopen(url, timeout=12) as response:
            payload = response.read().decode("utf-8")

        rows = list(csv.DictReader(io.StringIO(payload)))
        if not rows:
            return {
                "live": True,
                "source": f"NASA FIRMS {FIRMS_DATASET}",
                "frp": 0.0,
                "fire_last_1d": 0.0,
                "fire_last_3d": 0.0,
                "fire_last_7d": 0.0,
                "latest_detection_date": None,
            }

        filtered = []
        for row in rows:
            try:
                row_lat = round(float(row["latitude"]), 1)
                row_lon = round(float(row["longitude"]), 1)
                if row_lat != lat_bin or row_lon != lon_bin:
                    continue
                filtered.append(
                    {
                        "acq_date": pd.to_datetime(row["acq_date"]).normalize(),
                        "frp": float(row.get("frp") or 0.0),
                    }
                )
            except Exception:
                continue

        today = pd.Timestamp.utcnow().normalize().tz_localize(None)
        days = pd.date_range(end=today, periods=6, freq="D")

        if not filtered:
            timeline = pd.DataFrame({"acq_date": days, "frp": 0.0, "fire_today": 0.0})
        else:
            live_df = pd.DataFrame(filtered)
            per_day = live_df.groupby("acq_date", as_index=False).agg(frp=("frp", "mean"))
            per_day["fire_today"] = 1.0
            timeline = pd.DataFrame({"acq_date": days}).merge(per_day, on="acq_date", how="left")
            timeline["frp"] = timeline["frp"].fillna(0.0)
            timeline["fire_today"] = timeline["fire_today"].fillna(0.0)

        current = timeline.iloc[-1]
        prior = timeline.iloc[:-1]
        latest_detection = timeline.loc[timeline["fire_today"] > 0, "acq_date"]

        return {
            "live": True,
            "source": f"NASA FIRMS {FIRMS_DATASET} (5-day NRT window)",
            "frp": float(current["frp"]),
            "fire_last_1d": float(prior.tail(1)["fire_today"].sum()),
            "fire_last_3d": float(prior.tail(3)["fire_today"].sum()),
            "fire_last_7d": float(prior.tail(5)["fire_today"].sum()),
            "latest_detection_date": latest_detection.max().date().isoformat() if not latest_detection.empty else None,
        }
    except Exception as exc:
        return {
            "live": False,
            "source": f"historical fallback ({exc})",
            "frp": 0.0,
            "fire_last_1d": 0.0,
            "fire_last_3d": 0.0,
            "fire_last_7d": 0.0,
            "latest_detection_date": None,
        }


def _bbox_for_cell(lat_bin: float, lon_bin: float) -> tuple[float, float, float, float]:
    half = 0.5
    min_lon = round(max(-180.0, lon_bin - half), 4)
    min_lat = round(max(-90.0, lat_bin - half), 4)
    max_lon = round(min(180.0, lon_bin + half), 4)
    max_lat = round(min(90.0, lat_bin + half), 4)
    return min_lon, min_lat, max_lon, max_lat


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


def _risk_level(score: float) -> str:
    if score >= 0.58:
        return "High"
    if score >= 0.34:
        return "Moderate"
    return "Low"
