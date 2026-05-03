from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.earthquake_service import predict_earthquake
from backend.flood_service import flood_risk_for_location, get_model_status
from backend.hotspots import get_hotspots
from backend.schemas import EarthquakeRequest, LocationRequest


app = FastAPI(
    title="Disaster Intelligence API",
    description="Flood and earthquake prediction API for the Minor Project dashboard.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "flood_model": get_model_status().__dict__,
    }


@app.post("/api/flood/location")
def flood_location(payload: LocationRequest) -> dict[str, object]:
    return flood_risk_for_location(payload.latitude, payload.longitude)


@app.post("/api/earthquake/predict")
def earthquake_prediction(payload: EarthquakeRequest) -> dict[str, object]:
    return predict_earthquake(payload.latitude, payload.longitude, payload.depth)


@app.get("/api/hotspots")
def hotspots() -> dict[str, list[dict[str, object]]]:
    return get_hotspots()
