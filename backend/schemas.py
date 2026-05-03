from __future__ import annotations

from pydantic import BaseModel, Field


class LocationRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class EarthquakeRequest(LocationRequest):
    depth: float = Field(10.0, ge=0, le=700)
