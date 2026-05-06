from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLOOD_CHECKPOINT = PROJECT_ROOT / "outputs" / "checkpoints" / "best_unet.pt"
FLOOD_DATA_ROOT = PROJECT_ROOT / "dataset" / "dataset" / "sen1floods11_india"
EARTHQUAKE_DIR = PROJECT_ROOT / "earthquake"
if not EARTHQUAKE_DIR.exists():
    EARTHQUAKE_DIR = PROJECT_ROOT / "Earthquake"
EARTHQUAKE_MODEL = EARTHQUAKE_DIR / "smarter_earthquake_model.h5"
EARTHQUAKE_DATA = EARTHQUAKE_DIR / "cleaned_earthquake_data.csv"
USE_EARTHQUAKE_MODEL = os.getenv("USE_EARTHQUAKE_MODEL", "true").lower() in {"1", "true", "yes", "on"}
WILDFIRE_DIR = PROJECT_ROOT / "firemodel"
WILDFIRE_MODEL = WILDFIRE_DIR / "model.pkl"
WILDFIRE_DATA = WILDFIRE_DIR / "data" / "fires.csv"
WILDFIRE_METADATA = WILDFIRE_DIR / "metadata.json"
WILDFIRE_FEATURES = WILDFIRE_DIR / "features.json"
WILDFIRE_REPORT = WILDFIRE_DIR / "model_report.txt"
FIRMS_API_KEY = "ba75bc8791b6771d3592af2360f3c0f6"
FIRMS_DATASET = "VIIRS_SNPP_NRT"
