from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLOOD_CHECKPOINT = PROJECT_ROOT / "outputs" / "checkpoints" / "best_unet.pt"
FLOOD_DATA_ROOT = PROJECT_ROOT / "dataset" / "dataset" / "sen1floods11_india"
EARTHQUAKE_DIR = PROJECT_ROOT / "earthquake"
EARTHQUAKE_MODEL = EARTHQUAKE_DIR / "smarter_earthquake_model.h5"
EARTHQUAKE_DATA = EARTHQUAKE_DIR / "cleaned_earthquake_data.csv"
