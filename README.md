# India Flood Segmentation from Sentinel-1

This project trains a small U-Net model to predict flood masks for India-specific Sentinel-1 SAR regions from the Sen1Floods11-style dataset in `dataset/dataset/sen1floods11_india`.

The folder has 110 hand-labeled image/mask pairs in total, including 68 India pairs. Training uses random patches from each 512 x 512 scene, which increases the effective training size without duplicating files on disk.

## Dataset Layout

Expected structure:

```text
dataset/dataset/sen1floods11_india/
  images/       *_S1Hand.tif
  masks/        *_LabelHand.tif
  weak_images/  *_S1Weak.tif
  weak_masks/   *_S1OtsuLabelWeak.tif
```

Only `India_*.tif` files are used by default.

## Train

Create and activate the project environment:

```bash
python -m venv .venv --system-site-packages
.\.venv\Scripts\activate
```

Quick CPU smoke run:

```bash
python src/train.py --epochs 1 --patches-per-image 2 --batch-size 4 --device cpu
```

Better training run:

```bash
python src/train.py --epochs 25 --patch-size 256 --patches-per-image 24 --batch-size 8 --use-weak
```

The best checkpoint is saved to:

```text
outputs/checkpoints/best_unet.pt
```

## Predict a Flood Mask for a Region

```bash
python src/predict.py ^
  --checkpoint outputs/checkpoints/best_unet.pt ^
  --input dataset/dataset/sen1floods11_india/images/India_1017769_S1Hand.tif ^
  --output outputs/predictions/India_1017769_pred.tif
```

The output is a single-band mask where `1` means predicted flood and `0` means not flooded.

## Disaster Intelligence Website

The project now includes a FastAPI backend and React frontend for:

- live-location flood situation in India using rainfall and regional risk signals,
- trained Sentinel-1 flood model status,
- earthquake magnitude/risk prediction using the pasted LSTM `.h5` model.

Install dependencies:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend
npm install
```

Run the backend:

```bash
.\run_backend.ps1
```

Run the frontend in another terminal:

```bash
.\run_frontend.ps1
```

Open:

```text
http://127.0.0.1:5173
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Notes

- Input Sentinel-1 files are expected as two SAR bands shaped like `(2, H, W)`.
- Mask labels use `1` for flood, `0` for non-flood, and `-1` for ignored/unlabeled pixels.
- Weak masks are optional because they are noisier than hand labels.
