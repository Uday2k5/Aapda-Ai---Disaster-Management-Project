from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
import torch

from data import normalize_sar, read_sentinel
from model import SmallUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict flood mask for a Sentinel-1 region.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    model = SmallUNet(in_channels=2).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image = normalize_sar(read_sentinel(args.input))
    tensor = torch.from_numpy(image[None]).to(args.device)
    logits = model(tensor)
    probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    prediction = (probability >= args.threshold).astype(np.uint8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(args.output, prediction)
    probability_path = args.output.with_name(args.output.stem + "_probability.tif")
    tifffile.imwrite(probability_path, probability.astype(np.float32))
    print(f"wrote mask: {args.output}")
    print(f"wrote probability: {probability_path}")


if __name__ == "__main__":
    main()
