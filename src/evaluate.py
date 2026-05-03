from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import FloodPatchDataset, discover_pairs, split_pairs
from model import SmallUNet
from train import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained flood segmentation checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/best_unet.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("dataset/dataset/sen1floods11_india"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    saved_args = checkpoint.get("args", {})

    seed = int(saved_args.get("seed", 42))
    val_fraction = float(saved_args.get("val_fraction", 0.2))
    patch_size = int(saved_args.get("patch_size", 256))
    patches_per_image = max(1, int(saved_args.get("patches_per_image", 16)) // 4)
    use_weak = bool(saved_args.get("use_weak", False))
    include_non_india = bool(saved_args.get("include_non_india", False))

    pairs = discover_pairs(
        args.data_root,
        india_only=not include_non_india,
        use_weak=use_weak,
    )
    train_pairs, val_pairs = split_pairs(pairs, val_fraction=val_fraction, seed=seed)
    val_ds = FloodPatchDataset(
        val_pairs,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        training=False,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SmallUNet(in_channels=2).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    loss, scores = evaluate(model, val_loader, args.device)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset pairs: {len(pairs)} total, {len(train_pairs)} train, {len(val_pairs)} validation")
    print(f"Validation patches: {len(val_ds)}")
    print(f"Validation loss: {loss:.4f}")
    print(f"Pixel accuracy: {scores['accuracy']:.4f} ({scores['accuracy'] * 100:.2f}%)")
    print(f"Flood IoU: {scores['iou']:.4f}")
    print(f"Flood F1/Dice: {scores['f1']:.4f}")


if __name__ == "__main__":
    main()
