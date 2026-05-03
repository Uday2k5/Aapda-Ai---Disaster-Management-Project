from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import FloodPatchDataset, discover_pairs, split_pairs
from metrics import masked_bce_with_logits, segmentation_scores
from model import SmallUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flood segmentation model for India Sentinel-1 regions.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset/dataset/sen1floods11_india"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patches-per-image", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-weak", action="store_true", help="Add weak Otsu masks as extra noisy training labels.")
    parser.add_argument("--include-non-india", action="store_true", help="Also use non-India files in this folder.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/checkpoints"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(args.data_root, india_only=not args.include_non_india, use_weak=args.use_weak)
    train_pairs, val_pairs = split_pairs(pairs, args.val_fraction, args.seed)

    train_ds = FloodPatchDataset(
        train_pairs,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        training=True,
    )
    val_ds = FloodPatchDataset(
        val_pairs,
        patch_size=args.patch_size,
        patches_per_image=max(1, args.patches_per_image // 4),
        training=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SmallUNet(in_channels=2).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_iou = -1.0
    best_path = args.output_dir / "best_unet.pt"
    print(f"Found {len(pairs)} pairs: {len(train_pairs)} train, {len(val_pairs)} val")
    print(f"Patch-expanded training samples per epoch: {len(train_ds)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs} train"):
            images = images.to(args.device)
            targets = targets.to(args.device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = masked_bce_with_logits(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * images.size(0)

        val_loss, scores = evaluate(model, val_loader, args.device)
        train_loss /= max(1, len(train_ds))
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"iou={scores['iou']:.4f} f1={scores['f1']:.4f} acc={scores['accuracy']:.4f}"
        )

        if scores["iou"] > best_iou:
            best_iou = scores["iou"]
            saved_args = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            }
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": saved_args,
                    "best_iou": best_iou,
                },
                best_path,
            )
            print(f"saved {best_path}")


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_scores = {"iou": 0.0, "f1": 0.0, "accuracy": 0.0}
    batches = 0

    for images, targets in tqdm(loader, desc="validate"):
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = masked_bce_with_logits(logits, targets)
        scores = segmentation_scores(logits, targets)

        total_loss += float(loss) * images.size(0)
        for key, value in scores.items():
            total_scores[key] += value
        batches += 1

    total_loss /= max(1, len(loader.dataset))
    total_scores = {key: value / max(1, batches) for key, value in total_scores.items()}
    return total_loss, total_scores


if __name__ == "__main__":
    main()
