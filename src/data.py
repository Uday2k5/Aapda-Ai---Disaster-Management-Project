from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FloodPair:
    image: Path
    mask: Path
    weak: bool = False


def discover_pairs(root: Path, india_only: bool = True, use_weak: bool = False) -> list[FloodPair]:
    pairs: list[FloodPair] = []
    image_dir = root / "images"
    mask_dir = root / "masks"

    for image_path in sorted(image_dir.glob("*_S1Hand.tif")):
        if india_only and not image_path.name.startswith("India_"):
            continue
        mask_name = image_path.name.replace("_S1Hand.tif", "_LabelHand.tif")
        mask_path = mask_dir / mask_name
        if mask_path.exists():
            pairs.append(FloodPair(image_path, mask_path, weak=False))

    if use_weak:
        weak_image_dir = root / "weak_images"
        weak_mask_dir = root / "weak_masks"
        for image_path in sorted(weak_image_dir.glob("*_S1Weak.tif")):
            if india_only and not image_path.name.startswith("India_"):
                continue
            mask_name = image_path.name.replace("_S1Weak.tif", "_S1OtsuLabelWeak.tif")
            mask_path = weak_mask_dir / mask_name
            if mask_path.exists():
                pairs.append(FloodPair(image_path, mask_path, weak=True))

    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found under {root}")
    return pairs


def split_pairs(
    pairs: list[FloodPair], val_fraction: float = 0.2, seed: int = 42
) -> tuple[list[FloodPair], list[FloodPair]]:
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_fraction))
    return shuffled[val_count:], shuffled[:val_count]


def read_sentinel(path: Path) -> np.ndarray:
    image = tifffile.imread(path).astype(np.float32)
    if image.ndim == 2:
        image = image[None, ...]
    if image.shape[0] not in (1, 2, 3) and image.shape[-1] in (1, 2, 3):
        image = np.moveaxis(image, -1, 0)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return image


def read_mask(path: Path) -> np.ndarray:
    mask = tifffile.imread(path).astype(np.int64)
    if mask.ndim == 3:
        mask = mask[0]
    return mask


def normalize_sar(image: np.ndarray) -> np.ndarray:
    normalized = image.copy()
    for band in range(normalized.shape[0]):
        channel = normalized[band]
        lo, hi = np.percentile(channel, (1, 99))
        if hi <= lo:
            normalized[band] = 0.0
            continue
        channel = np.clip(channel, lo, hi)
        normalized[band] = (channel - lo) / (hi - lo)
    return normalized.astype(np.float32)


class FloodPatchDataset(Dataset):
    def __init__(
        self,
        pairs: list[FloodPair],
        patch_size: int = 256,
        patches_per_image: int = 16,
        training: bool = True,
        min_labeled_fraction: float = 0.1,
    ) -> None:
        self.pairs = pairs
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.training = training
        self.min_labeled_fraction = min_labeled_fraction

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[index // self.patches_per_image]
        image = normalize_sar(read_sentinel(pair.image))
        mask = read_mask(pair.mask)
        image_patch, mask_patch = self._crop(image, mask)

        if self.training:
            image_patch, mask_patch = self._augment(image_patch, mask_patch)

        valid = mask_patch >= 0
        target = (mask_patch == 1).astype(np.float32)
        ignore = (~valid).astype(np.float32)
        stacked_target = np.stack([target, ignore], axis=0)

        return torch.from_numpy(image_patch), torch.from_numpy(stacked_target)

    def _crop(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, height, width = image.shape
        patch = min(self.patch_size, height, width)

        best_y = 0
        best_x = 0
        best_score = -1.0
        attempts = 12 if self.training else 1

        for _ in range(attempts):
            y = 0 if height == patch else random.randint(0, height - patch)
            x = 0 if width == patch else random.randint(0, width - patch)
            mask_patch = mask[y : y + patch, x : x + patch]
            labeled_fraction = float((mask_patch >= 0).mean())
            flood_fraction = float((mask_patch == 1).mean())
            score = labeled_fraction + flood_fraction
            if labeled_fraction >= self.min_labeled_fraction:
                best_y, best_x = y, x
                break
            if score > best_score:
                best_score = score
                best_y, best_x = y, x

        return (
            image[:, best_y : best_y + patch, best_x : best_x + patch],
            mask[best_y : best_y + patch, best_x : best_x + patch],
        )

    @staticmethod
    def _augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = image[:, :, ::-1]
            mask = mask[:, ::-1]
        if random.random() < 0.5:
            image = image[:, ::-1, :]
            mask = mask[::-1, :]
        rotations = random.randint(0, 3)
        if rotations:
            image = np.rot90(image, rotations, axes=(1, 2))
            mask = np.rot90(mask, rotations, axes=(0, 1))
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)
