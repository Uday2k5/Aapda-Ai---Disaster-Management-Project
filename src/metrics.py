from __future__ import annotations

import torch


def masked_bce_with_logits(logits: torch.Tensor, target_and_ignore: torch.Tensor) -> torch.Tensor:
    target = target_and_ignore[:, :1]
    ignore = target_and_ignore[:, 1:2].bool()
    valid = ~ignore
    if valid.sum() == 0:
        return logits.sum() * 0.0
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits[valid], target[valid]
    )


@torch.no_grad()
def segmentation_scores(
    logits: torch.Tensor, target_and_ignore: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    target = target_and_ignore[:, :1].bool()
    ignore = target_and_ignore[:, 1:2].bool()
    valid = ~ignore
    pred = torch.sigmoid(logits) >= threshold

    pred = pred[valid]
    target = target[valid]
    if pred.numel() == 0:
        return {"iou": 0.0, "f1": 0.0, "accuracy": 0.0}

    tp = (pred & target).sum().float()
    fp = (pred & ~target).sum().float()
    fn = (~pred & target).sum().float()
    correct = (pred == target).sum().float()

    iou = tp / (tp + fp + fn + 1e-7)
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    accuracy = correct / pred.numel()
    return {"iou": float(iou), "f1": float(f1), "accuracy": float(accuracy)}
