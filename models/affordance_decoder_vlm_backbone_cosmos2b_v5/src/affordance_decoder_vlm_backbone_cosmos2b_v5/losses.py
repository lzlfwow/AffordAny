"""Loss functions for v3 affordance decoder.

Identical to v1 except compute_loss passes projection coords to the model.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def binarize_heatmap(heatmap: torch.Tensor, positive_threshold: float) -> torch.Tensor:
    return (heatmap > positive_threshold).float()


def masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return (values * valid).sum() / valid.sum().clamp_min(1.0)


def dice_loss_with_logits(
    logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor,
) -> torch.Tensor:
    prob = torch.sigmoid(logits) * valid
    labels = labels.clamp(0.0, 1.0) * valid
    intersection = (prob * labels).sum(dim=1)
    denominator = prob.sum(dim=1) + labels.sum(dim=1)
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def focal_loss_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    labels = labels.clamp(0.0, 1.0)
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * labels + (1.0 - prob) * (1.0 - labels)
    alpha_t = alpha * labels + (1.0 - alpha) * (1.0 - labels)
    return masked_mean(alpha_t * (1.0 - p_t).pow(gamma) * bce, valid)


def compute_pos_weight(
    labels: torch.Tensor, valid: torch.Tensor, configured_weight: float,
) -> torch.Tensor:
    if configured_weight > 0.0:
        return torch.full((), float(configured_weight), device=labels.device)
    positives = (labels * valid).sum()
    negatives = ((1.0 - labels) * valid).sum()
    return (negatives / positives.clamp_min(1.0)).clamp_min(1.0)


def compute_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device,
    positive_threshold: float,
    pos_weight: float,
    lambda_bce: float,
    lambda_focal: float,
    lambda_dice: float,
    focal_alpha: float,
    focal_gamma: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    features = batch["features"].to(device)
    xyz = batch["xyz"].to(device)
    aff_hidden = batch["aff_hidden"].to(device)
    visual_tokens = batch["visual_tokens"].to(device)
    proj_coords = batch["proj_coords"].to(device)
    proj_visible = batch["proj_visible"].to(device)
    heatmap = batch["heatmap"].to(device).clamp(0.0, 1.0)

    labels = binarize_heatmap(heatmap, positive_threshold)
    valid = torch.ones_like(labels)

    logits = model(
        features, xyz, aff_hidden, visual_tokens,
        proj_coords=proj_coords, proj_visible=proj_visible,
    )["logits"]

    weight = compute_pos_weight(labels, valid, pos_weight)
    bce_loss = masked_mean(
        F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none", pos_weight=weight,
        ),
        valid,
    )
    focal_loss = focal_loss_with_logits(
        logits, labels, valid, alpha=focal_alpha, gamma=focal_gamma,
    )
    dice_loss = dice_loss_with_logits(logits, labels, valid)

    loss = lambda_bce * bce_loss + lambda_focal * focal_loss + lambda_dice * dice_loss
    return loss, {
        "loss": float(loss.detach().item()),
        "bce_loss": float(bce_loss.detach().item()),
        "focal_loss": float(focal_loss.detach().item()),
        "dice_loss": float(dice_loss.detach().item()),
        "pos_weight": float(weight.detach().item()),
        "truth_pos": float(
            ((labels * valid).sum() / valid.sum().clamp_min(1.0)).detach().item()
        ),
    }
