"""Loss functions for self-training with soft pseudo-labels + confidence weighting.

Based on v5 losses.py. Key changes:
  - use_soft_labels: skip binarization, use continuous [0,1] heatmap as target
  - confidence: per-sample [B] weight multiplied into the valid mask
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
    use_soft_labels: bool = False,
    confidence: torch.Tensor | None = None,
    pseudo_bce_scale: float = 0.3,
    pseudo_focal_scale: float = 1.0,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    features = batch["features"].to(device)
    xyz = batch["xyz"].to(device)
    aff_hidden = batch["aff_hidden"].to(device)
    visual_tokens = batch["visual_tokens"].to(device)
    proj_coords = batch["proj_coords"].to(device)
    proj_visible = batch["proj_visible"].to(device)
    heatmap = batch["heatmap"].to(device).clamp(0.0, 1.0)

    if use_soft_labels:
        labels = heatmap
    else:
        labels = binarize_heatmap(heatmap, positive_threshold)

    valid = torch.ones_like(labels)

    point_mask = batch.get("point_mask")
    if point_mask is not None:
        valid = valid * point_mask.to(device).float()

    ignore_mask = batch.get("ignore_mask")
    if ignore_mask is not None:
        valid = valid * (~ignore_mask.to(device)).float()

    if confidence is not None:
        conf = confidence.to(device).clamp(0.0, 1.0).unsqueeze(1)  # [B, 1]
        valid = valid * conf

    logits = model(
        features, xyz, aff_hidden, visual_tokens,
        proj_coords=proj_coords, proj_visible=proj_visible,
    )["logits"]

    is_pseudo = batch.get("is_pseudo", [False] * logits.shape[0])
    has_pseudo = any(is_pseudo)

    if has_pseudo:
        pseudo_mask = torch.tensor(is_pseudo, device=device).unsqueeze(1).float()
        real_mask = 1.0 - pseudo_mask

        # Label smoothing: only for pseudo samples (0→ε, 1→1-ε)
        if label_smoothing > 0:
            smoothed = labels * (1 - 2 * label_smoothing) + label_smoothing
            labels = torch.where(pseudo_mask > 0.5, smoothed, labels)

        real_valid = valid * real_mask
        pseudo_valid = valid * pseudo_mask

        weight = compute_pos_weight(labels, real_valid, pos_weight) if real_valid.sum() > 0 else torch.tensor(pos_weight, device=device)
        bce_raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=weight)
        bce_real = masked_mean(bce_raw, real_valid) if real_valid.sum() > 0 else torch.tensor(0.0, device=device)
        bce_pseudo = masked_mean(bce_raw, pseudo_valid) if pseudo_valid.sum() > 0 else torch.tensor(0.0, device=device)
        bce_loss = bce_real + pseudo_bce_scale * bce_pseudo

        if pseudo_focal_scale < 1.0:
            focal_real = focal_loss_with_logits(logits, labels, real_valid, alpha=focal_alpha, gamma=focal_gamma) if real_valid.sum() > 0 else torch.tensor(0.0, device=device)
            focal_pseudo = focal_loss_with_logits(logits, labels, pseudo_valid, alpha=focal_alpha, gamma=focal_gamma) if pseudo_valid.sum() > 0 and pseudo_focal_scale > 0 else torch.tensor(0.0, device=device)
            focal_loss = focal_real + pseudo_focal_scale * focal_pseudo
        else:
            focal_loss = focal_loss_with_logits(logits, labels, valid, alpha=focal_alpha, gamma=focal_gamma)
        dice_loss = dice_loss_with_logits(logits, labels, valid)

        loss = lambda_bce * bce_loss + lambda_focal * focal_loss + lambda_dice * dice_loss
    else:
        weight = compute_pos_weight(labels, valid, pos_weight)
        bce_loss = masked_mean(
            F.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=weight),
            valid,
        )
        focal_loss = focal_loss_with_logits(logits, labels, valid, alpha=focal_alpha, gamma=focal_gamma)
        dice_loss = dice_loss_with_logits(logits, labels, valid)
        loss = lambda_bce * bce_loss + lambda_focal * focal_loss + lambda_dice * dice_loss

    return loss, {
        "loss": float(loss.detach().item()),
        "bce_loss": float(bce_loss.detach().item()),
        "focal_loss": float(focal_loss.detach().item()),
        "dice_loss": float(dice_loss.detach().item()),
        "pos_weight": float(weight.detach().item()) if not has_pseudo else 0.0,
        "truth_pos": float(
            ((labels * valid).sum() / valid.sum().clamp_min(1.0)).detach().item()
        ),
    }
