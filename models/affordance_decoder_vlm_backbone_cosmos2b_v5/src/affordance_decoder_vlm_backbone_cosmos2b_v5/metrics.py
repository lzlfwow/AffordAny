from __future__ import annotations

from collections.abc import Sequence

import torch

from .losses import binarize_heatmap


def binary_auc(prob: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> float:
    mask = valid > 0.5
    scores = prob[mask].detach().flatten()
    target = labels[mask].detach().flatten()
    positives = target.sum()
    negatives = target.numel() - positives
    if positives <= 0 or negatives <= 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float32)
    pos_ranks = ranks[target > 0.5].sum()
    auc = (pos_ranks - positives * (positives + 1) / 2) / (positives * negatives)
    return float(auc.item())


def compute_basic_metrics(
    prob: torch.Tensor,
    heatmap: torch.Tensor,
    *,
    threshold: float,
    positive_threshold: float,
    categories: Sequence[str] | None = None,
) -> dict[str, float]:
    heatmap = heatmap.clamp(0.0, 1.0)
    labels = binarize_heatmap(heatmap, positive_threshold)
    valid = torch.ones_like(labels)
    pred = (prob >= threshold).float()
    inter = ((pred * labels) * valid).sum()
    union = ((((pred + labels) > 0).float()) * valid).sum().clamp_min(1.0)
    mae = ((prob - labels).abs() * valid).sum() / valid.sum().clamp_min(1.0)
    return {
        "iou": float((inter / union).item()),
        "miou": category_miou(pred, labels, valid, categories, inter, union),
        "auc": binary_auc(prob, labels, valid),
        "mae": float(mae.item()),
        "sim": distribution_similarity(prob, heatmap, valid),
        "pred_pos": float(((pred * valid).sum() / valid.sum().clamp_min(1.0)).item()),
        "truth_pos": float(((labels * valid).sum() / valid.sum().clamp_min(1.0)).item()),
        "prob_mean": float(((prob * valid).sum() / valid.sum().clamp_min(1.0)).item()),
        "prob_max": float(prob.max().item()),
        "valid_ratio": float(valid.mean().item()),
    }


def category_miou(
    pred: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    categories: Sequence[str] | None,
    global_inter: torch.Tensor,
    global_union: torch.Tensor,
) -> float:
    if categories is None:
        return float((global_inter / global_union).item())
    if len(categories) != int(pred.shape[0]):
        raise ValueError(
            f"categories length {len(categories)} does not match batch size {pred.shape[0]}"
        )
    values: list[float] = []
    for category in sorted(set(str(item) for item in categories)):
        mask = torch.tensor(
            [str(item) == category for item in categories],
            device=pred.device,
            dtype=torch.bool,
        )
        if not bool(mask.any()):
            continue
        cat_pred = pred[mask]
        cat_labels = labels[mask]
        cat_valid = valid[mask]
        inter = ((cat_pred * cat_labels) * cat_valid).sum()
        union = ((((cat_pred + cat_labels) > 0).float()) * cat_valid).sum().clamp_min(1.0)
        values.append(float((inter / union).item()))
    return sum(values) / len(values) if values else float("nan")


def distribution_similarity(
    prob: torch.Tensor,
    heatmap: torch.Tensor,
    valid: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> float:
    pred_dist = (prob.clamp(0.0, 1.0) * valid).flatten(start_dim=1)
    true_dist = (heatmap.clamp(0.0, 1.0) * valid).flatten(start_dim=1)
    pred_sum = pred_dist.sum(dim=1, keepdim=True)
    true_sum = true_dist.sum(dim=1, keepdim=True)
    pred_has_mass = pred_sum.squeeze(1) > eps
    true_has_mass = true_sum.squeeze(1) > eps
    pred_norm = torch.where(pred_sum > eps, pred_dist / pred_sum.clamp_min(eps), torch.zeros_like(pred_dist))
    true_norm = torch.where(true_sum > eps, true_dist / true_sum.clamp_min(eps), torch.zeros_like(true_dist))
    sim = torch.minimum(pred_norm, true_norm).sum(dim=1)
    sim = torch.where(
        pred_has_mass & true_has_mass,
        sim,
        torch.where(~pred_has_mass & ~true_has_mass, torch.ones_like(sim), torch.zeros_like(sim)),
    )
    return float(sim.mean().item())


def merge_metric_lists(items: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for item in items for key in item})
    merged: dict[str, float] = {}
    for key in keys:
        values = [float(item[key]) for item in items if key in item and float(item[key]) == float(item[key])]
        merged[key] = sum(values) / len(values) if values else float("nan")
    return merged
