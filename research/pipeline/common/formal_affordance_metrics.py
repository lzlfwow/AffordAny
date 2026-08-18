from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class FormalMetricConfig:
    score_threshold: float = 0.5
    support_eps: float = 1e-6


def parse_thresholds(raw: str | list[float] | tuple[float, ...] | None) -> list[float]:
    """Compatibility shim for older callers; threshold grids are no longer used."""
    if raw is None:
        return [FormalMetricConfig().score_threshold]
    if isinstance(raw, str):
        return [float(item) for item in raw.split(",") if item.strip()]
    return [float(item) for item in raw]


def _as_numpy_vector(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().float().reshape(-1).numpy()
    else:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    return array.astype(np.float32, copy=False)


def _align_prob_target(prob: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    prob_array = _as_numpy_vector(prob)
    target_array = _as_numpy_vector(target)
    count = min(prob_array.shape[0], target_array.shape[0])
    return prob_array[:count], target_array[:count]


def _binary_metrics(prob: np.ndarray, target: np.ndarray, *, score_threshold: float, support_eps: float) -> dict[str, float]:
    pred = prob >= float(score_threshold)
    truth = target > float(support_eps)
    tp = float(np.logical_and(pred, truth).sum())
    fp = float(np.logical_and(pred, ~truth).sum())
    fn = float(np.logical_and(~pred, truth).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1.0)
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _average_precision(prob: np.ndarray, target: np.ndarray, *, support_eps: float) -> float:
    truth = target > float(support_eps)
    num_pos = int(truth.sum())
    if prob.size == 0 or num_pos <= 0:
        return 0.0
    order = np.argsort(-prob, kind="mergesort")
    sorted_labels = truth[order].astype(np.float64, copy=False)
    cumulative_tp = np.cumsum(sorted_labels)
    precision = cumulative_tp / np.arange(1, sorted_labels.size + 1, dtype=np.float64)
    return float(precision[sorted_labels > 0.5].sum() / max(num_pos, 1))


def _roc_auc(prob: np.ndarray, target: np.ndarray, *, support_eps: float) -> float:
    truth = target > float(support_eps)
    num_pos = int(truth.sum())
    num_neg = int(truth.size - num_pos)
    if prob.size == 0 or num_pos <= 0 or num_neg <= 0:
        return 0.0
    order = np.argsort(prob, kind="mergesort")
    sorted_prob = prob[order]
    ranks = np.empty(prob.shape[0], dtype=np.float64)
    start = 0
    while start < sorted_prob.shape[0]:
        end = start + 1
        while end < sorted_prob.shape[0] and sorted_prob[end] == sorted_prob[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(ranks[truth].sum())
    auc = (positive_rank_sum - num_pos * (num_pos + 1) / 2.0) / float(num_pos * num_neg)
    return float(auc)


def _weighted_metrics(prob: np.ndarray, target: np.ndarray, *, support_eps: float) -> dict[str, float]:
    prob = np.clip(prob.astype(np.float64, copy=False), 0.0, 1.0)
    target = np.clip(target.astype(np.float64, copy=False), 0.0, 1.0)
    support = target > float(support_eps)
    weights = target
    intersection = float(np.minimum(prob, target).sum())
    union = float(np.maximum(prob, target).sum())
    weighted_iou = intersection / max(union, 1e-8)
    weight_sum = float(weights.sum())
    if weight_sum > 1e-8:
        weighted_mae = float((np.abs(prob - target) * weights).sum() / weight_sum)
    else:
        weighted_mae = float(np.abs(prob - target).mean()) if prob.size else 0.0
    support_mae = float(np.abs(prob[support] - target[support]).mean()) if support.any() else 0.0
    background_mean = float(prob[~support].mean()) if (~support).any() else 0.0
    support_mean = float(prob[support].mean()) if support.any() else 0.0
    return {
        "weighted_iou": float(weighted_iou),
        "weighted_mae": float(weighted_mae),
        "support_mae": float(support_mae),
        "support_score_mean": float(support_mean),
        "background_score_mean": float(background_mean),
        "score_margin_mean": float(support_mean - background_mean),
    }


def compute_sample_formal_metrics(
    prob: Any,
    target: Any,
    *,
    fixed_threshold: float = 0.5,
    support_eps: float = 1e-6,
) -> dict[str, float]:
    prob_array, target_array = _align_prob_target(prob, target)
    binary = _binary_metrics(prob_array, target_array, score_threshold=fixed_threshold, support_eps=support_eps)
    soft = _weighted_metrics(prob_array, target_array, support_eps=support_eps)
    mae = float(np.abs(prob_array - target_array).mean()) if prob_array.size else 0.0
    return {
        **binary,
        **soft,
        "pr_auc": _average_precision(prob_array, target_array, support_eps=support_eps),
        "roc_auc": _roc_auc(prob_array, target_array, support_eps=support_eps),
        "mae": mae,
    }


def average_metric_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item.keys()})
    aggregated: dict[str, float] = {}
    for key in keys:
        values = [float(item[key]) for item in items if key in item]
        aggregated[key] = float(sum(values) / max(len(values), 1))
    return aggregated


def evaluate_formal_rows(
    rows: list[dict[str, Any]],
    *,
    calibration_rows: list[dict[str, Any]] | None = None,
    fixed_threshold: float = 0.5,
    calibration_thresholds: list[float] | None = None,
    support_eps: float = 1e-6,
) -> dict[str, float]:
    del calibration_rows, calibration_thresholds
    sample_metrics = [
        compute_sample_formal_metrics(
            row["prob"],
            row["target"],
            fixed_threshold=fixed_threshold,
            support_eps=support_eps,
        )
        for row in rows
    ]
    payload = average_metric_dicts(sample_metrics)
    payload["score_threshold"] = float(fixed_threshold)
    payload["support_eps"] = float(support_eps)
    payload["samples"] = len(rows)
    return payload


def build_instruction_deltas(
    seen_metrics: dict[str, float],
    unseen_metrics: dict[str, float],
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for name in ("iou", "weighted_iou", "pr_auc", "roc_auc", "precision", "recall", "f1"):
        if name in seen_metrics and name in unseen_metrics:
            deltas[f"delta_{name}"] = float(seen_metrics[name] - unseen_metrics[name])
    for name in ("mae", "weighted_mae", "support_mae"):
        if name in seen_metrics and name in unseen_metrics:
            deltas[f"delta_{name}"] = float(unseen_metrics[name] - seen_metrics[name])
    return deltas
