"""Mixed real + pseudo-label dataset with per-sample confidence weighting."""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[4]
V5_SRC = REPO_ROOT / "models" / "affordance_decoder_vlm_backbone_cosmos2b_v5" / "src"
for p in (REPO_ROOT, V5_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from affordance_decoder_vlm_backbone_cosmos2b_v5.dataset import (  # noqa: E402
    VLMBackboneDatasetV5,
)


class MixedAffordanceDataset(Dataset):
    """Concatenates real-label and pseudo-label datasets with confidence weighting.

    Real samples get confidence=1.0. Pseudo samples get confidence derived from
    the heatmap's prediction certainty, scaled by ``pseudo_weight``.

    ``pseudo_manifests`` accepts a single path or a list of paths. Multiple
    manifests are loaded independently and concatenated in order.
    """

    def __init__(
        self,
        real_manifest: str | Path,
        pseudo_manifests: str | Path | list[str | Path] | None = None,
        pseudo_weight: float = 0.5,
        pseudo_high_thresh: float = 0.8,
        pseudo_low_thresh: float = 0.2,
        pseudo_sample_ratio: float = 1.0,
        seed: int = 42,
    ):
        self.real_ds = VLMBackboneDatasetV5(real_manifest)
        self.pseudo_weight = pseudo_weight
        self.pseudo_high_thresh = pseudo_high_thresh
        self.pseudo_low_thresh = pseudo_low_thresh
        self._real_len = len(self.real_ds)

        self._pseudo_datasets: list[VLMBackboneDatasetV5] = []
        self._pseudo_offsets: list[int] = []
        if pseudo_manifests is not None:
            if isinstance(pseudo_manifests, (str, Path)):
                pseudo_manifests = [pseudo_manifests]
            offset = 0
            for m in pseudo_manifests:
                ds = VLMBackboneDatasetV5(m)
                self._pseudo_datasets.append(ds)
                self._pseudo_offsets.append(offset)
                offset += len(ds)
        total_pseudo = sum(len(ds) for ds in self._pseudo_datasets)

        if pseudo_sample_ratio < 1.0 and total_pseudo > 0:
            k = max(1, int(total_pseudo * pseudo_sample_ratio))
            rng = random.Random(seed)
            self._pseudo_indices = sorted(rng.sample(range(total_pseudo), k))
        else:
            self._pseudo_indices = list(range(total_pseudo))
        self._pseudo_len = len(self._pseudo_indices)
        self._heatmap_overrides: dict[int, torch.Tensor] = {}

    @property
    def pseudo_count(self) -> int:
        return self._pseudo_len

    def update_pseudo_heatmaps(self, overrides: dict[int, torch.Tensor]) -> None:
        """Replace pseudo-label heatmaps with refreshed predictions."""
        self._heatmap_overrides = overrides

    def clear_pseudo_heatmap_overrides(self) -> None:
        self._heatmap_overrides = {}

    def __len__(self) -> int:
        return self._real_len + self._pseudo_len

    def _get_pseudo_sample(self, pseudo_index: int) -> dict[str, Any]:
        for ds, offset in zip(self._pseudo_datasets, self._pseudo_offsets):
            if pseudo_index < offset + len(ds):
                return ds[pseudo_index - offset]
        raise IndexError(f"pseudo_index {pseudo_index} out of range")

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < self._real_len:
            sample = self.real_ds[index]
            sample["confidence"] = torch.tensor(1.0)
            sample["is_pseudo"] = False
            sample["ignore_mask"] = torch.zeros(sample["heatmap"].shape[0], dtype=torch.bool)
        else:
            pseudo_rel_idx = index - self._real_len
            raw_pseudo_idx = self._pseudo_indices[pseudo_rel_idx]
            sample = self._get_pseudo_sample(raw_pseudo_idx)
            if pseudo_rel_idx in self._heatmap_overrides:
                sample["heatmap"] = self._heatmap_overrides[pseudo_rel_idx]
            heatmap = sample["heatmap"]
            confident = (heatmap >= self.pseudo_high_thresh) | (heatmap <= self.pseudo_low_thresh)
            sample["ignore_mask"] = ~confident
            certainty = (heatmap - 0.5).abs() * 2.0
            sample["confidence"] = certainty.mean().clamp(0.0, 1.0) * self.pseudo_weight
            sample["is_pseudo"] = True
        return sample


def _pad_to(tensor: torch.Tensor, target_n: int, dim: int = 0) -> torch.Tensor:
    """Zero-pad a tensor along `dim` to reach `target_n`."""
    n = tensor.shape[dim]
    if n >= target_n:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_n - n
    return torch.cat([tensor, torch.zeros(pad_shape, dtype=tensor.dtype)], dim=dim)


def collate_mixed_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_n = max(b["features"].shape[0] for b in batch)

    point_mask = torch.zeros(len(batch), max_n, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["features"].shape[0]
        point_mask[i, :n] = True

    return {
        "features": torch.stack([_pad_to(b["features"], max_n, 0) for b in batch]),
        "xyz": torch.stack([_pad_to(b["xyz"], max_n, 0) for b in batch]),
        "heatmap": torch.stack([_pad_to(b["heatmap"], max_n, 0) for b in batch]),
        "aff_hidden": torch.stack([b["aff_hidden"] for b in batch]),
        "visual_tokens": torch.stack([b["visual_tokens"] for b in batch]),
        "proj_coords": torch.stack([_pad_to(b["proj_coords"], max_n, 0) for b in batch]),
        "proj_visible": torch.stack([_pad_to(b["proj_visible"], max_n, 0) for b in batch]),
        "confidences": torch.stack([b["confidence"] for b in batch]),
        "ignore_mask": torch.stack([_pad_to(b["ignore_mask"], max_n, 0) for b in batch]),
        "point_mask": point_mask,
        "sample_ids": [b["sample_id"] for b in batch],
        "object_ids": [b["object_id"] for b in batch],
        "category_names": [b["category_name"] for b in batch],
        "part_names": [b["part_name"] for b in batch],
        "instructions": [b["instruction"] for b in batch],
        "eval_groups": [b["eval_group"] for b in batch],
        "instruction_splits": [b["instruction_split"] for b in batch],
        "object_splits": [b["object_split"] for b in batch],
        "category_splits": [b["category_split"] for b in batch],
        "is_pseudo": [b["is_pseudo"] for b in batch],
    }
