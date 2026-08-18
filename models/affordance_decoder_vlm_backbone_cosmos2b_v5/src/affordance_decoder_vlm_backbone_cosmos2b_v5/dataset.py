"""Dual-cache dataset for v4: source-image VLM features with pre-computed projection.

Changes from v3:
  - No P2b augmentation (removed entirely).
  - Projection coords loaded from VLM cache (pre-computed), not computed on-the-fly.
  - No dependency on augmentation.py or projection.py at dataset level.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[4]
for p in (REPO_ROOT,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from repo_layout import resolve_repo_path  # noqa: E402


class VLMBackboneDatasetV5(Dataset):
    """Dual-cache dataset with pre-computed source-image projection (no augmentation)."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(resolve_repo_path(manifest_path))
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.rows: list[dict[str, Any]] = payload.get("rows", [])
        if not isinstance(self.rows, list):
            raise ValueError(f"manifest {self.manifest_path} lacks a rows list")
        self._geo_cache: dict[str, dict[str, Any]] = {}
        self._vlm_cache: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _load_geo_cache(self, cache_path: str) -> dict[str, Any]:
        resolved = str(resolve_repo_path(cache_path))
        if resolved not in self._geo_cache:
            self._geo_cache[resolved] = torch.load(
                resolved, map_location="cpu", weights_only=False,
            )
        return self._geo_cache[resolved]

    def _load_vlm_cache(self, cache_path: str) -> dict[str, Any]:
        resolved = str(resolve_repo_path(cache_path))
        if resolved not in self._vlm_cache:
            self._vlm_cache[resolved] = torch.load(
                resolved, map_location="cpu", weights_only=False,
            )
        return self._vlm_cache[resolved]

    def _load_heatmap(
        self, row: dict[str, Any], geo: dict[str, Any],
    ) -> torch.Tensor:
        target_index = int(row["target_index"])
        target_heatmaps = geo["target_heatmaps"].float()
        teacher_scores = geo["teacher_scores"].float()
        features = geo["features"].float()
        if target_index < int(target_heatmaps.shape[0]):
            return target_heatmaps[target_index].float().clamp(0.0, 1.0)
        part_name = str(row.get("part_name", ""))
        source_part_names = list(geo.get("part_names", []))
        if part_name in source_part_names and teacher_scores.numel() > 0:
            return teacher_scores[source_part_names.index(part_name)].float().clamp(0.0, 1.0)
        if teacher_scores.numel() > 0:
            return teacher_scores.max(dim=0).values.float().clamp(0.0, 1.0)
        return torch.zeros(features.shape[0], dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        geo = self._load_geo_cache(row["geo_cache_path"])
        vlm = self._load_vlm_cache(row["vlm_cache_path"])
        vlm_idx = int(row["vlm_cache_index"])

        aff_hidden = vlm["aff_hidden"][vlm_idx].float()
        visual_tokens = vlm["visual_tokens"][vlm_idx].float()
        proj_coords = vlm["source_proj_coords"][vlm_idx].float()
        proj_visible = vlm["source_proj_visible"][vlm_idx].float()

        features = geo["features"].float()
        xyz = geo["xyz"].float()

        return {
            "features": features,
            "xyz": xyz,
            "heatmap": self._load_heatmap(row, geo),
            "aff_hidden": aff_hidden,
            "visual_tokens": visual_tokens,
            "proj_coords": proj_coords,
            "proj_visible": proj_visible,
            "sample_id": row["sample_id"],
            "object_id": row["object_id"],
            "category_name": row.get("category_name", ""),
            "part_name": row.get("part_name", ""),
            "instruction": row.get("instruction", ""),
            "eval_group": row.get("eval_group", ""),
            "instruction_split": row.get("instruction_split", "seen"),
            "object_split": row.get("object_split", ""),
            "category_split": row.get("category_split", ""),
        }


def collate_vlm_backbone_v5_batch(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "xyz": torch.stack([b["xyz"] for b in batch]),
        "heatmap": torch.stack([b["heatmap"] for b in batch]),
        "aff_hidden": torch.stack([b["aff_hidden"] for b in batch]),
        "visual_tokens": torch.stack([b["visual_tokens"] for b in batch]),
        "proj_coords": torch.stack([b["proj_coords"] for b in batch]),
        "proj_visible": torch.stack([b["proj_visible"] for b in batch]),
        "sample_ids": [b["sample_id"] for b in batch],
        "object_ids": [b["object_id"] for b in batch],
        "category_names": [b["category_name"] for b in batch],
        "part_names": [b["part_name"] for b in batch],
        "instructions": [b["instruction"] for b in batch],
        "eval_groups": [b["eval_group"] for b in batch],
        "instruction_splits": [b["instruction_split"] for b in batch],
        "object_splits": [b["object_split"] for b in batch],
        "category_splits": [b["category_split"] for b in batch],
    }
