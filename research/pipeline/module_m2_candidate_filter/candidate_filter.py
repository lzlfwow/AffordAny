from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image

from research.pipeline.common.segmentation import decode_segmentation_mask
from research.pipeline.module_m1_instance_extraction.instance_extraction import LVISInstanceRecord


@dataclass(frozen=True)
class CandidateFilterConfig:
    interactive_categories: tuple[str, ...]
    min_area: float = 1024.0
    min_clarity_score: float = 5.0
    min_mask_pixels: int = 256
    reject_border_touch: bool = True


@dataclass(frozen=True)
class CandidateFilterResult:
    object_id: str
    category_name: str
    passed: bool
    reasons: tuple[str, ...]
    metrics: dict

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "category_name": self.category_name,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "metrics": self.metrics,
        }


def _bbox_touches_border(record: LVISInstanceRecord) -> bool:
    x, y, w, h = record.bbox_xywh
    width = record.image_width
    height = record.image_height
    return x <= 0 or y <= 0 or (x + w) >= width or (y + h) >= height


def _masked_clarity_score(record: LVISInstanceRecord) -> tuple[float, int]:
    image = np.asarray(Image.open(record.image_path).convert("L"), dtype=np.float32)
    mask = decode_segmentation_mask(record.segmentation, record.image_height, record.image_width)
    mask_pixels = int(mask.sum())
    if mask_pixels == 0:
        return 0.0, 0

    gy, gx = np.gradient(image)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    score = float(magnitude[mask.astype(bool)].mean())
    return score, mask_pixels


def evaluate_candidate(
    record: LVISInstanceRecord,
    config: CandidateFilterConfig,
) -> CandidateFilterResult:
    reasons: list[str] = []
    category_ok = record.category_name.lower() in {
        name.lower() for name in config.interactive_categories
    }
    if not category_ok:
        reasons.append("non_interactive_category")

    if record.area < config.min_area:
        reasons.append("small_area")

    border_touch = _bbox_touches_border(record)
    if config.reject_border_touch and border_touch:
        reasons.append("touches_border")

    clarity_score, mask_pixels = _masked_clarity_score(record)
    if mask_pixels < config.min_mask_pixels:
        reasons.append("small_mask_pixels")
    if clarity_score < config.min_clarity_score:
        reasons.append("low_clarity")

    metrics = {
        "area": float(record.area),
        "touches_border": border_touch,
        "clarity_score": clarity_score,
        "mask_pixels": mask_pixels,
    }
    return CandidateFilterResult(
        object_id=record.object_id,
        category_name=record.category_name,
        passed=not reasons,
        reasons=tuple(reasons),
        metrics=metrics,
    )


def filter_candidates(
    records: list[LVISInstanceRecord],
    config: CandidateFilterConfig,
) -> list[CandidateFilterResult]:
    return [evaluate_candidate(record, config) for record in records]


def export_candidate_results(
    results: list[CandidateFilterResult],
    *,
    export_root: str | Path,
) -> dict:
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    candidate_records = []
    for result in results:
        object_dir = export_root / result.object_id
        candidate_dir = object_dir / "candidate"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        candidate_meta = result.to_dict()
        candidate_meta_path = candidate_dir / "candidate_meta.json"
        with candidate_meta_path.open("w", encoding="utf-8") as handle:
            json.dump(candidate_meta, handle, indent=2, ensure_ascii=True)

        candidate_records.append(
            {
                "object_id": result.object_id,
                "category_name": result.category_name,
                "passed": result.passed,
                "reasons": list(result.reasons),
                "candidate_meta_path": f"{result.object_id}/candidate/candidate_meta.json",
            }
        )

    summary = {
        "num_records": len(candidate_records),
        "num_passed": sum(1 for item in candidate_records if item["passed"]),
        "records": candidate_records,
    }
    summary_path = export_root / "candidate_records.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    return {
        "export_root": str(export_root),
        "summary_path": str(summary_path),
        "num_records": summary["num_records"],
        "num_passed": summary["num_passed"],
    }
