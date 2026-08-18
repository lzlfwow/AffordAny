from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Protocol
from urllib.parse import urlencode

import numpy as np
from PIL import Image
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_layout import (
    LVIS_RAW_ROOT,
    SAM3D_OBJECTS_ROOT,
    default_sam3d_objects_env_path,
    lvis_run_root,
    repo_relative,
)
from research.pipeline.common.segmentation import decode_segmentation_mask
from research.pipeline.module_m1_instance_extraction.instance_extraction import (
    InstanceExtractionConfig,
    LVISInstanceRecord,
    build_source_meta,
    extract_lvis_records,
    load_lvis_json,
)
from research.pipeline.module_m2_candidate_filter.gemini_filter import (
    GeminiCandidateFilterClient,
    GeminiFilterConfig,
    create_masked_instance_view,
    decision_to_reasons,
)
from research.pipeline.module_m3_object_reconstruction.reconstruction import (
    LocalSam3DObjectsExecutor,
    ReconstructionConfig,
    build_reconstruction_request,
    execute_reconstruction,
)
from research.pipeline.module_m4_multiview_render.render import (
    LocalRenderExecutor,
    RenderConfig,
    build_render_request,
    execute_render,
)
from research.pipeline.module_m5_part_prompt.part_prompt import (
    GeminiPartPromptClient,
    PartPromptConfig,
    execute_part_prompt_generation,
)
from research.pipeline.module_m6_part_segmentation.segmentation import (
    LocalSam3SegmentationExecutor,
    SegmentationConfig,
    build_segmentation_request,
    execute_segmentation,
)
from research.pipeline.module_m7_lift_2d_to_3d.lift import (
    LiftConfig,
    LocalLiftExecutor,
    build_lift_request,
    execute_lift,
)
from research.pipeline.module_m8_multiview_fusion.fusion import (
    FusionConfig,
    LocalFusionExecutor,
    build_fusion_request,
    execute_fusion,
)
from research.pipeline.module_m9_packaging_qc.packaging import (
    LocalPackagingExecutor,
    PackagingConfig,
    build_packaging_request,
    execute_packaging,
)
from research.pipeline.module_m11_dit_part_segmentation import (
    DitPartSegmentationConfig,
    ExternalDitPartSegmentationExecutor,
    Label3DAdapterDitPartSegmentationExecutor,
    Seed3DReproDitPartSegmentationExecutor,
    build_dit_part_segmentation_request,
    execute_dit_part_segmentation,
)


DEFAULT_ANNOTATION_JSON = (LVIS_RAW_ROOT / "valid" / "annotation" / "lvis_v1_val.json").as_posix()
DEFAULT_IMAGES_ROOT = (LVIS_RAW_ROOT / "valid" / "image" / "val2017").as_posix()
DEFAULT_STAGE1_ROOT = str(lvis_run_root("default") / "stage1_candidates")
DEFAULT_STAGE2_ROOT = str(lvis_run_root("default") / "stage2_dataset")
DEFAULT_APPROVAL_JSON = str(lvis_run_root("default") / "stage1_candidates" / "approval_template.json")
DEFAULT_SAM3D_ROOT = repo_relative(SAM3D_OBJECTS_ROOT)
DEFAULT_SAM3D_CONDA_ENV = default_sam3d_objects_env_path()


def _repo_root() -> Path:
    return REPO_ROOT


def _default_run_root(run_name: str) -> Path:
    return lvis_run_root(run_name)


def _serialize_tuple(value: tuple[str, ...]) -> list[str]:
    return list(value)


@dataclass(frozen=True)
class InteractiveCategoryDecision:
    interactive_object: bool
    short_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "interactive_object": self.interactive_object,
            "short_reason": self.short_reason,
        }


class InteractiveCategoryInferenceClient(Protocol):
    def infer(
        self,
        *,
        category_name: str,
        config: GeminiFilterConfig,
    ) -> InteractiveCategoryDecision:
        ...


def build_interactive_category_prompt(category_name: str) -> str:
    return (
        "You are filtering category names for a robot embodied interaction dataset. "
        "Keep only categories that usually have meaningful human-interaction or operable regions "
        "that a robot may need to localize from language instructions. "
        "Reject passive scenery, materials, abstract classes, or categories without a stable operable-part concept. "
        'Return JSON only with keys: interactive_object, short_reason. '
        f"Category name: {category_name}."
    )


def build_interactive_category_payload(
    config: GeminiFilterConfig,
    *,
    category_name: str,
) -> dict[str, Any]:
    return {
        "contents": [
            {
                "parts": [
                    {
                        "text": build_interactive_category_prompt(category_name)
                    }
                ]
            }
        ],
        "generationConfig": {"responseMimeType": "application/json"},
    }


def _parse_json_text_from_gemini_response(response_json: dict[str, Any]) -> dict[str, Any]:
    try:
        parts = response_json["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unsupported response structure") from exc

    text_chunks = []
    for item in parts:
        if isinstance(item, dict) and "text" in item:
            text_chunks.append(item["text"])
    content_text = "".join(text_chunks).strip()
    if not content_text:
        raise ValueError("response content must include text")

    if content_text.startswith("```"):
        blocks = content_text.split("```")
        if len(blocks) >= 2:
            content_text = blocks[1]
        content_text = content_text.removeprefix("json").strip()
    parsed = json.loads(content_text)
    if isinstance(parsed, list):
        if not parsed or not isinstance(parsed[0], dict):
            raise ValueError("response JSON list must contain an object")
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        raise ValueError("response JSON must decode to an object")
    return parsed


def parse_interactive_category_decision(response_json: dict[str, Any]) -> InteractiveCategoryDecision:
    parsed = _parse_json_text_from_gemini_response(response_json)
    return InteractiveCategoryDecision(
        interactive_object=bool(parsed["interactive_object"]),
        short_reason=str(parsed["short_reason"]),
    )


class GeminiInteractiveCategoryClient:
    def get_base_url(self, config: GeminiFilterConfig) -> str:
        value = os.environ.get(config.base_url_env, "").rstrip("/")
        if not value:
            raise RuntimeError(f"missing env var: {config.base_url_env}")
        return value

    def get_api_key(self, config: GeminiFilterConfig) -> str:
        value = os.environ.get(config.api_key_env, "")
        if not value:
            raise RuntimeError(f"missing env var: {config.api_key_env}")
        return value

    def build_request(
        self,
        *,
        category_name: str,
        config: GeminiFilterConfig,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        endpoint = config.endpoint_path_template.format(model=config.model_name)
        query = urlencode({"key": self.get_api_key(config)})
        url = f"{self.get_base_url(config)}{endpoint}?{query}"
        headers = {"Content-Type": "application/json"}
        payload = build_interactive_category_payload(config, category_name=category_name)
        return url, headers, payload

    def infer(
        self,
        *,
        category_name: str,
        config: GeminiFilterConfig,
    ) -> InteractiveCategoryDecision:
        url, headers, payload = self.build_request(category_name=category_name, config=config)
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
        return parse_interactive_category_decision(response.json())


@dataclass(frozen=True)
class Stage1Config:
    annotation_json_path: str = DEFAULT_ANNOTATION_JSON
    images_root: str = DEFAULT_IMAGES_ROOT
    split_name: str = "lvis_val"
    stage1_root: str = DEFAULT_STAGE1_ROOT
    allowed_category_names: tuple[str, ...] = ()
    start_index: int = 0
    limit: int | None = None
    geometry_min_mask_area_ratio: float = 0.03
    geometry_max_mask_aspect_ratio: float = 4.0
    geometry_max_border_touch_count: int = 1
    geometry_border_margin_ratio: float = 0.02
    geometry_border_margin_min_px: int = 2
    target_num_candidates: int | None = None
    quota_image_count_power: float = 0.5
    quota_min_per_category: int = 1
    reuse_existing_selection_quotas: bool = False
    gemini_config: GeminiFilterConfig = GeminiFilterConfig()
    overwrite: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_category_names"] = _serialize_tuple(self.allowed_category_names)
        payload["gemini_config"] = self.gemini_config.to_dict()
        return payload


@dataclass(frozen=True)
class Stage2Config:
    stage1_root: str = DEFAULT_STAGE1_ROOT
    stage2_root: str = DEFAULT_STAGE2_ROOT
    approval_json_path: str = DEFAULT_APPROVAL_JSON
    prefetched_prompt_root: str | None = None
    overwrite: bool = False
    skip_existing: bool = True
    max_objects: int | None = None
    hf_endpoint: str = "https://huggingface.co"
    reconstruction_config: ReconstructionConfig = ReconstructionConfig(
        sam3d_root=DEFAULT_SAM3D_ROOT,
        checkpoint_tag="hf",
        conda_env_path=DEFAULT_SAM3D_CONDA_ENV,
    )
    render_config: RenderConfig = RenderConfig(
        sam3d_root=DEFAULT_SAM3D_ROOT,
        conda_env_path=DEFAULT_SAM3D_CONDA_ENV,
    )
    part_prompt_config: PartPromptConfig = PartPromptConfig()
    segmentation_config: SegmentationConfig = SegmentationConfig()
    lift_config: LiftConfig = LiftConfig()
    fusion_config: FusionConfig = FusionConfig()
    dit_part_segmentation_config: DitPartSegmentationConfig = DitPartSegmentationConfig()
    packaging_config: PackagingConfig = PackagingConfig()

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage1_root": self.stage1_root,
            "stage2_root": self.stage2_root,
            "approval_json_path": self.approval_json_path,
            "prefetched_prompt_root": self.prefetched_prompt_root,
            "overwrite": self.overwrite,
            "skip_existing": self.skip_existing,
            "max_objects": self.max_objects,
            "hf_endpoint": self.hf_endpoint,
            "reconstruction_config": self.reconstruction_config.to_dict(),
            "render_config": self.render_config.to_dict(),
            "part_prompt_config": self.part_prompt_config.to_dict(),
            "segmentation_config": self.segmentation_config.to_dict(),
            "lift_config": self.lift_config.to_dict(),
            "fusion_config": self.fusion_config.to_dict(),
            "dit_part_segmentation_config": self.dit_part_segmentation_config.to_dict(),
            "packaging_config": self.packaging_config.to_dict(),
        }


def _slice_items(items, *, start_index: int, limit: int | None):
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    start = start_index
    end = None if limit is None else start_index + limit
    return items[start:end]


def _ensure_clean_root(path: Path, *, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _progress(message: str) -> None:
    print(message, flush=True)


def _group_records_by_category(records: list[LVISInstanceRecord]) -> tuple[list[str], dict[str, list[LVISInstanceRecord]]]:
    ordered_categories: list[str] = []
    grouped: dict[str, list[LVISInstanceRecord]] = {}
    for record in records:
        if record.category_name not in grouped:
            ordered_categories.append(record.category_name)
            grouped[record.category_name] = []
        grouped[record.category_name].append(record)
    return ordered_categories, grouped


def _count_images_by_category(
    records_by_category: dict[str, list[LVISInstanceRecord]],
) -> dict[str, int]:
    return {
        category_name: len({record.image_id for record in category_records})
        for category_name, category_records in records_by_category.items()
    }


def _allocate_category_selection_quotas(
    *,
    category_names: list[str],
    category_records_by_name: dict[str, dict[str, Any]],
    category_image_counts: dict[str, int],
    config: Stage1Config,
) -> dict[str, int]:
    text_pass_category_names = [
        category_name
        for category_name in category_names
        if category_records_by_name.get(category_name, {}).get("interactive_object")
    ]
    if not text_pass_category_names:
        return {}

    if config.reuse_existing_selection_quotas:
        quotas: dict[str, int] = {}
        for category_name in text_pass_category_names:
            quota = int(category_records_by_name.get(category_name, {}).get("selection_quota", 0))
            if quota <= 0:
                raise ValueError(
                    "reuse_existing_selection_quotas requires positive selection_quota "
                    f"for category '{category_name}'"
                )
            quotas[category_name] = quota
        return quotas

    target_num_candidates = config.target_num_candidates
    if target_num_candidates is None:
        return {category_name: 1 for category_name in text_pass_category_names}
    if target_num_candidates < len(text_pass_category_names):
        raise ValueError(
            "target_num_candidates must be at least the number of text-pass categories "
            f"({len(text_pass_category_names)})"
        )
    if config.quota_min_per_category < 0:
        raise ValueError("quota_min_per_category must be non-negative")

    base_total = config.quota_min_per_category * len(text_pass_category_names)
    if base_total > target_num_candidates:
        raise ValueError(
            "target_num_candidates is smaller than quota_min_per_category * num_text_pass_categories"
        )

    remaining = target_num_candidates - base_total
    quotas = {
        category_name: config.quota_min_per_category
        for category_name in text_pass_category_names
    }
    if remaining == 0:
        return quotas

    if config.quota_image_count_power < 0:
        raise ValueError("quota_image_count_power must be non-negative")

    weights = {
        category_name: float(max(category_image_counts.get(category_name, 0), 1))
        ** config.quota_image_count_power
        for category_name in text_pass_category_names
    }
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        weights = {category_name: 1.0 for category_name in text_pass_category_names}
        weight_sum = float(len(text_pass_category_names))

    raw_extras = {
        category_name: remaining * (weights[category_name] / weight_sum)
        for category_name in text_pass_category_names
    }
    floored_extras = {
        category_name: int(math.floor(raw_extras[category_name]))
        for category_name in text_pass_category_names
    }
    for category_name in text_pass_category_names:
        quotas[category_name] += floored_extras[category_name]

    leftover = remaining - sum(floored_extras.values())
    if leftover > 0:
        ranked = sorted(
            text_pass_category_names,
            key=lambda category_name: (
                raw_extras[category_name] - floored_extras[category_name],
                weights[category_name],
                category_name,
            ),
            reverse=True,
        )
        for category_name in ranked[:leftover]:
            quotas[category_name] += 1

    return quotas


def _materialize_record_assets(
    record: LVISInstanceRecord,
    *,
    object_dir: Path,
    split_name: str,
    object_id: str | None = None,
) -> None:
    source_dir = object_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(record.image_path) as image:
        image.convert("RGB").save(source_dir / "image.png")

    mask = decode_segmentation_mask(
        record.segmentation,
        int(record.image_height),
        int(record.image_width),
    )
    Image.fromarray((mask.astype(np.uint8) * 255)).save(source_dir / "instance_mask.png")

    source_meta = build_source_meta(record, split_name=split_name)
    if object_id is not None:
        source_meta["object_id"] = object_id
    with (source_dir / "source_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(source_meta, handle, indent=2, ensure_ascii=True)


def _stable_stage1_object_id(annotation_id: int) -> str:
    if annotation_id < 0:
        raise ValueError("annotation_id must be non-negative")
    return f"lvis_ann_{annotation_id:08d}"


def _stage1_progress_uses_stable_object_ids(progress_payload: dict[str, Any] | None) -> bool:
    if progress_payload is None:
        return False
    for item in progress_payload.get("candidate_records", []):
        object_id = str(item.get("object_id", ""))
        if object_id.startswith("lvis_ann_"):
            return True
    for item in progress_payload.get("category_records", []):
        for object_id in item.get("selected_object_ids", []):
            if str(object_id).startswith("lvis_ann_"):
                return True
        selected_object_id = item.get("selected_object_id")
        if str(selected_object_id or "").startswith("lvis_ann_"):
            return True
    return False


def _stage1_object_id_for_record(
    record: LVISInstanceRecord,
    *,
    use_stable_object_ids: bool,
) -> str:
    if not use_stable_object_ids:
        return record.object_id
    return _stable_stage1_object_id(int(record.annotation_id))


def _build_stage1_object_records_from_root(
    stage1_root: Path,
    *,
    candidate_records: list[dict[str, Any]],
    split_name: str,
) -> dict[str, Any]:
    records = []
    for item in candidate_records:
        object_id = str(item["object_id"])
        source_meta_path = stage1_root / object_id / "source" / "source_meta.json"
        source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
        records.append(
            {
                "object_id": object_id,
                "category_name": source_meta["category_name"],
                "annotation_id": source_meta["annotation_id"],
                "image_id": source_meta["image_id"],
                "object_dir": object_id,
                "source_meta_path": f"{object_id}/source/source_meta.json",
                "original_image_path": source_meta["original_image_path"],
            }
        )
    return {
        "split_name": split_name,
        "num_records": len(records),
        "records": records,
    }


def _evaluate_geometric_filter(
    record: LVISInstanceRecord,
    *,
    config: Stage1Config,
) -> dict[str, Any]:
    mask = decode_segmentation_mask(
        record.segmentation,
        int(record.image_height),
        int(record.image_width),
    ).astype(bool)
    image_h = int(record.image_height)
    image_w = int(record.image_width)
    image_area = max(image_h * image_w, 1)
    mask_pixels = int(mask.sum())
    mask_area_ratio = mask_pixels / float(image_area)

    ys, xs = np.where(mask)
    if len(xs) and len(ys):
        bbox_w = int(xs.max() - xs.min() + 1)
        bbox_h = int(ys.max() - ys.min() + 1)
    else:
        bbox_w = 0
        bbox_h = 0
    aspect_ratio = (
        max(bbox_w / max(bbox_h, 1), bbox_h / max(bbox_w, 1))
        if bbox_w and bbox_h
        else float("inf")
    )

    margin_px = max(
        int(round(min(image_h, image_w) * config.geometry_border_margin_ratio)),
        config.geometry_border_margin_min_px,
    )
    touches = {
        "top": bool(mask[:margin_px, :].any()),
        "bottom": bool(mask[-margin_px:, :].any()),
        "left": bool(mask[:, :margin_px].any()),
        "right": bool(mask[:, -margin_px:].any()),
    }
    border_touch_count = sum(1 for value in touches.values() if value)

    reasons: list[str] = []
    if mask_area_ratio < config.geometry_min_mask_area_ratio:
        reasons.append("mask_too_small")
    if aspect_ratio > config.geometry_max_mask_aspect_ratio:
        reasons.append("mask_aspect_too_large")
    if border_touch_count > config.geometry_max_border_touch_count:
        reasons.append("touches_too_many_borders")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "metrics": {
            "mask_area_ratio": mask_area_ratio,
            "mask_pixels": mask_pixels,
            "bbox_width": bbox_w,
            "bbox_height": bbox_h,
            "aspect_ratio": aspect_ratio,
            "border_margin_px": margin_px,
            "border_touch_count": border_touch_count,
            "touches": touches,
        },
    }


def _build_stage1_category_summary(
    stage1_root: Path,
    *,
    category_records: list[dict[str, Any]],
    num_input_records: int,
    num_categories_in_batch: int,
    num_text_pass_categories: int,
    num_selected_candidates: int,
) -> dict[str, Any]:
    return {
        "stage": "stage1_category_text_filter",
        "stage1_root": str(stage1_root),
        "num_input_records": num_input_records,
        "num_categories_in_batch": num_categories_in_batch,
        "num_text_pass_categories": num_text_pass_categories,
        "num_selected_candidates": num_selected_candidates,
        "records": category_records,
    }


def _stage1_progress_path(stage1_root: Path) -> Path:
    return stage1_root / "stage1_progress.json"


def _write_stage1_progress(
    stage1_root: Path,
    *,
    config: Stage1Config,
    status: str,
    num_input_records: int,
    num_categories_in_batch: int,
    category_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> str:
    completed_categories = [
        item["category_name"]
        for item in category_records
        if item.get("status") == "completed"
    ]
    payload = {
        "stage": "stage1_candidate_screening_progress",
        "status": status,
        "config": config.to_dict(),
        "num_input_records": num_input_records,
        "num_categories_in_batch": num_categories_in_batch,
        "num_text_pass_categories": sum(
            1 for item in category_records if item.get("interactive_object")
        ),
        "num_selected_candidates": len(candidate_records),
        "completed_categories": completed_categories,
        "category_records": category_records,
        "candidate_records": candidate_records,
    }
    progress_path = _stage1_progress_path(stage1_root)
    progress_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return str(progress_path)


def _load_stage1_progress(stage1_root: Path) -> dict[str, Any] | None:
    progress_path = _stage1_progress_path(stage1_root)
    if not progress_path.is_file():
        return None
    return json.loads(progress_path.read_text(encoding="utf-8"))


def _build_stage1_review_summary(stage1_root: Path, candidate_records_payload: dict[str, Any]) -> dict[str, Any]:
    records = []
    for item in candidate_records_payload["records"]:
        object_id = item["object_id"]
        records.append(
            {
                "object_id": object_id,
                "category_name": item["category_name"],
                "passed": item["passed"],
                "short_reason": item["decision"]["short_reason"],
                "reasons": item["reasons"],
                "masked_view_path": item["masked_view_path"],
                "source_image_path": f"{object_id}/source/image.png",
                "source_mask_path": f"{object_id}/source/instance_mask.png",
                "candidate_meta_path": f"{object_id}/candidate/candidate_meta.json",
            }
        )
    return {
        "stage": "stage1_candidate_screening",
        "stage1_root": str(stage1_root),
        "num_records": candidate_records_payload["num_selected_candidates"],
        "num_passed": candidate_records_payload["num_selected_candidates"],
        "num_text_pass_categories": candidate_records_payload["num_text_pass_categories"],
        "candidate_records_path": str(stage1_root / "candidate_records.json"),
        "category_filter_records_path": str(stage1_root / "category_filter_records.json"),
        "records": records,
    }


def _build_stage1_approval_template(stage1_root: Path, candidate_records_payload: dict[str, Any]) -> dict[str, Any]:
    approved_object_ids = [
        item["object_id"]
        for item in candidate_records_payload["records"]
        if item["passed"]
    ]
    return {
        "stage": "stage1_review_gate",
        "stage1_root": str(stage1_root),
        "candidate_records_path": str(stage1_root / "candidate_records.json"),
        "approved_object_ids": approved_object_ids,
        "notes": "Review candidate records and edit approved_object_ids before running stage2.",
    }


def run_stage1_candidate_screening(
    config: Stage1Config,
    *,
    client=None,
    category_client: InteractiveCategoryInferenceClient | None = None,
) -> dict[str, Any]:
    stage1_root = Path(config.stage1_root)
    _ensure_clean_root(stage1_root, overwrite=config.overwrite)

    extraction_config = InstanceExtractionConfig(
        annotation_json_path=config.annotation_json_path,
        images_root=config.images_root,
        split_name=config.split_name,
        allowed_category_names=config.allowed_category_names,
    )
    annotation_data = load_lvis_json(extraction_config.annotation_json_path)
    records = extract_lvis_records(
        annotation_data,
        images_root=extraction_config.images_root,
        allowed_category_names=extraction_config.allowed_category_names,
    )
    if not records:
        raise ValueError("no records selected for stage1")

    ordered_categories, records_by_category = _group_records_by_category(records)
    category_image_counts = _count_images_by_category(records_by_category)
    category_names = _slice_items(
        ordered_categories,
        start_index=config.start_index,
        limit=config.limit,
    )
    if not category_names:
        raise ValueError("no category names selected for stage1")

    _progress(
        f"[stage1] loaded {len(records)} records across {len(ordered_categories)} categories; "
        f"current batch has {len(category_names)} categories"
    )

    category_client = category_client or GeminiInteractiveCategoryClient()
    client = client or GeminiCandidateFilterClient(config.gemini_config)
    progress_payload = _load_stage1_progress(stage1_root)
    use_stable_object_ids = _stage1_progress_uses_stable_object_ids(progress_payload)
    if progress_payload is not None:
        category_filter_records = list(progress_payload.get("category_records", []))
        candidate_records = list(progress_payload.get("candidate_records", []))
        completed_categories = set(progress_payload.get("completed_categories", []))
        _progress(
            f"[stage1] resume detected: {len(completed_categories)}/{len(category_names)} categories already completed"
        )
    else:
        category_filter_records = []
        candidate_records = []
        completed_categories = set()

    category_records_by_name = {
        item["category_name"]: item for item in category_filter_records
    }
    for category_name in category_names:
        existing = category_records_by_name.get(category_name)
        if existing is None:
            continue
        selected_object_ids = list(existing.get("selected_object_ids", []))
        if not selected_object_ids and existing.get("selected_object_id"):
            selected_object_ids = [existing["selected_object_id"]]
        existing["selected_object_ids"] = selected_object_ids
        existing["selected_object_id"] = selected_object_ids[0] if selected_object_ids else None
        existing["num_selected_objects"] = len(selected_object_ids)
        existing["num_available_objects"] = len(records_by_category[category_name])
        existing["num_available_images"] = category_image_counts[category_name]
        existing.setdefault("selection_quota", 0)

    for idx, category_name in enumerate(category_names, start=1):
        if category_name in category_records_by_name:
            _progress(
                f"[stage1][text {idx}/{len(category_names)}] reuse cached text decision for '{category_name}'"
            )
            continue

        _progress(
            f"[stage1][text {idx}/{len(category_names)}] screening category '{category_name}'"
        )
        category_decision = category_client.infer(
            category_name=category_name,
            config=config.gemini_config,
        )
        category_record = {
            "category_name": category_name,
            "interactive_object": category_decision.interactive_object,
            "short_reason": category_decision.short_reason,
            "num_available_objects": len(records_by_category[category_name]),
            "num_available_images": category_image_counts[category_name],
            "selection_quota": 0,
            "selected_object_id": None,
            "selected_object_ids": [],
            "num_selected_objects": 0,
            "num_image_attempts": 0,
            "num_missing_images": 0,
            "num_geometry_rejects": 0,
            "num_llm_rejects": 0,
            "status": "text_completed" if category_decision.interactive_object else "completed",
        }
        category_records_by_name[category_name] = category_record
        category_filter_records.append(category_record)
        _progress(
            f"[stage1][text {idx}/{len(category_names)}] "
            f"{'keep' if category_decision.interactive_object else 'drop'} '{category_name}': "
            f"{category_decision.short_reason}"
        )
        _write_stage1_progress(
            stage1_root,
            config=config,
            status="running",
            num_input_records=len(records),
            num_categories_in_batch=len(category_names),
            category_records=category_filter_records,
            candidate_records=candidate_records,
        )

    category_selection_quotas = _allocate_category_selection_quotas(
        category_names=category_names,
        category_records_by_name=category_records_by_name,
        category_image_counts=category_image_counts,
        config=config,
    )
    for category_name in category_names:
        category_record = category_records_by_name[category_name]
        category_record["selection_quota"] = category_selection_quotas.get(category_name, 0)
        if category_record.get("interactive_object"):
            selected_object_ids = list(category_record.get("selected_object_ids", []))
            if len(selected_object_ids) >= int(category_record["selection_quota"]):
                category_record["status"] = "completed"
                completed_categories.add(category_name)
            else:
                category_record["status"] = "text_completed"
                completed_categories.discard(category_name)

    _write_stage1_progress(
        stage1_root,
        config=config,
        status="running",
        num_input_records=len(records),
        num_categories_in_batch=len(category_names),
        category_records=category_filter_records,
        candidate_records=candidate_records,
    )

    text_pass_category_names = [
        category_name
        for category_name in category_names
        if category_records_by_name.get(category_name, {}).get("interactive_object")
    ]
    candidate_record_ids = {
        str(item["object_id"])
        for item in candidate_records
    }
    for interactive_position, category_name in enumerate(text_pass_category_names, start=1):
        category_record = category_records_by_name[category_name]
        if category_name in completed_categories:
            _progress(
                f"[stage1][image {interactive_position}/{len(text_pass_category_names)}] skip completed category '{category_name}'"
            )
            continue

        records_for_category = records_by_category[category_name]
        selection_quota = int(category_record.get("selection_quota", 0))
        selected_object_ids = list(category_record.get("selected_object_ids", []))
        _progress(
            f"[stage1][image {interactive_position}/{len(text_pass_category_names)}] "
            f"screening category '{category_name}' with {len(records_for_category)} objects; "
            f"target quota {selection_quota}"
        )

        for attempt_idx, record in enumerate(records_for_category, start=1):
            if len(selected_object_ids) >= selection_quota:
                break

            category_record["num_image_attempts"] = attempt_idx
            stage1_object_id = _stage1_object_id_for_record(
                record,
                use_stable_object_ids=use_stable_object_ids,
            )
            if stage1_object_id in selected_object_ids:
                continue
            if not Path(record.image_path).is_file():
                category_record["num_missing_images"] += 1
                _progress(
                    f"[stage1][image {interactive_position}/{len(text_pass_category_names)}]"
                    f"[try {attempt_idx}/{len(records_for_category)}] "
                    f"{record.object_id} -> skip missing image"
                )
                continue

            geometry = _evaluate_geometric_filter(record, config=config)
            if not geometry["passed"]:
                category_record["num_geometry_rejects"] += 1
                _progress(
                    f"[stage1][image {interactive_position}/{len(text_pass_category_names)}]"
                    f"[try {attempt_idx}/{len(records_for_category)}] "
                    f"{record.object_id} -> reject geometry: {', '.join(geometry['reasons'])}"
                )
                continue

            object_dir = stage1_root / stage1_object_id
            if object_dir.exists():
                shutil.rmtree(object_dir)

            _materialize_record_assets(
                record,
                object_dir=object_dir,
                split_name=config.split_name,
                object_id=stage1_object_id,
            )
            source_dir = object_dir / "source"
            candidate_dir = object_dir / "candidate"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            masked_view_path = candidate_dir / "masked_instance.png"
            create_masked_instance_view(
                source_dir / "image.png",
                source_dir / "instance_mask.png",
                masked_view_path,
            )
            decision = client.infer(
                category_name=record.category_name,
                masked_image_path=masked_view_path,
                original_image_path=source_dir / "image.png",
            )
            candidate_record = {
                "object_id": stage1_object_id,
                "category_name": record.category_name,
                "model_name": config.gemini_config.model_name,
                "passed": decision.passed,
                "reasons": list(decision_to_reasons(decision)),
                "geometry_filter": geometry,
                "decision": decision.to_dict(),
                "masked_view_path": f"{stage1_object_id}/candidate/masked_instance.png",
                "original_image_path": f"{stage1_object_id}/source/image.png",
                "image_attempt_index": attempt_idx,
                "selection_slot_index": len(selected_object_ids) + 1,
                "selection_quota": selection_quota,
                "used_llm_quality_check": True,
            }
            with (candidate_dir / "candidate_meta.json").open("w", encoding="utf-8") as handle:
                json.dump(candidate_record, handle, indent=2, ensure_ascii=True)

            _progress(
                f"[stage1][image {interactive_position}/{len(text_pass_category_names)}]"
                f"[try {attempt_idx}/{len(records_for_category)}] "
                f"{record.object_id} -> {'pass' if decision.passed else 'reject'}"
            )
            if decision.passed:
                if stage1_object_id not in candidate_record_ids:
                    candidate_records.append(candidate_record)
                    candidate_record_ids.add(stage1_object_id)
                selected_object_ids.append(stage1_object_id)
                category_record["selected_object_ids"] = selected_object_ids
                category_record["selected_object_id"] = selected_object_ids[0]
                category_record["num_selected_objects"] = len(selected_object_ids)
                continue

            category_record["num_llm_rejects"] += 1
            shutil.rmtree(object_dir)

        category_record["selected_object_ids"] = selected_object_ids
        category_record["selected_object_id"] = selected_object_ids[0] if selected_object_ids else None
        category_record["num_selected_objects"] = len(selected_object_ids)
        category_record["status"] = "completed"
        completed_categories.add(category_name)
        _write_stage1_progress(
            stage1_root,
            config=config,
            status="running",
            num_input_records=len(records),
            num_categories_in_batch=len(category_names),
            category_records=category_filter_records,
            candidate_records=candidate_records,
        )

    num_text_pass_categories = sum(
        1 for item in category_filter_records if item.get("interactive_object")
    )

    object_records = _build_stage1_object_records_from_root(
        stage1_root,
        candidate_records=candidate_records,
        split_name=config.split_name,
    )
    (stage1_root / "object_records.json").write_text(
        json.dumps(object_records, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    category_summary = _build_stage1_category_summary(
        stage1_root,
        category_records=category_filter_records,
        num_input_records=len(records),
        num_categories_in_batch=len(category_names),
        num_text_pass_categories=num_text_pass_categories,
        num_selected_candidates=len(candidate_records),
    )
    (stage1_root / "category_filter_records.json").write_text(
        json.dumps(category_summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    candidate_summary = {
        "model_name": config.gemini_config.model_name,
        "num_selected_candidates": len(candidate_records),
        "num_text_pass_categories": num_text_pass_categories,
        "records": candidate_records,
    }
    (stage1_root / "candidate_records.json").write_text(
        json.dumps(candidate_summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    review_summary = _build_stage1_review_summary(stage1_root, candidate_summary)
    (stage1_root / "review_summary.json").write_text(
        json.dumps(review_summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    approval_template = _build_stage1_approval_template(stage1_root, candidate_summary)
    (stage1_root / "approval_template.json").write_text(
        json.dumps(approval_template, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    stage1_meta = {
        "stage": "stage1_candidate_screening",
        "config": config.to_dict(),
        "num_input_records": len(records),
        "num_categories_in_batch": len(category_names),
        "num_text_pass_categories": num_text_pass_categories,
        "num_selected_candidates": len(candidate_records),
        "artifacts": {
            "object_records_path": str(stage1_root / "object_records.json"),
            "category_filter_records_path": str(stage1_root / "category_filter_records.json"),
            "candidate_records_path": str(stage1_root / "candidate_records.json"),
            "review_summary_path": str(stage1_root / "review_summary.json"),
            "approval_template_path": str(stage1_root / "approval_template.json"),
        },
    }
    (stage1_root / "stage1_meta.json").write_text(
        json.dumps(stage1_meta, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    _write_stage1_progress(
        stage1_root,
        config=config,
        status="completed",
        num_input_records=len(records),
        num_categories_in_batch=len(category_names),
        category_records=category_filter_records,
        candidate_records=candidate_records,
    )
    return {
        "stage1_root": str(stage1_root),
        "num_input_records": len(records),
        "num_categories_in_batch": len(category_names),
        "num_text_pass_categories": num_text_pass_categories,
        "num_selected_candidates": len(candidate_records),
        "review_summary_path": str(stage1_root / "review_summary.json"),
        "approval_template_path": str(stage1_root / "approval_template.json"),
    }


def _load_approved_object_ids(approval_json_path: Path) -> list[str]:
    payload = json.loads(approval_json_path.read_text(encoding="utf-8"))
    approved_object_ids = payload.get("approved_object_ids", [])
    if not isinstance(approved_object_ids, list) or not approved_object_ids:
        raise ValueError("approval file contains no approved_object_ids")
    return [str(item) for item in approved_object_ids]


def _copy_candidate_object(src_object_dir: Path, dst_object_dir: Path, *, overwrite: bool) -> None:
    if dst_object_dir.exists() and overwrite:
        shutil.rmtree(dst_object_dir)
    dst_object_dir.mkdir(parents=True, exist_ok=True)
    for dirname in ["source", "candidate"]:
        src_dir = src_object_dir / dirname
        dst_dir = dst_object_dir / dirname
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)


def _copy_prefetched_prompt_artifacts(
    prefetched_prompt_root: Path | None,
    *,
    object_id: str,
    dst_object_dir: Path,
) -> bool:
    if prefetched_prompt_root is None:
        return False
    src_prompt_dir = prefetched_prompt_root / "objects" / object_id / "prompt"
    if not src_prompt_dir.is_dir():
        return False
    dst_prompt_dir = dst_object_dir / "prompt"
    if dst_prompt_dir.exists():
        shutil.rmtree(dst_prompt_dir)
    shutil.copytree(src_prompt_dir, dst_prompt_dir)
    return True


def _has_reusable_part_prompt(dst_object_dir: Path, *, prompt_filename: str) -> bool:
    prompt_path = dst_object_dir / "prompt" / prompt_filename
    if not prompt_path.is_file():
        return False
    try:
        payload = json.loads(prompt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and isinstance(payload.get("parts", []), list)


def _stage2_object_matches_fusion_config(
    dst_object_dir: Path,
    *,
    fusion_config: FusionConfig,
) -> bool:
    fusion_meta_path = dst_object_dir / "label3d" / fusion_config.fusion_meta_filename
    if not fusion_meta_path.is_file():
        return False
    try:
        payload = json.loads(fusion_meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if payload.get("status") != "completed":
        return False
    recorded_config = payload.get("fusion_config")
    if not isinstance(recorded_config, dict):
        return False
    return recorded_config == fusion_config.to_dict()


def _stage2_object_is_complete(
    dst_object_dir: Path,
    *,
    fusion_config: FusionConfig,
    packaging_config: PackagingConfig,
) -> bool:
    if not (dst_object_dir / "package" / "part_instances.json").is_file():
        return False
    if not (dst_object_dir / "label3d" / fusion_config.part_membership_filename).is_file():
        return False
    if not (dst_object_dir / "label3d" / fusion_config.unknown_mask_filename).is_file():
        return False
    if not _stage2_object_matches_fusion_config(
        dst_object_dir,
        fusion_config=fusion_config,
    ):
        return False
    if packaging_config.colored_ply_enabled:
        if not (dst_object_dir / "project3d" / packaging_config.colored_ply_filename).is_file():
            return False
    if packaging_config.visualization_enabled:
        prompt_path = dst_object_dir / "prompt" / "part_prompts.json"
        summary_path = dst_object_dir / packaging_config.visualization_output_dirname / "part_label_rotation_summary.json"
        if prompt_path.is_file():
            try:
                prompt_payload = json.loads(prompt_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return False
            if prompt_payload.get("parts") and not summary_path.is_file():
                return False
    return True


def _aggregate_stage2_dataset(stage2_root: Path, completed_object_ids: list[str], failed_objects: list[dict[str, str]]) -> dict[str, Any]:
    objects_root = stage2_root / "objects"
    object_manifests = []
    all_part_samples = []
    for object_id in completed_object_ids:
        object_dir = objects_root / object_id
        manifest_path = object_dir / "package" / "object_manifest.json"
        part_instances_path = object_dir / "package" / "part_instances.json"
        stats_path = object_dir / "package" / "stats_report.json"
        object_manifests.append(
            {
                "object_id": object_id,
                "object_dir": str(object_dir),
                "object_manifest_path": str(manifest_path),
                "part_instances_path": str(part_instances_path),
                "stats_report_path": str(stats_path),
            }
        )
        all_part_samples.extend(json.loads(part_instances_path.read_text(encoding="utf-8")))

    dataset_index = {
        "stage": "stage2_dataset_build",
        "num_objects": len(completed_object_ids),
        "num_part_samples": len(all_part_samples),
        "objects": object_manifests,
        "part_samples": all_part_samples,
    }
    (stage2_root / "dataset_index.json").write_text(
        json.dumps(dataset_index, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    build_summary = {
        "num_completed": len(completed_object_ids),
        "num_failed": len(failed_objects),
        "completed_object_ids": completed_object_ids,
        "failed_objects": failed_objects,
        "dataset_index_path": str(stage2_root / "dataset_index.json"),
    }
    (stage2_root / "build_summary.json").write_text(
        json.dumps(build_summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (stage2_root / "failures.json").write_text(
        json.dumps({"failed_objects": failed_objects}, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return build_summary


def run_stage2_dataset_build(
    config: Stage2Config,
    *,
    reconstruction_executor=None,
    render_executor=None,
    part_prompt_client=None,
    segmentation_executor=None,
    lift_executor=None,
    fusion_executor=None,
    dit_part_segmentation_executor=None,
    packaging_executor=None,
) -> dict[str, Any]:
    os.environ.setdefault("HF_ENDPOINT", config.hf_endpoint)

    stage1_root = Path(config.stage1_root)
    stage2_root = Path(config.stage2_root)
    approval_json_path = Path(config.approval_json_path)
    prefetched_prompt_root = Path(config.prefetched_prompt_root).resolve() if config.prefetched_prompt_root else None
    if not stage1_root.is_dir():
        raise FileNotFoundError(f"missing stage1 root: {stage1_root}")
    if not approval_json_path.is_file():
        raise FileNotFoundError(f"missing approval json: {approval_json_path}")
    if prefetched_prompt_root is not None and not prefetched_prompt_root.is_dir():
        raise FileNotFoundError(f"missing prefetched prompt root: {prefetched_prompt_root}")

    if stage2_root.exists() and config.overwrite:
        shutil.rmtree(stage2_root)
    objects_root = stage2_root / "objects"
    objects_root.mkdir(parents=True, exist_ok=True)

    approved_object_ids = _load_approved_object_ids(approval_json_path)
    if config.max_objects is not None:
        approved_object_ids = approved_object_ids[: config.max_objects]

    reconstruction_executor = reconstruction_executor or LocalSam3DObjectsExecutor()
    render_executor = render_executor or LocalRenderExecutor()
    part_prompt_client = part_prompt_client or GeminiPartPromptClient()
    segmentation_executor = segmentation_executor or LocalSam3SegmentationExecutor()
    lift_executor = lift_executor or LocalLiftExecutor()
    fusion_executor = fusion_executor or LocalFusionExecutor()
    if dit_part_segmentation_executor is None:
        if config.dit_part_segmentation_config.backend == "seed3d_partseg_repro":
            dit_part_segmentation_executor = Seed3DReproDitPartSegmentationExecutor()
        elif config.dit_part_segmentation_config.backend == "label3d_adapter":
            dit_part_segmentation_executor = Label3DAdapterDitPartSegmentationExecutor()
        else:
            dit_part_segmentation_executor = ExternalDitPartSegmentationExecutor()
    packaging_executor = packaging_executor or LocalPackagingExecutor()

    completed_object_ids: list[str] = []
    failed_objects: list[dict[str, str]] = []
    for idx, object_id in enumerate(approved_object_ids, start=1):
        _progress(f"[stage2][{idx}/{len(approved_object_ids)}] building object '{object_id}'")
        src_object_dir = stage1_root / object_id
        dst_object_dir = objects_root / object_id
        if not src_object_dir.is_dir():
            failed_objects.append({"object_id": object_id, "error": "missing_stage1_object_dir"})
            continue
        if dst_object_dir.exists() and config.skip_existing and _stage2_object_is_complete(
            dst_object_dir,
            fusion_config=config.fusion_config,
            packaging_config=config.packaging_config,
        ):
            _progress(f"[stage2][{idx}/{len(approved_object_ids)}] skip existing '{object_id}'")
            completed_object_ids.append(object_id)
            continue
        if dst_object_dir.exists() and config.skip_existing:
            _progress(
                f"[stage2][{idx}/{len(approved_object_ids)}] rebuild stale or incomplete '{object_id}'"
            )

        try:
            _copy_candidate_object(
                src_object_dir,
                dst_object_dir,
                overwrite=config.overwrite or dst_object_dir.exists(),
            )
            _copy_prefetched_prompt_artifacts(
                prefetched_prompt_root,
                object_id=object_id,
                dst_object_dir=dst_object_dir,
            )
            source_meta = json.loads((dst_object_dir / "source" / "source_meta.json").read_text(encoding="utf-8"))
            category_name = source_meta["category_name"]

            recon_request = build_reconstruction_request(dst_object_dir, config=config.reconstruction_config)
            execute_reconstruction(
                recon_request,
                config=config.reconstruction_config,
                executor=reconstruction_executor,
            )

            render_request = build_render_request(dst_object_dir, config=config.render_config)
            execute_render(
                render_request,
                config=config.render_config,
                executor=render_executor,
            )

            if _has_reusable_part_prompt(
                dst_object_dir,
                prompt_filename=config.part_prompt_config.prompt_filename,
            ):
                _progress(f"[stage2][{idx}/{len(approved_object_ids)}] reuse prompt '{object_id}'")
            else:
                execute_part_prompt_generation(
                    dst_object_dir,
                    object_id=object_id,
                    object_name=category_name,
                    config=config.part_prompt_config,
                    client=part_prompt_client,
                )

            seg_request = build_segmentation_request(dst_object_dir, config=config.segmentation_config)
            execute_segmentation(
                seg_request,
                config=config.segmentation_config,
                executor=segmentation_executor,
            )

            lift_request = build_lift_request(dst_object_dir, config=config.lift_config)
            execute_lift(
                lift_request,
                config=config.lift_config,
                executor=lift_executor,
            )

            fusion_request = build_fusion_request(dst_object_dir, config=config.fusion_config)
            execute_fusion(
                fusion_request,
                config=config.fusion_config,
                executor=fusion_executor,
            )

            if config.packaging_config.part_label_source == config.dit_part_segmentation_config.output_dirname:
                dit_request = build_dit_part_segmentation_request(
                    dst_object_dir,
                    config=config.dit_part_segmentation_config,
                )
                execute_dit_part_segmentation(
                    dit_request,
                    config=config.dit_part_segmentation_config,
                    executor=dit_part_segmentation_executor,
                )

            packaging_request = build_packaging_request(dst_object_dir, config=config.packaging_config)
            execute_packaging(
                packaging_request,
                config=config.packaging_config,
                executor=packaging_executor,
            )
            completed_object_ids.append(object_id)
            _progress(f"[stage2][{idx}/{len(approved_object_ids)}] completed '{object_id}'")
        except Exception as exc:  # noqa: BLE001
            failed_objects.append({"object_id": object_id, "error": str(exc)})
            _progress(f"[stage2][{idx}/{len(approved_object_ids)}] failed '{object_id}': {exc}")

    build_summary = _aggregate_stage2_dataset(stage2_root, completed_object_ids, failed_objects)
    stage2_meta = {
        "stage": "stage2_dataset_build",
        "config": config.to_dict(),
        "approved_object_ids": approved_object_ids,
        "build_summary_path": str(stage2_root / "build_summary.json"),
        "dataset_index_path": str(stage2_root / "dataset_index.json"),
    }
    (stage2_root / "stage2_meta.json").write_text(
        json.dumps(stage2_meta, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return build_summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the two-stage real LVIS dataset pipeline.")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    stage1 = subparsers.add_parser(
        "stage1",
        help="text-filter interactive categories, then keep image-qualified objects per category based on quota",
    )
    stage1.add_argument("--run-name", default="default")
    stage1.add_argument("--annotation-json", default=DEFAULT_ANNOTATION_JSON)
    stage1.add_argument("--images-root", default=DEFAULT_IMAGES_ROOT)
    stage1.add_argument("--split-name", default="lvis_val")
    stage1.add_argument("--start-index", type=int, default=0)
    stage1.add_argument("--limit", type=int)
    stage1.add_argument("--target-num-candidates", type=int)
    stage1.add_argument("--quota-image-count-power", type=float, default=0.5)
    stage1.add_argument("--quota-min-per-category", type=int, default=1)
    stage1.add_argument("--reuse-existing-selection-quotas", action="store_true")
    stage1.add_argument("--overwrite", action="store_true")
    stage1.add_argument("--allowed-categories", nargs="*", default=[])
    stage1.add_argument("--allowed-categories-file")

    stage2 = subparsers.add_parser("stage2", help="build the final dataset from approved stage1 candidates")
    stage2.add_argument("--run-name", default="default")
    stage2.add_argument("--approval-json")
    stage2.add_argument("--prefetched-prompt-root")
    stage2.add_argument("--overwrite", action="store_true")
    stage2.add_argument("--no-skip-existing", action="store_true")
    stage2.add_argument("--max-objects", type=int)
    stage2.add_argument("--reconstruction-seed", type=int, default=42)
    stage2.add_argument(
        "--part-label-source",
        choices=["label3d", "dit_partseg", "dit_seed3d_partseg"],
        default="label3d",
        help="Use legacy projected labels or DIT-style part segmentation outputs for packaging.",
    )
    stage2.add_argument(
        "--dit-command-template",
        default="",
        help="External Seed3D-style PartSeg/PartDiT command template for --part-label-source dit_partseg.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_root = _default_run_root(args.run_name)

    if args.stage == "stage1":
        allowed_categories = list(args.allowed_categories)
        if args.allowed_categories_file:
            allowed_path = Path(args.allowed_categories_file)
            raw_text = allowed_path.read_text(encoding="utf-8").strip()
            if raw_text:
                if raw_text.startswith("["):
                    payload = json.loads(raw_text)
                    if not isinstance(payload, list):
                        raise ValueError("allowed categories file JSON must be a list")
                    allowed_categories.extend(str(item) for item in payload)
                else:
                    allowed_categories.extend(
                        line.strip()
                        for line in raw_text.splitlines()
                        if line.strip()
                    )
        result = run_stage1_candidate_screening(
            Stage1Config(
                annotation_json_path=args.annotation_json,
                images_root=args.images_root,
                split_name=args.split_name,
                stage1_root=str(run_root / "stage1_candidates"),
                allowed_category_names=tuple(allowed_categories),
                start_index=args.start_index,
                limit=args.limit,
                target_num_candidates=args.target_num_candidates,
                quota_image_count_power=args.quota_image_count_power,
                quota_min_per_category=args.quota_min_per_category,
                reuse_existing_selection_quotas=args.reuse_existing_selection_quotas,
                overwrite=args.overwrite,
            )
        )
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0

    approval_json = args.approval_json or str(run_root / "stage1_candidates" / "approval_template.json")
    result = run_stage2_dataset_build(
        Stage2Config(
            stage1_root=str(run_root / "stage1_candidates"),
            stage2_root=str(run_root / "stage2_dataset"),
            approval_json_path=approval_json,
            prefetched_prompt_root=args.prefetched_prompt_root,
            overwrite=args.overwrite,
            skip_existing=not args.no_skip_existing,
            max_objects=args.max_objects,
            reconstruction_config=ReconstructionConfig(
                sam3d_root=DEFAULT_SAM3D_ROOT,
                checkpoint_tag="hf",
                conda_env_path=DEFAULT_SAM3D_CONDA_ENV,
                seed=args.reconstruction_seed,
            ),
            dit_part_segmentation_config=DitPartSegmentationConfig(
                backend="external_seed3d_partdit"
                if args.dit_command_template
                else "seed3d_partseg_repro"
                if args.part_label_source in {"dit_partseg", "dit_seed3d_partseg"}
                else "label3d_adapter",
                output_dirname=args.part_label_source
                if args.part_label_source in {"dit_partseg", "dit_seed3d_partseg"}
                else "dit_partseg",
                command_template=args.dit_command_template,
            ),
            packaging_config=PackagingConfig(part_label_source=args.part_label_source),
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
