from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image

from research.pipeline.common.segmentation import decode_segmentation_mask
from research.pipeline.module_m0_data_spec.data_spec import build_object_id


@dataclass(frozen=True)
class InstanceExtractionConfig:
    annotation_json_path: str
    images_root: str
    split_name: str = "lvis"
    allowed_category_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class LVISInstanceRecord:
    object_id: str
    annotation_id: int
    image_id: int
    category_id: int
    category_name: str
    image_path: str
    image_width: int
    image_height: int
    bbox_xywh: list[float]
    area: float
    segmentation: list
    object_dir_name: str
    source_image_relpath: str
    source_mask_relpath: str
    source_meta_relpath: str

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "annotation_id": self.annotation_id,
            "image_id": self.image_id,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "image_path": self.image_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "bbox_xywh": self.bbox_xywh,
            "area": self.area,
            "segmentation": self.segmentation,
            "object_dir_name": self.object_dir_name,
            "source": {
                "image": self.source_image_relpath,
                "instance_mask": self.source_mask_relpath,
                "meta": self.source_meta_relpath,
            },
        }


def load_lvis_json(annotation_json_path: str | Path) -> dict:
    path = Path(annotation_json_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_source_meta(
    record: LVISInstanceRecord,
    *,
    split_name: str = "lvis",
) -> dict:
    segmentation = record.segmentation
    num_polygons = len(segmentation) if isinstance(segmentation, list) else 0
    first_polygon_points = 0
    if num_polygons and isinstance(segmentation[0], list):
        first_polygon_points = len(segmentation[0]) // 2

    return {
        "object_id": record.object_id,
        "split_name": split_name,
        "annotation_id": record.annotation_id,
        "image_id": record.image_id,
        "category_id": record.category_id,
        "category_name": record.category_name,
        "original_image_path": record.image_path,
        "image_size": {
            "width": record.image_width,
            "height": record.image_height,
        },
        "bbox_xywh": record.bbox_xywh,
        "area": record.area,
        "segmentation_summary": {
            "num_polygons": num_polygons,
            "first_polygon_points": first_polygon_points,
        },
        "planned_source_paths": {
            "image": record.source_image_relpath,
            "instance_mask": record.source_mask_relpath,
            "meta": record.source_meta_relpath,
        },
    }


def extract_lvis_records(
    annotation_data: dict,
    *,
    images_root: str | Path,
    allowed_category_names: tuple[str, ...] = (),
    limit: int | None = None,
) -> list[LVISInstanceRecord]:
    images_by_id = {image["id"]: image for image in annotation_data.get("images", [])}
    categories_by_id = {
        category["id"]: category for category in annotation_data.get("categories", [])
    }
    allowed_names = {name.lower() for name in allowed_category_names}

    records: list[LVISInstanceRecord] = []
    for annotation in annotation_data.get("annotations", []):
        category = categories_by_id.get(annotation["category_id"])
        image = images_by_id.get(annotation["image_id"])
        if category is None or image is None:
            continue

        category_name = category["name"]
        if allowed_names and category_name.lower() not in allowed_names:
            continue

        image_file_name = image.get("coco_url", "").rstrip("/").split("/")[-1]
        if not image_file_name:
            image_file_name = f"{image['id']:012d}.jpg"
        image_path = str(Path(images_root) / image_file_name)

        object_id = build_object_id(len(records))
        record = LVISInstanceRecord(
            object_id=object_id,
            annotation_id=annotation["id"],
            image_id=annotation["image_id"],
            category_id=annotation["category_id"],
            category_name=category_name,
            image_path=image_path,
            image_width=image["width"],
            image_height=image["height"],
            bbox_xywh=list(annotation.get("bbox", [])),
            area=float(annotation.get("area", 0.0)),
            segmentation=list(annotation.get("segmentation", [])),
            object_dir_name=object_id,
            source_image_relpath="source/image.png",
            source_mask_relpath="source/instance_mask.png",
            source_meta_relpath="source/source_meta.json",
        )
        records.append(record)
        if limit is not None and len(records) >= limit:
            break

    return records


def export_object_records(
    records: list[LVISInstanceRecord],
    *,
    export_root: str | Path,
    split_name: str = "lvis",
) -> dict:
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    object_index = []
    for record in records:
        object_dir = export_root / record.object_id
        source_dir = object_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)

        source_meta = build_source_meta(record, split_name=split_name)
        source_meta_path = source_dir / "source_meta.json"
        with source_meta_path.open("w", encoding="utf-8") as handle:
            json.dump(source_meta, handle, indent=2, ensure_ascii=True)

        object_index.append(
            {
                "object_id": record.object_id,
                "category_name": record.category_name,
                "annotation_id": record.annotation_id,
                "image_id": record.image_id,
                "object_dir": record.object_id,
                "source_meta_path": f"{record.object_id}/source/source_meta.json",
                "original_image_path": record.image_path,
            }
        )

    index_payload = {
        "split_name": split_name,
        "num_records": len(object_index),
        "records": object_index,
    }
    index_path = export_root / "object_records.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index_payload, handle, indent=2, ensure_ascii=True)

    return {
        "export_root": str(export_root),
        "index_path": str(index_path),
        "num_records": len(object_index),
    }


def _segmentation_to_mask(record: LVISInstanceRecord) -> np.ndarray:
    mask = decode_segmentation_mask(
        record.segmentation,
        int(record.image_height),
        int(record.image_width),
    )
    return mask * 255


def materialize_source_assets(
    records: list[LVISInstanceRecord],
    *,
    export_root: str | Path,
) -> dict:
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    written = []
    for record in records:
        object_dir = export_root / record.object_id
        source_dir = object_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)

        src_image_path = Path(record.image_path)
        dst_image_path = source_dir / "image.png"
        with Image.open(src_image_path) as image:
            image.convert("RGB").save(dst_image_path)

        mask = _segmentation_to_mask(record)
        dst_mask_path = source_dir / "instance_mask.png"
        Image.fromarray(mask).save(dst_mask_path)

        written.append(
            {
                "object_id": record.object_id,
                "image_path": str(dst_image_path),
                "mask_path": str(dst_mask_path),
            }
        )

    return {
        "export_root": str(export_root),
        "num_records": len(written),
        "written_assets": written,
    }
