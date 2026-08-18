from __future__ import annotations

import json
from pathlib import Path
import shutil

from research.pipeline.module_m1_instance_extraction.instance_extraction import (
    export_object_records,
    extract_lvis_records,
    load_lvis_json,
    materialize_source_assets,
)


def prepare_live_chain_object(
    root: str | Path,
    *,
    category_name: str = "chair",
    split_name: str = "lvis_val",
) -> tuple[Path, dict]:
    root = Path(root)
    artifact_root = root / "pipeline" / "outputs" / "artifacts" / "live_chain"
    object_dir = artifact_root / "object_000000"
    if object_dir.exists():
        shutil.rmtree(object_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)

    annotation_path = root / "dataset" / "lvis" / "valid" / "annotation" / "lvis_v1_val.json"
    images_root = root / "dataset" / "lvis" / "valid" / "image" / "val2017"
    data = load_lvis_json(annotation_path)
    records = extract_lvis_records(
        data,
        images_root=images_root,
        allowed_category_names=(category_name,),
        limit=1,
    )
    if not records:
        raise RuntimeError(f"no LVIS record found for category: {category_name}")

    export_object_records(records, export_root=artifact_root, split_name=split_name)
    materialize_source_assets(records, export_root=artifact_root)

    source_meta_path = object_dir / "source" / "source_meta.json"
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    return object_dir, source_meta
