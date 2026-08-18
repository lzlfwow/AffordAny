from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Protocol

from repo_layout import default_sam3_env_path, default_sam3d_objects_env_path


DEFAULT_SAM3_CONDA_ENV = default_sam3_env_path(
    fallback=default_sam3d_objects_env_path()
)


@dataclass(frozen=True)
class SegmentationConfig:
    model_name: str = "sam3"
    conda_env_path: str = DEFAULT_SAM3_CONDA_ENV
    checkpoint_path: str = ""
    image_ext: str = ".png"
    seg_meta_filename: str = "seg_meta.json"
    low_confidence_threshold: float = 0.1
    model_confidence_threshold: float = 0.5
    max_selected_parts: int = 3
    max_masks_per_part: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SegmentationTarget:
    view_id: str
    image_path: str
    part_name: str
    prompt: str
    mask_path: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SegmentationTargetResult:
    view_id: str
    part_name: str
    prompt: str
    mask_path: str
    score: float | None
    box_xyxy: list[float] | None
    has_mask: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SegmentationRequest:
    object_id: str
    render_dir: str
    cameras_json_path: str
    prompt_path: str
    seg_dir: str
    seg_meta_path: str
    targets: tuple[SegmentationTarget, ...]
    max_successes: int | None = None

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "render_dir": self.render_dir,
            "cameras_json_path": self.cameras_json_path,
            "prompt_path": self.prompt_path,
            "seg_dir": self.seg_dir,
            "seg_meta_path": self.seg_meta_path,
            "targets": [target.to_dict() for target in self.targets],
            "max_successes": self.max_successes,
        }


@dataclass(frozen=True)
class SegmentationExecutionResult:
    object_id: str
    status: str
    seg_meta_path: str
    target_results: tuple[SegmentationTargetResult, ...]
    stdout: str = ""

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "status": self.status,
            "seg_meta_path": self.seg_meta_path,
            "target_results": [item.to_dict() for item in self.target_results],
            "stdout": self.stdout,
        }


class SegmentationExecutor(Protocol):
    def run(
        self,
        request: SegmentationRequest,
        config: SegmentationConfig,
    ) -> SegmentationExecutionResult:
        ...


def build_base_prompt(object_name: str, part_name: str) -> str:
    return f"the {part_name.replace('_', ' ')} of the {object_name.replace('_', ' ')}"


def build_segmentation_request(
    object_dir: str | Path,
    *,
    config: SegmentationConfig | None = None,
) -> SegmentationRequest:
    config = config or SegmentationConfig()
    object_dir = Path(object_dir)
    render_dir = object_dir / "render"
    cameras_json_path = render_dir / "cameras.json"
    prompt_path = object_dir / "prompt" / "part_prompts.json"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"missing part prompt file: {prompt_path}")
    if not cameras_json_path.is_file():
        raise FileNotFoundError(f"missing cameras json: {cameras_json_path}")

    payload = json.loads(prompt_path.read_text(encoding="utf-8"))
    cameras_payload = json.loads(cameras_json_path.read_text(encoding="utf-8"))
    parts = payload.get("parts", [])
    object_name = str(payload.get("object_name", object_dir.name))
    seg_dir = object_dir / "seg2d"

    targets = []
    for view in cameras_payload.get("views", []):
        view_id = str(view["view_id"])
        view_path = render_dir / f"{view_id}{config.image_ext}"
        if not view_path.is_file():
            raise FileNotFoundError(f"missing render view for segmentation: {view_path}")
        for part in parts:
            part_name = part["part_name"]
            prompt = str(part.get("prompt") or build_base_prompt(object_name, part_name))
            mask_path = seg_dir / view_id / f"{part_name}_mask{config.image_ext}"
            targets.append(
                SegmentationTarget(
                    view_id=view_id,
                    image_path=str(view_path),
                    part_name=part_name,
                    prompt=prompt,
                    mask_path=str(mask_path),
                )
            )

    if parts and not targets:
        raise ValueError("no render views found for segmentation request")

    return SegmentationRequest(
        object_id=object_dir.name,
        render_dir=str(render_dir),
        cameras_json_path=str(cameras_json_path),
        prompt_path=str(prompt_path),
        seg_dir=str(seg_dir),
        seg_meta_path=str(seg_dir / config.seg_meta_filename),
        targets=tuple(targets),
    )


def write_segmentation_meta(
    request: SegmentationRequest,
    *,
    config: SegmentationConfig | None = None,
    status: str = "planned",
    target_results: tuple[SegmentationTargetResult, ...] | None = None,
) -> str:
    config = config or SegmentationConfig()
    seg_dir = Path(request.seg_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "object_id": request.object_id,
        "status": status,
        "segmentation_config": config.to_dict(),
        "input": {
            "render_dir": request.render_dir,
            "cameras_json_path": request.cameras_json_path,
            "prompt_path": request.prompt_path,
        },
        "targets": [target.to_dict() for target in request.targets],
    }
    if target_results is not None:
        payload["target_results"] = [item.to_dict() for item in target_results]
    path = Path(request.seg_meta_path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    return str(path)


def _subset_segmentation_request(
    request: SegmentationRequest,
    targets: tuple[SegmentationTarget, ...],
    *,
    max_successes: int | None = None,
) -> SegmentationRequest:
    return SegmentationRequest(
        object_id=request.object_id,
        render_dir=request.render_dir,
        cameras_json_path=request.cameras_json_path,
        prompt_path=request.prompt_path,
        seg_dir=request.seg_dir,
        seg_meta_path=request.seg_meta_path,
        targets=targets,
        max_successes=max_successes,
    )


def _group_targets_by_part(
    targets: tuple[SegmentationTarget, ...],
) -> list[tuple[str, tuple[SegmentationTarget, ...]]]:
    order: list[str] = []
    grouped: dict[str, list[SegmentationTarget]] = {}
    for target in targets:
        if target.part_name not in grouped:
            grouped[target.part_name] = []
            order.append(target.part_name)
        grouped[target.part_name].append(target)
    return [(part_name, tuple(grouped[part_name])) for part_name in order]


def _rewrite_prompt_file_with_selected_parts(
    prompt_path: str | Path,
    *,
    selected_part_names: tuple[str, ...],
) -> None:
    path = Path(prompt_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = set(selected_part_names)
    payload["parts"] = [
        part
        for part in payload.get("parts", [])
        if part.get("part_name") in selected
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _delete_result_masks(results: tuple[SegmentationTargetResult, ...]) -> None:
    for item in results:
        path = Path(item.mask_path)
        if path.is_file():
            path.unlink()


def _segmentation_script() -> str:
    return r"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

root = Path(sys.argv[1])
targets = json.loads(sys.argv[2])
threshold = float(sys.argv[3])
model_confidence_threshold = float(sys.argv[4])
checkpoint_path = sys.argv[5].strip()
max_successes = int(sys.argv[6])

sys.path.insert(0, str(root / "third_party" / "sam3"))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def to_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def run_prompt(processor, image, prompt, threshold):
    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=prompt)

    masks = output.get("masks")
    boxes = output.get("boxes")
    scores = output.get("scores")

    has_mask = False
    best_score = None
    best_box = None
    best_mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)

    if masks is not None and len(masks) > 0 and scores is not None and len(scores) > 0:
        if hasattr(scores, "detach"):
            score_values = scores.detach().cpu().flatten().tolist()
        else:
            score_values = list(scores)
        best_idx = max(range(len(score_values)), key=lambda i: score_values[i])
        best_score = float(score_values[best_idx])
        if best_score >= threshold:
            chosen = masks[best_idx]
            if hasattr(chosen, "detach"):
                chosen = chosen.detach().cpu().numpy()
            chosen = np.asarray(chosen).squeeze()
            best_mask = (chosen > 0).astype(np.uint8) * 255
            has_mask = True
            if boxes is not None and len(boxes) > best_idx:
                best_box = to_list(boxes[best_idx])

    return {
        "has_mask": has_mask,
        "score": best_score,
        "box_xyxy": best_box,
        "mask": best_mask,
    }


device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_sam3_image_model(
    device=device,
    checkpoint_path=checkpoint_path or None,
)
processor = Sam3Processor(model, confidence_threshold=model_confidence_threshold)

results = []
current_image_path = None
current_image = None
current_size = None
success_count = 0

for target in targets:
    image_path = Path(target["image_path"])
    prompt = target["prompt"]
    mask_path = Path(target["mask_path"])
    if current_image_path != image_path:
        current_image = Image.open(image_path).convert("RGB")
        current_size = current_image.size
        current_image_path = image_path

    final_result = run_prompt(
        processor,
        current_image,
        prompt,
        threshold,
    )

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final_result["mask"]).save(mask_path)
    results.append(
        {
            "view_id": target["view_id"],
            "part_name": target["part_name"],
            "prompt": prompt,
            "mask_path": str(mask_path),
            "score": final_result["score"],
            "box_xyxy": final_result["box_xyxy"],
            "has_mask": final_result["has_mask"],
        }
    )
    if final_result["has_mask"]:
        success_count += 1
        if max_successes > 0 and success_count >= max_successes:
            break

print(json.dumps({"target_results": results}, ensure_ascii=True))
"""


class LocalSam3SegmentationExecutor:
    def run(
        self,
        request: SegmentationRequest,
        config: SegmentationConfig,
    ) -> SegmentationExecutionResult:
        if not config.conda_env_path:
            raise RuntimeError(
                "conda_env_path is required for SAM3 segmentation execution; "
                "install the 'sam3' env or set SegmentationConfig.conda_env_path"
            )

        sam3_env = Path(config.conda_env_path) / "bin" / "python"
        if not sam3_env.is_file():
            raise FileNotFoundError(f"missing sam3 env python: {sam3_env}")

        checkpoint_path = config.checkpoint_path or os.environ.get("SAM3_CHECKPOINT_PATH", "")
        seg_dir = Path(request.seg_dir)
        seg_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(sam3_env),
            "-c",
            _segmentation_script(),
            os.getcwd(),
            json.dumps([target.to_dict() for target in request.targets], ensure_ascii=True),
            str(config.low_confidence_threshold),
            str(config.model_confidence_threshold),
            checkpoint_path,
            str(request.max_successes or 0),
        ]
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ},
        )
        payload = json.loads(completed.stdout.strip())
        target_results = tuple(SegmentationTargetResult(**item) for item in payload["target_results"])
        return SegmentationExecutionResult(
            object_id=request.object_id,
            status="completed",
            seg_meta_path=request.seg_meta_path,
            target_results=target_results,
            stdout=completed.stdout.strip(),
        )


def execute_segmentation(
    request: SegmentationRequest,
    *,
    config: SegmentationConfig | None = None,
    executor: SegmentationExecutor | None = None,
) -> SegmentationExecutionResult:
    config = config or SegmentationConfig()
    executor = executor or LocalSam3SegmentationExecutor()
    seg_dir = Path(request.seg_dir)
    if seg_dir.exists():
        shutil.rmtree(seg_dir)
    if not request.targets:
        empty_result = SegmentationExecutionResult(
            object_id=request.object_id,
            status="completed",
            seg_meta_path=request.seg_meta_path,
            target_results=tuple(),
            stdout="no segmentation targets",
        )
        _rewrite_prompt_file_with_selected_parts(request.prompt_path, selected_part_names=tuple())
        write_segmentation_meta(
            request,
            config=config,
            status=empty_result.status,
            target_results=empty_result.target_results,
        )
        return empty_result
    write_segmentation_meta(request, config=config, status="running")
    selected_part_names: list[str] = []
    selected_results: list[SegmentationTargetResult] = []
    stdout_chunks: list[str] = []
    for part_name, part_targets in _group_targets_by_part(request.targets):
        if len(selected_part_names) >= config.max_selected_parts:
            break
        part_request = _subset_segmentation_request(
            request,
            part_targets,
            max_successes=config.max_masks_per_part,
        )
        part_result = executor.run(part_request, config)
        if part_result.stdout:
            stdout_chunks.append(part_result.stdout)
        successful_results: list[SegmentationTargetResult] = []
        discarded_results: list[SegmentationTargetResult] = []
        for item in part_result.target_results:
            if item.has_mask and len(successful_results) < config.max_masks_per_part:
                successful_results.append(item)
            else:
                discarded_results.append(item)
        _delete_result_masks(tuple(discarded_results))
        if successful_results:
            selected_part_names.append(part_name)
            selected_results.extend(successful_results)
    _rewrite_prompt_file_with_selected_parts(
        request.prompt_path,
        selected_part_names=tuple(selected_part_names),
    )
    result = SegmentationExecutionResult(
        object_id=request.object_id,
        status="completed",
        seg_meta_path=request.seg_meta_path,
        target_results=tuple(selected_results),
        stdout="\n".join(chunk for chunk in stdout_chunks if chunk),
    )
    write_segmentation_meta(
        request,
        config=config,
        status=result.status,
        target_results=result.target_results,
    )
    return result
