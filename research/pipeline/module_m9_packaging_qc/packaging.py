from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class PackagingConfig:
    part_label_source: str = "label3d"
    object_manifest_filename: str = "object_manifest.json"
    part_instances_filename: str = "part_instances.json"
    stats_report_filename: str = "stats_report.json"
    colored_ply_enabled: bool = True
    colored_ply_filename: str = "gaussian_part_scores_colored.ply"
    colored_ply_min_score: float = 1e-6
    colored_ply_unlabeled_color: str = "255,255,255"
    colored_ply_scale_color_by_score: bool = True
    colored_ply_min_brightness: float = 0.65
    visualization_enabled: bool = True
    visualization_output_dirname: str = "visualization/part_gifs"
    visualization_resolution: int = 512
    visualization_num_frames: int = 72
    visualization_fps: int = 18
    visualization_radius: float = 1.25
    visualization_fov_deg: float = 55.0
    visualization_pitch_deg: float = 0.0
    visualization_yaw_start_deg: float = 0.0
    visualization_first_frame_hold_seconds: float = 0.5
    visualization_context_opacity_scale: float = 0.05
    visualization_target_min_opacity_scale: float = 0.6

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PartInstanceSample:
    part_sample_id: str
    object_id: str
    category_name: str
    part_name: str
    gaussian_path: str
    camera_params_path: str
    render_views: tuple[str, ...]
    visibility_paths: tuple[str, ...]
    prompt: str
    score_path: str
    unknown_path: str

    def to_dict(self) -> dict:
        return {
            "part_sample_id": self.part_sample_id,
            "object_id": self.object_id,
            "category_name": self.category_name,
            "part_name": self.part_name,
            "gaussian_path": self.gaussian_path,
            "camera_params_path": self.camera_params_path,
            "render_views": list(self.render_views),
            "visibility_paths": list(self.visibility_paths),
            "prompt": self.prompt,
            "score_path": self.score_path,
            "unknown_path": self.unknown_path,
        }


@dataclass(frozen=True)
class PackagingRequest:
    object_id: str
    object_dir: str
    category_name: str
    package_dir: str
    object_manifest_path: str
    part_instances_path: str
    part_samples: tuple[PartInstanceSample, ...]

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "object_dir": self.object_dir,
            "category_name": self.category_name,
            "package_dir": self.package_dir,
            "object_manifest_path": self.object_manifest_path,
            "part_instances_path": self.part_instances_path,
            "part_samples": [sample.to_dict() for sample in self.part_samples],
        }


@dataclass(frozen=True)
class PackagingExecutionResult:
    object_id: str
    status: str
    object_manifest_path: str
    part_instances_path: str
    num_part_samples: int
    stats_report_path: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PackagingStatsReport:
    object_id: str
    category_name: str
    num_part_samples: int
    num_render_views: int
    num_visibility_files: int
    num_segmentation_targets: int
    num_segmentation_masks: int
    num_unknown: int
    part_names: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "category_name": self.category_name,
            "num_part_samples": self.num_part_samples,
            "num_render_views": self.num_render_views,
            "num_visibility_files": self.num_visibility_files,
            "num_segmentation_targets": self.num_segmentation_targets,
            "num_segmentation_masks": self.num_segmentation_masks,
            "num_unknown": self.num_unknown,
            "part_names": list(self.part_names),
        }


class PackagingExecutor(Protocol):
    def run(
        self,
        request: PackagingRequest,
        config: PackagingConfig,
    ) -> PackagingExecutionResult:
        ...


def _is_view_visibility_image(path: Path) -> bool:
    return re.fullmatch(r"view_\d+_visibility\.png", path.name) is not None


def _build_part_sample_id(category_name: str, object_id: str, part_name: str) -> str:
    return f"{category_name}__{object_id}__{part_name}"


def _load_available_part_names(scores_path: Path) -> set[str]:
    payload = np.load(scores_path, allow_pickle=True)
    return {str(item) for item in payload["part_names"].tolist()}


def build_packaging_request(
    object_dir: str | Path,
    *,
    config: PackagingConfig | None = None,
) -> PackagingRequest:
    config = config or PackagingConfig()
    object_dir = Path(object_dir)
    source_meta_path = object_dir / "source" / "source_meta.json"
    prompt_path = object_dir / "prompt" / "part_prompts.json"
    camera_params_path = object_dir / "recon3d" / "lifting_camera_params.json"
    gaussian_path = object_dir / "recon3d" / "splat.ply"
    label_dir = object_dir / config.part_label_source
    scores_path = label_dir / "part_membership_scores.npz"
    unknown_path = label_dir / "unknown_mask.npz"
    visibility_dir = object_dir / "project3d" / "visibility"

    for path in [source_meta_path, prompt_path, camera_params_path, gaussian_path, scores_path, unknown_path]:
        if not path.is_file():
            raise FileNotFoundError(f"missing required input: {path}")
    if not visibility_dir.is_dir():
        raise FileNotFoundError(f"missing visibility dir: {visibility_dir}")

    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
    category_name = source_meta["category_name"]
    available_part_names = _load_available_part_names(scores_path)
    render_views = tuple(
        str(path)
        for path in sorted((object_dir / "render").glob("view_*.png"))
    )
    visibility_paths = tuple(
        str(path)
        for path in sorted(path for path in visibility_dir.glob("*.png") if _is_view_visibility_image(path))
    )
    part_samples = []
    for part in prompts["parts"]:
        part_name = part["part_name"]
        if part_name not in available_part_names:
            continue
        part_samples.append(
            PartInstanceSample(
                part_sample_id=_build_part_sample_id(category_name, object_dir.name, part_name),
                object_id=object_dir.name,
                category_name=category_name,
                part_name=part_name,
                gaussian_path=str(gaussian_path),
                camera_params_path=str(camera_params_path),
                render_views=render_views,
                visibility_paths=visibility_paths,
                prompt=part["prompt"],
                score_path=str(scores_path),
                unknown_path=str(unknown_path),
            )
        )

    package_dir = object_dir / "package"
    return PackagingRequest(
        object_id=object_dir.name,
        object_dir=str(object_dir),
        category_name=category_name,
        package_dir=str(package_dir),
        object_manifest_path=str(package_dir / config.object_manifest_filename),
        part_instances_path=str(package_dir / config.part_instances_filename),
        part_samples=tuple(part_samples),
    )


def write_packaging_outputs(
    request: PackagingRequest,
    *,
    config: PackagingConfig | None = None,
) -> tuple[str, str]:
    config = config or PackagingConfig()
    package_dir = Path(request.package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    object_manifest = {
        "object_id": request.object_id,
        "object_dir": request.object_dir,
        "category_name": request.category_name,
        "num_part_samples": len(request.part_samples),
        "part_label_source": config.part_label_source,
        "config": config.to_dict(),
    }
    with Path(request.object_manifest_path).open("w", encoding="utf-8") as handle:
        json.dump(object_manifest, handle, indent=2, ensure_ascii=True)

    with Path(request.part_instances_path).open("w", encoding="utf-8") as handle:
        json.dump([sample.to_dict() for sample in request.part_samples], handle, indent=2, ensure_ascii=True)

    return request.object_manifest_path, request.part_instances_path


def build_stats_report(
    request: PackagingRequest,
    *,
    config: PackagingConfig | None = None,
) -> PackagingStatsReport:
    config = config or PackagingConfig()
    object_dir = Path(request.object_dir)
    seg_meta = json.loads((object_dir / "seg2d" / "seg_meta.json").read_text(encoding="utf-8"))
    unknown = np.load(
        object_dir / config.part_label_source / "unknown_mask.npz",
        allow_pickle=True,
    )
    part_names = tuple(sample.part_name for sample in request.part_samples)
    return PackagingStatsReport(
        object_id=request.object_id,
        category_name=request.category_name,
        num_part_samples=len(request.part_samples),
        num_render_views=len(list((object_dir / "render").glob("view_*.png"))),
        num_visibility_files=len({path for sample in request.part_samples for path in sample.visibility_paths}),
        num_segmentation_targets=len(seg_meta.get("targets", [])),
        num_segmentation_masks=sum(1 for item in seg_meta.get("target_results", []) if item.get("has_mask")),
        num_unknown=int(unknown["unknown_mask"].sum()),
        part_names=part_names,
    )


def write_stats_report(
    request: PackagingRequest,
    *,
    config: PackagingConfig | None = None,
) -> str:
    config = config or PackagingConfig()
    package_dir = Path(request.package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    report = build_stats_report(request, config=config)
    path = package_dir / config.stats_report_filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, ensure_ascii=True)
    return str(path)


def write_colored_ply_output(
    request: PackagingRequest,
    *,
    config: PackagingConfig | None = None,
) -> str:
    config = config or PackagingConfig()
    from research.pipeline.tools.export_gaussian_part_ply import export_gaussian_part_ply

    object_dir = Path(request.object_dir)
    summary = export_gaussian_part_ply(
        scores_npz=object_dir / config.part_label_source / "part_membership_scores.npz",
        source_ply=object_dir / "recon3d" / "splat.ply",
        output_ply=object_dir / "project3d" / config.colored_ply_filename,
        min_score=config.colored_ply_min_score,
        unlabeled_color=np.asarray(
            [int(item.strip()) for item in config.colored_ply_unlabeled_color.split(",")],
            dtype=np.uint8,
        ),
        scale_color_by_score=config.colored_ply_scale_color_by_score,
        min_brightness=config.colored_ply_min_brightness,
        binary=False,
    )
    return str(summary["output_ply"])


def write_visualization_outputs(
    request: PackagingRequest,
    *,
    config: PackagingConfig | None = None,
) -> str:
    config = config or PackagingConfig()
    from research.pipeline.tools.visualize_stage2_part_gif import render_part_label_gif

    object_dir = Path(request.object_dir)
    output_dir = object_dir / config.visualization_output_dirname
    summary = render_part_label_gif(
        object_dir,
        output_dir=output_dir,
        resolution=config.visualization_resolution,
        num_frames=config.visualization_num_frames,
        fps=config.visualization_fps,
        radius=config.visualization_radius,
        fov_deg=config.visualization_fov_deg,
        pitch_deg=config.visualization_pitch_deg,
        yaw_start_deg=config.visualization_yaw_start_deg,
        first_frame_hold_seconds=config.visualization_first_frame_hold_seconds,
        context_opacity_scale=config.visualization_context_opacity_scale,
        target_min_opacity_scale=config.visualization_target_min_opacity_scale,
    )
    summary_path = output_dir / "part_label_rotation_summary.json"
    if not summary_path.is_file():
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(summary_path)


class LocalPackagingExecutor:
    def run(
        self,
        request: PackagingRequest,
        config: PackagingConfig,
    ) -> PackagingExecutionResult:
        object_manifest_path, part_instances_path = write_packaging_outputs(request, config=config)
        stats_report_path = write_stats_report(request, config=config)
        if config.colored_ply_enabled:
            write_colored_ply_output(request, config=config)
        if config.visualization_enabled and request.part_samples:
            write_visualization_outputs(request, config=config)
        return PackagingExecutionResult(
            object_id=request.object_id,
            status="completed",
            object_manifest_path=object_manifest_path,
            part_instances_path=part_instances_path,
            num_part_samples=len(request.part_samples),
            stats_report_path=stats_report_path,
        )


def execute_packaging(
    request: PackagingRequest,
    *,
    config: PackagingConfig | None = None,
    executor: PackagingExecutor | None = None,
) -> PackagingExecutionResult:
    config = config or PackagingConfig()
    executor = executor or LocalPackagingExecutor()
    return executor.run(request, config)
