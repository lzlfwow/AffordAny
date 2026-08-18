from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "unknown"


def build_object_id(index: int) -> str:
    if index < 0:
        raise ValueError("index must be non-negative")
    return f"object_{index:06d}"


def build_part_sample_id(category_name: str, object_id: str, part_name: str) -> str:
    return f"{_slugify(category_name)}__{object_id}__{_slugify(part_name)}"


@dataclass(frozen=True)
class DataSpecConfig:
    pipeline_root: str = "pipeline"
    object_prefix: str = "object"
    object_id_width: int = 6
    num_render_views: int = 6
    num_visibility_views: int = 3
    image_ext: str = ".png"
    gaussian_ext: str = ".ply"
    score_ext: str = ".npz"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ObjectPaths:
    root: Path
    object_id: str
    num_render_views: int = 6
    num_visibility_views: int = 3
    render_view_names: tuple[str, ...] = field(init=False)
    visibility_view_names: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "render_view_names",
            tuple(f"view_{i:03d}" for i in range(self.num_render_views)),
        )
        object.__setattr__(
            self,
            "visibility_view_names",
            tuple(f"view_{i:03d}" for i in range(self.num_visibility_views)),
        )

    @property
    def object_dir(self) -> Path:
        return self.root / self.object_id

    def to_manifest(self) -> dict:
        views = {
            view_name: {
                "image": f"render/{view_name}.png",
                "seg_dir": f"seg2d/{view_name}",
            }
            for view_name in self.render_view_names
        }
        visibility_views = {
            view_name: f"project3d/visibility/{view_name}_visibility.png"
            for view_name in self.visibility_view_names
        }
        return {
            "object_id": self.object_id,
            "object_dir": str(self.object_dir),
            "source": {
                "image": "source/image.png",
                "instance_mask": "source/instance_mask.png",
                "meta": "source/source_meta.json",
            },
            "candidate": {"meta": "candidate/candidate_meta.json"},
            "recon3d": {
                "gaussian": "recon3d/splat.ply",
                "meta": "recon3d/recon_meta.json",
                "camera_params": "recon3d/lifting_camera_params.json",
            },
            "render": {
                "views": views,
                "cameras": "render/cameras.json",
            },
            "prompt": {"parts": "prompt/part_prompts.json"},
            "seg2d": {"meta": "seg2d/seg_meta.json"},
            "project3d": {
                "votes": "project3d/part_votes.json",
                "scores": "project3d/gaussian_part_scores.npz",
                "visibility_views": visibility_views,
            },
            "label3d": {
                "part_membership_scores": "label3d/part_membership_scores.npz",
                "unknown_mask": "label3d/unknown_mask.npz",
            },
            "package": {
                "object_manifest": "package/object_manifest.json",
                "part_instances": "package/part_instances.json",
            },
            "instruction": {
                "part_instructions": "instruction/part_instructions.json",
            },
        }


@dataclass(frozen=True)
class PartInstanceRecord:
    category_name: str
    object_id: str
    part_name: str
    prompt: str
    object_paths: ObjectPaths

    @property
    def part_sample_id(self) -> str:
        return build_part_sample_id(
            self.category_name,
            self.object_id,
            self.part_name,
        )

    def to_dict(self) -> dict:
        views = [
            f"render/{view_name}.png"
            for view_name in self.object_paths.render_view_names
        ]
        visibility_paths = [
            f"project3d/visibility/{view_name}_visibility.png"
            for view_name in self.object_paths.visibility_view_names
        ]
        return {
            "part_sample_id": self.part_sample_id,
            "object_id": self.object_id,
            "category_name": _slugify(self.category_name),
            "part_name": _slugify(self.part_name),
            "gaussian_path": "recon3d/splat.ply",
            "camera_params_path": "recon3d/lifting_camera_params.json",
            "render_views": views,
            "visibility_paths": visibility_paths,
            "prompt": self.prompt,
            "score_path": "label3d/part_membership_scores.npz",
            "unknown_path": "label3d/unknown_mask.npz",
        }
