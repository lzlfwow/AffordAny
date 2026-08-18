from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Protocol

import numpy as np
from PIL import Image
from plyfile import PlyData


@dataclass(frozen=True)
class LiftConfig:
    visibility_dirname: str = "visibility"
    part_votes_filename: str = "part_votes.json"
    gaussian_scores_filename: str = "gaussian_part_scores.npz"
    project_meta_filename: str = "project3d_meta.json"
    visibility_ext: str = ".png"
    visibility_unlabeled_alpha: float = 0.18
    visibility_labeled_alpha: float = 1.0
    footprint_scale: float = 1.75
    min_projected_radius_px: float = 0.0
    max_projected_radius_px: float = 12.0
    depth_tolerance_scale: float = 0.75
    depth_tolerance_min: float = 1e-3
    footprint_coverage_threshold: float = 0.02
    normalize_geometry_for_render_alignment: bool = True
    visualization_yaws_deg: tuple[float, ...] = (30.0, 150.0, 270.0)
    visualization_pitches_deg: tuple[float, ...] = (15.0, -10.0, 20.0)
    visualization_radius: float | None = None
    visualization_fov_deg: float | None = None
    visualization_resolution: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LiftTarget:
    view_id: str
    part_name: str
    mask_path: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LiftVisibilityView:
    view_id: str
    visibility_path: str
    yaw_deg: float
    pitch_deg: float
    radius: float
    fov_deg: float
    resolution: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LiftRequest:
    object_id: str
    gaussian_path: str
    camera_params_path: str
    cameras_json_path: str
    seg_meta_path: str
    project3d_dir: str
    part_votes_path: str
    gaussian_scores_path: str
    project_meta_path: str
    targets: tuple[LiftTarget, ...]
    visibility_views: tuple[LiftVisibilityView, ...]

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "gaussian_path": self.gaussian_path,
            "camera_params_path": self.camera_params_path,
            "cameras_json_path": self.cameras_json_path,
            "seg_meta_path": self.seg_meta_path,
            "project3d_dir": self.project3d_dir,
            "part_votes_path": self.part_votes_path,
            "gaussian_scores_path": self.gaussian_scores_path,
            "project_meta_path": self.project_meta_path,
            "targets": [target.to_dict() for target in self.targets],
            "visibility_views": [view.to_dict() for view in self.visibility_views],
        }


@dataclass(frozen=True)
class LiftExecutionResult:
    object_id: str
    status: str
    part_votes_path: str
    gaussian_scores_path: str
    project_meta_path: str
    visibility_paths: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


class LiftExecutor(Protocol):
    def run(
        self,
        request: LiftRequest,
        config: LiftConfig,
    ) -> LiftExecutionResult:
        ...


def build_lift_request(
    object_dir: str | Path,
    *,
    config: LiftConfig | None = None,
) -> LiftRequest:
    config = config or LiftConfig()
    object_dir = Path(object_dir)
    gaussian_path = object_dir / "recon3d" / "splat.ply"
    camera_params_path = object_dir / "recon3d" / "lifting_camera_params.json"
    cameras_json_path = object_dir / "render" / "cameras.json"
    seg_meta_path = object_dir / "seg2d" / "seg_meta.json"

    for path in [gaussian_path, camera_params_path, cameras_json_path, seg_meta_path]:
        if not path.is_file():
            raise FileNotFoundError(f"missing required input: {path}")

    seg_meta = json.loads(seg_meta_path.read_text(encoding="utf-8"))
    target_results = seg_meta.get("target_results", [])
    cameras_payload = json.loads(cameras_json_path.read_text(encoding="utf-8"))

    project3d_dir = object_dir / "project3d"
    visibility_dir = project3d_dir / config.visibility_dirname
    targets = []
    for item in target_results:
        view_id = item["view_id"]
        part_name = item["part_name"]
        targets.append(
            LiftTarget(
                view_id=view_id,
                part_name=part_name,
                mask_path=item["mask_path"],
            )
        )
    visibility_views = _build_visibility_views(
        visibility_dir=visibility_dir,
        cameras_payload=cameras_payload,
        config=config,
    )

    return LiftRequest(
        object_id=object_dir.name,
        gaussian_path=str(gaussian_path),
        camera_params_path=str(camera_params_path),
        cameras_json_path=str(cameras_json_path),
        seg_meta_path=str(seg_meta_path),
        project3d_dir=str(project3d_dir),
        part_votes_path=str(project3d_dir / config.part_votes_filename),
        gaussian_scores_path=str(project3d_dir / config.gaussian_scores_filename),
        project_meta_path=str(project3d_dir / config.project_meta_filename),
        targets=tuple(targets),
        visibility_views=tuple(visibility_views),
    )


def write_project3d_meta(
    request: LiftRequest,
    *,
    config: LiftConfig | None = None,
    status: str = "planned",
    execution: LiftExecutionResult | None = None,
) -> str:
    config = config or LiftConfig()
    project3d_dir = Path(request.project3d_dir)
    project3d_dir.mkdir(parents=True, exist_ok=True)
    (project3d_dir / config.visibility_dirname).mkdir(parents=True, exist_ok=True)
    payload = {
        "object_id": request.object_id,
        "status": status,
        "lift_config": config.to_dict(),
        "input": {
            "gaussian_path": request.gaussian_path,
            "camera_params_path": request.camera_params_path,
            "cameras_json_path": request.cameras_json_path,
            "seg_meta_path": request.seg_meta_path,
        },
        "output": {
            "part_votes_path": request.part_votes_path,
            "gaussian_scores_path": request.gaussian_scores_path,
        },
        "targets": [target.to_dict() for target in request.targets],
        "visibility_views": [view.to_dict() for view in request.visibility_views],
    }
    if execution is not None:
        payload["execution"] = execution.to_dict()
    path = Path(request.project_meta_path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    return str(path)


def _load_gaussian_points(gaussian_path: str | Path) -> np.ndarray:
    ply = PlyData.read(str(gaussian_path))
    x = np.asarray(ply.elements[0]["x"], dtype=np.float32)
    y = np.asarray(ply.elements[0]["y"], dtype=np.float32)
    z = np.asarray(ply.elements[0]["z"], dtype=np.float32)
    return np.stack([x, y, z], axis=1)


def _load_gaussian_geometry(
    gaussian_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ply = PlyData.read(str(gaussian_path))
    vertex = ply.elements[0]
    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )
    property_names = {prop.name for prop in vertex.properties}
    scale_names = sorted(
        [name for name in property_names if name.startswith("scale_")],
        key=lambda item: int(item.split("_")[-1]),
    )
    if scale_names:
        scales_log = np.stack(
            [np.asarray(vertex[name], dtype=np.float32) for name in scale_names],
            axis=1,
        )
        scales = np.exp(scales_log).astype(np.float32)
    else:
        scales = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    if "opacity" in property_names:
        opacity_raw = np.asarray(vertex["opacity"], dtype=np.float32)
        opacity = 1.0 / (1.0 + np.exp(-opacity_raw))
    else:
        opacity = np.ones(xyz.shape[0], dtype=np.float32)
    return xyz, scales.astype(np.float32), opacity.astype(np.float32)


def _normalize_gaussian_geometry_for_render(
    points_xyz: np.ndarray,
    scales_xyz: np.ndarray,
    opacity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if points_xyz.shape[0] == 0:
        return points_xyz.astype(np.float32), scales_xyz.astype(np.float32)

    active_mask = opacity > np.float32(0.9)
    if not np.any(active_mask):
        active_mask = opacity > np.float32(0.0)
    if not np.any(active_mask):
        active_mask = np.ones(points_xyz.shape[0], dtype=bool)

    active_points = points_xyz[active_mask]
    mins = active_points.min(axis=0)
    maxs = active_points.max(axis=0)
    inv_scale = float(np.max(np.maximum(maxs - mins, np.float32(1e-6))))
    inv_scale = max(inv_scale, 1e-6)

    norm_points = points_xyz / np.float32(inv_scale)
    norm_scales = scales_xyz / np.float32(inv_scale)

    active_norm_points = norm_points[active_mask]
    center = (active_norm_points.min(axis=0) + active_norm_points.max(axis=0)) * np.float32(0.5)
    norm_points = norm_points - center.astype(np.float32)
    return norm_points.astype(np.float32), norm_scales.astype(np.float32)


def _project_points(
    points_xyz: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([points_xyz, ones], axis=1)
    camera_xyz = homo @ extrinsics.T
    xyz = camera_xyz[:, :3]
    z = xyz[:, 2]
    valid = z > 1e-6
    proj = np.full((points_xyz.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        x = xyz[valid, 0] / z[valid]
        y = xyz[valid, 1] / z[valid]
        u = (intrinsics[0, 0] * x + intrinsics[0, 2]) * width
        v = (intrinsics[1, 1] * y + intrinsics[1, 2]) * height
        proj[valid, 0] = u
        proj[valid, 1] = v
    return proj, z


def _compute_projected_radii_px(
    *,
    scales_xyz: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    config: LiftConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if scales_xyz.shape[0] == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    focal_x = abs(float(intrinsics[0, 0])) * float(width)
    focal_y = abs(float(intrinsics[1, 1])) * float(height)
    max_scale_world = np.max(scales_xyz, axis=1).astype(np.float32)
    valid_depth = depth > 1e-6
    safe_depth = np.where(valid_depth, depth, 1.0).astype(np.float32)
    radius_x = focal_x * max_scale_world / safe_depth
    radius_y = focal_y * max_scale_world / safe_depth
    radii_px = np.maximum(radius_x, radius_y).astype(np.float32) * np.float32(config.footprint_scale)
    radii_px = np.clip(
        radii_px,
        np.float32(config.min_projected_radius_px),
        np.float32(config.max_projected_radius_px),
    )
    radii_px = np.where(valid_depth, radii_px, 0.0).astype(np.float32)
    return radii_px, max_scale_world


def _build_depth_buffer(
    *,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    depth: np.ndarray,
    valid_mask: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    flat_depth = np.full(width * height, np.inf, dtype=np.float32)
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        return flat_depth.reshape(height, width)
    flat_indices = pixel_y[valid_indices].astype(np.int64) * np.int64(width) + pixel_x[valid_indices].astype(np.int64)
    np.minimum.at(flat_depth, flat_indices, depth[valid_indices].astype(np.float32))
    return flat_depth.reshape(height, width)


def _compute_visibility_mask(
    *,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    depth: np.ndarray,
    radii_px: np.ndarray,
    max_scale_world: np.ndarray,
    depth_buffer: np.ndarray,
    valid_mask: np.ndarray,
    width: int,
    height: int,
    config: LiftConfig,
) -> np.ndarray:
    visibility = np.zeros_like(valid_mask, dtype=bool)
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        return visibility

    depth_tol = np.maximum(
        max_scale_world[valid_indices] * np.float32(config.depth_tolerance_scale),
        np.float32(config.depth_tolerance_min),
    )
    sample_step = np.maximum(np.ceil(radii_px[valid_indices] * 0.5).astype(np.int32), 1)
    sample_x = np.stack(
        [
            pixel_x[valid_indices],
            pixel_x[valid_indices] - sample_step,
            pixel_x[valid_indices] + sample_step,
            pixel_x[valid_indices],
            pixel_x[valid_indices],
        ],
        axis=1,
    )
    sample_y = np.stack(
        [
            pixel_y[valid_indices],
            pixel_y[valid_indices],
            pixel_y[valid_indices],
            pixel_y[valid_indices] - sample_step,
            pixel_y[valid_indices] + sample_step,
        ],
        axis=1,
    )
    sample_x = np.clip(sample_x, 0, width - 1)
    sample_y = np.clip(sample_y, 0, height - 1)
    sampled_depth = depth_buffer[sample_y, sample_x]
    visible_valid = np.any(
        depth[valid_indices, None] <= sampled_depth + depth_tol[:, None],
        axis=1,
    )
    visibility[valid_indices] = visible_valid
    return visibility


def _build_integral_image(mask: np.ndarray) -> np.ndarray:
    mask_u32 = mask.astype(np.uint32)
    integral = np.pad(mask_u32, ((1, 0), (1, 0)), mode="constant", constant_values=0)
    return integral.cumsum(axis=0).cumsum(axis=1)


def _query_box_mean(
    integral: np.ndarray,
    *,
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
) -> np.ndarray:
    x0_i = x0.astype(np.int64)
    y0_i = y0.astype(np.int64)
    x1_i = x1.astype(np.int64) + 1
    y1_i = y1.astype(np.int64) + 1
    sums = (
        integral[y1_i, x1_i]
        - integral[y0_i, x1_i]
        - integral[y1_i, x0_i]
        + integral[y0_i, x0_i]
    ).astype(np.float32)
    areas = ((x1 - x0 + 1) * (y1 - y0 + 1)).astype(np.float32)
    return sums / np.maximum(areas, 1.0)


def _compute_visible_indices(
    points_xyz: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    proj, depth = _project_points(points_xyz, extrinsics, intrinsics, width, height)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    owner = np.full((height, width), -1, dtype=np.int32)

    for idx, ((u, v), z) in enumerate(zip(proj, depth)):
        if not np.isfinite(u) or not np.isfinite(v) or z <= 1e-6:
            continue
        px = int(np.clip(round(float(u)), 0, width - 1))
        py = int(np.clip(round(float(v)), 0, height - 1))
        if z < depth_buffer[py, px]:
            depth_buffer[py, px] = z
            owner[py, px] = idx

    visible = owner >= 0
    visible_indices = owner[visible]
    return visible_indices, owner


def _part_color(part_name: str) -> np.ndarray:
    palette = {
        "seat": np.array([255, 80, 80], dtype=np.uint8),
        "backrest": np.array([80, 180, 255], dtype=np.uint8),
        "armrest": np.array([120, 255, 120], dtype=np.uint8),
    }
    if part_name in palette:
        return palette[part_name]
    hashed = abs(hash(part_name)) % 200
    return np.array(
        [
            55 + (hashed * 37) % 200,
            55 + (hashed * 67) % 200,
            55 + (hashed * 97) % 200,
        ],
        dtype=np.uint8,
    )


def _build_visibility_views(
    *,
    visibility_dir: Path,
    cameras_payload: dict,
    config: LiftConfig,
) -> tuple[LiftVisibilityView, ...]:
    if len(config.visualization_yaws_deg) != len(config.visualization_pitches_deg):
        raise ValueError("visualization_yaws_deg and visualization_pitches_deg must have the same length")

    source_views = cameras_payload.get("views", [])
    default_radius = float(config.visualization_radius) if config.visualization_radius is not None else 2.0
    default_fov = float(config.visualization_fov_deg) if config.visualization_fov_deg is not None else 40.0
    default_resolution = int(config.visualization_resolution) if config.visualization_resolution is not None else 512
    if source_views:
        default_radius = float(config.visualization_radius) if config.visualization_radius is not None else float(
            np.mean([float(view.get("radius", 2.0)) for view in source_views])
        )
        default_fov = float(config.visualization_fov_deg) if config.visualization_fov_deg is not None else float(
            np.mean([float(view.get("fov_deg", 40.0)) for view in source_views])
        )
        default_resolution = int(config.visualization_resolution) if config.visualization_resolution is not None else int(
            source_views[0].get("resolution", 512)
        )

    return tuple(
        LiftVisibilityView(
            view_id=f"view_{idx:03d}",
            visibility_path=str(visibility_dir / f"view_{idx:03d}_visibility{config.visibility_ext}"),
            yaw_deg=float(yaw_deg),
            pitch_deg=float(pitch_deg),
            radius=default_radius,
            fov_deg=default_fov,
            resolution=default_resolution,
        )
        for idx, (yaw_deg, pitch_deg) in enumerate(zip(config.visualization_yaws_deg, config.visualization_pitches_deg))
    )


def _visibility_extrinsics_intrinsics(view: LiftVisibilityView) -> tuple[np.ndarray, np.ndarray]:
    yaw = np.deg2rad(np.float32(view.yaw_deg))
    pitch = np.deg2rad(np.float32(view.pitch_deg))
    eye = np.array(
        [
            np.sin(yaw) * np.cos(pitch),
            np.cos(yaw) * np.cos(pitch),
            np.sin(pitch),
        ],
        dtype=np.float32,
    ) * np.float32(view.radius)

    target = np.zeros(3, dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    forward = target - eye
    forward /= max(float(np.linalg.norm(forward)), 1e-6)
    right = np.cross(forward, up)
    right /= max(float(np.linalg.norm(right)), 1e-6)
    down = np.cross(forward, right)

    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[0, :3] = right
    extrinsics[1, :3] = down
    extrinsics[2, :3] = forward
    extrinsics[:3, 3] = -(extrinsics[:3, :3] @ eye)

    fov_rad = np.deg2rad(np.float32(view.fov_deg))
    focal = np.float32(0.5 / np.tan(fov_rad * 0.5))
    intrinsics = np.array(
        [
            [focal, 0.0, 0.5],
            [0.0, focal, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return extrinsics, intrinsics


def _render_view_visibility(
    owner: np.ndarray,
    part_names: list[str],
    best_indices: np.ndarray,
    max_scores: np.ndarray,
    *,
    unlabeled_alpha: float = 0.18,
    labeled_alpha: float = 1.0,
) -> np.ndarray:
    height, width = owner.shape
    vis_img = np.zeros((height, width, 4), dtype=np.uint8)
    vis_img[..., 3] = 255
    visible_mask = owner >= 0
    if not np.any(visible_mask):
        return vis_img
    vis_img[visible_mask] = np.array(
        [255, 255, 255, int(np.clip(round(unlabeled_alpha * 255.0), 0, 255))],
        dtype=np.uint8,
    )
    if not part_names:
        return vis_img

    point_indices = owner[visible_mask]
    pixel_best_indices = best_indices[point_indices]
    pixel_scores = np.clip(max_scores[point_indices], 0.0, 1.0)
    pixel_colors = np.zeros((point_indices.shape[0], 3), dtype=np.float32)
    labeled_mask = (pixel_best_indices >= 0) & (pixel_scores > 0.0)

    for part_idx, part_name in enumerate(part_names):
        matched = labeled_mask & (pixel_best_indices == part_idx)
        if not np.any(matched):
            continue
        pixel_colors[matched] = _part_color(part_name).astype(np.float32) * pixel_scores[matched, None]

    colored_values = np.clip(pixel_colors, 0.0, 255.0).astype(np.uint8)
    visible_pixels = vis_img[visible_mask]
    visible_pixels[:, :3] = np.where(
        labeled_mask[:, None],
        colored_values,
        visible_pixels[:, :3],
    )
    visible_pixels[:, 3] = np.where(
        labeled_mask,
        int(np.clip(round(labeled_alpha * 255.0), 0, 255)),
        visible_pixels[:, 3],
    )
    vis_img[visible_mask] = visible_pixels
    return vis_img


class LocalLiftExecutor:
    def run(
        self,
        request: LiftRequest,
        config: LiftConfig,
    ) -> LiftExecutionResult:
        project3d_dir = Path(request.project3d_dir)
        visibility_dir = project3d_dir / config.visibility_dirname
        if visibility_dir.exists():
            shutil.rmtree(visibility_dir)
        visibility_dir.mkdir(parents=True, exist_ok=True)

        points_xyz, scales_xyz, opacity = _load_gaussian_geometry(request.gaussian_path)
        if config.normalize_geometry_for_render_alignment:
            points_xyz, scales_xyz = _normalize_gaussian_geometry_for_render(
                points_xyz,
                scales_xyz,
                opacity,
            )
        cameras = json.loads(Path(request.cameras_json_path).read_text(encoding="utf-8"))
        seg_meta = json.loads(Path(request.seg_meta_path).read_text(encoding="utf-8"))

        part_names = []
        for item in seg_meta["target_results"]:
            if item["part_name"] not in part_names:
                part_names.append(item["part_name"])
        part_index = {name: idx for idx, name in enumerate(part_names)}

        positive_counts = np.zeros((len(part_names), len(points_xyz)), dtype=np.float32)
        visible_counts = np.zeros(len(points_xyz), dtype=np.float32)
        visibility_paths = []
        view_cache: dict[str, dict[str, np.ndarray | tuple[int, int]]] = {}
        camera_views = {view["view_id"]: view for view in cameras["views"]}
        targets_by_view: dict[str, list[LiftTarget]] = {}
        for target in request.targets:
            targets_by_view.setdefault(target.view_id, []).append(target)

        for view_id, view_targets in targets_by_view.items():
            if view_id not in view_cache:
                reference_mask_path = next(
                    (Path(target.mask_path) for target in view_targets if Path(target.mask_path).is_file()),
                    None,
                )
                if reference_mask_path is None:
                    continue
                reference_mask = np.asarray(Image.open(reference_mask_path).convert("L"), dtype=np.uint8) > 0
                height, width = reference_mask.shape
                view = camera_views[view_id]
                extrinsics = np.asarray(view["extrinsics"], dtype=np.float32)
                intrinsics = np.asarray(view["intrinsics"], dtype=np.float32)
                visible_indices, owner = _compute_visible_indices(
                    points_xyz,
                    extrinsics,
                    intrinsics,
                    width,
                    height,
                )
                view_cache[view_id] = {
                    "shape": (height, width),
                    "owner": owner,
                    "visible_indices": visible_indices,
                }
                if visible_indices.size:
                    visible_counts[visible_indices] += 1.0

            view_state = view_cache.get(view_id)
            if view_state is None:
                continue
            visible_indices = np.asarray(view_state["visible_indices"], dtype=np.int32)
            if visible_indices.size == 0:
                continue
            owner = np.asarray(view_state["owner"], dtype=np.int32)
            height, width = tuple(view_state["shape"])
            for target in view_targets:
                mask = np.asarray(Image.open(target.mask_path).convert("L"), dtype=np.uint8) > 0
                if mask.shape != (height, width):
                    raise ValueError(
                        f"mask shape mismatch for {target.mask_path}: expected {(height, width)}, got {mask.shape}"
                    )
                part_idx = part_index[target.part_name]
                owner_hits = owner[mask & (owner >= 0)]
                if owner_hits.size == 0:
                    continue
                np.add.at(
                    positive_counts[part_idx],
                    owner_hits.astype(np.int64),
                    np.float32(1.0),
                )

        scores = np.zeros_like(positive_counts)
        visible_safe = np.where(visible_counts > 0, visible_counts, 1.0)
        for idx in range(len(part_names)):
            scores[idx] = positive_counts[idx] / visible_safe

        if len(part_names):
            best_indices = scores.argmax(axis=0).astype(np.int32)
            max_scores = scores.max(axis=0).astype(np.float32)
        else:
            best_indices = np.full(len(points_xyz), -1, dtype=np.int32)
            max_scores = np.zeros(len(points_xyz), dtype=np.float32)

        np.savez(
            request.gaussian_scores_path,
            part_names=np.asarray(part_names, dtype=object),
            scores=scores,
            positive_counts=positive_counts,
            visible_counts=visible_counts,
        )

        part_votes = {
            "object_id": request.object_id,
            "part_names": part_names,
            "summary": [
                {
                    "part_name": part_name,
                    "positive_count_total": float(positive_counts[idx].sum()),
                    "visible_count_total": float(visible_counts.sum()),
                    "score_mean": float(scores[idx].mean()),
                }
                for idx, part_name in enumerate(part_names)
            ],
        }
        with Path(request.part_votes_path).open("w", encoding="utf-8") as handle:
            json.dump(part_votes, handle, indent=2, ensure_ascii=True)

        # Render view-level visibility after scores are finalized so intensity reflects final part scores.
        for visibility_view in request.visibility_views:
            extrinsics, intrinsics = _visibility_extrinsics_intrinsics(visibility_view)
            _, owner = _compute_visible_indices(
                points_xyz,
                extrinsics,
                intrinsics,
                visibility_view.resolution,
                visibility_view.resolution,
            )
            visibility_path = Path(visibility_view.visibility_path)
            visibility_path.parent.mkdir(parents=True, exist_ok=True)
            vis_img = _render_view_visibility(
                owner,
                part_names,
                best_indices,
                max_scores,
                unlabeled_alpha=config.visibility_unlabeled_alpha,
                labeled_alpha=config.visibility_labeled_alpha,
            )
            Image.fromarray(vis_img).save(visibility_path)
            visibility_paths.append(str(visibility_path))

        return LiftExecutionResult(
            object_id=request.object_id,
            status="completed",
            part_votes_path=request.part_votes_path,
            gaussian_scores_path=request.gaussian_scores_path,
            project_meta_path=request.project_meta_path,
            visibility_paths=tuple(sorted(set(visibility_paths))),
        )


def execute_lift(
    request: LiftRequest,
    *,
    config: LiftConfig | None = None,
    executor: LiftExecutor | None = None,
) -> LiftExecutionResult:
    config = config or LiftConfig()
    executor = executor or LocalLiftExecutor()
    write_project3d_meta(request, config=config, status="running")
    result = executor.run(request, config)
    write_project3d_meta(request, config=config, status=result.status, execution=result)
    return result
