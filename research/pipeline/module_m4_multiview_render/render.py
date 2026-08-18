from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import numpy as np
import os
from pathlib import Path
import subprocess
from typing import Protocol


@dataclass(frozen=True)
class RenderConfig:
    sam3d_root: str = "third_party/sam-3d-objects"
    conda_env_path: str = ""
    num_views: int = 6
    seed: int = 42
    resolution: int = 512
    radius: float = 2.0
    fov_deg: float = 40.0
    standard_view_names: tuple[str, ...] = ("front", "right", "back", "left", "top", "bottom")
    standard_yaws_deg: tuple[float, ...] = (0.0, 90.0, 180.0, -90.0, 0.0, 0.0)
    standard_pitches_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 89.0, -89.0)
    diagonal_pose_enabled: bool = True
    diagonal_pose_target_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    image_ext: str = ".png"
    cameras_filename: str = "cameras.json"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RenderViewSpec:
    view_id: str
    image_path: str
    canonical_view: str
    yaw_deg: float
    pitch_deg: float
    radius: float
    fov_deg: float
    resolution: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RenderRequest:
    object_id: str
    object_dir: str
    gaussian_path: str
    camera_params_path: str
    render_dir: str
    cameras_json_path: str
    views: tuple[RenderViewSpec, ...]

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "object_dir": self.object_dir,
            "gaussian_path": self.gaussian_path,
            "camera_params_path": self.camera_params_path,
            "render_dir": self.render_dir,
            "cameras_json_path": self.cameras_json_path,
            "views": [view.to_dict() for view in self.views],
        }


@dataclass(frozen=True)
class RenderExecutionResult:
    object_id: str
    status: str
    render_dir: str
    cameras_json_path: str
    view_paths: tuple[str, ...]
    stdout: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class RenderExecutor(Protocol):
    def run(
        self,
        request: RenderRequest,
        config: RenderConfig,
    ) -> RenderExecutionResult:
        ...


def _normalize_vector(values: tuple[float, float, float] | list[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        raise ValueError("rotation target vector must be non-zero")
    return vector / norm


def build_object_pre_rotation_matrix(
    config: RenderConfig | None = None,
) -> np.ndarray:
    config = config or RenderConfig()
    matrix = np.eye(4, dtype=np.float32)
    if not config.diagonal_pose_enabled:
        return matrix

    source = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    target = _normalize_vector(config.diagonal_pose_target_xyz)
    cross = np.cross(source, target)
    cross_norm = float(np.linalg.norm(cross))
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))

    if cross_norm <= 1e-6:
        if dot < 0.0:
            rotation = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        else:
            rotation = np.eye(3, dtype=np.float32)
    else:
        skew = np.array(
            [
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ],
            dtype=np.float32,
        )
        rotation = (
            np.eye(3, dtype=np.float32)
            + skew
            + (skew @ skew) * ((1.0 - dot) / (cross_norm ** 2))
        )

    matrix[:3, :3] = rotation
    return matrix


def sample_view_specs(
    *,
    render_dir: str | Path,
    config: RenderConfig | None = None,
) -> tuple[RenderViewSpec, ...]:
    config = config or RenderConfig()
    render_dir = Path(render_dir)
    if len(config.standard_view_names) != len(config.standard_yaws_deg) or len(config.standard_view_names) != len(config.standard_pitches_deg):
        raise ValueError("standard view names, yaws, and pitches must have the same length")
    if config.num_views > len(config.standard_view_names):
        raise ValueError("num_views exceeds configured standard view definitions")
    views = []
    for idx, (canonical_view, yaw_deg, pitch_deg) in enumerate(
        zip(
            config.standard_view_names[:config.num_views],
            config.standard_yaws_deg[:config.num_views],
            config.standard_pitches_deg[:config.num_views],
        )
    ):
        view_id = f"view_{idx:03d}"
        views.append(
            RenderViewSpec(
                view_id=view_id,
                image_path=str(render_dir / f"{view_id}{config.image_ext}"),
                canonical_view=canonical_view,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                radius=config.radius,
                fov_deg=config.fov_deg,
                resolution=config.resolution,
            )
        )
    return tuple(views)


def build_render_request(
    object_dir: str | Path,
    *,
    config: RenderConfig | None = None,
) -> RenderRequest:
    config = config or RenderConfig()
    object_dir = Path(object_dir)
    gaussian_path = object_dir / "recon3d" / "splat.ply"
    camera_params_path = object_dir / "recon3d" / "lifting_camera_params.json"
    if not gaussian_path.is_file():
        raise FileNotFoundError(f"missing gaussian file: {gaussian_path}")
    if not camera_params_path.is_file():
        raise FileNotFoundError(f"missing camera params: {camera_params_path}")

    render_dir = object_dir / "render"
    views = sample_view_specs(render_dir=render_dir, config=config)
    return RenderRequest(
        object_id=object_dir.name,
        object_dir=str(object_dir),
        gaussian_path=str(gaussian_path),
        camera_params_path=str(camera_params_path),
        render_dir=str(render_dir),
        cameras_json_path=str(render_dir / config.cameras_filename),
        views=views,
    )


def write_cameras_metadata(
    request: RenderRequest,
    *,
    config: RenderConfig | None = None,
    status: str = "planned",
) -> str:
    config = config or RenderConfig()
    render_dir = Path(request.render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "object_id": request.object_id,
        "status": status,
        "render_config": config.to_dict(),
        "input": {
            "gaussian_path": request.gaussian_path,
            "camera_params_path": request.camera_params_path,
        },
        "object_pre_rotation_matrix": build_object_pre_rotation_matrix(config).tolist(),
        "views": [view.to_dict() for view in request.views],
    }
    path = Path(request.cameras_json_path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    return str(path)


def _render_script() -> str:
    return r"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
import torch

gaussian_path = Path(sys.argv[1])
cameras_json_path = Path(sys.argv[2])
views = json.loads(sys.argv[3])
sam3d_root = Path(sys.argv[4])
resolution = int(sys.argv[5])
object_pre_rotation = json.loads(sys.argv[6])

sys.path.insert(0, str(sam3d_root))
sys.path.insert(0, str(sam3d_root / "notebook"))

from inference import ready_gaussian_for_video_rendering
from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian import Gaussian
from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils


ply = PlyData.read(str(gaussian_path))
x = np.asarray(ply.elements[0]["x"])
y = np.asarray(ply.elements[0]["y"])
z = np.asarray(ply.elements[0]["z"])
mins = np.array([x.min(), y.min(), z.min()], dtype=np.float32)
maxs = np.array([x.max(), y.max(), z.max()], dtype=np.float32)
sizes = np.maximum(maxs - mins, 1e-6)
aabb = [*mins.tolist(), *sizes.tolist()]

device = "cuda" if torch.cuda.is_available() else "cpu"
gaussian = Gaussian(aabb=aabb, sh_degree=0, device=device)
gaussian.load_ply(str(gaussian_path))
gaussian = ready_gaussian_for_video_rendering(gaussian, in_place=False, fix_alignment=False)

yaws = [math.radians(float(v["yaw_deg"])) for v in views]
pitchs = [math.radians(float(v["pitch_deg"])) for v in views]
rs = [float(v["radius"]) for v in views]
fovs = [float(v["fov_deg"]) for v in views]
extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
    yaws, pitchs, rs, fovs
)
object_pre_rotation = torch.tensor(object_pre_rotation, dtype=torch.float32, device=extrinsics[0].device)
extrinsics = [extr @ object_pre_rotation for extr in extrinsics]
frames = render_utils.render_frames(
    gaussian,
    extrinsics,
    intrinsics,
    {"resolution": resolution, "bg_color": (0, 0, 0), "backend": "gsplat"},
    verbose=False,
)

camera_payload = {"status": "completed", "object_pre_rotation_matrix": object_pre_rotation.detach().cpu().tolist(), "views": []}
for idx, (view, image) in enumerate(zip(views, frames["color"])):
    image_path = Path(view["image_path"])
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(image_path)
    camera_payload["views"].append(
        {
            **view,
            "extrinsics": extrinsics[idx].detach().cpu().tolist(),
            "intrinsics": intrinsics[idx].detach().cpu().tolist(),
        }
    )

cameras_json_path.write_text(json.dumps(camera_payload, indent=2, ensure_ascii=True), encoding="utf-8")
print(json.dumps({"num_views": len(views), "cameras_json_path": str(cameras_json_path)}, ensure_ascii=True))
"""


class LocalRenderExecutor:
    def run(
        self,
        request: RenderRequest,
        config: RenderConfig,
    ) -> RenderExecutionResult:
        if not config.conda_env_path:
            raise RuntimeError("conda_env_path is required for real render execution")

        python_path = Path(config.conda_env_path) / "bin" / "python"
        if not python_path.is_file():
            raise FileNotFoundError(f"missing env python: {python_path}")

        render_dir = Path(request.render_dir)
        render_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(python_path),
            "-c",
            _render_script(),
            request.gaussian_path,
            request.cameras_json_path,
            json.dumps([view.to_dict() for view in request.views], ensure_ascii=True),
            config.sam3d_root,
            str(config.resolution),
            json.dumps(build_object_pre_rotation_matrix(config).tolist(), ensure_ascii=True),
        ]
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "CONDA_PREFIX": str(config.conda_env_path)},
        )
        return RenderExecutionResult(
            object_id=request.object_id,
            status="completed",
            render_dir=request.render_dir,
            cameras_json_path=request.cameras_json_path,
            view_paths=tuple(view.image_path for view in request.views),
            stdout=completed.stdout.strip(),
        )


def execute_render(
    request: RenderRequest,
    *,
    config: RenderConfig | None = None,
    executor: RenderExecutor | None = None,
) -> RenderExecutionResult:
    config = config or RenderConfig()
    executor = executor or LocalRenderExecutor()
    write_cameras_metadata(request, config=config, status="running")
    result = executor.run(request, config)
    return result
