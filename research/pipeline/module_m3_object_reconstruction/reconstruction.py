from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Protocol


@dataclass(frozen=True)
class ReconstructionConfig:
    sam3d_root: str = "third_party/sam-3d-objects"
    checkpoint_tag: str = "hf"
    conda_env_path: str = ""
    seed: int = 42
    compile_model: bool = False
    gaussian_filename: str = "splat.ply"
    recon_meta_filename: str = "recon_meta.json"
    camera_params_filename: str = "lifting_camera_params.json"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ReconstructionRequest:
    object_id: str
    object_dir: str
    image_path: str
    mask_path: str
    recon_dir: str
    gaussian_path: str
    recon_meta_path: str
    camera_params_path: str
    model_config_path: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ReconstructionExecutionResult:
    object_id: str
    status: str
    gaussian_path: str
    camera_params_path: str
    recon_meta_path: str
    stdout: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ReconstructionExecutor(Protocol):
    def run(
        self,
        request: ReconstructionRequest,
        config: ReconstructionConfig,
    ) -> ReconstructionExecutionResult:
        ...


def build_reconstruction_request(
    object_dir: str | Path,
    *,
    config: ReconstructionConfig | None = None,
) -> ReconstructionRequest:
    config = config or ReconstructionConfig()
    object_dir = Path(object_dir)
    image_path = object_dir / "source" / "image.png"
    mask_path = object_dir / "source" / "instance_mask.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"missing source image: {image_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"missing source mask: {mask_path}")

    recon_dir = object_dir / "recon3d"
    model_config_path = Path(config.sam3d_root) / "checkpoints" / config.checkpoint_tag / "pipeline.yaml"
    return ReconstructionRequest(
        object_id=object_dir.name,
        object_dir=str(object_dir),
        image_path=str(image_path),
        mask_path=str(mask_path),
        recon_dir=str(recon_dir),
        gaussian_path=str(recon_dir / config.gaussian_filename),
        recon_meta_path=str(recon_dir / config.recon_meta_filename),
        camera_params_path=str(recon_dir / config.camera_params_filename),
        model_config_path=str(model_config_path),
    )


def build_reconstruction_meta(
    request: ReconstructionRequest,
    *,
    config: ReconstructionConfig | None = None,
    status: str = "planned",
) -> dict:
    config = config or ReconstructionConfig()
    return {
        "object_id": request.object_id,
        "status": status,
        "backend": {
            "tool_root": config.sam3d_root,
            "checkpoint_tag": config.checkpoint_tag,
            "conda_env_path": config.conda_env_path,
            "model_config_path": request.model_config_path,
            "seed": config.seed,
            "compile_model": config.compile_model,
        },
        "input": {
            "image_path": request.image_path,
            "mask_path": request.mask_path,
        },
        "output": {
            "recon_dir": request.recon_dir,
            "gaussian_path": request.gaussian_path,
            "camera_params_path": request.camera_params_path,
            "recon_meta_path": request.recon_meta_path,
        },
    }


def write_reconstruction_meta(
    request: ReconstructionRequest,
    *,
    config: ReconstructionConfig | None = None,
    status: str = "planned",
) -> str:
    meta = build_reconstruction_meta(request, config=config, status=status)
    recon_dir = Path(request.recon_dir)
    recon_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(request.recon_meta_path)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=True)
    return str(meta_path)


def _execution_script() -> str:
    return r"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

image_path = Path(sys.argv[1])
mask_path = Path(sys.argv[2])
gaussian_path = Path(sys.argv[3])
camera_params_path = Path(sys.argv[4])
model_config_path = Path(sys.argv[5])
sam3d_root = Path(sys.argv[6])
seed = int(sys.argv[7])
compile_model = bool(int(sys.argv[8]))

sys.path.insert(0, str(sam3d_root / "notebook"))
sys.path.insert(0, str(sam3d_root))

from inference import Inference
from sam3d_objects.pipeline.utils.pointmap import infer_intrinsics_from_pointmap


def to_serializable(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


image = np.asarray(Image.open(image_path).convert("RGB"))
mask = (np.asarray(Image.open(mask_path).convert("L")) > 0).astype(np.uint8)

set_all_seeds(seed)
inference = Inference(str(model_config_path), compile=compile_model)
output = inference(image, mask, seed=seed)
output["gs"].save_ply(gaussian_path)

camera_payload = {
    "intrinsics": None,
    "rotation": to_serializable(output.get("rotation")),
    "translation": to_serializable(output.get("translation")),
    "scale": to_serializable(output.get("scale")),
    "shape": to_serializable(output.get("shape")),
}

pointmap = output.get("pointmap")
if pointmap is not None:
    pointmap_tensor = pointmap if isinstance(pointmap, torch.Tensor) else torch.as_tensor(pointmap)
    intrinsics_result = infer_intrinsics_from_pointmap(pointmap_tensor)
    camera_payload["intrinsics"] = to_serializable(intrinsics_result.get("intrinsics"))

camera_params_path.write_text(json.dumps(camera_payload, indent=2, ensure_ascii=True), encoding="utf-8")

summary = {
    "gaussian_path": str(gaussian_path),
    "camera_params_path": str(camera_params_path),
    "gaussian_exists": gaussian_path.is_file(),
    "camera_params_exists": camera_params_path.is_file(),
    "camera_keys": sorted(camera_payload.keys()),
}
print(json.dumps(summary, ensure_ascii=True))
"""


class LocalSam3DObjectsExecutor:
    def run(
        self,
        request: ReconstructionRequest,
        config: ReconstructionConfig,
    ) -> ReconstructionExecutionResult:
        if not config.conda_env_path:
            raise RuntimeError("conda_env_path is required for real reconstruction execution")

        python_path = Path(config.conda_env_path) / "bin" / "python"
        if not python_path.is_file():
            raise FileNotFoundError(f"missing env python: {python_path}")

        recon_dir = Path(request.recon_dir)
        recon_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(python_path),
            "-c",
            _execution_script(),
            request.image_path,
            request.mask_path,
            request.gaussian_path,
            request.camera_params_path,
            request.model_config_path,
            config.sam3d_root,
            str(config.seed),
            "1" if config.compile_model else "0",
        ]
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "CONDA_PREFIX": str(config.conda_env_path),
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1"),
            },
        )
        return ReconstructionExecutionResult(
            object_id=request.object_id,
            status="completed",
            gaussian_path=request.gaussian_path,
            camera_params_path=request.camera_params_path,
            recon_meta_path=request.recon_meta_path,
            stdout=completed.stdout.strip(),
        )


def execute_reconstruction(
    request: ReconstructionRequest,
    *,
    config: ReconstructionConfig | None = None,
    executor: ReconstructionExecutor | None = None,
) -> ReconstructionExecutionResult:
    config = config or ReconstructionConfig()
    executor = executor or LocalSam3DObjectsExecutor()
    write_reconstruction_meta(request, config=config, status="running")
    result = executor.run(request, config)
    write_reconstruction_meta(request, config=config, status=result.status)
    return result
