from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class DitPartSegmentationConfig:
    backend: str = "label3d_adapter"
    output_dirname: str = "dit_partseg"
    part_membership_filename: str = "part_membership_scores.npz"
    unknown_mask_filename: str = "unknown_mask.npz"
    part_segments_filename: str = "part_segments.ply"
    meta_filename: str = "dit_partseg_meta.json"
    command_template: str = ""
    timeout_seconds: int = 3600
    score_threshold: float = 0.3
    require_part_segments: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DitPartSegmentationRequest:
    object_id: str
    object_dir: str
    gaussian_path: str
    prompt_path: str
    output_dir: str
    part_membership_path: str
    unknown_mask_path: str
    part_segments_path: str
    meta_path: str
    part_names: tuple[str, ...]
    legacy_part_membership_path: str
    legacy_unknown_mask_path: str

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "object_dir": self.object_dir,
            "gaussian_path": self.gaussian_path,
            "prompt_path": self.prompt_path,
            "output_dir": self.output_dir,
            "part_membership_path": self.part_membership_path,
            "unknown_mask_path": self.unknown_mask_path,
            "part_segments_path": self.part_segments_path,
            "meta_path": self.meta_path,
            "part_names": list(self.part_names),
            "legacy_part_membership_path": self.legacy_part_membership_path,
            "legacy_unknown_mask_path": self.legacy_unknown_mask_path,
        }


@dataclass(frozen=True)
class DitPartSegmentationExecutionResult:
    object_id: str
    status: str
    part_membership_path: str
    unknown_mask_path: str
    part_segments_path: str
    part_names: tuple[str, ...]
    scores_shape: tuple[int, ...]
    num_unknown: int
    backend: str
    stdout: str = ""

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "status": self.status,
            "part_membership_path": self.part_membership_path,
            "unknown_mask_path": self.unknown_mask_path,
            "part_segments_path": self.part_segments_path,
            "part_names": list(self.part_names),
            "scores_shape": list(self.scores_shape),
            "num_unknown": self.num_unknown,
            "backend": self.backend,
            "stdout": self.stdout,
        }


class DitPartSegmentationExecutor(Protocol):
    def run(
        self,
        request: DitPartSegmentationRequest,
        config: DitPartSegmentationConfig,
    ) -> DitPartSegmentationExecutionResult:
        ...


def _load_prompt_part_names(prompt_path: Path) -> tuple[str, ...]:
    payload = json.loads(prompt_path.read_text(encoding="utf-8"))
    return tuple(str(item["part_name"]) for item in payload.get("parts", []))


def build_dit_part_segmentation_request(
    object_dir: str | Path,
    *,
    config: DitPartSegmentationConfig | None = None,
) -> DitPartSegmentationRequest:
    config = config or DitPartSegmentationConfig()
    object_dir = Path(object_dir)
    gaussian_path = object_dir / "recon3d" / "splat.ply"
    prompt_path = object_dir / "prompt" / "part_prompts.json"
    if not gaussian_path.is_file():
        raise FileNotFoundError(f"missing gaussian input: {gaussian_path}")
    if not prompt_path.is_file():
        raise FileNotFoundError(f"missing part prompt file: {prompt_path}")

    output_dir = object_dir / config.output_dirname
    return DitPartSegmentationRequest(
        object_id=object_dir.name,
        object_dir=str(object_dir),
        gaussian_path=str(gaussian_path),
        prompt_path=str(prompt_path),
        output_dir=str(output_dir),
        part_membership_path=str(output_dir / config.part_membership_filename),
        unknown_mask_path=str(output_dir / config.unknown_mask_filename),
        part_segments_path=str(output_dir / config.part_segments_filename),
        meta_path=str(output_dir / config.meta_filename),
        part_names=_load_prompt_part_names(prompt_path),
        legacy_part_membership_path=str(object_dir / "label3d" / "part_membership_scores.npz"),
        legacy_unknown_mask_path=str(object_dir / "label3d" / "unknown_mask.npz"),
    )


def _load_result_summary(
    request: DitPartSegmentationRequest,
    *,
    backend: str,
    stdout: str = "",
) -> DitPartSegmentationExecutionResult:
    membership = np.load(request.part_membership_path, allow_pickle=True)
    unknown = np.load(request.unknown_mask_path, allow_pickle=True)
    part_names = tuple(str(item) for item in membership["part_names"].tolist())
    scores_shape = tuple(int(item) for item in membership["scores"].shape)
    unknown_mask = unknown["unknown_mask"].astype(bool)
    return DitPartSegmentationExecutionResult(
        object_id=request.object_id,
        status="completed",
        part_membership_path=request.part_membership_path,
        unknown_mask_path=request.unknown_mask_path,
        part_segments_path=request.part_segments_path,
        part_names=part_names,
        scores_shape=scores_shape,
        num_unknown=int(unknown_mask.sum()),
        backend=backend,
        stdout=stdout,
    )


def _write_default_unknown_mask(
    request: DitPartSegmentationRequest,
    *,
    scores: np.ndarray,
    visible_counts: np.ndarray,
    score_threshold: float,
) -> None:
    max_scores = scores.max(axis=0) if scores.shape[0] else np.zeros(scores.shape[1], dtype=np.float32)
    invisible = visible_counts <= 0
    low_confidence = max_scores < float(score_threshold)
    unknown_mask = invisible | low_confidence
    np.savez(
        request.unknown_mask_path,
        unknown_mask=unknown_mask.astype(np.uint8),
        visible_counts=visible_counts.astype(np.float32),
        low_confidence=low_confidence.astype(np.uint8),
        invisible=invisible.astype(np.uint8),
        part_names=np.asarray([], dtype=object),
    )


class Label3DAdapterDitPartSegmentationExecutor:
    def run(
        self,
        request: DitPartSegmentationRequest,
        config: DitPartSegmentationConfig,
    ) -> DitPartSegmentationExecutionResult:
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        legacy_membership = Path(request.legacy_part_membership_path)
        legacy_unknown = Path(request.legacy_unknown_mask_path)
        if not legacy_membership.is_file():
            raise FileNotFoundError(
                "label3d adapter needs an existing membership file; "
                f"missing: {legacy_membership}"
            )
        if not legacy_unknown.is_file():
            raise FileNotFoundError(
                "label3d adapter needs an existing unknown mask file; "
                f"missing: {legacy_unknown}"
            )
        shutil.copy2(legacy_membership, request.part_membership_path)
        shutil.copy2(legacy_unknown, request.unknown_mask_path)
        return _load_result_summary(request, backend=config.backend)


class ExternalDitPartSegmentationExecutor:
    def run(
        self,
        request: DitPartSegmentationRequest,
        config: DitPartSegmentationConfig,
    ) -> DitPartSegmentationExecutionResult:
        if not config.command_template.strip():
            raise ValueError("command_template is required for external DIT part segmentation")
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        values = request.to_dict()
        command = config.command_template.format(**values)
        completed = subprocess.run(
            shlex.split(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
        )
        stdout = (completed.stdout or "") + (completed.stderr or "")
        if completed.returncode != 0:
            raise RuntimeError(
                "external DIT part segmentation failed "
                f"with code {completed.returncode}: {stdout[-4000:]}"
            )
        if not Path(request.part_membership_path).is_file():
            raise FileNotFoundError(
                "external DIT command did not write membership file: "
                f"{request.part_membership_path}"
            )
        membership = np.load(request.part_membership_path, allow_pickle=True)
        scores = membership["scores"].astype(np.float32)
        visible_counts = membership.get("visible_counts")
        if visible_counts is None:
            visible_counts = np.ones(scores.shape[1], dtype=np.float32)
        visible_counts = np.asarray(visible_counts, dtype=np.float32)
        if not Path(request.unknown_mask_path).is_file():
            _write_default_unknown_mask(
                request,
                scores=scores,
                visible_counts=visible_counts,
                score_threshold=config.score_threshold,
            )
        if config.require_part_segments and not Path(request.part_segments_path).is_file():
            raise FileNotFoundError(
                "external DIT command did not write part segments: "
                f"{request.part_segments_path}"
            )
        return _load_result_summary(request, backend=config.backend, stdout=stdout)


class Seed3DReproDitPartSegmentationExecutor:
    def run(
        self,
        request: DitPartSegmentationRequest,
        config: DitPartSegmentationConfig,
    ) -> DitPartSegmentationExecutionResult:
        from research.dit.seed3d_partseg_repro import PartSegReproConfig, run_partseg_repro

        result = run_partseg_repro(
            request.object_dir,
            config=PartSegReproConfig(output_dirname=config.output_dirname),
        )
        part_segments = Path(result.output_dir) / "part_segments_colored.ply"
        if part_segments.is_file() and part_segments != Path(request.part_segments_path):
            shutil.copy2(part_segments, request.part_segments_path)
        return _load_result_summary(request, backend=config.backend)


def write_dit_part_segmentation_meta(
    request: DitPartSegmentationRequest,
    *,
    config: DitPartSegmentationConfig | None = None,
    status: str = "planned",
    execution: DitPartSegmentationExecutionResult | None = None,
) -> str:
    config = config or DitPartSegmentationConfig()
    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "object_id": request.object_id,
        "status": status,
        "dit_part_segmentation_config": config.to_dict(),
        "input": {
            "object_dir": request.object_dir,
            "gaussian_path": request.gaussian_path,
            "prompt_path": request.prompt_path,
        },
        "output": {
            "part_membership_path": request.part_membership_path,
            "unknown_mask_path": request.unknown_mask_path,
            "part_segments_path": request.part_segments_path,
        },
        "part_names": list(request.part_names),
        "method_note": (
            "Seed3D 2.0 describes PartSeg followed by PartDiT; this module is "
            "the pipeline integration point and expects external model outputs "
            "unless backend='label3d_adapter'."
        ),
    }
    if execution is not None:
        payload["execution"] = execution.to_dict()
    path = Path(request.meta_path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(path)


def execute_dit_part_segmentation(
    request: DitPartSegmentationRequest,
    *,
    config: DitPartSegmentationConfig | None = None,
    executor: DitPartSegmentationExecutor | None = None,
) -> DitPartSegmentationExecutionResult:
    config = config or DitPartSegmentationConfig()
    if executor is None:
        if config.backend == "seed3d_partseg_repro":
            executor = Seed3DReproDitPartSegmentationExecutor()
        else:
            executor = Label3DAdapterDitPartSegmentationExecutor()
    write_dit_part_segmentation_meta(request, config=config, status="running")
    result = executor.run(request, config)
    write_dit_part_segmentation_meta(request, config=config, status=result.status, execution=result)
    return result
