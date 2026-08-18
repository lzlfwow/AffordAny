from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from plyfile import PlyData


@dataclass(frozen=True)
class FusionConfig:
    postprocess_variant: str = "object1_v3"
    score_threshold: float = 0.3
    ambiguity_margin: float = 0.05
    duplicate_part_prune_enabled: bool = True
    duplicate_part_count_tolerance: float = 0.3
    duplicate_part_overlap_threshold: float = 0.5
    hole_fill_enabled: bool = True
    hole_fill_knn_k: int = 32
    hole_fill_min_support_neighbors: int = 2
    hole_fill_min_support_ratio: float = 0.3
    hole_fill_assigned_score: float = 0.1  # legacy compatibility field
    hole_fill_max_distance_factor: float = 7.0
    hole_fill_max_iterations: int = 32
    part_membership_filename: str = "part_membership_scores.npz"
    unknown_mask_filename: str = "unknown_mask.npz"
    fusion_meta_filename: str = "fusion_meta.json"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FusionRequest:
    object_id: str
    gaussian_path: str
    part_votes_path: str
    gaussian_scores_path: str
    visibility_dir: str
    label3d_dir: str
    part_membership_path: str
    unknown_mask_path: str
    fusion_meta_path: str
    part_names: tuple[str, ...]
    scores_shape: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "gaussian_path": self.gaussian_path,
            "part_votes_path": self.part_votes_path,
            "gaussian_scores_path": self.gaussian_scores_path,
            "visibility_dir": self.visibility_dir,
            "label3d_dir": self.label3d_dir,
            "part_membership_path": self.part_membership_path,
            "unknown_mask_path": self.unknown_mask_path,
            "fusion_meta_path": self.fusion_meta_path,
            "part_names": list(self.part_names),
            "scores_shape": list(self.scores_shape),
        }


@dataclass(frozen=True)
class FusionExecutionResult:
    object_id: str
    status: str
    part_membership_path: str
    unknown_mask_path: str
    fusion_meta_path: str
    part_names: tuple[str, ...]
    num_unknown: int
    num_filled_unknown: int
    dropped_part_names: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "status": self.status,
            "part_membership_path": self.part_membership_path,
            "unknown_mask_path": self.unknown_mask_path,
            "fusion_meta_path": self.fusion_meta_path,
            "part_names": list(self.part_names),
            "num_unknown": self.num_unknown,
            "num_filled_unknown": self.num_filled_unknown,
            "dropped_part_names": list(self.dropped_part_names),
        }


class FusionExecutor(Protocol):
    def run(
        self,
        request: FusionRequest,
        config: FusionConfig,
    ) -> FusionExecutionResult:
        ...


def _load_gaussian_points(path: str | Path) -> np.ndarray:
    vertex = PlyData.read(str(path))["vertex"]
    return np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )


def _query_knn_indices(
    points_xyz: np.ndarray,
    query_xyz: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points_xyz) == 0 or len(query_xyz) == 0:
        return (
            np.zeros((len(query_xyz), 0), dtype=np.float32),
            np.zeros((len(query_xyz), 0), dtype=np.int64),
        )
    k = max(1, min(int(k), int(points_xyz.shape[0])))
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(points_xyz)
        distances, indices = tree.query(query_xyz, k=k, workers=-1)
    except Exception:
        from sklearn.neighbors import NearestNeighbors

        neighbors = NearestNeighbors(n_neighbors=k, algorithm="auto")
        neighbors.fit(points_xyz)
        distances, indices = neighbors.kneighbors(query_xyz)
    distances = np.asarray(distances, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
    if indices.ndim == 1:
        indices = indices[:, None]
    return distances, indices


def _fill_unknown_holes(
    *,
    part_scores: np.ndarray,
    initial_unknown_mask: np.ndarray,
    points_xyz: np.ndarray | None,
    config: FusionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    num_points = int(initial_unknown_mask.shape[0])
    filled = np.zeros(num_points, dtype=bool)
    filled_scores = np.zeros(num_points, dtype=np.float32)
    if (
        not config.hole_fill_enabled
        or points_xyz is None
        or num_points == 0
        or points_xyz.shape[0] != num_points
    ):
        return filled, filled_scores

    part_scores = np.asarray(part_scores, dtype=np.float32)
    positive_seed = np.isfinite(part_scores) & (part_scores > 0.0)
    state = np.asarray(positive_seed, dtype=bool)
    current_scores = np.where(positive_seed, part_scores, np.float32(0.0)).astype(np.float32)
    if not np.any(state):
        return filled, filled_scores

    base_distances, _ = _query_knn_indices(points_xyz, points_xyz, k=2)
    if base_distances.shape[1] < 2:
        return filled, filled_scores
    local_spacing = float(np.median(base_distances[:, 1]))
    if not np.isfinite(local_spacing) or local_spacing <= 0.0:
        return filled, filled_scores
    max_support_distance = local_spacing * float(config.hole_fill_max_distance_factor)

    knn_distances, knn_indices = _query_knn_indices(
        points_xyz,
        points_xyz,
        k=config.hole_fill_knn_k + 1,
    )
    neighbor_indices = knn_indices[:, 1:]
    neighbor_distances = knn_distances[:, 1:]
    neighbor_valid = neighbor_distances <= max_support_distance
    neighbor_total = neighbor_valid.sum(axis=1)
    candidate_mask = initial_unknown_mask.copy()
    for _ in range(max(int(config.hole_fill_max_iterations), 1)):
        propagated_support = state[neighbor_indices] & neighbor_valid
        support_count = propagated_support.sum(axis=1)
        required_count = np.ceil(neighbor_total.astype(np.float32) * float(config.hole_fill_min_support_ratio))
        add_mask = (
            candidate_mask
            & (~state)
            & (neighbor_total >= config.hole_fill_min_support_neighbors)
            & (support_count >= config.hole_fill_min_support_neighbors)
            & (support_count.astype(np.float32) >= required_count)
        )
        if not np.any(add_mask):
            break
        labeled_support = positive_seed[neighbor_indices] & neighbor_valid
        labeled_count = labeled_support.sum(axis=1)
        labeled_score_sum = np.where(
            labeled_support,
            part_scores[neighbor_indices],
            np.float32(0.0),
        ).sum(axis=1, dtype=np.float32)
        propagated_score_sum = np.where(
            propagated_support,
            current_scores[neighbor_indices],
            np.float32(0.0),
        ).sum(axis=1, dtype=np.float32)
        labeled_mean = labeled_score_sum / np.maximum(labeled_count.astype(np.float32), 1.0)
        propagated_mean = propagated_score_sum / np.maximum(support_count.astype(np.float32), 1.0)
        assigned_scores = np.where(labeled_count > 0, labeled_mean, propagated_mean).astype(np.float32)
        current_scores[add_mask] = assigned_scores[add_mask]
        filled_scores[add_mask] = assigned_scores[add_mask]
        state = state | add_mask
        filled = filled | add_mask
    return filled, filled_scores


def _prune_duplicate_parts(
    scores: np.ndarray,
    part_names: tuple[str, ...],
    *,
    config: FusionConfig,
) -> dict[str, object]:
    scores = np.asarray(scores, dtype=np.float32)
    part_names = tuple(str(item) for item in part_names)
    if scores.shape[0] != len(part_names):
        raise ValueError(
            f"scores rows ({scores.shape[0]}) must match part_names ({len(part_names)})"
        )
    if not config.duplicate_part_prune_enabled or scores.shape[0] <= 1:
        return {
            "scores": scores,
            "part_names": part_names,
            "keep_mask": np.ones(scores.shape[0], dtype=bool),
            "meta": {
                "enabled": bool(config.duplicate_part_prune_enabled),
                "num_parts_before": int(scores.shape[0]),
                "num_parts_after": int(scores.shape[0]),
                "dropped_parts": [],
                "dropped_part_names": [],
            },
        }

    positive_masks = np.isfinite(scores) & (scores > 0.0)
    point_counts = positive_masks.sum(axis=1).astype(np.int64)
    keep_mask = np.ones(scores.shape[0], dtype=bool)
    dropped_parts: list[dict[str, object]] = []
    ordered_indices = sorted(
        range(scores.shape[0]),
        key=lambda idx: (int(point_counts[idx]), part_names[idx], idx),
    )
    count_tolerance = float(config.duplicate_part_count_tolerance)
    overlap_threshold = float(config.duplicate_part_overlap_threshold)

    for base_order_index, base_idx in enumerate(ordered_indices):
        if not keep_mask[base_idx]:
            continue
        base_count = int(point_counts[base_idx])
        if base_count <= 0:
            continue
        base_mask = positive_masks[base_idx]
        for candidate_idx in ordered_indices[base_order_index + 1 :]:
            if not keep_mask[candidate_idx]:
                continue
            candidate_count = int(point_counts[candidate_idx])
            if candidate_count <= 0:
                continue
            diff_ratio = abs(base_count - candidate_count) / float(max(base_count, candidate_count))
            if diff_ratio >= count_tolerance:
                continue
            overlap_count = int(np.logical_and(base_mask, positive_masks[candidate_idx]).sum())
            overlap_ratio = overlap_count / float(min(base_count, candidate_count))
            if overlap_ratio <= overlap_threshold:
                continue
            keep_mask[candidate_idx] = False
            dropped_parts.append(
                {
                    "kept_part_name": part_names[base_idx],
                    "dropped_part_name": part_names[candidate_idx],
                    "kept_point_count": base_count,
                    "dropped_point_count": candidate_count,
                    "count_diff_ratio": diff_ratio,
                    "overlap_count": overlap_count,
                    "overlap_ratio_over_smaller": overlap_ratio,
                }
            )

    kept_indices = np.flatnonzero(keep_mask)
    kept_names = tuple(part_names[idx] for idx in kept_indices.tolist())
    dropped_names = [part_names[idx] for idx in np.flatnonzero(~keep_mask).tolist()]
    return {
        "scores": scores[keep_mask],
        "part_names": kept_names,
        "keep_mask": keep_mask,
        "meta": {
            "enabled": True,
            "num_parts_before": int(scores.shape[0]),
            "num_parts_after": int(len(kept_names)),
            "dropped_parts": dropped_parts,
            "dropped_part_names": dropped_names,
        },
    }


def build_membership_and_unknown_arrays(
    scores: np.ndarray,
    visible_counts: np.ndarray,
    *,
    config: FusionConfig,
    points_xyz: np.ndarray | None = None,
    part_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float32)
    visible_counts = np.asarray(visible_counts, dtype=np.float32)
    if part_names is None:
        part_names = tuple(f"part_{index:03d}" for index in range(scores.shape[0]))
    else:
        part_names = tuple(str(item) for item in part_names)
    if len(part_names) != scores.shape[0]:
        raise ValueError(
            f"part_names ({len(part_names)}) must align with score rows ({scores.shape[0]})"
        )
    if scores.shape[0] == 0:
        num_points = visible_counts.shape[0]
        updated_scores = scores.copy()
        max_scores = np.zeros(num_points, dtype=np.float32)
        best_indices = np.full(num_points, -1, dtype=np.int32)
        ambiguous = np.zeros(num_points, dtype=bool)
        low_conf = np.ones(num_points, dtype=bool)
        invisible = visible_counts <= 0
        initial_unknown_mask = np.ones(num_points, dtype=bool)
        candidate_unknown = initial_unknown_mask.copy()
        filled_unknown = np.zeros(num_points, dtype=bool)
        filled_by_part = np.zeros((0, num_points), dtype=bool)
        duplicate_pruning = {
            "enabled": bool(config.duplicate_part_prune_enabled),
            "num_parts_before": 0,
            "num_parts_after": 0,
            "dropped_parts": [],
            "dropped_part_names": [],
        }
    else:
        pruned = _prune_duplicate_parts(scores, part_names, config=config)
        scores = np.asarray(pruned["scores"], dtype=np.float32)
        part_names = tuple(str(item) for item in pruned["part_names"])
        duplicate_pruning = dict(pruned["meta"])
        original_max_scores = scores.max(axis=0)
        second_scores = (
            np.partition(scores, -2, axis=0)[-2]
            if scores.shape[0] > 1
            else np.zeros_like(original_max_scores)
        )
        original_low_conf = original_max_scores < config.score_threshold
        invisible = visible_counts <= 0
        # Keep ambiguous as a diagnostic signal, but do not drop it from supervision.
        initial_unknown_mask = invisible | original_low_conf
        candidate_unknown = initial_unknown_mask.copy()
        filled_by_part = np.zeros_like(scores, dtype=bool)
        updated_scores = scores.copy()
        for part_index in range(scores.shape[0]):
            part_filled, part_filled_scores = _fill_unknown_holes(
                part_scores=scores[part_index],
                initial_unknown_mask=initial_unknown_mask,
                points_xyz=points_xyz,
                config=config,
            )
            filled_by_part[part_index] = part_filled
            if np.any(part_filled):
                updated_scores[part_index, part_filled] = np.maximum(
                    updated_scores[part_index, part_filled],
                    part_filled_scores[part_filled],
                )
        filled_unknown = filled_by_part.any(axis=0)
        max_scores = updated_scores.max(axis=0)
        best_indices = updated_scores.argmax(axis=0).astype(np.int32)
        updated_second_scores = (
            np.partition(updated_scores, -2, axis=0)[-2]
            if updated_scores.shape[0] > 1
            else np.zeros_like(max_scores)
        )
        ambiguous = (max_scores - updated_second_scores) < config.ambiguity_margin
        low_conf = max_scores < config.score_threshold
    unknown_mask = invisible | low_conf
    return {
        "scores": updated_scores,
        "max_scores": max_scores,
        "best_indices": best_indices,
        "ambiguous": ambiguous,
        "low_conf": low_conf,
        "invisible": invisible,
        "initial_unknown_mask": initial_unknown_mask,
        "candidate_unknown": candidate_unknown,
        "filled_unknown": filled_unknown,
        "filled_by_part": filled_by_part,
        "unknown_mask": unknown_mask,
        "part_names": part_names,
        "duplicate_pruning": duplicate_pruning,
    }


def build_fusion_request(
    object_dir: str | Path,
    *,
    config: FusionConfig | None = None,
) -> FusionRequest:
    config = config or FusionConfig()
    object_dir = Path(object_dir)
    project3d_dir = object_dir / "project3d"
    gaussian_path = object_dir / "recon3d" / "splat.ply"
    part_votes_path = project3d_dir / "part_votes.json"
    gaussian_scores_path = project3d_dir / "gaussian_part_scores.npz"
    visibility_dir = project3d_dir / "visibility"

    for path in [part_votes_path, gaussian_scores_path]:
        if not path.is_file():
            raise FileNotFoundError(f"missing required input: {path}")
    if not visibility_dir.is_dir():
        raise FileNotFoundError(f"missing visibility directory: {visibility_dir}")

    scores = np.load(gaussian_scores_path, allow_pickle=True)
    part_names = tuple(scores["part_names"].tolist())
    scores_shape = tuple(scores["scores"].shape)

    label3d_dir = object_dir / "label3d"
    return FusionRequest(
        object_id=object_dir.name,
        gaussian_path=str(gaussian_path),
        part_votes_path=str(part_votes_path),
        gaussian_scores_path=str(gaussian_scores_path),
        visibility_dir=str(visibility_dir),
        label3d_dir=str(label3d_dir),
        part_membership_path=str(label3d_dir / config.part_membership_filename),
        unknown_mask_path=str(label3d_dir / config.unknown_mask_filename),
        fusion_meta_path=str(label3d_dir / config.fusion_meta_filename),
        part_names=part_names,
        scores_shape=scores_shape,
    )


def write_fusion_meta(
    request: FusionRequest,
    *,
    config: FusionConfig | None = None,
    status: str = "planned",
    execution: FusionExecutionResult | None = None,
) -> str:
    config = config or FusionConfig()
    label3d_dir = Path(request.label3d_dir)
    label3d_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "object_id": request.object_id,
        "status": status,
        "fusion_config": config.to_dict(),
        "input": {
            "gaussian_path": request.gaussian_path,
            "part_votes_path": request.part_votes_path,
            "gaussian_scores_path": request.gaussian_scores_path,
            "visibility_dir": request.visibility_dir,
        },
        "output": {
            "part_membership_path": request.part_membership_path,
            "unknown_mask_path": request.unknown_mask_path,
        },
        "part_names": list(request.part_names),
        "scores_shape": list(request.scores_shape),
    }
    if execution is not None:
        payload["execution"] = execution.to_dict()
    path = Path(request.fusion_meta_path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    return str(path)


class LocalFusionExecutor:
    def run(
        self,
        request: FusionRequest,
        config: FusionConfig,
    ) -> FusionExecutionResult:
        label3d_dir = Path(request.label3d_dir)
        label3d_dir.mkdir(parents=True, exist_ok=True)

        project_scores = np.load(request.gaussian_scores_path, allow_pickle=True)
        part_names = tuple(project_scores["part_names"].tolist())
        scores = project_scores["scores"].astype(np.float32)
        visible_counts = project_scores["visible_counts"].astype(np.float32)

        points_xyz = None
        if config.hole_fill_enabled and scores.shape[0] > 0:
            gaussian_path = Path(request.gaussian_path)
            if not gaussian_path.is_file():
                raise FileNotFoundError(
                    f"missing gaussian file for fusion postprocess: {gaussian_path}"
                )
            points_xyz = _load_gaussian_points(gaussian_path)

        payload = build_membership_and_unknown_arrays(
            scores,
            visible_counts,
            config=config,
            points_xyz=points_xyz,
            part_names=part_names,
        )
        part_names = tuple(payload["part_names"])
        updated_scores = payload["scores"]
        max_scores = payload["max_scores"]
        best_indices = payload["best_indices"]
        ambiguous = payload["ambiguous"]
        low_conf = payload["low_conf"]
        invisible = payload["invisible"]
        initial_unknown_mask = payload["initial_unknown_mask"]
        candidate_unknown = payload["candidate_unknown"]
        filled_unknown = payload["filled_unknown"]
        filled_by_part = payload["filled_by_part"]
        unknown_mask = payload["unknown_mask"]
        duplicate_pruning = dict(payload["duplicate_pruning"])

        np.savez(
            request.part_membership_path,
            part_names=np.asarray(part_names, dtype=object),
            scores=updated_scores,
            best_part_index=best_indices,
            max_scores=max_scores,
            visible_counts=visible_counts,
        )
        np.savez(
            request.unknown_mask_path,
            unknown_mask=unknown_mask.astype(np.uint8),
            visible_counts=visible_counts,
            low_confidence=low_conf.astype(np.uint8),
            ambiguous=ambiguous.astype(np.uint8),
            invisible=invisible.astype(np.uint8),
            initial_unknown_mask=initial_unknown_mask.astype(np.uint8),
            postprocess_candidate=candidate_unknown.astype(np.uint8),
            postprocess_filled=filled_unknown.astype(np.uint8),
            postprocess_filled_by_part=filled_by_part.astype(np.uint8),
            part_names=np.asarray(part_names, dtype=object),
        )

        return FusionExecutionResult(
            object_id=request.object_id,
            status="completed",
            part_membership_path=request.part_membership_path,
            unknown_mask_path=request.unknown_mask_path,
            fusion_meta_path=request.fusion_meta_path,
            part_names=part_names,
            num_unknown=int(unknown_mask.sum()),
            num_filled_unknown=int(filled_unknown.sum()),
            dropped_part_names=tuple(
                str(item) for item in duplicate_pruning.get("dropped_part_names", [])
            ),
        )


def execute_fusion(
    request: FusionRequest,
    *,
    config: FusionConfig | None = None,
    executor: FusionExecutor | None = None,
) -> FusionExecutionResult:
    config = config or FusionConfig()
    executor = executor or LocalFusionExecutor()
    write_fusion_meta(request, config=config, status="running")
    result = executor.run(request, config)
    write_fusion_meta(request, config=config, status=result.status, execution=result)
    return result
