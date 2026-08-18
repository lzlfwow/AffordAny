"""Projection utilities for source-image P1 alignment.

In v4, projection coordinates are pre-computed during VLM cache building
(using SAM3D's lifting_camera_params). This module only provides the
bilinear sampling function used by the model at training time.

Pre-computation formula (in build_cosmos_vlm_cache.py):
    P_cam = xyz_local * scale @ R + translation
    u = cx - fx * P_cam_x / P_cam_z
    v = cy - fy * P_cam_y / P_cam_z
where R = quaternion_to_matrix(rotation) from lifting_camera_params_fixed.json
and intrinsics = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_vlm_features(
    visual_tokens: torch.Tensor,
    proj_coords: torch.Tensor,
    proj_visible: torch.Tensor,
    grid_size: int = 12,
    num_views: int = 1,
) -> torch.Tensor:
    """Bilinear-sample VLM features at projected 2D locations.

    Args:
        visual_tokens: [B, num_views*grid_size^2, D]
        proj_coords:   [B, N, num_views, 2] normalized (u, v) in [0, 1]
        proj_visible:  [B, N, num_views] visibility mask
        grid_size:     spatial resolution per view
        num_views:     number of views (1 for source image)

    Returns:
        [B, N, D] per-point sampled features (visibility-weighted average).
    """
    B, _, D = visual_tokens.shape

    per_view_feats = []
    for vi in range(num_views):
        start = vi * grid_size * grid_size
        end = start + grid_size * grid_size
        grid = visual_tokens[:, start:end, :].view(
            B, grid_size, grid_size, D,
        ).permute(0, 3, 1, 2)  # [B, D, H, W]

        uv = proj_coords[:, :, vi, :]  # [B, N, 2]
        grid_xy = (2.0 * uv - 1.0).unsqueeze(1)  # [B, 1, N, 2]

        sampled = F.grid_sample(
            grid, grid_xy, mode="bilinear", padding_mode="zeros", align_corners=False,
        ).squeeze(2).permute(0, 2, 1)  # [B, N, D]

        per_view_feats.append(sampled)

    stacked = torch.stack(per_view_feats, dim=2)  # [B, N, V, D]
    mask = proj_visible.unsqueeze(-1).float()
    vis_count = mask.sum(dim=2).clamp(min=1.0)
    return (stacked * mask).sum(dim=2) / vis_count
