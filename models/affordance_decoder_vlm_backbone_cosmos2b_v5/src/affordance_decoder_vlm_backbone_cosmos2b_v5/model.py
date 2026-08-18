"""Cosmos-2B VLM-backbone affordance decoder v5.

Changes from v4:
  - P3 text-conditioned prototypes: FiLM conditioning via aff_token.
  - GPBlock between P3 and cross-attention: bidirectional fusion of
    point_tokens and compressed prototypes (not raw VLM tokens).
  - No P2b augmentation (same as v4).
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualPointMLPBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.net(self.norm(tokens))


class ProjectionFeatureInjector(nn.Module):
    """P1: Sample VLM features at projected 2D locations."""

    def __init__(self, hidden_size: int, num_views: int = 1, grid_size: int = 12):
        super().__init__()
        self.num_views = num_views
        self.grid_size = grid_size
        if num_views > 1:
            self.view_embed = nn.Embedding(num_views, hidden_size)
        self.combine = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        proj_coords: torch.Tensor,
        proj_visible: torch.Tensor,
    ) -> torch.Tensor:
        B, _, D = visual_tokens.shape
        per_view_feats = []
        for vi in range(self.num_views):
            start = vi * self.grid_size * self.grid_size
            end = start + self.grid_size * self.grid_size
            grid = visual_tokens[:, start:end, :].view(
                B, self.grid_size, self.grid_size, D,
            ).permute(0, 3, 1, 2)
            uv = proj_coords[:, :, vi, :]
            grid_xy = (2.0 * uv - 1.0).unsqueeze(1)
            sampled = F.grid_sample(
                grid, grid_xy, mode="bilinear", padding_mode="zeros", align_corners=False,
            ).squeeze(2).permute(0, 2, 1)
            if hasattr(self, 'view_embed'):
                sampled = sampled + self.view_embed.weight[vi]
            per_view_feats.append(sampled)
        stacked = torch.stack(per_view_feats, dim=2)
        mask = proj_visible.unsqueeze(-1).float()
        vis_count = mask.sum(dim=2).clamp(min=1.0)
        aggregated = (stacked * mask).sum(dim=2) / vis_count
        return self.combine(aggregated)


class SemanticBottleneckConditioned(nn.Module):
    """P3 with FiLM text conditioning: prototypes are modulated by aff_token."""

    def __init__(
        self,
        hidden_size: int,
        num_prototypes: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_text_cond: bool = True,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.use_text_cond = use_text_cond
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_size) * 0.02)
        if use_text_cond:
            self.cond_scale = nn.Linear(hidden_size, hidden_size)
            self.cond_shift = nn.Linear(hidden_size, hidden_size)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_size),
        )

    def forward(self, vlm_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vlm_tokens: [B, V+1, D] where vlm_tokens[:, 0] is aff_token.
        Returns:
            [B, K, D] text-conditioned compressed prototypes.
        """
        B = vlm_tokens.shape[0]
        aff_token = vlm_tokens[:, 0]
        if self.use_text_cond:
            scale = self.cond_scale(aff_token).unsqueeze(1)
            shift = self.cond_shift(aff_token).unsqueeze(1)
            queries = self.prototypes.unsqueeze(0) * (1.0 + scale) + shift
        else:
            queries = self.prototypes.unsqueeze(0).expand(B, -1, -1)
        return self.decoder(tgt=queries, memory=vlm_tokens)


class PrototypePointGPBlock(nn.Module):
    """Bidirectional fusion between point_tokens and prototypes.

    Group:   prototypes(Q) attend to points(KV) → geometry-aware prototypes
    Mix:     token-mixing + channel-mixing on prototypes
    Ungroup: points(Q) attend to mixed prototypes(KV) → semantically enriched points
    """

    def __init__(self, dim: int, num_prototypes: int = 16, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        # Group: prototypes attend to points
        self.group_norm = nn.LayerNorm(dim)
        self.group_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )

        # Mix: token + channel mixing on prototypes
        token_hidden = max(num_prototypes // 2, 4)
        self.mix_token_norm = nn.LayerNorm(dim)
        self.mix_token_fc = nn.Sequential(
            nn.Linear(num_prototypes, token_hidden), nn.GELU(),
            nn.Linear(token_hidden, num_prototypes), nn.Dropout(dropout),
        )
        self.mix_channel_norm = nn.LayerNorm(dim)
        self.mix_channel_fc = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

        # Ungroup: points attend to mixed prototypes
        self.ungroup_norm = nn.LayerNorm(dim)
        self.ungroup_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.ungroup_ffn_norm = nn.LayerNorm(dim)
        self.ungroup_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(
        self, point_tokens: torch.Tensor, prototypes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Group: prototypes learn about 3D geometry
        q = self.group_norm(prototypes)
        grouped, _ = self.group_attn(q, point_tokens, point_tokens)
        proto_out = prototypes + grouped

        # Mix: refine prototype representations
        residual = proto_out
        mixed = self.mix_token_norm(proto_out).transpose(1, 2)
        proto_out = residual + self.mix_token_fc(mixed).transpose(1, 2)
        residual = proto_out
        proto_out = residual + self.mix_channel_fc(self.mix_channel_norm(proto_out))

        # Ungroup: points absorb geometry-aware semantic info
        q = self.ungroup_norm(point_tokens)
        ungrouped, _ = self.ungroup_attn(q, proto_out, proto_out)
        point_out = point_tokens + ungrouped
        point_out = point_out + self.ungroup_ffn(self.ungroup_ffn_norm(point_out))

        return point_out, proto_out


class AffordanceVLMBackboneDecoderV5(nn.Module):
    """V5 decoder: P1 + conditioned-P3 + GPBlock(pts↔protos) + cross-attention.

    Flow:
      1. Point embedding + 3x ResidualMLP
      2. VLM projection (aff_token + visual_tokens)
      3. P1: bilinear-sample VLM at projected 2D → add to points
      4. P3-cond: text-conditioned prototypes compress VLM tokens
      5. GPBlock: bidirectional fusion between points and prototypes
      6. Cross-attention: Q=points, KV=[aff_token, prototypes]
      7. Head: per-point logits
    """

    def __init__(
        self,
        *,
        point_feature_size: int = 13,
        vlm_hidden_size: int = 2048,
        hidden_size: int = 256,
        point_depth: int = 3,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        dropout: float = 0.1,
        instruction_dropout_rate: float = 0.0,
        num_prototypes: int = 16,
        bottleneck_layers: int = 2,
        num_views: int = 1,
        grid_size: int = 12,
        use_projection: bool = True,
        use_bottleneck: bool = True,
        use_gpblock: bool = True,
        use_text_cond: bool = True,
    ):
        super().__init__()
        self.instruction_dropout_rate = instruction_dropout_rate
        self.num_views = num_views
        self.grid_size = grid_size
        self.use_projection = use_projection
        self.use_bottleneck = use_bottleneck
        self.use_gpblock = use_gpblock

        self.point_proj = nn.Sequential(
            nn.Linear(point_feature_size, hidden_size),
            nn.GELU(), nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self.xyz_proj = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.GELU(), nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self.vlm_proj = nn.Sequential(
            nn.Linear(vlm_hidden_size, hidden_size),
            nn.GELU(), nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        self.point_blocks = nn.ModuleList(
            ResidualPointMLPBlock(hidden_size, dropout=dropout) for _ in range(point_depth)
        )

        if use_projection:
            self.proj_injector = ProjectionFeatureInjector(
                hidden_size, num_views=num_views, grid_size=grid_size,
            )

        if use_bottleneck:
            self.bottleneck = SemanticBottleneckConditioned(
                hidden_size, num_prototypes=num_prototypes,
                num_heads=decoder_heads, num_layers=bottleneck_layers,
                dropout=dropout, use_text_cond=use_text_cond,
            )

        if use_gpblock:
            self.gp_block = PrototypePointGPBlock(
                hidden_size, num_prototypes=num_prototypes,
                num_heads=decoder_heads, dropout=dropout,
            )

        cross_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=decoder_heads,
            dim_feedforward=hidden_size * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.cross_decoder = nn.TransformerDecoder(
            cross_layer, num_layers=decoder_depth,
            norm=nn.LayerNorm(hidden_size),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        xyz: torch.Tensor,
        aff_hidden: torch.Tensor,
        visual_tokens: torch.Tensor,
        proj_coords: torch.Tensor | None = None,
        proj_visible: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # 1. Point embedding
        point_tokens = self.point_proj(features) + self.xyz_proj(xyz)
        for block in self.point_blocks:
            point_tokens = block(point_tokens)

        B = aff_hidden.shape[0]

        # Instruction dropout
        if self.training and self.instruction_dropout_rate > 0.0:
            drop_mask = torch.rand(B, device=aff_hidden.device) < self.instruction_dropout_rate
            if drop_mask.any():
                aff_hidden = aff_hidden.clone()
                visual_tokens = visual_tokens.clone()
                aff_hidden[drop_mask] = 0.0
                visual_tokens[drop_mask] = 0.0

        # 2. VLM projection
        aff_token = aff_hidden.unsqueeze(1)
        vlm_tokens = torch.cat([aff_token, visual_tokens], dim=1)
        vlm_tokens = self.vlm_proj(vlm_tokens)

        # 3. P1: projection-aware injection
        if self.use_projection and proj_coords is not None and proj_visible is not None:
            proj_feats = self.proj_injector(
                vlm_tokens[:, 1:], proj_coords, proj_visible,
            )
            point_tokens = point_tokens + proj_feats

        # 4. P3-cond: text-conditioned semantic bottleneck
        if self.use_bottleneck:
            proto_feats = self.bottleneck(vlm_tokens)  # [B, K, D]
        else:
            proto_feats = vlm_tokens

        # 5. GPBlock: bidirectional fusion (points ↔ prototypes)
        if self.use_gpblock and self.use_bottleneck:
            point_tokens, proto_feats = self.gp_block(point_tokens, proto_feats)

        # 6. Cross-attention: Q=points, KV=[aff_token, prototypes]
        kv = torch.cat([vlm_tokens[:, :1], proto_feats], dim=1)
        fused_tokens = self.cross_decoder(tgt=point_tokens, memory=kv)

        # 7. Head
        logits = self.head(fused_tokens).squeeze(-1)
        return {"logits": logits, "aff_token": vlm_tokens[:, 0]}
