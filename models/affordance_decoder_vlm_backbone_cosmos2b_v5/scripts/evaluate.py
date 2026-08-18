"""Evaluate a VLM-backbone v3 affordance decoder checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "models" / "affordance_decoder_vlm_backbone_cosmos2b_v5" / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from affordance_decoder_vlm_backbone_cosmos2b_v5.dataset import (  # noqa: E402
    VLMBackboneDatasetV5,
    collate_vlm_backbone_v5_batch,
)
from affordance_decoder_vlm_backbone_cosmos2b_v5.metrics import compute_basic_metrics  # noqa: E402
from affordance_decoder_vlm_backbone_cosmos2b_v5.model import AffordanceVLMBackboneDecoderV5  # noqa: E402
from affordance_decoder_vlm_backbone_cosmos2b_v5.utils import (  # noqa: E402
    save_json,
    validate_cache_config,
)


METRIC_KEYS = [
    "iou", "miou", "auc", "mae", "sim", "pred_pos", "truth_pos",
    "prob_mean", "prob_max", "valid_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Cosmos-2B VLM-backbone v3 affordance decoder checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--positive-threshold", type=float, default=None)
    return parser.parse_args()


def _ckpt_float(args: dict, key: str, default: float) -> float:
    return float(args.get(key, default))


def _ckpt_int(args: dict, key: str, default: int) -> int:
    return int(args.get(key, default))


def build_model(checkpoint: dict, device: torch.device) -> AffordanceVLMBackboneDecoderV5:
    ckpt_args = checkpoint.get("args", {})
    disable_p1 = ckpt_args.get("disable_p1", False)
    disable_p3 = ckpt_args.get("disable_p3", False)
    model = AffordanceVLMBackboneDecoderV5(
        point_feature_size=13,
        vlm_hidden_size=_ckpt_int(ckpt_args, "vlm_hidden_size", 2048),
        hidden_size=_ckpt_int(ckpt_args, "hidden_size", 256),
        point_depth=_ckpt_int(ckpt_args, "point_depth", 3),
        decoder_depth=_ckpt_int(ckpt_args, "decoder_depth", 4),
        decoder_heads=_ckpt_int(ckpt_args, "decoder_heads", 8),
        dropout=_ckpt_float(ckpt_args, "dropout", 0.1),
        instruction_dropout_rate=0.0,
        num_prototypes=_ckpt_int(ckpt_args, "num_prototypes", 16),
        bottleneck_layers=_ckpt_int(ckpt_args, "bottleneck_layers", 2),
        num_views=_ckpt_int(ckpt_args, "num_views", 4),
        grid_size=_ckpt_int(ckpt_args, "grid_size", 12),
        use_projection=not disable_p1,
        use_bottleneck=not disable_p3,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def write_markdown(path: Path, payload: dict) -> None:
    metrics = payload["metrics"]
    lines = [
        "# Cosmos-2B VLM Backbone V3 Affordance Decoder Evaluation",
        "",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Manifest: `{payload['manifest']}`",
        f"- Samples: `{payload['num_samples']}`",
        f"- Eval threshold: `{payload['threshold']}`",
        f"- Positive threshold: `{payload['positive_threshold']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in METRIC_KEYS:
        lines.append(f"| {key} | {float(metrics[key]):.6f} |")
    path.write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    threshold = args.threshold if args.threshold is not None else _ckpt_float(ckpt_args, "threshold", 0.5)
    positive_threshold = (
        args.positive_threshold if args.positive_threshold is not None
        else _ckpt_float(ckpt_args, "positive_threshold", 0.0)
    )
    num_views = _ckpt_int(ckpt_args, "num_views", 4)
    grid_size = _ckpt_int(ckpt_args, "grid_size", 12)
    vlm_hidden_size = _ckpt_int(ckpt_args, "vlm_hidden_size", 2048)
    validate_cache_config(
        args.manifest.parent,
        vlm_hidden_size=vlm_hidden_size,
        num_views=num_views,
        grid_size=grid_size,
    )

    dataset = VLMBackboneDatasetV5(
        args.manifest, num_views=num_views, grid_size=grid_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_vlm_backbone_v5_batch,
        pin_memory=torch.cuda.is_available(),
    )
    model = build_model(checkpoint, device)

    prob_list: list[torch.Tensor] = []
    target_list: list[torch.Tensor] = []
    category_list: list[str] = []
    for batch in loader:
        logits = model(
            batch["features"].to(device),
            batch["xyz"].to(device),
            batch["aff_hidden"].to(device),
            batch["visual_tokens"].to(device),
            proj_coords=batch["proj_coords"].to(device),
            proj_visible=batch["proj_visible"].to(device),
        )["logits"]
        prob_list.append(torch.sigmoid(logits).detach().cpu())
        target_list.append(batch["heatmap"].detach().cpu().clamp(0.0, 1.0))
        category_list.extend(batch["category_names"])

    prob = torch.cat(prob_list, dim=0)
    target = torch.cat(target_list, dim=0)
    metrics = compute_basic_metrics(
        prob, target, threshold=threshold, positive_threshold=positive_threshold,
        categories=category_list,
    )

    payload = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "num_samples": len(dataset),
        "num_points_per_sample": int(target.shape[1]),
        "threshold": threshold,
        "positive_threshold": positive_threshold,
        "metrics": metrics,
    }
    save_json(args.output_json, payload)
    write_markdown(args.output_md or args.output_json.with_suffix(".md"), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
