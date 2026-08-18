"""Self-training with soft pseudo-labels + confidence weighting + DDP.

Based on v5 train.py. Key additions:
  - --pseudo-manifest: pseudo-label VLM cache manifest
  - --pseudo-weight: confidence scaling for pseudo-label samples
  - --use-soft-labels: use continuous [0,1] heatmaps instead of binarized
  - MixedAffordanceDataset for combined real + pseudo training
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[3]
V5_SRC = REPO_ROOT / "models" / "affordance_decoder_vlm_backbone_cosmos2b_v5" / "src"
SELF_SRC = REPO_ROOT / "models" / "affordance_decoder_selftraining_v1" / "src"
for path in (REPO_ROOT, V5_SRC, SELF_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from affordance_decoder_vlm_backbone_cosmos2b_v5.dataset import (  # noqa: E402
    VLMBackboneDatasetV5,
    collate_vlm_backbone_v5_batch,
)
from affordance_decoder_vlm_backbone_cosmos2b_v5.distributed import (  # noqa: E402
    DistributedState,
    add_metric_sums,
    barrier,
    build_train_sampler,
    cleanup_distributed,
    init_distributed,
    maybe_wrap_model,
    rank0_print,
    reduce_metric_sums,
    unwrap_model,
)
from affordance_decoder_vlm_backbone_cosmos2b_v5.metrics import compute_basic_metrics  # noqa: E402
from affordance_decoder_vlm_backbone_cosmos2b_v5.model import AffordanceVLMBackboneDecoderV5  # noqa: E402
from affordance_decoder_vlm_backbone_cosmos2b_v5.utils import save_json  # noqa: E402

from affordance_decoder_selftraining_v1.losses import compute_loss  # noqa: E402
from affordance_decoder_selftraining_v1.mixed_dataset import (  # noqa: E402
    MixedAffordanceDataset,
    collate_mixed_batch,
)


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.module.load_state_dict(state_dict)


METRIC_KEYS = [
    "iou", "miou", "auc", "mae", "sim", "pred_pos", "truth_pos",
    "prob_mean", "prob_max", "valid_ratio",
]


def parse_args() -> argparse.Namespace:
    default_cache = (
        REPO_ROOT / "models" / "affordance_decoder_vlm_backbone_cosmos2b_v4" / "artifacts" / "vlm_cache"
    )
    default_output = (
        REPO_ROOT / "models" / "affordance_decoder_selftraining_v1" / "artifacts" / "experiments"
        / "selftraining_round1"
    )
    parser = argparse.ArgumentParser(description="Self-training with soft pseudo-labels.")

    parser.add_argument("--cache-root", type=Path, default=default_cache)
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pseudo-manifest", type=Path, nargs="*", default=None,
                        help="One or more pseudo-label VLM cache manifests")
    parser.add_argument("--pseudo-weight", type=float, default=0.5, help="Max confidence scaling for pseudo samples")
    parser.add_argument("--pseudo-high-thresh", type=float, default=0.8, help="Pseudo-label high confidence threshold")
    parser.add_argument("--pseudo-low-thresh", type=float, default=0.2, help="Pseudo-label low confidence threshold")
    parser.add_argument("--pseudo-sample-ratio", type=float, default=1.0, help="Fraction of pseudo samples to use (1.0=all)")
    parser.add_argument("--pseudo-bce-scale", type=float, default=0.3, help="BCE weight multiplier for pseudo samples (0=skip, 1=full)")
    parser.add_argument("--pseudo-focal-scale", type=float, default=1.0, help="Focal loss weight multiplier for pseudo samples (0=skip, 1=full)")
    parser.add_argument("--curriculum-start", type=int, default=0, help="Epoch to start introducing pseudo data (0=from start)")
    parser.add_argument("--curriculum-rampup", type=int, default=0, help="Epochs over which pseudo_weight ramps from 0 to target (0=instant)")
    parser.add_argument("--use-soft-labels", action="store_true", help="Use continuous [0,1] soft labels for pseudo data")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Initialize from a pretrained checkpoint")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--vlm-hidden-size", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--point-depth", type=int, default=3)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--instruction-dropout-rate", type=float, default=0.2)
    parser.add_argument("--num-prototypes", type=int, default=16)
    parser.add_argument("--bottleneck-layers", type=int, default=2)
    parser.add_argument("--num-views", type=int, default=1)
    parser.add_argument("--grid-size", type=int, default=12)

    parser.add_argument("--positive-threshold", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--lambda-bce", type=float, default=0.2)
    parser.add_argument("--lambda-focal", type=float, default=1.0)
    parser.add_argument("--lambda-dice", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=8.0)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--selection-metric", default="val_iou")
    parser.add_argument("--selection-mode", choices=["max", "min"], default="max")

    parser.add_argument("--formal-val-unseen-object-manifest", type=Path, default=None)
    parser.add_argument("--formal-val-unseen-category-manifest", type=Path, default=None)
    parser.add_argument("--formal-test-unseen-object-manifest", type=Path, default=None)
    parser.add_argument("--formal-test-unseen-category-manifest", type=Path, default=None)
    parser.add_argument("--formal-train-seen-manifest", type=Path, default=None)
    parser.add_argument("--formal-train-unseen-manifest", type=Path, default=None)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--log-every-batches", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--ema-decay", type=float, default=0.0, help="EMA decay rate (0=disabled, typical: 0.999)")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon for pseudo labels (e.g. 0.05)")
    parser.add_argument("--refresh-every", type=int, default=0, help="Refresh pseudo labels every N epochs using EMA model (0=disabled)")
    parser.add_argument("--refresh-batch-size", type=int, default=64, help="Batch size for pseudo-label refresh inference")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    def _convert(v: Any) -> Any:
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, list):
            return [_convert(x) for x in v]
        return v
    return {key: _convert(value) for key, value in vars(args).items()}


def make_eval_loader(
    manifest: Path, batch_size: int, num_workers: int,
) -> DataLoader:
    dataset = VLMBackboneDatasetV5(manifest)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_vlm_backbone_v5_batch,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module, loader: DataLoader, *, device: torch.device,
    threshold: float, positive_threshold: float,
) -> dict[str, float]:
    model.eval()
    prob_list, target_list, category_list = [], [], []
    for batch in loader:
        logits = model(
            batch["features"].to(device), batch["xyz"].to(device),
            batch["aff_hidden"].to(device), batch["visual_tokens"].to(device),
            proj_coords=batch["proj_coords"].to(device),
            proj_visible=batch["proj_visible"].to(device),
        )["logits"]
        prob_list.append(torch.sigmoid(logits).detach().cpu())
        target_list.append(batch["heatmap"].detach().cpu().clamp(0.0, 1.0))
        category_list.extend(batch["category_names"])
    return compute_basic_metrics(
        torch.cat(prob_list), torch.cat(target_list),
        threshold=threshold, positive_threshold=positive_threshold,
        categories=category_list,
    )


@torch.no_grad()
def evaluate_manifest(
    model: nn.Module, manifest: Path, *, batch_size: int, num_workers: int,
    device: torch.device, threshold: float, positive_threshold: float,
) -> dict[str, float]:
    loader = make_eval_loader(manifest, batch_size, num_workers)
    metrics = evaluate_model(
        model, loader, device=device, threshold=threshold,
        positive_threshold=positive_threshold,
    )
    metrics["samples"] = len(loader.dataset)
    return metrics


def current_is_better(value: float, best: float, mode: str) -> bool:
    if value != value:
        return False
    return value <= best if mode == "min" else value >= best


@torch.no_grad()
def refresh_pseudo_labels(
    ema_model: nn.Module,
    train_dataset: MixedAffordanceDataset,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict[str, float]:
    """Re-infer all pseudo-label samples with the EMA model and update heatmaps."""
    ema_model.eval()
    pseudo_count = train_dataset.pseudo_count
    if pseudo_count == 0:
        return {}

    pseudo_indices = list(range(train_dataset._real_len, len(train_dataset)))
    pseudo_subset = Subset(train_dataset, pseudo_indices)
    loader = DataLoader(
        pseudo_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_mixed_batch,
        pin_memory=torch.cuda.is_available(),
    )

    overrides: dict[int, torch.Tensor] = {}
    total_change = 0.0
    sample_count = 0

    for batch in loader:
        logits = ema_model(
            batch["features"].to(device),
            batch["xyz"].to(device),
            batch["aff_hidden"].to(device),
            batch["visual_tokens"].to(device),
            proj_coords=batch["proj_coords"].to(device),
            proj_visible=batch["proj_visible"].to(device),
        )["logits"]
        new_heatmaps = torch.sigmoid(logits).cpu()
        old_heatmaps = batch["heatmap"]

        for i in range(new_heatmaps.shape[0]):
            n_valid = batch["point_mask"][i].sum().int().item()
            new_h = new_heatmaps[i, :n_valid].clone()
            old_h = old_heatmaps[i, :n_valid]
            total_change += (new_h - old_h).abs().mean().item()
            overrides[sample_count] = new_h
            sample_count += 1

    train_dataset.update_pseudo_heatmaps(overrides)

    return {
        "refreshed_samples": sample_count,
        "mean_heatmap_change": total_change / max(sample_count, 1),
    }


def main() -> None:
    args = parse_args()
    state = init_distributed(args.device, args.backend)
    try:
        set_seed(args.seed + state.rank)
        args.cache_root = args.cache_root.resolve()

        args.train_manifest = args.train_manifest or args.cache_root / "train_unseen_instruction_manifest.json"
        args.val_manifest = args.val_manifest or args.cache_root / "val_unseen_object_seen_instruction_manifest.json"
        args.test_manifest = args.test_manifest or args.cache_root / "test_unseen_object_seen_instruction_manifest.json"
        for attr in [
            "formal_val_unseen_object_manifest", "formal_val_unseen_category_manifest",
            "formal_test_unseen_object_manifest", "formal_test_unseen_category_manifest",
            "formal_train_seen_manifest", "formal_train_unseen_manifest",
        ]:
            if getattr(args, attr) is None:
                suffix_map = {
                    "formal_val_unseen_object_manifest": "val_unseen_object_seen_instruction_manifest.json",
                    "formal_val_unseen_category_manifest": "val_unseen_category_seen_instruction_manifest.json",
                    "formal_test_unseen_object_manifest": "test_unseen_object_seen_instruction_manifest.json",
                    "formal_test_unseen_category_manifest": "test_unseen_category_seen_instruction_manifest.json",
                    "formal_train_seen_manifest": "train_seen_instruction_manifest.json",
                    "formal_train_unseen_manifest": "train_unseen_instruction_manifest.json",
                }
                setattr(args, attr, args.cache_root / suffix_map[attr])

        if state.is_rank0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        barrier(state)

        # --- Build mixed training dataset ---
        train_dataset = MixedAffordanceDataset(
            real_manifest=args.train_manifest,
            pseudo_manifests=args.pseudo_manifest,
            pseudo_weight=args.pseudo_weight,
            pseudo_high_thresh=args.pseudo_high_thresh,
            pseudo_low_thresh=args.pseudo_low_thresh,
            pseudo_sample_ratio=args.pseudo_sample_ratio,
        )
        train_sampler = build_train_sampler(train_dataset, state, shuffle=True) if state.distributed else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_mixed_batch,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = None
        test_loader = None
        if state.is_rank0:
            val_loader = make_eval_loader(args.val_manifest, args.batch_size, args.num_workers)
            if args.test_manifest and args.test_manifest.exists():
                test_loader = make_eval_loader(args.test_manifest, args.batch_size, args.num_workers)

        real_count = train_dataset._real_len
        pseudo_count = train_dataset._pseudo_len
        rank0_print(
            state,
            f"[selftraining] train: {len(train_dataset)} total "
            f"(real={real_count}, pseudo={pseudo_count}, weight={args.pseudo_weight}) "
            f"soft_labels={args.use_soft_labels} ddp={state.distributed} "
            f"world_size={state.world_size}",
        )

        # --- Build model ---
        model = AffordanceVLMBackboneDecoderV5(
            point_feature_size=13,
            vlm_hidden_size=args.vlm_hidden_size,
            hidden_size=args.hidden_size,
            point_depth=args.point_depth,
            decoder_depth=args.decoder_depth,
            decoder_heads=args.decoder_heads,
            dropout=args.dropout,
            instruction_dropout_rate=args.instruction_dropout_rate,
            num_prototypes=args.num_prototypes,
            bottleneck_layers=args.bottleneck_layers,
            num_views=args.num_views,
            grid_size=args.grid_size,
        ).to(state.device)

        if args.init_checkpoint and args.init_checkpoint.exists():
            ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"], strict=True)
            rank0_print(state, f"[selftraining] initialized from {args.init_checkpoint}")
            del ckpt

        model = maybe_wrap_model(model, state)
        total_params = sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad)
        rank0_print(state, f"[selftraining] trainable params: {total_params:,}")

        # --- EMA ---
        ema: EMAModel | None = None
        if args.ema_decay > 0:
            ema = EMAModel(unwrap_model(model), decay=args.ema_decay)
            rank0_print(state, f"[selftraining] EMA enabled, decay={args.ema_decay}")

        # --- Optimizer + scheduler ---
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        )

        def warmup_cosine_lr(epoch: int) -> float:
            if epoch < args.warmup_epochs:
                return (epoch + 1) / max(args.warmup_epochs, 1)
            progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
            return args.min_lr / args.learning_rate + (1.0 - args.min_lr / args.learning_rate) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lr)
        use_amp = torch.cuda.is_available() and state.device.type == "cuda" and not args.disable_amp
        scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
        autocast_dtype = torch.float16 if state.device.type == "cuda" else torch.bfloat16

        history: list[dict[str, Any]] = []
        best_value = float("inf") if args.selection_mode == "min" else float("-inf")
        best_epoch: int | None = None
        best_checkpoint = args.output_dir / "best_selftraining.pt"
        latest_checkpoint = args.output_dir / "latest_selftraining.pt"
        start_epoch = 1

        if args.resume and latest_checkpoint.exists():
            ckpt = torch.load(latest_checkpoint, map_location="cpu", weights_only=False)
            unwrap_model(model).load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            for _ in range(ckpt["epoch"]):
                scheduler.step()
            if best_checkpoint.exists():
                best_ckpt = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
                best_epoch = best_ckpt["epoch"]
                best_value = float(best_ckpt.get("val_metrics", {}).get(
                    args.selection_metric.replace("val_", ""), best_value,
                ))
            history_path = args.output_dir / "training_history.json"
            if history_path.exists():
                history = json.load(open(history_path))["history"]
            rank0_print(state, f"[selftraining] resumed from epoch {ckpt['epoch']}")
            del ckpt

        # --- Training loop ---
        for epoch in range(start_epoch, args.epochs + 1):
            start = perf_counter()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            metric_sums: dict[str, float] = {}
            metric_count = 0

            # Curriculum: dynamic pseudo_weight based on epoch
            if epoch < args.curriculum_start:
                epoch_pseudo_weight = 0.0
            elif args.curriculum_rampup > 0 and epoch < args.curriculum_start + args.curriculum_rampup:
                progress = (epoch - args.curriculum_start) / args.curriculum_rampup
                epoch_pseudo_weight = args.pseudo_weight * progress
            else:
                epoch_pseudo_weight = args.pseudo_weight

            for batch_index, batch in enumerate(train_loader, start=1):
                # Apply curriculum weight: scale pseudo confidences
                if epoch_pseudo_weight != args.pseudo_weight and "confidences" in batch:
                    is_pseudo = batch.get("is_pseudo", [])
                    scale = epoch_pseudo_weight / max(args.pseudo_weight, 1e-8)
                    confs = batch["confidences"].clone()
                    for i, p in enumerate(is_pseudo):
                        if p:
                            confs[i] = confs[i] * scale
                    batch["confidences"] = confs

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=state.device.type, dtype=autocast_dtype, enabled=use_amp,
                ):
                    loss, loss_metrics = compute_loss(
                        model, batch, device=state.device,
                        positive_threshold=args.positive_threshold,
                        pos_weight=args.pos_weight,
                        lambda_bce=args.lambda_bce,
                        lambda_focal=args.lambda_focal,
                        lambda_dice=args.lambda_dice,
                        focal_alpha=args.focal_alpha,
                        focal_gamma=args.focal_gamma,
                        use_soft_labels=args.use_soft_labels,
                        confidence=batch.get("confidences"),
                        pseudo_bce_scale=args.pseudo_bce_scale,
                        pseudo_focal_scale=args.pseudo_focal_scale,
                        label_smoothing=args.label_smoothing,
                    )
                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad(set_to_none=True)
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    if ema is not None:
                        ema.update(unwrap_model(model))
                add_metric_sums(metric_sums, loss_metrics)
                metric_count += 1

                if state.is_rank0 and (
                    batch_index == 1
                    or batch_index % max(args.log_every_batches, 1) == 0
                    or batch_index == len(train_loader)
                ):
                    n_pseudo = sum(1 for p in batch["is_pseudo"] if p)
                    print(
                        f"[selftraining] epoch {epoch}/{args.epochs} "
                        f"batch {batch_index}/{len(train_loader)} "
                        f"loss={loss_metrics['loss']:.4f} "
                        f"pseudo={n_pseudo}/{len(batch['is_pseudo'])} "
                        f"pw={epoch_pseudo_weight:.3f}",
                        flush=True,
                    )

            train_losses = reduce_metric_sums(metric_sums, metric_count, state)
            scheduler.step()
            should_eval = epoch == 1 or epoch == args.epochs or epoch % max(args.eval_every, 1) == 0

            if state.is_rank0:
                val_metrics: dict[str, float] = {}
                eval_model = ema.module if ema is not None else unwrap_model(model)
                if should_eval and val_loader is not None:
                    val_metrics = evaluate_model(
                        eval_model, val_loader,
                        device=state.device, threshold=args.threshold,
                        positive_threshold=args.positive_threshold,
                    )

                record = {
                    "epoch": epoch,
                    "seconds": perf_counter() - start,
                    "lr": optimizer.param_groups[0]["lr"],
                    **{f"train_loss_{k}": v for k, v in train_losses.items()},
                    **({f"val_{k}": v for k, v in val_metrics.items()} if should_eval else {}),
                }
                history.append(record)

                checkpoint = {
                    "model": unwrap_model(model).state_dict(),
                    "ema_model": ema.state_dict() if ema is not None else None,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "args": serialize_args(args),
                    "train_losses": train_losses,
                    "val_metrics": val_metrics,
                }
                torch.save(checkpoint, latest_checkpoint)

                if should_eval and val_metrics:
                    current_value = float(record.get(args.selection_metric, float("nan")))
                    print(
                        f"[selftraining] epoch {epoch}/{args.epochs} "
                        f"val_iou={val_metrics.get('iou', 0):.4f} "
                        f"selected={args.selection_metric}:{current_value:.4f}",
                        flush=True,
                    )
                    if current_is_better(current_value, best_value, args.selection_mode):
                        best_value = current_value
                        best_epoch = epoch
                        torch.save(checkpoint, best_checkpoint)
                        print(f"[selftraining] new best epoch={epoch} value={best_value:.6f}", flush=True)
                else:
                    print(
                        f"[selftraining] epoch {epoch}/{args.epochs} "
                        f"loss={train_losses.get('loss', float('nan')):.4f}",
                        flush=True,
                    )

                save_json(args.output_dir / "training_history.json", {"history": history})

            # --- Pseudo-label refresh ---
            should_refresh = (
                args.refresh_every > 0
                and ema is not None
                and epoch > 0
                and epoch % args.refresh_every == 0
                and epoch < args.epochs
            )
            if should_refresh:
                refresh_stats = refresh_pseudo_labels(
                    ema.module, train_dataset, state.device,
                    batch_size=args.refresh_batch_size,
                    num_workers=args.num_workers,
                )
                rank0_print(
                    state,
                    f"[selftraining] pseudo-label refresh at epoch {epoch}: "
                    f"samples={refresh_stats.get('refreshed_samples', 0)}, "
                    f"mean_change={refresh_stats.get('mean_heatmap_change', 0):.4f}",
                )

            barrier(state)

        # --- Final evaluation ---
        if state.is_rank0:
            test_metrics = None
            if best_checkpoint.exists() and test_loader is not None:
                best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
                # Use EMA weights for final eval if available
                ema_sd = best_payload.get("ema_model")
                if ema_sd is not None:
                    unwrap_model(model).load_state_dict(ema_sd)
                else:
                    unwrap_model(model).load_state_dict(best_payload["model"])
                final_eval_model = unwrap_model(model)
                test_metrics = evaluate_model(
                    final_eval_model, test_loader,
                    device=state.device, threshold=args.threshold,
                    positive_threshold=args.positive_threshold,
                )

            generalization_metrics = {}
            for name, attr in [
                ("test_unseen_object", "formal_test_unseen_object_manifest"),
                ("test_unseen_category", "formal_test_unseen_category_manifest"),
            ]:
                m = getattr(args, attr)
                if m and m.exists():
                    generalization_metrics[name] = evaluate_manifest(
                        final_eval_model, m,
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        device=state.device, threshold=args.threshold,
                        positive_threshold=args.positive_threshold,
                    )

            summary = {
                "args": serialize_args(args),
                "best": {"epoch": best_epoch, "value": best_value},
                "test_metrics": test_metrics,
                "generalization_metrics": generalization_metrics,
            }
            save_json(args.output_dir / "training_summary.json", summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
