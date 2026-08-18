"""Train Cosmos-2B VLM-backbone affordance decoder v4 (source image + P1 + condP3 + GPBlock).

Key differences from v1 train.py:
  - Uses VLMBackboneDatasetV5 with source-image VLM features and pre-computed projection.
  - Uses AffordanceVLMBackboneDecoderV5 with P1 injector + P3 bottleneck.
  - New model hyperparameters: num-prototypes, bottleneck-layers, num-views, grid-size.
  - Augmentation hyperparameters: scale-range, jitter-std, disable-rotation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
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
from affordance_decoder_vlm_backbone_cosmos2b_v5.losses import compute_loss  # noqa: E402
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
    default_cache = (
        REPO_ROOT / "models" / "affordance_decoder_vlm_backbone_cosmos2b_v5" / "artifacts" / "vlm_cache"
    )
    default_output = (
        REPO_ROOT / "models" / "affordance_decoder_vlm_backbone_cosmos2b_v5" / "artifacts" / "experiments"
        / "vlm_backbone_cosmos2b_v5_decoder"
    )
    parser = argparse.ArgumentParser(
        description="Train Cosmos-2B VLM-backbone affordance decoder v4 (source image, P1+P3).",
    )
    parser.add_argument("--cache-root", type=Path, default=default_cache)
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
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


    parser.add_argument("--disable-p1", action="store_true", help="Ablation: disable P1 ProjectionFeatureInjector")
    parser.add_argument("--disable-p3", action="store_true", help="Ablation: disable P3 SemanticBottleneck")
    parser.add_argument("--disable-text-cond", action="store_true", help="Ablation: disable text-conditioned P3")
    parser.add_argument("--disable-gpblock", action="store_true", help="Ablation: disable GPBlock bidirectional fusion")

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

    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output-dir")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--log-every-batches", type=int, default=30)
    parser.add_argument(
        "--eval-every", type=int, default=10,
        help="Evaluate every N epochs plus epoch 1 and final epoch.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def make_loader(
    manifest: Path,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    *,
    state: DistributedState | None = None,
    distributed_train: bool = False,
) -> tuple[DataLoader, torch.utils.data.distributed.DistributedSampler | None]:
    dataset = VLMBackboneDatasetV5(manifest)
    sampler = None
    if distributed_train:
        assert state is not None
        sampler = build_train_sampler(dataset, state, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_vlm_backbone_v5_batch,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, sampler


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    threshold: float,
    positive_threshold: float,
) -> dict[str, float]:
    model.eval()
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
    return compute_basic_metrics(
        prob, target, threshold=threshold, positive_threshold=positive_threshold,
        categories=category_list,
    )


def current_is_better(value: float, best_value: float, mode: str) -> bool:
    if value != value:
        return False
    return value <= best_value if mode == "min" else value >= best_value


def initial_best_value(mode: str) -> float:
    return float("inf") if mode == "min" else float("-inf")


@torch.no_grad()
def evaluate_manifest(
    model: nn.Module,
    manifest: Path,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    threshold: float,
    positive_threshold: float,
) -> dict[str, float]:
    loader, _ = make_loader(
        manifest, batch_size, num_workers, False,
    )
    metrics = evaluate_model(
        model, loader, device=device,
        threshold=threshold, positive_threshold=positive_threshold,
    )
    metrics["samples"] = len(loader.dataset)
    return metrics


def metric_deltas(
    seen: dict[str, float], unseen: dict[str, float],
) -> dict[str, float]:
    delta_keys = [
        "iou", "miou", "auc", "mae", "sim", "pred_pos", "truth_pos",
        "prob_mean", "prob_max", "valid_ratio",
    ]
    return {
        f"delta_{k}": float(seen[k]) - float(unseen[k])
        for k in delta_keys if k in seen and k in unseen
    }


def evaluate_formal_protocol(
    model: nn.Module, args: argparse.Namespace, device: torch.device,
) -> tuple[dict[str, dict[str, float]], dict[str, Any], dict[str, dict[str, float]]]:
    generalization_specs = {
        "test_unseen_object_seen_instruction": args.formal_test_unseen_object_manifest,
        "test_unseen_category_seen_instruction": args.formal_test_unseen_category_manifest,
    }
    validation_specs = {
        "val_unseen_object_seen_instruction": args.formal_val_unseen_object_manifest,
        "val_unseen_category_seen_instruction": args.formal_val_unseen_category_manifest,
    }
    instruction_specs = {
        "train_seen_instruction": args.formal_train_seen_manifest,
        "train_unseen_instruction": args.formal_train_unseen_manifest,
    }
    eval_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.num_workers,
        device=device, threshold=args.threshold,
        positive_threshold=args.positive_threshold,
    )
    generalization_metrics = {
        name: evaluate_manifest(model, m, **eval_kwargs)
        for name, m in generalization_specs.items()
        if m is not None and m.exists()
    }
    validation_metrics = {
        name: evaluate_manifest(model, m, **eval_kwargs)
        for name, m in validation_specs.items()
        if m is not None and m.exists()
    }
    instruction_metrics: dict[str, Any] = {
        name: evaluate_manifest(model, m, **eval_kwargs)
        for name, m in instruction_specs.items()
        if m is not None and m.exists()
    }
    if "train_seen_instruction" in instruction_metrics and "train_unseen_instruction" in instruction_metrics:
        instruction_metrics["deltas"] = metric_deltas(
            instruction_metrics["train_seen_instruction"],
            instruction_metrics["train_unseen_instruction"],
        )
    return generalization_metrics, instruction_metrics, validation_metrics


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    best = payload.get("best", {})
    test_metrics = payload.get("test_metrics")
    generalization_metrics = payload.get("generalization_metrics") or {}
    instruction_metrics = payload.get("instruction_metrics") or {}
    validation_metrics = payload.get("validation_metrics") or {}
    args = payload["args"]
    lines = [
        "# Cosmos-2B VLM Backbone Affordance Decoder V3 Report",
        "",
        "## Setting",
        f"- Cache root: `{args['cache_root']}`",
        f"- Epochs: `{args['epochs']}`",
        f"- Batch size per GPU: `{args['batch_size']}`",
        f"- VLM hidden size: `{args['vlm_hidden_size']}`",
        f"- Decoder hidden: `{args['hidden_size']}`",
        f"- Point depth: `{args['point_depth']}`",
        f"- Decoder depth: `{args['decoder_depth']}`",
        f"- Decoder heads: `{args['decoder_heads']}`",
        f"- Dropout: `{args['dropout']}`",
        f"- Instruction dropout: `{args['instruction_dropout_rate']}`",
        f"- Num prototypes: `{args['num_prototypes']}`",
        f"- Bottleneck layers: `{args['bottleneck_layers']}`",
        f"- Num views: `{args['num_views']}`",
        f"- Grid size: `{args['grid_size']}`",
        f"- Input: source image (masked, white outside)",
        "",
        "## Best",
        f"- Epoch: `{best.get('epoch')}`",
        f"- Metric: `{args['selection_metric']}`",
        f"- Value: `{best.get('value')}`",
    ]
    if test_metrics:
        lines.extend(["", "## Test Metrics", "| Metric | Value |", "|---|---:|"])
        for key in METRIC_KEYS:
            lines.append(f"| {key} | {float(test_metrics[key]):.6f} |")
    for title, group in [
        ("Generalization Metrics", generalization_metrics),
        ("Instruction Metrics", {k: v for k, v in instruction_metrics.items() if k != "deltas"}),
        ("Validation Generalization Metrics", validation_metrics),
    ]:
        if group:
            lines.extend([
                "", f"## {title}",
                "| Group | IoU | mIoU | AUC | MAE | SIM | Pred Pos | Truth Pos | Samples |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for name, metrics in group.items():
                lines.append(
                    f"| {name} | {float(metrics['iou']):.6f} | {float(metrics['miou']):.6f} | "
                    f"{float(metrics['auc']):.6f} | {float(metrics['mae']):.6f} | "
                    f"{float(metrics['sim']):.6f} | {float(metrics['pred_pos']):.6f} | "
                    f"{float(metrics['truth_pos']):.6f} | {int(metrics.get('samples', 0))} |"
                )
    if instruction_metrics.get("deltas"):
        lines.extend(["", "## Instruction Deltas", "| Metric | Delta |", "|---|---:|"])
        for key, value in instruction_metrics["deltas"].items():
            lines.append(f"| {key} | {float(value):.6f} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    state = init_distributed(args.device, args.backend)
    try:
        set_seed(args.seed + state.rank)
        args.cache_root = args.cache_root.resolve()

        args.train_manifest = args.train_manifest or args.cache_root / "train_unseen_instruction_manifest.json"
        args.val_manifest = args.val_manifest or args.cache_root / "val_unseen_object_seen_instruction_manifest.json"
        args.test_manifest = args.test_manifest or args.cache_root / "test_unseen_object_seen_instruction_manifest.json"
        args.formal_val_unseen_object_manifest = args.formal_val_unseen_object_manifest or args.cache_root / "val_unseen_object_seen_instruction_manifest.json"
        args.formal_val_unseen_category_manifest = args.formal_val_unseen_category_manifest or args.cache_root / "val_unseen_category_seen_instruction_manifest.json"
        args.formal_train_seen_manifest = args.formal_train_seen_manifest or args.cache_root / "train_seen_instruction_manifest.json"
        args.formal_train_unseen_manifest = args.formal_train_unseen_manifest or args.cache_root / "train_unseen_instruction_manifest.json"
        args.formal_test_unseen_object_manifest = args.formal_test_unseen_object_manifest or args.cache_root / "test_unseen_object_seen_instruction_manifest.json"
        args.formal_test_unseen_category_manifest = args.formal_test_unseen_category_manifest or args.cache_root / "test_unseen_category_seen_instruction_manifest.json"
        cache_config = validate_cache_config(
            args.cache_root,
            vlm_hidden_size=args.vlm_hidden_size,
            num_views=args.num_views,
            grid_size=args.grid_size,
        )

        if state.is_rank0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        barrier(state)

        train_loader, train_sampler = make_loader(
            args.train_manifest, args.batch_size, args.num_workers, True,
            state=state, distributed_train=state.distributed,
        )
        eval_train_loader = None
        val_loader = None
        test_loader = None
        if state.is_rank0:
            eval_train_loader, _ = make_loader(
                args.train_manifest, args.batch_size, args.num_workers, False,
            )
            val_loader, _ = make_loader(
                args.val_manifest, args.batch_size, args.num_workers, False,
            )
            if args.test_manifest.exists():
                test_loader, _ = make_loader(
                    args.test_manifest, args.batch_size, args.num_workers, False,
                )

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
            use_projection=not args.disable_p1,
            use_bottleneck=not args.disable_p3,
            use_gpblock=not args.disable_gpblock,
            use_text_cond=not args.disable_text_cond,
        ).to(state.device)
        model = maybe_wrap_model(model, state)

        total_params = sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad)
        if cache_config:
            rank0_print(
                state,
                "[cosmos2b-v3] cache: "
                f"model={cache_config.get('vlm_model')} "
                f"hidden={cache_config.get('hidden_size')} "
                f"visual_tokens={cache_config.get('total_visual_tokens')}",
            )
        rank0_print(state, f"[cosmos2b-v3] trainable params: {total_params:,}")
        rank0_print(
            state,
            f"[cosmos2b_v5] ablation: P1={'OFF' if args.disable_p1 else 'ON'} "
            f"P3={'OFF' if args.disable_p3 else 'ON'} "
            f"TextCond={'OFF' if args.disable_text_cond else 'ON'} "
            f"GPBlock={'OFF' if args.disable_gpblock else 'ON'}",
        )

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
        best_value = initial_best_value(args.selection_mode)
        best_epoch: int | None = None
        best_checkpoint = args.output_dir / "best_vlm_backbone_cosmos2b_v5_decoder.pt"
        latest_checkpoint = args.output_dir / "latest_vlm_backbone_cosmos2b_v5_decoder.pt"
        start_epoch = 1

        if args.resume and latest_checkpoint.exists():
            resume_ckpt = torch.load(latest_checkpoint, map_location="cpu", weights_only=False)
            unwrap_model(model).load_state_dict(resume_ckpt["model"])
            if "optimizer" in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
            if "scaler" in resume_ckpt:
                scaler.load_state_dict(resume_ckpt["scaler"])
            start_epoch = resume_ckpt["epoch"] + 1
            for _ in range(resume_ckpt["epoch"]):
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
            rank0_print(state, f"[cosmos2b-v3] resumed from epoch {resume_ckpt['epoch']}, starting epoch {start_epoch}")
            del resume_ckpt

        rank0_print(
            state,
            f"[cosmos2b-v3] train={len(train_loader.dataset)} "
            f"val={len(val_loader.dataset) if val_loader is not None else 0} "
            f"test={len(test_loader.dataset) if test_loader is not None else 0} "
            f"ddp={state.distributed} world_size={state.world_size} device={state.device}",
            flush=True,
        )

        for epoch in range(start_epoch, args.epochs + 1):
            start = perf_counter()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            metric_sums: dict[str, float] = {}
            metric_count = 0

            for batch_index, batch in enumerate(train_loader, start=1):
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
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                add_metric_sums(metric_sums, loss_metrics)
                metric_count += 1

                if state.is_rank0 and (
                    batch_index == 1
                    or batch_index % max(args.log_every_batches, 1) == 0
                    or batch_index == len(train_loader)
                ):
                    print(
                        f"[cosmos2b-v3] epoch {epoch}/{args.epochs} "
                        f"batch {batch_index}/{len(train_loader)} "
                        f"loss={loss_metrics['loss']:.4f} "
                        f"bce={loss_metrics['bce_loss']:.4f} "
                        f"focal={loss_metrics['focal_loss']:.4f} "
                        f"dice={loss_metrics['dice_loss']:.4f}",
                        flush=True,
                    )

            train_losses = reduce_metric_sums(metric_sums, metric_count, state)
            scheduler.step()
            should_eval = epoch == 1 or epoch == args.epochs or epoch % max(args.eval_every, 1) == 0

            if state.is_rank0:
                train_metrics: dict[str, float] = {}
                val_metrics: dict[str, float] = {}
                if should_eval:
                    assert eval_train_loader is not None
                    assert val_loader is not None
                    train_metrics = evaluate_model(
                        unwrap_model(model), eval_train_loader,
                        device=state.device, threshold=args.threshold,
                        positive_threshold=args.positive_threshold,
                    )
                    val_metrics = evaluate_model(
                        unwrap_model(model), val_loader,
                        device=state.device, threshold=args.threshold,
                        positive_threshold=args.positive_threshold,
                    )

                record = {
                    "epoch": epoch,
                    "seconds": perf_counter() - start,
                    "lr": optimizer.param_groups[0]["lr"],
                    **{f"train_loss_{k}": v for k, v in train_losses.items()},
                    **({f"train_{k}": v for k, v in train_metrics.items()} if should_eval else {}),
                    **({f"val_{k}": v for k, v in val_metrics.items()} if should_eval else {}),
                }
                history.append(record)

                checkpoint = {
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "args": serialize_args(args),
                    "train_losses": train_losses,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                }
                torch.save(checkpoint, latest_checkpoint)

                if should_eval:
                    current_value = float(record[args.selection_metric])
                    print(
                        f"[cosmos2b-v3] epoch {epoch}/{args.epochs} done "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} "
                        f"train_loss={train_losses.get('loss', float('nan')):.4f} "
                        f"train_iou={train_metrics['iou']:.4f} "
                        f"val_iou={val_metrics['iou']:.4f} "
                        f"selected={args.selection_metric}:{current_value:.4f}",
                        flush=True,
                    )
                    if current_is_better(current_value, best_value, args.selection_mode):
                        best_value = current_value
                        best_epoch = epoch
                        torch.save(checkpoint, best_checkpoint)
                        print(
                            f"[cosmos2b-v3] new best epoch={epoch} value={best_value:.6f}",
                            flush=True,
                        )
                else:
                    print(
                        f"[cosmos2b-v3] epoch {epoch}/{args.epochs} done "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} "
                        f"train_loss={train_losses.get('loss', float('nan')):.4f} eval_skipped",
                        flush=True,
                    )

                save_json(args.output_dir / "training_history.json", {"history": history})
            barrier(state)

        if state.is_rank0:
            test_metrics = None
            generalization_metrics: dict[str, dict[str, float]] = {}
            instruction_metrics: dict[str, Any] = {}
            validation_metrics: dict[str, dict[str, float]] = {}
            if best_checkpoint.exists() and test_loader is not None:
                best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
                unwrap_model(model).load_state_dict(best_payload["model"])
                test_metrics = evaluate_model(
                    unwrap_model(model), test_loader,
                    device=state.device, threshold=args.threshold,
                    positive_threshold=args.positive_threshold,
                )
                generalization_metrics, instruction_metrics, validation_metrics = evaluate_formal_protocol(
                    unwrap_model(model), args, state.device,
                )
                if generalization_metrics:
                    save_json(args.output_dir / "final_eval_generalization.json", generalization_metrics)
                if instruction_metrics:
                    save_json(args.output_dir / "final_eval_instruction.json", instruction_metrics)

            summary = {
                "args": serialize_args(args),
                "best": {"epoch": best_epoch, "value": best_value, "checkpoint": str(best_checkpoint)},
                "test_metrics": test_metrics,
                "generalization_metrics": generalization_metrics,
                "instruction_metrics": instruction_metrics,
                "validation_metrics": validation_metrics,
            }
            save_json(args.output_dir / "training_summary.json", summary)
            write_markdown(args.output_dir / "final_report.md", summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
