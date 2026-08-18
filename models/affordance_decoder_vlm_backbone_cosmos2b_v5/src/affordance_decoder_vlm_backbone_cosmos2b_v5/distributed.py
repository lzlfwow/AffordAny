from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


@dataclass
class DistributedState:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_rank0(self) -> bool:
        return (not self.distributed) or self.rank == 0


def init_distributed(device_arg: str, backend: str | None = None) -> DistributedState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = torch.device(device_arg)
        return DistributedState(False, 0, 1, 0, device)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend is None:
        backend = "nccl" if device_arg.startswith("cuda") and torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if torch.cuda.is_available() and device_arg.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(device_arg)
    return DistributedState(True, rank, world_size, local_rank, device)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(state: DistributedState) -> None:
    if state.distributed:
        dist.barrier()


def rank0_print(state: DistributedState, *args: Any, **kwargs: Any) -> None:
    if state.is_rank0:
        print(*args, **kwargs)


def build_train_sampler(dataset: Dataset, state: DistributedState, *, shuffle: bool = True) -> DistributedSampler | None:
    if not state.distributed:
        return None
    return DistributedSampler(dataset, shuffle=shuffle)


def maybe_wrap_model(model: nn.Module, state: DistributedState) -> nn.Module:
    if not state.distributed:
        return model
    if state.device.type == "cuda":
        return DistributedDataParallel(model, device_ids=[state.local_rank], output_device=state.local_rank)
    return DistributedDataParallel(model)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def reduce_metric_sums(metric_sums: dict[str, float], count: int, state: DistributedState) -> dict[str, float]:
    keys = sorted(metric_sums)
    values = [float(metric_sums[key]) for key in keys] + [float(count)]
    tensor = torch.tensor(values, dtype=torch.float64, device=state.device)
    if state.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total_count = int(tensor[-1].item())
    if total_count <= 0:
        return {key: float("nan") for key in keys}
    return {key: float(tensor[index].item() / total_count) for index, key in enumerate(keys)}


def add_metric_sums(metric_sums: dict[str, float], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        value = float(value)
        if value == value:
            metric_sums[key] = metric_sums.get(key, 0.0) + value

