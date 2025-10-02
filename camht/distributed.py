"""Distributed training utilities for extreme performance.

Multi-GPU setup with DDP/FSDP, communication optimization, and zero-overhead distributed primitives.
极致性能的分布式训练工具：DDP/FSDP、通信优化、零开销分布式原语
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False

from rich.console import Console

console = Console()


@dataclass
class DistConfig:
    """Distributed training configuration."""
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    init_method: str = "env://"  # environment variable based init
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    use_fsdp: bool = False
    fsdp_sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    gradient_as_bucket_view: bool = True  # reduce memory copies in DDP
    broadcast_buffers: bool = False  # reduce communication
    find_unused_parameters: bool = False  # strict graph requirement
    static_graph: bool = True  # enable static graph optimization in DDP


def is_dist_available_and_initialized() -> bool:
    """Check if torch.distributed is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get global rank. Returns 0 if not distributed."""
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """Get local rank (GPU index on this node). Returns 0 if not distributed."""
    if not is_dist_available_and_initialized():
        return 0
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """Get total number of processes. Returns 1 if not distributed."""
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes. No-op if not distributed."""
    if is_dist_available_and_initialized():
        dist.barrier()


def setup_distributed(config: Optional[DistConfig] = None) -> DistConfig:
    """Initialize distributed training environment.

    Reads from environment variables set by torch.distributed.launch or torchrun:
    - RANK: global rank
    - LOCAL_RANK: local rank (GPU index on this node)
    - WORLD_SIZE: total number of processes
    - MASTER_ADDR: master node address
    - MASTER_PORT: master node port
    """
    if config is None:
        config = DistConfig()

    # Read from environment if available
    config.rank = int(os.environ.get("RANK", 0))
    config.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    config.world_size = int(os.environ.get("WORLD_SIZE", 1))

    if config.world_size > 1:
        if not dist.is_available():
            raise RuntimeError("torch.distributed not available")

        # Set device before init_process_group to avoid NCCL issues
        torch.cuda.set_device(config.local_rank)

        # Initialize process group
        dist.init_process_group(
            backend=config.backend,
            init_method=config.init_method,
            world_size=config.world_size,
            rank=config.rank,
        )

        if is_main_process():
            console.print(f"[green]Initialized distributed training: world_size={config.world_size}, backend={config.backend}")
    else:
        if is_main_process():
            console.print("[yellow]Running in single-process mode (no distributed training)")

    return config


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


def get_device(local_rank: Optional[int] = None) -> torch.device:
    """Get the appropriate device for this process."""
    if local_rank is None:
        local_rank = get_local_rank()

    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def wrap_model_ddp(
    model: torch.nn.Module,
    device_ids: Optional[list[int]] = None,
    gradient_as_bucket_view: bool = True,
    broadcast_buffers: bool = False,
    find_unused_parameters: bool = False,
    static_graph: bool = True,
    bucket_cap_mb: int = 25,
) -> DDP:
    """Wrap model with DistributedDataParallel.

    Performance optimizations:
    - gradient_as_bucket_view=True: avoid extra memory copy
    - broadcast_buffers=False: reduce communication if buffers don't change
    - static_graph=True: enable graph-based optimizations
    - bucket_cap_mb=25: balance communication granularity
    """
    if not is_dist_available_and_initialized():
        raise RuntimeError("Distributed not initialized. Call setup_distributed() first.")

    local_rank = get_local_rank()
    if device_ids is None:
        device_ids = [local_rank]

    ddp_model = DDP(
        model,
        device_ids=device_ids,
        output_device=local_rank,
        gradient_as_bucket_view=gradient_as_bucket_view,
        broadcast_buffers=broadcast_buffers,
        find_unused_parameters=find_unused_parameters,
        static_graph=static_graph,
        bucket_cap_mb=bucket_cap_mb,
    )

    if is_main_process():
        console.print("[green]Wrapped model with DDP")

    return ddp_model


def wrap_model_fsdp(
    model: torch.nn.Module,
    sharding_strategy: str = "FULL_SHARD",
    mixed_precision_dtype: Optional[torch.dtype] = None,
    transformer_layer_cls: Optional[type] = None,
    cpu_offload: bool = False,
) -> FSDP:
    """Wrap model with FullyShardedDataParallel.

    FSDP shards model parameters, gradients, and optimizer states across GPUs,
    enabling training of models that don't fit on a single GPU.

    Args:
        model: Model to wrap
        sharding_strategy: FULL_SHARD, SHARD_GRAD_OP, or NO_SHARD
        mixed_precision_dtype: dtype for mixed precision (bf16 recommended)
        transformer_layer_cls: transformer layer class for auto wrapping
        cpu_offload: offload parameters to CPU (slow but saves GPU memory)
    """
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP not available in this PyTorch version")

    if not is_dist_available_and_initialized():
        raise RuntimeError("Distributed not initialized. Call setup_distributed() first.")

    # Map string to ShardingStrategy enum
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    strategy = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)

    # Mixed precision configuration
    mp_policy = None
    if mixed_precision_dtype is not None:
        mp_policy = MixedPrecision(
            param_dtype=mixed_precision_dtype,
            reduce_dtype=mixed_precision_dtype,
            buffer_dtype=mixed_precision_dtype,
        )

    # Auto wrap policy for transformer models
    auto_wrap_policy = None
    if transformer_layer_cls is not None:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_layer_cls},
        )

    # CPU offload
    cpu_offload_config = None
    if cpu_offload:
        from torch.distributed.fsdp import CPUOffload
        cpu_offload_config = CPUOffload(offload_params=True)

    fsdp_model = FSDP(
        model,
        sharding_strategy=strategy,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=cpu_offload_config,
        device_id=get_local_rank(),
    )

    if is_main_process():
        console.print(f"[green]Wrapped model with FSDP (strategy={sharding_strategy})")

    return fsdp_model


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce tensor across all processes and compute mean."""
    if not is_dist_available_and_initialized():
        return tensor

    world_size = get_world_size()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / world_size
    return tensor


def all_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensor from all processes."""
    if not is_dist_available_and_initialized():
        return tensor

    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)


class GradAccumulationContext:
    """Context manager for gradient accumulation with DDP.

    Disables gradient synchronization during accumulation steps,
    only syncing on the final step. This reduces communication overhead.
    """
    def __init__(self, model: torch.nn.Module, sync: bool = True):
        self.model = model
        self.sync = sync
        self.is_ddp = isinstance(model, DDP)

    def __enter__(self):
        if self.is_ddp and not self.sync:
            # Disable gradient synchronization
            self.model.require_backward_grad_sync = False
        return self

    def __exit__(self, *args):
        if self.is_ddp and not self.sync:
            # Re-enable gradient synchronization
            self.model.require_backward_grad_sync = True


def print_once(msg: str, *args, **kwargs) -> None:
    """Print message only on main process."""
    if is_main_process():
        console.print(msg, *args, **kwargs)


def reduce_dict(input_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reduce dictionary of tensors across all processes."""
    if not is_dist_available_and_initialized():
        return input_dict

    world_size = get_world_size()
    if world_size == 1:
        return input_dict

    names = sorted(input_dict.keys())
    values = torch.stack([input_dict[k] for k in names])
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values / world_size

    reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def save_on_main(state: dict[str, Any], path: str) -> None:
    """Save checkpoint only on main process."""
    if is_main_process():
        torch.save(state, path)


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast Python object from src rank to all ranks."""
    if not is_dist_available_and_initialized():
        return obj

    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]

