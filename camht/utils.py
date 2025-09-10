from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from rich.console import Console


console = Console()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class SDPPolicy:
    force_math: bool = False
    enable_flash: bool = True
    enable_mem_efficient: bool = True


def configure_sdp(policy: Optional[SDPPolicy] = None) -> None:
    """Configure scaled_dot_product_attention kernel preferences.

    Keep shapes static and use torch.compile for best performance.
    """
    if policy is None:
        policy = SDPPolicy()
    torch.backends.cuda.enable_flash_sdp(policy.enable_flash)
    torch.backends.cuda.enable_mem_efficient_sdp(policy.enable_mem_efficient)
    torch.backends.cuda.enable_math_sdp(policy.force_math)


def maybe_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if not enabled:
        return model
    try:
        model = torch.compile(model, mode="max-autotune")
    except Exception as err:  # noqa: BLE001
        console.print(f"[yellow]torch.compile failed: {err}. Continuing without compile.")
    return model


def env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "y"}

