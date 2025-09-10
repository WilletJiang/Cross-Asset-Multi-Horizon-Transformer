from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """Time2Vec module.

    Paper: https://arxiv.org/abs/1907.05321
    Implementation: simple linear + periodic components.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.periodic = nn.Linear(in_features, out_features - 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B, T, 1] or [B, T, C]
        return torch.cat([self.linear(t), torch.sin(self.periodic(t))], dim=-1)


@dataclass
class PatchSpec:
    size: int
    stride: int


class Patchify(nn.Module):
    """Patchify along time dimension.

    Input:  [B, T, C]
    Output: [B, N, P, C] where P=size, N=number of patches
    """

    def __init__(self, patch_size: int, stride: int):
        super().__init__()
        self.spec = PatchSpec(patch_size, stride)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, T, C = x.shape
        P, S = self.spec.size, self.spec.stride
        if T < P:
            # left-pad with zeros to reach at least one patch
            pad = P - T
            x = torch.nn.functional.pad(x, (0, 0, pad, 0))
            T = x.shape[1]
        n = 1 + (T - P) // S
        idx = torch.arange(0, n * S, S, device=x.device)
        patches = torch.stack([x[:, i : i + P, :] for i in idx], dim=1)
        return patches, n


class FlattenPatches(nn.Module):
    """Flatten [B, N, P, C] -> [B, N, P*C] with linear projection to d_model.
    """

    def __init__(self, in_channels: int, patch_size: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(in_channels * patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, P, C = x.shape
        x = x.reshape(B, N, P * C)
        return self.proj(x)

