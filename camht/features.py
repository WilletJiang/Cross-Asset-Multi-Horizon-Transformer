from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            pad = P - T
            # 仅在时间维度左侧做零填充，保持首端信息不丢失
            x = F.pad(x, (0, 0, pad, 0))
            T = x.shape[1]
        n = 1 + (T - P) // S
        # `unfold` 在 C++ 内部完成窗口抽取，比 Python for 循环快几个数量级
        patches = x.unfold(dimension=1, size=P, step=S).permute(0, 1, 3, 2).contiguous()
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
