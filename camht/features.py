from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Time2Vec(nn.Module):
    """Learnable time embedding with linear + periodic components / 可学习时间嵌入模块

    This implementation follows the Time2Vec formulation (Kazemi et al., 2019)
    and adds performance-oriented engineering for this project.

    Args:
        in_features: number of input time features.
        out_features: total embedding size (>= 1). One slot is reserved for the
            linear component by default; the remaining slots become periodic.
        periodic_activation: name of the periodic activation function applied to
            the periodic component. Currently supports {"sin", "cos"}.
        include_linear: whether to keep the linear trend component.
        freq_init: strategy for initializing the periodic frequencies. Options:
            - "random": default PyTorch init.
            - "harmonic": deterministic 2π/period spacing based on
              `freq_range` (requires in_features == 1).
        freq_range: expected period range `(min_period, max_period)` when using
            harmonic initialization. The values are expressed in input time
            units. Example: (4.0, 512.0) for intra-day to quarterly cycles.

    性能注意事项:
        - 所有算子均为张量级操作，适配 torch.compile / Inductor。
        - 无显式 Python 循环；梯度计算与混合精度训练安全。
        - 初始化时可选学习到的频率覆盖范围，便于跨尺度金融时间序列。
    """

    _ACT_MAP = {
        "sin": torch.sin,
        "cos": torch.cos,
    }

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        periodic_activation: str = "sin",
        include_linear: bool = True,
        freq_init: str = "random",
        freq_range: Tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        if out_features < 1:
            raise ValueError("`out_features` must be >= 1 for Time2Vec")
        if periodic_activation not in self._ACT_MAP:
            raise ValueError(f"Unsupported periodic activation: {periodic_activation}")
        if freq_init not in {"random", "harmonic"}:
            raise ValueError(f"Unsupported freq_init: {freq_init}")

        self.in_features = in_features
        self.out_features = out_features
        self.include_linear = include_linear
        self.freq_init = freq_init
        parsed_range: Tuple[float, float] | None = None
        if freq_range is not None:
            if not isinstance(freq_range, (tuple, list)):
                raise TypeError("`freq_range` must be a tuple/list of (min_period, max_period)")
            if len(freq_range) != 2:
                raise ValueError("`freq_range` must contain exactly two elements")
            parsed_range = (float(freq_range[0]), float(freq_range[1]))
        self.freq_range = parsed_range
        self._periodic_activation = self._ACT_MAP[periodic_activation]

        linear_dim = 1 if include_linear else 0
        if include_linear and out_features < 1:
            raise ValueError("`out_features` must be >= 1 when include_linear is True")

        periodic_dim = out_features - linear_dim
        if periodic_dim < 0:
            raise ValueError("`out_features` is too small for the selected configuration")
        if freq_init == "harmonic" and periodic_dim == 0:
            raise ValueError("Harmonic initialization requires at least one periodic unit")
        if freq_init == "harmonic" and in_features != 1:
            raise ValueError("Harmonic initialization assumes scalar time input (in_features == 1)")
        if freq_init == "harmonic" and (
            self.freq_range is None
            or self.freq_range[0] <= 0
            or self.freq_range[1] <= 0
        ):
            raise ValueError("Harmonic initialization needs positive freq_range (min_period, max_period)")

        self.linear = nn.Linear(in_features, linear_dim, bias=True) if linear_dim else None
        self.periodic = nn.Linear(in_features, periodic_dim, bias=True) if periodic_dim else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Use PyTorch defaults for stability; override periodic weights if harmonic init is requested.
        if self.linear is not None:
            self.linear.reset_parameters()
        if self.periodic is not None:
            self.periodic.reset_parameters()
            if self.freq_init == "harmonic":
                min_period, max_period = self.freq_range  # type: ignore[type-var]
                # 生成等比（对数）间隔的周期 / Log-spaced periods to cover multiple seasonalities
                periods = torch.logspace(
                    start=float(torch.log10(torch.tensor(min_period, device=self.periodic.weight.device))),
                    end=float(torch.log10(torch.tensor(max_period, device=self.periodic.weight.device))),
                    steps=self.periodic.out_features,
                    base=10.0,
                    dtype=self.periodic.weight.dtype,
                    device=self.periodic.weight.device,
                )
                # ω = 2π / period
                omegas = 2.0 * torch.pi / periods
                with torch.no_grad():
                    self.periodic.weight.copy_(omegas.unsqueeze(1))
                    self.periodic.bias.zero_()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute time embeddings / 计算时间嵌入.

        Args:
            t: time tensor shaped [..., in_features]. Supports batched sequences such
               as [B, T, C] or [B, T, 1]. 输入张量最后一维必须匹配 in_features。

        Returns:
            Tensor with shape [..., out_features], concatenating linear + periodic
            components (if enabled). 输出张量保持输入批次维度不变。
        """

        if t.size(-1) != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {t.size(-1)}"
            )

        outputs = []
        if self.linear is not None:
            outputs.append(self.linear(t))
        if self.periodic is not None:
            periodic_term = self._periodic_activation(self.periodic(t))
            outputs.append(periodic_term)
        if not outputs:
            raise RuntimeError("Time2Vec has neither linear nor periodic components enabled")
        return torch.cat(outputs, dim=-1)


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
