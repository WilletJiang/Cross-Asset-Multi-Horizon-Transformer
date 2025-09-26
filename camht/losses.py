from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchsort  # type: ignore
except Exception:  # noqa: BLE001
    torchsort = None


class DiffSpearmanLoss(nn.Module):
    """Differentiable Spearman via torchsort (SoftRank).

    If torchsort is unavailable, fallback to negative Spearman computed with rankdata+pearson
    which is not strictly differentiable; warn user.
    """

    def __init__(self, temperature: float = 0.1, regularization: str = "none") -> None:
        super().__init__()
        self.temperature = temperature
        self.regularization = regularization

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute 1 - Spearman(preds, targets) across the last dimension.

        Args:
            preds: [B, A]
            targets: [B, A]
        """
        # Row-wise masking of NaNs in targets (and corresponding preds)
        # 使用布尔 mask 一次性筛掉缺失值，避免 Python 循环导致的图中断
        mask = ~torch.isnan(targets)
        mask_f = mask.to(preds.dtype)
        valid_counts = mask.sum(dim=-1, keepdim=True)
        valid_rows = valid_counts.squeeze(-1) >= 2
        if valid_rows.sum() == 0:
            return torch.zeros((), device=preds.device)

        fill_value = torch.finfo(preds.dtype).min
        fill = torch.full_like(preds, fill_value)
        p_masked = torch.where(mask, preds, fill)
        t_masked = torch.where(mask, targets, fill)

        if torchsort is not None:
            sr_p = torchsort.soft_rank(p_masked, regularization=self.regularization, temperature=self.temperature)
            sr_t = torchsort.soft_rank(t_masked, regularization=self.regularization, temperature=self.temperature)
        else:
            sr_p = torch.argsort(torch.argsort(p_masked, dim=-1), dim=-1).float()
            sr_t = torch.argsort(torch.argsort(t_masked, dim=-1), dim=-1).float()

        invalid_counts = (~mask).sum(dim=-1, keepdim=True).float()
        sr_p = sr_p - invalid_counts
        sr_t = sr_t - invalid_counts

        sr_p = torch.where(mask, sr_p, torch.zeros_like(sr_p)).float()
        sr_t = torch.where(mask, sr_t, torch.zeros_like(sr_t)).float()

        safe_counts = valid_counts.clamp_min(1).float()
        mean_p = (sr_p * mask_f).sum(dim=-1, keepdim=True) / safe_counts
        mean_t = (sr_t * mask_f).sum(dim=-1, keepdim=True) / safe_counts
        xp = (sr_p - mean_p) * mask_f
        xt = (sr_t - mean_t) * mask_f
        eps = 1e-8
        cov = (xp * xt).sum(dim=-1) / safe_counts.squeeze(-1)
        var_p = (xp.pow(2).sum(dim=-1) / safe_counts.squeeze(-1)).clamp_min(eps)
        var_t = (xt.pow(2).sum(dim=-1) / safe_counts.squeeze(-1)).clamp_min(eps)
        rho = cov / (var_p.sqrt() * var_t.sqrt() + eps)
        rho = torch.where(valid_rows, rho, torch.zeros_like(rho))
        return (1.0 - rho).mean()


class StabilityRegularizer(nn.Module):
    """Penalize day-to-day volatility of predictions per-asset.

    If batch groups multiple consecutive days, this computes ||s_t - s_{t-1}||^2 across assets.
    """

    def __init__(self, weight: float = 0.2) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, preds: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        # preds: [B, A] for a given horizon, where B groups consecutive days
        if preds.shape[0] < 2 or self.weight <= 0.0:
            return torch.tensor(0.0, device=preds.device)
        # 直接用张量差分并结合掩码，彻底消除逐样本 for 循环
        diff = preds[1:] - preds[:-1]
        if targets is not None:
            mask = (~torch.isnan(targets[1:])) & (~torch.isnan(targets[:-1]))
            valid_counts = mask.sum(dim=-1).clamp_min(1).float()
            diff = torch.where(mask, diff, torch.zeros_like(diff))
            penalty = (diff.pow(2).sum(dim=-1) / valid_counts)
        else:
            penalty = diff.pow(2).mean(dim=-1)
        if penalty.numel() == 0:
            return torch.zeros((), device=preds.device)
        return self.weight * penalty.mean()
