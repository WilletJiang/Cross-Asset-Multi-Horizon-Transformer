from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    import torchsort  # type: ignore
except Exception:  # noqa: BLE001
    torchsort = None


class DiffSpearmanLoss(nn.Module):
    """Differentiable Spearman via torchsort (SoftRank) with optional row weights."""

    def __init__(self, temperature: float = 0.1, regularization: str = "none") -> None:
        super().__init__()
        self.temperature = temperature
        self.regularization = regularization

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        *,
        row_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if preds.ndim > 2:
            preds = preds.reshape(-1, preds.shape[-1])
            targets = targets.reshape(-1, targets.shape[-1])
            if row_weights is not None:
                row_weights = row_weights.reshape(-1)
        return self._loss_2d(preds, targets, row_weights)

    def _loss_2d(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        row_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = ~torch.isnan(targets)
        mask_f = mask.to(preds.dtype)
        valid_counts = mask.sum(dim=-1)
        valid_rows = valid_counts >= 2
        if valid_rows.sum() == 0:
            return torch.zeros((), device=preds.device, dtype=preds.dtype)

        fill_value = torch.finfo(preds.dtype).min
        fill = torch.full_like(preds, fill_value)
        p_masked = torch.where(mask, preds, fill)
        t_masked = torch.where(mask, targets, fill)

        if torchsort is not None:
            sr_p = torchsort.soft_rank(
                p_masked,
                regularization=self.regularization,
                temperature=self.temperature,
            )
            sr_t = torchsort.soft_rank(
                t_masked,
                regularization=self.regularization,
                temperature=self.temperature,
            )
        else:
            sr_p = torch.argsort(torch.argsort(p_masked, dim=-1), dim=-1).to(preds.dtype)
            sr_t = torch.argsort(torch.argsort(t_masked, dim=-1), dim=-1).to(preds.dtype)

        invalid_counts = (~mask).sum(dim=-1, keepdim=True).to(preds.dtype)
        sr_p = sr_p - invalid_counts
        sr_t = sr_t - invalid_counts

        sr_p = torch.where(mask, sr_p, torch.zeros_like(sr_p))
        sr_t = torch.where(mask, sr_t, torch.zeros_like(sr_t))

        safe_counts = valid_counts.clamp_min(1).to(preds.dtype)
        mean_p = (sr_p * mask_f).sum(dim=-1) / safe_counts
        mean_t = (sr_t * mask_f).sum(dim=-1) / safe_counts
        xp = (sr_p - mean_p.unsqueeze(-1)) * mask_f
        xt = (sr_t - mean_t.unsqueeze(-1)) * mask_f
        eps = torch.tensor(1e-8, dtype=preds.dtype, device=preds.device)
        cov = (xp * xt).sum(dim=-1) / safe_counts
        var_p = (xp.pow(2).sum(dim=-1) / safe_counts).clamp_min(eps)
        var_t = (xt.pow(2).sum(dim=-1) / safe_counts).clamp_min(eps)
        rho = cov / (var_p.sqrt() * var_t.sqrt() + eps)
        rho = torch.where(valid_rows, rho, torch.zeros_like(rho))
        rho = torch.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)

        if row_weights is not None:
            weights = row_weights.to(device=preds.device, dtype=preds.dtype)
            weights = torch.where(valid_rows, weights, torch.zeros_like(weights))
            weight_sum = weights.sum()
            if weight_sum <= 0:
                return torch.zeros((), device=preds.device, dtype=preds.dtype)
            loss = (1.0 - rho) * weights
            return loss.sum() / weight_sum

        return (1.0 - rho[valid_rows]).mean()


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
            return preds.new_zeros(())
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
            return preds.new_zeros(())
        return self.weight * penalty.mean()
