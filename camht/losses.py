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
        B, A = preds.shape
        losses = []
        for i in range(B):
            mask = ~torch.isnan(targets[i])
            if mask.sum() < 2:
                continue
            p = preds[i, mask]
            t = targets[i, mask]
            if torchsort is not None:
                sr_p = torchsort.soft_rank(p[None, :], regularization=self.regularization, temperature=self.temperature)
                sr_t = torchsort.soft_rank(t[None, :], regularization=self.regularization, temperature=self.temperature)
                x = sr_p - sr_p.mean(dim=-1, keepdim=True)
                y = sr_t - sr_t.mean(dim=-1, keepdim=True)
                vx = torch.sqrt((x**2).mean(dim=-1) + 1e-8)
                vy = torch.sqrt((y**2).mean(dim=-1) + 1e-8)
                rho = (x * y).mean(dim=-1) / (vx * vy + 1e-8)
                losses.append(1.0 - rho.squeeze(0))
            else:
                r_t = torch.argsort(torch.argsort(t))
                r_p = torch.argsort(torch.argsort(p))
                r_t = r_t - r_t.float().mean()
                r_p = r_p - r_p.float().mean()
                rho = (r_p * r_t).float().mean() / (
                    torch.sqrt((r_p.float() ** 2).mean() + 1e-8) * torch.sqrt((r_t.float() ** 2).mean() + 1e-8)
                )
                losses.append(1.0 - rho)
        if len(losses) == 0:
            return torch.tensor(0.0, device=preds.device)
        return torch.stack(losses).mean()


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
        losses = []
        for i in range(1, preds.shape[0]):
            p1 = preds[i - 1]
            p2 = preds[i]
            if targets is not None:
                m = (~torch.isnan(targets[i - 1])) & (~torch.isnan(targets[i]))
                if m.sum() < 1:
                    continue
                p1 = p1[m]
                p2 = p2[m]
            losses.append((p2 - p1).pow(2).mean())
        if len(losses) == 0:
            return torch.tensor(0.0, device=preds.device)
        return self.weight * torch.stack(losses).mean()
