from __future__ import annotations

import torch


@torch.no_grad()
def spearman_rho(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute Spearman rho across last dim and average across batch.

    Args:
      preds: [B, A]
      targets: [B, A]
    Returns:
      scalar tensor
    """
    r_t = torch.argsort(torch.argsort(targets, dim=-1), dim=-1).float()
    r_p = torch.argsort(torch.argsort(preds, dim=-1), dim=-1).float()
    r_t = r_t - r_t.mean(dim=-1, keepdim=True)
    r_p = r_p - r_p.mean(dim=-1, keepdim=True)
    num = (r_p * r_t).mean(dim=-1)
    den = torch.sqrt((r_p**2).mean(dim=-1) + 1e-8) * torch.sqrt((r_t**2).mean(dim=-1) + 1e-8)
    return (num / (den + 1e-8)).mean()

