from __future__ import annotations

import torch


@torch.no_grad()
def spearman_rho(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute Spearman rho across last dim and average across batch.

    NaN values in either tensor are ignored per-row so we keep AMP/TensorCore
    friendly code paths without graph breaks.

    Args:
      preds: [B, A]
      targets: [B, A]
    Returns:
      scalar tensor
    """
    mask = (~torch.isnan(targets)) & (~torch.isnan(preds))
    valid_counts = mask.sum(dim=-1, keepdim=True)
    valid_rows = valid_counts.squeeze(-1) >= 2
    if not torch.any(valid_rows):
        return torch.zeros((), device=preds.device, dtype=preds.dtype)
    fill_value = torch.finfo(preds.dtype).min
    fill = torch.full_like(preds, fill_value)
    preds_masked = torch.where(mask, preds, fill)
    targets_masked = torch.where(mask, targets, fill)
    r_t = torch.argsort(torch.argsort(targets_masked, dim=-1), dim=-1).float()
    r_p = torch.argsort(torch.argsort(preds_masked, dim=-1), dim=-1).float()
    r_t = torch.where(mask, r_t, torch.zeros_like(r_t))
    r_p = torch.where(mask, r_p, torch.zeros_like(r_p))
    safe_counts = valid_counts.clamp_min(1).float()
    r_t = r_t - (r_t.sum(dim=-1, keepdim=True) / safe_counts)
    r_p = r_p - (r_p.sum(dim=-1, keepdim=True) / safe_counts)
    num = (r_p * r_t).sum(dim=-1)
    den = (
        torch.sqrt((r_p.pow(2).sum(dim=-1)).clamp_min(1e-8))
        * torch.sqrt((r_t.pow(2).sum(dim=-1)).clamp_min(1e-8))
    )
    rho = torch.zeros_like(num)
    rho[valid_rows] = (num[valid_rows] / (den[valid_rows] + 1e-8))
    return rho.mean()

