from __future__ import annotations

import torch


@torch.no_grad()
def masked_spearman(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise Spearman rho under an optional validity mask.

    Args:
        preds: [..., A]
        targets: [..., A]
        mask: [..., A] boolean mask where True marks valid entries. If None, derive from
            ~isnan(targets).

    Returns:
        rho: tensor of shape preds.shape[:-1] with per-row rho (invalid rows filled with 0)
        valid_rows: boolean tensor indicating rows with >=2 valid entries
    """

    if mask is None:
        mask = ~torch.isnan(targets)
    mask_f = mask.to(dtype=preds.dtype)
    valid_counts = mask.sum(dim=-1)
    valid_rows = valid_counts >= 2

    fill_value = torch.finfo(preds.dtype).min
    fill = torch.full_like(preds, fill_value)
    preds_masked = torch.where(mask, preds, fill)
    targets_masked = torch.where(mask, targets, fill)

    rank_preds = torch.argsort(torch.argsort(preds_masked, dim=-1), dim=-1).to(preds.dtype)
    rank_targets = torch.argsort(torch.argsort(targets_masked, dim=-1), dim=-1).to(preds.dtype)

    rank_preds = torch.where(mask, rank_preds, torch.zeros_like(rank_preds))
    rank_targets = torch.where(mask, rank_targets, torch.zeros_like(rank_targets))

    safe_counts = valid_counts.clamp_min(1).to(dtype=preds.dtype)
    mean_preds = (rank_preds * mask_f).sum(dim=-1, keepdim=True) / safe_counts.unsqueeze(-1)
    mean_targets = (rank_targets * mask_f).sum(dim=-1, keepdim=True) / safe_counts.unsqueeze(-1)

    xc = (rank_preds - mean_preds) * mask_f
    yc = (rank_targets - mean_targets) * mask_f
    eps = torch.tensor(1e-8, dtype=preds.dtype, device=preds.device)
    cov = (xc * yc).sum(dim=-1) / safe_counts
    var_p = (xc.pow(2).sum(dim=-1) / safe_counts).clamp_min(eps)
    var_t = (yc.pow(2).sum(dim=-1) / safe_counts).clamp_min(eps)
    rho = cov / (var_p.sqrt() * var_t.sqrt() + eps)
    rho = torch.where(valid_rows, rho, torch.zeros_like(rho))
    return rho, valid_rows


@torch.no_grad()
def spearman_rho(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    reduce: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Spearman rho averaged across rows with NaN-aware masking."""

    rho, valid_rows = masked_spearman(preds, targets, mask)
    if not reduce:
        return rho, valid_rows
    if valid_rows.any():
        return rho[valid_rows].mean()
    return torch.zeros((), device=preds.device, dtype=preds.dtype)

