from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from .features import Patchify, FlattenPatches, Time2Vec
from .model import EncoderStack


class TiMAE(nn.Module):
    """Time-series Masked AutoEncoder (Ti-MAE-like).

    - Patchify along time; add Time2Vec; project to d_model tokens
    - Randomly mask a ratio of patch tokens, replace by mask token
    - Encode (intra) and decode tokens; project back to raw patch content (without time2vec)
    - Loss is MSE over masked tokens only
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        n_heads: int,
        n_layers_enc: int,
        n_layers_dec: int,
        patch_len: int,
        patch_stride: int,
        time2vec_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_flash_attn: bool = True,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.patchify = Patchify(patch_len, patch_stride)
        self.time2vec = Time2Vec(1, time2vec_dim)
        self.flatten = FlattenPatches(in_channels + time2vec_dim, patch_len, d_model)
        self.encoder = EncoderStack(n_layers_enc, d_model, n_heads, dropout, activation, use_flash_attn, grad_ckpt=grad_checkpointing)
        self.decoder = EncoderStack(n_layers_dec, d_model, n_heads, dropout, activation, use_flash_attn, grad_ckpt=grad_checkpointing)
        self.proj_out = nn.Linear(d_model, in_channels * patch_len)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor, times: torch.Tensor, mask_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: [B, A, T, C]
            times: [B, A, T, 1]
            mask_ratio: fraction of patch tokens to mask
        Returns:
            recon: [B, A, N, P*C]
            mask:  [B, A, N] boolean mask where True = was masked
        """
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        if times.ndim == 3:
            times = times.unsqueeze(-1)

        if x.ndim != 4:
            raise ValueError(f"Expected x to be 4D [B, A, T, C], got {x.shape}")
        if times.ndim != 4:
            raise ValueError(f"Expected times to be 4D [B, A, T, 1], got {times.shape}")
        if x.shape[:3] != times.shape[:3]:
            raise ValueError(f"Mismatched shapes: x {x.shape} vs times {times.shape}")

        B, A, T, C = x.shape
        xa = rearrange(x, "B A T C -> (B A) T C")
        ta = rearrange(times, "B A T C -> (B A) T C")
        t2v = self.time2vec(ta)
        x_aug = torch.cat([xa, t2v], dim=-1)  # [BA, T, C+Tv]
        patches, _ = self.patchify(x_aug)  # [BA, N, P, C+Tv]
        tokens = self.flatten(patches)  # [BA, N, D]

        # targets from raw x (without time2vec)
        raw_patches, _ = self.patchify(xa)  # [BA, N, P, C]
        target = raw_patches.reshape(raw_patches.shape[0], raw_patches.shape[1], -1)  # [BA, N, P*C]

        # random mask
        N = tokens.shape[1]
        m = torch.rand((tokens.shape[0], N), device=tokens.device) < mask_ratio
        tokens_masked = tokens.clone()
        tokens_masked[m] = self.mask_token.expand_as(tokens)[m]

        enc = self.encoder(tokens_masked)
        dec = self.decoder(enc)
        recon = self.proj_out(dec)  # [BA, N, P*C]

        recon = recon.reshape(B, A, N, -1)
        mask = m.reshape(B, A, N)
        target = target.reshape(B, A, N, -1)
        return recon, target, mask

    def loss(self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return (recon - target).pow(2).mean()
        diff = (recon - target).pow(2)
        diff = diff.mean(dim=-1)  # [B, A, N]
        return (diff[mask]).mean()

