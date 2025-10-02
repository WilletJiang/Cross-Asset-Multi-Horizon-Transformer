from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from camht.data import build_normalized_windows
from camht.targets import parse_target_pairs, compute_target_matrix
from camht.timae import TiMAE
from camht.utils import SDPPolicy, configure_sdp, console, get_device, maybe_compile, set_seed


class SeriesDataset(Dataset):
    def __init__(self, series_df: pl.DataFrame, date_col: str, window: int, batch_days: int):
        df = series_df.sort(date_col)
        values = df.drop([date_col]).to_numpy()
        self.features = torch.from_numpy(build_normalized_windows(values, window, 0.0))  # [D, A, W]
        self.dates = torch.from_numpy(df[date_col].to_numpy().astype(np.int64))
        self.batch_days = batch_days
        self.window = window
        self.num_assets = self.features.shape[1]
        self.time_grid = torch.linspace(0.0, 1.0, window, dtype=torch.float32).view(1, 1, window, 1)
        self.indices = torch.arange(self.features.shape[0], dtype=torch.long)

    def __len__(self) -> int:
        return max(0, (self.indices.numel() + self.batch_days - 1) // self.batch_days)

    def __getitem__(self, idx: int):
        start = idx * self.batch_days
        end = min(self.indices.numel(), start + self.batch_days)
        sel = self.indices[start:end]
        x = self.features.index_select(0, sel).unsqueeze(-1)  # [B, A, W, 1]
        t = self.time_grid.expand(x.shape[0], self.num_assets, self.window, 1)
        return x, t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="camht/configs/camht.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg)

    set_seed(int(cfg.seed))
    device = get_device()
    configure_sdp(SDPPolicy())

    # Build series_df from train.csv using target_pairs
    train = pl.read_csv(cfg.data.train_csv)
    defs = parse_target_pairs(cfg.data.target_pairs_csv)
    series_df, order = compute_target_matrix(train, cfg.data.date_column, defs)
    ds = SeriesDataset(series_df, cfg.data.date_column, cfg.pretrain.window, cfg.pretrain.batch_days)
    def _collate(batch):
        if len(batch) != 1:
            raise ValueError("SeriesDataset expects batch_size=1")
        return batch[0]

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=_collate)

    model = TiMAE(
        in_channels=1,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers_enc=cfg.model.n_layers_intra,
        n_layers_dec=4,
        patch_len=cfg.pretrain.patch_len,
        patch_stride=cfg.pretrain.patch_stride,
        time2vec_dim=cfg.pretrain.time2vec_dim,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation,
        use_flash_attn=bool(cfg.model.use_flash_attn),
        grad_checkpointing=False,
    ).to(device)
    model = maybe_compile(model, bool(cfg.model.use_compile))

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.pretrain.lr), weight_decay=float(cfg.pretrain.weight_decay))

    for epoch in range(int(cfg.pretrain.epochs)):
        model.train()
        total = 0.0
        steps = 0
        for x, t in loader:
            x = x.to(device)
            t = t.to(device)
            opt.zero_grad(set_to_none=True)
            recon, target, mask = model(x, t, mask_ratio=float(cfg.pretrain.mask_ratio))
            loss = model.loss(recon, target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            steps += 1
        console.print(f"[TiMAE] epoch {epoch}: loss={total/max(1,steps):.4f}")

    # Save encoder-related weights for fine-tuning
    out = Path(cfg.train.pretrain_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "time2vec": model.time2vec.state_dict(),
        "flatten": model.flatten.state_dict(),
        "encoder": model.encoder.state_dict(),
        "patchify": {"spec.size": cfg.pretrain.patch_len, "spec.stride": cfg.pretrain.patch_stride},
    }
    torch.save(state, out)
    console.print(f"[TiMAE] saved encoder checkpoint to {out}")


if __name__ == "__main__":
    main()
