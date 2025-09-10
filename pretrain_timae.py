from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from camht.targets import parse_target_pairs, compute_target_matrix
from camht.timae import TiMAE
from camht.utils import SDPPolicy, configure_sdp, console, get_device, maybe_compile, set_seed


class SeriesDataset(Dataset):
    def __init__(self, series_df: pl.DataFrame, date_col: str, window: int, batch_days: int):
        self.df = series_df.sort(date_col)
        self.date_col = date_col
        self.window = window
        self.batch_days = batch_days
        self.dates = self.df[date_col].unique(maintain_order=True).to_list()

    def __len__(self):
        return max(1, (len(self.dates) + self.batch_days - 1) // self.batch_days)

    def __getitem__(self, idx):
        start = idx * self.batch_days
        end = min(len(self.dates), start + self.batch_days)
        dates = self.dates[start:end]
        xs, ts = [], []
        for d in dates:
            hist = self.df.filter(pl.col(self.date_col) <= d)
            X = hist.drop([self.date_col]).to_numpy()  # [T, A]
            X = X[-self.window :]
            mu = X.mean(axis=0, keepdims=True)
            sd = X.std(axis=0, keepdims=True) + 1e-8
            X = (X - mu) / sd
            X = np.expand_dims(X.T, -1)  # [A, T, 1]
            xs.append(X)
            T = X.shape[1]
            ts.append(np.linspace(0, 1, T, dtype=np.float32)[None, :, None].repeat(X.shape[0], axis=0))
        A = max(x.shape[0] for x in xs)
        T = xs[0].shape[1]
        B = len(xs)
        x_pad = np.zeros((B, A, T, 1), dtype=np.float32)
        t_pad = np.zeros((B, A, T, 1), dtype=np.float32)
        for i in range(B):
            a = xs[i].shape[0]
            x_pad[i, :a] = xs[i]
            t_pad[i, :a] = ts[i]
        return (
            torch.from_numpy(x_pad),
            torch.from_numpy(t_pad),
        )


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
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

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

