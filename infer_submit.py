from __future__ import annotations

import argparse
from pathlib import Path
import os
from typing import List

import numpy as np
import pandas as pd
import polars as pl
import torch

from camht.model import CAMHT
from camht.targets import TargetDef, compute_target_matrix, parse_target_pairs
from camht.utils import SDPPolicy, configure_sdp, get_device


def _prepare_daily_window(series_df: pl.DataFrame, date_col: str, window: int) -> np.ndarray:
    series_df = series_df.sort(date_col)
    # last `window` rows
    sub = series_df[-window:]
    X = sub.drop([date_col]).to_numpy()  # [T, A]
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - mu) / sd
    X = np.expand_dims(X.T, -1)  # [A, T, 1]
    return X


def predict_one_day(model: CAMHT, series_df: pl.DataFrame, date_id: int, date_col: str, window: int) -> np.ndarray:
    X = _prepare_daily_window(series_df.filter(pl.col(date_col) <= date_id), date_col, window)
    T = X.shape[1]
    times = np.linspace(0, 1, T, dtype=np.float32)[None, :, None].repeat(X.shape[0], axis=0)
    device = next(model.parameters()).device
    with torch.no_grad():
        preds = model(
            torch.from_numpy(X).unsqueeze(0).to(device),  # [1, A, T, 1]
            torch.from_numpy(times).unsqueeze(0).to(device),  # [1, A, T, 1]
        )  # list of [1, A]
        preds = [p.squeeze(0).float().cpu().numpy() for p in preds]
    return preds  # list of [A]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=str, default="checkpoints/best.ckpt")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--data", type=str, default="data/test.csv")
    parser.add_argument("--target_pairs", type=str, default="data/target_pairs.csv")
    parser.add_argument("--date_col", type=str, default="date_id")
    args = parser.parse_args()

    device = get_device()
    configure_sdp(SDPPolicy())

    # Load model arch (match training defaults)
    group_size = int(os.getenv("CAMHT_CROSS_GROUP_SIZE", "0"))
    model = CAMHT(
        in_channels=1,
        d_model=640,
        n_heads=10,
        n_layers_intra=10,
        n_layers_cross=6,
        patch_len=16,
        patch_stride=8,
        time2vec_dim=8,
        use_flash_attn=True,
        grad_checkpointing=False,
        cross_group_size=group_size,
    ).to(device)
    ckpt = torch.load(args.snapshot, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Build target series
    test = pl.read_csv(args.data)
    defs = parse_target_pairs(args.target_pairs)
    series_df, order = compute_target_matrix(test, args.date_col, defs)

    predictions = []
    for date_id in series_df[args.date_col].unique(maintain_order=True).to_list():
        preds_list = predict_one_day(model, series_df, int(date_id), args.date_col, args.window)
        # day-level standardization across assets to stabilize ranks
        std_preds_list = []
        for p in preds_list:
            mu = p.mean()
            sd = p.std() + 1e-8
            std_preds_list.append((p - mu) / sd)
        # optional tanh clip from env var
        clip = float(os.getenv("CAMHT_TANH_CLIP", "0.0"))
        if clip > 0:
            std_preds_list = [np.tanh(p / clip) for p in std_preds_list]
        predictions.append(std_preds_list)  # list of 4 arrays [A]

    # Prepare submission-like frame (not strictly Kaggle format here)
    # Save for inspection
    out_dir = Path("preds")
    out_dir.mkdir(exist_ok=True)
    for k, lag in enumerate([1, 2, 3, 4]):
        rows = []
        for i, date_id in enumerate(series_df[args.date_col].unique(maintain_order=True).to_list()):
            row = {args.date_col: int(date_id)}
            for j, tname in enumerate(order):
                row[tname] = predictions[i][k][j]
            rows.append(row)
        pl.DataFrame(rows).write_csv(out_dir / f"preds_lag_{lag}.csv")


if __name__ == "__main__":
    main()
