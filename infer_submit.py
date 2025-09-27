from __future__ import annotations

import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
import polars as pl
import torch

from camht.data import build_normalized_windows
from camht.model import CAMHT
from camht.targets import TargetDef, compute_target_matrix, parse_target_pairs
from camht.utils import SDPPolicy, configure_sdp, get_device


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

    base = series_df.sort(args.date_col)
    values = base.drop(args.date_col).to_numpy()
    dates = base[args.date_col].to_numpy().astype(np.int64)
    windows = build_normalized_windows(values, args.window, 0.0)
    feature_tensor = torch.from_numpy(windows).float()  # [D, A, W]
    num_dates, num_assets, _ = feature_tensor.shape
    time_grid = torch.linspace(0.0, 1.0, args.window, dtype=torch.float32, device=device).view(1, 1, args.window, 1)

    predictions: list[list[np.ndarray]] = []
    clip = float(os.getenv("CAMHT_TANH_CLIP", "0.0"))
    with torch.no_grad():
        for idx in range(num_dates):
            x = feature_tensor[idx].unsqueeze(0).unsqueeze(-1).to(device, non_blocking=True)
            times = time_grid.expand(1, num_assets, args.window, 1)
            preds = model(x, times)  # [H, 1, A]
            preds_np = preds[:, 0].float().cpu().numpy()  # [H, A]
            # 横截面 z-score 稳定秩序
            std_preds = [(p - p.mean()) / (p.std() + 1e-8) for p in preds_np]
            if clip > 0:
                std_preds = [np.tanh(p / clip) for p in std_preds]
            predictions.append(std_preds)

    # Prepare submission-like frame (not strictly Kaggle format here)
    # Save for inspection
    out_dir = Path("preds")
    out_dir.mkdir(exist_ok=True)
    date_list = [int(d) for d in dates]
    for k, lag in enumerate([1, 2, 3, 4]):
        rows = []
        for i, date_id in enumerate(date_list):
            row = {args.date_col: date_id}
            for j, tname in enumerate(order):
                row[tname] = predictions[i][k][j]
            rows.append(row)
        pl.DataFrame(rows).write_csv(out_dir / f"preds_lag_{lag}.csv")


if __name__ == "__main__":
    main()
