from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch

import kaggle_evaluation.core.templates
from kaggle_evaluation.core.templates import InferenceServer

from camht.data import build_normalized_windows
from camht.model import CAMHT
from camht.targets import compute_target_matrix, parse_target_pairs
from camht.utils import SDPPolicy, configure_sdp, get_device


def build_model(snapshot: str, device: torch.device) -> CAMHT:
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
    ckpt = torch.load(snapshot, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def predict_endpoint(model: CAMHT, test_hist: pl.DataFrame, label_lag_1: pl.DataFrame, label_lag_2: pl.DataFrame, label_lag_3: pl.DataFrame, label_lag_4: pl.DataFrame) -> pd.DataFrame:
    # Build per-target series up to current date
    date_col = "date_id"
    # Locate target_pairs.csv from Kaggle input path or local data
    candidates = [
        "/kaggle/input/mitsui-commodity-prediction-challenge/target_pairs.csv",
        "data/target_pairs.csv",
    ]
    target_pairs_path = next((p for p in candidates if Path(p).exists()), None)
    if target_pairs_path is None:
        raise FileNotFoundError("target_pairs.csv not found in expected locations")
    defs = parse_target_pairs(str(target_pairs_path))
    series_df, order = compute_target_matrix(test_hist, date_col, defs)
    window = 512
    base = series_df.sort(date_col)
    values = base.drop(date_col).to_numpy()
    # 复用训练端的滑窗标准化，确保线上和离线统计完全一致
    windows = build_normalized_windows(values, window, 0.0)
    latest = torch.from_numpy(windows[-1]).float()  # [A, W]
    times_grid = torch.linspace(0.0, 1.0, window, dtype=torch.float32)
    times = times_grid.view(1, 1, window, 1).expand(1, latest.shape[0], window, 1)

    device = next(model.parameters()).device
    with torch.no_grad():
        preds = model(
            latest.unsqueeze(0).unsqueeze(-1).to(device),
            times.to(device),
        )  # [H, 1, A]
        preds = preds[:, 0].float().cpu().numpy()
    # Day-level standardization (stabilize ranks)
    preds = [((p - p.mean()) / (p.std() + 1e-8)) for p in preds]
    # optional tanh clip for stability
    clip = float(os.getenv("CAMHT_TANH_CLIP", "0.0"))
    if clip > 0:
        preds = [np.tanh(p / clip) for p in preds]

    # Build single-row DataFrame in the exact column order expected today (duplicates allowed)
    def _cols(df: pl.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in ("date_id", "label_date_id")]

    cols1 = _cols(label_lag_1)
    cols2 = _cols(label_lag_2)
    cols3 = _cols(label_lag_3)
    cols4 = _cols(label_lag_4)
    cols_all = cols1 + cols2 + cols3 + cols4
    # Map target name -> index in 'order'
    idx = {name: i for i, name in enumerate(order)}
    values: list[float] = []
    for k, cols in enumerate([cols1, cols2, cols3, cols4]):
        for c in cols:
            v = preds[k][idx[c]] if c in idx else 0.0
            values.append(float(v))
    # Single-row pandas DataFrame without date_id column
    return pd.DataFrame([values], columns=cols_all)


class CAMHTServer(InferenceServer):
    def __init__(self, snapshot: str):
        self.device = get_device()
        configure_sdp(SDPPolicy())
        self.model = build_model(snapshot, self.device)
        super().__init__(predict)  # register endpoint

    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None, *args, **kwargs):
        import kaggle_evaluation.mitsui_gateway as mitsui_gateway

        return mitsui_gateway.MitsuiGateway(data_paths)


_SERVER: CAMHTServer | None = None


def predict(*args) -> pd.DataFrame:
    global _SERVER
    if _SERVER is None:
        raise RuntimeError("Server not initialized")
    return predict_endpoint(_SERVER.model, *args)


if __name__ == "__main__":
    snapshot = os.getenv("CAMHT_SNAPSHOT", "checkpoints/best.ckpt")
    _SERVER = CAMHTServer(snapshot)
    # Run local gateway for test with local data
    if os.getenv("CAMHT_LOCAL_GATEWAY", "0") == "1":
        _SERVER.run_local_gateway(data_paths=(".",))
    else:
        _SERVER.serve()
