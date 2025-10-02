from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class DataSpec:
    window: int
    patch_len: int
    patch_stride: int
    date_col: str
    feature_cols: Optional[List[str]]
    normalize_per_day: bool = True


def infer_feature_columns(df: pl.DataFrame, date_col: str) -> List[str]:
    cols = [c for c in df.columns if c != date_col]
    # only numeric columns
    num_cols = [c for c in cols if pl.datatypes.is_numeric(df.schema[c])]
    return num_cols


def winsorize(x: np.ndarray, p: float = 0.005) -> np.ndarray:
    if p <= 0:
        return x
    lo, hi = np.quantile(x, [p, 1 - p])
    return np.clip(x, lo, hi)


def build_normalized_windows(
    values: np.ndarray,
    window: int,
    winsorize_p: float = 0.0,
) -> np.ndarray:
    """Return [D, A, window] tensor化滑窗并做逐资产标准化。

    参数:
        values: [D, A] 原始矩阵（按日期排序）
        window: 滑窗长度
        winsorize_p: 每侧截尾百分比
    """
    if window <= 0:
        raise ValueError("window must be positive")
    values = np.asarray(values, dtype=np.float32)
    pad = max(window - 1, 0)
    if pad:
        head = np.repeat(values[:1], pad, axis=0)
        values = np.concatenate([head, values], axis=0)
    windows = sliding_window_view(values, window_shape=(window,), axis=0)  # [D, A, window]
    windows = windows.astype(np.float32, copy=False)
    if winsorize_p > 0:
        lo = np.quantile(windows, winsorize_p, axis=-1, keepdims=True)
        hi = np.quantile(windows, 1 - winsorize_p, axis=-1, keepdims=True)
        windows = np.clip(windows, lo, hi)
    mean = windows.mean(axis=-1, keepdims=True)
    std = windows.std(axis=-1, keepdims=True) + 1e-6
    return (windows - mean) / std


class DailyRollingDataset(Dataset):
    """Return per-day batches with rolling window history for all targets/assets.

    Shape convention for model input:
      x: [B=days, A=assets, T=window, C=features]
      times: [B, A, T, 1] normalized [0,1]
      labels_h[k]: [B, A] where A equals number of targets
    """

    def __init__(
        self,
        train: pl.DataFrame,
        labels: pl.DataFrame,
        spec: DataSpec,
        batch_days: int,
        winsorize_p: float = 0.0,
    ) -> None:
        self.train = train
        self.labels = labels
        self.spec = spec
        self.batch_days = batch_days
        self.winsorize_p = winsorize_p

        # Align unique dates present in both frames
        dates = sorted(set(train[spec.date_col].unique()) & set(labels[spec.date_col].unique()))
        self.dates = [int(d) for d in dates]
        self.target_cols = [c for c in labels.columns if c != spec.date_col]

        if spec.feature_cols is None:
            self.feature_cols = infer_feature_columns(train, spec.date_col)
        else:
            self.feature_cols = spec.feature_cols

        # Precompute per-date->frame for quick slicing
        self._date_to_frame_idx = {int(d): i for i, d in enumerate(self.dates)}
        self._train_groups = train.groupby(spec.date_col, maintain_order=True)
        self._label_groups = labels.groupby(spec.date_col, maintain_order=True)

    def __len__(self) -> int:
        return max(0, (len(self.dates) + self.batch_days - 1) // self.batch_days)

    def _slice_window(self, upto_date: int) -> pl.DataFrame:
        # take last `window` days from train up to `upto_date`
        start_idx = max(0, self._date_to_frame_idx[upto_date] - self.spec.window + 1)
        end_idx = self._date_to_frame_idx[upto_date]
        keep = set(self.dates[start_idx : end_idx + 1])
        return self.train.filter(pl.col(self.spec.date_col).is_in(keep))

    def _per_day_tensor(self, date: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window_df = self._slice_window(date)
        day_df = self._train_groups.get_group(date)
        # assets as rows for the given day
        assets_today = day_df.shape[0]

        # Build x for each asset: recent window x features
        # We will left-align by asset identity via row order in target columns assumption.
        # Expect labels order matches target asset order; we will map by row order.
        # Stack per-asset tensor
        # For speed, use numpy then convert
        feats = []
        for _, row in day_df.iter_rows(named=True):
            # filter window_df by same asset signature via row position. Here we assume same column schema,
            # use all rows of window for all assets (proxy when unique asset id is not provided).
            # Practical: we treat all assets identically using same window (cross-sectional); order will match target columns.
            w = window_df.select(self.feature_cols).to_numpy()
            if self.spec.normalize_per_day:
                mu = w.mean(axis=0, keepdims=True)
                sd = w.std(axis=0, keepdims=True) + 1e-8
                w = (w - mu) / sd
            if self.winsorize_p > 0:
                w = np.apply_along_axis(lambda a: winsorize(a, self.winsorize_p), 0, w)
            feats.append(w[-self.spec.window :])
        x = np.stack(feats, axis=0)  # [A, T, C]
        times = np.linspace(0, 1, x.shape[1], dtype=np.float32)[None, :, None]
        times = np.repeat(times, x.shape[0], axis=0)

        # labels for this date
        y = self._label_groups.get_group(date).select(self.target_cols).to_numpy()
        # expected single row for the date
        y = y.squeeze(0)
        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(times.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
        )

    def __getitem__(self, idx: int):
        start = idx * self.batch_days
        end = min(len(self.dates), start + self.batch_days)
        batch_dates = self.dates[start:end]
        xs, ts, ys = [], [], []
        for d in batch_dates:
            x, t, y = self._per_day_tensor(d)
            if not torch.isfinite(x).all():
                x = torch.nan_to_num(x, nan=0.0)
            if not torch.isfinite(t).all():
                t = torch.nan_to_num(t, nan=0.0)
            if not torch.isfinite(y).all():
                y = torch.nan_to_num(y, nan=0.0)
            xs.append(x)
            ts.append(t)
            ys.append(y)
        # pad batch if necessary (for compile stability)
        max_A = max(x.shape[0] for x in xs)
        T = xs[0].shape[1]
        C = xs[0].shape[2]
        B = len(xs)
        x_pad = torch.zeros((B, max_A, T, C), dtype=torch.float32)
        t_pad = torch.zeros((B, max_A, T, 1), dtype=torch.float32)
        y_pad = torch.full((B, max_A), float("nan"), dtype=torch.float32)
        for i in range(B):
            a = xs[i].shape[0]
            x_pad[i, :a] = xs[i]
            t_pad[i, :a] = ts[i]
            # labels length should equal number of targets; if mismatch, trim/pad
            y_len = min(a, ys[i].shape[0])
            y_pad[i, :y_len] = ys[i][:y_len]
        return x_pad, t_pad, y_pad, batch_dates


def load_frames(
    train_csv: str,
    labels_csv: str,
    date_col: str,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    train = pl.read_csv(train_csv)
    labels = pl.read_csv(labels_csv)
    return train, labels
