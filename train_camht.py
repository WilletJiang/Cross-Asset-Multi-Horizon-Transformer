from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from camht.cpcv import CPCVSpec, cpcv_splits
from camht.data import DataSpec, DailyRollingDataset, load_frames
from camht.losses import DiffSpearmanLoss, StabilityRegularizer
from camht.metrics import spearman_rho
from camht.model import CAMHT
from camht.targets import TargetDef, compute_pair_series, compute_target_matrix, parse_target_pairs
from camht.utils import SDPPolicy, configure_sdp, console, get_device, maybe_compile, set_seed


def cosine_scheduler(t: int, T: int, lr_max: float, warmup_pct: float) -> float:
    warmup = max(1, int(T * warmup_pct))
    if t < warmup:
        return lr_max * (t + 1) / warmup
    q = (t - warmup) / max(1, T - warmup)
    return 0.5 * lr_max * (1 + math.cos(math.pi * q))


def build_datasets(cfg) -> Tuple[DailyRollingDataset, List[int]]:
    train, labels_given = load_frames(cfg.data.train_csv, cfg.data.labels_csv, cfg.data.date_column)
    target_defs = parse_target_pairs(cfg.data.target_pairs_csv)

    # Compute per-target series from train.csv
    series_df, order = compute_target_matrix(train, cfg.data.date_column, target_defs)

    # Derive multi-horizon labels from series_df (percentage returns)
    # y_k[date] = s_{date+k}/s_{date} - 1
    date_col = cfg.data.date_column
    base = series_df.sort(date_col)
    targets_only = base.drop(date_col)
    Ys = {}
    for k in [1, 2, 3, 4]:
        shifted = targets_only.select(pl.all().shift(-k))
        y = (shifted - targets_only) / (targets_only.abs() + 1e-12)
        Ys[k] = pl.DataFrame({date_col: base[date_col]}).with_columns(y)

    # Use k=1 for now to drive supervised training; others can be included via multi-task
    labels = Ys[1]

    spec = DataSpec(
        window=cfg.data.window,
        patch_len=cfg.data.patch_len,
        patch_stride=cfg.data.patch_stride,
        date_col=cfg.data.date_column,
        feature_cols=None,
        normalize_per_day=False,
    )

    # Build dataset that uses only per-target pair series as a single feature channel
    # Construct synthetic train frame with just date + each target series as its own column; we will map to [A, T, 1]
    # The dataset class currently expects per-day rows of multiple assets; adapt by "assets == number of targets".
    # We'll create a thin adapter dataset below.

    return (build_adapter_dataset_multi(series_df, Ys, cfg), base[date_col].to_list())


class TargetsAdapterDataset(DailyRollingDataset):
    """Adapter to interpret target series as assets with single-channel feature.

    Overridden _per_day_tensor to use target series window across targets.
    """

    def __init__(self, series_df: pl.DataFrame, labels_df: pl.DataFrame, spec: DataSpec, batch_days: int, winsorize_p: float):
        super().__init__(series_df, labels_df, spec, batch_days, winsorize_p)
        # here feature_cols are targets themselves
        self.feature_cols = [c for c in series_df.columns if c != spec.date_col]
        self.target_cols = self.feature_cols

    def _per_day_tensor(self, date: int):
        window_df = self._slice_window(date)
        # x: assets == number of targets, feature channel C=1 from its own column
        W = window_df.select(self.feature_cols).to_numpy()  # [T, A]
        if self.winsorize_p > 0:
            W = np.apply_along_axis(lambda a: np.clip(a, np.quantile(a, self.winsorize_p), np.quantile(a, 1 - self.winsorize_p)), 0, W)
        W = W[-self.spec.window :]
        # normalize each asset series over the window
        mu = W.mean(axis=0, keepdims=True)
        sd = W.std(axis=0, keepdims=True) + 1e-8
        W = (W - mu) / sd
        # [A, T, 1]
        x = np.expand_dims(W.T, -1)
        times = np.linspace(0, 1, x.shape[1], dtype=np.float32)[None, :, None].repeat(x.shape[0], axis=0)
        # label vector for the date
        y = self._label_groups.get_group(date).select(self.target_cols).to_numpy().squeeze(0)
        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(times.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
        )


def build_adapter_dataset(series_df: pl.DataFrame, labels: pl.DataFrame, cfg) -> TargetsAdapterDataset:
    spec = DataSpec(
        window=cfg.data.window,
        patch_len=cfg.data.patch_len,
        patch_stride=cfg.data.patch_stride,
        date_col=cfg.data.date_column,
        feature_cols=None,
        normalize_per_day=False,
    )
    return TargetsAdapterDataset(series_df, labels, spec, cfg.train.batch_days, cfg.data.winsorize_p)


class TargetsAdapterDatasetMulti(TargetsAdapterDataset):
    def __init__(self, series_df: pl.DataFrame, labels_by_lag: dict[int, pl.DataFrame], spec: DataSpec, batch_days: int, winsorize_p: float):
        # use lag-1 to initialize parent
        super().__init__(series_df, labels_by_lag[1], spec, batch_days, winsorize_p)
        self.label_groups_list = [labels_by_lag[k].groupby(spec.date_col, maintain_order=True) for k in [1, 2, 3, 4]]

    def _per_day_tensor_multi(self, date: int):
        # Reuse super to get x,t and ignore single y
        x, t, _ = super()._per_day_tensor(date)
        ys = []
        for g in self.label_groups_list:
            y = g.get_group(date).select(self.target_cols).to_numpy().squeeze(0)
            ys.append(torch.from_numpy(y.astype(np.float32)))
        return x, t, ys

    def __getitem__(self, idx: int):
        start = idx * self.batch_days
        end = min(len(self.dates), start + self.batch_days)
        batch_dates = self.dates[start:end]
        xs, ts, ys_list = [], [], []  # ys_list: list over days of list over horizon
        for d in batch_dates:
            x, t, ys = self._per_day_tensor_multi(d)
            xs.append(x)
            ts.append(t)
            ys_list.append(ys)
        # pad batch dims
        max_A = max(x.shape[0] for x in xs)
        T = xs[0].shape[1]
        B = len(xs)
        x_pad = torch.zeros((B, max_A, T, 1), dtype=torch.float32)
        t_pad = torch.zeros((B, max_A, T, 1), dtype=torch.float32)
        # prepare y pads per horizon
        H = len(ys_list[0])
        y_pads = [torch.full((B, max_A), float("nan"), dtype=torch.float32) for _ in range(H)]
        for i in range(B):
            a = xs[i].shape[0]
            x_pad[i, :a] = xs[i]
            t_pad[i, :a] = ts[i]
            for h in range(H):
                y = ys_list[i][h]
                y_len = min(a, y.shape[0])
                y_pads[h][i, :y_len] = y[:y_len]
        return x_pad, t_pad, y_pads, batch_dates


def build_adapter_dataset_multi(series_df: pl.DataFrame, labels_by_lag: dict[int, pl.DataFrame], cfg) -> TargetsAdapterDatasetMulti:
    spec = DataSpec(
        window=cfg.data.window,
        patch_len=cfg.data.patch_len,
        patch_stride=cfg.data.patch_stride,
        date_col=cfg.data.date_column,
        feature_cols=None,
        normalize_per_day=False,
    )
    return TargetsAdapterDatasetMulti(series_df, labels_by_lag, spec, cfg.train.batch_days, cfg.data.winsorize_p)


def train_one_epoch(model, loader, optimizer, scaler, loss_main, loss_stab, device, epoch, total_epochs, lr_max, warmup_pct):
    model.train()
    total_loss = 0.0
    total_rho = 0.0
    steps = 0
    for x, t, ys, batch_dates in loader:
        x = x.to(device)
        t = t.to(device)
        ys = [y.to(device) for y in ys]
        lr = cosine_scheduler(epoch, total_epochs, lr_max, warmup_pct)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=scaler is not None):
            preds_list = model(x, t)
            # multi-horizon joint loss
            losses = []
            stabs = []
            for k in range(min(4, len(preds_list))):
                losses.append(loss_main(preds_list[k], ys[k]))
                stabs.append(loss_stab(preds_list[k], ys[k]))
            loss = torch.stack(losses).mean() + torch.stack(stabs).mean()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        with torch.no_grad():
            total_loss += loss.item()
            total_rho += spearman_rho(preds_list[0], ys[0]).item()
            steps += 1
    return total_loss / max(1, steps), total_rho / max(1, steps)


@torch.no_grad()
def evaluate(model, loader, loss_main, loss_stab, device, return_daily: bool = False):
    model.eval()
    total_loss = 0.0
    total_rho = 0.0
    steps = 0
    daily_rhos: list[float] = []
    for x, t, ys, batch_dates in loader:
        x = x.to(device)
        t = t.to(device)
        ys = [y.to(device) for y in ys]
        preds_list = model(x, t)
        loss = 0.0
        for k in range(min(4, len(preds_list))):
            loss = loss + loss_main(preds_list[k], ys[k]) + loss_stab(preds_list[k], ys[k])
        preds = preds_list[0]
        total_loss += loss.item()
        # compute per-day rho within this sample (B days inside)
        P = preds
        T = ys[0]
        for i in range(P.shape[0]):
            m = ~torch.isnan(T[i])
            if m.sum() < 2:
                continue
            rho_i = spearman_rho(P[i][None, m], T[i][None, m]).item()
            daily_rhos.append(rho_i)
        total_rho += spearman_rho(preds, ys[0]).item()
        steps += 1
    if return_daily:
        return total_loss / max(1, steps), total_rho / max(1, steps), daily_rhos
    return total_loss / max(1, steps), total_rho / max(1, steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="camht/configs/camht.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    set_seed(cfg.seed)
    configure_sdp(SDPPolicy())
    device = get_device()

    dataset, date_list = build_datasets(cfg)

    def subset(ds, idxs):
        ds2 = ds
        ds2.dates = [ds.dates[i] for i in idxs]
        return ds2

    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg.logging.log_dir)

    use_cpcv = bool(cfg.cv.use_cpcv)
    all_snapshots: list[Path] = []
    best_global_score = -1e9
    best_global_path: Path | None = None

    if use_cpcv:
        # CPCV outer loop
        dates = dataset.dates
        from camht.cpcv import CPCVSpec, cpcv_splits

        spec = CPCVSpec(n_splits=int(cfg.cv.n_splits), embargo_days=int(cfg.cv.embargo_days))
        for fold_id, (tr_idx, te_idx) in enumerate(cpcv_splits(dates, spec)):
            train_loader = DataLoader(subset(dataset, tr_idx), batch_size=1, shuffle=False, num_workers=0)
            val_loader = DataLoader(subset(dataset, te_idx), batch_size=1, shuffle=False, num_workers=0)

            model = CAMHT(
                in_channels=1,
                d_model=cfg.model.d_model,
                n_heads=cfg.model.n_heads,
                n_layers_intra=cfg.model.n_layers_intra,
                n_layers_cross=cfg.model.n_layers_cross,
                patch_len=cfg.data.patch_len,
                patch_stride=cfg.data.patch_stride,
                time2vec_dim=cfg.model.time2vec_dim,
                dropout=cfg.model.dropout,
                activation=cfg.model.activation,
                use_flash_attn=bool(cfg.model.use_flash_attn),
                grad_checkpointing=bool(cfg.train.grad_checkpointing),
                cross_group_size=int(cfg.model.cross_group_size),
            ).to(device)
            model = maybe_compile(model, cfg.model.use_compile)
            # optional: load TiMAE encoder weights
            if bool(cfg.train.use_pretrain) and Path(cfg.train.pretrain_ckpt).exists():
                try:
                    t = torch.load(cfg.train.pretrain_ckpt, map_location="cpu")
                    model.backbone.time2vec.load_state_dict(t["time2vec"])  # type: ignore
                    model.backbone.flatten.load_state_dict(t["flatten"])  # type: ignore
                    model.backbone.intra.load_state_dict(t["encoder"], strict=False)  # type: ignore
                    console.print("[green]Loaded TiMAE encoder weights")
                except Exception as e:  # noqa: BLE001
                    console.print(f"[yellow]Pretrain load failed: {e}")
            opt = optim.AdamW(model.parameters(), lr=cfg.train.lr_max, weight_decay=cfg.train.weight_decay)
            scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.train.amp))
            loss_main = DiffSpearmanLoss(cfg.loss.diffsort_temperature, cfg.loss.diffsort_regularization)
            loss_stab = StabilityRegularizer(cfg.loss.stability_lambda)

            best_fold_score = -1e9
            best_fold_epoch = -1
            patience = int(cfg.train.early_stop_patience)
            wait = 0
            for epoch in range(cfg.train.epochs):
                train_loss, train_rho = train_one_epoch(
                    model, train_loader, opt, scaler, loss_main, loss_stab, device, epoch, cfg.train.epochs, cfg.train.lr_max, cfg.train.warmup_pct
                )
                val_loss, val_rho, daily = evaluate(model, val_loader, loss_main, loss_stab, device, return_daily=True)
                mean = float(np.mean(daily)) if len(daily) else 0.0
                std = float(np.std(daily) + 1e-6)
                sharpe_like = mean / std if std > 0 else 0.0
                writer.add_scalar(f"fold{fold_id}/loss/train", train_loss, epoch)
                writer.add_scalar(f"fold{fold_id}/rho/train", train_rho, epoch)
                writer.add_scalar(f"fold{fold_id}/loss/val", val_loss, epoch)
                writer.add_scalar(f"fold{fold_id}/rho/val", val_rho, epoch)
                writer.add_scalar(f"fold{fold_id}/sharpe_like/val", sharpe_like, epoch)
                console.print(f"fold {fold_id} epoch {epoch}: train_loss={train_loss:.4f} rho={train_rho:.4f} | val_loss={val_loss:.4f} rho={val_rho:.4f} sharpe_like={sharpe_like:.4f}")
                improved = sharpe_like > best_fold_score
                if improved:
                    best_fold_score = sharpe_like
                    best_fold_epoch = epoch
                    wait = 0
                    path = ckpt_dir / f"best_fold_{fold_id}.ckpt"
                    torch.save({"model": model.state_dict(), "score": best_fold_score}, path)
                else:
                    wait += 1
                    if wait >= patience:
                        break
            # track global best
            if best_fold_score > best_global_score:
                best_global_score = best_fold_score
                best_global_path = ckpt_dir / f"best_fold_{fold_id}.ckpt"
            all_snapshots.append(ckpt_dir / f"best_fold_{fold_id}.ckpt")

        # choose best fold snapshot as checkpoints/best.ckpt for serving
        if best_global_path is not None:
            final_path = ckpt_dir / "best.ckpt"
            import shutil

            shutil.copy2(best_global_path, final_path)
    else:
        # Simple holdout
        n = len(dataset.dates)
        val_cut = int(n * (1 - cfg.cv.test_size_fraction))
        train_dates_idx = list(range(0, val_cut))
        val_dates_idx = list(range(val_cut, n))
        train_loader = DataLoader(subset(dataset, train_dates_idx), batch_size=1, shuffle=False, num_workers=0)
        val_loader = DataLoader(subset(dataset, val_dates_idx), batch_size=1, shuffle=False, num_workers=0)

        model = CAMHT(
            in_channels=1,
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_layers_intra=cfg.model.n_layers_intra,
            n_layers_cross=cfg.model.n_layers_cross,
            patch_len=cfg.data.patch_len,
            patch_stride=cfg.data.patch_stride,
            time2vec_dim=cfg.model.time2vec_dim,
            dropout=cfg.model.dropout,
            activation=cfg.model.activation,
            use_flash_attn=bool(cfg.model.use_flash_attn),
            grad_checkpointing=bool(cfg.train.grad_checkpointing),
            cross_group_size=int(cfg.model.cross_group_size),
        ).to(device)
        model = maybe_compile(model, cfg.model.use_compile)
        if bool(cfg.train.use_pretrain) and Path(cfg.train.pretrain_ckpt).exists():
            try:
                t = torch.load(cfg.train.pretrain_ckpt, map_location="cpu")
                model.backbone.time2vec.load_state_dict(t["time2vec"])  # type: ignore
                model.backbone.flatten.load_state_dict(t["flatten"])  # type: ignore
                model.backbone.intra.load_state_dict(t["encoder"], strict=False)  # type: ignore
                console.print("[green]Loaded TiMAE encoder weights")
            except Exception as e:  # noqa: BLE001
                console.print(f"[yellow]Pretrain load failed: {e}")
        opt = optim.AdamW(model.parameters(), lr=cfg.train.lr_max, weight_decay=cfg.train.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.train.amp))
        loss_main = DiffSpearmanLoss(cfg.loss.diffsort_temperature, cfg.loss.diffsort_regularization)
        loss_stab = StabilityRegularizer(cfg.loss.stability_lambda)

        best_metric = -1e9
        patience = int(cfg.train.early_stop_patience)
        wait = 0
        for epoch in range(cfg.train.epochs):
            train_loss, train_rho = train_one_epoch(
                model, train_loader, opt, scaler, loss_main, loss_stab, device, epoch, cfg.train.epochs, cfg.train.lr_max, cfg.train.warmup_pct
            )
            val_loss, val_rho, daily = evaluate(model, val_loader, loss_main, loss_stab, device, return_daily=True)
            mean = float(np.mean(daily)) if len(daily) else 0.0
            std = float(np.std(daily) + 1e-6)
            sharpe_like = mean / std if std > 0 else 0.0
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("rho/train", train_rho, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("rho/val", val_rho, epoch)
            writer.add_scalar("sharpe_like/val", sharpe_like, epoch)
            console.print(f"epoch {epoch}: train_loss={train_loss:.4f} rho={train_rho:.4f} | val_loss={val_loss:.4f} rho={val_rho:.4f} sharpe_like={sharpe_like:.4f}")
            if sharpe_like > best_metric:
                best_metric = sharpe_like
                torch.save({"model": model.state_dict(), "score": best_metric}, ckpt_dir / "best.ckpt")
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    writer.close()


if __name__ == "__main__":
    main()
