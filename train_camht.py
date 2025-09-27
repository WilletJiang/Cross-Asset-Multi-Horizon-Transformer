from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from camht.cpcv import CPCVSpec, cpcv_splits
from camht.data import build_normalized_windows, load_frames
from camht.losses import DiffSpearmanLoss, StabilityRegularizer
from camht.metrics import spearman_rho
from camht.model import CAMHT
from camht.targets import TargetDef, compute_pair_series, compute_target_matrix, parse_target_pairs
from camht.utils import SDPPolicy, configure_sdp, console, get_device, maybe_compile, set_seed


def _resolve_amp_dtype(cfg_value: str | None) -> torch.dtype:
    if cfg_value is None or str(cfg_value).lower() == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    value = str(cfg_value).lower()
    if value in {"bf16", "bfloat16"}:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        console.print("[yellow]Requested bf16 AMP but device不支持，自动改用 float16")
        return torch.float16
    if value in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"Unsupported amp_dtype: {cfg_value}")


def cosine_scheduler(t: int, T: int, lr_max: float, warmup_pct: float) -> float:
    warmup = max(1, int(T * warmup_pct))
    if t < warmup:
        return lr_max * (t + 1) / warmup
    q = (t - warmup) / max(1, T - warmup)
    return 0.5 * lr_max * (1 + math.cos(math.pi * q))


def build_datasets(cfg):
    train, _ = load_frames(cfg.data.train_csv, cfg.data.labels_csv, cfg.data.date_column)
    target_defs = parse_target_pairs(cfg.data.target_pairs_csv)

    # Compute per-target price series from train.csv（用于特征窗口）
    series_df, order = compute_target_matrix(train, cfg.data.date_column, target_defs)

    # Derive multi-horizon labels from series_df (percentage returns)
    # 对齐目标定义：使用 log-return
    date_col = cfg.data.date_column
    base = series_df.sort(date_col)
    horizons = [1, 2, 3, 4]

    # 预取所需基础资产列，计算 log-price，再组合成目标 log-return
    eps = 1e-6
    price_cols: set[str] = set()
    for tdef in target_defs.values():
        expr = tdef.expr
        if "-" in expr:
            left, right = [s.strip() for s in expr.split("-")]
            price_cols.add(left)
            price_cols.add(right)
        else:
            price_cols.add(expr.strip())
    price_cols = sorted(price_cols)
    train_sorted = train.sort(date_col)
    price_mat = train_sorted.select(price_cols).to_numpy()
    price_mat = np.clip(price_mat.astype(np.float64), eps, None)
    log_prices = np.log(price_mat)
    col_to_idx = {col: idx for idx, col in enumerate(price_cols)}

    log_returns_by_h = {}
    n_dates = log_prices.shape[0]
    for k in horizons:
        diff = np.full_like(log_prices, np.nan)
        if k < n_dates:
            diff[:-k, :] = log_prices[k:, :] - log_prices[:-k, :]
        log_returns_by_h[k] = diff

    label_tensors = []
    for k in horizons:
        horizon_returns = []
        diff = log_returns_by_h[k]
        for target_name in order:
            expr = target_defs[target_name].expr
            if "-" in expr:
                left, right = [s.strip() for s in expr.split("-")]
                ret = diff[:, col_to_idx[left]] - diff[:, col_to_idx[right]]
            else:
                ret = diff[:, col_to_idx[expr.strip()]]
            horizon_returns.append(ret.astype(np.float32))
        horizon_stack = np.stack(horizon_returns, axis=1)  # [D, A]
        label_tensors.append(horizon_stack)
    labels_tensor_np = np.stack(label_tensors, axis=0)  # [H, D, A]

    values = base.drop(date_col).to_numpy()
    dates = base[date_col].to_numpy()
    # 一次性构建所有日期的滑窗并标准化，避免训练过程中重复做 numpy 拷贝
    features = build_normalized_windows(values, cfg.data.window, cfg.data.winsorize_p)
    feature_tensor = torch.from_numpy(features)  # [D, A, W]
    time_grid = torch.linspace(0.0, 1.0, cfg.data.window, dtype=torch.float32).view(1, 1, cfg.data.window, 1)

    labels_tensor = torch.from_numpy(labels_tensor_np)

    dataset = TargetsWindowDataset(
        features=feature_tensor,
        labels=labels_tensor,
        dates=torch.from_numpy(dates.astype(np.int64)),
        batch_days=int(cfg.train.batch_days),
        time_grid=time_grid,
    )
    return dataset, dataset.dates.tolist()


class TargetsWindowDataset(torch.utils.data.Dataset):
    """针对目标对序列的高性能滑窗数据集，彻底移除 Python for 循环。"""

    def __init__(
        self,
        *,
        features: torch.Tensor,
        labels: torch.Tensor,
        dates: torch.Tensor,
        batch_days: int,
        time_grid: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> None:
        if features.ndim != 3:
            raise ValueError("features must be [D, A, W]")
        self.features = features.float()
        self.labels = labels.float()  # [H, D, A]
        self.dates = dates.long()
        self.batch_days = batch_days
        self.time_grid = time_grid.float()
        total = features.shape[0]
        base_idx = torch.arange(total, dtype=torch.long)
        self.indices = base_idx if indices is None else base_idx.index_select(0, indices.long())
        self.num_assets = features.shape[1]
        self.window = features.shape[2]

    def view(self, idxs: Sequence[int]) -> "TargetsWindowDataset":
        if isinstance(idxs, torch.Tensor):
            idx_tensor = idxs.to(torch.long)
        else:
            idx_tensor = torch.as_tensor(list(idxs), dtype=torch.long)
        return TargetsWindowDataset(
            features=self.features,
            labels=self.labels,
            dates=self.dates,
            batch_days=self.batch_days,
            time_grid=self.time_grid,
            indices=self.indices.index_select(0, idx_tensor),
        )

    def __len__(self) -> int:
        return max(0, (self.indices.numel() + self.batch_days - 1) // self.batch_days)

    def __getitem__(self, idx: int):
        start = idx * self.batch_days
        end = min(self.indices.numel(), start + self.batch_days)
        sel = self.indices[start:end]
        # 全部 tensor 操作，无 Python 循环，保持 torch.compile 友好
        x = self.features.index_select(0, sel).unsqueeze(-1)  # [B, A, W, 1]
        times = self.time_grid.expand(x.shape[0], self.num_assets, self.window, 1)
        y_block = self.labels.index_select(1, sel)  # [H, B, A]
        ys = list(y_block.unbind(0))
        counts = (~torch.isnan(y_block)).sum(dim=-1).float()
        counts_list = list(counts.unbind(0))
        batch_dates = self.dates.index_select(0, sel).tolist()
        return x, times, ys, counts_list, batch_dates


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    loss_main,
    loss_stab,
    device,
    epoch,
    total_epochs,
    lr_max,
    warmup_pct,
    amp_dtype,
    amp_enabled,
):
    model.train()
    total_loss = 0.0
    total_rho = 0.0
    steps = 0
    steps_per_epoch = max(1, len(loader))
    total_steps = max(1, total_epochs * steps_per_epoch)
    base_step = epoch * steps_per_epoch
    for step, (x, t, ys, counts, batch_dates) in enumerate(loader):
        x = x.to(device)
        t = t.to(device)
        if x.dim() > 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        if t.dim() > 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        ys = [y.to(device) for y in ys]
        ys = [y.squeeze(0) if y.dim() > 2 and y.shape[0] == 1 else y for y in ys]
        counts = [c.to(device) for c in counts]
        counts = [c.squeeze(0) if c.dim() > 1 and c.shape[0] == 1 else c for c in counts]
        total_weight = sum(float(c.sum().item()) for c in counts)
        if total_weight <= 0:
            continue
        lr = cosine_scheduler(base_step + step, total_steps, lr_max, warmup_pct)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            preds_list = model(x, t)
            weighted_terms = []
            for k in range(min(len(preds_list), len(counts))):
                weight = float(counts[k].sum().item()) / total_weight
                if weight <= 0:
                    continue
                loss_k = loss_main(preds_list[k], ys[k])
                stab_k = loss_stab(preds_list[k], ys[k])
                weighted_terms.append(weight * (loss_k + stab_k))
            if not weighted_terms:
                continue
            loss = torch.stack(weighted_terms).sum()
        if scaler is not None and scaler.is_enabled():
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
    if steps == 0:
        return 0.0, 0.0
    return total_loss / steps, total_rho / steps


@torch.no_grad()
def evaluate(model, loader, loss_main, loss_stab, device, amp_dtype, amp_enabled, return_daily: bool = False):
    model.eval()
    total_loss = 0.0
    total_rho = 0.0
    steps = 0
    daily_rhos: list[float] = []
    for x, t, ys, counts, batch_dates in loader:
        x = x.to(device)
        t = t.to(device)
        if x.dim() > 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        if t.dim() > 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        ys = [y.to(device) for y in ys]
        ys = [y.squeeze(0) if y.dim() > 2 and y.shape[0] == 1 else y for y in ys]
        counts = [c.to(device) for c in counts]
        counts = [c.squeeze(0) if c.dim() > 1 and c.shape[0] == 1 else c for c in counts]
        total_weight = sum(float(c.sum().item()) for c in counts)
        if total_weight <= 0:
            continue
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            preds_list = model(x, t)
        loss_val = 0.0
        for k in range(min(len(preds_list), len(counts))):
            weight = float(counts[k].sum().item()) / total_weight
            if weight <= 0:
                continue
            loss_val += weight * (
                loss_main(preds_list[k], ys[k]).item() + loss_stab(preds_list[k], ys[k]).item()
            )
        total_loss += loss_val
        preds = preds_list[0]
        # compute per-day rho within这个 batch（完全向量化处理缺失值）
        target = ys[0]
        mask = ~torch.isnan(target)
        valid = mask.sum(dim=-1) >= 2
        fill_value = torch.finfo(preds.dtype).min
        preds_masked = torch.where(mask, preds, torch.full_like(preds, fill_value))
        target_masked = torch.where(mask, target, torch.full_like(target, fill_value))
        rank_preds = torch.argsort(torch.argsort(preds_masked, dim=-1), dim=-1).float()
        rank_target = torch.argsort(torch.argsort(target_masked, dim=-1), dim=-1).float()
        rank_preds = torch.where(mask, rank_preds, torch.zeros_like(rank_preds))
        rank_target = torch.where(mask, rank_target, torch.zeros_like(rank_target))
        mask_f = mask.to(preds.dtype)
        valid_counts = mask.sum(dim=-1).clamp_min(1).float()
        mean_p = (rank_preds * mask_f).sum(dim=-1, keepdim=True) / valid_counts
        mean_t = (rank_target * mask_f).sum(dim=-1, keepdim=True) / valid_counts
        xc = (rank_preds - mean_p) * mask_f
        yc = (rank_target - mean_t) * mask_f
        eps = 1e-8
        cov = (xc * yc).sum(dim=-1) / valid_counts.squeeze(-1)
        var_p = (xc.pow(2).sum(dim=-1) / valid_counts.squeeze(-1)).clamp_min(eps)
        var_t = (yc.pow(2).sum(dim=-1) / valid_counts.squeeze(-1)).clamp_min(eps)
        rho_day = cov / (var_p.sqrt() * var_t.sqrt() + eps)
        valid_rho = rho_day[valid]
        if return_daily:
            daily_rhos.extend(valid_rho.detach().cpu().tolist())
        batch_mean_rho = valid_rho.mean().item() if valid_rho.numel() > 0 else 0.0
        total_rho += batch_mean_rho
        steps += 1
    if steps == 0:
        if return_daily:
            return 0.0, 0.0, []
        return 0.0, 0.0
    if return_daily:
        return total_loss / steps, total_rho / steps, daily_rhos
    return total_loss / steps, total_rho / steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="camht/configs/camht.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    set_seed(cfg.seed)
    configure_sdp(SDPPolicy())
    device = get_device()

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")

    amp_dtype = _resolve_amp_dtype(cfg.train.get("amp_dtype", "auto"))
    amp_enabled = bool(cfg.train.amp) and device.type == "cuda"

    dataset, date_list = build_datasets(cfg)

    def subset(ds: TargetsWindowDataset, idxs):
        return ds.view(idxs)

    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg.logging.log_dir)

    use_cpcv = bool(cfg.cv.use_cpcv)
    all_snapshots: list[Path] = []
    best_global_score = -1e9
    best_global_path: Path | None = None

    if use_cpcv:
        # CPCV outer loop
        dates = dataset.dates.tolist()
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
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
            loss_main = DiffSpearmanLoss(cfg.loss.diffsort_temperature, cfg.loss.diffsort_regularization)
            loss_stab = StabilityRegularizer(cfg.loss.stability_lambda)

            best_fold_score = -1e9
            best_fold_epoch = -1
            patience = int(cfg.train.early_stop_patience)
            wait = 0
            for epoch in range(cfg.train.epochs):
                train_loss, train_rho = train_one_epoch(
                    model,
                    train_loader,
                    opt,
                    scaler,
                    loss_main,
                    loss_stab,
                    device,
                    epoch,
                    cfg.train.epochs,
                    cfg.train.lr_max,
                    cfg.train.warmup_pct,
                    amp_dtype,
                    amp_enabled,
                )
                val_loss, val_rho, daily = evaluate(model, val_loader, loss_main, loss_stab, device, amp_dtype, amp_enabled, return_daily=True)
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
        n = dataset.dates.numel()
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
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
        loss_main = DiffSpearmanLoss(cfg.loss.diffsort_temperature, cfg.loss.diffsort_regularization)
        loss_stab = StabilityRegularizer(cfg.loss.stability_lambda)

        best_metric = -1e9
        patience = int(cfg.train.early_stop_patience)
        wait = 0
        for epoch in range(cfg.train.epochs):
            train_loss, train_rho = train_one_epoch(
                model,
                train_loader,
                opt,
                scaler,
                loss_main,
                loss_stab,
                device,
                epoch,
                cfg.train.epochs,
                cfg.train.lr_max,
                cfg.train.warmup_pct,
                amp_dtype,
                amp_enabled,
            )
            val_loss, val_rho, daily = evaluate(model, val_loader, loss_main, loss_stab, device, amp_dtype, amp_enabled, return_daily=True)
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
