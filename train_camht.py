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
from camht.distributed import (
    DistConfig,
    GradAccumulationContext,
    all_reduce_mean,
    cleanup_distributed,
    get_device,
    get_rank,
    get_world_size,
    is_main_process,
    print_once,
    save_on_main,
    setup_distributed,
    wrap_model_ddp,
    wrap_model_fsdp,
)
from camht.losses import DiffSpearmanLoss, StabilityRegularizer
from camht.metrics import spearman_rho
from camht.model import CAMHT, TransformerEncoder
from camht.targets import TargetDef, compute_pair_series, compute_target_matrix, parse_target_pairs
from camht.utils import (
    SDPPolicy,
    configure_sdp,
    console,
    maybe_compile,
    resolve_time2vec_kwargs,
    set_seed,
)


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
    """针对目标对序列的高性能滑窗数据集，彻底移除 Python for 循环。

    Fully compiled-friendly dataset with zero Python overhead in hot paths.
    完全编译友好的数据集，热路径零Python开销。
    """

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
        self.features = features.float().contiguous()
        self.labels = labels.float().contiguous()  # [H, D, A]
        self.label_mask = (~torch.isnan(self.labels)).to(torch.bool)
        self.dates = dates.long().contiguous()
        self.batch_days = int(batch_days)
        self.time_grid = time_grid.float().contiguous()
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
        x = self.features.index_select(0, sel).unsqueeze(-1).contiguous()  # [B, A, W, 1]
        times = self.time_grid.expand(x.shape[0], self.num_assets, self.window, 1)
        y_block = self.labels.index_select(1, sel).contiguous()  # [H, B, A]
        mask_block = self.label_mask.index_select(1, sel)
        counts = mask_block.sum(dim=-1).float().contiguous()  # [H, B]
        batch_dates = self.dates.index_select(0, sel)
        return x, times, y_block, counts, batch_dates


def _passthrough_collate(batch):
    if len(batch) != 1:
        raise ValueError("TargetsWindowDataset expects batch_size=1 in DataLoader")
    return batch[0]


def _suggest_worker_count(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    cpu_count = os.cpu_count() or 1
    # Use half of logical cores to avoid host saturation while keeping the GPU fed.
    return max(1, cpu_count // 2)


def _normalize_loader_section(section: object) -> Dict[str, object]:
    if section is None:
        return {}
    if isinstance(section, dict):
        return dict(section)
    if OmegaConf.is_config(section):
        resolved = OmegaConf.to_container(section, resolve=True)
        if isinstance(resolved, dict):
            return resolved
    raise TypeError(f"Unsupported loader configuration type: {type(section)!r}")


def _resolve_loader_options(cfg, role: str) -> Dict[str, object]:
    keys = (
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "pin_memory_device",
    )
    options: Dict[str, object] = {}
    for key in keys:
        if cfg.train.get(key, None) is not None:
            options[key] = cfg.train.get(key)
    section_key = f"{role}_loader"
    section = cfg.train.get(section_key, None)
    try:
        options.update(_normalize_loader_section(section))
    except TypeError:
        if section is not None:
            console.print(
                f"[yellow]Ignoring unsupported config for {section_key}: {type(section)!r}"
            )
    return options


def _build_loader(
    dataset: TargetsWindowDataset,
    device: torch.device,
    *,
    loader_cfg: Dict[str, object],
    is_distributed: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    """Build high-performance DataLoader with distributed support.

    Optimizations:
    - Multi-worker data loading with persistent workers
    - Pinned memory for fast H2D transfers
    - Prefetching to keep GPU fed
    - DistributedSampler for multi-GPU training
    """
    suggested_workers = _suggest_worker_count(device)
    num_workers = loader_cfg.get("num_workers")
    if num_workers is None:
        num_workers = suggested_workers
    num_workers = int(max(0, int(num_workers)))
    pin_memory_default = device.type == "cuda"
    pin_memory = bool(loader_cfg.get("pin_memory", pin_memory_default)) and device.type == "cuda"
    persistent_default = num_workers > 0
    persistent_workers = bool(loader_cfg.get("persistent_workers", persistent_default)) and num_workers > 0
    prefetch_factor = loader_cfg.get("prefetch_factor")
    if prefetch_factor is None and num_workers > 0:
        prefetch_factor = 4
    pin_memory_device = loader_cfg.get("pin_memory_device")

    # Distributed sampler
    sampler = None
    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            drop_last=False,
        )
        shuffle = False  # Sampler handles shuffling

    loader_kwargs: Dict[str, object] = {
        "batch_size": 1,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": _passthrough_collate,
    }
    if num_workers > 0:
        if persistent_workers:
            loader_kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    if pin_memory and pin_memory_device:
        loader_kwargs["pin_memory_device"] = str(pin_memory_device)
    return DataLoader(dataset, **loader_kwargs)


def _make_optimizer(model: nn.Module, *, lr: float, weight_decay: float, device: torch.device, use_foreach: bool = True) -> optim.Optimizer:
    """Create optimizer with fused kernels for extreme performance.

    Optimizations:
    - Fused AdamW on Ampere+ GPUs (SM 8.0+)
    - foreach=True for vectorized parameter updates
    - Capturable for CUDA graph compatibility
    """
    extra: Dict[str, object] = {}
    fused_ok = False
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            index = device.index if device.index is not None else torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(index)
            fused_ok = major >= 8  # Ampere and newer
        except Exception:  # noqa: BLE001
            fused_ok = False

    if fused_ok:
        extra["fused"] = True
        if is_main_process():
            console.print("[green]Using fused AdamW optimizer")
    elif use_foreach:
        # foreach=True uses vectorized ops, faster than default
        extra["foreach"] = True
        if is_main_process():
            console.print("[green]Using foreach AdamW optimizer")

    try:
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **extra)
    except TypeError:
        # Fallback for older PyTorch versions
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


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
    grad_accum_steps: int = 1,
    is_distributed: bool = False,
):
    """Training loop with extreme performance optimizations.

    Optimizations:
    - Non-blocking H2D transfers
    - Gradient accumulation with no_sync
    - Minimal host-device synchronization
    - Fused optimizer step
    - AMP with BF16
    """
    model.train()
    total_loss = 0.0
    total_rho = 0.0
    steps = 0
    steps_per_epoch = max(1, len(loader))
    total_steps = max(1, total_epochs * steps_per_epoch)
    base_step = epoch * steps_per_epoch

    # Performance tracking
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    for step, (x, t, ys, counts, batch_dates) in enumerate(loader):
        # Non-blocking H2D transfers for maximum overlap
        x = x.to(device=device, non_blocking=True)
        t = t.to(device=device, non_blocking=True)
        ys = ys.to(device=device, non_blocking=True)
        counts = counts.float()
        per_horizon_weight = counts.sum(dim=-1)
        total_weight = float(per_horizon_weight.sum().item())
        if total_weight <= 0:
            continue

        # Cosine learning rate schedule with warmup
        lr = cosine_scheduler(base_step + step, total_steps, lr_max, warmup_pct)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation: only sync on last step
        is_accum_step = (step + 1) % grad_accum_steps != 0
        sync_this_step = not is_accum_step

        # Zero gradients (set_to_none=True for better performance)
        if not is_accum_step or step == 0:
            optimizer.zero_grad(set_to_none=True)

        # Forward pass with gradient accumulation context
        with GradAccumulationContext(model, sync=sync_this_step):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                preds = model(x, t)
                weighted_terms = []
                horizons = min(preds.shape[0], ys.shape[0])
                for k in range(horizons):
                    weight = float(per_horizon_weight[k].item()) / total_weight
                    if weight <= 0:
                        continue
                    loss_k = loss_main(preds[k], ys[k])
                    stab_k = loss_stab(preds[k], ys[k])
                    weighted_terms.append(weight * (loss_k + stab_k))
                if not weighted_terms:
                    continue
                loss = torch.stack(weighted_terms).sum()
                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps

            # Backward pass
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # Optimizer step only on sync steps
        if sync_this_step:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        # Metrics (avoid sync where possible)
        with torch.no_grad():
            # Rescale loss for logging
            total_loss += loss.item() * grad_accum_steps
            total_rho += spearman_rho(preds[0], ys[0]).item()
            steps += 1

    if steps == 0:
        return 0.0, 0.0

    # Reduce metrics across processes
    avg_loss = total_loss / steps
    avg_rho = total_rho / steps
    if is_distributed:
        metrics = torch.tensor([avg_loss, avg_rho], device=device)
        metrics = all_reduce_mean(metrics)
        avg_loss, avg_rho = metrics[0].item(), metrics[1].item()

    return avg_loss, avg_rho


@torch.no_grad()
def evaluate(model, loader, loss_main, loss_stab, device, amp_dtype, amp_enabled, return_daily: bool = False, is_distributed: bool = False):
    """Evaluation loop with minimal synchronization overhead."""
    model.eval()
    total_loss = 0.0
    total_rho = 0.0
    steps = 0
    daily_rhos: list[float] = []
    for x, t, ys, counts, batch_dates in loader:
        x = x.to(device=device, non_blocking=True)
        t = t.to(device=device, non_blocking=True)
        ys = ys.to(device=device, non_blocking=True)
        counts = counts.float()
        per_horizon_weight = counts.sum(dim=-1)
        total_weight = float(per_horizon_weight.sum().item())
        if total_weight <= 0:
            continue
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            preds = model(x, t)
        loss_val = 0.0
        horizons = min(preds.shape[0], ys.shape[0])
        for k in range(horizons):
            weight = float(per_horizon_weight[k].item()) / total_weight
            if weight <= 0:
                continue
            loss_val += weight * (
                loss_main(preds[k], ys[k]).item() + loss_stab(preds[k], ys[k]).item()
            )
        total_loss += loss_val
        preds_h0 = preds[0]
        # compute per-day rho within这个 batch（完全向量化处理缺失值）
        target = ys[0]
        mask = ~torch.isnan(target)
        valid = mask.sum(dim=-1) >= 2
        fill_value = torch.finfo(preds_h0.dtype).min
        preds_masked = torch.where(mask, preds_h0, torch.full_like(preds_h0, fill_value))
        target_masked = torch.where(mask, target, torch.full_like(target, fill_value))
        rank_preds = torch.argsort(torch.argsort(preds_masked, dim=-1), dim=-1).float()
        rank_target = torch.argsort(torch.argsort(target_masked, dim=-1), dim=-1).float()
        rank_preds = torch.where(mask, rank_preds, torch.zeros_like(rank_preds))
        rank_target = torch.where(mask, rank_target, torch.zeros_like(rank_target))
        mask_f = mask.to(preds_h0.dtype)
        valid_counts = mask.sum(dim=-1).clamp_min(1).float()
        mean_p = (rank_preds * mask_f).sum(dim=-1, keepdim=True) / valid_counts.unsqueeze(-1)
        mean_t = (rank_target * mask_f).sum(dim=-1, keepdim=True) / valid_counts.unsqueeze(-1)
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

    # Reduce metrics across processes
    avg_loss = total_loss / steps
    avg_rho = total_rho / steps
    if is_distributed:
        metrics = torch.tensor([avg_loss, avg_rho], device=device)
        metrics = all_reduce_mean(metrics)
        avg_loss, avg_rho = metrics[0].item(), metrics[1].item()

    if return_daily:
        return avg_loss, avg_rho, daily_rhos
    return avg_loss, avg_rho


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="camht/configs/camht.yaml")
    parser.add_argument("--local_rank", type=int, default=None, help="Local rank for distributed training")
    args = parser.parse_args()

    # Initialize distributed training
    dist_config = setup_distributed()
    is_distributed = dist_config.world_size > 1

    cfg = OmegaConf.load(args.cfg)

    # Set seed (with rank offset for data augmentation diversity)
    set_seed(cfg.seed + get_rank())

    # Configure SDP backends
    configure_sdp(SDPPolicy())

    # Get device for this process
    device = get_device(dist_config.local_rank)

    # Enable TF32 for Ampere+ GPUs (extreme performance boost)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")
        # cuDNN autotuner for stable shapes
        torch.backends.cudnn.benchmark = True
        if is_main_process():
            console.print("[green]Enabled TF32 and cuDNN autotuning for extreme performance")

    amp_dtype = _resolve_amp_dtype(cfg.train.get("amp_dtype", "auto"))
    amp_enabled = bool(cfg.train.amp) and device.type == "cuda"

    # Gradient accumulation steps
    grad_accum = int(cfg.train.get("grad_accum", 1))

    dataset, date_list = build_datasets(cfg)

    def subset(ds: TargetsWindowDataset, idxs):
        return ds.view(idxs)

    # Only create dirs and writer on main process
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg.logging.log_dir) if is_main_process() else None

    use_cpcv = bool(cfg.cv.use_cpcv)
    all_snapshots: list[Path] = []
    best_global_score = -1e9
    best_global_path: Path | None = None
    train_loader_cfg = _resolve_loader_options(cfg, "train")
    eval_loader_cfg = _resolve_loader_options(cfg, "eval")
    if not eval_loader_cfg:
        eval_loader_cfg = dict(train_loader_cfg)

    if use_cpcv:
        # CPCV outer loop
        dates = dataset.dates.tolist()
        from camht.cpcv import CPCVSpec, cpcv_splits

        spec = CPCVSpec(n_splits=int(cfg.cv.n_splits), embargo_days=int(cfg.cv.embargo_days))
        for fold_id, (tr_idx, te_idx) in enumerate(cpcv_splits(dates, spec)):
            train_loader = _build_loader(
                subset(dataset, tr_idx),
                device,
                loader_cfg=train_loader_cfg,
                is_distributed=is_distributed,
                shuffle=True,
            )
            val_loader = _build_loader(
                subset(dataset, te_idx),
                device,
                loader_cfg=eval_loader_cfg,
                is_distributed=is_distributed,
                shuffle=False,
            )

            model = CAMHT(
                in_channels=1,
                d_model=cfg.model.d_model,
                n_heads=cfg.model.n_heads,
                n_layers_intra=cfg.model.n_layers_intra,
                n_layers_cross=cfg.model.n_layers_cross,
                patch_len=cfg.data.patch_len,
                patch_stride=cfg.data.patch_stride,
                time2vec_dim=cfg.model.time2vec_dim,
                time2vec_kwargs=resolve_time2vec_kwargs(cfg.model),
                dropout=cfg.model.dropout,
                activation=cfg.model.activation,
                use_flash_attn=bool(cfg.model.use_flash_attn),
                grad_checkpointing=bool(cfg.train.grad_checkpointing),
                cross_group_size=int(cfg.model.cross_group_size),
            ).to(device)

            # Wrap model with DDP/FSDP before compilation
            use_fsdp = cfg.train.get("use_fsdp", False)
            if is_distributed:
                if use_fsdp:
                    model = wrap_model_fsdp(
                        model,
                        sharding_strategy=cfg.train.get("fsdp_sharding_strategy", "FULL_SHARD"),
                        mixed_precision_dtype=amp_dtype if amp_enabled else None,
                        transformer_layer_cls=TransformerEncoder,
                    )
                else:
                    model = wrap_model_ddp(
                        model,
                        device_ids=[dist_config.local_rank],
                        gradient_as_bucket_view=True,
                        broadcast_buffers=False,
                        static_graph=True,
                    )

            # torch.compile AFTER wrapping with DDP/FSDP
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
            opt = _make_optimizer(
                model,
                lr=cfg.train.lr_max,
                weight_decay=cfg.train.weight_decay,
                device=device,
            )
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
                    grad_accum_steps=grad_accum,
                    is_distributed=is_distributed,
                )
                val_loss, val_rho, daily = evaluate(model, val_loader, loss_main, loss_stab, device, amp_dtype, amp_enabled, return_daily=True, is_distributed=is_distributed)
                mean = float(np.mean(daily)) if len(daily) else 0.0
                std = float(np.std(daily) + 1e-6)
                sharpe_like = mean / std if std > 0 else 0.0

                # Log metrics only on main process
                if writer is not None:
                    writer.add_scalar(f"fold{fold_id}/loss/train", train_loss, epoch)
                    writer.add_scalar(f"fold{fold_id}/rho/train", train_rho, epoch)
                    writer.add_scalar(f"fold{fold_id}/loss/val", val_loss, epoch)
                    writer.add_scalar(f"fold{fold_id}/rho/val", val_rho, epoch)
                    writer.add_scalar(f"fold{fold_id}/sharpe_like/val", sharpe_like, epoch)

                print_once(f"fold {fold_id} epoch {epoch}: train_loss={train_loss:.4f} rho={train_rho:.4f} | val_loss={val_loss:.4f} rho={val_rho:.4f} sharpe_like={sharpe_like:.4f}")

                improved = sharpe_like > best_fold_score
                if improved:
                    best_fold_score = sharpe_like
                    best_fold_epoch = epoch
                    wait = 0
                    # Save checkpoint only on main process
                    path = ckpt_dir / f"best_fold_{fold_id}.ckpt"
                    # Extract state dict from DDP/FSDP wrapper
                    state_dict = model.module.state_dict() if is_distributed else model.state_dict()
                    save_on_main({"model": state_dict, "score": best_fold_score}, path)
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
        train_loader = _build_loader(
            subset(dataset, train_dates_idx),
            device,
            loader_cfg=train_loader_cfg,
            is_distributed=is_distributed,
            shuffle=True,
        )
        val_loader = _build_loader(
            subset(dataset, val_dates_idx),
            device,
            loader_cfg=eval_loader_cfg,
            is_distributed=is_distributed,
            shuffle=False,
        )

        model = CAMHT(
            in_channels=1,
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_layers_intra=cfg.model.n_layers_intra,
            n_layers_cross=cfg.model.n_layers_cross,
            patch_len=cfg.data.patch_len,
            patch_stride=cfg.data.patch_stride,
            time2vec_dim=cfg.model.time2vec_dim,
            time2vec_kwargs=resolve_time2vec_kwargs(cfg.model),
            dropout=cfg.model.dropout,
            activation=cfg.model.activation,
            use_flash_attn=bool(cfg.model.use_flash_attn),
            grad_checkpointing=bool(cfg.train.grad_checkpointing),
            cross_group_size=int(cfg.model.cross_group_size),
        ).to(device)

        # Wrap model with DDP/FSDP before compilation
        use_fsdp = cfg.train.get("use_fsdp", False)
        if is_distributed:
            if use_fsdp:
                model = wrap_model_fsdp(
                    model,
                    sharding_strategy=cfg.train.get("fsdp_sharding_strategy", "FULL_SHARD"),
                    mixed_precision_dtype=amp_dtype if amp_enabled else None,
                    transformer_layer_cls=TransformerEncoder,
                )
            else:
                model = wrap_model_ddp(
                    model,
                    device_ids=[dist_config.local_rank],
                    gradient_as_bucket_view=True,
                    broadcast_buffers=False,
                    static_graph=True,
                )

        # torch.compile AFTER wrapping with DDP/FSDP
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
        opt = _make_optimizer(
            model,
            lr=cfg.train.lr_max,
            weight_decay=cfg.train.weight_decay,
            device=device,
        )
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
                grad_accum_steps=grad_accum,
                is_distributed=is_distributed,
            )
            val_loss, val_rho, daily = evaluate(model, val_loader, loss_main, loss_stab, device, amp_dtype, amp_enabled, return_daily=True, is_distributed=is_distributed)
            mean = float(np.mean(daily)) if len(daily) else 0.0
            std = float(np.std(daily) + 1e-6)
            sharpe_like = mean / std if std > 0 else 0.0

            # Log metrics only on main process
            if writer is not None:
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("rho/train", train_rho, epoch)
                writer.add_scalar("loss/val", val_loss, epoch)
                writer.add_scalar("rho/val", val_rho, epoch)
                writer.add_scalar("sharpe_like/val", sharpe_like, epoch)

            print_once(f"epoch {epoch}: train_loss={train_loss:.4f} rho={train_rho:.4f} | val_loss={val_loss:.4f} rho={val_rho:.4f} sharpe_like={sharpe_like:.4f}")

            if sharpe_like > best_metric:
                best_metric = sharpe_like
                # Extract state dict from DDP/FSDP wrapper
                state_dict = model.module.state_dict() if is_distributed else model.state_dict()
                save_on_main({"model": state_dict, "score": best_metric}, ckpt_dir / "best.ckpt")
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    if writer is not None:
        writer.close()

    # Clean up distributed training
    cleanup_distributed()


if __name__ == "__main__":
    main()
