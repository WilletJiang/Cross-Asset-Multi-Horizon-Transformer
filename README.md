CAMHT：Cross-Asset Multi‑Horizon Transformer（竞赛级实现）

简介
- 任务：对 400+ 目标在每个交易日输出 1–4 天前瞻分数。评分为按天 Spearman 的均值/波动（Sharpe-like）。
- 方案：纯 Transformer Encoder（PatchTST 风格 Intra 时序 + Cross 资产注意力），训练直接优化可微 Spearman，叠加“稳定性正则”降低日度波动。外层使用 Purged CPCV + Embargo，内层以 Sharpe-like 提前停止。推理严格按 Kaggle 网关逐日滚动，无泄露。

目录
- `camht/`：核心库
  - `model.py`：CAMHT 主干与多期头；支持 FlashAttention‑2 与梯度检查点；可选“跨资产分组注意力”。
  - `features.py`：Time2Vec、时间 patch 化与线性投影。
  - `losses.py`：可微 Spearman（torchsort）+ 稳定性正则（支持掩码）。
  - `metrics.py`：Spearman ρ。
  - `data.py`：按日滚动数据集；“targets‑as‑assets”适配（以 pair 系列作为资产单通道特征）。
  - `targets.py`：解析与计算 `target_pairs.csv` 中的 pair（如 A-B）。
  - `cpcv.py`：CombinatorialPurgedCV（优先用 skfolio，不可用则回退简单时间切分+Embargo）。
  - `timae.py`：Ti‑MAE 风格预训练（遮盖重建）。
  - `configs/camht.yaml`：超参与开关配置。
- 训练/推理
  - `train_camht.py`：CPCV 外环 + Sharpe-like 早停；可选加载 Ti‑MAE 预训练权重。
  - `pretrain_timae.py`：Ti‑MAE 预训练（仅历史重建），产出 encoder 权重。
  - `infer_submit.py`：本地离线按日推理（输出每日 1–4 天分数 csv）。
  - `serve_predict.py`：Kaggle 网关对接（单行 DataFrame，列序与当日 label lag 列严格一致）。
- `kaggle_evaluation/`：官方风格网关/Relay；`mitsui_gateway.py` 定义逐日批喂接口。

环境
- Python 3.10+；CUDA（建议 Linux 上装 `flash-attn`）
- 依赖：`pip install -r requirements.txt`
  - 关键库：`torch`, `polars`, `pyarrow`, `torchsort`, `skfolio`, `einops`, `tensorboard`, `flash-attn`（Linux）

配置要点（`camht/configs/camht.yaml`）
- `model.use_flash_attn`: true/false，优先用 FlashAttention‑2，失败自动回落到 PyTorch SDPA。
- `model.cross_group_size`: >0 时启用跨资产分组注意力（每组大小），限制 Cross‑Asset token 数；0 为关闭。
- `train.use_pretrain` + `train.pretrain_ckpt`: 是否加载 Ti‑MAE 预训练 encoder 权重。
- `cv.use_cpcv`: true/false；true 为 Purged CPCV 外环，内层以 Sharpe-like（日度 mean/std）早停与选优。
- `train.early_stop_patience`: 早停耐心步数。
- `inference.tanh_clip` 或环境变量 `CAMHT_TANH_CLIP`: 推理阶段对日内标准化分数做可选 `tanh` 压缩，稳秩。

数据说明
- `data/train.csv`: 历史特征（含 date_id）；
- `data/target_pairs.csv`: 每个 `target_i` 对应的 pair 定义与 lag；
- `data/train_labels.csv`: 可做 sanity check，但训练主监督来自按 pair 计算的多滞后收益 `y_k = s[t+k]/s[t]-1`；
- `lagged_test_labels/test_labels_lag_k.csv`: 线上每天列序模板（serve 时严格对齐）。

流程（一条龙）
1) 可选预训练（Ti‑MAE）
- 仅历史自监督遮盖重建，训练 encoder 的长窗表示与缺失鲁棒性。
- 命令：
  - `python pretrain_timae.py --cfg camht/configs/camht.yaml`
  - 产出：`checkpoints/timae_encoder.ckpt`

2) 监督训练（CPCV + Sharpe-like 早停）
- 直接对齐比赛目标：可微 Spearman 主损失 + 稳定性正则（降低日度波动）。
- 命令：
- `python train_camht.py --cfg camht/configs/camht.yaml`
- 针对 RTX 5090 32G 进行了专门调优，可直接使用 `python train_camht.py --cfg camht/configs/camht_rtx5090.yaml`
- DataLoader 通过 `train.train_loader`/`train.eval_loader` 节点配置（工作线程、pin memory、prefetch），可按机器带宽灵活调节。
- 结果：
  - 每折最佳：`checkpoints/best_fold_{k}.ckpt`
  - 全局最佳（用于 serve）：`checkpoints/best.ckpt`
- 可视化：TensorBoard（`runs/camht`）记录 loss、ρ、Sharpe-like。

3) 本地离线推理
- 命令：
  - `python infer_submit.py --snapshot checkpoints/best.ckpt --data data/test.csv --target_pairs data/target_pairs.csv`
- 稳定器（可选）：
  - `CAMHT_TANH_CLIP=2.0 python infer_submit.py ...`
- 产出：`preds/preds_lag_{1..4}.csv`

4) Kaggle 网关联调
- 命令：
  - `CAMHT_SNAPSHOT=checkpoints/best.ckpt CAMHT_LOCAL_GATEWAY=1 python serve_predict.py`
  - 可选：`CAMHT_TANH_CLIP=2.0`、`CAMHT_CROSS_GROUP_SIZE=16`
- 说明：`serve_predict.py` 会严格按照 `mitsui_gateway.py` 的逐日列序构造单行 DataFrame 并返回；支持在线稳定器。

实现与原则（关键点）
- 目标对齐：训练/选模直接优化/评估与 Kaggle 度量同构（Spearman mean/std）。
- 工程无泄露：
  - 训练标签来自 pair 系列的前瞻收益，只使用过去→未来的位移；
  - 数据集按日滚动窗口；
  - 推理严格按“≤当日”构造窗口；
  - 标准化仅用日内横截面统计。
- 性能：
  - SDPA/FlashMHA；
  - 可选“跨资产分组注意力”控制 Cross‑Asset token；
  - AMP(bf16)、grad checkpoint、torch.compile。

常见问题
- FlashAttention‑2 安装失败：自动回退到 SDPA，性能稍降但功能完整。
- `torchsort` 不可用：损失退化为非严格可微的硬排序 Spearman（会告警），建议安装 `torchsort` 获得平滑梯度。
- 资产数量很大：开启 `model.cross_group_size`（例如 16/32）以缩减 Cross‑Asset 注意力序列长度。

许可证
- 依仓库默认，未添加额外版权头。

