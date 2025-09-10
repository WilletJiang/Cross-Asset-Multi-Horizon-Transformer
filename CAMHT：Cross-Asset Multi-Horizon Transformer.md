下面是【CAMHT：Cross-Asset Multi-Horizon Transformer】的“竞赛规则定制、纯 Transformer Encoder”版完整设计。它把**时间维度的长期依赖**、**资产间横截面信息**、**多预测期（1–4 天**)和**排行榜度量（按天的Spearman/稳定性**)放到一个统一训练框架里，并在工程上确保**无泄露**、**可复现**与**推理稳定**。

> 参考脉络（仅列关键载荷出处）：  
> 多步多变量预测的 Transformer 代表作 TFT（多视角、多步预测）([arXiv](https://arxiv.org/abs/1912.09363?utm_source=chatgpt.com "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"), [科学直通车](https://www.sciencedirect.com/science/article/pii/S0169207021000637?utm_source=chatgpt.com "Temporal Fusion Transformers for interpretable multi ..."))；长窗口建模与打补丁（Patch）思想：PatchTST（ICLR’23）([arXiv](https://arxiv.org/abs/2211.14730?utm_source=chatgpt.com "A Time Series is Worth 64 Words: Long-term Forecasting ..."), [GitHub](https://github.com/yuqinie98/PatchTST?utm_source=chatgpt.com "PatchTST (ICLR 2023)"))；变量维度反转的 iTransformer（ICLR’24/24年版本）([arXiv](https://arxiv.org/abs/2310.06625?utm_source=chatgpt.com "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"), [GitHub](https://github.com/thuml/iTransformer?utm_source=chatgpt.com "thuml/iTransformer"))；跨变量（Cross-Dimension）注意力的 Crossformer（ICLR’23）([OpenReview](https://openreview.net/forum?id=vSVLM2j9eie&utm_source=chatgpt.com "Crossformer: Transformer Utilizing Cross-Dimension ..."), [GitHub](https://github.com/Thinklab-SJTU/Crossformer?utm_source=chatgpt.com "Official implementation of our ICLR 2023 paper \"Crossformer"))；时间嵌入 Time2Vec（通用可学习时间编码）([arXiv](https://arxiv.org/abs/1907.05321?utm_source=chatgpt.com "Time2Vec: Learning a Vector Representation of Time"))；可微分排序/排名（为 Spearman 构造可训练代理）([arXiv](https://arxiv.org/abs/2002.08871?utm_source=chatgpt.com "Fast Differentiable Sorting and Ranking"), [NeurIPS Papers](https://papers.neurips.cc/paper/8910-differentiable-ranking-and-sorting-using-optimal-transport.pdf?utm_source=chatgpt.com "Differentiable Ranking and Sorting using Optimal Transport"))；金融时序的“Purged/CPCV 带 Embargo”验证（防泄露）与开源实现([Agorism](https://agorism.dev/book/finance/ml/Marcos%20Lopez%20de%20Prado%20-%20Advances%20in%20Financial%20Machine%20Learning-Wiley%20%282018%29.pdf?utm_source=chatgpt.com "Advances in Financial Machine Learning"), [skfolio](https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html?utm_source=chatgpt.com "skfolio.model_selection.CombinatorialPurgedCV"), [维基百科](https://en.wikipedia.org/wiki/Purged_cross-validation?utm_source=chatgpt.com "Purged cross-validation"))；序列注意力的高效实现 FlashAttention-2（训练/推理加速）([arXiv](https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"))；本赛题数据与评测要点（类 Sharpe：按天的 Spearman 均值 / 标准差；多滞后标签）([Kaggle](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge?utm_source=chatgpt.com "MITSUI&CO. Commodity Prediction Challenge"), [DEV Community](https://dev.to/nk_maker/kaggle-diary-mitsuico-commodity-prediction-challenge-day-1-3kl4?utm_source=chatgpt.com "Kaggle Diary: MITSUI&CO. Commodity Prediction Challenge"))。

---

# 一、问题重述与目标函数对齐

**任务本质**：每天对 400+ 个目标（单品与价差）分别给出 1–4 天前瞻的**相对排序**信号，平台以“每日 Spearman 的均值 / 波动”**评估（Sharpe-like）。因此我们要优化的是**“横截面秩相关 + 稳定性”**，而不是单点回归误差。比赛同时提供多套“lag 标签”（1-4 天），且存在**交易日错位/假日**等导致滞后不齐的现实性约束（不要泄露到未来）。([Kaggle](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge?utm_source=chatgpt.com "MITSUI&CO. Commodity Prediction Challenge"), [DEV Community](https://dev.to/nk_maker/kaggle-diary-mitsuico-commodity-prediction-challenge-day-1-3kl4?utm_source=chatgpt.com "Kaggle Diary: MITSUI&CO. Commodity Prediction Challenge"))

**设计原则**：

1. **模型头*度量一致性**：训练用可微分“排序/相关”损失逼近 Spearman；
    
2. **稳定性正则**：惩罚“按天横截面相关的波动”（让日度 IC 更平滑）；
    
3. **多期联合**：共享骨干，分期（1–4 天）多头联合优化；
    
4. **横截面-时间双通道**：既学长程时序，又学资产间共振/相对强弱。
    

---

# 二、总体架构：CAMHT（纯 Encoder）

## 2.1 输入与打补丁（patching）

- **窗口**：对每个目标，取最近 _L_ 天的历史特征序列（含价格/量/技术因子/外部宏微变量等；仅用**可在 t 日已知**的信息）。
    
- **Patch 化**：借鉴 PatchTST，把长度 _L_ 切成长度 _P_、步长 _S_ 的不重叠/轻重叠小片，**每个 patch 作为 token**，显著降低注意力复杂度并保留局部语义。证明与代码生态见 PatchTST（ICLR’23）。([arXiv](https://arxiv.org/abs/2211.14730?utm_source=chatgpt.com "A Time Series is Worth 64 Words: Long-term Forecasting ..."))
    

## 2.2 时间编码与缺失模式编码

- **时间嵌入**：Time2Vec（线性分量+周期分量）与绝对/相对日历特征拼接，适配交易日不齐与多尺度周期。([arXiv](https://arxiv.org/abs/1907.05321?utm_source=chatgpt.com "Time2Vec: Learning a Vector Representation of Time"))
    
- **缺失/不规则采样**：对每个原始特征引入 **mask 与 Δtime 通道**（GRU-D 启发，但我们在前端特征层做，仍保持纯 Transformer 骨干），让模型知道“多久未更新/是否缺失”，利于金融数据常见的稀疏/节假日。([Nature](https://www.nature.com/articles/s41598-018-24271-9?utm_source=chatgpt.com "Recurrent Neural Networks for Multivariate Time Series ..."))
    

## 2.3 单资产（Intra-asset）时序编码层

- **Channel-independent 编码**（PatchTST 思想）：共享权重对每个目标的 patch-tokens 做时序自注意力，擅长长窗口记忆与局部语义。([arXiv](https://arxiv.org/abs/2211.14730?utm_source=chatgpt.com "A Time Series is Worth 64 Words: Long-term Forecasting ..."))


- 注意力实现：优先采用 **FlashAttention-2**（精确注意力、显存/吞吐优势），窗口较长时再选 **Performer（FAVOR+）** 近似以进一步降复杂度。([arXiv](https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"))
    

## 2.4 跨资产（Cross-asset）横截面编码层

- **动机**：比赛度量是**按天横截面秩相关**，因此“资产间的相对信号”**是核心。

- **做法**：将同一交易日 _t_、所有目标（单品&价差）的**时序编码摘要**（如[CLS]）组成一批 token，进入**跨资产自注意力**（ Crossformer 的“跨维度注意力”思想） ，刻画“共振/替代/对冲/曲线联动”。([OpenReview](https://openreview.net/forum?id=vSVLM2j9eie&utm_source=chatgpt.com "Crossformer: Transformer Utilizing Cross-Dimension ..."))

- **额外**：在此层加入**分组或分簇**（市场/品类/期限结构）以控制计算与提升同类互动质量；也可做**层次式**：先组内注意力，再组间稀疏注意力。这里你自己深度思考下作出权衡分析，给出最合适的选择。   

## 2.5 多期联合预测头（1–4 天）

- 一个共享的骨干后，接 **四个并行线性头**，分别输出 horizon ∈ {1,2,3,4} 的**原始分数**（logit），只用于**排序**。
    
- 训练时对每个 horizon 计算**当日横截面 Spearman 代理损失**（见 §3），并加上**稳定性正则**与**轻量回归校准**。

---

# 三、损失函数：对齐“榜单指标”

## 3.1 可微分 Spearman 代理（Listwise）

- 目标是**最大化**“按日的横截面 Spearman”。我们使用**可微分排序/排名**的连续松弛：
    
    - **Fast Differentiable Sorting/Ranking**（投影到置换多面体，O(n log n)；论文展示了**可微 Spearman**的构造）；

> 训练张量化：对某天 _t_、horizon _h_，取该天全部目标的预测分数 **p_{t,h}∈ℝ^{N_t}** 与标签 **y_{t,h}**，用可微排序近似 **rank(p_{t,h})**、**rank(y_{t,h})**，最小化 **1 − ρ_s** 的平滑代理（或最大化 ρ_s）。

## 3.2 稳定性（Sharpe-like）正则

- 赛事排行榜是“日度相关的均值/标准差”。在训练期，把每个 batch（跨多天样本）计算得到的**日度相关序列的方差作为正则项 **λ·Var(ρ_s,day)**，抑制过度抖动，让“信息比率”型目标更优。

- 为防止“过稳导致均值下降”，采用**双目标权衡**：
    $$L=∑h(ListwiseLossh⏟_提升相关均值+λ_hVarPerDay(ρ_s,h)⏟_降低相关波动)+μ⋅CalibLoss.\mathcal{L} = \sum_h \Big(\underbrace{\text{ListwiseLoss}_h}\_{\text{提升相关均值}} + \lambda\_h\underbrace{\text{VarPerDay}(ρ\_{s,h})}\_{\text{降低相关波动}}\Big) + \mu\cdot \text{CalibLoss}.$$
- **CalibLoss**：对真实连续收益给一层轻量回归（Huber），权重很小，仅帮助分数尺度与标签数量级对齐，防止“极端排行但数值无意义”。


---

# 四、验证与防泄露（与规则强绑定）

- 使用 **Purged K-Fold / CPCV + Embargo** 的**按时间切块**验证：
    
    - **Purge**：去除“测试区间标签构造所覆盖的训练样本”；
        
    - **Embargo**：在训练和测试块边界加缓冲带（覆盖 1–4 天乃至更长，视滞后设定）；
        
    - **CPCV**：组合式多路径回测，获得稳健方差估计，显著优于简单 Walk-Forward。开源实现可用 `skfolio.CombinatorialPurgedCV`。([Agorism](https://agorism.dev/book/finance/ml/Marcos%20Lopez%20de%20Prado%20-%20Advances%20in%20Financial%20Machine%20Learning-Wiley%20%282018%29.pdf?utm_source=chatgpt.com "Advances in Financial Machine Learning"), [skfolio](https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html?utm_source=chatgpt.com "skfolio.model_selection.CombinatorialPurgedCV"), [维基百科](https://en.wikipedia.org/wiki/Purged_cross-validation?utm_source=chatgpt.com "Purged cross-validation"))
        
- **评分对齐**：在 CV 中**逐日**计算 ρ_s（横截面），再计算**均值/标准差**与**加权平均**（若天数不等），保证**线下–线上一致性**。
    
- **滞后对齐**：对 horizon=1..4，严格按官方 lag 定义对齐训练天与标签天（考虑假日错位），相关经验参见社区整理。([DEV Community](https://dev.to/nk_maker/kaggle-diary-mitsuico-commodity-prediction-challenge-day-1-3kl4?utm_source=chatgpt.com "Kaggle Diary: MITSUI&CO. Commodity Prediction Challenge"))
    

---

# 五、特征工程（尽量“已知于当日”）

1. **基础价量/收益谱系**：对数收益、波动率、ATR、成交量变化、盘口/期限点位、跨市场映射（FX→金属等）。
    
2. **去趋势与标准化**：**仅用过去窗内**做 z-score/robust-scale；跨资产横截面**当日去均值/除标准差**，避免“市值/波动主导”的虚假秩。
    
3. **频域/多尺度通道**：STFT/小波/多窗口均线（与 Patch 化天然互补）；近期研究表明频域注意力有益于周期依赖捕捉。([arXiv](https://arxiv.org/html/2407.13806v1?utm_source=chatgpt.com "Revisiting Attention for Multivariate Time Series Forecasting"))
    
4. **缺失与对齐**：保留 **mask/Δtime** 两通道（不要静默填补为0）；节假日/停牌天以**占位+Δtime 增大**传达“久未更新”。([Nature](https://www.nature.com/articles/s41598-018-24271-9?utm_source=chatgpt.com "Recurrent Neural Networks for Multivariate Time Series ..."))
    
5. **横截面派生**：同日的**横截面分位、行业/品类中性化**、主成分（仅用当日横截面）等——**允许在推理日使用**，不会泄露未来，因为这些都是 **t 日“同时可观测”** 的横截面运算。
    
6. **价差目标**：对价差类标签，输入可加入两腿资产的**成分特征拼接 + 相对价量指标**（如价差、相对动量、协方差滑窗）。
    

---

# 六、训练细节（高效与稳健）

- **骨干**：24 层 Encoder，dim 768，heads 12；Patch 长度 P=16/24，步长 S=8/12；窗口 L 依资源 256–1024。
    
- **注意力**：默认 FlashAttention-2；当 _#tokens_ 很大时切换 Performer/FAVOR+。([arXiv](https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"))
    
- **优化器**：AdamW，OneCycle 或 Cosine + warmup（5–10% steps）；dropout/ stochastic depth 适度（0.1–0.2）。
    
- **批组织**：以“天”为最小单元（方便日度 Spearman 与稳定性正则），一个 batch 含若干天。
    
- **预训练**：在训练前做 **Ti-MAE/TS-MAE** 风格的**遮盖重建**（只用历史数据），再微调到排序目标，有利于长窗口表示与缺失鲁棒性。([arXiv](https://arxiv.org/abs/2301.08871?utm_source=chatgpt.com "Ti-MAE: Self-Supervised Masked Time Series Autoencoders"), [OpenReview](https://openreview.net/forum?id=9AuIMiZhkL2&utm_source=chatgpt.com "Ti-MAE: Self-Supervised Masked Time Series Autoencoders"))
    
- **数据增广**：随机窗口抖动、轻微噪声、时间遮盖（Span-mask），避免对特定日期/周度过拟合。
    
- **早停准则**：基于 **CPCV 外层**的“Sharpe-like（均值/波动）”综合指标，而非纯均值。
    

---

# 七、推理与提交后处理（与平台评分一致）

- **逐日推理**：对需要预测的每个交易日，先做**当日横截面标准化**（对预测分数再次去均值/除标准差），再输出；
    
- **秩保真**：提交的是**分数**不是名次，但我们要保证分数的**单调与横截面可比较性**，避免因尺度漂移拉低 Spearman；
    
- **多头融合**：对 1–4 天分别输出；若线上观察到某期波动异常，可调低该期稳定性正则权重重训或做**快照/种子平均**。
    
- **线上稳定器**：对每日分数做**winsorize**（分位剪裁）或**tanh 压缩**，能抑制极值导致的秩抖动（不改变大多数对比关系）。
    
- **注意**：官方评测为“按天 Spearman 的均值/波动”，社区提醒“**Public LB 不代表最终泛化**”，因此我们**不以短期 LB 波动为调参目标**。([DEV Community](https://dev.to/nk_maker/kaggle-diary-mitsuico-commodity-prediction-challenge-day-1-3kl4?utm_source=chatgpt.com "Kaggle Diary: MITSUI&CO. Commodity Prediction Challenge"))
    

---

# 八、消融与指标面板（建议线下看板）

- **主面板**：
    
    - 日度 ρ_s（mean/std）、按品类分组 ρ_s、各 horizon ρ_s；
        
    - 稳定性正则曲线：λ 变化对（mean, std）的权衡；
        
    - 误差分解：强势/弱势分位命中率（Top-k/Bottom-k 命中 Spearman/NDCG）。
        
- **消融**：
    
    1. 去掉跨资产层（只保留单资产编码）；
        
    2. 去掉可微排序，改用回归损失后再排序；
        
    3. 去掉稳定性正则；
        
    4. PatchTST vs iTransformer vs 混合；
        
    5. 频域通道/时间嵌入/缺失通道的贡献。
        
- **预期**：跨资产层与可微排序通常对“横截面 ρ_s 均值”最关键；稳定性正则能显著降“std”，提升“Sharpe-like”。
    

---

# 九、工程落地（目录与复现）

- **数据层**：`DatasetDaily`（按天聚合，返回{tokens, mask, Δt, horizon 标签}）、`LagAligner`（1–4天标签对齐）、`NoLeakScaler`（仅用历史/当日横截面统计）。
    
- **模型层**：`CAMHTBackbone`（Patch/Emb/Time2Vec → Intra Encoder → Cross-Asset Encoder）、`HorizonHeads`（4 头）；
    
- **损失层**：`DiffSortSpearmanLoss`（Blondel/Cuturi/SoftSort 任一实现）、`StabilityRegularizer`；
    
- **验证器**：`CPCVRunner`（外层 CPCV + 内层早停），`DailyMetrics`；
    
- **加速**：FlashAttention-2 kernel 开启，AMP/梯度检查点/ZeRO（按资源）；([arXiv](https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"))
    
- **日志**：逐日 ρ_s 热力图、各组别箱线、训练/验证 Sharpe-like 曲线。
    
- **再现脚本**：`train_camht.py --cfg camht.yaml`，`infer_submit.py`（逐日读入当日可用特征，输出全目标 4 期分数）。
    

---

# 十、资源规划与超参起点

- **显存预算**（参考）：A100-40GB：L=512，P=16，S=8，batch≈（天数×目标）=（8×全体）切片后通过分组注意力可跑满；
    
- **起始超参**：dim=640，layers=16（Intra 10 + Cross 6），heads=10，dropout=0.1；λ_stab=0.2；warmup 8%；Max LR=1.5e-3；权重衰减 0.05；训练轮次 20–30（以外层 CPCV 早停为准）。
    
- **集成**：不同随机种子×2，PatchTST 主干 与 iTransformer 主干各 1 份，共 4 模型做 **分数平均**（保持秩结构最稳）。([arXiv](https://arxiv.org/abs/2211.14730?utm_source=chatgpt.com "A Time Series is Worth 64 Words: Long-term Forecasting ..."))
    

---

# 十一、风险点与对策

- **LB 诱导过拟合**：只盯短期 Public LB 会牺牲最终稳健性 → 以 **CPCV 的 Sharpe-like** 早停与选择；([DEV Community](https://dev.to/nk_maker/kaggle-diary-mitsuico-commodity-prediction-challenge-day-1-3kl4?utm_source=chatgpt.com "Kaggle Diary: MITSUI&CO. Commodity Prediction Challenge"))
    
- **滞后错配/假日**：严格使用官方 lag 定义 + Embargo≥max(lag) 天；([Agorism](https://agorism.dev/book/finance/ml/Marcos%20Lopez%20de%20Prado%20-%20Advances%20in%20Financial%20Machine%20Learning-Wiley%20%282018%29.pdf?utm_source=chatgpt.com "Advances in Financial Machine Learning"))
    
- **可微排序数值不稳**：采用 O(n log n) 的投影法实现（Blondel’20）或温度退火的 SoftSort，并用分段/分组排序减少一次排序规模。([arXiv](https://arxiv.org/abs/2002.08871?utm_source=chatgpt.com "Fast Differentiable Sorting and Ranking"))
    
- **算力瓶颈**：优先 FlashAttention-2；序列过长再切 Performer；必要时减少 Cross-Asset 的 token（分组/Top-k 稀疏）。([arXiv](https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"))
    

---

# 十二、为什么这套方案“贴规则且有上限”

- **目标对齐**：用**可微 Spearman**直攻“横截面秩相关”，并用**稳定性正则**推升“均值/波动”的比值——这与平台度量同构。([arXiv](https://arxiv.org/abs/2002.08871?utm_source=chatgpt.com "Fast Differentiable Sorting and Ranking"))
    
- **结构对齐**：**跨资产注意力**直接学习“相对强弱/共振/替代”，这是 Spearman 的物理来源（相对排序）；Patch 化与 iTransformer 保障**长程与维度相关**同时被学到。([OpenReview](https://openreview.net/forum?id=vSVLM2j9eie&utm_source=chatgpt.com "Crossformer: Transformer Utilizing Cross-Dimension ..."), [arXiv](https://arxiv.org/abs/2211.14730?utm_source=chatgpt.com "A Time Series is Worth 64 Words: Long-term Forecasting ..."))
    
- **验证对齐**：**Purged/CPCV+Embargo** 准确反映“滚动日度评测”，避免信息泄露与过乐观。([Agorism](https://agorism.dev/book/finance/ml/Marcos%20Lopez%20de%20Prado%20-%20Advances%20in%20Financial%20Machine%20Learning-Wiley%20%282018%29.pdf?utm_source=chatgpt.com "Advances in Financial Machine Learning"))
