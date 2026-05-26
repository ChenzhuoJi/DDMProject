# MDLM 实验流程与可复现成果分析

> 论文：Simple and Effective Masked Diffusion Language Models (NeurIPS 2024)
> 代码：https://github.com/kuleshov-group/mdlm
> 项目官网：https://s-sahoo.com/mdlm/

---

## 1. 论文核心贡献

1. **SUBS 参数化**：将吸收态扩散的 reverse process 简化为两个核心操作（Zero Masking + Carry-Over Unmasking），使得 NELBO 退化为加权的 MLM loss
2. **连续时间 ELBO**：推导了连续时间下的简化 ELBO，避免了 CTMC 理论
3. **高效采样器**：`ddpm_cache` 采样器利用 SUBS 的 time-independent 特性缓存 denoising network 输出，加速 3-4x
4. **SAR 生成**：半自回归解码，支持变长文本生成

---

## 2. 代码结构全景

```
src/
├── main.py                 # 入口：train / ppl_eval / sample_eval
├── diffusion.py            # 核心：Diffusion LightningModule
│   ├── _loss()             # 训练损失计算
│   ├── _sample()           # 采样主循环 (t=1 → t=0)
│   ├── _ddpm_update()      # 标准 DDPM 去噪一步
│   ├── _ddpm_caching_update() # 带缓存的加速去噪
│   ├── sample_subs_guidance() # SAR 生成（已含中间状态收集）
│   └── compute_generative_perplexity() # 用 GPT-2 评估生成质量
├── noise_schedule.py       # 噪声调度：loglinear / linear / polynomial
├── dataloader.py           # 数据加载：支持 OWT / LM1B / text8 / PTB 等
├── models/
│   ├── dit.py              # DiT 骨干网络（主架构）
│   ├── dimamba.py          # Mamba SSM 骨干
│   └── autoregressive.py   # AR 基线
├── configs/
│   ├── config.yaml         # 主配置（默认 small + OWT + subs）
│   ├── model/              # tiny / small / medium 模型尺寸
│   ├── data/               # 各数据集配置
│   └── noise/              # 噪声调度配置
└── scripts/                # Slurm 训练/评估脚本
```

---

## 3. 作者实验流程

### 3.1 训练流程

```
数据准备 → 模型初始化 → 前向扩散(加噪) → 网络预测 → 计算简化ELBO → 反向传播
```

**一步训练细节** (`_forward_pass_diffusion` in `diffusion.py:847`)：

```
1. 采样时间 t ~ Uniform(eps, 1)          #  antithetic sampling 降低方差
2. 计算噪声参数: sigma, dsigma = noise(t)
3. 前向加噪:  xt = q_xt(x0, move_chance)  # 以概率 move_chance 将 token 替换为 [MASK]
4. 网络预测:  model_output = backbone(xt, sigma)
5. SUBS 参数化: 将 unmasked token logits 强制为 one-hot，[MASK] logit 设为 -inf
6. 损失计算:  loss = -log_p_theta * (dsigma / expm1(sigma))
   → 等价于对 masked token 加权的交叉熵损失（MLM loss 的加权平均）
```

### 3.2 采样流程

```
全 [MASK] → 迭代去噪 (t=1 → t=0) → 最终文本
```

**`_sample()` 方法 (`diffusion.py:658`)**：

```
1. x = [MASK] × sequence_length          # 初始化为全掩码
2. timesteps = linspace(1, eps, num_steps)  # 离散化时间步
3. for i in range(num_steps):
     t = timesteps[i]
     x = ddpm_update(x, t, dt)            # 去噪一步
4. noise_removal:  x = argmax(backbone(x, sigma))  # 最终去噪
5. return x                               # 输出 token IDs
```

**`ddpm_caching_update` 的加速原理**：当去噪一步后 token 序列没有变化时，直接复用之前缓存的 `p_x0`（模型预测的 clean x 分布），跳过模型的 forward pass。只有在 token 变化时才重新调用网络。当 `time_conditioning=False` 时，模型输出只依赖 token 状态，不依赖时间，因此缓存有效。

### 3.3 评估流程

**Perplexity 评估** (`_ppl_eval`)：
```
1. 加载 checkpoint
2. 遍历 validation set
3. 对每个 batch: 前向扩散 → 计算 NLL → 累加
4. PPL = exp(mean NLL)
```

**生成评估** (`generate_samples`)：
```
1. 加载 checkpoint
2. 采样生成文本
3. 用 GPT-2 Large 计算 generative perplexity（评估生成质量的指标）
```

---

## 4. 作者论文中报告的实验结果

### 4.1 语言建模 Perplexity

| 实验 | 数据集 | 模型 | 训练 tokens | PPL | 可复现？ |
|------|--------|------|-------------|-----|:-------:|
| Table 1 | LM1B | MDLM | 33B | ≤27.04 | ✅ 需多卡训练 |
| Table 1 | LM1B | MDLM | 327B | ≤23.00 | ❌ 10M steps |
| Table 2 | OWT | MDLM | 262B | ≤23.21 | ✅ 需多卡训练 |
| Table 8 | LM1B | MDLM (continuous) | 33B | 27.04±.01 | ✅ |
| Table 8 | LM1B | w/o continuous time | 33B | 27.19±.07 | ✅ |
| Table 8 | LM1B | & w/o carry-over | 33B | 28.56±.15 | ✅ |
| Table 8 | LM1B | & w/o zero masking | 33B | 28.51±.15 | ✅ |

### 4.2 Zero-shot 跨数据集评估

| 数据集 | AR | SEDD | MDLM |
|--------|:--:|:----:|:----:|
| PTB | 82.05 | 100.09 | **95.26** |
| Wikitext | 25.75 | 34.28 | **32.83** |
| LM1B | 51.25 | 68.20 | **67.01** |
| Lambada | 51.28 | 49.86 | **47.52** |
| AG News | 52.09 | 62.09 | **61.15** |
| Pubmed | 49.01 | 44.53 | **41.89** |
| Arxiv | 41.73 | 38.48 | **37.37** |

### 4.3 采样速度对比

| 采样器 | T=5000 | T=10000 |
|--------|:------:|:-------:|
| SEDD | 127.1s | 229.3s |
| MDLM + ddpm | 113.8s | 206.6s |
| MDLM + ddpm_cache | **40.1s** | **60.4s** |

### 4.4 SAR 生成（表5）

| 模型 | Gen. PPL | Sec/Seq |
|------|:--------:|:-------:|
| SSD-LM | 35.43 | 2473.9 |
| MDLM | **27.18** | **89.3** |

---

## 5. 可展现的实验成果（按复现难度排序）

### 5.1 成果一：扩散过程可视化 ★☆☆☆☆（1-2 小时）

**核心**：展示 [MASK] 逐步替换为真实 token 的去噪全过程。

**方法**：修改 `_sample()` 在 `diffusion.py:658`，在采样循环的特定时间点保存 `x` 快照：

| 时间 t | 去噪进度 | 预期视觉效果 |
|:------:|:--------:|-------------|
| 1.00 | 初始 | 全 [MASK] |
| 0.80 | 少数高频词浮现 | the, a, of, and 等开始出现 |
| 0.50 | 约一半 token 已确定 | 实词开始出现，句子骨架可见 |
| 0.20 | 大部分已确定 | 句子基本成型 |
| 0.05 | 微调阶段 | 少数实词从模糊到精确 |
| 0.00 | 最终 | 完整可读句子 |

**可输出的展现形式**：

```
Step 0   (t=1.00): [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]
Step 100 (t=0.90): [MASK] [MASK] the [MASK] [MASK] [MASK] of [MASK] [MASK] [MASK]
Step 300 (t=0.70): [MASK] is the [MASK] [MASK] [MASK] of [MASK] language [MASK]
Step 500 (t=0.50): This is the [MASK] [MASK] model of natural language [MASK]
Step 800 (t=0.20): This is the power of a language model for natural language
Step 1000(t=0.00): This is the power of a language model for natural language
```

**多维对比展示**：
- 不同 seed → 随机去噪顺序不同 → 展示扩散模型的非自回归特性
- 不同采样器对比（ddpm / ddpm_cache / analytic）→ 速度与质量
- 不同采样步数（128 / 500 / 1000）→ 步数对质量的影响

### 5.2 成果二：HuggingFace 预训练权重推理 ★☆☆☆☆（30 分钟）

直接用作者提供的 HuggingFace 权重跑采样：

```bash
python main.py \
  mode=sample_eval \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  data=openwebtext-split \
  model.length=1024 \
  sampling.predictor=ddpm_cache \
  sampling.steps=1000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=10 \
  backbone=hf_dit
```

**可产出**：
- 10 段模型生成的文本样本
- Generative perplexity 指标（GPT-2 Large 评估）
- 多组 seed 下不同样本展示多样性

### 5.3 成果三：Zero-shot 跨数据集评估 ★★☆☆☆（2-3 小时）

用预训练权重在多个数据集上评估 perplexity：

```bash
python main.py mode=ppl_eval ... data=ptb eval.checkpoint_path=kuleshov-group/mdlm-owt
python main.py mode=ppl_eval ... data=wikitext ...
python main.py mode=ppl_eval ... data=lambada ...
```

**可产出**：作者论文 Table 3 的复现表（7 个数据集的 PPL 对比）

### 5.4 成果四：采样器速度/质量对比 ★★☆☆☆（1-2 小时）

```bash
# 对比三种采样器
sampling.predictor=ddpm_cache  # 最快（作者推荐）
sampling.predictor=ddpm        # 标准
sampling.predictor=analytic    # SEDD 式
```

**可产出**：
- 作者论文 Table 5 的复现（但以你的硬件为准）
- 不同采样步数 T 下的速度-质量 tradeoff 曲线

### 5.5 成果五：text8 小规模训练 ★★★☆☆（30 分钟 - 1 小时）

```bash
python main.py \
  model=tiny \
  data=text8 \
  parameterization=subs \
  model.length=256 \
  trainer.max_steps=50000 \
  loader.batch_size=32
```

**可产出**：
- Training / Validation loss 曲线（ELBO 收敛过程）
- Validation perplexity 指标
- 字符级生成样本（从随机噪声到可读文本的渐进过程）

### 5.6 成果六：Ablation 分析复现 ★★★★☆（需训练多个模型）

复现论文 Table 8 的消融实验：

| 配置 | 预期效果 |
|------|---------|
| MDLM (full) | 最佳 PPL |
| w/o continuous time | PPL 差约 0.15 |
| & w/o carry-over | PPL 差约 1.5 |
| & w/o zero masking | PPL 差约 1.5（与上一条接近） |

**需训练 4 个模型**，但可在 text8 上快速验证趋势而非完整复现数值。

### 5.7 成果七：OpenWebText 全量训练 ★★★★★（需多卡长时间）

```bash
python main.py \
  model=small \
  data=openwebtext-split \
  parameterization=subs \
  model.length=1024 \
  trainer.max_steps=1000000 \
  loader.batch_size=16 \
  loader.eval_batch_size=16
```

**硬件需求与耗时估算**（global batch size = 512）：

| GPU 配置 | 每 GPU batch | 梯度累积 | 预估时间（1M steps） |
|---------|:-----------:|:-------:|:----------------:|
| 1×5090 (32GB) | 8 | ×64 | ~14 天 |
| 1×5090 (32GB) | 16 | ×32 | ~7 天 |
| 4×5090 (NVLink) | 16 | ×8 | ~2 天 |
| 4×5090 (以太网) | 16 | ×8 | ~3-4 天 |

**可产出**：
- 论文主要定量结果（OWT PPL ≤ 23.21）
- 完整的训练 checkpoints
- 验证集 loss 曲线

---

## 6. 推荐实验方案（课程项目版）

基于单卡 5090 的推荐方案：

```
第1步：环境搭建 + flash-attn 编译   (~2h)
第2步：HuggingFace 权重推理采样       (~30min)  → 成果一 + 成果二
第3步：可视化扩散过程                  (~1h)     → 修改采样代码 + 生成逐行展示
第4步：采样器速度对比                  (~1h)     → 成果四
第5步：text8 小规模训练                (~1h)     → 成果五
第6步：跨数据集评估                    (~2h)     → 成果三
                                    ─────────
                                    总计约 7-8 小时工作
```

## 7. 关键技术细节

### 7.1 SUBS 参数化实现 (`_subs_parameterization` at `diffusion.py:261`)

```python
def _subs_parameterization(self, logits, xt):
    # 1) Zero Masking：将 [MASK] token 的 logits 设为 -∞
    logits[:, :, self.mask_index] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    # 2) Carry-Over Unmasking：保留已 unmask 的 token
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits
```

### 7.2 损失计算 (`_forward_pass_diffusion` at `diffusion.py:847`)

```python
# 连续时间 SUBS 损失：
# L = -log p_θ(x|z_t) × (dsigma / expm1(sigma))
#   = -log(模型预测的 clean token 概率) × (噪声导数 / 指数变换)
return - log_p_theta * (dsigma / torch.expm1(sigma))[:, None]
```

### 7.3 Caching 采样 (`_ddpm_caching_update` at `diffusion.py:592`)

```python
# 如果 p_x0 已缓存（即 token 序列自上次更新以来未改变），
# 则跳过 backbone forward pass，直接复用之前的预测结果
if p_x0 is None:
    p_x0 = self.forward(x, sigma_t).exp()  # 新预测
# q(x_s | x_t) 的计算不依赖 backbone，由 cached p_x0 直接得到
q_xs = p_x0 * (move_chance_t - move_chance_s)
q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
```

---

## 8. 关键文件与函数索引

| 功能 | 文件 | 行号 | 函数/类 |
|------|------|:----:|---------|
| 训练入口 | `main.py` | 186 | `main()` |
| 采样入口 | `main.py` | 86 | `generate_samples()` |
| 评估入口 | `main.py` | 119 | `_ppl_eval()` |
| 核心模型 | `diffusion.py` | 68 | `class Diffusion` |
| 训练损失 | `diffusion.py` | 847 | `_forward_pass_diffusion()` |
| SUBS 参数化 | `diffusion.py` | 261 | `_subs_parameterization()` |
| 采样主循环 | `diffusion.py` | 658 | `_sample()` |
| DDPM 更新 | `diffusion.py` | 612 | `_ddpm_update()` |
| 缓存更新 | `diffusion.py` | 592 | `_ddpm_caching_update()` |
| SAR 生成（含中间状态） | `diffusion.py` | 958 | `sample_subs_guidance()` |
| Generative PPL | `diffusion.py` | 515 | `compute_generative_perplexity()` |
| DiT 网络 | `models/dit.py` | — | `class DIT` |
| 噪声调度 | `noise_schedule.py` | — | `get_noise()` |
| 主配置 | `configs/config.yaml` | — | — |
| 模型 small | `configs/model/small.yaml` | — | 12层, 768 hidden, 12头 |
| 模型 tiny | `configs/model/tiny.yaml` | — | 8层, 512 hidden, 8头 |

---

## 9. 与课程 "随机过程" 的联系

| 论文概念 | 课程对应知识点 |
|---------|--------------|
| 前向扩散过程 q(z_t | x) | Markov 链，转移概率矩阵 |
| 连续时间极限 T → ∞ | 连续时间 Markov 链 (CTMC) |
| 反向过程 p_θ(z_s | z_t) | 逆 Markov 链，Bayesian 推理 |
| ELBO 推导 | 变分推断，KL 散度 |
| 噪声调度 α_t | 时齐/非时齐 Markov 链 |
| 吸收态 [MASK] | 吸收态 Markov 链 |
| SUBS 参数化 | 条件概率分解，因子图 |
