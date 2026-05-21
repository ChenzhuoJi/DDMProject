# 05 — 图像任务整体 Pipeline

> CIFAR-10 图像生成在 CTMC 框架中的完整数据流
>
> 从原始像素 → 前向 CTMC → 训练 → 反向采样 → 评估

## 目录

- [1. Pipeline 全景图](#1-pipeline-全景图)
- [2. 数据层：从像素到离散状态](#2-数据层从像素到离散状态)
- [3. 前向 CTMC：连续时间扩散过程](#3-前向-ctmc连续时间扩散过程)
- [4. 去噪网络：UNet + 截断逻辑分布](#4-去噪网络unet--截断逻辑分布)
- [5. 损失函数：CT-ELBO 的计算路径](#5-损失函数ct-elbo-的计算路径)
- [6. 训练循环：组件的组装与协同](#6-训练循环组件的组装与协同)
- [7. 反向采样：Tau-Leaping 与 PC](#7-反向采样tau-leaping-与-pc)
- [8. 评估管线：ELBO 与 FID](#8-评估管线elbo-与-fid)
- [9. 端到端数据流总结](#9-端到端数据流总结)

---

## 1. Pipeline 全景图

整个图像任务由三条并行的执行路径构成，共享同一组训练好的模型参数：

```
┌────────────────────────────────────────────────────────────────────┐
│                        配置层 (config/train/cifar10.py)            │
│  ml_collections.ConfigDict 驱动所有组件的行为                     │
└────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│  路径 A: 训练        │   │  路径 B: 采样        │   │  路径 C: 评估        │
│                     │   │                     │   │                     │
│  train.py           │   │  scripts/sample.py  │   │  elbo_evaluation.py │
│                     │   │                     │   │                     │
│  DiscreteCIFAR10    │   │  PCTauLeaping        │   │  DiscreteCIFAR10    │
│  → CT-ELBO          │   │  → 50k PNG          │   │  → CT-ELBO          │
│  → UNet params      │   │  → FID 计算         │   │  → bits/dim        │
│  → EMA shadow       │   │                     │   │                     │
│  → Checkpoint       │   │                     │   │                     │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
```

三条路径共享的核心组件：

| 组件 | 训练 | 采样 | 评估 |
|------|------|------|------|
| `GaussianTargetRate` (rate + transition) | 计算 $R_t$, $q_{t\|0}$ | 计算 $R_t$, $q_{t\|0}$ | 计算 $R_t$, $q_{t\|0}$ |
| `ImageX0PredBase` (UNet + logistic) | 梯度更新 | 推理 (eval mode) | 推理 (eval mode) |
| `EMA` | 更新 shadow | 加载 shadow | 加载 shadow |
| `GenericAux` (CT-ELBO) | 前向传播 | — | 前向传播 |

---

## 2. 数据层：从像素到离散状态

### 2.1 CIFAR-10 原始数据

```
原始格式: (N, 32, 32, 3), uint8, range [0, 255]
Pytorch 加载后: torchvision.datasets.CIFAR10
```

### 2.2 DiscreteCIFAR10 预处理

`lib/datasets/datasets.py:11-51`:

```python
class DiscreteCIFAR10(CIFAR10):
    def __init__(self, cfg, device):
        # 1. 从 numpy 加载
        self.data = from_numpy(self.data)    # (N, 32, 32, 3)
        # 2. 转换为 PyTorch 通道优先格式
        self.data = self.data.transpose(1, 3)  # (N, 3, 32, 32)
        self.data = self.data.transpose(2, 3)  # (N, 3, 32, 32) [确认不变]
        # 3. 预加载到 GPU
        self.data = self.data.to(device).view(-1, 3, 32, 32)
        # 4. 随机水平翻转 (训练)
        self.flip = RandomHorizontalFlip()
```

关键设计：

1. **状态空间 S = 256**: 每个像素保留原始 256 级离散值（对应 RGB 0-255），不做二值化或量化。这使得模型必须学习完整的色彩分布。

2. **数据预加载到 GPU**: 整个训练集（50k × 3 × 32 × 32 ≈ 150MB）一次性放在 GPU 显存中，消除训练时 CPU-GPU 传输瓶颈。

3. **DataLoader 直接返回离散像素值**: `__getitem__` 直接返回 `img` 张量，不返回标签。模型做的是无条件生成，不需要类别信息。

4. **数据在模型中始终作为离散索引使用**: 像素值 `[0, 255]` 在模型中作为 `q_{t|0}` 的条件行索引、以及在采样时作为 `x_t` 的类别使用。

### 2.3 数据流图示

```
磁盘上的 CIFAR-10                   训练中的 minibatch         模型中
  (uint8, HWC)        DiscreteCIFAR10         DataLoader           前向
                                │                    │
  50000×32×32×3 ──────────────→│  (N,3,32,32)        │
                                │  GPU Tensor         │
                                │  uint8 → float      │
                                │  (隐式, torch 自动)  │
                                │                    │
                                │ 每步 128 张 → DataLoader → model(B,3072)
```

---

## 3. 前向 CTMC：连续时间扩散过程

### 3.1 核心思想

CTMC (Continuous-Time Markov Chain) 将图像生成建模为状态空间 `{0, ..., 255}` 上的连续时间马尔可夫链。前向过程从干净图像 $x_0$ 开始，按照速率矩阵 $R_t$ 不断跳跃，到 $t=1$ 时达到噪声分布 $q_1$。

### 3.2 GaussianTargetRate 的数学构造

`lib/models/models.py:192-267`:

```
时间调度:   β(t) = time_base * log(time_exponential) * time_exponential^t
            ∫β(s)ds = time_base * (time_exponential^t - 1)

基速率矩阵: R_b ∈ ℝ^{256×256}
            - 高斯偏好: 只有"数值相近"的状态间有显著转移率
            - 细节平衡: 满足平稳分布 N(S/2, Q_sigma²)

实际速率:   R_t = β(t) · R_b
转移矩阵:   q_{t|0} = exp(∫β(s)ds · Λ)   其中 Λ = Q^{-1} R_b Q
```

配置参数:

| 参数 | 值 | 含义 |
|------|-----|------|
| `rate_sigma = 6.0` | 基速率矩阵的宽度 | 决定哪些状态间可以跳转 |
| `Q_sigma = 512.0` | 平稳分布宽度 | 最终噪声分布接近均匀但带高斯偏好 |
| `time_exponential = 100.0` | 速率增长指数 | 速率随时间指数增长 |
| `time_base = 3.0` | 速率基准 | 控制整体速率量级 |

### 3.3 R_t 与 q_{t|0} 的作用

这两个对象是整个代码中**最频繁使用的计算**，出现在损失函数和采样器中：

```
模型推理一次 → 调一次 model.forward(x, t)
损失计算一步 → 调 model.transition(t) + model.rate(t) + model.forward(x, t)
采样一步     → 调 model.transition(t) + model.rate(t) + model.forward(x, t)
```

- `R_t(i, j)` = 在时刻 $t$ 从状态 $i$ 跳转到 $j$ 的速率
- `q_{t|0}(j | i)` = 给定初始状态 $i$，在时刻 $t$ 处于状态 $j$ 的概率

两者通过矩阵指数相关：$q_{t|0} = \exp(\int_0^t R_s ds)$。

---

## 4. 去噪网络：UNet + 截断逻辑分布

### 4.1 整体功能

去噪网络（UNet）的核心任务是预测 $p_{0|t}^\theta(x_0 | x_t)$，即给定被噪声污染的 $x_t$，预测原始干净像素 $x_0$ 在 256 个值上的分布。

### 4.2 数据格式转换链

```
输入: (B, 3072)                     — 展平的离散像素索引 [0, 255]
  │  view(B, 3, 32, 32)
  ▼
(B, 3, 32, 32)                      — 重塑为图像
  │  UNet.forward
  ▼
(B, 6, 32, 32)                      — 原始网络输出
  │  前 3 通道 = μ, 后 3 通道 = log_scale
  ▼
截断逻辑分布计算                     — 将连续分布离散化为 256 bins
  │
  ▼
(B, 3072, 256)                      — 每个像素 256 个 logits
```

### 4.3 截断逻辑分布详解

`lib/models/models.py:47-90`:

这种方法的关键动机：**CNN 天然适合输出连续的图像特征**（如边缘、纹理），但扩散过程需要离散类别概率。截断逻辑分布提供了一条从连续到离散的平滑桥梁。

```python
# 核心步骤
mu = net_out[:, 0:3]               # (B, 3, 32, 32) 均值
log_scale = net_out[:, 3:6]         # (B, 3, 32, 32) 对数尺度
inv_scale = exp(-(log_scale - 2))   # 尺度倒数

# 在 [-1, 1] 上放置 256 个等宽 bin
bin_width = 2.0 / 256
bin_centers = linspace(-1 + bin_width/2, 1 - bin_width/2, 256)

# 计算每个 bin 的 logistic CDF 差值
logits = log(Φ(right) - Φ(left))    # (B, 3, 32, 32, 256)
```

**为什么用逻辑分布而非正态分布**：逻辑分布的尾部更重，能更好地处理离散化边界上的概率质量。这在图像建模中是已知的优点（PixelCNN++ 也使用 logistic 分布）。

### 4.4 与 DDPM/score-SDE 的核心差异

| | DDPM | CTMC (本代码) |
|--|------|---------------|
| 网络输出 | 预测噪声 $\epsilon \in \mathbb{R}^{3\times 32\times 32}$ | 预测连续分布参数 $\mu,\log\_scale \in \mathbb{R}^{6\times 32\times 32}$ |
| 损失 | MSE($\epsilon$, $\hat{\epsilon}$) | CT-ELBO + 辅助交叉熵 |
| 数据空间 | 连续像素 [-1, 1] | 离散像素 {0, ..., 255} |
| 物理含义 | 直接预测噪声 | 预测 $p_{0\|t}(x_0\|x_t)$ |

---

## 5. 损失函数：CT-ELBO 的计算路径

### 5.1 数学形式

`lib/losses/losses.py:11-246`:

论文推导的连续时间 ELBO:

```
L_CT(θ) = E[ Reg项 + Sig项 ] + weight · NLL辅助项

Reg项 = Σ_{x'≠x_t} R̂_θ_t(x_t, x')          ← 正则化，防止反向速率过大
Sig项 = -Z(x_t) · log R̂_θ_t(x̃, x_t)        ← 主要学习信号
NLL   = CrossEntropy(p_θ(x_0|x_t), x_0)     ← 辅助监督
```

### 5.2 单步前向技巧 (One Forward Pass)

`loss.one_forward_pass = True` (对应论文 §C.4):

```
常规做法 (两次前向):                         单次前向 (本代码):
  logits_reg = model(x_t, t)       ──→      logits = model(x_tilde, t)
  logits_sig = model(x_tilde, t)            p0t_reg = softmax(logits)  # 用于 reg 项
                                            p0t_sig = softmax(logits)  # 用于 sig 项
```

将 `x_tilde` 而非 `x_t` 作为 reg 项的输入。这牺牲了少量 ELBO 紧度（因为 reg 项本应在 $x_t$ 上计算），但节省了一半计算量。

### 5.3 每个 minibatch 的计算流程

```
输入: minibatch (B, 3, 32, 32), 像素值 [0, 255]
         │
         │ view(B, 3072)
         ▼
    ┌─────────────────────────────────────┐
    │ ① 采样时间 t ~ U(0.01, 1.0)         │
    │    ts = rand(B) * 0.99 + 0.01       │
    └─────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │ ② 计算 q_{t|0} 和 R_t               │
    │    qt0 = model.transition(ts)       │  ← (B, 256, 256)
    │    rate = model.rate(ts)            │  ← (B, 256, 256)
    └─────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │ ③ 采样 x_t ~ q_{t|0}(·|x_0)        │
    │    每像素独立的类别采样              │  ← (B, 3072)
    └─────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │ ④ 采样 x_tilde (单维度跳跃)         │
    │    选维度: P(d) ∝ Σ R_t(x_t^d, s)   │
    │    选新值: P(s) ∝ R_t(x_t^d, s)     │  ← (B, 3072)
    └─────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │ ⑤ 模型前向                          │
    │    logits = model(x_tilde, ts)      │  ← (B, 3072, 256)
    │    p0t = softmax(logits, dim=2)     │  ← p_θ(x_0|x_t)
    └─────────────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌──────────┐ ┌──────────┐
│ ⑥ Reg项   │ │ ⑦ Sig项  │
│ 计算       │ │ 计算       │
│ R̂_t总和   │ │ Z·log R̂_t │
└──────────┘ └──────────┘
    │         │
    └────┬────┘
         ▼
    ┌─────────────────────────────────────┐
    │ ⑧ 辅助 NLL                          │
    │ cross_entropy(logits.permute, x_0)  │
    └─────────────────────────────────────┘
         │
         ▼
    return neg_elbo + 0.001 * nll
```

### 5.4 x_t 和 x_tilde 采样的物理意义

为什么需要 $x_t$ 和 $\tilde x_t$ 这一对状态？

```
前向 CTMC 中的一次跳跃:
  x_0 ──t──→ x_t ──h──→ x_tilde
              ↑            ↑
          采样自       从 x_t 出发
          q_{t|0}     按 R_t 跳一次

反向过程需要学习:
  给定 (x_tilde, t), 预测应该跳回 x_t 的速率
  → 模型预测 p_{0|t}(x_0 | x_tilde)
  → 通过公式转换为反向速率 R̂_t(x_tilde → x_t)
```

### 5.5 辅助 NLL 的作用

```python
perm_x_logits = logits.permute(0, 2, 1)  # (B, S, D) → CrossEntropy 需要
nll = cross_entropy(perm_x_logits, minibatch.long())
```

这是一项直接监督：要求模型在给定 $x_t$ 时准确预测原始 $x_0$，与 CT-ELBO 框架正交。设 `weight = 0.001` 使其在早期训练中稳定过程，不至于干扰 ELBO 梯度方向。

---

## 6. 训练循环：组件的组装与协同

### 6.1 组件创建序列

`train.py:95-118`:

```
main(cfg)
  │
  ├── bookkeeping.create_experiment_folder()     — 创建 save_dir / checkpoint_dir / config_dir
  ├── bookkeeping.setup_tensorboard()            — TensorBoard writer
  ├── model_utils.create_model(cfg, device)      — GaussianTargetRateImageX0PredEMA
  │     ├── ImageX0PredBase.__init__             — 创建 UNet (740 万参数)
  │     ├── GaussianTargetRate.__init__          — 预计算 R_b 的特征分解
  │     └── EMA.__init__ + init_ema()            — 初始化 shadow params
  │
  ├── dataset_utils.get_dataset(cfg, device)     — DiscreteCIFAR10 (预加载到 GPU)
  ├── DataLoader(dataset, batch_size=128)
  │
  ├── losses_utils.get_loss(cfg)                 — GenericAux
  ├── training_utils.get_train_step(cfg)         — Standard
  └── optimizers_utils.get_optimizer(...)        — Adam(lr=2e-4)
```

### 6.2 单步训练展开

`lib/training/training.py:13-39`:

```python
Standard.step(state, minibatch, loss, writer):
    状态: state = {'model': model, 'optimizer': optimizer, 'n_iter': k}

    ① optimizer.zero_grad()
    ② l = loss.calc_loss(minibatch, state, writer)  ← 计算 CT-ELBO
    ③ assert not isnan(l)
    ④ l.backward()                                    ← 反向传播
    ⑤ clip_grad_norm_(params, 1.0)                    ← 梯度裁剪
    ⑥ 线性 warmup: lr = 2e-4 * min(k / 5000, 1.0)   ← LR 调度
    ⑦ optimizer.step()                                ← 参数更新
    ⑧ model.update_ema()                               ← EMA 更新
    ⑨ writer.add_scalar('loss', l)                    ← 日志
```

### 6.3 训练状态演化

```
迭代 0        →  模型参数 (随机初始化) →  EMA shadow (copy)
迭代 1        →    ↓ optimizer.step()     ↓ update_ema()
迭代 2        →    ↓                      ↓
...
迭代 2,000,000 →   训练完成

优化器状态:
  - Adam 一阶/二阶动量: 正常维护
  - lr: 前 5000 步从 0 线性升到 2e-4，之后保持

EMA 状态:
  - decay = 0.9999, 实际随迭代调整: decay = min(0.9999, (1+k)/(10+k))
  - 评估/采样时: 加载 EMA shadow → 模型参数 (更平滑、更稳健)
```

### 6.4 模型保存与恢复

```python
# 每 1000 步保存 (checkpoint_freq = 1000)
bookkeeping.save_checkpoint(checkpoint_dir, state, 2)
  └─ state 包含: model.state_dict(), optimizer.state_dict(), n_iter

# 恢复训练 (抢占恢复)
bookkeeping.check_for_preempted_run(...)
  └─ 检测 save_dir 下最近的 checkpoint
  └─ 恢复 state → 继续训练

# 低频率日志 (log_low_freq = 10000)
logger = denoisingImages(state, cfg, writer, minibatch, dataset)
  └─ 在几个时间点 t = [0.01, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0] 可视化去噪过程
  └─ 对 3 张验证图像: 采样 x_t → 模型预测 x_0 → 可视化 (x_t, x_0_pred)
  └─ 写入 TensorBoard
```

---

## 7. 反向采样：Tau-Leaping 与 PC

### 7.1 采样原理

训练完成后，EMA 参数为 $p_{0|t}^\theta$ 提供了准确的估计。反向采样从 $t=1$ 的噪声 $x_1$ 开始，沿着反向 CTMC 逐步生成 $x_0$。

### 7.2 Tau-Leaping 采样

`lib/sampling/sampling.py:30-120` (`config/eval/cifar10.py` 默认使用 PCTauLeaping):

```python
x = get_initial_samples(N=50, D=3072, device, S=256,
    initial_dist='gaussian', std=512.0)
    # 从高斯偏好分布采样初始状态 (接近均匀，但中心值略高)

ts = linspace(1.0, 0.01, 500)  # 500 步从 t=1 到 t=0.01

for t in ts[0:-1]:
    h = t_prev - t                                     # 步长
    qt0 = model.transition(t)                          # (N, 256, 256)
    rate = model.rate(t)                                # (N, 256, 256)
    p0t = softmax(model(x, t), dim=2)                   # (N, 3072, 256)

    # 计算近似反向速率 R̂_t
    forward_rates = R_t(·, x)                           # (N, 3072, 256)
    inner_sum = (p0t / q_{t|0}(·|x)) @ q_{t|0}         # (N, 3072, 256)
    reverse_rates = forward_rates * inner_sum           # (N, 3072, 256)
    reverse_rates[:, :, x] = 0                          # 禁止自己跳自己

    # Poisson 跳跃 (Tau-Leaping)
    jump_nums ~ Poisson(reverse_rates * h)              # (N, 3072, 256)
    total_jumps = Σ jump_nums * (target_state - x)      # (N, 3072)
    x_new = clamp(x + total_jumps, 0, 255)

# 最后一步 t=0.01: 直接 argmax
p0_final = softmax(model(x, 0.01), dim=2)
x_final = argmax(p0_final, dim=2)                      # (N, 3072)
```

关键洞察：**Tau-Leaping 用 Poisson 分布近似连续时间过程**。在时间步长 $h$ 内，每个像素 $d$ 的目标状态 $s$ 的跳跃次数服从 $\text{Poisson}(\hat R_t(x^d, s) \cdot h)$。所有跳跃的效果叠加得到新状态。

### 7.3 Predictor-Corrector (PC) 采样

`lib/sampling/sampling.py:122-240`:

PC 在 Tau-Leaping 的"预测"步之后添加"校正"步：

```python
for t in ts:
    # === 预测步 (Predictor) ===
    reverse_rates = compute_reverse_rates(x, t)
    x = take_poisson_step(x, reverse_rates, h)

    # === 校正步 (Corrector) [仅在 t ≤ 0.1 时生效] ===
    if t <= corrector_entry_time:
        for _ in range(10):  # num_corrector_steps = 10
            _, reverse_rates, _ = get_rates(x, t-h)
            transpose_rates = R_t(x, ·)                # 前向速率的转置
            corrector_rate = transpose_rates + reverse_rates
            x = take_poisson_step(x, corrector_rate,
                corrector_step_size_multiplier * h)    # 步长 ×1.5
```

配置对比:

| 参数 | TauLeaping | PCTauLeaping |
|------|-----------|--------------|
| `num_steps` | 500 | 500 |
| `corrector_steps` | 0 | 10 |
| `entry_time` | — | 0.1 |
| `step_multiplier` | — | 1.5 |
| `initial_dist` | gaussian | gaussian |

PC 在小时间区域 (t < 0.1) 的每个 Tau 步后，额外运行 10 步校正器。校正器使用 **前向 + 反向速率的和** 作为速率，这对应 Langevin 动力学的离散模拟。校正步的存在显著提升了采样质量。

### 7.4 从采样到图像

`scripts/sample.py:47-61`:

```
sampler.sample(model, batch=50, num_intermediates=1)
  → samples: (50, 3072), dtype=int, range [0, 255]
  → reshape(50, 3, 32, 32)
  → astype(uint8)
  → Image.fromarray(transpose_to_HWC)  # (3,32,32) → (32,32,3)
  → save as PNG (50,000 张)
```

---

## 8. 评估管线：ELBO 与 FID

### 8.1 ELBO 评估路径

`elbo_evaluation.py:27-71`:

```python
main(eval_cfg):
    # 1. 加载训练配置 (config.yaml)
    train_cfg = load_ml_collections(config_dir)
    for override in eval_cfg.train_config_overrides:
        set_in_nested_dict(train_cfg, override)

    # 2. 重建模型并加载 EMA 权重
    model = create_model(train_cfg, device)
    model.load_state_dict(torch.load(checkpoint)['model'])
    model.eval()  # → 自动切换到 EMA shadow params

    # 3. 运行 ELBO Logger
    logger = ELBO(state={'model': model}, ...)
      └─ 遍历数据集前 total_B 个样本
      └─ 每个样本重复 total_N 次蒙特卡洛采样
      └─ 计算 -ELBO (与训练相同，但使用 EMA + 无梯度)
      └─ 输出 bits/dim = -ELBO / (3*32*32)
```

`config/eval/cifar10_elbo.py` 指定评估细节（如采样次数、batch 大小）。

### 8.2 FID 评估路径

通过 `scripts/sample.py` 生成 50,000 张图像后，使用标准工具 (clean-fid, pytorch-fid) 计算与真实 CIFAR-10 数据的 Frechet Inception Distance (FID)。

### 8.3 两种评估的差异

| | ELBO | FID |
|--|------|-----|
| 度量 | 似然 (likelihood) | 分布距离 |
| 实现位置 | `elbo_evaluation.py` | 外部工具 |
| 需要数据 | 需要测试集图像 | 需要测试集统计量 |
| 计算量 | 适中 (几百次前向) | 大 (50k 生成 + Inception) |
| 意义 | 模型对数据的拟合度 | 生成样本的视觉质量 |

---

## 9. 端到端数据流总结

### 9.1 训练时数据流

```
训练配置 (ml_collections)
  ↓
train.py main()
  ├── model_utils.create_model()    → UNet (随机初始化)
  ├── dataset_utils.get_dataset()   → DiscreteCIFAR10 on GPU
  ├── losses_utils.get_loss()       → GenericAux (CT-ELBO)
  ├── training_utils.get_train_step() → Standard
  └── optimizers_utils.get_optimizer() → Adam(2e-4)
        ↓
训练循环 (2,000,000 步):
  for minibatch in DataLoader:           # 每步取出 (128, 3, 32, 32) 离散像素
    loss.calc_loss(minibatch, state)      # 计算 CT-ELBO → 标量
      ├── model.transition(t)             # q_{t|0} 矩阵
      ├── model.rate(t)                   # R_t 矩阵
      ├── model.forward(x_tilde, t)       # UNet 前向 → 截断逻辑 → logits
      └── reg + sig + aux_nll             # 三项求和
    l.backward()                           # 梯度
    optimizer.step()                       # Adam 更新
    model.update_ema()                     # EMA shadow

  loggers: denoisingImages                # 每 10k 步 TensorBoard 可视化
    ├── 加噪: x_t = sample_from_q_{t|0}(x_0, t)
    ├── 预测: x_0_pred = argmax(model(x_t, t))
    └── 展示: 3张图 × 7个时间点 × 2行 (x_t, x_0_pred)
```

### 9.2 采样时数据流

```
采样配置 (ml_collections)
  ↓
scripts/sample.py:
  ├── model = create_model(train_cfg)      # 重建模型
  ├── model.load_state_dict(checkpoint)     # 加载训练好的权重
  └── model.eval()                          # 切换到 EMA 参数 (关键!)
        ↓
采样 (50,000 张, batch=50):
  sampler = PCTauLeaping(cfg)
  sampler.sample(model, batch, 1):
    x = get_initial_samples()              # 从高斯偏好分布采样 (50, 3072)
    for t in ts:                           # 500 步从 1.0 → 0.01
      p0t = model(x, t)                    # UNet 推理 → logits → softmax
      reverse_rates = p0t @ q_{t|0}        # 构建反向速率 (核心公式)
      x = tau_leap(x, reverse_rates)        # Poisson 近似跳跃
      if t < 0.1:                          # 校正步
        x = correct(x, reverse_rates)      # Langevin 风格校正
    return x_final = argmax(model(x, 0.01)) # 最终预测
        ↓
  后处理:
    reshape(50, 3, 32, 32) → uint8 → PNG
```

### 9.3 评估时数据流

```
评估配置 (ml_collections)
  ↓
elbo_evaluation.py:
  ├── model = load_model(checkpoint, EMA=True)
  ├── dataset = DiscreteCIFAR10(test_set)
  └── logger = ELBO(model, dataset)
        ↓
  每个测试样本:
    for _ in range(MC_repeats):           # 多次蒙特卡洛
      x_t ~ q_{t|0}(·|x_0)                 # 加噪
      x_tilde = jump(x_t, R_t)             # 单步跳跃
      logits = model(x_tilde, t)           # 预测
      compute CT-ELBO bits/dim             # 累积
```

### 9.4 关键决策点总结

| 决策 | 选择 | 位置 | 理由 |
|------|------|------|------|
| 状态空间大小 | S=256 | 配置 | 保留完整像素精度，避免量化损失 |
| 速率矩阵 | GaussianTargetRate | 配置 | 可交换性保证 $q_{t\|0}$ 解析解 |
| 网络架构 | UNet + 截断逻辑 | 配置 | 适合图像空间结构的归纳偏置 |
| 损失函数 | CT-ELBO + 辅助NLL | 配置 | 连续时间 ELBO + 稳定训练 |
| 训练步 | Standard + EMA | 配置 | 简单有效 |
| 采样算法 | PCTauLeaping | 配置 | PC 校正提升质量 |
| EMA 切换 | `model.eval()` 自动完成 | `EMA.train()` | 训练/评估无缝切换 |
| 一次前向 | `one_forward_pass=True` | 配置 | 节省 50% 计算量 |
