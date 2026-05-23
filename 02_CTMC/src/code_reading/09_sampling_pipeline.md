# 09 — 反向采样 Pipeline：Tau-Leaping 与 Predictor-Corrector

> 对应论文 §4.3-§4.4，代码 `lib/sampling/sampling.py:11-240`
>
> 从噪声到图像：反向 CTMC 的数值模拟

## 目录

- [1. 采样的数学基础](#1-采样的数学基础)
- [2. 初始噪声采样](#2-初始噪声采样)
- [3. 反向速率 \(\hat R_t^\theta\) 的计算](#3-反向速率-hat-r_tθ-的计算)
- [4. Tau-Leaping：Poisson 近似跳跃](#4-tau-leapingpoisson-近似跳跃)
- [5. TauLeaping 采样器完整流程](#5-tauleaping-采样器完整流程)
- [6. Predictor-Corrector 扩展](#6-predictor-corrector-扩展)
- [7. 配置系统与采样脚本](#7-配置系统与采样脚本)
- [8. 中间状态可视化](#8-中间状态可视化)

---

## 1. 采样的数学基础

### 1.1 反向 CTMC

训练完成后，模型学会了 $p_{0|t}^\theta(x_0 | x_t)$——给定被污染图像 $x_t$，预测原始干净图像 $x_0$ 在 256 个像素值上的分布。

生成新图像需要**反向运行扩散过程**：从 $t=1$ 的纯噪声 $x_1$ 开始，沿反向 CTMC 逐步去噪，到 $t=0$ 得到干净图像 $x_0$。

反向 CTMC 的速率矩阵 $\hat R_t^\theta$ 由前向速率 $R_t$ 和数据分布隐式决定。论文的关键公式：

$$
\hat R_t^\theta(x, \tilde x) = R_t(\tilde x, x) \cdot \sum_{x_0} \frac{q_{t|0}(\tilde x|x_0)}{q_{t|0}(x|x_0)} \, p_{0|t}^\theta(x_0|x)
$$

其中 $R_t(\tilde x, x)$ 是前向过程中从 $\tilde x$ 跳到 $x$ 的速率，$p_{0|t}^\theta$ 由去噪 UNet 提供。

### 1.2 数值模拟的挑战

CTMC 是连续时间过程，在计算机上只能做离散时间近似。给定时间步长 $h$，如何从 $x_t$ 模拟到 $x_{t-h}$？

**精确做法**：在一个无穷小时间 $h$ 内，每个像素 $d$ 的跳跃次数服从 Poisson 分布。如果 $h$ 足够小，每个像素最多发生一次跳跃；如果 $h$ 较大，可能需要考虑多次跳跃。

**Tau-Leaping** 正是针对此问题的近似算法：它假设在 $h$ 内反向速率 $\hat R_t^\theta$ 近似常数，然后从 Poisson 分布中采样跳跃次数。

---

## 2. 初始噪声采样

`lib/sampling/sampling.py:11-27`:

```python
def get_initial_samples(N, D, device, S, initial_dist, initial_dist_std=None):
    if initial_dist == 'uniform':
        x = torch.randint(low=0, high=S, size=(N, D), device=device)
    elif initial_dist == 'gaussian':
        target = np.exp(
            - ((np.arange(1, S+1) - S//2)**2) / (2 * initial_dist_std**2)
        )
        target = target / np.sum(target)
        cat = torch.distributions.categorical.Categorical(
            torch.from_numpy(target)
        )
        x = cat.sample((N*D,)).view(N, D)
        x = x.to(device)
```

CIFAR-10 使用 `initial_dist = 'gaussian'`，`initial_dist_std = model.Q_sigma = 512.0`。

### 2.1 高斯偏好分布

$$
\pi(k) \propto \exp\left(-\frac{(k - 128)^2}{2 \times 512^2}\right), \quad k = 1, 2, \dots, 256
$$

`initial_dist_std = 512.0` 远大于状态空间 `256`，因此分布接近均匀：

| 像素值 k | 相对概率 $\pi(k)$ |
|---------|-------------------|
| 1 | 0.972 |
| 64 | 0.996 |
| 128 | 1.000 |
| 192 | 0.996 |
| 255 | 0.972 |

**设计动机**：前向过程 $q_1$（$t=1$ 时的噪声分布）不是精确的均匀分布，而是带有轻微高斯偏好的分布。使用相同的高斯偏好采样初始状态，确保 $x_1$ 与 $q_1$ 匹配，减少采样起始阶段的分布偏移。

### 2.2 采样规模

每批生成 50 张图像，每张图像 3072 个像素，因此产生一个 `(50, 3072)` 的张量，每个元素独立从高斯偏好分布中采样。

---

## 3. 反向速率 $\hat R_t^\theta$ 的计算

这是采样中最核心的步骤。代码中的实现涉及三个关键张量和复杂的索引操作。

### 3.1 输入准备

`sampling.py:163-202` (`PCTauLeaping` 的 `get_rates` 辅助函数):

```python
def get_rates(in_x, in_t):
    # 1. 计算 q_{t|0} 和 R_t
    qt0 = model.transition(in_t)    # (N, S, S)
    rate = model.rate(in_t)         # (N, S, S)

    # 2. 模型预测 p_{0|t}
    p0t = F.softmax(model(in_x, in_t), dim=2)  # (N, D, S)

    # 3. 提取 q_{t|0}(·|x) — 给定当前状态 x 的条件概率
    qt0_denom = qt0[
        arange(N).repeat_interleave(D*S),
        arange(S).repeat(N*D),
        in_x.long().flatten().repeat_interleave(S)
    ].view(N, D, S)  # (N, D, S)

    # 4. q_{t|0}(x'|x_0) — 原始过渡矩阵
    qt0_numer = qt0  # (N, S, S)

    # 5. R_t(·, x) — 跳转到当前状态的速率
    forward_rates = rate[
        arange(N).repeat_interleave(D*S),
        arange(S).repeat(N*D),
        in_x.long().flatten().repeat_interleave(S)
    ].view(N, D, S)
```

### 3.2 复杂索引的展开

`qt0_denom` 的索引操作可能是代码中最难理解的部分。展开来看：

```
qt0: (N, S, S)   其中 qt0[b, i, j] = q_{t|0}(j|i)
in_x: (N, D)     当前状态值

我们要提取: 对于每个 batch b, 每个像素 d:
  qt0_denom[b, d, s] = q_{t|0}(s | in_x[b, d])
  即 qt0[b, :, in_x[b,d]] 这个列向量

索引构造:
  arange(N).repeat_interleave(D*S)
    → [0,0,...,0, 1,1,...,1, ..., N-1,...,N-1]
       └─── D*S ───┘

  arange(S).repeat(N*D)
    → [0,1,...,S-1, 0,1,...,S-1, ...]  共 N*D*S 个元素

  in_x.flatten().repeat_interleave(S)
    → [x_0, x_0, ..., x_0, x_1, x_1, ...]  每个 x_d 重复 S 次

所以:
  qt0_denom[k] = qt0[ batch_idx=b,  row=in_x[b,d],  col=s ]
  遍历所有 (b,d,s) 组合的笛卡尔积
```

### 3.3 反向速率的闭式表达式

```python
inner_sum = (p0t / qt0_denom) @ qt0_numer  # (N, D, S)

reverse_rates = forward_rates * inner_sum  # (N, D, S)

reverse_rates[arange(N).repeat(D), arange(D).repeat(N), in_x.flatten()] = 0.0
```

数学上：

$$
\text{inner\_sum}[b,d,s] = \sum_{x_0} \frac{p_{0|t}^\theta(x_0 | x^d)}{q_{t|0}(s | x^d)} \cdot q_{t|0}(s | x_0)
$$

$$
\hat R_t^\theta(x^d, s) = R_t(s, x^d) \cdot \text{inner\_sum}[b,d,s]
$$

最后将**自己跳自己**的对角线置零：$\hat R_t^\theta(x^d, x^d) = 0$。

**关键洞察**：反向速率的计算与训练时损失函数中的计算公式完全一致。区别在于：
- 训练时只需要对**一个选中的像素** $\tilde x$ 计算信号
- 采样时需要对**所有像素的所有目标状态**计算出完整 $S \times S$ 反向速率矩阵

### 3.4 转置前向速率

```python
transpose_forward_rates = rate[
    arange(N).repeat_interleave(D*S),
    in_x.long().flatten().repeat_interleave(S),
    arange(S).repeat(N*D)
].view(N, D, S)
```

这是 $R_t(x^d, s)$——从当前状态 $x^d$ 跳到 $s$ 的前向速率。与上面的 `forward_rates = R_t(s, x^d)` 正好是转置关系。它用于 PC 采样中的校正步骤。

---

## 4. Tau-Leaping：Poisson 近似跳跃

### 4.1 数学原理

给定反向速率矩阵 $\hat R_t^\theta$，在一个无穷小时间 $h$ 内发生 $k$ 次跳跃的概率服从 Poisson 分布：

$$
P(\text{从 } x^d \text{ 跳到 } s \text{ 恰好 } k \text{ 次}) = \frac{(\hat R_t^\theta(x^d,s) \cdot h)^k}{k!} e^{-\hat R_t^\theta(x^d,s) \cdot h}
$$

Tau-Leaping 假设 $h$ 足够小使得速率在 $[t-h, t]$ 内近似常数，然后对每个 $(d, s)$ 对采样跳跃次数。

### 4.2 代码实现

`sampling.py:105-111` (TauLeaping) / `sampling.py:204-213` (PCTauLeaping):

```python
def take_poisson_step(in_x, in_reverse_rates, in_h):
    # 1. 目标减源 = 跳转方向
    diffs = torch.arange(S, device=device).view(1, 1, S) - in_x.view(N, D, 1)
    # diffs: (N, D, S)
    # diffs[b, d, s] = s - in_x[b, d]

    # 2. Poisson 采样跳跃次数
    poisson_dist = torch.distributions.poisson.Poisson(in_reverse_rates * in_h)
    jump_nums = poisson_dist.sample()  # (N, D, S)

    # 3. 跳跃方向 × 跳跃次数
    adj_diffs = jump_nums * diffs  # (N, D, S)

    # 4. 所有目标状态的跳跃累计
    overall_jump = torch.sum(adj_diffs, dim=2)  # (N, D)
    # overall_jump[b, d] = Σ_s jump_nums[b,d,s] × (s - in_x[b,d])

    # 5. 新状态 = 旧状态 + 累计偏移, 裁剪到合法范围
    x_new = torch.clamp(in_x + overall_jump, min=0, max=S-1)
    return x_new
```

### 4.3 数值示例

假设 $S=4$，当前状态 $x^d = 1$，反向速率 $\hat R_t^\theta(1, :) = [0, 0, 3.0, 1.0]$，步长 $h=0.01$：

```
Poisson 速率: [0, 0, 0.03, 0.01]
一次采样可能的结果:

场景 1: jump_nums = [0, 0, 1, 0]
  adj_diffs = [0, 0, 1×(2-1), 0] = [0, 0, 1, 0]
  total_jump = 1
  x_new = 1 + 1 = 2   ← 跳到状态 2

场景 2: jump_nums = [0, 0, 0, 1]
  adj_diffs = [0, 0, 0, 1×(3-1)] = [0, 0, 0, 2]
  total_jump = 2
  x_new = 1 + 2 = 3   ← 跳到状态 3

场景 3 (罕见): jump_nums = [0, 0, 2, 0]
  adj_diffs = [0, 0, 2×1, 0] = [0, 0, 2, 0]
  total_jump = 2
  x_new = 1 + 2 = 3   ← 两次跳到 2, 净效果是跳到了 3

场景 4 (最可能): jump_nums = [0, 0, 0, 0]
  total_jump = 0
  x_new = 1           ← 没有跳跃, 维持原状
```

**Tau-Leaping 与精确 CTMC 的关系**：

| 特性   | 精确 CTMC             | Tau-Leaping |
| ---- | ------------------- | ----------- |
| 跳跃时间 | 指数分布                | 固定步长 $h$    |
| 跳跃次数 | 最多 1 次 (当 $h\to 0$) | 可能多次        |
| 实现   | Gillespie 算法        | Poisson 采样  |
| 计算效率 | 每步 O(1) 次跳跃         | 批量 O(S) 次跳跃 |

当 $h \to 0$ 时，Poisson 过程 → 伯努利过程，最多发生一次跳跃，此时 Tau-Leaping 收敛到精确 CTMC。代码中使用 500 步，$h \approx 0.002$，在这个步长下多次跳跃的概率很小。

### 4.4 同步更新

所有 $N \times D$ 个像素**同时**进行 Poisson 采样和跳跃。这意味着一个步长内可能多个像素同时发生变化。这是标准 Tau-Leaping 的并行实现——利用 GPU 的矩阵运算一次性处理所有像素。

---

## 5. TauLeaping 采样器完整流程

`sampling.py:31-120`:

### 5.1 时间网格

```python
ts = np.concatenate((np.linspace(1.0, min_t, num_steps), np.array([0])))
```

`num_steps=500`, `min_t=0.01`：

```
ts = [1.000, 0.998, 0.996, ..., 0.012, 0.010, 0.000]
      ↑                                            ↑
  t=1.0 (噪声)                              t=0 (干净)
      └────────── 501 个时间点, 500 个步长 ──────────┘
```

每个步长 $h = \frac{1.0 - 0.01}{500} \approx 0.00198$。

### 5.2 每步迭代

```python
for idx, t in tqdm(enumerate(ts[0:-1])):  # 500 步
    h = ts[idx] - ts[idx+1]                # ≈ 0.00198

    # 1. 获取当前时间的转移矩阵和速率矩阵
    qt0 = model.transition(t)            # (N, S, S)
    rate = model.rate(t)                 # (N, S, S)

    # 2. 模型去噪预测
    p0t = softmax(model(x, t), dim=2)    # (N, D, S)

    # 3. 计算反向速率
    qt0_denom = extract_columns(qt0, x)  # (N, D, S): q_{t|0}(·|x)
    forward_rates = extract_columns(rate, x)  # (N, D, S): R_t(·, x)
    inner_sum = (p0t / qt0_denom) @ qt0  # (N, D, S)
    reverse_rates = forward_rates * inner_sum  # (N, D, S)

    # 4. Poisson 跳跃
    x = take_poisson_step(x, reverse_rates, h)
```

### 5.3 最终去噪

```python
# 在 t=min_t=0.01 时, 再做一次预测
p_0gt = softmax(model(x, min_t), dim=2)  # (N, D, S)
x_0max = argmax(p_0gt, dim=2)            # (N, D)
```

在 $t=0.01$ 处的模型预测 $p_{0|t}^\theta(x_0 | x_{0.01})$ 已经是干净图像的估计。取 argmax 得到最终的离散像素值。

**为什么不在 $t=0$ 处做**：$t=0$ 时 $q_{t|0}$ 退化为单位矩阵，速率 $R_t$ 为零矩阵，模型在 $t=0$ 附近没有训练数据（`min_time=0.01` 截断了 $[0, 0.01)$）。因此用 $t=0.01$ 的预测作为最终结果。

### 5.4 采样过程中的维度追踪

```
输入: x (N, 3072), t 标量
  │
  ├── model.forward(x, t) → (N, 3072, 256)   logits
  ├── softmax → (N, 3072, 256)                p0t
  │
  ├── qt0:  (N, 256, 256)
  ├── rate: (N, 256, 256)
  │
  ├── qt0_denom:         (N, 3072, 256)  ← 从 qt0 中按 x 取值
  ├── forward_rates:     (N, 3072, 256)  ← 从 rate 中按 x 取值
  ├── inner_sum = (p0t / qt0_denom) @ qt0  →  (N, 3072, 256)
  ├── reverse_rates = forward_rates * inner_sum  →  (N, 3072, 256)
  │
  ├── Poisson(rates * h) → jump_nums (N, 3072, 256)
  ├── sum(jump_nums * diffs) → overall_jump (N, 3072)
  └── x_new = clamp(x + overall_jump)  →  (N, 3072)
```

---

## 6. Predictor-Corrector 扩展

### 6.1 核心思想

PC 采样（论文 §4.4）受数值求解随机微分方程的 Predictor-Corrector 方法启发。在 CTMC 语境下：

- **Predictor**：Tau-Leaping 步，用 $\hat R_t^\theta$ 模拟反向扩散
- **Corrector**：额外的 Langevin 风格步，用前向 + 反向速率的和修正采样轨迹

`sampling.py:122-240`:

```python
class PCTauLeaping():
    def sample(self, model, N, num_intermediates):
        ...
        x = get_initial_samples(...)

        h = 1.0 / num_steps
        ts = np.linspace(1.0, min_t + h, num_steps)

        for idx, t in enumerate(ts[0:-1]):
            h = ts[idx] - ts[idx+1]

            # === Predictor: 标准 Tau-Leaping ===
            _, reverse_rates, _ = get_rates(x, t)
            x = take_poisson_step(x, reverse_rates, h)

            # === Corrector: 额外 Langevin 步 ===
            if t <= corrector_entry_time:              # t ≤ 0.1
                for _ in range(num_corrector_steps):   # 10 次
                    transpose_rates, reverse_rates, _ = get_rates(x, t-h)
                    corrector_rate = transpose_rates + reverse_rates
                    corrector_rate[:, :, x] = 0.0
                    x = take_poisson_step(x, corrector_rate,
                        corrector_step_size_multiplier * h)  # 步长 ×1.5
```

### 6.2 校正器的物理含义

校正器速率：

$$
R_t^{\text{corr}}(x, s) = R_t(x, s) + \hat R_t^\theta(x, s)
$$

- $R_t(x, s)$：前向速率（从 $x$ 跳到 $s$）
- $\hat R_t^\theta(x, s)$：反向速率（从 $s$ 跳到 $x$）

加法的对称性意味着：如果模型认为 $s$ 和 $x$ 是"互为反向"的两个状态，校正器会加速在这两个状态之间的抖动，类似 Langevin 动力学中的噪声注入——帮助采样逃离局部模式。

### 6.3 PC 的时间调度

```python
corrector_entry_time = 0.1
```

这意味着：

```
时间轴: t=1.0 ────── t=0.1 ────── t=0.01
                  ↑              ↑
          只有 Predictor    Predictor + Corrector(×10)
```

**为什么只在 $t$ 小时校正**：

- 在 $t$ 大（接近 1.0）时，模型预测 $p_{0|t}^\theta$ 还很模糊，反向速率 $\hat R_t^\theta$ 不可靠。此时强行校正可能引入误差。
- 在 $t$ 小（接近 0.0）时，图像结构已经初步形成，Model 的预测比较准确。校正器可以精细调整局部细节。

### 6.4 步长倍增

```python
corrector_step_size_multiplier = 1.5
# → 校正器的有效步长: h_corr = 1.5 × h ≈ 0.00297
```

校正器的步长比预测器大 50%。因为校正器使用 $R_t + \hat R_t^\theta$ 作为速率，这个和通常大于单独的 $\hat R_t^\theta$，因此需要更大的步长来产生足够的探索。

### 6.5 TauLeaping 与 PCTauLeaping 的配置对比

```
TauLeaping:                              PCTauLeaping:
  总前向次数: 500                           总前向次数: 500 + 500 × 10 ≈ 5500
  每步模型调用: 1                           每步模型调用: 1 + 11 ≈ 12 (在 t≤0.1)
  质量: 基线                                质量: 显著提升
  速度: 快                                 速度: ~10x 更慢
```

| 采样器 | 步数 | 每大步模型调用 | 总前向次数 | 采样 50k 图像 (batch=50) |
|--------|------|-----------|-----------|------------------------|
| TauLeaping | 500 | 1 | 500 | ~30 秒 (GPU) |
| PCTauLeaping | 500 | 1 (pred) + 10 (corr) 在 t≤0.1 | 500 + 460 ≈ 960 | ~60 秒 (GPU) |

### 6.6 get_rates 函数的关键差异

对比 TauLeaping（内联代码）和 PCTauLeaping（局部函数）中反向速率的计算：

```python
# TauLeaping — 仅计算反向速率
reverse_rates = forward_rates * ((p0t / qt0_denom) @ qt0_numer)

# PCTauLeaping — 额外返回转置前向速率（供校正器使用）
reverse_rates = forward_rates * ((p0t / qt0_denom) @ qt0_numer)
transpose_forward_rates = R_t(x, ·)  # ← 新增
return transpose_forward_rates, reverse_rates, x_0max
```

---

## 7. 配置系统与采样脚本

### 7.1 评估配置

`config/eval/cifar10.py`:

```python
config.sampler = sampler = ml_collections.ConfigDict()
sampler.name = 'PCTauLeaping'
sampler.num_steps = 500
sampler.min_t = 0.01
sampler.eps_ratio = 1e-9
sampler.initial_dist = 'gaussian'
sampler.num_corrector_steps = 10
sampler.corrector_step_size_multiplier = 1.5
sampler.corrector_entry_time = 0.1
```

与训练配置的覆盖关系：

```python
config.train_config_overrides = [
    [['device'], 'cpu'],           # 训练配置中的 device 被覆盖
    [['data', 'root'], datasets_folder],
    [['distributed'], False]
]
```

评估脚本首先加载训练配置，然后应用覆盖（将设备从 `cpu` 修改为 `cuda`，数据路径修改为评估路径），最后用合并后的配置重建模型。

### 7.2 采样脚本完整流程

`scripts/sample.py:24-63`:

```python
# 1. 加载评估配置
eval_cfg = get_eval_config()

# 2. 加载训练配置并应用覆盖
train_cfg = bookkeeping.load_ml_collections(Path(eval_cfg.train_config_path))
for item in eval_cfg.train_config_overrides:
    utils.set_in_nested_dict(train_cfg, item[0], item[1])

# 3. 重建模型
model = model_utils.create_model(train_cfg, device)

# 4. 加载训练好的权重（移除 DDP 的 "module." 前缀）
loaded_state = torch.load(Path(eval_cfg.checkpoint_path), map_location=device)
modified_model_state = utils.remove_module_from_keys(loaded_state['model'])
model.load_state_dict(modified_model_state)

# 5. 切换到评估模式 → 自动使用 EMA shadow 参数
model.eval()

# 6. 批量生成 50,000 张图像
total_samples = 0
batch = 50
sampler = sampling_utils.get_sampler(eval_cfg)

while True:
    samples, _, _ = sampler.sample(model, batch, 1)
    # samples: (50, 3072), dtype=int, range [0, 255]

    samples = samples.reshape(batch, 3, 32, 32)
    samples_uint8 = samples.astype(np.uint8)

    for i in range(batch):
        img = Image.fromarray(imgtrans(samples_uint8[i]))
        img.save(f'{save_samples_path}/{total_samples + i}.png')

    total_samples += batch
    if total_samples >= 50000:
        break
```

### 7.3 模型加载中的去前缀

```python
# utils.py
def remove_module_from_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_state_dict[key[7:]] = value  # 去掉 "module." 前缀
        else:
            new_state_dict[key] = value
    return new_state_dict
```

如果训练时使用了 `DistributedDataParallel`（`dist_train.py`），检查点中所有参数键名带 `module.` 前缀。单 GPU 推理时需要去掉。

### 7.4 图像后处理

```python
def imgtrans(x):
    # (C, H, W) → (H, W, C)
    x = np.transpose(x, (1, 2, 0))
    return x
```

PyTorch 的 `(3, 32, 32)` 格式 → PIL 的 `(32, 32, 3)` RGB 格式 → PNG 保存。

---

## 8. 中间状态可视化

采样器返回三个值：

```python
return x_0max, x_hist, x0_hist
```

```python
samples, x_hist, x0_hist = sampler.sample(model, batch, num_intermediates)
```

- `x_hist`: 中间状态 $x_t$ 的时间序列
- `x0_hist`: 中间预测 $x_0^{\text{pred}}$ 的时间序列（取 argmax）

对于 `num_intermediates=5`，可以可视化采样过程：

```
t=1.0:  ████████████  ← 纯噪声 (高斯偏好)
t=0.75: ████░░████░░  ← 轮廓隐约可见
t=0.5:  ░░░████░░░░  ← 主体结构形成
t=0.25: ░░░░██░░░░░░  ← 细节补充
t=0.01: ░░░░░░░░░░░░  ← 最终图像

x_t:    随着去噪推进，噪声逐渐减少
x0_pred:从模糊轮廓逐渐锐化为清晰图像
```

这种可视化既是调试工具（检查采样是否正常收敛），也是论文中常见的"去噪过程展示"。
