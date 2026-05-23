# 02 — 前向过程（Forward Diffusion）详解

## 概述

前向过程（forward process / diffusion process）定义了如何从干净数据 $x_0$ 逐步加噪得到 $x_t$。UD3 的前向过程核心公式为：

$$
q(x_t \mid x_0) = \bar\alpha_t \cdot \delta(x_t, x_0) + (1 - \bar\alpha_t) \cdot m(x_t)
$$

其中 $\bar\alpha_t$ 是噪声调度中的累积保留概率，$m$ 是稳态噪声分布。前向采样使用**分支指示变量**技巧，而非直接采样完整的 categorical 分布。

本文档聚焦代码中前向过程相关的函数，按调用链路自底向上解读。

---

## 1. 概率辅助函数（8–26 行）

```python
def sample_uniform_categorical(x_shape, num_classes, device="cuda"):
    return torch.randint(num_classes, size=x_shape, device=device)

def sample_bernoulli(prob, x_shape, device="cuda"):
    u = torch.rand(x_shape, device=device)
    b = u.clamp(min=EPS) < prob
    return b

def sample_categorical(prob):
    ind_sample = torch.multinomial(prob.flatten(end_dim=-2), num_samples=1).view(prob.shape[:-1])
    return ind_sample
```

| 函数                           | 用途                                                 |
| ---------------------------- | -------------------------------------------------- |
| `sample_uniform_categorical` | 从均匀离散分布采样（当 $m$ 为 None 时使用）                        |
| `sample_bernoulli`           | 生成分支指示变量 $b_t \sim \text{Bernoulli}(\bar\alpha_t)$ |
| `sample_categorical`         | 从任意概率向量采样，用于反向步和初始 $x_T$                           |

**关键细节**：`sample_bernoulli` 使用 `u.clamp(min=EPS)` 确保数值稳定性，避免 `prob=0` 时出错。`sample_categorical` 先 `flatten` 再 `multinomial`，支持任意形状输入（B, N1, ..., Nk, C）。

---

## 2. 索引辅助函数（28–62 行）

```python
def get_broadcast_idx(shape):
    return [torch.arange(s).view([1]*i+[-1]+[1]*(len(shape)-1-i)) for i, s in enumerate(shape)]

def index_last_dim(x, idx):
    """ Equivalent to torch.gather """
    broadcast_idx = get_broadcast_idx(idx.shape)
    return x[broadcast_idx + [idx]]

def set_last_dim(x, idx, value=0, inplace_add=False):
    """ Equivalent to torch.scatter """
    broadcast_idx = get_broadcast_idx(idx.shape)
    if inplace_add:
        x[broadcast_idx + [idx]] += value
    else:
        x[broadcast_idx + [idx]] = value
    return x
```

这些函数在整个代码中高频使用，核心作用是在 categorical 分布的最后一维（类别维）上按索引取值或赋值。

**工作原理**：
- `get_broadcast_idx` 为张量的每一维生成一个带有适当广播形状的 arange 张量。例如形状 `(B, N, C)` 会生成 `[arange(B).view(B,1,1), arange(N).view(1,N,1)]`。
- `index_last_dim(x, idx)` 等价于 `torch.gather(x, dim=-1, idx.unsqueeze(-1)).squeeze(-1)`，即取每个位置 `idx` 对应的类别概率值 $f_\theta(x_t)[x_t]$。
- `set_last_dim` 等价于 `torch.scatter`，在指定索引位置设置（或累加）值。

**前向过程中的使用**：
- `get_m_dot_xt` 用 `index_last_dim` 获取 $m[x_t]$：噪声分布在当前 token 处的概率。
- `qt_0_prob` 用 `set_last_dim` 将 $\bar\alpha_t$ 设置到 $x_0$ 对应的类别位置。
- `qs_t0_prob` 用两者构造条件概率矩阵。

---

## 3. 噪声调度 `noise_schedule`（64–118 行）

```python
def noise_schedule(t_step, s_step=None, schedule_type="cosine", N=1000, Tmax=1, ...):
    step_to_time = lambda step: step if N == 0 else step / N * Tmax
    t = step_to_time(t_step)
    ...
    return alphabar_t, beta_t
```

前向过程依赖两个返回值：
- `alphabar_t`：累积保留概率 $\bar\alpha_t$
- `beta_t`：瞬时噪声率（连续时间损失需要）

**与前向过程的关系**：
- $\bar\alpha_t$ 越大 → 越大概率保留 $x_0$（噪声小）
- $\bar\alpha_t \to 0$ 当 $t \to T$ → $x_t$ 完全来自噪声 $m$
- `s_step` 参数用于条件概率 `qs_t0_prob`，计算 $\bar\alpha_s / \bar\alpha_t$

---

## 4. 前向采样 `qt_0_sample`（206–233 行）

这是训练时最常调用的前向函数，对 batch 中的每个样本 $x_0$ 和随机时间步 $t$，采样 $x_t$。

```python
def qt_0_sample(self, x_0, t, m=None, conditional_mask=None):
    alphabar_t, _ = self.get_alphabar_beta(t)
    alphabar_t = alphabar_t.view([-1]+[1]*(x_0.dim()-1))

    # 采样 m0 ~ m（噪声分布）
    if m is None:
        m0 = sample_uniform_categorical(sample_shape, self.num_classes, device=x_0.device)
    elif m.dim() == 1:
        m0 = torch.multinomial(m, num_samples=sample_shape.numel(), replacement=True).view(sample_shape)
    else:
        m0 = sample_categorical(m)

    # 分支指示变量：bt=1 保留原值，bt=0 从噪声采样
    bt = sample_bernoulli(alphabar_t, sample_shape, device=x_0.device)
    sample = torch.where(bt, x_0, m0)

    # 条件掩码：被 mask 的位置强制保持 x_0
    if conditional_mask is not None:
        sample[conditional_mask] = x_0[conditional_mask]
    return sample
```

### 算法流程

```
输入: x_0 (B, N1, ..., Nk), t (B,)
1. 计算 alphabar_t = 噪声调度(t)        # (B,) → (B,1,...,1)
2. 从 m 采样 m0 与 x_0 同形状          # 噪声候选
3. bt = Bernoulli(alphabar_t)          # 分支指示, True=保留, False=替换
4. x_t = bt ? x_0 : m0                # 逐位置选择
5. 若 conditional_mask 存在，掩码位置强制 = x_0
输出: x_t (B, N1, ..., Nk)
```

### 三种 m 的处理

| m 类型 | 采样方式 | 典型场景 |
|--------|---------|---------|
| `None` | `sample_uniform_categorical` → 均匀噪声 | 无先验知识 |
| `(C,)` 1D 张量 | `torch.multinomial` 按权重采样 | 预定义噪声分布（如均匀、频率分布） |
| `(B,...,C)` 完整张量 | `sample_categorical` 逐位置采样 | 位置相关的噪声（如 absorbing [MASK]） |

### 分支指示变量的意义

直接采样 $q(x_t \mid x_0) = \bar\alpha_t \delta_{x_0} + (1-\bar\alpha_t)m$ 需要构造完整的 categorical 概率向量再调用 `sample_categorical`。分支技巧将过程分解为两步：

1. 以概率 $\bar\alpha_t$ 决定是保留还是替换
2. 若替换，再从 $m$ 中采样

这等价于原始分布，但避免了大张量的构造，且与掩码扩散（masked diffusion）框架自然对齐。

### 条件掩码

`conditional_mask` 参数用于两种场景：
- **条件生成**：某些位置给定（prompt/class label），不参与加噪
- **填充掩码**：padding 区域不贡献损失

前向过程中，被掩码位置在整个扩散过程中保持不变（始终为 $x_0$），这与 `sample_step` 中反向过程的处理一致。

---

## 5. 前向概率 `qt_0_prob`（235–254 行）

计算 $q(x_t \mid x_0)$ 的完整概率分布，仅用于连续时间模式的损失计算。

```python
def qt_0_prob(self, x_0, t, m=None, return_beta=False):
    alphabar_t, beta_t = self.get_alphabar_beta(t)
    alphabar_t, beta_t = alphabar_t.view(shape), beta_t.view(shape)

    if m is None:
        m = torch.full_like(x_0, 1/self.num_classes, dtype=torch.float32).unsqueeze(-1).repeat_interleave(self.num_classes, -1)
    elif m.dim() == 1:
        m = torch.broadcast_to(m, list(x_0.shape)+[self.num_classes])

    prob = (1 - alphabar_t) * m
    prob = set_last_dim(prob, x_0, value=alphabar_t.squeeze(-1), inplace_add=True)
    return prob
```

### 算法解析

该函数构造形状为 `(B, N1, ..., Nk, C)` 的概率张量，其计算逻辑与采样公式完全对应：

$$
q(x_t \mid x_0)[i] =
\begin{cases}
\bar\alpha_t + (1-\bar\alpha_t) \cdot m[i] & i = x_0 \\
(1-\bar\alpha_t) \cdot m[i] & i \neq x_0
\end{cases}
$$

代码实现分为两步：
1. `prob = (1 - alphabar_t) * m` — 所有类别初始化为 $(1-\bar\alpha_t)m$
2. `set_last_dim(prob, x_0, value=alphabar_t, inplace_add=True)` — 在 $x_0$ 位置**累加** $\bar\alpha_t$

### 与采样的关系

`qt_0_sample` 是该分布的高效采样版本，而 `qt_0_prob` 返回完整概率向量。两者数学上等价，但 `qt_0_prob` 仅在需要概率值计算损失时使用（连续时间），训练中的采样加噪则始终使用 `qt_0_sample`。

---

## 6. 条件前向概率 `qs_t0_prob`（256–289 行）

计算 $q(x_s \mid x_t, x_0)$，即给定 $x_0$ 和更嘈杂的 $x_t$ 后，中间步 $x_s$（$s < t$）的条件分布。用于离散时间模式的 KL 损失。

```python
def qs_t0_prob(self, x_t, x_0, t, s, m=None):
    alphabar_t, _ = self.get_alphabar_beta(t)
    alphabar_s, _ = self.get_alphabar_beta(s)
    ...
    mu_alphabar_t_s = self.get_mu_times_alphabar(alphabar_t, alphabar_s)
    mu_t_s = self.get_mu(alphabar_t, alphabar_s)
    lambda_t_s = self.get_lambda(alphabar_t, alphabar_s, x_t, m)

    # prob_eq: 当 x_t == x_0
    prob_eq = lambda_t_s[..., None]
    prob_eq = prob_eq * m ...
    prob_eq[broadcast_idx+[x_t]] += 1 - lambda_t_s

    # prob_neq: 当 x_t != x_0
    prob_neq = (mu_t_s - mu_alphabar_t_s)[..., None]
    prob_neq = prob_neq * m ...
    prob_neq[broadcast_idx+[x_t]] += mu_alphabar_t_s
    prob_neq[broadcast_idx+[x_0]] += 1 - mu_t_s

    prob = torch.where((x_t == x_0).unsqueeze(-1), prob_eq, prob_neq)
    return prob
```

### 三个核心系数

| 系数 | 公式 | 含义 |
|------|------|------|
| $\lambda_{t\|s}$ | $\frac{(1-\bar\alpha_s)(1-\bar\alpha_t/\bar\alpha_s) m[x_t]}{\bar\alpha_t + (1-\bar\alpha_t)m[x_t]}$ | $x_t$ 与 $x_0$ 相同的概率中，来自噪声的部分 |
| $\mu_{t\|s}$ | $(1-\bar\alpha_s)/(1-\bar\alpha_t)$ | $x_s$ 从噪声 $m$ 采样的概率 |
| $\mu^{\alpha}_{t\|s}$ | $\frac{\bar\alpha_t - \bar\alpha_s\bar\alpha_t}{\bar\alpha_s - \bar\alpha_s\bar\alpha_t}$ | $x_s$ 保留 $x_t$ 且 $x_t$ 来自 $x_0$ 的概率 |

### $\text{prob\_eq}$ vs $\text{prob\_neq}$

代码根据 $x_t$ 是否等于 $x_0$ 分情况构造概率，这是 UD3 推导中的关键技巧：

**当 $x_t = x_0$**（同一 token 未被噪声替换）：
- 大概率 $x_s$ 也保留该 token
- $\text{prob\_eq}$ 在 $x_t$ 位置设 $1-\lambda_{t\|s}$，其余为 $\lambda_{t\|s} \cdot m$

**当 $x_t \neq x_0$**（token 已被替换）：
- $x_s$ 有三种去向：保留 $x_t$、回到 $x_0$、从 $m$ 采样
- $\text{prob\_neq}$ 分别处理这三项

---

## 7. 辅助系数函数（用于前向过程的 q 分布）

### `get_m_dot_xt`（151–163 行）

获取噪声分布在当前 token $x_t$ 处的概率 $m[x_t]$，是前向过程中的核心操作，被 `get_lambda`、`get_gamma_coef`、`_gt_inner` 等反复调用。

```python
def get_m_dot_xt(self, x_t, m=None):
    if m is None:
        m_dot_xt = torch.full_like(x_t, 1/self.num_classes, dtype=torch.float32)
    elif m.dim() == 1:
        m_dot_xt = m[x_t]           # (C,) 索引
    else:
        m_dot_xt = index_last_dim(m, x_t)  # (B,...,C) gather
    return m_dot_xt
```

### `get_lambda`（165–180 行）

$$
\lambda_{t|s} = \frac{(1-\bar\alpha_s)(1-\bar\alpha_t/\bar\alpha_s) \cdot m[x_t]}{\bar\alpha_t + (1-\bar\alpha_t) \cdot m[x_t]}
$$

对应概率 $\text{prob\_eq}$ 中，$m$ 部分的系数。

### `get_mu`（182–189 行）

$$
\mu_{t|s} = \frac{1-\bar\alpha_s}{1-\bar\alpha_t}
$$

$x_s$ 从噪声 $m$ 采样的概率，越大表示 $x_s$ 越不确定。

### `get_mu_times_alphabar`（191–194 行）

$$
\mu^{\alpha}_{t|s} = \frac{\bar\alpha_t - \bar\alpha_s\bar\alpha_t}{\bar\alpha_s - \bar\alpha_s\bar\alpha_t}
$$

$x_s$ 保留 $x_t$ 且 $x_t$ 来自 $x_0$ 的概率。

---

## 8. 前向过程完整数据流

```
训练循环:
  1. 从数据集采样 x_0, 随机采样 t ~ Uniform{1,...,T}
  2. x_t = qt_0_sample(x_0, t, m)       # 前向加噪
  3. fprob_t = denoising_fn(x_t, t)      # 去噪网络预测
  4. loss = compute_loss(fprob_t, x_t, x_0, t, m)
     ├── 离散模式: qs_t0_prob (解析 KL 的后项)
     └── 连续模式: qt_0_prob (Proposition 4 的辅助项)

前向过程涉及的关键函数调用链:
  qt_0_sample
    ├── get_alphabar_beta → noise_schedule    # 计算 alphabar_t
    ├── sample_bernoulli                       # 分支指示 bt
    ├── sample_uniform_categorical / sample_categorical / multinomial  # 噪声 m0
    └── torch.where(bt, x_0, m0)              # 最终采样

  qs_t0_prob (离散损失的反向条件概率)
    ├── get_alphabar_beta (t 和 s)
    ├── get_mu / get_mu_times_alphabar / get_lambda  # 三个系数
    ├── get_m_dot_xt                                  # m[x_t]
    ├── 构造 prob_eq（x_t == x_0 分支）
    └── 构造 prob_neq（x_t != x_0 分支）

  qt_0_prob (连续损失的概率向量)
    ├── get_alphabar_beta
    └── set_last_dim(prob, x_0, alphabar_t)  # 在 x_0 位置累加保留概率
```

### 离散时间 vs 连续时间的前向差异

| 方面 | 离散时间 (num_steps > 0) | 连续时间 (num_steps = 0) |
|------|--------------------------|--------------------------|
| 时间步 | 整数 $t \in [1, N]$ | 浮点数 $t \in [0, T_\text{max}]$ |
| 前向采样 | `qt_0_sample` | `qt_0_sample`（与离散相同！） |
| 损失中的前向概率 | `qs_t0_prob`（解析条件概率） | `qt_0_prob`（完整概率向量） |
| 调度参数 | `N=num_steps, Tmax=1` | `N=0, Tmax=1` |

两种模式的前向采样实现**完全一致**（`qt_0_sample` 共享），区别仅在于损失计算时如何使用前向概率。

---

## 9. 与论文公式的对应

| 代码函数 | 论文符号 | 论文位置 |
|----------|----------|----------|
| `qt_0_sample` | $x_t \sim q(x_t \mid x_0)$ | §3.1, Eq. (1) |
| `qt_0_prob` | $q(x_t \mid x_0)$ 概率 | §3.1 |
| `qs_t0_prob` | $q(x_s \mid x_t, x_0)$ | §3.3, Eq. (7) |
| `get_alphabar_beta` | $\bar\alpha_t, \beta_t$ | §3.1 |
| `get_m_dot_xt` | $m[x_t]$ | §2 |
| `get_lambda` | $\lambda_{t\|s}$ | §3.3, Lemma 3 |
| `get_mu` | $\mu_{t\|s}$ | §3.3, Lemma 3 |
| `get_mu_times_alphabar` | $\mu^{\alpha}_{t\|s}$ | §3.3, Lemma 3 |
| `sample_bernoulli` | $b_t \sim \text{Bernoulli}(\bar\alpha_t)$ | §3.1 |
