# 07 — CT-ELBO 损失函数：数学推导与逐行代码解读

> 对应论文 §3.2 + Appendices C/D，代码 `lib/losses/losses.py:11-246`
>
> 聚焦图像任务（CIFAR-10）中的 `GenericAux` 实现

## 目录

- [1. 从离散时间 ELBO 到连续时间 ELBO](#1-从离散时间-elbo-到连续时间-elbo)
- [2. 损失函数的完整数据流](#2-损失函数的完整数据流)
- [3. 步骤①：采样时间 t](#3-步骤采样时间-t)
- [4. 步骤②：采样 x_t——从 q_{t|0} 中抽取噪声图像](#4-步骤采样-x_t从-q_t0-中抽取噪声图像)
- [5. 步骤③：采样 x_tilde——单维度的速率加权跳跃](#5-步骤采样-x_tilde单维度的速率加权跳跃)
- [6. 反向速率 R̂_t^θ 的隐式定义](#6-反向速率-r̂_t^θ-的隐式定义)
- [7. Reg 项：反向速率的正则化](#7-reg-项反向速率的正则化)
- [8. Sig 项：主要学习信号](#8-sig-项主要学习信号)
- [9. 辅助 NLL 与最终损失](#9-辅助-nll-与最终损失)
- [10. 单次前向优化 (One Forward Pass)](#10-单次前向优化-one-forward-pass)
- [11. 配置参数的影响](#11-配置参数的影响)

---

## 1. 从离散时间 ELBO 到连续时间 ELBO

### 1.1 论文核心推导

标准扩散模型的 ELBO 在离散时间步上定义。CTMC 论文的核心贡献之一是将 ELBO 推广到连续时间。

**离散时间 ELBO**（D3PM，$T$ 个时间步）：

$$
\mathcal{L}_T = \sum_{k=1}^T \mathbb{E}_{x_k \sim q_{k|0}} \left[ D_{\mathrm{KL}}(q_{k-1|k,0} \parallel p_\theta(x_{k-1}|x_k)) \right]
$$

**连续时间 ELBO**（CTMC，论文 Theorem 1）：

$$
\mathcal{L}_\mathrm{CT}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0,T),\, x_t \sim q_t,\, \tilde x \sim r_t(\cdot|x_t)} \left[ \sum_{x' \neq x_t} \hat R^\theta_t(x_t,x') \;-\; \mathcal{Z}^t(x_t) \log \hat R^\theta_t(\tilde x, x_t) \right] + C
$$

其中 $\mathcal{Z}^t(x_t) = \sum_{x' \neq x_t} R_t(x_t, x')$ 是状态 $x_t$ 的总出率。

直观理解：

| 项 | 作用 | 类比 |
|----|------|------|
| $\sum_{x'\neq x_t} \hat R^\theta_t$ | 让反向速率尽可能小 | 正则化 |
| $-\mathcal{Z}^t \log \hat R^\theta_t(\tilde x, x_t)$ | 最大化特定跳跃的反向速率 | 似然拟合 |

两项的平衡：模型必须把"概率质量"集中在正确的跳跃上，而不是均匀分散到所有状态。

### 1.2 代码的整体入口

`lib/losses/losses.py:23-246`:

```python
class GenericAux():
    def calc_loss(self, minibatch, state, writer):
        model = state['model']
        B, D = minibatch.shape     # (128, 3072)

        # ③ 采样 t ∼ U(0.01, 1.0)
        ts = torch.rand((B,), device=device) * 0.99 + 0.01

        # ④ 计算 q_{t|0} 和 R_t
        qt0 = model.transition(ts)    # (B, S, S)
        rate = model.rate(ts)         # (B, S, S)

        # ⑤ 采样 x_t 和 x_tilde
        x_t = sample_xt(minibatch, qt0)           # (B, D)
        x_tilde = sample_xtilde(x_t, rate)        # (B, D)

        # ⑥ 模型前向
        logits = model(x_tilde, ts)   # (B, D, S)

        # ⑦ Reg 项 + Sig 项 + NLL
        reg_term = compute_reg(p0t, qt0, rate, x_tilde)
        sig_term = compute_sig(p0t, qt0, rate, minibatch, x_tilde)
        nll = cross_entropy(logits, minibatch)

        return reg_term + sig_term + 0.001 * nll
```

---

## 2. 损失函数的完整数据流

整个 `calc_loss` 涉及 6 个 PyTorch 张量的交互。理解它们的关系是读懂代码的关键：

```
符号约定:
  B = batch size (128)
  D = 像素总数 (3×32×32 = 3072)
  S = 状态空间大小 (256)
  N = B*D (简化)

参与计算的张量:
  qt0:   (B, S, S)    转移概率矩阵 q_{t|0}(x_t | x_0)
  rate:  (B, S, S)    速率矩阵 R_t(x', x)

  minibatch:  (B, D)    原始图像 x_0 (像素值 [0,255])
  x_t:        (B, D)    加噪图像 (在时刻 t)
  x_tilde:    (B, D)    从 x_t 再跳一步后的图像

  logits:  (B, D, S)   模型输出的预测 logits
  p0t:     (B, D, S)   softmax 后的 p_θ(x_0|x_t)

索引操作模式:
  代码大量使用 arange + repeat_interleave 来做"查表"
  例如: qt0[arange(B).repeat(D), x_0, :]  → 取出每个像素对应的转移行向量
```

---

## 3. 步骤①：采样时间 t

`lib/losses/losses.py:32`:

```python
ts = torch.rand((B,), device=device) * (1.0 - self.min_time) + self.min_time
```

等价于 $t \sim \mathcal{U}(0.01, 1.0)$。

**为什么避开 $t=0$**：当 $t \to 0$ 时，$q_{t|0}$ 趋于单位矩阵，$R_t$ 趋于零矩阵。在 $t=0$ 附近的数值计算会遭遇除零和 log(0) 问题。设置 `min_time = 0.01` 是一个实用主义的截断。

**为什么 $t=1$ 是终点**：论文将时间区间归一化为 $[0,1]$。$t=1$ 对应"充分扩散"，$q_1$ 接近平稳分布。

每个 batch 的 $B$ 个样本独立采样时间，这实现了论文中的"每个样本使用不同的随机时间"——这是连续时间 ELBO 的蒙特卡洛估计。

---

## 4. 步骤②：采样 x_t——从 q_{t|0} 中抽取噪声图像

`lib/losses/losses.py:41-48`:

```python
qt0_rows_reg = qt0[
    torch.arange(B, device=device).repeat_interleave(D),   # batch 索引: [0,0,...,0, 1,1,...,1, ...]
    minibatch.flatten().long(),                            # x_0 值: [pixel_0, pixel_1, ...]
    :                                                       # 所有目标状态
]  # 结果: (B*D, S)
```

这是代码中第一处复杂的索引操作。展开来看：

```
qt0: (B, S, S)
  qt0[b, i, j] = 第 b 个样本中，给定初始状态 i，t 时刻处于 j 的概率

我们要为每个像素独立采样 x_t:
  对于样本 b 中的像素 d，其 x_0 值为 v = minibatch[b, d]
  需要 qt0[b, v, :] — 即第 b 个转移矩阵的第 v 行

arange(B).repeat_interleave(D) 产生:
  [0,0,...,0, 1,1,...,1, ..., B-1,...,B-1]  共 B*D 个元素
  └─D个─┘  └─D个─┘          └─D个─┘

minibatch.flatten() 产生:
  [x_0^{0,0}, x_0^{0,1}, ..., x_0^{0,D-1}, x_0^{1,0}, ...]  共 B*D 个元素

所以:
  qt0_rows_reg[k, :] = qt0[ batch_idx, x_0_value, : ]
  其中 batch_idx = k // D,  x_0_value = minibatch[batch_idx, k % D]
```

采样过程：

```python
x_t_cat = torch.distributions.categorical.Categorical(qt0_rows_reg)
x_t = x_t_cat.sample().view(B, D)
```

每个像素独立地从其条件分布 $q_{t|0}(\cdot | x_0^d)$ 中采样。这意味着：
- 同一个图像的不同像素可能经历不同程度的噪声
- 但由于 $q_{t|0}$ 只依赖于初始像素值，像素之间条件独立

---

## 5. 步骤③：采样 x_tilde——单维度的速率加权跳跃

论文 §4.2 使用"维度分解"技巧：每次只改变一个维度的值。这对应在 CT-ELBO 中，蒙特卡洛估计反向速率项时只选一个维度做重要性采样。

### 5.1 第一步：选择跳哪个维度

`lib/losses/losses.py:50-64`:

```python
rate_vals_square = rate[
    torch.arange(B, device=device).repeat_interleave(D),   # batch 索引
    x_t.long().flatten(),                                   # 当前状态值 x_t^d
    :
]  # (B*D, S)

rate_vals_square[arange(B*D), x_t.long().flatten()] = 0.0  # 对角线置零
rate_vals_square = rate_vals_square.view(B, D, S)
```

这一步取出 $R_t(x_t^d, :)$，即每个像素当前值的**出率向量**。将对角线（跳到自身）设为 0。

```python
rate_vals_square_dimsum = torch.sum(rate_vals_square, dim=2).view(B, D)
# (B, D)  每个像素的总出率 Z_t(x_t^d) = Σ_{s≠x_t^d} R_t(x_t^d, s)

square_dimcat = torch.distributions.categorical.Categorical(rate_vals_square_dimsum)
square_dims = square_dimcat.sample()  # (B,)
```

**关键设计**：$P(\text{选维度 } d) \propto \mathcal{Z}^t(x_t^d) = \sum_{s \neq x_t^d} R_t(x_t^d, s)$。总出率越大的像素，越可能被选中发生跳跃。这是**重要性采样**：高活跃度的像素提供了更大的学习信号。

### 5.2 第二步：选择新值

`lib/losses/losses.py:65-78`:

```python
rate_new_val_probs = rate_vals_square[
    torch.arange(B, device=device),
    square_dims,       # 每个样本选中的维度
    :
]  # (B, S)

square_newvalcat = torch.distributions.categorical.Categorical(rate_new_val_probs)
square_newval_samples = square_newvalcat.sample()  # (B,) 新值

x_tilde = x_t.clone()
x_tilde[torch.arange(B), square_dims] = square_newval_samples
```

新值 $s$ 的概率正比于 $R_t(x_t^d, s)$。由于 `GaussianTargetRate` 只有"相近状态"有显著转移率，新值通常与旧值相近。

**最终效果**：$x_{\text{tilde}}$ 与 $x_t$ 恰好差一个像素不一样。这个"被改变的像素"就是模型要学习的关键点。

### 5.3 维度分解的物理意义

```
前向 CTMC 的一个无穷小时间段 dt 内:
  - 每个像素独立地以速率 R_t(x_t^d, ·) 发生跳跃
  - 两个像素同时跳跃的概率是 O(dt²)，可忽略

因此"一次只改一个维度"是 CTMC 的精确模拟:
  x_t  ──────── dt ────────→  x_{t+dt}
  (一个像素变了，其他不变)
```

---

## 6. 反向速率 R̂_t^θ 的隐式定义

在理解 Reg 和 Sig 项之前，必须先理解反向速率 $\hat R_t^\theta$ 在代码中如何表示。

**$\hat R_t^\theta$ 不是直接参数化的**。论文中，反向速率通过 $p_{0|t}^\theta$ 隐式定义：

$$
\hat R_t^\theta(x, \tilde x) = R_t(\tilde x, x) \cdot \sum_{x_0} \frac{q_{t|0}(\tilde x | x_0)}{q_{t|0}(x | x_0)} \, p_{0|t}^\theta(x_0 | x)
$$

其中：
- $R_t(\tilde x, x)$：前向速率（从 $\tilde x$ 跳到 $x$）
- $p_{0|t}^\theta(x_0 | x)$：去噪网络的预测
- $q_{t|0}(\tilde x|x_0) / q_{t|0}(x|x_0)$：概率比率

这个定义的来源是**时间反演公式**（论文 Eq. 7）：反向过程也是 CTMC，其速率由前向速率和数据密度之比决定。

代码中，这个表达式被分解到 Reg 和 Sig 项的计算中：

```python
# 反向速率的核心计算 (sampling.py 中也有完全相同的形式)
inner_sum = (p0t / qt0_denom) @ qt0_numer   # (B, D, S)
reverse_rates = forward_rates * inner_sum     # (B, D, S)
```

---

## 7. Reg 项：反向速率的正则化

从论文 Eq. 14：

$$
\text{Reg} = T \cdot \mathbb{E}_{t, x_t, \tilde x} \left[ \sum_{x' \neq x_t} \hat R_t^\theta(x_t, x') \right]
$$

这个项惩罚**所有可能跳跃的反向速率之和**，防止 $\hat R_t^\theta$ 无限增长。

### 7.1 计算流程

`lib/losses/losses.py:85-122`:

```python
# 模型前向 (以 one_forward_pass=True 为例)
x_logits = model(x_tilde, ts)               # (B, D, S)
p0t_reg = F.softmax(x_logits, dim=2)        # (B, D, S)  ← p_θ(x_0|x_tilde)
reg_x = x_tilde                              # 使用 x_tilde 作为条件
```

```python
# 构建 mask: 排除自身 (对角线)
mask_reg = torch.ones((B, D, S))
mask_reg[arange(B).repeat(D), arange(D).repeat(B), reg_x.flatten()] = 0.0
```

```python
# qt0_numer = q_{t|0}(x'|x_0)   形状: (B, S, S)
qt0_numer_reg = qt0.view(B, S, S)

# qt0_denom = q_{t|0}(· | reg_x)   形状: (B, D, S)
qt0_denom_reg = qt0[torch.arange(B, device=device).repeat_interleave(D), :, reg_x.flatten()].view(B, D, S) + eps
```

`qt0[:, :, reg_x]` 的意思是：对于每个像素 $d$，取 $q_{t|0}(\cdot, \text{reg\_x}^d)$——即从所有初始状态 $x_0$ 出发、到达 $\text{reg\_x}^d$ 的概率。

```python
# rate_vals = R_t(·, reg_x)  形状: (B, D, S)
rate_vals_reg = rate[torch.arange(B, device=device).repeat_interleave(D), :, reg_x.flatten()].view(B, D, S)
```

这里 `rate[:, :, reg_x]` 取的是 $R_t(x', \text{reg\_x}^d)$——从任意状态 $x'$ 跳到当前状态 $\text{reg\_x}^d$ 的**前向**速率。

```python
# 核心计算
reg_tmp = (mask_reg * rate_vals_reg) @ qt0_numer_reg.transpose(1,2)
# (B, D, S) = (B, D, S) @ (B, S, S)

reg_term = torch.sum(
    (p0t_reg / qt0_denom_reg) * reg_tmp,
    dim=(1,2)
)  # (B,)
```

### 7.2 数学对应

展开 `reg_tmp[i,d,s]`：

$$
\text{reg\_tmp}[b,d,s] = \sum_{x_0} \left( \sum_{x' \neq \text{reg\_x}^d} \underbrace{R_t(x', \text{reg\_x}^d)}_{\text{rate\_vals\_reg}} \cdot \underbrace{q_{t|0}(s|x_0)}_{\text{qt0\_numer}} \right)
$$

这实际上是 $\sum_{x' \neq x} R_t(x', x) \cdot q_{t|0}(s|x_0)$，然后对所有 $x_0$ 求和。

最终：

$$
\text{reg\_term}[b] = \sum_{d,s} \frac{p_{0|t}^\theta(s|\text{reg\_x})}{q_{t|0}(s|\text{reg\_x})} \cdot \text{reg\_tmp}[b,d,s]
$$

这等价于 $\sum_{x'} \hat R_t^\theta(\text{reg\_x}, x')$——即 reg 项需要的反向速率总和。

---

## 8. Sig 项：主要学习信号

从论文 Eq. 14：

$$
\text{Sig} = -T \cdot \mathbb{E}_{t, x_t, \tilde x} \left[ \mathcal{Z}^t(x_t) \log \hat R_t^\theta(\tilde x, x_t) \right]
$$

其中 $\mathcal{Z}^t(x_t) = \sum_{x' \neq x_t} R_t(x_t, x')$ 是归一化常数。

Sig 项是最重要的项：它最大化**从 $\tilde x$ 跳回 $x_t$ 的反向速率**。这正是"去噪"的核心——给定一个稍稍更噪声的状态，模型应该预测如何回到稍稍更干净的状态。

### 8.1 Outer 部分：速率 × 概率比率

`lib/losses/losses.py:136-180`:

```python
# outer_qt0_numer:  q_{t|0}(x_0 | x')  形状 (B, D, S)
# 固定 batch b 和像素 d:
#   outer_qt0_numer[b, d, s] = q_{t|0}(minibatch[b,d] | s)
outer_qt0_numer_sig = qt0[
    arange(B).repeat_interleave(D*S),
    minibatch.flatten().long().repeat_interleave(S),
    arange(S).repeat(B*D)
].view(B, D, S)
```

这个复杂的索引等价于：

```python
for b in range(B):
    for d in range(D):
        for s in range(S):
            outer_qt0_numer_sig[b,d,s] = qt0[b, minibatch[b,d], s]
```

即 $q_{t|0}(x_0^d | s)$——给定初始状态 $x_0^d$，处于状态 $s$ 的概率。注意这里 "s" 是**目标状态**（第二个下标），而非初始状态。

```python
# outer_qt0_denom:  q_{t|0}(x_0 | x_tilde)  形状 (B, D)
outer_qt0_denom_sig = qt0[
    arange(B).repeat_interleave(D),
    minibatch.long().flatten(),
    x_tilde.long().flatten()
] + eps  # (B, D)
```

这对应 $q_{t|0}(x_0^d | \tilde x^d)$，即在给定 $x_0=x_0^d$ 的条件下，恰好处于 $x_{\text{tilde}}^d$ 的概率。

```python
inner_log_sig = torch.log(
    (p0t_sig / qt0_denom_sig) @ qt0_numer_sig + eps
)  # (B, D, S)
```

展开：

$$
\text{inner\_log}[b,d,s] = \log \left( \sum_{x_0'} \frac{p_{0|t}^\theta(x_0'|\tilde x)}{q_{t|0}(x_0'|\tilde x)} \cdot q_{t|0}(s|x_0') \right)
$$

这正是 $\log \hat R_t^\theta(\tilde x, s)$ 内部的结构——反向速率中关于 $x_0$ 求和的部分。

### 8.2 Rate × 概率比率的乘积

```python
outer_rate_sig = rate[
    arange(B).repeat_interleave(D*S),
    arange(S).repeat(B*D),
    x_tilde.long().flatten().repeat_interleave(S)
].view(B, D, S)
```

这提取 $R_t(s, \tilde x)$——从状态 $s$ 跳到 $\tilde x$ 的前向速率。注意 `rate[b, s, x_tilde]` 的第一个下标是源状态 s，第二个是目标状态 x_tilde。

### 8.3 Sig 项的主求和

```python
outer_sum_sig = torch.sum(
    x_tilde_mask * outer_rate_sig * (outer_qt0_numer_sig / outer_qt0_denom_sig.view(B,D,1)) * inner_log_sig,
    dim=(1,2)
)  # (B,)
```

展开为数学公式：

$$
\text{outer\_sum}[b] = \sum_{d=1}^D \sum_{s \neq \tilde x^d} \underbrace{R_t(s, \tilde x^d)}_{\text{outer\_rate}} \cdot \frac{q_{t|0}(x_0^d | s)}{q_{t|0}(x_0^d | \tilde x^d)} \cdot \underbrace{\log \hat R_t^\theta(\tilde x^d, s)}_{\text{inner\_log}}
$$

这与论文中 sig 项的被积函数完全一致。

### 8.4 归一化常数 Z

`lib/losses/losses.py:184-227`:

```python
# 每行的总和 = 总出率 Z_t(s) = Σ_{s'≠s} R_t(s, s')
rate_row_sums = - rate[
    arange(B).repeat_interleave(S),
    arange(S).repeat(B),
    arange(S).repeat(B)
].view(B, S)
# 等价的: rate_row_sums[b, s] = -rate[b, s, s]
# 由于 R_t 的行和为 0，-R[s,s] = Σ_{s'≠s} R[s,s']
```

```python
# 每个像素的出率
base_Z_tmp = rate_row_sums[
    arange(B).repeat_interleave(D),
    x_tilde.long().flatten()
].view(B, D)
# base_Z_tmp[b, d] = Z_t(x_tilde^d) = Σ_{s'} R_t(x_tilde^d, s')

# 全局归一化常数 (per sample)
base_Z = torch.sum(base_Z_tmp, dim=1)  # (B,) = Σ_d Z_t(x_tilde^d)
```

归一化常数 Z 涉及复杂的索引操作：

```python
Z_sig_norm = base_Z.view(B, 1, 1) - \
    Z_subtraction.view(B, D, 1) + \
    Z_addition.view(B, 1, S)
# (B, D, S)  外积风格: base_Z 减去 x_tilde 部分, 加上 s 部分
```

这对应：

$$
\mathcal{Z}^{t,d}(s) = \sum_{d'} \mathcal{Z}^t(\tilde x^{d'}) - \mathcal{Z}^t(\tilde x^d) + \mathcal{Z}^t(s)
$$

即替换当前像素 d 的出率 $\mathcal{Z}^t(\tilde x^d)$ 为目标状态 s 的出率 $\mathcal{Z}^t(s)$，而其他像素保持不变。

最终的 sig 项：

```python
sig_norm = torch.sum(
    (rate_sig_norm * qt0_sig_norm_numer * x_tilde_mask) /
    (Z_sig_norm * qt0_sig_norm_denom.view(B,D,1)),
    dim=(1,2)
)

sig_mean = torch.mean(-outer_sum_sig / sig_norm)
```

这与论文 Eq. 14 完全一致。

---

## 9. 辅助 NLL 与最终损失

`lib/losses/losses.py:242-246`:

```python
perm_x_logits = torch.permute(x_logits, (0, 2, 1))  # (B, S, D)
nll = self.cross_ent(perm_x_logits, minibatch.long())  # 标量
```

辅助 NLL（论文 §D）是一层直接的监督：要求模型在给定 $x_t$ 时预测 $x_0$。`CrossEntropyLoss` 期望输入 `(B, S, D)` 和目标 `(B, D)`。

```python
return neg_elbo + self.nll_weight * nll
```

`nll_weight = 0.001` 使其仅作为辅助信号。在训练早期，当 CT-ELBO 的梯度还不够稳定时，NLL 项提供直接的去噪目标。

**为什么要保留如此小的 NLL 权重**：如果 NLL 权重过大，模型会退化为"一步去噪器"，忽略连续时间动态。CT-ELBO 才是核心损失。

---

## 10. 单次前向优化 (One Forward Pass)

论文 §C.4 提出的计算优化：

```python
if self.one_forward_pass:   # cfg.loss.one_forward_pass = True
    x_logits = model(x_tilde, ts)    # 只做一次前向
    p0t_reg = F.softmax(x_logits, dim=2)  # 用于 reg 项
    p0t_sig = F.softmax(x_logits, dim=2)  # 用于 sig 项 (与 reg 项共享)
    reg_x = x_tilde
else:
    x_logits_reg = model(x_t, ts)       # 两次独立前向
    x_logits_sig = model(x_tilde, ts)
```

**常规做法**：reg 项使用 $x_t$ 做条件，sig 项使用 $\tilde x$ 做条件。这需要两次模型前向。

**单次前向技巧**：reg 项和 sig 项都使用 $\tilde x$ 做条件，共用一次模型前向。这会引入小的偏差（reg 项理论上应该在 $x_t$ 上计算），但实验证明不影响最终性能。

计算量对比（B=128, UNet 约 7.4M 参数）：

| | 常规做法 | 单次前向 |
|--|---------|---------|
| 模型前向次数 | 2 | 1 |
| 一次前向的 FLOPs | ~28 GFLOPs | ~14 GFLOPs |
| 总损失计算 | ~56 GFLOPs | ~14 GFLOPs |

**节省 50% 计算量**。

---

## 11. 配置参数的影响

来自 `config/train/cifar10.py`:

| 参数 | 值 | 影响 |
|------|-----|------|
| `loss.name` | `GenericAux` | 使用无条件 CT-ELBO（非条件版） |
| `loss.eps_ratio` | `1e-9` | 所有除法中的数值稳定常数 |
| `loss.nll_weight` | `0.001` | 辅助交叉熵权重 |
| `loss.min_time` | `0.01` | 避开 $t=0$ 奇点 |
| `loss.one_forward_pass` | `True` | 节省 50% 计算量 |

`eps_ratio = 1e-9` 的作用：

```python
qt0_denom = qt0[..., reg_x] + self.ratio_eps
# 避免 q_{t|0} 中某些条目为零导致的除零
```

当 $t$ 很小或 $x_t$ 远离 $x_0$ 时，$q_{t|0}(x_t|x_0)$ 可能接近零。`eps` 防止分母为零。

### 11.1 监控指标

训练过程中 TensorBoard 记录：

```python
writer.add_scalar('sig', sig_mean.detach(), state['n_iter'])
writer.add_scalar('reg', reg_mean.detach(), state['n_iter'])
```

- `sig`: 通常为负值，绝对值越大表示学习信号越强
- `reg`: 正值，保持在合理范围内防止过拟合
- `loss = sig + reg + 0.001 * nll`: 整体 ELBO 估计

### 11.2 调试技巧

如果在训练中观察到 loss 异常：

```
NaN loss:
  → 检查 eps_ratio 是否太小
  → 检查 min_time 是否太小 (导致 qt0 对角化过度)
  → 检查 rate_sigma 是否太大 (导致速率爆炸)

reg 项远大于 sig 项:
  → 模型过度正则化, 检查 nll_weight
  → 可能的学习率过大

sig 项接近 0:
  → 模型停止学习, 检查 min_time 是否过大
  → 检查数据归一化
```
