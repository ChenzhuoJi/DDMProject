# 06 — 前向 CTMC：GaussianTargetRate 数学原理与源码实现

> 对应论文 §4.1，代码 `lib/models/models.py:192-267`
>
> 从数学定义到逐行代码，解析前向速率矩阵的设计与计算

## 目录

- [1. 设计目标](#1-设计目标)
- [2. 时间无关的基速率矩阵 \(R_b\)](#2-时间无关的基速率矩阵-r_b)
  - [2.1 高斯偏好掩码](#21-高斯偏好掩码)
  - [2.2 细节平衡修正](#22-细节平衡修正)
  - [2.3 对角化收尾](#23-对角化收尾)
- [3. 时间调度函数 \(\beta(t)\)](#3-时间调度函数-beta_t)
- [4. 全速率矩阵 \(R_t = \beta(t) R_b\)](#4-全速率矩阵-r_t--beta_t-r_b)
- [5. 转移概率矩阵 \(q_{t\|0}\) 的解析计算](#5-转移概率矩阵-q_t0-的解析计算)
  - [5.1 可交换性条件](#51-可交换性条件)
  - [5.2 矩阵指数与特征分解](#52-矩阵指数与特征分解)
  - [5.3 数值稳定性处理](#53-数值稳定性处理)
- [6. 三种前向过程对比](#6-三种前向过程对比)
- [7. 完整计算示例](#7-完整计算示例)

---

## 1. 设计目标

CTMC 框架需要一个**连续时间、离散状态空间**的前向扩散过程。具体要求：

1. **速率矩阵 $R_t$ 必须被显式定义**，因为 CT-ELBO 和 Tau-Leaping 采样都需要它
2. **转移概率 $q_{t|0}$ 必须有闭式解**，否则无法高效训练
3. **$q_1$ 应接近均匀分布**，从而可以从简单先验采样开始反向过程
4. $R_t$ 与 $R_{t'}$ 对任意 $t, t'$ **可交换**（commute），保证矩阵指数计算可行

`GaussianTargetRate` 的设计围绕这些目标展开。整体策略是**将速率分解为时间部分和状态部分**：

$$
R_t = \beta(t) \cdot R_b
$$

其中 $R_b$ 是**时间无关的基速率矩阵**，$\beta(t)$ 是标量时间调度函数。所有设计工作集中在 $R_b$ 的构造上。

---

## 2. 时间无关的基速率矩阵 $R_b$

`lib/models/models.py:201-218`:

```python
rate = np.zeros((S, S))             # S = 256
vals = np.exp(-np.arange(0, S)**2 / (self.rate_sigma**2))
```

### 2.1 高斯偏好掩码

核心思想：**数值相近的状态之间应有更高的转移速率**。对于像素值，这意味着 $127 \to 128$ 的速率远高于 $0 \to 255$。

```python
for i in range(S):
    for j in range(S):
        if i < S // 2:                    # 上半区 (状态值较小)
            if j > i and j < S - i:       # 对称区间内
                rate[i, j] = vals[j - i - 1]
        elif i > S // 2:                  # 下半区 (状态值较大)
            if j < i and j > -i + S - 1:  # 对称区间内
                rate[i, j] = vals[i - j - 1]
```

这段代码对 $R_b$ 的非对角元施加了一个三角掩码。以 $S=256$ 为例，$R_b[i,j]$ 非零的区域是：

```
      j → 0  63  127 128 191 255
i ↓
0          ████████████████           ← 前一半行: j > i 且 j < S-i
63         ████████████████             即 j ∈ (i, S-i-1]
127                     █               i=127 时只有 j=128
128                   █                 ← 后一半行: j < i 且 j > S-i-1
191         ████████████████            即 j ∈ (S-i-1, i)
255 ████████████████
```

这种构造的结果是：**$R_b$ 只允许状态向"中间值"移动**。上三角中非零元素随 $|i-j|$ 增大按高斯核衰减：

```python
vals[d] = exp(-d² / rate_sigma²)
```

其中 `d = j - i - 1`（上三角）或 `d = i - j - 1`（下三角），`rate_sigma = 6.0`。这意味着：

| 距离 d | 相对速率 $e^{-d^2/36}$ |
|--------|----------------------|
| 0 | 1.000 |
| 1 | 0.973 |
| 2 | 0.895 |
| 3 | 0.779 |
| 5 | 0.500 |
| 10 | 0.062 |

**直观理解**：`rate_sigma` 控制每次跳跃的"胆量"。$\sigma=6$ 意味着状态 $i$ 主要向 $i\pm3$ 范围内跳，跨越 10 个以上等级的跳跃几乎不会发生。

### 2.2 细节平衡修正

高斯掩码构造的矩阵是不对称的，因为它只填充了非零区域的上/下三角。为了满足**物理上合理的平稳分布**，必须施加**细节平衡**条件：

$$
\pi(i) \cdot R_b[i,j] = \pi(j) \cdot R_b[j,i]
$$

代码通过以下循环实现：

```python
for i in range(S):
    for j in range(S):
        if rate[j, i] > 0.0:
            rate[i, j] = rate[j, i] * np.exp(
                -((j+1)**2 - (i+1)**2 + S*(i+1) - S*(j+1)) / (2 * Q_sigma**2)
            )
```

这段代码的数学来源是：假设平稳分布为

$$
\pi(k) \propto \exp\left(-\frac{(k - S/2)^2}{2 Q_\text{sigma}^2}\right)
$$

即中心在 $S/2 = 128$、宽度 $Q_\text{sigma} = 512$ 的高斯分布。那么细节平衡条件给出：

$$
\frac{R_b[i,j]}{R_b[j,i]} = \frac{\pi(j)}{\pi(i)} = \exp\left(-\frac{(j - S/2)^2 - (i - S/2)^2}{2 Q_\text{sigma}^2}\right)
$$

展开后：

$$
(j - S/2)^2 - (i - S/2)^2 = (j^2 - i^2) - S(j - i) = (j+1)^2 - (i+1)^2 + S(i+1) - S(j+1)
$$

后一个等号成立是因为在校正 $0$-indexing（Python 数组从 0 开始，而数学公式从 1 开始）。

**物理含义**：

- `Q_sigma = 512` 远大于状态空间大小 256，因此平稳分布几乎是均匀的
- 但严格来说，平稳分布在 $k=128$ 附近有微弱的钟形峰
- 这保证了 $t=1$ 时的噪声分布 $q_1$ 接近均匀但略偏向中间灰度值，符合"自然图像像素值集中在中间范围"的先验

### 2.3 对角化收尾

```python
rate = rate - np.diag(np.diag(rate))       # 清除前面可能产生的对角元
rate = rate - np.diag(np.sum(rate, axis=1)) # 行和置零: R[i,i] = -Σ_{j≠i} R[i,j]
```

这是速率矩阵的核心约束：**每行之和必须为零**。因为概率质量守恒：

$$
\sum_j R[i,j] = 0 \quad \Longrightarrow \quad R[i,i] = -\sum_{j \neq i} R[i,j]
$$

构造完成后进行特征分解：

```python
eigvals, eigvecs = np.linalg.eig(rate)
inv_eigvecs = np.linalg.inv(eigvecs)

self.base_rate = torch.from_numpy(rate).float()
self.eigvals = torch.from_numpy(eigvals).float()
self.eigvecs = torch.from_numpy(eigvecs).float()
self.inv_eigvecs = torch.from_numpy(inv_eigvecs).float()
```

`np.linalg.eig`（而非对称的 `eigh`）是因为 $R_b$ 被细节平衡修正后**不再对称**。特征分解在 `__init__` 中完成，后续 `rate()` 和 `transition()` 调用时直接使用预计算的结果。

---

## 3. 时间调度函数 $\beta(t)$

`lib/models/models.py:228-236`:

```python
def _integral_rate_scalar(self, t):
    return self.time_base * (self.time_exponential ** t) - self.time_base

def _rate_scalar(self, t):
    return self.time_base * math.log(self.time_exponential) * (self.time_exponential ** t)
```

这是 $\beta(t)$ 及其积分：

$$
\beta(t) = T_b \cdot \ln(T_e) \cdot T_e^{\,t}
$$

$$
\int_0^t \beta(s) \, ds = T_b \cdot (T_e^{\,t} - 1)
$$

代入配置值 $T_b = 3.0$, $T_e = 100.0$：

| t | $\beta(t)$ | $\int_0^t \beta(s) ds$ |
|---|-----------|----------------------|
| 0.0 | 13.82 | 0.0 |
| 0.25 | 43.49 | 3.0 |
| 0.5 | 136.9 | 13.82 |
| 0.75 | 430.8 | 46.49 |
| 1.0 | 1356.0 | 138.2 |

**设计原理**：

- $\beta(t)$ 随时间**指数增长**：早期 $t \to 0$ 变化慢，后期 $t \to 1$ 变化快
- 这对应于"先探索局部、后扩散全局"的物理过程
- 在 $t=0$ 附近，速率很小，状态主要在局部跳跃（高斯核约束）
- 在 $t=1$ 附近，速率很大，状态可以快速跨越整个空间
- `time_base` 控制整体速率量级，`time_exponential` 控制增长曲率

与常用的方差调度类比：

| 扩散模型 | 时间调度 | 增长率 |
|---------|---------|--------|
| DDPM (离散) | $\beta_t$ 线性/余弦 | 线性 |
| score-SDE (连续) | $\beta(t)$ 指数 | 指数 |
| **CTMC (本代码)** | $\beta(t) = T_b \ln T_e \cdot T_e^t$ | **指数** |

---

## 4. 全速率矩阵 $R_t = \beta(t) R_b$

`lib/models/models.py:238-244`:

```python
def rate(self, t):
    B = t.shape[0]       # batch 大小
    S = self.S

    rate_scalars = self._rate_scalar(t)            # (B,)

    return self.base_rate.view(1, S, S) * rate_scalars.view(B, 1, 1)
    # 返回 (B, S, S)
```

展开为数学公式：

$$
R_t = \beta(t) \cdot R_b
$$

$$
R_t[i,j] = \beta(t) \cdot R_b[i,j] \quad \text{对于每个 batch 样本}
$$

等价地说，速率矩阵随时间**整体缩放**，但状态之间的**相对偏好不变**。

---

## 5. 转移概率矩阵 $q_{t|0}$ 的解析计算

### 5.1 可交换性条件

前向 CTMC 的转移概率 $q_{t|0}$ 满足 Kolmogorov 前向方程：

$$
\frac{d}{dt} q_{t|0} = q_{t|0} \cdot R_t
$$

如果 $R_t$ 与 $R_{t'}$ 对所有 $t, t'$ **可交换**（即 $R_{t_1} R_{t_2} = R_{t_2} R_{t_1}$），则有闭式解：

$$
q_{t|0} = \exp\left(\int_0^t R_s \, ds\right)
$$

由于 $R_t = \beta(t) R_b$，而 $\beta(t)$ 是标量，可交换性自然得到满足。

### 5.2 矩阵指数与特征分解

`lib/models/models.py:246-267`:

```python
def transition(self, t):
    B = t.shape[0]
    S = self.S

    integral_rate_scalars = self._integral_rate_scalar(t)   # (B,)

    adj_eigvals = integral_rate_scalars.view(B, 1) * self.eigvals.view(1, S)
    # (B, S)  每个 batch 样本的积分值 × 每个特征值

    transitions = self.eigvecs.view(1, S, S) @ \
        torch.diag_embed(torch.exp(adj_eigvals)) @ \
        self.inv_eigvecs.view(1, S, S)
    # (B, S, S)

    return transitions
```

数学推导：

$$
q_{t|0} = \exp\left(\int_0^t \beta(s) ds \cdot R_b\right)
$$

对 $R_b$ 进行特征分解 $R_b = Q \Lambda Q^{-1}$，则：

$$
q_{t|0} = Q \cdot \exp\left(\int_0^t \beta(s) ds \cdot \Lambda\right) \cdot Q^{-1}
$$

其中 $\exp(\cdot)$ 是逐元素的标量指数（因为 $\Lambda$ 是对角矩阵）：

$$
\exp\left(\int_0^t \beta(s) ds \cdot \Lambda\right) = \text{diag}\left(e^{\lambda_1 \int \beta}, e^{\lambda_2 \int \beta}, \dots, e^{\lambda_S \int \beta}\right)
$$

代码中的实现完全对应：

```
integral_rate_scalars  →  ∫₀ᵗ β(s) ds
adj_eigvals             →  ∫₀ᵗ β(s) ds · Λ      (逐元素乘法)
torch.exp(adj_eigvals)  →  diag(exp(∫β · Λ))
eigvecs @ diag(exp) @ inv_eigvecs  →  Q · exp(∫β · Λ) · Q⁻¹
```

### 5.3 数值稳定性处理

```python
if torch.min(transitions) < -1e-6:
    print(f"[Warning] GaussianTargetRate, large negative transition values ...")

transitions[transitions < 1e-8] = 0.0
```

浮点误差可能导致极小负值。代码在模型初始化时一次性完成特征分解（`np.linalg.eig` + `np.linalg.inv`），运行时只需 O(B·S²) 的矩阵乘法，避免了每次调用都做昂贵的矩阵指数计算。

**S=256 时的计算量**：

| 操作 | 复杂度 | 频率 |
|------|--------|------|
| 特征分解 | O(S³) = 16.8M | 一次 (`__init__`) |
| `transition()` | O(B·S²) = B × 65.5K | 每步 loss + 每步采样 |

---

## 6. 三种前向过程对比

代码中实现了三种 `rate/transition` 类，`GaussianTargetRate` 是 CIFAR-10 的默认选择：

| 特性 | GaussianTargetRate | UniformRate | BirthDeathForwardBase |
|------|-------------------|-------------|----------------------|
| **代码位置** | `models.py:192-267` | `models.py:155-190` | `models.py:99-153` |
| **基速率 $R_b$** | 高斯偏好 + 细节平衡 | 全连接常数速率 | 仅相邻状态 (1D 生灭) |
| $R_b$ **形态** | 满矩阵（大多数元素 >0） | 满矩阵 | 三对角 |
| **细节平衡** | ✅ 显式满足 | ✅ (对称矩阵自然满足) | ✅ (对称矩阵) |
| **平稳分布** | $N(S/2, Q_\text{sigma}^2)$ | 均匀 | 均匀 |
| **跳跃模式** | 偏好相近值 | 等概率跳所有值 | 只能 ±1 |
| **适用场景** | CIFAR-10 图像 | Piano (序列) | Piano 消融 |
| **特征分解方法** | `np.linalg.eig` | `np.linalg.eigh` | `np.linalg.eigh` |

```
GaussianTargetRate (S=256):
  跳跃偏好: 127 ↔ 128 (高), 0 ↔ 255 (无)
  视觉:    "局部探索 → 逐步扩散"

UniformRate (S=128, 钢琴):
  跳跃偏好: 所有状态等概率
  视觉:    "全局噪声"

BirthDeath (S=128, 钢琴):
  跳跃偏好: 只能 i → i±1
  视觉:    "梯度下降式漂移"
```

---

## 7. 完整计算示例

以 CIFAR-10 配置为例，追踪 $t=0.5$ 时一步 `transition` 的计算：

```python
# 输入:
t = torch.tensor([0.5])  # (1,)

# 1. 积分标量
integral_scalar = time_base * (time_exponential ** 0.5 - 1)
                = 3.0 * (100.0 ** 0.5 - 1)
                = 3.0 * (10.0 - 1)
                = 27.0

# 2. 调整特征值
adj_eigvals = 27.0 * eigvals  # (256,) 逐元素乘

# 3. 矩阵指数
exp_diag = diag(exp(adj_eigvals))  # (256, 256) 对角阵

# 4. 重构 q_{t|0}
qt0 = eigvecs @ exp_diag @ inv_eigvecs  # (256, 256)

# qt0[i, j] = 从状态 i 出发，t=0.5 时处于状态 j 的概率
# 当 i=128 时: qt0[128, :] 是以 128 为中心的宽高斯状分布
# 当 i=0 时:   qt0[0, :] 也是以 128 为中心 (因为平稳分布偏向中间)
```

速率矩阵在同一时刻的值：

```python
# 5. 速率标量
rate_scalar = time_base * ln(time_exponential) * (time_exponential ** 0.5)
             = 3.0 * ln(100.0) * 10.0
             = 3.0 * 4.605 * 10.0
             = 138.15

# 6. 全速率
Rt = rate_scalar * base_rate  # (256, 256)
# Rt[127, 128] ≈ 138.15 × 1.0         (相邻值, 最大速率)
# Rt[0, 1]     ≈ 138.15 × e^{-1/36}   (边缘值, 稍小)
# Rt[0, 255]   = 0                     (跨度过大, 不允许)
```

两者关系验证：

$$
q_{0.5|0} = \exp(27.0 \cdot R_b) \neq 27.0 \cdot R_b \ (\text{注意: 这是矩阵指数, 不是标量乘})
$$

直观上，$q_{t|0}$ 是 $R_t$ 在时间区间上的累积效应，而非瞬时速率本身。
