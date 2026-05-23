# 03 — 前向转移概率：从 t 到 $q(x_t \mid x_0)$

## 核心问题

给定时间步 $t$，前向过程定义了一个转移概率 $q(x_t \mid x_0)$，描述干净数据 $x_0$ 被噪声破坏为 $x_t$ 的概率分布。这个分布是**逐位置独立**的：每个 token 在给定 $x_0$ 的条件下独立转移。

## 1. 从 t 到 $\bar\alpha_t$

转移概率的唯一时间相关参数是累积保留概率 $\bar\alpha_t$。计算链路为：

```
t (标量或 B 维向量)
  │
  ▼
step_to_time(t)         # 将离散步映射到连续时间
  │  t_cont = t / N * Tmax (离散模式)
  │  t_cont = t           (连续模式)
  ▼
schedule_fn(t_cont)      # 根据调度类型计算
  │
  ▼
alphabar_t               # (B,) 张量, 值域 (0, 1]
```

### 调度函数的具体形式

| 调度类型                | $\bar\alpha_t$ 公式                                                                               | 关键性质                                         |
| ------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------- |
| cosine              | $\frac{\cos(\frac{t/T + a}{1+a} \cdot \frac{\pi}{2})}{\cos(\frac{a}{1+a} \cdot \frac{\pi}{2})}$ | $\bar\alpha_0 = 1$，$\bar\alpha_T \to 0$，平滑衰减 |
| exponential         | $\exp\left(a \cdot t \cdot (1 - b^{t/T})\right)$                                                | 可调衰减速率                                       |
| linear              | $1 - t/T$                                                                                       | $\bar\alpha_T = 0$                           |
| constant            | $e^{-a t}$                                                                                      | 指数衰减                                         |
| geometric/loglinear | 见 `noise_schedule`                                                                              | 特殊用途                                         |

**实现位置**：`discrete_diffusion.py:65-118`，函数 `noise_schedule`。

### 关键保证

$$
\bar\alpha_0 = 1, \quad \bar\alpha_T \approx 0, \quad \bar\alpha_t \in (0, 1], \quad \bar\alpha_t \text{ 关于 } t \text{ 单调递减}
$$

代码中通过 `torch.clip(alphabar_t, min=min_alphabar, max=1-min_alphabar)` 保证数值稳定性。

## 2. 从 $\bar\alpha_t$ 到转移概率

UD3 定义转移概率的核心公式为：

$$
q(x_t \mid x_0) = \bar\alpha_t \cdot \delta(x_t, x_0) + (1 - \bar\alpha_t) \cdot m(x_t)
$$

其中：

- $\delta(x_t, x_0)$：Kronecker delta，当 $x_t = x_0$ 时为 1，否则为 0
- $m(x_t)$：稳态噪声分布在 $x_t$ 处的概率
- $\bar\alpha_t$：**控制噪声强度的唯一参数**

### 逐类别展开

对类别 $c \in \{1, \dots, C\}$：

$$
q(x_t = c \mid x_0) =
\begin{cases}
\bar\alpha_t + (1 - \bar\alpha_t) \cdot m[c], & c = x_0 \\[4pt]
(1 - \bar\alpha_t) \cdot m[c], & c \neq x_0
\end{cases}
$$

### 物理含义

- $\bar\alpha_t \to 1$（$t$ 接近 0）：$x_t$ 几乎必然等于 $x_0$，噪声项可忽略
- $\bar\alpha_t \to 0$（$t$ 接近 $T$）：$x_t$ 完全由噪声分布 $m$ 决定，与 $x_0$ 无关
- 中间 $t$：两者混合，$\bar\alpha_t$ 决定保留与替换的比例

## 3. 三种噪声分布 $m$ 的处理

$m$ 的三种形式影响转移概率的具体计算：

| $m$ 的形式        | 含义    | $q(x_t \mid x_0)$ 具体形式                                                        |
| -------------- | ----- | ----------------------------------------------------------------------------- |
| `None`         | 均匀噪声  | $\bar\alpha_t \cdot \delta_{x_0} + (1-\bar\alpha_t) \cdot \frac{1}{C}$        |
| 1D `(C,)`      | 全局共享  | $\bar\alpha_t \cdot \delta_{x_0} + (1-\bar\alpha_t) \cdot m[x_t]$             |
| ND `(B,...,C)` | 逐位置依赖 | $\bar\alpha_t \cdot \delta_{x_0} + (1-\bar\alpha_t) \cdot m[\text{pos}, x_t]$ |

## 4. 代码实现：概率路径 vs 采样路径

### 4a. 概率路径：`qt_0_prob`（第 235-254 行）

计算完整的 $C$ 维概率向量，用于连续时间损失：

```python
def qt_0_prob(self, x_0, t, m=None):
    alphabar_t, _ = self.get_alphabar_beta(t)   # t → alphabar_t
    prob = (1 - alphabar_t) * m                  # 所有类别初始化为 (1-α_t)·m
    prob = set_last_dim(prob, x_0,               # 在 x_0 位置累加 α_t
           value=alphabar_t, inplace_add=True)
    return prob                                  # (B, N1, ..., Nk, C)
```

**两步构造**精确对应数学公式：

1. `prob = (1 - alphabar_t) * m`：所有类别设置为 $(1-\bar\alpha_t) \cdot m[c]$
2. `set_last_dim(prob, x_0, alphabar_t, inplace_add=True)`：在 $x_0$ 位置加上 $\bar\alpha_t$，得到 $(1-\bar\alpha_t) \cdot m[x_0] + \bar\alpha_t$

### 4b. 采样路径：`qt_0_sample`（第 206-233 行）

**不构造概率向量，而是直接采样 $x_t$**，用于训练加噪：

```python
def qt_0_sample(self, x_0, t, m=None):
    alphabar_t, _ = self.get_alphabar_beta(t)   # t → alphabar_t
    m0 = sample_from_noise(m)                    # 从 m 采样一个候选
    bt = sample_bernoulli(alphabar_t)            # bt ~ Bernoulli(ᾱ_t)
    sample = torch.where(bt, x_0, m0)            # bt=1 保留 x_0, bt=0 替换为 m0
    return sample
```

**分支指示变量技巧**：

$$
b_t \sim \text{Bernoulli}(\bar\alpha_t), \qquad
x_t = b_t \cdot x_0 + (1 - b_t) \cdot m_0
$$

等价于 $x_t \sim q(x_t \mid x_0)$，但避免了构造 $C$ 维概率张量。

### 4c. 等价性证明

对任意类别 $c$：

$$
\begin{aligned}
P(x_t = c \mid x_0)
&= P(b_t=1) \cdot P(x_t = c \mid b_t=1, x_0) + P(b_t=0) \cdot P(x_t = c \mid b_t=0) \\
&= \bar\alpha_t \cdot \delta_{c, x_0} + (1-\bar\alpha_t) \cdot m[c]
\end{aligned}
$$

与公式完全一致。

## 5. 完整数据流总结

```
输入: t (标量或 (B,) 向量), x_0 (干净数据), m (噪声分布)
───
步骤1: 调用 noise_schedule(t)
         → 计算 alphabar_t
         → 若离散模式 (num_steps > 0): t_cont = t / N * Tmax
         → 若连续模式 (num_steps = 0): t_cont = t
         → 代入调度函数返回 alphabar_t

步骤2: 使用 alphabar_t
         ┌── 训练加噪: qt_0_sample
         │      alphabar_t → Bernoulli(alphabar_t) → bt
         │      x_t = bt ? x_0 : m0
         │
         └── 概率计算: qt_0_prob
                prob[:, x_0] = alphabar_t + (1-alphabar_t) * m[:, x_0]
                prob[:, c≠x_0] = (1-alphabar_t) * m[:, c]

输出: x_t (采样路径) 或 prob (概率路径)
───
核心公式: q(x_t | x_0) = ᾱ_t · δ(x_t, x_0) + (1-ᾱ_t) · m(x_t)
```

## 6. 与论文的对应

| 代码实体 | 论文符号 | 含义 |
|---------|---------|------|
| `noise_schedule(t)` → `alphabar_t` | $\bar\alpha_t = \prod_{i=1}^t (1-\beta_i)$ | 累积保留概率，控制噪声强度 |
| `sample_bernoulli(alphabar_t)` → `bt` | $b_t \sim \text{Bernoulli}(\bar\alpha_t)$ | 分支指示变量 |
| `qt_0_prob` 的返回值 | $q(x_t \mid x_0)$ | 精确转移概率 |
| `qt_0_sample` 的返回值 | $x_t \sim q(x_t \mid x_0)$ | 前向加噪样本 |
| $m$ | $m$ | 稳态噪声分布，$p(x_T)$ 的先验 |
