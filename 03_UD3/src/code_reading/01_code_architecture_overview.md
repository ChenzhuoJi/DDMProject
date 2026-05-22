# 01 — UD3 代码架构总览

## 文件结构

USD3 (Unified Discrete Diffusion for Categorical Data) 的实现仅一个文件：

```
discrete_diffusion.py  (699 行)
```

该文件不含任何神经网络定义，仅包含扩散**过程**的逻辑。用户需自行提供 `denoising_fn`（PyTorch 网络）。

## 代码布局

```
1. 概率辅助函数     (8–26 行)   — 采样、softmax 工具
2. 索引辅助函数     (28–62 行)  — tensor gather / scatter
3. 噪声调度         (64–118 行) — 6 种调度类型
4. UnifiedDiscreteDiffusion (120–699 行) — 主类
   ├── 前向过程     (q 分布)
   ├── 反向过程     (p 分布，含概率空间与对数空间)
   ├── 损失：离散时间 (ELBO + CE)
   ├── 损失：连续时间 (ELBO + CE)
   ├── MCMC 校正器  (类似 Gibbs 的细化解)
   └── 采样循环     (T → 0 祖先采样)
```

## 核心设计：离散与连续时间统一

`num_steps` 参数控制时间模式：

| `num_steps` | 时间模式 | 使用损失                     |     |
| ----------- | ---- | ------------------------ | --- |
| `0`         | 连续时间 | `continuous_time_loss()` |     |
| `> 0`       | 离散时间 | `discrete_time_loss()`   |     |

这一统一是 UD3 论文的核心贡献之一。

## 噪声分布 `m`

`m` 参数表示**稳态噪声分布**，有三种传参方式：

- `None` → 在 `num_classes` 上均匀分布
- 1D 张量 `(C,)` → 所有位置共享
- 完整张量 `(B, ..., C)` → 逐位置噪声

它出现在每个前向、反向、损失函数中，同时也作为先验 `p(x_T)`。

## 噪声调度（6 种）

| 调度类型          | 关键公式                                                          | 适用场景    |
| ------------- | ------------------------------------------------------------- | ------- |
| `cosine`      | $\bar\alpha_t = \cos(\frac{t+\alpha}{1+\alpha}\frac{\pi}{2})$ | 默认，平滑衰减 |
| `exponential` | $\bar\alpha_t = \exp(a t (b^0 - b^{t/T}))$                    | 灵活速率控制  |
| `linear`      | $\bar\alpha_t = 1 - t/T$                                      | 简单基线    |
| `constant`    | $\bar\alpha_t = e^{-a t}$                                     | 均匀速率    |
| `geometric`   | $\sigma$ 插值                                                   | 方差保持    |
| `loglinear`   | $1/(1-(1-\epsilon)t)$                                         | 分数匹配常用  |

## 前向过程 (q)

| 方法            | 计算内容                           | 用途            |
| ------------- | ------------------------------ | ------------- |
| `qt_0_sample` | 采样 $x_t \sim q(x_t \mid x_0)$  | 训练数据加噪        |
| `qt_0_prob`   | $q(x_t \mid x_0)$ 完整概率         | 连续时间损失        |
| `qs_t0_prob`  | $q(x_s \mid x_t, x_0)$ (s < t) | 离散时间损失（解析 KL） |

**采样技巧**：使用**分支指示变量** $b_t \sim \text{Bernoulli}(\bar\alpha_t)$：
- $b_t = 1$ → 保留 $x_0$（未加噪）
- $b_t = 0$ → 从噪声 $m$ 中采样

这使前向采样高效且简洁。

## 反向过程 (p)

| 方法             | 计算内容                                |          |
| -------------- | ----------------------------------- | -------- |
| `ps_t_prob`    | $p_\theta(x_s \mid x_t)$（概率空间）      |          |
| `ps_t_logprob` | $p_\theta(x_s \mid x_t)$（对数空间，数值稳定） |          |
| `ps_t0_delta`  | $p_\theta - q_{s\mid t,0}$ 差值       | 用于简化 VLB |

反向步使用三个系数分解转移：

- $\mu_{t|s}$：保留当前 token 的概率
- $\lambda_{t|s}$：$x_t$ 来自 $x_0$ 而非噪声的概率
- $\gamma_{t|s}$：去噪网络预测的校正项

## 损失函数

### 离散时间 (389–429 行)
$$
\mathcal{L} = \mathbb{E}_{t\sim[1,T]}\big[ \underbrace{\text{KL}(p_\theta(x_s|x_t) \| q(x_s|x_t,x_0))}_{\text{VLB}} + \underbrace{\mathbb{1}_{t=1} \cdot (-\log p_\theta(x_0|x_1))}_{\text{最终步 CE}} \big] + \frac{1}{T}\underbrace{\text{KL}(q(x_T|x_0)\| p(x_T))}_{\text{先验}}
$$

### 连续时间 (475–525 行)
$$
\mathcal{L} = \mathbb{E}_{t\sim[0,T]}\big[ \beta_t \cdot g_\theta(x_t, t) \big]
$$

其中 $g_\theta$ 源自论文 Proposition 4，含一个涉及 $q(z_t|x_0)$ 比值的辅助项用于方差缩减。

两者均支持 `simplified_vlb` 模式（L2 近似，加速训练）。

## MCMC 校正器 (553–605 行)

采样时使用的类似 Gibbs 的细化过程：

```
for n in range(max_steps):
    fprob = denoising_fn(z_n, t)
    z_{n+1} ~ p(z | fprob, t)    # coef=2 加快混合
```

步长 $\delta_n$ 被自适应裁剪，确保停留概率 $\ge$ `min_stay_prob`。

## 采样循环 (607–698 行)

```
x_T ~ m (噪声)
for t in reversed(time_steps):
    fprob = denoising_fn(x_t, t/num_steps)
    prob_s = p_theta(x_s | x_t)   # s=0 时直接使用 fprob
    x_s ~ Cat(prob_s)
    if 启用 MCMC: x_s = mcmc_corrector(x_s, s)
return x_0
```

关键细节：
- 时间步从 `T` 到 `0` 线性等距排列
- `s=0` 时 `prob_s` 直接使用去噪网络输出（最终的干净预测）
- 离散模式下 `denoising_fn` 接收归一化时间 `t/num_steps`
- 全程支持条件掩码：被掩码位置强制保持输入值

## 整体数据流

```
训练:
  x_0 ──t──► qt_0_sample ──► x_t ──► denoising_fn ──► fprob_t ──► compute_loss ──► ∇θ
         随机 t                                           │
                                               ┌───────────┤
                                               ▼           ▼
                                          VLB 项      CE 项
                                     (KL 或 Proposition 4) ( -log fprob_t[x_0] )

采样:
  m ──► sample_categorical ──► x_T ──► for t in reverse: ──► x_0
                                       │
                                       ├── denoising_fn(x_t, t)
                                       ├── ps_t_prob → sample_categorical
                                       └── [可选] MCMC 校正器
```
