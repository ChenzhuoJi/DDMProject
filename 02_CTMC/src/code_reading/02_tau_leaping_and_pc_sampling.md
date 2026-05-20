# 02 — Tau-Leaping + Predictor-Corrector 采样精读

> CTMC 论文代码精读 · 反向采样
>
> 对应论文 §4.3–§4.5 + Appendices F/G

## 目录

- [1. 采样配置与入口](#1-采样配置与入口)
- [2. 独立采样脚本 `scripts/sample.py`](#2-独立采样脚本-scriptssamplepy)
- [3. TauLeaping 采样器](#3-tauleaping-采样器)
- [4. PCTauLeaping 采样器](#4-pctauleaping-采样器)
- [5. 条件采样器 ConditionalTauLeaping / ConditionalPCTauLeaping](#5-条件采样器)
- [6. 速率矩阵的实现](#6-速率矩阵的实现)
- [7. 论文 §4.5 误差界与代码的关系](#7-论文-45-误差界与代码的关系)
- [8. 论文 ↔ 代码对照表](#8-论文--代码对照表)

---

## 1. 采样配置与入口

### 1.1 评估配置

**文件**: `config/eval/cifar10.py`

```python
config.sampler.name = 'PCTauLeaping'       # TauLeaping / PCTauLeaping
sampler.num_steps = 500                     # 总步数 (tau = T / num_steps)
sampler.min_t = 0.01                        # 最小时间 (避免 t=0)
sampler.eps_ratio = 1e-9                    # 数值稳定性
sampler.initial_dist = 'gaussian'           # 初始分布类型
sampler.num_corrector_steps = 10            # 每步 corrector 次数
sampler.corrector_step_size_multiplier = 1.5  # corrector 步长倍率
sampler.corrector_entry_time = 0.1          # 何时开始 corrector (t < 0.1T)
```

**关键参数**:

| 参数 | 代码变量 | 论文符号 | 含义 |
|------|---------|---------|------|
| `num_steps` | `h = 1/num_steps` | $\tau$ | tau-leaping 步长 (步数=1000→$\tau=0.001$) |
| `min_t` | `min_t` | $t_{\min}$ | 终止时间，避免 $t=0$ 奇点 |
| `initial_dist` | — | $p_{\text{ref}}$ | $t=T$ 时的先验分布 (uniform/gaussian) |
| `num_corrector_steps` | — | $N_c$ | 每个 predictor 步后的 corrector 次数 |
| `corrector_entry_time` | — | $t_{\text{entry}}$ | 从此时间开始启用 corrector |

**Piano 配置** (`config/eval/piano.py`) 类似，但额外有 `condition_dim` 和 `reject_multiple_jumps` 参数。

### 1.2 采样器注册

采样器通过注册表模式按名称实例化:

```python
# sampling_utils.py
_SAMPLERS = {}

def register_sampler(cls):
    _SAMPLERS[cls.__name__] = cls
    return cls

def get_sampler(cfg):
    return _SAMPLERS[cfg.sampler.name](cfg)
```

---

## 2. 独立采样脚本 `scripts/sample.py`

```python
def main():
    # 1. 加载配置
    eval_cfg = get_eval_config()
    train_cfg = bookkeeping.load_ml_collections(eval_cfg.train_config_path)
    
    # 2. 覆盖训练配置
    for item in eval_cfg.train_config_overrides:
        set_in_nested_dict(train_cfg, item[0], item[1])
    
    # 3. 重建模型 + 加载检查点 + 切换到评估模式 (EMA)
    model = model_utils.create_model(train_cfg, device)
    loaded_state = torch.load(eval_cfg.checkpoint_path)
    model.load_state_dict(remove_module_from_keys(loaded_state['model']))
    model.eval()   # ← 切换为 EMA shadow params
    
    # 4. 采样 50000 张图像
    sampler = sampling_utils.get_sampler(eval_cfg)
    total_samples = 0
    while total_samples < 50000:
        samples, _, _ = sampler.sample(model, batch=50, num_intermediates=1)
        samples = samples.reshape(50, 3, 32, 32).astype(np.uint8)
        for i in range(batch):
            Image.fromarray(imgtrans(samples[i])).save(f'{save_path}/{total_samples+i}.png')
        total_samples += batch
```

---

## 3. TauLeaping 采样器

**文件**: `lib/sampling/sampling.py:30-120`

### 3.1 整体流程

```
输入: 模型, 样本数 N, 中间步数 num_intermediates
1. 初始化 x ~ p_ref (uniform 或 gaussian)
2. ts = linspace(1.0, min_t, num_steps) + [0]
3. 对每个 t = ts[0], ts[1], ..., ts[-2]:
   a. 计算 q_{t|0}, R_t, p_{0|t}^θ
   b. 计算反向速率 R̂_t^θ
   c. Poisson 跳跃采样 → x_new
4. 最后一步: 用 p_{0|min_t}^θ 输出最终 x_0
```

### 3.2 初始分布采样

```python
def get_initial_samples(N, D, device, S, initial_dist, initial_dist_std=None):
    if initial_dist == 'uniform':
        x = torch.randint(low=0, high=S, size=(N, D), device=device)
    elif initial_dist == 'gaussian':
        # 以 S//2 为中心的高斯分布
        target = exp(-((arange(1, S+1) - S//2)²) / (2 * initial_dist_std²))
        target = target / sum(target)
        cat = Categorical(torch.from_numpy(target))
        x = cat.sample((N*D,)).view(N, D)
    return x
```

对 CIFAR-10 (`S=256`): Gaussian 初始分布在中心值 128 附近集中，反映像素值在 128 附近的天然分布。`initial_dist_std = model.Q_sigma = 512.0`。

### 3.3 主循环

```python
class TauLeaping:
    def sample(self, model, N, num_intermediates):
        # 准备时间网格
        ts = np.concatenate((
            np.linspace(1.0, min_t, num_steps),
            np.array([0])
        ))
        
        x = get_initial_samples(N, D, device, S, initial_dist, initial_dist_std)
        
        for idx, t in enumerate(ts[0:-1]):
            h = ts[idx] - ts[idx+1]  # tau-leaping 步长 τ
            
            # ===== ① 计算前向量 =====
            qt0 = model.transition(t * torch.ones((N,), device))     # (N, S, S)
            rate = model.rate(t * torch.ones((N,), device))          # (N, S, S)
            p0t = F.softmax(model(x, t * torch.ones((N,), device)), dim=2)  # (N, D, S)
            
            # ===== ② 计算反向速率 R̂_t^θ =====
            # R̂_t^θ(x, x̃) = R_t(x̃, x) · Σ_{x0} (p0t / q_denom) @ q_numer
            
            # q_t|0(· | x_t^d): (N, D, S)
            qt0_denom = qt0[:, :, x.flatten().repeat_interleave(S)].view(N, D, S) + eps
            
            # R_t(·, x_t^d): (N, D, S)
            forward_rates = rate[:, :, x.flatten().repeat_interleave(S)].view(N, D, S)
            
            # 内积: Σ_{x0} p0t(x0|x_t) · q_t|0(x̃|x0) / q_t|0(x_t|x0)
            inner_sum = (p0t / qt0_denom) @ qt0  # (N, D, S)
            
            # 反向速率: R̂_t^θ(x_t, x̃) = R_t(x̃, x_t) · inner_sum
            reverse_rates = forward_rates * inner_sum  # (N, D, S)
            
            # 排除对角线 (x̃ == x_t 的速率为 0)
            reverse_rates[arange(N).repeat(D), arange(D).repeat(N), x.flatten()] = 0.0
            
            # ===== ③ Tau-Leaping Poisson 跳跃 =====
            diffs = arange(S).view(1,1,S) - x.view(N,D,1)  # 每个目标状态的距离
            poisson_dist = Poisson(reverse_rates * h)       # Poisson(τ · R̂_t^θ)
            jump_nums = poisson_dist.sample()               # (N, D, S)
            
            adj_diffs = jump_nums * diffs
            overall_jump = sum(adj_diffs, dim=2)            # 所有跳跃合并
            xp = x + overall_jump
            x_new = torch.clamp(xp, min=0, max=S-1)         # 确保在 [0, S-1]
            
            x = x_new
        
        # ===== ④ 最终去噪 =====
        p_0gt = F.softmax(model(x, min_t * torch.ones((N,), device)), dim=2)
        x_0max = torch.max(p_0gt, dim=2)[1]  # argmax → 最终样本
        return x_0max.detach().cpu().numpy().astype(int)
```

### 3.4 核心步骤详解

#### 步骤 ②: 反向速率的计算

**理论依据** (论文 Proposition 1, 式 6):

$$
\hat R_t^\theta(x, \tilde x) = R_t(\tilde x, x) \sum_{x_0} \frac{q_{t|0}(\tilde x | x_0)}{q_{t|0}(x | x_0)} p_{0|t}^\theta(x_0 | x)
$$

代码逐行分解:

```python
# ┌─ R_t(x̃, x) = rate[:, :, x.flatten()].view(N,D,S)
# │  注意: rate 的 index 是 [batch, x̃, x]
# │  所以 rate[:, :, x.flatten()] 得到 R_t(·, x) — 跳转到 x 的速率
# │
# ├─ inner_sum = Σ_{x0} p0t(x0|x) · q_t|0(x̃|x0) / q_t|0(x|x0)
# │  其中:
# │    p0t       = (p0t / qt0_denom)  — 归一化后的 p0t
# │    @ qt0     = Σ_{x0} p0t(x0|x) · q_t|0(x̃|x0)
# │
# └─ 两者逐元素相乘即得 R̂_t^θ
```

> **直觉**: $p_{0\|t}^\theta(x_0|x)$ 是模型对"清晰原图"的预测。$\hat R_t^\theta$ 本质上是说：如果 $x_t$ 中的某个像素被污染成了某个值 $x$，那么它应该以多快的速率跳回 $\tilde x$（接近 $x_0$ 的某个候选值）。

#### 步骤 ③: Poisson 跳跃

```python
diffs = arange(S).view(1,1,S) - x.view(N,D,1)
# diffs[d, s] = s - x^d, 即从当前值到目标值 s 需要移动的距离
# 例如: x^d = 100, diffs = [..., -100, -99, ..., 0, ..., 155]

poisson_dist = Poisson(reverse_rates * h)
# 对每个 (batch, 维度, 目标状态) 三元组:
# 跳跃次数 ~ Poisson(τ · R̂_t^θ(x_t, s))
# τ = h 是 tau-leaping 步长

jump_nums = poisson_dist.sample()
# (N, D, S) 整数矩阵
# jump_nums[n,d,s] = 在 [t-τ, t] 内从 x_t^d 到 s 的跳跃次数

adj_diffs = jump_nums * diffs
# 加权位移: 每次跳跃的贡献

overall_jump = sum(adj_diffs, dim=2)
# 所有目标状态的跳跃合并成一个总位移
```

**为什么可以一次改多个维度**: 由于每个维度独立采样 Poisson 跳跃，不同维度可能同时发生多次跳跃。

**为什么对 ordinal 数据有物理意义**: 对像素值 (0-255)，$s - x^d$ 有明确的数值含义。例如从 100 跳 3 次到 103 等价于一次跳 3。但对分类数据 (如钢琴卷帘的 128 个音高)，多次跳跃无意义，所以用 `reject_multiple_jumps` 选项。

### 3.5 Tau-Leaping 示意图

```
时间 t ──────────────────────────────────→ 0
        |    |    |    |    |    |    |
        τ₀   τ₁   τ₂   τ₃   τ₄   τ₅   τ₆

在每个区间 [t_i, t_{i+1}]:
  ┌─────────────────────────────┐
  │  R̂_t 在区间内假设为常数      │
  │  每个 (d, s) 独立采样 Poisson │
  │  所有跳跃同时应用到 x         │
  └─────────────────────────────┘

对比精确模拟 (Gillespie):
  ┌─────────────────────────────┐
  │  一次只发生一个跳跃           │
  │  每个跳跃后重新计算 R̂_t       │
  │  状态更新频率 = 跳跃次数      │
  └─────────────────────────────┘
```

---

## 4. PCTauLeaping 采样器

**文件**: `lib/sampling/sampling.py:122-240`

### 4.1 与 TauLeaping 的区别

| 特性 | TauLeaping | PCTauLeaping |
|------|-----------|-------------|
| Predictor step | ✅ Poisson 跳跃 | ✅ Poisson 跳跃 |
| Corrector step | ❌ | ✅ $R_t^c = R_t + \hat R_t^\theta$ |
| Corrector 时机 | — | `t <= corrector_entry_time` (默认 $t < 0.1$) |
| Corrector 次数 | — | `num_corrector_steps` (默认 10) |

### 4.2 整体流程

```python
class PCTauLeaping:
    def sample(self, model, N, num_intermediates):
        x = get_initial_samples(...)
        
        # 时间网格 (不同: 使用 linspace 而不是手动)
        h = 1.0 / num_steps  # 约等于
        ts = np.linspace(1.0, min_t + h, num_steps)
        
        for idx, t in enumerate(ts[0:-1]):
            h = ts[idx] - ts[idx+1]
            
            # ===== Predictor Step =====
            transpose_forward_rates, reverse_rates, x_0max = get_rates(x, t)
            x = take_poisson_step(x, reverse_rates, h)  # 同 TauLeaping
            
            # ===== Corrector Steps (仅在 t <= entry_time) =====
            if t <= corrector_entry_time:
                for _ in range(num_corrector_steps):
                    tfr, rr, _ = get_rates(x, t - h)  # 在 predictor 后的时间点
                    corrector_rate = tfr + rr          # R_t^c = R_t + R̂_t
                    corrector_rate[:, :, x.flatten()] = 0.0
                    x = take_poisson_step(x, corrector_rate, 
                        corrector_step_size_multiplier * h)
        
        # 最终去噪
        p_0gt = F.softmax(model(x, min_t), dim=2)
        return argmax(p_0gt, dim=2)
```

### 4.3 Corrector 的理论依据

**Proposition 4**: 速率 $R_t^c = R_t + \hat R_t$ 以 $q_t(x_t)$ 为平稳分布。

证明: $R_t$ 控制前向过程，$\hat R_t$ 控制反向过程。它们的和是一个**可逆**的速率矩阵，其平稳分布正是前向过程的边际分布 $q_t(x_t)$。

**代码实现**:

```python
# get_rates 返回:
#   transpose_forward_rates = R_t(x̃, x)  [前向速率, 但索引转置]
#   reverse_rates = R̂_t^θ(x, x̃)          [反向速率]

corrector_rate = transpose_forward_rates + reverse_rates
# = R_t(x̃, x) + R̂_t^θ(x, x̃) = R_t^c

corrector_rate[:, :, x.flatten()] = 0.0  # 对角线置零
```

**为什么 corrector 有效**: tau-leaping 的 predictor 步是近似的（假设速率在区间内常数），累积误差会使采样分布偏离真实的 $q_t$。Corrector 步用 $R_t^c$ 做 MCMC 采样，把分布"拉回"正确的边际分布。

**为何只在 $t < 0.1T$ 启用**: 在反向过程的早期（$t$ 接近 $T$），$x$ 接近初始分布，误差较小。后期（$t$ 接近 $0$）误差累积最严重，所以只在最后 10% 的时间范围内启用 corrector。

### 4.4 Corrector 步长

```python
corrector_step_size_multiplier = 1.5  # 默认
x = take_poisson_step(x, corrector_rate, corrector_step_size_multiplier * h)
```

Corrector 步长通常比 predictor 略大 ($1.5\tau$)，因为 corrector 需要更快地混合到平稳分布。

### 4.5 辅助函数: get_rates

PC 采样器将速率计算提取为独立函数 `get_rates`，因为它需要在 predictor 和 corrector 中重复调用:

```python
def get_rates(in_x, in_t):
    qt0 = model.transition(in_t)    # (N, S, S)
    rate = model.rate(in_t)         # (N, S, S)
    p0t = F.softmax(model(in_x, in_t), dim=2)  # (N, D, S)
    
    # 计算反向速率 (同 TauLeaping)
    qt0_denom = qt0[:, :, in_x.flatten().repeat_interleave(S)].view(N, D, S)
    forward_rates = rate[:, :, in_x.flatten().repeat_interleave(S)].view(N, D, S)
    reverse_rates = forward_rates * ((p0t / qt0_denom) @ qt0)
    reverse_rates[:, :, in_x.flatten()] = 0.0
    
    # 转置前向速率 (用于 corrector)
    transpose_forward_rates = rate[
        :, in_x.flatten().repeat_interleave(S), arange(S).repeat(N*D)
    ].view(N, D, S)
    
    return transpose_forward_rates, reverse_rates, x_0max
```

### 4.6 辅助函数: take_poisson_step

```python
def take_poisson_step(in_x, in_reverse_rates, in_h):
    diffs = arange(S).view(1,1,S) - in_x.view(N,D,1)
    poisson_dist = Poisson(in_reverse_rates * in_h)
    jump_nums = poisson_dist.sample()
    adj_diffs = jump_nums * diffs
    overall_jump = sum(adj_diffs, dim=2)
    x_new = torch.clamp(in_x + overall_jump, min=0, max=S-1)
    return x_new
```

---

## 5. 条件采样器

### 5.1 ConditionalTauLeaping

**文件**: `sampling.py:242-351`

用于 Piano 音乐的条件生成：给定前 `condition_dim` 个 token（前 2 小节），生成后 `sample_D` 个 token（后 14 小节）。

与 `TauLeaping` 的关键区别:

```python
class ConditionalTauLeaping:
    def sample(self, model, N, num_intermediates, conditioner):
        # conditioner: (N, condition_dim) — 已知的前缀
        
        # 初始: 只采样未知部分
        x = get_initial_samples(N, sample_D, S, ...)
        
        for t in ts:
            # 模型输入 = 条件 + 当前采样
            model_input = torch.concat((conditioner, x), dim=1)
            p0t = F.softmax(model(model_input, t), dim=2)
            p0t = p0t[:, condition_dim:, :]  # 只取采样部分
            
            # ... 同 TauLeaping, 但只在 sample_D 维度上操作
        
        # 最终输出 = 条件 + 采样
        output = torch.concat((conditioner, x_0max), dim=1)
```

### 5.2 ConditionalPCTauLeaping

在 ConditionalTauLeaping 基础上增加 corrector 步骤，逻辑与 PCTauLeaping 完全相同。

### 5.3 reject_multiple_jumps

对分类数据（钢琴卷帘，$S=129$）的额外选项:

```python
if reject_multiple_jumps:
    jump_num_sum = torch.sum(jump_nums, dim=2)  # 每维度的总跳跃次数
    jump_num_sum_mask = jump_num_sum <= 1        # 只保留 0 或 1 次跳跃
    masked_jump_nums = jump_nums * jump_num_sum_mask.view(N, sample_D, 1)
    adj_diffs = masked_jump_nums * diffs         # 拒绝同一维度多次跳跃
```

**原因**: 对分类数据（如音高 0-128），状态到整数的映射是任意的，多次跳跃（如 3→5→7）不等于一次跳跃到 7。拒绝多跳保证语义合理。

---

## 6. 速率矩阵的实现

三种速率矩阵，均在 `models.py` 中实现:

### 6.1 GaussianTargetRate (CIFAR-10)

**物理意义**: 像素值接近的"颜色"之间更容易互相转换。

```python
class GaussianTargetRate:
    def __init__(self, cfg):
        S = cfg.data.S  # 256
        rate_sigma = cfg.model.rate_sigma   # 6.0 — 高斯宽度
        Q_sigma = cfg.model.Q_sigma          # 512.0 — 平稳分布宽度
        
        rate = np.zeros((S, S))  # (256, 256)
        
        # 先构造上三角: 基于高斯核 exp(-d²/σ²)
        vals = np.exp(-np.arange(0, S)² / rate_sigma²)
        for i in range(S):
            for j in range(S):
                if i < S//2 and j > i and j < S-i:
                    rate[i, j] = vals[j-i-1]      # 距离越近速率越高
                elif i > S//2 and j < i and j > -i+S-1:
                    rate[i, j] = vals[i-j-1]
        
        # 细节平衡 (detailed balance): 
        # R(i,j)·π(i) = R(j,i)·π(j), 其中 π 是平稳分布 (高斯)
        for i in range(S):
            for j in range(S):
                if rate[j, i] > 0.0:
                    rate[i, j] = rate[j, i] * exp(-((j+1)²-(i+1)²+S*(i+1)-S*(j+1))/(2*Q_sigma²))
        
        # 对角线 = -行和 (保证行和为 0)
        rate = rate - np.diag(np.diag(rate))
        rate = rate - np.diag(np.sum(rate, axis=1))
```

**矩阵 $R_b$ 可视化**:

```
对 S=256:
  - R[i,j] 当 |i-j| 小时 → 大值 (近邻转移)
  - R[i,j] 当 |i-j| 大时 → 接近 0 (远邻几乎不可能)
  - 平稳分布 π(i) ∝ exp(-(i-128)²/(2·512²)) ~ uniform
  
时间标量 β(t):
  β(t) = time_base · log(time_exponential) · time_exponential^t
       = 3.0 · log(100) · 100^t
  ∫₀ᵗ β(s) ds = 3.0 · (100^t - 1)
```

**$q_{t\|0}$ 计算**: 利用特征分解 $R_b = Q \Lambda Q^{-1}$:

```python
def transition(self, t):
    integral_rate_scalars = self.time_base * (self.time_exponential ** t - 1)
    adj_eigvals = integral_rate_scalars * self.eigvals
    # exp(∫β · Λ) = diag(exp(∫β · λ_i))
    transitions = self.eigvecs @ diag_embed(exp(adj_eigvals)) @ self.inv_eigvecs
    # = Q · exp(∫β · Λ) · Q⁻¹
    transitions[transitions < 1e-8] = 0.0
    return transitions  # (B, S, S)
```

### 6.2 UniformRate (Piano)

**物理意义**: 所有状态之间均匀转移，适合无序结构的分类数据。

```python
class UniformRate:
    def __init__(self, cfg):
        S = cfg.data.S
        rate_const = cfg.model.rate_const  # 转移常数
        rate = rate_const * (ones((S,S)) - I)  # 非对角为常数
        rate = rate - diag(sum(rate, axis=1))   # 对角线归一化
        # 预计算特征分解 (对称矩阵, 可正交对角化)
        eigvals, eigvecs = eigh(rate)
```

### 6.3 BirthDeathForwardBase (Piano 消融)

**物理意义**: 只有相邻状态可以互相转换 (1D 生灭过程)。

```python
class BirthDeathForwardBase:
    def __init__(self, cfg):
        S = cfg.data.S
        base_rate = diag(ones(S-1), 1) + diag(ones(S-1), -1)  # 邻接矩阵
        base_rate -= diag(sum(base_rate, axis=1))
        # σ²(t) = σ_min² · (σ_max/σ_min)^{2t}
        # ∫σ² = 0.5·σ_min²·((σ_max/σ_min)^{2t} - 1)
```

### 6.4 三种速率对比

| 特性 | GaussianTargetRate | UniformRate | BirthDeathForwardBase |
|------|-------------------|-------------|----------------------|
| 非零项 | 近距离高，远距离低 | 全连接常数 | 仅相邻 |
| 数据结构 | 有序 (像素) | 无序 (分类) | 有序 (消融) |
| 对应 D3PM | 离散化高斯核 | 均匀核 | — |
| 实现复杂度 | 复杂 (特征分解) | 简单 | 简单 |
| 适用场景 | CIFAR-10 图像 | Piano 音乐 | 消融实验 |

---

## 7. 论文 §4.5 误差界与代码的关系

论文定理 1 的误差界:

$$
\|\mathcal{L}(y_0) - p_{\text{data}}\|_{\text{TV}} \leq 3MT + \left\{ \left(|R|SDC_1\right)^2 + \frac{1}{2}C_2(M + C_1SD|R|) \right\}\tau T + 2\exp\left\{-\frac{T\log^2 2}{t_{\text{mix}}\log 4D}\right\}
$$

三项误差:

| 误差项 | 含义 | 对应代码中的因素 |
|--------|------|----------------|
| $3MT$ | 反向速率近似误差 ($\|\hat R_t - \hat R_t^\theta\|$) | 模型 $p_{0\|t}^\theta$ 的近似能力 (UNet/Transformer) |
| $O(\tau T)$ | Tau-leaping 离散化误差 | `num_steps` 控制 $\tau = 1/\text{num\_steps}$ |
| $2\exp(-T/(t_{\text{mix}}\log 4D))$ | 前向过程混合误差 ($p_{\text{ref}}$ 与 $q_T$ 的差距) | `T=1` + 速率参数 (`time_exponential`, `time_base`)|

代码中的实际操作:
- $\tau$ 由 `sampler.num_steps` 控制: 从 100 到 1000 步均可配置
- $T$ 固定为 1.0 (论文所有实验)
- $t_{\text{mix}}$ 由速率矩阵的参数间接控制 (`rate_sigma`, `time_exponential`)

---

## 8. 论文 ↔ 代码对照表

| 论文部分 | 代码位置 | 说明 |
|----------|---------|------|
| §4.1 $R_t = \beta(t)R_b$ | `GaussianTargetRate.rate()`: `models.py:238-244` | 速率标量 × 基速率 |
| §4.1 $q_{t\|0}$ 解析解 | `GaussianTargetRate.transition()`: `models.py:246-267` | 矩阵指数 $e^{\int\beta \cdot \Lambda}$ |
| §4.2 维度分解 → 单维跳跃 | `TauLeaping.sample()` step ②: `sampling.py:79-103` | 仅 $D(S-1)+1$ 个非零速率 |
| §4.3 Tau-leaping | `TauLeaping.sample()`: `sampling.py:30-120` | Poisson 跳跃 + 同时更新 |
| §4.3 Gillespie 精确模拟 | 仅论文讨论未实现 | — |
| §4.3 算法 1 | `TauLeaping.sample()`: `sampling.py:56-113` | 完整实现 |
| §4.4 Predictor-Corrector | `PCTauLeaping.sample()`: `sampling.py:127-240` | $R_t^c = R_t + \hat R_t$ |
| §4.4 Proposition 4 | `PCTauLeaping`: `sampling.py:225-226` | `corrector_rate = tfr + rr` |
| §4.5 误差界定理 1 | — | 理论保证，非代码 |
| §6.1 2D 演示 | `GaussianRateResidualMLP` (`models.py:476`) + ResidualMLP (`networks.py:593`) | 验证 tau-leaping 精度 |
| §6.2 CIFAR-10 | `scripts/sample.py` | 50000 样本 → FID 计算 |
| §6.2 图 4 right: τ 与 FID | `sampler.num_steps` 控制 | NFE = T/τ |
| §6.3 Piano 条件生成 | `ConditionalPCTauLeaping`: `sampling.py:354-490` | 给定前 2 小节生成后 14 小节 |
| §F Tau-leaping 算法细节 | `TauLeaping.sample()` | 完整实现 |
| §G 离散时间的隐式假设 | — | 仅在论文讨论 |
| Appendix A CTMC 入门 | — | 理论背景 |
| Appendix H 实验细节 | `config/eval/cifar10.py`, `config/train/piano.py` | 超参数 |
