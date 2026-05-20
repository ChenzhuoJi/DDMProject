# 01 — 神经网络训练部分精读

> CTMC 论文代码精读 · 训练流水线
>
> 对应论文 §2–§4.2 + Appendices C/D/E/H

## 目录

- [1. 配置系统全景](#1-配置系统全景)
- [2. 训练入口 `train.py`](#2-训练入口-trainpy)
- [3. 模型组合机制 `models.py`](#3-模型组合机制-moviespy)
- [4. 神经网络架构 `networks.py`](#4-神经网络架构-networkspy)
- [5. 时间嵌入 `network_utils.py`](#5-时间嵌入-network_utilspy)
- [6. CT-ELBO 损失函数 `losses.py`](#6-ct-elbo-损失函数-lossespy)
- [7. 训练步骤 `training.py`](#7-训练步骤-trainingpy)
- [8. 数据集 `datasets.py`](#8-数据集-datasetspy)
- [9. ELBO 评估 `elbo_evaluation.py`](#9-elbo-评估-elbo_evaluationpy)
- [10. 论文 ↔ 代码对照表](#10-论文--代码对照表)

---

## 1. 配置系统全景

**文件**: `config/train/cifar10.py`

整个实验由一个 `ml_collections.ConfigDict` 对象驱动。核心配置字段：

```python
# 设备 & 分布式
config.device = 'cpu'           # 训练设备
config.distributed = False      # 是否 DDP
config.num_gpus = 0

# 损失函数
loss.name = 'GenericAux'        # 损失类注册名
loss.eps_ratio = 1e-9           # 数值稳定性
loss.nll_weight = 0.001         # 辅助交叉熵权重
loss.min_time = 0.01            # 最小时间 (避免 t=0 奇点)
loss.one_forward_pass = True    # 单次前向传播技巧 (§C.4)

# 训练
training.train_step_name = 'Standard'
training.n_iters = 2000000      # 总迭代数
training.clip_grad = True       # 梯度裁剪 1.0
training.warmup = 5000          # LR warmup 步数

# 数据
data.name = 'DiscreteCIFAR10'
data.S = 256                    # 状态空间大小 (每个通道 256 级)
data.batch_size = 128
data.shape = [3, 32, 32]       # C, H, W

# 模型
model.name = 'GaussianTargetRateImageX0PredEMA'
model.ema_decay = 0.9999
model.ch = 128                 # UNet 基础通道数
model.num_scales = 4           # 下采样层级
model.ch_mult = [1, 2, 2, 2]  # 每层通道倍数

# 前向速率矩阵参数 (§4.1)
model.rate_sigma = 6.0         # 高斯速率宽度
model.Q_sigma = 512.0          # 平稳分布宽度
model.time_exponential = 100.0 # β(t) 指数底数
model.time_base = 3.0          # β(t) 基数
```

> **设计意图**: 将所有超参数集中在一个 ConfigDict 中，配合注册表模式，切换实验只需改配置，无需修改代码。

---

## 2. 训练入口 `train.py`

### 2.1 主流程

```python
def main(cfg):
    # 1. 创建实验文件夹 / 恢复抢占的检查点
    save_dir, checkpoint_dir, config_dir = bookkeeping.create_experiment_folder(...)
    
    # 2. 初始化 TensorBoard
    writer = bookkeeping.setup_tensorboard(save_dir, 0)
    
    # 3. 从注册表创建所有组件
    model = model_utils.create_model(cfg, device)
    dataset = dataset_utils.get_dataset(cfg, device)
    dataloader = DataLoader(dataset, ...)
    loss = losses_utils.get_loss(cfg)
    training_step = training_utils.get_train_step(cfg)
    optimizer = optimizers_utils.get_optimizer(model.parameters(), cfg)
    
    # 4. 准备状态字典 (包含模型、优化器、迭代计数)
    state = {'model': model, 'optimizer': optimizer, 'n_iter': 0}
    
    # 5. 训练循环
    while True:
        for minibatch in dataloader:
            training_step.step(state, minibatch, loss, writer)
            if n_iter % checkpoint_freq == 0:
                bookkeeping.save_checkpoint(...)
            if n_iter % log_low_freq == 0:
                for logger in low_freq_loggers:
                    logger(state, cfg, writer, minibatch, dataset)
            state['n_iter'] += 1
```

### 2.2 注册表模式

所有可替换组件（model / dataset / loss / trainer / optimizer / sampler / logger）共用一套注册机制：

```python
# model_utils.py
_MODELS = {}

def register_model(cls):
    name = cls.__name__
    _MODELS[name] = cls
    return cls

def create_model(cfg, device, rank=None):
    model = _MODELS[cfg.model.name](cfg, device, rank)
    return model.to(device)
```

使用 `@register_model` 装饰器将类注册到全局 `_MODELS` 字典，通过配置中的字符串名称按需实例化。

---

## 3. 模型组合机制 `models.py`

### 3.1 多继承架构

这是代码最精巧的设计：用 **Python 多继承** 将**网络架构**、**前向速率矩阵**、**EMA** 组合成最终模型。

```python
@model_utils.register_model
class GaussianTargetRateImageX0PredEMA(
    EMA,                    # ← 指数移动平均 (Mixin)
    ImageX0PredBase,        # ← UNet 网络包装
    GaussianTargetRate      # ← 前向速率矩阵
):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        ImageX0PredBase.__init__(self, cfg, device, rank)
        GaussianTargetRate.__init__(self, cfg, device)
        self.init_ema()
```

### 3.2 三个基类的职责

| 基类 | 文件 | 职责 | 提供的接口 |
|------|------|------|-----------|
| `ImageX0PredBase` | `models.py:14` | UNet 包装 + 截断逻辑分布 → logits | `forward(x, times) → (B,D,S)` |
| `GaussianTargetRate` | `models.py:192` | 高斯偏好速率矩阵 + 解析 $q_{t\|0}$ | `rate(t)`, `transition(t)` |
| `EMA` | `models.py:354` | Shadow params + 训练/评估切换 | `update_ema()`, `train()` |

### 3.3 ImageX0PredBase: UNet → logits

```python
class ImageX0PredBase(nn.Module):
    def forward(self, x, times):
        # x: (B, D) 离散像素值 [0, 255], D = 3*32*32 = 3072
        B, D = x.shape
        C, H, W = self.data_shape   # (3, 32, 32)
        x = x.view(B, C, H, W)      # 重塑为图像
        net_out = self.net(x, times) # UNet 输出 (B, 2*C, H, W)
        
        # 截断逻辑分布 → 256 个离散 logits
        mu = net_out[:, 0:C, :, :].unsqueeze(-1)
        log_scale = net_out[:, C:, :, :].unsqueeze(-1)
        inv_scale = torch.exp(-(log_scale - 2))
        
        bin_width = 2. / self.S     # S = 256
        bin_centers = linspace(-1 + bin_width/2, 1 - bin_width/2, S)
        
        # 计算每个 bin 的 CDF 差值 → 类别概率
        sig_in_left = (bin_centers - bin_width/2 - mu) * inv_scale
        bin_left_logcdf = logsigmoid(sig_in_left)
        sig_in_right = (bin_centers + bin_width/2 - mu) * inv_scale
        bin_right_logcdf = logsigmoid(sig_in_right)
        
        logits = log(exp(bin_right_logcdf) - exp(bin_left_logcdf))
        return logits.view(B, D, S)  # (B, 3072, 256)
```

**关键设计**: 使用 **截断逻辑分布 (discretized logistic)** 将连续输出离散化为 256 个 bin。UNet 输出 `2C` 个通道：前 `C` 是每个通道的均值 `μ`，后 `C` 是 `log_scale`。然后在 `[-1, 1]` 区间上等距放置 256 个 bin 中心，计算每个 bin 内 logistic CDF 的差值作为该类的 log-probability。

> 这项技术来自 D3PM 论文 [Austin et al., 2021] 以及 PixelCNN++ [Salimans et al., 2017]。

### 3.4 GaussianTargetRate: 前向速率矩阵

论文 §4.1 要求：
1. $R_t$ 与 $R_{t'}$ 对所有时间可交换 → 令 $R_t = \beta(t) R_b$
2. $q_{t\|0}$ 有解析解 → 矩阵指数 $e^{\int_0^t \beta(s) ds \cdot \Lambda}$

```python
class GaussianTargetRate:
    def __init__(self, cfg):
        # 构造时间无关的基速率矩阵 R_b (S×S)
        # 高斯偏好: 只有"附近"的状态间有显著转移率
        rate = np.zeros((S, S))
        vals = exp(-arange(0, S)² / rate_sigma²)
        for i in range(S):
            for j in range(S):
                if i < S//2 and j > i and j < S-i:
                    rate[i, j] = vals[j-i-1]   # 上三角
                elif i > S//2 and j < i and j > -i+S-1:
                    rate[i, j] = vals[i-j-1]   # 下三角
        # 细节平衡条件: rate[i,j] = rate[j,i] * exp(Δ能量)
        for i in range(S):
            for j in range(S):
                if rate[j, i] > 0:
                    rate[i, j] = rate[j,i] * exp(-((j+1)²-(i+1)²+S*(i+1)-S*(j+1))/(2*Q_sigma²))
        # 对角项 = -行和
        rate = rate - diag(diag(rate))
        rate = rate - diag(sum(rate, axis=1))
        
        # 预计算特征分解 R_b = Q Λ Q⁻¹
        eigvals, eigvecs = eig(rate)
        inv_eigvecs = inv(eigvecs)
```

时间依赖部分：

```python
def _rate_scalar(self, t):
    # β(t) = time_base · log(time_exponential) · time_exponential^t
    return self.time_base * math.log(self.time_exponential) * (self.time_exponential ** t)

def _integral_rate_scalar(self, t):
    # ∫₀ᵗ β(s) ds = time_base · (time_exponential^t - 1)
    return self.time_base * (self.time_exponential ** t) - self.time_base

def rate(self, t):        # R_t = β(t) · R_b
    return self.base_rate.view(1, S, S) * rate_scalars.view(B, 1, 1)

def transition(self, t):  # q_{t|0}(j|i) = [Q · exp(∫β · Λ) · Q⁻¹]_{ij}
    adj_eigvals = integral_rate_scalars.view(B, 1) * self.eigvals.view(1, S)
    transitions = self.eigvecs @ diag_embed(exp(adj_eigvals)) @ self.inv_eigvecs
    transitions[transitions < 1e-8] = 0.0
    return transitions
```

> **解释**: `transition(t)` 计算的是 $q_{t\|0}(x_t=j \mid x_0=i)$，即给定初始状态 $i$ 时刻 $t$ 的状态分布。这个矩阵在损失函数中需要频繁使用。

### 3.5 EMA: 指数移动平均

```python
class EMA:
    def init_ema(self):
        self.shadow_params = [p.clone().detach() for p in self.parameters()]
    
    def update_ema(self):
        decay = min(self.decay, (1 + num_updates) / (10 + num_updates))
        for s_param, param in zip(shadow_params, parameters):
            s_param.sub_((1 - decay) * (s_param - param))
    
    def train(self, mode=True):
        if mode:   # 训练模式: 从收集的参数恢复
            self.move_collected_params_to_model_params()
        else:      # 评估模式: 保存当前参数, 加载 EMA shadow params
            self.move_model_params_to_collected_params()
            self.move_shadow_params_to_model_params()
```

> **注意**: `GaussianTargetRateImageX0PredEMA` 的 MRO 中 `EMA` 在第一位，这样 EMA 的 `train()` 方法会覆盖 `nn.Module.train()`，实现训练/评估时参数的无缝切换。

### 3.6 其他模型组合

| 注册名称 | 架构 | 速率 | 用途 |
|----------|------|------|------|
| `UniformRateSequenceTransformerEMA` | TransformerEncoder | UniformRate | Piano 音乐生成 |
| `BirthDeathRateSequenceTransformerEMA` | TransformerEncoder | BirthDeathForwardBase | Piano 消融 |
| `GaussianRateResidualMLP` | ResidualMLP | GaussianTargetRate | 2D 演示实验 |

---

## 4. 神经网络架构 `networks.py`

### 4.1 UNet (CIFAR-10)

代码移植自 [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)，结构与 DDPM 的 UNet 一致。

**结构参数** (从配置):
- 基础通道 `ch = 128`
- 4 个分辨率层级: `[128, 256, 256, 256]` (ch_mult = [1, 2, 2, 2])
- 每层 2 个残差块
- 第 2 层 (`scale_count=1`) 加自注意力
- 时间嵌入: FiLM 风格

**前向路径**:

```
输入 x (B, 3, 32, 32) [0, 255]
  → _center_data: 归一化到 [-1, 1]
  → _time_embedding: 时间 t → 正弦嵌入 → MLP → (B, 4*ch)
  → _do_input_conv: Conv2D 3→128
  → _do_downsampling: 
      Scale 0: ResBlock×2 (128→128) 
      Scale 1: ResBlock×2 (128→256) + AttnBlock → Downsample 32→16
      Scale 2: ResBlock×2 (256→256) → Downsample 16→8
      Scale 3: ResBlock×2 (256→256) → Downsample 8→4
  → _do_middle: ResBlock → AttnBlock → ResBlock (256)
  → _do_upsampling: (每步 concat 对应下采样的 skip 连接)
      Scale 2: ResBlock×3 (512→256) → Upsample 4→8
      Scale 1: ResBlock×3 (512→256) + AttnBlock → Upsample 8→16
      Scale 0: ResBlock×3 (512→128) → Upsample 16→32
  → _do_output: GroupNorm → SiLU → Conv2D (128→6)
  → _logistic_output_res: 加入残差连接 (centered_x_in + h[:,0:3])
输出 (B, 6, 32, 32)  ← 前3通道 μ, 后3通道 log_scale
```

**ResBlock** 细节:

```python
class ResBlock(nn.Module):
    def forward(self, x, temb=None):
        h = GroupNorm(x) → SiLU → Conv2D(in_ch, out_ch, 3×3)
        if temb: h += dense0(SiLU(temb))[:, :, None, None]  # FiLM 偏置
        h = GroupNorm(h) → SiLU → Dropout → Conv2D(out_ch, out_ch, 3×3)
        if in_ch ≠ out_ch: x = NiN(x)  # 1×1 卷积投影
        return (x + h) / √2  # 跳跃连接 + 重缩放
```

**AttnBlock**: 通道维度的自注意力 (非空间维度):

```python
q = NIN0(h)  # 1×1 conv → query
k = NIN1(h)  # 1×1 conv → key
v = NIN2(h)  # 1×1 conv → value
w = einsum('bchw,bcij→bhwij', q, k) / √C
w = softmax(w.reshape(B, H, W, H*W)).reshape(B, H, W, H, W)
h = einsum('bhwij,bcij→bchw', w, v)
return (x + h) / √2
```

### 4.2 TransformerEncoder (Piano)

```python
class TransformerEncoder:
    def forward(self, x, times):
        # x: (B, L) 离散 token IDs, L = 256
        # 时间嵌入: sine → MLP → 4*temb_dim
        temb = temb_net(transformer_timestep_embedding(times, temb_dim))
        
        # 输入嵌入: one-hot → Linear → d_model
        one_hot_x = one_hot(x, S)  # (B, L, S)
        x = input_embedding(one_hot_x.float())  # (B, L, d_model)
        x = pos_embed(x)  # 正弦位置编码
        
        for layer in encoder_layers:  # 6 层
            x = layer(x, temb)
        for resid in output_resid_layers:  # 输出残差
            x = resid(x, temb)
        x = output_linear(x)  # (B, L, S)
        x = x + one_hot_x     # 残差连接到输入 one-hot
        return x
```

**TransformerEncoderLayer** 使用 FiLM 进行时间条件化:

```python
class TransformerEncoderLayer:
    def forward(self, x, temb):
        film_params = self.film_from_temb(temb)  # (B, 2*d_model)
        gamma, beta = film_params[:, :K], film_params[:, K:]
        
        x = norm1(x + self_attention(x))
        x = gamma * x + beta  # FiLM: 仿射变换
        x = norm2(x + FFN(x))
        x = gamma * x + beta  # FiLM: 第二次
        return x
```

### 4.3 ResidualMLP (2D 演示)

两层残差 MLP + FiLM 时间条件化:

```python
h = normalize_input(x)        # (B, D) → [-1, 1]
h = input_layer(h)            # (B, d_model)
for n in range(num_layers):
    h = norm(h + layer2(ReLU(layer1(h))))  # 残差块
    film_params = temb_layers[n](temb)
    h = gamma * h + beta                   # FiLM
h = output_layer(h)           # (B, D*S)
h = h.reshape(B, D, S)
logits = h + one_hot_x        # 残差
```

---

## 5. 时间嵌入 `network_utils.py`

```python
def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    half_dim = embedding_dim // 2
    emb = log(max_positions) / (half_dim - 1)
    emb = exp(arange(half_dim) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = cat([sin(emb), cos(emb)], dim=1)
    return emb  # (B, embedding_dim)
```

这是 Transformer 中的标准正弦位置编码，将标量时间 $t$ 编码为 `embedding_dim` 维向量。在 UNet 中 `embedding_dim = ch = 128`，在 Transformer 中 `temb_dim` 可配置。

输入 $t$ 乘以 `time_scale_factor = 1000`，将 $[0, 1]$ 映射到 $[0, 1000]$。

---

## 6. CT-ELBO 损失函数 `losses.py`

### 6.1 整体结构

论文第 3.2 节推导的连续时间 ELBO:

$$
\mathcal{L}_{\text{CT}}(\theta) = T \, \mathbb{E}_{t \sim \mathcal{U}(0,T),\, x_t \sim q_t,\, \tilde x \sim r_t(\cdot|x_t)} \left[ \underbrace{\sum_{x'\neq x_t} \hat R_t^\theta(x_t, x')}_{\text{reg 项}} - \underbrace{\mathcal{Z}^t(x_t) \log \hat R_t^\theta(\tilde x, x_t)}_{\text{sig 项}} \right] + C
$$

代码中实现为 `GenericAux.calc_loss()`:

```python
def calc_loss(self, minibatch, state, writer):
    B, D = minibatch.shape        # (batch, 3072)
    
    # --- ① 采样时间 ---
    ts = torch.rand((B,)) * (1.0 - self.min_time) + self.min_time
    
    # --- ② 计算 q_{t|0} 和 R_t ---
    qt0 = model.transition(ts)   # (B, S, S) 每行 q_t(·|x_0)
    rate = model.rate(ts)        # (B, S, S)
    
    # --- ③ 采样 x_t ~ q_{t|0}(·|x_0) ---
    # 对每个 batch 每个维度，从 qt0 对应行采样
    x_t = sample_from_qt0(minibatch, qt0)
    
    # --- ④ 采样 x_tilde (改变一个维度) ---
    # 加权选维度: P(d) ∝ Σ_s R_t(x_t^d, s)
    square_dims = sample_dim_by_rate(x_t, rate)
    # 选新值: P(s) ∝ R_t(x_t^d, s)
    square_newval = sample_val_by_rate(x_t, rate, square_dims)
    x_tilde = x_t.clone()
    x_tilde[:, square_dims] = square_newval
    
    # --- ⑤ 模型前向 (one_forward_pass 优化) ---
    if one_forward_pass:
        logits = model(x_tilde, ts)  # 只用一次前向
        p0t = softmax(logits, dim=2) # (B, D, S)
        reg_x = x_tilde
    else:
        logits_reg = model(x_t, ts)
        p0t_reg = softmax(logits_reg, dim=2)
        logits_sig = model(x_tilde, ts)
        p0t_sig = softmax(logits_sig, dim=2)
    
    # --- ⑥ reg 项 (第一项) ---
    reg_term = compute_reg_term(p0t, qt0, rate, reg_x)
    
    # --- ⑦ sig 项 (第二项) ---
    sig_term = compute_sig_term(p0t, qt0, rate, minibatch, x_tilde)
    
    # --- ⑧ 辅助 NLL ---
    nll = cross_entropy(logits.permute(0,2,1), minibatch)
    
    return (reg_term + sig_term) + nll_weight * nll
```

### 6.2 详细步骤分解

#### 步骤 ③: 采样 $x_t$

```python
# qt0: (B, S, S), 对于 batch 中每个样本是一个 S×S 转移矩阵
# 需要从 qt0[x_0, :] 采样 (每个维度独立)
qt0_rows_reg = qt0[
    arange(B).repeat_interleave(D),  # batch 索引
    minibatch.flatten().long(),      # x_0 值 (0..255)
    :                                 # 所有目标状态
]  # → (B*D, S)

x_t_cat = Categorical(qt0_rows_reg)
x_t = x_t_cat.sample().view(B, D)  # (B, D)
```

#### 步骤 ④: 采样 $\tilde x$

$\tilde x$ 是从 $x_t$ 出发、按照前向速率 $R_t$ 发生一次跳跃后的状态。分两步:

**第一步**: 选哪个维度发生跳跃。在维度 $d$ 上发生跳跃的概率正比于该维度的总出率 $\sum_{s \neq x_t^d} R_t(x_t^d, s)$。

```python
rate_vals_square = rate[arange(B).repeat_interleave(D), x_t.flatten(), :]  # (B*D, S)
rate_vals_square[arange(B*D), x_t.flatten()] = 0.0  # 对角线置零
rate_vals_square_dimsum = sum(rate_vals_square, dim=2)  # (B, D) 每维度总出率
square_dimcat = Categorical(rate_vals_square_dimsum)
square_dims = square_dimcat.sample()  # (B,) 每个样本选一个维度
```

**第二步**: 选这个维度上的新值。概率正比于 $R_t(x_t^d, s)$。

```python
rate_new_val_probs = rate_vals_square[arange(B), square_dims, :]  # (B, S)
square_newvalcat = Categorical(rate_new_val_probs)
square_newval_samples = square_newvalcat.sample()  # (B,)

x_tilde = x_t.clone()
x_tilde[arange(B), square_dims] = square_newval_samples
```

> **$x_t$ 与 $\tilde x$ 的关系**: 它们是一个 CTMC 中前后两个时刻的状态对。在反向过程中，模型需要学习从 $\tilde x$（稍后的状态）跳回 $x_t$（稍早的状态），即 $\hat R_t^\theta$。

#### 步骤 ⑤ + ⑥: Reg 项

实现论文中 CT-ELBO 的第一项 $\sum_{x' \neq x} \hat R_t^\theta(x, x')$。

```python
def compute_reg_term(p0t, qt0, rate, reg_x):
    # p0t: (B, D, S) — 模型预测的 p_0|t(x_0 | x_t)
    # reg_x: 使用 x_t 或 x_tilde (取决于 one_forward_pass)
    
    # 构建 mask: 排除当前状态 (对角线)
    mask_reg = ones((B, D, S))
    mask_reg[arange(B).repeat(D), arange(D).repeat(B), reg_x.flatten()] = 0.0
    
    # qt0_numer: 转置后的 qt0 (B, S, S)
    qt0_numer_reg = qt0.view(B, S, S)
    
    # qt0_denom: q_t|0(· | reg_x) (B, D, S)
    qt0_denom_reg = qt0[:, :, reg_x.flatten()].view(B, D, S) + eps
    
    # R_t(·, reg_x): 跳转到 reg_x 的速率 (B, D, S)
    rate_vals_reg = rate[:, :, reg_x.flatten()].view(B, D, S)
    
    # reg_tmp = (mask * R_t(·, reg_x)) × qt0_numer^T  (B, D, S)
    reg_tmp = (mask_reg * rate_vals_reg) @ qt0_numer_reg.transpose(1, 2)
    
    # reg_term = Σ (p0t / qt0_denom) * reg_tmp  (B,)
    reg_term = sum((p0t_reg / qt0_denom_reg) * reg_tmp, dim=(1, 2))
    return reg_term
```

**物理含义**: 正则化项惩罚 $\hat R_t^\theta(x, x')$ 的总和。直观上，它防止模型学习过大的反向速率，起到正则化作用。代码中 $\hat R_t^\theta$ 不是直接参数化的，而是通过 $p_{0\|t}^\theta$ 隐式定义的：$\hat R_t^\theta(x, \tilde x) = R_t(\tilde x, x) \sum_{x_0} \frac{q_{t\|0}(\tilde x|x_0)}{q_{t\|0}(x|x_0)} p_{0\|t}^\theta(x_0|x)$。

#### 步骤 ⑤ + ⑦: Sig 项

第二项 $\mathcal{Z}^t(x) \log \hat R_t^\theta(\tilde x, x)$，是主要的"学习信号"——它最大化特定反向跳跃 $(\tilde x \to x)$ 的速率。

```python
def compute_sig_term(p0t, qt0, rate, minibatch, x_tilde):
    # --- 内部部分: log(p0t / qt0_denom @ qt0_numer) ---
    inner_log = log((p0t / qt0_denom_sig) @ qt0_numer_sig + eps)  # (B, D, S)
    
    # --- 外部部分: R_t(x', x_tilde) × q_t|0(x_0|x') / q_t|0(x_0|x_tilde) ---
    # x' 遍历所有 S 个状态
    outer_part = x_tilde_mask * outer_rate_sig * (outer_qt0_numer / qt0_denom_sig)
    
    outer_sum = sum(outer_part * inner_log, dim=(1, 2))
    
    # --- 归一化 Z ---
    Z = compute_Z(rate, x_tilde)
    sig_term = outer_sum / Z
    
    return -mean(sig_term)  # 负号: 最小化 ELBO
```

#### 步骤 ⑧: 辅助 NLL

```python
perm_x_logits = logits.permute(0, 2, 1)  # (B, S, D)
nll = cross_entropy(perm_x_logits, minibatch.long())  # 标准交叉熵
```

这是论文 §D 中的"直接去噪模型监督"项。使用很小的权重 (`nll_weight = 0.001`) 来稳定训练。

---

## 7. 训练步骤 `training.py`

```python
class Standard:
    def step(self, state, minibatch, loss, writer):
        state['optimizer'].zero_grad()
        l = loss.calc_loss(minibatch, state, writer)
        
        l.backward()
        
        if self.clip_grad:
            clip_grad_norm_(state['model'].parameters(), 1.0)
        
        if self.warmup > 0:
            for g in state['optimizer'].param_groups:
                g['lr'] = self.lr * min(state['n_iter'] / self.warmup, 1.0)
        
        state['optimizer'].step()
        
        if self.do_ema:
            state['model'].update_ema()  # 更新 shadow params
```

**关键点**:
- 梯度裁剪 1.0 (`training.clip_grad = True`)
- LR warmup 5000 步 (`training.warmup = 5000`)，从 0 线性增加到 $2\times 10^{-4}$
- 每一步后更新 EMA (`ema_decay = 0.9999`)

---

## 8. 数据集 `datasets.py`

### DiscreteCIFAR10

```python
class DiscreteCIFAR10(CIFAR10):
    def __init__(self, cfg, device):
        super().__init__(root, train, download)
        self.data = from_numpy(self.data)      # (N, 32, 32, 3)
        self.data = self.data.transpose(1, 3)  # (N, 3, 32, 32)
        self.data = self.data.to(device)
        self.random_flips = cfg.data.random_flips
        self.flip = RandomHorizontalFlip()
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.random_flips:
            img = self.flip(img)
        return img  # 直接返回离散像素值 (0..255)
```

> 数据被直接预加载到 GPU (line 23: `self.data = self.data.to(device)`)，消除 CPU-GPU 传输瓶颈。CIFAR-10 的 50000 张训练图像仅需 ~1.5GB。

### LakhPianoroll

```python
class LakhPianoroll(Dataset):
    def __init__(self, cfg, device):
        np_data = np.load(cfg.data.path)  # (N, 256) in [0, 128]
        self.data = from_numpy(np_data).to(device)
```

---

## 9. ELBO 评估 `elbo_evaluation.py`

```python
def main(eval_cfg):
    # 1. 加载训练配置并覆盖评估参数
    train_cfg = load_ml_collections(most_recent_config)
    for item in eval_cfg.train_config_overrides:
        set_in_nested_dict(train_cfg, item[0], item[1])
    
    # 2. 重建模型并加载权重
    model = create_model(train_cfg, device)
    model_state = torch.load(checkpoint_path)['model']
    if is_model_state_DDP(model_state):
        model_state = remove_module_from_keys(model_state)
    model.load_state_dict(model_state)
    model.eval()  # 切换到 EMA 参数
    
    # 3. 运行 ELBO logger (在 loggers.py 中)
    for logger_name in eval_cfg.loggers:
        logging_func(state={'model': model}, cfg=eval_cfg, ...)
        # 这会遍历数据集, 计算 -ELBO bits/dim
```

---

## 10. 论文 ↔ 代码对照表

| 论文部分 | 代码位置 | 说明 |
|----------|---------|------|
| §2 DT-ELBO 回顾 | — | 背景知识，非代码 |
| §3.1 前向 CTMC + 时间反演 | `models.py:192-267` (GaussianTargetRate) | 速率矩阵 $R_t$ + $q_{t\|0}$ 解析解 |
| §3.1 反向速率 $\hat R_t^\theta$ | `losses.py:85-122` (reg term) / `sampling.py:79-97` | 通过 $p_{0\|t}^\theta$ 隐式定义 |
| §3.2 CT-ELBO | `losses.py:23-246` (GenericAux) | 两项期望 + 辅助 NLL |
| §4.1 前向过程设计 | `models.py:114-267` | β(t) 调度 + 特征分解 |
| §4.2 维度分解 | `losses.py:50-78` (采样 x_tilde) | 每次只改一个维度 |
| §4.3 Tau-Leaping | `sampling.py:30-120` | — (见 02.md) |
| §4.4 Predictor-Corrector | `sampling.py:122-240` | — (见 02.md) |
| §C.4 One Forward Pass | `losses.py:85-91, 129-132` | 是否复用 x_tilde 的输出 |
| §D 直接去噪模型监督 | `losses.py:242-246` | nll_weight = 0.001 |
| §E 前向过程选择 | `models.py:99-267` | Uniform/Gaussian/BirthDeath |
| Appendix A CTMC 入门 | — | 理论背景 |
| Appendix H 实验细节 | `config/` | CIFAR-10 / Piano 配置 |
