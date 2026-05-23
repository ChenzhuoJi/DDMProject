# 11 — 音频任务整体 Pipeline

> Lakh Pianoroll 单声道音乐条件生成在 CTMC 框架中的完整数据流
>
> 从乐谱序列 → 前向 CTMC → 条件训练 → 条件反向采样 → 评估

## 目录

- [1. 任务概述](#1-任务概述)
- [2. 数据层：Lakh Pianoroll 数据集](#2-数据层lakh-pianoroll-数据集)
- [3. 前向过程：UniformRate](#3-前向过程uniformrate)
- [4. 去噪网络：TransformerEncoder](#4-去噪网络transformerencoder)
- [5. 条件损失函数：ConditionalAux](#5-条件损失函数conditionalaux)
- [6. 训练配置与入口](#6-训练配置与入口)
- [7. 条件采样：ConditionalTauLeaping](#7-条件采样conditionaltauleaping)
- [8. 条件 PC 采样：ConditionalPCTauLeaping](#8-条件-pc-采样conditionalpctauleaping)
- [9. 评估指标](#9-评估指标)
- [10. 图像 vs 音频任务全景对比](#10-图像-vs-音频任务全景对比)

---

## 1. 任务概述

音频任务的核心目标是**条件音乐生成**（conditional music generation）：给定一首曲子的前 2 小节（32 个时间步），模型生成后续 14 小节（224 个时间步）的完整单声道旋律。

与图像任务的对比：

| | 图像任务 | 音频任务 |
|--|---------|---------|
| 任务类型 | 无条件生成 | 条件生成 |
| 数据 | CIFAR-10 (50k 张图) | Lakh Pianoroll (6k 训练, 950 测试) |
| 维度 D | 3072 (3×32×32) | 256 (时间步) |
| 状态空间 S | 256 (像素值) | 129 (128 音符 + 1 休止符) |
| 前向速率 | GaussianTargetRate | UniformRate |
| 网络架构 | UNet + 截断逻辑分布 | TransformerEncoder + FiLM |
| 损失函数 | GenericAux (无条件 CT-ELBO) | ConditionalAux (条件 CT-ELBO) |
| 采样算法 | TauLeaping / PCTauLeaping | ConditionalTauLeaping / ConditionalPCTauLeaping |

---

## 2. 数据层：Lakh Pianoroll 数据集

### 2.1 数据来源与筛选

`A Continuous Time Framework for Discrete Denoising Models.md:1588-1600`:

从 Lakh Pianoroll 数据集（174,154 个多轨 piano rolls）中筛选满足以下条件的序列：
1. **单声道**（monophonic）：每个时间步最多只有一个音符被弹奏
2. **无长静音**：不存在超过一小节（16 个时间步）的连续休止
3. **多样性**：序列中包含超过一种音符
4. **无长持续音**：没有同一个音符持续超过 50 个时间步

最终得到 **6000 个训练样本** 和 **950 个测试样本**。

### 2.2 数据格式

```
每个样本: 256 个时间步 (16 步/小节 × 16 小节)
每步状态: 129 个值 (128 种音符 + 1 个休止符)

D = 256  (序列长度)
S = 129  (状态空间大小)
```

### 2.3 状态空间加扰

论文中强调：**打乱 129 个状态的编号顺序，破坏天然的序数结构**（`scramble the ordering of the state space when mapping to Z to destroy any ordinal structure`）。这是因为音乐数据本质上是分类的（categorical），音符编号的数值顺序没有实际意义。

代码中通过 `descramble_key.txt` 文件实现加扰/解扰：

```python
# notebook (piano.ipynb:55-58):
descramble_key = np.loadtxt(path + '/descramble_key.txt')

def descramble(samples):
    return descramble_key[samples.flatten()].reshape(*samples.shape)
```

### 2.4 LakhPianoroll Dataset 类

`lib/datasets/datasets.py:54-68`:

```python
class LakhPianoroll(Dataset):
    def __init__(self, cfg, device):
        S = cfg.data.S           # 129
        L = cfg.data.shape[0]    # 256
        np_data = np.load(cfg.data.path)  # (N, 256), range [0, 128]
        self.data = torch.from_numpy(np_data).to(device)

    def __getitem__(self, index):
        return self.data[index]   # (256,) 整数序列
```

与图像任务的关键区别：

1. **数据是 1D 序列**而非 2D 图像，形状为 `(N, 256)`
2. **直接加载 numpy 文件**而非使用 torchvision
3. **无数据增强**（图像任务有随机水平翻转）
4. **不需要 reshape**，直接作为 flat 序列使用

---

## 3. 前向过程：UniformRate

`lib/models/models.py:155-190`:

### 3.1 为什么选择 UniformRate？

音频数据经过状态空间加扰后，相邻状态之间没有序数关系。因此：

- **GaussianTargetRate**（图像任务使用）只允许"数值相近"的状态间跳转，不适合分类数据
- **UniformRate** 允许状态空间中任意两个状态以相等速率相互跳转，非常适合分类数据

论文 Table 2 的消融实验证实：UniformRate (Hellinger=0.3765) 优于 BirthDeathRate (Hellinger=0.3928)。

### 3.2 数学定义

```
R_base(i, j) = rate_const     (i ≠ j)    ∀i,j ∈ {0, ..., 128}
R_base(i, i) = -Σ_{j≠i} rate_const

R_t = R_base                   (不随时间变化)
q_{t|0} = exp(t · R_base)     (矩阵指数)
```

### 3.3 配置参数

`config/train/piano.py:53-57`:

```python
model.rate_const = 0.03       # 均匀转移速率常数
model.rate_sigma = 3.0        # 不使用 (GaussianTargetRate 的参数)
model.Q_sigma = 20.0          # 不使用
model.time_exponential = 1000.0  # 不使用
model.time_base = 0.5         # 不使用
```

注意：`rate_const = 0.03` 控制噪声添加的速度。对于 UniformRate，速率矩阵是常数的，不随时间变化。

### 3.4 特征分解

`UniformRate.__init__` 中预计算了速率矩阵的特征分解：
```python
rate = rate_const * np.ones((S,S))
rate = rate - np.diag(np.diag(rate))        # 对角置零
rate = rate - np.diag(np.sum(rate, axis=1)) # 对角 = -行和
eigvals, eigvecs = np.linalg.eigh(rate)     # 对称矩阵，使用 eigh
```

由于 UniformRate 是对称矩阵（所有非对角元素相等），特征分解可以在初始化时一次性完成，后续 `transition(t)` 只需 O(S²) 的矩阵乘法。

---

## 4. 去噪网络：TransformerEncoder

### 4.1 架构概述

`lib/networks/networks.py:511-591`:

音频任务使用 Transformer 代替图像任务的 UNet，因为序列数据需要捕获长距离依赖关系（256 个时间步的上下文）。

```
输入: (B, 256), int, range [0, 128]
  │
  ├── One-hot 编码: (B, 256, 129)
  ├── Input Embedding: Linear(129 → 128)  →  (B, 256, 128)
  ├── Positional Encoding: 正弦位置编码    →  (B, 256, 128)
  │
  ├── ×6 TransformerEncoderLayer:
  │     ├── Multihead Self-Attention (8 heads)
  │     ├── FiLM: film_params × h + film_bias  (时间条件)
  │     └── Feed-Forward (128 → 2048 → 128)
  │
  ├── ×2 FFResidual:                      
  │     ├── Feed-Forward (128 → 2048 → 128)
  │     ├── LayerNorm + Residual
  │     └── FiLM (时间条件)
  │
  ├── Output Linear: Linear(128 → 129)     →  (B, 256, 129)
  └── Residual: + one_hot_x               →  (B, 256, 129)
```

### 4.2 关键设计细节

#### 4.2.1 One-hot 输入与残差连接

`TransformerEncoder.forward:564-584`:

```python
one_hot_x = F.one_hot(x, num_classes=S)   # (B, 256, 129)
x = self.input_embedding(one_hot_x.float()) # (B, 256, 128)

# ... 经过 Transformer ...

x = self.output_linear(x)                  # (B, 256, 129)
x = x + one_hot_x                          # 残差连接 (skip connection)
```

这被称为**残差偏置**（residual bias）：将输入的 one-hot 编码加到输出 logits 上。这强制模型以输入为基准做调整，类似图像任务中的截断逻辑分布偏置，但实现更简单。

#### 4.2.2 FiLM 层 (Feature-wise Linear Modulation)

`TransformerEncoderLayer:467-490`:

```python
class TransformerEncoderLayer:
    def forward(self, x, temb):
        film_params = self.film_from_temb(temb)  # (B, 2*K)
        x = self.norm1(x + self_attention(x))
        x = film_params[:, None, 0:K] * x + film_params[:, None, K:]
        x = self.norm2(x + ff_block(x))
        x = film_params[:, None, 0:K] * x + film_params[:, None, K:]
        return x
```

时间信息通过 FiLM 注入到每一层，而不是像 UNet 那样只在 ResBlock 中加一次。这使得 Transformer 每一层都能感知时间步。

#### 4.2.3 时间嵌入

`TransformerEncoder.forward:559-563`:

```python
temb = self.temb_net(
    network_utils.transformer_timestep_embedding(
        times * self.time_scale_factor, self.temb_dim  # 128
    )
)
# temb_net: Linear(128 → 2048) → ReLU → Linear(2048 → 512)
```

与图像任务的 UNet 使用相同的时间嵌入，但通过一个 MLP 将维度从 128 拓展到 512（= 4×128），因为 Transformer 的 FiLM 需要 2×d_model = 256 的参数，而每个 FiLM 层都有一份。

#### 4.2.4 自注意力

`TransformerEncoderLayer:450-490`:

使用标准 PyTorch `nn.MultiheadAttention`，8 头注意力。整个序列长度为 256，注意力矩阵大小为 256×256，计算复杂度 O(L²) = O(256²) 在可接受范围内。

### 4.3 模型复合

`lib/models/models.py:455-462`:

```python
@model_utils.register_model
class UniformRateSequenceTransformerEMA(EMA, SequenceTransformer, UniformRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        SequenceTransformer.__init__(self, cfg, device, rank)
        UniformRate.__init__(self, cfg, device)
        self.init_ema()
```

三重继承：
1. **EMA**: 指数移动平均（训练稳定，采样时使用平滑参数）
2. **SequenceTransformer**: Transformer 架构，实现 `forward(x, t) → logits`
3. **UniformRate**: 前向速率矩阵，提供 `rate(t)` 和 `transition(t)`

`SequenceTransformer.forward` 直接将离散整数输入传入网络：
```python
def forward(self, x, times):
    B, D = x.shape
    logits = self.net(x.long(), times.long())  # 直接传 long 类型
    return logits
```

---

## 5. 条件损失函数：ConditionalAux

### 5.1 与 GenericAux 的异同

`lib/losses/losses.py:250-493`:

`ConditionalAux` 是 `GenericAux` 的条件版本。核心差异：

```
GenericAux (无条件):                ConditionalAux (条件):
  minibatch: (B, D)                   minibatch: (B, D)
  ↓                                   ↓
  D 个维度全部参与损失计算              conditioner = minibatch[:, 0:32]
                                      data = minibatch[:, 32:]      (d = 224)
                                      ↓
                                      x_t, x_tilde 只在 data 部分采样
                                      ↓
                                      model_input = concat(conditioner, data_noisy)
                                      ↓
                                      模型输出前 32 维 masked 掉
                                      只取后 224 维计算 loss
```

### 5.2 核心代码逻辑

`ConditionalAux.calc_loss:263-493`:

```python
def calc_loss(self, minibatch, state, writer):
    B, D = minibatch.shape
    conditioner = minibatch[:, 0:self.condition_dim]  # (B, 32)
    data = minibatch[:, self.condition_dim:]           # (B, 224)
    d = data.shape[1]                                  # 224

    # === 从前向 CTMC 采样 x_t (加噪) ===
    qt0_rows_reg = qt0[..., data, :]      # (B*d, S)
    x_t = Categorical(qt0_rows_reg).sample().view(B, d)

    # === 采样 x_tilde (再跳一步) ===
    # [同 GenericAux, 只在 data 维度上操作]

    # === 模型前向 (条件) ===
    model_input = torch.concat((conditioner, x_tilde), dim=1)  # (B, 256)
    x_logits_full = model(model_input, ts)                     # (B, 256, 129)
    x_logits = x_logits_full[:, self.condition_dim:, :]        # (B, 224, 129)

    # === 计算 Reg 项 (只在 data 部分) ===
    # [同 GenericAux, d=224, S=129]

    # === 计算 Sig 项 (只在 data 部分) ===
    # [同 GenericAux, d=224, S=129]

    # === 辅助 NLL (只在 data 部分) ===
    nll = self.cross_ent(x_logits.permute(0,2,1), data.long())

    return neg_elbo + self.nll_weight * nll
```

### 5.3 配置参数

`config/train/piano.py:16-22`:

```python
loss.name = 'ConditionalAux'
loss.eps_ratio = 1e-9
loss.nll_weight = 0.001          # 辅助 NLL 权重 λ = 0.001
loss.min_time = 0.01             # 最小时间，避免 t=0 处的不稳定性
loss.condition_dim = 32          # 条件维度（前 32 个时间步 = 2 小节）
loss.one_forward_pass = True     # 单次前向技巧（同图像任务）
```

---

## 6. 训练配置与入口

### 6.1 完整训练配置

`config/train/piano.py`:

```python
config.experiment_name = 'piano'
config.device = 'cpu'             # 可改为 GPU
config.data.name = 'LakhPianoroll'
config.data.path = 'path/to/train.npy'
config.data.S = 129
config.data.batch_size = 64
config.data.shape = [256]         # 一维序列

training.n_iters = 1000000        # 1M 步
training.warmup = 5000            # 线性 warmup
training.clip_grad = True         # 梯度裁剪 (norm=1.0)

optimizer.name = 'Adam'
optimizer.lr = 2e-4               # 学习率

model.name = 'UniformRateSequenceTransformerEMA'
model.num_layers = 6              # Transformer 层数
model.d_model = 128               # 模型维度
model.num_heads = 8               # 注意力头数
model.dim_feedforward = 2048      # FFN 隐藏层维度
model.dropout = 0.1               # Dropout 率
model.temb_dim = 128              # 时间嵌入维度
model.num_output_FFresiduals = 2  # 输出残差 FFN 层数
model.time_scale_factor = 1000    # 时间缩放因子
model.use_one_hot_input = True    # 使用 one-hot 输入
model.ema_decay = 0.9999          # EMA 衰减率
model.rate_const = 0.03           # UniformRate 速率常数
```

### 6.2 训练入口

`train.py:129-130`:

```python
elif args.config == 'piano':
    from config.train.piano import get_config
```

命令行启动：`python train.py piano`

训练循环与图像任务完全一致（`Standard.step`），唯一区别是 `ConditionalAux.calc_loss` 的内部逻辑不同。

---

## 7. 条件采样：ConditionalTauLeaping

### 7.1 采样器注册

`lib/sampling/sampling.py:242-351`:

```python
@sampling_utils.register_sampler
class ConditionalTauLeaping():
    def sample(self, model, N, num_intermediates, conditioner):
```

与 `TauLeaping` 的三点核心差异：

### 7.2 差异 1：条件-噪声拼接

无条件采样：模型输入的只是噪声 `x`
条件采样：模型输入 = `[conditioner, x_noise]` 拼接

```python
# ConditionalTauLeaping:
model_input = torch.concat((conditioner, x), dim=1)  # (N, 256)
p0t = F.softmax(model(model_input, t), dim=2)        # (N, 256, 129)
p0t = p0t[:, condition_dim:, :]                      # (N, 224, 129) 只取采样部分
```

### 7.3 差异 2：只对采样部分做 Tau-Leaping

```python
# 初始采样维度 = sample_D = 256 - 32 = 224
x = get_initial_samples(N, sample_D, device, S, initial_dist, ...)

# 反向速率只对 224 维计算
reverse_rates.shape  # (N, 224, 129)
```

条件部分（前 32 维）在整个采样过程中保持不变。

### 7.4 差异 3：拒绝多次跳跃

`reject_multiple_jumps = True`（仅音频任务使用）：

```python
jump_nums = poisson_dist.sample()  # (N, 224, 129)

if reject_multiple_jumps:
    jump_num_sum = torch.sum(jump_nums, dim=2)           # 每个维度的总跳跃次数
    jump_num_sum_mask = jump_num_sum <= 1                # 只保留 ≤1 次跳跃的维度
    masked_jump_nums = jump_nums * jump_num_sum_mask.unsqueeze(-1)
    adj_diffs = masked_jump_nums * diffs
```

为什么需要拒绝多次跳跃？因为**音频状态是纯分类的**（categorical），跳跃到状态 42 再跳到状态 7 和直接跳到状态 7 是等价的。允许多次跳跃相当于允许一步内跳过多步，这在分类空间中没有物理意义。

论文 §H.3 指出拒绝率通常为 0（Figure 12），仅在少数步骤略有上升。

### 7.5 采样输出

```python
# 最终预测 p_{0|t} (t=min_time)
model_input = torch.concat((conditioner, x), dim=1)
p_0gt = F.softmax(model(model_input, min_t), dim=2)
p_0gt = p_0gt[:, condition_dim:, :]
x_0max = torch.max(p_0gt, dim=2)[1]

# 拼接条件部分，形成完整输出
output = torch.concat((conditioner, x_0max), dim=1)  # (N, 256)
```

### 7.6 采样配置

`config/eval/piano.py`:

```python
sampler.name = 'ConditionalTauLeaping'  # 或 ConditionalPCTauLeaping
sampler.num_steps = 1000
sampler.min_t = 0.01
sampler.initial_dist = 'uniform'        # 均匀初始分布
sampler.condition_dim = 32
sampler.reject_multiple_jumps = True
```

---

## 8. 条件 PC 采样：ConditionalPCTauLeaping

`lib/sampling/sampling.py:354-490`:

在条件 Tau-Leaping 的基础上添加 Predictor-Corrector 校正步。

### 8.1 PC 校正

```python
if t <= corrector_entry_time:
    for _ in range(num_corrector_steps):
        transpose_forward_rates, reverse_rates, _ = get_rates(x, t-h)
        corrector_rate = transpose_forward_rates + reverse_rates
        # 取对应采样维度
        corrector_rate = corrector_rate[:, condition_dim:, :]
        x = take_poisson_step(x, corrector_rate, multiplier * h)
```

### 8.2 配置参数

```python
sampler.num_corrector_steps = 2          # 每步 2 次校正
sampler.corrector_step_size_multiplier = 0.1  # 校正步长缩小系数
sampler.corrector_entry_time = 0.9       # t < 0.9T 后才开始校正
sampler.reject_multiple_jumps = True     # 拒绝多次跳跃
```

与图像任务（`PCTauLeaping`）的核心差异：校正步长缩小了 10 倍（`0.1` 对比图像任务的 `1.5`），因为音频数据的跳跃更敏感。

---

## 9. 评估指标

### 9.1 评估配置

`config/eval/piano.py:31-43`:

```python
sampler.test_dataset = pianoroll_dataset_path + '/test.npy'
sampler.condition_dim = 32
```

### 9.2 Notebook 采样流程

`notebooks/piano.ipynb`:

```python
# 从测试集取一个样本的前 32 步作为条件
test_data_idx = 8
conditioner = torch.from_numpy(
    test_dataset[test_data_idx, 0:condition_dim]
).to(device).view(1, condition_dim)

# 条件采样
sampler = sampling_utils.get_sampler(eval_cfg)
samples, x_hist, x0_hist = sampler.sample(model, 1, 10, conditioner)

# 解扰后可视化
samples = descramble(samples)
plt.scatter(np.arange(256), samples[0], alpha=0.5)  # 生成
plt.scatter(np.arange(256), test_data[test_data_idx], alpha=0.5)  # 真实
```

### 9.3 论文中的评估指标

论文 Table 2 使用两种指标：

1. **Hellinger Distance**: 衡量生成音符分布与真实分布之间的差异
   ```
   H(p||q) = (1/√2) · √(Σ(√p_i - √q_i)²)
   ```
   值越小越好。

2. **Proportion of Outliers**: 生成样本中出现在真实样本之外的音符比例
   值越小越好。

| 模型 | Hellinger Distance ↓ | Outliers ↓ |
|-----|---------------------|------------|
| τLDR-0 Birth/Death | 0.3928 | 0.1316 |
| τLDR-0 Uniform | **0.3765** | **0.1106** |
| τLDR-2 Uniform | **0.3762** | **0.1091** |
| D3PM Uniform [8] | 0.3839 | 0.1137 |

结论：UniformRate 优于 Birth/Death 速率，τLDR-2（2 步校正器）进一步提升质量，整体优于离散时间方法 D3PM。

---

## 10. 图像 vs 音频任务全景对比

### 10.1 配置对比

| 配置项 | CIFAR-10 (图像) | Lakh Pianoroll (音频) |
|-------|----------------|---------------------|
| 配置类 | `config/train/cifar10.py` | `config/train/piano.py` |
| 模型 | `GaussianTargetRateImageX0PredEMA` | `UniformRateSequenceTransformerEMA` |
| 数据类 | `DiscreteCIFAR10` | `LakhPianoroll` |
| 损失类 | `GenericAux` | `ConditionalAux` |
| 训练步 | `Standard` | `Standard` |
| 优化器 | `Adam(lr=2e-4)` | `Adam(lr=2e-4)` |
| 采样器 | `TauLeaping` / `PCTauLeaping` | `ConditionalTauLeaping` / `ConditionalPCTauLeaping` |

### 10.2 模型架构对比

| 组件 | UNet (图像) | TransformerEncoder (音频) |
|-----|------------|--------------------------|
| 主干 | 卷积 ResBlock + 跳跃连接 | 自注意力 + 逐位 FFN |
| 位置信息 | 隐式 (卷积空间结构) | 显式 (PositionalEncoding) |
| 时间条件 | ResBlock 中加 temb | 每层通过 FiLM 注入 |
| 输出 | 截断逻辑分布参数 (μ, log_scale) | 直接 logits + one-hot 残差 |
| 参数量 | ~7.4M | ~7.6M (估计) |

### 10.3 速率矩阵对比

| 属性 | GaussianTargetRate (图像) | UniformRate (音频) |
|-----|--------------------------|-------------------|
| 非对角元素 | 高斯距离衰减 | 全相等常数 |
| 时间依赖 | β(t) 指数增长 | 不随时间变化 |
| 适用场景 | 序数数据 (ordinal) | 分类数据 (categorical) |
| 特征分解 | 非对称 (eig + inv) | 对称 (eigh) |
| 数学形式 | R_t = β(t) · R_base | R_t = R_base (常数) |

### 10.4 采样器对比

| 特性 | TauLeaping (图像) | ConditionalTauLeaping (音频) |
|-----|------------------|------------------------------|
| 起始分布 | gaussian (高斯偏好) | uniform (均匀) |
| 维度 | 3072 (全部) | 224 (后 14 小节) |
| 条件 | 无 | 前 32 维固定 |
| 拒绝多跳 | 否 | 是 (reject_multiple_jumps) |
| 校正步步长 | 1.5× 预测步长 | 0.1× 预测步长 |

### 10.5 端到端数据流图示

```
训练:
  LakhPianoroll: (N, 256) 离散整数 [0, 128]
    │
    ├── conditioner = [:, 0:32]   (前 2 小节，固定)
    ├── data = [:, 32:]           (后 14 小节，待生成)
    │
    ▼
  ConditionalAux.calc_loss:
    ├── x_t ~ q_{t|0}(·|data)          (加噪)
    ├── x_tilde = jump(x_t, R_t)       (再跳一步)
    ├── model(concat(conditioner, x_tilde), t)  (前向)
    ├── 只取后 224 维 → Reg + Sig + NLL
    └── neg_elbo + 0.001 × nll

采样:
  conditioner = test_data[0:32]    (条件)
  x ~ Uniform(0, 128)              (初始噪声, 224 维)
    │
    for t in [1.0, 0.01]:
    │   model_input = concat(conditioner, x)
    │   p0t = softmax(model(model_input, t))[:, 32:, :]
    │   reverse_rates = fwd_rates × (p0t / q_{t|0}) @ q_{t|0}
    │   x = tau_leap(x, reverse_rates, h)
    │
    x_final = argmax(model(concat(conditioner, x), 0.01))[:, 32:, :]
    output = concat(conditioner, x_final)  # (256,)
    │
    ▼
  descramble(output) → 恢复原始音符编号
    │
    ▼
  Hellinger Distance / Outlier Proportion vs ground truth
```
