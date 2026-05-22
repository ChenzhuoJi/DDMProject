# 03 — UNet 架构精读

> CTMC 论文代码中的 UNet 实现 (CIFAR-10 图像生成)
>
> 文件: `lib/networks/networks.py:189-424`, `lib/models/models.py:14-97`
> 配置: `config/train/cifar10.py`

## 目录

- [1. 设计背景](#1-设计背景)
- [2. 配置参数](#2-配置参数)
- [3. 整体结构总览](#3-整体结构总览)
- [4. 核心组件详解](#4-核心组件详解)
- [5. 前向路径逐层分解](#5-前向路径逐层分解)
- [6. 时间嵌入机制](#6-时间嵌入机制)
- [7. 输出层：截断逻辑分布](#7-输出层截断逻辑分布)
- [8. 模型包装器 ImageX0PredBase](#8-模型包装器-imagex0predbase)
- [9. 与 DDPM / score-SDE UNet 的对比](#9-与-ddpm--score-sde-unet-的对比)

---

## 1. 设计背景

本 UNet 代码移植自 [Yang Song 的 score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)，而后者又源自 DDPM [Ho et al., 2020] 的 UNet 实现。

**关键词**: score-SDE 风格 UNet

与标准 UNet [Ronneberger et al., 2015] 的关键区别:

| 特性 | 标准 UNet (医学图像) | 本 UNet (扩散模型) |
|------|---------------------|-------------------|
| 下采样 | Max Pooling | Strided Conv (stride=2) |
| 上采样 | Transposed Conv | Nearest Interpolation + Conv |
| 跳跃连接 | Concat | Concat (但 ResBlock 内有 `/√2` 重缩放) |
| 时间条件 | ❌ | ✅ FiLM 风格 (偏置 + 缩放) |
| 归一化 | BatchNorm | GroupNorm |
| 激活 | ReLU | SiLU (Swish) |
| 自注意力 | ❌ | ✅ 在中层使用 |
| 输出 | 分割图 (单值) | 截断逻辑分布参数 (μ, log_scale) |

---

## 2. 配置参数

来自 `config/train/cifar10.py`:

```python
model.name = 'GaussianTargetRateImageX0PredEMA'

# UNet 结构参数
model.ch = 128                    # 基础通道数
model.num_res_blocks = 2          # 每层残差块数
model.num_scales = 4              # 下采样层级数
model.ch_mult = [1, 2, 2, 2]      # 每层通道倍数 → [128, 256, 256, 256]
model.input_channels = 3          # 输入通道 (RGB)
model.scale_count_to_put_attn = 1 # 在第 2 层 (0-indexed) 加自注意力
model.data_min_max = [0, 255]     # 输入数据范围
model.dropout = 0.1               # Dropout 率
model.skip_rescale = True         # 跳跃连接重缩放
model.time_embed_dim = model.ch   # 时间嵌入维度 (= 128)
model.time_scale_factor = 1000    # 时间缩放因子
model.fix_logistic = False        # 截断逻辑修正

# 输出通道: 2 * input_channels = 6 (前 3 为 μ, 后 3 为 log_scale)
```

**结构推导**:

```
ch = 128
num_scales = 4
ch_mult = [1, 2, 2, 2]

每层通道数:
  Scale 0: 128 * 1 = 128
  Scale 1: 128 * 2 = 256
  Scale 2: 128 * 2 = 256
  Scale 3: 128 * 2 = 256

每层特征图尺寸:
  Input:  32 × 32
  Scale 0: 32 × 32 (下采样到 16 × 16)
  Scale 1: 16 × 16 (下采样到 8 × 8)
  Scale 2: 8 × 8   (下采样到 4 × 4)
  Scale 3: 4 × 4   (最低分辨率)
```

---

## 3. 整体结构总览

```
输入: (B, 3, 32, 32), 像素值 [0, 255]
     │
     │ _center_data: 归一化到 [-1, 1]
     │ x ← (x - 0) / (255 - 0) * 2 - 1
     ▼
┌──────────────────────────────────────────────────────┐
│ 1. _time_embedding(t)                                 │
│     t ∈ [0, 1] → t × 1000 → sine embed (128)         │
│     → Linear(128, 512) → SiLU → Linear(512, 512)     │
│     → temb (B, 512)                                   │
└──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│ 2. _do_input_conv                                     │
│     Conv2D(3 → 128, 3×3, pad=1)                      │
│     → h (B, 128, 32, 32)                              │
│     → hs = [h]  ← 存储跳跃连接                        │
└──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│ 3. _do_downsampling                                   │
│                                                       │
│  [Scale 0: 32×32, ch=128]                            │
│    ├─ ResBlock(128→128) + temb                        │
│    ├─ ResBlock(128→128) + temb                        │
│    └─ Downsample(128, 32→16)  → hs += [128, 16]      │
│                                                       │
│  [Scale 1: 16×16, ch=256]  ← Attn 在此层              │
│    ├─ ResBlock(128→256) + temb  ← 通道倍增             │
│    ├─ AttnBlock(256)            ← 自注意力              │
│    ├─ ResBlock(256→256) + temb                        │
│    └─ Downsample(256, 16→8)  → hs += [256, 8]        │
│                                                       │
│  [Scale 2: 8×8, ch=256]                              │
│    ├─ ResBlock(256→256) + temb                        │
│    ├─ ResBlock(256→256) + temb                        │
│    └─ Downsample(256, 8→4)  → hs += [256, 4]         │
│                                                       │
│  [Scale 3: 4×4, ch=256]                              │
│    ├─ ResBlock(256→256) + temb                        │
│    ├─ ResBlock(256→256) + temb                        │
│    └─ (无下采样, 最低分辨率)                            │
│                                                       │
└──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│ 4. _do_middle                                         │
│    ├─ ResBlock(256→256) + temb                        │
│    ├─ AttnBlock(256)            ← 自注意力              │
│    └─ ResBlock(256→256) + temb                        │
└──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│ 5. _do_upsampling                                     │
│                                                       │
│  [Scale 2: 4→8, ch=256]                              │
│    ├─ ResBlock(256+256→256) + temb  ← 跳跃连接拼接      │
│    ├─ ResBlock(256→256) + temb                        │
│    ├─ ResBlock(256→256) + temb                        │
│    └─ Upsample(256, 4→8)                              │
│                                                       │
│  [Scale 1: 8→16, ch=256]  ← Attn 在此层               │
│    ├─ ResBlock(256+256→256) + temb                    │
│    ├─ AttnBlock(256)                                  │
│    ├─ ResBlock(256→256) + temb                        │
│    ├─ ResBlock(256→256) + temb                        │
│    └─ Upsample(256, 8→16)                             │
│                                                       │
│  [Scale 0: 16→32, ch=128]                             │
│    ├─ ResBlock(256+128→128) + temb  ← 通道降回 128      │
│    ├─ ResBlock(128→128) + temb                        │
│    ├─ ResBlock(128→128) + temb                        │
│    └─ (无上采样, 回到原分辨率 32×32)                    │
│                                                       │
└──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│ 6. _do_output                                         │
│     GroupNorm(128) → SiLU → Conv2D(128→6, 3×3)      │
│     → h (B, 6, 32, 32)                                 │
└──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│ 7. _logistic_output_res                               │
│     h[:, 0:3] ← tanh(centered_x_in + h[:, 0:3])      │
│     → h (B, 6, 32, 32)  ← 前 3 μ, 后 3 log_scale     │
└──────────────────────────────────────────────────────┘
     │
     ▼
输出: (B, 6, 32, 32) → 在 ImageX0PredBase 中转换为 (B, 3072, 256) logits
```

---

## 4. 核心组件详解

### 4.1 ResBlock (`networks.py:98-158`)

```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim, dropout=0.1, skip_rescale=True):
        self.groupnorm0 = GroupNorm(min(in_ch//4, 32), in_ch)
        self.conv0 = Conv2d(in_ch, out_ch, 3, padding=1)
        
        if temb_dim is not None:
            self.dense0 = Linear(temb_dim, out_ch)  # 时间条件偏置
            nn.init.zeros_(self.dense0.bias)
        
        self.groupnorm1 = GroupNorm(min(out_ch//4, 32), out_ch)
        self.dropout0 = Dropout(dropout)
        self.conv1 = Conv2d(out_ch, out_ch, 3, padding=1)
        
        if out_ch != in_ch:
            self.nin = NiN(in_ch, out_ch)  # 1×1 卷积投影
    
    def forward(self, x, temb=None):
        h = GroupNorm(x) → SiLU → Conv2d(in_ch, out_ch)
        
        if temb is not None:
            h += self.dense0(SiLU(temb))[:, :, None, None]
        
        h = GroupNorm(h) → SiLU → Dropout → Conv2d(out_ch, out_ch)
        
        if in_ch != out_ch:
            x = self.nin(x)  # 1×1 卷积匹配通道
        
        if self.skip_rescale:
            return (x + h) / sqrt(2)  # 重缩放保持方差
        else:
            return x + h
```

**设计要点**:

| 设计 | 原因 |
|------|------|
| GroupNorm (min(ch//4, 32)) | BatchNorm 不适合小 batch 或可变 batch 大小 |
| SiLU (Swish) | 比 ReLU 更平滑，扩散模型中广泛使用 |
| 跳跃连接 `/√2` | 保持通道拼接后的方差稳定，避免深层梯度爆炸 |
| FiLM 偏置 `dense0(temb)[:,:,None,None]` | 时间信息通过广播加到所有空间位置 |
| `nn.init.zeros_(dense0.bias)` | 初始化时间条件为 0，使早期训练不被时间信号主导 |

### 4.2 AttnBlock (`networks.py:64-95`)

```python
class AttnBlock(nn.Module):
    def __init__(self, channels, skip_rescale=True):
        self.GroupNorm_0 = GroupNorm(min(channels//4, 32), channels)
        self.NIN_0 = NiN(channels, channels)  # Q
        self.NIN_1 = NiN(channels, channels)  # K
        self.NIN_2 = NiN(channels, channels)  # V
        self.NIN_3 = NiN(channels, channels, init_scale=0.)  # 输出投影
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)  # (B, C, H, W)
        k = self.NIN_1(h)  # (B, C, H, W)
        v = self.NIN_2(h)  # (B, C, H, W)
        
        # 注意力权重: (B, H, W, H, W)
        w = einsum('bchw,bcij→bhwij', q, k) * (C ** (-0.5))
        w = softmax(w.reshape(B, H, W, H*W), dim=-1)
        w = w.reshape(B, H, W, H, W)
        h = einsum('bhwij,bcij→bchw', w, v)
        h = self.NIN_3(h)  # 输出投影, init_scale=0 从零开始
        
        if self.skip_rescale:
            return (x + h) / sqrt(2)
        else:
            return x + h
```

**设计要点**:

| 设计 | 说明 |
|------|------|
| **通道注意力** (非空间) | 注意力是在空间维度 (H×W) 上的，即每个像素关注其他所有像素 |
| **NiN (Network-in-Network)** | 1×1 卷积实现线性投影，替代独立的 Linear 层 |
| **output init scale = 0** | `NIN_3` 的输出投影初始化为零，使注意力模块从残差开始 |
| **`/√2` 重缩放** | 同 ResBlock，稳定深层梯度 |
| **位置** | 在 Scale 1 和 Middle 各放一个 AttnBlock |

**注意力计算过程**:

```
q, k, v: (B, C, H, W)
  
w[b, h, w, i, j] = Σ_c q[b,c,h,w] · k[b,c,i,j] / √C
  → 位置 (h,w) 和位置 (i,j) 之间的相似度
  
softmax 在最后一个维度归一化:
  w[b, h, w, :] = softmax over all (i,j)

输出:
  h[b, c, h, w] = Σ_{i,j} w[b, h, w, i, j] · v[b, c, i, j]
  → 每个位置聚合所有位置的特征
```

### 4.3 Downsample (`networks.py:160-173`)

```python
class Downsample(nn.Module):
    def __init__(self, channels):
        self.conv = Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = F.pad(x, (0, 1, 0, 1))  # pad right+bottom 各 1 像素
        x = self.conv(x)             # stride=2 → 尺寸减半
        return x
```

> 用 **Strided Conv (stride=2)** 替代 MaxPool 做下采样。先 pad (0,1,0,1) 再卷积，相当于 padding=1 但更精细地控制边界。

### 4.4 Upsample (`networks.py:175-187`)

```python
class Upsample(nn.Module):
    def __init__(self, channels):
        self.conv = Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = F.interpolate(x, (H*2, W*2), mode='nearest')  # 最近邻放大
        h = self.conv(h)                                    # 抗锯齿卷积
        return h
```

> 用 `nearest` 插值 + Conv2d 替代转置卷积。这种组合在实践中能减少棋盘 artifacts。

### 4.5 NiN (`networks.py:49-62`)

```python
class NiN(nn.Module):
    def __init__(self, in_ch, out_ch, init_scale=0.1):
        self.W = Parameter(default_init(scale=init_scale)((in_ch, out_ch)))
        self.b = Parameter(zeros(out_ch))
```

等价于 `Conv2d(in_ch, out_ch, 1, 1)`，用于 ResBlock 的通道投影和 AttnBlock 的 Q/K/V 投影。

---

## 5. 前向路径逐层分解

以 batch=4 为例:

### 输入

```
x_in: (4, 3, 32, 32), dtype=uint8, range [0, 255]
```

### Layer 1: 输入预处理

```python
h = _center_data(x_in)  # (4, 3, 32, 32), dtype=float32, range [-1, 1]
centered_x_in = h       # 保存用于 logistic 残差

temb = _time_embedding(times)  # (4, 512)
```

### Layer 2: 输入卷积

```python
h = input_conv(h)  # (4, 128, 32, 32)
hs = [h]            # 存储跳跃连接
```

### Layer 3: 下采样

```
模块列表: downsampling_modules = [16 个模块]

索引 | 模块          | 输入形状           | 输出形状            | hs 追加
-----|---------------|--------------------|--------------------|---------
 0   | ResBlock + t  | (4, 128, 32, 32)   | (4, 128, 32, 32)   | ✓
 1   | ResBlock + t  | (4, 128, 32, 32)   | (4, 128, 32, 32)   | ✓
 2   | Downsample    | (4, 128, 32, 32)   | (4, 128, 16, 16)   | ✓
     |               | [Scale 0 结束]      |                     |
 3   | ResBlock + t  | (4, 128, 16, 16)   | (4, 256, 16, 16)   | ✓ ← 通道翻倍
 4   | AttnBlock     | (4, 256, 16, 16)   | (4, 256, 16, 16)   | ✓
 5   | ResBlock + t  | (4, 256, 16, 16)   | (4, 256, 16, 16)   | ✓
 6   | Downsample    | (4, 256, 16, 16)   | (4, 256, 8, 8)     | ✓
     |               | [Scale 1 结束]      |                     |
 7   | ResBlock + t  | (4, 256, 8, 8)     | (4, 256, 8, 8)     | ✓
 8   | ResBlock + t  | (4, 256, 8, 8)     | (4, 256, 8, 8)     | ✓
 9   | Downsample    | (4, 256, 8, 8)     | (4, 256, 4, 4)     | ✓
     |               | [Scale 2 结束]      |                     |
10   | ResBlock + t  | (4, 256, 4, 4)     | (4, 256, 4, 4)     | ✓
11   | ResBlock + t  | (4, 256, 4, 4)     | (4, 256, 4, 4)     | ✓
     |               | [Scale 3 结束]      |                     |

hs 最终长度: 1 (input_conv) + 5×2 (每层 2 ResBlock) + 4 (AttnBlock) + 3 (Downsample) = 16
```

### Layer 4: Middle

```
模块列表: middle_modules = [3 个模块]

索引 | 模块          | 输入形状           | 输出形状
-----|---------------|--------------------|--------------------
 0   | ResBlock + t  | (4, 256, 4, 4)     | (4, 256, 4, 4)
 1   | AttnBlock     | (4, 256, 4, 4)     | (4, 256, 4, 4)
 2   | ResBlock + t  | (4, 256, 4, 4)     | (4, 256, 4, 4)
```

### Layer 5: 上采样

```
模块列表: upsampling_modules = [16 个模块]
(与下采样对称, 但每层有 num_res_blocks+1 = 3 个 ResBlock)

索引 | 模块          | 跳跃连接 (hs pop)    | 输入→输出形状       | 备注
-----|---------------|---------------------|--------------------|------
     | [Scale 2: 4→8] |                      |                    |
 0   | ResBlock + t  | (4, 256, 4, 4)     | (4, 512→256, 4,4)  | concat 后 512→256
 1   | ResBlock + t  | —                   | (4, 256, 4, 4)     |
 2   | ResBlock + t  | —                   | (4, 256, 4, 4)     |
 3   | Upsample      | —                   | (4, 256, 4→8, 4→8) |
     | [Scale 1: 8→16] |                     |                     |
 4   | ResBlock + t  | (4, 256, 8, 8)     | (4, 512→256, 8, 8) |
 5   | AttnBlock     | —                   | (4, 256, 8, 8)     |
 6   | ResBlock + t  | —                   | (4, 256, 8, 8)     |
 7   | ResBlock + t  | —                   | (4, 256, 8, 8)     |
 8   | Upsample      | —                   | (4, 256, 8→16, 8→16)|
     | [Scale 0: 16→32] |                     |                     |  ← ch_mult[0] = 1
 9   | ResBlock + t  | (4, 128, 16, 16)   | (4, 384→128, 16,16) | concat: 256+128=384
10   | ResBlock + t  | —                   | (4, 128, 16, 16)   |
11   | ResBlock + t  | —                   | (4, 128, 16, 16)   |
     | (无 Upsample) | 已回到 32×32       |                     |

hs 最终应为空列表
```

### Layer 6: 输出

```python
h = GroupNorm(128)(h)    # (4, 128, 32, 32)
h = SiLU(h)
h = Conv2d(128→6, 3, 1)(h)  # (4, 6, 32, 32)
```

### Layer 7: 截断逻辑残差

```python
h[:, 0:3] = tanh(centered_x_in + h[:, 0:3])
# → 输出 (4, 6, 32, 32)
#   前 3 通道: μ (每个通道一个)
#   后 3 通道: log_scale (每个通道一个)
```

---

## 6. 时间嵌入机制

### 6.1 正弦嵌入

```python
# network_utils.py:7
def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    half_dim = embedding_dim // 2  # 64
    emb = log(max_positions) / (half_dim - 1)  # log(10000) / 63 ≈ 0.146
    emb = exp(arange(64) * -emb)  # 指数衰减频率
    emb = timesteps[:, None] * emb[None, :]  # (B, 64)
    emb = cat([sin(emb), cos(emb)], dim=1)   # (B, 128)
    return emb
```

### 6.2 MLP 投影

```python
# UNet._time_embedding
temb = transformer_timestep_embedding(times * 1000, 128)  # 输入 [0,1] → [0,1000]
temb = Linear(128, 512)(temb)       # 升维
temb = Linear(512, 512)(SiLU(temb)) # 非线性变换
# → temb (B, 512)
```

### 6.3 FiLM 注入

在每个 ResBlock 中:

```python
h += self.dense0(SiLU(temb))[:, :, None, None]
```

`dense0`: `Linear(512 → out_ch)`，输出通过广播加到每个空间位置。

> **为什么不使用完整的 FiLM (γh + β) 而只用偏置?** 这是 score-SDE UNet 的设计选择。部分扩散模型实现使用完整的缩放 + 偏置，但本代码仅使用偏置注入。

---

## 7. 输出层：截断逻辑分布

这部分在 `ImageX0PredBase.forward` (`models.py:47-90`) 中，不在 UNet 内部，但必须理解才能看懂完整的前向路径。

### 7.1 动机

UNet 输出 6 个通道: 前 3 个是 RGB 各通道的均值 `μ`，后 3 个是 `log_scale` (对数尺度)。图像有 256 个离散像素值，因此需要将连续分布离散化为 256 个类别。

### 7.2 离散化过程

```python
# 将 UNet 输出转换为每个像素 256 个 logits

# 1. 从 UNet 提取参数
mu = net_out[:, 0:C, :, :].unsqueeze(-1)       # (B, 3, 32, 32, 1)
log_scale = net_out[:, C:, :, :].unsqueeze(-1)  # (B, 3, 32, 32, 1)
inv_scale = exp(-(log_scale - 2))               # 尺度倒数

# 2. 定义 256 个 bin 中心 (在 [-1, 1] 区间上等距)
bin_width = 2.0 / 256  # 0.0078125
bin_centers = linspace(-1 + bin_width/2, 1 - bin_width/2, 256)
# → (1, 1, 1, 1, 256)

# 3. 计算每个 bin 的 CDF 差值
# 左边界 CDF
sig_in_left = (bin_centers - bin_width/2 - mu) * inv_scale
bin_left_logcdf = logsigmoid(sig_in_left)
# 右边界 CDF
sig_in_right = (bin_centers + bin_width/2 - mu) * inv_scale
bin_right_logcdf = logsigmoid(sig_in_right)

# log(p(bin)) = log(Φ(right) - Φ(left))
logits = log(exp(bin_right_logcdf) - exp(bin_left_logcdf))  # (B, 3, 32, 32, 256)

# 4. 展平为 (B, D, S) 其中 D = 3×32×32 = 3072, S = 256
logits = logits.view(B, D, S)
```

### 7.3 可视化

```
logistic 分布:
         Φ(right) - Φ(left)
              ▼
  ┌───────────────────────────┐
  │   ░░░░░                   │ ← 每个 bin 的概率
  │ ░░░░░░░░░                 │
  │░░░░░░░░░░░░░             │
──┴───────┴───────┴───────┴──→
-1       bin_centers      +1
         (256 个)
```

### 7.4 数值稳定性

```python
# _log_minus_exp: log(exp(a) - exp(b)) 的稳定计算
def _log_minus_exp(self, a, b, eps=1e-6):
    return a + torch.log1p(-torch.exp(b - a) + eps)
```

---

## 8. 模型包装器 ImageX0PredBase

完整前向路径 (`models.py:47-90`):

```python
class ImageX0PredBase(nn.Module):
    def forward(self, x, times):
        B, D = x.shape              # (B, 3072) — 展平的离散像素
        C, H, W = self.data_shape   # (3, 32, 32)
        S = self.S                  # 256
        
        # 重塑为图像
        x = x.view(B, C, H, W)      # (B, 3, 32, 32)
        
        # UNet 前向
        net_out = self.net(x, times)  # (B, 6, 32, 32)
        
        # 截断逻辑分布 → logits
        logits = self._discretized_logistic(net_out, S)
        # logits (B, 3, 32, 32, 256)
        
        return logits.view(B, D, S)  # (B, 3072, 256)
```

---

## 9. 与 DDPM / score-SDE UNet 的对比

| 特性 | DDPM UNet | score-SDE UNet (本代码) | 说明 |
|------|-----------|----------------------|------|
| 归一化 | GroupNorm | GroupNorm | 相同 |
| 激活 | SiLU/Swish | SiLU/Swish | 相同 |
| 时间嵌入 | Transformer PE + MLP | Transformer PE + MLP | 相同 |
| 时间注入方式 | FiLM (γ·h + β) | 偏置 (h + β) | 简化版 |
| 跳跃连接重缩放 | ✅ `/√2` | ✅ `/√2` | 相同 |
| 注意力位置 | 16×16 层 + Middle | 16×16 层 + Middle | 相同 |
| 下采样 | Strided Conv | Strided Conv | 相同 |
| 上采样 | Interpolate + Conv | Interpolate + Conv | 相同 |
| 输出 | 预测噪声 ε | 截断逻辑 μ, log_scale | ❌ 关键差异 |
| 通道配置 | ch=128, mult=[1,2,2,2] | ch=128, mult=[1,2,2,2] | 相同 |
| 每层 ResBlock 数 | 2 | 2 | 相同 |
| 上采样 ResBlock 数 | num_res_blocks+1=3 | num_res_blocks+1=3 | 相同 |

**核心差异**: DDPM 的 UNet 预测添加的噪声 $\epsilon$ (与输入相同形状)，而这个 UNet 预测**截断逻辑分布的参数** (μ, log_scale)，因为数据是离散像素值而非连续噪声。

---

## 附录: 完整模块索引表

```
downsampling_modules (索引 0-15):
  [0]  ResBlock(in=128, out=128, t)
  [1]  ResBlock(in=128, out=128, t)
  [2]  Downsample(128, 32→16)
  [3]  ResBlock(in=128, out=256, t)    ← 通道翻倍
  [4]  AttnBlock(256)
  [5]  ResBlock(in=256, out=256, t)
  [6]  Downsample(256, 16→8)
  [7]  ResBlock(in=256, out=256, t)
  [8]  ResBlock(in=256, out=256, t)
  [9]  Downsample(256, 8→4)
  [10] ResBlock(in=256, out=256, t)
  [11] ResBlock(in=256, out=256, t)

middle_modules (索引 0-2):
  [0] ResBlock(in=256, out=256, t)
  [1] AttnBlock(256)
  [2] ResBlock(in=256, out=256, t)

upsampling_modules (索引 0-11):
  [0]  ResBlock(in=512, out=256, t)    ← hs pop (256) + h (256) = 512
  [1]  ResBlock(in=256, out=256, t)
  [2]  ResBlock(in=256, out=256, t)
  [3]  Upsample(256, 4→8)
  [4]  ResBlock(in=512, out=256, t)    ← hs pop (256) + h (256) = 512
  [5]  AttnBlock(256)
  [6]  ResBlock(in=256, out=256, t)
  [7]  ResBlock(in=256, out=256, t)
  [8]  Upsample(256, 8→16)
  [9]  ResBlock(in=384, out=128, t)    ← hs pop (128) + h (256) = 384
  [10] ResBlock(in=128, out=128, t)
  [11] ResBlock(in=128, out=128, t)

output_modules (索引 0-1):
  [0] GroupNorm(128)
  [1] Conv2d(128→6, 3, 1)
```
