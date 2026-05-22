# 04 — UNet Engineering Design Notes

> 工程实现的设计哲学——为什么扩散模型中的 UNet 长这样
>
> 对应代码: `lib/networks/networks.py`, `lib/models/models.py`

---

## 摘要

扩散模型的 UNet 并非诞生于某个单一的数学原理，而是多年工程经验的结晶。本文档不重复逐层结构，而是聚焦于**每一个工程设计决策背后的权衡与推演**：为什么选 GroupNorm 而非 BatchNorm？为什么残差连接要除以 $\sqrt{2}$？为什么注意力模块输出要初始化为零？为什么用插值卷积而非转置卷积做上采样？

每一个看似琐碎的选择，都是对**信号稳定性、训练收敛性、实现简洁性**三者进行的精确权衡。

---

## 第一章：残差连接与方差守恒

### 1.1 残差连接的标准形式

残差网络的核心公式是 $y = x + F(x)$。当 $F$ 是恒等映射时，网络退化问题消失。但在扩散模型中，残差连接面临一个更微妙的问题：**信号方差随深度增长**。

考虑一个简单情况：假设 $x$ 和 $F(x)$ 是均值为零、方差为 $\sigma^2$ 的不相关随机变量，则：

$$Var(x + F(x)) = Var(x) + Var(F(x)) = 2\sigma^2$$

每经过一个残差块，方差膨胀为原来的两倍。经过 $L$ 个残差块后，信号方差为 $2^L \sigma^2$，指数级增长。这对后续层的数值稳定性、梯度传播和初始化调参都构成巨大挑战。

### 1.2 方差重缩放：除以 $\sqrt{2}$

代码中的解决方案 (`ResBlock.forward`, `networks.py:155-156`):

```python
if self.skip_rescale:
    return (x + h) / np.sqrt(2.)
else:
    return x + h
```

设 $x$ 和 $h$ 是方差为 $\sigma^2$ 的独立随机变量，则:

$$\frac{x + h}{\sqrt{2}} \sim \mathcal{N}(0, \sigma^2)$$

方差被严格保持。无论经过多少个 ResBlock，信号幅度的统计期望不变。

这个设计是 score-SDE UNet 移植自 DDPM 论文的一个关键改进。早期残差网络（如 ResNet）不使用这种重缩放，而是依赖 BatchNorm 来控制信号幅度，但 BatchNorm 本身引入的均值和方差估计又依赖 batch size，并不纯粹。**本质上：除以 $\sqrt{2}$ 是以零额外计算代价实现的确定性方差正则化。**

### 1.3 同类设计扩散

同样的重缩放出现在注意力模块中 (`AttnBlock.forward`, `networks.py:92-93`):

```python
if self.skip_rescale:
    return (x + h) / np.sqrt(2.)
```

这说明设计者将此模式视为通用原则：**所有残差路径的分支合并处都应保持方差守恒**。

### 1.4 Conv2D 输出的方差

注意力模块的输出投影 `NIN_3` 使用 `init_scale=0`（`networks.py:74`），即输出投影的权重初始化为零。这使得注意力模块在训练开始时是恒等映射（$h = 0$），完全靠残差 `x` 传递信号。随着训练进行，注意力逐渐参与。这种"从恒等起步"的策略在 Transformer 的 Pre-LN 架构中有类似动机：**初始阶段优先保证信号通路的通畅，再将复杂功能逐步引入。**

---

## 第二章：GroupNorm：Batch 无关的归一化

### 2.1 BatchNorm 的困境

BatchNorm 在标准 CNN 中是默认选择，但在扩散模型场景中失效。原因是**全局 batch size 不稳定**:

- 训练时使用 batch size 128（`data.batch_size = 128`）
- 但评估和采样时 batch 可能很小（如 `scripts/sample.py` 中 sampler 内部逐条处理）
- 分布式训练时每个 GPU 上的 batch 更小

BatchNorm 对 batch 中的样本做统计量估计，当 batch size 小时估计方差不稳定，推理时用累积的全局统计量与训练时的分布也可能不匹配。扩散模型对生成质量的敏感性放大了这种不匹配。

### 2.2 GroupNorm 的策略

代码中的 GroupNorm (`networks.py:109-111`):

```python
nn.GroupNorm(
    num_groups=min(in_ch // 4, 32),
    num_channels=in_ch,
    eps=1e-6
)
```

GroupNorm 将通道划分为 `G` 组，对每组内所有空间位置做归一化。当 $C = 128$ 时，`groups = min(32, 32) = 32`，每组 4 个通道；当 $C = 256$ 时，`groups = min(64, 32) = 32`，每组 8 个通道。

与 BatchNorm 的关键差异:

| 特性 | BatchNorm | GroupNorm |
|------|-----------|-----------|
| 归一化轴 | N, H, W | C/G, H, W |
| 依赖 batch size | ✅ 是 | ❌ 否 |
| 训练/推理差异 | ✅ 有 (running stats) | ❌ 无 |
| 参数量 | $O(4C)$ | $O(4C)$ |
| 对 batch 的鲁棒性 | 低 | 高 |

GroupNorm 的决策独立于 batch size，极大简化了训练到评估的转换。代价是失去了 BatchNorm 的隐式正则化（batch-level 噪声），但扩散模型的损失函数本身已经提供了充分的训练信号。

### 2.3 GroupNorm 的位置

在 ResBlock 中 GroupNorm 放置在卷积之前（即 Pre-Norm 风格，`networks.py:139`）:

```python
h = self.groupnorm0(x)
h = self.act(h)
h = self.conv0(h)
```

而非传统 BN 放在卷积之后的 Post-Norm 风格。Pre-Norm 已被 Transformer 和 ResNet 的后续变体证明比 Post-Norm 更稳定：**归一化确保激活值在进入非线性函数前保持良好范围，使梯度在反传过程中不受归一化层本身的缩放影响。**

---

## 第三章：SiLU：平滑激活的隐性收益

### 3.1 选择 SiLU

使用 `nn.functional.silu`（Swish, `networks.py:107`），定义为:

$$SiLU(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

对比 ReLU:

- **ReLU** 在 $x \leq 0$ 时梯度为 0，导致神经元死亡
- **SiLU** 在 $x \to -\infty$ 时梯度趋近 $0$ 但非精确 $0$，在 $x=0$ 处梯度约 0.5，在 $x \to +\infty$ 时梯度趋近 $1$

### 3.2 梯度连续性的价值

扩散模型的 UNet 深度通常在 50-100 层（所有残差块内的卷积层累加）。当使用 ReLU 时，大量梯度为零的神经元叠加可能导致信号快速衰减。SiLU 的梯度处处非零且连续，使得深层网络中每一层都能接收到有意义的梯度更新。

此外，SiLU 的"非单调"特性（在 $x \approx -1.5$ 处 $y < 0$）提供了轻微的去相关效果：**不同层级的神经元可以编码正负不同的偏差，而 ReLU 强制所有非活跃输出为精确零。**

### 3.3 没有"最佳"激活函数

这是一个经验选择而非理论驱动的设计。如扩散模型的 DDPM 和 ADM 实验中，SiLU 和 ReLU 的差异在最终 FID 上约为 0.1-0.2，不足以定性改变结果，但 SiLU 的一致收敛速度略优。代码库中 SiLU 的选择更反映了"跟随已被验证的实验设定"的工程务实风格。

---

## 第四章：时间编码与注入：标量的维度升维

### 4.1 从标量到向量的必要性

时间 $t$ 是一个 $[0, 1]$ 区间的标量。直接将标量输入神经网络的问题是**标量的表达能力有限**——单个数字只能编码一个全局的"进度"信息，但不同层可能需要不同的时间感知粒度：浅层需要精细的时间分辨率，深层需要粗略的阶段信息。

### 4.2 Transformer 正弦嵌入

代码 (`network_utils.py:7-21`):

```python
def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb
```

生成一个 $d$ 维向量，其中第 $i$ 个分量为：

$$
\begin{cases}
\sin(t \cdot e^{-i \cdot \frac{\log(10000)}{d/2-1}}), & i < d/2 \\
\cos(t \cdot e^{-(i-d/2) \cdot \frac{\log(10000)}{d/2-1}}), & i \geq d/2
\end{cases}
$$

这本质上是将标量 $t$ 编码为不同频率的正弦/余弦信号。高频分量（维度索引大）区分微小的时间变化，低频分量（维度索引小）编码全局范围。这种编码最初来自 Transformer 中的位置编码，被移植到扩散模型中的原因相同：**模型通过不同频率的信号可以提取不同粒度的位置信息。**

### 4.3 MLP 升维

正弦嵌入后经过两层 MLP（`UNet._time_embedding`, `networks.py:314-324`）:

```python
temb = transformer_timestep_embedding(timesteps * 1000, self.time_embed_dim)
temb = self.temb_modules[0](temb)  # Linear(128, 512)
temb = self.temb_modules[1](SiLU(temb))  # Linear(512, 512)
```

嵌入维度从 128 升到 512。两层 MLP 的作用是**非线性变换**，将正弦编码映射到各层 ResBlock 实际需要的激活模式。每层 ResBlock 的 `dense0` 再从中提取偏置信号，形成多级解耦：正弦编码负责频率结构，MLP 负责任务适配，dense0 负责逐层特化。

### 4.4 FiLM 注入：偏置 vs 完整仿射

每个 ResBlock 中的注入 (`networks.py:143-144`):

```python
if temb is not None:
    h += self.dense0(self.act(temb))[:, :, None, None]
```

这是简化版的 FiLM（Feature-wise Linear Modulation）。标准的 FiLM 是 $y = \gamma \cdot x + \beta$，但这里只用了偏置 $\beta$。对比 TransformerEncoderLayer 中的完整 FiLM（`networks.py:477-478`）:

```python
x = film_params[:, None, 0:K] * x + film_params[:, None, K:]
```

为什么 UNet 用简化版？可能原因：UNet 的通道数较大（128~256），如果同时缩放，$\gamma$ 可能将某些通道的信号强度调到过小或过大，破坏方差守恒。仅用偏置限制了时间条件的影响范围，降低了和残差连接的交互复杂度。

### 4.5 时间箱（Temporal Bucketing）

实现在 `UNet.forward` 中使用 `time_scale_factor = 1000`（`networks.py:317`）:

```python
timesteps * self.time_scale_factor  # [0, 1] → [0, 1000]
```

将 $[0, 1]$ 映射到 $[0, 1000]$，这与 DDPM 训练时的离散时间步数一致。即使模型用连续时间训练，时间表示仍保留了对原始离散设计的兼容性——这是工程演进的痕迹：**连续时间框架继承了离散时间的实现习惯。**

---

## 第五章：注意力模块的防御性设计

### 5.1 注意力前的 GroupNorm

注意力模块中 Q/K/V 投影前的 GroupNorm（`networks.py:69-70`）:

```python
self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels//4, 32), num_channels=channels)
h = self.GroupNorm_0(x)
q = self.NIN_0(h)
k = self.NIN_1(h)
v = self.NIN_2(h)
```

注意力前的归一化是**防御性设计**：注意力权重对 Q/K 的幅度极端敏感。即使前一层输出方差完美保持，跨层的微小偏移也会被注意力机制放大。在 Q/K 之前插入 GroupNorm 确保相似度计算不受输入幅度的干扰。

### 5.2 1×1 卷积作为线性投影

Q/K/V 投影使用 `NiN`（Network-in-Network），等价于 `Conv2d(C, C, 1)`。使用 1×1 卷积而非 `nn.Linear` 的原因是**保持空间结构的可解释性**：Q(b,c,h,w) 只和同一位置的其他通道做交互，不跨空间位置（跨空间交互留给注意力计算本身）。这符合卷积神经网络的空间局部性原则。

### 5.3 Attention Logit 缩放

```python
w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
```

缩放因子 $\sqrt{C}^{-1}$ 是标准 Transformer 设计，防止高维空间中点积的值过大导致 softmax 输出坍缩为 one-hot。代码中 $C = 256$ 时缩放因子为 $1/16$。

### 5.4 输出投影的零初始化

`NIN_3` 的 `init_scale=0` (`networks.py:74`) 意味着其权重初始化为零，因此注意力模块初始时输出为零，整个 AttnBlock 退化为恒等映射 $x + 0 = x$。

这是极其谨慎的设计——**注意力是网络中计算最昂贵的组件，应该在训练初期确保主干网络正常收敛后再逐步引入注意力的贡献。** 如果注意力模块一开始就输出非零信号，它会干扰刚初始化的卷积层学习基本的特征提取。

---

## 第六章：采样层：下采样与上采样的工程选择

### 6.1 下采样：Strided Conv 而非 MaxPool

```python
class Downsample(nn.Module):
    def __init__(self, channels):
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = nn.functional.pad(x, (0, 1, 0, 1))
        x = self.conv(x)
        return x
```

使用 `padding=0` + 手动 pad (0,1,0,1) 是为了精确控制边界条件：**pad 方式等价于右侧和底部各补一个像素，使 stride=2 卷积的输出尺寸精确为 $\lfloor (H+1)/2 \rfloor$。** 当 H=32 时输出 16，H=16 时输出 8，以此类推。

与传统 MaxPool 相比，**Strided Conv 是可学习的下采样**——网络可以自主选择保留哪些信息。对于像素级生成任务，这种灵活性比 max 操作的硬选择更适合保留图像细节。

### 6.2 上采样：插值 + Conv 而非 Transposed Conv

```python
class Upsample(nn.Module):
    def __init__(self, channels):
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = F.interpolate(x, (H*2, W*2), mode='nearest')
        h = self.conv(h)
        return h
```

先 `nearest` 插值放大，再用普通 Conv 平滑。对比转置卷积:

- **转置卷积**存在"棋盘效应"——当 kernel size 不能被 stride 整除时，某些像素获得不均匀的贡献
- **插值 + Conv** 分两步：插值是确定性的（无参数），Conv 是可学习的平滑滤波器。两步分离让网络专注于学习特征精炼而非上采样的插值模式

这是一个典型的"**分离关注点**"工程模式——将上采样的"几何放大"和"特征精炼"解耦为两个独立阶段。

---

## 第七章：主干连接与跳跃连接

### 7.1 下采样中的跳跃连接收集

`h_cs` 列表记录了每经过一个模块后的通道数（`networks.py:227`）:

```python
h_cs = [self.ch]  # 初始值: input_conv 输出的通道数

# 每添加一个 ResBlock 或 Downsample 后:
h_cs.append(in_ch)
```

上采样阶段通过 `h_cs.pop()` 从列表末端取出对应的跳跃连接通道数（`networks.py:277`）:

```python
ResBlock(in_ch + h_cs.pop(), out_ch, ...)  # 拼接后通道数 = h 的 in_ch + skip 的 ch
```

这种设计使得 UNet 可以动态确定每层拼接后的输入通道数，而不需要硬编码。

### 7.2 上采样的额外 ResBlock

上采样每层使用 `num_res_blocks + 1 = 3` 个 ResBlock，而下采样只用 `num_res_blocks = 2` 个。这是 DDPM 中继承的对称性破缺设计：**上采样路径承担了更多参数，因为生成阶段的细节重建比下采样阶段的抽象压缩更难。**

### 7.3 跳跃连接的通道拼接

上采样的 ResBlock 输入是 `h`（上采样的输出）和跳跃连接 `skip` 的拼接（`networks.py:368`）:

```python
h = self.upsampling_modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
```

通道翻倍后，ResBlock 再从高通道降低到 `out_ch`。这与 U-Net 原始论文一致，差异在于本 UNet 在拼接后立即用 ResBlock 处理（而非两个单独的卷积），使得信息融合发生在残差结构的非线性变换中。

---

## 第八章：输出头——离散化的平滑桥梁

### 8.1 截断逻辑分布

UNet 输出 `2C = 6` 个通道：前 3 个为均值 `μ`，后 3 个为 `log_scale`（`networks.py:201`）:

```python
self.output_channels = 2 * input_channels  # = 6
```

这两组参数定义了一个连续逻辑分布，再离散化为 256 个 bins。为什么这样设计而不是直接输出 256 个 logits？

**连续到离散的平滑桥梁**：直接输出 256 个 logits 意味着模型要学习 256 个类别的独立概率，忽略了像素值的序结构（0 和 1 相邻，0 和 255 相远）。截断逻辑分布通过连续的 μ 和 σ 编码了序结构——μ 指示"大概的像素值"，σ 指示"不确定性"——模型仅需预测 2 个参数而非 256 个。

### 8.2 残差跳跃连接

输出层之前的特殊残差（`networks.py:398`）:

```python
h[:, 0:C, :, :] = torch.tanh(centered_x_in + h[:, 0:C, :, :])
```

μ 的最终值通过 `tanh` 约束在 $[-1, 1]$ 范围内，与输入数据的归一化范围一致。`centered_x_in` 是输入图像归一化后的值，这种**输入到输出的跨层残差连接**在 pixelCNN 中也有类似设计，本质上是告诉网络："输出 μ 应该在输入像素值附近做微调，而不是从零开始预测"。

---

## 第九章：初始化哲学——从保守到学习

### 9.1 Variance Scaling

使用 `variance_scaling`（`networks.py:10-41`）源自 JAX 的实现，采用了 `fan_avg` 模式和 `uniform` 分布：

```python
def variance_scaling(scale, mode, distribution):
    # mode='fan_avg': denominator = (fan_in + fan_out) / 2
    # distribution='uniform': uniform [-√(3v), √(3v)] where v = scale/denom
```

核心思想：**权重的方差反比于输入/输出维度**。这在深层网络中维持了前向和反向传播的信号方差。

### 9.2 零偏置的偏置

时间嵌入的 MLP 偏置和 ResBlock 中 `dense0` 的偏置均显式初始化为零（`networks.py:118, 215, 217`）:

```python
nn.init.zeros_(self.dense0.bias)
nn.init.zeros_(self.temb_modules[-1].bias)
```

零偏置初始化确保训练初期时间信号的偏置不在网络中引入系统性偏移。这是"**从零开始，逐步学习**"设计哲学的又一实例。

### 9.3 输出模块的零初始化

`default_init(scale=0)` 在 AttnBlock 的输出投影中使用（`networks.py:74`）:

```python
self.NIN_3 = NiN(channels, channels, init_scale=0.)
```

当 `scale = 0` 时，`variance_scaling` 返回的初始化方差为 0，即权重精确为零。这意味着注意力模块最开始不贡献任何信号，梯度也只流经主干路径。随着训练的进行，零初始化权重通过梯度更新获得非零值，注意力模块逐渐启用。

这种"**锁定再解锁**"的初始化策略在大型模型中是常见的防御性设计，防止复杂子模块在训练初期的不成熟信号干扰主干网络。

---

## 第十章：各组件关系总图

所有工程化设计最终是为了解决同一个核心矛盾：**需要足够深的网络来表达复杂的去噪映射，但深网络必然面临梯度衰减、信号爆炸、收敛不稳定等问题。**

```
                 ┌───────────────┐
                 │   输入离散像素   │
                 └───────┬───────┘
                         │
                    ┌────┴────┐
                    │ _center │ 数据归一化, 统一动态范围
                    │ _data   │
                    └────┬────┘
                         │
               ┌─────────┴─────────┐
               │    _time_embed     │ 标量 t → (B,512)
               │  (正弦 + 2×MLP)    │ 高频/低频混合表示
               └─────────┬─────────┘
                         │
               ┌─────────┴─────────┐
               │  GroupNorm + SiLU  │ Pre-Norm 稳定激活
               │  + /√2 残差重缩放 │ 方差守恒
               │  + FiLM 偏置注入   │ 时间条件逐层适配
               │                   │
               │ ┌───────────────┐ │
               │ │ Downsample    │ │ Strided Conv 可学习下采样
               │ │ (pad+stride2) │ │
               │ └───────┬───────┘ │
               │         │         │
               │ ┌───────┴───────┐ │
               │ │ AttnBlock     │ │ GroupNorm+零初始化
               │ │ (中层,16×16)  │ │ 从恒等起步
               │ └───────────────┘ │
               │         │         │
               │ ┌───────┴───────┐ │
               │ │ Middle (Attn) │ │ 最低分辨率全局交互
               │ └───────┬───────┘ │
               │         │         │
               │ ┌───────┴───────┐ │
               │ │ Upsample      │ │ Interpolate+Conv
               │ │ (nearest+conv)│ │ 避免棋盘效应
               │ └───────┬───────┘ │
               │         │         │
               │ ┌───────┴───────┐ │
               │ │ skip concat   │ │ 跳跃连接:保留细节
               │ │ + ResBlock×3  │ │ 比下采样多一个 block
               │ └───────────────┘ │
               └─────────┬─────────┘
                         │
               ┌─────────┴─────────┐
               │  Conv2d(128→6)     │
               │  + tanh μ 残差     │ 输出范围约束
               └─────────┬─────────┘
                         │
               ┌─────────┴─────────┐
               │ 截断逻辑分布离散化   │ μ,log_scale → 256 logits
               │ (logistic CDF 差)  │ 序结构编码
               └─────────┬─────────┘
                         │
                 ┌───────┴───────┐
                 │  (B,3072,256)  │
                 │  每个像素 256 logits
                 └───────────────┘
```

---

## 附录：设计原则总结

| 设计模式 | 解决的问题 | 应用位置 | 代码行 |
|----------|-----------|---------|--------|
| **除以 √2** | 残差方差增长 | ResBlock, AttnBlock 跳跃连接 | 92-93, 155-156 |
| **GroupNorm Pre-Norm** | 归一化稳定性 + batch 无关 | 所有 ResBlock, AttnBlock, 输出头 | 69, 108-111, 301 |
| **SiLU** | 连续梯度流 | 所有激活函数 | 107, 210 |
| **正弦 + MLP 时间嵌入** | 标量→向量多粒度编码 | UNet 顶层 | 314-324 |
| **零初始化的注意力输出** | 避免早期干扰 | AttnBlock 的输出投影 | 74 |
| **zero bias 初始化** | 不引入系统偏移 | 所有 Linear 偏置 | 118, 215, 217 |
| **Strided Conv 下采样** | 可学习信息筛选 | Downsample | 160-173 |
| **Interpolate+Conv 上采样** | 避免棋盘效应 | Upsample | 175-187 |
| **跳跃连接通道拼接** | 保留精细细节 | 上采样 ResBlock 输入 | 368 |
| **上与下 ResBlock 数不对称** | 生成需更多参数 | 上采样 3 个 / 下采样 2 个 | 234, 273-274 |
| **tanh μ 残差** | 输出范围约束 + 跨层连接 | _logistic_output_res | 392-399 |
| **截断逻辑分布** | 序结构编码 + 参数效率 | ImageX0PredBase.forward | models.py:60-90 |
| **variance_scaling 初始化** | 维持信号方差 | 所有权重 | 10-41 |
| **数据预加载到 GPU** | 消除 CPU-GPU 瓶颈 | DiscreteCIFAR10 | datasets.py:23 |
