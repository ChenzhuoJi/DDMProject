
# D3PM (Structured Denoising Diffusion Models in Discrete State-Spaces) 源码教程

## 目录

1. [项目概述](#1-项目概述)
2. [环境配置与运行](#2-环境配置与运行)
3. [images 模块：图像离散扩散模型](#3-images-模块图像离散扩散模型)
   - 3.1 [入口与配置 (entry_point.py / config.py)](#31-入口与配置)
   - 3.2 [数据集 (datasets.py)](#32-数据集)
   - 3.3 [扩散过程 (diffusion_categorical.py)](#33-扩散过程)
   - 3.4 [模型架构 (model.py)](#34-模型架构)
   - 3.5 [训练框架 (gm.py / main.py)](#35-训练框架)
4. [text 模块：文本离散扩散模型](#4-text-模块文本离散扩散模型)
   - 4.1 [配置文件系统 (configs.py)](#41-配置文件系统)
   - 4.2 [数据集加载 (datasets.py)](#42-数据集加载)
   - 4.3 [扩散过程 (diffusion.py)](#43-扩散过程)
   - 4.4 [Transformer 模型 (models.py)](#44-transformer-模型)
   - 4.5 [损失函数 (losses.py)](#45-损失函数)
   - 4.6 [训练器 (trainers.py)](#46-训练器)
   - 4.7 [主执行入口 (main.py)](#47-主执行入口)
5. [insertdelete 模块：插入与删除扩展](#5-insertdelete-模块插入与删除扩展)
   - 5.1 [转移算子 (transition_operator.py)](#51-转移算子)
   - 5.2 [前向过程 (forward_process.py)](#52-前向过程)
   - 5.3 [调度器 (schedules.py)](#53-调度器)
   - 5.4 [训练设置 (training_setup.py)](#54-训练设置)
   - 5.5 [概率分布与动态规划 (distributions.py / dynamic_programs.py)](#55-概率分布与动态规划)
6. [关键概念总结](#6-关键概念总结)
7. [三模块对比](#7-三模块对比)

---

## 1. 项目概述

D3PM (**S**tructured **D**enoising **D**iffusion Models in **D**iscrete **S**tate-**S**paces) 是 Google Research 在 NeurIPS 2021 上发表的论文的官方实现。它将扩散模型从连续状态空间（如图像像素的连续值）推广到**离散状态空间**（如分类像素值、文本 token）。

整个代码库使用 **JAX** + **Flax (Linen)** 作为深度学习框架，有三个功能模块：

| 模块 | 目录 | 功能 |
|------|------|------|
| **images** | `01_D3PM/src/images/` | CIFAR-10 图像生成，像素值作为离散类别 |
| **text** | `01_D3PM/src/text/` | LM1B / text8 文本生成，token 作为离散类别 |
| **insertdelete** | `01_D3PM/src/insertdelete/` | 插入和删除操作扩展，实现更灵活的序列生成 |

---

## 2. 环境配置与运行

### 2.1 依赖安装

核心依赖在 [requirements.txt](file:///d:/Desktop/随机过程大作业/01_D3PM/src/requirements.txt) 中指定：

| 包 | 版本 | 用途 |
|----|------|------|
| `jax` / `jaxlib` | 0.3.14 | 自动微分与 XLA 编译 |
| `flax` | 0.5.1 | 神经网络框架（Linen API） |
| `tensorflow` | 2.8.0 | 数据加载 |
| `gin-config` | 0.5.0 | 超参数配置 |
| `ml_collections` | 0.1.1 | 配置字典 |
| `seqio` | 0.0.16 | 分词器（SentencePiece） |
| `flaxformer` | git commit | Transformer 架构 |

运行方式见 [run.sh](file:///d:/Desktop/随机过程大作业/01_D3PM/src/run.sh)：

```bash
virtualenv -p python3 .venv_d3pm
source .venv_d3pm/bin/activate
pip install --upgrade pip
pip install -r d3pm/requirements.txt

python -m d3pm.images.main_test
python -m d3pm.text.main_test
```

### 2.2 代码架构概览

```
src/
├── requirements.txt          # 依赖清单
├── run.sh                    # 安装与测试脚本
├── README.md                 # 根 README
│
├── images/                   # === 图像 D3PM ===
│   ├── config.py             # 超参数配置（ml_collections）
│   ├── datasets.py           # CIFAR-10 数据集加载
│   ├── diffusion_categorical.py  # 离散扩散过程（核心算法）
│   ├── model.py              # UNet 架构（Flax Linen）
│   ├── entry_point.py        # 入口调度
│   ├── gm.py                 # 通用训练器（TrainableModel）
│   ├── main.py               # CIFAR-10 主实验文件
│   ├── main_test.py          # 集成测试
│   ├── main_test_config.py   # 测试用轻量配置
│   └── utils.py              # 工具函数
│
├── text/                     # === 文本 D3PM ===
│   ├── configs.py            # Gin 配置与实验预设
│   ├── datasets.py           # LM1B / text8 数据加载
│   ├── diffusion.py          # 离散扩散（基类 + 多种实现）
│   ├── models.py             # Transformer 模型（Flaxformer）
│   ├── losses.py             # 损失函数（CE, KL）
│   ├── trainers.py           # 训练器（Trainer 类）
│   ├── tasks.py              # 任务注册
│   ├── main.py               # 主训练入口
│   ├── model_utils.py        # 模型工具（幂运算、嵌入等）
│   ├── types.py              # 类型定义
│   ├── metrics.py            # 评估指标
│   ├── preprocessors.py      # 数据预处理器
│   └── utils.py              # 工具函数
│
└── insertdelete/             # === 插入删除扩展 ===
    ├── transition_operator.py    # 转移算子（矩阵/掩码/均匀）
    ├── forward_process.py        # 带 sentinel 的前向过程
    ├── schedules.py              # 插入删除调度器
    ├── training_setup.py         # 训练损失设置
    ├── distributions.py          # 概率分布（二项/几何/负超几何）
    ├── dynamic_programs.py       # 动态规划调度
    ├── math_util.py              # 数学工具（log-space 安全运算）
    └── util.py                   # 杂项工具
```

---

## 3. images 模块：图像离散扩散模型

### 3.1 入口与配置

[entry_point.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/images/entry_point.py) 是整个实验的调度器。它处理命令行参数、读取配置、设置随机种子，然后调用指定的可执行函数：

```
main(executable_dict, argv)
  ├── 读取 JSON 配置或命令行配置
  ├── 设置 JAX / TensorFlow / 随机种子
  └── 调用 executable_dict[FLAGS.executable_name](config, ...)
```

[config.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/images/config.py) 定义了 CIFAR-10 实验的默认配置：

```python
dataset = config_dict(
    name='CIFAR10',
    args=config_dict(class_conditional=False, randflip=True))

model = config_dict(
    name='unet0',
    args=config_dict(ch=128, out_ch=3, ch_mult=[1,2,2,2], ...),
    diffusion_betas=config_dict(type='linear', start=1e-4, stop=0.02, num_timesteps=1000),
    model_prediction='x_start',
    transition_mat_type='gaussian',
    loss_type='hybrid',
    hybrid_coeff=0.001)

train = config_dict(
    batch_size=128, learning_rate=2e-4, num_train_steps=1500000, ...)
```

关键超参数说明：
- **`transition_mat_type`**：转移矩阵类型，可选 `'gaussian'` / `'uniform'` / `'absorbing'`
- **`model_prediction`**：模型预测目标，`'x_start'` 预测原始数据，`'xprev'` 预测上一步
- **`loss_type`**：损失类型，`'kl'` / `'cross_entropy_x_start'` / `'hybrid'`
- **`diffusion_betas`**：噪声调度，可选 `'linear'` / `'cosine'` / `'jsd'`

### 3.2 数据集

[datasets.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/images/datasets.py) 定义了数据集基类和 CIFAR-10 实现。

`Dataset` 基类接口：

```python
class Dataset:
    @property
    def data_shape(self):     # 返回数据形状，如 (32, 32, 3)
    @property
    def num_train(self):      # 训练集大小
    @property
    def num_eval(self):       # 评估集大小
    @property
    def num_classes(self):    # 类别数（条件生成用）
    def get_tf_dataset(self, *, batch_shape, split, ...):  # 返回 tf.data.Dataset
```

`CIFAR10` 类加载 TFDS 的 cifar10 数据集，支持：
- **随机水平翻转**（数据增强）
- **跨设备分片**（`shard_dataset`）
- **批处理**（多维 batch shape：`[device_count, substeps, batch_per_device]`）

`MockCIFAR10` 是测试用的小型模拟数据集（8x8 图像，仅 10 个样本）。

### 3.3 扩散过程

[diffusion_categorical.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/images/diffusion_categorical.py) 是 **images 模块的核心**，实现了离散状态空间的扩散过程。

#### 3.3.1 噪声调度 (`get_diffusion_betas`)

三种 β 调度：

```python
def get_diffusion_betas(spec):
    if spec.type == 'linear':       # DDPM 的线性调度
        return onp.linspace(spec.start, spec.stop, spec.num_timesteps)
    elif spec.type == 'cosine':     # Hoogeboom 等的余弦调度
        alpha_bar = onp.cos((steps + 0.008) / 1.008 * onp.pi / 2) ** 2
        betas = onp.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
    elif spec.type == 'jsd':        # Sohl-Dickstein 的 1/T 调度
        return 1. / onp.linspace(num_timesteps, 1., num_timesteps)
```

#### 3.3.2 CategoricalDiffusion 类

这是核心类，管理离散扩散的全部数学运算。

**转移矩阵构建**：`q(x_t | x_{t-1})` 有三种类型：

| 类型 | 行为 | 稳态分布 |
|------|------|---------|
| `'uniform'` | 以概率 β 均匀跳到任意状态 | 均匀分布 |
| `'gaussian'` | 以高斯核权重跳到邻近状态 | 均匀分布 |
| `'absorbing'` | 以概率 β 跳到吸收态（如 [128,128,128]） | 吸收态上的 Delta 分布 |

**核心方法**：

```python
def q_probs(self, x_start, t):
    """计算 q(x_t | x_start) 的概率"""
    return self._at(self.q_mats, t, x_start)

def q_sample(self, x_start, t, noise):
    """从 q(x_t | x_start) 采样（添加噪声）"""
    # 使用 Gumbel-max 技巧进行离散采样
    logits = jnp.log(self.q_probs(x_start, t) + eps)
    gumbel_noise = -jnp.log(-jnp.log(noise))
    return jnp.argmax(logits + gumbel_noise, axis=-1)

def q_posterior_logits(self, x_start, x_t, t):
    """计算 q(x_{t-1} | x_t, x_start) 的对数概率"""
    fact1 = self._at(self.transpose_q_onestep_mats, t, x_t)
    fact2 = self._at(self.q_mats, t-1, x_start)
    return jnp.log(fact1 + eps) + jnp.log(fact2 + eps)
```

**训练损失**：

```python
def training_losses(self, model_fn, x_start, rng):
    # 1. 采样时间步 t
    # 2. 采样噪声数据 x_t
    # 3. 计算模型预测 p_theta(x_{t-1} | x_t)
    # 4. 计算 KL 散度或交叉熵损失
    # 5. 可选择计算 bits-per-dimension (BPD)
```

**采样过程**：

```python
def p_sample_loop(self, model_fn, *, shape, rng):
    """祖先采样：从 x_T 逐步去噪到 x_0"""
    # 根据转移矩阵类型初始化 x_T：
    #   'uniform' / 'gaussian' → 均匀随机像素值
    #   'absorbing' → 全部设为吸收态
    for t in reversed(range(num_timesteps)):
        x, _ = self.p_sample(model_fn, x=x, t=t, noise=gumbel_noise)
    return x
```

#### 3.3.3 Gumbel-max 技巧

在离散采样中，代码使用 Gumbel-max 技巧代替直接从类别分布采样。这种方法使得采样过程可微：

```python
gumbel_noise = -jnp.log(-jnp.log(uniform_noise))
sample = jnp.argmax(logits + gumbel_noise, axis=-1)
```

### 3.4 模型架构

[model.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/images/model.py) 实现了 `UNet`，这是标准 DDPM UNet 的 Flax Linen 版本。

#### 3.4.1 时间步嵌入

```python
def get_timestep_embedding(timesteps, embedding_dim, max_time=1000.):
    """构建正弦位置编码（来自 Fairseq/Transformer）"""
    half_dim = embedding_dim // 2
    emb = jnp.exp(jnp.arange(half_dim) * -log(10000) / (half_dim - 1))
    emb = timesteps[:, None] * emb[None, :]
    return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
```

这与 Transformer 中的位置编码类似，但应用于时间步。

#### 3.4.2 UNet 结构

```
输入 x (int32, shape=[B, H, W, 3])
  │
  ├── one_hot → float (256 类)
  ├── normalize_data → [-1, 1]
  │
  ├── Conv 3x3 (ch)
  │
  ├── 下采样路径 (Downsampling)
  │   └── 每层: ResnetBlock × num_res_blocks + (可选) AttnBlock
  │   └── 下采样 (Conv stride 2)
  │
  ├── 中间层 (Middle)
  │   └── ResnetBlock → AttnBlock → ResnetBlock
  │
  ├── 上采样路径 (Upsampling)
  │   └── 每层: ResnetBlock × (num_res_blocks + 1) + (可选) AttnBlock
  │   └── 上采样 (最近邻插值 + Conv)
  │
  └── 输出层
      ├── model_output='logits' → [B, H, W, 3, 256] (逐像素 logits)
      └── model_output='logistic_pars' → loc, log_scale (逻辑斯谛分布参数)
```

#### 3.4.3 ResnetBlock

```python
class ResnetBlock(nn.Module):
    """带有时间步和类别条件注入的残差块"""
    def __call__(self, x, *, temb, y, deterministic):
        # 1. GroupNorm → Swish → Conv 3x3
        # 2. + 时间步嵌入投影 (Dense)
        # 3. + 类别嵌入投影 (Dense) [条件生成时]
        # 4. GroupNorm → Swish → Dropout → Conv 3x3 (零初始化)
        # 5. 残差连接 (shortcut 维度匹配时)
```

#### 3.4.4 AttnBlock

```python
class AttnBlock(nn.Module):
    """自注意力残差块"""
    def __call__(self, x):
        # LayerNorm → 多头注意力 (QKV) → 输出投影 (零初始化)
        # 残差连接
```

### 3.5 训练框架

[gm.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/images/gm.py) 定义了 `TrainableModel` 基类，[main.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/images/main.py) 定义了 `Cifar10DiffusionModel`。

#### 3.5.1 TrainState

```python
@flax.struct.dataclass
class TrainState:
    step: int                      # 当前步数
    optimizer: flax.optim.Optimizer  # 优化器（含参数）
    ema_params: Any                # 指数移动平均参数（用于评估）
```

#### 3.5.2 训练循环

```python
def run_train(self, experiment_dir, work_unit_dir, rng):
    # 1. 构建数据流水线（train + eval）
    # 2. 初始化模型参数
    # 3. 恢复检查点（如有）
    # 4. pmap 跨设备并行训练
    # 5. 循环：
    #    a. train_step: 前向 → 损失 → 梯度裁剪 → 优化器更新 → EMA 更新
    #    b. 周期性评估：计算损失 + 生成样本
    #    c. 保存检查点
```

#### 3.5.3 损失计算 (CIFAR-10)

```python
def loss_fn(self, rng, train, batch, params):
    def model_fn(x, t):
        return self.model.apply({'params': params}, x=x, t=t, y=label, train=train)

    dif = make_diffusion(self.config.model, num_bits=8)
    loss = dif.training_losses(model_fn, x_start=img, rng=next(rng)).mean()
    # 训练时返回 loss
    # 评估时额外计算 prior_bpd 和 total_bpd
```

#### 3.5.4 日志与检查点

- **EMA 参数**：评估和采样时使用 EMA 参数（`ema_decay=0.9999`）
- **梯度裁剪**：按全局范数裁剪（`grad_clip=1.0`）
- **更新跳过**：若梯度包含 NaN，则跳过该步更新
- **检查点**：周期性保存，保留最近 3 个

---

## 4. text 模块：文本离散扩散模型

text 模块比 images 模块更复杂，使用了 Gin 配置系统、Flaxformer Transformer 架构，并支持多种扩散类型。

### 4.1 配置文件系统

[configs.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/text/configs.py) 使用 Gin 配置库管理超参数。它定义了多个预设实验配置：

**模型大小预设**：

```python
def _model_gpt(size=0):
    """GPT 风格的模型大小"""
    num_layers = [1, 3, 6, 12, 24, 36, 48][size]
    dim = [64, 128, 512, 768, 1024, 1280, 1600][size]
    num_heads = dim / 64

gpt_extra_tiny  # size=0: 1层, 64维
gpt_tiny         # size=1: 3层, 128维
gpt_small        # size=2: 6层, 512维
gpt_base         # size=3: 12层, 768维
gpt_large        # size=4: 24层, 1024维
```

**完整实验预设**（组合模型大小 + 数据集 + 扩散类型）：

| 预设 | 模型 | 数据集 | 扩散类型 | 步数 |
|------|------|--------|---------|------|
| `lm1b_tiny` | extra tiny | LM1B | mask | 32 |
| `lm1b_base` | base | LM1B | mask | 1000 |
| `text8_tiny` | extra tiny | text8 | mask | 32 |
| `text8_base` | base | text8 | mask | 1000 |

### 4.2 数据集加载

[datasets.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/text/datasets.py) 实现了两个数据集的加载：

#### LM1B

- 使用 SentencePiece 分词器（词汇表：`lm1b-sentencepiece-8k.model`，~8000 token）
- 支持句子打包（packing）：多个句子拼接到 `max_length`
- 默认 `max_length=128`
- 可添加额外 token（用于 mask 扩散的 `[MASK]` 标记）

#### text8

- 27 字符词汇表（a-z + 空格）
- 数据下载自 http://mattmahoney.net/dc/text8.zip
- 自动切分为 train/valid/test（9000万/500万/剩余字符）
- 训练集支持随机裁剪
- 默认 `max_length=256`

#### 词汇表封装

代码定义了多层词汇表封装：

```
文本 → TFTextVocabulary (27字符基础分词)
     → D3PMVocabulary (可添加额外 token，如 [MASK])
     → PermutationVocab (可对 token ID 施加置换)
```

### 4.3 扩散过程

[diffusion.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/text/diffusion.py) 是 text 模块最庞大的文件，约 3200 行，实现了多种离散扩散算法。

#### 4.3.1 类层次结构

```
DiscreteDiffusionBase (抽象基类)
  │
  └── DiscreteDiffusionMatrixBase (矩阵基类)
        ├── BetaDiagonalDiffusion (β-对角扩散)
        ├── BandDiagonalDiffusion (带状对角扩散)
        ├── MaskDiffusion (掩码扩散)
        └── ...其他实现
```

#### 4.3.2 BetaDiagonalDiffusion

最简单的扩散方式，转移矩阵为：

```
Q_t = (1 - β_t) I + β_t * (1/D) * 1 1^T
```

即以概率 `(1 - β_t)` 保持不动，以概率 `β_t` 均匀跳到任意 token。

**高效推理**：利用多项式恒等式 `(1-β)I + βJ/D` 的幂运算有闭式解，避免昂贵的矩阵乘法。

```python
def custom_product_fn(self, t):
    if self.schedule.is_constant:
        beta = self.schedule(0)
        return (1 - beta)**t * I + (1 - (1 - beta)**t) / D * J
    else:
        # 使用预计算的多项式系数
        p = self.state[t]
        return p[1] * I + p[0] * J / D
```

#### 4.3.3 MaskDiffusion

转移到特殊 `[MASK]` token 的扩散方式（类似 BERT 的掩码语言建模）：

```
Q_t = (1 - β_t) I + β_t * e_mask
```

即每个 token 以概率 `β_t` 被替换为 `[MASK]`。

#### 4.3.4 扩散调度器创建

`create_discrete_diffusion` 函数根据 `kind` 参数选择扩散类型，根据 `schedule_type` 选择 β 调度：

```
schedule_type = 'standard' → β_t = 1 / (T - t)
schedule_type = 'linear'   → β 从 beta_min 线性增加到 beta_max
schedule_type = 'cosine'   → 余弦调度（OpenAI I-DDPM）
```

### 4.4 Transformer 模型

[models.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/text/models.py) 实现了基于 **Flaxformer**（Google 的 Transformer 库）的编码器-解码器架构。

#### 4.4.1 TransformerConfig

```python
@struct.dataclass
class TransformerConfig:
    vocab_size: int               # 词汇表大小
    emb_dim: int = 512            # 嵌入维度
    num_heads: int = 8            # 注意力头数
    num_encoder_layers: int = 6   # 编码器层数
    num_decoder_layers: int = 6   # 解码器层数
    qkv_dim: int = 512            # QKV 维度
    mlp_dim: int = 2048           # FFN 中间维度
    dropout_rate: float = 0.1     # Dropout 率
    ...
```

#### 4.4.2 CategoricalDiffusionModel

这是用于扩散的模型类，包装了编码器-解码器：

```python
class CategoricalDiffusionModel(nn.Module):
    config: TransformerConfig
    num_steps: int = 1000
    use_timestep_embeddings: bool = True
    use_film_layers: bool = True

    def encode(self, encoder_input_tokens, encoder_padding_mask):
        """编码条件输入"""

    def decode(self, encoded, decoder_input_tokens, ...):
        """解码目标序列，可选时间步条件"""
```

#### 4.4.3 时间步条件注入

模型支持两种时间步条件注入方式：

1. **Timestep Embedding**：将时间步嵌入加到 token 嵌入上
   ```python
   embed = Embed(num_steps, dim)(timestep)
   x = x + embed[:, None]  # 广播加到所有位置
   ```

2. **FiLM Block**：特征线性调制
   ```python
   embed = Embed(num_steps, 2*dim)(timestep)
   x = x * gamma[:, None] + beta[:, None]  # 仿射变换
   ```

### 4.5 损失函数

[losses.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/text/losses.py) 实现了标准的损失函数：

| 函数 | 用途 |
|------|------|
| `cross_entropy_with_logits` | 带标签平滑的交叉熵 |
| `cross_entropy_with_probs` | 对概率分布的交叉熵 |
| `kl_divergence_with_logits` | 从 logits 计算的 KL 散度 |
| `kl_divergence_with_probs` | 从概率计算的 KL 散度 |
| `weighted_accuracy` | 加权准确率 |

**自定义 VJP**：交叉熵使用了自定义的前向/反向传播（`@jax.custom_vjp`）以提高数值稳定性：

```python
@jax.custom_vjp
def _cross_entropy_with_logits(logits, targets):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_shifted = jnp.exp(shifted)
    sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
    log_softmax = shifted - jnp.log(sum_exp)
    return -jnp.sum(targets * log_softmax, axis=-1)
```

### 4.6 训练器

[trainers.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/text/trainers.py) 定义了 `Trainer` 类，管理训练循环。

#### 4.6.1 TrainState

```python
@flax.struct.dataclass
class TrainState:
    optimizer: flax.optim.Optimizer   # 优化器
    step: chex.Array                  # 当前步数
    ema_loss: chex.Array              # 损失 EMA（用于异常值检测）
    ema_variance: chex.Array          # 方差 EMA
```

#### 4.6.2 训练步骤

```python
def standard_train_step(state, batch, rng_key, dynamic_state, ...):
    # 1. 前向传播计算损失
    # 2. 反向传播计算梯度
    # 3. pmean 跨设备平均梯度
    # 4. 梯度裁剪
    # 5. 优化器更新
    # 6. EMA 更新（异常值检测）
    return new_state, metrics, rng_key
```

**异常值拒绝**：通过 EMA 维护损失的均值和方差，当损失异常高时跳过该步更新：

```python
normal_pdf = jax.scipy.stats.norm.pdf(loss, loc=ema_loss, scale=jnp.sqrt(ema_variance))
should_replace = (normal_pdf > threshold) | (step < warmup)
```

#### 4.6.3 训练循环

`run_experiment` 函数在 [main.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/text/main.py) 中实现，工作流程：

```python
def run_experiment(model_dir, model_cls, task_name, dataset_name, ...):
    # 1. 加载任务和数据集
    task = tasks.load(name=task_name)
    ds = datasets.load(name=dataset_name, batch_size=...)

    # 2. 初始化训练器
    trainer = trainers.Trainer(dataset_info=train_ds.info, task=task, model_cls=model_cls)

    # 3. 训练循环
    for step, batch in zip(range(start_step, max_train_steps), train_iter):
        metrics = trainer.fit_batch(batch)     # 训练一步
        if step % validate_every == 0:
            eval_summary = evaluate(trainer, valid_ds)  # 评估
        if step % checkpoint_frequency == 0:
            trainer.save_checkpoint(model_dir)          # 保存检查点
```

### 4.7 任务注册

[tasks.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/text/tasks.py) 和 [diffusion.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/text/diffusion.py) 末尾定义了扩散任务，包括：

- **`discrete_diffusion_loss_fn`**：扩散模型的损失函数（计算 ELBO）
- **`discrete_diffusion_predict_fn`**：扩散模型的预测/采样函数
- 这些函数通过 `@gin.configurable` 和 `tasks.register()` 注册

---

## 5. insertdelete 模块：插入与删除扩展

insertdelete 模块是 D3PM 的扩展，允许扩散过程不仅改变 token 的值，还可以**改变序列长度**（插入和删除 token）。

### 5.1 转移算子

[transition_operator.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/images/transition_operator.py) 定义了操作离散 token 的转移算子。

#### 5.1.1 算子接口

```python
class TransitionOperator(abc.ABC):
    def prob_matrix(self, log=False):    # 返回转移概率矩阵
    def apply(self, before, is_distn, log=False):    # 左乘（前向传播）
    def observe(self, after, is_distn, log=False):   # 右乘（后向推断）
    def then(self, other):               # 组合两个算子
    def left_fold_identity(self):        # 左折叠的单位元
```

#### 5.1.2 实现类

| 算子 | 说明 |
|------|------|
| `IdentityOperator` | 恒等映射 |
| `MatrixOperator` | 显式转移矩阵 |
| `LogMatrixOperator` | 对数空间的转移矩阵 |
| `UniformDiffusionOperator` | 均匀扩散：`p(diag) = exp(lp_no_randomize) + exp(lp_off_diag)` |
| `MaskDiffusionOperator` | 掩码扩散：转移到 `mask_token` |
| `RerollOperator` | 以一定概率重新采样（组合子算子 + 重采样分布） |

算子支持 **组合**（`then` 方法），例如 `MaskOperator.then(MaskOperator)` 仍然是 `MaskOperator`，只需将 `lp_no_mask` 相加。

### 5.2 前向过程

[forward_process.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/insertdelete/forward_process.py) 实现了带 **sentinel（哨兵）标记** 的插入/删除前向过程。

#### 5.2.1 核心思想

为了在保持确定性对齐的同时建模插入和删除，代码引入了两种 sentinel 标记：

| 标记 | 含义 |
|------|------|
| `INSERT_SENTINEL = -1` | 表示该位置的 token 将在下一步被插入 |
| `DELETE_SENTINEL = -2` | 表示该位置将在下一步被删除 |

示例：

```
t=1    A  B  C          D  E  F         ← 原始序列
t=2    A  B DEL INS INS D DEL F     INS  ← 标记哪些要删除/插入
t=3    A DEL     G   H  D     F INS  J   ← 实际插入和删除后
t=4    A         G   H  D     F  K   J   ← 继续扩散
```

其中 INS 标记消失变为新 token，DEL 标记消失（该位置被删除）。

#### 5.2.2 OneStepDistn 和 ManyStepDistn

- **`OneStepDistn`**：描述单步转移的参数（`lp_delete`, `lp_insert`, `A`, `D_insert_logits`）
- **`ManyStepDistn`**：描述多步累积转移的参数（多了 `lp_silent_delete`, `lp_silent_insert`, `lp_reroll` 等）

`ManyStepDistn.then(after)` 方法组合两步转移：

```python
def then(self, after):
    """组合 ManyStepDistn 和 OneStepDistn"""
    # 计算新的 sentinel_delete 概率：
    #   = (保持不动 → 删除) OR (删除 → 不插入 → 删除)
    lp_sentinel_delete = logaddexp(
        self.lp_no_delete + after.lp_delete,
        self.lp_sentinel_delete + ... + after.lp_delete)
    # 同样计算 silent_delete, sentinel_insert, silent_insert 等
    return ManyStepDistn(...)
```

#### 5.2.3 动态长度序列

```python
class DynamicLengthSentinelSequence:
    INSERT_SENTINEL = -1
    DELETE_SENTINEL = -2

    tokens: NDArray     # int32[max_len]，标记或 token ID
    length: NDArray     # 当前有效长度
```

提供了 sentinel 检测、padding 填充、sentinel 剥离等方法。

### 5.3 调度器

[schedules.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/insertdelete/schedules.py) 构建插入-删除的噪声调度。

#### SentinelInsertDeleteSchedule

```python
class SentinelInsertDeleteSchedule:
    steps: OneStepDistn           # 每个时间步的分布（堆叠）
    cumulative: ManyStepDistn     # 累积分布
    weights: NDArray              # 采样权重

    def sample_step_number(self, rng):
        """按权重采样一个时间步"""
    def distns_at_step(self, step_number):
        """获取指定步的分布"""
```

#### 调度构建函数

- **`build_schedule`**：堆叠单步分布为完整调度，支持添加最终全删除步
- **`build_uniform_insert_delete_schedule`**：构建均匀调度（均匀插入、均匀删除）
- **`schedule_from_interpolators`**：通过插值函数构建调度（可自定义转移、插入、相对大小、刷新概率）

### 5.4 训练设置

[training_setup.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/insertdelete/training_setup.py) 实现了插入-删除扩散的训练逻辑。

#### 5.4.1 核心损失函数

```python
def sample_based_elbo_term_loss(x0, schedule, denoise_prediction_fn, rng, ...):
    """单步 ELBO 项估计

    1. 采样时间步 t
    2. 从 q(x_{t+1} | x_0) 采样
    3. 计算对齐 alignment(0, t+1)
    4. 可选：重采样多个对齐（降低方差）
    5. 运行模型预测 p_theta(x_t | x_{t+1})
    6. 计算 ELBO = log q(x_{t+1} | x_t) - log p_theta(x_t | x_{t+1})
    """
```

#### 5.4.2 完整 ELBO 估计

```python
def sample_based_elbo(x0, schedule, denoise_prediction_fn, rng, num_samples):
    """完整 ELBO 估计（所有步）

    使用拒绝采样确保序列长度不超过 max_len，
    并修正 log q'(x_{t+1} | x_t) = log q(x_{t+1} | x_t) - log in_bounds_prob
    """
```

#### 5.4.3 预处理器

```python
def preprocess_targets(sequence, how='fixed', pad_to=None):
    """将原始 token 序列转为 DynamicLengthSentinelSequence

    how='fixed': 使用固定长度（pad_to 参数）
    how='padding': 根据 PAD token 自动检测长度
    """
```

### 5.5 概率分布与动态规划

#### 5.5.1 概率分布

[distributions.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/insertdelete/distributions.py) 实现了可用于插入-删除过程的概率分布：

| 分布 | 说明 |
|------|------|
| `binomial_log_pdf` | 二项分布：`n` 次试验，成功概率 `exp(success_log_prob)` |
| `geometric_log_pdf` | 几何分布：直到第一次成功的试验次数 |
| `negative_binomial_log_pdf` | 负二项分布：达到 `r` 次失败前的成功次数 |
| `negative_hypergeometric_log_pdf` | 负超几何分布：不放回抽样的负二项版本 |

分布以 `RandomVariablePDF` 形式返回（包装了 log_probs 数组），支持 `shift` 和 `mixture_of` 操作。

#### 5.5.2 动态规划

[dynamic_programs.py](file:///d:/Desktop/随机过程大作业/01_D3PM/src/insertdelete/dynamic_programs.py) 用于在插入删除过程中进行高效的动态规划计算。

**调度类型**：

- `SerialSchedule`：串行遍历所有元素
- `LookupSchedule`：基于预计算查找表的并行调度
- `packed_block_schedule`：自动构建并行依赖调度

**核心函数**：

```python
def dynamic_program(destination, lookback_fn, kernel_fn, schedule, rewrite_vjp=True):
    """通用动态规划执行器

    - destination: 初始表（空）
    - lookback_fn: 读取依赖位置
    - kernel_fn: 根据依赖计算新值
    - schedule: 访问调度
    - rewrite_vjp: 使用自定义 VJP 节省内存
    """
```

该函数支持自定义 VJP，在前向传播时不保存中间结果，反向传播时重新计算，实现**常数内存**的自动微分。

---

## 6. 关键概念总结

### 6.1 离散扩散的核心数学

D3PM 将连续扩散中的高斯噪声替换为离散空间的**转移矩阵** `Q_t`：

```
连续:  x_t = √(α_t) x_0 + √(1-α_t) ε
离散:  q(x_t | x_0) = one_hot(x_0) · Q_1 · Q_2 · ... · Q_t
```

### 6.2 损失函数

三种损失类型：

| 损失类型 | 公式 | 说明 |
|---------|------|------|
| KL | `D_KL(q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t))` | 精确 ELBO |
| Cross-entropy (x_start) | `CE(x_0, p_θ(x_0|x_t))` | 直接预测原始数据 |
| Hybrid | `KL + λ * CE(x_0)` | 两者混合 |

### 6.3 转移矩阵类型

| 类型 | 公式 | 稳态 | 适用场景 |
|------|------|------|---------|
| Uniform | `(1-β)I + β·J/D` | 均匀分布 | 通用离散扩散 |
| Gaussian | 带权重的带状转移 | 均匀分布 | 有序离散空间（像素值） |
| Absorbing | `(1-β)I + β·e_absorb` | 吸收态 | 掩码扩散 |
| Mask | `(1-β)I + β·e_mask` | 全 [MASK] | 文本扩散（类似 BERT） |
| Band-diagonal | 有限带宽转移 | 近似均匀 | 文本/小词汇表 |

### 6.4 sentinel 机制（insertdelete）

InsertDelete 模块引入 sentinel 标记来对齐变长序列：

```
x_t:     A  B  DEL  INS  C      ← sentinel 标记
          \  \   |    |  /
x_{t+1}:  A  B   X    Y  C      ← 实际序列
```

- **DEL sentinel**：下一步被删除
- **INS sentinel**：下一步变为新 token

---

## 7. 三模块对比

| 方面 | images | text | insertdelete |
|------|--------|------|-------------|
| **数据** | CIFAR-10 图像 | LM1B / text8 文本 | 通用离散序列 |
| **词汇表** | 256 像素值 | 8K / 27 token | 通用 |
| **序列长度** | 固定 (32x32x3) | 固定 (128/256) | **可变** |
| **扩散类型** | uniform/gaussian/absorbing | beta-diagonal/mask/band | 插入+删除+替换 |
| **模型** | UNet | Transformer (Flaxformer) | 通用模型接口 |
| **配置** | ml_collections | Gin | Gin |
| **框架复杂度** | 中等 | 高 | 高 |
| **论文对应** | NeurIPS 2021 §6 | NeurIPS 2021 §5 | INNF+ 2021 Workshop |

### 设计模式总结

1. **配置驱动**：images 用 `ml_collections`，text 和 insertdelete 用 `gin-config`
2. **函数式编程**：JAX 的纯函数风格，`TrainState` 是不可变 dataclass
3. **pmap 并行**：通过 `jax.pmap` 实现数据并行，`jax.lax.pmean` 跨设备平均
4. **Gumbel-max 技巧**：离散采样的可微近似
5. **对数空间运算**：大量使用对数概率以避免数值下溢
6. **注册机制**：数据集、任务都使用注册模式（registry pattern）

---

> **编写日期**：2026-05-20
> **代码版本**：Google Research D3PM (NeurIPS 2021)
