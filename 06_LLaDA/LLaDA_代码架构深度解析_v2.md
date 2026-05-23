# LLaDA 代码架构深度解析

> **版本**: 2.0 - 书级别技术文档  
> **目标读者**: 希望深入理解LLaDA实现的开发者、研究人员  
> **阅读建议**: 配合论文《Large Language Diffusion Models》(NeurIPS 2025) 一同阅读

---

## 目录

- [前言：为什么需要这份文档](#前言为什么需要这份文档)
- [第1章 整体架构概览](#第1章-整体架构概览)
- [第2章 核心采样算法：generate.py 完全解析](#第2章-核心采样算法generatepy-完全解析)
- [第3章 似然评估：get_log_likelihood.py 完全解析](#第3章-似然评估get_log_likelihoodpy-完全解析)
- [第4章 对话系统：chat.py 完全解析](#第4章-对话系统chatpy-完全解析)
- [第5章 评估框架：eval_llada.py 完全解析](#第5章-评估框架eval_lladapy-完全解析)
- [第6章 可视化界面：app.py 完全解析](#第6章-可视化界面apppy-完全解析)
- [第7章 训练实现原理](#第7章-训练实现原理)
- [第8章 工程技巧与最佳实践](#第8章-工程技巧与最佳实践)
- [附录 核心公式速查表](#附录-核心公式速查表)

---

## 前言：为什么需要这份文档

LLaDA（Large Language Diffusion with mAsking）是首个达到8B参数规模的扩散语言模型，它证明了扩散模型在大语言模型领域可以达到与自回归模型相媲美的性能。理解LLaDA的实现不仅需要掌握扩散模型的理论基础，更需要深入理解其工程实现中的各种技巧和设计决策。

本文档的目标是从**代码实现角度**系统性拆解LLaDA，不仅展示完整代码，更重要的是：

1. **逐行解析**：解释每一行代码背后的原理
2. **数学公式推导**：将代码与理论公式对应起来
3. **工程技巧分析**：说明为什么选择这样的实现方式
4. **设计权衡讨论**：分析不同参数和策略的trade-off

---

## 第1章 整体架构概览

### 1.1 项目文件结构

```
06_LLaDA/LLaDA/
├── generate.py                    # 核心采样算法
│   ├── add_gumbel_noise()         # Gumbel采样
│   ├── get_num_transfer_tokens()  # 计算每步揭示token数
│   └── generate()                 # 主生成函数
│
├── get_log_likelihood.py          # 对数似然评估
│   ├── forward_process()          # 前向加噪过程
│   ├── get_logits()               # 带CFG的logits计算
│   └── get_log_likelihood()       # 蒙特卡洛估计
│
├── chat.py                        # 对话系统
│   └── chat()                     # 多轮对话循环
│
├── eval_llada.py                  # 评估框架
│   └── LLaDAEvalHarness           # lm-evaluation-harness集成类
│
├── app.py                         # Gradio可视化界面
│   ├── generate_response_with_visualization()  # 可视化生成
│   └── create_chatbot_demo()      # UI构建
│
└── GUIDELINES.md                  # 训练指南
```

### 1.2 核心设计哲学

LLaDA的代码设计遵循以下原则：

**1. 理论与实践的统一**
- 每个实现细节都能在论文公式中找到对应
- 代码注释中明确引用论文的公式编号

**2. 工程实用性优先**
- 使用bfloat16而非float32节省显存
- 支持batch处理提高效率
- 提供多种采样策略（贪婪、采样、CFG等）

**3. 模块化设计**
- 采样、评估、对话、可视化各司其职
- 核心函数可独立使用

---

## 第2章 核心采样算法：generate.py 完全解析

`generate.py`是LLaDA最核心的文件，实现了从全掩码序列逐步恢复文本的反向扩散过程。这一章我们将逐行解析整个文件。

### 2.1 导入与依赖

```python
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
```

**依赖分析：**

1. **PyTorch (`torch`)**: 深度学习框架，用于张量计算和自动微分
2. **NumPy (`np`)**: 数值计算库，主要用于`-np.inf`常量和随机数
3. **PyTorch Functional (`F`)**: 提供softmax、cross_entropy等函数
4. **Transformers**: Hugging Face库，用于加载预训练模型和tokenizer

**为什么不使用其他库？**
- LLaDA的设计尽量保持简洁，避免过度依赖
- 核心算法完全用PyTorch实现，便于理解和修改

---

### 2.2 Gumbel噪声机制：`add_gumbel_noise`函数

```python
def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves 
    perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
```

#### 2.2.1 数学原理：Gumbel-Max Trick

**问题背景：**

给定一个分类分布 $p = (p_1, p_2, ..., p_K)$，如何从中采样？最直接的方法是根据概率累积分布进行采样，但这种方法是不可导的。

**Gumbel-Max Trick** 提供了一种重参数化的方法：

$$
\text{sample} = \arg\max_i \left( \log p_i + G_i \right)
$$

其中 $G_i$ 是服从**Gumbel分布**的随机变量。

**Gumbel分布：**

Gumbel分布的CDF为 $F(g) = e^{-e^{-g}}$，可以通过逆变换采样得到：

如果 $U \sim \text{Uniform}(0, 1)$，则：

$$
G = -\log(-\log U)
$$

服从标准Gumbel分布。

**代码实现分析：**

第14行：`if temperature == 0: return logits`

当温度为0时，直接返回原始logits，相当于贪婪解码（greedy decoding）。

第16行：`logits = logits.to(torch.float64)`

**关键技巧！** 这里使用float64而非默认的float32。为什么？

根据论文arXiv:2409.02908的研究，低精度的Gumbel Max虽然可以提高perplexity分数，但会降低实际生成质量。这是因为：

1. float32的精度有限，当logits值域跨度大时，softmax计算会有数值误差
2. Gumbel噪声涉及对数运算，需要高精度避免数值不稳定
3. float64提供了约15-17位十进制精度，足以处理绝大多数情况

第17行：`noise = torch.rand_like(logits, dtype=torch.float64)`

生成与logits形状相同的随机数，均匀分布在$(0, 1)$区间。

第18行：`gumbel_noise = (- torch.log(noise)) ** temperature`

这里有两个数学变形：

**变形1：对数的处理**

原始公式：$G = -\log(-\log U)$

代码实现：`(- torch.log(noise))` 对应 $-\log U$

但是代码中没有外层的 $-\log$，这是因为：

$$\arg\max_i (\log p_i + G_i) = \arg\max_i (\log p_i - \log(-\log U_i))$$

注意到：

$$\log p_i - \log(-\log U_i) = \log \frac{p_i}{-\log U_i}$$

因此：

$$\arg\max_i \left( \log p_i + G_i \right) = \arg\max_i \frac{p_i}{-\log U_i}$$

**变形2：温度参数**

代码使用 `** temperature` 而非加法。为什么？

这是Gumbel分布的**温度缩放**（temperature scaling）技巧：

标准Gumbel采样：$\arg\max_i (\log p_i + G_i)$

带温度$\tau$的采样：$\arg\max_i (\log p_i / \tau + G_i)$

等价于：$\arg\max_i (p_i^{1/\tau} \cdot e^{G_i})$

代码中的实现：

$$(-\log U)^{\tau} = \exp(\tau \cdot \log(-\log U))$$

当$\tau \to 0$时，分布趋于尖锐（贪婪）；当$\tau \to \infty$时，分布趋于均匀（随机）。

第19行：`return logits.exp() / gumbel_noise`

返回的是：`exp(logits) / (-log(noise))^temperature`

这等价于：$p_i / (-\log U_i)^\tau$

然后在外部调用`torch.argmax`得到最终采样结果。

#### 2.2.2 数值稳定性分析

为什么这行代码是数值稳定的？

1. **除法vs乘法**：使用除法而非减法避免精度损失
2. **float64保障**：高精度确保中间计算不会下溢或上溢
3. **温度控制**：通过temperature参数可以平滑控制采样随机性

**工程技巧总结：**

| 技巧 | 目的 | 效果 |
|------|------|------|
| float64 | 避免精度损失 | 生成质量提升 |
| 温度参数 | 控制采样多样性 | 灵活调节 |
| 惰性求值 | temperature=0时跳过计算 | 提升效率 |

---

### 2.3 Token转移数量计算：`get_num_transfer_tokens`函数

```python
def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized 
    into steps intervals. Furthermore, because LLaDA employs a linear 
    noise schedule (as defined in Eq. (8)), the expected number of tokens 
    transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that 
    need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)
    
    base = mask_num // steps
    remainder = mask_num % steps
    
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, 
        device=mask_index.device, dtype=torch.int64
    ) + base
    
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    
    return num_transfer_tokens
```

#### 2.3.1 数学背景：线性噪声调度

根据论文公式(8)，LLaDA使用**线性掩码调度**：

$$
q_{t|0}(x_t^i = \mathbf{M} | x_0^i) = t, \quad t \in [0, 1]
$$

这意味着在时间$t$，每个token被掩码的概率与时间成正比。

**反向过程的离散化：**

在反向采样时，我们需要将连续区间$[0, 1]$离散为`steps`个步骤。由于噪声调度是线性的，每步应该揭示的token数量应该是**均匀的**。

#### 2.3.2 算法逻辑解析

第30行：`mask_num = mask_index.sum(dim=1, keepdim=True)`

计算每个batch中剩余掩码token的数量。

- `mask_index`: shape为`(batch_size, seq_len)`的布尔张量
- `sum(dim=1)`: 沿序列维度求和，得到每个样本的掩码数
- `keepdim=True`: 保持维度为`(batch_size, 1)`，便于后续广播

第32-33行：
```python
base = mask_num // steps
remainder = mask_num % steps
```

这是整数除法，计算：
- `base`: 每步基础揭示数量（向下取整）
- `remainder`: 余数（无法整除的部分）

**示例：**
如果`mask_num = 10`, `steps = 3`:
- `base = 10 // 3 = 3`
- `remainder = 10 % 3 = 1`

第35-38行：
```python
num_transfer_tokens = torch.zeros(...) + base
for i in range(mask_num.size(0)):
    num_transfer_tokens[i, :remainder[i]] += 1
```

**分配策略：**

将`remainder`个额外的token分配到前`remainder`步，使得：
- 前`remainder`步：每步`base + 1`个token
- 后`steps - remainder`步：每步`base`个token

**为什么这样分配？**

这种分配方式确保了：
1. **总和正确**：`(base + 1) * remainder + base * (steps - remainder) = mask_num`
2. **尽可能均匀**：余数分散在前几步，避免某一步负担过重
3. **确定性**：对于相同的`mask_num`和`steps`，结果始终一致（利于复现）

**继续示例：**
- 第0步：3 + 1 = 4个token
- 第1步：3个token  
- 第2步：3个token
- 总计：4 + 3 + 3 = 10 ✓

#### 2.3.3 与论文的联系

论文Appendix B.4提到，LLaDA在实验中尝试了不同的token揭示策略：

1. **均匀揭示**（本实现）：每步揭示固定数量
2. **置信度排序揭示**：按模型置信度排序揭示
3. **随机揭示**：随机选择揭示顺序

实验结果表明，均匀揭示配合置信度排序的选择策略效果最佳。

**工程技巧总结：**

| 技巧 | 说明 |
|------|------|
| 整数运算 | 使用`//`和`%`避免浮点误差 |
| 广播机制 | `keepdim=True`便于后续广播运算 |
| 就地修改 | `+= 1`避免创建新张量，节省内存 |

---

---

### 2.4 主生成函数：`generate`

`generate`函数是LLaDA采样算法的核心实现，它完成了从全掩码序列逐步恢复文本的完整反向扩散过程。

#### 2.4.1 完整代码

```python
@torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, 
             block_length=128, temperature=0., cfg_scale=0., 
             remasking='low_confidence', mask_id=126336, 
             logits_eos_inf=False, confidence_eos_eot_inf=False):
    '''
    Args:
        model: Mask predictor (Transformer Encoder).
        prompt: A tensor of shape (batch_size, prompt_length).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length for semi-autoregressive generation.
                     If < gen_length, uses block-wise generation.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        logits_eos_inf: Whether to set EOS token logits to -inf.
        confidence_eos_eot_inf: Whether to set EOS/EoT confidence to -inf.
    '''
    # ========== 阶段1：初始化 ==========
    # 创建全掩码序列，前部填充prompt
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length), 
        mask_id, dtype=torch.long
    ).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    # 处理attention mask（用于batch处理变长序列）
    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask, 
            torch.ones((prompt.shape[0], gen_length), 
                      dtype=attention_mask.dtype, device=model.device)
        ], dim=-1)
    
    # 标记prompt位置（用于CFG和保序）
    prompt_index = (x != mask_id)
    
    # ========== 阶段2：分块参数计算 ==========
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    
    # ========== 阶段3：反向扩散过程 ==========
    for num_block in range(num_blocks):
        # 当前块的掩码索引
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        
        # 计算每步应揭示的token数量
        num_transfer_tokens = get_num_transfer_tokens(
            block_mask_index, steps_per_block
        )
        
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            
            # ========== Classifier-Free Guidance ==========
            if cfg_scale > 0.:
                # 构造无条件版本（mask掉prompt）
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)  # batch加倍
                
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                    logits = model(x_, attention_mask=attention_mask_).logits
                else:
                    logits = model(x_).logits
                
                # 拆分条件/无条件logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                # CFG公式
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits
            
            # 可选：抑制EOS token
            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf
            
            # ========== Gumbel采样 ==========
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = -torch.inf
                logits[:, :, 126348] = -torch.inf
            
            # ========== 置信度计算 ==========
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            
            # 只考虑当前块内的位置（半自回归）
            x0_p[:, block_end:] = -np.inf
            
            # 保持已揭示token不变
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            # ========== 选择要揭示的token ==========
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(
                    confidence[j], k=num_transfer_tokens[j, i]
                )
                transfer_index[j, select_index] = True
            
            # 应用揭示
            x[transfer_index] = x0[transfer_index]
    
    return x
```

#### 2.4.2 逐阶段深度解析

##### 阶段1：初始化（第60-66行）

```python
x = torch.full(
    (prompt.shape[0], prompt.shape[1] + gen_length), 
    mask_id, dtype=torch.long
).to(model.device)
x[:, :prompt.shape[1]] = prompt.clone()
```

**代码解析：**

这里创建了反向扩散的初始状态`x`：

1. **形状计算**：`(batch_size, prompt_len + gen_len)`
   - 总长度 = prompt长度 + 生成长度
   - 支持batch并行处理

2. **全掩码初始化**：
   - 所有位置初始化为`mask_id`（126336）
   - 对应扩散过程的起始点 $t=1$（全掩码）

3. **条件填充**：
   - 前`prompt_len`个位置填充真实的prompt token
   - 这些位置在后续过程中不会被重新掩码
   - 对应条件生成 $p_\theta(x_0 | \text{prompt})$

**与论文的联系：**

这对应论文中的反向过程初始化：

$$
x_1 \sim q_{1|0}(\cdot | x_0) \quad \text{即全掩码状态}
$$

但因为我们做的是**条件生成**，所以保留了prompt部分不掩码：

$$
x_1^{(i)} = \begin{cases}
x_0^{(i)} & \text{if } i \leq \text{prompt_len} \\
\mathbf{M} & \text{otherwise}
\end{cases}
$$

---

```python
if attention_mask is not None:
    attention_mask = torch.cat([
        attention_mask, 
        torch.ones((prompt.shape[0], gen_length), 
                  dtype=attention_mask.dtype, device=model.device)
    ], dim=-1)
```

**代码解析：**

处理attention mask以支持变长序列的batch处理：

1. **输入mask**：`attention_mask`形状为`(batch_size, prompt_len)`
   - 1表示有效token
   - 0表示padding

2. **生成部分mask**：创建全1的mask `(batch_size, gen_length)`
   - 生成部分所有位置都是有效的

3. **拼接**：沿序列维度(dim=-1)拼接
   - 结果形状：`(batch_size, prompt_len + gen_length)`

**为什么需要这个？**

在Transformer中，attention mask用于告诉模型哪些位置是真实的token，哪些是padding。这对于batch处理不同长度的prompt至关重要。

**工程技巧：**

- 使用`torch.ones`创建生成部分的mask（所有位置都有效）
- 显式指定`device`确保张量在正确的设备上（CPU/GPU）
- 使用`dtype=attention_mask.dtype`保持数据类型一致

---

```python
prompt_index = (x != mask_id)
```

**代码解析：**

创建布尔掩码`prompt_index`，标记哪些位置是prompt（非掩码）。

**用途：**

1. **CFG实现**：在构造无条件输入时，需要mask掉prompt
2. **保持约束**：确保prompt在生成过程中不会被修改
3. **置信度计算**：在计算置信度时，prompt位置应该被忽略

**实现细节：**

- `x != mask_id`逐元素比较
- 结果形状与`x`相同：`(batch_size, seq_len)`
- prompt位置为`True`，待生成位置为`False`

---

##### 阶段2：分块参数计算（第68-72行）

```python
assert gen_length % block_length == 0
num_blocks = gen_length // block_length

assert steps % num_blocks == 0
steps_per_block = steps // num_blocks
```

**代码解析：**

这里实现了**半自回归（Semi-Autoregressive）生成**的参数计算。

**为什么要分块？**

全并行生成（所有位置同时去噪）虽然速度快，但长序列质量会下降。完全串行（一次揭示一个token）质量最好但速度极慢。

半自回归是折中方案：
- 将生成序列分成多个块
- 块内并行去噪
- 块间串行处理

**参数计算：**

1. `num_blocks = gen_length // block_length`：计算块数
   - 必须整除，否则assert失败

2. `steps_per_block = steps // num_blocks`：每块的步数
   - 总步数均匀分配到每个块

**示例：**
```
gen_length = 128
block_length = 32
steps = 128

num_blocks = 128 // 32 = 4块
steps_per_block = 128 // 4 = 32步/块
```

**设计权衡：**

| block_length | 特点 | 适用场景 |
|-------------|------|----------|
| = gen_length | 全并行，最快 | 短序列（<64 tokens） |
| = gen_length / 4 | 平衡速度和效果 | 中等长度（128-256 tokens） |
| = 1 | 完全串行，最慢但最好 | 需要最高质量时 |

---

##### 阶段3：反向扩散过程（第74-118行）

这是整个算法的核心循环。让我们逐行分析：

**外层循环（分块）：**

```python
for num_block in range(num_blocks):
    block_start = prompt.shape[1] + num_block * block_length
    block_end = prompt.shape[1] + (num_block + 1) * block_length
    block_mask_index = (x[:, block_start:block_end] == mask_id)
```

**代码解析：**

1. **块边界计算**：
   - `block_start`：当前块起始位置（相对于整个序列）
   - `block_end`：当前块结束位置

2. **块内掩码索引**：
   - `x[:, block_start:block_end]`：截取当前块
   - `== mask_id`：找出块内仍为掩码的位置
   - `block_mask_index`形状：`(batch_size, block_length)`

**注意：** 这里只考虑当前块内的掩码，之前块已经揭示的位置不参与当前块的计算。

---

```python
    num_transfer_tokens = get_num_transfer_tokens(
        block_mask_index, steps_per_block
    )
```

**代码解析：**

计算当前块每步应揭示的token数量。

**返回值**：形状`(batch_size, steps_per_block)`
- `num_transfer_tokens[j, i]`：第j个样本在第i步应揭示的数量

---

**内层循环（逐步去噪）：**

```python
    for i in range(steps_per_block):
        mask_index = (x == mask_id)
```

**代码解析：**

`mask_index`标记整个序列中所有仍为掩码的位置。这是为了后续计算中保持已揭示token不变。

---

**Classifier-Free Guidance（CFG）：**

```python
        if cfg_scale > 0.:
            # 构造无条件版本（mask掉prompt）
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)  # batch加倍
```

**代码解析：**

CFG需要同时计算条件生成和无条件生成的logits：

1. **克隆条件输入**：`un_x = x.clone()`
2. **构造无条件输入**：`un_x[prompt_index] = mask_id`
   - 将prompt位置也变成掩码
   - 这相当于"没有条件信息"
3. **拼接**：`torch.cat([x, un_x], dim=0)`
   - 沿batch维度(dim=0)拼接
   - batch大小从B变为2B

**工程技巧：**

- 将条件和无条件输入拼接后一次性forward，比两次forward更高效
- GPU并行计算可以同时处理2B个样本

---

```python
            if attention_mask is not None:
                attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
            else:
                logits = model(x_).logits
```

**代码解析：**

如果使用了attention mask，也需要相应地将mask加倍，以匹配拼接后的输入。

---

```python
            # 拆分条件/无条件logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            # CFG公式
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
```

**代码解析：**

1. **`torch.chunk(logits, 2, dim=0)`**：将logits沿batch维度拆分为两份
   - `logits`：条件生成的logits（前B个）
   - `un_logits`：无条件生成的logits（后B个）

2. **CFG公式应用**：

$$
\hat{\ell} = \ell_{\text{uncond}} + (1 + s) \cdot (\ell_{\text{cond}} - \ell_{\text{uncond}})
$$

其中$s$是`cfg_scale`。

**原理分析：**

CFG通过放大条件与无条件的差异来增强条件控制：
- 当$\ell_{\text{cond}} > \ell_{\text{uncond}}$（条件支持该token），会进一步放大
- 当$\ell_{\text{cond}} < \ell_{\text{uncond}}$（条件不支持），会进一步抑制
- 效果：让生成更"听话"地遵循条件

---

**Gumbel采样：**

```python
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
```

**代码解析：**

1. **添加Gumbel噪声**：实现温度控制的采样
2. **贪婪选择**：`torch.argmax`选择概率最大的token

**注意：** 这里的`x0`是模型预测的**所有位置**的token，包括已揭示的位置。

---

**置信度计算：**

```python
        if remasking == 'low_confidence':
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
            )
        elif remasking == 'random':
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
```

**代码解析：**

**策略1：low_confidence（低置信度优先揭示）**

计算每个预测位置的置信度（概率）：

1. `F.softmax(logits, dim=-1)`：计算概率分布，形状`(B, L, V)`
2. `torch.unsqueeze(x0, -1)`：增加维度，形状`(B, L, 1)`
3. `torch.gather(p, dim=-1, index=...)`：收集预测token的概率
   - 对于每个位置，取出预测token对应的概率值
4. `torch.squeeze(..., -1)`：去掉最后一维，形状`(B, L)`

**为什么这样实现？**

我们希望优先揭示模型**最确定**的token。置信度越高，说明模型对该预测越有信心。

**策略2：random（随机揭示）**

直接使用随机数作为"置信度"，完全不考虑模型预测。

**用于ablation study**，证明low_confidence策略的有效性。

---

```python
        x0_p[:, block_end:] = -np.inf
```

**代码解析：**

**关键步骤！** 将当前块之后的位置的置信度设为负无穷。

**作用：**

1. **半自回归约束**：确保在当前块完成之前，不会揭示后续块的位置
2. **注意力聚焦**：让模型专注于当前块的去噪

**数学表达：**

$$
\text{confidence}_i = \begin{cases}
\text{original} & i \in [\text{block_start}, \text{block_end}] \\
-\infty & \text{otherwise}
\end{cases}
$$

---

```python
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)
```

**代码解析：**

1. **保持已揭示token不变**：
   - `torch.where(mask_index, x0, x)`
   - 如果位置i仍是掩码(`mask_index[i]=True`)，使用新预测`x0[i]`
   - 否则，保持原值`x[i]`

2. **置信度掩码**：
   - 已揭示位置的置信度设为`-inf`，确保不会被再次选中

---

```python
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
            transfer_index[j, select_index] = True
```

**代码解析：**

选择要揭示的token：

1. **创建布尔索引**：`transfer_index`形状`(B, L)`，初始全False

2. **对每个样本**：
   - `torch.topk(confidence[j], k=...)`：找出置信度最高的k个位置
   - `select_index`：这些位置的索引
   - `transfer_index[j, select_index] = True`：标记为要揭示

**注意：** 由于之前将当前块外和已揭示位置的置信度设为`-inf`，`topk`会自动跳过它们。

---

```python
        x[transfer_index] = x0[transfer_index]
```

**代码解析：**

**揭示操作**：将选中的位置从掩码替换为预测值。

这行代码完成了反向扩散的一步：从$t$到$t-\Delta t$的状态转移。

---

#### 2.4.3 算法流程总结

```
输入: prompt, model
输出: 完整序列x

1. 初始化:
   - x = [prompt] + [MASK] * gen_length
   - prompt_index = (x != MASK)

2. 分块:
   - num_blocks = gen_length / block_length
   - steps_per_block = steps / num_blocks

3. 对每个块:
   a. 计算该块每步应揭示的token数
   b. 对每步:
      i.   模型前向：获取logits
      ii.  [可选] CFG增强
      iii. Gumbel采样：得到预测x0
      iv.  计算置信度
      v.    选择top-k置信度位置
      vi.  揭示这些位置（更新x）

4. 返回x
```

#### 2.4.4 与论文公式的对应

| 代码概念 | 论文符号 | 公式 |
|---------|---------|------|
| `x` | $x_t$ | 时间t的状态 |
| `mask_index` | $\mathbf{1}[x_t^i = \mathbf{M}]$ | 掩码指示函数 |
| `logits` | $p_\theta(x_0 \| x_t)$ | 条件分布 |
| `num_transfer_tokens` | - | 离散化的转移数量 |
| `remasking` | - | 对应公式(10)的重掩码策略 |

**论文公式(10) - 反向条件分布：**

$$
q_{s|t}(x_s^i | x_t) = \begin{cases}
1 & x_t^i \neq \mathbf{M}, x_s^i = x_t^i \\
\frac{s}{t} & x_t^i = \mathbf{M}, x_s^i = \mathbf{M} \\
\frac{t-s}{t} q_{0|t}(x_s^i | x_t) & x_t^i = \mathbf{M}, x_s^i \neq \mathbf{M} \\
0 & \text{otherwise}
\end{cases}
$$

代码实现中的对应：

1. **情况1**（未掩码保持不变）：`x0 = torch.where(mask_index, x0, x)`
2. **情况2**（继续掩码）：通过`confidence = -inf`实现
3. **情况3**（揭示）：`x[transfer_index] = x0[transfer_index]`

**论文公式(14) - 低方差训练目标：**

虽然这是训练目标，但采样时也遵循类似思想：

$$
-\mathbb{E}_{l, x_0, x_l}\left[ \frac{L}{l} \sum_{i=1}^{L} \mathbf{1}[x_l^i = \mathbf{M}] \log p_\theta(x_0^i | x_l) \right]
$$

代码中的`num_transfer_tokens`计算实现了$l$的均匀采样。

---

---

## 第3章 似然评估：get_log_likelihood.py 完全解析

`get_log_likelihood.py`实现了计算条件对数似然 $\log p(\text{answer} | \text{prompt})$ 的功能，这是评估模型性能的关键指标。

### 3.1 前向加噪过程：`forward_process`函数

```python
def forward_process(batch, prompt_index, mask_id):
    '''
    前向扩散过程：对answer部分进行随机掩码
    使用低方差版本：每次恰好掩码k个token（k从1到target_len均匀采样）
    '''
    b, l = batch.shape
    
    # answer部分的长度
    target_len = (l - prompt_index.sum()).item()
    
    # 采样掩码数量k（1到target_len之间）
    k = torch.randint(1, target_len + 1, (), device=batch.device)
    
    # 为batch中每个样本创建不同的k值（线性递增后循环）
    x = torch.round(torch.linspace(
        float(k), 
        k + (b - 1) * (target_len / b), 
        steps=b, 
        device=batch.device
    )).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len
    
    # 创建掩码矩阵
    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)  # 前x[i]个为True
    
    # 对每个样本的掩码位置随机打乱（确保随机性）
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]
    
    # 合并prompt部分（不掩码）
    is_mask = torch.cat((
        torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device),
        is_mask
    ), dim=1)
    
    noisy_batch = torch.where(is_mask, mask_id, batch)
    
    # 返回加噪序列和掩码比例
    return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)
```

#### 3.1.1 数学原理：低方差前向过程

**论文公式(3) vs 公式(14)：**

标准前向过程（公式3）：

$$
t \sim U[0,1], \quad x_t \sim q_{t|0}(\cdot | x_0)
$$

掩码数量是随机变量（服从二项分布），方差较大。

低方差版本（公式14）：

$$
l \sim U\{1, 2, ..., L\}, \quad x_l \sim \text{UniformSample}(l \text{ positions})
$$

掩码数量是确定性的（恰好l个），方差小。

**代码实现对应：**

第11行：`k = torch.randint(1, target_len + 1, (), device=batch.device)`

从$[1, \text{target_len}]$均匀随机采样一个整数$k$，对应公式中的$l$。

第13-14行：
```python
x = torch.round(torch.linspace(
    float(k), 
    k + (b - 1) * (target_len / b), 
    steps=b, 
    device=batch.device
)).long()
```

**核心技巧！** 这里为batch中的每个样本创建**不同**的$k$值：

- 起始：$k$
- 结束：$k + (b-1) \cdot \frac{\text{target_len}}{b}$
- 在区间内均匀采样$b$个点

**为什么这样做？**

这样可以在**单次forward**中覆盖不同的噪声水平，提高蒙特卡洛估计的效率。

**示例：** target_len=100, batch_size=4, k=10
- k值: [10, 35, 60, 85] （均匀分布在1-100之间）

第17-18行：
```python
indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
is_mask = indices < x.unsqueeze(1)
```

创建掩码矩阵：
- `indices`: 形状`(b, target_len)`，每行是`[0, 1, 2, ..., target_len-1]`
- `x.unsqueeze(1)`: 形状`(b, 1)`
- 广播后比较：前`x[i]`个位置为True

第20-21行：
```python
for i in range(b):
    is_mask[i] = is_mask[i][torch.randperm(target_len)]
```

**关键步骤！** 对每行的掩码位置进行随机打乱：

- 如果不大乱，总是掩码前$k$个位置，模型可能学到位置bias
- `torch.randperm`生成随机排列，确保掩码位置完全随机

第26行：
```python
return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)
```

返回两个值：
1. `noisy_batch`: 加噪后的序列
2. `p_mask`: 掩码比例 $t = k / \text{target_len}$

**为什么需要p_mask？**

因为损失函数需要除以掩码比例（公式3中的$1/t$权重）：

$$
\mathcal{L} = \mathbb{E}\left[ \frac{1}{t} \sum_i \mathbf{1}[x_t^i = \mathbf{M}] \log p_\theta(x_0^i | x_t) \right]
$$

---

### 3.2 对数似然计算：`get_log_likelihood`函数

```python
@torch.no_grad()
def get_log_likelihood(model, prompt, answer, mc_num=128, batch_size=16, 
                       cfg_scale=0., mask_id=126336):
    '''
    使用蒙特卡洛估计计算条件对数似然 log p(answer|prompt)
    
    Args:
        mc_num: 蒙特卡洛采样次数（论文推荐128次足够稳定）
        batch_size: 每次处理的样本数（mc_num必须是batch_size倍数）
    
    理论基础: 公式(14)的低方差形式
    '''
    # 拼接prompt和answer
    seq = torch.concatenate([prompt, answer])[None, :]
    seq = seq.repeat((batch_size, 1)).to(model.device)
    
    # prompt位置的mask
    prompt_index = torch.arange(seq.shape[1], device=model.device) < len(prompt)
    
    loss_ = []
    for _ in range(mc_num // batch_size):
        # 前向加噪
        perturbed_seq, p_mask = forward_process(seq, prompt_index, mask_id)
        mask_index = perturbed_seq == mask_id
        
        # 模型预测
        logits = get_logits(model, perturbed_seq, prompt_index, cfg_scale, mask_id)
        
        # 计算加权交叉熵损失
        # 关键：除以掩码比例p_mask（公式中的1/t权重）
        loss = F.cross_entropy(
            logits[mask_index], 
            seq[mask_index], 
            reduction='none'
        ) / p_mask[mask_index]
        
        loss = loss.sum() / batch_size
        loss_.append(loss.item())
    
    # 返回负对数似然（越高越好）
    return -sum(loss_) / len(loss_)
```

#### 3.2.1 算法逻辑解析

**目标：** 计算 $\log p_\theta(\text{answer} | \text{prompt})$

**理论基础：**

根据论文公式(4)，对数似然的上界为：

$$
-\log p_\theta(x_0) \leq \mathcal{L}(\theta) = -\mathbb{E}_{t, x_t}\left[ \frac{1}{t} \sum_i \mathbf{1}[x_t^i = \mathbf{M}] \log p_\theta(x_0^i | x_t) \right]
$$

因此，对数似然的估计为：

$$
\log p_\theta(x_0) \approx -\frac{1}{N} \sum_{n=1}^N \frac{1}{t_n} \sum_i \mathbf{1}[x_{t_n}^i = \mathbf{M}] \log p_\theta(x_0^i | x_{t_n})
$$

第61-63行：
```python
seq = torch.concatenate([prompt, answer])[None, :]
seq = seq.repeat((batch_size, 1)).to(model.device)
prompt_index = torch.arange(seq.shape[1], device=model.device) < len(prompt)
```

**数据准备：**
1. 拼接prompt和answer
2. 重复`batch_size`次（为了并行计算多个MC样本）
3. 创建prompt位置的布尔mask

第66-76行：蒙特卡洛循环

```python
for _ in range(mc_num // batch_size):
    perturbed_seq, p_mask = forward_process(seq, prompt_index, mask_id)
    mask_index = perturbed_seq == mask_id
    
    logits = get_logits(model, perturbed_seq, prompt_index, cfg_scale, mask_id)
    
    loss = F.cross_entropy(logits[mask_index], seq[mask_index], reduction='none')
    loss = loss / p_mask[mask_index]  # 除以t
    loss = loss.sum() / batch_size
    loss_.append(loss.item())
```

**关键步骤分析：**

1. **前向加噪**：`forward_process`返回加噪序列和掩码比例$p$

2. **模型预测**：`get_logits`获取预测logits（支持CFG）

3. **交叉熵计算**：`F.cross_entropy`计算每个掩码位置的负对数似然
   - `logits[mask_index]`: 只考虑掩码位置的预测
   - `seq[mask_index]`: 对应的真实token
   - `reduction='none'`: 返回每个位置的损失，不求和

4. **加权**：`loss / p_mask[mask_index]`
   - 这是公式中的$1/t$权重！
   - 当$t$小（掩码少）时，权重$1/t$大，强调"困难"样本

5. **归一化**：`loss.sum() / batch_size`
   - 对整个batch求平均

第78行：
```python
return -sum(loss_) / len(loss_)
```

返回负的平均损失，即对数似然的估计值。

**为什么是负的？**

- `cross_entropy`返回的是**负**对数似然
- 我们取负号，得到**正**的对数似然（越高越好）

#### 3.2.2 超参数选择

**论文Appendix B.5的建议：**

| 任务类型 | mc_num推荐 | 说明 |
|---------|-----------|------|
| MMLU/CMMLU/C-EVAL | 1 | 只需单个token的似然 |
| 其他benchmark | 128 | 足够稳定的结果 |

**batch_size的选择：**
- 必须是`mc_num`的约数
- 越大越高效，但受限于显存
- 论文使用16-32

---

## 第4章 对话系统：chat.py 完全解析

`chat.py`实现了与LLaDA模型的多轮对话功能，支持保持对话历史上下文。

### 4.1 完整代码

```python
import torch
from generate import generate
from transformers import AutoTokenizer, AutoModel


def chat():
    device = 'cuda'
    model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', 
        trust_remote_code=True
    )
    
    gen_length = 128
    steps = 128
    print('*' * 66)
    print(f'**  Answer Length: {gen_length}  |  Sampling Steps: {steps}  **')
    print('*' * 66)
    
    conversation_num = 0
    while True:
        user_input = input("Enter your question: ")
        
        # 应用对话模板
        m = [{"role": "user", "content": user_input}]
        user_input = tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False
        )
        input_ids = tokenizer(user_input)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        
        # 对话历史拼接
        if conversation_num == 0:
            prompt = input_ids
        else:
            # 关键：去掉新的input_ids的BOS token避免重复
            prompt = torch.cat([prompt, input_ids[:, 1:]], dim=1)
        
        # 生成回复
        out = generate(
            model, prompt, 
            steps=steps, gen_length=gen_length, block_length=32,
            temperature=0., cfg_scale=0., remasking='low_confidence'
        )
        
        answer = tokenizer.batch_decode(
            out[:, prompt.shape[1]:], 
            skip_special_tokens=True
        )[0]
        print(f"Bot's reply: {answer}")
        
        # 更新prompt：移除EOS后作为下一轮的历史
        prompt = out[out != 126081].unsqueeze(0)
        conversation_num += 1
        print('-' * 70)


if __name__ == "__main__":
    chat()
```

### 4.2 关键技术解析

#### 4.2.1 对话历史管理

```python
if conversation_num == 0:
    prompt = input_ids
else:
    # 去掉BOS token（通常索引为0）避免序列中出现多个BOS
    prompt = torch.cat([prompt, input_ids[:, 1:]], dim=1)
```

**关键技巧：移除BOS token**

**问题：** 每个序列通常以BOS (Beginning of Sequence, 如`<s>`) token开始。如果直接拼接：

```
第一轮: <s> User: Hello Assistant: Hi there!
第二轮输入: <s> User: How are you?
直接拼接: <s> Hello... <s> How are you?
         ↑ 多余的BOS！
```

这会导致模型困惑，因为序列中间出现了BOS。

**解决方案：**

去掉新输入的BOS token：`input_ids[:, 1:]`（从索引1开始取）

```
正确拼接: <s> Hello... User: How are you?
```

#### 4.2.2 EOS处理

```python
prompt = out[out != 126081].unsqueeze(0)
```

**126081是EOS token的ID**（End of Sequence）。

**为什么要移除EOS？**

模型生成时会输出`<EOS>`标记表示生成结束：

```
生成结果: ... response text <EOS> <PAD> <PAD> ...
```

如果直接把整个`out`作为下一轮的历史，`<EOS>`会在序列中间，导致模型认为对话已经结束。

**过滤操作：**

- `out != 126081`: 创建布尔mask，标记非EOS位置
- `out[...]`: 只保留非EOS的token
- `.unsqueeze(0)`: 增加batch维度，形状从`(L,)`变为`(1, L)`

---

## 第5章 评估框架：eval_llada.py 完全解析

`eval_llada.py`实现了与`lm-evaluation-harness`库的集成，支持标准化评估。

### 5.1 核心类：`LLaDAEvalHarness`

```python
@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(self, model_path='', mask_id=126336, max_length=4096,
                 batch_size=32, mc_num=128, is_check_greedy=True, cfg=0.,
                 steps=1024, gen_length=1024, block_length=1024,
                 remasking='low_confidence', device="cuda", **kwargs):
        # ... 初始化代码
```

**装饰器`@register_model("llada_dist")`：**

这是`lm-evaluation-harness`的机制，将此类注册为名为"llada_dist"的模型，可以通过命令行调用。

### 5.2 关键方法解析

#### 5.2.1 贪婪解码验证

```python
@torch.no_grad()
def suffix_greedy_prediction(self, prefix, target):
    '''
    验证target是否可通过贪婪解码从prefix生成
    用于LAMBADA等需要验证的任务
    '''
    if not self.is_check_greedy:
        return False
    
    # 初始化全掩码序列
    seq = torch.full((1, len(prefix) + len(target)), self.mask_id, ...)
    seq[0, :len(prefix)] = prefix
    
    for i in range(len(target)):
        mask_index = (seq == self.mask_id)
        logits = self.get_logits(seq, prompt_index)[mask_index]
        x0 = torch.argmax(logits, dim=-1)
        
        # 只保留置信度最高的一个，其余重新掩码
        p = torch.softmax(logits.to(torch.float32), dim=-1)
        confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(-1)
        _, index = torch.sort(confidence, descending=True)
        x0[index[1:]] = self.mask_id
        
        seq[mask_index] = x0.clone()
    
    return target == seq[0, len(prefix):]
```

**算法：顺序贪婪解码**

与`generate`的并行揭示不同，这里使用**顺序揭示**：

1. 每步只揭示**一个**最高置信度的token
2. 其余预测重新掩码
3. 重复直到全部揭示

**为什么这样做？**

某些任务（如LAMBADA）需要验证答案是否可通过**贪婪采样**获得，而非条件生成。这检查的是模型的确定性行为。

#### 5.2.2 多GPU支持

```python
accelerator = accelerate.Accelerator()
if accelerator.num_processes > 1:
    self.accelerator = accelerator
    model_kwargs = {'device_map': {'': f'{self.accelerator.device}'}}
    # ...
    self.model = self.accelerator.prepare(self.model)
```

使用Hugging Face Accelerate库：
- 自动检测多GPU环境
- 自动处理分布式数据并行
- 简化多卡评估代码

---

## 第6章 可视化界面：app.py 完全解析

`app.py`使用Gradio构建交互式Web界面，展示扩散生成过程的动态可视化。

### 6.1 核心可视化函数

```python
def generate_response_with_visualization(model, tokenizer, device, messages, 
                                         gen_length=64, steps=32, 
                                         constraints=None, temperature=0.0, 
                                         cfg_scale=0.0, block_length=32,
                                         remasking='low_confidence'):
    '''
    生成并记录每一步的状态用于可视化
    '''
    visualization_states = []
    
    # 初始状态：全掩码
    initial_state = [(MASK_TOKEN, "#444444") for _ in range(gen_length)]
    visualization_states.append(initial_state)
    
    # ... 生成逻辑 ...
    
    # 每步记录状态
    for each_step:
        current_state = []
        for position in range(gen_length):
            if x[0, pos] == MASK_ID:
                current_state.append((MASK_TOKEN, "#444444"))  # 深灰
            elif newly_revealed:
                # 根据置信度着色
                if confidence < 0.3:
                    color = "#FF6666"  # 红色：低置信
                elif confidence < 0.7:
                    color = "#FFAA33"  # 橙色：中置信
                else:
                    color = "#66CC66"  # 绿色：高置信
                current_state.append((token, color))
            else:
                current_state.append((token, "#6699CC"))  # 蓝色：已揭示
        
        visualization_states.append(current_state)
    
    return visualization_states, final_text
```

### 6.2 颜色编码设计

```
颜色编码:
┌────────────────────────────────────────────────┐
│  #444444 (深灰) - [MASK] 待揭示                │
│  #FF6666 (红色) - 新揭示，低置信度 (<0.3)      │
│  #FFAA33 (橙色) - 新揭示，中置信度 (0.3-0.7)   │
│  #66CC66 (绿色) - 新揭示，高置信度 (>0.7)      │
│  #6699CC (蓝色) - 已揭示（历史）               │
└────────────────────────────────────────────────┘
```

**设计意图：**

1. **直观展示扩散过程**：用户可以看到token是如何一步步被揭示的
2. **置信度可视化**：颜色反映模型对每个预测的信心
3. **教学工具**：帮助理解扩散模型的去噪过程

---

## 第7章 训练实现原理

虽然训练代码未开源，但`GUIDELINES.md`提供了训练的核心逻辑。

### 7.1 预训练核心循环

```python
def forward_process(input_ids, eps=1e-3):
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, 126336, input_ids)
    return noisy_batch, masked_indices, p_mask


# 训练循环
input_ids = batch["input_ids"]

# 1%概率随机截断长度
if torch.rand(1) < 0.01:
    random_length = torch.randint(1, input_ids.shape[1] + 1, (1,))
    input_ids = input_ids[:, :random_length]

noisy_batch, masked_indices, p_mask = forward_process(input_ids)
logits = model(input_ids=noisy_batch).logits

# 加权交叉熵
token_loss = F.cross_entropy(
    logits[masked_indices], 
    input_ids[masked_indices], 
    reduction='none'
) / p_mask[masked_indices]

loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
```

### 7.2 关键技巧解析

#### 7.2.1 eps=1e-3的作用

```python
p_mask = (1 - eps) * t + eps
```

当$t \to 0$时，$p_{\text{mask}} \to \epsilon = 10^{-3}$，而非0。

**为什么？**

避免$t=0$时的数值不稳定：
- 损失函数需要除以$t$
- 如果$t=0$，会导致除以零错误
- $10^{-3}$足够小，不影响训练效果，但保证数值稳定

#### 7.2.2 随机截断策略

```python
if torch.rand(1) < 0.01:
    random_length = torch.randint(1, input_ids.shape[1] + 1, (1,))
    input_ids = input_ids[:, :random_length]
```

**1%的概率随机截断序列长度。**

**目的：**

让模型学习生成**变长序列**。如果不这样做，所有序列都是固定长度（如4096），模型可能学到长度偏差。

#### 7.2.3 SFT的特殊处理

```python
# SFT时不掩码prompt
noisy_batch[prompt_mask] = input_ids[prompt_mask]
```

与预训练不同，SFT需要保持prompt不变，只对answer部分进行掩码和预测。

---

## 第8章 工程技巧与最佳实践

### 8.1 数值稳定性

| 技巧 | 实现 | 效果 |
|------|------|------|
| float64 for Gumbel | `logits.to(torch.float64)` | 避免精度损失，提升生成质量 |
| eps防护 | `p_mask = (1-ε)*t + ε` | 避免t=0时的数值溢出 |
| attention mask | 显式处理padding | 正确处理变长序列 |

### 8.2 内存优化

| 技巧 | 实现 | 效果 |
|------|------|------|
| bfloat16 | `torch_dtype=torch.bfloat16` | 节省50%显存，保持精度 |
| @torch.no_grad() | 装饰器 | 禁用梯度，节省显存 |
| 及时清空 | `torch.cuda.empty_cache()` | 防止OOM |

### 8.3 效率优化

| 技巧 | 实现 | 效果 |
|------|------|------|
| batch加倍for CFG | `torch.cat([x, un_x])` | 单次forward处理2B样本 |
| left padding | `tokenizer.padding_side='left'` | 简化采样逻辑 |
| 块级并行 | `block_length < gen_length` | 平衡速度和质量 |

---

## 附录 核心公式速查表

### 前向过程

$$
q_{t|0}(x_t | x_0) = \prod_{i=1}^L q_{t|0}(x_t^i | x_0^i)
$$

$$
q_{t|0}(x_t^i | x_0^i) = \begin{cases}
1 - t & x_t^i = x_0^i \\
t & x_t^i = \mathbf{M}
\end{cases}
$$

### 反向过程

$$
q_{s|t}(x_s | x_t) = \prod_{i=1}^L q_{s|t}(x_s^i | x_t)
$$

### 训练目标

$$
\mathcal{L}(\theta) = -\mathbb{E}_{t, x_0, x_t}\left[ \frac{1}{t} \sum_{i=1}^{L} \mathbf{1}[x_t^i = \mathbf{M}] \log p_\theta(x_0^i | x_t) \right]
$$

### CFG公式

$$
\hat{\ell} = \ell_{\text{uncond}} + (1 + s) \cdot (\ell_{\text{cond}} - \ell_{\text{uncond}})
$$

---

**文档版本**: 2.0  
**最后更新**: 2025年  
**基于**: LLaDA官方实现 (https://github.com/ML-GSAI/LLaDA)

**贡献者**: OpenCode Agent  
**特别感谢**: LLaDA研究团队的开源工作
