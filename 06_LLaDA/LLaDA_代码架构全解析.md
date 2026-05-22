# LLaDA 代码架构全景解析

> **文档定位**: 从代码实现角度深入理解 LLaDA (Large Language Diffusion with mAsking) 的模型架构、采样算法和工程技巧
> 
> **适合读者**: 已掌握扩散模型理论基础，希望深入理解代码实现的开发者

---

## 目录

1. [项目架构概览](#1-项目架构概览)
2. [核心采样算法: `generate.py`](#2-核心采样算法-generatepy)
3. [似然评估: `get_log_likelihood.py`](#3-似然评估-get_log_likelihoodpy)
4. [对话系统: `chat.py`](#4-对话系统-chatpy)
5. [评估框架: `eval_llada.py`](#5-评估框架-eval_lladapy)
6. [可视化与交互: `app.py`](#6-可视化与交互-apppy)
7. [训练实现: `GUIDELINES.md` 解读](#7-训练实现-guidelinesmd-解读)
8. [工程技巧总结](#8-工程技巧总结)

---

## 1. 项目架构概览

### 1.1 文件结构

```
LLaDA/
├── generate.py              # 核心采样算法（扩散反向过程）
├── get_log_likelihood.py    # 对数似然计算（用于评估）
├── chat.py                  # 多轮对话系统
├── app.py                   # Gradio 可视化界面
├── eval_llada.py            # lm-evaluation-harness 集成
├── eval_reverse.py          # 逆转诅咒测试
├── GUIDELINES.md            # 训练和架构指南
├── data/
│   └── poem_data.json       # 逆转诅咒测试数据
└── visualization/           # 可视化工具
    ├── generate.py          # 带日志的生成（用于可视化）
    └── visualization_*.py   # 可视化脚本
```

### 1.2 核心设计理念

```
┌─────────────────────────────────────────────────────────────┐
│                    LLaDA 核心设计                            │
├─────────────────────────────────────────────────────────────┤
│  模型架构: Transformer Encoder (非 Decoder!)                 │
│  ├─ 移除因果掩码 (causal mask)                               │
│  ├─ 双向注意力机制                                          │
│  └─ 标准位置编码                                            │
│                                                             │
│  关键超参数:                                                │
│  ├─ mask_id = 126336  (专用 [MASK] token)                   │
│  ├─ 词汇表大小: ~100K                                       │
│  └─ 模型规模: 8B 参数                                       │
│                                                             │
│  扩散过程:                                                 │
│  ├─ 前向: 线性掩码调度 (t ~ U[0,1])                         │
│  ├─ 反向: 逐步去噪 + 重掩码策略                             │
│  └─ 采样: Gumbel噪声 + 置信度排序                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 核心采样算法: `generate.py`

### 2.1 完整代码

```python
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max 
    improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


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
    # ========== 初始化阶段 ==========
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
    
    # ========== 分块参数计算 ==========
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    
    # ========== 反向扩散过程 ==========
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
                x_ = torch.cat([x, un_x], dim=0)
                
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                    logits = model(x_, attention_mask=attention_mask_).logits
                else:
                    logits = model(x_).logits
                
                # 拆分条件/无条件logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                # CFG公式: un_cond + (1 + scale) * (cond - un_cond)
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
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf
            
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


def main():
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
    
    # 使用left-padding（简化采样逻辑）
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    
    assert tokenizer.pad_token_id != 126336
    
    prompts = [
        "Lily can run 12 kilometers per hour for 4 hours...",
        # ... more prompts
    ]
    
    # 添加对话模板
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    prompts = [
        tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) 
        for message in messages
    ]
    
    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)
    
    out = generate(
        model, input_ids, attention_mask, 
        steps=128, gen_length=128, block_length=32, 
        temperature=0., cfg_scale=0., remasking='low_confidence'
    )
    
    output = tokenizer.batch_decode(
        out[:, input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    for o in output:
        print(o)
        print('-' * 50)


if __name__ == '__main__':
    main()
```

### 2.2 代码深度解析

#### 2.2.1 Gumbel噪声机制 (`add_gumbel_noise`)

```python
def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)  # 关键！使用float64
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
```

**原理解析:**

**Gumbel-Max Trick** 是从分类分布中采样的重参数化技巧：

$$
\text{sample} = \arg\max_i \left( \log \pi_i + G_i \right)
$$

其中 $G_i = -\log(-\log(U_i))$ 是Gumbel随机变量，$U_i \sim \text{Uniform}(0,1)$。

**实现技巧:**

1. **float64精度**: 论文 arXiv:2409.02908 发现，低精度Gumbel Max虽然提高perplexity但降低生成质量。LLaDA使用`float64`确保数值稳定性。

2. **温度参数控制**: 
   - `temperature=0`: 贪婪解码（无噪声）
   - `temperature>0`: 采样多样化程度
   - 温度越高，采样越随机

3. **数学变形**: 代码使用等价形式避免对数运算:
   ```
   argmax(log π + G) = argmax(π * exp(-G))
   ```
   其中 `gumbel_noise = (-log(U))^temperature`

#### 2.2.2 Token转移数量计算 (`get_num_transfer_tokens`)

```python
def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)  # 每样本掩码数
    
    base = mask_num // steps      # 每步基础数量
    remainder = mask_num % steps  # 余数
    
    # 初始化：每步都是base
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    
    # 余数分配：前remainder步每步多1个
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    
    return num_transfer_tokens
```

**设计原理:**

由于LLaDA使用**线性掩码调度**（公式8）：$q_{t|0}(x_t^i = \mathbf{M} | x_0^i) = t$

时间区间 $[0,1]$ 被均匀离散为 `steps` 个区间，每步应揭示的token数量应**均匀分布**。

**示例:** 
- 剩余掩码数 = 10, 步数 = 3
- 基础 = 3, 余数 = 1
- 每步揭示: [4, 3, 3] （尽可能均匀）

#### 2.2.3 核心生成循环解析

**初始化阶段:**

```python
x = torch.full((batch_size, prompt_len + gen_length), mask_id, ...)
x[:, :prompt_len] = prompt.clone()
```

- `x`: 当前状态序列，初始为全掩码
- 前部填充prompt作为条件

**分块半自回归:**

```
总长度 = prompt + gen_length
块数 = gen_length / block_length
每块步数 = steps / num_blocks

对于每块:
  对于每步:
    1. 模型预测所有掩码位置
    2. 计算置信度
    3. 按置信度排序，揭示top-k
    4. 剩余位置保持掩码（remasking）
```

**关键决策点:**

1. **为什么用半自回归?** 
   - 全并行(block_length=gen_length): 速度最快，但长序列质量下降
   - 块级并行: 平衡速度和质量，是当前最佳实践
   - 完全串行(block_length=1): 质量最好但极慢

2. **置信度策略:**
   - `low_confidence`: 优先揭示模型最确定的token（基于softmax概率）
   - `random`: 随机选择，用于ablation study

#### 2.2.4 Classifier-Free Guidance (CFG) 实现

```python
if cfg_scale > 0.:
    # 构造无条件输入（mask掉prompt）
    un_x = x.clone()
    un_x[prompt_index] = mask_id
    x_ = torch.cat([x, un_x], dim=0)  # 2x batch
    
    logits = model(x_).logits
    logits, un_logits = torch.chunk(logits, 2, dim=0)
    
    # CFG公式
    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
```

**原理:**

CFG通过对比**条件生成**和**无条件生成**来增强条件控制：

$$
\hat{v} = v_{\text{uncond}} + (1 + s) \cdot (v_{\text{cond}} - v_{\text{uncond}})
$$

**工程实现技巧:**

1. **batch加倍**: 同时处理条件和无条件，利用GPU并行
2. **mask prompt作为无条件**: 简单但有效的无条件近似
3. **scale控制**: `cfg_scale=0` 禁用，`>0` 增强控制强度

---

## 3. 似然评估: `get_log_likelihood.py`

### 3.1 完整代码

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


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


def get_logits(model, batch, prompt_index, cfg_scale, mask_id):
    '''获取模型logits，支持CFG'''
    if cfg_scale > 0.:
        assert len(prompt_index) == batch.shape[1]
        prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
        un_batch = batch.clone()
        un_batch[prompt_index] = mask_id
        batch = torch.cat([batch, un_batch])
    
    input = batch
    logits = model(input).logits
    
    if cfg_scale > 0.:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    return logits


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


def main():
    device = 'cuda'
    
    model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-8B-Base', 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Base', 
        trust_remote_code=True
    )
    
    # Hellaswag数据集示例
    prompt = 'Roof shingle removal: A man is sitting on a roof. He'
    answer = ' is using wrap to wrap a pair of skis.'
    
    prompt = torch.tensor(tokenizer(prompt)['input_ids']).to(device)
    answer = torch.tensor(tokenizer(answer)['input_ids']).to(device)
    
    print(get_log_likelihood(model, prompt, answer, mc_num=128))


if __name__ == '__main__':
    main()
```

### 3.2 关键实现技巧

#### 3.2.1 低方差前向过程

```python
# 高方差版本（公式3）:
t ~ U[0,1]  # 随机掩码比例
# 掩码数量 ~ Binomial(L, t)  # 随机变量，方差大

# 低方差版本（公式14）:
k ~ U{1, 2, ..., L}  # 确定性的掩码数量
positions = random.sample(L, k)  # 随机选择位置
# 掩码数量 = k  # 确定性，方差小
```

**代码实现:**

```python
# 关键：每个batch使用不同的k值（覆盖整个范围）
x = torch.round(torch.linspace(
    float(k), 
    k + (b - 1) * (target_len / b), 
    steps=b, 
    device=batch.device
)).long()
x = ((x - 1) % target_len) + 1  # 确保在[1, target_len]范围内
```

**示例** (target_len=100, batch_size=4, 初始k=10):
- k值: [10, 35, 60, 85] （均匀分布在整个范围）
- 每个样本不同k，单次forward覆盖多样噪声水平

#### 3.2.2 加权损失计算

```python
loss = F.cross_entropy(logits[mask_index], seq[mask_index], reduction='none')
loss = loss / p_mask[mask_index]  # 关键：除以掩码比例！
```

**理论依据:**

变分下界（公式3/4）包含 $1/t$ 权重：

$$
\mathcal{L}(\theta) = -\mathbb{E}_{t, x_0, x_t}\left[ \frac{1}{t} \sum_{i=1}^{L} \mathbf{1}[x_t^i = \mathbf{M}] \log p_\theta(x_0^i | x_t) \right]
$$

除以 `p_mask` (即 $t$) 实现了这个权重！

---

## 4. 对话系统: `chat.py`

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

### 4.2 多轮对话关键技术

#### 4.2.1 对话历史管理

```python
if conversation_num == 0:
    prompt = input_ids
else:
    # 去掉BOS token（通常索引为0）避免序列中出现多个BOS
    prompt = torch.cat([prompt, input_ids[:, 1:]], dim=1)
```

**问题**: 为什么不直接 `cat([prompt, input_ids])`?

**解答**: 每个序列通常以BOS (Beginning of Sequence) token开始。如果直接拼接，中间会出现多余的BOS，导致模型困惑。

#### 4.2.2 EOS处理

```python
prompt = out[out != 126081].unsqueeze(0)  # 126081是EOS token ID
```

**作用**: 
- 模型生成时会输出 `<EOS>` 标记结束
- 下一轮对话需要移除这个标记，否则模型会认为是新的独立序列

---

## 5. 评估框架: `eval_llada.py`

### 5.1 架构概览

```python
@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    '''
    lm-evaluation-harness 集成类
    支持: loglikelihood评估、greedy解码验证、文本生成
    '''
    
    def __init__(self, model_path, mask_id=126336, ...):
        # 模型加载
        # 多GPU支持 (accelerate)
        # 评估参数配置
        
    def get_loglikelihood(self, prefix, target):
        # 蒙特卡洛估计对数似然
        
    def suffix_greedy_prediction(self, prefix, target):
        # 验证target是否可通过贪婪解码获得
        # 用于LAMBADA等需要验证的任务
        
    def loglikelihood(self, requests):
        # 批量处理likelihood请求
        
    def generate_until(self, requests):
        # 生成直到满足停止条件
```

### 5.2 关键实现细节

#### 5.2.1 贪婪解码验证

```python
@torch.no_grad()
def suffix_greedy_prediction(self, prefix, target):
    '''
    验证target是否可通过贪婪解码从prefix生成
    用于LAMBADA等任务（需要判断模型是否能生成正确答案）
    '''
    if not self.is_check_greedy:
        return False  # 默认跳过加速
    
    # 初始化全掩码序列
    seq = torch.full((1, len(prefix) + len(target)), self.mask_id, ...)
    seq[0, :len(prefix)] = prefix
    
    for i in range(len(target)):
        mask_index = (seq == self.mask_id)
        logits = self.get_logits(seq, prompt_index)[mask_index]
        
        # 贪婪选择：选择最高置信度的token
        x0 = torch.argmax(logits, dim=-1)
        
        # 只保留置信度最高的一个，其余重新掩码
        p = torch.softmax(logits.to(torch.float32), dim=-1)
        confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(-1)
        _, index = torch.sort(confidence, descending=True)
        x0[index[1:]] = self.mask_id  # 只保留最高置信度的一个
        
        seq[mask_index] = x0.clone()
    
    return target == seq[0, len(prefix):]
```

**算法解析:**

这是**顺序贪婪解码**（Sequential Greedy Decoding）:
1. 每步只揭示**一个**最高置信度的token
2. 其余预测的token重新掩码
3. 重复直到全部揭示

与并行生成不同，这模拟了逐步生成过程，用于判断模型是否"能"生成目标序列。

#### 5.2.2 多GPU支持

```python
accelerator = accelerate.Accelerator()
if accelerator.num_processes > 1:
    self.accelerator = accelerator
    model_kwargs = {'device_map': {'': f'{self.accelerator.device}'}}
    # ...
    self.model = self.accelerator.prepare(self.model)
```

使用Hugging Face Accelerate库实现分布式评估。

---

## 6. 可视化与交互: `app.py`

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
    
    # 约束处理
    if constraints:
        for pos, word in constraints.items():
            tokens = tokenizer.encode(" " + word, add_special_tokens=False)
            for i, token_id in enumerate(tokens):
                processed_constraints[pos + i] = token_id
    
    # 生成循环（与generate.py类似，但添加可视化记录）
    for num_block in range(num_blocks):
        for i in range(steps_per_block):
            # ... 生成逻辑 ...
            
            # 记录状态
            current_state = []
            for j in range(gen_length):
                pos = prompt_length + j
                
                if x[0, pos] == MASK_ID:
                    # 仍是掩码
                    current_state.append((MASK_TOKEN, "#444444"))
                elif old_x[0, pos] == MASK_ID:
                    # 本轮新揭示
                    token = tokenizer.decode([x[0, pos].item()])
                    confidence = float(x0_p[0, pos].cpu())
                    
                    # 根据置信度着色
                    if confidence < 0.3:
                        color = "#FF6666"  # 红色：低置信
                    elif confidence < 0.7:
                        color = "#FFAA33"  # 橙色：中置信
                    else:
                        color = "#66CC66"  # 绿色：高置信
                    
                    current_state.append((token, color))
                else:
                    # 之前已揭示
                    token = tokenizer.decode([x[0, pos].item()])
                    current_state.append((token, "#6699CC"))  # 蓝色
            
            visualization_states.append(current_state)
    
    return visualization_states, final_text
```

### 6.2 可视化设计

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

---

## 7. 训练实现: `GUIDELINES.md` 解读

### 7.1 预训练核心代码

```python
def forward_process(input_ids, eps=1e-3):
    '''
    前向加噪过程
    eps=1e-3确保t不会精确为0（数值稳定性）
    '''
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)  # t ~ U[0,1]
    
    # p_mask = (1-ε)*t + ε，确保最小掩码率
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    
    # 伯努利采样掩码位置
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, 126336, input_ids)
    
    return noisy_batch, masked_indices, p_mask


# 训练循环
input_ids = batch["input_ids"]

# 1%概率随机截断长度（用于学习变长序列）
if torch.rand(1) < 0.01:
    random_length = torch.randint(1, input_ids.shape[1] + 1, (1,))
    input_ids = input_ids[:, :random_length]

noisy_batch, masked_indices, p_mask = forward_process(input_ids)
logits = model(input_ids=noisy_batch).logits

# 关键：加权交叉熵（除以p_mask实现1/t权重）
token_loss = F.cross_entropy(
    logits[masked_indices], 
    input_ids[masked_indices], 
    reduction='none'
) / p_mask[masked_indices]

# 按序列长度归一化
loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
```

### 7.2 SFT（监督微调）修改

```python
input_ids, prompt_lengths = batch["input_ids"], batch["prompt_lengths"]

noisy_batch, _, p_mask = forward_process(input_ids)

# ========== 关键区别：不掩码prompt ==========
token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device)
token_positions = token_positions.expand(noisy_batch.size(0), noisy_batch.size(1))
prompt_mask = (token_positions < prompt_length.unsqueeze(1))
noisy_batch[prompt_mask] = input_ids[prompt_mask]

# 计算answer长度（用于归一化）
prompt_mask = prompt_mask.to(torch.int64)
answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
answer_lengths = answer_length.repeat(1, noisy_batch.shape[1])

masked_indices = (noisy_batch == 126336)
logits = model(input_ids=noisy_batch).logits

token_loss = F.cross_entropy(
    logits[masked_indices], 
    input_ids[masked_indices], 
    reduction='none'
) / p_mask[masked_indices]

# 按answer长度归一化（而非总长度）
ce_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
```

### 7.3 训练技巧总结

| 技巧 | 目的 | 实现 |
|------|------|------|
| `eps=1e-3` | 避免t=0时的数值溢出 | `p_mask = (1-ε)*t + ε` |
| 随机截断 | 学习变长生成 | 1%概率随机长度 |
| Prompt保护 | 条件生成 | SFT中不掩码prompt |
| Answer归一化 | 公平比较不同长度 | 按answer长度而非总长度归一化 |

---

## 8. 工程技巧总结

### 8.1 数值稳定性

```python
# 1. Gumbel噪声使用float64
logits = logits.to(torch.float64)

# 2. 避免log(0) in Gumbel
noise = torch.rand_like(logits, dtype=torch.float64)  # (0,1)
gumbel_noise = (- torch.log(noise)) ** temperature     # 无log(0)风险

# 3. 前向过程eps防护
p_mask = (1 - eps) * t + eps  # 最小掩码率ε
```

### 8.2 内存优化

```python
# 1. 使用bfloat16（平衡精度和速度）
model = AutoModel.from_pretrained(..., torch_dtype=torch.bfloat16)

# 2. @torch.no_grad()装饰器
@torch.no_grad()
def generate(...):
    # 禁用梯度计算，节省显存

# 3. 及时清空缓存
torch.cuda.empty_cache()
```

### 8.3 Batch处理技巧

```python
# 1. Left padding（简化采样逻辑）
tokenizer.padding_side = 'left'

# 2. Attention mask支持变长序列
attention_mask = torch.cat([prompt_mask, gen_mask], dim=-1)
logits = model(x, attention_mask=attention_mask).logits

# 3. CFG batch加倍技巧
x_ = torch.cat([x, un_x], dim=0)  # 2x batch
logits = model(x_).logits
logits, un_logits = torch.chunk(logits, 2, dim=0)
```

### 8.4 超参数设计原则

```
关键关系式:
┌────────────────────────────────────────────────────────┐
│ gen_length % block_length == 0                         │
│ steps % num_blocks == 0                                │
│                                                        │
│ 推荐配置:                                              │
│ ├─ 短生成 (64-128 tokens): steps = gen_length        │
│ ├─ 中等生成 (256-512): steps = gen_length            │
│ └─ 长生成 (1024+): steps = gen_length / 2            │
│                                                        │
│ block_length权衡:                                      │
│ ├─ = gen_length: 全并行，最快，长序列质量下降          │
│ ├─ = gen_length / 4: 平衡，推荐默认值                  │
│ └─ = 1: 完全串行，最慢，质量最好                       │
└────────────────────────────────────────────────────────┘
```

### 8.5 调试技巧

```python
# 1. 检查mask_id不冲突
assert tokenizer.pad_token_id != 126336  # mask_id

# 2. 验证steps/block整除关系
assert gen_length % block_length == 0
assert steps % num_blocks == 0

# 3. 可视化生成过程（visualization/generate.py）
# 每步输出到sample_process.txt，便于调试
```

---

## 附录: 核心公式对照表

| 代码变量 | 论文符号 | 含义 |
|----------|----------|------|
| `mask_id = 126336` | M | 掩码token |
| `p_mask` | t | 掩码比例 |
| `steps` | T | 采样步数 |
| `block_length` | - | 半自回归块大小 |
| `cfg_scale` | s | CFG缩放因子 |
| `temperature` | τ | Gumbel采样温度 |
| `remasking` | - | 重掩码策略 |
| `num_transfer_tokens` | - | 每步揭示token数 |

---

**文档版本**: 1.0  
**最后更新**: 2025年  
**基于**: LLaDA官方实现 (https://github.com/ML-GSAI/LLaDA)
