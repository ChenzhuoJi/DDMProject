# 10 — 评估管线：ELBO 与 FID

> `elbo_evaluation.py` + `lib/loggers/loggers.py:117-373` + `scripts/sample.py`
>
> 训练结束后如何衡量生成质量

## 目录

- [1. 评估全景](#1-评估全景)
- [2. ELBO 评估入口](#2-elbo-评估入口)
- [3. ELBO Logger 详解](#3-elbo-logger-详解)
- [4. ELBO 与训练 CT-ELBO 的差异](#4-elbo-与训练-ct-elbo-的差异)
- [5. FID 评估管线](#5-fid-评估管线)
- [6. 结果持久化：NumpyWriter](#6-结果持久化numpywriter)
- [7. 评估配置](#7-评估配置)

---

## 1. 评估全景

训练完成后，CTMC 框架提供两种互补的质量度量：

| 度量 | 实现位置 | 衡量 | 需要数据 | 计算量 |
|------|---------|------|---------|--------|
| **ELBO** (bits/dim) | `elbo_evaluation.py` + `ELBO` logger | 似然：模型对数据的拟合程度 | 测试集 | 每张图 ~100 次前向 |
| **FID** | `scripts/sample.py` + 外部工具 | 分布距离：生成样本的视觉质量 | 测试集统计量 | 50k 生成 + Inception 特征 |

**两者互补**：ELBO 反映模型是否学到了真实数据分布，FID 反映生成的图像有多逼真。前者是概率论意义上的评估，后者是感知意义上的评估。

```
                训练目录
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
elbo_evaluation.py  │      scripts/sample.py
    │               │               │
    ▼               │               ▼
ELBO Logger        │          50k PNG
    │               │               │
    ▼               │               ▼
bits/dim            │          clean-fid
    │               │               │
    └───────────────┼───────────────┘
                    ▼
          实验目录/eval/
         YYYY-MM-DD_HH-MM-SS_IvI.../
           ├── config/config_001.yaml
           ├── git_hash.txt
           ├── neg_elbo            (numpy .txt)
           ├── full_elbos_0        (numpy .npy)
           └── (多个 eval 子目录)
```

---

## 2. ELBO 评估入口

### 2.1 执行流程

`elbo_evaluation.py:27-71`:

```bash
python elbo_evaluation.py
```

```python
def main(eval_cfg, job_id=0, task_id=0):
    # 1. 创建评估目录
    eval_folder, eval_named_folder, eval_named_folder_configs = \
        bookkeeping.setup_eval_folders(
            Path(eval_cfg.experiment_dir), eval_cfg.eval_name,
            job_id, task_id
        )
    # → 实验目录/eval/2026-05-21_15-30-00_IvICIFAR_elboIvI_0_0/

    # 2. 保存评估配置和 git hash
    bookkeeping.save_config_as_yaml(eval_cfg, eval_named_folder_configs)
    bookkeeping.save_git_hash(eval_named_folder)

    # 3. 加载训练配置
    train_cfg = bookkeeping.load_ml_collections(
        bookkeeping.get_most_recent_config(
            Path(eval_cfg.experiment_dir) / 'config'
        )
    )
    # 4. 应用评估覆盖
    for item in eval_cfg.train_config_overrides:
        utils.set_in_nested_dict(train_cfg, item[0], item[1])

    # 5. 重建模型并加载 EMA 权重
    device = torch.device(eval_cfg.device)
    model = model_utils.create_model(train_cfg, device)
    ...
    model.load_state_dict(model_state)           # 加载 checkpoint
    model.eval()                                 # 切换到 EMA 参数

    # 6. 加载测试集
    dataset = dataset_utils.get_dataset(eval_cfg, device)
    # → data.train = False (使用测试集)

    # 7. 运行 ELBO logger
    writer = bookkeeping.NumpyWriter(eval_named_folder)
    for logger_name in eval_cfg.loggers:         # ['ELBO']
        logging_func = logger_utils.get_logger(logger_name)
        logging_func(state={'model': model, 'n_iter': 1337},
                     cfg=eval_cfg, writer=writer, dataset=dataset)

    # 8. 保存结果
    writer.save_to_disk()
    return eval_named_folder
```

### 2.2 与训练共用的配置覆盖

```python
# cifar10_elbo.py
config.train_config_overrides = [
    [['device'], 'cuda'],           # 覆盖训练配置中的 device
    [['data', 'root'], cifar10_path],  # 数据路径
    [['distributed'], False]         # 单卡评估
]
```

`load_ml_collections` 从训练实验目录的 `config/config_001.yaml` 加载训练配置，然后通过 `set_in_nested_dict` 逐项覆盖。

### 2.3 模型状态加载

```python
model_state = torch.load(Path(eval_cfg.checkpoint_path),
    map_location=eval_cfg.device)['model']

if utils.is_model_state_DDP(model_state):
    model_state = utils.remove_module_from_keys(model_state)

model.load_state_dict(model_state)
model.eval()
```

`model.eval()` 触发 `EMA.train(False)` —— 将训练参数暂存到 collected，加载 EMA shadow 参数。确保评估使用的是平滑后的参数。

---

## 3. ELBO Logger 详解

`lib/loggers/loggers.py:117-373`:

### 3.1 配置参数

`config/eval/cifar10_elbo.py`:

```python
config.logging = logging = ml_collections.ConfigDict()
logging.total_N = 100       # 每张图像的蒙特卡洛样本数
logging.total_B = 10000     # 评估的测试图像总数
logging.B = 50              # 每次内批处理大小
logging.min_t = 0.01        # 最小时间
logging.eps = 1e-9          # 数值稳定常数
logging.initial_dist = 'gaussian'  # 先验分布
```

| 参数 | 值 | 含义 |
|------|-----|------|
| `total_N = 100` | 每张图像重复 100 次 MC 采样 | 降低蒙特卡洛噪声 |
| `total_B = 10000` | 评估 10000 张测试图像 | CIFAR-10 测试集 10000 张 |
| `B = 50` | 每次处理 50 张 | 显存控制 |

### 3.2 外层循环

```python
def ELBO(state, cfg, writer, dataset):
    C, H, W = cfg.data.shape            # (3, 32, 32)
    D = C * H * W                       # 3072
    S = cfg.data.S                      # 256

    total_B = cfg.logging.total_B       # 10000
    B = cfg.logging.B                   # 50
    total_N = cfg.logging.total_N       # 100

    data = dataset.data[0:total_B, ...]  # 取前 10000 张
    elbos = np.zeros((total_B // B, total_N))  # (200, 100)

    for b_repeat in range(total_B // B):       # 200 批
        x_0 = data[b_repeat*B:(b_repeat+1)*B]  # (50, 3, 32, 32)
        x_0 = x_0.view(B, D)                   # (50, 3072)

        for n_repeat in range(total_N):         # 100 次 MC
            neg_elbo = compute_one_elbo(x_0, model, cfg)
            elbos[b_repeat, n_repeat] = neg_elbo.detach().cpu().numpy()

    elbos = elbos / (3 * 32 * 32)   # → bits/dim
    writer.add_scalar('neg_elbo', np.mean(elbos), 0)
    writer.add_numpy_data('full_elbos', elbos, 0)
```

### 3.3 单次 ELBO 估计

`loggers.py:158-368`:

ELBO 的分解公式：

$$
-\text{ELBO} = \underbrace{\mathbb{E}_{x_T \sim q_{T|0}}[-\log p_T(x_T)]}_{\text{prefterm}} + \underbrace{\mathbb{E}_{t, x_t, \tilde x}[\text{reg} + \text{sig}]}_{\text{neg\_elbo}}
$$

```python
def compute_one_elbo(x_0, model, cfg):
    # ===== 前缀项: -log p_T(x_T) =====
    qT0 = model.transition(torch.tensor([1.0]))[0, :, :]  # (S, S)
    # 从 q_{1|0}(·|x_0) 采样 x_T
    x_T = Categorical(qT0[x_0.flatten().long(), :].view(B, D, S)).sample()
    logpref = initial_dist.log_prob(x_T)     # (B, D)
    prefterm = -torch.mean(torch.sum(logpref, dim=1))  # 标量

    # ===== CT-ELBO 项 (与训练相同) =====
    ts = torch.rand((B,), device=device) * 0.99 + 0.01  # t ~ U(0.01, 1.0)
    qt0 = model.transition(ts)    # (B, S, S)
    rate = model.rate(ts)         # (B, S, S)

    # 采样 x_t 和 x_tilde
    x_t = sample_xt(x_0, qt0)
    x_tilde = sample_xtilde(x_t, rate)

    # 模型前向 (两次，不使用 one_forward_pass)
    p0t_reg = softmax(model(x_t, ts), dim=2)       # reg 用 x_t
    p0t_sig = softmax(model(x_tilde, ts), dim=2)   # sig 用 x_tilde

    reg_term = compute_reg(p0t_reg, qt0, rate, x_t)
    sig_term = compute_sig(p0t_sig, qt0, rate, x_0, x_tilde)

    neg_elbo = prefterm + reg_term + sig_term
    return neg_elbo  # (B,) 或标量
```

### 3.4 前缀项详解

`loggers.py:166-174`:

```python
qT0 = model.transition(torch.tensor([1.0]))[0, :, :]  # t=1 的转移矩阵

# 从 q_{1|0}(·|x_0) 采样 x_T
qT0dist = Categorical(qT0[x_0_D.flatten().long(), :].view(B, D, S))
x_T = qT0dist.sample()

# 计算先验分布下 x_T 的 log 概率
logpref = initial_dist.log_prob(x_T)  # (B, D)
prefterm = - torch.mean(torch.sum(logpref, dim=1))
```

**物理含义**：

- $-\log p_T(x_T)$ 衡量"给定的噪声图像 $x_T$ 在先验分布下有多不可能"
- 如果 $q_{1|0}$ 恰好将数据映射到先验分布（即 $q_1$ 严格等于先验），此项为零
- 实际上 $q_1$ 只是近似先验，此项贡献少量 bits（约 0.5 bits/dim）

### 3.5 CT-ELBO 项

与训练时的 `GenericAux.calc_loss` 几乎一致，但有两个关键差异：

| 方面 | 训练 | 评估 |
|------|------|------|
| `one_forward_pass` | `True` | 未使用（两次前向） |
| 模型模式 | `train()` | `eval()` (EMA) |
| 梯度 | 计算梯度 | `torch.no_grad()` |
| 输出 | 加 NLL 辅助项 | 纯 ELBO |

### 3.6 最终归一化

```python
elbos = elbos / (3 * 32 * 32)  # → bits per dimension
```

对于 CIFAR-10，$3 \times 32 \times 32 = 3072$ 个维度。bits/dim 是语言模型 perplexity 的图像版：

- 均匀分布（每像素 256 个值）：$\log_2 256 = 8$ bits/dim
- 随机猜测：约 8 bits/dim
- 好的扩散模型：约 3-4 bits/dim
- CTMC 论文：约 3.16 bits/dim

---

## 4. ELBO 与训练 CT-ELBO 的差异

评估时的 `ELBO` logger 与训练时的 `GenericAux.calc_loss` 有相同核心——但并非完全相同。

| 差异 | 训练 (`GenericAux`) | 评估 (`ELBO` logger) |
|------|---------------------|----------------------|
| 前缀项 | 无 | ✅ 包含 $-\log p_T(x_T)$ |
| 反向前向次数 | 1 次 (`one_forward_pass`) | 2 次 (reg 用 x_t, sig 用 x_tilde) |
| 辅助 NLL | ✅ 包含 (weight=0.001) | ❌ 不包含 |
| 梯度 | 计算 | `torch.no_grad()` |
| 参数 | 原始参数 | EMA shadow |
| 重复次数 | 1 | `total_N = 100` MC 平均 |
| 数据 | 训练集 (shuffle) | 测试集 (前 10000 张) |

**为什么评估不用 one_forward_pass**：

`one_forward_pass` 在训练中是节省计算量的近似。在评估中，计算精度优先于效率，所以使用标准的两次前向计算。这也意味着评估的 ELBO 比训练损失更接近真实 ELBO（没有近似偏差）。

**为什么包含前缀项**：

完整 ELBO 的一个组成部分是 $-D_{\mathrm{KL}}(q_T \parallel p_T)$，对应于 $-\mathbb{E}_{q_{T|0}}[\log p_T(x_T) - \log q_{T|0}(x_T|x_0)]$。其中 $-\log q_{T|0}$ 项常数可忽略，$-\log p_T$ 即 `prefterm`。训练中这项可以被吸收到常数 $C$ 中，但评估时需要显式计算。

**80 万次前向的构成**：

```
10000 张图像 × 100 次 MC × 2 次前向 = 2,000,000 次 UNet 前向
每次前向: ~14 GFLOPS
总计算量: ~28 PFLOPS → 单 GPU 约 1 小时
```

---

## 5. FID 评估管线

### 5.1 采样入口

`scripts/sample.py` 是独立的采样脚本，与 ELBO 评估完全分离。

```bash
python scripts/sample.py
```

需要修改脚本中的路径：

```python
save_samples_path = 'path/to/tauLDR_samples'   # 输出目录
eval_cfg = get_eval_config()                     # 从 config/eval/cifar10.py 加载
```

### 5.2 采样参数

`config/eval/cifar10.py` (与 ELBO 不同的配置):

```python
config.sampler.name = 'PCTauLeaping'
config.sampler.num_steps = 500
config.sampler.min_t = 0.01
config.sampler.initial_dist = 'gaussian'
config.sampler.num_corrector_steps = 10
config.sampler.corrector_step_size_multiplier = 1.5
config.sampler.corrector_entry_time = 0.1
```

采样器使用 PCTauLeaping（默认）而非 TauLeaping，每张图像约 960 次前向。

### 5.3 采样循环

```python
total_samples = 0
batch = 50
sampler = sampling_utils.get_sampler(eval_cfg)

while total_samples < 50000:
    samples, _, _ = sampler.sample(model, batch, 1)
    # samples: (50, 3072), int, [0, 255]

    samples = samples.reshape(batch, 3, 32, 32)
    samples_uint8 = samples.astype(np.uint8)

    for i in range(batch):
        img = Image.fromarray(imgtrans(samples_uint8[i]))
        img.save(f'{save_samples_path}/{total_samples + i}.png')

    total_samples += batch
```

每次循环 50 张，共 1000 次循环，产生 50000 张 PNG 图像。

### 5.4 FID 计算

生成 50000 张图像后，使用标准工具计算 FID：

```bash
# 方案 1: clean-fid (推荐)
pip install clean-fid
python -c "from cleanfid import fid; fid.compute_fid('generated/path', dataset_name='cifar10')"

# 方案 2: pytorch-fid
python -m pytorch_fid path/to/real_cifar10_statistics.npz path/to/generated_images/
```

FID 基于 Inception-v3 特征空间的 Fréchet 距离：

$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})
$$

CTMC 论文报告的 CIFAR-10 结果：

| 采样器 | FID |
|--------|-----|
| TauLeaping (500步) | 约 8.0 |
| PCTauLeaping (500步, 10校正) | 约 5.0 |
| PCTauLeaping (1000步, 10校正) | 约 4.5 |

---

## 6. 结果持久化：NumpyWriter

ELBO 评估不使用 TensorBoard，而是用 `NumpyWriter` 将结果保存为 numpy 格式。

`lib/utils/bookkeeping.py:136-168`:

```python
class NumpyWriter():
    def __init__(self, save_dir):
        self.scalar_data = {}    # 标量记录
        self.figure_data = []    # 图片记录
        self.numpy_data = []     # numpy 数组记录

    def add_scalar(self, name, value, idx):
        # → 保存为文本: 如 neg_elbo 文件, 每行 [idx, value]

    def add_numpy_data(self, name, nparray, idx):
        # → 保存为 .npy: 如 full_elbos_0.npy

    def save_to_disk(self):
        # 遍历所有记录，写入 eval 目录
```

评估目录结构：

```
experiment_dir/eval/
└── 2026-05-21_15-30-00_IvICIFAR_elboIvI_0_0/
    ├── config/config_001.yaml
    ├── git_hash.txt
    ├── neg_elbo              # 文本文件: 0  3.16
    └── full_elbos_0.npy      # (10000/50, 100) 所有原始 ELBO 值
```

与 TensorBoard 的对比：

| | TensorBoard (训练) | NumpyWriter (评估) |
|--|-------------------|-------------------|
| 格式 | `events.out.tfevents.*` | `.txt` + `.npy` |
| 查看方式 | `tensorboard --logdir ...` | 直接 `numpy.load` 或 `cat` |
| 可移植性 | 需要 TensorBoard | 纯文本 + numpy |
| 适用场景 | 实时监控 | 离线评估结果 |

---

## 7. 评估配置

### 7.1 ELBO 配置 (`config/eval/cifar10_elbo.py`)

| 字段 | 值 | 含义 |
|------|-----|------|
| `eval_name` | `'CIFAR_elbo'` | 评估运行名称 |
| `loggers` | `['ELBO']` | 要运行的 logger 列表 |
| `experiment_dir` | `'path/to/...'` | 训练实验目录 |
| `checkpoint_path` | `'.../ckpt_0001999999.pt'` | 模型检查点 |
| `device` | `'cuda'` | 评估设备 |
| `logging.total_N` | `100` | MC 重复次数 |
| `logging.total_B` | `10000` | 测试图像数 |
| `logging.B` | `50` | batch size |
| `logging.min_t` | `0.01` | 最小时间 |
| `logging.initial_dist` | `'gaussian'` | 先验分布 |

### 7.2 采样配置 (`config/eval/cifar10.py`)

| 字段 | 值 | 含义 |
|------|-----|------|
| `sampler.name` | `'PCTauLeaping'` | 采样器类型 |
| `sampler.num_steps` | `500` | 时间步数 |
| `sampler.initial_dist` | `'gaussian'` | 初始分布 |
| `sampler.num_corrector_steps` | `10` | 每步校正次数 |
| `sampler.corrector_entry_time` | `0.1` | 校正起始时间 |

### 7.3 两套配置独立设计

```
ELBO 评估:                         采样评估:
  ┌──────────────────────┐          ┌──────────────────────┐
  │ ELBO 评估需要测试集    │          │ 采样评估不需要数据    │
  │ 小心地高保真 ELBO     │          │ 大批量生成 50k 图像   │
  │ 不关心采样质量        │          │ 不关心似然值          │
  │ 使用 TauLeaping      │          │ 使用 PCTauLeaping     │
  │ 8-16 batch, MC 重复  │          │ 50 batch, 无重复      │
  └──────────────────────┘          └──────────────────────┘
```

两者使用相同的模型权重，但评估维度和计算策略完全不同。
