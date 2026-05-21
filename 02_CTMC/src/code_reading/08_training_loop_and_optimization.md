# 08 — 训练循环与优化策略

> 训练入口 `train.py` / `dist_train.py`，训练步 `lib/training/training.py`
>
> 从启动到收敛，完整追踪 200 万次迭代的训练过程

## 目录

- [1. 训练启动与配置解析](#1-训练启动与配置解析)
- [2. 实验目录与持久化](#2-实验目录与持久化)
- [3. 组件创建：注册表模式的组装工厂](#3-组件创建注册表模式的组装工厂)
- [4. 抢占恢复机制](#4-抢占恢复机制)
- [5. 主训练循环](#5-主训练循环)
- [6. Standard 训练步详解](#6-standard-训练步详解)
- [7. Adam 优化器与 Warmup 策略](#7-adam-优化器与-warmup-策略)
- [8. 指数移动平均 (EMA)](#8-指数移动平均-ema)
- [9. 检查点系统](#9-检查点系统)
- [10. 日志与可视化](#10-日志与可视化)
- [11. 分布式训练](#11-分布式训练)
- [12. 200 万步训练的时间线](#12-200-万步训练的时间线)

---

## 1. 训练启动与配置解析

### 1.1 命令行入口

`train.py:123-135`:

```bash
python train.py cifar10
```

```python
parser = argparse.ArgumentParser()
parser.add_argument('config')
args, unknown_args = parser.parse_known_args()

if args.config == 'cifar10':
    from config.train.cifar10 import get_config
elif args.config == 'piano':
    from config.train.piano import get_config
```

配置名称 `cifar10` 通过 `importlib` 风格的条件导入，加载对应的 `get_config()` 函数。配置是一个 `ml_collections.ConfigDict` 对象，以点号访问所有超参数。

### 1.2 配置结构

`config/train/cifar10.py` 定义的整个配置树：

```
cfg
├── experiment_name       "cifar10"
├── save_location         "path/to/output"
├── init_model_path       None
├── device                "cpu"
├── distributed           False
├── num_gpus              0
│
├── loss
│   ├── name              "GenericAux"
│   ├── eps_ratio         1e-9
│   ├── nll_weight        0.001
│   ├── min_time          0.01
│   └── one_forward_pass  True
│
├── training
│   ├── train_step_name   "Standard"
│   ├── n_iters           2_000_000
│   ├── clip_grad         True
│   └── warmup            5000
│
├── data
│   ├── name              "DiscreteCIFAR10"
│   ├── root              "path/to/datasets"
│   ├── train             True
│   ├── download          True
│   ├── S                 256
│   ├── batch_size        128
│   ├── shuffle           True
│   ├── shape             [3, 32, 32]
│   └── random_flips      True
│
├── model
│   ├── name              "GaussianTargetRateImageX0PredEMA"
│   ├── ema_decay         0.9999
│   ├── ch                128
│   ├── num_res_blocks    2
│   ├── num_scales        4
│   ├── ch_mult           [1,2,2,2]
│   ├── time_embed_dim    128
│   ├── time_scale_factor 1000
│   ├── rate_sigma        6.0
│   ├── Q_sigma           512.0
│   ├── time_exponential  100.0
│   └── time_base         3.0
│
├── optimizer
│   ├── name              "Adam"
│   └── lr                2e-4
│
└── saving
    ├── enable_preemption_recovery      False
    ├── checkpoint_freq                 1000
    ├── num_checkpoints_to_keep         2
    ├── log_low_freq                    10000
    ├── low_freq_loggers                ["denoisingImages"]
    └── prepare_to_resume_after_timeout False
```

---

## 2. 实验目录与持久化

### 2.1 目录结构

`lib/utils/bookkeeping.py:15-40`:

```
save_location/
└── YYYY-MM-DD/
    └── HH-MM-SS_experiment_name/
        ├── config/
        │   └── config_001.yaml        ← 配置的 YAML 快照
        ├── checkpoints/
        │   ├── ckpt_0000000000.pt
        │   ├── ckpt_0000001000.pt
        │   ├── ...                     ← 保留最近 2 个
        │   └── archive/                ← 低频归档
        ├── tensorboard/
        │   └── events.out.tfevents.*  ← TensorBoard 日志
        └── preemption_log.txt         ← 抢占恢复日志
```

创建时机：

```python
save_dir, checkpoint_dir, config_dir = bookkeeping.create_experiment_folder(
    cfg.save_location,
    cfg.experiment_name,
    include_time=True
)
# 结果: save_dir = "output/2026-05-21/14-30-00_cifar10/"
```

### 2.2 配置持久化

```python
# bookkeeping.py:43-54
bookkeeping.save_config_as_yaml(cfg, config_dir)
# → config/config_001.yaml

# 若恢复训练时有新配置:
# → config/config_002.yaml (递增编号)
```

每次实验运行都会把当前配置保存为 YAML，便于后续评估脚本加载同一配置。

---

## 3. 组件创建：注册表模式的组装工厂

`train.py:58-78`:

```python
model = model_utils.create_model(cfg, device)         # ①
dataset = dataset_utils.get_dataset(cfg, device)       # ②
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
loss = losses_utils.get_loss(cfg)                      # ③
training_step = training_utils.get_train_step(cfg)     # ④
optimizer = optimizers_utils.get_optimizer(model.parameters(), cfg)  # ⑤
```

五个组件通过**注册表模式**统一管理。每个 `lib/*/ *_utils.py` 维护一个全局字典：

```python
# model_utils.py — 以模型为例
_MODELS = {}

def register_model(cls):
    name = cls.__name__               # 用类名作为键
    _MODELS[name] = cls
    return cls

def create_model(cfg, device, rank=None):
    return _MODELS[cfg.model.name](cfg, device, rank)
```

使用 `@register_model` 装饰器注册：

```python
@model_utils.register_model
class GaussianTargetRateImageX0PredEMA(EMA, ImageX0PredBase, GaussianTargetRate):
    ...
```

**通过 `cfg.model.name` 字符串选择实现类，完全解耦配置与代码**。

### 3.1 模型的组合构造

`GaussianTargetRateImageX0PredEMA` 的多继承初始化：

```python
def __init__(self, cfg, device, rank=None):
    EMA.__init__(self, cfg)                           # ← 初始化 EMA 参数
    ImageX0PredBase.__init__(self, cfg, device, rank) # ← 创建 UNet
    GaussianTargetRate.__init__(self, cfg, device)     # ← 预计算 R_b 特征分解
    self.init_ema()                                   # ← 复制当前参数到 shadow
```

MRO (Method Resolution Order)：

```
GaussianTargetRateImageX0PredEMA
  → EMA          ← 第一位，覆盖 nn.Module.train() 实现 EMA 切换
  → ImageX0PredBase  ← nn.Module，提供 forward()
  → GaussianTargetRate  ← 提供 rate() 和 transition()
```

`EMA` 在第一位确保其 `train()` 方法覆盖 `nn.Module.train()`。

---

## 4. 抢占恢复机制

### 4.1 信号处理

`bookkeeping.py:187-229`:

云环境（如 Google Cloud TPU）中实例可能被抢占。代码通过信号处理实现优雅退出：

```python
def signal_handler(sig, frame, checkpoint_dir, state, ...):
    # 收到 SIGTERM/SIGINT/SIGCONT 时：
    # 1. 记录信号到 preemption_log.txt
    # 2. 立即保存 checkpoint
    # 3. sys.exit(0)
    save_checkpoint(checkpoint_dir, state, num_checkpoints_to_keep)
    sys.exit(0)

def setup_preemption(save_dir, checkpoint_dir, state, ...):
    signal.signal(signal.SIGCONT, handler)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
```

### 4.2 恢复检测

`train.py:31-38`:

```python
if cfg.saving.enable_preemption_recovery:
    preempted_path = bookkeeping.check_for_preempted_run(
        cfg.save_location,
        cfg.saving.preemption_start_day_YYYYhyphenMMhyphenDD,
        cfg,
        cfg.saving.prepare_to_resume_after_timeout
    )
```

恢复检测逻辑：

```python
# bookkeeping.py:231-271
def check_for_preempted_run(save_location, start_date, cfg, ...):
    check_dir = Path(save_location).joinpath(start_date)

    for inner_run_path in reversed(sorted(glob(check_dir + '/*'))):
        # 跳过没有 preemption_log 的目录（可能是正在运行的并行任务）
        if not os.path.isfile(inner_run_path / 'preemption_log.txt'):
            continue

        # 检查日志：最后一行必须包含 SIGCONT 或 Expecting Timeout
        if not ('SIGCONT' in preemption_log[-1] or 'Expecting Timeout' in preemption_log[-1]):
            continue

        # 检查配置是否一致
        run_config = load_ml_collections(inner_run_path / 'config/config_001.yaml')
        if not run_config == cfg:
            continue

        return inner_run_path  # 找到可恢复的实验
    return Path("null")         # 没有可恢复的实验
```

### 4.3 恢复执行

```python
if preempted_path.as_posix() == 'null':
    # 全新实验
    save_dir = create_experiment_folder(...)
else:
    # 恢复实验
    save_dir = preempted_path          # 复用原目录
    checkpoint_dir = create_inner_experiment_folders(save_dir)  # 确保子目录存在

# 如果是从恢复路径启动，加载状态
if not preempted_path.as_posix() == 'null':
    state = bookkeeping.resume_training(preempted_path, state, cfg.device)
    # → 加载最近 checkpoint 的 model, optimizer, n_iter
```

---

## 5. 主训练循环

`train.py:95-118`:

```python
state = {'model': model, 'optimizer': optimizer, 'n_iter': 0}
exit_flag = False

while True:                                    # epoch 循环
    for minibatch in tqdm(dataloader):          # batch 循环
        training_step.step(state, minibatch, loss, writer)  # ← 核心

        if n_iter % checkpoint_freq == 0:       # 每 1000 步存一次
            save_checkpoint(checkpoint_dir, state, 2)

        if n_iter % log_low_freq == 0:          # 每 10000 步日志一次
            for logger in low_freq_loggers:
                logger(state, cfg, writer, minibatch, dataset)

        state['n_iter'] += 1
        if state['n_iter'] > n_iters - 1:
            exit_flag = True; break

    if exit_flag:
        break
```

**设计特点**：

- `while True` 配合 `DataLoader(shuffle=True)`：CIFAR-10 只有 50000 张图，`batch_size=128` 时每 epoch 约 391 步。2,000,000 步需要约 5115 个 epoch。每个 epoch 数据顺序被重新打乱。
- 训练步数 `n_iter` 而非 epoch 计数作为基本单位：这是扩散模型的常见做法，因为每个 minibatch 使用不同的随机时间 $t$，相当于无限的训练数据。
- `tqdm` 包装 dataloader：提供每步的进度条，包括已用时间和预估剩余时间。

### 5.1 CIFAR-10 数据循环的量化

```
数据集大小:     50,000 张
Batch size:    128
每 epoch 步数: 50,000 / 128 ≈ 391
总步数:         2,000,000
总 epoch 数:   2,000,000 / 391 ≈ 5,115
每步时间:      ≈ 0.2 秒 (GPU) / ~2 秒 (CPU)
总训练时间:    ≈ 111 小时 (GPU) / ~1100 小时 (CPU)
```

---

## 6. Standard 训练步详解

`lib/training/training.py:6-39`:

```python
class Standard():
    def __init__(self, cfg):
        self.do_ema = 'ema_decay' in cfg.model    # 检测是否使用 EMA
        self.clip_grad = cfg.training.clip_grad     # True
        self.warmup = cfg.training.warmup           # 5000
        self.lr = cfg.optimizer.lr                  # 2e-4
```

### 6.1 单步展开

每一步按时间顺序执行 8 个操作：

```
① optimizer.zero_grad()         ← 清空梯度 (0.01 ms)
        │
② l = loss.calc_loss(minibatch) ← 前向: CT-ELBO (~200 ms)
        │
③ assert not isnan(l)           ← NaN 防御
        │
④ l.backward()                  ← 反向传播 (~400 ms)
        │
⑤ clip_grad_norm_(params, 1.0)  ← 梯度裁剪 (~10 ms)
        │
⑥ warmup lr 调度                ← 前 5000 步线性增 LR (~0.01 ms)
        │
⑦ optimizer.step()              ← Adam 参数更新 (~50 ms)
        │
⑧ model.update_ema()            ← EMA shadow 更新 (~40 ms)
        │
⑨ writer.add_scalar('loss', l)  ← 日志 (~0.1 ms)
```

### 6.2 梯度裁剪

```python
if self.clip_grad:
    torch.nn.utils.clip_grad_norm_(state['model'].parameters(), 1.0)
```

将所有参数的梯度 L2 范数裁剪到 1.0。这是防止梯度爆炸的标准技术，尤其重要因为 CT-ELBO 的梯度可能因 $q_{t|0}$ 分母接近零而变得很大。

### 6.3 NaN 防御

```python
if l.isnan().any() or l.isinf().any():
    print("Loss is nan")
    assert False
```

训练早期偶尔会遇到 NaN（尤其当 $t$ 接近 `min_time=0.01` 且 $q_{t|0}$ 极度对角化时）。直接 `assert False` 终止训练而非静默继续——因为一旦 NaN 进入参数，模型无法恢复。

---

## 7. Adam 优化器与 Warmup 策略

### 7.1 优化器配置

`lib/optimizers/optimizers.py:5-7`:

```python
@optimizers_utils.register_optimizer
def Adam(params, cfg):
    return torch.optim.Adam(params, cfg.optimizer.lr)
```

使用 PyTorch 的默认 Adam 参数（$\beta_1=0.9, \beta_2=0.999, \epsilon=1\times10^{-8}$），仅覆盖学习率 `lr=2e-4`。

### 7.2 Warmup 实现

`training.py:27-29`:

```python
if self.warmup > 0:
    for g in state['optimizer'].param_groups:
        g['lr'] = self.lr * np.minimum(state['n_iter'] / self.warmup, 1.0)
```

Warmup 曲线：

```
lr
│
2e-4 ─────────────────────────────────────────
│                                               plateau
│                                      ╱
│                                   ╱
│                                ╱
│                             ╱
│                          ╱                  lr(k) = 2e-4 * min(k/5000, 1.0)
│                       ╱
│                    ╱
│                 ╱
│              ╱
│           ╱
│        ╱
│     ╱
│  ╱                   线性 warmup
│╱
└───────────────────────────────────────────→ steps
0     5000                              2,000,000
```

**为什么需要 warmup**：CT-ELBO 在训练早期的损失值非常大（因为随机初始化的模型预测接近均匀分布，导致 sig 项中 log 的参数接近 0，产生大梯度）。Warmup 防止早期大梯度破坏模型参数。

### 7.3 学习率的演变

```
阶段 1 (迭代 0 - 5,000):   lr: 0 → 2e-4   线性增长
阶段 2 (迭代 5,001 - 2,000,000): lr: 2e-4   保持不变
```

代码使用**固定学习率**而非余弦退火或分段衰减。这与扩散模型训练的经验一致——长时间以固定中等学习率训练通常优于衰减策略。

---

## 8. 指数移动平均 (EMA)

### 8.1 参数管理

`lib/models/models.py:354-438`:

EMA 管理三组参数：

```
模型当前参数:  model.parameters()      ← optimizer.step() 更新
  ↓ copy at init
EMA shadow:  ema.shadow_params       ← update_ema() 更新
  ↓ swap on eval
EMA collected: ema.collected_params  ← 训练时暂存当前参数
```

### 8.2 更新规则

```python
def update_ema(self):
    decay = self.decay                    # 0.9999
    self.num_updates += 1
    decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
    one_minus_decay = 1.0 - decay

    for s_param, param in zip(self.shadow_params, self.parameters()):
        s_param.sub_(one_minus_decay * (s_param - param))
```

等价于：

$$
\hat\theta_{k+1} = \alpha_k \cdot \hat\theta_k + (1 - \alpha_k) \cdot \theta_{k+1}
$$

其中 $\alpha_k = \min(0.9999, \frac{1+k}{10+k})$。前几步的衰减率：

| 步数 k | $\alpha_k$ |
|--------|-----------|
| 0 | 0.1000 |
| 1 | 0.1818 |
| 10 | 0.5500 |
| 100 | 0.9083 |
| 1,000 | 0.9901 |
| 10,000 | 0.9990 |
| 100,000 | 0.9999 |

这种调度称为**偏置校正**，防止 EMA 在训练初期被早期参数主导。

### 8.3 训练/评估切换

`models.py:423-438`:

```python
def train(self, mode=True):
    if mode:  # 切换回训练模式
        # 从 collected 恢复原始参数
        self.move_collected_params_to_model_params()
    else:     # 切换到评估模式
        # 保存当前参数到 collected
        self.move_model_params_to_collected_params()
        # 加载 EMA shadow 到模型参数
        self.move_shadow_params_to_model_params()
```

使用流程：

```python
# 训练
model.train()      # 正常训练参数
  ...
  optimizer.step()
  model.update_ema()  # EMA shadow 跟随
  ...

# 评估
model.eval()       # 自动替换为 EMA shadow
  samples = model(x, t)  # 使用平滑参数
  ...

# 继续训练
model.train()      # 自动恢复原始参数
```

**关键保证**：`model.train()` 连续调用两次相同模式会报错，防止意外覆盖：

```python
if self.training == mode:
    raise ValueError("Dont call model.train() with the same mode twice!")
```

---

## 9. 检查点系统

### 9.1 保存时机

`train.py:100-102`:

```python
if state['n_iter'] % cfg.saving.checkpoint_freq == 0 \
   or state['n_iter'] == cfg.training.n_iters - 1:
    bookkeeping.save_checkpoint(checkpoint_dir, state,
        cfg.saving.num_checkpoints_to_keep)
```

`checkpoint_freq = 1000`，即每 1000 步保存一次。最后一次迭代（第 1,999,999 步）也一定保存。

### 9.2 保存内容

`bookkeeping.py:74-86`:

```python
def save_checkpoint(checkpoint_dir, state, num_checkpoints_to_keep):
    state_to_save = {
        'model': state['model'].state_dict(),       # 含 EMA shadow params
        'optimizer': state['optimizer'].state_dict(), # Adam 动量
        'n_iter': state['n_iter']                     # 恢复起点
    }
    torch.save(state_to_save,
        checkpoint_dir / f'ckpt_{state["n_iter"]:010d}.pt'
    )
```

### 9.3 轮换策略

```python
all_ckpts = sorted(glob(checkpoint_dir / 'ckpt_*.pt'))
if len(all_ckpts) > num_checkpoints_to_keep:  # 2
    for i in range(len(all_ckpts) - num_checkpoints_to_keep):
        os.remove(all_ckpts[i])
```

磁盘占用估算：

```
每个 checkpoint:
  model:  ~85 MB (UNet 7.4M params × 4 bytes × ~3 倍状态)
  optimizer: ~60 MB (Adam: 参数 × 2 倍动量)
  总计: ~150 MB

保留 2 个: ~300 MB
归档: 每 200,000 步另存一个永久副本
```

### 9.4 检查点文件命名

```
checkpoints/
├── ckpt_0000000000.pt       ← 迭代 0
├── ckpt_0000001000.pt       ← 迭代 1000
├── ckpt_0000002000.pt       ← 迭代 2000 (覆盖 0)
├── ckpt_0000003000.pt       ← 迭代 3000 (覆盖 1000)
└── ...
```

`num_checkpoints_to_keep=2` 意味着始终只保留最近的两个检查点，加上最老的一个在轮换前被删除。

### 9.5 归档

```python
# dist_train.py:161-162 (仅分布式版本支持)
if state['n_iter'] % cfg.saving.checkpoint_archive_freq == 0:
    bookkeeping.save_archive_checkpoint(checkpoint_dir, state)
    # → checkpoints/archive/ckpt_0200000.pt (永久保留)
```

归档频率 `checkpoint_archive_freq = 200000`，每 20 万步生成一个不可删除的快照。

---

## 10. 日志与可视化

### 10.1 实时日志

```python
writer.add_scalar('loss', l.detach(), state['n_iter'])
# TensorBoard 记录每一步的 loss 值
```

### 10.2 低频日志

`low_freq_loggers = ['denoisingImages']`, `log_low_freq = 10000`

`lib/loggers/loggers.py:20-59`:

每 10000 步对 3 张验证图像执行一次去噪可视化：

```python
def denoisingImages(state, cfg, writer, minibatch, dataset):
    ts = [0.01, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0]  # 7 个时间点

    for img_idx in range(3):          # 选 3 张图
        for t_idx in range(len(ts)):   # 每个时间点
            # 1) 加噪: x_t ~ q_{t|0}(· | x_0)
            qt0_rows = qt0[0, minibatch[img_idx].flatten().long(), :]
            x_t = Categorical(qt0_rows).sample().view(1, 3072)

            # 2) 去噪: 模型预测 x_0
            x_0_logits = model(x_t, t)
            x_0_pred = argmax(x_0_logits, dim=4)  # (1, 3, 32, 32)

            # 3) 可视化: 上行 = x_t, 下行 = x_0_pred
            ax[2*img_idx, t_idx].imshow(x_t)          # 噪声图像
            ax[2*img_idx+1, t_idx].imshow(x_0_pred)   # 预测去噪

    writer.add_figure('denoisingImages', fig, state['n_iter'])
```

输出是一个 `6×7` 的子图网格：

```
           t=0.01   t=0.3   t=0.5   t=0.6   t=0.7   t=0.8   t=1.0
图0 x_t    [噪]    [噪]    [噪]    [噪]    [噪]    [噪]    [噪]
图0 x_0    [清]    [清]    [模]    [模]    [糊]    [糊]    [噪]
图1 x_t    ...
图1 x_0    ...
图2 x_t    ...
图2 x_0    ...
```

训练早期（~10k 步）：所有 $x_0$ 预测都模糊不清
训练中期（~500k 步）：$t \leq 0.3$ 的预测变清晰
训练后期（~1500k 步）：$t=0.7$ 的预测也开始清晰

**这是一个极其有效的调试工具**：一眼就能看出模型是否学会了去噪。

### 10.3 TensorBoard 目录

```
tensorboard/
└── events.out.tfevents.*
```

使用方式：

```bash
tensorboard --logdir save_location/YYYY-MM-DD/HH-MM-SS_cifar10/tensorboard
```

可追踪的指标：

| 指标 | 频率 | 来源 |
|------|------|------|
| `loss` | 每步 | `training.py:37` |
| `sig` | 每步 | `losses.py:234` |
| `reg` | 每步 | `losses.py:235` |
| `denoisingImages` | 每 10k 步 | `loggers.py:59` |

---

## 11. 分布式训练

### 11.1 启动方式

`dist_train.py:186-203`:

```bash
python dist_train.py cifar10
```

```python
cfg = get_config()
world_size = cfg.num_gpus                      # 从配置中读取 GPU 数量
mp.spawn(main, args=(world_size, cfg, 0), nprocs=world_size, join=True)
```

使用 `torch.multiprocessing.spawn` 启动多个进程，每个进程对应一个 GPU。

### 11.2 DDP 初始化

`dist_train.py:34-40`:

```python
def setup(rank, world_size, unique_num):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12355 + unique_num)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
```

每个进程的 `rank` 决定使用的 GPU 设备：

```python
device = torch.device(f"cuda:{rank}")
```

### 11.3 与单 GPU 训练的关键差异

| 方面 | 单 GPU (`train.py`) | 多 GPU (`dist_train.py`) |
|------|---------------------|--------------------------|
| 模型包装 | — | `DDP(model, device_ids=[rank])` |
| DataLoader | `shuffle=True` | `DistributedSampler(shuffle=True)` |
| Batch size | 128 | 128 / world_size |
| Checkpoint | 全部 rank | 仅 rank 0 |
| Logger | TensorBoard | rank 0 TensorBoard, 其余 DummyWriter |
| 初始化模型 | — | 支持 `init_model_path` 加载预训练 |
| 归档 | — | 额外支持 archive checkpoint |
| 目录创建 | rank 0 独有 | rank 0 创建 + `FileStore` 广播 |

### 11.4 FileStore 跨进程通信

`dist_train.py:54`:

```python
ddp_store = dist.FileStore(f"/tmp/tldr_ddpstore_{unique_num}")
```

rank 0 创建实验目录后，将路径写入 `FileStore`，其他 rank 从中读取：

```python
# rank 0 写入
ddp_store.set("save_dir", save_dir.as_posix())

# 其他 rank 读取
save_dir = Path(ddp_store.get("save_dir").decode("utf-8"))
```

---

## 12. 200 万步训练的时间线

### 12.1 完整训练生命周期

```python
main(cfg)
│
├── 启动阶段
│   ├── 检查抢占恢复
│   ├── 创建实验目录 (或复用)
│   ├── 创建 TensorBoard writer
│   ├── 构建模型 (UNet ~7.4M 参数)
│   ├── 加载数据集到 GPU
│   ├── 创建 DataLoader
│   ├── 初始化损失函数
│   ├── 初始化训练步
│   ├── 创建 Adam 优化器
│   ├── 设置信号处理 (SIGTERM/SIGINT/SIGCONT)
│   └── 加载 checkpoint (如果需要恢复)
│
├── 训练阶段 (2,000,000 步)
│   │
│   ├── 步 0 ~ 5,000 (Warmup)
│   │   ├── lr: 0 → 2e-4 线性增长
│   │   ├── loss: 从 ~10⁴ 下降到 ~10²
│   │   ├── EMA decay: 0.1 → 0.99+
│   │   └── 模型从随机噪声开始学习
│   │
│   ├── 步 5,000 ~ 100,000 (早期训练)
│   │   ├── lr: 固定在 2e-4
│   │   ├── loss: 继续下降到 ~10¹
│   │   ├── EMA: decay 逐渐接近 0.9999
│   │   ├── checkpoint: 每千步保存
│   │   └── 去噪可视化: t=0.01 的预测开始清晰
│   │
│   ├── 步 100,000 ~ 1,000,000 (中期训练)
│   │   ├── loss: 进一步下降到 ~1
│   │   ├── EMA: decay > 0.999
│   │   ├── sig 项: 绝对值增大 (学习信号变强)
│   │   ├── reg 项: 稳定在某个水平
│   │   └── 去噪可视化: t=0.5 开始可辨认
│   │
│   ├── 步 1,000,000 ~ 2,000,000 (后期收敛)
│   │   ├── loss: 缓慢下降到 ~0.1 量级
│   │   ├── 模型参数趋于稳定
│   │   ├── EMA shadow 显著优于原始参数
│   │   ├── 归档: 每 20 万步保留永久副本
│   │   └── 去噪可视化: t=0.7 也能恢复
│   │
│   └── 步 2,000,000 (训练完成)
│       ├── 最后 checkpoint 保存
│       ├── TensorBoard 关闭
│       └── preemption_log 标记完成
│
└── 评估阶段 (外部执行)
    ├── scripts/sample.py → 50k 图像 → FID
    └── elbo_evaluation.py → bits/dim
```

### 12.2 关键资源消耗

| 资源 | 单 GPU (B=128) | 4 GPU (B=32/GPU) |
|------|----------------|------------------|
| GPU 显存 | ~4 GB | ~2 GB/GPU |
| 每步时间 (V100) | ~0.2s | ~0.06s |
| 总训练时间 | ~111 h | ~33 h |
| 磁盘/checkpoint | ~150 MB | ~150 MB |
| 总磁盘 (含归档) | ~3 GB | ~3 GB |
