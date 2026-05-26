# MDLM 实验复现指南 —— 逐命令执行手册

> 基于 **RTX PRO 6000 (Blackwell, 96GB)** + PyTorch 2.7 / CUDA 12.8，成果 1-5（轻量级复现）
> 参考代码库：https://github.com/kuleshov-group/mdlm

---

## 目录

1. [环境搭建](#1-环境搭建)
2. [成果一：HF 预训练权重推理采样](#2-成果一hf-预训练权重推理采样)
3. [成果二：扩散过程可视化](#3-成果二扩散过程可视化)
4. [成果三：跨数据集 Zero-shot 评估](#4-成果三跨数据集-zero-shot-评估)
5. [成果四：采样器速度/质量对比](#5-成果四采样器速度质量对比)
6. [成果五：text8 小规模训练](#6-成果五text8-小规模训练)
7. [附录：常见问题](#7-附录常见问题)

---

## 1. 环境搭建

> ⚠️ 本指南针对 RTX PRO 6000（Blackwell, 96GB）编写。
> 容器已预装 PyTorch 2.7.0 + CUDA 12.8 + Python 3.12，**无需重复安装 PyTorch**。

### 1.1 前置检查

SSH 登录后，先确认 GPU 和 PyTorch：

```bash
# 检查 GPU 型号、驱动版本
nvidia-smi

# 检查容器内 PyTorch 状态
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

预期输出：`PyTorch 2.7.0, CUDA: True, GPU: RTX PRO 6000`

### 1.2 克隆代码库

```bash
cd ~
git clone https://github.com/kuleshov-group/mdlm.git
cd mdlm
```

### 1.3 安装 Python 依赖

创建一个 pip-only requirements 文件 `requirements-rtx6000.yaml`（已准备好在 `src/` 目录下），或直接 pip 安装：

```bash
# 方式 A：用准备好的 yaml（推荐）
conda env create -f requirements-rtx6000.yaml
conda activate mdlm

# 或方式 B：直接用 pip（如果不想用 conda 环境）
pip install \
  'lightning>=2.5.0' \
  hydra-core==1.3.2 \
  omegaconf==2.3.0 \
  'transformers>=4.45.0' \
  datasets==2.18.0 \
  einops==0.7.0 \
  fsspec==2024.2.0 \
  packaging==23.2 \
  rich==13.7.1 \
  h5py==3.10.0 \
  pandas==2.2.1 \
  scikit-learn==1.4.0 \
  seaborn==0.13.2 \
  ipdb==0.13.13 \
  nvitop==1.3.2 \
  timm==0.9.16 \
  git-lfs==1.6 \
  'wandb>=0.18.0'
```

> 注意：`lightning>=2.5.0` 和 `transformers>=4.45.0` 是为了兼容 PyTorch 2.7 + Python 3.12。

### 1.4 编译安装 flash-attn

Blackwell + CUDA 12.8 需要用最新版 flash-attn：

```bash
pip install flash-attn --no-build-isolation
```

编译耗时约 10-30 分钟。如果失败，可临时跳过（MDLM 的 DiT 在 A100/Blackwell 上用 PyTorch SDP 也很快）。

### 1.5 验证安装

```bash
python -c "
import torch, lightning, hydra, transformers, datasets
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'Lightning {lightning.__version__}, Transformers {transformers.__version__}')
try:
  import flash_attn
  print('flash-attn: OK')
except:
  print('flash-attn: skipped')
"
```

### 1.6 应用可视化补丁

代码已内置 snapshot 可视化功能。在你的服务器上，进入代码目录并应用补丁：

> 补丁文件在 `src/mdlm_snapshot_save.patch`

```bash
cd ~/mdlm
# 将补丁传到服务器（scp 或 wget），然后：
git apply /path/to/mdlm_snapshot_save.patch
```

如提示冲突，手动将 `diffusion.py` 和 `main.py` 按 patch 内容修改（仅涉及 snapshot 保存功能）。

### 1.7 创建必要目录

```bash
mkdir -p outputs watch_folder
```

---

## 2. 成果一：HF 预训练权重推理采样

### 2.1 命令

直接用作者上传到 HuggingFace 的预训练权重采样，无需本地训练：

```bash
# 激活环境（如果已激活则跳过）
conda activate mdlm

# 进入代码目录
cd ~/mdlm

# 生成 10 个 batch 的样本（每个 batch 大小受 GPU 显存限制）
python main.py \
  mode=sample_eval \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  data=openwebtext-split \
  model.length=1024 \
  sampling.predictor=ddpm_cache \
  sampling.steps=1000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=10 \
  backbone=hf_dit
```

### 2.2 预期输出

```
Text samples: ['...', '...', ...]  # 10 段生成的文本
Generative perplexity: tensor(X.XXXX)  # GPT-2 Large 评估的生成困惑度
```

### 2.3 第一次下载模型

首次运行会自动从 HuggingFace 下载权重（约 2GB），取决于网络速度，可能需要数分钟。

### 2.4 尝试不同采样器

将 `sampling.predictor` 替换为 `ddpm` 或 `analytic`：

```bash
# 标准 DDPM 采样器（较慢）
python main.py \
  mode=sample_eval \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  data=openwebtext-split \
  model.length=1024 \
  sampling.predictor=ddpm \
  sampling.steps=1000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=2 \
  backbone=hf_dit

# Analytic 采样器（SEDD 式）
python main.py \
  mode=sample_eval \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  data=openwebtext-split \
  model.length=1024 \
  sampling.predictor=analytic \
  sampling.steps=1000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=2 \
  backbone=hf_dit
```

---

## 3. 成果二：扩散过程可视化

### 3.1 开启可视化开关

通过 `sampling.visualize=True` 开启。快照会自动保存到 `snapshots.txt`，也可通过 `sampling.visualize_save_path` 指定路径：

```bash
# 默认保存到 snapshots.txt
python main.py \
  mode=sample_eval \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  data=openwebtext-split \
  model.length=256 \
  sampling.predictor=ddpm_cache \
  sampling.steps=1000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=1 \
  sampling.visualize=True \
  backbone=hf_dit

# 指定自定义保存路径
python main.py \
  mode=sample_eval \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  data=openwebtext-split \
  model.length=256 \
  sampling.predictor=ddpm_cache \
  sampling.steps=1000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=1 \
  sampling.visualize=True \
  sampling.visualize_save_path=outputs/snapshots/diffusion_process.txt \
  backbone=hf_dit
```

### 3.2 预期输出

会打印类似下面的过程：

```
======================================================================
Diffusion Reverse Process (t=1.0 → t=0.0)
======================================================================
t=1.000 (  0% done): [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] ...
t=0.900 ( 10% done): the [MASK] [MASK] of [MASK] [MASK] [MASK] and ...
t=0.800 ( 20% done): the [MASK] [MASK] of [MASK] language [MASK] ...
...
t=0.200 ( 80% done): This is the power of a language model for ...
t=0.050 ( 95% done): This is the power of a language model for ...
t=0.000 (100% done): This is the power of a language model for ...
======================================================================
```

### 3.3 多对比展示（可选）

```bash
# 不同 seed → 不同的去噪顺序 → 展示非自回归特性
python main.py ... seed=42 sampling.visualize=True
python main.py ... seed=123 sampling.visualize=True
python main.py ... seed=999 sampling.visualize=True

# 不同采样步数对比
python main.py ... sampling.steps=128 sampling.visualize=True
python main.py ... sampling.steps=500 sampling.visualize=True
python main.py ... sampling.steps=1000 sampling.visualize=True
```

> **注意**：`model.length` 设为 256 以加快输出读取；若需展示长文本，改为 512 或 1024。

---

## 4. 成果三：跨数据集 Zero-shot 评估

### 4.1 全部 7 个数据集评估命令

用作者预训练的 OWT 权重，在 7 个数据集上评估 perplexity：

```bash
# 依次评估每个数据集
# 注意：每个命令会下载对应的数据集（首次需要网络）

# 1. PTB (Penn Treebank)
python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=ptb \
  model=small \
  parameterization=subs \
  backbone=hf_dit \
  model.length=1024 \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  +wandb.offline=true

# 2. Wikitext-2
python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=wikitext2 \
  model=small \
  parameterization=subs \
  backbone=hf_dit \
  model.length=1024 \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  +wandb.offline=true

# 3. LM1B
python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=lm1b-gpt2 \
  model=small \
  parameterization=subs \
  backbone=hf_dit \
  model.length=1024 \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  +wandb.offline=true

# 4. Lambada
python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=lambada \
  model=small \
  parameterization=subs \
  backbone=hf_dit \
  model.length=1024 \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  +wandb.offline=true

# 5. AG News
python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=ag_news \
  model=small \
  parameterization=subs \
  backbone=hf_dit \
  model.length=1024 \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  +wandb.offline=true

# 6. Pubmed
python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=scientific_papers_pubmed \
  model=small \
  parameterization=subs \
  backbone=hf_dit \
  model.length=1024 \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  +wandb.offline=true

# 7. Arxiv
python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=scientific_papers_arxiv \
  model=small \
  parameterization=subs \
  backbone=hf_dit \
  model.length=1024 \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  +wandb.offline=true
```

### 4.2 预期结果对照表

| 数据集 | 论文值 | 你的结果 | 备注 |
|--------|:------:|:--------:|------|
| PTB | 95.26 | | 首个跑，耗时较长 |
| Wikitext-2 | 32.83 | | |
| LM1B | 67.01 | | |
| Lambada | 47.52 | | |
| AG News | 61.15 | | |
| Pubmed | 41.89 | | |
| Arxiv | 37.37 | | |

### 4.3 批量运行脚本（可选）

可以将上述 7 个命令写入一个 shell 脚本 `/tmp/eval_all.sh`，一次性提交：

```bash
#!/bin/bash
# eval_all.sh — 在 SSH 服务器上逐数据集评估
conda activate mdlm
cd ~/mdlm

DATASETS=(
  "ptb"
  "wikitext2"
  "lm1b-gpt2"
  "lambada"
  "ag_news"
  "scientific_papers_pubmed"
  "scientific_papers_arxiv"
)

for ds in "${DATASETS[@]}"; do
  echo "===== Evaluating $ds ====="
  python main.py \
    mode=ppl_eval \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    data=$ds \
    model=small \
    parameterization=subs \
    backbone=hf_dit \
    model.length=1024 \
    eval.checkpoint_path=kuleshov-group/mdlm-owt \
    +wandb.offline=true
  echo "===== Done $ds ====="
done
```

运行方式：

```bash
bash /tmp/eval_all.sh 2>&1 | tee eval_results.log
```

最后从日志中提取每个数据集的 PPL 值。

---

## 5. 成果四：采样器速度/质量对比

### 5.1 三组采样器 × 多种步数

```bash
# 快速大致对比 (steps=128)
for predictor in ddpm_cache ddpm analytic; do
  echo "===== predictor=$predictor, steps=128 ====="
  time python main.py \
    mode=sample_eval \
    eval.checkpoint_path=kuleshov-group/mdlm-owt \
    data=openwebtext-split \
    model.length=1024 \
    sampling.predictor=$predictor \
    sampling.steps=128 \
    loader.eval_batch_size=1 \
    sampling.num_sample_batches=1 \
    backbone=hf_dit
done
```

### 5.2 详细对比（含多组步数）

```bash
# 分别测试 128 / 500 / 1000 / 5000 步
for steps in 128 500 1000 5000; do
  for predictor in ddpm_cache ddpm analytic; do
    echo "===== predictor=$predictor, steps=$steps ====="
    # 用 time 命令计时
    TIMEFORMAT='Wall time: %R seconds'
    time python main.py \
      mode=sample_eval \
      eval.checkpoint_path=kuleshov-group/mdlm-owt \
      data=openwebtext-split \
      model.length=1024 \
      sampling.predictor=$predictor \
      sampling.steps=$steps \
      loader.eval_batch_size=1 \
      sampling.num_sample_batches=1 \
      backbone=hf_dit
  done
done
```

### 5.3 记录结果表格

将输出中的时间 + generative perplexity 填入下表：

| 采样器 | steps=128 | steps=500 | steps=1000 | steps=5000 |
|--------|:---------:|:---------:|:----------:|:----------:|
| **ddpm_cache** | (s) / PPL | (s) / PPL | (s) / PPL | (s) / PPL |
| **ddpm** | (s) / PPL | (s) / PPL | (s) / PPL | (s) / PPL |
| **analytic** | (s) / PPL | (s) / PPL | (s) / PPL | (s) / PPL |

---

## 6. 成果五：text8 小规模训练

### 6.1 训练命令

```bash
python main.py \
  model=tiny \
  data=text8 \
  parameterization=subs \
  model.length=256 \
  trainer.max_steps=50000 \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  +wandb.offline=true
```

### 6.2 训练过程中的监控

训练时会输出类似：

```
Epoch 0:  5%|████▊                              | 2500/50000 [02:30<47:30, 16.67it/s, loss=3.45, v_num=...]
Epoch 0: 10%|████████▌                           | 5000/50000 [05:00<45:00, 16.67it/s, loss=2.89, v_num=...]
...
```

### 6.3 调整 batch size（如果 OOM）

如果 CUDA OOM，减小 batch size 并增加梯度累积：

```bash
python main.py \
  model=tiny \
  data=text8 \
  parameterization=subs \
  model.length=256 \
  trainer.max_steps=50000 \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  +wandb.offline=true
```

注意：当 `loader.batch_size * num_gpus < loader.global_batch_size` 时，Lightning 自动做梯度累积。默认 `global_batch_size=512`，单卡 batch_size=32 时累积 16 步。

### 6.4 训练完成后采样

训练结束后 checkpoints 保存在 `outputs/text8/` 下（具体路径看输出日志）。找到最新的 `.ckpt` 文件后：

```bash
# 列出 checkpoints
ls -la outputs/text8/*/checkpoints/*.ckpt

# 用最佳 checkpoint 采样
python main.py \
  mode=sample_eval \
  eval.checkpoint_path=outputs/text8/2026.05.25/XXXXXX/checkpoints/last.ckpt \
  data=text8 \
  model=tiny \
  parameterization=subs \
  model.length=256 \
  sampling.predictor=ddpm_cache \
  sampling.steps=1000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=5 \
  backbone=dit
```

> **注意**：训练模型时 `backbone=dit`（默认），采样时也要用 `backbone=dit`（不是 `hf_dit`）。

### 6.5 text8 训练预期耗时

| GPU | batch_size | 预估时间（50K steps） |
|-----|:----------:|:-------------------:|
| RTX 5090 (32GB) | 32 | ~45 分钟 |
| RTX 5090 (32GB) | 16 | ~90 分钟 |

---

## 7. 附录：常见问题

### Q1: CUDA out of memory

降低 `loader.batch_size` 和 `loader.eval_batch_size`。对于 32GB 5090：

| model.length | batch_size=32 | batch_size=16 | batch_size=8 |
|:------------:|:-------------:|:-------------:|:------------:|
| 256 | ✅ 安全 | ✅ 安全 | ✅ 安全 |
| 512 | ⚠️ 注意 | ✅ 安全 | ✅ 安全 |
| 1024 | ❌ 可能 OOM | ⚠️ 注意 | ✅ 安全 |

### Q2: datasets 下载慢

HuggingFace 数据集首次下载需要网络。可预先下载：

```bash
# 预先下载 text8 数据集
python -c "from datasets import load_dataset; load_dataset('text8')"

# 预先下载 OpenWebText
python -c "from datasets import load_dataset; load_dataset('openwebtext', split='train', streaming=True)"
```

### Q3: wandb 报错

已使用 `+wandb.offline=true` 禁用 wandb 上传。如果仍然报错：

```bash
wandb offline
```

### Q4: hydra 路径错误

如果运行时报错 `cfg not found` 或路径问题，确保在 `~/mdlm` 目录下运行（即 `main.py` 同级目录）。

### Q5: flash-attn 编译失败 on Blackwell (5090)

```bash
# 尝试不使用 flash-attn（性能下降但可运行）
pip uninstall flash-attn -y
# 或安装开发版
pip install flash-attn --no-build-isolation --no-cache-dir
```

---

## 执行总清单

| # | 步骤 | 预计耗时 | 完成状态 |
|:-:|------|:--------:|:-------:|
| 1 | SSH 登录 + 检查 GPU | 5min | ☐ |
| 2 | conda create + 安装依赖 | 30-60min | ☐ |
| 3 | 下载 HF 权重 + 推理采样 | 10min | ☐ |
| 4 | 扩散过程可视化 | 5min | ☐ |
| 5 | 跨数据集评估（7个） | 1-2h | ☐ |
| 6 | 采样器速度对比 | 30-60min | ☐ |
| 7 | text8 小规模训练 | 45-90min | ☐ |

**当你在服务器上执行完每一步后，把终端输出粘贴给我，我来分析结果并给出下一步指令。**
