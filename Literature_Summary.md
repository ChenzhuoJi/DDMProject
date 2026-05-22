# 离散扩散模型文献调研 — 论文摘要汇总

> Spring 2026《随机过程》课程项目

---

## 1. Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM)

**作者**: Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow & Rianne van den Berg (Google Research)

**会议**: NeurIPS 2021

**摘要**: 将连续空间中的 DDPM 推广到离散数据，提出 **D3PM**。核心贡献在于设计了多种结构化转移矩阵（均匀/离散化高斯/吸收态[MASK]），而不仅是之前简单的均匀转移。吸收态矩阵建立了扩散模型与自回归/掩码生成模型的联系。引入辅助交叉熵损失稳定训练。在 CIFAR-10 上接近连续 DDPM 的样本质量并超越其 log-likelihood，在 LM1B 字符级文本生成上取得强结果。

---

## 2. A Continuous Time Framework for Discrete Denoising Models

**作者**: Andrew Campbell, Joe Benton, Valentin De Bortoli, Tom Rainforth, George Deligiannidis, Arnaud Doucet (Oxford & ENS Ulm)

**会议**: NeurIPS 2022

**摘要**: 首个完整的**连续时间**离散扩散框架。将前向噪声过程和反向生成过程建模为**连续时间马尔可夫链 (CTMC)**，用连续时间 ELBO 高效训练。利用化学物理中的 tau-leaping 技术模拟高维 CTMC，并提出 predictor-corrector 采样器，在 CIFAR-10 上超越离散时间方法的样本质量。此外导出了生成分布与真实数据分布之间误差的理论界。

---

## 3. Unified Discrete Diffusion for Categorical Data (UD3)

**作者**: Lingxiao Zhao, Xueying Ding, Lijun Yu, Leman Akoglu (CMU)

**会议**: JMLR 2025

**摘要**: 统一了**离散时间**与**连续时间**离散扩散框架。通过对 VLB 进行简化和泛化，使得连续时间扩散可以使用离散时间中精确高效的反向采样，避免昂贵近似；离散时间扩散也可以使用原属连续时间的 MCMC corrector。核心创新包括：简化的封闭形式后验概率、统一的 forward/backward 过程、双向收益的框架整合。

---

## 4. Simplified and Generalized Masked Diffusion for Discrete Data (MD4)

**作者**: Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, Michalis K. Titsias (Google DeepMind)

**会议**: NeurIPS 2024

**摘要**: 大幅简化了 masked diffusion 的公式和训练目标。证明连续时间 masked diffusion 的变分目标本质上是一个**加权的交叉熵积分**。支持状态相关的掩码调度（state-dependent masking schedules）。在 OpenWebText 上以 GPT-2 规模训练，困惑度超越此前所有扩散语言模型；在像素级图像建模上达到 2.75 (CIFAR-10) 和 3.40 (ImageNet 64×64) bits/dim，超越同规模自回归模型。

---

## 5. Simple and Effective Masked Diffusion Language Models (MDLM)

**作者**: Subham Sekhar Sahoo, Aaron Gokaslan, Marianne Arriola, Edgar Marroquin, Yair Schiff, Justin T Chiu, Alexander Rush, Volodymyr Kuleshov (Cornell Tech)

**会议**: NeurIPS 2024

**摘要**: 证明简单的 masked 离散扩散比此前认为的更有效。提出一个**Rao-Blackwellized 连续时间 ELBO**，其形式简洁——是经典掩码语言模型 (MLM) 损失的加权混合。可以用 encoder-only 模型（如 BERT 风格）实现原则性的生成能力。设计了支持**半自回归 (SAR)** 生成的高效采样器。在 LM1B、OWT 等语言建模基准上达到扩散模型 SOTA，困惑度接近自回归模型的 85%。

---

## 6. Large Language Diffusion Models (LLaDA)

**作者**: Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, Chongxuan Li (中国人民大学)

**会议**: NeurIPS 2025

**摘要**: 挑战"LLM 的核心能力必须依赖自回归模型"这一共识。提出 **LLaDA**，一个从零训练的 8B 参数扩散语言模型，采用前向掩码 + 反向生成范式，优化似然下界。在 MMLU、GSM8K 等广泛基准上与 LLaMA3 8B 表现相当。SFT 后展示出多轮对话等指令跟随能力。特别地，LLaDA 克服了**反转诅咒 (reversal curse)**，在反转诗歌补全任务上超越 GPT-4o。

---

## 7. Dream 7B: Diffusion Large Language Models

**作者**: Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, Lingpeng Kong (HKU & Huawei)

**时间**: 2025

**摘要**: **目前最强的开源扩散大语言模型**（7B 参数）。通过 AR-LLM 初始化 + 上下文自适应 token 级噪声重调度等训练技术，在通用、数学、编程任务上持续超越所有现有扩散语言模型。展示了扩散范式的独特优势：**任意顺序生成、文本填充 (infilling)、可调节的质量-速度权衡**，以及更强的规划能力。

---

## 8. Continuous Latent Diffusion Language Model (Cola DLM)

**作者**: Hongcan Guo, Qinyu Zhao, Yian Zhao, Shen Nie, Rui Zhu, Qiushan Guo, Feng Wang, Tao Yang, Hengshuang Zhao, Guoqiang Wei, Yan Zeng (ByteDance Seed, HKU, ANU, PKU, 中国人民大学)

**时间**: 2026.5

**摘要**: 提出一种**层次化潜变量扩散语言模型**，将文本生成分解为两个层次：(1) **全局语义组织** — 通过 Text VAE 将文本压缩到连续潜空间，用 block-causal DiT 建模语义先验；(2) **局部文本实现** — 通过条件解码器生成 token。扩散过程不再做 token 级观测恢复，而是做**潜先验传输 (latent prior transport)**。从统一 Markov 路径视角，它比离散扩散更适合表达全局语义结构，且天然支持任意顺序生成。在 ~2B 参数下与自回归和 LLaDA 严格对比，验证了 strong scaling behavior。此外还自然扩展到图像等多模态。

**关键见解**：生成质量与 scaling behavior 比 perplexity 更能反映模型能力（likelihood-oriented 评估与生成质量存在结构性偏差）。

---

## 脉络关系（更新）

```
D3PM (NeurIPS 2021)
  └── 提出离散扩散基础框架，引入多种转移矩阵（含[MASK]吸收态）
       │
CTMC Framework (NeurIPS 2022)
  └── 将离散扩散推广到连续时间 CTMC，tau-leaping 高效采样
       │
UD3 (JMLR 2025)
  └── 统一离散时间与连续时间框架，简化 VLB 与反向采样
       │
MD4 / MDLM (NeurIPS 2024) ──→ LLaDA / Dream 7B (2025) ──→ Cola DLM (2026)
  └── 简化 masked diffusion         └── 7-8B 参数级别              └── 潜空间扩散
      ELBO = 加权交叉熵               挑战自回归范式                    层次化语义建模
                                                                        任意顺序生成
```
