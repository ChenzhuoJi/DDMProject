# 随机过程课程项目：离散扩散（语言）模型文献调研

Spring 2026《随机过程》— 问题导向的研讨式学习

## 项目结构

```
.
├── 00_ProjectRequirement/         # 大作业要求
│   ├── *.pdf                      # 要求原文
│   ├── *.md                       # MinerU 解析
│   └── ...
├── 01_D3PM/ ~ 07_Dream7B/       # 7篇必读参考文献（大作业指定），各含：
├── 08_ColaDLM/                   # 附加论文（非大作业指定，前沿扩展），含：
│   ├── *.pdf                      # 论文原文
│   ├── *.md                       # MinerU 解析
│   ├── *.py / *.ipynb             # 官方源码（D3PM、CTMC 已有，其余可补充）
│   └── ...
├── translated/                    # pdf2zh 中英对照翻译版
│   ├── *-dual.pdf                 # 左右对照版（原文 + 中文）
│   ├── *-mono.pdf                 # 纯中文版
│   └── ...
├── Literature_Summary.md          # 8篇论文摘要汇总与脉络关系
└── README.md                      # 本文件
```

### 论文清单

| # | 简称 | 论文 | 发表 | 源码 |
|---|------|------|------|------|
| 00 | Requirement | 大作业要求文档 | — | — |
| 01 | D3PM | Structured Denoising Diffusion Models in Discrete State-Spaces | NeurIPS 2021 | ✅ `src/` |
| 02 | CTMC | A Continuous Time Framework for Discrete Denoising Models | NeurIPS 2022 | ✅ `src/` |
| 03 | UD3 | Unified Discrete Diffusion for Categorical Data | JMLR 2025 | — |
| 04 | MD4 | Simplified and Generalized Masked Diffusion for Discrete Data | NeurIPS 2024 | — |
| 05 | MDLM | Simple and Effective Masked Diffusion Language Models | NeurIPS 2024 | — |
| 06 | LLaDA | Large Language Diffusion Models | NeurIPS 2025 | — |
| 07 | Dream7B | Dream 7B: Diffusion Large Language Models | 2025 | — |
| 08 | ColaDLM ⚡ | Continuous Latent Diffusion Language Model（非必读，前沿扩展） | 2026 | — |

### 文件说明

- 每篇论文目录下：**原文 PDF** + **MinerU 解析的 Markdown** + （可选）**官方源码**
- `translated/`：所有论文（除 Cola DLM）的 pdf2zh 中英对照翻译版，保留排版与公式
- `Literature_Summary.md`：各论文摘要及技术脉络图
- `08_ColaDLM/` 为自行补充的前沿论文，**非大作业要求中的必读文献**

### 技术脉络

```
D3PM (2021) ──→ CTMC (2022) ──→ UD3 (2025) ──→ MDLM/MD4 (2024) ──→ LLaDA/Dream 7B (2025) ──→ Cola DLM (2026)
```

从离散转移矩阵 → 连续时间 CTMC → 离散/连续统一 → 简化掩码扩散 → 大规模扩散 LM → 潜空间扩散。
