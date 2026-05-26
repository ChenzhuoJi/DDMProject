# AGENTS.md — DDMProject

Spring 2026《随机过程》课程项目：离散扩散（语言）模型文献调研。

## Repo layout

```
00_ProjectRequirement/   大作业要求（PDF + MinerU 解析 MD）
01_D3PM/ ~ 08_ColaDLM/  每篇论文各一目录，含：
  ├── *.pdf               论文原文
  ├── *.md                MinerU 解析的 Markdown
  └── src/                官方源码
translated/               pdf2zh 翻译：*-dual.pdf（中英对照）| *-mono.pdf（纯中文）
Literature_Summary.md     8篇论文摘要 + 技术脉络图
```

## Git

- 远程: `git@github.com:ChenzhuoJi/DDMProject.git`
- **提交信息格式**：`<prefix>: <中文描述>`
  | prefix | 适用场景 |
  |--------|----------|
  | `feat` | 新增论文、源码、功能文件 |
  | `fix` | 修复错误或格式问题 |
  | `docs` | 更新文档（README、AGENTS.md、Summary、code_reading） |
  | `chore` | 仓库维护（gitignore、结构整理） |
  - 示例：`feat: 添加 Cola DLM 论文及摘要` / `docs: 更新 README 论文清单`

## 多人协作规范

### 工作流程

```bash
# 1. 开始新任务前：拉取远程最新更改
git fetch origin
git pull               # fetch + merge

# 2. 创建个人分支（可选，推荐但非强制）
git checkout -b feat/my-part  # 以具体任务命名

# 3. 提交更改
git add <文件>
git commit -m "<prefix>: <描述>"

# 4. 推送到远程（在分支上）
git push origin feat/my-part

# 5. 在 GitHub 上创建 Pull Request → 合并到 main
#    或直接推送到 main（小团队可简化）
```

### 冲突处理

```bash
# pull 时若提示冲突：
git pull
# → 编辑冲突文件，搜索 <<<<<<< / ======= / >>>>>>> 标记
# → 手动合并后：
git add <冲突文件>
git commit -m "fix: 解决 xxx 冲突"
git push
```

### 原则

- **改自己负责的目录**：每人负责 1-3 篇论文目录（如 `03_UD3/` ~ `05_MDLM/`），尽量避免同时修改同一文件
- **每次 push 前先 pull**：养成习惯，减少冲突
- **不强制使用分支**：小团队（2-3人）可直接在 main 上协作，但需频繁 pull
- **发现问题及时沟通**：避免两人同时大幅修改同一篇论文目录

## Writer Agent（报告撰稿员）

### 使用方式

撰写报告章节时，通过 Task tool 调用 writer agent，Prompt 模板如下：

```
请按 writer agent 规范撰写报告章节。Writer 规范见 .opencode/skills/report-writer/instructions.md。

【待写章节】3.X 标题
【核心公式与概念】列出需覆盖的公式和概念
【论文来源】列出该节主要参考的论文及章节
【前置知识】已写好的前一节内容摘要
【输出路径】Draft/CTMC_DDM/XX_topic.md
```

### 写作范围

Writer agent 负责撰写 `Draft/CTMC_DDM/` 下的章节草稿，当前规划：

| 文件 | 内容 | 状态 |
|------|------|------|
| `01_basic.md` | CTMC 基础知识与记号 | ✅ 完成 |
| `02_forward.md` | 前向噪声过程构建 | ✅ 完成 |
| `03_reverse.md` | 反向过程与参数化 | ⬜ 待写 |
| `04_elbo.md` | 连续时间 ELBO 推导 | ⬜ 待写 |
| `05_sampling.md` | 高效采样算法 | ✅ 完成 |
| `06_ud3.md` | UD3 统一视角 | ✅ 完成 |

### 引用规范（统一格式）

```
*Austin et al. (D3PM)*   = Austin et al., Structured Denoising Diffusion Models in Discrete State-Spaces (NeurIPS 2021)
*Campbell et al. (CTMC)* = Campbell et al., A Continuous Time Framework for Discrete Denoising Models (NeurIPS 2022)
*Zhao et al. (UD3)*      = Zhao et al., Unified Discrete Diffusion for Categorical Data (JMLR 2025)
```

引用示例：`见 *Campbell et al. (CTMC) §4.2 Proposition 3* 给出` / `这一记法沿用 *Austin et al. (D3PM) §2*。`

### 行向量左乘约定

$$\boldsymbol{x}_t \sim \text{Cat}(\boldsymbol{x}_{t-1} Q_t)$$

所有转移矩阵使用行向量左乘约定（从左到右的马尔可夫链演化）。
