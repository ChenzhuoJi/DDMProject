# AGENTS.md — DDMProject

Spring 2026《随机过程》课程项目：离散扩散（语言）模型文献调研。

## Repo layout

```
00_ProjectRequirement/   大作业要求（PDF + MinerU 解析 MD）
01_D3PM/ ~ 08_ColaDLM/  每篇论文各一目录，含：
  ├── *.pdf               论文原文
  ├── *.md                MinerU 解析的 Markdown
  └── src/                官方源码（仅 D3PM 已添加）
translated/               pdf2zh 翻译：*-dual.pdf（中英对照）| *-mono.pdf（纯中文）
Literature_Summary.md     8篇论文摘要 + 技术脉络图
```

## Key commands

```bash
# 解析 PDF → Markdown（免登录，限 10MB/20页）
mineru-open-api flash-extract paper.pdf --pages 1-20 -o ./parsed/

# 翻译 PDF → 中英对照版
pip install pdf2zh
pdf2zh paper.pdf -o translated/

# 国内下载 HuggingFace 模型时指定镜像
HF_ENDPOINT=https://hf-mirror.com pdf2zh paper.pdf -o translated/

# Python venv 路径
/tmp/pdfvenv/bin/python3
/tmp/pdfvenv/bin/pip
```

## D3PM 源码（01_D3PM/src/）

- Google Research 官方实现（JAX + Flax）
- 子目录：`images/`（图像实验）, `text/`（文本实验）, `insertdelete/`（插入删除扩展）
- 测试：`python -m d3pm.images.main_test` / `python -m d3pm.text.main_test`
- 使用 venv 隔离安装依赖：`requirements.txt` 含 JAX 0.3.14 + TensorFlow 2.8

## Git

- 远程: `git@github.com:ChenzhuoJi/DDMProject.git`
- 根目录残留 `Dream 7B Diffusion Large Language Models.pdf` 因权限无法删除，已在 `.gitignore` 中排除
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

## Tech stack

- **PDF解析**: MinerU (`mineru-open-api flash-extract`)
- **翻译**: pdf2zh (Google 翻译后端)
- **D3PM源码**: JAX / Flax / TensorFlow
- **所有论文主题**: 离散扩散模型 → Masked Diffusion → Large Diffusion LM → 潜空间扩散
