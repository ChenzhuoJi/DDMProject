# LaTeX 代码规范

团队协作撰写报告时，统一的代码风格能减少引用冲突、提高可维护性。以下规范供全体成员遵守。

---

## 1 参考文献

### 1.1 BibTeX 管理

所有参考文献条目统一放入 `reference.bib`，**不用**在正文中手动拼写参考文献信息。

```bib
@inproceedings{austin2021d3pm,
  author    = {Jacob Austin and Daniel D. Johnson and Jonathan Ho and Daniel Tarlow and Rianne van den Berg},
  title     = {Structured Denoising Diffusion Models in Discrete State-Spaces},
  booktitle = {Advances in Neural Information Processing Systems 34 (NeurIPS 2021)},
  pages     = {17981--17993},
  year      = {2021},
}
```

### 1.2 引用命令

```latex
% 基本引用（无页码/章节）
\cite{austin2021d3pm}

% 带章节或页码
\cite[§4.2 Proposition 3]{campbell2022ctmc}
\cite[p.~12]{zhao2025ud3}

% 多个引用
\cite{austin2021d3pm, campbell2022ctmc}
```

### 1.3 综述引用原则

综述性内容（如"在离散扩散模型中……"）只需引用论文本身，**不需要**精确到章节。仅当引用某篇论文的特定公式、命题或定理时，才使用可选的章节参数。

```latex
% 综述引用（正确）
离散扩散模型将连续空间中的 DDPM 推广到离散数据 \cite{austin2021d3pm}。

% 特定引用（正确）
反向过程的速率矩阵由 \cite[§3.1 Proposition 1]{campbell2022ctmc} 给出。

% 错误
离散扩散模型将连续空间中的 DDPM 推广到离散数据 \cite[§2]{austin2021d3pm}。
（综述内容不需要具体到章节）
```

---

## 2 公式环境

### 2.1 行间公式

**必须**使用 `\begin{equation}` 环境，**不**使用 `$$ ... $$`。

```latex
% 正确
\begin{equation}
\frac{d}{dt} q_t = q_t R_t. \label{eq:kolmogorov-fwd-mat}
\end{equation}

% 错误
$$
\frac{d}{dt} q_t = q_t R_t.
$$

% 正确（多行公式）
\begin{equation}
\begin{aligned}
\mathcal{L}_{\text{DT}} &= \mathbb{E}_{p_{\text{data}}(x_0)}\Big[ \cdots \Big] \\
&\quad + \sum_{t=2}^T \mathbb{E}_{q(x_t \mid x_0)}\big[ \cdots \big].
\end{aligned}
\label{eq:elbo-discrete}
\end{equation}

% 正确（无编号公式，慎用）
\begin{equation*}
\text{NFE} = \frac{T}{\tau}.
\end{equation*}
```

### 2.2 行内公式

使用 `$...$`。

```latex
设 $\boldsymbol{x}_t$ 为时刻 $t$ 的随机变量，其中 $t \in [0, T]$。
```

### 2.3 公式标签

每个可被引用的公式都必须有 `\label{eq:...}`，标签名称用英文小写，连字符分隔。

```latex
\label{eq:separable-rate}
\label{eq:time-reversal}
\label{eq:elbo-ct}
```

### 2.4 公式引用

```latex
% 推荐（带括号）
由 \eqref{eq:time-reversal} 可知……

% 也可用
由式~(\ref{eq:time-reversal}) 可知……
```

同一页内不应出现两个无标签的相同公式。

---

## 3 定理类环境

### 3.1 环境命令

| 环境 | 用途 | 标签前缀 |
|------|------|---------|
| `\begin{theorem}` | 定理 | `thm:` |
| `\begin{proposition}` | 命题 | `prop:` |
| `\begin{lemma}` | 引理 | `lem:` |
| `\begin{corollary}` | 推论 | `cor:` |
| `\begin{definition}` | 定义 | `def:` |
| `\begin{remark}` | 注解 | `rmk:` |
| `\begin{example}` | 例题 | `ex:` |
| `\begin{proof}` | 证明 | （不加标签） |

### 3.2 用法示例

```latex
\begin{proposition}[反向速率的维度分解]
\label{prop:dim-decomp}
若前向过程因子化为各维度的独立 CTMC，则反向过程也具有相同的因子化结构：
\begin{equation}
\hat{R}_t^{1:D}(\boldsymbol{x}^{1:D}, \tilde{\boldsymbol{x}}^{1:D}) = \sum_{d=1}^D \hat{R}_t^d(\boldsymbol{x}^{1:D}, \tilde{x}^d) \;\delta_{\tilde{\boldsymbol{x}}^{1:D \setminus d}, \boldsymbol{x}^{1:D \setminus d}}.
\label{eq:reverse-dim-decomp}
\end{equation}
\end{proposition}
```

引用时统一使用 `\ref`：

```latex
由 \ref{prop:dim-decomp} 可知……
如 \ref{thm:error-bound} 所示……
见 \ref{rmk:schedule-flexibility}。
```

### 3.3 注意事项

- 定理的可选标题（方括号内）只用来说明该定理的核心结论，不写"来源"信息
- 证明环境末尾自动带 $\square$，无需手动添加
- 同一命题内若有多个公式，公式标签前缀用 `eq:`，命题标签用 `prop:`

---

## 4 章节标签

### 4.1 前缀规范

| 对象 | 标签前缀 | 示例 |
|------|---------|------|
| `\chapter` | `chap:` | `\label{chap:ctmc}` |
| `\section` | `sec:` | `\label{sec:forward}` |
| `\subsection` | `sec:` | `\label{sec:dim-decomp}` |
| `\subsubsection` | `sec:` | `\label{sec:base-rate}` |

### 4.2 示例

```latex
\section{前向噪声过程的 CTMC 构建}
\label{sec:forward}

\subsection{维度分解}
\label{sec:dim-decomp}
```

引用：

```latex
详见 \ref{sec:forward} 节。
如 \ref{sec:dim-decomp} 所述。
```

---

## 5 图表标签

### 5.1 前缀规范

| 对象 | 标签前缀 | 示例 |
|------|---------|------|
| 图 | `fig:` | `\label{fig:architecture}` |
| 表 | `tab:` | `\label{tab:cifar10}` |

### 5.2 示例

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/architecture.pdf}
\caption{CTMC 离散扩散模型的整体架构}
\label{fig:architecture}
\end{figure}
```

```latex
\begin{table}[htbp]
\centering
\caption{CIFAR-10 上的性能对比}
\label{tab:cifar10}
\begin{tabular}{lcc}
\toprule
方法 & IS ($\uparrow$) & FID ($\downarrow$) \\
\midrule
D3PM & 8.56 & 7.34 \\
$\tau$LDR-10 & 9.49 & 3.74 \\
\bottomrule
\end{tabular}
\end{table}
```

引用：

```latex
如图 \ref{fig:architecture} 所示。
见表 \ref{tab:cifar10}。
```

---

## 6 交叉引用总览

| 对象 | 标签前缀 | 引用命令 | 示例 |
|------|---------|---------|------|
| 公式 | `eq:` | `\eqref` | `\eqref{eq:time-reversal}` |
| 定理 | `thm:` | `\ref` | `\ref{thm:error-bound}` |
| 命题 | `prop:` | `\ref` | `\ref{prop:dim-decomp}` |
| 引理 | `lem:` | `\ref` | `\ref{lem:xxx}` |
| 推论 | `cor:` | `\ref` | `\ref{cor:xxx}` |
| 定义 | `def:` | `\ref` | `\ref{def:xxx}` |
| 注解 | `rmk:` | `\ref` | `\ref{rmk:xxx}` |
| 章节 | `sec:` | `\ref` | `\ref{sec:forward}` |
| 图 | `fig:` | `\ref` | `\ref{fig:architecture}` |
| 表 | `tab:` | `\ref` | `\ref{tab:cifar10}` |

---

## 7 其他约定

### 7.1 空格与换行

- 中文与英文/公式之间加一个空格（XeLaTeX 会自动处理，但源码中保持可读性）
- 每个句号后换行，便于 diff 和审阅
- 每个公式独立成段，前后留空行

### 7.2 数学符号

- 向量使用 `\boldsymbol{x}` 或 `\bm{x}`
- 矩阵使用 `\mathbf{R}` 或 `\mathsf{Q}`
- 集合使用 `\mathcal{S}`, `\mathcal{X}`
- 期望使用 `\mathbb{E}`
- 分类分布使用 `\text{Cat}`
- 常用运算符自定义命令（如 `\KL`、`\ELBO`）需在导言区定义
