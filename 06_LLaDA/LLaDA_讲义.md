# LLaDA: Large Language Diffusion with mAshing — 数学统计讲义

> **论文来源**: *Large Language Diffusion Models* (NeurIPS 2025)
> **作者**: Shen Nie, Fengqi Zhu, Zebin You, et al. (中国人民大学 & 蚂蚁集团)
> **本文档定位**: 以数学/统计专业研究生课程讲义风格，对 LLaDA 论文进行系统解读，每节包含背景动机、原理架构、公式推导与解释讲义。

---

## 第1章 生成式建模基础与动机

### 1.1 语言生成的概率框架

**背景与动机.** 大语言模型（LLM）的核心问题是：如何从真实但未知的语言数据分布 $p_{\text{data}}(\cdot)$ 中学习，使模型分布 $p_\theta(\cdot)$ 尽可能逼近它。

**定义 1.1（生成式建模原理）** 最优参数 $\theta$ 通过极大似然估计获得，等价于最小化模型分布与真实分布之间的 Kullback-Leibler 散度：

$$
\boxed{\max_{\theta} \mathbb{E}_{p_{\text{data}}(x)} \log p_\theta(x) \;\Longleftrightarrow\; \min_{\theta} \mathrm{KL}\big(p_{\text{data}}(x) \,\|\, p_\theta(x)\big)} \tag{1}
$$

**定义 1.2（自回归建模, ARM）** 当前主流 LLM 使用自回归分解（"下一个 token 预测"）：

$$
\boxed{p_\theta(x) = p_\theta(x^1) \prod_{i=2}^{L} p_\theta(x^i \mid x^1, \ldots, x^{i-1})} \tag{2}
$$

其中 $x = (x^1, \ldots, x^L)$ 是长度为 $L$ 的 token 序列。

> **【讲义 1.1】**
>
> **公式 (1) 的含义**：生成式建模的目标无非是极大化观测数据在模型下的对数似然。Fisher 一致性（Fisher Consistency）告诉我们，当模型容量和数据量趋于无穷时，极大似然估计可以恢复真实的 $p_{\text{data}}$。这是所有生成模型——无论自回归还是扩散——共同的理论根基。
>
> **公式 (2) 的含义**：自回归模型将联合概率按**因果顺序**（从左到右）分解为条件概率的乘积。$x^i$ 只依赖 $x^1,\ldots,x^{i-1}$。这既是它的力量（简化为逐 token 预测），也是它的局限（无法利用右侧上下文）。
>
> **论文的核心问题**：公式 (1) 和 (2) 哪个才是 LLM 能力的**本质来源**？作者的答案是：** (1) 才是根本，(2) 只是实现方式之一。** 因此，用扩散模型来定义 $p_\theta(x)$——只要它同样优化公式 (1)——理论上也能获得 LLM 的能力。

### 1.2 自回归范式的固有限制：逆转诅咒

**定义 1.3（逆转诅咒, Reversal Curse）** 如果模型在训练数据中见过 "$A$ 是 $B$" 的句式，但测试时无法从 "$B$ 是 $A$" 的角度正确回答，则称模型遭受逆转诅咒。这源于自回归分解的单向性：

$$
p_\theta(B \mid A) \text{ 被优化 } \quad\not\Rightarrow\quad p_\theta(A \mid B) \text{ 被优化 }
$$

> **【讲义 1.2】**
>
> **直觉解释**：自回归模型总是按固定方向（左→右）编码依赖关系。当它在训练时看到 "Paris is the capital of France"，它学会了在 "Paris is the capital of" 之后预测 "France"。但当你问 "What is the capital of France?" 时，模型需要反向推理——**自回归的因果结构没有为此提供直接的参数通路**。
>
> **数学解释**：公式 (2) 中，$p_\theta(x^i \mid x^{<i})$ 仅依赖左侧变量。如果数据中信息流是双向的，自回归分解就引入了**归纳偏置（inductive bias）**，在对称任务上造成系统性偏差。LLaDA 的掩码扩散则不同（见第 2 章），它天然支持**双向依赖**。

### 1.3 LLaDA 的设计哲学

LLaDA（Large Language Diffusion with mAsking）的设计理念可概括为：

1. **用掩码扩散替换自回归**：定义前向掩码过程和反向生成过程
2. **优化似然下界而非精确似然**：以 $\mathcal{L}(\theta)$ 作为负对数似然的上界
3. **保留标准预训练-SFT 范式**：数据准备、训练、评估流程与现有 LLM 保持一致
4. **达到 8B 参数规模**：在 2.3T tokens 上训练，验证扩散模型的可扩展性

> **【讲义 1.3】**
>
> LLaDA 的核心洞察是：生成式建模原理（公式 (1)）提供的 Fisher 一致性 + Transformer 架构 + 大量数据 = LLM 能力，**而不是**自回归公式 (2) 本身。视觉领域的 Diffusion Transformer (DiT) 的成功为这一观点提供了佐证——图像也没有天然的顺序，但扩散模型依然表现出色。

---

## 第2章 掩码扩散模型的理论基础

### 2.1 前向过程（数据掩码过程）

**背景与动机.** 要摆脱自回归的因果分解，我们需要一种新的方式定义模型分布 $p_\theta(x_0)$。掩码扩散的思路是：定义一个**前向过程**，将干净数据 $x_0$ 逐步转化为全掩码序列 $x_1$；再学习一个**反向过程**，从 $x_1$ 逐步恢复 $x_0$。模型分布 $p_\theta(x_0)$ 就是反向过程在 $t=0$ 时的边际分布。

**定义 2.1（前向掩码过程）** 对 $t \in [0, 1]$，定义随机过程 $\{x_t\}$。在时间 $t$，给定 $x_0$，$x_t$ 的条件概率为完全因子化形式：

$$
q_{t|0}(x_t \mid x_0) = \prod_{i=1}^{L} q_{t|0}(x_t^i \mid x_0^i) \tag{7}
$$

其中每个 token 独立地以概率 $t$ 被掩码：

$$
q_{t|0}(x_t^i \mid x_0^i) =
\begin{cases}
1 - t, & x_t^i = x_0^i, \\[4pt]
t,     & x_t^i = \mathbf{M}.
\end{cases} \tag{8}
$$

> **【讲义 2.1】**
>
> **公式 (7) 的意义**：每个 token 的掩码与否**相互独立**。这极大简化了计算——前向过程可以高效地并行采样。
>
> **公式 (8) 的意义**：掩码概率 $t$ 是**线性**地从 0 增长到 1：
> - $t=0$：所有 token 保持原值（$x_0$）
> - $t=1$：所有 token 都被掩码（$x_1$ 是纯掩码序列）
> - $t \in (0,1)$：每个 token 以概率 $t$ 变为 $\mathbf{M}$，以概率 $1-t$ 不变
>
> **为什么要用线性调度？** 文本的信息量大致与 token 数量成正比。线性地增加掩码比例意味着信息量线性地减少——这与连续扩散模型中的噪声调度（如 DDPM 的余弦调度）角色类似，但这里是**离散状态**上的版本。
>
> **与 BERT 的关键区别**：BERT 使用固定的 15% 掩码率，这意味着 BERT 只学习"填空"（fill-in-the-blank）。而 LLaDA 在 $t \sim U[0,1]$ 下训练，模型必须能够在**所有噪声水平**下进行预测——这正是它成为**生成式模型**（而非判别式模型）的关键。
>
> **与 MaskGIT 的关键区别**：MaskGIT 的损失缺少 $1/t$ 权重项，没有理论保证与极大似然之间的联系。LLaDA 的 $1/t$ 项来自变分下界的推导（见第 2.3 节）。

### 2.2 反向过程与数据预测函数

**背景与动机.** 给定前向过程，反向过程的目的是从 $t=1$（全掩码）逐步恢复到 $t=0$（干净数据）。需要确定反向转移核 $q_{s|t}$。

**定义 2.2（反向条件分布）** 对于 $0 \leq s < t \leq 1$，反向转移核同样完全因子化：

$$
q_{s|t}(x_s \mid x_t) = \prod_{i=1}^{L} q_{s|t}(x_s^i \mid x_t) \tag{9}
$$

其中每个 token 的转移概率为：

$$
q_{s|t}(x_s^i \mid x_t) =
\begin{cases}
1, & x_t^i \neq \mathbf{M},\; x_s^i = x_t^i, \\[8pt]
\dfrac{s}{t}, & x_t^i = \mathbf{M},\; x_s^i = \mathbf{M}, \\[12pt]
\dfrac{t-s}{t}\, q_{0|t}(x_s^i \mid x_t), & x_t^i = \mathbf{M},\; x_s^i \neq \mathbf{M}, \\[4pt]
0, & \text{otherwise}.
\end{cases} \tag{10}
$$

**定义 2.3（数据预测函数）** $q_{0|t}(x_s^i \mid x_t)$ 是反向过程的核心，表示给定部分观测 $x_t$ 后预测原始 token 的条件分布。

> **【讲义 2.2】**
>
> **公式 (10) 的四种情况解读**：
> 1. **未掩码的 token 保持不变**：如果 $x_t^i$ 不是掩码，那么它在反向过程中确定性地保持原值——一旦信息在正向过程中幸存，反向中就不需要改变它。
> 2. **掩码 token 继续保持掩码**（概率 $s/t$）：这是为了与前向过程的一致性——如果前向过程以概率 $t$ 掩码了一个 token，那么在反向时间 $s$，它还有 $s/t$ 的概率尚未被恢复。
> 3. **掩码 token 被恢复为某个值**（概率 $(t-s)/t$）：这是真正的"去噪"步骤，需要 $q_{0|t}$ 来预测应该恢复成什么。
> 4. **不可能情况**：反向过程中，一个未掩码的 token 不能变为其他值。
>
> **核心函数 $q_{0|t}$**：这是神经网络需要学习的对象。给定部分观测 $x_t$，对于每个被掩码的位置 $i$，模型需要预测原始 token $x_0^i$ 的分布。

**定理 2.1（时间无关性）** 数据预测函数 $q_{0|t}$ 实际上等价于**干净数据上的条件分布**，且与时间 $t$ 无关（Ou et al., 2024）：

$$
\boxed{q_{0|t}(x_s^i \mid x_t) = p_{\text{data}}(x_0^i \mid x_t^{\text{UM}})}, \quad \forall i \text{ 满足 } x_t^i = \mathbf{M}. \tag{11}
$$

其中 $x_t^{\text{UM}}$ 表示 $x_t$ 中所有未掩码 token 的集合。

> **【讲义 2.2 续】**
>
> **公式 (11) 的革命性意义**：
>
> - **时间信息是多余的**：给定未掩码 token 的集合，预测被掩码位置的条件分布**并不需要知道 $t$ 是多少**。这意味着神经网络的输入不需要时间嵌入（time embedding）。
> - **直觉解释**：假设序列中有 5 个可见 token 和 3 个掩码 token。作为预测器，你只需要知道哪些位置可见以及它们是什么值，而不需要知道"是因为 30% 掩码率还是 50% 掩码率导致了这么多掩码"。
> - **实际简化**：这大大降低了模型复杂度——LLaDA 的 Transformer 不需要像 DDPM 那样将 $t$ 编码后注入网络。这也意味着同一个 $p_\theta(\cdot|x_t)$ 可以在任意时间 $t$ 使用。

### 2.3 训练目标：变分下界

**背景与动机.** 我们无法直接计算 $p_\theta(x_0)$ 的精确对数似然（因为 $p_\theta$ 是通过反向扩散过程的边际分布定义的）。因此需要优化对数似然的**变分下界**。

**定理 2.2（变分上界）** 定义损失函数：

$$
\boxed{\mathcal{L}(\theta) \triangleq -\mathbb{E}_{t, x_0, x_t}\left[ \frac{1}{t} \sum_{i=1}^{L} \mathbf{1}[x_t^i = \mathbf{M}] \log p_\theta(x_0^i \mid x_t) \right]} \tag{3}
$$

其中 $t \sim U[0,1]$，$x_t \sim q_{t|0}(x_t|x_0)$，则 $\mathcal{L}(\theta)$ 是负对数似然的**上界**：

$$
\boxed{-\mathbb{E}_{p_{\text{data}}(x_0)}[\log p_\theta(x_0)] \leq \mathcal{L}(\theta)} \tag{4}
$$

> **【讲义 2.3】**
>
> **公式 (3) 的结构**：
> - **效率性**：$\mathbf{1}[x_t^i = \mathbf{M}]$ 表示**只对掩码 token 计算交叉熵损失**——这很直观，模型只负责预测被掩码的部分。
> - **权重 $1/t$**：这是与 MaskGIT 等启发式方法的根本区别。$1/t$ 来自变分下界的推导。当 $t$ 很小（只有少数 token 被掩码）时，$1/t$ 很大，迫使模型在这些"容易"的情况下也提供高精度的预测。
> - **全局均匀的 $t$**：$t \sim U[0,1]$ 确保模型在所有掩码水平上都被训练，这与 BERT（固定 15%）不同。
>
> **公式 (4) 的意义**：最小化 $\mathcal{L}(\theta)$ 等价于**最小化负对数似然的上界**——这是原则性的生成式建模。虽然我们无法直接优化 $-\log p_\theta(x_0)$，但我们可以优化一个保证不会比它小的上界。
>
> **与 ARM 的对比**：ARM 的交叉熵损失直接对应于精确的负对数似然（因为 $p_\theta$ 可以直接写出）。扩散模型只能优化一个上界——这解释了为何 Nie et al. (2024) 发现 MDM 需要更多计算量才能达到相同的似然值。但论文指出，**下游任务性能与似然值并不直接相关**。

 **预备知识：离散马尔可夫轨迹的 ELBO 分解**

在推导连续积分前，我们必须先看清在离散时间步下（假设从时刻 $0$ 到时刻 $1$ 被等分成 $N$ 个微小步，步长 $\Delta t = 1/N$），这个界是怎么构建的。设前向加噪轨迹为 $x_0 \to x_{\Delta t} \to x_{2\Delta t} \dots \to x_1$。

由于前向马尔可夫链的转移概率 $q(x_{\Delta t:1} | x_0)$ 完全已知，根据标准的 Jensen 不等式，负对数似然可以写出如下上界（传统的离散变分下界）：

$$
- \log p_\theta(x_0) \le \mathbb{E}_{q(x_{\Delta t:1}|x_0)} \left[ \log \frac{q(x_{\Delta t:1} | x_0)}{p_\theta(x_{0:1})} \right]
$$

利用马尔可夫链的转移性质，前向联合概率可以展开为条件概率连乘：

$$
q(x_{\Delta t:1} | x_0) = \prod_{k=1}^N q(x_{k\Delta t} | x_{(k-1)\Delta t})
$$

同理，逆向生成过程也是一个马尔可夫链：

$$
p_\theta(x_{0:1}) = p(x_1) \prod_{k=1}^N p_\theta(x_{(k-1)\Delta t} | x_{k\Delta t})
$$

将连乘代入对数中，展开为求和，并将属于相同时间步的项合并，离散形式的标准上界为：

$$
- \log p_\theta(x_0) \le \mathbb{E}_{q} \left[ \sum_{k=1}^N D_{\text{KL}}\big( q(x_{(k-1)\Delta t} | x_{k\Delta t}, x_0) \ || \ p_\theta(x_{(k-1)\Delta t} | x_{k\Delta t}) \big) \right] + \text{端点项}
$$

---

**核心推导：从离散求和向连续积分的极限跨越**

现在，我们让时间切片变得无限稠密，即令步数 $N \to \infty$，步长 $\Delta t \to 0$。此时，上面的离散求和 $\sum_{k=1}^N$ 自然收敛为连续积分 $\int_0^1 \frac{1}{\Delta t} \dots dt$。

我们聚焦于在任意绝对时间点 $t$（对应离散步的 $k\Delta t$）上的那一个微元时间段 $[t - \Delta t, t]$ 内的局部 KL 散度项。我们定义局部损失速率 $\mathcal{L}_{\text{local}}$ 为单位时间内的 KL 散度增量：

$$
\mathcal{L}_{\text{local}}(x_0, x_t, t) \triangleq \lim_{\Delta t \to 0} \frac{1}{\Delta t} D_{\text{KL}}\big( q(x_{t-\Delta t} | x_t, x_0) \ || \ p_\theta(x_{t-\Delta t} | x_t) \big)
$$

只要我们能严格推导出这个极限的解析形式，将其放回积分中，就能得到你问的那个始发式。下面我们针对 LLaDA 的独立掩码机制来彻底算死这个极限。

**第一步：拆解贝叶斯后验 $q(x_{t-\Delta t} | x_t, x_0)$**

在掩码扩散中，前向过程是单向污染的（健康的词只能变成 [MASK]，而 [MASK] 一旦产生就无法复原）。因此，我们要计算从当前残缺状态 $x_t$ 倒退回上一个更清醒状态 $x_{t-\Delta t}$ 的后验概率。根据贝叶斯公式，对于序列中的第 $i$ 个位置：

$$
q(x_{t-\Delta t}^i | x_t^i, x_0^i) = \frac{q(x_t^i | x_{t-\Delta t}^i) \cdot q(x_{t-\Delta t}^i | x_0^i)}{q(x_t^i | x_0^i)}
$$

由于位置之间相互独立，我们只需考察两种物理状态：

**状态 A**：当前位置已经是 MASK（$x_t^i = \text{M}$）

此时，这个位置在更早的时刻 $t-\Delta t$ 时，有两种可能：要么已经是 $\text{M}$，要么还是清醒的 $x_0^i$。

若 $x_{t-\Delta t}^i = \text{M}$：说明在 $\Delta t$ 这段时间内没有发生新掩码。根据前向定义，$q(x_t^i=\text{M} | x_{t-\Delta t}^i=\text{M}) = 1$。代入贝叶斯公式：

$$
q(x_{t-\Delta t}^i = \text{M} | x_t^i = \text{M}, x_0^i) = \frac{1 \cdot t}{t} = 1 - \frac{\Delta t}{t} + o(\Delta t)
$$

若 $x_{t-\Delta t}^i = x_0^i$：说明掩码恰好发生在 $[t-\Delta t, t]$ 这段微元时间内。根据前向定义，转移概率为 $\frac{\Delta t}{1 - (t - \Delta t)}$。代入贝叶斯公式展开并略去高阶无穷小量后，其解析概率为：

$$
q(x_{t-\Delta t}^i = x_0^i | x_t^i = \text{M}, x_0^i) = \frac{\Delta t}{t} + o(\Delta t)
$$

**状态 B**：当前位置没有被掩码（$x_t^i = x_0^i$）

由于掩码是单向不可逆的，如果现在都是清醒的，那么过去必然也是清醒的。因此该后验概率是绝对确定性的：

$$
q(x_{t-\Delta t}^i = x_0^i | x_t^i = x_0^i, x_0^i) = 1
$$

**第二步：参数化反向模型 $p_\theta(x_{t-\Delta t} | x_t)$ 的微元构造**

为了能和上面的前向后验进行对齐，神经网络在反向推进一个微元步 $\Delta t$ 时的条件概率分布必须采取对称的结构设计：

如果当前位置没被掩码（$x_t^i = x_0^i$）：直接继承，不做多余动作：

$$
p_\theta(x_{t-\Delta t}^i = x_0^i | x_t^i = x_0^i) = 1
$$

如果当前位置是 MASK（$x_t^i = \text{M}$）：网络根据当前的残缺上下文 $x_t$，以预测概率 $p_\theta(v | x_t)$ 尝试去将 $\text{M}$ 还原为词表中的某个词 $v$。如果在微元时间 $\Delta t$ 内突变恢复成功，则变为 $v$；若未成功，则保持 $\text{M}$。其微元转移构型为：

$$
p_\theta(x_{t-\Delta t}^i = v | x_t^i = \text{M}) = \Delta t \cdot p_\theta(v | x_t)
$$

$$
p_\theta(x_{t-\Delta t}^i = \text{M} | x_t^i = \text{M}) = 1 - \Delta t \cdot \sum_v p_\theta(v | x_t) = 1 - \Delta t
$$

在实际计算中，如果采用更精确的归一化（考虑时间比例），这两个公式通常简化为：

$$
p_\theta(x_{t-\Delta t}^i = v | x_t^i = \text{M}) = \frac{\Delta t}{t} p_\theta(v | x_t)
$$

$$
p_\theta(x_{t-\Delta t}^i = \text{M} | x_t^i = \text{M}) = 1 - \frac{\Delta t}{t}
$$

---

**第三步：极限显式计算与导数项的涌现**

现在我们将第一步（前向真实后验 $q$）和第二步（模型反向步 $p_\theta$）的微元解析式，正式代入局部 KL 散度的定义式中，并执行 $\lim_{\Delta t \to 0} \frac{1}{\Delta t}$ 算子。

由于只有当前状态 $x_t^i = \text{M}$ 的位置才会对散度产生非零贡献（清醒位置的 KL 散度项为 $\log(1/1) = 0$），我们对掩码位置展开两个分支（变为真实词 $x_0^i$ 或保持 $\text{M}$）的对数求和：

$$
\begin{aligned}
D_{\text{KL}}(q \ || \ p_\theta) &= \sum_{i \in \text{Masked}} \left[ q(x_0^i | \text{M}) \log \frac{q(x_0^i | \text{M})}{p_\theta(x_0^i | \text{M})} + q(\text{M} | \text{M}) \log \frac{q(\text{M} | \text{M})}{p_\theta(\text{M} | \text{M})} \right] \\
&= \sum_{i=1}^L \mathbf{1}[x_t^i = \text{M}] \left[ \left(\frac{\Delta t}{t}\right) \log \frac{\frac{\Delta t}{t}}{\frac{\Delta t}{t} p_\theta(x_0^i | x_t)} + \left(1 - \frac{\Delta t}{t}\right) \log \frac{1 - \frac{\Delta t}{t}}{1 - \frac{\Delta t}{t}} \right] \\
&= \sum_{i=1}^L \mathbf{1}[x_t^i = \text{M}] \left[ \frac{\Delta t}{t} \log \frac{1}{p_\theta(x_0^i | x_t)} + 0 \right] \\
&= \Delta t \cdot \sum_{i=1}^L \mathbf{1}[x_t^i = \text{M}] \cdot \frac{1}{t} \left( - \log p_\theta(x_0^i | x_t) \right)
\end{aligned}
$$

最后，我们将这个展开式代回 $\mathcal{L}_{\text{local}}$ 的极限定义中，分母的 $\Delta t$ 与分子整式提取出来的 $\Delta t$ 完美相消：

$$
\begin{aligned}
\mathcal{L}_{\text{local}}(x_0, x_t, t) &= \lim_{\Delta t \to 0} \frac{1}{\Delta t} \left[ \Delta t \cdot \sum_{i=1}^L \mathbf{1}[x_t^i = \text{M}] \cdot \frac{1}{t} \left( - \log p_\theta(x_0^i | x_t) \right) \right] \\
&= \frac{1}{t} \sum_{i=1}^L \mathbf{1}[x_t^i = \text{M}] \left( - \log p_\theta(x_0^i | x_t) \right)
\end{aligned}
$$

注意到我们在前一节课推导中说过的，前向掩码概率 $q(x_t^i = \text{M} | x_0^i) = t$，其对时间的微分变化率 $\frac{d}{dt}(t) = 1$。因此，上面式子中脱落出来的系数 $1$，本质上就是导数项 $\frac{d}{dt} q(x_t^i = \text{M} | x_0^i)$。而括号里的 $\log \frac{1}{p_\theta}$ 本质上就是狄拉克独热分布下的标准 KL 散度。

**终点站：公式汇聚**

我们将求得的连续极限下的 $\mathcal{L}_{\text{local}}(x_0, x_t, t)$ 重新放回最外层的时间全积分中，并对真实数据分布 $p_{\text{data}}(x_0)$ 以及前向加噪路径 $q(x_t|x_0)$ 全局求期望：

$$
-\mathbb{E}_{x_0 \sim p_{\text{data}}} [\log p_\theta(x_0)] \le \int_0^1 \mathbb{E}_{x_0 \sim p_{\text{data}}} \mathbb{E}_{x_t \sim q(x_t|x_0)} \left[ \mathcal{L}_{\text{local}}(x_0, x_t, t) \right] dt
$$

这个优雅的连续时间不等式，就是这样通过将离散马尔可夫链的每一步变分亏损（KL 散度）除以步长 $\Delta t$，在时空无限稠密的连续极限下积分而来。它为整个 LLaDA 扩散模型锁定了无懈可击的极大似然数学边界。

**定义 2.4（等价损失形式）** 公式 (3) 的一个等价形式具有更低的方差：

$$
\boxed{-\mathbb{E}_{l, x_0, x_l}\left[ \frac{L}{l} \sum_{i=1}^{L} \mathbf{1}[x_l^i = \mathbf{M}] \log p_\theta(x_0^i \mid x_l) \right]} \tag{14}
$$

其中 $l \sim U\{1,2,\ldots,L\}$，$x_l$ 是从 $x_0$ 中**无放回均匀采样** $l$ 个位置掩码得到。

> **【讲义 2.3 续】**
>
> **公式 (3) vs 公式 (14)**：
> - 公式 (3) 中，$x_t$ 的掩码数量是**随机变量**（二项分布），方差大
> - 公式 (14) 中，$x_l$ 的掩码数量是**确定性的**（恰好 $l$ 个），方差小
> - 经验结果：公式 (3) 需要 1000+ 次蒙特卡洛估计才能稳定，而公式 (14) 仅需约 128 次
>
> **条件版本**：将公式 (14) 中的无条件分布改为条件分布，得到用于 SFT 评估的公式 (6)（见第 3.3 节）。

### 2.4 与 Any-Order Autoregressive Models 的联系

**定理 2.3（与 AO-ARM 的等价性）** 掩码扩散模型的训练目标等价于任意阶自回归模型（AO-ARM）的期望负对数似然：

$$
-\mathbb{E}_{x_0, \pi \sim U_\pi}\left[ \sum_{i=1}^{L} \log p_\theta(x_0^{\pi(i)} \mid x_0^{\pi(<i)}; \pi) \right] \tag{15}
$$

其中 $\pi$ 是从 $L!$ 种排列中均匀采样的顺序，$U_\pi$ 是均匀分布。

> **【讲义 2.4】**
>
> **公式 (15) 的含义**：如果一个模型对**所有可能的 token 排列顺序**都进行自回归预测，那么它的训练目标与 LLaDA 的掩码扩散目标 (3) 等价。这揭示了 LLaDA 的**双向推理能力**的数学根源——它隐式地学会了在所有条件化方向下进行预测。
>
> **直觉解释**：在 LLaDA 的训练中，$x_t$ 中的未掩码 token 可以是**任意位置的任意子集**。模型必须学会根据"左侧的 token"预测"右侧的掩码"**和**根据"右侧的 token"预测"左侧的掩码"。这就是双向性的来源，也是 LLaDA 能缓解逆转诅咒的根本原因。

---

## 第3章 LLaDA 的算法框架

### 3.1 预训练算法

**背景与动机.** LLaDA 的预训练与标准 LLM 预训练遵循相同的数据准备和训练流程，但用掩码扩散目标替代了自回归的因果语言模型目标。

**算法 1（LLaDA 预训练）**

$$
\begin{aligned}
&\textbf{输入：} \text{掩码预测器 } p_\theta,\; \text{数据分布 } p_{\text{data}} \\
&\textbf{重复：} \\
&\quad 1.\; x_0 \sim p_{\text{data}} \quad (\text{以 }1\%\text{ 概率使用随机长度 } U[1, 4096]) \\
&\quad 2.\; t \sim U(0, 1] \\
&\quad 3.\; x_t \sim q_{t|0}(x_t \mid x_0) \quad \text{(公式 (7)-(8))} \\
&\quad 4.\; \text{计算 } \mathcal{L} = -\frac{1}{t \cdot L}\sum_{i=1}^{L} \mathbf{1}[x_t^i = \mathbf{M}] \log p_\theta(x_0^i \mid x_t) \\
&\quad 5.\; \text{计算 } \nabla_\theta \mathcal{L} \text{ 并更新参数} \\
&\textbf{返回：} p_\theta
\end{aligned}
$$

> **【讲义 3.1】**
>
> **架构细节**：
> - 使用标准的 Transformer（LLaMA 架构：RMSNorm, SwiGLU, RoPE）
> - **不使用因果掩码**：所有 token 之间都可以互相注意力——因为公式 (3) 允许双向上下文
> - **不使用 KV 缓存**：因为 LLaDA 并行预测所有掩码 token，而非逐 token 生成
> - 使用 vanilla multi-head attention（而非 GQA），因为 KV 缓存不兼容
> - 降低 FFN 维度以补偿注意力层增加的参数量
>
> **训练细节**：
> - 2.3T tokens，序列长度 4096，计算量 13 万 H800 GPU 小时
> - Warmup-Stable-Decay 学习率：$0 \to 4\times10^{-4}$（2000 步），保持至 1.2T tokens，衰减至 $1\times10^{-4}$ 保持至 2.0T tokens，最后 0.3T 衰减至 $1\times10^{-5}$
> - AdamW, weight decay 0.1, batch size 1280
>
> **变长训练技巧**：1% 的数据使用随机长度 $U[1, 4096]$，增强模型处理变长输入的能力。这在下游任务中至关重要，因为推理时的生成长度是超参数。

### 3.2 监督微调（SFT）

**背景与动机.** 为使 LLaDA 具备指令遵循能力，使用提示-响应对 $(p_0, r_0)$ 进行监督微调。

**定义 3.1（SFT 损失函数）** 保持提示 $p_0$ 不变，仅对响应 $r_0$ 进行随机掩码：

$$
\boxed{-\mathbb{E}_{t, p_0, r_0, r_t}\left[ \frac{1}{t} \sum_{i=1}^{L'} \mathbf{1}[r_t^i = \mathbf{M}] \log p_\theta(r_0^i \mid p_0, r_t) \right]} \tag{5}
$$

其中 $L'$ 是响应的长度。

> **【讲义 3.2】**
>
> **与预训练的统一性**：这是 LLaDA 设计中最优雅之处：
>
> $$
> \underbrace{(p_0, r_0)}_{\text{SFT 数据}} \;\equiv\; \underbrace{x_0}_{\text{预训练数据}}
> $$
> $$
> \underbrace{(p_0, r_t)}_{\text{SFT 输入}} \;\equiv\; \underbrace{x_t}_{\text{预训练输入}}
> $$
>
> 唯一的区别是 SFT 中所有掩码恰好落在 $r_0$ 部分。**无需任何架构修改。**
>
> **SFT 数据格式**：450 万对数据（100 万人工标注 + 350 万合成），涵盖代码、数学、指令遵循。短响应用 |EOS| token 补齐到等长。|EOS| 在训练中被视为普通 token，但在采样时用于控制生成长度。
>
> **多轮对话处理**：将 $n$ 轮对话 $(p_0^0, r_0^0, p_0^1, r_0^1, \ldots)$ 拆分为 $n$ 个单轮样本 $(p_0^0, r_0^0), (p_0^0 r_0^0 p_0^1, r_0^1), \ldots$，随机采样一个进行训练。

### 3.3 推理：条件似然评估

**背景与动机.** 对于需要计算候选答案似然的任务（如 MMLU 的选择题），LLaDA 需要评估条件对数似然 $\log p_\theta(r_0 \mid p_0)$。

**定义 3.2（条件似然评估）** 使用公式 (14) 的条件版本（Ou et al., 2024）：

$$
\boxed{-\mathbb{E}_{l, r_0, r_l}\left[ \frac{L}{l} \sum_{i=1}^{L} \mathbf{1}[r_l^i = \mathbf{M}] \log p_\theta(r_0^i \mid p_0, r_l) \right]} \tag{6}
$$

其中 $L$ 是响应长度，$l \sim U\{1,\ldots,L\}$，$r_l$ 是无放回采样 $l$ 个位置掩码得到。

> **【讲义 3.3】**
>
> **评估方法**：
> - 对每个候选答案，进行多次蒙特卡洛采样（通常 $n_{\text{mc}}=128$ 次）
> - 每条样本：随机采样 $l$ 个位置掩码，计算加权交叉熵 $\frac{L}{l} \sum \mathbf{1}[\cdot] \log p_\theta(\cdot)$
> - 取 $n_{\text{mc}}$ 次估计的平均值作为 $\log p_\theta(r_0 \mid p_0)$ 的近似
> - 选择似然最大的候选答案作为最终输出
>
> **为什么用这个形式而非公式 (5)？** 公式 (6) 的掩码数量是确定性的（恰好 $l$ 个），方差更小，128 次估计即可稳定。

### 3.4 推理：反向扩散采样

**背景与动机.** 对于需要生成自由文本的任务，LLaDA 模拟反向扩散过程，从全掩码序列逐步生成文本。

#### 3.4.1 随机重掩码策略

**算法 4（随机重掩码采样）**

$$
\begin{aligned}
&\textbf{输入：} \text{掩码预测器 } p_\theta,\; \text{提示 } p_0,\; \text{生成长度 } L,\; \text{采样步数 } N \\
&\textbf{初始化：} r_1 \gets \text{长度为 } L \text{ 的全掩码序列} \\
&\textbf{对 } t = 1, \tfrac{N-1}{N}, \ldots, \tfrac{1}{N}: \\
&\quad s = t - \tfrac{1}{N} \\
&\quad \hat{r}_0 = \arg\max p_\theta(r_0 \mid p_0, r_t) \quad \text{(贪心解码)} \\
&\quad \textbf{对每个位置 } i = 1, \ldots, L: \\
&\qquad \textbf{如果 } r_t^i \neq \mathbf{M}:\; \hat{r}_0^i = r_t^i \\
&\qquad \textbf{否则}:\; \hat{r}_0^i \gets \mathbf{M} \text{ 以概率 } \tfrac{s}{t} \\
&\quad r_s = \hat{r}_0 \\
&\textbf{返回：} r_0
\end{aligned}
$$

#### 3.4.2 低置信度重掩码策略

**算法 5（低置信度重掩码）**

$$
\begin{aligned}
&\textbf{输入：} \text{掩码预测器 } p_\theta,\; \text{提示 } p_0,\; \text{生成长度 } L,\; \text{采样步数 } N \\
&\textbf{初始化：} r_1 \gets \text{长度为 } L \text{ 的全掩码序列} \\
&\textbf{对 } t = 1, \tfrac{N-1}{N}, \ldots, \tfrac{1}{N}: \\
&\quad s = t - \tfrac{1}{N} \\
&\quad \textbf{对每个位置 } i = 1, \ldots, L: \\
&\qquad \textbf{如果 } r_t^i \neq \mathbf{M}:\; \hat{r}_0^i = r_t^i,\; c_i = 1 \\
&\qquad \textbf{否则}:\; \hat{r}_0^i = \arg\max p_\theta(\cdot \mid p_0, r_t),\; c_i = p_\theta(\hat{r}_0^i \mid p_0, r_t) \\
&\quad n_{\text{un}} = \lfloor L(1 - s) \rfloor \\
&\quad \text{将 } \hat{r}_0 \text{ 中置信度最低的 } (L - n_{\text{un}}) \text{ 个位置置为 } \mathbf{M} \\
&\quad r_s = \hat{r}_0 \\
&\textbf{返回：} r_0
\end{aligned}
$$

> **【讲义 3.4】**
>
> **离散化**：连续时间 $[0,1]$ 被离散为 $N$ 个等距步长。$N$ 是速度-质量权衡的超参数。
>
> **并行预测**：每步对所有掩码位置同时预测——这与 ARM 逐 token 生成有本质区别。
>
> **重掩码的意义**：以概率 $s/t$ 重新掩码部分已预测的 token，保证了反向过程与前向过程的一致性。没有这一步，采样就会偏离真实的数据分布。
>
> **随机 vs 低置信度**：
> - **随机重掩码**：均匀随机选择哪些已预测 token 重新掩码
> - **低置信度重掩码**：选择预测置信度最低的 token 重新掩码——这类似于 LLM 中的退火采样（temperature annealing）。Tab. 9 显示低置信度策略显著优于随机策略：
>
> | 策略 | BBH | GSM8K | Math | HumanEval | MBPP |
> |:---|:---:|:---:|:---:|:---:|:---:|
> | 随机重掩码 | 32.1 | 21.3 | 9.2 | 11.6 | 21.0 |
> | **低置信度** | **45.0** | **70.0** | **30.3** | **32.9** | **40.2** |
>
> **多种采样方式的灵活性**：LLaDA 支持三种采样方式，**无需重新训练**：
> 1. **自回归采样**：每步只 unmask 一个 token → 退化为 ARM
> 2. **块扩散采样**：在块内进行扩散，块间自回归
> 3. **纯扩散采样**：标准反向过程——**性能最佳**

### 3.5 无分类器引导（Classifier-Free Guidance）

**定义 3.3（无监督 CFG）** LLaDA 可以使用以下修改后的掩码预测器进行推理：

$$
\boxed{\tilde{p}_\theta(r_0 \mid p_0, r_t) \propto \frac{p_\theta(r_0 \mid p_0, r_t)^{1+w}}{p_\theta(r_0 \mid \mathbf{m}, r_t)^w}} \tag{16}
$$

其中 $\mathbf{m}$ 是与 $p_0$ 等长的掩码序列，$w$ 是控制提示强度的超参数。

> **【讲义 3.5】**
>
> **CFG 的作用**：通过放大条件信号（提示 $p_0$）、减弱无条件信号（用 $\mathbf{m}$ 替代 $p_0$），来提高生成与提示的对齐程度。Tab. 6 显示 CFG 一致性地提升了 LLaDA 在多个基准上的性能（如 ARC-C 从 45.9 提升至 47.9，Hellaswag 从 70.5 提升至 72.5）。
>
> **与 ARM 的公平比较**：为保证公平，论文主要实验**未使用 CFG**。CFG 的提升是额外的增益。

---

## 第4章 实验设计与结果分析

### 4.1 可扩展性验证

**背景与动机.** 证明 LLaDA 在不同计算量下展现出与 ARM 相当的性能提升趋势。

> **【讲义 4.1】**
>
> **实验设置**：
> - 在 1B 规模：LLaDA 与 ARM 共享**完全相同**的架构、数据和配置
> - 在大规模：因计算资源限制，LLaDA 8B vs ARM 7B（略微不同）
> - 使用预训练 FLOPs 作为统一的扩展性度量
> - 评估 6 个任务：MMLU, ARC-C, CMMLU, PIQA, GSM8K, HumanEval
>
> **关键发现**：
> - LLaDA 的整体扩展性曲线与 ARM 高度竞争（Fig. 3）
> - 在 MMLU 和 GSM8K 上，LLaDA 展现出**更强的**扩展性
> - 在 PIQA 等较弱任务上，差距随规模增大而缩小
>
> **性能增益假设**：ARM 只优化从左到右的条件概率，而 LLaDA 被训练考虑**多种条件化方向**（公式 (15) 的等价性）。这可能在逻辑推理和数学任务上带来更好的泛化性。

### 4.2 基准测试结果

**背景与动机.** 在标准 LLM 基准上比较 LLaDA 8B 与主流模型。

**表 1（预训练模型关键对比）**

| 任务 | LLaDA 8B | LLaMA3 8B | LLaMA2 7B |
|:---|---:|:---:|:---:|
| MMLU (5-shot) | **65.9** | 65.4 | 45.9 |
| GSM8K (4-shot) | **70.3** | 48.7 | 13.1 |
| HumanEval (0-shot) | **35.4** | 34.8 | 12.8 |
| CMMLU (5-shot) | **69.9** | 50.7 | 32.5 |
| C-Eval (5-shot) | **70.5** | 51.7 | 34.0 |

**表 4（逆转诅咒对比）**

| 模型 | 前向 | 反向 |
|:---|---:|:---:|
| GPT-4o | 82.7 | 34.3 |
| Qwen2.5-7B Instruct | 75.9 | 38.0 |
| **LLaDA-8B Instruct** | **51.8** | **45.6** |

> **【讲义 4.2】**
>
> **表 1 解读**：
> - LLaDA 8B 的训练数据量只有 LLaMA3 8B 的约 $2.3/15 \approx 15\%$，但在 MMLU 上持平甚至略超
> - GSM8K 的 70.3 大幅领先 LLaMA3 的 48.7——双向建模可能在数学推理上有优势
> - 在 BBH、ARC-C、Hellaswag 等任务上差距较大，可能源于数据质量和分布差异
> - 数据泄露已被排除（通过 iGSM 数据集验证，见附录 B.8）
>
> **表 4 解读**：
> - GPT-4o 的前向 82.7 远超 LLaDA 的 51.8——这体现了更大数据量和计算量的"知识"优势
> - 但 GPT-4o 反向仅 34.3，与正向差距达 48.4 个百分点——**逆转诅咒的典型表现**
> - LLaDA 的正反向差距仅 6.2 个百分点（51.8 vs 45.6）——**双向建模的天然优势**
> - 在逆转任务上，LLaDA 以 45.6 超过 GPT-4o 的 34.3 和 Qwen2.5 的 38.0

### 4.3 采样效率分析

**背景与动机.** 评估 LLaDA 在实际推理中的速度和资源消耗。

> **【讲义 4.3】**
>
> **速度分析**（Fig. 5）：
> - LLaDA 通过调整采样步数 $N$ 实现灵活的**速度-质量权衡**
> - 在 GSM8K 和 Math 上，LLaDA 在达到可比性能的同时，吞吐量是 LLaMA3（使用 KV Cache）的 1.5-1.8 倍
> - 例如 $N=32$ 时每步解码 8 个 token，$N=256$ 时每步解码 1 个 token
>
> **内存消耗**（Tab. 11）：
> - LLaDA 与 LLaMA3 无 KV Cache 时内存消耗相当
> - 比 LLaMA3 有 KV Cache 时略高，但差距在可接受范围内
>
> **重要声明**：论文的目标不是提出一个比 ARM 更快的模型，而是**证明扩散模型在语言建模大规模化上的潜力**。效率优化留给未来工作。

### 4.4 采样策略消融研究

> **【讲义 4.4】**
>
> **Tab. 7 & 8 的关键结论**：
> - **纯扩散采样**在所有策略中表现最佳
> - 块扩散采样随块长度增大而性能提升
> - 自回归采样在 LLaDA 上效果最差（尤其是 SFT 模型，因为 SFT 数据中每个样本都是完整句子，自回归采样会过早遇到 |EOS|）
>
> **SFT 模型的特殊处理**：由于 SFT 数据大量使用 |EOS| 补齐，纯扩散采样中 |EOS| token 会大量出现导致生成长度过短。解决方式：在采样时将 |EOS| 的置信度分数设为零。

---

## 第5章 理论性质深度分析

### 5.1 变分下界的推导思路

**定理 5.1（损失函数 $\mathcal{L}(\theta)$ 的变分起源）** 掩码扩散模型的负对数似然 $\mathbb{E}_{p_{\text{data}}(x_0)}[-\log p_\theta(x_0)]$ 可以被变分下界（ELBO）上界化，其中 $\mathcal{L}(\theta)$ 是经过简化后的形式。

**推导概要**：对于反向过程 $p_\theta(x_0) = \int p_\theta(x_1) \prod_{k=1}^{N} p_\theta(x_{s_k} \mid x_{t_k}) \, dx_1 \cdots$，ELBO 可写为各步 KL 散度之和。当使用公式 (8) 和 (10) 的具体形式时，KL 散度退化为公式 (3) 中的加权交叉熵形式。$\frac{1}{t}$ 项来自 KL 散度中对前向/反向概率比的泰勒展开。

> **【讲义 5.1】**
>
> 这解释了为什么 $\frac{1}{t}$ 项是关键——它直接来自变分推断的理论推导，而不是启发式添加的。MaskGIT 忽略了这一项，因此在严格意义上不是一个遵循极大似然原则的生成模型。

### 5.2 前向-反向一致性条件

**定理 5.2（重掩码概率的推导）** 反向步骤中重掩码概率 $\frac{s}{t}$ 满足详细平衡条件：

$$
q_{t|0}(x_t \mid x_0) \cdot q_{s|t}(x_s \mid x_t) = q_{s|0}(x_s \mid x_0) \cdot q_{t|s}(x_t \mid x_s)
$$

这保证了反向过程的边际分布与反向时间 $s$ 的前向过程边际分布一致，从而使得从 $t=1$ 到 $t=0$ 的采样收敛到真实数据分布。

> **【讲义 5.2】**
>
> 这是扩散模型的核心理论保证：只要反向转移核满足这个一致性条件，**无论模型 $p_\theta$ 是否完美**，采样过程的边际分布都会逼近真实数据分布（在 $p_\theta = q_{0|t}$ 的理想情况下）。

### 5.3 逆转诅咒的理论解释

**性质 5.1（LLaDA 的对称归纳偏置）** 由公式 (15) 的等价性，LLaDA 的训练目标等价于在所有排列顺序 $\pi$ 下进行自回归预测。因此，模型在训练中同时学习到：

$$
p_\theta(x_0^i \mid x_0^{\pi(<i)}) \quad \forall \pi \in \text{Perm}(L)
$$

这意味着对于任意两个 token $x^i$ 和 $x^j$，模型既学会了 $p_\theta(x^j \mid x^i)$ 也学会了 $p_\theta(x^i \mid x^j)$。在逆转任务（给定 $B$ 预测 $A$）中，自回归模型从未被训练过预测"左侧"的内容，而 LLaDA 在训练中不断这样做。

> **【讲义 5.3】**
>
> **自回归 vs 扩散的逆转推理对比**：
>
> ARM：训练 $\underbrace{\texttt{窈窕淑女}}_{\text{上下文}} \to \underbrace{\texttt{君子好逑}}_{\text{预测}}$ → 推理 $\underbrace{\texttt{君子好逑}}_{\text{上下文}} \to \;???$ 失败
>
> LLaDA：训练中大量出现 $\texttt{[M] 淑女 君子 [M]}$ → 同时预测前向和反向，因此推理时两种方向都能处理。
>
> 这揭示了**归纳偏置（inductive bias）** 的重要性：ARM 的因果偏置在大多数情况下是有益的（因为它符合文本生成的自然顺序），但在需要对称推理的任务中成为了障碍。LLaDA 通过双向条件化消除了这一偏置。

---

## 第6章 讨论与展望

### 6.1 对比连续扩散语言模型

> **【讲义 6.1】**
>
> 已有的文本扩散方法主要分为两类：
>
> 1. **连续状态方法**：将离散文本嵌入到连续空间（如 Diffusion-LM, DiffuSeq），然后使用标准连续扩散模型（DDPM, Score Matching）。计算量大，1B 模型可能需要 64× ARM 的计算量（Gulrajani & Hashimoto, 2024）。
>
> 2. **离散状态方法**：直接在离散 token 空间中进行扩散（如 D3PM, MDM）。计算效率更高。LLaDA 属于此类，并将其规模扩展到了前所未有的 8B 参数。
>
> LLaDA 的成功表明，离散掩码扩散在语言建模上远比连续方法更具可行性。

### 6.2 局限性与未来方向

> **【讲义 6.2】**
>
> **已承认的局限性**：
> 1. **生成长度为超参数**：需要用户预先指定，虽然结果对此不敏感
> 2. **计算约束下的对比限制**：与 ARM 的直接对比只在 $10^{23}$ FLOPs 以下进行
> 3. **无 RL 对齐**：仅有 SFT，缺少 RLHF/DPO，这在性能上与 SFT+RL 的模型有差距
> 4. **无专门架构优化**：未设计特殊的注意力机制、位置编码或推理优化（如 KV cache 不适用）
> 5. **未扩展到多模态**
>
> **未来方向**：
> - RL 对齐（DPO, RLHF）
> - 自适应生成长度
> - 高效的采样算法（蒸馏、高阶求解器、一致性模型）
> - O1-like 推理系统集成
> - 多模态扩展
> - Agent 系统集成

### 6.3 核心结论

> **【讲义 6.3】**
>
> LLaDA 的贡献可概括为：
>
> 1. **首次证明**：扩散语言模型在 8B 规模上可以达到与主流自回归 LLM（LLaMA3 8B）竞争的性能
> 2. **挑战常识**：LLM 的核心能力（可扩展性、上下文学习、指令遵循）**并非自回归范式独有的**
> 3. **独特优势**：扩散模型天然具备双向建模能力，有效解决逆转诅咒
> 4. **范式意义**：为大语言模型的生成式建模开辟了新的概率范式

---

## 附录：算法汇总

### Algorithm 1: 预训练

$$
\begin{aligned}
&\text{Require: } p_\theta,\; p_{\text{data}} \\
&\text{repeat} \\
&\qquad x_0 \sim p_{\text{data}} \\
&\qquad t \sim U(0, 1] \\
&\qquad x_t \sim q_{t|0}(x_t \mid x_0) \\
&\qquad \mathcal{L} = -\frac{1}{t L}\sum_{i=1}^{L} \mathbf{1}[x_t^i = \mathbf{M}] \log p_\theta(x_0^i \mid x_t) \\
&\qquad \nabla_\theta \mathcal{L} \to \text{optimizer} \\
&\text{until converged} \\
&\text{Return } p_\theta
\end{aligned}
$$

### Algorithm 2: 监督微调

$$
\begin{aligned}
&\text{Require: } p_\theta,\; \text{pair data } p_{\text{data}} \\
&\text{repeat} \\
&\qquad (p_0, r_0) \sim p_{\text{data}} \\
&\qquad t \sim U(0, 1] \\
&\qquad r_t \sim q_{t|0}(r_t \mid r_0) \\
&\qquad \mathcal{L} = -\frac{1}{t L'}\sum_{i=1}^{L'} \mathbf{1}[r_t^i = \mathbf{M}] \log p_\theta(r_0^i \mid p_0, r_t) \\
&\qquad \nabla_\theta \mathcal{L} \to \text{optimizer} \\
&\text{until converged} \\
&\text{Return } p_\theta
\end{aligned}
$$

### Algorithm 3: 条件似然评估

$$
\begin{aligned}
&\text{Require: } p_\theta,\; p_0,\; r_0,\; n_{\text{mc}} \\
&\text{log\_likelihood} = 0 \\
&\text{for } i = 1 \text{ to } n_{\text{mc}} \text{ do} \\
&\qquad l \sim U\{1, \ldots, L\} \\
&\qquad r_l \gets \text{从 } r_0 \text{ 中无放回采样 } l \text{ 个位置掩码} \\
&\qquad \text{log\_likelihood} \mathrel{+}= \frac{L}{l} \sum_i \mathbf{1}[r_l^i = \mathbf{M}] \log p_\theta(r_0^i \mid p_0, r_l) \\
&\text{end for} \\
&\text{Return log\_likelihood} / n_{\text{mc}}
\end{aligned}
$$

### Algorithm 4: 随机重掩码采样

$$
\begin{aligned}
&\text{Require: } p_\theta,\; p_0,\; L,\; N \\
&r_1 \gets \text{全掩码序列, 长度 } L \\
&\text{for } t = 1, \frac{N-1}{N}, \ldots, \frac{1}{N} \text{ do} \\
&\qquad s = t - \frac{1}{N} \\
&\qquad \hat{r}_0 = \arg\max p_\theta(r_0 \mid p_0, r_t) \\
&\qquad \text{for } i = 1 \text{ to } L \text{ do} \\
&\qquad\qquad \text{if } r_t^i \neq \mathbf{M}:\; \hat{r}_0^i = r_t^i \\
&\qquad\qquad \text{else}:\; \hat{r}_0^i \gets \mathbf{M} \text{ w.p. } \frac{s}{t} \\
&\qquad r_s = \hat{r}_0 \\
&\text{Return } r_0
\end{aligned}
$$

### Algorithm 5: 低置信度重掩码采样

$$
\begin{aligned}
&\text{Require: } p_\theta,\; p_0,\; L,\; N \\
&r_1 \gets \text{全掩码序列, 长度 } L \\
&\text{for } t = 1, \frac{N-1}{N}, \ldots, \frac{1}{N} \text{ do} \\
&\qquad s = t - \frac{1}{N} \\
&\qquad \text{for } i = 1 \text{ to } L \text{ do} \\
&\qquad\qquad \text{if } r_t^i \neq \mathbf{M}:\; \hat{r}_0^i = r_t^i,\; c_i = 1 \\
&\qquad\qquad \text{else}:\; \hat{r}_0^i = \arg\max p_\theta(\cdot \mid p_0, r_t),\; c_i = p_\theta(\hat{r}_0^i \mid p_0, r_t) \\
&\qquad n_{\text{un}} = \lfloor L(1 - s) \rfloor \\
&\qquad \hat{r}_0[\text{lowest\_confidence\_indices}] = \mathbf{M} \\
&\qquad r_s = \hat{r}_0 \\
&\text{Return } r_0
\end{aligned}
$$

---

## 术语表

| 英文（论文） | 中文 | 数学符号 | 简释 |
|:---|:---|:---:|:---|
| Autoregressive Model | 自回归模型 | $p_\theta(x) = \prod p_\theta(x^i \mid x^{<i})$ | 按顺序逐 token 生成 |
| Masked Diffusion Model | 掩码扩散模型 | $p_\theta(x_0)$ 通过正反向过程定义 | 并行预测掩码 token |
| Forward Process | 前向过程 | $q_{t|0}(x_t \mid x_0)$ | 逐渐掩码数据 |
| Reverse Process | 反向过程 | $p_\theta(x_0)$ 通过迭代去噪 | 从全掩码恢复数据 |
| Mask Predictor | 掩码预测器 | $p_\theta(\cdot \mid x_t)$ | 预测掩码 token 的 Transformer |
| Variational Upper Bound | 变分上界 | $\mathcal{L}(\theta) \geq -\log p_\theta(x_0)$ | 可优化的似然上界 |
| Remasking | 重掩码 | 以概率 $s/t$ 重新掩码 | 保证前向-反向一致性 |
| Low-confidence Remasking | 低置信度重掩码 | 选择置信度最低的 token 重掩码 | 类似退火采样 |
| Reversal Curse | 逆转诅咒 | $A \to B$ 学习 ≠ $B \to A$ 推理 | 单向分解的固有限制 |
| AO-ARM | 任意阶自回归模型 | 所有排列顺序的自回归 | 与 MDM 等价的框架 |
| CFG | 无分类器引导 | $\tilde{p} \propto p_\theta^{1+w} / p_\theta^w$ | 增强条件信号的技术 |

---

> **参考文献**: Nie, S., Zhu, F., You, Z., et al. (2025). Large Language Diffusion Models. *NeurIPS 2025*. arXiv:2502.09992v3.
>
> **讲义编制说明**: 本文档以高校数学/统计专业研究生课程讲义风格撰写，每节包含背景动机、定义定理、公式推导与解释讲义，旨在系统化呈现 LLaDA 的理论基础与技术细节。
