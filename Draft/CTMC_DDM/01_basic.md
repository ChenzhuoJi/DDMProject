# 3.1 基础知识

在建立连续时间离散扩散模型之前，有必要回顾连续时间马尔可夫链的基本理论。这些内容既是《随机过程》课程的核心知识，也是后续所有推导的数学基础。本节将从扩散模型的实际需求出发，系统梳理 CTMC 的定义、基本方程、转移概率的闭式表达，以及时间反转等关键概念，并建立全文统一使用的数学记号。

## 3.1.1 数学记号

**状态空间与数据**。考虑一个有限离散状态空间 $\mathcal{S} = \{1, 2, \dots, K\}$，其中 $K$ 为状态总数。一条具有 $D$ 个维度的数据样本表示为 $\boldsymbol{x} = (x^1, x^2, \dots, x^D) \in \mathcal{S}^D$。在文本生成场景中，每个维度对应一个 token 位置，$K$ 为词表大小；在像素级图像生成中，每个维度对应一个像素位置，$K = 256$ 为像素值的离散级数。我们将时刻 $t$ 的随机变量记为 $\boldsymbol{x}_t$，其中 $t$ 为连续时间，取值范围为 $[0, T]$，$T$ 为终止时刻（通常取 $T = 1$）。

**概率分布**。用 $q_t(\boldsymbol{x})$ 表示前向过程在时刻 $t$ 的边缘概率分布，$q_{t|s}(\boldsymbol{x}_t \mid \boldsymbol{x}_s)$ 表示从时刻 $s$ 到 $t$ 的转移核。特别地，$q_{t|0}(\boldsymbol{x}_t \mid \boldsymbol{x}_0)$ 表示从初始干净数据出发经过时间 $t$ 后的分布。模型分布的记号采用 $p_\theta$，其中 $\theta$ 为可学习参数。神经网络输出去噪预测表示为 $p_{0|t}^\theta(\boldsymbol{x}_0 \mid \boldsymbol{x}_t)$，即在给定带噪数据 $\boldsymbol{x}_t$ 的条件下对原始干净数据 $\boldsymbol{x}_0$ 的预测分布，这一参数化方式源自 *Austin et al. (D3PM) §2* 的 $x_0$ 参数化，在连续时间框架中沿用自 *Campbell et al. (CTMC) §3.1*。

**矩阵基础**。速率矩阵（又称生成元矩阵）记为 $R_t \in \mathbb{R}^{K \times K}$，其元素 $[R_t]_{ij} = R_t(i, j)$ 表示时刻 $t$ 从状态 $i$ 跳转到状态 $j$ 的瞬时速率，这一记法沿用 *Campbell et al. (CTMC) §3.1*。离散时间的单步转移矩阵记为 $Q_t$，其元素 $[Q_t]_{ij} = q(x_t = j \mid x_{t-1} = i)$，这一记法沿用 *Austin et al. (D3PM) §2*。累积转移矩阵 $\overline{Q}_{t} = Q_1 Q_2 \cdots Q_t$ 刻画从初始时刻到时刻 $t$ 的多步转移。

**噪声调度相关量**。连续时间噪声调度函数记为 $\beta(t)$，其累积积分 $\overline{\beta}_{t|s} = \int_s^t \beta(\tau) \, d\tau$，这一记法沿用 *Zhao et al. (UD3) §2.1*。离散时间的噪声参数 $\beta_t$ 与连续时间的关联为 $\beta_t = 1 - \exp\bigl(-\int_{t-1}^t \beta(\tau)\,d\tau\bigr)$，即连续速率在单位时间间隔上的累积，参见 *Campbell et al. (CTMC) §4.1*。

**基本向量与矩阵**。$\mathbf{1}$ 表示全 1 列向量，$\mathbf{e}_i$ 表示第 $i$ 个标准基向量（$i$ 处为 1，其余为 0），$\delta_{ij}$ 为 Kronecker delta 符号，$\odot$ 表示逐元素乘积，$\langle \cdot, \cdot \rangle$ 表示向量内积。分类分布记为 $\text{Cat}(\boldsymbol{x}; \boldsymbol{p})$，其中 $\boldsymbol{p}$ 为概率向量。为简明起见，当一个行向量与一个转移矩阵相乘时，我们默认使用行向量左乘的约定，即 $\boldsymbol{x}_t \sim \text{Cat}(\boldsymbol{x}_{t-1} Q_t)$。

## 3.1.2 连续时间马尔可夫链的基本定义

连续时间马尔可夫链 $(\boldsymbol{x}_t)_{t \ge 0}$ 是一个在有限状态空间 $\mathcal{S}^D$ 上取值的随机过程，满足马尔可夫性：对于任意 $0 \le s \le t$ 和任意状态 $\boldsymbol{y} \in \mathcal{S}^D$，

$$
\mathbb{P}(\boldsymbol{x}_t = \boldsymbol{y} \mid \{\boldsymbol{x}_u : u \le s\}) = \mathbb{P}(\boldsymbol{x}_t = \boldsymbol{y} \mid \boldsymbol{x}_s).
\tag{3.1}
$$

与离散时间马尔可夫链不同，CTMC 的状态可以在任意连续时刻发生变化，而非仅在预设的离散时间步上。这一特性使其天然适合描述连续的噪声注入过程：在扩散模型的语境中，我们希望噪声随时间连续地、渐进地破坏数据结构，而非仅在有限个时间离散点上发生跳变。

**速率矩阵的定义与性质**。CTMC 的局部动力学完全由一个 $K \times K$ 的速率矩阵 $R_t$ 刻画。对于 $i \neq j$，$R_t(i, j) \ge 0$ 表示从状态 $i$ 到 $j$ 的单位时间跳转强度；对角线元素则定义为

$$
R_t(i, i) = -\sum_{j \neq i} R_t(i, j),
\tag{3.2}
$$

以保证矩阵的每一行之和为零。这一行和为零的条件是概率守恒的自然体现：从某状态出发的"总流出速率"必须等于该状态自身的"总流入速率"的相反数，因为概率的总和始终为 1。

速率矩阵的直观含义可以通过无限小时间间隔上的转移概率来理解，这一关系见 *Campbell et al. (CTMC) §3.1*。在长度为 $\Delta t$ 的极限短的时间内，

$$
q_{t+\Delta t \mid t}(j \mid i) = \delta_{ij} + R_t(i, j) \Delta t + o(\Delta t), \quad \Delta t \to 0.
\tag{3.3}
$$

换言之，对于 $j \neq i$，过程以概率 $R_t(i, j) \Delta t + o(\Delta t)$ 从 $i$ 跳转到 $j$；以概率 $1 + R_t(i, i) \Delta t + o(\Delta t)$ 停留在 $i$。这里的 $o(\Delta t)$ 表示比 $\Delta t$ 更高阶的无穷小量，$\delta_{ij}$ 当 $i = j$ 时为 1，否则为 0。

## 3.1.3 Kolmogorov 方程

CTMC 的边缘分布随时间的演化服从 Kolmogorov 前向方程（亦称 Fokker-Planck 方程），见 *Campbell et al. (CTMC) Appendix A*：

$$
\frac{\partial}{\partial t} q_t(\boldsymbol{x}) = \sum_{\boldsymbol{y}} q_t(\boldsymbol{y}) R_t(\boldsymbol{y}, \boldsymbol{x}).
\tag{3.4}
$$

这一方程的直观含义是：时刻 $t$ 处于状态 $\boldsymbol{x}$ 的概率密度的时间变化率，等于从所有其他状态 $\boldsymbol{y}$ 流入 $\boldsymbol{x}$ 的速率之和减去从 $\boldsymbol{x}$ 流出到所有其他状态的速率之和。若写成矩阵形式，以 $q_t$ 为行向量，则式 (3.4) 可简洁地写为

$$
\frac{d}{dt} q_t = q_t R_t.
\tag{3.5}
$$

对应的 Kolmogorov 后向方程为

$$
\frac{\partial}{\partial t} q_{t|s}(\boldsymbol{x}_t \mid \boldsymbol{x}_s) = -\sum_{\boldsymbol{y}} R_t(\boldsymbol{x}_t, \boldsymbol{y}) \, q_{t|s}(\boldsymbol{y} \mid \boldsymbol{x}_s),
\tag{3.6}
$$

其矩阵形式为 $\frac{d}{dt} P_{s,t} = -R_t P_{s,t}$，其中 $[P_{s,t}]_{ij} = q_{t|s}(j \mid i)$。

在扩散模型的框架中，前向方程描述了噪声如何随时间逐渐破坏原始数据的结构，而后向方程则在反向过程的推导中扮演关键角色——时间反转过程的速率矩阵正是通过前向与后向方程的结合导出的。

## 3.1.4 转移概率与矩阵指数

对于时间齐次的情形，即速率矩阵 $R$ 与时间 $t$ 无关，转移概率矩阵可以解析地表示为矩阵指数：

$$
P(t) = \exp(tR) = \sum_{n=0}^{\infty} \frac{(tR)^n}{n!}.
\tag{3.7}
$$

其元素 $[P(t)]_{ij} = q_{t|0}(j \mid i)$ 给出了从初态 $i$ 出发经过时间 $t$ 后处于状态 $j$ 的概率。矩阵指数满足半群性质 $P(s + t) = P(s) P(t)$，并且满足 Kolmogorov 前向方程 $\frac{d}{dt} P(t) = P(t) R$。

当速率矩阵与时间相关时，情况更为复杂。扩散模型中通常使用一种具有可分离结构的形式，见 *Campbell et al. (CTMC) §4.1* 和 *Zhao et al. (UD3) §2.1*：

$$
R_t = \beta(t) R_b,
\tag{3.8}
$$

其中 $\beta(t) > 0$ 是一个标量函数，控制噪声注入的速率随时间的变化，$R_b$ 是一个固定的基速率矩阵，决定了跳转的结构模式。在这一设定下，转移概率仍然具有闭式表达：

$$
q_{t|s}(\boldsymbol{x}_t \mid \boldsymbol{x}_s) = \exp\!\Bigl( \overline{\beta}_{t|s} R_b \Bigr), \quad \overline{\beta}_{t|s} = \int_s^t \beta(\tau)\, d\tau.
\tag{3.9}
$$

式 (3.9) 的核心在于：对矩阵指数 $\exp(\alpha R_b)$ 的计算，其中 $\alpha = \overline{\beta}_{t|s}$ 是一个标量。当 $R_b$ 具有简单的代数结构（如幂等性 $R_b^2 \propto R_b$）时，矩阵指数可以进一步化简为仅含两项的解析形式。这一性质将成为后续统一离散与连续框架的关键。

## 3.1.5 平稳分布与可逆性

对于不可约、非周期的 CTMC，当 $t \to \infty$ 时，$q_t$ 收敛到平稳分布 $\pi$，满足 $\pi R = 0$。在扩散模型中，平稳分布对应"完全噪声"的状态分布——前向过程的终点 $q_T$ 应接近于一个易于采样的简单分布（如均匀分布）。

若 CTMC 满足细致平衡条件

$$
\pi(i) R(i, j) = \pi(j) R(j, i), \quad \forall i \neq j,
\tag{3.10}
$$

则称该链关于 $\pi$ 是可逆的。可逆性意味着在稳态下，从 $i$ 到 $j$ 的概率流等于从 $j$ 到 $i$ 的概率流，从而正向过程与时间反转过程在分布上不可区分。这一概念在生成模型的时间反转构造中扮演了基础性角色。

## 3.1.6 与离散时间马尔可夫链的关系

CTMC 与离散时间马尔可夫链之间存在自然的联系，见 *Austin et al. (D3PM) Appendix A*。给定一个 CTMC 及其速率矩阵 $R$，以任意 $\Delta t > 0$ 为步长，离散时间转移矩阵为

$$
Q = \exp(\Delta t R) = I + R \Delta t + o(\Delta t).
\tag{3.11}
$$

反之，给定一个离散时间转移矩阵 $Q$，可以寻找其生成元 $R$ 使得 $Q = \exp(R)$。当 $\Delta t$ 很小时，一阶近似 $Q \approx I + R \Delta t$ 成立，这恰好对应了连续过程的欧拉离散化。在 D3PM 等离散时间模型中，$Q_t$ 被直接参数化；而在连续时间框架中，我们则直接定义速率矩阵 $R_t$，再通过式 (3.11) 导出离散化方案。两种视角的统一构成了本报告第 3.6 节的核心主题。

## 3.1.7 时间反转

CTMC 的一个深刻性质是，其时间反转过程仍然是一个 CTMC，这是扩散模型反向过程构造的出发点。设正向过程为 $(\boldsymbol{x}_t)_{t \in [0, T]}$，定义时间反转过程 $(\bar{\boldsymbol{x}}_t)_{t \in [0, T]}$ 为 $\bar{\boldsymbol{x}}_t = \boldsymbol{x}_{T - t}$。反转过程的速率矩阵 $\hat{R}_t$ 与正向速率矩阵 $R_t$ 之间存在如下关系，由 *Campbell et al. (CTMC) §3.1 Proposition 1* 给出：

$$
\hat{R}_t(\boldsymbol{x}, \tilde{\boldsymbol{x}}) = R_{T-t}(\tilde{\boldsymbol{x}}, \boldsymbol{x}) \frac{q_{T-t}(\tilde{\boldsymbol{x}})}{q_{T-t}(\boldsymbol{x})}, \quad \boldsymbol{x} \neq \tilde{\boldsymbol{x}}.
\tag{3.12}
$$

其直观含义是：反向过程从 $\tilde{\boldsymbol{x}}$ 跳转到 $\boldsymbol{x}$ 的速率，等于正向过程中从 $\boldsymbol{x}$ 跳转到 $\tilde{\boldsymbol{x}}$ 的速率乘以目标状态与当前状态在正向过程中的边缘概率之比。当正向过程将数据逐渐破坏为均匀噪声时，这个比率自然引导反向过程从噪声向数据恢复。然而，式 (3.12) 中的边缘概率 $q_t$ 通常是不可知的，这正是需要引入神经网络进行近似的原因——我们将在第 3.3 节中详细展开这一思路。
