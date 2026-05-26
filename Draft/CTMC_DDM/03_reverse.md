# 3.3 反向过程与参数化

在构造了前向噪声过程之后，我们的目标是构建一个反向生成过程，将纯噪声分布 $q_T$ 逐步恢复为数据分布 $q_0$。这要求我们回答两个问题：第一，给定前向 CTMC 的速率矩阵 $R_t$，精确的反向过程具有怎样的形式？第二，如何用可学习的神经网络近似这个反向过程？本节将依次处理这两个问题，并讨论反向过程的维度分解与离散时间对应。

## 3.3.1 精确反向速率

我们在第 3.1.7 节已经看到，一个 CTMC 的时间反转过程仍然是一个 CTMC。设前向过程为 $(\boldsymbol{x}_t)_{t \in [0, T]}$，其时间反转过程 $\bar{\boldsymbol{x}}_t = \boldsymbol{x}_{T-t}$ 的速率矩阵 $\hat{R}_t$ 与前向速率矩阵 $R_t$ 之间的关系由 *Campbell et al. (CTMC) §3.1 Proposition 1* 给出：

$$
\hat{R}_t(\boldsymbol{x}, \tilde{\boldsymbol{x}}) = R_{T-t}(\tilde{\boldsymbol{x}}, \boldsymbol{x}) \frac{q_{T-t}(\tilde{\boldsymbol{x}})}{q_{T-t}(\boldsymbol{x})}, \quad \boldsymbol{x} \neq \tilde{\boldsymbol{x}}.
\tag{3.26}
$$

在扩散模型的语境中，我们关心的是从 $T$ 时刻向 $0$ 时刻反向演化的过程。重新参数化时间坐标，令正向时间为 $t$，反向时间为 $T-t$，则在正向时间 $t$ 处的反向速率可以写为

$$
\hat{R}_t(\boldsymbol{x}, \tilde{\boldsymbol{x}}) = R_t(\tilde{\boldsymbol{x}}, \boldsymbol{x}) \frac{q_t(\tilde{\boldsymbol{x}})}{q_t(\boldsymbol{x})}, \quad \boldsymbol{x} \neq \tilde{\boldsymbol{x}}.
\tag{3.27}
$$

这里 $\hat{R}_t$ 是反向过程在"正向时间 $t$"处的速率——即从正向第 $t$ 时刻向第 $0$ 时刻演化的瞬时速率。这一公式的直觉是深刻的：反向过程跳转到某个状态的速率，等于正向过程从该状态跳出的速率，乘以该状态在当前时刻的边缘概率相对于当前状态的比率。换言之，当 $q_t(\tilde{\boldsymbol{x}}) \gg q_t(\boldsymbol{x})$ 时，即 $\tilde{\boldsymbol{x}}$ 在时刻 $t$ 远比 $\boldsymbol{x}$ 更可能时，反向过程倾向于从 $\boldsymbol{x}$ 跳转到 $\tilde{\boldsymbol{x}}$。

这一公式的推导可以从 Kolmogorov 后向方程出发。考虑联合分布 $q(\boldsymbol{x}_t, \boldsymbol{x}_{t+\Delta t})$，按照贝叶斯规则和无穷小展开，可以得到反向转移核的表达式；再取 $\Delta t \to 0$ 的极限，即得到式 (3.27)。完整的推导过程可以参见 *Campbell et al. (CTMC) Appendix B.1*。

式 (3.27) 虽然形式简洁，但其直接的实用性受到一个根本性障碍的限制：边缘分布 $q_t$ 通常是未知且不可计算的。$q_t$ 由前向过程从初始数据分布 $q_0$ 演化而来，但 $q_0$ 正是我们需要学习的未知数据分布。因此，必须引入近似。

## 3.3.2 基于去噪网络的反向速率参数化

为了获得可计算的反向速率，我们用 $q_{t|0}$ 将式 (3.27) 中的边缘概率比展开：

$$
\frac{q_t(\tilde{\boldsymbol{x}})}{q_t(\boldsymbol{x})} = \frac{\sum_{\boldsymbol{x}_0} q_{t|0}(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}_0) q_0(\boldsymbol{x}_0)}{\sum_{\boldsymbol{x}_0} q_{t|0}(\boldsymbol{x} \mid \boldsymbol{x}_0) q_0(\boldsymbol{x}_0)}.
\tag{3.28}
$$

虽然 $q_0$ 仍然未知，但我们注意到 $q_{t|0}$ 是已知的前向转移核（由 $R_t$ 唯一确定），而 $q_0(\boldsymbol{x}_0)$ 恰好是我们在生成过程中希望学习的对象。按照 *Austin et al. (D3PM) §2* 提出的 $x_0$ 参数化思路，我们引入一个神经网络 $p_{0|t}^\theta(\boldsymbol{x}_0 \mid \boldsymbol{x}_t)$，它试图从带噪数据 $\boldsymbol{x}_t$ 中预测出原始的干净数据 $\boldsymbol{x}_0$。利用这个网络，我们可以将式 (3.27) 中不可知的 $q_t(\tilde{\boldsymbol{x}})/q_t(\boldsymbol{x})$ 替换为可计算的估计，由 *Campbell et al. (CTMC) §3.1* 给出：

$$
\hat{R}_t^\theta(\boldsymbol{x}, \tilde{\boldsymbol{x}}) = R_t(\tilde{\boldsymbol{x}}, \boldsymbol{x}) \sum_{\boldsymbol{x}_0} \frac{q_{t|0}(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}_0)}{q_{t|0}(\boldsymbol{x} \mid \boldsymbol{x}_0)} p_{0|t}^\theta(\boldsymbol{x}_0 \mid \boldsymbol{x}), \quad \boldsymbol{x} \neq \tilde{\boldsymbol{x}}.
\tag{3.29}
$$

这一参数化的直观含义如下：给定当前状态 $\boldsymbol{x}$，网络预测可能的 $\boldsymbol{x}_0$ 上的分布 $p_{0|t}^\theta(\boldsymbol{x}_0 \mid \boldsymbol{x})$；对于每个候选的 $\boldsymbol{x}_0$，前向核的比值 $q_{t|0}(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}_0) / q_{t|0}(\boldsymbol{x} \mid \boldsymbol{x}_0)$ 衡量了从 $\boldsymbol{x}_0$ 出发，到达目标状态 $\tilde{\boldsymbol{x}}$ 相对于当前状态 $\boldsymbol{x}$ 的可能性；再将这两者加权求和，得到从 $\boldsymbol{x}$ 跳转到 $\tilde{\boldsymbol{x}}$ 的反向速率。

对角线元素由行和归零条件确定：

$$
\hat{R}_t^\theta(\boldsymbol{x}, \boldsymbol{x}) = -\sum_{\tilde{\boldsymbol{x}} \neq \boldsymbol{x}} \hat{R}_t^\theta(\boldsymbol{x}, \tilde{\boldsymbol{x}}).
\tag{3.30}
$$

至此，我们得到了一个完全可计算的反向过程近似。当 $p_{0|t}^\theta$ 精确地逼近真实的后验分布 $q_{0|t}$ 时，式 (3.29) 退化为精确反向速率式 (3.27)；而当 $p_{0|t}^\theta$ 存在误差时，它提供了对反向过程的一个变分近似。

## 3.3.3 去噪网络 $p_{0|t}^\theta$ 的结构

去噪网络 $p_{0|t}^\theta(\boldsymbol{x}_0 \mid \boldsymbol{x}_t)$ 是整个反向过程的核心引擎。它接收带噪数据 $\boldsymbol{x}_t$ 和时间 $t$ 作为输入，输出一个 $K$ 维概率向量，表示对原始干净数据 $\boldsymbol{x}_0$ 的预测分布。具体而言，对于文本数据，$p_{0|t}^\theta$ 通常基于 Transformer 架构，输出词表上的 softmax 分布；对于图像数据，则基于 UNet 架构，输出每个像素位置上 256 个像素值的分类分布。

网络输出的分布 $p_{0|t}^\theta$ 在各维度上是条件独立的——给定 $\boldsymbol{x}_t$ 的条件下，不同维度上的预测因子化：

$$
p_{0|t}^\theta(\boldsymbol{x}_0^{1:D} \mid \boldsymbol{x}_t^{1:D}) = \prod_{d=1}^D p_{0|t}^{\theta,d}(x_0^d \mid \boldsymbol{x}_t^{1:D}).
\tag{3.31}
$$

需要注意，虽然在给定 $\boldsymbol{x}_t$ 的条件下各维度的预测是独立的，但 $\boldsymbol{x}_t$ 本身包含了跨维度的信息。因此，网络需要通过自注意力等机制捕捉维度之间的依赖关系，从而实现对原始数据的联合预测。

训练目标的核心部分是鼓励 $p_{0|t}^\theta$ 对 $\boldsymbol{x}_0$ 的预测尽可能准确。这可以通过在每个时间步 $t$ 上最小化交叉熵损失来实现：

$$
L_{\text{CE}}(\theta) = \mathbb{E}_{t, q_{t|0}(\boldsymbol{x}_t \mid \boldsymbol{x}_0) q_0(\boldsymbol{x}_0)}\left[-\log p_{0|t}^\theta(\boldsymbol{x}_0 \mid \boldsymbol{x}_t)\right].
\tag{3.32}
$$

然而，完整的训练目标并非仅有这一交叉熵项——反向过程的总体训练是通过最大化连续时间 ELBO 来进行的，交叉熵损失将作为其中的一个重要组分。我们将在第 3.4 节中详细展开 ELBO 的推导。

## 3.3.4 反向过程的维度分解

如前所述，前向过程在各维度之间是独立的，一次跳转只能改变一个维度的状态。这一性质对于反向过程同样成立——因为反向过程的速率矩阵继承了正向速率矩阵的稀疏性结构。根据 *Campbell et al. (CTMC) §4.2 Proposition 3*，若前向过程因子化为各维度的独立 CTMC，则反向过程也具有相同的因子化结构：

$$
\hat{R}_t^{1:D}(\boldsymbol{x}^{1:D}, \tilde{\boldsymbol{x}}^{1:D}) = \sum_{d=1}^D \hat{R}_t^d(\boldsymbol{x}^{1:D}, \tilde{x}^d) \;\delta_{\tilde{\boldsymbol{x}}^{1:D \setminus d}, \boldsymbol{x}^{1:D \setminus d}}.
\tag{3.33}
$$

其中每个维度的偏反向速率为

$$
\hat{R}_t^d(\boldsymbol{x}^{1:D}, \tilde{x}^d) = R_t^d(\tilde{x}^d, x^d) \sum_{x_0^d} \frac{q_{t|0}(\tilde{x}^d \mid x_0^d)}{q_{t|0}(x^d \mid x_0^d)} \; q_{0|t}(x_0^d \mid \boldsymbol{x}^{1:D}).
\tag{3.34}
$$

这里的关键区别在于，虽然 $R_t^d$ 和 $q_{t|0}$ 都是各维度独立的，但 $q_{0|t}(x_0^d \mid \boldsymbol{x}^{1:D})$ 是**跨维度耦合**的——它需要利用所有维度的观测值 $\boldsymbol{x}^{1:D}$ 来推断单一维度 $d$ 的原始状态 $x_0^d$。这正是生成模型需要学习的内容：各维度之间的统计依赖关系。

将式 (3.34) 中的精确后验 $q_{0|t}$ 替换为网络预测 $p_{0|t}^\theta$，得到参数化的偏反向速率：

$$
\hat{R}_t^{\theta,d}(\boldsymbol{x}^{1:D}, \tilde{x}^d) = R_t^d(\tilde{x}^d, x^d) \sum_{x_0^d} \frac{q_{t|0}(\tilde{x}^d \mid x_0^d)}{q_{t|0}(x^d \mid x_0^d)} \; p_{0|t}^{\theta,d}(x_0^d \mid \boldsymbol{x}^{1:D}).
\tag{3.35}
$$

在实际实现中，式 (3.35) 的计算可以通过一次网络前向完成：给定当前完整的带噪数据 $\boldsymbol{x}^{1:D}$，网络输出每个维度上对 $\boldsymbol{x}_0$ 的预测分布 $p_{0|t}^{\theta,d}$；随后，对每个维度 $d$ 和每个候选状态 $\tilde{x}^d \neq x^d$，利用已知的前向核比值计算反向速率。因此，单次网络前向即可得到整个 $D \times (K-1)$ 维的反向速率张量。

## 3.3.5 与离散时间反向过程的联系

在 D3PM 等离散时间框架中，反向过程被参数化为

$$
p_\theta(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t) \propto \sum_{\boldsymbol{x}_0} q(\boldsymbol{x}_{t-1}, \boldsymbol{x}_t \mid \boldsymbol{x}_0) \, \tilde{p}_\theta(\boldsymbol{x}_0 \mid \boldsymbol{x}_t),
\tag{3.36}
$$

其中 $\tilde{p}_\theta$ 是去噪网络，$q(\boldsymbol{x}_{t-1}, \boldsymbol{x}_t \mid \boldsymbol{x}_0)$ 是已知的前向联合分布。这一参数化方式称为 $x_0$ 参数化，由 *Austin et al. (D3PM) §2* 提出。

连续时间参数化式 (3.29) 可以视为式 (3.36) 在 $\Delta t \to 0$ 时的极限。当时间步长趋于零时，$q(\boldsymbol{x}_{t-1}, \boldsymbol{x}_t \mid \boldsymbol{x}_0)$ 中的信息退化到仅包含"是否发生跳转"的局部行为，恰好由速率矩阵 $R_t$ 捕捉。两者的核心思想一致：利用去噪网络对 $\boldsymbol{x}_0$ 的预测来构造反向转移，区别仅在于连续时间框架中我们需要处理的是速率而非概率，在计算上需要针对 $\Delta t$ 下的局部行为进行适当缩放。

这一联系不仅验证了连续时间框架的合理性，也为后续第 3.4 节中从离散 ELBO 到连续 ELBO 的极限推导奠定了基础。
