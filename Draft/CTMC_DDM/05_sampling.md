# 3.5 反向采样算法

在定义了反向速率矩阵 $\hat{R}_t^\theta$ 之后，我们面临一个计算问题：如何从这一高维 CTMC 中高效地采样生成样本？理论上，任何 CTMC 都可以通过 Gillespie 算法精确模拟——每步从指数分布中采样停留时间，再按转移概率选择下一个状态。然而，在扩散模型中，数据维度 $D$ 通常为数千甚至数万，而 Gillespie 算法每步只改变一个维度，生成一张 $32 \times 32$ 的图像就需要数百万步——这是完全不可行的。本节介绍两种从反向 CTMC 高效采样的技术：tau-leaping 算法和 Predictor-Corrector 采样器，并讨论生成分布的误差界。

## 3.5.1 Tau-Leaping 算法

Tau-leaping 的核心思想是牺牲局部的精确性来换取全局的效率。与 Gillespie 算法每步只处理一次跳转不同，tau-leaping 假设在时间窗口 $[t-\tau, t]$ 内速率矩阵 $\hat{R}_t^{\theta, 1:D}$ 和当前状态 $\boldsymbol{x}_t^{1:D}$ 保持恒定，然后在此窗口内所有可能发生的跳转**批量处理**：对每种可能的跳转，其发生次数服从 Poisson 分布，由 *Campbell et al. (CTMC) §4.3* 给出。

**引入这一思想的目的是**：将原本逐个处理的 $D$ 维指数分布采样，转化为一次性的 Poisson 采样，从而利用并行计算大幅加速。

具体而言，利用第 3.3.4 节的维度分解——反向过程每次只改变一个维度——我们可以对每个维度 $d$ 和每个目标状态 $s \neq x_t^d$，独立采样 Poisson 计数：

$$
P_{ds} \sim \text{Poisson}\Big( \tau \,\hat{R}_t^{\theta,d}(\boldsymbol{x}_t^{1:D}, s) \Big),
\tag{3.50}
$$

其中 $\tau$ 是 tau-leaping 的步长，$\hat{R}_t^{\theta,d}$ 由式 (3.35) 给出。采样完成后，按如下规则更新状态：

$$
\boldsymbol{x}_{t-\tau}^{1:D} = \boldsymbol{x}_t^{1:D} + \sum_{d=1}^D \sum_{s \neq x_t^d} P_{ds} \, (s - x_t^d) \, \boldsymbol{e}^d,
\tag{3.51}
$$

其中 $\boldsymbol{e}^d$ 是第 $d$ 个维度的单位向量。这一更新的含义是：对每个维度 $d$，检查该维度上所有可能的目标状态 $s$ 的 Poisson 计数 $P_{ds}$，然后将每个计数对应的跳转施加到当前状态上。

**类别数据的特殊处理**。对于文本等名义数据（即状态之间的标签没有序数关系），在同一维度上发生多次跳转是没有意义的——从 token A 跳到 B 再跳到 C，与直接从 A 跳到 C 不同，而状态标签的任意性使得这种区分无意义。因此，当同一维度上的 Poisson 计数之和超过 1 时，对该维度放弃本次更新：

$$
\boldsymbol{x}_{t-\tau}^d \leftarrow \boldsymbol{x}_t^d, \quad \text{若} \sum_{s \neq x_t^d} P_{ds} > 1.
\tag{3.52}
$$

在实际中，采用均匀速率矩阵时，这一拒绝率非常低——因为 Poisson 分布的均值 $\tau \hat{R}_t^{\theta,d}$ 很小，同一维度出现多次跳转的概率可以忽略。对于图像等有序数据，则无需此拒绝规则，因为多次跳转可以累积为有意义的数值变化。

**步长与计算量**。Tau-leaping 的每次迭代只需要一次网络前向来计算 $p_{0|t}^\theta$，随后对 $D \times (K-1)$ 种候选跳转计算 Poisson 计数。单次迭代的计算量与 $D$ 呈线性关系，且完全可并行。记 $T$ 为总时间（通常取 $T=1$），则总的网络前向次数（NFE）为

$$
\text{NFE} = \frac{T}{\tau}.
\tag{3.53}
$$

步长 $\tau$ 控制着效率与精度之间的权衡：$\tau$ 越大，每次迭代跨越的时间越长，NFE 越小，但 tau-leaping 的"速率恒定"假设越不准确；$\tau \to 0$ 时则恢复为精确的 Gillespie 模拟。

## 3.5.2 Predictor-Corrector 采样器

Tau-leaping 提供了从 $\hat{R}_t^\theta$ 采样的手段，但仅此一步可能不够精确——因为 $\hat{R}_t^\theta$ 只是真实反向速率 $\hat{R}_t$ 的近似，每次 tau-leaping 都会引入近似误差。Predictor-Corrector 采样器通过引入额外的高斯蒙特卡洛步骤来修正这一偏差，其核心发现由 *Campbell et al. (CTMC) §4.4 Proposition 4* 给出：

**校正子速率的不变性**。对于前向 CTMC 的速率 $R_t$ 和精确反向速率 $\hat{R}_t$，它们的和

$$
R_t^c = R_t + \hat{R}_t
\tag{3.54}
$$

以 $q_t$ 为平稳分布。这一结论的直接推论是：当我们用 $\hat{R}_t^\theta$ 近似 $\hat{R}_t$ 时，组合速率 $R_t^{c\theta} = R_t + \hat{R}_t^\theta$ 近似以 $q_t$ 为平稳分布。

**实际采样流程**（见 *Campbell et al. (CTMC) Appendix F.2*）交替执行以下两步：

1. **Prediction（预测步）**：使用反向速率 $\hat{R}_t^\theta$ 执行一次 tau-leaping，从 $t$ 推进到 $t-\tau$，生成一个初步的样本。
2. **Correction（校正步）**：使用校正子速率 $R_t^{c\theta} = \hat{R}_t^\theta + R_t$ 执行一次 tau-leaping，以当前时刻的 $q_t$ 为平稳目标进行马尔可夫链蒙特卡洛修正。

**校正步的目的是**：将反向采样过程中的近似误差向 $q_t$ 的方向"拉回"——由于 $R_t^{c\theta}$ 以 $q_t$ 为平稳分布，每次校正步都相当于在正确的边际分布下进行一次 MCMC 采样。这类似于连续扩散模型中 Predictor-Corrector 采样器的思想（Song et al., 2021），只不过此处使用的是离散状态空间中的速率矩阵而非连续空间中的随机微分方程。

在离散状态空间中，校正步的效果可以理解为：在 $\hat{R}_t^\theta$ 定义的反向分布基础上，通过叠加前向噪声 $R_t$ 使各维度的分类分布略微向均匀分布"退火"，从而抵消近似误差带来的偏差。

实验表明，校正步显著提升了样本质量。*Campbell et al. (CTMC)* 在 CIFAR-10 上的结果如表 3.1 所示：

| 方法 | IS ($\uparrow$) | FID ($\downarrow$) |
|------|:-:|:-:|
| D3PM Gaussian | 8.56 | 7.34 |
| τLDR-0（无校正步） | 8.74 | 8.10 |
| τLDR-10（10 校正步） | **9.49** | **3.74** |
| DDPM（连续） | 9.46 | 3.17 |

τLDR-0 在 IS 上已超越 D3PM，但 FID 略差；增加 10 次校正步后（仅在 $t < 0.1T$ 时使用），FID 从 8.10 降至 3.74，几乎追平了连续 DDPM 的水平。这一结果凸显了校正步在弥合离散与连续扩散模型性能差距中的关键作用。

## 3.5.3 误差界

为理解反向采样中的各类误差来源，*Campbell et al. (CTMC) §4.5 Theorem 1* 给出了生成分布 $\mathcal{L}(y_0)$ 与真实数据分布 $p_{\text{data}}$ 之间的总变差（Total Variation）误差上界：

$$
\begin{aligned}
\|\mathcal{L}(y_0) - p_{\text{data}}\|_{\text{TV}} \le\; &3MT \\
&+ \Big\{ \big(|R| S D C_1\big)^2 + \frac{1}{2} C_2 (M + C_1 S D |R|) \Big\} \tau T \\
&+ 2 \exp\!\Big\{ -\frac{T \log^2 2}{t_{\text{mix}} \log 4D} \Big\}.
\end{aligned}
\tag{3.55}
$$

**三部分误差来源的意图是**：将总误差分解为三个可独立分析的分量，分别对应反向速率近似误差、tau-leaping 离散化误差和初始分布不匹配误差。

三个误差项的意义如下：

- **第一项 $3MT$**：来源于学习到的反向速率 $\hat{R}_t^\theta$ 与精确反向速率 $\hat{R}_t$ 之间的差距。$M$ 衡量了每单位时间上两个速率矩阵行和的平均绝对差。这一项随总时间 $T$ 线性增长，因为近似误差在反向采样的过程中持续累积。

- **第二项 $\mathcal{O}((SD)^2 \tau T)$**：来源于 tau-leaping 的有限步长近似。当 $\tau \to 0$ 时该项消失，对应精确模拟。常数 $C_1$ 和 $C_2$ 分别控制边缘概率比值的界和速率的 Lipschitz 连续性（见 *Campbell et al. (CTMC) Appendix B.5* 的假设 2 和 3）。实际中，这一项会随维度 $D$ 和状态空间大小 $S$ 的平方增长，但由于 $S$ 通常较小（文本词表 $S \sim 10^4$ 但实际每步只考虑一个维度），且 $D$ 的影响是多项式而非指数级的，这使得 tau-leaping 在高维场景下仍然可行。

- **第三项 $\mathcal{O}(\exp(-T / (t_{\text{mix}} \log D)))$**：来源于参考分布 $p_{\text{ref}}$ 与真实前向终点分布 $q_T$ 之间的不匹配。$t_{\text{mix}}$ 是单维度 CTMC 的混合时间（mixing time）。由于 $T$ 通常取得足够大以确保前向过程已接近平稳，这一项的贡献可以忽略。

**误差界最重要的结论是**：总误差关于维度 $D$ 的增长最多是二次的（$\mathcal{O}(D^2)$），而非指数级的（$\mathcal{O}(S^D)$）。这意味着我们不需要为了在高维数据上获得近似的精确采样而将 $\tau$ 取到难以实现的小值。这一多项式可扩展性（polynomial scalability）是连续时间 CTMC 框架相较于某些离散时间方法的一个关键优势。

**对采样实践的建议**。综合以上分析，在实际采样中需要权衡三项误差：选择足够小的 $\tau$ 以控制离散化误差，选择合适的 $T$（通常 $T=1$ 已然足够）以使初始分布充分混合，同时通过增加校正步数来降低反向速率的近似误差。在计算预算固定时，Prediction 和 Correction 之间的步数分配是一个重要的超参数。
