# 3.4 连续时间 ELBO 推导

在定义了反向过程的参数化形式 $\hat{R}_t^\theta$ 之后，下一个核心问题是如何训练这个模型。训练的目标是使生成分布 $p_0^\theta$ 尽可能接近真实数据分布 $p_{\text{data}}$。在扩散模型中，这一目标通过最大化证据下界（Evidence Lower Bound, ELBO）来实现。本节将从变分原理出发，回顾离散时间 ELBO，然后推导其连续时间极限，得到连续时间 ELBO 的闭式表达，最后讨论其实用近似形式。

## 3.4.1 变分下界：从似然到 ELBO

记生成模型在数据 $x_0$ 上的对数似然为 $\log p_0^\theta(x_0)$。由于 $p_0^\theta$ 涉及对反向过程所有可能路径的积分，直接计算是极困难的。变分方法引入一个已知的、易处理的前向过程 $q$ 作为辅助分布，将对数似然分解为 ELBO 和 KL 散度之和：

$$
-\log p_0^\theta(x_0) \leq \mathbb{E}_{q(x_{1:K} \mid x_0)}\left[-\log \frac{p_{0:K}^\theta(x_{0:K})}{q(x_{1:K} \mid x_0)}\right] =: \mathcal{L}_{\text{DT}}(\theta).
\tag{3.37}
$$

不等号来源于 Jensen 不等式——左侧是负对数似然，右侧是它的变分上界。因此，最小化 $\mathcal{L}_{\text{DT}}$ 等价于最大化对数似然的下界。

在离散时间扩散模型中，这一 ELBO 可以进一步分解为三项之和，由 *Austin et al. (D3PM) §2* 给出：

$$
\begin{aligned}
\mathcal{L}_{\text{DT}}(\theta) = \mathbb{E}_{p_{\text{data}}(x_0)}\Big[
&\underbrace{D_{\text{KL}}(q(x_T \mid x_0) \parallel p_{\text{ref}}(x_T))}_{L_T} \\
&+ \sum_{t=2}^T \underbrace{\mathbb{E}_{q(x_t \mid x_0)}\big[D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \parallel p_\theta(x_{t-1} \mid x_t))\big]}_{L_{t-1}} \\
&+ \underbrace{\big(-\mathbb{E}_{q(x_1 \mid x_0)}[\log p_\theta(x_0 \mid x_1)]\big)}_{L_0}
\Big].
\end{aligned}
\tag{3.38}
$$

其中 $L_T$ 衡量前向终点的分布与先验之间的差距，$L_{t-1}$ 衡量每步反向转移的近似误差，$L_0$ 是最后一步的似然项。在连续时间框架中，我们将取 $T \to \infty$ 且时间步长 $\Delta t \to 0$ 的极限，此时 $L_T$ 项消失（因为 $q_T$ 充分接近 $p_{\text{ref}}$），而 $L_{t-1}$ 项的求和转化为一个关于时间的积分。这正是下一小节的核心内容。

## 3.4.2 用速率矩阵表示离散 ELBO 的局部项

为推导连续时间极限，我们需要将离散 ELBO 中的每项 KL 散度用速率矩阵的语言重新表达。将时间区间 $[0, T]$ 均匀划分为 $K$ 步，步长 $\Delta t = T/K$，离散时间步索引为 $k = 0, 1, \dots, K$。相邻时间步之间的转移核可以用 $R_t$ 和 $\hat{R}_t^\theta$ 展开为无限小形式。

考虑离散 ELBO 中的一个典型项。对于 $k \ge 1$，由 *Campbell et al. (CTMC) Appendix B.2*，反向转移的近似 $p_{k|k+1}^\theta$ 和前向转移 $q_{k+1|k}$ 可以展开为

$$
p_{k|k+1}^\theta(x_k \mid x_{k+1}) = \delta_{x_k, x_{k+1}} + \hat{R}_k^\theta(x_{k+1}, x_k) \Delta t + o(\Delta t),
\tag{3.39}
$$

$$
q_{k+1|k}(x_{k+1} \mid x_k) = \delta_{x_k, x_{k+1}} + R_k(x_k, x_{k+1}) \Delta t + o(\Delta t).
\tag{3.40}
$$

**这里每一步的目的是**：建立起离散时间核与连续时间速率矩阵之间的桥梁，使得我们可以将每步的 KL 散度写为关于 $\Delta t$ 的显式展开，从而为取极限做好准备。

将式 (3.39) 取对数，得到两项的贡献分离：

$$
\log p_{k|k+1}^\theta(x_k \mid x_{k+1}) = \delta_{x_k, x_{k+1}} \hat{R}_k^\theta(x_k, x_k) \Delta t + (1 - \delta_{x_k, x_{k+1}}) \log\big(\hat{R}_k^\theta(x_{k+1}, x_k) \Delta t + o(\Delta t)\big) + o(\Delta t).
\tag{3.41}
$$

当 $x_k = x_{k+1}$ 时，对数概率正比于 $\hat{R}_k^\theta \Delta t$；当 $x_k \neq x_{k+1}$ 时，则涉及 $\log(\hat{R}_k^\theta \Delta t)$ 项。将后者的对数进一步展开：

$$
\log(\hat{R}_k^\theta \Delta t + o(\Delta t)) = \log \Delta t + \log \hat{R}_k^\theta + o(1).
\tag{3.42}
$$

这一分离是推导的关键：$\log \Delta t$ 项与 $\theta$ 无关，在 $\Delta t \to 0$ 时趋于负无穷，但它将被期望操作中的同阶无穷小抵消，最终仅贡献一个可忽略的加性常数。

将式 (3.40) 和式 (3.41) 代入 $L_{k}$ 的表达式，展开并保留主导阶项，经过整理后得到：

$$
L_k = -\mathbb{E}_{q_k(x_k)}\Big[ \hat{R}_k^\theta(x_k, x_k) \Delta t + \sum_{x_{k+1} \neq x_k} R_k(x_k, x_{k+1}) \Delta t \log \hat{R}_k^\theta(x_{k+1}, x_k) + o(\Delta t) \Big] + C.
\tag{3.43}
$$

这里 $C$ 包含所有与 $\theta$ 无关的常数项。式 (3.43) 具有清晰的解释：第一项 $\hat{R}_k^\theta(x_k, x_k)$ 是反向过程中"停留在原处"的负速率（因为对角线是负值），鼓励反向过程在应该停留时不要跳转；第二项是对正向观察到的跳转施加对数似然惩罚，确保反向过程在跳转发生时选择正确的方向。

## 3.4.3 取连续极限

现在我们取 $\Delta t \to 0$（等价于 $K \to \infty$）的极限。离散求和转化为积分，时间索引 $k$ 变为连续时间 $t$。累积所有 $L_k$ 项并取极限，得到连续时间 ELBO 的闭式表达，由 *Campbell et al. (CTMC) §3.2 Proposition 2* 给出：

$$
\mathcal{L}_{\text{CT}}(\theta) = T \,\mathbb{E}_{t \sim \mathcal{U}(0,T),\; q_t(x),\; r_t(\tilde{x} \mid x)}\Big[ \Big\{\sum_{x' \neq x} \hat{R}_t^\theta(x, x')\Big\} - \mathcal{Z}^t(x) \log \big(\hat{R}_t^\theta(\tilde{x}, x)\big) \Big] + C.
\tag{3.44}
$$

其中引入了两个辅助量来简化表达式：

$$
\mathcal{Z}^t(x) = \sum_{x' \neq x} R_t(x, x'),
\qquad
r_t(\tilde{x} \mid x) = \frac{R_t(x, \tilde{x})}{\mathcal{Z}^t(x)}\; (\tilde{x} \neq x).
\tag{3.45}
$$

$\mathcal{Z}^t(x)$ 是从状态 $x$ 出发的总跳出速率，$r_t(\tilde{x} \mid x)$ 是在发生一次跳转的条件下跳转到 $\tilde{x}$ 的条件概率分布。因此，期望 $\mathbb{E}_{r_t(\tilde{x}|x)}$ 正是对"正向过程发生一次跳转的目的地"取平均。

**直观理解**。式 (3.44) 由两项构成，它们分别对应反向过程的两种行为：

- **第一项 $\sum_{x' \neq x} \hat{R}_t^\theta(x, x')$**：这是反向过程中从 $x$ 跳出的总速率。由于 $\hat{R}_t^\theta(x, x)$ 是负的，最大化 $\hat{R}_t^\theta(x, x')$ 的求和等价于最小化反向过程的"原地停留"速率。换言之，当反向过程不应当跳转时，我们希望它的总跳转速率趋近于零。

- **第二项 $\mathcal{Z}^t(x) \log \hat{R}_t^\theta(\tilde{x}, x)$**：这是对正向过程实际发生的一次跳转（从 $x$ 到 $\tilde{x}$）的监督。它鼓励反向过程在观察到正向跳转时，从 $\tilde{x}$ 跳回 $x$ 的速率尽可能大。$\mathcal{Z}^t(x)$ 作为权重，确保跳转越频繁的时刻对训练的贡献越大。

两项之间形成了一种平衡：反向过程不应无理由地跳转（第一项的惩罚），但当正向过程确实发生了跳转时，反向过程必须学会从跳转的目的地跳回来（第二项的鼓励）。

**另一种推导路径**。除了上述的离散极限方法，*Campbell et al. (CTMC) Appendix B.2* 还提供了另一种基于路径测度的推导。该路径将前向和反向过程视为路径空间上的概率测度 $\mathbb{Q}$ 和 $\mathbb{P}^\theta$，利用 Cameron-Martin-Girsanov 公式写出两个 CTMC 路径测度之间的 Radon-Nikodym 导数，再通过 Dynkin 引理将跳跃求和转化为速率积分，最终得到与式 (3.44) 完全相同的形式。这种推导在数学上更为严格，但需要更多的随机分析工具。

## 3.4.4 高效单次前向近似

式 (3.44) 在实际训练中面临一个问题：期望操作同时涉及 $q_t(x)$ 和 $r_t(\tilde{x} \mid x)$，直觉上需要先采样 $x \sim q_t$，再采样 $\tilde{x} \sim r_t(\cdot \mid x)$，然后对 $x$ 和 $\tilde{x}$ 分别计算 $\hat{R}_t^\theta$，即需要两次网络前向。这增加了训练成本。

*Campbell et al. (CTMC) Appendix C.4* 提出了一个高效的单次前向近似。注意到 $q_t(x) r_t(\tilde{x} \mid x)$ 近似等于 $q_{t+\delta t}(\tilde{x})$，其中 $\delta t = 1 / \mathcal{Z}^t(\tilde{x})$ 是平均停留时间的量级——在图像数据上这一时间尺度约为 $10^{-6}T$ 到 $10^{-8}T$，非常微小。因此，将第一项中的 $x$ 替换为 $\tilde{x}$，可以得到近似：

$$
\mathcal{L}_{\text{eCT}}(\theta) = T \,\mathbb{E}_{t \sim \mathcal{U}(0,T),\; q_t(x),\; r_t(\tilde{x} \mid x)}\Big[ \Big\{\sum_{x' \neq \tilde{x}} \hat{R}_t^\theta(\tilde{x}, x')\Big\} - \mathcal{Z}^t(x) \log \big(\hat{R}_t^\theta(\tilde{x}, x)\big) \Big] + C.
\tag{3.46}
$$

**这样做的目的**：两项现在都只要求对 $\tilde{x}$ 进行一次网络前向，计算量减半。同时，作者在实验中发现 $\mathcal{L}_{\text{eCT}}$ 的性能甚至略优于 $\mathcal{L}_{\text{CT}}$，可能的原因是两项共享同一个 $\tilde{x}$ 降低了蒙特卡洛估计的方差。

## 3.4.5 辅助去噪损失

实践表明，仅使用 $\mathcal{L}_{\text{CT}}$ 训练可能不够稳定——它涉及 $\hat{R}_t^\theta$ 中对 $p_{0|t}^\theta$ 的间接优化，梯度信号相对隐晦。为此，*Campbell et al. (CTMC) Appendix D* 建议叠加一个直接的去噪交叉熵损失：

$$
L_{\text{ll}}(\theta) = T \,\mathbb{E}_{t \sim \mathcal{U}(0,T),\; p_{\text{data}}(x_0),\; q_{t|0}(x \mid x_0)}\big[ -\log p_{0|t}^\theta(x_0 \mid x) \big].
\tag{3.47}
$$

这个损失直接鼓励网络准确预测原始数据 $x_0$，与 D3PM 中的辅助交叉熵损失（*Austin et al. (D3PM) §2* 的混合损失）一脉相承。事实上，当 $p_{0|t}^\theta$ 等于真实后验 $q_{0|t}$ 时，$L_{\text{ll}}$ 达到最小值，见 *Campbell et al. (CTMC) Proposition 8*。

完整的训练目标为二者的加权组合：

$$
\min_\theta \mathcal{L}_{\text{CT}}(\theta) + \lambda L_{\text{ll}}(\theta),
\tag{3.48}
$$

其中 $\lambda$ 是一个平衡超参数。在 CIFAR-10 实验中通常取 $\lambda = 0.001$。**这一组合的目的是**：$\mathcal{L}_{\text{CT}}$ 负责从变分角度修正反向过程的整体行为，而 $L_{\text{ll}}$ 直接引导 $p_{0|t}^\theta$ 的预测准确性，二者互补。

## 3.4.6 高维数据的因子化 ELBO

对于具有 $D$ 个维度的数据，利用第 3.2.3 节和第 3.3.4 节的维度分解，ELBO 可以进一步因子化。*Campbell et al. (CTMC) Appendix C.3 Proposition 7* 给出了因子化形式：

$$
\begin{aligned}
\mathcal{L}_{\text{CT}} = T \,\mathbb{E}_{t, p_{\text{data}}(\boldsymbol{x}_0^{1:D}), q_{t|0}(\boldsymbol{x}^{1:D} \mid \boldsymbol{x}_0^{1:D})}
&\Big[ \sum_{d=1}^D \sum_{x'^d \neq x^d} \hat{R}_t^{\theta d}(\boldsymbol{x}^{1:D}, x'^d) \Big] \\
&- T \,\mathbb{E}_{t, p_{\text{data}}(\boldsymbol{x}_0^{1:D}), \psi_t(\tilde{\boldsymbol{x}}^{1:D} \mid \boldsymbol{x}_0^{1:D})}
\Big[ \sum_{d=1}^D \sum_{x^d \neq \tilde{x}^d} \phi_t(x^d \mid \tilde{\boldsymbol{x}}^{1:D}, \boldsymbol{x}_0^{1:D}) \mathcal{Z}^t(\tilde{\boldsymbol{x}}^{1:D/d} \circ x^d) \log \hat{R}_t^{\theta d}(\tilde{\boldsymbol{x}}^{1:D}, x^d) \Big] + C.
\end{aligned}
\tag{3.49}
$$

**引入这一形式的目的**是避免在全空间 $\mathcal{S}^D$ 上求和，转而利用维度分解将计算复杂度降到 $O(DK)$。虽然表达式看似复杂，但实现时只需对每个维度独立计算 $\hat{R}_t^{\theta d}$，而 $\hat{R}_t^{\theta d}$ 本身又通过单次网络前向得到。因此，整个 ELBO 的计算可以高效地实施。
