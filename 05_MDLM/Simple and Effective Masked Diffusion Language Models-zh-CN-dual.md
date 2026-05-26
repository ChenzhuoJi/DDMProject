# Simple and Effective Masked Diffusion Language Models
简单有效的掩码扩散语言模型
# Subham Sekhar Sahoo

Cornell Tech, NYC, USA. [ssahoo@cs.cornell.edu](mailto:ssahoo@cs.cornell.edu)
纽约市，美国康奈尔科技学院。ssahoo@cs.cornell.edu

# Marianne Arriola

Cornell Tech, NYC, USA. [ma2238@cornell.edu](mailto:ma2238@cornell.edu)
康奈尔科技学院，纽约，美国。ma2238@cornell.edu

# Yair Schiff
亚伊尔·谢夫

Cornell Tech, NYC, USA. [yzs2@cornell.edu](mailto:yzs2@cornell.edu)
康奈尔科技学院，纽约，美国。yzs2@cornell.edu

# Aaron Gokaslan
亚伦·戈卡斯兰

Cornell Tech, NYC, USA. [akg87@cs.cornell.edu](mailto:akg87@cs.cornell.edu)
康奈尔科技学院，纽约，美国。akg87@cs.cornell.edu

# Edgar Marroquin
埃德加·马罗昆

Cornell Tech, NYC, USA. [emm392@cornell.edu](mailto:emm392@cornell.edu)
康奈尔科技学院，纽约，美国。emm392@cornell.edu

# Justin T Chiu
贾斯汀·T·周

Cornell Tech, NYC, USA. [jtc257@cornell.edu](mailto:jtc257@cornell.edu)
康奈尔科技学院，纽约，美国。jtc257@cornell.edu

# Alexander Rush
亚历山大·拉什

Cornell Tech, NYC, USA. [ar459@cornell.edu](mailto:ar459@cornell.edu)
康奈尔科技学院，纽约，美国。ar459@cornell.edu

# Volodymyr Kuleshov
沃洛迪米尔·库列绍夫

Cornell Tech, NYC, USA. [kuleshov@cornell.edu](mailto:kuleshov@cornell.edu)
康奈尔科技学院，纽约，美国。kuleshov@cornell.edu

# Abstract
摘要

While diffusion models excel at generating high-quality images, prior work reports a significant performance gap between diffusion and autoregressive (AR) methods in language modeling. In this work, we show that simple masked discrete diffusion is more performant than previously thought. We apply an effective training recipe that improves the performance of masked diffusion models and derive a simplified, Rao-Blackwellized objective that results in additional improvements. Our objective has a simple form—it is a mixture of classical masked language modeling losses— and can be used to train encoder-only language models that admit efficient samplers, including ones that can generate arbitrary lengths of text semi-autoregressively like a traditional language model. On language modeling benchmarks, a range of masked diffusion models trained with modern engineering practices achieves a new state-of-the-art among diffusion models, and approaches AR perplexity. We provide the code1, along with a blog post and video tutorial2 on the project page:
尽管扩散模型在生成高质量图像方面表现出色，但先前研究报道了扩散模型与自回归（AR）方法在语言建模方面存在显著的性能差距。在这项工作中，我们表明简单的掩码离散扩散比之前认为的更有效。我们应用了一种有效的训练配方，提高了掩码扩散模型的性能，并推导出一个简化的、Rao-Blackwell 化的目标，从而实现了进一步的改进。我们的目标形式简单——它是由经典的掩码语言建模损失混合而成的——并且可以用于训练仅编码器语言模型，这些模型允许高效的采样器，包括那些可以像传统语言模型一样半自回归地生成任意长度文本的采样器。在语言建模基准测试中，使用现代工程实践训练的一系列掩码扩散模型在扩散模型中达到了新的最先进水平，并接近 AR 困惑度。我们在项目页面上提供了代码 1，以及一篇博客文章和视频教程 2。

[https://s-sahoo.com/mdlm](https://s-sahoo.com/mdlm)

# 1 Introduction
1 引言

Diffusion models excel at producing realistic, high-quality images and have received significant attention as potential tools for generating discrete data, such as text \[1, 31, 33\], biological sequences \[2, 47\], and graphs \[60, 63\]. Unlike autoregressive (AR) approaches, diffusion-based methods are not constrained to generate data sequentially, and therefore have the potential to improve long-term planning, controllable generation, and sampling speed. However, discrete diffusion methods exhibit a performance gap relative to AR models \[1, 23, 26, 33\], especially in language modeling. The standard measure of language modeling performance is log-likelihood: when controlling for parameter count, prior work reports a sizable log-likelihood gap between AR and diffusion models.
扩散模型在生成逼真、高质量的图像方面表现出色，并作为生成离散数据（如文本\[1, 31, 33\]、生物序列\[2, 47\]和图\[60, 63\]）的潜在工具而受到广泛关注。与自回归（AR）方法不同，基于扩散的方法不受生成数据顺序的限制，因此有潜力提高长期规划、可控生成和采样速度。然而，离散扩散方法相对于 AR 模型表现出性能差距\[1, 23, 26, 33\]，尤其是在语言建模方面。语言建模性能的标准衡量指标是对数似然：在控制参数数量的情况下，已有研究报道 AR 模型与扩散模型之间存在显著的对数似然差距。

In this work, we show that simple masked diffusion language modeling (MDLM) combined with effective training recipes is more performant than previously thought \[1, 26, 69\]. We develop a wellengineered MDLM implementation that significantly improves discrete diffusion log-likelihood; we further improve likelihood using a simple substitution-based parameterization of the reverse diffusion process that enables deriving a Rao-Blackwellized continuous-time variational lower bound (ELBO) with improved tightness \[49\]. Interestingly, our objective has a simple form: it is a weighted average of masked language modeling (MLM) losses \[15\], and can be used to endow BERT-style, encoder-only models with principled generation capabilities. We complement this framework with efficient samplers—including ones that can generate semi-autoregressively like a typical language model.
在这项工作中，我们证明了简单的掩码扩散语言模型（MDLM）结合有效的训练配方比之前认为的更具性能\[1, 26, 69\]。我们开发了一个精心设计的 MDLM 实现，显著提高了离散扩散对数似然；我们进一步使用基于简单替换的逆向扩散过程的参数化方法改进了似然，该方法能够推导出具有改进紧度的 Rao-Blackwell 化连续时间变分下界（ELBO）\[49\]。有趣的是，我们的目标具有简单的形式：它是掩码语言建模（MLM）损失的加权平均\[15\]，并且可以用来赋予 BERT 风格的、仅编码器模型以原则性的生成能力。我们用高效的采样器补充了这个框架，包括那些可以像典型语言模型一样进行半自回归生成的采样器。

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-23/18fab104-cc56-43d9-8508-e6cf151b27e7/a132b3450e8e85f0b0a70bee0cced179c05c3a116ac2319d079cddbc765bb2b7.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-23/18fab104-cc56-43d9-8508-e6cf151b27e7/841319f2952f0f68a3436a54b1086e05023c37b96fbfa14ab84f9d10d387b293.jpg)

Figure 1: (Left) Our proposed masked diffusion language model (MDLM) is trained using a weighted average of masked cross entropy losses. (Top Right) In comparison to masked language models (MLM), MDLM’s objective correspond to a principled variational lower bound, and supports generation via ancestral sampling. (Bottom Right) Perplexity (PPL) on One Billion Words (LM1B) benchmark.
图 1：（左）我们提出的掩码扩散语言模型（MDLM）使用掩码交叉熵损失的加权平均值进行训练。（右上）与掩码语言模型（MLM）相比，MDLM 的目标对应于原则性的变分下界，并支持通过祖先采样进行生成。（右下）在 One Billion Words（LM1B）基准上的困惑度（PPL）。

Our masked diffusion models achieve a new state-of-the-art among diffusion models on language modeling benchmarks and approach the perplexity of AR models within 15-25%. Surprisingly, simple engineering choices significantly improve performance in both our models and simple baselines that were previously thought to perform poorly. Our framework also extends to non-language domains, including biological sequence modeling. We pre-train DNA sequence models and observe similar or higher downstream performance compared to classical BERT-style training, while also introducing generative capabilities that classical masked DNA language models lack.
我们的掩码扩散模型在语言建模基准上取得了扩散模型的最新技术水平，并使困惑度在 AR 模型的 15-25%范围内。令人惊讶的是，简单的工程选择显著提高了我们模型和先前被认为表现不佳的简单基线的性能。我们的框架还扩展到非语言领域，包括生物序列建模。我们预训练 DNA 序列模型，观察到与经典 BERT 风格训练相比相似或更高的下游性能，同时引入了经典掩码 DNA 语言模型所缺乏的生成能力。

Contributions We describe (1) a simple masked diffusion language modeling (MDLM) framework with a well-engineered implementation that outperforms all existing diffusion models across language modeling benchmarks (LM1B \[8\], OWT \[18\], DNA \[12\]), and that significantly improves the performance of existing baselines \[1, 26\]. Our MDLM framework implements (2a) a substitution-based parameterization (SUBS) of the reverse unmasking diffusion process; SUBS allows us to derive (2b) a simple, continuous-time, Rao-Blackwellized objective that improves tightness and variance of the ELBO, further increasing performance. We complement MDLM with (3) fast samplers that support semi-autoregressive (SAR) generation and outperform previous SAR models.
贡献 我们描述了（1）一个简单的掩码扩散语言模型（MDLM）框架，其具有精心设计的实现，在语言建模基准（LM1B \[8\], OWT \[18\], DNA \[12\]）上优于所有现有的扩散模型，并显著提高了现有基线的性能\[1, 26\]。我们的 MDLM 框架实现了（2a）基于替换的参数化（SUBS）的逆向掩码扩散过程；SUBS 使我们能够推导出（2b）一个简单、连续时间的 Rao-Blackwell 化目标，该目标提高了 ELBO 的紧密度和方差，进一步提升了性能。我们通过（3）支持半自回归（SAR）生成的快速采样器来补充 MDLM，其性能优于之前的 SAR 模型。

# 2 Background
2 背景

# 2.1 Diffusion Models
2.1 扩散模型

Diffusion models are trained to iteratively undo a forward corruption process q that takes clean data x drawn from the data distribution $q ( \mathbf { x } )$ and defines latent variables $\mathbf { z } _ { t }$ for $t \in [ 0 , \bar { 1 } ]$ \] that represent progressively noisy versions of x \[27, 54, 56, 66, 48, 19\]. The standard forward process for continuous x is
扩散模型被训练以迭代地撤销一个正向 corruption 过程 q，该过程将来自数据分布 $q ( \mathbf { x } )$ 的干净数据 x 定义了表示 x 的逐步噪声版本的潜变量 $\mathbf { z } _ { t }$ 对于 $t \in [ 0 , \bar { 1 } ]$ \] \[27, 54, 56, 66, 48, 19\]。连续 x 的标准正向过程是

$$
\mathbf {z} _ {t} = \sqrt {\alpha_ {t}} \mathbf {x} + \sqrt {1 - \alpha_ {t}} \boldsymbol {\epsilon} \tag {1}
$$

where ${ \epsilon \sim \mathcal { N } ( \bf { 0 } , \bf { I } ) }$ and $( \alpha _ { t } ) _ { t \in [ 0 , 1 ] }$ is a noise schedule, monotonically decreasing in t. The parameterized reverse diffusion model pθ over x and $\mathbf { z } _ { t }$ is trained to maximize a variational lower bound on loglikelihood (ELBO). Given a number of discretization steps T, defining $s ( i ) = ( i - 1 ) / T$ and $t ( i ) = i / { \check { T } }$ , and using $D _ { \mathrm { K L } } [ \cdot ]$ to denote the Kullback–Leibler divergence, the Negative ELBO (NELBO) equals \[54\]:
${ \epsilon \sim \mathcal { N } ( \bf { 0 } , \bf { I } ) }$ 和 $( \alpha _ { t } ) _ { t \in [ 0 , 1 ] }$ 是一个随时间 t 单调递减的噪声调度。参数化的逆向扩散模型 pθ关于 x 和 $\mathbf { z } _ { t }$ 被训练以最大化对数似然变分下界（ELBO）。给定离散化步数 T，定义 $s ( i ) = ( i - 1 ) / T$ 和 $t ( i ) = i / { \check { T } }$ ，并使用 $D _ { \mathrm { K L } } [ \cdot ]$ 表示 Kullback-Leibler 散度，负 ELBO（NELBO）等于\[54\]：

$$
\mathbb {E} _ {q} \left[ \underbrace {- \log p _ {\theta} (\mathbf {x} | \mathbf {z} _ {t (0)})} _ {\mathcal {L} _ {\text {recons}}} + \underbrace {\sum_ {i = 1} ^ {T} D _ {\mathrm{KL}} \left[ q \left(\mathbf {z} _ {s (i)} \mid \mathbf {z} _ {t (i)} , \mathbf {x}\right) \| p _ {\theta} \left(\mathbf {z} _ {s (i)} \mid \mathbf {z} _ {t (i)}\right) \right]} _ {\mathcal {L} _ {\text {diffusion}}} \right] + \underbrace {D _ {\mathrm{KL}} \left[ q \left(\mathbf {z} _ {t (T)} \mid \mathbf {x}\right) \| p _ {\theta} \left(\mathbf {z} _ {t (T)}\right) \right]} _ {\mathcal {L} _ {\text {prior}}} \tag {2}
$$

For brevity, we drop i from $t ( i )$ and s(i) below; in general, s will denote the time step before t.
为简洁起见，我们从 $t ( i )$ 和 s(i)中省略 i；通常，s 将表示 t 之前的时间步。

# 2.2 Discrete Diffusion Models
2.2 离散扩散模型

Applications of diffusion modeling to discrete data can be broken into two broad categories. First are works that embed discrete structures in continuous space and then perform the Gaussian diffusion defined above on these continuous representations \[9, 16, 23, 24, 30, 34, 57\]. More related to our method are works that define a diffusion process directly on discrete structures. D3PM \[1\] introduces a framework with a Markov forward process $q ( \mathbf { z } _ { t } | \mathbf { z } _ { t - 1 } ) { \dot { = } } \mathbf { C a t } ( \mathbf { z } _ { t } ; Q _ { t } \mathbf { z } _ { t - 1 } )$ defined by the multiplication of matrices $Q _ { t }$ over $T$ discrete time steps. This process induces marginals
将扩散建模应用于离散数据可以分为两个广泛的类别。首先是嵌入离散结构到连续空间并在这些连续表示上执行上述高斯扩散的工作\[9, 16, 23, 24, 30, 34, 57\]。与我们的方法更相关的是直接在离散结构上定义扩散过程的工作。D3PM \[1\]引入了一个框架，其中马尔可夫前向过程 $q ( \mathbf { z } _ { t } | \mathbf { z } _ { t - 1 } ) { \dot { = } } \mathbf { C a t } ( \mathbf { z } _ { t } ; Q _ { t } \mathbf { z } _ { t - 1 } )$ 由在 $T$ 个离散时间步上的矩阵 $Q _ { t }$ 的乘积定义。该过程诱导了边缘

$$
q \left(\mathbf {z} _ {t} \mid \mathbf {x}\right) = \operatorname{Cat} \left(\mathbf {z} _ {t}; \bar {Q} _ {t} \mathbf {x}\right) = \operatorname{Cat} \left(\mathbf {z} _ {t}; Q _ {t} \cdot Q _ {t - 1} \dots Q _ {1} \mathbf {x}\right) \tag {3}
$$

that represent the discrete-state form of (1). Extending this formalism to continuous time (as in (1)) relies on continuous time Markov chain (CTMC) theory \[5\]. The CTMC framework in turns leads to generalizations of the score matching perspective on diffusion modeling \[55\] to discrete data \[33, 59\]. Notably, SEDD \[33\] connects score-based approaches with ELBO maximization, enabling performant likelihood-based training of score-based models.
表示（1）的离散状态形式。将这种形式主义扩展到连续时间（如（1））依赖于连续时间马尔可夫链（CTMC）理论\[5\]。CTMC 框架反过来又导致扩散建模中得分匹配视角\[55\]对离散数据的推广\[33, 59\]。值得注意的是，SEDD\[33\]将基于得分的 方法与 ELBO 最大化联系起来，从而实现了基于得分的模型的性能良好的似然训练。

# 3 Simple Masked Diffusion Models
3 简单掩码扩散模型

While previous work on discrete diffusion supports general forward processes (e.g., general $Q _ { t }$ in D3PM), absorbing state (i.e., masking) diffusion consistently achieves the best performance $[ 1 , 3 3 ]$ . In this work, instead of supporting general noise processes, we focus on masking and derive tight Rao-Blackwellized objectives that outperform general approaches and do not require CTMC theory. In this section, we first define the diffusion process for a categorical random variable. Later in Sec. 3.5, we extend this process to sequences containing multiple such categorical variables. We denote our overall approach as Masked Diffusion Language Models (MDLM).
虽然之前的离散扩散工作支持一般的正向过程（例如，D3PM 中的通用 $Q _ { t }$ ），但吸收状态（即掩码）扩散始终能获得最佳性能 $[ 1 , 3 3 ]$ 。在这项工作中，我们没有支持一般的噪声过程，而是专注于掩码，并推导出紧密的 Rao-Blackwell 化目标，这些目标优于通用方法，并且不需要 CTMC 理论。在本节中，我们首先定义分类随机变量的扩散过程。在 3.5 节后面，我们将此过程扩展到包含多个此类分类变量的序列。我们将我们的整体方法称为掩码扩散语言模型（MDLM）。

Notation. We denote scalar discrete random variables with K categories as ‘one-hot’ column vectors and define $\mathcal { V } \in \{ \mathbf { x } \in \{ 0 , 1 \} ^ { K } : \sum _ { i = 1 } ^ { K } \mathbf { x } _ { i } = 1 \}$ as the set of all such vectors. Define $\operatorname { C a t } ( \cdot ; \pi )$ as the categorical distribution over K classes with probabilities given by $\pi \in \Delta ^ { K }$ , where $\Delta ^ { K }$ denotes the K-simplex. We also assume that the K-th category corresponds to a special \[MASK\] token and let $\mathbf { m } \in \mathcal { V }$ be the one-hot vector for this mask, i.e., m $\kappa = 1$ . Additionally, let $\mathbf { 1 } = \left\{ 1 \right\} ^ { K }$ and $^ { \prime } ( \mathbf { a } , \mathbf { b } )$ ⟩ and a⊙b respectively denote the dot and Hadamard products between two vectors a and b.
符号。我们将具有 K 类别的标量离散随机变量表示为“one-hot”列向量，并将 $\mathcal { V } \in \{ \mathbf { x } \in \{ 0 , 1 \} ^ { K } : \sum _ { i = 1 } ^ { K } \mathbf { x } _ { i } = 1 \}$ 定义为所有此类向量的集合。定义 $\operatorname { C a t } ( \cdot ; \pi )$ 为具有 $\pi \in \Delta ^ { K }$ 给定概率的 K 类上的分类分布，其中 $\Delta ^ { K }$ 表示 K 单纯形。我们还假设第 K 类对应于一个特殊的@ token，并让 $\mathbf { m } \in \mathcal { V }$ 为此掩码的 one-hot 向量，即 m $\kappa = 1$ 。此外，让 $\mathbf { 1 } = \left\{ 1 \right\} ^ { K }$ 和 $^ { \prime } ( \mathbf { a } , \mathbf { b } )$ ⟨和 a⊙b 分别表示两个向量 a 和 b 之间的点积和 Hadamard 积。

# 3.1 Interpolating Discrete Diffusion
3.1 插值离散扩散

We restrict our attention to forward processes q that interpolate between clean data $\mathbf { x } \in \nu$ and a target distribution $\operatorname { C a t } ( . ; \pi )$ , forming a direct extension of Gaussian diffusion in (1). Let q define a sequence of increasingly noisy latent variables $\mathbf { z } _ { t } \in \mathcal { V }$ , where the time step t runs from t = 0 (least noisy) to t = 1 (most noisy). The marginal of $\mathbf { z } _ { t }$ conditioned on x at time t is
我们将注意力限制在从干净数据 $\mathbf { x } \in \nu$ 到目标分布 $\operatorname { C a t } ( . ; \pi )$ 插值的正向过程 q 上，形成(1)中高斯扩散的直接扩展。让 q 定义一个越来越嘈杂的潜变量序列 $\mathbf { z } _ { t } \in \mathcal { V }$ ，其中时间步长 t 从 t = 0（最不嘈杂）到 t = 1（最嘈杂）运行。在时间 t 下， $\mathbf { z } _ { t }$ 在 x 上的边缘是

$$
q \left(\mathbf {z} _ {t} \mid \mathbf {x}\right) = \operatorname{Cat} \left(\mathbf {z} _ {t}; \alpha_ {t} \mathbf {x} + (1 - \alpha_ {t}) \boldsymbol {\pi}\right), \tag {4}
$$

where $\alpha _ { t } \in [ 0 , 1 ]$ is a strictly decreasing function in t, with α0 ≈ 1 and $\alpha _ { 1 } \approx 0 ;$ see Suppl. E.1 for details. This implies transition probabilities $q ( \mathbf { z } _ { t } | \mathbf { z } _ { s } ) = \mathrm { C a t } ( \mathbf { z } _ { t } ; \alpha _ { t | s } \mathbf { z } _ { s } + ( 1 - \alpha _ { t | s } ) \pmb { \pi } )$ , where $\alpha _ { t | s } = \alpha _ { t } / \alpha _ { s }$ . This indicates that during each diffusion step from $s \to t ,$ a fraction $( 1 - \dot { \alpha } _ { t | s } )$ of the probability mass is transferred to the prior distribution π. The reverse posterior is given as (see Suppl. 16 for details):
$\alpha _ { t } \in [ 0 , 1 ]$ 是一个在 t 上严格递减的函数，α0 ≈ 1， $\alpha _ { 1 } \approx 0 ;$ 请参见补充材料 E.1 的详细说明。这意味着转换概率 $q ( \mathbf { z } _ { t } | \mathbf { z } _ { s } ) = \mathrm { C a t } ( \mathbf { z } _ { t } ; \alpha _ { t | s } \mathbf { z } _ { s } + ( 1 - \alpha _ { t | s } ) \pmb { \pi } )$ ，其中 $\alpha _ { t | s } = \alpha _ { t } / \alpha _ { s }$ 。这表明在每个扩散步骤中，从 $s \to t ,$ 有一个分数 $( 1 - \dot { \alpha } _ { t | s } )$ 的概率质量被转移到先验分布 π。反向后验给出如下（请参见补充材料 16 的详细说明）：

$$
q \left(\mathbf {z} _ {s} \mid \mathbf {z} _ {t}, \mathbf {x}\right) = \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {\left[ \alpha_ {t \mid s} \mathbf {z} _ {t} + \left(1 - \alpha_ {t \mid s}\right) \mathbf {1} \boldsymbol {\pi} ^ {\top} \mathbf {z} _ {t} \right] \odot \left[ \alpha_ {s} \mathbf {x} + \left(1 - \alpha_ {s}\right) \boldsymbol {\pi} \right]}{\alpha_ {t} \mathbf {z} _ {t} ^ {\top} \mathbf {x} + \left(1 - \alpha_ {t}\right) \mathbf {z} _ {t} ^ {\top} \boldsymbol {\pi}}\right). \tag {5}
$$

While (4) and (5) represent a special case of the more general diffusion processes proposed in D3PM \[1\], we show below that they yield a simplified variational lower bound objective and admit straightforward continuous time extensions.
虽然 (4) 和 (5) 是 D3PM \[1\] 中提出的更一般扩散过程的一个特例，但我们将在下文证明它们会产生一个简化的变分下界目标，并允许直接的连续时间扩展。

# 3.2 Masked Diffusion
3.2 掩码扩散

Next, we focus on masking processes and derive a simple Rao-Blackwellized objective for this choice of q. This objective incurs lower variance during training and improves tightness.
接下来，我们关注掩码过程，并为这个 q 的选择推导出一个简单的 Rao-Blackwellized 目标。该目标在训练过程中具有更低的方差，并提高了紧致性。

# 3.2.1 Forward Masking Process
3.2.1 前向掩码过程

In masked (i.e., absorbing state) diffusion, we set π = m. At each noising step, t, the input x transitions to a ‘masked’ state m with some probability. If an input transitions to m at any time $t ^ { \prime } { . }$ , it will remain in this state for all $t > t ^ { \prime } : q ( \mathbf { z } _ { t } | \mathbf { z } _ { t ^ { \prime } } = \mathbf { m } ) { = } \mathrm { C a t } ( \mathbf { z } _ { t } ; \mathbf { m } )$ . At time T , all inputs are masked with probability 1.
在掩码（即吸收状态）扩散中，我们设置 π = m。在每个加噪步骤 t 中，输入 x 以一定概率过渡到“掩码”状态 m。如果输入在任何时间 $t ^ { \prime } { . }$ 过渡到 m，它将保持在此状态直到 $t > t ^ { \prime } : q ( \mathbf { z } _ { t } | \mathbf { z } _ { t ^ { \prime } } = \mathbf { m } ) { = } \mathrm { C a t } ( \mathbf { z } _ { t } ; \mathbf { m } )$ 。在时间 T ，所有输入都以概率 1 被掩码。

The marginal of the forward process (4) is given by $q ( \mathbf { z } _ { t } | \mathbf { x } ) = \mathrm { C a t } ( \mathbf { z } _ { t } ; \alpha _ { t } \mathbf { x } + ( 1 - \alpha _ { t } ) \mathbf { m } )$ . Using properties of the masking process, the posterior $q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } )$ simplifies (5); see Suppl. A.2:
前向过程（4）的边缘分布由 $q ( \mathbf { z } _ { t } | \mathbf { x } ) = \mathrm { C a t } ( \mathbf { z } _ { t } ; \alpha _ { t } \mathbf { x } + ( 1 - \alpha _ { t } ) \mathbf { m } )$ 给出。利用掩码过程的性质，后验分布 $q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } )$ 简化为（5）；参见补充材料 A.2：

$$
q (\mathbf {z} _ {s} | \mathbf {z} _ {t}, \mathbf {x}) = \left\{ \begin{array}{l l} \operatorname{Cat} \left(\mathbf {z} _ {s}; \mathbf {z} _ {t}\right) & \mathbf {z} _ {t} \neq \mathbf {m}, \\ \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {(1 - \alpha_ {s}) \mathbf {m} + (\alpha_ {s} - \alpha_ {t}) \mathbf {x}}{1 - \alpha_ {t}}\right) & \mathbf {z} _ {t} = \mathbf {m}. \end{array} \right. \tag {6}
$$

# 3.2.2 Reverse Unmasking Process
3.2.2 逆向去掩码过程

The reverse process inverts the noise process defined by q. We consider both a finite number of steps T , as well as a continuous time model corresponding to $T \to \infty$ . We begin with the discrete-time case for which the generative model is expressed as $\begin{array} { r } { p _ { \theta } ( \mathbf { x } ) = \int _ { \mathbf { z } } p _ { \theta } ( \mathbf { z } _ { 1 } ) p _ { \theta } ( \mathbf { x } | \mathbf { z } _ { 0 } ) \prod _ { i = 1 } ^ { T } p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } ) \mathrm { d } \mathbf { z } _ { 0 : 1 } } \end{array}$ .
反向过程会逆转由 q 定义的噪声过程。我们考虑有限步数 T 以及对应于 $T \to \infty$ 的连续时间模型。我们从离散时间情况开始，其中生成模型表示为 $\begin{array} { r } { p _ { \theta } ( \mathbf { x } ) = \int _ { \mathbf { z } } p _ { \theta } ( \mathbf { z } _ { 1 } ) p _ { \theta } ( \mathbf { x } | \mathbf { z } _ { 0 } ) \prod _ { i = 1 } ^ { T } p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } ) \mathrm { d } \mathbf { z } _ { 0 : 1 } } \end{array}$ 。

The optimal form for $p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$ matches the true posterior in (6): this follows immediately from the definition of the diffusion objective in (2), which is a sum of terms of the form $\begin{array} { r } { \operatorname { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) \big | \big | p _ { \theta } \big ( \mathbf { z } _ { s } | \mathbf { z } _ { t } \big ) \big ) } \end{array}$ . However, (6) is conditioned on x, which we do not know. Therefore, we introduce a model $\mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) : \mathcal { V } \times [ 0 , 1 ] \to \Delta ^ { K }$ that approximates x with a neural network. We can also omit explicit dependence of $\mathbf { x } _ { \theta }$ on time t, which simplifies sampling, yielding a 2x inference speed-up (see Suppl. E.2).
$p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$ 的最优形式与公式 (6) 中的真实后验分布相匹配：这直接源于公式 (2) 中扩散目标的定义，后者是形式为 $\begin{array} { r } { \operatorname { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) \big | \big | p _ { \theta } \big ( \mathbf { z } _ { s } | \mathbf { z } _ { t } \big ) \big ) } \end{array}$ 的项的和。然而，公式 (6) 以 x 为条件，而我们知道 x。因此，我们引入一个模型 $\mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) : \mathcal { V } \times [ 0 , 1 ] \to \Delta ^ { K }$ ，它使用神经网络来近似 x。我们还可以省略 $\mathbf { x } _ { \theta }$ 对时间 t 的显式依赖，这简化了采样，实现了 2 倍的推理速度提升（参见补充材料 E.2）。

# 3.2.3 SUBS Parameterization
3.2.3 子参数化

The specific parameterization for $p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$ that we use is
我们使用的 $p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$ 的特定参数化形式是

$$
p _ {\theta} \left(\mathbf {z} _ {s} \mid \mathbf {z} _ {t}\right) = q \left(\mathbf {z} _ {s} \mid \mathbf {z} _ {t}, \mathbf {x} = \mathbf {x} _ {\theta} \left(\mathbf {z} _ {t}, t\right)\right) = \left\{ \begin{array}{l l} \operatorname{Cat} \left(\mathbf {z} _ {s}; \mathbf {z} _ {t}\right), & \mathbf {z} _ {t} \neq \mathbf {m}, \\ \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {(1 - \alpha_ {s}) \mathbf {m} + (\alpha_ {s} - \alpha_ {t}) \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t)}{1 - \alpha_ {t}}\right). & \mathbf {z} _ {t} = \mathbf {m}, \end{array} \right. \tag {7}
$$

Furthermore, we induce 2 key properties of the absorbing state diffusion process into our denoising model, ${ \bf x } _ { \theta } ( { \bf z } _ { t } , t )$ : an unmasked token remains unchanged during reverse diffusion, and the clean input is never masked. We implement these as substitutions to the output of ${ \bf x } _ { \theta } ( { \bf z } _ { t } , t )$ , hence we call our parameterization SUBS.
此外，我们将吸收态扩散过程中的 2 个关键特性引入我们的去噪模型 ${ \bf x } _ { \theta } ( { \bf z } _ { t } , t )$ ：未遮盖的标记在反向扩散过程中保持不变，干净的输入永远不会被遮盖。我们将这些实现为对 ${ \bf x } _ { \theta } ( { \bf z } _ { t } , t )$ 输出的替换，因此我们将我们的参数化称为 SUBS。

Zero Masking Probabilities First, notice that by definition, $\langle \mathbf { x } , \mathbf { m } \rangle = 0$ . For this reason, we design the denoising network such that $\begin{array} { r } { \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle = 0 , \mathrm { i . e . } } \end{array}$ ., we substitute the logit index corresponding to the \[MASK\] token with −∞.
零遮盖概率 首先，根据定义， $\langle \mathbf { x } , \mathbf { m } \rangle = 0$ 。因此，我们设计去噪网络使得 $\begin{array} { r } { \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle = 0 , \mathrm { i . e . } } \end{array}$ 。，我们将对应于遮盖标记的 logit 索引替换为−∞。

Carry-Over Unmasking Second, if $\mathbf { z } _ { t }$ is unmasked, then we desire ${ \bf x } _ { \theta } ( { \bf z } _ { t } , t ) = { \bf z } _ { t } ,$ i.e., unmasked latents are ‘carried over’. We accomplish this by substituting the output of our network to simply copy unmasked inputs.
传递遮盖解除 第二，如果 $\mathbf { z } _ { t }$ 被解除遮盖，那么我们希望 ${ \bf x } _ { \theta } ( { \bf z } _ { t } , t ) = { \bf z } _ { t } ,$ ，即未遮盖的潜空间被“传递”。我们通过将网络输出替换为简单地复制未遮盖输入来实现这一点。

In Suppl. B.1, we show that “Zero Masking Probabilities” property simplifies the D3PM’s NELBO (39) to (41), and “Carry-Over Unmasking” futher simplifies (41) to (43) whose continuous time equivalent is the simplified NELBO (10). Table 8 shows that each simplification leads to an improved likelihood.
在补充材料 B.1 中，我们展示了“零遮盖概率”特性将 D3PM 的 NELBO（39）简化为（41），而“传递遮盖解除”进一步将（41）简化为（43），其连续时间等价于简化的 NELBO（10）。表 8 表明，每次简化都导致似然性提高。

# 3.3 Rao-Blackwellized Likelihood Bounds

Recall from (2) that the diffusion traning objective has the form $\mathcal { L } _ { \mathrm { r e c o n s } } + \mathcal { L } _ { \mathrm { d i f f u s i o n } } + \mathcal { L } _ { \mathrm { p r i o r } }$ . For the simplified reverse process in (7), the discrete-time diffusion loss for finite T simplifies to (Suppl. B.1.3):
由(2)式可知，扩散训练目标的形式为 $\mathcal { L } _ { \mathrm { r e c o n s } } + \mathcal { L } _ { \mathrm { d i f f u s i o n } } + \mathcal { L } _ { \mathrm { p r i o r } }$ 。对于(7)式中的简化反向过程，有限 T 时间的离散时间扩散损失简化为（补充材料 B.1.3）：

$$
\mathcal {L} _ {\text { diffusion }} = \sum_ {i = 1} ^ {T} \mathbb {E} _ {q} \left[ \mathrm{D} _ {\mathrm{KL}} \left(q \left(\mathbf {z} _ {s (i)} \mid \mathbf {z} _ {t (i)}, \mathbf {x}\right) \| p _ {\theta} \left(\mathbf {z} _ {s (i)} \mid \mathbf {z} _ {t (i)}\right)\right) \right] = \sum_ {i = 1} ^ {T} \mathbb {E} _ {q} \left[ \frac {\alpha_ {t (i)} - \alpha_ {s (i)}}{1 - \alpha_ {t (i)}} \log \left\langle \mathbf {x} _ {\theta} \left(\mathbf {z} _ {t (i)}\right), \mathbf {x} \right\rangle \right] \tag {8}
$$

Note that this objective is simpler and more well-behaved than the expression one would obtain for $\mathrm { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) \big | \big | \overline { { p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } ) } } \big )$ under the parameterization induced by using $p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } ) = q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } =$ $\mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) )$ from (5), which is similar to what is used by D3PM \[1\] (see Suppl. A.2.4):
请注意，此目标比在(5)式中使用 $p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } ) = q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } =$ $\mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) )$ 所诱导的参数化下得到的 $\mathrm { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) \big | \big | \overline { { p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } ) } } \big )$ 表达式更简单、行为更良好，这与 D3PM \[1\]所使用的类似（参见补充材料 A.2.4）：

$$
\left[ \frac {\alpha_ {s} - \alpha_ {t}}{1 - \alpha_ {t}} \log \frac {\alpha_ {t} \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {m} \rangle + (1 - \alpha_ {t})}{(1 - \alpha_ {t}) \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {x} \rangle} + \frac {1 - \alpha_ {s}}{1 - \alpha_ {t}} \log \frac {(1 - \alpha_ {s}) (\alpha_ {t} \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {m} \rangle + (1 - \alpha_ {t}))}{(1 - \alpha_ {t}) (\alpha_ {s} \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {m} \rangle + (1 - \alpha_ {s}))} \right] \langle \mathbf {z} _ {t}, \mathbf {m} \rangle \tag {9}
$$

We refer to the process of obtaining (8) in lieu of (9) as a form of Rao-Blackwellization. Specifically, we analytically compute expectations such as $\langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle = 0$ in order to simplify objective (9) to obtain (8). Without analytical simplifications, a model must learn θ such that $\langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle = 0$ holds. Unlike in regular Rao-Blackwellization, simplifications are possible because of modeling choices for ${ \bf x } _ { \theta } ( { \bf z } _ { t } , t )$ (zero masking probabilities and carry-over unmasking). In that sense, our approach has similarities to graphical modeling, where incorporating conditional independencies into $p _ { \theta }$ sets certain log-likelihood terms to zero. However, our approach also empirically helps reduce variance, hence we refer to it as Rao-Blackwellization, somewhat abusing the usual terminology.
我们称通过（8）代替（9）的过程为一种形式的罗-布莱克威尔化。具体而言，我们通过解析计算诸如 $\langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle = 0$ 的期望，以简化目标（9）从而得到（8）。如果没有解析简化，模型必须学习θ使得 $\langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle = 0$ 成立。与常规的罗-布莱克威尔化不同，由于 ${ \bf x } _ { \theta } ( { \bf z } _ { t } , t )$ 的建模选择（零掩码概率和传递掩码解除），简化是可能的。从这个意义上说，我们的方法与图模型有相似之处，在图模型中，将条件独立性纳入 $p _ { \theta }$ 将某些对数似然项设为零。然而，我们的方法还通过经验帮助减少方差，因此我们将其称为罗-布莱克威尔化，这在某种程度上有些滥用通常的术语。

# 3.4 Continuous-Time Likelihood Bounds
3.4 连续时间似然界

Previous works have shown empirically and mathematically that increasing the number of steps T yields a tighter approximation to the ELBO \[29\]. Following a similar argument, we form an continuous extension of (8) by taking $T \to \infty$ (see Suppl. B.2), which yields the following NELBO, $\mathcal { L } _ { \mathrm { N E L B O } } ^ { \infty } \mathrm { : }$
先前的工作已经通过实证和数学证明，增加步数 T 可以得到对 ELBO 更紧密的近似\[29\]。遵循类似的论证，我们通过取 $T \to \infty$ （参见补充材料 B.2）对（8）进行连续扩展，从而得到以下 NELBO， $\mathcal { L } _ { \mathrm { N E L B O } } ^ { \infty } \mathrm { : }$

$$
\mathcal {L} _ {\mathrm{NELBO}} ^ {\infty} = \mathbb {E} _ {q} \int_ {t = 0} ^ {t = 1} \frac {\alpha_ {t} ^ {\prime}}{1 - \alpha_ {t}} \log \left\langle \mathbf {x} _ {\theta} \left(\mathbf {z} _ {t}, t\right), \mathbf {x} \right\rangle \mathrm{d} t \tag {10}
$$

Invariance to the noise schedule The function $\alpha _ { t }$ is invertible due to the monotonicity assumption in Sec. 3.1, and so we can perform the following change of variables in (10): $\gamma \equiv \log ( 1 - \alpha _ { t } )$ . Thus, the diffusion loss can be equivalently expressed as L∞NELBO = −EqR γ=0γ=−∞l $\begin{array} { r } { \mathcal { L } _ { \mathrm { N E L B O } } ^ { \infty } = - \mathbb { E } _ { q } \int _ { \gamma = - \infty } ^ { \gamma = 0 } \log \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { \gamma } , \gamma ) , \mathbf { x } \rangle \mathrm { d } \gamma ; } \end{array}$ γ=0 see Suppl. E.1.1 for details. This new formulation demonstrates that the diffusion loss is invariant to the functional form of $\alpha _ { t } ,$ , which we verify empirically in Suppl. E.1.
对噪声调度的不变性 由于 Sec. 3.1 中的单调性假设，函数 $\alpha _ { t }$ 是可逆的，因此我们可以在式 (10) 中进行以下变量替换： $\gamma \equiv \log ( 1 - \alpha _ { t } )$ 。因此，扩散损失可以等价地表示为 L∞NELBO = −EqR γ=0γ=−∞l $\begin{array} { r } { \mathcal { L } _ { \mathrm { N E L B O } } ^ { \infty } = - \mathbb { E } _ { q } \int _ { \gamma = - \infty } ^ { \gamma = 0 } \log \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { \gamma } , \gamma ) , \mathbf { x } \rangle \mathrm { d } \gamma ; } \end{array}$ γ=0 参见补充材料 E.1.1 了解详情。这种新的公式表明扩散损失对 $\alpha _ { t } ,$ 的函数形式是不变的，我们在补充材料 E.1 中通过实验验证了这一点。

# 3.5 Masked Diffusion Language Models
3.5 带掩码的扩散语言模型

Next, we apply masked diffusion to language modeling over sequences $\mathbf { x } ^ { 1 : L }$ of L tokens, with $\mathbf { x } ^ { \ell }$ denoting the ℓ-th token. We make the assumption that the forward noising process is applied independently across a sequence and that, conditioned on a sequence of latents $\bar { \mathbf { z } _ { t } ^ { 1 : L } }$ , the denoising process factorizes independently across tokens, i.e., $\begin{array} { r } { p _ { \theta } \big ( \mathbf { z } _ { s } ^ { 1 : L } \mid \mathbf { z } _ { t } ^ { 1 : L } \big ) = \prod _ { \ell = 1 } ^ { L } p _ { \theta } \big ( \mathbf { z } _ { s } ^ { \ell } \mid \mathbf { z } _ { t } ^ { 1 : L } \big ) } \end{array}$ . Thus, we use a single model to compute $\mathbf { x } _ { \theta } ^ { \ell } ( \mathbf { z } _ { t } ^ { 1 : L } , t )$ for each ℓ from a masked sequence $\mathbf { z } _ { t } ,$ optimizing:
接下来，我们将带掩码的扩散应用于长度为 L 的序列 $\mathbf { x } ^ { 1 : L }$ 的语言建模，其中 $\mathbf { x } ^ { \ell }$ 表示第 ℓ 个标记。我们假设正向噪声过程在序列中独立应用，并且在给定潜在序列 $\bar { \mathbf { z } _ { t } ^ { 1 : L } }$ 的条件下，去噪过程在标记上独立因子化，即 $\begin{array} { r } { p _ { \theta } \big ( \mathbf { z } _ { s } ^ { 1 : L } \mid \mathbf { z } _ { t } ^ { 1 : L } \big ) = \prod _ { \ell = 1 } ^ { L } p _ { \theta } \big ( \mathbf { z } _ { s } ^ { \ell } \mid \mathbf { z } _ { t } ^ { 1 : L } \big ) } \end{array}$ 。因此，我们使用单个模型来计算从掩码序列 $\mathbf { z } _ { t } ,$ 优化的每个 ℓ 的 $\mathbf { x } _ { \theta } ^ { \ell } ( \mathbf { z } _ { t } ^ { 1 : L } , t )$ 。

$$
\mathcal {L} _ {\mathrm{NELBO}} ^ {\infty} = \mathbb {E} _ {q} \int_ {t = 0} ^ {t = 1} \frac {\alpha_ {t} ^ {\prime}}{1 - \alpha_ {t}} \sum_ {\ell} \log \left\langle \mathbf {x} _ {\theta} ^ {\ell} \left(\mathbf {z} _ {t} ^ {1: L}, t\right), \mathbf {x} ^ {\ell} \right\rangle \mathrm{d} t \tag {11}
$$

Interestingly, our objective has a simple form: it is the weighted average of masked language modeling (MLM) losses \[15\]. Thus our work establishes a connection between generative diffusion models and encoder-only BERT models. Our objective enables principled selection of a (randomized) masking rate, and also endows BERT-style models with principled generation capabilities; see Sec. 6. The full training algorithm is provided in Suppl. B.3.
有趣的是，我们的目标具有简单的形式：它是掩码语言建模（MLM）损失的加权平均\[15\]。因此，我们的工作在生成扩散模型和仅编码器 BERT 模型之间建立了联系。我们的目标能够对（随机化的）掩码率进行原则性选择，同时也赋予 BERT 风格模型原则性生成能力；参见第 6 节。完整的训练算法在补充材料 B.3 中提供。

Note: Although (11) imposes a loss on all tokens, unmasked tokens don’t contribute to the loss, as they are copied over by the denoising network due to “carry-over unmasking” (Sec. 3.2.3), effectively reducing log $\langle \mathbf { x } _ { \theta } ^ { \ell } ( \mathbf { z } _ { t } ^ { 1 : L } , \dot { \mathbf { \xi } } _ { t } ) , \mathbf { x } ^ { \ell } \rangle$ to zero.
注意：尽管公式(11)对所有标记施加了损失，但未掩码的标记不会对损失做出贡献，因为它们由于“携带过掩码”（第 3.2.3 节）而被去噪网络复制，有效地将 log $\langle \mathbf { x } _ { \theta } ^ { \ell } ( \mathbf { z } _ { t } ^ { 1 : L } , \dot { \mathbf { \xi } } _ { t } ) , \mathbf { x } ^ { \ell } \rangle$ 减少到零。

# 3.5.1 Training Considerations for Masked Diffusion
3.5.1 掩码扩散的训练考虑

One of the key contributions of our work is a well-engineered implementation of masked diffusion models. Our experiments demonstrate that these improvements greatly boost performance even for methods previously thought to perform poorly, e.g., Austin et al. \[1\]. Below we briefly summarize these implementation details. First, we find that tokenization is critical to performance. Small vocabularies, such as the 8k vocabulary in Austin et al. \[1\], result in longer-range dependencies that decrease the performance of both diffusion and AR models. Additionally, by focusing on masked diffusion, we are able to provide a numerically stable implementation of the objective function. Namely, since previous formulations of discrete diffusion were constructed to accommodate a wide range of limiting distributions \[1\], the objective was implemented by materializing the full transition matrices $\bar { Q } _ { t }$ and posterior probabilities. In contrast, we evaluate $\bar { D _ { \mathrm { K L } } } [ q ( \mathbf { z } _ { s } \mid \mathbf { z } _ { t } , \mathbf { x } ) \bar { | } | p _ { \theta } ( \mathbf { z } _ { s } \mid \mathbf { z } _ { t } ) ]$ by examining only the masked token indices rather than comparing the full true and approximate posterior distributions.
我们工作的一个关键贡献是精心设计的掩码扩散模型实现。我们的实验表明，这些改进即使对于先前被认为表现不佳的方法（例如 Austin 等人 \[1\]），也能显著提升性能。下面我们简要总结这些实现细节。首先，我们发现分词对性能至关重要。小词汇表，如 Austin 等人 \[1\] 中的 8k 词汇表，会导致长距离依赖，从而降低扩散模型和 AR 模型的性能。此外，通过专注于掩码扩散，我们能够提供一个数值稳定的目標函数实现。具体来说，由于先前离散扩散的公式是为了适应广泛的极限分布 \[1\] 而构建的，因此目標函数是通过显式构建完整的转换矩阵 $\bar { Q } _ { t }$ 和后验概率来实现的。相比之下，我们通过仅检查掩码标记索引来评估 $\bar { D _ { \mathrm { K L } } } [ q ( \mathbf { z } _ { s } \mid \mathbf { z } _ { t } , \mathbf { x } ) \bar { | } | p _ { \theta } ( \mathbf { z } _ { s } \mid \mathbf { z } _ { t } ) ]$ ，而不是比较完整真实和近似后验分布。

Furthermore, we modernize the architecture for the denoising network relative to D3PM \[1\]. In lieu of the T5 architecture used in D3PM, we use the diffusion transformer (DiT) introduced in Peebles & Xie \[42\], which integrates time step conditioning into a standard encoder-only transformer \[62\] and uses rotary positional embeddings \[58\]. In addition, we implement a low-discrepancy sampler that reduces the variance of the ELBO, similar to Kingma et al. \[29\] and draws correlated samples $t _ { i }$ rather than performing i.i.d. sampling.
此外，我们针对去噪网络相对于 D3PM \[1\] 进行了现代化改造。在 D3PM 中使用的 T5 架构的基础上，我们采用了 Peebles & Xie \[42\] 中引入的扩散 Transformer（DiT），该架构将时间步长条件整合到标准的编码器-only Transformer \[62\] 中，并使用旋转位置编码 \[58\]。此外，我们实现了一种低差异采样器，以降低 ELBO 的方差，类似于 Kingma 等人 \[29\] 的方法，并抽取相关样本 $t _ { i }$ 而不是执行独立同分布采样。

# 4 Inference and Sampling in Masked Diffusion Language Models
4 在掩码扩散语言模型中的推理和采样

# 4.1 Efficient Ancestral Sampling
4.1 高效的祖先采样

To generate a sequence of length $L$ the reverse diffusion process starts with the sequence $\mathbf { z } _ { t = 1 } ^ { 1 : L }$ where $\mathbf { z } _ { t = 1 } ^ { \ell } = \mathbf { m }$ , for all $\ell \in \{ 1 , \ldots , L \}$ . Then the subsequent latents, $\mathbf { z } _ { t } ^ { 1 : L }$ are generated by discretizing the t=1 reverse diffusion process with some finite T. Given $\mathbf { z } _ { t } ^ { 1 : L }$ 1 t, we construct $\mathbf { \widetilde { z } } _ { s } ^ { 1 : L }$ by sampling each token $\mathbf { z } _ { s } ^ { \ell }$ independently from the distribution $p _ { \theta } ( \mathbf { z } _ { s } ^ { \ell } | \mathbf { z } _ { t } ^ { 1 : L } )$ given in (7).
为了生成长度为 $L$ 的序列，反向扩散过程从序列 $\mathbf { z } _ { t = 1 } ^ { 1 : L }$ 开始，其中 $\mathbf { z } _ { t = 1 } ^ { \ell } = \mathbf { m }$ ，对所有 $\ell \in \{ 1 , \ldots , L \}$ 适用。然后，通过将时间步长为 1 的反向扩散过程离散化到某个有限值 T 来生成后续的潜空间 $\mathbf { z } _ { t } ^ { 1 : L }$ 。给定 $\mathbf { z } _ { t } ^ { 1 : L }$ 1 t，我们通过独立地从公式（7）中给出的分布 $p _ { \theta } ( \mathbf { z } _ { s } ^ { \ell } | \mathbf { z } _ { t } ^ { 1 : L } )$ 中采样每个标记 $\mathbf { z } _ { s } ^ { \ell }$ 来构建 $\mathbf { \widetilde { z } } _ { s } ^ { 1 : L }$ 。

Note that in the reverse process, unmasked tokens remain unchanged. Thus, if no new tbecome unmasked (which can occur often in early denoising stages for large T ), then $\smash { \mathbf { z } _ { c } ^ { 1 : L } }$ . $\mathbf { z } _ { s } ^ { 1 : L } = \mathbf { z } _ { t } ^ { \hat { 1 } : L }$ Additionally if the denoising model, $\mathbf { x } _ { \theta } ( \mathbf { z } _ { t } ^ { 1 : L } )$ is not conditioned on time, then we can simply draw a new sample from $p _ { \theta } \big ( \mathbf { z } _ { s - 1 / T } ^ { 1 : L } \big | \mathbf { z } _ { s } ^ { 1 : L } \big )$ 1:L) using the previously computed and cached value $\mathbf { x } _ { \theta } ( \mathbf { z } _ { t } ^ { 1 : L } )$ . This means we have effectively “skipped” over the time step s, saving a function call to the denoising network. Note that SEDD \[33\] does not support this caching because the denoising network models time-dependent rates, which requires conditioning on time.
请注意，在逆向过程中，未遮盖的标记保持不变。因此，如果没有任何新的标记被遮盖（这在早期去噪阶段对于较大的 T 经常发生），那么 $\smash { \mathbf { z } _ { c } ^ { 1 : L } }$ $\mathbf { z } _ { s } ^ { 1 : L } = \mathbf { z } _ { t } ^ { \hat { 1 } : L }$ 此外，如果去噪模型 $\mathbf { x } _ { \theta } ( \mathbf { z } _ { t } ^ { 1 : L } )$ 不依赖于时间，那么我们可以直接从 $p _ { \theta } \big ( \mathbf { z } _ { s - 1 / T } ^ { 1 : L } \big | \mathbf { z } _ { s } ^ { 1 : L } \big )$ 1:L)中抽取一个新的样本，使用先前计算并缓存的值 $\mathbf { x } _ { \theta } ( \mathbf { z } _ { t } ^ { 1 : L } )$ 。这意味着我们实际上“跳过”了时间步长 s，节省了对去噪网络的函数调用。请注意，SEDD \[33\] 不支持这种缓存，因为去噪网络对时间相关的速率进行建模，这需要依赖于时间。

# 4.2 Semi-Autoregressive Masked Diffusion Language Models
4.2 半自回归遮盖扩散语言模型

Our method also admits an effective semi-autoregressive (SAR) decoding method that allows the model to generate sequences of arbitrary length \[24, 52, 53\]. Let $\tilde { \mathbf { x } } ^ { 1 : L }$ represent the output from sampling a sequence of L tokens using the reverse diffusion process described above. To generate additional $\bar { L } ^ { \prime } < L$ tokens, we propose a generation algorithm in which the latter $L - L ^ { \prime }$ tokens $\tilde { \mathbf { x } } ^ { \bar { L } ^ { \prime } : L }$ are used as a prefix for an additional round of generation. Given the carry-over unmasking described in Sec. 3.2.3, these prefix tokens will simply be copied over at each decoding step. The remaining tokens are generated as above withto b $\mathbf { z } _ { s } ^ { \ell } \sim p _ { \theta } ( \mathbf { z } _ { s } ^ { \ell } | \mathbf { \bar { z } } _ { t } ^ { L : L + L ^ { \prime } } )$ for all sked to $\ell { \in } \left\{ L { + } 1 , { \ldots } , L { + } L ^ { \prime } \right\}$ , withf this $\mathbf { z } _ { t = 1 } ^ { L - L ^ { \prime } : L }$ initialized to we have pro $\tilde { \mathbf { x } } ^ { L - L ^ { \prime } : L }$ pposed tokens $L { + } L ^ { \prime }$ concat $\left[ \tilde { \mathbf { x } } ^ { 1 : L } , \tilde { \mathbf { x } } ^ { L + 1 : L + L ^ { \prime } } \right]$ , where concat\[·\] denotes concatenation along the sequence length dimension. This process can repeat indefinitely, with the prefix shifted for every new round of generation.
我们的方法还包含一种有效的半自回归（SAR）解码方法，该方法允许模型生成任意长度的序列\[24, 52, 53\]。令 $\tilde { \mathbf { x } } ^ { 1 : L }$ 表示使用上述反向扩散过程对 L 个标记的序列进行采样后的输出。为了生成额外的 $\bar { L } ^ { \prime } < L$ 标记，我们提出了一种生成算法，其中后面的 $L - L ^ { \prime }$ 标记 $\tilde { \mathbf { x } } ^ { \bar { L } ^ { \prime } : L }$ 被用作新一轮生成的前缀。根据第 3.2.3 节中描述的传递式去掩码，这些前缀标记在每个解码步骤中将被简单地复制。其余标记按照上述方法生成，对于所有 sked 到 $\ell { \in } \left\{ L { + } 1 , { \ldots } , L { + } L ^ { \prime } \right\}$ ，使用 $\mathbf { z } _ { s } ^ { \ell } \sim p _ { \theta } ( \mathbf { z } _ { s } ^ { \ell } | \mathbf { \bar { z } } _ { t } ^ { L : L + L ^ { \prime } } )$ 进行初始化，我们提出了将 $\mathbf { z } _ { t = 1 } ^ { L - L ^ { \prime } : L }$ 初始化为 $\tilde { \mathbf { x } } ^ { L - L ^ { \prime } : L }$ 的标记 $L { + } L ^ { \prime }$ ，其中 concat\[·\]表示沿序列长度维度进行拼接。这个过程可以无限重复，每次生成新的一轮时前缀都会进行移动。

# 5 Experiments
5 实验部分

# 5.1 Masked Diffusion Language Models
5.1 掩码扩散语言模型

Experimental Setup We evaluate MDLM as a generative model of language and as a representation model via fine-tuning on downstream tasks.
实验设置 我们将 MDLM 评估为语言生成模型，并通过在下游任务上进行微调将其评估为表示模型。

For language modeling likelihood evaluation, we conduct experiments on two datasets: The One Billion Words Dataset (LM1B; \[8\]) and OpenWebText (OWT; \[18\]). We use the bert-base-uncased tokenizer for LM1B, and report perplexities on the test split. Models have a context size of 128. For OWT, which does not have a pre-defined split, we reserve the last 100K documents as a held-out validation set and report perplexities on this set. We use the GPT2 tokenizer \[45\] for OWT. Models have a context size of 1,024. We utilize the transformer architecture from Lou et al. \[33\], which augments the diffusion transformer \[42\] with rotary embeddings \[58\]. MDLM was trained for 1M or 10M steps (corresponding to 33B, 327B tokens, respectively) on LM1B and 1M steps on OWT (which corresponds to 262B tokens). The corresponding AR baseline was trained for half the number of steps to ensure similar number of tokens seen (details in Suppl. D.2). Full hyperparameters are given in Suppl. D.4. On OWT, we train with and without time step conditioning.
在语言模型似然度评估方面，我们在两个数据集上进行了实验：十亿词数据集（LM1B；\[8\]）和开放网络文本（OWT；\[18\]）。我们使用 bert-base-uncased 分词器对 LM1B 进行处理，并在测试集上报告困惑度。模型的上下文大小为 128。对于没有预定义分割的 OWT，我们将其最后 100K 个文档保留为独立的验证集，并在该集上报告困惑度。我们使用 GPT2 分词器\[45\]对 OWT 进行处理。模型的上下文大小为 1,024。我们采用了 Lou 等人\[33\]提出的 Transformer 架构，该架构通过旋转嵌入\[58\]增强了扩散 Transformer\[42\]。MDLM 在 LM1B 上训练了 1M 或 10M 步（分别对应 33B、327B 个 token），在 OWT 上训练了 1M 步（对应 262B 个 token）。相应的 AR 基线训练步数减半，以确保看到的 token 数量相似（详细信息见补充材料 D.2）。完整超参数在补充材料 D.4 中给出。在 OWT 上，我们进行了带有时步条件处理和不带时步条件处理的训练。

Table 1: Test perplexities (PPL; ↓) on LM1B. †Reported in He et al. \[26\]. Best diffusion value is bolded.
表 1：LM1B 上的测试困惑度（PPL；↓）†

<table><tbody><tr><td colspan="2"></td><td data-imt-p="1">Parameters参数</td><td data-imt-p="1">PPL (↓)PPL（↓）</td></tr><tr><td rowspan="2" data-imt-p="1">Autoregressive自回归</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">Transformer-X Base [13]</td><td>0.46B</td><td>23.5</td></tr><tr><td data-imt-p="1" data-imt_insert_failed_reason="same_text">OmniNet_T[61]</td><td>100M</td><td>21.5</td></tr><tr><td rowspan="5" data-imt-p="1">Diffusion扩散</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">BERT-Mouth [64]^{\dagger}</td><td>110M</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">≤142.89</td></tr><tr><td data-imt-p="1">D3PM (absorb) [1]D3PM（吸收）[1]</td><td>70M</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">≤76.90</td></tr><tr><td data-imt-p="1">Diffusion-LM [30]^{\dagger}Diffusion-LM [30]^{†}</td><td>80M</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">≤118.62</td></tr><tr><td data-imt-p="1" data-imt_insert_failed_reason="same_text">DiffusionBert [26]</td><td>110M</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">≤63.78</td></tr><tr><td data-imt-p="1" data-imt_insert_failed_reason="same_text">SEDD [33] (33B tokens)</td><td>110M</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">≤32.79</td></tr><tr><td rowspan="2" data-imt-p="1">Autoregressive (Retrained)自回归（重新训练）</td><td data-imt-p="1">Transformer (33B tokens)Transformer（33B 个 token）</td><td rowspan="2">110M</td><td>22.32</td></tr><tr><td data-imt-p="1">Transformer (327B tokens)Transformer（327B 个 token）</td><td>20.86</td></tr><tr><td rowspan="2" data-imt-p="1">Diffusion (Ours)扩散（我们的方法）</td><td data-imt-p="1">MDLM (33B tokens)MDLM（33B 个 token）</td><td rowspan="2">110M</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">≤27.04</td></tr><tr><td data-imt-p="1">MDLM (327B tokens)MDLM（327B 个 token）</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">≤23.00</td></tr></tbody></table>

For representation learning, we pre-train models on the C4 dataset \[46\], then fine-tune and evaluate models on the GLUE benchmark \[65\]. Models have a context size of 128. We use the bert-base-uncased tokenizer for the representation learning experiments. We utilize the MosaicBERT architecture from Portes et al. \[43\], an extension of the original BERT architecture \[15\]. We pre-train a bidirectional MosaicBERT using an MLM objective for 37B tokens of C4, as well as a causal variant on the same data. We further fine-tune MosaicBERT model using the MDLM for 327M tokens, less than 1% of the pre-training data. We provide the full hyperparameters in Suppl. D.6.
在表征学习方面，我们在 C4 数据集\[46\]上预训练模型，然后在 GLUE 基准\[65\]上微调和评估模型。模型的上下文大小为 128。我们使用 bert-base-uncased 分词器进行表征学习实验。我们采用 Portes 等人提出的 MosaicBERT 架构\[43\]，这是原始 BERT 架构\[15\]的扩展。我们使用 MLM 目标在 C4 数据集上预训练一个双向 MosaicBERT，共 37B 个 token，并在相同数据上训练一个因果变体。我们进一步使用 MDLM 对 MosaicBERT 模型进行微调，使用 327M 个 token，不到预训练数据的 1%。完整的超参数参数在补充材料 D.6 中提供。

Likelihood Evaluation On LM1B, MDLM outperforms all previous diffusion methods (Table 1). Compared to the SEDD baseline reported by Lou et al. \[33\], trained for 33B tokens, MDLM, which we train for the same amount, achieves a 17% improvement on the perplexity bound. Finally, MDLM gets within 14% of an AR baseline and continues to improve with more training. We see the same trend for models trained on OWT, a larger dataset, shown in Table 2 – MDLM outperforms prior diffusion methods, closing the gap towards AR models. In Table 12 we find that models trained with and without time conditioning attain similar perplexities on OWT. Additionally, Figure 3 demonstrates the reduced variance we achieve from our objective, when compared to previous masked diffusion models such as SEDD \[33\].
似然评估在 LM1B 上，MDLM 的表现优于所有先前的扩散方法（表 1）。与 Lou 等人报告的 SEDD 基线\[33\]相比，该基线经过 33B 个 token 的训练，而 MDLM 经过相同数量的 token 训练，在困惑度界限上实现了 17%的提升。最后，MDLM 在 AR 基线上的表现接近 14%，并且随着更多训练的进行持续改进。对于在 OWT（一个更大的数据集）上训练的模型，我们观察到相同趋势（表 2）——MDLM 优于先前的扩散方法，并逐渐缩小与 AR 模型的差距。在表 12 中我们发现，有时间和无时间条件训练的模型在 OWT 上的困惑度相似。此外，图 3 展示了与先前的掩码扩散模型（如 SEDD\[33\]）相比，我们通过目标函数实现的方差降低。

Zero-Shot Likelihood Evaluation We also explore models’ ability to generalize by taking models trained on OWT and evaluating how well they model unseen datasets. We compare the perplexities of our MDLM with SEDD \[1\] and an AR Transformer language model. Our zero-shot datasets include the validation splits of Penn Tree Bank (PTB; \[36\]), Wikitext \[38\], LM1B, Lambada \[41\], AG News \[68\], and Scientific Papers (Pubmed and Arxiv subsets; \[10\]). Full experimental details are available in Suppl. D.4.
零样本似然评估 我们还研究了模型泛化能力，通过使用在 OWT 上训练的模型，评估它们对未见数据集建模的效果。我们将我们的 MDLM 与 SEDD \[1\]以及一个 AR Transformer 语言模型的困惑度进行比较。我们的零样本数据集包括宾夕法尼亚树库（PTB; \[36\]）、维基文本\[38\]、LM1B、Lambada \[41\]、AG 新闻\[68\]以及科学论文（PubMed 和 Arxiv 子集；\[10\]）。完整的实验细节可在 Suppl. D.4 中找到。

MDLM consistently outperforms the SEDD diffusion parameterization. In some cases, e.g., for Lambada and Scientific Papers, MDLM attains better perplexity than AR. We hypothesize that these datasets are farther
MDLM 始终优于 SEDD 扩散参数化。在某些情况下，例如对于 Lambada 和科学论文，MDLM 的困惑度优于 AR。我们假设这些数据集更加

from OWT, and that diffusion models may be more robust to out-of-domain evaluation due to the unmasking-based objective.
从 OWT 中，并且由于基于去遮蔽的目标，扩散模型可能对域外评估更加稳健。

Table 2: Test perplexities (PPL; ↓) on OWT for models trained for 262B tokens. † denotes retrained models.
表 2：在 OWT 上测试的困惑度（PPL；↓）针对训练了 262B 个 token 的模型。†表示重新训练的模型。

<table><tbody><tr><td></td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">PPL (↓)</td></tr><tr><td data-imt-p="1" data-imt_insert_failed_reason="same_text">AR†</td><td>17.54</td></tr><tr><td data-imt-p="1" data-imt_insert_failed_reason="same_text">SEDD†</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">≤24.10</td></tr><tr><td data-imt-p="1">MDLM (Ours)MDLM（我们的模型）</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">≤23.21</td></tr></tbody></table>

Downstream Task Evaluation We find that BERT fine-tuned with MDLM to be a generative model results in strong perplexities while preserving performance on downstream tasks. On the C4 validation set, the AR model attains perplexity (PPL) of 22, the pre-trained BERT attains a PPL upper bound of 78 (evaluated using the MDLM variational bound), and BERT + MDLM-FT attains a PPL upper bound of 35. In Table 4, we further find that BERT + MDLM fine-tuning has no degradation in downstream
下游任务评估我们发现，使用 MDLM 微调 BERT 作为生成模型，能够在保留下游任务性能的同时，产生较强的困惑度。在 C4 验证集上，AR 模型的困惑度（PPL）为 22，预训练的 BERT 的 PPL 上限为 78（使用 MDLM 变分界限评估），而 BERT+MDLM-FT 的 PPL 上限为 35。在表 4 中，我们进一步发现，BERT+MDLM 微调在下游任务中没有性能下降。

Table 3: Zero-shot perplexities (↓) of models trained for 524B tokens on OWT. All perplexities for diffusion models are upper bounds.
表 3：在 OWT 上训练 524B token 的模型的零样本困惑度（↓）。所有扩散模型的困惑度均为上限。

<table><tbody><tr><td></td><td>PTB</td><td data-imt-p="1">Wikitext维基文本</td><td>LM1B</td><td data-imt-p="1">LambadaLambdaDA</td><td data-imt-p="1">AG NewsAG 新闻</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">Pubmed</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">Arxiv</td></tr><tr><td data-imt-p="1">AR (Retrained)AR（重新训练）</td><td>82.05</td><td>25.75</td><td>51.25</td><td>51.28</td><td>52.09</td><td>49.01</td><td>41.73</td></tr><tr><td data-imt-p="1">SEDD (Retrained)SEDD（重新训练）</td><td>100.09</td><td>34.28</td><td>68.20</td><td>49.86</td><td>62.09</td><td>44.53</td><td>38.48</td></tr><tr><td data-imt-p="1">MDLM (Ours)MDLM（我们）</td><td>95.26</td><td>32.83</td><td>67.01</td><td>47.52</td><td>61.15</td><td>41.89</td><td>37.37</td></tr></tbody></table>

Table 4: GLUE evaluation results. Evaluation measures (↑) are F1 score for QQP and MRPC, Spearman correlations for STS-B, and accuracy for the rest. For MNLI, we report match/mismatch accuracies.
表 4：GLUE 评估结果。评估指标（↑）为 QQP 和 MRPC 的 F1 分数、STS-B 的 Spearman 相关性以及其余指标的准确率。对于 MNLI，我们报告匹配/不匹配的准确率。

<table><tbody><tr><td></td><td data-imt-p="1">MNLI (m/mm)MNLI（m/mm）</td><td>QQP</td><td>QNLI</td><td>SST-2</td><td>COLA</td><td>STS-B</td><td>MRPC</td><td>RTE</td><td data-imt-p="1">Avg平均</td></tr><tr><td>AR</td><td>80.94/80.78</td><td>86.98</td><td>86.16</td><td>90.14</td><td>33.43</td><td>84.32</td><td>83.88</td><td>47.29</td><td>74.88</td></tr><tr><td>BERT</td><td>84.43/85.35</td><td>88.41</td><td>90.46</td><td>92.20</td><td>54.81</td><td>88.41</td><td>89.16</td><td>61.37</td><td>81.62</td></tr><tr><td data-imt-p="1" data-imt_insert_failed_reason="same_text">+MDLM-FT</td><td>84.76/85.07</td><td>88.49</td><td>90.30</td><td>92.20</td><td>57.69</td><td>87.48</td><td>90.53</td><td>62.09</td><td>82.06</td></tr></tbody></table>

GLUE performance compared to the BERT initialization. While the perplexity of our method is higher than the AR baseline, the downstream task performance is significantly better.
与 BERT 初始化的 GLUE 性能比较。尽管我们的方法的困惑度高于 AR 基线，但下游任务的性能显著更好。

Semi-Autoregressive Modeling To test the SAR decoding algorithm presented in Sec. 4.2, we compare to SSD-LM \[24\] a diffusion model that was designed to generate blocks of text autoregressively. We generate 200 sequences of length 2048 tokens on a single 3090 GPU and evaluate generative perplexity under a pre-trained GPT-2 \[45\] model. The SSD-LM sequences are generated using blocks of
半自回归建模 为了测试第 4.2 节中提出的 SAR 解码算法，我们将 SSD-LM \[24\]进行了比较，SSD-LM 是一种设计用于自回归生成文本块的扩散模型。我们在单个 3090 GPU 上生成了 200 个长度为 2048 个 token 的序列，并在预训练的 GPT-2 \[45\]模型下评估了生成困惑度。SSD-LM 序列使用块生成

Table 5: Semi-AR generative perplexity (Gen. PPL; ↓) for sequences of 2048 tokens.
表 5：2048 个 token 序列的半 AR 生成困惑度（Gen. PPL；↓）。

<table><tbody><tr><td></td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">Gen. PPL (↓)</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">Sec/Seq (↓)</td></tr><tr><td>SSD-LM</td><td>35.43</td><td>2473.9</td></tr><tr><td data-imt-p="1">MDLM (Ours)MDLM (我们的方法)</td><td>27.18</td><td>89.3</td></tr></tbody></table>

25 tokens (as implemented in their pre-trained model) and the MDLM sequences are generated using L′ = 512. In Table 5, we find that in addition to achieving better generative perplexity, MDLM enables ∼25-30x faster SAR decoding relative to SSD-LM.
25 个 token（在其预训练模型中实现）以及 MDLM 序列使用 L′ = 512 生成。在表 5 中，我们发现除了实现更好的生成困惑度外，MDLM 相对于 SSD-LM 实现了约 25-30 倍的 SAR 解码速度提升。

# 5.2 Masked Diffusion DNA Models
5.2 带掩码扩散 DNA 模型

We also explore applications to the generative modeling of biological sequences \[14, 47\] using a state space model (SSM) backbone \[22\]. Namely, we build on the recently-proposed Caduceus DNA language model \[50\], which uses as a backbone the data-dependent SSM Mamba block \[21\].
我们还探索了使用状态空间模型（SSM）骨干\[22\]对生物序列生成建模的应用\[14, 47\]。具体来说，我们基于最近提出的 Caduceus DNA 语言模型\[50\]进行构建，该模型使用数据相关的 SSM Mamba 块\[21\]作为骨干。

Experimental Setup We pre-train the encoder-only Caduceus \[50\], which is an MLM, on the HG38 human reference genome \[11\] and perform fine-tuning using our diffusion parameterization. We use a context length of 1024 tokens and follow Schiff et al. \[50\] for the experimental setup, other than learning rate which was reduced to 1e-3. See Suppl. D.7 for full experimental details. We assess both generative performance using perplexity and downstream performance on Genomics Benchmarks \[20\] across language diffusion paradigms and AR models.
实验设置 我们在 HG38 人类参考基因组\[11\]上预训练仅编码器的 Caduceus\[50\]，这是一个语言模型（MLM），并使用我们的扩散参数化进行微调。我们使用 1024 个 token 的上下文长度，并遵循 Schiff 等人\[50\]的实验设置，除了学习率降低到 1e-3。有关完整实验细节，请参见补充材料 D.7。我们在语言扩散范式和 AR 模型上使用困惑度评估生成性能，并在基因组基准测试\[20\]上评估下游性能。

Generative Performance We fine-tune the Caduceus MLM across diffusion parameterizations and compare perplexities against AR models. We report perplexity values in Table 6. MDLM outperforms all other diffusion language modeling schemes.
生成性能 我们在扩散参数化上微调 Caduceus MLM，并与 AR 模型比较困惑度。我们在表 6 中报告了困惑度值。MDLM 优于所有其他扩散语言建模方案。

Downstream Task Fine-tuning We perform downstream evaluation with the Genomics Benchmarks \[20\], a recently proposed benchmark with eight regulatory element classification tasks. As shown in Table 7, our generative fine-tuning paradigm preserves or improves upon downstream performance from MLM pre-training. Absorbing-state diffusion methods outperform Plaid across tasks except for the simplest task Human vs. Worm, where all methods have roughly the same performance. For tasks where the input is a biased subsample of the full genome, we observe that the correlation between perplexity and downstream performance is weaker; see Suppl. D.7.
下游任务微调 我们使用基因组基准 \[20\] 进行下游评估，这是一个最近提出的包含八个调控元件分类任务的基准。如表 7 所示，我们的生成式微调范式在下游性能上保持了或优于 MLM 预训练。吸收态扩散方法在除最简单的任务人类 vs. 蠕虫之外的所有任务中都优于 Plaid，在人类 vs. 蠕虫任务中，所有方法的表现大致相同。对于输入是全基因组有偏子样本的任务，我们观察到困惑度与下游性能之间的相关性较弱；参见补充材料 D.7。

Table 6: Test perplexities $( \mathrm { P P L } ; \downarrow )$ of generative fine-tuning of the Caduceus MLM \[50\] on the HG38 reference genome. Best diffusion model values are bolded. Error bars indicate the difference between the maximum and minimum values across 5 random seeds used for fine-tuning.
表 6：Caduceus MLM \[50\] 在 HG38 参考基因组上生成式微调的测试困惑度 $( \mathrm { P P L } ; \downarrow )$ 。最佳扩散模型值加粗显示。误差线表示在用于微调的 5 个随机种子中最大值和最小值之间的差异。

<table><tbody><tr><td></td><td></td><td data-imt-p="1">Params参数</td><td data-imt-p="1">PPL (↓)困惑度（↓）</td></tr><tr><td rowspan="2" data-imt-p="1">Autoregressive (Retrained)自回归（重新训练）</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">Mamba</td><td>465K</td><td data-imt-p="1">3.067 ± .0103.067 ± 0.010</td></tr><tr><td data-imt-p="1" data-imt_insert_failed_reason="same_text">HyenaDNA</td><td>433K</td><td data-imt-p="1">3.153 ± .0013.153 ± 0.001</td></tr><tr><td rowspan="2" data-imt-p="1">Diffusion (Retrained)扩散（重新训练）</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">Plaid</td><td>507K</td><td data-imt-p="1">≤ 3.240 ± .005≤ 3.240 ± 0.005</td></tr><tr><td>SEDD</td><td>467K</td><td data-imt-p="1">≤ 3.216 ± .003≤ 3.216 ± 0.003</td></tr><tr><td data-imt-p="1">Diffusion (Ours)扩散（我们的方法）</td><td>MDLM</td><td>467K</td><td data-imt-p="1">≤ 3.199 ± .010≤ 3.199 ± 0.010</td></tr></tbody></table>

Table 7: Genomic Benchmarks. Top-1 accuracy (↑) across 5-fold cross-validation (CV) for a pre-trained AR Mamba, and a pre-trained Caduceus model fine-tuned with different diffusion parameterizations. The best values per task are bolded and the second best are italicized. Error bars indicate the difference between the maximum and minimum values across 5 random seeds used for CV.
表 7：基因组基准测试。在 5 折交叉验证下，预训练的 AR Mamba 和经过不同扩散参数化微调的 Caduceus 模型的 Top-1 准确率（↑）。每个任务的最佳值加粗显示，第二好的值用斜体表示。误差线表示在用于交叉验证的 5 个随机种子中最大值和最小值之间的差异。

<table><tbody><tr><td>Model</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">Mamba</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">Caduceus</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">Caduceus</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">Caduceus</td><td data-imt-p="1">Caduceus蛇杖</td></tr><tr><td data-imt-p="1">Fine-Tuning Objective (Parameter Count)微调目标（参数数量）</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">AR (465K)</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">MLM (467K)</td><td data-imt-p="1">Plaid (507k)条纹 (507k)</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">SEDD (467k)</td><td data-imt-p="1">MDLM (ours) (467k)MDLM (我们) (467k)</td></tr><tr><td data-imt-p="1">Mouse Enhancers鼠标增强器</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.763 {±0.008}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.810 {±0.016}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.745 {±0.079}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.784 {±0.058}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.795 {±0.029}</td></tr><tr><td data-imt-p="1">Coding vs. Intergenomic编码与基因组间</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.897 {±0.004}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.913 {±0.003}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.908 {±0.003}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.913 {±0.005}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.913 {±0.003}</td></tr><tr><td data-imt-p="1">Human vs. Worm人类与蠕虫</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.967 {±0.002}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.970 {±0.002}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.971 {±0.001}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.970 {±0.003}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.970 {±0.003}</td></tr><tr><td data-imt-p="1">Human Enhancers Cohn人类增强者 Cohn</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.734 {±0.027}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.737 {±0.001}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.743 {±0.010}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.746 {±0.015}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.743 {±0.016}</td></tr><tr><td data-imt-p="1">Human Enhancer Ensembl人类增强器集成</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.856 {±0.003}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.907 {±0.000}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.885 {±0.003}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.905 {±0.006}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.899 {±0.004}</td></tr><tr><td data-imt-p="1">Human Regulatory人类调控</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.861 {±0.008}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.874 {±0.003}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.868 {±0.010}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.828 {±0.037}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.868 {±0.004}</td></tr><tr><td data-imt-p="1" data-imt_insert_failed_reason="same_text">Human OCR Ensembl</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.806 {±0.005}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.821 {±0.000}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.820 {±0.004}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.816 {±0.008}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.823 {±0.008}</td></tr><tr><td data-imt-p="1">Human NonTATA Promoters人类非 TATA 启动子</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.926 {±0.008}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.935 {±0.014}</td><td data-imt-p="1">0.935 {±0l007}0.935 {±0.007}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.935 {±0.014}</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">0.940 {±0.007}</td></tr></tbody></table>

# 5.3 Ablation Analysis
5.3 消融分析

In Table 8, we can see the effect of our streamlined masked diffusion implementation. The improvements described in Sec. 3.5.1 allow us to greatly reduce perplexity of previously discounted models, such as D3PM (see the bottom row of this table, which is mathematically equivalent to the D3PM formulation). While most works assumed that D3PM achieves mediocre log-likelihoods, we show that is incorrect: our re-implementation almost matches state-of-the-art score-based methods. This introduces a new strong baseline that opens new research opportunities. Additionally, in Table 8, we ablate
在表 8 中，我们可以看到我们精简的掩码扩散实现的效应。第 3.5.1 节中描述的改进使我们能够显著降低先前被忽视的模型的困惑度，例如 D3PM（参见本表的最后一行，它在数学上等同于 D3PM 公式）。虽然大多数工作假设 D3PM 实现了平庸的对数似然值，但我们证明这是不正确的：我们的重新实现几乎与最先进的基于分数的方法相匹配。这引入了一个新的强基线，开启了新的研究机会。此外，在表 8 中，我们消融

Table 8: Test perplexities $( \mathrm { { P P L } ; \downarrow ) }$ for MDLM ablations on LM1B. For the discrete-time models, we use $T = 1 0 0 0$ . Standard deviation is measured over 5 seeds during evaluation.
表 8：MDLM 在 LM1B 上的测试困惑度 $( \mathrm { { P P L } ; \downarrow ) }$ 。对于离散时间模型，我们使用 $T = 1 0 0 0$ 。标准差是在评估期间对 5 个种子进行测量的。

<table><tbody><tr><td></td><td data-imt-p="1">PPL (≤)PPL（≤）</td></tr><tr><td data-imt-p="1">MDLM (47)MDLM（47）</td><td data-imt-p="1">27.04±.0127.04±0.01</td></tr><tr><td data-imt-p="1">w/o continuous time (43)不含连续时间（43）</td><td data-imt-p="1">27.19±.0727.19±0.07</td></tr><tr><td data-imt-p="1">&amp; w/o carry-over (41)及不含进位（41）</td><td data-imt-p="1">28.56±.1528.56±0.15</td></tr><tr><td data-imt-p="1">&amp; w/o zero masking (39)&amp; 无零掩码（39）</td><td data-imt-p="1">28.51±.1528.51±0.15</td></tr></tbody></table>

different components of MDLM. We observe that the perplexity for MDLM trained with a discrete T = 1000 marginally worsens by 0.1 compared to MDLM trained in continuous time. Additionally, removing the “carry over” operation from the SUBS parameterization increases the perplexity by 1.5 points. However, further removing the “zero masking” operation does not lead to any meaningful change in perplexity. We provide further ablations for the continuous time formulation in the Appendix, showing in Table 11 that for a pre-trained model, at inference, increasing T yields better likelihoods.
MDLM 的不同组件。我们观察到，与连续时间训练的 MDLM 相比，使用离散 T=1000 训练的 MDLM 的困惑度略微恶化了 0.1。此外，从 SUBS 参数化中移除“进位”操作使困惑度增加了 1.5 点。然而，进一步移除“零掩码”操作并没有引起困惑度的任何有意义的变化。我们在附录中提供了关于连续时间公式的进一步消融实验，表 11 表明，对于预训练模型，在推理时增加 T 可以获得更好的似然值。

# 6 Related Work
6 相关工作

Comparison to D3PM Masked diffusion is a strict subset of D3PM \[1\]; setting $Q _ { t | s } = \alpha _ { t | s } { \bf I } + ( 1 -$ $\alpha _ { t | s } ) \mathbf { 1 m } ^ { \top }$ in their framework yields our forward diffusion. We improve over D3PM in three ways: (1) we adopt the SUBS parameterization; (2) this allows us to derive a simplified objective that analytically simplifies certain expectations to zero; (3) we adopt well-engineered training recipes that improve performance. Both (1) and (2) are possible because we focus on masking instead of developing a general discrete diffusion framework. Surprisingly, (3) has the largest contribution to performance.
与 D3PM 的比较掩码扩散是 D3PM \[1\] 的严格子集；在其框架中设置 $Q _ { t | s } = \alpha _ { t | s } { \bf I } + ( 1 -$ $\alpha _ { t | s } ) \mathbf { 1 m } ^ { \top }$ 可得我们的正向扩散。我们在三个方面改进了 D3PM：(1) 我们采用了 SUBS 参数化；(2) 这使我们能够推导出一个简化的目标，将某些期望解析简化为零；(3) 我们采用了精心设计的训练配方来提高性能。第(1)点和第(2)点之所以可能，是因为我们专注于掩码而不是开发一个通用的离散扩散框架。令人惊讶的是，第(3)点对性能的提升贡献最大。

Comparison to CTMC Most implementations of diffusion work best in continuous time. However, extending D3PM in this way requires computing the limit of the product of an infinite number of matrices $Q _ { T } { \cdot } Q _ { T - 1 } { \cdots } Q _ { t }$ as $T \to \infty$ , which requires advanced CTMC theory \[5\]. Our work describes simple continuous-time formulations for the most common noise processes (e.g., masking and uniform π), thus helping make an important part of the literature more accessible. In Suppl. C, we show that our results are compatible with CTMC, using the rate forward matrix $\begin{array} { r } { R _ { t } = \frac { \alpha _ { t } ^ { \prime } } { \alpha _ { t } } ( \mathbf { I } - \mathbf { 1 m } ^ { \top } ) } \end{array}$ and the reverse rate $\tilde { R } _ { t } ( \mathbf { y } ^ { \prime } , \mathbf { y } )$ for the transition $\mathbf { y } \to \mathbf { y } ^ { \prime }$ , where $\mathbf { y } , \mathbf { y } ^ { \prime } \in \mathcal { V } ;$ :
与 CTMC 的比较大多数扩散实现最好在连续时间下工作。然而，以这种方式扩展 D3PM 需要计算无限多个矩阵 $Q _ { T } { \cdot } Q _ { T - 1 } { \cdots } Q _ { t }$ 的乘积的极限 $T \to \infty$ ，这需要高级的 CTMC 理论 \[5\]。我们的工作描述了最常见的噪声过程（例如，掩码和均匀π）的简单连续时间公式，从而帮助使文献中的重要部分更加易于访问。在补充材料 C 中，我们展示了我们的结果与 CTMC 兼容，使用转换 $\mathbf { y } \to \mathbf { y } ^ { \prime }$ 的速率正向矩阵 $\begin{array} { r } { R _ { t } = \frac { \alpha _ { t } ^ { \prime } } { \alpha _ { t } } ( \mathbf { I } - \mathbf { 1 m } ^ { \top } ) } \end{array}$ 和反向速率 $\tilde { R } _ { t } ( \mathbf { y } ^ { \prime } , \mathbf { y } )$ ，其中 $\mathbf { y } , \mathbf { y } ^ { \prime } \in \mathcal { V } ;$ :

$$
\tilde {R} _ {t} (\mathbf {y} ^ {\prime}, \mathbf {y}) = - \frac {\alpha_ {t} ^ {\prime}}{1 - \alpha_ {t}} [ \mathbf {y} ^ {\prime} ] ^ {\top} [ \mathbf {x} _ {\theta} (\mathbf {y}, t) - \mathbf {m} ] \langle \mathbf {y}, \mathbf {m} \rangle \tag {12}
$$

Comparison to Score Estimation Score-based approaches to diffusion \[55\] extend to discrete states, although they typically further build upon advanced CTMC theory. In particular, SEDD \[33\] optimizes an ELBO3 that is a function of the score model, obtaining state-of-the-art log-likelihoods among diffusion models. Our approach, however, is much simpler and does not require advanced theory. Furthermore, we can extract the score for MDLM (76), as demonstrated in Suppl. C.3, making it compatible with various techniques designed for score-based algorithms, such as samplers \[5\], score parameterization \[33\], efficient designs of the denoising network \[59\], guidance techniques, and more.
与评分估计方法的比较：基于评分的扩散方法\[55\]可以扩展到离散状态，尽管它们通常进一步建立在先进的 CTMC 理论上。特别是，SEDD\[33\]优化了一个依赖于评分模型的 ELBO3，在扩散模型中获得了最先进的对数似然值。然而，我们的方法要简单得多，并且不需要先进的理论。此外，我们可以提取 MDLM（76）的评分，如补充材料 C.3 所示，使其与各种为基于评分的算法设计的技术兼容，例如采样器\[5\]、评分参数化\[33\]、去噪网络的效率设计\[59\]、引导技术等。

Comparison to BERT Our work provides a principled way of making BERT generative when trained with randomized masking rates. Previous work on generating from BERT used Gibbs sampling or ad-hoc methods \[17, 32, 64\]. The connection between BERT and diffusion was first made by Austin et al. \[1\]: their objective effectively involves unmasking. He et al. \[26\] additionally starts training from a pretrained BERT. However, both works use an objective that is similar to (9), which is less numerically stable than our objective (see Section 3.5.1). Austin et al. \[1\] mention in their appendix that their ELBO simplifies to a weighted masking (MLM) loss similar to (8), but it uses a more complex formula for the weights and is limited to the discrete time setting unlike our work. Furthermore, they do not train with that objective. Our work derives a simpler expression for the average of MLM losses, implements it, and obtains better likelihoods.
与 BERT 的比较 我们的成果提供了一种在随机掩码率训练下使 BERT 具有生成能力的基本方法。以往从 BERT 生成内容的工作使用了吉布斯采样或特设方法\[17, 32, 64\]。BERT 与扩散之间的联系最初由 Austin 等人\[1\]建立：他们的目标实际上涉及去掩码。He 等人\[26\]额外从预训练的 BERT 开始训练。然而，这两项工作使用的目标与(9)相似，其数值稳定性不如我们的目标（见第 3.5.1 节）。Austin 等人\[1\]在附录中提到他们的 ELBO 简化为与(8)类似的加权掩码（MLM）损失，但其权重公式更复杂，并且仅限于离散时间设置，这与我们的工作不同。此外，他们并未使用该目标进行训练。我们的工作推导出 MLM 损失的更简单表达式，实现并获得了更好的似然值。

Comparision to Latent Diffusion LMs In contrast to this work, which defines diffusion over discrete structures, Plaid \[23\] and Diffusion LM \[30\] define a Gaussian diffusion process over word embeddings. Zhang et al. \[67\] and Hu et al. \[28\] extend this approach to flow matching over word embeddings, enabling the design of faster samplers. Discrete Flow Matching (DFM) \[6\] applies flow matching to discrete structures, using a cross-entropy loss as their training objective: $- \mathbb { E } _ { q , t } ^ { \mathbf { \phi } } \mathrm { l o g } p _ { \theta } ( \mathbf { x } ^ { 1 : L } | \mathbf { z } _ { t } ^ { 1 : L } )$ . Similar to Chang et al. \[7\], DFM’s objective, while effective, is not weighted to serve as a proper ELBO. In MDLM, however, we derive a tight, principled lower bound on the log-likelihood.
与潜在扩散语言模型的比较 与本研究将扩散定义在离散结构上不同，Plaid \[23\] 和 Diffusion LM \[30\] 将高斯扩散过程定义在词嵌入上。Zhang 等人 \[67\] 和 Hu 等人 \[28\] 将这种方法扩展到词嵌入上的流匹配，从而能够设计更快的采样器。离散流匹配 (DFM) \[6\] 将流匹配应用于离散结构，使用交叉熵损失作为其训练目标： $- \mathbb { E } _ { q , t } ^ { \mathbf { \phi } } \mathrm { l o g } p _ { \theta } ( \mathbf { x } ^ { 1 : L } | \mathbf { z } _ { t } ^ { 1 : L } )$ 。类似于 Chang 等人 \[7\]，DFM 的目标虽然有效，但并未加权以作为适当的 ELBO。然而，在 MDLM 中，我们推导出对数似然的一个紧密、原则性的下界。

Concurrent Works Concurrent to our work, Shi et al. \[51\] and Ou et al. \[40\] derive a similar simplified objective for masked diffusion processes. While Ou et al. \[40\] start from a score matching perspective, we tackle this problem from a variational lens similar to Shi et al. \[51\]. Similar to Ou et al. \[40\], we formulate efficient samplers in Section 4.1 by leveraging a time-independent denoising network.
并行工作 与我们的工作并行，Shi 等人 \[51\] 和 Ou 等人 \[40\] 为掩码扩散过程推导出类似的简化目标。虽然 Ou 等人 \[40\] 从分数匹配的角度出发，但我们从类似于 Shi 等人 \[51\] 的变分视角来处理这个问题。类似于 Ou 等人 \[40\]，我们在第 4.1 节通过利用时间无关的降噪网络，制定了高效的采样器。

A key differentiation between our work and that of Shi et al. \[51\], Ou et al. \[40\] is the semi-autoregressive decoding method we present in Section 4.2. While \[51, 40\] are restricted to sample sequences of a fixed length, we propose samplers to generate arbitrary lengths of text like a traditional language model. Furthermore, we establish the connection between our simplified objective and the masked language modeling (MLM) objective. As a result, we endow BERT-style models with principled generation capabilities while maintaining representation learning capabilities. Whereas \[51, 40\] only evaluate on NLP datasets, we show that masked diffusion is also effective in modeling biological sequences.
我们工作与 Shi 等人\[51\]和 Ou 等人\[40\]的主要区别在于我们在第 4.2 节中提出的半自回归解码方法。虽然\[51, 40\]仅限于生成固定长度的样本序列，但我们提出了采样器来生成任意长度的文本，类似于传统语言模型。此外，我们建立了我们简化目标与掩码语言建模（MLM）目标之间的联系。因此，我们在保持表示学习能力的同时，赋予 BERT 风格模型原则性生成能力。而\[51, 40\]仅在 NLP 数据集上评估，我们展示了掩码扩散在建模生物序列方面的有效性。

# 7 Conclusion
7 结论

In this work, we explore masked diffusion. With a well-engineered implementation that supports a simple variational objective, we attain state-of-the-art diffusion perplexities on language benchmarks and demonstrate how to efficiently convert BERT-style encoders into generative models. Given we are working on language modeling, we carry any of the inherent risks and opportunities that come with this line of research.
在这项工作中，我们探索了掩码扩散。通过一个支持简单变分目标的精心设计的实现，我们在语言基准上达到了最先进的扩散困惑度，并展示了如何高效地将 BERT 风格编码器转换为生成模型。鉴于我们从事的是语言建模，我们承担了这一研究方向所固有的风险和机遇。

# Acknowledgments and Disclosure of Funding
致谢与资金披露

This work was partially funded by the National Science Foundation under awards DGE-1922551, CAREER awards 2046760 and 2145577, and by the National Institute of Health under award MIRA R35GM151243. Marianne Arriola is supported by a NSF Graduate Research Fellowship under award DGE-2139899 and a Hopper-Dean/Bowers CIS Deans Excellence Fellowship.
这项工作得到了美国国家科学基金会部分资助，项目编号为 DGE-1922551、CAREER 奖项 2046760 和 2145577，以及美国国立卫生研究院项目 MIRA R35GM151243 的资助。Marianne Arriola 由美国国家科学基金会研究生研究奖学金（项目编号 DGE-2139899）和 Hopper-Dean/Bowers CIS 院长卓越奖学金支持。

# References
参考文献

\[1\] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in Neural Information Processing Systems, 34:17981–17993, 2021.
\[1\] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, 和 Rianne Van Den Berg. 基于离散状态空间的结构化去噪扩散模型. Neural Information Processing Systems, 34:17981–17993, 2021.

\[2\] Pavel Avdeyev, Chenlai Shi, Yuhao Tan, Kseniia Dudnyk, and Jian Zhou. Dirichlet diffusion score model for biological sequence generation. In International Conference on Machine Learning, pp. 1276–1301. PMLR, 2023.
\[2\] Pavel Avdeyev, Chenlai Shi, Yuhao Tan, Kseniia Dudnyk, 和 Jian Zhou. 用于生物序列生成的 Dirichlet 扩散评分模型. 国际机器学习会议, pp. 1276–1301. PMLR, 2023.

\[3\] Žiga Avsec, Vikram Agarwal, Daniel Visentin, Joseph R Ledsam, Agnieszka Grabska-Barwinska, Kyle R Taylor, Yannis Assael, John Jumper, Pushmeet Kohli, and David R Kelley. Effective gene expression prediction from sequence by integrating long-range interactions. Nature methods, 18 (10):1196–1203, 2021.
\[3\] Žiga Avsec, Vikram Agarwal, Daniel Visentin, Joseph R Ledsam, Agnieszka Grabska-Barwinska, Kyle R Taylor, Yannis Assael, John Jumper, Pushmeet Kohli, and David R Kelley. 通过整合长程交互实现有效的基因表达预测. Nature methods, 18 (10):1196–1203, 2021.

\[4\] Joe Benton, Yuyang Shi, Valentin De Bortoli, George Deligiannidis, and Arnaud Doucet. From denoising diffusions to denoising markov models. arXiv preprint arXiv:2211.03595, 2022.
\[4\] Joe Benton, Yuyang Shi, Valentin De Bortoli, George Deligiannidis, and Arnaud Doucet. 从去噪扩散到去噪马尔可夫模型. arXiv preprint arXiv:2211.03595, 2022.

\[5\] Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. Advances in Neural Information Processing Systems, 35:28266–28279, 2022.
\[5\] Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. 连续时间框架用于离散去噪模型. Advances in Neural Information Processing Systems, 35:28266–28279, 2022.

\[6\] Andrew Campbell, Jason Yim, Regina Barzilay, Tom Rainforth, and Tommi Jaakkola. Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design. arXiv preprint arXiv:2402.04997, 2024.
\[6\] Andrew Campbell, Jason Yim, Regina Barzilay, Tom Rainforth, and Tommi Jaakkola. 离散状态空间上的生成流：实现多模态流及其在蛋白质协同设计中的应用. arXiv preprint arXiv:2402.04997, 2024.

\[7\] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11315–11325, 2022.

\[8\] Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony Robinson. One billion word benchmark for measuring progress in statistical language modeling, 2014.

\[9\] Ting Chen, Ruixiang Zhang, and Geoffrey Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. arXiv preprint arXiv:2208.04202, 2022.

\[10\] Arman Cohan, Franck Dernoncourt, Doo Soon Kim, Trung Bui, Seokhwan Kim, Walter Chang, and Nazli Goharian. A discourse-aware attention model for abstractive summarization of long documents. Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), 2018. doi: 10.18653/v1/n18-2097. URL [http://dx.doi.org/10.18653/v1/n18-2097](http://dx.doi.org/10.18653/v1/n18-2097).

\[11\] Genome Reference Consortium. Genome reference consortium human build 37 (grch37. Database (GenBank or RefSeq), 2009.
\[11\] Genome Reference Consortium. Human build 37 (grch37)基因组参考联盟数据库（GenBank 或 RefSeq），2009 年。

\[12\] Genome Reference Consortium et al. Genome reference consortium human build 37 (grch37). Database (GenBank or RefSeq), 2009.
\[12\] Genome Reference Consortium 等. Human build 37 (grch37)基因组参考联盟数据库（GenBank 或 RefSeq），2009 年。

\[13\] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860, 2019.
\[13\] 戴志航, 杨志林, 杨一鸣, 卡尔本内尔, 莱克沃克, 和萨拉赫丁诺夫. Transformer-xl: 超越固定长度上下文的注意力语言模型. arXiv 预印本 arXiv:1901.02860, 2019 年。

\[14\] Shachi Deshpande, Kaiwen Wang, Dhruv Sreenivas, Zheng Li, and Volodymyr Kuleshov. Deep multi-modal structural equations for causal effect estimation with unstructured proxies. Advances in Neural Information Processing Systems, 35:10931–10944, 2022.
\[14\] 德什潘德·沙奇, 王凯文, 德鲁夫·斯里尼瓦斯, 李铮, 和库列霍夫·沃洛迪米尔. 用于非结构化代理的因果效应估计的深度多模态结构方程. 神经信息处理系统进展, 35:10931–10944, 2022 年。

\[15\] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.
\[15\] Jacob Devlin, Ming-Wei Chang, Kenton Lee 和 Kristina Toutanova. BERT：用于语言理解的深度双向 Transformer 的预训练。arXiv 预印本 arXiv:1810.04805，2018。

\[16\] Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, et al. Continuous diffusion for categorical data. arXiv preprint arXiv:2211.15089, 2022.
\[16\] Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan 等. 用于分类数据的连续扩散。arXiv 预印本 arXiv:2211.15089，2022。

\[17\] Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer. Mask-predict: Parallel decoding of conditional masked language models. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 6112–6121, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1633. URL [https://aclanthology.org/D19-1633](https://aclanthology.org/D19-1633).
\[17\] Marjan Ghazvininejad, Omer Levy, Yinhan Liu 和 Luke Zettlemoyer. Mask-predict：条件掩码语言模型的并行解码。收录于 Kentaro Inui, Jing Jiang, Vincent Ng 和 Xiaojun Wan 编的《2019 年实验方法自然语言处理会议暨第九届国际自然语言处理联合会议论文集》（EMNLP-IJCNLP），第 6112–6121 页，中国香港，2019 年 11 月。计算语言学协会。doi: 10.18653/v1/D19-1633。URL https://aclanthology.org/D19-1633。

\[18\] Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. Openwebtext corpus. http: //Skylion007.github.io/OpenWebTextCorpus, 2019.
\[18\] Aaron Gokaslan, Vanya Cohen, Ellie Pavlick 和 Stefanie Tellex. Openwebtext 语料库。http: //Skylion007.github.io/OpenWebTextCorpus，2019。

\[19\] Aaron Gokaslan, A Feder Cooper, Jasmine Collins, Landan Seguin, Austin Jacobson, Mihir Patel, Jonathan Frankle, Cory Stephenson, and Volodymyr Kuleshov. Commoncanvas: Open diffusion models trained on creative-commons images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8250–8260, 2024.
\[19\] Aaron Gokaslan, A Feder Cooper, Jasmine Collins, Landan Seguin, Austin Jacobson, Mihir Patel, Jonathan Frankle, Cory Stephenson, 和 Volodymyr Kuleshov. Commoncanvas: 在知识共享许可图像上训练的开源扩散模型。IEEE/CVF 计算机视觉与模式识别会议论文集，第 8250–8260 页，2024 年。

\[20\] Katarína Grešová, Vlastimil Martinek, David Cechák, Petr Šime ˇ cek, and Panagiotis Alexiou. ˇ Genomic benchmarks: a collection of datasets for genomic sequence classification. BMC Genomic Data, 24(1):25, 2023.
\[20\] Katarína Grešová, Vlastimil Martinek, David Cechák, Petr Šimeček, 和 Panagiotis Alexiou. ˇ 基因组基准：用于基因组序列分类的数据集集合。BMC 基因组数据，24(1):25，2023 年。

\[21\] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.
\[21\] Albert Gu 和 Tri Dao. Mamba: 基于选择性状态空间的线性时间序列建模。arXiv 预印本 arXiv:2312.00752，2023 年。

\[22\] Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021.
\[22\] Albert Gu, Karan Goel, 和 Christopher Ré. 基于结构化状态空间的高效长序列建模。arXiv 预印本 arXiv:2111.00396，2021 年。

\[23\] Ishaan Gulrajani and Tatsunori B Hashimoto. Likelihood-based diffusion language models. Advances in Neural Information Processing Systems, 36, 2024.
\[23\] Ishaan Gulrajani 和 Tatsunori B Hashimoto. 基于似然的扩散语言模型。神经信息处理系统进展，36，2024。

\[24\] Xiaochuang Han, Sachin Kumar, and Yulia Tsvetkov. Ssd-lm: Semi-autoregressive simplexbased diffusion language model for text generation and modular control. arXiv preprint arXiv:2210.17432, 2022.
\[24\] Xiaochuang Han，Sachin Kumar 和 Yulia Tsvetkov。Ssd-lm：基于半自回归 Simplex 的扩散语言模型，用于文本生成和模块化控制。arXiv 预印本 arXiv:2210.17432，2022。

\[25\] Floyd B. Hanson. Applied stochastic processes and control for jump-diffusions - modeling, analysis, and computation. In Advances in design and control, 2007. URL [https://api](https://api). semanticscholar.org/CorpusID:6689808.
\[25\] Floyd B. Hanson。应用随机过程与跳跃扩散控制 - 建模、分析与计算。在设计与控制进展中，2007。URL https://api.semanticscholar.org/CorpusID:6689808。

\[26\] Zhengfu He, Tianxiang Sun, Kuanning Wang, Xuanjing Huang, and Xipeng Qiu. Diffusionbert: Improving generative masked language models with diffusion models. arXiv preprint arXiv:2211.15029, 2022.
\[26\] Zhengfu He，Tianxiang Sun，Kuanning Wang，Xuanjing Huang 和 Xipeng Qiu。Diffusionbert：利用扩散模型改进生成式掩码语言模型。arXiv 预印本 arXiv:2211.15029，2022。

\[27\] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
\[27\] Jonathan Ho、Ajay Jain 和 Pieter Abbeel。去噪扩散概率模型。神经信息处理系统进展，33:6840–6851，2020。

\[28\] Vincent Hu, Di Wu, Yuki Asano, Pascal Mettes, Basura Fernando, Björn Ommer, and Cees Snoek. Flow matching for conditional text generation in a few sampling steps. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 380–392, 2024.
\[28\] Vincent Hu、Di Wu、Yuki Asano、Pascal Mettes、Basura Fernando、Björn Ommer 和 Cees Snoek。流匹配在少量采样步骤中的条件文本生成。欧洲计算语言学协会第 18 届会议论文集（第 2 卷：短论文），pp. 380–392，2024。

\[29\] Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. Advances in neural information processing systems, 34:21696–21707, 2021.
\[29\] Diederik Kingma、Tim Salimans、Ben Poole 和 Jonathan Ho。变分扩散模型。神经信息处理系统进展，34:21696–21707，2021。

\[30\] Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. Diffusionlm improves controllable text generation. Advances in Neural Information Processing Systems, 35:4328–4343, 2022.
\[30\] Xiang Li、John Thickstun、Ishaan Gulrajani、Percy S Liang 和 Tatsunori B Hashimoto。Diffusionlm 改进可控文本生成。神经信息处理系统进展，35:4328–4343，2022。

\[31\] Xuanlin Li, Brandon Trabucco, Dong Huk Park, Michael Luo, Sheng Shen, Trevor Darrell, and Yang Gao. Discovering non-monotonic autoregressive orderings with variational inference. arXiv preprint arXiv:2110.15797, 2021.
\[31\] 李宣林, 布兰登·特拉布科, 朴东旭, 罗迈克尔, 沈胜, 达雷尔·特雷弗, 和高杨. 基于变分推理发现非单调自回归排序. arXiv 预印本 arXiv:2110.15797, 2021.

\[32\] Yi Liao, Xin Jiang, and Qun Liu. Probabilistically masked language model capable of autoregressive generation in arbitrary word order. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (eds.), Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 263–274, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.24. URL [https://aclanthology.org/2020.acl-main.24](https://aclanthology.org/2020.acl-main.24).
\[32\] 廖怡, 蒋欣, 和刘群. 能够在任意词序中进行自回归生成的概率掩码语言模型. 在丹·朱拉夫斯基, 乔伊斯·柴, 娜塔莉·施卢特, 和乔尔·特雷罗 (编), 计算语言学协会第 58 届年会论文集, 第 263–274 页, 线上, 2020 年 7 月. 计算语言学协会. doi: 10.18653/v1/2020.acl-main.24. URL https://aclanthology.org/2020.acl-main.24.

\[33\] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion language modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023.
\[33\] 劳亚伦, 孟晨林, 和埃默生·斯特凡诺. 通过估计数据分布的比率进行离散扩散语言建模. arXiv 预印本 arXiv:2310.16834, 2023.

\[34\] Justin Lovelace, Varsha Kishore, Chao Wan, Eliot Shekhtman, and Kilian Q Weinberger. Latent diffusion for language generation. Advances in Neural Information Processing Systems, 36, 2024.
\[34\] 贾斯汀·洛夫莱斯, 瓦尔沙·基绍尔, 万超, 谢克特曼·艾略特, 和韦恩伯格·基利安·Q. 潜在扩散语言生成. 神经信息处理系统进展, 36, 2024.

\[35\] Vincent Mallet and Jean-Philippe Vert. Reverse-complement equivariant networks for dna sequences. Advances in neural information processing systems, 34:13511–13523, 2021.
\[35\] Vincent Mallet 和 Jean-Philippe Vert. 用于 DNA 序列的逆向互补等变网络. 神经信息处理系统进展, 34:13511–13523, 2021.

\[36\] Mitch Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. Building a large annotated corpus of english: The penn treebank. Computational linguistics, 19(2):313–330, 1993.
\[36\] Mitch Marcus, Beatrice Santorini 和 Mary Ann Marcinkiewicz. 构建大型英语标注语料库：宾夕法尼亚树库. 计算语言学, 19(2):313–330, 1993.

\[37\] Chenlin Meng, Kristy Choi, Jiaming Song, and Stefano Ermon. Concrete score matching: Generalized score matching for discrete data. Advances in Neural Information Processing Systems, 35:34532–34545, 2022.
\[37\] Chenlin Meng, Kristy Choi, Jiaming Song 和 Stefano Ermon. 具体分数匹配：离散数据的广义分数匹配. 神经信息处理系统进展, 35:34532–34545, 2022.

\[38\] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models, 2016.
\[38\] Stephen Merity, Caiming Xiong, James Bradbury 和 Richard Socher. 指针哨兵混合模型, 2016.

\[39\] Eric Nguyen, Michael Poli, Marjan Faizi, Armin Thomas, Michael Wornow, Callum Birch-Sykes, Stefano Massaroli, Aman Patel, Clayton Rabideau, Yoshua Bengio, et al. Hyenadna: Long-range genomic sequence modeling at single nucleotide resolution. Advances in neural information processing systems, 36, 2024.
\[39\] Eric Nguyen, Michael Poli, Marjan Faizi, Armin Thomas, Michael Wornow, Callum Birch-Sykes, Stefano Massaroli, Aman Patel, Clayton Rabideau, Yoshua Bengio, 等. Hyenadna：单核苷酸分辨率的长程基因组序列建模。神经信息处理系统进展，36，2024.

\[40\] Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data, 2024.
\[40\] Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, 和 Chongxuan Li. 您的吸收离散扩散在秘密地建模干净数据的条件分布，2024.

\[41\] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernandez. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1525–1534, Berlin, Germany, August 2016. Association for Computational Linguistics. URL [http://www.aclweb.org/anthology/P16-1144](http://www.aclweb.org/anthology/P16-1144).
\[41\] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, 和 Raquel Fernandez. LAMBADA 数据集：需要广泛话语上下文的单词预测。在计算语言学协会第 54 届年会论文集（第一卷：长篇论文）中，第 1525–1534 页，柏林，德国，2016 年 8 月。计算语言学协会。URL http://www.aclweb.org/anthology/P16-1144.

\[42\] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.
\[42\] William Peebles 和 Saining Xie. 基于变压器的可扩展扩散模型。在 IEEE/CVF 国际计算机视觉会议论文集中，第 4195–4205 页，2023.

\[43\] Jacob Portes, Alex Trott, Sam Havens, Daniel King, Abhinav Venigalla, Moin Nadeem, Nikhil Sardana, Daya Khudia, and Jonathan Frankle. Mosaicbert: A bidirectional encoder optimized for fast pretraining, 2024.
\[43\] Jacob Portes, Alex Trott, Sam Havens, Daniel King, Abhinav Venigalla, Moin Nadeem, Nikhil Sardana, Daya Khudia, 和 Jonathan Frankle. Mosaicbert: 一种为快速预训练优化的双向编码器，2024。

\[44\] Ofir Press, Noah A. Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation, 2022.
\[44\] Ofir Press, Noah A. Smith, 和 Mike Lewis. Train short, test long: 带线性偏差的注意力机制实现输入长度外推，2022。

\[45\] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.
\[45\] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, 和 Ilya Sutskever. 语言模型是无监督的多任务学习器，2019。

\[46\] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21(1), jan 2020. ISSN 1532-4435.
\[46\] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, 和 Peter J. Liu. 探索统一文本到文本转换器在迁移学习中的极限，机器学习研究杂志，21(1)，2020 年 1 月。ISSN 1532-4435。

\[47\] Richa Rastogi and Yair Schiff. Semi parametric inducing point networks and neural processes. In International Conference on Learning Representations, 2023.
\[47\] Richa Rastogi 和 Yair Schiff. 半参数诱导点网络和神经过程. 国际学习表征会议, 2023.

\[48\] Subham Sekhar Sahoo, Aaron Gokaslan, Chris De Sa, and Volodymyr Kuleshov. Diffusion models with learned adaptive noise. arXiv preprint arXiv:2312.13236, 2023.
\[48\] Subham Sekhar Sahoo, Aaron Gokaslan, Chris De Sa, 和 Volodymyr Kuleshov. 具有自适应噪声的扩散模型. arXiv 预印本 arXiv:2312.13236, 2023.

\[49\] Subham Sekhar Sahoo, Anselm Paulus, Marin Vlastelica, Vít Musil, Volodymyr Kuleshov, and Georg Martius. Backpropagation through combinatorial algorithms: Identity with projection works. In The Eleventh International Conference on Learning Representations, 2023. URL [https://openreview.net/forum?id=JZMR727O29](https://openreview.net/forum?id=JZMR727O29).
\[49\] Subham Sekhar Sahoo, Anselm Paulus, Marin Vlastelica, Vít Musil, Volodymyr Kuleshov, 和 Georg Martius. 组合算法中的反向传播：投影一致性有效. 第十一届国际学习表征会议, 2023. URL https://openreview.net/forum?id=JZMR727O29.

\[50\] Yair Schiff, Chia-Hsiang Kao, Aaron Gokaslan, Tri Dao, Albert Gu, and Volodymyr Kuleshov. Caduceus: Bi-directional equivariant long-range dna sequence modeling. arXiv preprint arXiv:2403.03234, 2024.
\[50\] Yair Schiff, Chia-Hsiang Kao, Aaron Gokaslan, Tri Dao, Albert Gu, 和 Volodymyr Kuleshov. Caduceus：双向等变长程 DNA 序列建模. arXiv 预印本 arXiv:2403.03234, 2024.

\[51\] Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K Titsias. Simplified and generalized masked diffusion for discrete data. Advances in neural information processing systems, 36, 2024.
\[51\] Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, 和 Michalis K Titsias. 离散数据的简化和通用掩码扩散. 神经信息处理系统进展, 36, 2024.

\[52\] Phillip Si, Allan Bishop, and Volodymyr Kuleshov. Autoregressive quantile flows for predictive uncertainty estimation. In International Conference on Learning Representations.
\[52\] Phillip Si, Allan Bishop, 和 Volodymyr Kuleshov. 自回归分位数流用于预测不确定性估计. 在国际学习表示会议.

\[53\] Phillip Si, Zeyi Chen, Subham Sekhar Sahoo, Yair Schiff, and Volodymyr Kuleshov. Semiautoregressive energy flows: exploring likelihood-free training of normalizing flows. In International Conference on Machine Learning, pp. 31732–31753. PMLR, 2023.
\[53\] Phillip Si, Zeyi Chen, Subham Sekhar Sahoo, Yair Schiff, 和 Volodymyr Kuleshov. 半自回归能量流：探索无似然归一化流的训练. 在国际机器学习会议, pp. 31732–31753. PMLR, 2023.

\[54\] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256–2265. PMLR, 2015.
\[54\] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, 和 Surya Ganguli. 基于非平衡热力学的深度无监督学习. 在国际机器学习会议, pp. 2256–2265. PMLR, 2015.

\[55\] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.
\[55\] 杨松和 Stefano Ermon. 通过估计数据分布的梯度进行生成建模. Neural Information Processing Systems 进展, 32, 2019.

\[56\] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.
\[56\] 杨松, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, 和 Ben Poole. 基于随机微分方程的评分生成建模. arXiv 预印本 arXiv:2011.13456, 2020.

\[57\] Robin Strudel, Corentin Tallec, Florent Altché, Yilun Du, Yaroslav Ganin, Arthur Mensch, Will Grathwohl, Nikolay Savinov, Sander Dieleman, Laurent Sifre, et al. Self-conditioned embedding diffusion for text generation. arXiv preprint arXiv:2211.04236, 2022.
\[57\] Robin Strudel, Corentin Tallec, Florent Altché, Yilun Du, Yaroslav Ganin, Arthur Mensch, Will Grathwohl, Nikolay Savinov, Sander Dieleman, Laurent Sifre, 等人. 用于文本生成的自条件嵌入扩散. arXiv 预印本 arXiv:2211.04236, 2022.

\[58\] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864, 2021.
\[58\] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, 和 Yunfeng Liu. Roformer: 增强型 Transformer 与旋转位置嵌入. arXiv 预印本 arXiv:2104.09864, 2021.

\[59\] Haoran Sun, Lijun Yu, Bo Dai, Dale Schuurmans, and Hanjun Dai. Score-based continuous-time discrete diffusion models. arXiv preprint arXiv:2211.16750, 2022.
\[59\] 孙浩然, 余立军, 戴博, Dale Schuurmans, 戴汉军. 基于分数的连续时间离散扩散模型. arXiv 预印本 arXiv:2211.16750, 2022.

\[60\] Zhiqing Sun and Yiming Yang. Difusco: Graph-based diffusion solvers for combinatorial optimization. Advances in Neural Information Processing Systems, 36:3706–3731, 2023.
\[60\] 孙志庆, 杨毅明. Difusco：基于图的扩散求解器用于组合优化. 神经信息处理系统进展, 36:3706–3731, 2023.

\[61\] Yi Tay, Mostafa Dehghani, Vamsi Aribandi, Jai Gupta, Philip M Pham, Zhen Qin, Dara Bahri, Da-Cheng Juan, and Donald Metzler. Omninet: Omnidirectional representations from transformers. In International Conference on Machine Learning, pp. 10193–10202. PMLR, 2021.
\[61\] Tay Yi, Dehghani Mostafa, Aribandi Vamsi, Gupta Jai, Pham Philip M, Qin Zhen, Bahri Dara, Juan Da-Cheng, Metzler Donald. Omninet：来自 Transformer 的全向表示. 在国际机器学习会议, pp. 10193–10202. PMLR, 2021.

\[62\] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
\[62\] Vaswani Ashish, Shazeer Noam, Parmar Niki, Uszkoreit Jakob, Jones Llion, Gomez Aidan N, Kaiser Łukasz, Polosukhin Illia. 注意力机制是所有你需要的东西. 神经信息处理系统进展, 30, 2017.

\[63\] Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard. Digress: Discrete denoising diffusion for graph generation. arXiv preprint arXiv:2209.14734, 2022.
\[63\] Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, 和 Pascal Frossard. Digress: 用于图生成的离散去噪扩散模型。arXiv 预印本 arXiv:2209.14734, 2022.

\[64\] Alex Wang and Kyunghyun Cho. Bert has a mouth, and it must speak: Bert as a markov random field language model. arXiv preprint arXiv:1902.04094, 2019.
\[64\] Alex Wang 和 Kyunghyun Cho. Bert 有嘴，它必须说话：Bert 作为马尔可夫随机场语言模型。arXiv 预印本 arXiv:1902.04094, 2019.

\[65\] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In International Conference on Learning Representations, 2019. URL [https://openreview](https://openreview). net/forum?id=rJ4km2R5t7.
\[65\] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, 和 Samuel R. Bowman. GLUE: 用于自然语言理解的多任务基准和分析平台。在 International Conference on Learning Representations, 2019. URL https://openreview. net/forum?id=rJ4km2R5t7.

\[66\] Yingheng Wang, Yair Schiff, Aaron Gokaslan, Weishen Pan, Fei Wang, Christopher De Sa, and Volodymyr Kuleshov. Infodiffusion: Representation learning using information maximizing diffusion models. In International Conference on Machine Learning, pp. 36336–36354. PMLR, 2023.
\[66\] Yingheng Wang, Yair Schiff, Aaron Gokaslan, Weishen Pan, Fei Wang, Christopher De Sa, 和 Volodymyr Kuleshov. Infodiffusion: 使用信息最大化扩散模型进行表示学习。在 International Conference on Machine Learning, pp. 36336–36354. PMLR, 2023.

\[67\] Shujian Zhang, Lemeng Wu, Chengyue Gong, and Xingchao Liu. Language rectified flow: Advancing diffusion language generation with probabilistic flows. arXiv preprint arXiv:2403.16995, 2024.
\[67\] 张树建，吴乐萌，龚成越，刘兴超. 语言校正流：使用概率流推进扩散语言生成. arXiv 预印本 arXiv:2403.16995, 2024.

\[68\] Xiang Zhang, Junbo Jake Zhao, and Yann LeCun. Character-level convolutional networks for text classification. In NIPS, 2015.
\[68\] 张翔，赵俊波，杨立昆. 基于字符的卷积神经网络用于文本分类. 在 NIPS 会议论文集中, 2015.

\[69\] Lin Zheng, Jianbo Yuan, Lei Yu, and Lingpeng Kong. A reparameterized discrete diffusion model for text generation. arXiv preprint arXiv:2302.05737, 2023.
\[69\] 郑琳，袁建波，余雷，孔令鹏. 用于文本生成的重新参数化离散扩散模型. arXiv 预印本 arXiv:2302.05737, 2023.

\[70\] Hannah Zhou, Avanti Shrikumar, and Anshul Kundaje. Towards a better understanding of reverse-complement equivariance for deep learning models in genomics. In Machine Learning in Computational Biology, pp. 1–33. PMLR, 2022.
\[70\] 周昊，Shrikumar Avanti，Kundaje Anshul. 深入理解基因组中深度学习模型的反向互补等变性质. 在计算生物学中的机器学习会议论文集中, 第 1-33 页. PMLR, 2022.

# Contents
目录

# 1 Introduction 1
1 引言 1

# 2 Background 2
2 背景 2

2.1 Diffusion Models . . 2
2.1 扩散模型 . . 2

2.2 Discrete Diffusion Models 3
2.2 离散扩散模型 3

# 3 Simple Masked Diffusion Models 3
3 简单掩码扩散模型 3

3.1 Interpolating Discrete Diffusion 3
3.1 插值离散扩散 3

3.2 Masked Diffusion 4
3.2 掩码扩散 4

3.3 Rao-Blackwellized Likelihood Bounds 4

3.4 Continuous-Time Likelihood Bounds . 5

3.5 Masked Diffusion Language Models 5

# 4 Inference and Sampling in Masked Diffusion Language Models 6

4.1 Efficient Ancestral Sampling 6
4.1 高效祖先采样 6

4.2 Semi-Autoregressive Masked Diffusion Language Models . . . . . 6
4.2 半自回归掩码扩散语言模型 . . . . . 6

# 5 Experiments 6
5 实验 6

5.1 Masked Diffusion Language Models 6
5.1 掩码扩散语言模型 6

5.2 Masked Diffusion DNA Models . 8
5.2 掩码扩散 DNA 模型 . 8

5.3 Ablation Analysis . . 9
5.3 消融分析 . . 9

# 6 Related Work 9
6 相关工作 9

# 7 Conclusion 10
7 结论 10

# Appendices 17
附录 17

# Appendix A Discrete time ELBO 17
附录 A 离散时间 ELBO 17

A.1 Generic case . 17
A.1 一般情况 17

A.2 Absorbing state . . . 18
A.2 吸收状态 18

# Appendix B MDLM 21
附录 B MDLM 21

B.1 Rao-Blackwellization 22

B.2 Continuous Time 22
B.2 连续时间 22

B.3 Final Algorithm . . 23
B.3 最终算法 . . 23

# Appendix C Concrete Score Matching 23
附录 C 具体得分匹配 23

C.1 Extracting the Rate Matrix 24
C.1 提取速率矩阵 24

C.2 NELBO 25

C.3 Concrete Score for MDLM 27
C.3 MDLM 的具体得分 27

C.4 Reverse Rate Matrix for MDLM 28
C.4 MDLM 28 的逆率矩阵

C.5 Deriving MDLM’s NELBO via CTMC . 29
C.5 通过 CTMC 推导 MDLM 的 NELBO . 29

# Appendix D Experimental details 31
附录 D 实验细节 31

D.1 Likelihood Evaluation . . 31
D.1 似然评估 . . 31

D.2 Avg. Number of Tokens seen 31
D.2 平均看到的 token 数量 31

D.3 Low discrepancy sampler 31
D.3 低差异采样器 31

D.4 Language Modeling 31
D.4 语言建模 31

D.5 Zeroshot Likelihood 32
D.5 零样本似然 32

D.6 Representation Learning 32
D.6 表示学习 32

D.7 Diffusion DNA Models 32
D.7 扩散 DNA 模型 32

# Appendix E Additional Experiments 33
附录 E 额外实验 33

E.1 Noise schedule parameterization 33
E.1 噪声调度参数化 33

E.2 Faster sampling with caching 34
E.2 基于缓存的快速采样 34

E.3 LM1B ablations 35
E.3 LM1B 消融实验 35

E.4 Train NLL curves on OWT 35
E.4 在 OWT 上训练 NLL 曲线 35

E.5 Time-conditioning ablation on OWT 36
E.5 OWT 上的时间条件消融实验 36

E.6 Unconditional Samples 36
E.6 无条件样本 36

# Appendices
附录

# Appendix A Discrete time ELBO
附录 A 离散时间 ELBO

This section is organized as follows: First, we derive the expressions for the true posterior and the approximate posterior as outlined in Suppl. A.1. We then simplify these expressions specifically for the case of absorbing state diffusion in Suppl. A.2. Finally, we derive the expression for the ELBO for absorbing state diffusion in Suppl. A.2.3.
本节的结构如下：首先，我们推导出真实后验和近似后验的表达式，如补充材料 A.1 中所述。然后，在补充材料 A.2 中，我们针对吸收状态扩散的情况简化这些表达式。最后，在补充材料 A.2.3 中，我们推导出吸收状态扩散的 ELBO 表达式。

# A.1 Generic case
A.1 一般情况

Given the state transition matrix $Q _ { t }$ , prior π, and the latent variables $\mathbf { z } _ { s }$ and $\mathbf { z } _ { t } ,$ where $s < t ,$ let
给定状态转移矩阵 $Q _ { t }$ ，先验 π 以及潜变量 $\mathbf { z } _ { s }$ 和 $\mathbf { z } _ { t } ,$ ，其中 $s < t ,$ 使得

$$
Q _ {t \mid s} = \alpha_ {t \mid s} \mathbf {I} + (1 - \alpha_ {t \mid s}) \mathbf {1} \boldsymbol {\pi} ^ {\top}. \tag {13}
$$

# A.1.1 $q ( \mathbf { z } _ { t } | \mathbf { z } _ { s } )$

Thus, the marginals in (3) correspond to the following forward process:
因此，(3) 中的边缘分布对应于以下前向过程：

$$
\begin{array}{l} q (\mathbf {z} _ {t} | \mathbf {z} _ {s}) = \operatorname{Cat} (\mathbf {z} _ {t}; Q _ {t | s} ^ {\top} \mathbf {z} _ {s}) \\ = \operatorname{Cat} \left(\mathbf {z} _ {t}; \left[ \alpha_ {t | s} \mathbf {I} + \left(1 - \alpha_ {t | s}\right) \mathbf {1} \boldsymbol {\pi} ^ {\top} \right] ^ {\top} \mathbf {z} _ {s}\right) \\ = \operatorname{Cat} \left(\mathbf {z} _ {t}; \alpha_ {t | s} \mathbf {z} _ {s} + \left(1 - \alpha_ {t | s}\right) \boldsymbol {\pi} \mathbf {1} ^ {\top} \mathbf {z} _ {s}\right) \quad \because \mathbf {1} ^ {\top} \mathbf {z} _ {s} = 1 \\ = \operatorname{Cat} \left(\mathbf {z} _ {t}; \alpha_ {t | s} \mathbf {z} _ {s} + \left(1 - \alpha_ {t | s}\right) \boldsymbol {\pi}\right). \tag {14} \\ \end{array}
$$

The above equation indicates that during each diffusion step from $s \to t ,$ a fraction $\left( 1 - \alpha _ { t | s } \right)$ of the probability mass is transferred to the prior distribution π.
上述公式表明，在每个从 $s \to t ,$ 开始的扩散步骤中，有 $\left( 1 - \alpha _ { t | s } \right)$ 比例的概率质量被转移到先验分布π。

# A.1.2 $q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } )$

Austin et al. \[1\] show that the posterior corresponding to (14) is given as follows:
Austin 等人\[1\]表明，对应于(14)的后验分布给出如下：

$$
q \left(\mathbf {z} _ {s} \mid \mathbf {z} _ {t}, \mathbf {x}\right) = \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {Q _ {t \mid s} \mathbf {z} _ {t} \odot Q _ {s} ^ {\top} \mathbf {x}}{\mathbf {z} _ {t} ^ {\top} Q _ {t} ^ {\top} \mathbf {x}}\right), \tag {15}
$$

which we simplify to the following:
我们将其简化为以下形式：

$$
\begin{array}{l} q (\mathbf {z} _ {s} | \mathbf {z} _ {t}, \mathbf {x}) \\ = \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {\left[ \alpha_ {t | s} \mathbf {I} + (1 - \alpha_ {t | s}) \mathbf {1} \boldsymbol {\pi} ^ {\top} \right] \mathbf {z} _ {t} \odot \left[ \alpha_ {s} \mathbf {I} + (1 - \alpha_ {s}) \mathbf {1} \boldsymbol {\pi} ^ {\top} \right] ^ {\top} \mathbf {x}}{\mathbf {z} _ {t} ^ {\top} \left[ \alpha_ {t} \mathbf {I} + (1 - \alpha_ {t}) \mathbf {1} \boldsymbol {\pi} ^ {\top} \right] ^ {\top} \mathbf {x}}\right) \\ = \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {\left[ \alpha_ {t | s} \mathbf {z} _ {t} + (1 - \alpha_ {t | s}) \mathbf {1} \boldsymbol {\pi} ^ {\top} \mathbf {z} _ {t} \right] \odot \left[ \alpha_ {s} \mathbf {x} + (1 - \alpha_ {s}) \boldsymbol {\pi} \right]}{\mathbf {z} _ {t} ^ {\top} \left[ \alpha_ {t} \mathbf {x} + (1 - \alpha_ {t}) \boldsymbol {\pi} \mathbf {1} ^ {\top} \mathbf {x} \right]}\right) \\ = \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {\left[ \alpha_ {t | s} \mathbf {z} _ {t} + (1 - \alpha_ {t | s}) \mathbf {1} \boldsymbol {\pi} ^ {\top} \mathbf {z} _ {t} \right] \odot \left[ \alpha_ {s} \mathbf {x} + (1 - \alpha_ {s}) \boldsymbol {\pi} \right]}{\alpha_ {t} \mathbf {z} _ {t} ^ {\top} \mathbf {x} + (1 - \alpha_ {t}) \mathbf {z} _ {t} ^ {\top} \boldsymbol {\pi}}\right). \quad \because \mathbf {1} ^ {\top} \mathbf {x} = 1 \tag {16} \\ \end{array}
$$

# A.1.3 $p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$

Austin et al. \[1\] approximate the reverse process in the following manner:
奥斯汀等人\[1\]以如下方式近似了逆过程：

$$
p _ {\theta} \left(\mathbf {z} _ {s} \mid \mathbf {z} _ {t}\right) = q \left(\mathbf {z} _ {s} \mid \mathbf {z} _ {t}, \mathbf {x} = \mathbf {x} _ {\theta} \left(\mathbf {z} _ {t}, t\right)\right) = \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {Q _ {t \mid s} \mathbf {z} _ {t} \odot Q _ {s} ^ {\top} \mathbf {x} _ {\theta} \left(\mathbf {z} _ {t} , t\right)}{\mathbf {z} _ {t} ^ {\top} Q _ {t} ^ {\top} \mathbf {x} _ {\theta} \left(\mathbf {z} _ {t} , t\right)}\right). \tag {17}
$$

where $\mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) : \mathcal { V } \times [ 0 , 1 ] \Delta ^ { K }$ is an approximation for x.
其中 $\mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) : \mathcal { V } \times [ 0 , 1 ] \Delta ^ { K }$ 是 x 的一个近似值。

# A.2 Absorbing state
A.2 吸收态

For the absorbing state diffusion process we have ${ \boldsymbol { \pi } } { = } \mathbf { m }$ .
对于吸收状态扩散过程，我们有 ${ \boldsymbol { \pi } } { = } \mathbf { m }$ 。

# A.2.1 $q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } )$

Since, $\mathbf { z } _ { t } \in \{ \mathbf { x } , \mathbf { m } \}$ , takes only 2 values we consider the separate cases: $\mathbf { z } _ { t } = \mathbf { x }$ and $\mathbf { z } _ { t } = \mathbf { m }$ .
由于 $\mathbf { z } _ { t } \in \{ \mathbf { x } , \mathbf { m } \}$ 只取 2 个值，我们考虑分别考虑以下情况： $\mathbf { z } _ { t } = \mathbf { x }$ 和 $\mathbf { z } _ { t } = \mathbf { m }$ 。

Case 1. Consider the case $\mathbf { z } _ { t } = \mathbf { x } \mathbf { i } . \mathbf { e } . \mathbf { z } _ { t }$ is unmasked. From (16), we have the following:
情况 1。考虑 $\mathbf { z } _ { t } = \mathbf { x } \mathbf { i } . \mathbf { e } . \mathbf { z } _ { t }$ 未被遮蔽的情况。根据(16)，我们有以下公式：

$$
\begin{array}{l} q (\mathbf {z} _ {s} | \mathbf {z} _ {t} = \mathbf {x}, \mathbf {x}) \\ = \mathrm{Cat} \left(\mathbf {z} _ {s}; \frac {[ \alpha_ {t | s} \mathbf {x} + (1 - \alpha_ {t | s}) \mathbf {1 m} ^ {\top} \mathbf {x} ] \odot [ \alpha_ {s} \mathbf {x} + (1 - \alpha_ {s}) \mathbf {m} ]}{\alpha_ {t} \mathbf {x} ^ {\top} \mathbf {x} + (1 - \alpha_ {t}) \mathbf {x} ^ {\top} \mathbf {m}}\right) \\ = \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {\left[ \alpha_ {t | s} \mathbf {x} \right] \odot \left[ \alpha_ {s} \mathbf {x} + (1 - \alpha_ {s}) \mathbf {m} \right]}{\alpha_ {t}}\right) \quad \because \mathbf {x} ^ {\top} \mathbf {m} = 0 \\ = \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {\alpha_ {t} \mathbf {x}}{\alpha_ {t}}\right) \quad \because \mathbf {x} \odot \mathbf {m} = \mathbf {0} \text { and } \alpha_ {t} = \alpha_ {t | s} \alpha_ {s} \\ = \operatorname{Cat} \left(\mathbf {z} _ {s}; \mathbf {x}\right) \quad \because \alpha_ {t} = \alpha_ {t | s} \alpha_ {s} \tag {18} \\ \end{array}
$$

Thus, we have the following:
因此，我们有如下：

$$
q \left(\mathbf {z} _ {s} \mid \mathbf {z} _ {t} = \mathbf {x}, \mathbf {x}\right) = \operatorname{Cat} \left(\mathbf {z} _ {s}; \mathbf {x}\right). \tag {19}
$$

Case 2. Consider the case $\mathbf { z } _ { t } = \mathbf { m }$ . By substituting $\mathbf { z } _ { t } = \mathbf { m }$ and π =m in $( 1 6 ) , q ( { \bf z } _ { s } | { \bf z } _ { t } , { \bf x } )$ simplifies to the following:
情况 2。考虑情况 $\mathbf { z } _ { t } = \mathbf { m }$ 。通过代入 $\mathbf { z } _ { t } = \mathbf { m }$ 和π=m 到 $( 1 6 ) , q ( { \bf z } _ { s } | { \bf z } _ { t } , { \bf x } )$ 中，简化如下：

$$
\begin{array}{l} q (\mathbf {z} _ {s} | \mathbf {z} _ {t} = \mathbf {m}, \mathbf {x}) = \operatorname{Cat} \left(\frac {\left(\alpha_ {t | s} \mathbf {m} + (1 - \alpha_ {t | s}) \mathbf {1}\right) \odot \left(\alpha_ {s} \mathbf {x} + (1 - \alpha_ {s}) \mathbf {m}\right)}{(1 - \alpha_ {t})}\right) \\ = \operatorname{Cat} \left(\frac {\left(\alpha_ {t | s} \left(1 - \alpha_ {s}\right) \mathbf {m} + \left(1 - \alpha_ {t | s}\right) \left(1 - \alpha_ {s}\right) \mathbf {m} + \left(\alpha_ {s} - \alpha_ {t}\right) \mathbf {x}\right)}{\left(1 - \alpha_ {t}\right)}\right) \\ \end{array}
$$

 

$$
= \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {\left(1 - \alpha_ {s}\right) \mathbf {m} + \left(\alpha_ {s} - \alpha_ {t}\right) \mathbf {x}}{1 - \alpha_ {t}}\right) \tag {20}
$$

Note that the above categorical distribution is non-zero for $\mathbf { z } _ { s } \in \{ \mathbf { x } , \mathbf { m } \}$ and zero for every other value. The non-zero values are specified as follows:
请注意，上述分类分布对于 $\mathbf { z } _ { s } \in \{ \mathbf { x } , \mathbf { m } \}$ 非零，对于其他值均为零。非零值指定如下：

$$
q (\mathbf {z} _ {s} = \mathbf {x} | \mathbf {z} _ {t} = \mathbf {m}, \mathbf {x}) = \frac {\alpha_ {s} - \alpha_ {t}}{1 - \alpha_ {t}} \tag {21}
$$

 

$$
q \left(\mathbf {z} _ {s} = \mathbf {m} \mid \mathbf {z} _ {t} = \mathbf {m}, \mathbf {x}\right) = \frac {1 - \alpha_ {s}}{1 - \alpha_ {t}} \tag {22}
$$

Combining Cases 1 and 2, we get:
结合情况 1 和情况 2，我们得到：

$$
q (\mathbf {z} _ {s} | \mathbf {z} _ {t}, \mathbf {x}) = \left\{ \begin{array}{l l} \operatorname{Cat} \left(\mathbf {z} _ {s}; \mathbf {z} _ {t}\right) & \mathbf {z} _ {t} \neq \mathbf {m}, \\ \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {(1 - \alpha_ {s}) \mathbf {m} + (\alpha_ {s} - \alpha_ {t}) \mathbf {x}}{1 - \alpha_ {t}}\right) & \mathbf {z} _ {t} = \mathbf {m}. \end{array} \right. \tag {23}
$$

# A.2.2 $p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$

For the absorbing state diffusion process with ${ \boldsymbol { \pi } } = \mathbf { m } .$ , we want to simplify the (17). For this reason, we consider 2 cases: first, when $\mathbf { z } _ { t } \neq \mathbf { m }$ (case 1), second, when $\mathbf { z } _ { t } \neq \mathbf { m } \left( \mathbf { c a s e } \ 2 \right)$ .
对于具有 ${ \boldsymbol { \pi } } = \mathbf { m } .$ 的吸收态扩散过程，我们希望简化(17)。为此，我们考虑两种情况：首先，当 $\mathbf { z } _ { t } \neq \mathbf { m }$ （情况 1），其次，当 $\mathbf { z } _ { t } \neq \mathbf { m } \left( \mathbf { c a s e } \ 2 \right)$ 。

Case 1. Consider the case when $\mathbf { z } _ { t } \neq \mathbf { m } .$ . (17) simplifies to the following:
情况 1。考虑 $\mathbf { z } _ { t } \neq \mathbf { m } .$ 的情况。(17) 简化为如下：

$$
\begin{array}{l} p _ {\theta} (\mathbf {z} _ {s} | \mathbf {z} _ {t} \neq \mathbf {m}) = \mathrm{Cat} \left(\mathbf {z} _ {s}; \frac {Q _ {t | s} \mathbf {z} _ {t} \odot Q _ {s} ^ {\top} \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t)}{\mathbf {z} _ {t} ^ {\top} Q _ {t} ^ {\top} \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t)}\right) \\ = \mathrm{Cat} \left(\mathbf {z} _ {s}; \frac {Q _ {t | s} \mathbf {z} _ {t} \odot Q _ {s} ^ {\top} \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t)}{\left[ Q _ {t} \mathbf {z} _ {t} \right] ^ {\top} \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t)}\right) \\ = \mathrm{Cat} \left(\mathbf {z} _ {s}; \frac {[ \alpha_ {t | s} \mathbf {z} _ {t} ] \odot [ \alpha_ {s} \mathbf {I} + (1 - \alpha_ {s}) \mathbf {m 1} ^ {\top} ] \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t)}{[ \alpha_ {t} \mathbf {z} _ {t} ] ^ {\top} \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t)}\right) \\ = \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {\left[ \alpha_ {t | s} \mathbf {z} _ {t} \right] \odot \left[ \alpha_ {s} \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) + (1 - \alpha_ {s}) \mathbf {m} \langle \mathbf {1} , \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) \rangle \right]}{\alpha_ {t} \langle \mathbf {z} _ {t} , \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) \rangle}\right) \\ \end{array}
$$

since $\langle \mathbf { 1 } , \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) \rangle = 1$ , we have the following:
由于 $\langle \mathbf { 1 } , \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) \rangle = 1$ ，我们有以下：

$$
= \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {\left[ \alpha_ {t | s} \mathbf {z} _ {t} \right] \odot \left[ \alpha_ {s} \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) + (1 - \alpha_ {s}) \mathbf {m} \right]}{\alpha_ {t} \langle \mathbf {z} _ {t} , \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) \rangle}\right)
$$

since $\mathbf { z } _ { t } \odot \mathbf { m } { = } \mathbf { 0 }$ , we have the following:
自 $\mathbf { z } _ { t } \odot \mathbf { m } { = } \mathbf { 0 }$ 以来，我们有以下内容：

$$
= \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {\alpha_ {t} \mathbf {z} _ {t} \odot \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t)}{\alpha_ {t} \langle \mathbf {z} _ {t} , \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) \rangle}\right)
$$

 

$$
= \operatorname{Cat} \left(\mathbf {z} _ {s}; \mathbf {z} _ {t}\right) \tag {24}
$$

Case 2. Consider the case when ${ \bf z } _ { t } = { \bf m } . ( \mathrm { ~ ~ \omega ~ } )$ simplifies to the following:
情况 2。考虑 ${ \bf z } _ { t } = { \bf m } . ( \mathrm { ~ ~ \omega ~ } )$ 简化为以下情况：

$$
\begin{array}{l} p _ {\theta} (\mathbf {z} _ {s} | \mathbf {z} _ {t} = \mathbf {m}) = \mathrm{Cat} \left(\mathbf {z} _ {s}; \frac {Q _ {t | s} \mathbf {m} \odot Q _ {s} ^ {\top} \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t)}{\mathbf {m} ^ {\top} Q _ {t} ^ {\top} \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t)}\right) \\ = \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {Q _ {t | s} \mathbf {m} \odot Q _ {s} ^ {\top} \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t)}{\left[ Q _ {t} \mathbf {m} \right] ^ {\top} \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t)}\right) \\ = \mathrm{Cat} \left(\mathbf {z} _ {s}; \frac {[ \alpha_ {t | s} \mathbf {m} + (1 - \alpha_ {t | s}) \mathbf {1} ] \odot [ \alpha_ {s} \mathbf {I} + (1 - \alpha_ {s}) \mathbf {m} \mathbf {1} ^ {\top} ] \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t)}{[ \alpha_ {t} \mathbf {m} + (1 - \alpha_ {t}) \mathbf {1} ] ^ {\top} \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t)}\right) \\ = \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {\left[ \alpha_ {t | s} \mathbf {m} + (1 - \alpha_ {t | s}) \mathbf {1} \right] \odot \left[ \alpha_ {s} \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) + (1 - \alpha_ {s}) \mathbf {m} \langle \mathbf {1} , \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) \rangle \right]}{\alpha_ {t} \langle \mathbf {m} , \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) \rangle + (1 - \alpha_ {t}) \langle \mathbf {1} , \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) \rangle}\right) \\ = \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {\left[ \alpha_ {t | s} \mathbf {m} + (1 - \alpha_ {t | s}) \mathbf {1} \right] \odot \left[ \alpha_ {s} \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) + (1 - \alpha_ {s}) \mathbf {m} \right]}{\alpha_ {t} \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {m} \rangle + (1 - \alpha_ {t})}\right) \\ \end{array}
$$

 

$$
= \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {\alpha_ {t} \mathbf {m} \odot \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) + \left(\alpha_ {s} - \alpha_ {t}\right) \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) + \left(1 - \alpha_ {s}\right) \mathbf {m}}{\alpha_ {t} \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {m} \rangle + (1 - \alpha_ {t})}\right) \tag {25}
$$

Note that the above categorical distribution, we can obtain the values for $p _ { \theta } ( \mathbf { z } _ { s } = \mathbf { x } | \mathbf { z } _ { t } = \mathbf { m } )$ and $p _ { \theta } ( \mathbf { z } _ { s } = \mathbf { m } | \mathbf { z } _ { t } = \mathbf { m } )$ which are as follows:
请注意，上述的范畴分布，我们可以得到 $p _ { \theta } ( \mathbf { z } _ { s } = \mathbf { x } | \mathbf { z } _ { t } = \mathbf { m } )$ 和 $p _ { \theta } ( \mathbf { z } _ { s } = \mathbf { m } | \mathbf { z } _ { t } = \mathbf { m } )$ 的值，如下所示：

$$
p _ {\theta} \left(\mathbf {z} _ {s} = \mathbf {x} \mid \mathbf {z} _ {t} = \mathbf {m}\right) = \frac {\left(\alpha_ {s} - \alpha_ {t}\right) \left\langle \mathbf {x} _ {\theta} \left(\mathbf {z} _ {t} , t\right) , \mathbf {x} \right\rangle}{\alpha_ {t} \left\langle \mathbf {x} _ {\theta} \left(\mathbf {z} _ {t} , t\right) , \mathbf {m} \right\rangle + (1 - \alpha_ {t})} \tag {26}
$$

 

$$
p _ {\theta} \left(\mathbf {z} _ {s} = \mathbf {m} \mid \mathbf {z} _ {t} = \mathbf {m}\right) = \frac {\alpha_ {s} \left\langle \mathbf {x} _ {\theta} \left(\mathbf {z} _ {t} , t\right) , \mathbf {m} \right\rangle + \left(1 - \alpha_ {s}\right)}{\alpha_ {t} \left\langle \mathbf {x} _ {\theta} \left(\mathbf {z} _ {t} , t\right) , \mathbf {m} \right\rangle + \left(1 - \alpha_ {t}\right)} \tag {27}
$$

As a sanity check, we can verify that (26) reduces to (21), and (27) reduces to (22) if our denoising network can reconstruct x perfectly, $\mathbf { i . e . , x } _ { \theta } ( \mathbf { z } _ { t } , t ) = \mathbf { x }$ .
作为一个合理性检查，我们可以验证，如果我们的去噪网络能够完美地重建 x， $\mathbf { i . e . , x } _ { \theta } ( \mathbf { z } _ { t } , t ) = \mathbf { x }$ ，则(26)简化为(21)，(27)简化为(22)。

Combining (24) and (25), we get the following expression for the reverse process parameterization:
结合(24)和(25)，我们得到反向过程参数化的如下表达式：

$$
p _ {\theta} \left(\mathbf {z} _ {s} \mid \mathbf {z} _ {t}\right) = \left\{ \begin{array}{l l} \operatorname{Cat} \left(\mathbf {z} _ {s}; \mathbf {z} _ {t}\right) & \mathbf {z} _ {t} \neq \mathbf {m}, \\ \operatorname{Cat} \left(\mathbf {z} _ {s}; \frac {\alpha_ {t} \mathbf {m} \odot \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) + \left(\alpha_ {s} - \alpha_ {t}\right) \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) + \left(1 - \alpha_ {s}\right) \mathbf {m}}{\alpha_ {t} \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {m} \rangle + (1 - \alpha_ {t})}\right) & \mathbf {z} _ {t} = \mathbf {m}. \end{array} \right. \tag {28}
$$

# A.2.3 Diffusion Loss
A.2.3 扩散损失

For a given T , Let $\begin{array} { r } { \mathcal { L } _ { T } = \mathbb { E } _ { t \in \{ \frac { 1 } { r } , \frac { 2 } { r } , \dots , 1 \} } \mathbb { E } _ { q ( \mathbf { z } _ { t } \mid \mathbf { x } ) } T \mathrm { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) \big | \big | p _ { \theta } \big ( \mathbf { z } _ { s } | \mathbf { z } _ { t } \big ) \big ) } \end{array}$ denote the diffusion loss. We break down the computation of DKL $( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) \| p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } ) )$ into 2 cases: $\mathbf { z } _ { t } = \mathbf { x }$ (case 1) and $\mathbf { z } _ { t } = \mathbf { m } \left( \mathbf { c a s e } \ 2 \right)$ .
对于给定的 T，令 $\begin{array} { r } { \mathcal { L } _ { T } = \mathbb { E } _ { t \in \{ \frac { 1 } { r } , \frac { 2 } { r } , \dots , 1 \} } \mathbb { E } _ { q ( \mathbf { z } _ { t } \mid \mathbf { x } ) } T \mathrm { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) \big | \big | p _ { \theta } \big ( \mathbf { z } _ { s } | \mathbf { z } _ { t } \big ) \big ) } \end{array}$ 表示扩散损失。我们将 DKL $( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) \| p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } ) )$ 的计算分解为 2 种情况： $\mathbf { z } _ { t } = \mathbf { x }$ （情况 1）和 $\mathbf { z } _ { t } = \mathbf { m } \left( \mathbf { c a s e } \ 2 \right)$ 。

Case 1: consider the case $\mathbf { z } _ { t } = \mathbf { x }$ . Let’s simplify $\operatorname { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } = \mathbf { x } , \mathbf { x } ) \big | | p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } = \mathbf { x } ) \big )$ .
情况 1：考虑 $\mathbf { z } _ { t } = \mathbf { x }$ 的情况。让我们简化 $\operatorname { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } = \mathbf { x } , \mathbf { x } ) \big | | p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } = \mathbf { x } ) \big )$ 。

$$
\mathrm{D} _ {\mathrm{KL}} \left(q \left(\mathbf {z} _ {s} \mid \mathbf {z} _ {t} = \mathbf {x}, \mathbf {x}\right) \| p _ {\theta} \left(\mathbf {z} _ {s} \mid \mathbf {z} _ {t} = \mathbf {x}\right)\right)
$$

 

$$
= \mathrm{D} _ {\mathrm{KL}} \left(\mathbf {z} _ {t} \| \mathbf {z} _ {t}\right) \quad \text { From (23) and (24) }
$$

  

$$
= 0 \tag {29}
$$

Case 2: Consider the case $\mathbf { z } _ { t } = \mathbf { m }$ . Let’s simplify $\mathrm { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } { = } \mathbf { m } , \mathbf { x } ) \big | \big | p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } { = } \mathbf { m } ) \big )$ .
案例 2：考虑情况 $\mathbf { z } _ { t } = \mathbf { m }$ 。让我们简化 $\mathrm { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } { = } \mathbf { m } , \mathbf { x } ) \big | \big | p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } { = } \mathbf { m } ) \big )$ 。

$$
\mathrm{D} _ {\mathrm{KL}} \left(q \left(\mathbf {z} _ {s} \mid \mathbf {z} _ {t} = \mathbf {m}, \mathbf {x}\right) \| p _ {\theta} \left(\mathbf {z} _ {s} \mid \mathbf {z} _ {t} = \mathbf {m}\right)\right)
$$

 

$$
= \sum_ {\mathbf {z} _ {s}} q (\mathbf {z} _ {s} | \mathbf {z} _ {t} = \mathbf {m}, \mathbf {x}) \log \frac {q (\mathbf {z} _ {s} | \mathbf {z} _ {t} = \mathbf {m} , \mathbf {x})}{p _ {\theta} (\mathbf {z} _ {s} | \mathbf {z} _ {t} = \mathbf {m})}
$$

 

$$
= \sum_ {\mathbf {z} _ {s} \in \{\mathbf {x}, \mathbf {m} \}} q (\mathbf {z} _ {s} | \mathbf {z} _ {t} = \mathbf {m}, \mathbf {x}) \log \frac {q (\mathbf {z} _ {s} | \mathbf {z} _ {t} = \mathbf {m} , \mathbf {x})}{p _ {\theta} (\mathbf {z} _ {s} | \mathbf {z} _ {t} = \mathbf {m})}
$$

 

$$
= \underbrace {q (\mathbf {z} _ {s} = \mathbf {x} | \mathbf {z} _ {t} = \mathbf {m} , \mathbf {x}) \log \frac {q (\mathbf {z} _ {s} = \mathbf {x} | \mathbf {z} _ {t} = \mathbf {m} , \mathbf {x})}{p _ {\theta} (\mathbf {z} _ {s} = \mathbf {x} | \mathbf {z} _ {t} = \mathbf {m})}} _ {\text {Simplify using (21) and (26)}}
$$

 

$$
+ \underbrace {q (\mathbf {z} _ {s} = \mathbf {m} | \mathbf {z} _ {t} = \mathbf {m} , \mathbf {x}) \log \frac {q (\mathbf {z} _ {s} = \mathbf {m} | \mathbf {z} _ {t} = \mathbf {m} , \mathbf {x})}{p _ {\theta} (\mathbf {z} _ {s} = \mathbf {m} | \mathbf {z} _ {t} = \mathbf {m})}} _ {\text { Simplify using (22) and (27) }}
$$

 

$$
= \frac {\alpha_ {s} - \alpha_ {t}}{1 - \alpha_ {t}} \log \frac {\alpha_ {t} \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {m} \rangle + (1 - \alpha_ {t})}{(1 - \alpha_ {t}) \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {x} \rangle}
$$

 

$$
+ \frac {1 - \alpha_ {s}}{1 - \alpha_ {t}} \log \frac {(1 - \alpha_ {s}) (\alpha_ {t} \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {m} \rangle + (1 - \alpha_ {t}))}{(1 - \alpha_ {t}) (\alpha_ {s} \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {m} \rangle + (1 - \alpha_ {s}))} \tag {30}
$$

Thus, $\begin{array} { r } { \operatorname { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) \big | \big | p _ { \theta } \big ( \mathbf { z } _ { s } | \mathbf { z } _ { t } \big ) \big ) } \end{array}$ can be written in the following manner where $\scriptstyle \left. \mathbf { z } _ { t } , \mathbf { x } \right.$ evaluates to 1 if $\mathbf { z } _ { t } = \mathbf { x }$ and $\mathbf { \delta } ( \mathbf { z } _ { t } , \mathbf { m } )$ evaluates to 1 if $\mathbf { z } _ { t } = \mathbf { m } \mathrm { . }$ :
因此， $\begin{array} { r } { \operatorname { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) \big | \big | p _ { \theta } \big ( \mathbf { z } _ { s } | \mathbf { z } _ { t } \big ) \big ) } \end{array}$ 可以以以下方式表示，其中 $\scriptstyle \left. \mathbf { z } _ { t } , \mathbf { x } \right.$ 在 $\mathbf { z } _ { t } = \mathbf { x }$ 时评估为 1， $\mathbf { \delta } ( \mathbf { z } _ { t } , \mathbf { m } )$ 在 $\mathbf { z } _ { t } = \mathbf { m } \mathrm { . }$ 时评估为 1：

$$
\mathrm{D} _ {\mathrm{KL}} (q (\mathbf {z} _ {s} | \mathbf {z} _ {t}, \mathbf {x}) \| p _ {\theta} (\mathbf {z} _ {s} | \mathbf {z} _ {t}))
$$

 

$$
= \underbrace {\mathrm{D} _ {\mathrm{KL}} \left(q \left(\mathbf {z} _ {s} \mid \mathbf {z} _ {t} = \mathbf {x} , \mathbf {x}\right) \| p _ {\theta} \left(\mathbf {z} _ {s} \mid \mathbf {z} _ {t} = \mathbf {x}\right)\right)} _ {= 0, \text {from (29)}} \langle \mathbf {z} _ {t}, \mathbf {x} \rangle + \underbrace {\mathrm{D} _ {\mathrm{KL}} \left(q \left(\mathbf {z} _ {s} \mid \mathbf {z} _ {t} = \mathbf {m} , \mathbf {x}\right) \| p _ {\theta} \left(\mathbf {z} _ {s} \mid \mathbf {z} _ {t} = \mathbf {m}\right)\right)} _ {\text {Given by (30)}} \langle \mathbf {z} _ {t}, \mathbf {m} \rangle \tag {31}
$$

Thus, we derive the diffusion loss, $\mathcal { L } _ { T }$ , in the following manner:
因此，我们以以下方式推导出扩散损失 $\mathcal { L } _ { T }$ ：

$$
\begin{array}{l} \mathcal {L} _ {T} = \mathbb {E} _ {t \in \left\{\frac {1}{T}, \frac {2}{T}, \dots , 1 \right\}} \mathbb {E} _ {q (\mathbf {z} _ {t} | \mathbf {x})} T \mathrm{D} _ {\mathrm{KL}} \big (q (\mathbf {z} _ {s} | \mathbf {z} _ {t}, \mathbf {x}) \| p _ {\theta} (\mathbf {z} _ {s} | \mathbf {z} _ {t}) \big) \\ = \mathbb {E} _ {t \in \left\{\frac {1}{T}, \frac {2}{T}, \dots , 1 \right\}} \mathbb {E} _ {q (\mathbf {z} _ {t} | \mathbf {x})} T \left[ \frac {\alpha_ {s} - \alpha_ {t}}{1 - \alpha_ {t}} \log \frac {\alpha_ {t} \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {m} \rangle + (1 - \alpha_ {t})}{(1 - \alpha_ {t}) \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {x} \rangle} \right. \\ \left. + \frac {1 - \alpha_ {s}}{1 - \alpha_ {t}} \log \frac {\left(1 - \alpha_ {s}\right) \left(\alpha_ {t} \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {m} \rangle + (1 - \alpha_ {t})\right)}{\left(1 - \alpha_ {t}\right) \left(\alpha_ {s} \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {m} \rangle + (1 - \alpha_ {s})\right)} \right] \langle \mathbf {z} _ {t}, \mathbf {m} \rangle \tag {32} \\ \end{array}
$$

Note that $\mathcal { L } _ { T }$ is 0 if $\mathbf { z } _ { t }$ is an unmasked token i.e. $\mathbf { z } _ { t } = \mathbf { x } .$
请注意，如果 $\mathbf { z } _ { t }$ 是一个未遮盖的标记，即 $\mathbf { z } _ { t } = \mathbf { x } .$ ，则 $\mathcal { L } _ { T }$ 为 0。

# A.2.4 NELBO

Austin et al. \[1\], Sohl-Dickstein et al. \[54\] model $\alpha _ { i }$ as $\begin{array} { r } { ( \alpha _ { i } ) _ { i \in \{ 1 , \dots , T \} } = 1 - \frac { i } { T } } \end{array}$ given latents $\mathbf { z } _ { 1 , \ldots , T } ,$ However, in this paper, we denote the latents as ${ \mathbf z } _ { t ( 0 ) , \dots , t ( T ) }$ ; and hence, the $\alpha _ { t ( i ) }$ are given as follows:
Austin 等人 \[1\]、Sohl-Dickstein 等人 \[54\] 将 $\alpha _ { i }$ 模型化为 $\begin{array} { r } { ( \alpha _ { i } ) _ { i \in \{ 1 , \dots , T \} } = 1 - \frac { i } { T } } \end{array}$ ，给定潜在变量 $\mathbf { z } _ { 1 , \ldots , T } ,$ 。然而，在本文中，我们将潜在变量表示为 ${ \mathbf z } _ { t ( 0 ) , \dots , t ( T ) }$ ；因此， $\alpha _ { t ( i ) }$ 如下所示：

$$
\begin{array}{l} (\alpha_ {i}) _ {i \in \{1, \dots , T \}} = 1 - \frac {i}{T} \\ \Longrightarrow (\alpha_ {i}) _ {k \in \{1, \dots , T + 1 \}} = 1 - \frac {i}{T + 1} \quad \text { For } T + 1 \text { latents } \\ \Longrightarrow (\alpha_ {i}) _ {i \in \{0, \dots , T \}} = 1 - \frac {i + 1}{T + 1} \quad \text { Offsetting the indices by } 1. \\ \end{array}
$$

 

$$
\Longrightarrow \left(\alpha_ {t (i)}\right) _ {i \in \{0, \dots , T \}} = 1 - \frac {i + 1}{T + 1} \quad \text { Switching the notations from } \alpha_ {i} \text { to } \alpha_ {t (i)}. \tag {33}
$$

Consequently, from Equation 33, we derive that
因此，根据方程式 33，我们推导出

$$
\alpha_ {t (0)} = \frac {T}{T + 1}, \tag {34}
$$

 

$$
\alpha_ {t (T)} = 0. \tag {35}
$$

Thus we have the following:
因此我们有以下公式：

$$
\mathbf {z} _ {t (0)} \sim \operatorname{Cat} \left(\cdot ; \alpha_ {t = 0} \mathbf {x} + \left(1 - \alpha_ {t = 0}\right) \mathbf {m}\right) = \operatorname{Cat} \left(\cdot ; \frac {T}{T + 1} \mathbf {x} + \frac {1}{T + 1} \mathbf {m}\right), \tag {36}
$$

 

$$
q \left(\mathbf {z} _ {t (T)} \mid \mathbf {x}\right) = \operatorname{Cat} \left(.; \alpha_ {t = 1} \mathbf {x} + \left(1 - \alpha_ {t = 1}\right) \mathbf {m}\right) = \operatorname{Cat} (.; \mathbf {m}), \tag {37}
$$

 

$$
p _ {\theta} \left(\mathbf {z} _ {t (T)}\right) = \operatorname{Cat} (.; \mathbf {m}) \tag {38}
$$

The NELBO (2) simplifies to the following:
NELBO（2）简化为如下：

$$
\begin{array}{l} \mathbb {E} _ {q} \left[ - \log p _ {\theta} (\mathbf {x} | \mathbf {z} _ {t (0)}) + \underbrace {\mathcal {L} _ {T}} _ {\text {Compute using (32)}} \right] + \underbrace {D _ {\mathrm{KL}} [ q (\mathbf {z} _ {t (T)} | \mathbf {x}) \| p _ {\theta} (\mathbf {z} _ {t (T)}) ]} _ {= 0 \text {using (37) and (38)}} \\ = \mathbb {E} _ {q, t} \left[ - \log p _ {\theta} (\mathbf {x} | \mathbf {z} _ {t (0)}) + T \left[ \frac {\alpha_ {s} - \alpha_ {t}}{1 - \alpha_ {t}} \log \frac {\alpha_ {t} \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {m} \rangle + (1 - \alpha_ {t})}{(1 - \alpha_ {t}) \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {x} \rangle} \right. \right. \\ \left. \left. + \frac {1 - \alpha_ {s}}{1 - \alpha_ {t}} \log \frac {\left(1 - \alpha_ {s}\right) \left(\alpha_ {t} \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {m} \rangle + (1 - \alpha_ {t})\right)}{\left(1 - \alpha_ {t}\right) \left(\alpha_ {s} \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {m} \rangle + (1 - \alpha_ {s})\right)} \right] \langle \mathbf {z} _ {t}, \mathbf {m} \rangle \right] \tag {39} \\ \end{array}
$$

# Appendix B MDLM
附录 B MDLM

In this section, we show how SUBS parameterization can simplify the functional form of the NELBO as defined in (39).
在本节中，我们展示了 SUBS 参数化如何简化（39）中定义的 NELBO 的函数形式。

# B.1 Rao-Blackwellization
B.1 Rao-Blackwell 化

We employ the RB techniques as described in Sec. 3.2.3 to simplify the NELBO (39) to (41) using RB2, and further to (43) using RB1.
我们采用第 3.2.3 节中描述的 RB 技术，使用 RB2 将 NELBO（39）简化为（41），并进一步使用 RB1 简化为（43）。

# B.1.1 Zero Masking Probabilities
B.1.1 零掩码概率

Using “Zero Masking Probabilities” (RB2) from Sec. 3.2.3, we set $\langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle = 0$ in (32) to obtain the following simplified diffusion loss:
使用第 3.2.3 节的“零掩码概率”（RB2），我们将（32）中的 $\langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle = 0$ 设置为以下简化扩散损失：

$$
\begin{array}{l} \mathcal {L} _ {T} ^ {\mathrm{RB2}} = \mathbb {E} _ {t \in \left\{\frac {1}{T}, \frac {2}{T}, \dots , 1 \right\}} \mathbb {E} _ {q (\mathbf {z} _ {t} | \mathbf {x})} T \left[ \frac {\alpha_ {s} - \alpha_ {t}}{1 - \alpha_ {t}} \log \frac {1}{\langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t) , \mathbf {x} \rangle} \right] \langle \mathbf {z} _ {t}, \mathbf {m} \rangle \\ = \mathbb {E} _ {t \in \left\{\frac {1}{T}, \frac {2}{T}, \dots , 1 \right\}} \mathbb {E} _ {q (\mathbf {z} _ {t} | \mathbf {x})} T \left[ \frac {\alpha_ {t} - \alpha_ {s}}{1 - \alpha_ {t}} \log \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t}, t), \mathbf {x} \rangle \right] \langle \mathbf {z} _ {t}, \mathbf {m} \rangle . \tag {40} \\ \end{array}
$$

The corresponding Rao-Blackwellized NELBO is given as:
相应的 Rao-Blackwell 化 NELBO 为：

$$
\begin{array}{l} \mathbb {E} _ {q} \left[ - \log p _ {\theta} (\mathbf {x} | \mathbf {z} _ {t (0)}) + \underbrace {\mathcal {L} _ {T} ^ {\mathrm{RB2}}} _ {\text { Compute using (40) }} \right] + \underbrace {D _ {\mathrm{KL}} [ q (\mathbf {z} _ {t (T)} | \mathbf {x}) \| p _ {\theta} (\mathbf {z} _ {t (T)}) ]} _ {= 0 \text { using (37) and (38) }} \\ = \mathbb {E} _ {q, t} \left[ - \log p _ {\theta} (\mathbf {x} | \mathbf {z} _ {t (0)}) + T \left[ \frac {\alpha_ {t} - \alpha_ {s}}{1 - \alpha_ {t}} \log \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t}, t), \mathbf {x} \rangle \right] \langle \mathbf {z} _ {t}, \mathbf {m} \rangle \right] \tag {41} \\ \end{array}
$$

# B.1.2 Carry Over Unmasking
B.1.2 携带过移除遮蔽

Notice that the term $\langle \mathbf { z } _ { t } , \mathbf { m } \rangle$ in (40) is intended to reduce the diffusion loss to zero when $\mathbf { z } _ { t } = \mathbf { x } .$ . Now, we will demonstrate that, by applying “Carry Over Unmasking” (RB1) from Sec. 3.2.3, $\langle \mathbf { z } _ { t } , \mathbf { m } \rangle$ can be removed from (40).
注意到公式(40)中的 $\langle \mathbf { z } _ { t } , \mathbf { m } \rangle$ 旨在当 $\mathbf { z } _ { t } = \mathbf { x } .$ 时将扩散损失降至零。现在，我们将证明通过应用第 3.2.3 节的“携带过移除遮蔽”（RB1）， $\langle \mathbf { z } _ { t } , \mathbf { m } \rangle$ 可以从(40)中移除。

Recall that RB1 guarantees ${ \bf x } _ { \theta } ( { \bf z } _ { t } , t ) = { \bf x }$ when $\mathbf { z } _ { t } = \mathbf { x }$ . Thus, with the RB1 parameterization, the diffusion loss in (40) becomes zero for $\mathbf { z } _ { t } = \mathbf { x } .$ , as $\log \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle = 0$ . Consequently, $\langle \mathbf { z } _ { t } , \mathbf { m } \rangle$ can be safely omitted from (41), yielding the following diffusion loss:
回想 RB1 保证了当 $\mathbf { z } _ { t } = \mathbf { x }$ 时 ${ \bf x } _ { \theta } ( { \bf z } _ { t } , t ) = { \bf x }$ 。因此，在 RB1 参数化下，对于 $\mathbf { z } _ { t } = \mathbf { x } .$ ，公式(40)中的扩散损失变为零，如 $\log \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle = 0$ 所示。因此， $\langle \mathbf { z } _ { t } , \mathbf { m } \rangle$ 可以从(41)中安全地省略，得到以下扩散损失：

$$
\mathcal {L} _ {T} ^ {\mathrm{RB2+RB1}} = \mathbb {E} _ {t \in \left\{\frac {1}{T}, \frac {2}{T}, \dots , 1 \right\}} \mathbb {E} _ {q (\mathbf {z} _ {t} | \mathbf {x})} T \left[ \frac {\alpha_ {t} - \alpha_ {s}}{1 - \alpha_ {t}} \log \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t}, t), \mathbf {x} \rangle \right] \tag {42}
$$

# B.1.3 NELBO

Thus, we have the following NELBO:
因此，我们有以下 NELBO：

$$
\begin{array}{l} \mathbb {E} _ {q} \left[ - \log p _ {\theta} (\mathbf {x} | \mathbf {z} _ {t (0)}) + \underbrace {\mathcal {L} _ {T} ^ {\mathrm{RB2+RB1}}} _ {\text {Compute using (42)}} \right] + \underbrace {D _ {\mathrm{KL}} [ q (\mathbf {z} _ {t (T)} | \mathbf {x}) \| p _ {\theta} (\mathbf {z} _ {t (T)}) ]} _ {= 0 \text {using (37) and (38)}} \\ = \mathbb {E} _ {q, t} \left[ - \log p _ {\theta} (\mathbf {x} | \mathbf {z} _ {t (0)}) + T \left[ \frac {\alpha_ {t} - \alpha_ {s}}{1 - \alpha_ {t}} \log \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t}, t), \mathbf {x} \rangle \right] \right] \tag {43} \\ \end{array}
$$

Comparing (43) and (41). Note that due to RB1, log $p _ { \theta } \big ( \mathbf { x } \big | \mathbf { z } _ { t ( 0 ) } \big )$ in (43) reduces to 0 every time zt(0) = x as explained in (45). However, this is not the case in (41), even though it has a functionally similar expression to (43). Because of this reason (43) should lead to a better likelihood estimate and we empirically verify this in Table 8.
比较 (43) 和 (41)。注意，由于 RB1，(43) 中的 log $p _ { \theta } \big ( \mathbf { x } \big | \mathbf { z } _ { t ( 0 ) } \big )$ 每当 zt(0) = x 时都会减少到 0，正如 (45) 中所解释的那样。然而，在 (41) 中情况并非如此，尽管它具有与 (43) 函数上相似的表示。由于这个原因，(43) 应该会导致更好的似然估计，我们在表 8 中进行了经验验证。

# B.2 Continuous Time
B.2 连续时间

# B.2.1 Diffusion Loss
B.2.1 扩散损失

To derive the continuous-time diffusion loss, $\mathcal { L } _ { \mathrm { d i f f u s i o n } } ^ { \infty }$ , we consider the limiting case lim $\mathbf { \Omega } ^ { \mathrm { i } } T \mathbf { } \infty \mathcal { L } _ { T } ^ { \mathrm { R B 2 + R B 1 } } ( 4 2 )$ :
为了推导连续时间扩散损失 $\mathcal { L } _ { \mathrm { d i f f u s i o n } } ^ { \infty }$ ，我们考虑极限情况 lim $\mathbf { \Omega } ^ { \mathrm { i } } T \mathbf { } \infty \mathcal { L } _ { T } ^ { \mathrm { R B 2 + R B 1 } } ( 4 2 )$ :

$$
\mathcal {L} _ {\text { diffusion }} ^ {\infty} = \lim _ {T \rightarrow \infty} \mathcal {L} _ {T} ^ {\mathrm{RB2+RB1}}
$$

 

$$
\begin{array}{l} = \mathbb {E} _ {t \in \left\{\frac {1}{T}, \frac {2}{T}, \dots , 1 \right\}, q (\mathbf {z} _ {t} | \mathbf {x})} \left[ \lim _ {T \rightarrow \infty} T \frac {\alpha_ {t} - \alpha_ {s}}{1 - \alpha_ {t}} \log \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t}, t), \mathbf {x} \rangle \right] \\ = \mathbb {E} _ {t \sim \mathcal {U} [ 0, 1 ], q (\mathbf {z} _ {t} | \mathbf {x})} \left[ \frac {\alpha_ {t} ^ {\prime}}{1 - \alpha_ {t}} \log \left\langle \mathbf {x} _ {\theta} \left(\mathbf {z} _ {t}, t\right), \mathbf {x} \right\rangle\right] \quad \text {Using} \lim _ {T \rightarrow \infty} T \left(\alpha_ {t} - \alpha_ {s}\right) = \alpha_ {t} ^ {\prime} \tag {44} \\ \end{array}
$$

# B.2.2 Reconstruction Loss
B.2.2 重建损失

For the continous time case, from (36), we have
对于连续时间情况，从(36)式，我们有

$$
\begin{array}{l} \mathbf {z} _ {t (0)} \sim \lim _ {T \rightarrow \infty} \operatorname{Cat} \left(\cdot ; \frac {T}{T + 1} \mathbf {x} + \frac {1}{T + 1} \mathbf {m}\right) \\ \Longrightarrow \mathbf {z} _ {t (0)} \sim \operatorname{Cat} (.; \mathbf {x}) \\ \Longrightarrow \mathbf {z} _ {t (0)} = \mathbf {x} \tag {45} \\ \end{array}
$$

Thus, the reconstruction loss reduces to 0 in the following manner:
因此，重建损失按照以下方式简化为 0：

$$
\begin{array}{l} \mathcal {L} _ {\text { recons }} = - \log p _ {\theta} (\mathbf {x} | \mathbf {z} _ {t (0)}) \\ = - \log p _ {\theta} (\mathbf {x} | \mathbf {z} _ {t (0)} = \mathbf {x}) \quad \text { From } (45) \\ = - \log \left\langle \mathbf {x} _ {\theta} (\mathbf {x}, t (0)), \mathbf {x} \right\rangle \\ = - \log \langle \mathbf {x}, \mathbf {x} \rangle \quad \text { Due to ``carry - over unmasking'' } \mathbf {x} _ {\theta} (\mathbf {x}, t (0)) = \mathbf {x} \\ = 0. (46) \\ \end{array}
$$

# B.2.3 NELBO

Thus, we have the following NELBO:
因此，我们得到以下 NELBO：

$$
\begin{array}{l} \mathbb {E} _ {q} \left[ - \underbrace {\log p _ {\theta} (\mathbf {x} | \mathbf {z} _ {t (0)})} _ {= 0 \text { from (46) }} + \underbrace {\mathcal {L} _ {\text { diffusion }} ^ {\infty}} _ {\text { Compute using (42) }} \right] + \underbrace {D _ {\mathrm{KL}} [ q (\mathbf {z} _ {t (T)} | \mathbf {x}) \| p _ {\theta} (\mathbf {z} _ {t (T)} , t) ]} _ {= 0 \text { using (37) and (38) }} \\ = \boxed {\mathbb {E} _ {q, t} \left[ \frac {\alpha_ {t} ^ {\prime}}{1 - \alpha_ {t}} \log \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t}, t), \mathbf {x} \rangle \right]} \tag {47} \\ \end{array}
$$

# B.3 Final Algorithm
B.3 最终算法

In Algorithm 1, we present the training algorithm for MDLM.
在算法 1 中，我们提出了 MDLM 的训练算法。

# Algorithm 1 Training MDLM
算法 1 训练 MDLM

1: repeat
1: 重复

2: $\bar { \mathbf { x } } ^ { 1 : L } \sim q ( \mathbf { x } )$ ▷ Sample a sentence.
2: $\bar { \mathbf { x } } ^ { 1 : L } \sim q ( \mathbf { x } )$ ▷ 采样一个句子。

3: $t \sim \mathcal { U } [ 0 , \bar { 1 } ]$ ▷ Sample a time step.
3: $t \sim \mathcal { U } [ 0 , \bar { 1 } ]$ ▷ 采样一个时间步。

4: $\mathbf { z } _ { t } ^ { \ell } \sim \dot { \mathrm { C a t } } ( \mathbf { \bar { z } } _ { t } ^ { \ell } ; \alpha _ { t } \mathbf { x } ^ { \ell } + ( 1 - \alpha _ { t } ) \mathbf { m } ) \forall 1 \le \ell \le L$ ▷ Mask Each token $\mathbf { x } ^ { \ell }$ independently to obtain the latent $\mathbf { z } _ { t } ^ { 1 : L } .$
4: $\mathbf { z } _ { t } ^ { \ell } \sim \dot { \mathrm { C a t } } ( \mathbf { \bar { z } } _ { t } ^ { \ell } ; \alpha _ { t } \mathbf { x } ^ { \ell } + ( 1 - \alpha _ { t } ) \mathbf { m } ) \forall 1 \le \ell \le L$ ▷ 对每个 token $\mathbf { x } ^ { \ell }$ 独立进行掩码，以获得潜在 $\mathbf { z } _ { t } ^ { 1 : L } .$

5: Take gradient descent step on
5: 对 @

$$
\nabla_ {\theta} \frac {\alpha_ {t} ^ {\prime}}{1 - \alpha_ {t}} \sum_ {\ell} \log \langle \mathbf {x} _ {\theta} ^ {\ell} (\mathbf {z} _ {t} ^ {1: L}, t), \mathbf {x} ^ {\ell} \rangle
$$

6: until converged
6: 直到收敛

# Appendix C Concrete Score Matching
附录 C 具体得分匹配

In the previous section, we defined the discrete diffusion process as a Discrete-Time Markov Chain (DTMC) with a finite set of T states, ${ \mathbf z } _ { \{ 0 , \frac { 1 } { T } , \ldots , 1 \} }$ , and a state transition matrix $Q _ { t }$ . To derive the continuous-time ELBO, we simply take the limit as $T \to \infty$ .
在上一节中，我们将离散扩散过程定义为一个具有有限状态集 T 的离散时间马尔可夫链（DTMC），以及状态转移矩阵 $Q _ { t }$ 。为了推导连续时间 ELBO，我们只需取 $T \to \infty$ 的极限。

In contrast, Campbell et al. \[5\] and Lou et al. \[33\] defined the discrete diffusion process as a Continuous-Time Markov Chain (CTMC), where the forward corruption process is specified by the rate change matrix $R _ { t } \in \mathbb { R } ^ { | \mathcal { V } | \times | \mathcal { V } | }$ |, which can be thought of as the instantaneous rate at which one state transitions to another. With this formulation, the forward posterior $q ( \mathbf { z } _ { t } | \mathbf { z } _ { s } )$ and the true reverse posterior $q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } )$ can be expressed in terms of the rate change matrix as follows:
相比之下，Campbell 等人 \[5\] 和 Lou 等人 \[33\] 将离散扩散过程定义为一个连续时间马尔可夫链（CTMC），其中正向污染过程由速率变化矩阵 $R _ { t } \in \mathbb { R } ^ { | \mathcal { V } | \times | \mathcal { V } | }$ | 指定，可以将其视为一个状态向另一个状态转换的瞬时速率。在这种公式下，正向后验 $q ( \mathbf { z } _ { t } | \mathbf { z } _ { s } )$ 和真实反向后验 $q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } )$ 可以用速率变化矩阵表示如下：

$$
q (\mathbf {z} _ {t} = \mathbf {y} ^ {\prime} | \mathbf {z} _ {s} = \mathbf {y}) = \delta_ {\mathbf {y} ^ {\prime}, \mathbf {y}} + R _ {t} (\mathbf {y} ^ {\prime}, \mathbf {y}) \frac {1}{T} + \mathcal {O} \left(\frac {1}{T ^ {2}}\right) \tag {48}
$$

 

$$
q \left(\mathbf {z} _ {s} = \mathbf {y} ^ {\prime} \mid \mathbf {z} _ {t} = \mathbf {y}, \mathbf {x}\right) = \delta_ {\mathbf {y} ^ {\prime}, \mathbf {y}} + \tilde {R} _ {t} \left(\mathbf {y} ^ {\prime}, \mathbf {y}\right) \frac {1}{T} + \mathcal {O} \left(\frac {1}{T ^ {2}}\right) \tag {49}
$$

where δ is the Kroenecker delta and $\scriptstyle { \mathcal { O } } ( { \frac { 1 } { T ^ { 2 } } } )$ represents higher order terms of $\scriptstyle { \frac { 1 } { T ^ { 2 } } }$ . In Sec. C.1, we show how to express the rate matrix $R _ { t }$ in terms of the state transition matrix $Q _ { t }$ . Lou et al. \[33\] propose a continuous time ELBO for this process. They mention that this expression can be derived from Benton et al. \[4\], though they do not provide an explicit derivation. For this reason, we present a rigorous derivation in Sec. C.2 and further demonstrate that, under the SUBS parameterization in Sec. 3.2.3, this formula reduces to our proposed continuous-time ELBO, given by (10).
其中 δ 是克罗内克δ函数， $\scriptstyle { \mathcal { O } } ( { \frac { 1 } { T ^ { 2 } } } )$ 表示 $\scriptstyle { \frac { 1 } { T ^ { 2 } } }$ 的高阶项。在 C.1 节中，我们展示了如何用状态转移矩阵 $Q _ { t }$ 表示速率矩阵 $R _ { t }$ 。Lou 等人 \[33\] 提出了该过程的连续时间 ELBO。他们提到这个表达式可以追溯到 Benton 等人 \[4\]，尽管他们没有提供明确的推导过程。因此，我们在 C.2 节中给出了严格的推导，并进一步证明，在 3.2.3 节的 SUBS 参数化下，该公式简化为我们提出的连续时间 ELBO，如公式 (10) 所示。

For the remainder of this section, we switch to the notation $q _ { t | s } ( \mathbf { y } ^ { \prime } | \mathbf { y } )$ to denote $q ( \mathbf { z } _ { t } = \mathbf { y } ^ { \prime } | \mathbf { z } _ { s } = \mathbf { y } )$ , $q _ { s | t } ( \mathbf { y } ^ { \prime } | \mathbf { y } )$ for ${ q } ( \mathbf { z } _ { s } = \mathbf { y } ^ { \prime } | \mathbf { z } _ { t } = \mathbf { y } )$ , and $q ( \mathbf { z } _ { t } = \mathbf { y } | \mathbf { x } )$ for $q _ { t } ( \mathbf { y } \vert \mathbf { x } )$ , aligning with the notation typically used in the CTMC literature.
在本节的其余部分，我们采用 $q _ { t | s } ( \mathbf { y } ^ { \prime } | \mathbf { y } )$ 表示 $q ( \mathbf { z } _ { t } = \mathbf { y } ^ { \prime } | \mathbf { z } _ { s } = \mathbf { y } )$ ， $q _ { s | t } ( \mathbf { y } ^ { \prime } | \mathbf { y } )$ 表示 ${ q } ( \mathbf { z } _ { s } = \mathbf { y } ^ { \prime } | \mathbf { z } _ { t } = \mathbf { y } )$ ， $q ( \mathbf { z } _ { t } = \mathbf { y } | \mathbf { x } )$ 表示 $q _ { t } ( \mathbf { y } \vert \mathbf { x } )$ ，以符合 CTMC 文献中常用的符号表示。

# C.1 Extracting the Rate Matrix
C.1 提取速率矩阵

Here, we aim to express the rate change matrix $R _ { t }$ in terms of the state transition matrix $Q _ { t }$ . To do this, we first represent the forward transition $q _ { t \mid s }$ in terms of $Q _ { t }$ and $R _ { t }$ separately, allowing us to illustrate their relationship.
在这里，我们的目标是用状态转移矩阵 $Q _ { t }$ 表示速率变化矩阵 $R _ { t }$ 。为此，我们首先分别用 $Q _ { t }$ 和 $R _ { t }$ 表示前向转移 $q _ { t \mid s }$ ，从而阐明它们之间的关系。

Using (13), we can write $q _ { t | s }$ as follows:
利用(13)式，我们可以将 $q _ { t | s }$ 表示如下：

$$
\begin{array}{l} q _ {t | s} \left(\mathbf {y} ^ {\prime} \mid \mathbf {y}\right) = \left[ \mathbf {y} ^ {\prime} \right] ^ {\top} \left[ \alpha_ {t | s} \mathbf {I} + \left(1 - \alpha_ {t | s}\right) \mathbf {1 m} ^ {\top} \right] ^ {\top} \mathbf {y} \\ = [ \mathbf {y} ^ {\prime} ] ^ {\top} [ \alpha_ {t | s} \mathbf {y} + (1 - \alpha_ {t | s}) \mathbf {m 1} ^ {\top} \mathbf {y} ] \\ = [ \mathbf {y} ^ {\prime} ] ^ {\top} [ \alpha_ {t | s} \mathbf {y} + (1 - \alpha_ {t | s}) \mathbf {m} ] \\ = \alpha_ {t | s} [ \mathbf {y} ^ {\prime} ] ^ {\top} \mathbf {y} + (1 - \alpha_ {t | s}) [ \mathbf {y} ^ {\prime} ] ^ {\top} \mathbf {m} \tag {50} \\ \end{array}
$$

Now let’s analyze all possible combinations for the tuple $\mathbf { \Sigma } ( \mathbf { y } ^ { \prime } , \mathbf { y } )$ :
现在让我们分析元组 $\mathbf { \Sigma } ( \mathbf { y } ^ { \prime } , \mathbf { y } )$ 的所有可能组合：

1.  Case $( \mathbf { y } ^ { \prime } = \mathbf { x } , \mathbf { y } = \mathbf { x } ) { \mathrm { : } }$ Using (50), we find that $q _ { t | s } ( \mathbf { x } | \mathbf { x } ) = \alpha _ { t | s }$ s for the DTMC. By (48), we have $q _ { t | s } ( \mathbf { x } | \mathbf { x } ) = 1 + R _ { t } ( \mathbf { x } , \mathbf { x } ) \frac { 1 } { T }$ as $T \to \infty$ , since the higher-order terms $\scriptstyle { \mathcal { O } } ( { \frac { 1 } { T ^ { 2 } } } )$ vanish in the limit. Thus, we get:
    情况 $( \mathbf { y } ^ { \prime } = \mathbf { x } , \mathbf { y } = \mathbf { x } ) { \mathrm { : } }$ 使用(50)式，我们找到 DTMC 的 $q _ { t | s } ( \mathbf { x } | \mathbf { x } ) = \alpha _ { t | s }$ 。根据(48)式，我们有 $q _ { t | s } ( \mathbf { x } | \mathbf { x } ) = 1 + R _ { t } ( \mathbf { x } , \mathbf { x } ) \frac { 1 } { T }$ 等于 $T \to \infty$ ，因为在极限中高阶项 $\scriptstyle { \mathcal { O } } ( { \frac { 1 } { T ^ { 2 } } } )$ 消失。因此，我们得到：

$$
\begin{array}{l} \lim _ {T \to \infty} \left[ 1 + R _ {t} (\mathbf {x}, \mathbf {x}) \frac {1}{T} \right] = \lim _ {T \to \infty} \alpha_ {t | s} \\ \Longrightarrow R _ {t} (\mathbf {x}, \mathbf {x}) = \lim _ {T \rightarrow \infty} T \left(\alpha_ {t | s} - 1\right) = \frac {\alpha_ {t} ^ {\prime}}{\alpha_ {t}} \tag {51} \\ \end{array}
$$

2.  Case $( \mathbf { y } ^ { \prime } = \mathbf { m } , \mathbf { y } \in \mathcal { V } - \{ \mathbf { m } \} ) \colon$ Similarly, using (50) and (48), we have $q _ { t | s } ( \mathbf { m } | \mathbf { y } \neq \mathbf { m } ) = 1 - \alpha _ { t | s }$ s and $q _ { t | s } ( \mathbf x | \mathbf y \neq \mathbf m ) = R _ { t } ( \mathbf m , \mathbf y \neq \mathbf m ) \frac { 1 } { T }$ . Thus,
    情况 $( \mathbf { y } ^ { \prime } = \mathbf { m } , \mathbf { y } \in \mathcal { V } - \{ \mathbf { m } \} ) \colon$ 类似地，使用(50)式和(48)式，我们有 $q _ { t | s } ( \mathbf { m } | \mathbf { y } \neq \mathbf { m } ) = 1 - \alpha _ { t | s }$ 和 $q _ { t | s } ( \mathbf x | \mathbf y \neq \mathbf m ) = R _ { t } ( \mathbf m , \mathbf y \neq \mathbf m ) \frac { 1 } { T }$ 。

$$
\begin{array}{l} \lim _ {T \to \infty} \left[ R _ {t} (\mathbf {m}, \mathbf {y} \neq \mathbf {m}) \frac {1}{T} \right] = \lim _ {T \to \infty} [ 1 - \alpha_ {t | s} ] \\ \Longrightarrow R _ {t} (\mathbf {m}, \mathbf {y} \neq \mathbf {m}) = \lim _ {T \rightarrow \infty} T (1 - \alpha_ {t | s}) = - \frac {\alpha_ {t} ^ {\prime}}{\alpha_ {t}} \tag {52} \\ \end{array}
$$

3.  Case $( \mathbf { y } ^ { \prime } = \mathbf { m } , \mathbf { y } = \mathbf { m } )$ : Using (50) and(48), we find $q _ { t | s } ( \mathbf { m } | \mathbf { m } ) = 1$ and $q _ { t | s } ( \mathbf { m } | \mathbf { m } ) = 1 +$ $\begin{array} { r l } { R _ { t } ( \mathbf { m } , \mathbf { m } ) { \frac { 1 } { T } } + { \mathcal { O } } \left( { \frac { 1 } { T ^ { 2 } } } \right) } & { { } } \end{array}$ . Since these two expressions must be equal for any T , it follows that
    情况 $( \mathbf { y } ^ { \prime } = \mathbf { m } , \mathbf { y } = \mathbf { m } )$ ：使用 (50) 和 (48)，我们得到 $q _ { t | s } ( \mathbf { m } | \mathbf { m } ) = 1$ 和 $q _ { t | s } ( \mathbf { m } | \mathbf { m } ) = 1 +$ $\begin{array} { r l } { R _ { t } ( \mathbf { m } , \mathbf { m } ) { \frac { 1 } { T } } + { \mathcal { O } } \left( { \frac { 1 } { T ^ { 2 } } } \right) } & { { } } \end{array}$ 。由于这两个表达式对于任何 T 都必须相等，因此

$$
R _ {t} (\mathbf {m}, \mathbf {m}) = 0. \tag {53}
$$

Note that when $R _ { t } ( \mathbf { m } , \mathbf { m } )$ is constant, the term $\scriptstyle { \mathcal { O } } \left( { \frac { 1 } { T ^ { 2 } } } \right)$ reduces to zero, as it includes higher-order time derivatives of $R _ { t }$ .
注意，当 $R _ { t } ( \mathbf { m } , \mathbf { m } )$ 为常数时，项 $\scriptstyle { \mathcal { O } } \left( { \frac { 1 } { T ^ { 2 } } } \right)$ 简化为零，因为它包含 $R _ { t }$ 的高阶时间导数。

4.  Case $( \mathbf { y } ^ { \prime } = \mathbf { x } , \mathbf { y } \in \mathcal { V } - \{ \mathbf { x } \} ) \colon$ In the context of absorbing state diffusion, these states are never observed. Thus,
    情况 $( \mathbf { y } ^ { \prime } = \mathbf { x } , \mathbf { y } \in \mathcal { V } - \{ \mathbf { x } \} ) \colon$ ：在吸收状态扩散的背景下，这些状态从未被观察到。因此，

$$
R _ {t} \left(\mathbf {y} ^ {\prime} = \mathbf {x}, \mathbf {y} \in \mathcal {V} - \{\mathbf {m}, \mathbf {x} \}\right) = 0 \tag {54}
$$

5.  Case $( \mathbf { y } ^ { \prime } \in \mathcal { V } - \{ \mathbf { m } , \mathbf { x } \} , \mathbf { y } \in \{ \mathbf { m } , \mathbf { x } \} )$ : In the context of absorbing state diffusion, these states are never observed. Thus,
    情况 $( \mathbf { y } ^ { \prime } \in \mathcal { V } - \{ \mathbf { m } , \mathbf { x } \} , \mathbf { y } \in \{ \mathbf { m } , \mathbf { x } \} )$ ：在吸收状态扩散的背景下，这些状态从未被观察到。因此，

$$
R _ {t} (\mathbf {y} ^ {\prime} \in \mathcal {V} - \{\mathbf {m}, \mathbf {x} \}, \mathbf {y} \in \{\mathbf {m}, \mathbf {x} \}) = 0 \tag {55}
$$

Finally, we can express the forward rate matrix as:
最后，我们可以将前向率矩阵表示为：

$$
\boxed {R _ {t} = \frac {\alpha_ {t} ^ {\prime}}{\alpha_ {t}} \left(\mathbf {I} - \mathbf {m 1} ^ {\top}\right)} \tag {56}
$$

It can be seen that the columns of this matrix sum to zero, i.e.,
可以看出，该矩阵的每一列之和为零，即：

$$
\sum_ {\mathbf {y} ^ {\prime} \in \mathcal {V}} R _ {t} (\mathbf {y} ^ {\prime}, \mathbf {y}) = 0 \Longrightarrow R _ {t} (\mathbf {y}, \mathbf {y}) = \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {y}} R _ {t} (\mathbf {y} ^ {\prime}, \mathbf {y}), \tag {57}
$$

which ensures that the probability mass is preserved in the forward diffusion process. Similarly, the reverse rate matrix $\tilde { R } _ { t }$ can be written in terms of the forward rate matrix $R _ { t }$ as follows \[33\]:
这确保了前向扩散过程中概率质量的守恒。类似地，反向率矩阵 $\tilde { R } _ { t }$ 可以用前向率矩阵 $R _ { t }$ 表示如下 \[33\]：

$$
\tilde {R} _ {t} \left(\mathbf {y} ^ {\prime}, \mathbf {y}\right) = \left\{ \begin{array}{l l} \frac {q _ {t} \left(\mathbf {y} ^ {\prime} \mid \mathbf {x}\right)}{q _ {t} (\mathbf {y} \mid \mathbf {x})} R _ {t} \left(\mathbf {y}, \mathbf {y} ^ {\prime}\right) & \mathbf {y} ^ {\prime} \neq \mathbf {y} \\ - \sum_ {\tilde {\mathbf {y}} \neq \mathbf {y}} \frac {q _ {t} (\tilde {\mathbf {y}} \mid \mathbf {x})}{q _ {t} (\mathbf {y} \mid \mathbf {x})} R _ {t} (\mathbf {y}, \tilde {\mathbf {y}}) & \mathbf {y} ^ {\prime} = \mathbf {y}. \end{array} \right. \tag {58}
$$

# C.2 NELBO

Meng et al. \[37\] introduced the term “concrete score” for the term $q _ { t } ( \mathbf { y } ^ { \prime } | \mathbf { x } ) / q _ { t } ( \mathbf { y } | \mathbf { x } )$ that appears in $\tilde { R } _ { t }$ . Since this quantity is not directly accessible in the reverse diffusion process, we approximate it using a neural network, $\mathbf { s } _ { \theta } : \mathcal { V } \xrightarrow { } \mathcal { V } .$ , with parameters θ. The approximate reverse posterior $p _ { s \mid t }$ can then be expressed in terms of the approximate reverse rate matrix $\tilde { R } _ { t }$ in the following manner:
孟等人\[37\]为 $\tilde { R } _ { t }$ 中出现的 $q _ { t } ( \mathbf { y } ^ { \prime } | \mathbf { x } ) / q _ { t } ( \mathbf { y } | \mathbf { x } )$ 术语引入了“具体分数”这一概念。由于该量在反向扩散过程中无法直接获取，我们使用参数为θ的神经网络 $\mathbf { s } _ { \theta } : \mathcal { V } \xrightarrow { } \mathcal { V } .$ 对其进行近似，然后可以用以下方式表示近似反向后验 $p _ { s \mid t }$ ，其中包含近似反向率矩阵 $\tilde { R } _ { t }$ ：

$$
p _ {s \mid t} (\mathbf {y} ^ {\prime} | \mathbf {y}) = \delta_ {y ^ {\prime}, y} + \tilde {R} _ {t} ^ {\theta} (\mathbf {y} ^ {\prime}, \mathbf {y}) \frac {1}{T} + \mathcal {O} \left(\frac {1}{T ^ {2}}\right) \tag {59}
$$

 

$$
\tilde {R} _ {t} ^ {\theta} \left(\mathbf {y} ^ {\prime}, \mathbf {y}\right) = \left\{ \begin{array}{l l} \mathbf {s} _ {\theta} (\mathbf {y}) _ {\mathbf {y} ^ {\prime}} R _ {t} \left(\mathbf {y}, \mathbf {y} ^ {\prime}\right) & \mathbf {y} ^ {\prime} \neq \mathbf {y} \\ - \sum_ {\tilde {\mathbf {y}} \neq \mathbf {y}} \mathbf {s} _ {\theta} (\mathbf {y}) _ {\tilde {\mathbf {y}}} R _ {t} (\mathbf {y}, \tilde {\mathbf {y}}) & \mathbf {y} ^ {\prime} = \mathbf {y} \end{array} \right. \tag {60}
$$

where $\mathbf { s } _ { \theta } ( \mathbf { y } ) _ { \mathbf { y } ^ { \prime } }$ denotes the approximate concrete score $q _ { t } ( \mathbf { y } ^ { \prime } | \mathbf { x } ) / q _ { t } ( \mathbf { y } | \mathbf { x } )$ . Lou et al. \[33\] propose the following NELBO to train such a model:
其中 $\mathbf { s } _ { \theta } ( \mathbf { y } ) _ { \mathbf { y } ^ { \prime } }$ 表示 $q _ { t } ( \mathbf { y } ^ { \prime } | \mathbf { x } ) / q _ { t } ( \mathbf { y } | \mathbf { x } )$ 的近似具体分数。刘等人\[33\]提出了以下负对数似然目标函数（NELBO）来训练此类模型：

$$
\mathbb {E} _ {t \in [ 0, 1 ], \mathbf {y} \sim q _ {t} (\cdot | \mathbf {x})} \left[ \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {y}} R _ {t} (\mathbf {y}, \mathbf {y} ^ {\prime}) \left(\mathrm{s} _ {\theta} (\mathbf {y}) _ {\mathbf {y} ^ {\prime}} - \frac {q _ {t} \left(\mathbf {y} ^ {\prime} | \mathbf {x}\right)}{q _ {t} (\mathbf {y} | \mathbf {x})} \log \mathrm{s} _ {\theta} (\mathbf {y}) _ {\mathbf {y} ^ {\prime}} + K \left(\frac {q _ {t} \left(\mathbf {y} ^ {\prime} | \mathbf {x}\right)}{q _ {t} (\mathbf {y} | \mathbf {x})}\right)\right) \right] \tag {61}
$$

where $K ( a ) = a \log a - a$ . They mention that this expression can be derived from Benton et al. \[4\], though they do not provide an explicit derivation. For this reason, we present a rigorous derivation in the following section.
其中 $K ( a ) = a \log a - a$ 。他们提到该表达式可源自本顿等人\[4\]的研究，尽管他们未提供明确的推导过程。因此，我们在下一节将给出严谨的推导。

Proof. Let’s focus on the diffusion loss for this process. As mentioned in the previous section, the reconstruction and prior loss terms reduce to zero. The continuous-time diffusion loss is given by:
证明。让我们关注这个过程的扩散损失。如前一节所述，重建和先验损失项简化为零。连续时间扩散损失由下式给出：

$$
\lim _ {T \to \infty} T \mathbb {E} _ {t \in \{\frac {1}{T}, \frac {2}{T}, \dots , 1 \}, \mathbf {y} \sim q _ {t} (. | \mathbf {x})} [ \mathrm{D} _ {\mathrm{KL}} (q _ {s | t} (\mathbf {y} ^ {\prime} | \mathbf {y}, \mathbf {x}) \| p _ {s | t} (\mathbf {y} ^ {\prime} | \mathbf {y})) ]
$$

 

$$
= \lim _ {T \to \infty} T \mathbb {E} _ {t \in \{\frac {1}{T}, \frac {2}{T}, \dots , 1 \}, \mathbf {y} \sim q _ {t} (. | \mathbf {x})} \left[ \sum_ {\mathbf {y} ^ {\prime}} q _ {s | t} (\mathbf {y} ^ {\prime} | \mathbf {y}, \mathbf {x}) \log \frac {q _ {s | t} (\mathbf {y} ^ {\prime} | \mathbf {y} , \mathbf {x})}{p _ {s | t} (\mathbf {y} ^ {\prime} | \mathbf {y} , \mathbf {x})} \right]
$$

 

$$
= \mathbb {E} _ {t \in [ 0, 1 ], \mathbf {y} \sim q _ {t} (. | \mathbf {x})} \left[ \lim _ {T \to \infty} T \sum_ {\mathbf {y} ^ {\prime}} q _ {s | t} (\mathbf {y} ^ {\prime} | \mathbf {y}, \mathbf {x}) \mathrm{log} \frac {q _ {s | t} (\mathbf {y} ^ {\prime} | \mathbf {y} , \mathbf {x})}{p _ {s | t} (\mathbf {y} ^ {\prime} | \mathbf {y} , \mathbf {x})} \right]
$$

 

$$
= \mathbb {E} _ {t \in [ 0, 1 ], \mathbf {y} \sim q _ {t} (., | \mathbf {x})} \left[ \underbrace {\lim _ {T \rightarrow \infty} T q _ {s \mid t} (\mathbf {y} \mid \mathbf {y} , \mathbf {x}) \log \frac {q _ {s \mid t} (\mathbf {y} \mid \mathbf {y} , \mathbf {x})}{p _ {s \mid t} (\mathbf {y} \mid \mathbf {y} , \mathbf {x})}} _ {\text {Term 1}} + \underbrace {\lim _ {T \rightarrow \infty} T \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {y}} q _ {s \mid t} (\mathbf {y} ^ {\prime} \mid \mathbf {y} , \mathbf {x}) \log \frac {q _ {s \mid t} (\mathbf {y} ^ {\prime} \mid \mathbf {y} , \mathbf {x})}{p _ {s \mid t} (\mathbf {y} ^ {\prime} \mid \mathbf {y} , \mathbf {x})}} _ {\text {Term 2}} \right] \tag {62}
$$

Let’s simplify these two terms separately. For the derivation, we’ll rely on two key observations: In the limiting case as $T \to \infty$ , it follows from (49) that lim $T \to \infty q _ { s | t } ( \mathbf { y } | \mathbf { y } , \mathbf { x } ) = 1$ and from (59) that lim $_ { { \bf \Pi } ^ { \perp } \partial \mathrm { \bf { \vec { \phi } } } \partial p _ { s \mid t } } ( \bf { y } | \bf { y } , \bf { x } ) = 1$ .
让我们分别简化这两个项。在推导过程中，我们将依赖于两个关键观察：在 $T \to \infty$ 的极限情况下，根据(49)式可知 lim $T \to \infty q _ { s | t } ( \mathbf { y } | \mathbf { y } , \mathbf { x } ) = 1$ ，根据(59)式可知 lim $_ { { \bf \Pi } ^ { \perp } \partial \mathrm { \bf { \vec { \phi } } } \partial p _ { s \mid t } } ( \bf { y } | \bf { y } , \bf { x } ) = 1$ 。

# Term 1:
第一项：

$$
\lim _ {T \to \infty} T q _ {s | t} (\mathbf {y} | \mathbf {y}, \mathbf {x}) \mathrm{log} \frac {q _ {s | t} (\mathbf {y} | \mathbf {y} , \mathbf {x})}{p _ {s | t} (\mathbf {y} | \mathbf {y} , \mathbf {x})}
$$

 

$$
\because \lim _ {T \rightarrow \infty} q _ {s | t} (\mathbf {y} | \mathbf {y}, \mathbf {x}) = 1; \text { hence },
$$

 

$$
= \lim _ {T \rightarrow \infty} T \log \frac {q _ {s | t} (\mathbf {y} | \mathbf {y} , \mathbf {x})}{p _ {s | t} (\mathbf {y} | \mathbf {y} , \mathbf {x})}
$$

The above term is in $\infty \times 0$ indeterminate form; therefore,
上述项是 $\infty \times 0$ 不定式；因此，

$$
= \lim _ {T \rightarrow \infty} T \left(\log q _ {s | t} (\mathbf {y} | \mathbf {y}, \mathbf {x}) - \log p _ {s | t} (\mathbf {y} | \mathbf {y}, \mathbf {x})\right)
$$

Substituting $q _ { s \mid t }$ and $p _ { s \mid t }$ from (49) and (59), we get:
将(49)和(59)中的 $q _ { s \mid t }$ 和 $p _ { s \mid t }$ 代入，我们得到：

$$
= \lim _ {T \rightarrow \infty} T \left[ \log \left(1 + \tilde {R} _ {t} (\mathbf {y}, \mathbf {y}) \frac {1}{T} + \mathcal {O} \left(\frac {1}{T ^ {2}}\right)\right) - \log \left(1 + \tilde {R} _ {t} ^ {\theta} (\mathbf {y}, \mathbf {y}) \frac {1}{T} + \mathcal {O} \left(\frac {1}{T ^ {2}}\right)\right)\right]
$$

Applying the Taylor series expansion for log(1+x), we get:
对 log(1+x)应用泰勒级数展开，我们得到：

$$
= \lim _ {T \rightarrow \infty} T \left[ \tilde {R} _ {t} (\mathbf {y}, \mathbf {y}) \frac {1}{T} + \mathcal {O} \left(\frac {1}{T ^ {2}}\right) - \tilde {R} _ {t} ^ {\theta} (\mathbf {y}, \mathbf {y}) \frac {1}{T} - \mathcal {O} \left(\frac {1}{T ^ {2}}\right)\right]
$$

 

$$
= \tilde {R} _ {t} (\mathbf {y}, \mathbf {y}) + \lim _ {T \rightarrow \infty} T \mathcal {O} \left(\frac {1}{T ^ {2}}\right) - \tilde {R} _ {t} ^ {\theta} (\mathbf {y}, \mathbf {y}) - \lim _ {T \rightarrow \infty} T \mathcal {O} \left(\frac {1}{T ^ {2}}\right)
$$

 

$$
\because \lim _ {T \rightarrow \infty} T \mathcal {O} \left(\frac {1}{T ^ {2}}\right) = 0, \text { we get: }
$$

  

$$
= \tilde {R} _ {t} (\mathbf {y}, \mathbf {y}) - \tilde {R} _ {t} ^ {\theta} (\mathbf {y}, \mathbf {y})
$$

Using (58) and (60), we get:
使用(58)和(60)，我们得到：

$$
= - \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {y}} R _ {t} (\mathbf {y}, \mathbf {y} ^ {\prime}) \left(\frac {q _ {t} \left(\mathbf {y} ^ {\prime} \mid \mathbf {x}\right)}{q _ {t} (\mathbf {y} \mid \mathbf {x})} - \mathrm{s} _ {\theta} (\mathbf {y}) _ {\mathbf {y} ^ {\prime}}\right) \tag {63}
$$

# Term 2:
项 2：

$$
\lim _ {T \to \infty} T \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {y}} q _ {s | t} (\mathbf {y} ^ {\prime} | \mathbf {y}, \mathbf {x}) \log \frac {q _ {s | t} (\mathbf {y} ^ {\prime} | \mathbf {y} , \mathbf {x})}{p _ {s | t} (\mathbf {y} ^ {\prime} | \mathbf {y} , \mathbf {x})}
$$

Substituting $q _ { s \mid t }$ and $p _ { s \mid t }$ from (49) and (59), we get:
将(49)和(59)中的 $q _ { s \mid t }$ 和 $p _ { s \mid t }$ 代入，我们得到：

$$
= \lim _ {T \to \infty} T \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {y}} \left[ \left[ \frac {q _ {s} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {y} | \mathbf {x})} R _ {t} (\mathbf {y}, \mathbf {y} ^ {\prime}) \frac {1}{T} + \mathcal {O} \left(\frac {1}{T ^ {2}}\right) \right] \log \frac {\frac {q _ {s} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {y} | \mathbf {x})} R _ {t} (\mathbf {y} , \mathbf {y} ^ {\prime}) \frac {1}{T} + \mathcal {O} \left(\frac {1}{T ^ {2}}\right)}{\mathbf {s} _ {\theta} (\mathbf {y}) _ {\mathbf {y} ^ {\prime}} R _ {t} (\mathbf {y} , \mathbf {y} ^ {\prime}) \frac {1}{T} + \mathcal {O} \left(\frac {1}{T ^ {2}}\right)} \right]
$$

 

$$
= \lim _ {T \to \infty} \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {y}} \left[ \left[ \frac {q _ {s} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {y} | \mathbf {x})} R _ {t} (\mathbf {y}, \mathbf {y} ^ {\prime}) + T \mathcal {O} \left(\frac {1}{T ^ {2}}\right) \right] \log \frac {\frac {q _ {s} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {y} | \mathbf {x})} R _ {t} (\mathbf {y} , \mathbf {y} ^ {\prime}) + T \mathcal {O} \left(\frac {1}{T ^ {2}}\right)}{\mathbf {s} _ {\theta} (\mathbf {y}) _ {\mathbf {y} ^ {\prime}} R _ {t} (\mathbf {y} , \mathbf {y} ^ {\prime}) + T \mathcal {O} \left(\frac {1}{T ^ {2}}\right)} \right]
$$

 

$$
\because \lim _ {T \rightarrow \infty} T \mathcal {O} \left(\frac {1}{T ^ {2}}\right) = 0, \text { we get: }
$$

  

$$
= \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {y}} \frac {q _ {s} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {y} | \mathbf {x})} R _ {t} (\mathbf {y}, \mathbf {y} ^ {\prime}) \log \frac {\frac {q _ {s} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {y} | \mathbf {x})} R _ {t} (\mathbf {y} , \mathbf {y} ^ {\prime})}{\mathbf {s} _ {\theta} (\mathbf {y}) _ {\mathbf {y} ^ {\prime}} R _ {t} (\mathbf {y} , \mathbf {y} ^ {\prime})}
$$

 

$$
= \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {y}} \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {y} | \mathbf {x})} R _ {t} (\mathbf {y}, \mathbf {y} ^ {\prime}) \left(\log \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {y} | \mathbf {x})} - \log \mathrm{s} _ {\theta} (\mathbf {y}) _ {\mathbf {y} ^ {\prime}}\right) \tag {64}
$$

Finally, plugging (63) and (64) into (62) yields us the NELBO as proposed in Lou et al. \[33\]:
最后，将(63)和(64)代入(62)得到所提出的 NELBO，如 Lou 等人\[33\]所述：

$$
\begin{array}{l} \mathbb {E} _ {t \in [ 0, 1 ], \mathbf {y} \sim q _ {t} (. | \mathbf {x})} \left[ - \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {y}} R _ {t} (\mathbf {y}, \mathbf {y} ^ {\prime}) \left(\frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {y} | \mathbf {x})} - \mathbf {s} _ {\theta} (\mathbf {y}) _ {\mathbf {y} ^ {\prime}}\right) \right. \\ \left. + \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {y}} \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {y} | \mathbf {x})} R _ {t} (\mathbf {y}, \mathbf {y} ^ {\prime}) \left(\log \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {y} | \mathbf {x})} - \log \mathrm{s} _ {\theta} (\mathbf {y}) _ {\mathbf {y} ^ {\prime}}\right) \right] \\ = \mathbb {E} _ {t \in [ 0, 1 ], \mathbf {y} \sim q _ {t} (. | \mathbf {x})} \left[ \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {y}} R _ {t} (\mathbf {y}, \mathbf {y} ^ {\prime}) \left(- \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {y} | \mathbf {x})} + \mathbf {s} _ {\theta} (\mathbf {y}) _ {\mathbf {y} ^ {\prime}} \right. \right. \\ \left. \left. + \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {y} | \mathbf {x})} \log \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {y} | \mathbf {x})} - \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {y} | \mathbf {x})} \log \mathrm{s} _ {\theta} (\mathbf {y}) _ {\mathbf {y} ^ {\prime}}\right) \right] \\ = \boxed {\mathbb {E} _ {t \in [ 0, 1 ], \mathbf {y} \sim q _ {t} (. | \mathbf {x})} \left[ \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {y}} R _ {t} (\mathbf {y}, \mathbf {y} ^ {\prime}) \left(\mathbf {s} _ {\theta} (\mathbf {y}) _ {\mathbf {y} ^ {\prime}} - \frac {q _ {t} \left(\mathbf {y} ^ {\prime} | \mathbf {x}\right)}{q _ {t} (\mathbf {y} | \mathbf {x})} \log \mathbf {s} _ {\theta} (\mathbf {y}) _ {\mathbf {y} ^ {\prime}} + K \left(\frac {q _ {t} \left(\mathbf {y} ^ {\prime} | \mathbf {x}\right)}{q _ {t} (\mathbf {y} | \mathbf {x})}\right)\right) \right]} \tag {65} \\ \end{array}
$$

where $K ( a ) = a \log a - a .$ This concludes the proof.
其中 $K ( a ) = a \log a - a .$ 证明完毕。

# C.3 Concrete Score for MDLM
C.3 MDLM 的具体得分

Given a latent variable $\mathbf { z } _ { t }$ and the output of the denoising model, ${ \bf x } _ { \theta } ( { \bf z } _ { t } , t )$ parameterized using SUBS, we aim to recover the concrete score $\mathbf { s } _ { \theta } ( \mathbf { z } _ { t } ) \in ( \mathbb { R } ^ { + } + \{ 0 \} ) ^ { | \nu | }$ . Note that ${ \bf s } _ { \theta } ( { \bf z } _ { t } ) _ { \bf y }$ is the ratio $\frac { p _ { t } ( \mathbf { y } ) } { p _ { t } ( \mathbf { z } _ { t } ) }$ in the reverse process. Since xθ approximates x, we use $p _ { t } ( \mathbf { y } ) { = } q _ { t } ( \mathbf { y } | \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) )$ ; therefore,
给定潜在变量 $\mathbf { z } _ { t }$ 和去噪模型的输出 ${ \bf x } _ { \theta } ( { \bf z } _ { t } , t )$ ，其中 ${ \bf x } _ { \theta } ( { \bf z } _ { t } , t )$ 使用 SUBS 参数化，我们旨在恢复具体分数 $\mathbf { s } _ { \theta } ( \mathbf { z } _ { t } ) \in ( \mathbb { R } ^ { + } + \{ 0 \} ) ^ { | \nu | }$ 。请注意 ${ \bf s } _ { \theta } ( { \bf z } _ { t } ) _ { \bf y }$ 是反向过程中的比率 $\frac { p _ { t } ( \mathbf { y } ) } { p _ { t } ( \mathbf { z } _ { t } ) }$ 。由于 xθ 近似于 x，我们使用 $p _ { t } ( \mathbf { y } ) { = } q _ { t } ( \mathbf { y } | \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) )$ ；因此，

$$
\mathbf {s} _ {\theta} (\mathbf {z} _ {t}) _ {\mathbf {y}} = \frac {p _ {t} (\mathbf {y})}{p _ {t} (\mathbf {z} _ {t})} = \frac {q _ {t} (\mathbf {y} | \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t))}{q _ {t} (\mathbf {z} _ {t} | \mathbf {x} _ {\theta} (\mathbf {z} _ {t} , t))}. \tag {66}
$$

To obtain the score, we first compute $q _ { t } ( \mathbf { y } | \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) )$ for all possible y and $\mathbf { z } _ { t }$ . Using (4), we derive the following expressions under the SUBS parameterization:
为了获得分数，我们首先对所有可能的 y 和 $\mathbf { z } _ { t }$ 计算 $q _ { t } ( \mathbf { y } | \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) )$ 。使用公式 (4)，在 SUBS 参数化下，我们推导出以下表达式：

$$
q _ {t} (\mathbf {y} \neq \mathbf {m} | \mathbf {x} _ {\theta} (\mathbf {z} _ {t} = \mathbf {m}, t)) = \alpha_ {t} \left\langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t}, t), \mathbf {y} \right\rangle \tag {67}
$$

 

$$
q _ {t} \left(\mathbf {y} \notin \{\mathbf {m}, \mathbf {z} _ {t} \} \mid \mathbf {x} _ {\theta} \left(\mathbf {z} _ {t} \neq \mathbf {m}, t\right)\right) = 0 \tag {68}
$$

 

$$
q _ {t} \left(\mathbf {y} = \mathbf {z} _ {t} \mid \mathbf {x} _ {\theta} \left(\mathbf {z} _ {t} \neq \mathbf {m}, t\right)\right) = \alpha_ {t} \tag {69}
$$

 

$$
q _ {t} (\mathbf {y} = \mathbf {m} | \mathbf {x} _ {\theta} (\mathbf {z} _ {t} \in \mathcal {V}, t)) = 1 - \alpha_ {t} \tag {70}
$$

Plugging these into (66), we get:
将这些代入公式 (66)，我们得到：

$$
\mathbf {s} _ {\theta} \left(\mathbf {z} _ {t} = \mathbf {m}\right) _ {\mathbf {y} \neq \mathbf {m}} = \frac {\alpha_ {t}}{1 - \alpha_ {t}} \left\langle \mathbf {x} _ {\theta} \left(\mathbf {z} _ {t}, t\right), \mathbf {y} \right\rangle \quad \text { Using (70) and (67) } \tag {71}
$$

 

$$
\mathbf {s} _ {\theta} \left(\mathbf {z} _ {t} = \mathbf {m}\right) _ {\mathbf {y} = \mathbf {m}} = 1 \quad \text { Using (70) } \tag {72}
$$

 

$$
\mathbf {s} _ {\theta} \left(\mathbf {z} _ {t} \neq \mathbf {m}\right) _ {\mathbf {y} = \mathbf {m}} = \frac {1 - \alpha_ {t}}{\alpha_ {t}} \quad \text { Using (69) and (70) } \tag {73}
$$

 

$$
\mathbf {s} _ {\theta} \left(\mathbf {z} _ {t} \neq \mathbf {m}\right) _ {\mathbf {y} = \mathbf {z} _ {t}} = 1 \quad \text {Using(69)} \tag {74}
$$

 

$$
\mathbf {s} _ {\theta} \left(\mathbf {z} _ {t} \neq \mathbf {m}\right) _ {\mathbf {y} \notin \{\mathbf {m}, \mathbf {z} _ {t} \}} = 0 \quad \text { Using (68) and (69) } \tag {75}
$$

These can be consolidated into the following expression:
这些可以整合为以下表达式：

$$
\boxed {\mathbf {s} _ {\theta} \left(\mathbf {z} _ {t}\right) _ {\mathbf {y}} = \mathbf {y} ^ {\top} \left[ \delta_ {\mathbf {z} _ {t}, \mathbf {m}} \frac {\alpha_ {t}}{1 - \alpha_ {t}} \mathbf {x} _ {\theta} \left(\mathbf {z} _ {t}, t\right) + \left(1 - \delta_ {\mathbf {z} _ {t}, \mathbf {m}}\right) \frac {1 - \alpha_ {t}}{\alpha_ {t}} \mathbf {m} + \mathbf {z} _ {t} \right]} \tag {76}
$$

# C.4 Reverse Rate Matrix for MDLM
C.4 MDLM 的逆向速率矩阵

We can formulate the reverse rate matrix for MDLM using (76) and (56). Recall that the reverse rate matrix $\tilde { R } _ { t } ( \mathbf { y } ^ { \prime } , \mathbf { y } )$ is given by:
我们可以使用公式(76)和(56)来构建 MDLM 的逆向速率矩阵。回想一下，逆向速率矩阵 $\tilde { R } _ { t } ( \mathbf { y } ^ { \prime } , \mathbf { y } )$ 由下式给出：

$$
\tilde {R} _ {t} (\mathbf {y} ^ {\prime}, \mathbf {y}) = \left\{ \begin{array}{l l} \mathbf {s} _ {\theta} (\mathbf {y}) _ {\mathbf {y} ^ {\prime}} R _ {t} (\mathbf {y}, \mathbf {y} ^ {\prime}) & \mathbf {y} ^ {\prime} \neq \mathbf {y} \\ - \sum_ {\tilde {\mathbf {y}} \neq \mathbf {y}} \mathbf {s} _ {\theta} (\mathbf {y}) _ {\mathbf {y} ^ {\prime}} R _ {t} (\mathbf {y}, \tilde {\mathbf {y}}) & \mathbf {y} ^ {\prime} = \mathbf {y}. \end{array} \right.
$$

Let’s examine the cases where $\mathbf { y } = \mathbf { m }$ and $\mathbf { y } \neq \mathbf { m }$ .
让我们考察 $\mathbf { y } = \mathbf { m }$ 和 $\mathbf { y } \neq \mathbf { m }$ 的情况。

Case y =m: For $\mathbf { y } ^ { \prime } \neq \mathbf { m }$ , the reverse rate $\tilde { R } _ { t } ( \mathbf { y } ^ { \prime } , \mathbf { y } = \mathbf { m } )$ is given by:
情况 y = m：对于 $\mathbf { y } ^ { \prime } \neq \mathbf { m }$ ，逆向速率 $\tilde { R } _ { t } ( \mathbf { y } ^ { \prime } , \mathbf { y } = \mathbf { m } )$ 由下式给出：

$$
\begin{array}{l} \tilde {R} _ {t} (\mathbf {y} ^ {\prime} \neq \mathbf {m}, \mathbf {y} = \mathbf {m}) \\ = \mathbf {s} _ {\theta} (\mathbf {y} = \mathbf {m}) _ {\mathbf {y} ^ {\prime} \neq \mathbf {m}} R _ {t} (\mathbf {y} = \mathbf {m}, \mathbf {y} ^ {\prime}) \\ \end{array}
$$

Using (71) and (56), we get:
利用(71)和(56)，我们得到：

$$
= \frac {\alpha_ {t}}{1 - \alpha_ {t}} \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t}, t), \mathbf {y} ^ {\prime} \rangle \left[ - \frac {\alpha_ {t} ^ {\prime}}{\alpha_ {t}} \right]
$$

 

$$
= - \frac {\alpha_ {t} ^ {\prime}}{1 - \alpha_ {t}} \left\langle \mathbf {x} _ {\theta} \left(\mathbf {z} _ {t}, t\right), \mathbf {y} ^ {\prime} \right\rangle \tag {77}
$$

For $\mathbf { y } ^ { \prime } { = } \mathbf { m }$ , the reverse rate $\tilde { R } _ { t } ( \mathbf { y } ^ { \prime } , \mathbf { y } = \mathbf { m } )$ is given by:
对于 $\mathbf { y } ^ { \prime } { = } \mathbf { m }$ ，反向速率 $\tilde { R } _ { t } ( \mathbf { y } ^ { \prime } , \mathbf { y } = \mathbf { m } )$ 由下式给出：

$$
\begin{array}{l} \tilde {R} _ {t} (\mathbf {y} ^ {\prime} = \mathbf {m}, \mathbf {y} = \mathbf {m}) \\ = - \sum_ {\tilde {\mathbf {y}} \neq \mathbf {m}} \tilde {R} _ {t} (\tilde {\mathbf {y}}, \mathbf {y} = \mathbf {m}) \\ \end{array}
$$

Using (77), we get:
利用(77)，我们得到：

$$
= \sum_ {\tilde {\mathbf {y}} \neq \mathbf {m}} \frac {\alpha_ {t} ^ {\prime}}{1 - \alpha_ {t}} \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t}, t), \tilde {\mathbf {y}} \rangle
$$

 

$$
= \frac {\alpha_ {t} ^ {\prime}}{1 - \alpha_ {t}} \sum_ {\tilde {\mathbf {y}} \neq \mathbf {m}} \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t}, t), \tilde {\mathbf {y}} \rangle
$$

“zero-masking probability” in Sec. $3 . 2 . 3 \Longrightarrow \sum _ { { \tilde { \mathbf { y } } } \neq \mathbf { m } } \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , { \tilde { \mathbf { y } } } \rangle = 1 ; { \mathrm { h e n c e } } ,$
第 $3 . 2 . 3 \Longrightarrow \sum _ { { \tilde { \mathbf { y } } } \neq \mathbf { m } } \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , { \tilde { \mathbf { y } } } \rangle = 1 ; { \mathrm { h e n c e } } ,$ 节中的“零掩码概率”

$$
= \frac {\alpha_ {t} ^ {\prime}}{1 - \alpha_ {t}}. \tag {78}
$$

Case y ̸=m: For $\mathbf { y } ^ { \prime } \neq \mathbf { y }$ we have:
当 y ≠ m 时：对于 $\mathbf { y } ^ { \prime } \neq \mathbf { y }$ 我们有：

$$
\begin{array}{l} \tilde {R} _ {t} (\mathbf {y} ^ {\prime} \notin \{\mathbf {y}, \mathbf {m} \}, \mathbf {y} \neq \mathbf {m}) \\ = \mathbf {s} _ {\theta} (\mathbf {y} \neq \mathbf {m}) _ {\mathbf {y} ^ {\prime} \notin \{\mathbf {y}, \mathbf {m} \}} \underbrace {R _ {t} (\mathbf {y} \neq \mathbf {m} , \mathbf {y} ^ {\prime} \notin \{\mathbf {y} , \mathbf {m} \})} _ {= 0 \text { from (56) }} \\ = 0 \tag {79} \\ \end{array}
$$

For $\mathbf { y } ^ { \prime } { = } \mathbf { m } .$ , we have:
对于 $\mathbf { y } ^ { \prime } { = } \mathbf { m } .$ ，我们有：

$$
\begin{array}{l} \tilde {R} _ {t} (\mathbf {y} ^ {\prime} = \mathbf {m}, \mathbf {y} \neq \mathbf {m}) \\ = \mathbf {s} _ {\theta} (\mathbf {y} \neq \mathbf {m}) _ {\mathbf {y} ^ {\prime} = \mathbf {m}} \underbrace {R _ {t} (\mathbf {y} \neq \mathbf {m} , \mathbf {y} ^ {\prime} = \mathbf {m})} _ {= 0 \text { from (56) }} \\ = 0 \tag {80} \\ \end{array}
$$

Thus, for $\mathbf { y } ^ { \prime } { = } \mathbf { y } .$ , we have:
因此，对于 $\mathbf { y } ^ { \prime } { = } \mathbf { y } .$ ，我们有：

$$
\begin{array}{l} \tilde {R} _ {t} (\mathbf {y} ^ {\prime} = \mathbf {y}, \mathbf {y} \neq \mathbf {m}) \\ = - \sum_ {\tilde {\mathbf {y}} \neq \mathbf {y}} \tilde {R} _ {t} (\tilde {\mathbf {y}}, \mathbf {y} \neq \mathbf {m}) \\ \end{array}
$$

 

$$
= - \underbrace {\tilde {R} _ {t} (\tilde {\mathbf {y}} = \mathbf {m} , \mathbf {y} \neq \mathbf {m})} _ {= 0 \text { from (80) }} - \underbrace {\sum_ {\tilde {\mathbf {y}} \notin \{\mathbf {y} , \mathbf {m} \}} \tilde {R} _ {t} (\tilde {\mathbf {y}} , \mathbf {y} \neq \mathbf {m})} _ {= 0 \text { from (79) }}
$$

 

$$
= 0 \tag {81}
$$

Summarizing (77), (78), (79), (80), (81), we have:
总结 (77)、(78)、(79)、(80)、(81)，我们得到：

$$
\tilde {R} _ {t} (\mathbf {y} ^ {\prime}, \mathbf {y}) = \left\{ \begin{array}{l l} - \langle \mathbf {x} _ {\theta} (\mathbf {y}, t), \mathbf {y} ^ {\prime} \rangle \frac {\alpha_ {t} ^ {\prime}}{1 - \alpha_ {t}} & \mathbf {y} ^ {\prime} \neq \mathbf {m}, \mathbf {y} = \mathbf {m} \\ \frac {\alpha_ {t} ^ {\prime}}{1 - \alpha_ {t}} & \mathbf {y} ^ {\prime} = \mathbf {m}, \mathbf {y} = \mathbf {m} \\ 0 & \text { otherwise. } \end{array} \right.
$$

 

$$
= \boxed {- \frac {\alpha_ {t} ^ {\prime}}{1 - \alpha_ {t}} \left[ \mathbf {y} ^ {\prime} \right] ^ {\top} \left[ \mathbf {x} _ {\theta} (\mathbf {y}, t) - \mathbf {m} \right] \langle \mathbf {y}, \mathbf {m} \rangle} \tag {82}
$$

# C.5 Deriving MDLM’s NELBO via CTMC
C.5 通过 CTMC 推导 MDLM 的 NELBO

Now, we aim to show that substituting the expression for the rate matrix $R _ { t }$ in terms of state transition matrix $Q _ { t }$ from (56) into (65) and switching from score-parameterization to the SUBS parameterization (Sec. 3.2.3) yields the simplified NELBO for MDLM as given by (10). We present the proof below. Recall that the term ⟨a,b⟩ denotes the dot product of two vectors a and b. When a and b represent two one-hot vectors, this quantity evaluates to 1 if a = b and 0 otherwise.
现在，我们旨在证明将(56)中的速率矩阵 $R _ { t }$ 用状态转移矩阵 $Q _ { t }$ 表示的表达式代入(65)，并从得分参数化切换到 SUBS 参数化（第 3.2.3 节），即可得到 MDLM 的简化 NELBO，如(10)所示。我们将在下面给出证明。回想一下，符号⟨a,b⟩表示向量 a 和 b 的点积。当 a 和 b 表示两个 one-hot 向量时，这个量在 a=b 时取值为 1，否则为 0。

Proof. Recall that for absorbing state diffusion, y takes only two possible values, i.e., $\mathbf { y } \in \{ \mathbf { x } , \mathbf { m } \}$ . Thus, we expand (65) as follows:
证明。回想一下，对于吸收状态扩散，y 只取两个可能的值，即 $\mathbf { y } \in \{ \mathbf { x } , \mathbf { m } \}$ 。因此，我们按如下方式展开(65)：

$$
\begin{array}{l} \mathbb {E} _ {t \in [ 0, 1 ], \mathbf {y} \sim q _ {t} (. | \mathbf {x})} \Bigg [ \langle \mathbf {y}, \mathbf {x} \rangle \Bigg [ \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {x}} R _ {t} (\mathbf {x}, \mathbf {y} ^ {\prime}) \bigg (\mathbf {s} _ {\theta} (\mathbf {x}) _ {\mathbf {y} ^ {\prime}} - \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {x} | \mathbf {x})} \mathrm{log} \mathbf {s} _ {\theta} (\mathbf {x}) _ {\mathbf {y} ^ {\prime}} + K \bigg (\frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {x} | \mathbf {x})} \bigg) \bigg) \Bigg ] \\ + \langle \mathbf {y}, \mathbf {m} \rangle \left[ \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {m}} R _ {t} (\mathbf {m}, \mathbf {y} ^ {\prime}) \left(\mathbf {s} _ {\theta} (\mathbf {m}) _ {\mathbf {y} ^ {\prime}} - \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})} \mathrm{log} \mathbf {s} _ {\theta} (\mathbf {m}) _ {\mathbf {y} ^ {\prime}} + K \left(\frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})}\right)\right) \right] \\ \end{array}
$$

 

$$
\because R _ {t} \left(\mathbf {x}, \mathbf {y} ^ {\prime} \neq \mathbf {x}\right) = 0 \text { from (54), we get: }
$$

  

$$
= \mathbb {E} _ {t \in [ 0, 1 ], \mathbf {y} \sim q _ {t} (. | \mathbf {x})} \langle \mathbf {y}, \mathbf {m} \rangle \left[ \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {m}} R _ {t} (\mathbf {m}, \mathbf {y} ^ {\prime}) \left(\mathbf {s} _ {\theta} (\mathbf {m}) _ {\mathbf {y} ^ {\prime}} - \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})} \log \mathbf {s} _ {\theta} (\mathbf {m}) _ {\mathbf {y} ^ {\prime}} + K \left(\frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})}\right)\right) \right]
$$

Substituting $R _ { t } ( \mathbf { m } , \mathbf { y } ^ { \prime } \neq \mathbf { m } )$ from (52), we get:
将(52)中的 $R _ { t } ( \mathbf { m } , \mathbf { y } ^ { \prime } \neq \mathbf { m } )$ 代入，我们得到：

$$
= \mathbb {E} _ {t \in [ 0, 1 ], \mathbf {y} \sim q _ {t} (. | \mathbf {x})} \langle \mathbf {y}, \mathbf {m} \rangle \left[ \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {m}} - \frac {\alpha_ {t} ^ {\prime}}{\alpha_ {t}} \left(\mathbf {s} _ {\theta} (\mathbf {m}) _ {\mathbf {y} ^ {\prime}} - \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})} \log \mathbf {s} _ {\theta} (\mathbf {m}) _ {\mathbf {y} ^ {\prime}} + K \left(\frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})}\right)\right) \right]
$$

 

$$
= \mathbb {E} _ {t \in [ 0, 1 ], \mathbf {y} \sim q _ {t} (\cdot | \mathbf {x})} \langle \mathbf {y}, \mathbf {m} \rangle \left[ - \frac {\alpha_ {t} ^ {\prime}}{\alpha_ {t}} \left(\underbrace {\sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {m}} \mathrm{s} _ {\theta} (\mathbf {m}) _ {\mathbf {y} ^ {\prime}}} _ {\text {Term 1}} - \underbrace {\sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {m}} \frac {q _ {t} \left(\mathbf {y} ^ {\prime} | \mathbf {x}\right)}{q _ {t} (\mathbf {m} | \mathbf {x})} \log \mathrm{s} _ {\theta} (\mathbf {m}) _ {\mathbf {y} ^ {\prime}}} _ {\text {Term 2}} + \underbrace {\sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {m}} K \left(\frac {q _ {t} \left(\mathbf {y} ^ {\prime} | \mathbf {x}\right)}{q _ {t} (\mathbf {m} | \mathbf {x})}\right)} _ {\text {Term 3}}\right) \right] \tag {83}
$$

Term 1:
术语 1：

$$
\sum_ {\mathbf {y} \neq \mathbf {m}} \mathbf {s} _ {\theta} (\mathbf {m}) _ {\mathbf {y}}
$$

Using (67), we get,
根据(67)，我们得到，

$$
= \sum_ {\mathbf {y} \neq \mathbf {m}} \frac {\alpha_ {t}}{1 - \alpha_ {t}} \langle \mathbf {x} _ {\theta} (\mathbf {m}, t), \mathbf {y} \rangle
$$

 

$$
= \frac {\alpha_ {t}}{1 - \alpha_ {t}} \sum_ {\mathbf {y} \neq \mathbf {m}} \left\langle \mathbf {x} _ {\theta} (\mathbf {m}, t), \mathbf {y} \right\rangle
$$

“zero-masking probability” in Sec. $3 . 2 . 3 \Longrightarrow \sum _ { \mathbf { y } \neq \mathbf { m } } \langle \mathbf { x } _ { \theta } ( \mathbf { m } , t ) , \mathbf { y } \rangle = 1 ;$ hence,
“零掩码概率”在节 $3 . 2 . 3 \Longrightarrow \sum _ { \mathbf { y } \neq \mathbf { m } } \langle \mathbf { x } _ { \theta } ( \mathbf { m } , t ) , \mathbf { y } \rangle = 1 ;$ 因此，

$$
= \frac {\alpha_ {t}}{1 - \alpha_ {t}} \tag {84}
$$

Term 2:
术语 2：

$$
\begin{array}{l} \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {m}} \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})} \mathrm{log} \mathbf {s} _ {\theta} (\mathbf {m}) _ {\mathbf {y} ^ {\prime}} \\ = \frac {q _ {t} (\mathbf {x} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})} \log \mathbf {s} _ {\theta} (\mathbf {m}) _ {\mathbf {x}} + \sum_ {\mathbf {y} ^ {\prime} \not \in \{\mathbf {m}, \mathbf {x} \}} \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})} \log \mathbf {s} _ {\theta} (\mathbf {m}) _ {\mathbf {y} ^ {\prime}} \\ \end{array}
$$

$\therefore q _ { t } ( \mathbf { y } ^ { \prime } | \mathbf { x } ) { = } 0 \operatorname { f o r } \mathbf { y } ^ { \prime } \notin \{ \mathbf { x } , \mathbf { m } \}$ from (4)) we get:
$\therefore q _ { t } ( \mathbf { y } ^ { \prime } | \mathbf { x } ) { = } 0 \operatorname { f o r } \mathbf { y } ^ { \prime } \notin \{ \mathbf { x } , \mathbf { m } \}$ 从（4）式中可得：

$$
= \frac {q _ {t} (\mathbf {x} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})} \mathrm{log} \mathbf {s} _ {\theta} (\mathbf {m}) _ {\mathbf {x}}
$$

Using (4), we get:
使用（4）式，我们得到：

$$
= \frac {\alpha_ {t}}{1 - \alpha_ {t}} \log \mathbf {s} _ {\theta} (\mathbf {m}) _ {\mathbf {x}}
$$

Using (67), we get:
使用（67）式，我们得到：

$$
\begin{array}{l} = \frac {\alpha_ {t}}{1 - \alpha_ {t}} \log \left[ \frac {\alpha_ {t}}{1 - \alpha_ {t}} \langle \mathbf {x} _ {\theta} (\mathbf {m}, t), \mathbf {x} \rangle \right] \\ = \frac {\alpha_ {t}}{1 - \alpha_ {t}} \log \frac {\alpha_ {t}}{1 - \alpha_ {t}} + \frac {\alpha_ {t}}{1 - \alpha_ {t}} \log \left\langle \mathbf {x} _ {\theta} (\mathbf {m}, t), \mathbf {x} \right\rangle \tag {85} \\ \end{array}
$$

Term 3:
项 3：

$$
\begin{array}{l} \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {m}} K \left(\frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})}\right) \\ = \sum_ {\mathbf {y} ^ {\prime} \neq \mathbf {m}} \left[ \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})} \log \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})} - \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})} \right] \\ = \frac {q _ {t} (\mathbf {x} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})} \log \frac {q _ {t} (\mathbf {x} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})} - \frac {q _ {t} (\mathbf {x} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})} + \sum_ {\mathbf {y} ^ {\prime} \not \in \{\mathbf {x}, \mathbf {m} \}} \left[ \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})} \log \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})} - \frac {q _ {t} (\mathbf {y} ^ {\prime} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})} \right] \\ \end{array}
$$

$\therefore q _ { t } ( \mathbf { y } ^ { \prime } | \mathbf { x } ) { = } 0 \mathrm { f o r } \mathbf { y } ^ { \prime } \not \in \{ \mathbf { x } , \mathbf { m } \} , \mathrm { w e ~ g e t } ,$

$$
= \frac {q _ {t} (\mathbf {x} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})} \log \frac {q _ {t} (\mathbf {x} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})} - \frac {q _ {t} (\mathbf {x} | \mathbf {x})}{q _ {t} (\mathbf {m} | \mathbf {x})}
$$

Substituting the values using (4), we get:
使用公式(4)代入数值后，我们得到：

$$
= \frac {\alpha_ {t}}{1 - \alpha_ {t}} \log \frac {\alpha_ {t}}{1 - \alpha_ {t}} - \frac {\alpha_ {t}}{1 - \alpha_ {t}} \tag {86}
$$

Substituing (84), (85), and (86) in (83) we get,
将(84)、(85)和(86)代入(83)后，我们得到，

$$
\mathbb {E} _ {t \in [ 0, 1 ], \mathbf {y} \sim q _ {t} (. | \mathbf {x})} \langle \mathbf {y}, \mathbf {m} \rangle \left[ - \frac {\alpha_ {t} ^ {\prime}}{\alpha_ {t}} \left(\frac {\alpha_ {t}}{1 - \alpha_ {t}} - \frac {\alpha_ {t}}{1 - \alpha_ {t}} \log \frac {\alpha_ {t}}{1 - \alpha_ {t}} - \frac {\alpha_ {t}}{1 - \alpha_ {t}} \log \langle \mathbf {x} _ {\theta} (\mathbf {m}, t), \mathbf {x} \rangle \right. \right.
$$

 

$$
\left. \left. + \frac {\alpha_ {t}}{1 - \alpha_ {t}} \log \frac {\alpha_ {t}}{1 - \alpha_ {t}} - \frac {\alpha_ {t}}{1 - \alpha_ {t}}\right) \right]
$$

 

$$
= \mathbb {E} _ {t \in [ 0, 1 ], \mathbf {y} \sim q _ {t} (. | \mathbf {x})} \langle \mathbf {y}, \mathbf {m} \rangle \left[ - \frac {\alpha_ {t} ^ {\prime}}{\alpha_ {t}} \left(- \frac {\alpha_ {t}}{1 - \alpha_ {t}} \log \langle \mathbf {x} _ {\theta} (\mathbf {m}, t), \mathbf {x} \rangle\right) \right]
$$

 

$$
= \mathbb {E} _ {t \in [ 0, 1 ], \mathbf {y} \sim q _ {t} (. | \mathbf {x})} \left\langle \mathbf {y}, \mathbf {m} \right\rangle \left[ \frac {\alpha_ {t} ^ {\prime}}{1 - \alpha_ {t}} \log \left\langle \mathbf {x} _ {\theta} (\mathbf {m}, t), \mathbf {x} \right\rangle \right]
$$

Under the SUBS parameterization, l $\scriptstyle \mathbf { y } \left. \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { x } \right. = 0$ when $\mathbf { z } _ { t } = \mathbf { x } ;$ hence ${ \bf \langle y , m \rangle }$ can be dropped:
在 SUBS 参数化下，l $\scriptstyle \mathbf { y } \left. \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { x } \right. = 0$ 当 $\mathbf { z } _ { t } = \mathbf { x } ;$ 因此 ${ \bf \langle y , m \rangle }$ 可以省略：

$$
= \boxed {\mathbb {E} _ {t \in [ 0, 1 ], \mathbf {y} \sim q _ {t} (. | \mathbf {x})} \left[ \frac {\alpha_ {t} ^ {\prime}}{1 - \alpha_ {t}} \log \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t}, t), \mathbf {x} \rangle \right]} \tag {87}
$$

This concludes the proof.
证明完毕。

# Appendix D Experimental details
附录 D 实验细节

# D.1 Likelihood Evaluation
D.1 似然评估

We use a single monte-carlo estimate for t to evaluate the likelihood. The low discrepancy sampler (D.3) plays a key role in reducing the variance of the estimate as seen in Table 8.
我们使用单个蒙特卡洛估计来评估似然。低差异采样器（D.3）在减少估计的方差中起着关键作用，如表 8 所示。

# D.2 Avg. Number of Tokens seen
D.2 已见到的平均令牌数

Given training\_steps, batch\_size, context\_length, the number of tokens seen by the AR model is given as:
给定训练步数（training\_steps）、批次大小（batch\_size）和上下文长度（context\_length），AR 模型所见到的 token 数量表示为：

 

$$
\text { training\_steps } \times \text { batch\_size } \times \text { context\_length }. \tag {88}
$$

However, this expression doesn’t hold for a diffusion model, since at each training step, a fraction of the input tokens are masked before being fed to the model. Let $p _ { m }$ be the probability of a token being masked at a timestep t. For the log-linear schedule in our experiments, $p _ { m } = t$ . Thus, the expected number of tokens seen by the diffusion model is:
然而，这一表达式对于扩散模型并不适用，因为在每个训练步骤中，输入 token 的一部分会在被输入模型之前被掩盖。令 $p _ { m }$ 表示在时间步 t 时一个 token 被掩盖的概率。在我们的实验中使用的对数线性调度下， $p _ { m } = t$ 。因此，扩散模型所见到的预期 token 数量为：

$$
\begin{array}{l} \mathbb {E} _ {t \sim \mathcal {U} [ 0, 1 ]} \left[ \text { training\_steps } \times \text { batch\_size } \times \text { context\_length } \times p _ {m} \right] \\ = \text { training\_steps } \times \text { batch\_size } \times \text { context\_length } \times \mathbb {E} _ {t \sim \mathcal {U} [ 0, 1 ]} [ p _ {m} ] \\ = \text { training\_steps } \times \text { batch\_size } \times \text { context\_length } \times \mathbb {E} _ {t \sim \mathcal {U} [ 0, 1 ]} [ t ] \quad \because p _ {m} = t \\ = \text { training\_steps } \times \text { batch\_size } \times \text { context\_length } \times 0. 5. \quad \because \mathbb {E} _ {t \sim \mathcal {U} [ 0, 1 ]} [ t ] = 0. 5 \tag {89} \\ \end{array}
$$

LM1B. Following \[1, 33, 26\], we train MDLM for 1M training steps with a batch\_size = 512, and a context length of 128. Like \[33\] we use a log-linear schedule and hence the number of tokens seen by our model is ≈ 33B (89). Similarly, MDLM trained for 10M steps, saw 327B tokens in expectation. The corresponding AR baseline was trained for 0.5M and 5M steps to ensure a similar number of tokens was seen.
LM1B. 根据\[1, 33, 26\]，我们使用 batch\_size = 512 和上下文长度为 128，对 MDLM 进行 1M 步数的训练。与\[33\]类似，我们采用对数线性调度，因此我们的模型所见到的 token 数量约为 33B（89）。类似地，经过 10M 步数训练的 MDLM，预期所见到的 token 数量为 327B。相应的 AR 基线模型则分别进行了 0.5M 和 5M 步数的训练，以确保所见到的 token 数量相似。

OWT. We train SEDD and MDLM for 1M training steps with a batch\_ $\mathsf { s i z e } ~ = ~ 5 1 2$ , context\_length = 1024, and log-linear schedule. Hence, these models saw 262B tokens during training. Similarly, the AR model saw the same number of tokens when trained for 0.5M steps with the same batch\_size and context\_length.
OWT. 我们使用 batch\_ $\mathsf { s i z e } ~ = ~ 5 1 2$ 、上下文长度为 1024 和对数线性调度，对 SEDD 和 MDLM 进行 1M 步数的训练。因此，这些模型在训练过程中所见到的 token 数量为 262B。类似地，当 AR 模型使用相同的 batch\_size 和上下文长度，进行 0.5M 步数的训练时，所见到的 token 数量也相同。

# D.3 Low discrepancy sampler
D.3 低差异采样器

To reduce variance during training we use a low-discrepancy sampler, similar to that proposed in Kingma et al. \[29\]. Specifically, when processing a minibatch of N samples, instead of independently sampling N from a uniform distribution, we partition the unit interval and sample the time step for each sequence $i \in \{ 1 , . . . , N \}$ from a different portion of the interval $\begin{array} { r } { t _ { i } \sim U [ \frac { i - 1 } { N } , \frac { i } { N } ] } \end{array}$ \]. This ensures that our sampled timesteps are more evenly spaced across the interval \[0,1\], reducing the variance of the ELBO.
为了在训练过程中减少方差，我们使用一种低差异采样器，类似于 Kingma 等人\[29\]提出的方案。具体来说，在处理一个包含 N 个样本的小批量时，我们不从均匀分布中独立地采样 N 个样本，而是将单位区间进行划分，并为每个序列 $i \in \{ 1 , . . . , N \}$ 从区间的不同部分 $\begin{array} { r } { t _ { i } \sim U [ \frac { i - 1 } { N } , \frac { i } { N } ] } \end{array}$ 中采样时间步。这确保了我们的采样时间步在区间\[0,1\]上更加均匀分布，从而减少了 ELBO 的方差。

# D.4 Language Modeling
D.4 语言建模

For our forward noise process, we use a log-linear noise schedule similar to Lou et al. \[33\].
对于我们的正向噪声过程，我们使用一种类似于 Lou 等人\[33\]提出的对数线性噪声调度。

We detokenize the One Billion Words dataset following Lou et al. \[33\], whose code can be found $\mathrm { h e r e } ^ { 4 }$ . We tokenize the One Billion Words dataset with the bert-base-uncased tokenizer, following He et al. \[26\]. We pad and truncate sequences to a length of 128.
我们遵循 Lou 等人 \[33\] 的方法对 One Billion Words 数据集进行去分词处理，其代码可在 $\mathrm { h e r e } ^ { 4 }$ 找到。我们使用 bert-base-uncased 分词器对 One Billion Words 数据集进行分词，遵循 He 等人 \[26\] 的方法。我们将序列填充和截断至长度为 128。

We tokenize OpenWebText with the GPT2 tokenizer. We do not pad or truncate sequences – we concatenate and wrap them to a length of 1,024. When wrapping, we add the eos token in-between concatenated. We additionally set the first and last token of every batch to be eos. Since OpenWebText does not have a validation split, we leave the last 100k docs as validation.
我们使用 GPT2 分词器对 OpenWebText 进行分词。我们不进行序列填充或截断——我们将它们连接并包装成长度为 1,024。在包装时，我们在连接的序列之间添加 eos 标记。此外，我们将每个批次的第一和最后一个标记设置为 eos。由于 OpenWebText 没有验证集划分，我们将最后 100k 篇文档作为验证集。

We parameterize our autoregressive baselines, SEDD, and MDLM with the transformer architecture from Lou et al. \[33\]. We use 12 layers, a hidden dimension of 768, 12 attention heads, and a timestep embedding of 128 when applicable. Word embeddings are not tied between the input and output.
我们使用来自 Lou 等人 \[33\] 的 transformer 架构对我们的自回归基线 SEDD 和 MDLM 进行参数化。我们使用 12 层、768 维的隐藏维度、12 个注意力头，并在适用时使用 128 维的时间步嵌入。词嵌入在输入和输出之间不共享。

We use the AdamW optimizer with a batch size of 512, constant learning rate warmup from 0 to a learning rate of 3e-4 for 2,500 steps. We use a constant learning rate for 1M, 5M, or 10M steps on One Billion Words, and 1M steps for OpenWebText. We use a dropout rate of 0.1.
我们使用 AdamW 优化器，批大小为 512，从 0 到 3e-4 的学习率进行 2,500 步的恒定学习率预热。我们在 One Billion Words 上使用 1M、5M 或 10M 步的恒定学习率，在 OpenWebText 上使用 1M 步。我们使用 0.1 的 dropout 率。

# D.5 Zeroshot Likelihood
D.5 零样本似然

We evaluate zeroshot likelihoods by taking the models trained on OpenWebText and evaluating likelihoods on the validation splits of 7 datasets: Penn Tree Bank (PTB; Marcus et al. \[36\]), Wikitext \[38\], One Billion Word Language Model Benchmark (LM1B; Chelba et al. \[8\]), Lambada \[41\], AG News \[68\], and Scientific Papers (Pubmed and Arxiv subsets; Cohan et al. \[10\]). We detokenize the datasets following Lou et al. \[33\]. For the AG News and Scientific Papers (Pubmed and Arxiv), we apply both the Wikitext and One Billion Words detokenizers. Since the zeroshot datasets have different conventions for sequence segmentation, we wrap sequences to 1024 and do not add eos tokens in between sequences.
我们通过在 OpenWebText 上训练的模型上评估 7 个数据集（Penn Tree Bank（PTB；Marcus 等人\[36\]）、Wikitext\[38\]、One Billion Word Language Model Benchmark（LM1B；Chelba 等人\[8\]）、Lambada\[41\]、AG News\[68\]以及科学论文（Pubmed 和 Arxiv 子集；Cohan 等人\[10\]））的验证集上的似然来评估零样本似然。我们按照 Lou 等人\[33\]的方法对数据集进行去分词处理。对于 AG News 和科学论文（Pubmed 和 Arxiv），我们应用了 Wikitext 和 One Billion Words 的去分词器。由于零样本数据集在序列分割方面有不同的规范，我们将序列包装为 1024 个，并且在序列之间不添加 eos 标记。

# D.6 Representation Learning
D.6 表示学习

Following Devlin et al. \[15\], we evaluate on all GLUE tasks \[65\], but exclude WNLI.
遵循 Devlin 等人\[15\]的方法，我们在所有 GLUE 任务\[65\]上进行评估，但排除了 WNLI。

We pre-train a MosaicBERT model on C4 \[46\] for 70k steps, corresponding to 36B tokens. We pad and truncate the data to 128 tokens using the bert-base-uncased tokenizer.
我们在 C4 \[46\]上预训练了一个 MosaicBERT 模型，共训练了 70k 步，相当于 36B 个 token。我们使用 bert-base-uncased 分词器将数据填充和截断到 128 个 token。

MosaicBERT \[43\] has a similar architecture to bert-base-uncased and has 137M parameters, 12 layers, 12 attention heads, a hidden dimension of 768, an intermediate size of 3072, and ALiBi attention bias \[44\].
MosaicBERT \[43\]的架构与 bert-base-uncased 相似，具有 137M 个参数，12 层，12 个注意力头，隐藏维度为 768，中间维度为 3072，以及 ALiBi 注意力偏置\[44\]。

For pre-training, we use the following hyperparameters: A global batch size of 4096 with gradient accumulation, a learning rate of 5e-4, linear decay to 0.02x of the learning rate with a warmup of 0.06x of the full training duration, and the decoupled AdamW optimizer with 1e-5 weight decay and betas 0.9 and 0.98.
对于预训练，我们使用以下超参数：全局批大小为 4096，带梯度累积，学习率为 5e-4，线性衰减到学习率的 0.02 倍，预热为完整训练时长的 0.06 倍，以及解耦的 AdamW 优化器，权重衰减为 1e-5，beta 值为 0.9 和 0.98。

For diffusion fine-tuning we use AdamW with a warmup of 2,500 steps from a learning rate of 0 to 5e-5, betas 0.95 and 0.999, and batch size 512. We train for 5k steps total, corresponding to 32M tokens.
对于扩散微调，我们使用 AdamW，从 0 到 5e-5 的学习率预热 2500 步，beta 值为 0.95 和 0.999，批大小为 512。我们总共训练 5k 步，相当于 32M 个 token。

For GLUE evaluation, we use the HuggingFace script found here5. We use the default parameters for all datasets, except for a batch size of 16, which we found helped with smaller datasets. This includes the default of 3 epochs for all datasets and learning rate of 2e-5.
在 GLUE 评估中，我们使用此处找到的 HuggingFace 脚本 5。我们对所有数据集使用默认参数，除了批量大小为 16，我们发现这有助于较小的数据集。这包括所有数据集的默认 3 个 epoch 和 2e-5 的学习率。

# D.7 Diffusion DNA Models
D.7 扩散 DNA 模型

Dataset We pre-train the Caduceus MLM \[50\] on the HG38 human reference genome \[11\]. Following Schiff et al. \[50\], we use character- / base pair-level tokenization. The dataset is based on the splits used in Avsec et al. \[3\]: the training split comprises of 35 billion tokens covering the human genome. This consists of 34,021 segments extended to a maximum length of 1,048,576 (220 segments). We maintain a constant $2 ^ { 2 0 }$ tokens per batch. For the Genomics Benchmark tasks, we use 5-fold cross-validation where we split the training set into 90/10 train/validation splits.
数据集 我们在 HG38 人类参考基因组\[11\]上预训练 Caduceus MLM\[50\]。遵循 Schiff 等人\[50\]，我们使用字符/碱基对级别的分词。该数据集基于 Avsec 等人\[3\]中使用的分割：训练集包含 350 亿个 token，覆盖人类基因组。这包括 34,021 个片段，扩展到最大长度 1,048,576（220 个片段）。我们保持每个批次 $2 ^ { 2 0 }$ 个 token 的常数。对于基因组基准任务，我们使用 5 折交叉验证，其中我们将训练集分成 90/10 的训练/验证分割。

Architecture The Caduceus MLM uses as a backbone a bi-directional variant of the data-dependent SSM Mamba block proposed in Gu et al. \[22\]. This architecture is ideal as it contains inductive biases that preserve reverse complement (RC) equviariance, respecting the inherent symmetry of double-stranded DNA molecules \[35, 50, 70\].
架构 Caduceus MLM 使用 Gu 等人\[22\]中提出的双向数据相关 SSM Mamba 块作为主干。这种架构非常适合，因为它包含保持反向互补（RC）等变性的归纳偏差，尊重双链 DNA 分子的固有对称性\[35, 50, 70\]。

Training details All models are pre-trained on 10B tokens (10K steps) and fine-tuned on a generative objective for an additional 50B tokens (50K steps). We use a global batch size of 1024 for a context length of 1024 tokens. Downstream task fine-tuning is performed for 16K steps ( 1B tokens).
训练细节 所有模型均在 10B 个 token（10K 步）上进行预训练，并在生成目标上进行微调，额外使用 50B 个 token（50K 步）。我们使用 1024 个 token 的上下文长度和 1024 的全局批处理大小。下游任务微调进行 16K 步（1B 个 token）。

For performing Caduceus MLM pre-training, we follow Schiff et al. \[50\] for the model size configuration, and hyperparameter selection. For pre-training, we use a fixed 15% mask rate as done in Devlin et al. \[15\]. Of the ’masked’ tokens, 80% are replaced with \[MASK\] , 10% are replaced with a random token from the vocabulary, and 10% are left unchanged.
为了执行 Caduceus MLM 预训练，我们遵循 Schiff 等人 \[50\] 的模型大小配置和超参数选择。在预训练过程中，我们使用固定的 15% 掩码率，如 Devlin 等人 \[15\] 所做的那样。在“掩码”的 token 中，80% 被替换为 \[MASK\]，10% 被替换为词汇表中的一个随机 token，10% 保持不变。

For fine-tuning all Mamba-based models (including Caduceus) on diffusion objectives, we lower the learning rate from 8e-3 to 1e-3. For fine-tuning HyenaDNA \[39\], we lower the learning rate from 6e-4 to 5e-5. Similar to Gu et al. \[22\], Schiff et al. \[50\], we found that Mamba-based models were robust to higher learning rates. We exclude timestep embeddings for all Diffusion DNA experiments, as we show it has minimal impact on generative performance (see Table 12, Suppl. E.5).
为了在扩散目标上微调所有基于 Mamba 的模型（包括 Caduceus），我们将学习率从 8e-3 降低到 1e-3。为了微调 HyenaDNA \[39\]，我们将学习率从 6e-4 降低到 5e-5。与 Gu 等人 \[22\]、Schiff 等人 \[50\] 类似，我们发现基于 Mamba 的模型对较高的学习率具有鲁棒性。我们排除了所有扩散 DNA 实验的时间步嵌入，因为我们表明它对生成性能的影响最小（见表 12，补充 E.5）。

We perform downstream task fine-tuning on the final hidden state embedding from pre-training. We perform mean pooling across the sequence length, which may vary from 200 to approximately 2,000 bps. We report the mean and ± on max/min classification accuracy over 5-fold cross-validation (CV) using different random seeds, with early stopping on validation accuracy. For each task, we do a hyperparameter sweep over batch size and learning rate and report the values of the 5-fold CV for the best configuration.
我们在预训练的最终隐藏状态嵌入上执行下游任务微调。我们对序列长度进行均值池化，该长度可在 200 到约 2000 bps 之间变化。我们使用不同的随机种子，通过 5 折交叉验证（CV）报告均值以及±最大/最小分类精度，并在验证精度上实施提前停止。对于每个任务，我们对批大小和学习率进行超参数扫描，并报告最佳配置的 5 折 CV 值。

Genomic Benchmark Task Distributions We use a subset of the Genomic Benchmark tasks with an emphasis on tasks from Human data. The positive samples for each dataset were generated by selecting samples that were annotated, either computationally or experimentally, in previous work (e.g enhancers, promoters, open chromatin regions (OCR)) \[20\]. These annotations each correspond to subsets of the genome of varying sizes that may exhibit different distributions of DNA than those observed globally over the reference genome. Due to this, the observed dataset may have a different distribution than the data used for pre-training and calculating perplexity. This might in turn lead to a case where perplexity and downstream performance may not necessarily correlate.
基因组基准任务分布 我们使用基因组基准任务的一个子集，特别关注来自人类数据的任务。每个数据集的正样本是通过选择先前工作中（例如增强子、启动子、开放染色质区域（OCR））被注释（无论是计算上还是实验上）的样本生成的\[20\]。这些注释各自对应于不同大小的基因组子集，这些子集可能表现出与参考基因组整体上观察到的 DNA 分布不同的分布。由于这一点，观察到的数据集的分布可能与用于预训练和计算困惑度（perplexity）的数据不同。这反过来可能导致困惑度（perplexity）和下游性能不一定相关的情况。

# Appendix E Additional Experiments
附录 E 额外实验

# E.1 Noise schedule parameterization
E.1 噪声调度参数化

As described in Sec. 3.4, the ELBO is invariant to the functional form of $\alpha _ { t }$ . To demonstrate this, we evaluate MDLM, initially trained using a log-linear schedule on OWT, by replacing the noise schedule with various other noise schedules as mentioned below. Following prior works \[1, 33, 54\], we parameterize $\alpha _ { t } = e ^ { - \sigma ( t ) }$ , where $\sigma ( t ) : [ 0 , 1 ] \to \mathbb { R } ^ { + }$ . Various functional forms of $\sigma ( t )$ are listed below:
如第 3.4 节所述，ELBO 对于 $\alpha _ { t }$ 的函数形式是不变的。为了证明这一点，我们评估 MDLM，该模型最初使用对数线性调度在 OWT 上训练，通过用以下提到的各种其他噪声调度替换噪声调度来评估。遵循先前的工作\[1, 33, 54\]，我们对 $\alpha _ { t } = e ^ { - \sigma ( t ) }$ 进行参数化，其中 $\sigma ( t ) : [ 0 , 1 ] \to \mathbb { R } ^ { + }$ 。以下是 $\sigma ( t )$ 的各种函数形式的列表：

Log Linear \[1, 33, 54\]. The log linear schedule is given as:
对数线性\[1, 33, 54\]。对数线性调度给出：

$$
\sigma (t) = - \log (1 - t) \tag {90}
$$

Cosine Squared schedule \[24\]. The Cosine Squared schedule is given as:
余弦平方调度\[24\]。余弦平方调度给出：

$$
\sigma (t) = - \log \cos^ {2} \left(\frac {\pi}{2} (1 - t)\right) \tag {91}
$$

Cosine schedule. The Cosine schedule is given as:
余弦调度。余弦调度给出：

$$
\sigma (t) = - \log \cos \left(\frac {\pi}{2} (1 - t)\right) \tag {92}
$$

Linear. The Linear schedule is given as:
线性。线性调度方案表示为：

$$
\sigma (t) = \sigma_ {\max} t \tag {93}
$$

where $\sigma _ { \mathrm { m a x } }$ is a very large number. In our experiments we set it to $1 0 ^ { 8 }$ .
其中 $\sigma _ { \mathrm { m a x } }$ 是一个非常大的数。在我们的实验中，我们将其设置为 $1 0 ^ { 8 }$ 。

# E.1.1 ELBO Invariance
E.1.1 ELBO 不变性

The function $\alpha _ { t }$ is invertible due to the monotonicity assumption in Sec. 3.1, and so we can perform the following change of variables in (10): $\gamma \equiv \log ( 1 - \alpha _ { t } )$ . Let $f : [ 0 , 1 ] \to \mathbb { R } ^ { - }$ be a function such that $\gamma = f ( t )$ . Note that $\alpha _ { t }$ goes through a monotonic transformation to obtain $\gamma ;$ hence, γ is also monotonic in t since $\alpha _ { t }$ is monotonic in t. This implies that the function $f$ is invertible. Let $t { = } \dot { f } ^ { - 1 } ( \gamma )$ . Then, we can we have the following diffusion loss:
由于第 3.1 节中的单调性假设，函数 $\alpha _ { t }$ 是可逆的，因此我们可以在公式 (10) 中进行以下变量替换： $\gamma \equiv \log ( 1 - \alpha _ { t } )$ 。令 $f : [ 0 , 1 ] \to \mathbb { R } ^ { - }$ 是一个函数，使得 $\gamma = f ( t )$ 。请注意， $\alpha _ { t }$ 经历单调变换得到 $\gamma ;$ ，因此 γ 在 t 上也是单调的，因为 $\alpha _ { t }$ 在 t 上是单调的。这意味着函数 $f$ 是可逆的。令 $t { = } \dot { f } ^ { - 1 } ( \gamma )$ 。然后，我们可以得到以下扩散损失：

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{NELBO}} ^ {\infty} = \mathbb {E} _ {q} \int_ {t = 0} ^ {t = 1} \frac {\alpha_ {t} ^ {\prime}}{1 - \alpha_ {t}} \log \langle \mathbf {x} _ {\theta} (\mathbf {z} _ {t}, t), \mathbf {x} \rangle \mathrm{d} t \\ = - \mathbb {E} _ {q} \int_ {t = 0} ^ {t = 1} \log \left\langle \mathbf {x} _ {\theta} \left(\mathbf {z} _ {t}, t\right), \mathbf {x} \right\rangle \frac {\mathrm{d}}{\mathrm{d} t} \left[ \log \left(1 - \alpha_ {t}\right) \right] \mathrm{d} t \\ = - \mathbb {E} _ {q} \int_ {t = 0} ^ {t = 1} \log \left\langle \mathbf {x} _ {\theta} \left(\mathbf {z} _ {t}, t\right), \mathbf {x} \right\rangle \frac {\mathrm{d}}{\mathrm{d} t} [ f (t) ] \mathrm{d} t \quad \text { Substituting } f (t) = \log \left(1 - \alpha_ {t}\right) \\ = - \mathbb {E} _ {q} \int_ {\boldsymbol {\gamma} = - \infty} ^ {\boldsymbol {\gamma} = 0} \log \left\langle \mathbf {x} _ {\theta} \left(\mathbf {z} _ {f ^ {- 1} (\boldsymbol {\gamma})}, f ^ {- 1} (\boldsymbol {\gamma})\right), \mathbf {x} \right\rangle d \boldsymbol {\gamma} \quad \text { Change of variables } \boldsymbol {\gamma} \equiv f (t) \\ = - \mathbb {E} _ {q} \int_ {\boldsymbol {\gamma} = - \infty} ^ {\boldsymbol {\gamma} = 0} \log \left\langle \mathbf {x} _ {\theta} \left(\tilde {\mathbf {z}} _ {\boldsymbol {\gamma}}, f ^ {- 1} (\boldsymbol {\gamma})\right), \mathbf {x} \right\rangle d \boldsymbol {\gamma} \quad \tilde {\mathbf {z}} _ {\boldsymbol {\gamma}} \equiv \mathbf {z} _ {f ^ {- 1} (\boldsymbol {\gamma})} \\ = - \mathbb {E} _ {q} \int_ {\boldsymbol {\gamma} = - \infty} ^ {\boldsymbol {\gamma} = 0} \log \left\langle \tilde {\mathbf {x}} _ {\boldsymbol {\theta}} \left(\tilde {\mathbf {z}} _ {\boldsymbol {\gamma}}, \boldsymbol {\gamma}\right), \mathbf {x} \right\rangle \mathrm{d} \boldsymbol {\gamma} \quad \tilde {\mathbf {x}} _ {\boldsymbol {\theta}} \left(\tilde {\mathbf {z}} _ {\boldsymbol {\gamma}}, \boldsymbol {\gamma}\right) \equiv \mathbf {x} _ {\boldsymbol {\theta}} \left(\tilde {\mathbf {z}} _ {\boldsymbol {\gamma}}, f ^ {- 1} (\boldsymbol {\gamma})\right) \tag {94} \\ \end{array}
$$

This new formulation demonstrates that the diffusion loss is invariant to the functional form of $\alpha _ { t }$ . In Table 9, we demonstrate empirically that noise schedules with different functional forms evaluate to the same Likelihood which is consistent with our theory. However, different schedules lead to different per data point variance. Notably, the log-linear schedule exhibits the lowest variance among all the noise schedules considered.
这种新的公式表明扩散损失与 $\alpha _ { t }$ 的函数形式无关。在表 9 中，我们通过实验证明，具有不同函数形式的噪声调度评估出的似然值相同，这与我们的理论一致。然而，不同的调度会导致每个数据点的方差不同。值得注意的是，对数线性调度在所有考虑的噪声调度中表现出最低的方差。

Table 9: Likelihood in bits per dimension (BPD) for different noise schedules on OWT dataset, is reported along with the mean and variance associated with each noise schedule per data point. We empirically observe that noise schedules with different functional forms yield the same likelihood, consistent with our theory in Sec. 3.4; however, different schedules result in different variances.
表 9：在 OWT 数据集上不同噪声调度每维度的似然值（BPD），同时报告了每个噪声调度每个数据点的均值和方差。我们通过实验观察到，具有不同函数形式的噪声调度产生相同的似然值，这与第 3.4 节中的理论一致；然而，不同的调度会导致不同的方差。

<table><tbody><tr><td data-imt-p="1" data-imt_insert_failed_reason="same_text">σ(t)</td><td data-imt-p="1">Mean均值</td><td data-imt-p="1">Variance per datapoint每个数据点的方差</td></tr><tr><td data-imt-p="1">Log Linear (90)对数线性（90）</td><td>3.30</td><td>1.81</td></tr><tr><td data-imt-p="1">Cosine (92)余弦（92）</td><td>3.30</td><td>3.30</td></tr><tr><td data-imt-p="1">Cosine Squared (91)余弦平方（91）</td><td>3.30</td><td>3.30</td></tr><tr><td data-imt-p="1">Linear (93)线性（93）</td><td>3.30</td><td>7.57</td></tr></tbody></table>

# E.2 Faster sampling with caching
E.2 基于缓存的快速采样

In Figure 10, we compare the wall clock times of variaous methods: AR, SEDD, MDLM with caching, and MDLM without caching for generating 64 samples on a single GPU. When sampling in batches, a change of 1 token would necessitate a call to the denoising model. Therefore, smaller batch sizes have a lower likelihood of a token being unmasked. This might lead one to prefer generating samples in smaller batches, as opposed to using a larger batch size that fully saturates the GPU. Table 10 shows that generating samples with a batch size of 1 and using caching is twice as fast as generating samples without caching while fully utilizing the GPU. In Fig. 2, we observe that MDLM without caching yields samples that consistently get better generative perplexity than SEDD. For $T = \{ 5 k , 1 0 k \}$ , both SEDD and MDLM get better generative perplexity than the AR model.
在图 10 中，我们比较了各种方法在单个 GPU 上生成 64 个样本的墙时钟时间：AR、SEDD、带缓存的 MDLM 和不带缓存的 MDLM。当批量采样时，1 个 token 的变化将需要调用去噪模型。因此，较小的批量大小使得 token 被未遮盖的可能性更低。这可能使得人们倾向于使用较小的批量生成样本，而不是使用能够完全饱和 GPU 的大批量大小。表 10 显示，使用批量大小为 1 并使用缓存生成样本的速度是不带缓存且完全利用 GPU 生成样本的两倍。在图 2 中，我们观察到不带缓存的 MDLM 生成的样本在生成困惑度上始终优于 SEDD。对于 $T = \{ 5 k , 1 0 k \}$ ，SEDD 和 MDLM 的生成困惑度都优于 AR 模型。

Table 10: Wall clock time reported in minutes to generate 64 samples on a single A5000 GPU.
表 10：在单个 A5000 GPU 上生成 64 个样本所报告的墙时钟时间（分钟）。

<table><tbody><tr><td></td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">T=5k(↓)</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">T=10k(↓)</td></tr><tr><td>SEDD</td><td>85.3</td><td>155.2</td></tr><tr><td>MDLM</td><td>70.3</td><td>127.9</td></tr><tr><td data-imt-p="1">+ caching+ 缓存</td><td>40.1</td><td>60.4</td></tr></tbody></table>

Generative perplexities across sample times on OpenWebText
在 OpenWebText 上，不同样本时间的生成困惑度

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-23/18fab104-cc56-43d9-8508-e6cf151b27e7/dc3840407abdd7c67920c2823ae066ba8888f4e1f00f4e5ecc268f099fc2a39b.jpg)

Figure 2: Generative perplexities across wall clock time for generating 64 samples on OWT using a single 32GB A5000 GPU are compared by varying $T \in \{ 1 0 \bar { 0 } , 5 0 0 , 1 0 \bar { 0 } 0 , 5 0 0 0 , \bar { 1 } 0 0 0 0 \}$ in the reverse diffusion process. The samples are generated in mini-batches with a batch size of 16 for AR, SEDD, and MDLM without caching, as it is the largest batch size that fits on this GPU. For MDLM with caching, we vary the batch size.
图 2：在 OWT 上使用单个 32GB A5000 GPU 生成 64 个样本时，通过改变反向扩散过程中的 $T \in \{ 1 0 \bar { 0 } , 5 0 0 , 1 0 \bar { 0 } 0 , 5 0 0 0 , \bar { 1 } 0 0 0 0 \}$ ，比较了生成过程中的生成困惑度。对于 AR、SEDD 和 MDLM，样本以 16 的批量大小进行生成，不使用缓存，因为这是该 GPU 上可以容纳的最大批量大小。对于使用缓存的 MDLM，我们变化批量大小。

# E.3 LM1B ablations
E.3 LM1B 消融实验

We assess the importance of our continuous-time framework by performing ablation on diffusion steps T . In Table 11, we compare NLL and PPL under continuous and discrete T in MDLM. We find that NLL consistently decreases as $T \to \infty$ .
我们通过在扩散步长 T 上执行消融实验来评估我们连续时间框架的重要性。在表 11 中，我们比较了 MDLM 在连续和离散 T 下的 NLL 和 PPL。我们发现 NLL 随着 $T \to \infty$ 的增大而持续减小。

Table 11: Discrete vs continuous time evaluation for MDLM w/o time-conditioning on OWT. MDLM was trained with $T = \infty$ . We report test perplexity for a discrete T .
表 11：MDLM 在 OWT 上无时间条件下的离散时间与连续时间评估。MDLM 使用 $T = \infty$ 进行训练。我们报告了离散 T 的测试困惑度。

<table><tbody><tr><td>T</td><td data-imt-p="1" data-imt_insert_failed_reason="same_text">PPL(≤)</td></tr><tr><td data-imt-p="1" data-imt_insert_failed_reason="same_text">∞</td><td>23.05</td></tr><tr><td>10</td><td>42.18</td></tr><tr><td>20</td><td>30.70</td></tr><tr><td>50</td><td>25.77</td></tr><tr><td>100</td><td>24.35</td></tr><tr><td>200</td><td>23.66</td></tr><tr><td>500</td><td>23.26</td></tr><tr><td>1000</td><td>23.15</td></tr></tbody></table>

# E.4 Train NLL curves on OWT
E.4 在 OWT 上训练 NLL 曲线

In Figure 3, we show that MDLM achieves lower variance loss during training compared to a previous diffusion language model, SEDD. Training is performed over 1M steps on OWT (which corresponds to 524B tokens).
在图 3 中，我们展示了 MDLM 在训练过程中相比于之前的扩散语言模型 SEDD 具有更低的方差损失。训练在 OWT 上进行了 100 万步（对应于 5240 亿个 token）。

Train Negative Log-Likelihood (NLL) on OpenWebText
在 OpenWebText 上训练负对数似然（NLL）

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-23/18fab104-cc56-43d9-8508-e6cf151b27e7/2cc6a49fbd209f2cb8fa212fb08ca2103b3feb3311204c460205659663274faa.jpg)

Figure 3: Train negative log-likelihood (NLL) curves across 1M gradient steps (524B tokens) on OpenWebText \[18\]. NLL is logged every 1K steps without value smoothing.
图 3：在 OpenWebText \[18\] 上跨 1M 梯度步长（524B 个 token）训练负对数似然（NLL）曲线。每 1K 步记录 NLL 值，不进行值平滑。

# E.5 Time-conditioning ablation on OWT
E.5 在 OWT 上的时间条件消融实验

In Table 12, we assess the importance of time conditioning in MDLM on OWT. We observe that time-conditioning has minimal impact on perplexity. Training is performed over 1M steps on OWT (which corresponds to 524B tokens).
在表 12 中，我们评估了时间条件在 MDLM 上 OWT 中的重要性。我们观察到时间条件对困惑度的影响最小。训练在 OWT 上进行了 1M 步（对应 524B 个 token）。

Table 12: Ablation on time-conditioning in MDLM on OWT.
表 12：在 OWT 上 MDLM 中时间调节的消融实验。

<table><tbody><tr><td data-imt-p="1">Method方法</td><td>PPL</td></tr><tr><td data-imt-p="1">MDLM w/ time-conditioning带时间调节的 MDLM</td><td>23.21</td></tr><tr><td data-imt-p="1">MDLM w/o time-conditioning不带时间调节的 MDLM</td><td>23.05</td></tr></tbody></table>

# E.6 Unconditional Samples
E.6 无条件样本

Here, we present some unconditional samples generated by MDLM trained on OWT with a context length of L= 1024 for T ={1000,10000}.
这里，我们展示了由在 OWT 上训练的 MDLM 生成的无条件样本，其中上下文长度为 L=1024，T={1000,10000}。

# E.6.1 T = 1000

Example 1 <|endoftext|> a 17-10 victory and a trip to the playoffs.
示例 1 17-10 胜利并晋级季后赛。

The last wildcard seed: New York Jets, Houston and the last potable playoff spot. The last-second home wins: New Orleans and Carolina, 21-21.
最后一个通配符种子：纽约喷气机队、休斯顿以及最后一个可用的季后赛席位。最后一秒主场获胜：新奥尔良和卡罗来纳，21-21。

The Saints finish sixth with the highest regular season (42) NFC wins. They lost 14 of their 13 games in the conference playoffs.
圣徒队以 42 场常规赛（NFC）胜利的成绩排名第六。他们在联盟季后赛中输掉了 13 场比赛中的 14 场。

The Cardinals were in Group A in Round 1 with Game 2, Round 3, Round 4 and Quarter Game 5, but they made their last trip to the playoffs off North Carolina on the road as even North Carolina.
红雀队在第一轮的 A 组中进行了第 2 场比赛、第 3 轮、第 4 轮和第 5 场四分之一决赛，但他们最后一次进入季后赛是在客场对阵北卡罗来纳州，甚至北卡罗来纳州也是如此。

True to their reputation, the Cards swept the Saints in the first round, but knocked it out at home. No Panthers went to the playoffs more than the Saints.
名副其实，红雀队在第一轮横扫了圣徒队，但在主场将其淘汰。没有哪个黑豹队的成绩能超过圣徒队进入季后赛。

Don Jean no longer is the South Carolina Panther.
Don Jean 不再是南卡罗来纳黑豹队的球员。

The Cardinals thought that provided that he had a chance to be an NFL player.
红雀队认为，只要他有机会成为 NFL 球员。

"I did," said defensive end Lorenzo Williams with a laugh as he exited his car at the airport. "Also, I won that game."
“我做到了，”防守端球员洛伦佐·威廉姆斯笑着走出机场汽车时说。“而且，我赢得了那场比赛。”

KC win brings Carson back home.
KC 的胜利让卡森回到了家乡。

Griffin made promise on Sunday to never exactly give up the dunk. Although he failed to score 40 points in the playoffs, he has had better luck in them this year.
格里芬在周日承诺永不彻底放弃扣篮。尽管他在季后赛中未能得到 40 分，但今年季后赛他运气更好。

With turnovers and fumbles returning, he has to play out because the team doesn’t trust him. He’s long years of injuries, turnovers and calls because he knows he can play that way for everybody.
随着失误和混乱的回归，他不得不打出表现，因为球队不信任他。他多年的伤病、失误和判罚，因为他知道他可以那样为所有人打球。

Griffin is no stranger to Saints fans.
格里芬对圣徒队的球迷并不陌生。

"Players want him to know them," someone from South Carolina said after coming out against the Panthers — in their best home home Week 7 win Sunday — in an 11-9 rout. South Carolina did win its final three and passed the Saints, 24-1.
“球员们希望他了解他们，”来自南卡罗来纳州的人在对抗黑豹队——他们在周日最佳主场第七周比赛中以 11-9 大胜——后说。南卡罗来纳州赢得了最后三场比赛，并以 24-1 超越圣徒队。

Although the Cardinals are in the South, they am a step behind.
尽管红雀队在南边，但他们仍然落后一步。

They still have little time left to take down the West Coast wild card. There is no chance they get another victory.
他们几乎没有时间击败西海岸的灰熊队。他们不可能再获得胜利。

The West was out by Beshear in their first round games last season, losing by 63 to the 49ers.
上赛季首轮比赛中，西边被贝希尔击败，以 63 比 49 输给了 49 人队。

The outcome will be tough.
结果将很艰难。

"Now we’re so close, let’s figure out the time to win," Brees said. "We still have a few games left; I’m glad about that."
现在我们离成功如此之近，让我们来计算获胜所需的时间，Brees 说道。"我们还有几场比赛要打；我很高兴。"

South Carolina takes the revenge.
南卡罗来纳州进行了复仇。

When asked about his second time since Super 4, Brees shot back that he understood.
当被问及自超级 4 以来的第二次机会时，Brees 回应说他理解。

"You can doubt the answer but I think that was a no-brainer. In time, you try to prove an answer wrong," Brees said. "I think his ability will be as cool as Julio Jones’ ability, but having that time \[out\] to my season was difficult overall. I did what I was expecting to do. Hopefully they’ll tell me to try again."
"你可以怀疑答案，但我认为这毫无疑问。随着时间的推移，你试图证明一个答案是错误的，" Brees 说道。"我认为他的能力会和朱利奥·琼斯的能力一样出色，但对我来说，那个时间\[段\]对我的整个赛季来说非常困难。我做了我预期要做的。希望他们会告诉我再试一次。"

After their late win, the NFL calls the Saints reschedule’must try.’
在他们的胜利之后，NFL 呼吁圣徒队必须重新安排比赛。

"Because Saints," those who am there still say it, "focus on defense and, offense is defense."
“因为圣徒队，”那些在场的人仍然说，“专注于防守，进攻就是防守。”

ESPN said Saints’s star receiver Dashon Jeffishard, turning heads on long passes and connecting with open defenders, already had their 20-yard overall score from the field. When Jeffishard finished with three passing he set up. He obviously had no difference; it was his first snap-off.
ESPN 表示，圣徒队的明星接球手 Dashon Jeffishard 在长传中吸引了注意力，并与防守中的空位接球手建立了联系，已经从场内得到了 20 码的整体得分。当 Jeffishard 完成三次传球后，他建立了基础。他显然没有区别；这是他的第一次接球。

With his changes in his starting lineup, Brees was just hoping they had little to prove against Carolina over the weekend.
随着他对首发阵容的调整，Brees 只是希望他们能在周末对卡罗来纳队证明自己。

"I felt like we didn’t have quite enough focus on and there was so puny coverage that I wanted more of our guys at the same position so we’d up our game," he said.
我觉得我们在某些方面的关注不够集中，覆盖范围也太小了，我希望我们能在同一位置上增加更多球员，这样我们才能提升竞争力，"他说。

Brees said South Carolina was well at linebacker.
布里斯表示南卡罗来纳州在后卫线上表现相当出色。

"If that’s part of it, if you’re going to try and stay with what you’re going \[out with\]. What’d you want? Smart play," Brees said. "What would you say? That you’re always ready to play. So you’re going out there strong and ready to go to play football."
"如果这是其中一部分，如果你打算坚持你所做的\[外传\]。你想要什么？聪明的比赛，"布里斯说。"你会怎么说？你总是准备好比赛。所以你出去时强大，准备好去踢足球。"

That said, New Orleans was damned shy when it came to Carolina.
话说回来，新奥尔良在对抗卡罗来纳州时表现得非常害羞。

"My guys admit to feeling it a little bit \[Sunday but\] I say to them that they always knew, ’I don’t think that was necessarily how I would beat you, that will give them their confidence," Brees said.
"我的队友们承认有点感觉\[周日\]，但我告诉他们他们一直都知道，'我不认为那 necessarily 是我击败你们的方式，那会给他们信心，" 布里斯说。

"It was really hard because I’ve obviously learned a lot of detail about how to deal with everyone and as hard as I have to be, I also feel part of the stuff that they’ve been through on the team, like they’re still going to go through things they know are somewhat right, but they feel a lot of pressure so it’s got to be important to get it right now to get it in the future."
"这真的很难，因为我显然学到了很多关于如何应对每个人的细节，尽管我必须很严格，但我也能感受到他们团队经历的一部分，比如他们仍然会经历他们知道是某种程度上正确的事情，但他们感到很多压力，所以现在必须把它弄对，未来也必须把它弄对。"

Could all ask for more roses?
所有人都能要求更多玫瑰吗？

Let’s just take a slip, South Carolina, and face the NFC<|endoftext|>
我们就拿一个滑倒，南卡罗来纳州，面对 NFC。

Example 2 <|endoftext|> Memorial Hospital.
示例 2 纪念医院。

Valia and Hill had been working with the Coast Guard in response to public questions, and when they were reached couldn’t comment on the new information, Chapman said.
瓦利亚和希尔一直在与海岸警卫队合作，回应公众的疑问，当被联系时，他们无法对新信息发表评论，查普曼说。

People referred to Valia during the years from Hill’s family in Ants, and she cut in contact with their family and friends in 2016.
在从安茨的希尔家族中，人们提到了瓦利亚，她在 2016 年与他们的家人和朋友断绝了联系。

"Each day they stepped on the bus, when they left they saw me on TV," she said.
“每天他们上车时，离开时看到我在电视上，”她说。

After separating from their family recently, Valia, 32, also moved into a Richmond house last October.
最近与家人分离后，32 岁的瓦利亚去年十月也搬进了里士满的房子。

Read or Share this story: [http://usat.ly/1NNC4zY](http://usat.ly/1NNC4zY)<|endoftext|>CIVIL C. "Marky" Hogan has been charged with homicides with a few days remaining after the April 2 purdade high school shooting where an undercover medical examiner and two other state and Illinois police officers was using heroin to go see a therapist.
阅读或分享这个故事：http://usat.ly/1NNC4zY CIVIL C."马克 y"霍根已被指控犯有杀人罪，在 4 月 2 日 purdue 高中枪击事件后几天内，一名卧底法医和两名其他州及伊利诺伊州警察正在使用海洛因去看心理治疗师。

DICEZ TV’s Zach Putler reported Tuesday that Hogan was charged with felony drug possession by the Chicago Police Department at the preliminary hearing on Monday. Putler interviewed on Monday. Authorities could offer a limit until Cook County takes Tuesday afternoon or they have to assign plea agreements.
DICEZ TV 的扎克·普特勒于周二报道，霍根在周一的初步听证会上被芝加哥警察局指控犯有非法持有毒品的重罪。普特勒于周一接受采访。当局可能直到周二下午库克县采取行动，或者他们必须分配认罪协议。

Dogan said in a news conference he made during a conference call Wednesday in Chicago that he believes the people who used him as a legal tool in the killing and fired employees hired for suffering also participated.
多گان在一项新闻发布会上表示，他在芝加哥周三举行的电话会议期间认为，那些将他作为法律工具参与谋杀并解雇因受苦而受雇的人也参与了其中。

He said the couple’s request to an attorney Monday will let the charges finally play out. Their lawyer did not respond Wednesday.
他说周一夫妇向律师提出的请求将使指控最终得以实施。他们的律师周三没有回应。

Dogan would not give away to possibility that he speculated in a statement that he would escape and return unless shot.
多گان不会放弃他在声明中推测自己除非被射击否则会逃脱并返回的可能性。

Chicago police initially said the other drug charges failed to raise enough evidence to establish why the killers were charged last year, raising the possibility that the drug mix contributed to a reason for their arrest. But a new statement was made Tuesday by a man
芝加哥警方最初表示，其他毒品指控未能提供足够的证据来证明为何去年对凶手进行指控，这增加了毒品混合物可能是他们被逮捕原因之一的可能性。但周二，一名男子发表了新的声明。

who worked as the shooter in his unit on campus, and suggested the charges might be related to his work at university supervision.
他在校园里担任射击手的职务，并建议指控可能与他在大学监管方面的工作有关。

His attorney and the university’s president last week signaled that the incident of the bat gun was not a police investigation at Wednesday’s conference.
他的律师和大学校长上周在周三的会议上暗示，蝙蝠枪事件并非警方调查的事。

Michael Durin, Illinois State University spokesman said he did not meet with university officials at the conference, and that university officials don’t have any updates yet either.
伊利诺伊州立大学发言人迈克尔·杜林表示，他在会议上没有与大学官员会面，而且大学官员目前也没有任何更新。

"The fact that the Defendants were charged is a major factor in why it would get this much attention," the university spokesperson said. "Given that all the matters are not being resolved for months and months, any new specific information and other concerns they may be tasked with investigating now are understandable."
大学发言人表示：“被告被指控的事实是导致此事受到如此关注的重要因素。” “考虑到所有问题都需要数月时间才能解决，他们现在可能被赋予调查任何新的具体信息和其他问题的任务，这是可以理解的。”

After city police began looking for evidence in connection to Hogan’s April shooting, Durin said he had not noticed it until the Chicago Police Department found a person who was producing a bind gun on Illinois State campus. That same department found that 14 officers shot and were injured during a standoff, but it led to the launch of a combination of unrelated and related investigations leading to homicides charges in May 2015.
在市警方开始寻找与霍根四月枪击事件有关的证据后，杜林表示，直到芝加哥警察局在伊利诺伊州立大学校园发现一名正在制造绑枪的人之前，他都没有注意到此事。该部门还发现，14 名警官在僵持中开枪受伤，但这导致了 2015 年 5 月一系列不相关和相关的调查，最终导致了谋杀指控。

A memo from special school investigators suggests it had identified the drug fentanyl, and says the department had described the individual-oriented and inconsistent use of the gun, as well as the substance administered by CODC.
一份来自特殊学校调查员的备忘录表明，他们已经识别出药物芬太尼，并称该部门描述了个人化和不一致的枪支使用情况，以及 CODC 施用的物质。

Dogan’s allegations claim that a 2009 police gravesite package showed water running over campus and shows that the dental show photos of supposed victims were reinterpreted.
多甘的指控声称，2009 年的警方墓地包裹显示校园有水流过，并表明所谓的受害者牙齿照片被重新解读。

David Mann, a member of the Police Department, said he spoke exclusively to News 1 on condition of anonymity because university officials can’t review documents immediately, and university officials had to change information that had been a consideration.
警察局成员大卫·曼表示，他匿名独家接受新闻 1 的采访，因为大学官员无法立即审查文件，并且大学官员不得不更改已被考虑的信息。

"We didn’t change our information until he personally told the drug overdose problem," he said. "He said that drug dealing wasn’t really a focal point."
他说：“直到他亲自告知药物过量的問題，我们才更改了信息。” “他说，贩毒并不是真正的焦点。”

He however, in a May 15 statement, also was "proleased" with university and state officialsin the U.S. Attorney’s Office. "We do not have any way to estimate the crime syndicate, given that the finding in the case does not preclude a separate review of standard CPD policy directed at any school employee, and whether that employee worked or went out of work," Mann said.
然而，在 5 月 15 日的声明中，他也被美国地区检察官办公室的大学和州官员“proleased”。“鉴于案件中的发现并未排除对任何学校员工的标准 CPD 政策进行单独审查，我们没有任何方法来估计犯罪集团，而且该员工是否工作或离职，”曼恩说。

The lawsuit alleges that donations from a full visit with Hogan’s workers and thousands of dollars, spent on gift cards, cigars, stock tickets to trips across the country, as well as other financial accounts and income from social members of former employees, were missing from buyers bought in stock.
诉讼指控说，与霍根的工人进行完整访问的捐款以及数千美元用于礼品卡、雪茄、全国旅行股票凭证，以及其他来自前员工社会成员的财务账户和收入，都从买家手中购买了股票。

Both employees and co-participants remain employed at the university.
员工和共同参与者仍然在大学任职。

On the Illinois StateUniversity Facebook page<|endoftext|>Image Steve Healey, the Cook Electronics Co. Ltd., general manager of Aug. 8-Feb. 16 at a recent internal session meeting (see link) held in The Apple Building in London. This July 1939 photo shows Peggy Deaver in 1986. (Photo courtesy the Cook Electronics Co. Ltd., displayed here under G. Healey in prison dress.)<|endoftext|>Steve GIRO’s H-P<|endoftext|>
在伊利诺伊州立大学 Facebook 页面上，8 月 8 日至 2 月 16 日在伦敦苹果大楼举行的一次内部会议（见链接）中，展示了史蒂夫·希利，库克电子有限公司总经理。这张 1939 年 7 月的照片显示了佩吉·戴弗在 1986 年。（照片由库克电子有限公司提供，根据 G.希利在监狱服刑的照片在此展示。）

# E.6.2 T =10000

Example 1 <|endoftext|> has been presenting a number of temporary measures to help help resolve a crisis.
示例 1 已经提出了一些临时措施来帮助解决危机。

The last crisis was like this, but we have to resolve it at the specified level; we have to stay low. The people of Greece will be interested in the effectiveness of the measures. They will not only manage these measures, but also they will help in order to cope with the problems of the fiscal stability.
上一次危机是这样的，但我们必须在指定级别上解决它；我们必须保持低调。希腊人民将关注这些措施的有效性。他们不仅会管理这些措施，而且将帮助应对财政稳定问题。

However, we do not want to dispose assets for the treasury. This also, so we will work on developing the national economy, and also paying on the national debt.
然而，我们不想出售资产来为财政部提供资金。这也一样，因此我们将致力于发展国民经济，并偿还国家债务。

This affects the national incomes
这会影响国民收入

So, as of 2007-2011, we use the government’s temporary measures as a measure, helping resolve the crisis. In addition, we are able to pay for the borrowing costs. Additionally, we pay $440 billion to settle debt issues, which can never be settled by default in a country.
因此，在 2007-2011 年期间，我们使用政府的临时措施作为指标，帮助解决危机。此外，我们能够支付借款成本。此外，我们支付了 4400 亿美元来处理债务问题，一个国家永远无法通过违约来解决这个问题。

These temporary measures will be aimed on several fronts, because the government will have three different partners in the system, in my right.
这些临时措施将针对多个方面，因为政府将在系统中拥有三个不同的合作伙伴，在我右边。

Firstly, we will be allowed to borrow a lot more upon the addition of the emergency measures and these temporary measures will help provide for the repayment of our debts before we are forced into a crisis as a result of our borrowing bills. Secondly, in the case of this, we will have resort to temporary measures in the revenue budget for Greece. The budget costs the government another $5.2 billion a year.
首先，在紧急措施和这些临时措施的基础上，我们将被允许借入更多资金。这些临时措施将有助于在我们因借款账单而被迫陷入危机之前偿还我们的债务。其次，在这种情况下，我们将诉诸于希腊的临时收入预算。该预算每年给政府带来 52 亿美元的成本。

So what you propose - if it happens again, does this mean that, since 2010, will you resolve the deficits which will occur on the basis of what we already have?
那么你提出的方案——如果再次发生，这意味着自 2010 年以来，你将基于我们已有的基础来解决出现的赤字吗？

The fiscal situation
财政状况

We would be able to settle our debts by the end of June which is the end. That said, we are taking our part as one of the most important countries in Europe, not only to make a proper transfer of the money but also to rely on it in the economy. However, first of all, we cannot achieve this on a day-to-day basis.
我们能够在 6 月底之前偿还债务，这就算结束了。话说回来，作为欧洲最重要的国家之一，我们不仅要在资金转移上做出适当的安排，还要依靠它来推动经济。然而，首先，我们无法在日常工作中实现这一点。

It is still true that we have decided to be able to deal with the economic situation of the country, but there might be another change in the fiscal situation, and therefore, we will try to negotiate on the situation at the end of June and over the summer.
我们仍然决定能够应对国家的经济状况，但财政状况可能会有另一变化，因此，我们将尝试在 6 月底和整个夏天就这种情况进行谈判。

The changes in the fiscal situation would be up to the parliament of management, bureaucrats, judges and a legitimate parliament of Greece.
财政状况的变化将由管理议会、官僚、法官以及合法的希腊议会决定。

Is the government planning to talk about thethe ’temporary measures’ of Greece?
政府是否计划讨论希腊的“临时措施”？

We will continue the process to operate through the temporary measures. This is not a temporary measure at this point, because after a crisis, not thereyet at crisis level, you can still have enough investment purchases until the end of the month.
我们将继续通过临时措施进行操作。目前这并非临时措施，因为在危机之后，即使尚未达到危机级别，你仍然可以进行足够的投资购买直至月底。

Again the government decided to create a temporary measure and now it depends upon a particular event, such as that there’s another liquidity crunch. It is better that the government and the authorities decided to create a temporary measure effective in June at fair sum monthly bond rates.
政府再次决定创建临时措施，现在这取决于特定事件，例如出现另一次流动性危机。政府与当局决定在六月实施临时措施，以公平的月度债券利率有效。

The temporary measures will also enhance the government’s economic status, especially when following the measures at the end of the month.
临时措施也将提升政府的经济地位，尤其是在月底实施这些措施时。

Temporary measures is a real tool for growth, not just for the economy.
临时措施是增长的真正工具，而不仅仅是为了经济。

Knowing that there are several measures in place to increase our supply, for example, the level offor profit on public sector enterprises is certain, under all of these temporary measures the increases in output, after that, will increase the external demand and the internal demand.
知道有几种措施来增加我们的供应，例如，公共部门企业的利润水平将确定，在所有这些临时措施下，产出的增加，之后将增加外部需求和内部需求。

We will be able to create the demand, and also strengthen the government’s credibility through fiscal organization. What is important here, here is that we will apply these measures to our reserves, and at the same time, we apply these measures to the debt level, which will also be the aim of debt-free Greece.
我们将能够创造需求，并通过财政组织加强政府的信誉。这里重要的是，我们将这些措施应用于我们的储备，同时，我们也将这些措施应用于债务水平，这也将是无债希腊的目标。

So, first of all, everything is certain ofwhat continues to be collected by the government. Given the situation and after the release of the last data on October 16, you also recognize that this will not be any kind of non-payment.
因此，首先，所有继续由政府收集的事项都是确定的。鉴于当前情况，并在 10 月 16 日发布最新数据后，您也认识到这不会是任何形式的未付款项。

In the case of the payment against the equipment, we will be able to manage with the measures.
对于设备付款的情况，我们将能够通过措施进行管理。

What does government expect in its plans to create a fiscal consolidation for the public sector and the new budget.
政府在其为公共部门制定财政整合计划和新预算方面，期望什么。

Regarding this is the temporary measure, we will be able to cope with the troubled finances. However, I do not think it is any measure which threatens the fiscal stability of the economy. However, that is not a temporary measure, a permanent measure.
关于这一点，这是临时措施，我们将能够应对困扰的财务状况。然而，我不认为这是任何威胁经济财政稳定的措施。但是，这不是临时措施，而是永久措施。

On the other hand, there will be our ongoing work on construction in the ministry. If this falls, we will continue work on job creation, the expansion of the economy.
另一方面，我们部门将持续进行建设工作。如果这项工作失败，我们将继续进行就业创造和经济扩张工作。

Also, also the government mentioned the new government reforms, which increased labor hours for the employees, which will further the economic growth, and the second aspect of budget as well and this is government welfare, which will improve the quality of life. We will<|endoftext|>
此外，政府还提到了新的政府改革，这将增加员工的劳动时间，从而进一步推动经济增长，以及预算的第二个方面，这也是政府福利，将提高生活质量。我们将

Example 2 <|endoftext|> him. He said: “What are you doing?”
示例 2 him. 他说：“你在做什么？”

I hesitated before answering. “Boy, this is so exciting. You need a better girl. Is she?”
我犹豫了一下才回答。“小子，这太令人兴奋了。你需要一个更好的女孩。她是吗？”

And I said, “You don’t have a brain. You have no brain anymore.”
我说：“你没有脑子。你早就没脑子了。”

After a minute, he had walked back and said on his own, working through that, he thought he had got himself going in a new direction.
过了一分钟，他走回来，自言自语地说，经过一番思考，他认为自己已经走上了新的方向。

He could’ve been a better boy in the first three years.
他前三年本可以成为一个更好的孩子。

“You’ll only have once before it starts.”
“你只有一次机会，在它开始之前。”

MVP

The story is always, “That’s what the other guy has to do.” He was the guy who had to do anything. He had to reason with school officials. My cousin mentioned to me that some of my friends almost doubled over at one meeting.
故事总是，“那是别人该做的事。”他是那个必须做任何事的人。他必须与学校官员讲道理。我的表哥告诉我，在某个会议上，我的几个朋友几乎要弯腰了。

I’d picked up a lot of the money I owed him from high people in me; he liked my grades. Drop-outs didn’t consider me high enough to let me go hang out. He hung up when I challenged him after practice to show a new talk. He started making, and, quite, never
我从一些对我很重要的人那里还清了我欠他的很多钱；他喜欢我的成绩。辍学者认为我不够重要，不能让我出去玩。他在练习后当我挑战他展示一个新演讲时挂断了电话。他开始制作，而且，相当，再也没有

I first saw him C. morning, in the sixth grade class. He wouldn’t hang up with him on point at team meetings. He started talking about things about me: “I’m an M, I’ll get an A. Tonight.” Having had that conversation over lunch, my heart touched mine with pride. He came, my boy. Now he looks like he’s going back to school. I don’t know if he’s going to sue. Let’s just have a two-bedroom apartment, a $500-dollar condo for renting, and a pool. And then he was back.
我第一次在 C.早上，在六年级的课堂上看到他。在团队会议上，他不会因为某个观点而挂断电话。他开始谈论关于我的事情：“我是一个 M，我会得到 A。今晚。”在午餐时有了那次谈话后，我的心被我的骄傲所触动。他来了，我的孩子。现在他看起来像是要回到学校了。我不知道他是否会起诉。让我们有一个两居室的公寓，一个 500 美元的公寓出租，还有一个游泳池。然后他又回来了。

That was a part of my life as I think about it. It was the school year.
那是我想起来的一部分生活。那是学年。

I never saw a guy come up at the locker room and show a new talk. That day one day, I told the high school, “We’ll show up one day right here, we can have a little fun,” and after this, I remember a small handful of the boys made friends, and they never, ever showed up for a new talk.
我从未见过有人在储物间出现并展示新的谈话。那天，我对高中说：“我们有一天会出现在这里，我们可以稍微玩玩，”之后，我记得有几个男孩交了朋友，他们再也没有出现过新的谈话。

I call them “M’s kids. I always remember him, I remember his ass up his ass, getting ready for a freshman orientation out there. He’ll show everything.
我称他们为“M 的孩子”。我一直记得他，我记得他翘着屁股，准备参加新生引导活动。他会展示一切。

He’ll show if I’m freshman, I’ll act I was going to play junior. In a few years I’ll try it, then he’ll make sure he’s going to judge me. He will come over to me one day.
如果我是新生，我会假装我要参加初三年级。过几年我会试试，然后他会确保他要评判我。有一天他会过来找我。

Then one day, the senior class was sitting on the bench, pressing his ball on the floor of the locker room, the referee was just standing the knelt it down.
然后有一天，高年级的学生坐在长椅上，把球压在储物间的地板上，裁判正站在那里跪下。

And when he heard about that, another boy, three of his friends, and one of his cousins were on the other side of the room. The boys’ class was filling in with his new brother and his new cousin and his new M.M’s player.
当他听到这个消息时，另一个男孩，他的三个朋友，以及他的一个表弟在房间的另一边。男孩们的班级正在填入他的新兄弟、他的新表弟和他的新 M.M 的球员。

The senior class watched me walk me through the chair to the bench. Everyone passed by the boy. Just on his toes on a foot, too.
高年级的学生看着我走下椅子到长椅上。每个人都从男孩身边经过。他的脚尖也刚刚够到。

He \[and a girl\] passed over his head and, as I looked at them, he carried me into the locker room. And the biggest part of the story, was a mistake.
他和一个女孩从他头顶上经过，当我看着他们时，他把我带进了储物间。而故事中最大的部分是一个错误。

With his elbows out, he pulled me down on my shoe, my other, sort of a- don’t know what they were; palebelly somethings, like bleeding very much, or on little toes. He was up on the stairs and everybody watches, with men and high school kids, who saw him in the locker room. And he caught a breath. Then his old man approached me and, disappeared into the middle of the room. He took off his vest, fast enough as to herd him into the locker room. I walked into the room and read him little cards with my own eyes to make notes, to pull him under my shoes.
肘部向外，他把我拉到鞋上，我的另一只脚，不知道是什么；像肚子苍白的东西，像是流血很多，或者在小脚趾上。他在楼梯上，每个人都看着，有男人和高中生，在更衣室里看到他。然后他喘了口气。接着他父亲向我走来，消失在房间中间。他脱下背心，足够快地把他赶到更衣室。我走进房间，用我自己的眼睛读给他看小卡片，做笔记，把他拉到我的鞋下。

I told them: “Listen, because I say this today, when you talk to ’em today, “Just make sure you talk is more than you’ll show. He’s just listening to me, and he’s telling me I’m going to be there for him.”
我告诉他们：“听着，因为今天我说这话，今天你们跟他们说话的时候，‘一定要确保你们说的话比你们表现出来的更多。他只是在听我说话，他告诉我我会为他加油。’”

I’m always the one who wants to do something important about you than I show up. I walk around and ask, I want a message from you, “Keep it going. It<|endoftext|>
我总是想为你做些重要的事情，而不是我表现出来的。我四处走动，我问，我想从你那里得到一条信息，“继续下去。”

# NeurIPS Paper Checklist
NeurIPS 论文检查清单

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.
该清单旨在鼓励负责任的机器学习研究，解决可复现性、透明度、研究伦理和社会影响等问题。请勿删除该清单：未包含该清单的论文将被直接拒稿。该清单应遵循参考文献，并遵循（可选的）补充材料。该清单不计入页数限制。

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:
请仔细阅读清单指南，了解如何回答这些问题。对于清单中的每个问题：

• You should answer \[Yes\] , \[No\] , or \[NA\] .
• 您应回答\[是\]、\[否\]或\[不适用\]。

• \[NA\] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
• \[不适用\]表示该问题对于该特定论文不适用，或相关信息不可用。

• Please provide a short (1–2 sentence) justification right after your answer (even for NA).
• 请在您的答案后提供一个简短的（1-2 句话）理由（即使对于“NA”也是如此）。

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.
检查表答案是您论文提交的重要组成部分。它们对审稿人、领域主席、高级领域主席和伦理审稿人是可见的。您将被要求在最终修订后将其（与论文的最终版本一起）提交，其最终版本将随论文一起发表。

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "\[Yes\] " is generally preferable to "\[No\] ", it is perfectly acceptable to answer "\[No\] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "\[No\] " or "\[NA\] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer \[Yes\] to a question, in the justification please point to the section(s) where related material for the question can be found.
审稿人将被要求在评估中使用检查清单作为其中一个因素。虽然“\[是\]”通常优于“\[否\]”，但如果提供合理的解释（例如，“由于计算成本过高，未报告误差线”或“我们无法找到所使用数据集的许可证”），回答“\[否\]”也是完全可接受的。通常情况下，回答“\[否\]”或“\[NA\]”并不构成拒绝的理由。虽然问题以二元方式提出，但我们承认真实答案往往更加复杂，因此请根据您的最佳判断进行回答，并撰写解释以阐述。所有支持证据可以出现在正文中或补充材料中，以附录形式提供。如果您对某个问题回答\[是\]，请在解释中指明可以在哪些部分找到与该问题相关的材料。

# 1\. Claims
1\. 声明

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?
问题：摘要和引言中提出的主要声明是否准确反映了论文的贡献和范围？

Answer: \[Yes\]
回答：\[是\]

Justification: Claims are addressed
理由：已回应相关主张

# 2\. Limitations
2\. 局限性

Question: Does the paper discuss the limitations of the work performed by the authors?
问题：论文是否讨论了作者所进行工作的局限性？

Answer: \[Yes\]
回答：\[是\]

Justification: Our method under-performs compared to autoregressive models. We also discuss other limitations in the paper.
理由：与自回归模型相比，我们的方法表现不佳。我们还在论文中讨论了其他局限性。

# 3\. Theory Assumptions and Proofs
3\. 理论假设与证明

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
问题：对于每个理论结果，论文是否提供了完整的假设集和完整的（且正确的）证明？

Answer: \[Yes\]
答案：\[是\]

Justification: They are in the proofs.
理由：它们在证明中。

Guidelines:
指南：

• The answer NA means that the paper does not include theoretical results.
• 答案 NA 表示该论文不包含理论结果。

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• 论文中的所有定理、公式和证明都应编号并交叉引用。

• All assumptions should be clearly stated or referenced in the statement of any theorems.
• 所有假设都应在任何定理的陈述中明确说明或引用。

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• 证明可以出现在正文中或补充材料中，但如果出现在补充材料中，作者应被鼓励提供简短的证明概要以提供直观理解。

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• 相反地，论文主体中提供的任何非正式证明都应通过附录或补充材料中提供的正式证明来补充。

• Theorems and Lemmas that the proof relies upon should be properly referenced.
• 证明所依赖的定理和引理应正确引用。

# 4\. Experimental Result Reproducibility
4\. 实验结果可复现性

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
问题：论文是否充分披露了所有必要信息，以便复现论文的主要实验结果，这些信息对论文的主要论点及/或结论有影响（无论是否提供代码和数据）？

Answer: \[Yes\]
答案：\[是\]

Justification: We provide all hyperparameters necesessary to reproduce the experiments and will provide code.
理由：我们提供了所有必要的超参数以复现实验，并将提供代码。

# 5\. Open access to data and code
5\. 数据和代码的开放获取

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?
问题：论文是否提供了对数据和代码的开放获取，并附有足够的说明以忠实地重现补充材料中描述的主要实验结果？

Answer: \[No\]
答案：\[否\]

Justification: We will release all code after the paper is accepted. The datasets are already public.
理由：我们将在论文被接受后发布所有代码。数据集已经公开。

Guidelines:
指南：

• The answer NA means that paper does not include experiments requiring code.
• 答案 NA 表示该论文不包含需要代码的实验。

• Please see the NeurIPS code and data submission guidelines ([https://nips.cc/](https://nips.cc/) public/guides/CodeSubmissionPolicy) for more details.
• 请参阅 NeurIPS 代码和数据提交指南（https://nips.cc/ public/guides/CodeSubmissionPolicy）了解更多详情。

• While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• 虽然我们鼓励发布代码和数据，但我们理解这可能并不可行，因此“否”是一个可接受的答案。除非这对于贡献至关重要（例如，为一个新的开源基准），否则不能仅因不包含代码而拒绝论文。

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• 说明应包含运行以复现结果所需的精确命令和环境。有关详细信息，请参阅 NeurIPS 代码和数据提交指南（https: //nips.cc/public/guides/CodeSubmissionPolicy）。

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• 作者应提供数据访问和准备说明，包括如何访问原始数据、预处理数据、中间数据和生成数据等。

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• 作者应提供脚本以复现新提出的方法和基线的所有实验结果。如果只有部分实验可复现，他们应说明哪些实验从脚本中省略了以及原因。

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• 提交时，为保持匿名性，作者应发布匿名版本（如适用）。

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
• 建议在补充材料（附加在论文中）中尽可能提供更多信息，但允许包含数据代码的 URL 链接。

# 6\. Experimental Setting/Details
6\. 实验设置/详细信息

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
问题：论文是否详细说明了理解结果所需的全部训练和测试细节（例如，数据分割、超参数、选择方法、优化器类型等）？

Answer: \[Yes\]
答案：\[是\]

Justification: We provide detailed hyperparameters for all experiments.
理由：我们为所有实验提供了详细的超参数。

Guidelines:
指南：

• The answer NA means that the paper does not include experiments.
• 答案 NA 表示该论文不包含实验。

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• 实验设置应在论文的核心部分呈现，详细程度应足以欣赏结果并理解它们。

• The full details can be provided either with the code, in appendix, or as supplemental material.
• 完整细节可以通过代码、附录或补充材料提供。

# 7\. Experiment Statistical Significance
7\. 实验统计显著性

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
问题：论文是否适当且正确地报告了实验误差线或其他关于实验统计显著性的适当信息？

# Answer: \[Yes\]
答案：\[是\]

Justification: Many of our tabels include error bars and standard deviations
理由：我们许多表格包含误差线和标准差

# Guidelines:
指南：

• The answer NA means that the paper does not include experiments.
• 答案 NA 表示该论文不包含实验。

• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• 如果结果伴随误差线、置信区间或统计显著性检验，作者应回答"是"，至少对于支持论文主要声明的实验。

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• 误差线所捕捉的变异性因素应明确说明（例如，训练/测试分割、初始化、某些参数的随机抽取，或给定实验条件下的整体运行）。

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
• 计算误差线的方法应予以解释（封闭形式公式、调用库函数、自助法等）。

• The assumptions made should be given (e.g., Normally distributed errors).
• 所做的假设应给出（例如，误差正态分布）。

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• 应明确误差线是标准差还是均值的标准误差。

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• 报告 1-σ误差线是可以的，但应该说明。如果未验证误差正态性假设，作者最好报告 2-σ误差线，而不是说明他们有一个 96%的置信区间。

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• 对于非对称分布，作者应谨慎避免在表格或图中展示对称误差线，这些误差线会导致结果超出范围（例如负误差率）。

# 8\. Experiments Compute Resources
8\. 实验计算资源

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
问题：对于每个实验，论文是否提供了足够的计算机资源信息（计算工作类型、内存、执行时间），以便重现实验？

# Answer: \[Yes\] .
答案：\[是\]。

Justification: We conduct all experiments on 8x 3090s, 8xA6000s, 8xA100s, or 8xH100s. The largest models on OpenWebText take 2 weeks to train on 8xA100, the LM1B models only take 2 days to train on the same hardware
理由：我们在 8x 3090s、8xA6000s、8xA100s 或 8xH100s 上进行了所有实验。在 OpenWebText 上，最大的模型在 8xA100 上训练需要 2 周时间，而 LM1B 模型在同一硬件上仅需 2 天即可训练完成。

# Guidelines:
指南：

• The answer NA means that the paper does not include experiments.
• 答案 NA 表示该论文不包含实验。

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• 论文应标明计算工作者的类型，包括 CPU 或 GPU、内部集群或云服务提供商，以及相关的内存和存储。

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• 论文应提供每个单独实验运行所需的计算量，并估计总计算量。

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).
• 论文应披露整个研究项目是否需要比论文中报告的实验更多的计算量（例如，未纳入论文的初步或失败的实验）。

# 9\. Code Of Ethics
9\. 道德准则

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics [https://neurips.cc/public/EthicsGuidelines](https://neurips.cc/public/EthicsGuidelines)?
问题：论文中所进行的研究是否在各个方面都与 NeurIPS 伦理准则 https://neurips.cc/public/EthicsGuidelines 相符？

# Answer: \[Yes\]
答案：\[是\]

Justification: We follow standard practices
理由：我们遵循标准实践

# Guidelines:
指南：

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• 答案 NA 表示作者未审查 NeurIPS 伦理准则。

• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• 如果作者回答否，他们应该解释需要偏离伦理准则的特殊情况。

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
• 作者应确保保持匿名（例如，如果由于当地法律或法规存在特殊考虑）。

# 10\. Broader Impacts
10\. 更广泛的影响

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
问题：论文是否讨论了所进行工作的潜在积极社会影响和消极社会影响？

# Answer: \[Yes\]
答案：\[是\]

Justification: Our model will allow for more controllable text generation models, and do not increase the capability of current autoregressive models
理由：我们的模型将允许更可控的文本生成模型，并且不会增加当前自回归模型的能力

# Guidelines:
指南：

• The answer NA means that there is no societal impact of the work performed.
• 答案 NA 表示所做工作没有社会影响。

• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• 如果作者回答 NA 或不，他们应该解释为什么他们的工作没有社会影响，或者为什么论文没有涉及社会影响。

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• 负面社会影响的例子包括潜在的恶意或非预期用途（例如，虚假信息、生成虚假资料、监控）、公平性考虑（例如，部署可能做出对特定群体产生不公平影响的决策的技术）、隐私考虑和安全考虑。

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• 会议期望许多论文是基础研究，与特定应用甚至部署无关。然而，如果有任何直接通往负面应用的途径，作者应该指出。例如，指出生成模型质量的提高可能被用于制造虚假信息以传播虚假信息是合理的。另一方面，指出用于优化神经网络的通用算法可能使人们更快地训练生成 Deepfakes 的模型则不是必需的。

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• 作者应当考虑在技术按预期使用且功能正常时可能产生的危害，在技术按预期使用但给出错误结果时可能产生的危害，以及（有意或无意）滥用技术所导致的危害。

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
• 如果存在负面社会影响，作者还可以讨论可能的缓解策略（例如，模型的分阶段发布，提供防御措施而非攻击手段，监控滥用的机制，监控系统如何随时间从反馈中学习，提高机器学习（ML）的效率和可访问性）。

# 11\. Safeguards
11\. 安全保障措施

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?
问题：论文是否描述了为负责任地发布具有高风险被滥用的数据或模型（例如，预训练语言模型、图像生成器或抓取的数据集）而采取的安全保障措施？

# Answer: \[Yes\]
答案：\[是\]

Justification: These models are trained on trivial datasets and unlikely to cause any harm compared to state of the art language models.
理由：这些模型在琐碎的数据集上进行训练，与最先进的语言模型相比，不太可能造成任何危害。

# Guidelines:
指南：

• The answer NA means that the paper poses no such risks.
• 答案 NA 表示该论文不存在此类风险。

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• 对于存在高风险被滥用或具有双重用途的已发布模型，应采取必要的保护措施以实现模型的受控使用，例如要求用户遵守使用指南或限制以访问模型，或实施安全过滤器。

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• 从互联网上抓取的数据集可能存在安全风险。作者应描述他们如何避免发布不安全的图像。

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
• 我们认识到提供有效的保护措施具有挑战性，并且许多论文并未要求这一点，但我们鼓励作者考虑这一点并尽最大努力。

# 12\. Licenses for existing assets
12\. 现有资产许可证

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
问题：论文中使用的资产（例如，代码、数据、模型）的创建者或原始所有者是否得到了适当的认可，并且许可证和使用条款是否明确说明并得到适当尊重？

# Answer: \[Yes\]
答案：\[是\]

Justification: All assets are publically available and we respect the licenses for all the data.
理由：所有资产都是公开可用的，我们尊重所有数据的许可证。

# Guidelines:
指南：

• The answer NA means that the paper does not use existing assets.
• 答案 NA 表示该论文未使用现有资源。

• The authors should cite the original paper that produced the code package or dataset.
• 作者应引用产生代码包或数据集的原始论文。

• The authors should state which version of the asset is used and, if possible, include a URL.
• 作者应说明所使用的资源版本，如果可能，应包含 URL。

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• 每个资源应包含许可证名称（例如，CC-BY 4.0）。

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• 对于从特定来源（例如网站）获取的数据，应提供该来源的版权和服务条款。

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• 如果资产被发布，应提供包中的许可证、版权信息和使用条款。对于流行的数据集，paperswithcode.com/datasets 为一些数据集整理了许可证。它们的许可证指南可以帮助确定数据集的许可证。

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• 对于重新打包的现有数据集，应同时提供原始许可证和派生资产的许可证（如果已更改）。

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.
• 如果在线上无法获取这些信息，鼓励作者联系资产的创建者。

# 13\. New Assets
13\. 新资产

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
问题：论文中引入的新资产是否得到了充分文档记录，并且资产是否提供了相应的文档？

Answer: \[NA\]
答案：\[NA\]

Justification: We provide no new assets.
理由：我们不提供新资产。

# Guidelines:
指南：

• The answer NA means that the paper does not release new assets.
• 答案 NA 表示该论文未发布新资产。

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• 研究人员应通过结构化模板在提交中详细说明数据集/代码/模型的细节。这包括关于训练、许可、限制等方面的信息。

• The paper should discuss whether and how consent was obtained from people whose asset is used.
• 论文应讨论是否以及如何从使用其资产的人员那里获得同意。

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
• 提交时，请记得匿名化您的资产（如适用）。您可以创建一个匿名化的 URL，或包含一个匿名化的 zip 文件。

# 14\. Crowdsourcing and Research with Human Subjects
14\. 群体外包与人本研究

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
问题：对于群体外包实验与人本研究，论文是否包含提供给参与者的完整指令文本和截图（如适用），以及关于补偿（如有）的详细信息？

Answer: \[NA\]
答案：\[NA\]

Justification: \[NA\]
理由：\[NA\]

# Guidelines:
指南：

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• 答案 NA 表示该论文不涉及众包或涉及人类受试者的研究。

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• 将此信息包含在补充材料中是可以的，但如果论文的主要贡献涉及人类受试者，那么应该在正文中尽可能详细地包含相关信息。

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
• 根据 NeurIPS 伦理准则，参与数据收集、整理或其他劳动的工作人员应至少获得数据收集所在国家的最低工资。

# 15\. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects
15\. 涉及人类受试者的研究机构审查委员会（IRB）批准或同等批准/审查

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?
问题：论文是否描述了研究参与者可能面临的风险，这些风险是否已告知受试者，以及是否已获得机构审查委员会（IRB）的批准（或根据您国家或机构的 要求获得同等批准/审查）？

Answer: \[NA\]
答案：\[NA\]

Justification: \[NA\]
理由：\[NA\]

# Guidelines:
指南：

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• 答案 NA 表示该论文不涉及众包或涉及人类受试者的研究。

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• 根据研究进行的国家，任何涉及人类受试者的研究可能需要 IRB 批准（或同等效力）。如果您获得了 IRB 批准，您应在论文中明确说明这一点。

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• 我们认识到，此程序在不同机构之间可能存在显著差异，我们期望作者遵守 NeurIPS 道德准则及其所在机构的指南。

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
• 对于初次提交，不要包含任何可能破坏匿名性的信息（如适用），例如进行评审的机构。

End of NEURIPS CHECKLIST. Must be at end of document after appendix
NEURIPS 清单结束。必须在附录之后作为文档的结尾。