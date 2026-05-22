# Simple and Effective Masked Diffusion Language Models

Subham Sekhar Sahoo Cornell Tech, NYC, USA. ssahoo@cs.cornell.edu

Aaron Gokaslan Cornell Tech, NYC, USA. akg87@cs.cornell.edu

Marianne Arriola Cornell Tech, NYC, USA. ma2238@cornell.edu

Edgar Marroquin Cornell Tech, NYC, USA. emm392@cornell.edu

Yair Schiff Cornell Tech, NYC, USA. yzs2@cornell.edu

Justin T Chiu Cornell Tech, NYC, USA. jtc257@cornell.edu

Alexander Rush Cornell Tech, NYC, USA. ar459@cornell.edu

Volodymyr Kuleshov Cornell Tech, NYC, USA. kuleshov@cornell.edu

## Abstract

While diffusion models excel at generating high-quality images, prior work reports a significant performance gap between diffusion and autoregressive (AR) methods in language modeling. In this work, we show that simple masked discrete diffusion is more performant than previously thought. We apply an effective training recipe that improves the performance of masked diffusion models and derive a simplified, Rao-Blackwellized objective that results in additional improvements. Our objective has a simple form—it is a mixture of classical masked language modeling losses— and can be used to train encoder-only language models that admit efficient samplers, including ones that can generate arbitrary lengths of text semi-autoregressively like a traditional language model. On language modeling benchmarks, a range of masked diffusion models trained with modern engineering practices achieves a new state-of-the-art among diffusion models, and approaches AR perplexity. We provide the code1, along with a blog post and video tutorial2 on the project page:

https://s-sahoo.com/mdlm

## 1 Introduction

Diffusion models excel at producing realistic, high-quality images and have received significant attention as potential tools for generating discrete data, such as text [1, 31, 33], biological sequences [2, 47], and graphs [60, 63]. Unlike autoregressive (AR) approaches, diffusion-based methods are not constrained to generate data sequentially, and therefore have the potential to improve long-term planning, controllable generation, and sampling speed. However, discrete diffusion methods exhibit a performance gap relative to AR models [1, 23, 26, 33], especially in language modeling. The standard measure of language modeling performance is log-likelihood: when controlling for parameter count, prior work reports a sizable log-likelihood gap between AR and diffusion models.

In this work, we show that simple masked diffusion language modeling (MDLM) combined with effective training recipes is more performant than previously thought [1, 26, 69]. We develop a wellengineered MDLM implementation that significantly improves discrete diffusion log-likelihood; we further improve likelihood using a simple substitution-based parameterization of the reverse diffusion process that enables deriving a Rao-Blackwellized continuous-time variational lower bound (ELBO) with improved tightness [49]. Interestingly, our objective has a simple form: it is a weighted average of masked language modeling (MLM) losses [15], and can be used to endow BERT-style, encoder-only models with principled generation capabilities. We complement this framework with efficient samplers—including ones that can generate semi-autoregressively like a typical language model.

<!-- image-->  
Figure 1: (Left) Our proposed masked diffusion language model (MDLM) is trained using a weighted average of masked cross entropy losses. (Top Right) In comparison to masked language models (MLM), MDLM’s objective correspond to a principled variational lower bound, and supports generation via ancestral sampling. (Bottom Right) Perplexity (PPL) on One Billion Words (LM1B) benchmark.

Our masked diffusion models achieve a new state-of-the-art among diffusion models on language modeling benchmarks and approach the perplexity of AR models within 15-25%. Surprisingly, simple engineering choices significantly improve performance in both our models and simple baselines that were previously thought to perform poorly. Our framework also extends to non-language domains, including biological sequence modeling. We pre-train DNA sequence models and observe similar or higher downstream performance compared to classical BERT-style training, while also introducing generative capabilities that classical masked DNA language models lack.

Contributions We describe (1) a simple masked diffusion language modeling (MDLM) framework with a well-engineered implementation that outperforms all existing diffusion models across language modeling benchmarks (LM1B [8], OWT [18], DNA [12]), and that significantly improves the performance of existing baselines [1, 26]. Our MDLM framework implements (2a) a substitution-based parameterization (SUBS) of the reverse unmasking diffusion process; SUBS allows us to derive (2b) a simple, continuous-time, Rao-Blackwellized objective that improves tightness and variance of the ELBO, further increasing performance. We complement MDLM with (3) fast samplers that support semi-autoregressive (SAR) generation and outperform previous SAR models.

## 2 Background

## 2.1 Diffusion Models

Diffusion models are trained to iteratively undo a forward corruption process q that takes clean data x drawn from the data distribution $q ( \mathbf { x } )$ and defines latent variables $\mathbf { z } _ { t }$ for $t \in [ 0 , \bar { 1 } ]$ that represent progressively noisy versions of x [27, 54, 56, 66, 48, 19]. The standard forward process for continuous x is

$$
\mathbf { z } _ { t } = \sqrt { \alpha _ { t } } \mathbf { x } + \sqrt { 1 - \alpha _ { t } } \epsilon\tag{1}
$$

where ${ \epsilon \sim \mathcal { N } ( \bf { 0 } , \bf { I } ) }$ and $( \alpha _ { t } ) _ { t \in [ 0 , 1 ] }$ is a noise schedule, monotonically decreasing in t. The parameterized reverse diffusion model pθ over x and $\mathbf { z } _ { t }$ is trained to maximize a variational lower bound on loglikelihood (ELBO). Given a number of discretization steps T, defining $s ( i ) = ( i - 1 ) / T \mathrm { a n d } t ( i ) = i / \breve { T }$

and using $D _ { \mathrm { K L } } [ \cdot ]$ to denote the Kullback–Leibler divergence, the Negative ELBO (NELBO) equals [54]:

$$
\mathbb { E } _ { q } \left[ \underbrace { - \log p \theta \left( \mathbf { x } | \mathbf { z } _ { t ( 0 ) } \right) } _ { \mathcal { L } _ { \mathrm { r e c o n s } } } + \underbrace { \sum _ { i = 1 } ^ { T } D _ { \mathrm { K L } } [ q ( \mathbf { z } _ { s ( i ) } | \mathbf { z } _ { t ( i ) } , \mathbf { x } ) | ] p \theta \left( \mathbf { z } _ { s ( i ) } | \mathbf { z } _ { t ( i ) } \right) ] } _ { \mathcal { L } _ { \mathrm { d i f f u s i o n } } } \right] + \underbrace { D _ { \mathrm { K L } } [ q ( \mathbf { z } _ { t ( T ) } | \mathbf { x } ) | ] p \theta \left( \mathbf { z } _ { t ( T ) } \right) } _ { \mathcal { L } _ { \mathrm { p r i o r } } }\tag{2}
$$

For brevity, we drop i from $t ( i )$ and $s ( i )$ below; in general, s will denote the time step before t.

## 2.2 Discrete Diffusion Models

Applications of diffusion modeling to discrete data can be broken into two broad categories. First are works that embed discrete structures in continuous space and then perform the Gaussian diffusion defined above on these continuous representations [9, 16, 23, 24, 30, 34, 57]. More related to our method are works that define a diffusion process directly on discrete structures. D3PM [1] introduces a framework with a Markov forward process $q ( \mathbf { z } _ { t } | \mathbf { z } _ { t - 1 } ) { \dot { = } } \mathbf { C a t } ( \mathbf { z } _ { t } ; Q _ { t } \mathbf { z } _ { t - 1 } )$ defined by the multiplication of matrices $Q _ { t }$ over $T$ discrete time steps. This process induces marginals

$$
q ( \mathbf { z } _ { t } | \mathbf { x } ) { = } \mathrm { C a t } ( \mathbf { z } _ { t } ; \bar { Q } _ { t } \mathbf { x } ) { = } \mathrm { C a t } ( \mathbf { z } _ { t } ; Q _ { t } \cdot Q _ { t - 1 } \cdots Q _ { 1 } \mathbf { x } )\tag{3}
$$

that represent the discrete-state form of (1). Extending this formalism to continuous time (as in (1)) relies on continuous time Markov chain (CTMC) theory [5]. The CTMC framework in turns leads to generalizations of the score matching perspective on diffusion modeling [55] to discrete data [33, 59]. Notably, SEDD [33] connects score-based approaches with ELBO maximization, enabling performant likelihood-based training of score-based models.

## 3 Simple Masked Diffusion Models

While previous work on discrete diffusion supports general forward processes (e.g., general $Q _ { t }$ in D3PM), absorbing state (i.e., masking) diffusion consistently achieves the best performance $[ 1 , 3 3 ]$ In this work, instead of supporting general noise processes, we focus on masking and derive tight Rao-Blackwellized objectives that outperform general approaches and do not require CTMC theory. In this section, we first define the diffusion process for a categorical random variable. Later in Sec. 3.5, we extend this process to sequences containing multiple such categorical variables. We denote our overall approach as Masked Diffusion Language Models (MDLM).

Notation. We denote scalar discrete random variables with K categories as ‘one-hot’ column vectors and define $\mathcal { V } \in \{ \mathbf { x } \in \{ 0 , 1 \} ^ { K } : \sum _ { i = 1 } ^ { K } \mathbf { x } _ { i } = 1 \}$ } as the set of all such vectors. Define $\operatorname { C a t } ( \cdot ; \pi )$ as the categorical distribution over K classes with probabilities given by $\pi \in \Delta ^ { K }$ , where $\Delta ^ { K }$ denotes the K-simplex. We also assume that the K-th category corresponds to a special [MASK] token and let $\mathbf { m } \in \mathcal { V }$ be the one-hot vector for this mask, i.e., m $\kappa = 1$ . Additionally, let $\mathbf { 1 } = \left\{ 1 \right\} ^ { K }$ and $^ { \prime } ( \mathbf { a } , \mathbf { b } )$ and a⊙b respectively denote the dot and Hadamard products between two vectors a and b.

## 3.1 Interpolating Discrete Diffusion

We restrict our attention to forward processes q that interpolate between clean data $\mathbf { x } \in \nu$ and a target distribution $\operatorname { C a t } ( . ; \pi )$ , forming a direct extension of Gaussian diffusion in (1). Let q define a sequence of increasingly noisy latent variables $\mathbf { z } _ { t } \in \mathcal { V }$ , where the time step t runs from $t = 0$ (least noisy) to $t = 1$ (most noisy). The marginal of $\mathbf { z } _ { t }$ conditioned on x at time t is

$$
q ( \mathbf { z } _ { t } | \mathbf { x } ) { = } \mathrm { C a t } ( \mathbf { z } _ { t } ; \alpha _ { t } \mathbf { x } { + } ( 1 { - } \alpha _ { t } ) \pi ) ,\tag{4}
$$

where $\alpha _ { t } \in [ 0 , 1 ]$ is a strictly decreasing function in t, with α0 ≈ 1 and $\alpha _ { 1 } \approx 0 ;$ see Suppl. E.1 for details. This implies transition probabilities $q ( \mathbf { z } _ { t } | \mathbf { z } _ { s } ) = \mathrm { C a t } ( \mathbf { z } _ { t } ; \alpha _ { t | s } \mathbf { z } _ { s } + ( 1 - \alpha _ { t | s } ) \pmb { \pi } )$ , where $\alpha _ { t | s } = \alpha _ { t } / \alpha _ { s }$ This indicates that during each diffusion step from $s \to t ,$ a fraction $( 1 - \dot { \alpha } _ { t | s } )$ of the probability mass is transferred to the prior distribution π. The reverse posterior is given as (see Suppl. 16 for details):

$$
q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) = \mathrm { C a t } \left( \mathbf { z } _ { s } ; \frac { [ \alpha _ { t | s } \mathbf { z } _ { t } + ( 1 - \alpha _ { t | s } ) \mathbf { 1 } \pi ^ { \top } \mathbf { z } _ { t } ] \odot [ \alpha _ { s } \mathbf { x } + ( 1 - \alpha _ { s } ) \pi ] } { \alpha _ { t } \mathbf { z } _ { t } ^ { \top } \mathbf { x } + ( 1 - \alpha _ { t } ) \mathbf { z } _ { t } ^ { \top } \pi } \right) .\tag{5}
$$

While (4) and (5) represent a special case of the more general diffusion processes proposed in D3PM [1], we show below that they yield a simplified variational lower bound objective and admit straightforward continuous time extensions.

## 3.2 Masked Diffusion

Next, we focus on masking processes and derive a simple Rao-Blackwellized objective for this choice of q. This objective incurs lower variance during training and improves tightness.

## 3.2.1 Forward Masking Process

$$
t ^ { \prime } { . }
$$

$$
t > t ^ { \prime } : q ( \mathbf { z } _ { t } | \mathbf { z } _ { t ^ { \prime } } = \mathbf { m } ) { = } \mathrm { C a t } ( \mathbf { z } _ { t } { ; } \mathbf { m } )
$$

The marginal of the forward process (4) is given by $q ( \mathbf { z } _ { t } | \mathbf { x } ) = \mathrm { C a t } ( \mathbf { z } _ { t } ; \alpha _ { t } \mathbf { x } + ( 1 - \alpha _ { t } ) \mathbf { m } )$ . Using properties of the masking process, the posterior $q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } )$ ) simplifies (5); see Suppl. A.2:

$$
\begin{array} { r } { q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) = \left\{ \begin{array} { l l } { \mathrm { C a t } ( \mathbf { z } _ { s } ; \mathbf { z } _ { t } ) } & { \mathbf { z } _ { t } \neq \mathbf { m } , } \\ { \mathrm { C a t } \Big ( \mathbf { z } _ { s } ; \frac { ( 1 - \alpha _ { s } ) \mathbf { m } + ( \alpha _ { s } - \alpha _ { t } ) \mathbf { x } } { 1 - \alpha _ { t } } \Big ) } & { \mathbf { z } _ { t } = \mathbf { m } . } \end{array} \right. } \end{array}\tag{6}
$$

## 3.2.2 Reverse Unmasking Process

The reverse process inverts the noise process defined by q. We consider both a finite number of steps T , as well as a continuous time model corresponding to $T \to \infty$ . We begin with the discrete-time case for which the generative model is expressed as $\begin{array} { r } { p _ { \theta } ( \mathbf { x } ) = \int _ { \mathbf { z } } p _ { \theta } ( \mathbf { z } _ { 1 } ) p _ { \theta } ( \mathbf { x } | \mathbf { z } _ { 0 } ) \prod _ { i = 1 } ^ { T } p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } ) \mathrm { d } \mathbf { z } _ { 0 : 1 } } \end{array}$

The optimal form for $p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$ matches the true posterior in (6): this follows immediately from the definition of the diffusion objective in (2), which is a sum of terms of the form $\begin{array} { r } { \operatorname { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) \big | \big | p _ { \theta } \big ( \mathbf { z } _ { s } | \mathbf { z } _ { t } \big ) \big ) } \end{array}$ However, (6) is conditioned on x, which we do not know. Therefore, we introduce a model $\mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) : \mathcal { V } \times [ 0 , 1 ] \to \Delta ^ { K }$ that approximates x with a neural network. We can also omit explicit dependence of $\mathbf { x } _ { \theta }$ on time t, which simplifies sampling, yielding a 2x inference speed-up (see Suppl. E.2).

## 3.2.3 SUBS Parameterization

The specific parameterization for $p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$ that we use is

$$
\begin{array} { r } { p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } ) = q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } = \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) ) = \left\{ \begin{array} { l l } { \mathrm { C a t } ( \mathbf { z } _ { s } ; \mathbf { z } _ { t } ) , } & { \mathbf { z } _ { t } \neq \mathbf { m } , } \\ { \mathrm { C a t } \Big ( \mathbf { z } _ { s } ; \frac { ( 1 - \alpha _ { s } ) \mathbf { m } + ( \alpha _ { s } - \alpha _ { t } ) \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) } { 1 - \alpha _ { t } } \Big ) . } & { \mathbf { z } _ { t } = \mathbf { m } , } \end{array} \right. } \end{array}\tag{7}
$$

Furthermore, we induce 2 key properties of the absorbing state diffusion process into our denoising model, ${ \bf x } _ { \theta } ( { \bf z } _ { t } , t )$ : an unmasked token remains unchanged during reverse diffusion, and the clean input is never masked. We implement these as substitutions to the output of ${ \bf x } _ { \theta } ( { \bf z } _ { t } , t )$ , hence we call our parameterization SUBS.

Zero Masking Probabilities First, notice that by definition, $\langle \mathbf { x } , \mathbf { m } \rangle = 0$ . For this reason, we design the denoising network such that $\begin{array} { r } { \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle = 0 , \mathrm { i . e . } } \end{array}$ , we substitute the logit index corresponding to the [MASK] token with −∞.

Carry-Over Unmasking Second, if $\mathbf { z } _ { t }$ is unmasked, then we desire ${ \bf x } _ { \theta } ( { \bf z } _ { t } , t ) = { \bf z } _ { t } ,$ i.e., unmasked latents are ‘carried over’. We accomplish this by substituting the output of our network to simply copy unmasked inputs.

In Suppl. B.1, we show that “Zero Masking Probabilities” property simplifies the D3PM’s NELBO (39) to (41), and “Carry-Over Unmasking” futher simplifies (41) to (43) whose continuous time equivalent is the simplified NELBO (10). Table 8 shows that each simplification leads to an improved likelihood.

## 3.3 Rao-Blackwellized Likelihood Bounds

Recall from (2) that the diffusion traning objective has the form $\mathcal { L } _ { \mathrm { r e c o n s } } + \mathcal { L } _ { \mathrm { d i f f u s i o n } } + \mathcal { L } _ { \mathrm { p r i o r } }$ . For the simplified reverse process in (7), the discrete-time diffusion loss for finite T simplifies to (Suppl. B.1.3):

$$
\mathcal { L } _ { \mathrm { d i f f u s i o n } } = \sum _ { i = 1 } ^ { T } \mathbb { E } _ { q } \big [ \mathrm { D } _ { \mathbf { K } \mathrm { L } } \big ( q ( \mathbf { z } _ { s ( i ) } | \mathbf { z } _ { t ( i ) } , \mathbf { x } ) \| p _ { \theta } ( \mathbf { z } _ { s ( i ) } | \mathbf { z } _ { t ( i ) } ) \big ) \big ] = \sum _ { i = 1 } ^ { T } \mathbb { E } _ { q } \bigg [ \frac { \alpha _ { t ( i ) } - \alpha _ { s ( i ) } } { 1 - \alpha _ { t ( i ) } } \mathrm { l o g } \big \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t ( i ) } ) , \mathbf { x } \big \rangle \bigg ]\tag{8}
$$

Note that this objective is simpler and more well-behaved than the expression one would obtain for $\mathrm { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) \big | \big | \overline { { p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } ) } } \big )$ under the parameterization induced by using pθ $\begin{array} { r } { ( { \bf z } _ { s } | { \bf z } _ { t } ) = q ( { \bf z } _ { s } | { \bf z } _ { t } , { \bf x } = } \end{array}$ ${ \bf x } _ { \theta } ( { \bf z } _ { t } , t ) )$ from (5), which is similar to what is used by D3PM [1] (see Suppl. A.2.4):

$$
\left[ \frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } } \log \frac { \alpha _ { t } \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle + ( 1 - \alpha _ { t } ) } { ( 1 - \alpha _ { t } ) \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { x } \rangle } + \frac { 1 - \alpha _ { s } } { 1 - \alpha _ { t } } \log \frac { ( 1 - \alpha _ { s } ) \big ( \alpha _ { t } \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle + ( 1 - \alpha _ { t } ) \big ) } { ( 1 - \alpha _ { t } ) \big ( \alpha _ { s } \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle + ( 1 - \alpha _ { s } ) \big ) } \right] \langle \mathbf { z } _ { t } , \mathbf { m } \rangle\tag{9}
$$

We refer to the process of obtaining (8) in lieu of (9) as a form of Rao-Blackwellization. Specifically, we analytically compute expectations such as $\langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle = 0$ in order to simplify objective (9) to obtain (8). Without analytical simplifications, a model must learn θ such that $\langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle = 0$ holds. Unlike in regular Rao-Blackwellization, simplifications are possible because of modeling choices for ${ \bf x } _ { \theta } ( { \bf z } _ { t } , t )$ (zero masking probabilities and carry-over unmasking). In that sense, our approach has similarities to graphical modeling, where incorporating conditional independencies into $p _ { \theta }$ sets certain log-likelihood terms to zero. However, our approach also empirically helps reduce variance, hence we refer to it as Rao-Blackwellization, somewhat abusing the usual terminology.

## 3.4 Continuous-Time Likelihood Bounds

Previous works have shown empirically and mathematically that increasing the number of steps T yields a tighter approximation to the ELBO [29]. Following a similar argument, we form an continuous extension of (8) by taking $T \to \infty$ (see Suppl. B.2), which yields the following NELBO, $\mathcal { L } _ { \mathrm { N E L B O } } ^ { \infty } \mathrm { : }$

$$
\mathcal { L } _ { \mathrm { N E L B O } } ^ { \infty } = \mathbb { E } _ { q } \int _ { t = 0 } ^ { t = 1 } \frac { \alpha _ { t } ^ { \prime } } { 1 - \alpha _ { t } } \mathrm { l o g } \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { x } \rangle \mathrm { d } t\tag{10}
$$

Invariance to the noise schedule The function $\alpha _ { t }$ is invertible due to the monotonicity assumption in Sec. 3.1, and so we can perform the following change of variables in (10): $\gamma \equiv \log ( 1 - \alpha _ { t } )$ Thus, the diffusion loss can be equivalently expressed as $\begin{array} { r } { \mathcal { L } _ { \mathrm { N E L B O } } ^ { \infty } = - \mathbb { E } _ { q } \int _ { \gamma = - \infty } ^ { \gamma = 0 } \log \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { \gamma } , \gamma ) , \mathbf { x } \rangle \mathrm { d } \gamma ; } \end{array}$ see Suppl. E.1.1 for details. This new formulation demonstrates that the diffusion loss is invariant to the functional form of $\alpha _ { t } ,$ , which we verify empirically in Suppl. E.1.

## 3.5 Masked Diffusion Language Models

Next, we apply masked diffusion to language modeling over sequences $\mathbf { x } ^ { 1 : L }$ of L tokens, with $\mathbf { x } ^ { \ell }$ denoting the ℓ-th token. We make the assumption that the forward noising process is applied independently across a sequence and that, conditioned on a sequence of latents $\bar { \mathbf { z } _ { t } ^ { 1 : L } }$ , the denoising process factorizes independently across tokens, i.e., $\begin{array} { r } { p _ { \theta } \big ( \mathbf { z } _ { s } ^ { 1 : L } \mid \mathbf { z } _ { t } ^ { 1 : L } \big ) = \prod _ { \ell = 1 } ^ { L } p _ { \theta } \big ( \mathbf { z } _ { s } ^ { \ell } \mid \mathbf { z } _ { t } ^ { 1 : L } \big ) } \end{array}$ . Thus, we use a single model to compute $\mathbf { x } _ { \theta } ^ { \ell } ( \mathbf { z } _ { t } ^ { 1 : L } , t )$ for each ℓ from a masked sequence $\mathbf { z } _ { t } ,$ optimizing:

$$
\mathcal { L } _ { \mathrm { N E L B O } } ^ { \infty } = \mathbb { E } _ { q } \int _ { t = 0 } ^ { t = 1 } \frac { \alpha _ { t } ^ { \prime } } { 1 - \alpha _ { t } } \sum _ { \ell } \log \langle \mathbf { x } _ { \theta } ^ { \ell } ( \mathbf { z } _ { t } ^ { 1 : L } , t ) , \mathbf { x } ^ { \ell } \rangle \mathrm { d } t\tag{11}
$$

Interestingly, our objective has a simple form: it is the weighted average of masked language modeling (MLM) losses [15]. Thus our work establishes a connection between generative diffusion models and encoder-only BERT models. Our objective enables principled selection of a (randomized) masking rate, and also endows BERT-style models with principled generation capabilities; see Sec. 6. The full training algorithm is provided in Suppl. B.3.

Note: Although (11) imposes a loss on all tokens, unmasked tokens don’t contribute to the loss, as they are copied over by the denoising network due to “carry-over unmasking” (Sec. 3.2.3), effectively reducing log $\langle \mathbf { x } _ { \theta } ^ { \ell } ( \mathbf { z } _ { t } ^ { 1 : L } , \dot { \mathbf { \xi } } _ { t } ) , \mathbf { x } ^ { \ell } \rangle$ to zero.

## 3.5.1 Training Considerations for Masked Diffusion

One of the key contributions of our work is a well-engineered implementation of masked diffusion models. Our experiments demonstrate that these improvements greatly boost performance even for methods previously thought to perform poorly, e.g., Austin et al. [1]. Below we briefly summarize these implementation details. First, we find that tokenization is critical to performance. Small vocabularies, such as the 8k vocabulary in Austin et al. [1], result in longer-range dependencies that decrease the performance of both diffusion and AR models. Additionally, by focusing on masked diffusion, we are able to provide a numerically stable implementation of the objective function. Namely, since previous formulations of discrete diffusion were constructed to accommodate a wide range of limiting distributions [1], the objective was implemented by materializing the full transition matrices $\bar { Q } _ { t }$ and posterior probabilities. In contrast, we evaluate $\bar { D _ { \mathrm { K L } } } [ q ( \mathbf { z } _ { s } \mid \mathbf { z } _ { t } , \mathbf { x } ) \bar { | } | p _ { \theta } ( \mathbf { z } _ { s } \mid \mathbf { z } _ { t } ) ]$ by examining only the masked token indices rather than comparing the full true and approximate posterior distributions.

Furthermore, we modernize the architecture for the denoising network relative to D3PM [1]. In lieu of the T5 architecture used in D3PM, we use the diffusion transformer (DiT) introduced in Peebles & Xie [42], which integrates time step conditioning into a standard encoder-only transformer [62] and uses rotary positional embeddings [58]. In addition, we implement a low-discrepancy sampler that reduces the variance of the ELBO, similar to Kingma et al. [29] and draws correlated samples $t _ { i }$ rather than performing i.i.d. sampling.

## 4 Inference and Sampling in Masked Diffusion Language Models

## 4.1 Efficient Ancestral Sampling

To generate a sequence of length $L$ , the reverse diffusion process starts with the sequence $\mathbf { z } _ { t = 1 } ^ { 1 : L }$ where $\mathbf { z } _ { t = 1 } ^ { \ell } = \mathbf { m }$ , for all $\ell \in \{ 1 , \ldots , L \}$ Then the subsequent latents, $\mathbf { z } _ { t } ^ { 1 : L }$ are generated by discretizing the reverse diffusion process with some finite T. Given $\mathbf { z } _ { t } ^ { 1 : L }$ , we construct $\mathbf { \widetilde { z } } _ { s } ^ { 1 : L }$ by sampling each token $\mathbf { z } _ { s } ^ { \ell }$ independently from the distribution $p _ { \theta } ( \mathbf { z } _ { s } ^ { \ell } | \mathbf { z } _ { t } ^ { 1 : L } )$ given in (7).

Note that in the reverse process, unmasked tokens remain unchanged. Thus, if no new tokens in $\smash { \mathbf { z } _ { c } ^ { 1 : L } }$ become unmasked (which can occur often in early denoising stages for large T ), then $\mathbf { z } _ { s } ^ { 1 : L } = \mathbf { z } _ { t } ^ { \hat { 1 } : L }$ Additionally if the denoising model, $\mathbf { x } _ { \theta } ( \mathbf { z } _ { t } ^ { 1 : L } )$ is not conditioned on time, then we can simply draw a new sample from $p _ { \theta } \big ( \mathbf { z } _ { s - 1 / T } ^ { 1 : L } \big | \mathbf { z } _ { s } ^ { 1 : L } \big )$ using the previously computed and cached value $\mathbf { x } _ { \theta } ( \mathbf { z } _ { t } ^ { 1 : L } )$ . This means we have effectively “skipped” over the time step s, saving a function call to the denoising network. Note that SEDD [33] does not support this caching because the denoising network models time-dependent rates, which requires conditioning on time.

## 4.2 Semi-Autoregressive Masked Diffusion Language Models

Our method also admits an effective semi-autoregressive (SAR) decoding method that allows the model to generate sequences of arbitrary length [24, 52, 53]. Let $\tilde { \mathbf { x } } ^ { 1 : L }$ represent the output from sampling a sequence of L tokens using the reverse diffusion process described above. To generate additional $\bar { L } ^ { \prime } < L$ tokens, we propose a generation algorithm in which the latter $L - L ^ { \prime }$ tokens $\tilde { \mathbf { x } } ^ { \bar { L } ^ { \prime } : L }$ are used as a prefix for an additional round of generation. Given the carry-over unmasking described in Sec. 3.2.3, these prefix tokens will simply be copied over at each decoding step. The remaining tokens are generated as above with $\mathbf { z } _ { s } ^ { \ell } \sim p _ { \theta } ( \mathbf { z } _ { s } ^ { \ell } | \mathbf { \bar { z } } _ { t } ^ { L : L + L ^ { \prime } } )$ for all $\ell { \in } \left\{ L { + } 1 , { \ldots } , L { + } L ^ { \prime } \right\}$ , with $\mathbf { z } _ { t = 1 } ^ { L - L ^ { \prime } : L }$ initialized to $\tilde { \mathbf { x } } ^ { L - L ^ { \prime } : L }$ as opposed to being initialized as masked tokens m. At the end of this process, we have produced $L { + } L ^ { \prime }$ tokens concat $\left[ \tilde { \mathbf { x } } ^ { 1 : L } , \tilde { \mathbf { x } } ^ { L + 1 : L + L ^ { \prime } } \right]$ , where concat[·] denotes concatenation along the sequence length dimension. This process can repeat indefinitely, with the prefix shifted for every new round of generation.

## 5 Experiments

## 5.1 Masked Diffusion Language Models

Experimental Setup We evaluate MDLM as a generative model of language and as a representation model via fine-tuning on downstream tasks.

For language modeling likelihood evaluation, we conduct experiments on two datasets: The One Billion Words Dataset (LM1B; [8]) and OpenWebText (OWT; [18]). We use the bert-base-uncased tokenizer for LM1B, and report perplexities on the test split. Models have a context size of 128. For OWT, which does not have a pre-defined split, we reserve the last 100K documents as a held-out validation set and report perplexities on this set. We use the GPT2 tokenizer [45] for OWT. Models have a context size of 1,024. We utilize the transformer architecture from Lou et al. [33], which augments the diffusion transformer [42] with rotary embeddings [58]. MDLM was trained for 1M or 10M steps (corresponding to 33B, 327B tokens, respectively) on LM1B and 1M steps on OWT (which corresponds to 262B tokens). The corresponding AR baseline was trained for half the number of steps to ensure similar number of tokens seen (details in Suppl. D.2). Full hyperparameters are given in Suppl. D.4. On OWT, we train with and without time step conditioning.

Table 1: Test perplexities (PPL; ↓) on LM1B. †Reported in He et al. [26]. Best diffusion value is bolded.
<table><tr><td colspan="2"></td><td>Parameters</td><td>PPL (↓)</td></tr><tr><td rowspan="2">Autoregressive</td><td>Transformer-X Base [13]</td><td>0.46B</td><td>23.5</td></tr><tr><td>OmniNetr [61]</td><td>100M</td><td>21.5</td></tr><tr><td rowspan="5">Diffusion</td><td>BERT-Mouth [64]t</td><td>110M</td><td>≤142.89</td></tr><tr><td>D3PM(absorb)[1]</td><td>70M</td><td>≤76.90</td></tr><tr><td>Diffusion-LM[30]t</td><td>80M</td><td>≤118.62</td></tr><tr><td>DiffusionBert [26]</td><td>110M</td><td>≤63.78</td></tr><tr><td>SEDD [33](33B tokens)</td><td>110M</td><td>≤ 32.79</td></tr><tr><td rowspan="2">Autoregressive (Retrained)</td><td>Transformer (33B tokens)</td><td>110M</td><td>22.32</td></tr><tr><td>Transformer (327B tokens)</td><td></td><td>20.86</td></tr><tr><td rowspan="2">Diffusion (Ours)</td><td>MDLM (33B tokens)</td><td>110M</td><td>≤27.04</td></tr><tr><td>MDLM (327B tokens)</td><td></td><td>≤23.00</td></tr></table>

For representation learning, we pre-train models on the C4 dataset [46], then fine-tune and evaluate models on the GLUE benchmark [65]. Models have a context size of 128. We use the bert-base-uncased tokenizer for the representation learning experiments. We utilize the MosaicBERT architecture from Portes et al. [43], an extension of the original BERT architecture [15]. We pre-train a bidirectional MosaicBERT using an MLM objective for 37B tokens of C4, as well as a causal variant on the same data. We further fine-tune MosaicBERT model using the MDLM for 327M tokens, less than 1% of the pre-training data. We provide the full hyperparameters in Suppl. D.6.

Likelihood Evaluation On LM1B, MDLM outperforms all previous diffusion methods (Table 1). Compared to the SEDD baseline reported by Lou et al. [33], trained for 33B tokens, MDLM, which we train for the same amount, achieves a 17% improvement on the perplexity bound. Finally, MDLM gets within 14% of an AR baseline and continues to improve with more training. We see the same trend for models trained on OWT, a larger dataset, shown in Table 2 – MDLM outperforms prior diffusion methods, closing the gap towards AR models. In Table 12 we find that models trained with and without time conditioning attain similar perplexities on OWT. Additionally, Figure 3 demonstrates the reduced variance we achieve from our objective, when compared to previous masked diffusion models such as SEDD [33].

Zero-Shot Likelihood Evaluation We also explore models’ ability to generalize by taking models trained on OWT and evaluating how well they model unseen datasets. We compare the perplexities of our MDLM with SEDD [1] and an AR Transformer language model. Our zero-shot datasets include the validation splits of Penn Tree Bank (PTB; [36]), Wikitext [38], LM1B, Lambada [41], AG News [68], and Scientific Papers (Pubmed and Arxiv subsets; [10]). Full experimental details are available in Suppl. D.4.

Table 2: Test perplexities (PPL; ↓) on OWT for models trained for 262B tokens. † denotes retrained models.

MDLM consistently outperforms the SEDD diffusion parameterization. In some cases, e.g., for Lambada and Scientific Papers, MDLM attains better perplexity than AR. We hypothesize that these datasets are farther from OWT, and that diffusion models may be more robust to out-of-domain evaluation due to the unmasking-based objective.

<table><tr><td></td><td>PPL (↓)</td></tr><tr><td>ARt</td><td>17.54</td></tr><tr><td>SEDD†</td><td>≤24.10</td></tr><tr><td>MDLM (Ours)</td><td>≤23.21</td></tr></table>

Downstream Task Evaluation We find that BERT fine-tuned with MDLM to be a generative model results in strong perplexities while preserving performance on downstream tasks. On the C4 validation set, the AR model attains perplexity (PPL) of 22, the pre-trained BERT attains a PPL upper bound of 78 (evaluated using the MDLM variational bound), and BERT + MDLM-FT attains a PPL upper bound of 35. In Table 4, we further find that BERT + MDLM fine-tuning has no degradation in downstream

Table 3: Zero-shot perplexities (↓) of models trained for 524B tokens on OWT. All perplexities for diffusion models are upper bounds.
<table><tr><td></td><td>PTB</td><td>Wikitext</td><td>LM1B</td><td>Lambada</td><td>AGNews</td><td>Pubmed</td><td>Arxiv</td></tr><tr><td>AR (Retrained)</td><td>82.05</td><td>25.75</td><td>51.25</td><td>51.28</td><td>52.09</td><td>49.01</td><td>41.73</td></tr><tr><td>SEDD (Retrained)</td><td>100.09</td><td>34.28</td><td>68.20</td><td>49.86</td><td>62.09</td><td>44.53</td><td>38.48</td></tr><tr><td>MDLM(Ours)</td><td>95.26</td><td>32.83</td><td>67.01</td><td>47.52</td><td>61.15</td><td>41.89</td><td>37.37</td></tr></table>

Table 4: GLUE evaluation results. Evaluation measures (↑) are F1 score for QQP and MRPC, Spearman correlations for STS-B, and accuracy for the rest. For MNLI, we report match/mismatch accuracies.
<table><tr><td colspan="10">MNLI</td></tr><tr><td></td><td>(m/mm)</td><td>QQP</td><td>QNLI</td><td>SST-2</td><td>COLA</td><td>STS-B</td><td>MRPC</td><td>RTE</td><td>Avg</td></tr><tr><td>AR</td><td>80.94/80.78</td><td>86.98</td><td>86.16</td><td>90.14</td><td>33.43</td><td>84.32</td><td>83.88</td><td>47.29</td><td>74.88</td></tr><tr><td>BERT</td><td>84.43/85.35</td><td>88.41</td><td>90.46</td><td>92.20</td><td>54.81</td><td>88.41</td><td>89.16</td><td>61.37</td><td>81.62</td></tr><tr><td>+MDLM-FT</td><td>84.76/85.07</td><td>88.49</td><td>90.30</td><td>92.20</td><td>57.69</td><td>87.48</td><td>90.53</td><td>62.09</td><td>82.06</td></tr></table>

GLUE performance compared to the BERT initialization. While the perplexity of our method is higher than the AR baseline, the downstream task performance is significantly better.

Semi-Autoregressive Modeling To test the SAR decoding algorithm presented in Sec. 4.2, we compare to SSD-LM [24] a diffusion model that was designed to generate blocks of text autoregressively. We generate 200 sequences of length 2048 tokens on a single 3090 GPU and evaluate generative perplexity under a pre-trained GPT-2 [45] model. The SSD-LM sequences are generated using blocks of

Table 5: Semi-AR generative perplexity (Gen. PPL; ↓) for sequences of 2048 tokens.
<table><tr><td></td><td>Gen. PPL (↓)</td><td>Sec/Seq (↓)</td></tr><tr><td>SSD-LM</td><td>35.43</td><td>2473.9</td></tr><tr><td>MDLM(Ours)</td><td>27.18</td><td>89.3</td></tr></table>

25 tokens (as implemented in their pre-trained model) and the MDLM sequences are generated using L′ = 512. In Table 5, we find that in addition to achieving better generative perplexity, MDLM enables ∼25-30x faster SAR decoding relative to SSD-LM.

## 5.2 Masked Diffusion DNA Models

We also explore applications to the generative modeling of biological sequences [14, 47] using a state space model (SSM) backbone [22]. Namely, we build on the recently-proposed Caduceus DNA language model [50], which uses as a backbone the data-dependent SSM Mamba block [21].

Experimental Setup We pre-train the encoder-only Caduceus [50], which is an MLM, on the HG38 human reference genome [11] and perform fine-tuning using our diffusion parameterization. We use a context length of 1024 tokens and follow Schiff et al. [50] for the experimental setup, other than learning rate which was reduced to 1e-3. See Suppl. D.7 for full experimental details. We assess both generative performance using perplexity and downstream performance on Genomics Benchmarks [20] across language diffusion paradigms and AR models.

Generative Performance We fine-tune the Caduceus MLM across diffusion parameterizations and compare perplexities against AR models. We report perplexity values in Table 6. MDLM outperforms all other diffusion language modeling schemes.

Downstream Task Fine-tuning We perform downstream evaluation with the Genomics Benchmarks [20], a recently proposed benchmark with eight regulatory element classification tasks. As shown in Table 7, our generative fine-tuning paradigm preserves or improves upon downstream performance from MLM pre-training. Absorbing-state diffusion methods outperform Plaid across tasks except for the simplest task Human vs. Worm, where all methods have roughly the same performance. For tasks where the input is a biased subsample of the full genome, we observe that the correlation between perplexity and downstream performance is weaker; see Suppl. D.7.

Table 6: Test perplexities $( \mathrm { P P L } ; \downarrow )$ of generative fine-tuning of the Caduceus MLM [50] on the HG38 reference genome. Best diffusion model values are bolded. Error bars indicate the difference between the maximum and minimum values across 5 random seeds used for fine-tuning.
<table><tr><td colspan="2"></td><td>Params</td><td>PPL (↓)</td></tr><tr><td rowspan="2">Autoregressive (Retrained)</td><td>Mamba</td><td>465K</td><td> $3 . 0 6 7 \pm . 0 1 0$ </td></tr><tr><td>HyenaDNA</td><td>433K</td><td> $3 . 1 5 3 \pm . 0 0 1$ </td></tr><tr><td rowspan="2">Diffusion (Retrained)</td><td>Plaid</td><td>507K</td><td> $\leq 3 . 2 4 0 \pm . 0 0 5$ </td></tr><tr><td>SEDD</td><td>467K</td><td> $\leq 3 . 2 1 6 \pm . 0 0 3$ </td></tr><tr><td>Diffusion (Ours)</td><td>MDLM</td><td>467K</td><td> $\leq 3 . 1 9 9 \pm . 0 1 0$ </td></tr></table>

Table 7: Genomic Benchmarks. Top-1 accuracy (↑) across 5-fold cross-validation (CV) for a pre-trained AR Mamba, and a pre-trained Caduceus model fine-tuned with different diffusion parameterizations. The best values per task are bolded and the second best are italicized. Error bars indicate the difference between the maximum and minimum values across 5 random seeds used for CV.
<table><tr><td>Model Fine-Tuning Objective (Parameter Count)</td><td>Mamba AR (465K)</td><td>Caduceus MLM (467K)</td><td>Caduceus Plaid (507k)</td><td>Caduceus SEDD (467k)</td><td>Caduceus MDLM(ours) (467k)</td></tr><tr><td>Mouse Enhancers</td><td>0.763{±0.008}</td><td>0.810 {±0.016}</td><td>0.745 {±0.079}</td><td>0.784{±0.058}</td><td>0.795 {±0.029}</td></tr><tr><td>Coding Vs.Intergenomic</td><td>0.897 {±0.004}</td><td>0.913{±0.003}</td><td>0.908 {±0.003}</td><td>0.913{±0.005}</td><td>0.913{±0.003}</td></tr><tr><td>Human vs.Worm</td><td>0.967 {±0.002}</td><td>0.970 {±0.002}</td><td>0.971 {±0.001}</td><td>0.970 {±0.003}</td><td>0.970 {±0.003}</td></tr><tr><td>Human Enhancers Cohn</td><td>0.734{±0.027}</td><td>0.737{±0.001}</td><td>0.743{±0.010}</td><td>0.746{±0.015}</td><td>0.743 {±0.016}</td></tr><tr><td>Human Enhancer Ensembl</td><td>0.856 {±0.003}</td><td>0.907 {±0.000}</td><td>0.885 {±0.003}</td><td>0.905 {±0.006}</td><td>0.899 {±0.004}</td></tr><tr><td>Human Regulatory</td><td>0.861{±0.008}</td><td>0.874{±0.003}</td><td>0.868{±0.010}</td><td>0.828 {±0.037}</td><td>0.868 {±0.004}</td></tr><tr><td>Human OCR Ensembl</td><td>0.806{±0.005}</td><td>0.821 {±0.000}</td><td>0.820 {±0.004}</td><td>0.816 {±0.008}</td><td>0.823{±0.008}</td></tr><tr><td>Human NonTATA Promoters</td><td>0.926 {±0.008}</td><td>0.935 {±0.014}</td><td>0.935 {±01007}</td><td>0.935 {±0.014}</td><td>0.940{±0.007}</td></tr></table>

## 5.3 Ablation Analysis

In Table 8, we can see the effect of our streamlined masked diffusion implementation. The improvements described in Sec. 3.5.1 allow us to greatly reduce perplexity of previously discounted models, such as D3PM (see the bottom row of this table, which is mathematically equivalent to the D3PM formulation). While most works assumed that D3PM achieves mediocre log-likelihoods, we show that is incorrect: our re-implementation almost matches state-of-the-art score-based methods. This introduces a new strong baseline that opens new research opportunities. Additionally, in Table 8, we ablate

Table 8: Test perplexities $( \mathrm { { P P L } ; \downarrow ) }$ for MDLM ablations on LM1B. For the discrete-time models, we use $T = 1 0 0 0$ . Standard deviation is measured over 5 seeds during evaluation.
<table><tr><td></td><td>PPL(≤)</td></tr><tr><td>MDLM (47)</td><td>27.04±.01</td></tr><tr><td>w/o continuous time (43)</td><td> $2 7 . 1 9 \pm . 0 7$ </td></tr><tr><td>&amp; w/o carry-over (41)</td><td> $2 8 . 5 6 \pm . 1 5$ </td></tr><tr><td>&amp; w/o zero masking (39)</td><td> $2 8 . 5 1 \pm . 1 5$ </td></tr></table>

different components of MDLM. We observe that the perplexity for MDLM trained with a discrete T = 1000 marginally worsens by 0.1 compared to MDLM trained in continuous time. Additionally, removing the “carry over” operation from the SUBS parameterization increases the perplexity by 1.5 points. However, further removing the “zero masking” operation does not lead to any meaningful change in perplexity. We provide further ablations for the continuous time formulation in the Appendix, showing in Table 11 that for a pre-trained model, at inference, increasing T yields better likelihoods.

## 6 Related Work

Comparison to D3PM Masked diffusion is a strict subset of D3PM [1]; setting $Q _ { t | s } = \alpha _ { t | s } { \bf I } + ( 1 -$ $\alpha _ { t | s } ) \mathbf { 1 m } ^ { \top }$ in their framework yields our forward diffusion. We improve over D3PM in three ways: (1) we adopt the SUBS parameterization; (2) this allows us to derive a simplified objective that analytically simplifies certain expectations to zero; (3) we adopt well-engineered training recipes that improve performance. Both (1) and (2) are possible because we focus on masking instead of developing a general discrete diffusion framework. Surprisingly, (3) has the largest contribution to performance.

Comparison to CTMC Most implementations of diffusion work best in continuous time. However, extending D3PM in this way requires computing the limit of the product of an infinite number of matrices $Q _ { T } { \cdot } Q _ { T - 1 } { \cdots } Q _ { t }$ as $T \to \infty$ , which requires advanced CTMC theory [5]. Our work describes simple continuous-time formulations for the most common noise processes (e.g., masking and uniform π), thus helping make an important part of the literature more accessible. In Suppl. C, we show that our results are compatible with CTMC, using the rate forward matrix $\begin{array} { r } { R _ { t } = \frac { \alpha _ { t } ^ { \prime } } { \alpha _ { t } } ( \mathbf { I } - \mathbf { 1 m } ^ { \top } ) } \end{array}$ and the reverse rate $\tilde { R } _ { t } ( \mathbf { y } ^ { \prime } , \mathbf { y } )$ for the transition $\mathbf { y } \to \mathbf { y } ^ { \prime }$ , where $\mathbf { y } , \mathbf { y } ^ { \prime } \in \mathcal { V } ;$

$$
\tilde { R } _ { t } ( \mathbf { y } ^ { \prime } , \mathbf { y } ) = - \frac { \alpha _ { t } ^ { \prime } } { 1 - \alpha _ { t } } [ \mathbf { y } ^ { \prime } ] ^ { \top } [ \mathbf { x } _ { \theta } ( \mathbf { y } , t ) - \mathbf { m } ] \langle \mathbf { y } , \mathbf { m } \rangle\tag{12}
$$

Comparison to Score Estimation Score-based approaches to diffusion [55] extend to discrete states, although they typically further build upon advanced CTMC theory. In particular, SEDD [33] optimizes an ELBO3 that is a function of the score model, obtaining state-of-the-art log-likelihoods among diffusion models. Our approach, however, is much simpler and does not require advanced theory. Furthermore, we can extract the score for MDLM (76), as demonstrated in Suppl. C.3, making it compatible with various techniques designed for score-based algorithms, such as samplers [5], score parameterization [33], efficient designs of the denoising network [59], guidance techniques, and more.

Comparison to BERT Our work provides a principled way of making BERT generative when trained with randomized masking rates. Previous work on generating from BERT used Gibbs sampling or ad-hoc methods [17, 32, 64]. The connection between BERT and diffusion was first made by Austin et al. [1]: their objective effectively involves unmasking. He et al. [26] additionally starts training from a pretrained BERT. However, both works use an objective that is similar to (9), which is less numerically stable than our objective (see Section 3.5.1). Austin et al. [1] mention in their appendix that their ELBO simplifies to a weighted masking (MLM) loss similar to (8), but it uses a more complex formula for the weights and is limited to the discrete time setting unlike our work. Furthermore, they do not train with that objective. Our work derives a simpler expression for the average of MLM losses, implements it, and obtains better likelihoods.

Comparision to Latent Diffusion LMs In contrast to this work, which defines diffusion over discrete structures, Plaid [23] and Diffusion LM [30] define a Gaussian diffusion process over word embeddings. Zhang et al. [67] and Hu et al. [28] extend this approach to flow matching over word embeddings, enabling the design of faster samplers. Discrete Flow Matching (DFM) [6] applies flow matching to discrete structures, using a cross-entropy loss as their training objective: $- \mathbb { E } _ { q , t } ^ { \mathbf { \phi } } \mathrm { l o g } p _ { \theta } ( \mathbf { x } ^ { 1 : L } | \mathbf { z } _ { t } ^ { 1 : L } )$ Similar to Chang et al. [7], DFM’s objective, while effective, is not weighted to serve as a proper ELBO. In MDLM, however, we derive a tight, principled lower bound on the log-likelihood.

Concurrent Works Concurrent to our work, Shi et al. [51] and Ou et al. [40] derive a similar simplified objective for masked diffusion processes. While Ou et al. [40] start from a score matching perspective, we tackle this problem from a variational lens similar to Shi et al. [51]. Similar to Ou et al. [40], we formulate efficient samplers in Section 4.1 by leveraging a time-independent denoising network.

A key differentiation between our work and that of Shi et al. [51], Ou et al. [40] is the semi-autoregressive decoding method we present in Section 4.2. While [51, 40] are restricted to sample sequences of a fixed length, we propose samplers to generate arbitrary lengths of text like a traditional language model. Furthermore, we establish the connection between our simplified objective and the masked language modeling (MLM) objective. As a result, we endow BERT-style models with principled generation capabilities while maintaining representation learning capabilities. Whereas [51, 40] only evaluate on NLP datasets, we show that masked diffusion is also effective in modeling biological sequences.

## 7 Conclusion

In this work, we explore masked diffusion. With a well-engineered implementation that supports a simple variational objective, we attain state-of-the-art diffusion perplexities on language benchmarks and demonstrate how to efficiently convert BERT-style encoders into generative models. Given we are working on language modeling, we carry any of the inherent risks and opportunities that come with this line of research.

## Acknowledgments and Disclosure of Funding

This work was partially funded by the National Science Foundation under awards DGE-1922551, CAREER awards 2046760 and 2145577, and by the National Institute of Health under award MIRA R35GM151243. Marianne Arriola is supported by a NSF Graduate Research Fellowship under award DGE-2139899 and a Hopper-Dean/Bowers CIS Deans Excellence Fellowship.

## References

[1] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in Neural Information Processing Systems, 34:17981–17993, 2021.

[2] Pavel Avdeyev, Chenlai Shi, Yuhao Tan, Kseniia Dudnyk, and Jian Zhou. Dirichlet diffusion score model for biological sequence generation. In International Conference on Machine Learning, pp. 1276–1301. PMLR, 2023.

[3] Žiga Avsec, Vikram Agarwal, Daniel Visentin, Joseph R Ledsam, Agnieszka Grabska-Barwinska, Kyle R Taylor, Yannis Assael, John Jumper, Pushmeet Kohli, and David R Kelley. Effective gene expression prediction from sequence by integrating long-range interactions. Nature methods, 18 (10):1196–1203, 2021.

[4] Joe Benton, Yuyang Shi, Valentin De Bortoli, George Deligiannidis, and Arnaud Doucet. From denoising diffusions to denoising markov models. arXiv preprint arXiv:2211.03595, 2022.

[5] Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. Advances in Neural Information Processing Systems, 35:28266–28279, 2022.

[6] Andrew Campbell, Jason Yim, Regina Barzilay, Tom Rainforth, and Tommi Jaakkola. Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design. arXiv preprint arXiv:2402.04997, 2024.

[7] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11315–11325, 2022.

[8] Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony Robinson. One billion word benchmark for measuring progress in statistical language modeling, 2014.

[9] Ting Chen, Ruixiang Zhang, and Geoffrey Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. arXiv preprint arXiv:2208.04202, 2022.

[10] Arman Cohan, Franck Dernoncourt, Doo Soon Kim, Trung Bui, Seokhwan Kim, Walter Chang, and Nazli Goharian. A discourse-aware attention model for abstractive summarization of long documents. Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), 2018. doi: 10.18653/v1/n18-2097. URL http://dx.doi.org/10.18653/v1/n18-2097.

[11] Genome Reference Consortium. Genome reference consortium human build 37 (grch37. Database (GenBank or RefSeq), 2009.

[12] Genome Reference Consortium et al. Genome reference consortium human build 37 (grch37). Database (GenBank or RefSeq), 2009.

[13] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860, 2019.

[14] Shachi Deshpande, Kaiwen Wang, Dhruv Sreenivas, Zheng Li, and Volodymyr Kuleshov. Deep multi-modal structural equations for causal effect estimation with unstructured proxies. Advances in Neural Information Processing Systems, 35:10931–10944, 2022.

[15] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

[16] Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, et al. Continuous diffusion for categorical data. arXiv preprint arXiv:2211.15089, 2022.

[17] Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer. Mask-predict: Parallel decoding of conditional masked language models. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 6112–6121, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1633. URL https://aclanthology.org/D19-1633.

[18] Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. Openwebtext corpus. http: //Skylion007.github.io/OpenWebTextCorpus, 2019.

[19] Aaron Gokaslan, A Feder Cooper, Jasmine Collins, Landan Seguin, Austin Jacobson, Mihir Patel, Jonathan Frankle, Cory Stephenson, and Volodymyr Kuleshov. Commoncanvas: Open diffusion models trained on creative-commons images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8250–8260, 2024.

[20] Katarína Grešová, Vlastimil Martinek, David Cechák, Petr Šime ˇ cek, and Panagiotis Alexiou. ˇ Genomic benchmarks: a collection of datasets for genomic sequence classification. BMC Genomic Data, 24(1):25, 2023.

[21] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

[22] Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021.

[23] Ishaan Gulrajani and Tatsunori B Hashimoto. Likelihood-based diffusion language models. Advances in Neural Information Processing Systems, 36, 2024.

[24] Xiaochuang Han, Sachin Kumar, and Yulia Tsvetkov. Ssd-lm: Semi-autoregressive simplexbased diffusion language model for text generation and modular control. arXiv preprint arXiv:2210.17432, 2022.

[25] Floyd B. Hanson. Applied stochastic processes and control for jump-diffusions - modeling, analysis, and computation. In Advances in design and control, 2007. URL https://api. semanticscholar.org/CorpusID:6689808.

[26] Zhengfu He, Tianxiang Sun, Kuanning Wang, Xuanjing Huang, and Xipeng Qiu. Diffusionbert: Improving generative masked language models with diffusion models. arXiv preprint arXiv:2211.15029, 2022.

[27] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[28] Vincent Hu, Di Wu, Yuki Asano, Pascal Mettes, Basura Fernando, Björn Ommer, and Cees Snoek. Flow matching for conditional text generation in a few sampling steps. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 380–392, 2024.

[29] Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. Advances in neural information processing systems, 34:21696–21707, 2021.

[30] Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. Diffusionlm improves controllable text generation. Advances in Neural Information Processing Systems, 35:4328–4343, 2022.

[31] Xuanlin Li, Brandon Trabucco, Dong Huk Park, Michael Luo, Sheng Shen, Trevor Darrell, and Yang Gao. Discovering non-monotonic autoregressive orderings with variational inference. arXiv preprint arXiv:2110.15797, 2021.

[32] Yi Liao, Xin Jiang, and Qun Liu. Probabilistically masked language model capable of autoregressive generation in arbitrary word order. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (eds.), Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 263–274, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.24. URL https://aclanthology.org/2020.acl-main.24.

[33] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion language modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023.

[34] Justin Lovelace, Varsha Kishore, Chao Wan, Eliot Shekhtman, and Kilian Q Weinberger. Latent diffusion for language generation. Advances in Neural Information Processing Systems, 36, 2024.

[35] Vincent Mallet and Jean-Philippe Vert. Reverse-complement equivariant networks for dna sequences. Advances in neural information processing systems, 34:13511–13523, 2021.

[36] Mitch Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. Building a large annotated corpus of english: The penn treebank. Computational linguistics, 19(2):313–330, 1993.

[37] Chenlin Meng, Kristy Choi, Jiaming Song, and Stefano Ermon. Concrete score matching: Generalized score matching for discrete data. Advances in Neural Information Processing Systems, 35:34532–34545, 2022.

[38] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models, 2016.

[39] Eric Nguyen, Michael Poli, Marjan Faizi, Armin Thomas, Michael Wornow, Callum Birch-Sykes, Stefano Massaroli, Aman Patel, Clayton Rabideau, Yoshua Bengio, et al. Hyenadna: Long-range genomic sequence modeling at single nucleotide resolution. Advances in neural information processing systems, 36, 2024.

[40] Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data, 2024.

[41] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernandez. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1525–1534, Berlin, Germany, August 2016. Association for Computational Linguistics. URL http://www.aclweb.org/anthology/P16-1144.

[42] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

[43] Jacob Portes, Alex Trott, Sam Havens, Daniel King, Abhinav Venigalla, Moin Nadeem, Nikhil Sardana, Daya Khudia, and Jonathan Frankle. Mosaicbert: A bidirectional encoder optimized for fast pretraining, 2024.

[44] Ofir Press, Noah A. Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation, 2022.

[45] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.

[46] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21(1), jan 2020. ISSN 1532-4435.

[47] Richa Rastogi and Yair Schiff. Semi parametric inducing point networks and neural processes. In International Conference on Learning Representations, 2023.

[48] Subham Sekhar Sahoo, Aaron Gokaslan, Chris De Sa, and Volodymyr Kuleshov. Diffusion models with learned adaptive noise. arXiv preprint arXiv:2312.13236, 2023.

[49] Subham Sekhar Sahoo, Anselm Paulus, Marin Vlastelica, Vít Musil, Volodymyr Kuleshov, and Georg Martius. Backpropagation through combinatorial algorithms: Identity with projection works. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=JZMR727O29.

[50] Yair Schiff, Chia-Hsiang Kao, Aaron Gokaslan, Tri Dao, Albert Gu, and Volodymyr Kuleshov. Caduceus: Bi-directional equivariant long-range dna sequence modeling. arXiv preprint arXiv:2403.03234, 2024.

[51] Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K Titsias. Simplified and generalized masked diffusion for discrete data. Advances in neural information processing systems, 36, 2024.

[52] Phillip Si, Allan Bishop, and Volodymyr Kuleshov. Autoregressive quantile flows for predictive uncertainty estimation. In International Conference on Learning Representations.

[53] Phillip Si, Zeyi Chen, Subham Sekhar Sahoo, Yair Schiff, and Volodymyr Kuleshov. Semiautoregressive energy flows: exploring likelihood-free training of normalizing flows. In International Conference on Machine Learning, pp. 31732–31753. PMLR, 2023.

[54] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256–2265. PMLR, 2015.

[55] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.

[56] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

[57] Robin Strudel, Corentin Tallec, Florent Altché, Yilun Du, Yaroslav Ganin, Arthur Mensch, Will Grathwohl, Nikolay Savinov, Sander Dieleman, Laurent Sifre, et al. Self-conditioned embedding diffusion for text generation. arXiv preprint arXiv:2211.04236, 2022.

[58] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864, 2021.

[59] Haoran Sun, Lijun Yu, Bo Dai, Dale Schuurmans, and Hanjun Dai. Score-based continuous-time discrete diffusion models. arXiv preprint arXiv:2211.16750, 2022.

[60] Zhiqing Sun and Yiming Yang. Difusco: Graph-based diffusion solvers for combinatorial optimization. Advances in Neural Information Processing Systems, 36:3706–3731, 2023.

[61] Yi Tay, Mostafa Dehghani, Vamsi Aribandi, Jai Gupta, Philip M Pham, Zhen Qin, Dara Bahri, Da-Cheng Juan, and Donald Metzler. Omninet: Omnidirectional representations from transformers. In International Conference on Machine Learning, pp. 10193–10202. PMLR, 2021.

[62] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[63] Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard. Digress: Discrete denoising diffusion for graph generation. arXiv preprint arXiv:2209.14734, 2022.

[64] Alex Wang and Kyunghyun Cho. Bert has a mouth, and it must speak: Bert as a markov random field language model. arXiv preprint arXiv:1902.04094, 2019.

[65] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In International Conference on Learning Representations, 2019. URL https://openreview. net/forum?id=rJ4km2R5t7.

[66] Yingheng Wang, Yair Schiff, Aaron Gokaslan, Weishen Pan, Fei Wang, Christopher De Sa, and Volodymyr Kuleshov. Infodiffusion: Representation learning using information maximizing diffusion models. In International Conference on Machine Learning, pp. 36336–36354. PMLR, 2023.

[67] Shujian Zhang, Lemeng Wu, Chengyue Gong, and Xingchao Liu. Language rectified flow: Advancing diffusion language generation with probabilistic flows. arXiv preprint arXiv:2403.16995, 2024.

[68] Xiang Zhang, Junbo Jake Zhao, and Yann LeCun. Character-level convolutional networks for text classification. In NIPS, 2015.

[69] Lin Zheng, Jianbo Yuan, Lei Yu, and Lingpeng Kong. A reparameterized discrete diffusion model for text generation. arXiv preprint arXiv:2302.05737, 2023.

[70] Hannah Zhou, Avanti Shrikumar, and Anshul Kundaje. Towards a better understanding of reverse-complement equivariance for deep learning models in genomics. In Machine Learning in Computational Biology, pp. 1–33. PMLR, 2022.

## Contents

1 Introduction 1   
2 Background 2   
2.1 Diffusion Models 2   
2.2 Discrete Diffusion Models 3   
3 Simple Masked Diffusion Models 3   
3.1 Interpolating Discrete Diffusion 3   
3.2 Masked Diffusion 4   
3.3 Rao-Blackwellized Likelihood Bounds 4   
3.4 Continuous-Time Likelihood Bounds . 5   
3.5 Masked Diffusion Language Models 5   
4 Inference and Sampling in Masked Diffusion Language Models 6   
4.1 Efficient Ancestral Sampling 6   
4.2 Semi-Autoregressive Masked Diffusion Language Models . 6   
5 Experiments 6   
5.1 Masked Diffusion Language Models 6   
5.2 Masked Diffusion DNA Models . 8   
5.3 Ablation Analysis 9   
6 Related Work 9   
Conclusion 10   
Appendices 17   
Appendix A Discrete time ELBO 17   
A.1 Generic case 17   
A.2 Absorbing state 18   
Appendix B MDLM 21   
B.1 Rao-Blackwellization 22   
B.2 Continuous Time 22   
B.3 Final Algorithm 23   
Appendix C Concrete Score Matching 23   
C.1 Extracting the Rate Matrix 24   
C.2 NELBO 25   
C.3 Concrete Score for MDLM 27   
C.4 Reverse Rate Matrix for MDLM 28   
C.5 Deriving MDLM’s NELBO via CTMC 29   
Appendix D Experimental details 31   
D.1 Likelihood Evaluation . . 31   
D.2 Avg. Number of Tokens seen 31   
D.3 Low discrepancy sampler 31   
D.4 Language Modeling . . 31   
D.5 Zeroshot Likelihood . 32   
D.6 Representation Learning 32   
D.7 Diffusion DNA Models 32   
Appendix E Additional Experiments 33   
E.1 Noise schedule parameterization 33   
E.2 Faster sampling with caching 34   
E.3 LM1B ablations . . 35   
E.4 Train NLL curves on OWT 35   
E.5 Time-conditioning ablation on OWT 36   
E.6 Unconditional Samples 36

## Appendices

## Appendix A Discrete time ELBO

This section is organized as follows: First, we derive the expressions for the true posterior and the approximate posterior as outlined in Suppl. A.1. We then simplify these expressions specifically for the case of absorbing state diffusion in Suppl. A.2. Finally, we derive the expression for the ELBO for absorbing state diffusion in Suppl. A.2.3.

## A.1 Generic case

Given the state transition matrix $Q _ { t }$ , prior $\pi ,$ , and the latent variables $\mathbf { z } _ { s }$ and $\mathbf { z } _ { t } ,$ where $s < t ,$ let

$$
\begin{array} { r } { Q _ { t | s } = \alpha _ { t | s } \mathbf { I } + ( 1 - \alpha _ { t | s } ) \mathbf { 1 } \pi ^ { \top } . } \end{array}\tag{13}
$$

$$
\mathbf { A . 1 . 1 } \quad q ( \mathbf { z } _ { t } | \mathbf { z } _ { s } )
$$

Thus, the marginals in (3) correspond to the following forward process:

$$
\begin{array} { r l r l } & { q ( \mathbf { z } _ { t } | \mathbf { z } _ { s } ) = \mathrm { C a t } ( \mathbf { z } _ { t } ; Q _ { t | s } ^ { \top } \mathbf { z } _ { s } ) } \\ & { ~ } & { ~ = \mathrm { C a t } ( \mathbf { z } _ { t } ; [ \alpha _ { t | s } \mathbf { I } + ( 1 - \alpha _ { t | s } ) \mathbf { 1 } \pi ^ { \top } ] ^ { \top } \mathbf { z } _ { s } ) } \\ & { ~ } & { ~ = \mathrm { C a t } ( \mathbf { z } _ { t } ; \alpha _ { t | s } \mathbf { z } _ { s } + ( 1 - \alpha _ { t | s } ) \pi \mathbf { 1 } ^ { \top } \mathbf { z } _ { s } ) } \\ & { ~ } & { ~ = \mathrm { C a t } ( \mathbf { z } _ { t } ; \alpha _ { t | s } \mathbf { z } _ { s } + ( 1 - \alpha _ { t | s } ) \pi ) . } \end{array}\tag{14}
$$

The above equation indicates that during each diffusion step from $s \to t ,$ a fraction $\left( 1 - \alpha _ { t | s } \right)$ of the probability mass is transferred to the prior distribution π.

A.1.2 $q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } )$

Austin et al. [1] show that the posterior corresponding to (14) is given as follows:

$$
q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) { = } \mathbf { C a t } \left( \mathbf { z } _ { s } ; \frac { Q _ { t | s } \mathbf { z } _ { t } { \odot } Q _ { s } ^ { \top } \mathbf { x } } { \mathbf { z } _ { t } ^ { \top } Q _ { t } ^ { \top } \mathbf { x } } \right) ,\tag{15}
$$

which we simplify to the following:

$$
\begin{array} { r l r l } & { q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) } \\ & { = \mathrm { C a t } \Bigg ( \mathbf { z } _ { s } ; \frac { [ \boldsymbol { \alpha } _ { t | s } \mathbf { I } + ( 1 - \alpha _ { t | s } ) \mathbf { 1 } \pi ^ { \top } ] \mathbf { z } _ { t } \odot [ \boldsymbol { \alpha } _ { s } \mathbf { I } + ( 1 - \alpha _ { s } ) \mathbf { 1 } \pi ^ { \top } ] ^ { \top } \mathbf { x } } { \mathbf { z } _ { t } ^ { \top } [ \boldsymbol { \alpha } _ { t } \mathbf { I } + ( 1 - \alpha _ { t } ) \mathbf { 1 } \pi ^ { \top } ] ^ { \top } \mathbf { x } } \Bigg ) } & & { } \\ & { = \mathrm { C a t } \Bigg ( \mathbf { z } _ { s } ; \frac { [ \boldsymbol { \alpha } _ { t | s } \mathbf { z } _ { t } + ( 1 - \alpha _ { t | s } ) \mathbf { 1 } \pi ^ { \top } \mathbf { z } _ { t } ] \odot [ \boldsymbol { \alpha } _ { s } \mathbf { x } + ( 1 - \alpha _ { s } ) \boldsymbol { \pi } ] } { \mathbf { z } _ { t } ^ { \top } [ \boldsymbol { \alpha } _ { t } \mathbf { x } + ( 1 - \alpha _ { t } ) \boldsymbol { \pi } ] ^ { \top } \mathbf { x } ] } \Bigg ) } & & { } \\ & { = \mathrm { C a t } \Bigg ( \mathbf { z } _ { s } ; \frac { [ \boldsymbol { \alpha } _ { t | s } \mathbf { z } _ { t } + ( 1 - \alpha _ { t | s } ) \mathbf { 1 } \pi ^ { \top } \mathbf { z } _ { t } ] ( \odot [ \boldsymbol { \alpha } _ { s } \mathbf { x } + ( 1 - \alpha _ { s } ) \boldsymbol { \pi } ] ) } { \mathbf { z } _ { t } ^ { \top } [ \alpha _ { t } \mathbf { x } + ( 1 - \alpha _ { t } ) \mathbf { z } _ { t } ] ( \Omega _ { s } \mathbf { x } + ( 1 - \alpha _ { s } ) \boldsymbol { \pi } ] } \Bigg ) . } & & { } \\ &  = \mathrm { C a t } \Bigg ( \mathbf { z } _  \end{array}\tag{16}
$$

## A.1.3 $p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$

Austin et al. [1] approximate the reverse process in the following manner:

$$
p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } ) = q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } = \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) ) = \mathbf { C a t } \left( \mathbf { z } _ { s } ; \frac { Q _ { t | s } \mathbf { z } _ { t } \odot Q _ { s } ^ { \top } \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) } { \mathbf { z } _ { t } ^ { \top } Q _ { t } ^ { \top } \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) } \right) .\tag{17}
$$

where $\mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) : \mathcal { V } \times [ 0 , 1 ]  \Delta ^ { K }$ is an approximation for x.

## A.2 Absorbing state

For the absorbing state diffusion process we have ${ \boldsymbol { \pi } } { = } \mathbf { m }$

A.2.1 $q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } )$

Since, $\mathbf { z } _ { t } \in \{ \mathbf { x } , \mathbf { m } \}$ , takes only 2 values we consider the separate cases: $\mathbf { z } _ { t } = \mathbf { x }$ and $\mathbf { z } _ { t } = \mathbf { m }$

Case 1. Consider the case $\mathbf { z } _ { t } = \mathbf { x } \mathbf { i } . \mathbf { e } . \mathbf { z } _ { t }$ is unmasked. From (16), we have the following:

$$
\begin{array} { r l r } & { q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } = \mathbf { x } , \mathbf { x } ) } \\ & { } & { = \mathrm { C a t } \Bigg ( \mathbf { z } _ { s } , \frac { [ \alpha _ { t } | _ { s } \mathbf { x } + ( 1 - \alpha _ { t | s } ) \mathbf { 1 } \mathbf { m } ^ { \top } \mathbf { x } ] \odot [ \alpha _ { s } \mathbf { x } + ( 1 - \alpha _ { s } ) \mathbf { m } ] } { \alpha _ { t } \mathbf { x } ^ { \top } \mathbf { x } + ( 1 - \alpha _ { t } ) \mathbf { x } ^ { \top } \mathbf { m } } \Bigg ) } \\ & { } & \\ & { } & { = \mathrm { C a t } \Bigg ( \mathbf { z } _ { s } , \frac { [ \alpha _ { t } | _ { s } \mathbf { x } ] \odot [ \alpha _ { s } \mathbf { x } + ( 1 - \alpha _ { s } ) \mathbf { m } ] } { \alpha _ { t } } \Bigg ) } \\ & { } & { = \mathrm { C a t } \Bigg ( \mathbf { z } _ { s } , \frac { \alpha _ { t } \mathbf { x } } { \alpha _ { t } } \Bigg ) } \\ & { } & { \quad \cdot \mathbf { \cdot x } \odot \mathbf { m } = \mathbf { 0 } \mathrm { a n d } \alpha _ { t } = \alpha _ { t | s } \alpha _ { s } } \\ & { } & { \quad \cdot \mathbf { \cdot x } \alpha _ { t } = \alpha _ { t | s } \alpha _ { s } } \end{array}\tag{18}
$$

Thus, we have the following:

$$
q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } = \mathbf { x } , \mathbf { x } ) = \mathbf { C } \mathbf { a } \ t ( \mathbf { z } _ { s } ; \mathbf { x } ) .\tag{19}
$$

Case 2. Consider the case $\mathbf { z } _ { t } = \mathbf { m }$ . By substituting $\mathbf { z } _ { t } = \mathbf { m }$ and ${ \boldsymbol { \pi } } = \mathbf { m }$ in $( 1 6 ) , q ( { \bf z } _ { s } | { \bf z } _ { t } , { \bf x } )$ simplifies to the following:

$$
\begin{array} { r l } & { q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } = \mathbf { m } , \mathbf { x } ) = C \mathrm { a t } \Bigg ( \frac { ( \alpha _ { t | s } \mathbf { m } + ( 1 - \alpha _ { t | s } ) \mathbf { 1 } ) \odot ( \alpha _ { s } \mathbf { x } + ( 1 - \alpha _ { s } ) \mathbf { m } ) } { ( 1 - \alpha _ { t } ) } \Bigg ) } \\ & { \qquad = C \mathrm { a t } \Bigg ( \frac { ( \alpha _ { t | s } ( 1 - \alpha _ { s } ) \mathbf { m } + ( 1 - \alpha _ { t | s } ) ( 1 - \alpha _ { s } ) \mathbf { m } + ( \alpha _ { s } - \alpha _ { t } ) \mathbf { x } ) } { ( 1 - \alpha _ { t } ) } \Bigg ) } \end{array}
$$

$$
= \mathrm { C a t } \Bigg ( \mathbf { z } _ { s } ; \frac { ( 1 - \alpha _ { s } ) \mathbf { m } + ( \alpha _ { s } - \alpha _ { t } ) \mathbf { x } } { 1 - \alpha _ { t } } \Bigg )\tag{20}
$$

Note that the above categorical distribution is non-zero for $\mathbf { z } _ { s } \in \{ \mathbf { x } , \mathbf { m } \}$ and zero for every other value. The non-zero values are specified as follows:

$$
q ( \mathbf { z } _ { s } = \mathbf { x } | \mathbf { z } _ { t } = \mathbf { m } , \mathbf { x } ) = \frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } }\tag{21}
$$

$$
q ( \mathbf { z } _ { s } = \mathbf { m } | \mathbf { z } _ { t } = \mathbf { m } , \mathbf { x } ) = \frac { 1 - \alpha _ { s } } { 1 - \alpha _ { t } }\tag{22}
$$

Combining Cases 1 and 2, we get:

$$
\begin{array} { r } { q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) = \left\{ \begin{array} { l l } { \mathrm { C a t } ( \mathbf { z } _ { s } ; \mathbf { z } _ { t } ) } & { \mathbf { z } _ { t } \neq \mathbf { m } , } \\ { \mathrm { C a t } \Big ( \mathbf { z } _ { s } ; \frac { ( 1 - \alpha _ { s } ) \mathbf { m } + ( \alpha _ { s } - \alpha _ { t } ) \mathbf { x } } { 1 - \alpha _ { t } } \Big ) } & { \mathbf { z } _ { t } = \mathbf { m } . } \end{array} \right. } \end{array}\tag{23}
$$

A.2.2 $p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$

For the absorbing state diffusion process with ${ \boldsymbol { \pi } } = \mathbf { m } .$ , we want to simplify the (17). For this reason, we consider 2 cases: first, when $\mathbf { z } _ { t } \neq \mathbf { m }$ (case 1), second, when $\mathbf { z } _ { t } \neq \mathbf { m } \left( \mathbf { c a s e } \ 2 \right)$ .

Case 1. Consider the case when $\mathbf { z } _ { t } \neq \mathbf { m }$ . (17) simplifies to the following:

$$
\begin{array} { r l } { P _ { \mathrm { P } } ( \mathbf { z } ) \mathbf { z } _ { \mathrm { i } } \neq \operatorname* { m a x } \Bigg \{ \mathrm { C a } \Bigg ( \mathbf { z } _ { < } \frac { \mathcal { G } _ { \mathrm { P } } ( \mathbf { z } _ { \mathrm { P } } , \mathbf { z } ) \cdot \nabla ( \mathbf { z } _ { \mathrm { P } } ( \mathbf { z } _ { \mathrm { P } } , \mathbf { z } ) ) } { \mathcal { G } _ { \mathrm { P } } ( \mathbf { z } , \mathbf { z } ) \cdot \nabla ( \mathbf { z } , \mathbf { z } ) } \Bigg ) } & { } \\ & { = \mathrm { C a } \Bigg ( \mathbf { z } _ { < } \cdot \frac { \mathcal { G } _ { \mathrm { P } } ( \mathbf { z } _ { \mathrm { P } } , \mathbf { z } ) \cdot \nabla ( \mathbf { z } _ { \mathrm { P } } ( \mathbf { z } , \mathbf { z } _ { \mathrm { P } } , \mathbf { z } ) ) } { \big ( \mathcal { G } _ { \mathrm { P } } ( \mathbf { z } , \mathbf { z } ) \cdot \nabla ( \mathbf { z } , \mathbf { z } ) \cdot \big ) } \Bigg ) } \\ & { - \mathrm { C a } \Bigg ( \mathbf { z } _ { < } \cdot \frac { \mathcal { G } _ { \mathrm { P } } ( \mathbf { z } , \mathbf { z } ) \cdot \nabla ( \mathbf { z } , \mathbf { z } ) \cdot \nabla ( \mathbf { z } , \mathbf { z } ) } { \big ( \mathcal { G } _ { \mathrm { P } } ( \mathbf { z } , \mathbf { z } ) \cdot \nabla ( \mathbf { z } , \mathbf { z } ) \cdot \nabla ( \mathbf { z } , \mathbf { z } ) \big ) } } \\ &  - \mathrm { C a } \Bigg ( \mathbf { z } _ { < } \cdot \frac { \mathcal { G } _ { \mathrm { P } } ( \mathbf { z } , \mathbf { z } ) \cdot \nabla ( \mathbf { z } , \mathbf { z } ) \cdot \nabla ( \mathbf { z } , \mathbf { z } ) \cdot \nabla ( \mathbf { z } , \mathbf { z } ) \cdot \nabla ( \mathbf { z } , \mathbf { z } ) }  \big ( \mathcal { G } _ { \mathrm { P } } ( \mathbf { z } , \mathbf { z } ) \cdot \nabla ( \mathbf { z } , \end{array}\tag{24}
$$

Case 2. Consider the case when $\mathbf { z } _ { t } = \mathbf { m }$ . (17) simplifies to the following:

$$
\begin{array} { r l } & { p _ { \theta } ( \mathbf z _ { s } | \mathbf z _ { t } = \mathbf { m } ) = \mathrm { C a t } \Bigg ( \mathbf z _ { s } , \frac { \boldsymbol { Q } _ { \mathrm { t i s } } \mathrm { m } \odot \boldsymbol { Q } _ { \mathrm { t } } ^ { \top } \mathbf { x } _ { s } ( \mathbf z _ { t } , t ) } { \mathrm { ~ m } ^ { \top } \boldsymbol { Q } _ { \mathrm { t } } ^ { \top } \times \boldsymbol { g } ( \mathbf z _ { t } , t ) } \Bigg ) } \\ & { \qquad = \mathrm { C a t } \Bigg ( \mathbf z _ { s } , \frac { \boldsymbol { Q } _ { \mathrm { t i s } } \mathrm { m } \odot \boldsymbol { Q } _ { \mathrm { t } } ^ { \top } \mathbf { x } _ { s } ( \mathbf z _ { t } , t ) } { [ \boldsymbol { Q } _ { \mathrm { t } } \mathrm { m } ] ^ { \top } \boldsymbol { S } _ { u } ( z _ { t } , t ) } \Bigg ) } \\ & { \qquad = \mathrm { C a t } \Bigg ( \mathbf z _ { s } , \frac { [ \boldsymbol { Q } _ { \mathrm { t i s } } \mathrm { m } ] + ( 1 - \alpha _ { \mathrm { t i s } } ] \mathbf { 1 } ] \odot [ \boldsymbol { \ O } _ { \mathrm { s } } \mathbf { I } + ( 1 - \alpha _ { s } ) \mathbf { m } \mathbf { 1 } ^ { \top } ] \mathbf { x } _ { \theta } ( \mathbf z _ { t } , t ) } { [ \alpha _ { t } \mathrm { m } + ( 1 - \alpha _ { \mathrm { t i s } } ) ] [ \mathrm { ~ m } ^ { \top } \mathbf { x } _ { \theta } ( \mathbf z _ { t } , t ) } \Bigg ) } \\ &  \qquad = \mathrm { C a t } \Bigg ( \mathbf z _ { s } , \frac { [ \boldsymbol { Q } _ { \mathrm { t i s } } \mathrm { m } ] + ( 1 - \alpha _ { \mathrm { t i s } } ] \mathbf { 1 } ] \odot [ \boldsymbol { \ O } _ { \mathrm { s } } \mathbf { u } _ { \boldsymbol { \theta } } ( \mathbf z _ { t } , t ) ] + ( 1 - \alpha _ { s } ) \mathbf { m } \{ 1 , \mathbf { x } _ { \theta } ( \mathbf z _ { t } , t ) \} }  \alpha _  \end{array}
$$

$$
= \mathrm { C a t } \Bigg ( \mathbf { z } _ { s } ; \frac { \alpha _ { t } \mathbf { m } \odot \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) + ( \alpha _ { s } - \alpha _ { t } ) \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) + ( 1 - \alpha _ { s } ) \mathbf { m } } { \alpha _ { t } \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle + ( 1 - \alpha _ { t } ) } \Bigg )\tag{25}
$$

Note that the above categorical distribution, we can obtain the values for $p _ { \theta } ( \mathbf { z } _ { s } = \mathbf { x } | \mathbf { z } _ { t } = \mathbf { m } )$ and $p _ { \theta } ( \mathbf { z } _ { s } = \mathbf { m } | \mathbf { z } _ { t } = \mathbf { m } )$ ) which are as follows:

$$
p _ { \theta } ( \mathbf { z } _ { s } = \mathbf { x } | \mathbf { z } _ { t } = \mathbf { m } ) = \frac { ( \alpha _ { s } - \alpha _ { t } ) \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { x } \rangle } { \alpha _ { t } \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle + ( 1 - \alpha _ { t } ) }\tag{26}
$$

$$
p _ { \theta } ( \mathbf { z } _ { s } = \mathbf { m } | \mathbf { z } _ { t } = \mathbf { m } ) { = } \frac { \alpha _ { s } \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle { + } ( 1 - \alpha _ { s } ) } { \alpha _ { t } \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle { + } ( 1 - \alpha _ { t } ) }\tag{27}
$$

As a sanity check, we can verify that (26) reduces to (21), and (27) reduces to (22) if our denoising network can reconstruct x perfectly, $\mathbf { i . e . , x } _ { \theta } ( \mathbf { z } _ { t } , t ) = \mathbf { x }$

Combining (24) and (25), we get the following expression for the reverse process parameterization:

$$
p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } ) = \left\{ \begin{array} { l l } { \mathrm { C a t } ( \mathbf { z } _ { s } ; \mathbf { z } _ { t } ) ~ } & { \mathbf { z } _ { t } \neq \mathbf { m } , } \\ { \mathrm { C a t } \Big ( \mathbf { z } _ { s } ; \frac { \alpha _ { t } \mathbf { m } \odot \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) + ( \alpha _ { s } - \alpha _ { t } ) \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) + ( 1 - \alpha _ { s } ) \mathbf { m } } { \alpha _ { t } \langle \mathbf { x } _ { \theta } ( \mathbf { z } _ { t } , t ) , \mathbf { m } \rangle + ( 1 - \alpha _ { t } ) } \Big ) ~ } & { \mathbf { z } _ { t } = \mathbf { m } . } \end{array} \right.\tag{28}
$$

## A.2.3 Diffusion Loss

For a given T , Let $\begin{array} { r } { \mathcal { L } _ { T } = \mathbb { E } _ { t \in \{ \frac { 1 } { r } , \frac { 2 } { r } , \dots , 1 \} } \mathbb { E } _ { q ( \mathbf { z } _ { t } \mid \mathbf { x } ) } T \mathrm { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) \big | \big | p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } ) \big ) } \end{array}$ denote the diffusion loss. We break down the computation of DKL $( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) \| p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } ) )$ into 2 cases: $\mathbf { z } _ { t } = \mathbf { x }$ (case 1) and $\mathbf { z } _ { t } = \mathbf { m } \left( \mathbf { c a s e } \ 2 \right)$ .

Case 1: consider the case $\mathbf { z } _ { t } = \mathbf { x }$ . Let’s simplify $\operatorname { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } = \mathbf { x } , \mathbf { x } ) \big | | p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } = \mathbf { x } ) \big )$

$$
\begin{array} { r l } & { \mathbf { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } = \mathbf { x } , \mathbf { x } ) \| p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } = \mathbf { x } ) \big ) } \\ & { = \mathbf { D } _ { \mathrm { K L } } \big ( \mathbf { z } _ { t } \| \mathbf { z } _ { t } \big ) } \\ & { = 0 } \end{array}
$$

From (23) and (24)

(29)

Case 2: Consider the case $\mathbf { z } _ { t } = \mathbf { m }$ . Let’s simplify $\mathrm { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } { = } \mathbf { m } , \mathbf { x } ) \big | \big | p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } { = } \mathbf { m } ) \big )$

$$
\begin{array} { r l } & { \mathbb { D } _ { \mathbf { R } } \{ \| \boldsymbol { \mathcal { Q } } ( z , \mathbf { z } ) \| _ { \mathcal { L } } ^ { \epsilon } = \operatorname* { m a x } _ { \mathbf { N } } \} | \rho | \boldsymbol { \mathcal { Q } } ( z , \mathbf { z } \mathbf { z } = \mathbf { m } ) \bigg \} } \\ & { = \sum _ { s , t } \tilde { \mathcal { Q } } \{ \| \boldsymbol { \mathcal { Q } } ( z , \mathbf { z } ) \| _ { \mathcal { L } } \{ \mathbf { z } ( \mathbf { z } , \mathbf { u } , \mathbf { x } ) \} \mathrm { i n } \frac { \phi } { \phi } \bigg \} \mathrm { i } \frac { \phi } { \phi } \bigg \{ \tilde { \mathcal { Q } } ( z , \mathbf { z } ( \mathbf { z } , \mathbf { u } , \mathbf { x } ) ) }  \\ & { = \sum _ { s , t } \tilde { \mathcal { Q } } \{ \| \boldsymbol { \mathcal { Q } } ( z , \mathbf { z } ) \mathrm { i } - \mathbf { i } \boldsymbol { \mathcal { Q } } ( \mathbf { z } , \mathbf { x } ) \mathrm { i } \phi \} \mathrm { i } \frac { \phi } { \phi } \mathrm { i } \frac { \phi } { \phi } \bigg \} \mathrm { i } \frac { \phi } { \phi } \bigg \{ \mathcal { Q } ( z , \mathbf { z } ( \mathbf { z } , \mathbf { u } ) ) }  \\ & { \quad \times \mathrm { i } \frac { \phi } { \phi } \mathrm { s } \bigg \{ i } \bigg \{ \phi \bigg \} \mathrm { i } \frac { \phi } { \phi } \mathrm { s } \bigg \{ i  \frac { \phi } { \phi } \bigg \{ \phi  \mathrm { i } \frac { \phi } { \phi } \mathrm { i } \frac { \phi } { \phi } \mathrm { i } \frac { \phi } { \phi } \mathrm { i } \frac { \phi } { \phi } \mathrm { i } \phi \mathrm { i } \overline { { \mathcal { Q } } ( \mathbf { z } , \mathbf { z } ( \mathbf { u } , \mathbf { x } ) ) }   \\ & { \quad \times \mathrm { i } \frac { \phi } { \phi } \mathrm { s } \bigg \{ \phi \mathrm { i } } \frac { \phi }  \phi \end{array}\tag{30}
$$

Thus, $\begin{array} { r } { \operatorname { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) \big | \big | p _ { \theta } \big ( \mathbf { z } _ { s } | \mathbf { z } _ { t } \big ) \big ) } \end{array}$ can be written in the following manner where $\scriptstyle \left. \mathbf { z } _ { t } , \mathbf { x } \right.$ evaluates to 1 if $\mathbf { z } _ { t } = \mathbf { x }$ and $\mathbf { \Psi } ( \mathbf { z } _ { t } , \mathbf { m } )$ evaluates to 1 if $\mathbf { z } _ { t } = \mathbf { m } .$ :

$$
\begin{array} { r } { \operatorname { D } _ { \mathrm { K L } } \big ( q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } ) \big | \big | p _ { \theta } \big ( \mathbf { z } _ { s } | \mathbf { z } _ { t } \big ) \big ) } \end{array}
$$