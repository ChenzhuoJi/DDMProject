# Structured Denoising Diffusion Models in Discrete State-Spaces

Jacob Austin∗, Daniel D. Johnson∗, Jonathan Ho, Daniel Tarlow & Rianne van den Berg† Google Research, Brain Team

{jaaustin,ddjohnson,jonathanho,dtarlow,riannevdberg}@google.com

## Abstract

Denoising diffusion probabilistic models (DDPMs) [19] have shown impressive results on image and waveform generation in continuous state spaces. Here, we introduce Discrete Denoising Diffusion Probabilistic Models (D3PMs), diffusionlike generative models for discrete data that generalize the multinomial diffusion model of Hoogeboom et al. [20], by going beyond corruption processes with uniform transition probabilities. This includes corruption with transition matrices that mimic Gaussian kernels in continuous space, matrices based on nearest neighbors in embedding space, and matrices that introduce absorbing states. The third allows us to draw a connection between diffusion models and autoregressive and mask-based generative models. We show that the choice of transition matrix is an important design decision that leads to improved results in image and text domains. We also introduce a new loss function that combines the variational lower bound with an auxiliary cross entropy loss. For text, this model class achieves strong results on character-level text generation while scaling to large vocabularies on LM1B. On the image dataset CIFAR-10, our models approach the sample quality and exceed the log-likelihood of the continuous-space DDPM model.

## 1 Introduction

Generative modeling is a core problem in machine learning, useful both for benchmarking our ability to capture statistics of natural datasets and for downstream applications that require generating high-dimensional data like images, text, and speech waveforms. There has been a great deal of progress with the development of methods like GANs [15, 4], VAEs [25, 35], large autoregressive neural network models [51, 50, 52], normalizing flows [34, 12, 24, 32], and others, each with their own tradeoffs in terms of sample quality, sampling speed, log-likelihoods, and training stability.

Recently, diffusion models [43] have emerged as a compelling alternative for image [19, 46] and audio [7, 26] generation, achieving comparable sample quality to GANs and log-likelihoods comparable to autoregressive models with fewer inference steps. A diffusion model is a parameterized Markov chain trained to reverse a predefined forward process, which is a stochastic process constructed to gradually corrupt training data into pure noise. Diffusion models are trained using a stable objective closely related to both maximum likelihood and score matching [21, 53], and they admit faster sampling than autoregressive models by using parallel iterative refinement [30, 45, 47, 44].

Although diffusion models have been proposed in both discrete and continuous state spaces [43], most recent work has focused on Gaussian diffusion processes that operate in continuous state spaces (e.g. for real-valued image and waveform data). Diffusion models with discrete state spaces have been explored for text and image segmentation domains [20], but they have not yet been demonstrated as a competitive model class for large scale text or image generation.

35th Conference on Neural Information Processing Systems (NeurIPS 2021).

<!-- image-->  
Figure 1: D3PM forward and (learned) reverse process applied to a quantized swiss roll. Each dot represents a 2D categorical variable. Top: samples from the uniform, discretized Gaussian, and absorbing state D3PM model forward processes, along with corresponding transition matrices Q. Bottom: samples from a learned discretized Gaussian reverse process.

Our aim in this work is to improve and extend discrete diffusion models by using a more structured categorical corruption process to shape data generation, as illustrated in Figure 1. Our models do not require relaxing or embedding discrete data (including images) into continuous spaces, and can embed structure or domain knowledge into the transition matrices used by the forward process. We achieve significantly improved results by taking advantage of this flexibility. We develop structured corruption processes appropriate for text data, using similarity between tokens to enable gradual corruption and denoising. Expanding further, we also explore corruption processes that insert [MASK] tokens, which let us draw parallels to autoregressive and mask-based generative models. Finally, we study discrete diffusion models for quantized images, taking inspiration from the locality exploited by continuous diffusion models. This leads to a particular choice of discrete corruption process that diffuses preferentially to more similar states and leads to much better results in the image domain.

Overall, we make a number of technical and conceptual contributions. Beyond designing several new structured diffusion models, we introduce a new auxiliary loss which stabilizes training of D3PMs and a family of noise schedules based on mutual information that lead to improved performance. We strongly outperform various non-autoregressive baselines for text generation on character-level text generation, and successfully scale discrete diffusion models to large vocabularies and long sequence lengths. We also achieve strong results on the image dataset CIFAR-10, approaching or exceeding the Gaussian diffusion model from Ho et al. [19] on log-likelihoods and sample quality.

## 2 Background: diffusion models

Diffusion models [43] are latent variable generative models characterized by a forward and a reverse Markov process. The forward process $\begin{array} { r } { q ( \pmb { x } _ { 1 : T } | \pmb { x } _ { 0 } ) = \prod _ { t = 1 } ^ { T } q ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } ) } \end{array}$ corrupts the data $\mathbf { \delta } _ { \mathbf { \mathcal { X } } 0 } \sim \mathbf { \delta } $ $q ( { \pmb x } _ { 0 } )$ into a sequence of increasingly noisy latent variables x ${ \bf \mathrm { 1 : } } T = { \bf x } _ { 1 } , { \bf x } _ { 2 } , . . . , { \bf x } _ { T }$ The learned reverse Markov process $\begin{array} { r } { p _ { \theta } ( \pmb { x } _ { 0 : T } ) = p ( \pmb { x } _ { T } ) \prod _ { t = 1 } ^ { T } p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } ) } \end{array}$ gradually denoises the latent variables towards the data distribution. For example, for continuous data, the forward process typically adds Gaussian noise, which the reverse process learns to remove.

In order to optimize the generative model $p _ { \theta } ( { \pmb x } _ { 0 } )$ to fit the data distribution $q ( { \pmb x } _ { 0 } )$ , we typically optimize a variational upper bound on the negative log-likelihood:

$$
\begin{array} { r l } & { L _ { \mathrm { v b } } = \mathbb { E } _ { q ( \boldsymbol { x } _ { 0 } ) } \Bigg [ \underbrace { D _ { \mathrm { K L } } \big [ q ( \boldsymbol { x } _ { T } | \boldsymbol { x } _ { 0 } ) \big | | p ( \boldsymbol { x } _ { T } ) \big ] } _ { L _ { T } } + \underset { t = 2 } { \overset { T } { \sum } } \underbrace { \mathbb { E } _ { q ( \boldsymbol { x } _ { t } | \boldsymbol { x } _ { 0 } ) } \big [ D _ { \mathrm { K L } } \big [ q ( \boldsymbol { x } _ { t - 1 } | \boldsymbol { x } _ { t } , \boldsymbol { x } _ { 0 } ) \big | \big | p _ { \theta } ( \boldsymbol { x } _ { t - 1 } | \boldsymbol { x } _ { t } ) \big ] \big ] } _ { L _ { t - 1 } } } \\ & { \quad \quad \quad \quad \quad \underbrace { - \mathbb { E } _ { q ( \boldsymbol { x } _ { 1 } | \boldsymbol { x } _ { 0 } ) } \big [ \log p _ { \theta } ( \boldsymbol { x } _ { 0 } | \boldsymbol { x } _ { 1 } ) \big ] } _ { L _ { 0 } } \Bigg ] . } \end{array}\tag{1}
$$

When the number of time steps $T$ goes to infinity, both the forward process and the reverse process share the same functional form [13], allowing the use of a learned reverse process from the same class of distributions as that of the forward process. Furthermore, for several choices of the forward process the distribution $q ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ converges to a stationary distribution $\pi ( { \pmb x } )$ in the limit $t \to \infty$ independent of the value of $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ . When the number of time steps $T$ is large enough and we choose $\pi ( { \pmb x } )$ as the prior $p ( { \pmb x } _ { T } )$ , we can guarantee that the $L _ { T }$ term in (1) will approach zero regardless of the data distribution $q ( \pmb { x } _ { 0 } )$ . (Alternatively, one can use a learned prior $p _ { \theta } ( { \pmb x } _ { T } ) . )$

While $q \big ( \mathbf { \boldsymbol { x } } _ { t } | \mathbf { \boldsymbol { x } } _ { t - 1 } \big )$ can in theory be arbitrary, efficient training of $p _ { \theta }$ is possible when $q \big ( \mathbf { \boldsymbol { x } } _ { t } | \mathbf { \boldsymbol { x } } _ { t - 1 } \big )$

1. Permits efficient sampling of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ from $q ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ for an arbitrary time t, allowing us to randomly sample timesteps and optimize each $L _ { t - 1 }$ term individually with stochastic gradient descent,

2. Has a tractable expression for the forward process posterior $q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } )$ , which allows us to compute the KL divergences present in the $L _ { t - 1 }$ term of (1).

The majority of recent work in continuous spaces [19, 44, 7, 30] defines the forward√ and reverse distributions as $\begin{array} { r l r } { q ( { \pmb x } _ { t } | { \pmb x } _ { t - 1 } ) } & { = } & { \sqrt { { \bf \alpha } } \sqrt { { \bf \alpha } } \big ( { \pmb x } _ { t } | \sqrt { 1 - \beta _ { t } } { \pmb x } _ { t - 1 } , \beta _ { t } { \pmb I } \big ) } \end{array}$ and $\begin{array} { r l } { p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } ) } & { = } \end{array}$ $\mathcal { N } \left( \boldsymbol { x } _ { t - 1 } | \mu _ { \boldsymbol { \theta } } ( \boldsymbol { x } _ { t } , t ) , \boldsymbol { \Sigma } _ { \boldsymbol { \theta } } ( \boldsymbol { x } _ { t } , t ) \right)$ , respectively. The aforementioned properties hold in the case of these Gaussian diffusion models: the forward process $q ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ ) converges to a stationary distribution, motivating the choice $p ( { \pmb x } _ { T } ) = \mathcal { N } \left( { \pmb x } _ { T } | \mathbf { 0 } , I \right)$ , and both $q ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ and $q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } )$ are tractable Gaussian distributions for which the KL divergence can be computed analytically.

## 3 Diffusion models for discrete state spaces

Diffusion models with discrete state spaces were first introduced by Sohl-Dickstein et al. [43], who considered a diffusion process over binary random variables. Hoogeboom et al. [20] extended the model class to categorical random variables with transition matrices characterized by uniform transition probabilities. In their supplementary material, Song et al. [44] also derived this extension, although no experiments were performed with this model class. Here, we briefly describe a more general framework for diffusion with categorical random variables which includes these models as special cases.

For scalar discrete random variables with K categories $x _ { t } , x _ { t - 1 } \in { 1 , . . . , K }$ the forward transition probabilities can be represented by matrices: $[ Q _ { t } ] _ { i j } = q ( x _ { t } = j | x _ { t - 1 } = i )$ . Denoting the one-hot version of x with the row vector x, we can write

$$
q ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } ) = \mathrm { C a t } ( \pmb { x } _ { t } ; \pmb { p } = \pmb { x } _ { t - 1 } \pmb { Q } _ { t } ) ,\tag{2}
$$

where $\operatorname { C a t } ( \pmb { x } ; \pmb { p } )$ is a categorical distribution over the one-hot row vector x with probabilities given by the row vector ${ \mathbf { } } p ,$ and ${ \pmb x } _ { t - 1 } { \pmb Q } _ { t }$ t is to be understood as a row vector-matrix product. We assume that $Q _ { t }$ is applied to each pixel of an image or each token in a sequence independently, and that q factorizes over these higher dimensions as well; we thus write $q ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } )$ in terms of a single element. Starting from $\scriptstyle { \mathbf { { \vec { x } } } } _ { 0 }$ , we obtain the following t-step marginal and posterior at time $t - 1 \colon$

$$
\begin{array} { r } { q ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) = \mathrm { C a t } \left( { \pmb x } _ { t } ; { \pmb p } = { \pmb x } _ { 0 } \overline { { \pmb Q } } _ { t } \right) , \quad \mathrm { w i t h } \quad \overline { { \pmb Q } } _ { t } = Q _ { 1 } { \pmb Q } _ { 2 } \ldots Q _ { t } } \end{array}
$$

$$
q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } ) = \frac { q ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } , \pmb { x } _ { 0 } ) q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { 0 } ) } { q ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) } = \mathrm { C a t } \left( \pmb { x } _ { t - 1 } ; \pmb { p } = \frac { \pmb { x } _ { t } Q _ { t } ^ { \top } \odot \pmb { x } _ { 0 } \overline { { Q } } _ { t - 1 } } { \pmb { x } _ { 0 } \overline { { Q } } _ { t } \pmb { x } _ { t } ^ { \top } } \right) .\tag{3}
$$

Note that due to the Markov property of the forward process $q ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } , \pmb { x } _ { 0 } ) = q ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } )$ . Assuming that the reverse process $p _ { \theta } ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } )$ is also factorized as conditionally independent over the image or sequence elements, the KL divergence between q and $p _ { \theta }$ can be computed by simply summing over all possible values of each random variable; we thus satisfy criteria 1 and 2 discussed in Section 2. Depending on $Q _ { t }$ , the cumulative products $\overline { { \boldsymbol { Q } } } _ { t }$ can often be computed in closed form, or simply precomputed for all t. However, for large K and large T this may be prohibitive. In Appendix A.4 we discuss how to ensure $\overline { { \pmb { Q } } } _ { t }$ can still be computed efficiently in this case, allowing the framework to scale to a larger number of categories.

In the next section we discuss the choice of the Markov transition matrices $Q _ { t }$ and corresponding stationary distributions. From here on, we refer to the general class of diffusion models with discrete state spaces as Discrete Denoising Diffusion Probabilistic Models (D3PMs).

## 3.1 Choice of Markov transition matrices for the forward process

An advantage of the D3PM framework described above is the ability to control the data corruption and denoising process by choosing $Q _ { t } ,$ , in notable contrast to continuous diffusion, for which only additive Gaussian noise has received significant attention. Besides the constraint that the rows of $Q _ { t }$ must sum to one to conserve probability mass, the only other constraint in choosing $Q _ { t }$ is that the rows of $\overline { { { \pmb { Q } } } } _ { t } = \pmb { Q } _ { 1 } \pmb { Q } _ { 2 } \dots \pmb { Q } _ { t }$ must converge to a known stationary distribution3 when t becomes large, which can be guaranteed while imposing minimal restrictions on $Q _ { t }$ (see Appendix A.1).

We argue that for most real-world discrete data, including images and text, it makes sense to add domain-dependent structure to the transition matrices $Q _ { t }$ as a way of controlling the forward corruption process and the learnable reverse denoising process. Below we briefly discuss the uniform transition matrices that have been studied in prior work [20], along with a set of structured transition matrices we have explored for our image and text dataset experiments; see Appendix A.2 for more details on each matrix type. We also note that this set is not exhaustive, and many other transition matrices could also be used within the D3PM framework.

Uniform (Appendix A.2.1). Sohl-Dickstein et al. [43] considered a simple $2 \times 2$ transition matrix for binary random variables. Hoogeboom et al. [20] later extended this to categorical variables, proposing a transition matrix $\pmb { Q } _ { t } = ( 1 - \beta _ { t } ) \pmb { I } + \beta _ { t } / K \bar { \parallel } \mathbb { 1 } ^ { T }$ with $\beta _ { t } \in [ 0 , 1 ]$ . Since this transition matrix is doubly stochastic with strictly positive entries, the stationary distribution is uniform. Because the transition probability to any other state is uniform, in this paper we equivalently refer to this discrete diffusion instance as D3PM-uniform.

Absorbing state (Appendix A.2.2). Motivated by the success of BERT [11] and recent work on Conditional Masked Language Models (CMLMs) in text, we consider a transition matrix with an absorbing state (called [MASK]), such that each token either stays the same or transitions to [MASK] with some probability $\beta _ { t }$ . This does not impose particular relationships between categories, similar to uniform diffusion, but still allows corrupted tokens to be distinguished from original ones. Moreover, the stationary distribution is not uniform but has all the mass on the [MASK] token. For images, we reuse the grey pixel as the [MASK] absorbing token.

Discretized Gaussian (Appendix A.2.3). Instead of transitioning uniformly to any other state, for ordinal data we propose imitating a continuous space diffusion model by using a discretized, truncated Gaussian distribution. We choose a normalization such that the transition matrix is doubly stochastic, leading to a uniform stationary distribution. This transition matrix will transition between more similar states with higher probability, and is well suited for quantized ordinal data such as images.

Token embedding distance (Appendix A.2.4). Textual data does not have ordinal structure, but there may still be interesting semantic relationships. For instance, in a character level vocabulary vowels may be more similar to each other than they are to consonants. As a demonstration of the generality of the D3PM framework, we explore using similarity in an embedding space to guide the forward process, and construct a doubly-stochastic transition matrix that transitions more frequently between tokens that have similar embeddings while maintaining a uniform stationary distribution.

For uniform and absorbing-state diffusion, the cumulative products $\overline { { \pmb { Q } } } _ { t }$ can be computed in closed form (see Appendix A.4.1); the remainder can be precomputed.

## 3.2 Noise schedules

We consider several different options for the noise schedule of the forward process. For discretized Gaussian diffusion, we explore linearly increasing the variance of the Gaussian before discretizing it. (Note that a linear schedule for $Q _ { t }$ leads to a nonlinear amount of cumulative noise in $\overline { { Q } } _ { t } . )$ For uniform diffusion we use the cosine schedule which sets the cumulative probability of a transition to a cosine function, as introduced by Nichol and Dhariwal [30] and adapted by Hoogeboom et al. [20]. For a general set of transition matrices $Q _ { t }$ (such as the one based on token embeddings), previously proposed schedules may not be directly applicable. We consider linearly interpolating the mutual information between $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ and $\scriptstyle { \pmb x } _ { 0 }$ to zero, i.e. $\begin{array} { r } { I ( \pmb { x } _ { t } ; \pmb { x } _ { 0 } ) \approx ( 1 - \frac { t } { T } ) H ( \pmb { \bar { x } } _ { 0 } ) } \end{array}$ ). Interestingly, for the specific case of absorbing-state D3PMs, this schedule reduces to exactly the $( T - t + 1 ) ^ { - 1 }$ schedule proposed by Sohl-Dickstein et al. [43] for a Bernoulli diffusion process. See Appendix A.7 for more details.

## 3.3 Parameterization of the reverse process

While it is possible to directly predict the logits of $p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } )$ using a neural network $\mathrm { n n } _ { \theta } ( { \pmb x } _ { t } )$ we follow Ho et al. [19] and Hoogeboom et al. [20] and focus on using a neural network $\mathrm { n n } _ { \theta } ( { \pmb x } _ { t } )$ to predict the logits of a distribution $\widetilde { p } _ { \theta } ( \widetilde { \pmb x } _ { 0 } | \pmb x _ { t } )$ , which we combine with $q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } )$ and a esummation over one-hot representations of $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ to obtain the following parameterization

$$
p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } ) \propto \sum _ { \widetilde { \pmb { x } } _ { 0 } } q ( \pmb { x } _ { t - 1 } , \pmb { x } _ { t } | \widetilde { \pmb { x } } _ { 0 } ) \widetilde { p } _ { \theta } ( \widetilde { \pmb { x } } _ { 0 } | \pmb { x } _ { t } ) .\tag{4}
$$

We note that under this $\scriptstyle { \pmb { x } } _ { 0 } \cdot$ -parameterization the KL divergence $D _ { \mathrm { K L } } [ q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } ) | | p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } ) ]$ will be zero if $\widetilde { p } _ { \boldsymbol { \theta } } ( \widetilde { \pmb { x } } _ { 0 } | \pmb { x } _ { t } )$ places all of its probability mass on the original value $\mathbf { \delta x } _ { 0 } .$ . The decomposition of $q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } )$ in (3) also provides us with a motivation for this parameterization. According to (3), in a given state $\mathbf { \Delta } \mathbf { x } _ { t } .$ , the optimal reverse process only takes into account transitions to states for which $q \big ( \mathbf { \boldsymbol { x } } _ { t } | \mathbf { \boldsymbol { x } } _ { t - 1 } \big )$ is non-zero. Therefore, the sparsity pattern of $Q _ { t }$ determines the sparsity pattern of the ideal reverse transition probabilities in $p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } )$ . The parameterization in (4) automatically ensures that the learned reverse probability distribution $\dot { p } _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } )$ has the correct sparsity pattern dictated by the choice of the Markov transition matrix $Q _ { t }$ . This parameterization also lets us perform inference with k steps at a time, by predicting $\begin{array} { r } { p _ { \theta } ( \pmb { x } _ { t - k } | \pmb { x } _ { t } ) = \sum q ( \pmb { x } _ { t - k } , \pmb { x } _ { t } | \widetilde { \pmb { x } } _ { 0 } ) \widetilde { p _ { \theta } } ( \widetilde { \pmb { x } } _ { 0 } | \pmb { x } _ { t } ) } \end{array}$

Finally, when modeling ordinal discrete data, instead of predicting the logits of $\widetilde { p } _ { \theta } ( \widetilde { \pmb { x } } _ { 0 } | \pmb { x } _ { t } )$ directly e ewith the output of a neural net, another option is to model the probabilities with a truncated discretized logistic distribution (see Appendix A.8). This provides an extra ordinal inductive bias to the reverse model and boosts FID and log-likelihood scores for images.

## 3.4 Loss function

While the original diffusion models introduced by Sohl-Dickstein et al. [43] were optimized with the negative variational lower bound $L _ { \mathrm { v b } }$ of (1), more recent diffusion models are optimized with different objectives. For instance, Ho et al. [19] derive a simplified loss function $( L _ { \mathrm { s i m p l e } } )$ that reweights the negative variational bound, and Nichol and Dhariwal [30] explore a hybrid loss $L _ { \mathrm { h y b r i d } } ~ = ~ L _ { \mathrm { s i m p l e } } + \lambda L _ { \mathrm { v b } }$ (using one term to learn the predicted mean and the other to learn predicted variance). Inspired by this recent work, we introduce an auxiliary denoising objective for the $\scriptstyle { \pmb { x } } _ { 0 } \cdot$ -parameterization of the reverse process, which encourages good predictions of the data $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ at each time step. We combine this with the negative variational lower bound, yielding the following alternative loss function:

$$
L _ { \lambda } = L _ { \mathrm { v b } } + \lambda \mathbb { E } _ { q ( \pmb { x } _ { 0 } ) } \mathbb { E } _ { q ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) } [ - \log \widetilde { p } _ { \theta } ( \pmb { x } _ { 0 } | \pmb { x } _ { t } ) ] .\tag{5}
$$

Note that the auxiliary loss coincides with the cross entropy term $L _ { 0 }$ in (1) at $t ~ = ~ 1$ . Furthermore, due to the x0-parameterization of $p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } )$ , both the auxiliary loss term and $D _ { \mathrm { K L } } [ q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } ) | | p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } ) ]$ in $L _ { \mathrm { v b } }$ are minimized exactly when $\widetilde { p } _ { \theta } ( \widetilde { \pmb { x } } _ { 0 } | \pmb { x } _ { t } )$ has all its mass on the datapoint $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ e e. We find that training with this loss leads to improved quality of image samples.

## 4 Connection to existing probabilistic models for text

In this section we expand on interesting connections between the D3PM framework and several existing probabilistic and language modeling approaches.

BERT is a one-step diffusion model: One possible D3PM transition matrix is a combination of a uniform transition matrix and an absorbing state at the [MASK] token (i.e. $Q = \alpha \mathbb { 1 } e _ { m } ^ { T } + \beta \mathbb { 1 } \mathbb { 1 } ^ { T } / K +$ $( 1 - \alpha - \beta ) I$ , where $e _ { m }$ is a one-hot vector on the [MASK] token). For a one-step diffusion process in which $q ( \pmb { x } _ { 1 } | \pmb { x } _ { 0 } )$ replaces 10% of tokens with [MASK] and 5% uniformly at random, this leads precisely to the BERT denoising objective, i.e. $L _ { v b } - L _ { T } = - \mathbb { E } _ { q ( \pmb { x } _ { 1 } | \pmb { x } _ { 0 } ) } [ \log p _ { \theta } ( \pmb { x } _ { 0 } | \pmb { x } _ { 1 } ) ] = L _ { B E R T }$ since $L _ { T }$ is a constant independent of θ (assuming a fixed prior).

Autoregressive models are (discrete) diffusion models: Consider a diffusion process that deterministically masks tokens one-by-one in a sequence of length $N = T \colon q ( [ \pmb { x } _ { t } ] _ { i } \mid \mathbf { \dot { x } } _ { 0 } ) = [ \pmb { x } _ { 0 } ] _ { i } \mathrm { i f } i <$

$N - t$ else [MASK] . This is a deterministic forward process, so $q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } )$ is a delta distribution on the $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ sequence with one fewer mask: $q ( [ \pmb { x } _ { t - 1 } ] _ { i } | \pmb { x } _ { t } , \pmb { x } _ { 0 } ) = \delta _ { [ \pmb { x } _ { t } ] _ { i } }$ if $i \neq T - t$ else $\delta _ { [ { \pmb x } _ { 0 } ] _ { i } }$ . While this process is not applied independently to each token, it can be recast as an independently-applied diffusion process on the product space $[ 0 . . . N ] \times { \mathcal { V } } .$ , where each token is tagged with its position in the sequence, V is the vocabulary, and Q is an $N \times | \mathcal { V } | \times N \times | \mathcal { V } |$ sparse matrix.

Because all tokens except the one at position $i = T - t$ have deterministic posteriors, the KL divergence $D _ { K L } \big ( q ( [ \pmb { x } _ { t - 1 } ] _ { j } | \pmb { x } _ { t } , \pmb { x } _ { 0 } ) \big | \big | \big | p _ { \theta } \big ( [ \pmb { x } _ { t - 1 } ] _ { j } | \pmb { x } _ { t } \big ) \big )$ is zero for all other positions. The only token for which this is not true is the token at position i, for which $D _ { K L } \bar { ( } q ( [ \mathbf { x } _ { t - 1 } ] _ { i } | \mathbf { x } _ { t } , \mathbf { x } _ { 0 } )$ $p _ { \theta } ( [ \pmb { x } _ { t - 1 } ] _ { i } | \pmb { x } _ { t } ) ) = - \log p _ { \theta } ( [ \pmb { x } _ { 0 } ] _ { i } | \pmb { x } _ { t } )$ , the standard cross entropy loss for an autoregressive model.

(Generative) Masked Language-Models (MLMs) are diffusion models: Generative Masked Language Models ([14], [54]) are generative models that generate text from a sequence of [MASK] tokens. They are usually trained by sampling a sequence ${ \pmb x } _ { 0 } ,$ , masking k tokens according to some schedule, and learning to predict the masked tokens given context. It turns out that a D3PM absorbing ([MASK]) model trained on the usual ELBO objective with the x0-parameterization from 3.3 reduces to a reweighted version of this MLM objective (see Appendix A.3 for a detailed derivation).

## 5 Text generation

For text, we experiment with generation on two datasets: text8 [28], a character-level dataset extracted from English-language Wikipedia, and the One Billion Word dataset (LM1B) [6], a large dataset of shuffled English-language sentences. For both, we train a D3PM uniform model based on the work by Hoogeboom et al. [20] (D3PM uniform) and a model that masks tokens (D3PM absorbing). We also consider a model that transitions uniformly to nearest neighbors in a token embedding space (D3PM NN). We follow Hoogeboom et al. [20] and use $T = 1 \bar { 0 } 0 0$ timesteps, although we are also able to evaluate on fewer due to the parameterization in Section 3.3.

## 5.1 Character-level generation on text8

text8 is a character-level text dataset consisting of a small vocabulary of 27 tokens: the letters $\mathbf { \hat { a } } _ { } ^ { \prime } - \mathbf { \hat { z } } _ { } ^ { \prime }$ and the ‘_’ whitespace token. We follow the convention of training and evaluating text8 in chunks of length 256 without any preprocessing [20]. For nearest-neighbor D3PM, our nearest neighbor graph in character-space is shown in Appendix B.2.1. D3PM uniform models were trained with a cosine schedule from Hoogeboom et al. [20] (ablations in Appendix B.2.1), while D3PM absorbing and D3PM NN models were trained with a mutual information schedule.

Table 1: Quantitative results on text8. NLL is reported on the entire test set. Sample times are for generating a single example of length 256. Results are reported on two seeds. All models are standard 12-layer transformers unless otherwise noted. †Transformer XL is a 24-layer transformer, using a 784 context window. ‡Results reported by [20] by running code from official repository.
<table><tr><td>Model</td><td>Model steps</td><td>NLL (bits/char) (↓)</td><td>Sample time (s) (↓)</td></tr><tr><td>Discrete Flow [49](8 × 3 layers)</td><td></td><td>1.23</td><td>0.16</td></tr><tr><td>Argmax Coupling Flow [20]</td><td></td><td>1.80</td><td> $0 . 4 0 \pm 0 . 0 3$ </td></tr><tr><td>IAF/SCF [57]*</td><td></td><td>1.88</td><td> $0 . 0 4 \pm 0 . 0 0 0 4$ </td></tr><tr><td>Multinomial Diffusion (D3PM uniform) [20]</td><td>1000</td><td> $\leq 1 . 7 2$ </td><td> $2 6 . 6 \pm 2 . 2$ </td></tr><tr><td>D3PM uniform [20] (ours)</td><td>1000</td><td> $\leq 1 . 6 1 \pm 0 . 0 2$ </td><td> $3 . 6 \pm 0 . 4$ </td></tr><tr><td>D3PM NN(Lvb) (ours)</td><td>1000</td><td> $\leq 1 . 5 9 \pm 0 . 0 3$ </td><td> $3 . 1 4 7 4 \pm 0 . 0 0 0 2$ </td></tr><tr><td>D3PM mask(Lλ=0.01)(ours)</td><td>1000</td><td> $\leq 1 . 4 5 \pm 0 . 0 2$ </td><td> $3 . 4 \pm 0 . 3$ </td></tr><tr><td>D3PM uniform [20] (ours)</td><td>256</td><td> $\leq 1 . 6 8 \pm 0 . 0 1$ </td><td> $0 . 5 8 0 1 \pm 0 . 0 0 0 1$ </td></tr><tr><td> $\mathrm { D 3 P M N N } \left( L _ { \mathrm { v b } } \right) \left( \mathrm { o u r s } \right)$ </td><td>256</td><td> $\overline { { \leq } } \ 1 . 6 4 \pm 0 . 0 2$ </td><td> $0 . 8 1 3 \pm 0 . 0 0 2$ </td></tr><tr><td>D3PMabsorbing (Lλ=o.o1) (ours)</td><td>256</td><td> $\leq 1 . 4 7 \pm 0 . 0 3$ </td><td> $0 . 5 9 8 \pm 0 . 0 0 2$ </td></tr><tr><td>Transformer decoder (ours)</td><td>256</td><td>1.23</td><td> $0 . 3 5 7 0 \pm 0 . 0 0 0 2$ </td></tr><tr><td>Transformer decoder [1]</td><td>256</td><td>1.18</td><td></td></tr><tr><td>Transformer XL [10]†</td><td>256</td><td>1.08</td><td></td></tr><tr><td>D3PM uniform [20](ours)</td><td>20</td><td> $\leq 1 . 7 9 \pm 0 . 0 3$ </td><td> $0 . 0 7 7 1 \pm 0 . 0 0 0 5$ </td></tr><tr><td>D3PM NN(Lvb)(ours)</td><td>20</td><td> $\overline { { \leq } } ~ 1 . 7 5 \pm 0 . 0 2$ </td><td> $0 . 1 1 1 0 \pm 0 . 0 0 0 1$ </td></tr><tr><td>D3PM absorbing (Lx=0.o1） (ours)</td><td>20</td><td> $\stackrel { - } { \leq } 1 . 5 6 \pm 0 . 0 4$ </td><td> $0 . 0 7 8 5 \pm 0 . 0 0 0 3$ </td></tr></table>

<!-- image-->

<!-- image-->  
Figure 2: Left: perplexity v.s. sampling iterations for LM1B. Right: Using a trained D3PM absorbing model for LM1B to (top) generate new sentences and (bottom) reconstruct corrupted examples.

Table 2: Quantitative results on LM1B. Perplexity reported on the test set. Results are reported on two seeds. All models have context window length 128 and 12 layers unless otherwise noted. †Transformer XL is a 24 layer transformer. ‡rounded for readability, see Appendix B.2.2.
<table><tr><td rowspan="2">Metric: inference steps:</td><td colspan="3">Perplexity (↓)</td><td colspan="3">Sample time* ( (s)(↓)</td></tr><tr><td>1000</td><td>128</td><td>64</td><td>1000</td><td>128</td><td>64</td></tr><tr><td>D3PM uniform</td><td> $1 3 7 . 9 \pm 2 . 1$ </td><td> $1 3 9 . 2 \pm 1 . 2$ </td><td> $1 4 5 . 0 \pm 1 . 2$ </td><td>1.82</td><td>0.21</td><td>0.08</td></tr><tr><td>D3PMNN</td><td> $1 4 9 . 5 \pm 1 . 3$ </td><td> $1 5 8 . 6 \pm 2 . 2$ </td><td> $1 6 0 . 4 \pm { 1 . 2 }$ </td><td>21.29</td><td>6.69</td><td>5.88</td></tr><tr><td>D3PM absorbing</td><td> $7 6 . 9 \pm 2 . 3$ </td><td> $8 0 . 1 \pm 1 . 2$ </td><td> $8 3 . 6 \pm 6 . 1$ </td><td>1.90</td><td>0.19</td><td>0.10</td></tr><tr><td>Transformer (ours)</td><td></td><td>43.6</td><td></td><td>1</td><td>0.26</td><td>-</td></tr><tr><td>Transformer XL [10]+</td><td></td><td>21.8</td><td></td><td>1</td><td>1</td><td>=</td></tr></table>

Table 1 shows that for D3PM, the D3PM absorbing model performed the best, exceeding the uniform and NN diffusion models. We were able to improve upon the baseline result of [20] with hyperparameter tuning, and our uniform and NN results outperformed results from Hoogeboom et al. [20] across all inference steps, down to as few as 20. We found that $L _ { \lambda = 0 . 0 1 }$ worked best for D3PM absorbing, while $L _ { \mathrm { v b } }$ was better for D3PM uniform. Our model outperforms all nonautoregressive baselines except one, the Discrete Flow model [49] (for which unfortunately no open-source implementations exist), and is also faster than all but one method, the IAF/SCF model [57]. It is also nearly 20x faster than an autoregressive transformer of the same size. We also include a plot of inference time as a function of iterations in Appendix B.2.1. D3PM with the mask absorbing token was by far the best performing model, which lends credibility to the use of masks in denoising auto-encoders. Nearest-neighbor diffusion only narrowly improves upon a D3PM-uniform model: this was a surprising negative result for us, suggesting that not all notions of structure are meaningful.

## 5.2 Text generation on LM1B

Text generation for large-scale text datasets and large vocabularies with discrete diffusion models has not been previously demonstrated. We include results from LM1B as a proof of concept, showing that these models can indeed scale (as discussed in Appendix A.4), and that the D3PM absorbing model continues to excel. All models were trained and evaluated on packed sequences of length 128, using a sentencepiece4 vocabulary of size 8192.

Table 2 contains results from experiments on LM1B. Overall, mask diffusion (D3PM absorbing) does relatively well, approaching the performance of a comparable autoregressive model of the same size, and scaling to far fewer steps, while uniform diffusion performs significantly worse. We find, surprisingly, that the D3PM NN model performs worse than the uniform model in terms of log likelihoods (although it demonstrates unique qualitative behavior). This suggests that word embedding similarity may not be a meaningful kind of locality in a diffusion process. We found the the $L _ { \lambda = 0 . 0 1 }$ loss worked best for the mask absorbing model, but reduced performance for the other models. We note the surprising scaling in perplexity in Figure 2, achieving strong results with as few as 10 inference steps. We also show samples from our model and completions from corrupted samples.

Table 3: Inception scores (IS), Frechet Inception Distance (FID) and negative log-likehood (NLL) on the image dataset CIFAR-10. The NLL is reported on the test set in bits per dimension. We report our results as averages with standard deviations, obtained by training five models with different seeds.
<table><tr><td>Model</td><td>IS (↑)</td><td> $\mathrm { F I D } \left( \downarrow \right)$ </td><td>NLL (↓)</td></tr><tr><td>Sparse Transformer [9]</td><td></td><td></td><td>2.80</td></tr><tr><td>NCSN [45]</td><td> $8 . 8 7 \pm 0 . 1 2$ </td><td>25.32</td><td></td></tr><tr><td>NCSNv2 [46]</td><td> $8 . 4 0 \pm 0 . 0 7$ </td><td>10.87</td><td></td></tr><tr><td>StyleGAN2 + ADA [22]</td><td> $9 . 7 4 \pm 0 . 0 5$ </td><td>3.26</td><td></td></tr><tr><td> $\mathrm { D i f f u s i o n ( o r i g i n a l ) } , L _ { \mathrm { v b } } [ 4 3 ]$ </td><td></td><td></td><td></td></tr><tr><td>DDPM  $L _ { \mathbf { v b } } \left[ 1 9 \right]$ </td><td> $7 . 6 7 \pm 0 . 1 3$ </td><td>13.51</td><td>3.70</td></tr><tr><td> $\mathrm { D D P M } \ L _ { \mathrm { s i m p l e } } \ [ 1 9 ]$ </td><td> $9 . 4 6 \pm 0 . 1 1$ </td><td>3.17</td><td>3.75</td></tr><tr><td>Improved DDPM  $L _ { \mathbf { v b } } \ [ 3 0 ]$ </td><td></td><td>11.47</td><td>2.94</td></tr><tr><td>Improved DDPM  $L _ { \mathrm { s i m p l e } } \ [ 3 0 ]$ </td><td></td><td>2.90</td><td></td></tr><tr><td>DDPM++ cont [47]</td><td></td><td>2.92</td><td>2.99</td></tr><tr><td>NCSN++ cont. [47]</td><td>9.89</td><td>2.20</td><td></td></tr><tr><td>D3PM uniform  $L _ { \mathbf { v b } }$ </td><td> $5 . 9 9 \pm 0 . 1 4$ </td><td> $5 1 . 2 7 \pm 2 . 1 5$ </td><td> $\leq 5 . 0 8 \pm 0 . 0 2$ </td></tr><tr><td>D3PM absorbing  $L _ { \mathbf { v b } }$ </td><td> $6 . 2 6 \pm 0 . 1 0$ </td><td> $4 1 . 2 8 \pm 0 . 6 5$ </td><td> $\leq 4 . 8 3 \pm 0 . 0 2$ </td></tr><tr><td>D3PM absorbing  $L _ { \lambda = 0 . 0 0 1 }$ </td><td> $6 . 7 8 \pm 0 . 0 8$ </td><td> $3 0 . 9 7 \pm 0 . 6 4$ </td><td> $\overline { { \leq } } ~ 4 . 4 0 \pm 0 . 0 2$ </td></tr><tr><td>D3PM Gauss  $L _ { \mathbf { v b } }$ </td><td> $7 . 7 5 \pm 0 . 1 3$ </td><td> $1 5 . 3 0 \pm 0 . 5 5$ </td><td> $\leq 3 . 9 6 6 \pm 0 . 0 0 5$ </td></tr><tr><td>D3PM Gauss  $L _ { \lambda = 0 . 0 0 1 }$ </td><td> $8 . 5 4 \pm 0 . 1 2$ </td><td> $8 . 3 4 \pm 0 . 1 0$ </td><td> $\leq 3 . 9 7 5 \pm 0 . 0 0 6$ </td></tr><tr><td>D3PM Gauss + logistic  $L _ { \lambda = 0 . 0 0 1 }$ </td><td> $8 . 5 6 \pm 0 . 1 0$ </td><td> $7 . 3 4 \pm 0 . 1 9$ </td><td> $\overline { { \leq } } 3 . 4 3 5 \pm 0 . 0 0 7$ </td></tr></table>

## 6 Image generation

We evaluate the performance of several D3PM models on the task of unconditional image generation with the dataset CIFAR-10 [27]. We follow Ho et al. [19] and use $T = 1 0 0 0$ timesteps for all models and verify that for all models the forward process converges to the stationary distribution within T steps, yielding a value of at most $L _ { T } \approx 1 0 ^ { - 5 }$ bits per dimension. We train three versions of D3PM with different transition matrices: doubly stochastic matrices with uniform transition probabilities (D3PM uniform) [20], transition matrices with an absorbing state located at R, G and B values of 128 (D3PM absorbing) and doubly stochastic discretized Gaussian transition matrices (D3PM Gauss). For the D3PM uniform model we experimented with a linear $\beta _ { t }$ schedule as well as the cosine schedule as proposed in [20], with the cosine schedule producing the best results. For D3PM absorbing we use the schedule $\ddot { \beta _ { t } } = ( T - t + 1 ) ^ { - 1 }$ as also proposed in [43], which corresponds to increasing the probability of being in the absorbing state linearly over time. For D3PM Gauss we use the same linear schedule as in [19]. See Appendix B.1 for more details on the experimental setup.

Table 3 shows that for D3PM models trained with the $L _ { \mathrm { v b } }$ objective, D3PM Gauss performs better than D3PM absorbing and uniform on all metrics: Inception score (IS), Frechet Inception Distance (FID) and negative log-likelihood (NLL). The IS score of the uniform and absorbing D3PM models are comparable, while the FID score and NLL of the D3PM absorbing model are slightly better. We trained both D3PM absorbing and D3PM Gauss with the alternative loss function $L _ { \lambda }$ of (5), and we found $\lambda = 0 . 0 0 1$ to work best. We have also experimented with larger values of λ and a model trained only with the auxiliary denoising term in (5). Although this led to a more rapid increase in performance early on in training, the NLL leveled off at higher values for larger λ and the FID even started increasing again. The results show that the models trained with $L _ { \lambda }$ perform significantly better than their counterparts trained with $L _ { \mathrm { v b } }$ . One explanation for this boost in performance is that the cross entropy term leads to gradient noise that varies less with the time step t, which is in contrast to the large change in magnitude of the $L _ { t - 1 }$ terms in $L _ { \mathrm { v b } }$ for smaller t, as demonstrated by Nichol and Dhariwal [30]. Finally, we achieve our best results by combining D3PM Gauss trained on $L _ { \lambda }$ with a truncated logistic parameterization of the reverse process distribution $p _ { \theta } ( \widetilde { \pmb x } _ { 0 } | \pmb x _ { t } )$ (D3PM Gauss e+ logistic). Figure 3 shows samples from our best model (D3PM Gauss + logistic), as well as the D3PM absorbing model.

<!-- image-->  
Figure 3: Left: progressive sampling at $t = 1 0 0 0 , 9 0 0 , 8 0 0 , . . . , 0$ for D3PM absorbing (top) and D3PM Gauss + logistic (bottom), trained with $L _ { \lambda }$ loss on CIFAR-10. These samples were cherry picked. Right: (non cherry picked) samples from the D3PM Gauss + logistic model.

## 7 Related Work

Diffusion generative models were first proposed by Sohl-Dickstein et al. [43] and have gained renewed attention recently due to strong results on image and waveform generation [19, 7]. Recent works have proposed improvements for diffusion model training, including importance sampling of the ELBO, better noise schedules [30] and implicit diffusion models [44]. Several works have also drawn connections to score matching [53, 21, 45], leading to improved sampling algorithms in the continuous-time limit [47].

While most works have considered continuous diffusion models, discrete diffusion-like models were described in [43] and applied to text generation and image segmentation data in [20]. Some works [31, 29] have dealt with discrete data by embedding it in continuous space and leveraging Gaussian diffusion, but have not applied this to text. Seff et al. [42] also considered generation of discrete structured objects using a diffusion-like Markov corruption process.

For text, denoising autoencoders have a long history both in representation learning [2, 11] and more recently as generative models [54]. These closely resemble our absorbing state diffusion variants for a particular schedule and transition matrix (see Section 4), although our framing allows us to compute log-likelihoods and experiment with alternative transition matrices. Other works have considered non-autoregressive translation and speech transcription via insertion and deletion [16, 37], masking [14], and iteratively-refined sequence alignments [5, 38].

## 8 Discussion

We have presented D3PMs, a class of models that improves diffusion models for discrete data by defining new kinds of discrete corruption processes. We achieve strong empirical results relative to previous work on discrete diffusion models, even surpassing performance of continuous diffusion models in terms of log-likelihoods for image generation. While these results are promising, one limitation is that—like much other work on non-autoregressive generative models—our models are still inferior to strong autoregressive models like Transformer XL for text generation, and continuous diffusion models still yield stronger results on image quality. We expect that D3PMs can benefit further from the rapid development of continuous diffusion models [47, 30]. For example, further research in alternative losses for D3PM’s can take inspiration from the reweighted $L _ { \mathrm { s i m p l e } }$ objective used in [19], or the resampled variational bound in Nichol and Dhariwal [30]. Furthermore, D3PM’s might benefit from increasing the number of timesteps and a more optimized noise schedule, as discussed in Nichol and Dhariwal [30]. Another limitation comes from the choice of evaluation metrics that we use (and that are standard for evaluation of generative models). Inception score and Frechet Inception Distance are based on neural networks that have been trained on a particular distribution of data, which is not representative for all use-cases, and focusing on average quality metrics may not accurately reflect performance across the wide diversity of settings where these generative models may be applied. This creates a risk of negative social impacts where advances disproportionately favor a subset of the population. Going forward, we are excited about the space of possibilities that arise within the D3PM framework. We have found successes in leveraging the flexibility that comes from defining discrete corruption processes for discrete data, but we believe that there are many more possibilities that make use of richer forms of structure to define even more powerful discrete diffusion models.

## Acknowledgments and Disclosure of Funding

We would like to thank Hugo Larochelle for providing high-level feedback during the project, and Ben Poole for reviewing a draft version of this manuscript. We would also like to thank Julia Kreutzer and Xavier Garcia for helpful conversations about language experiments. We, the authors, declare to have no competing interests. The research conducted for this paper was entirely supported by Google.

## References

[1] Rami Al-Rfou, Dokook Choe, Noah Constant, Mandy Guo, and Llion Jones. Character-Level language modeling with deeper Self-Attention. arXiv preprint arXiv:1808.04444, August 2018.

[2] Yoshua Bengio, Li Yao, Guillaume Alain, and Pascal Vincent. Generalized denoising Auto-Encoders as generative models. arXiv preprint arXiv:1305.6663, May 2013.

[3] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/google/jax.

[4] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. In International Conference on Learning Representations, 2019.

[5] William Chan, Chitwan Saharia, Geoffrey Hinton, Mohammad Norouzi, and Navdeep Jaitly. Imputer: Sequence modelling via imputation and dynamic programming. In International Conference on Machine Learning, pages 1403–1413. PMLR, 2020.

[6] Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony Robinson. One billion word benchmark for measuring progress in statistical language modeling. arXiv preprint arXiv:1312.3005, December 2013.

[7] Nanxin Chen, Yu Zhang, Heiga Zen, Ron J Weiss, Mohammad Norouzi, and William Chan. WaveGrad: Estimating gradients for waveform generation. arXiv preprint arXiv:2009.00713, September 2020.

[8] Xi Chen, Nikhil Mishra, Mostafa Rohaninejad, and Pieter Abbeel. PixelSNAIL: An improved autoregressive generative model. In International Conference on Machine Learning, pages 863–871, 2018.

[9] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.

[10] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. Transformer-XL: Attentive language models beyond a Fixed-Length context. arXiv preprint arXiv:1901.02860, January 2019.

[11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, October 2018.

[12] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using Real NVP. arXiv preprint arXiv:1605.08803, 2016.

[13] W Feller. On the theory of stochastic processes, with particular reference to applications. In Proceedings of the [First] Berkeley Symposium on Mathematical Statistics and Probability. The Regents of the University of California, 1949.

[14] Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer. Mask-Predict: Parallel decoding of conditional masked language models. arXiv preprint arXiv:1904.09324, April 2019.

[15] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, pages 2672–2680, 2014.

[16] Jiatao Gu, Changhan Wang, and Jake Zhao. Levenshtein transformer. arXiv preprint arXiv:1905.11006, May 2019.

[17] Jonathan Heek, Anselm Levskaya, Avital Oliver, Marvin Ritter, Bertrand Rondepierre, Andreas Steiner, and Marc van Zee. Flax: A neural network library and ecosystem for JAX, 2020. URL http://github.com/google/flax.

[18] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In Advances in Neural Information Processing Systems, pages 6626–6637, 2017.

[19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, pages 6840–6851, 2020.

[20] Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Towards non-autoregressive language models. arXiv preprint arXiv:2102.05379, 2021.

[21] Aapo Hyvärinen, Juha Karhunen, and Erkki Oja. Independent component analysis, volume 46. John Wiley & Sons, 2004.

[22] Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. arXiv preprint arXiv:2006.06676v1, 2020.

[23] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

[24] Diederik P Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. In Advances in Neural Information Processing Systems, pages 10215–10224, 2018.

[25] Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. arXiv preprint arXiv:1312.6114, 2013.

[26] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. arXiv preprint arXiv:2009.09761, 2020.

[27] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

[28] Matt Mahoney. Text8 dataset. http://mattmahoney.net/dc/textdata, 2011. Accessed: 2021-5-24.

[29] Gautam Mittal, Jesse Engel, Curtis Hawthorne, and Ian Simon. Symbolic music generation with diffusion models. arXiv preprint arXiv:2103.16091, March 2021.

[30] Alex Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. arXiv preprint arXiv:2102.09672, 2021.

[31] Chenhao Niu, Yang Song, Jiaming Song, Shengjia Zhao, Aditya Grover, and Stefano Ermon. Permutation invariant graph generation via score-based generative modeling. arXiv preprint arXiv:2003.00638, March 2020.

[32] George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. arXiv preprint arXiv:1912.02762, 2019.

[33] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683, 2020.

[34] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International Conference on Machine Learning, pages 1530–1538, 2015.

[35] Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In International Conference on Machine Learning, pages 1278–1286, 2014.

[36] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional networks for biomedical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 234–241. Springer, 2015.

[37] Laura Ruis, Mitchell Stern, Julia Proskurnia, and William Chan. Insertion-deletion transformer. arXiv preprint arXiv:2001.05540, 2020.

[38] Chitwan Saharia, William Chan, Saurabh Saxena, and Mohammad Norouzi. Non-autoregressive machine translation with latent alignments. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1098–1108, 2020.

[39] Tim Salimans and Durk P Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In Advances in Neural Information Processing Systems, pages 901–909, 2016.

[40] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In Advances in Neural Information Processing Systems, pages 2234–2242, 2016.

[41] Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P Kingma. PixelCNN++: Improving the PixelCNN with discretized logistic mixture likelihood and other modifications. In International Conference on Learning Representations, 2017.

[42] Ari Seff, Wenda Zhou, Farhan Damani, Abigail Doyle, and Ryan P Adams. Discrete object generation with reversible inductive construction. arXiv preprint arXiv:1907.08268, July 2019.

[43] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265, 2015.

[44] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021.

[45] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems, pages 11895–11907, 2019.

[46] Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. arXiv preprint arXiv:2006.09011, 2020.

[47] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, November 2020.

[48] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

[49] Dustin Tran, Keyon Vafa, Kumar Agrawal, Laurent Dinh, and Ben Poole. Discrete flows: Invertible generative models of discrete data. In Advances in Neural Information Processing Systems, volume 32, 2019.

[50] Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu. WaveNet: A generative model for raw audio. arXiv preprint arXiv:1609.03499, 2016.

[51] Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. International Conference on Machine Learning, 2016.

[52] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pages 5998–6008, 2017.

[53] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 23(7):1661–1674, 2011.

[54] Alex Wang and Kyunghyun Cho. BERT has a mouth, and it must speak: BERT as a markov random field language model. arXiv preprint arXiv:1902.04094, February 2019.

[55] Yuxin Wu and Kaiming He. Group normalization. In Proceedings of the European Conference on Computer Vision (ECCV), pages 3–19, 2018.

[56] Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016.

[57] Zachary M Ziegler and Alexander M Rush. Latent normalizing flows for discrete sequences. arXiv preprint arXiv:1901.10548, January 2019.

## A Additional details regarding D3PMs

## A.1 Doubly-stochastic matrices

As discussed in Section 3.1, there are two constraints on $Q _ { t }$ that allow it to be used within a D3PM: the rows of $Q _ { t }$ must sum to one to conserve probability mass, and the rows of $\overline { { { \pmb { Q } } } } _ { t } = \pmb { Q } _ { 1 } \pmb { Q } _ { 2 } \dots \pmb { Q } _ { t }$ must converge to a known stationary distribution as t becomes large. Technically, it is also possible to use a learned prior $p _ { \theta } ( { \pmb x } _ { T } )$ , but assuming this is still modeled under a conditional independence assumption, $q ( { \pmb x } _ { T } | { \pmb x } _ { 0 } )$ must still be close to a stationary distribution for the $L _ { T }$ loss term to be small.

One way to ensure that this occurs is to chose $Q _ { t }$ as increasing powers of a doubly stochastic base matrix $Q$ (rows and columns sum to 1) with strictly positive entries. This is enough to ensure that $Q$ is is irreducible and aperiodic and that product $\overline { { \mathbf { Q } } } _ { t }$ converges as $t \to \infty$ to a uniform distribution over all states. To show this, consider $\pi _ { i } = 1 / K$ for $i = 1 , . . . , K$ , and $\textstyle \sum _ { i = 1 } ^ { K } Q _ { i , : } = { \bf 1 }$ and $\textstyle \sum _ { j = 1 } ^ { K } Q _ { : , j } = { \bf 1 }$ then $\begin{array} { r } { [ \pmb { Q } \pmb { \pi } ] _ { i } = \sum _ { j = 1 } ^ { K } \pmb { Q } _ { i , j } \pi _ { j } = 1 / K \sum _ { i = 1 } ^ { K } \pmb { Q } _ { i , j } = 1 / K = \pi _ { i } } \end{array}$ , thus the uniform distribution is an eigenvector of the transition matrix with eigenvalue 1. Convergence to this distribution follows from the Perron-Frobenius theorem for positive square matrices.

More generally, a similar argument shows that even for $Q _ { t }$ that are not powers of the same base matrix, as long as each $Q _ { t }$ is doubly stochastic, irreducible, and aperiodic, the uniform distribution is the only possible stationary distribution, and as long as the second largest eigenvalue of $Q _ { t }$ is bounded below, the cumulative product $\overline { { \mathbf { Q } } } _ { t }$ will converge to the uniform distribution. In practice, we choose $Q _ { t }$ to add more noise as t increases, which ensures that $\overline { { Q } } _ { T }$ is very close to reaching a uniform stationary distribution.

## A.2 More details on possible choices of Markov transition matrices

## A.2.1 Uniform diffusion

The transition matrix described by Sohl-Dickstein et al. [43] for the binary case, and extended by Hoogeboom et al. [20], to the categorical case, can be represented using the following $K \times { \dot { K } }$ transition matrix

$$
\left[ \pmb { Q } _ { t } \right] _ { i j } = \left\{ \begin{array} { l l l } { 1 - \frac { K - 1 } { K } \beta _ { t } } & { \mathrm { i f } } & { i = j } \\ { \frac { 1 } { K } \beta _ { t } } & { \mathrm { i f } } & { i \neq j } \end{array} \right. ,\tag{6}
$$

This transition matrix can also be written as $( 1 - \beta _ { t } ) I + \beta _ { t } \mathbb { 1 } \mathbb { 1 } ^ { T } / K$ , where 1 is a column vector of all ones.

## A.2.2 Diffusion with an absorbing state

For our diffusion models with an absorbing state m, we use the following matrix:

$$
[ \pmb { Q } _ { t } ] _ { i j } = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f } \quad i = j = m } \\ { 1 - \beta _ { t } } & { \mathrm { i f } \quad i = j \neq m } \\ { \beta _ { t } } & { \mathrm { i f } \quad j = m , i \neq m } \end{array} \right.\tag{7}
$$

The transition matrix can also be written as $( 1 - \beta _ { t } ) I + \beta _ { t } \mathbb { 1 } e _ { m } ^ { T }$ , where $e _ { m }$ is a vector with a one on the absorbing state m and zeros elsewhere. Since m is an absorbing state, the corruption process converges not to a uniform distribution but to the point-mass distribution on m.

For text generation, we let m be the [MASK] token at index $K - 1 ;$ this leads to a BERT-like training objective, which masks tokens according to some schedule and learns to denoise them iteratively (see Section 4). For image generation, we set m to the gray RGB pixel (128, 128, 128) at index $\dot { K / / 2 }$

## A.2.3 Discretized Gaussian transition matrices

For our D3PM models applied to ordinal data, inspired by continuous-space diffusion models, we use the following $K \times K$ matrix:

$$
\left[ \pmb { Q } _ { t } \right] _ { i j } = \left\{ \begin{array} { l l } { \frac { \exp \left( - \frac { 4 | i - j | ^ { 2 } } { ( K - 1 ) ^ { 2 } \beta _ { t } } \right) } { \sum _ { n = - ( K - 1 ) } ^ { K - 1 } \exp \left( - \frac { 4 n ^ { 2 } } { ( K - 1 ) ^ { 2 } \beta _ { t } } \right) } } & { \mathrm { i f } \quad i \neq j } \\ { 1 - \sum _ { l = 0 , l \neq i } ^ { K - 1 } [ \pmb { Q } _ { t } ] _ { i l } } & { \mathrm { i f } \quad i = j } \end{array} \right.\tag{8}
$$

Normalization is ensured by assigning the diagonal values to one minus the sum of each row (not including the diagonal entry). Note that due to the normalization of the off-diagonal values over the range $\{ - K + \bar { 1 } , . . . , K - \bar { 1 } \}$ } the sum of each row excluding the diagonal entry is always smaller than 1. The result yields an irreducible doubly stochastic matrix and a forward process with a uniform stationary distribution. Similar to the continuous Gaussian diffusion model, the parameters $\beta _ { t }$ influence the variance of the forward process distributions.

## A.2.4 Structured diffusion in text: using word-embedding distance to introduce locality

For text, we construct a k-nearest neighbor adjacency matrix

$$
[ \mathbf { G } ] _ { i j } = 1 \mathrm { ~ i f ~ } w _ { i } \mathrm { ~ i s ~ a ~ k - n e a r e s t ~ n e i g h b o r ~ o f ~ } w _ { j } \mathrm { ~ e l s e ~ 0 ~ }
$$

constructed from a pre-trained embedding space over the vocabulary. Then we consider a symmetrized adjacency matrix of the form $\mathbf { A } = ( \mathbf { G } + \mathbf { \dot { G } } ^ { T } ) / ( 2 k )$ where k is the number of nearest neighbors of each node, and finally construct a doubly stochastic rate matrix with

$$
[ \pmb { R } ] _ { i j } = \left\{ \begin{array} { l l } { - \sum _ { l \neq i } A _ { i l } } & { \mathrm { i f } \quad i = j } \\ { A _ { i j } } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{9}
$$

Our final transition matrix is constructed as a matrix exponential of this rate matrix:

$$
\mathbf { Q } _ { t } = \exp ( \alpha _ { t } \mathbf { R } ) = \sum _ { n = 0 } ^ { \infty } \frac { \alpha _ { t } ^ { n } } { n ! } R ^ { n }
$$

Since R is symmetric and sums to zero along each row, $\mathbf { Q } _ { t }$ is doubly stochastic, which ensures we have a uniform stationary distribution (as long as G is connected). Increasing $\alpha _ { t }$ over time allows us to add more noise for larger values of t.

Assuming word embeddings are some metric for syntactic or semantic similarity, this results in a corruption process that gradually moves away from the ground-truth sentence, swapping words with nearest-neighbors in embedding space. For character level modeling, this is a graph over characters, which more often transitions for instance from vowels to other vowels than from vowels to consonants. For words, this could transition between semantically similar words.

For example, in Figure 4, we construct the forward process to diffuse from "dog" to "cat" or "cow", which are nearby in embedding space, but not to more distant words. We can either bootstrap this process by updating the transition matrix Q dynamically during training, or use pretrained embeddings; we use pretrained embeddings for all of our experiments.

## A.2.5 Band-diagonal transitions

A class of transition matrices that introduce local, ordinal inductive biases for structured data are banddiagonal transition matrices which only allow the corruption process to transition locally between states and biases the reverse process towards local iterative refinement. For example, in images, this can be used to allow transitions only between adjacent pixel values.

$$
\begin{array} { r } { [ Q _ { t } ] _ { i j } = \left\{ \begin{array} { l l } { \frac { 1 } { K } \beta _ { t } } & { \mathrm { i f } \quad 0 < | i - j | \leq v } \\ { 1 - \sum _ { l \neq i } Q _ { i l } } & { \mathrm { i f } \quad i = j } \end{array} \right. } \end{array}\tag{10}
$$

where v is the number of nonzero off-diagonal elements of Q above (and below) the main diagonal. Note that this is a doubly stochastic matrix, so the stationary distribution is uniform. We do not use these in our experiments.

p:0.01  
dog  
cat  
p:0.005  
cow

Figure 4: Two examples of noise schedules transforming text data. The top is a BERT-like absorbing + uniform diffusion which replaces tokens with [MASK] tokens (and occasionally with any other token, in black). The bottom is nearest-neighbor diffusion in embedding space. At left represents a possible column in the transition matrix.  
<!-- image-->  
Figure 5: The character-level symmetrized 5-NN graph.

## A.2.6 Combinations of absorbing diffusion and other diffusion

A few ablations in Appendix B.2.1 consider transition matrices that combine absorbing-state or nearest-neighbor and uniform D3PM models. For instance, an absorbing-uniform transition matrix can be constructed $\begin{array} { r } { \pmb { Q } = \alpha \mathbb { 1 } e _ { m } ^ { T } + \beta \mathbb { 1 } \mathbb { 1 } ^ { T } / K + ( 1 - \alpha - \beta ) I } \end{array}$ , where $e _ { m }$ is a one-hot vector on the [MASK] token.

## A.3 Generative Masked Language Models are Diffusion Models

Generative Masked Language Models [14, 54] are generative models that generate text from a sequence of [MASK] tokens. These are usually trained by sampling a sequence $\scriptstyle { \mathbf { { \vec { x } } } } _ { 0 }$ , masking tokens according to some schedule, and learning to predict the masked tokens given context. The actual masking procedure can either be done independently, i.e. by masking each token with probability $p = k / \bar { T }$ , like Devlin et al. [11], or by sampling exactly k tokens. The usual objective is5:

$$
\operatorname* { m i n } - \mathbb { E } _ { q ( \pmb { x } _ { 0 } ) } \left[ \mathbb { E } _ { k \in [ 1 . . . | \pmb { x } _ { 0 } | ] } \left[ \frac { 1 } { k } \mathbb { E } _ { \pmb { x } _ { k } \mathrm { w i t h } \ k \mathrm { \ m a s k e d } \mathrm { \ t o k e n s } } \left[ \sum _ { i } \sum _ { \mathrm { w i t h } \{ \pmb { x } _ { k } \} _ { i } = m } \log p _ { \theta } ( [ \pmb { x } _ { 0 } ] _ { i } | \pmb { x } _ { k } ) \right] \right] \right]\tag{11}
$$

where we first sample a datapoint $\scriptstyle { \mathbf { x } } _ { 0 } .$ , sample a number of tokens to mask k (either uniformly or according to some schedule), then mask that many tokens at random and compute a cross entropy loss over those masked tokens. We claim that this training objective is a (reweighted) absorbing-state D3PM objective with a particular noise schedule and the x0-parameterization from 3.3 (and indeed, that any absorbing-state D3PM model with [MASK] as the absorbing state will be a reweighted version of this loss with different weights assigned to different numbers of masked tokens k).

Consider a D3PM with a schedule that masks tokens with probability $\beta _ { t }$ . The reverse process predicts $\widetilde { p } _ { \theta } ( \widetilde { \pmb { x } _ { 0 } } | \pmb { x } _ { t } )$ , then uses the forward process to compute $\begin{array} { r } { p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } ) \propto \sum q ( \pmb { x } _ { t - 1 } , \pmb { x } _ { t } | \widetilde { \pmb { x } _ { 0 } } ) \widetilde { p } _ { \theta } ( \widetilde { \pmb { x } } _ { 0 } | \pmb { x } _ { t } ) } \end{array}$ e fIn the particular case of absorbing-state diffusion, for each masked token $[ { \pmb x } _ { t } ] _ { i } = m$ f in ${ \mathbf { \nabla } } _ { \mathbf { x } _ { t } } .$ e, we thus have

$$
p _ { \theta } ( [ \pmb { x } _ { t - 1 } ] _ { i } | \pmb { x } _ { t } ) \propto \left\{ \begin{array} { l l } { [ \beta _ { t } \prod _ { s < t } ( 1 - \beta _ { s } ) ] \widetilde { p } _ { \theta } ( [ \widetilde { \pmb { x } } _ { 0 } ] _ { i } = [ \pmb { x } _ { 0 } ] _ { i } | \pmb { x } _ { t } ) } & { \mathrm { f o r } [ \pmb { x } _ { t - 1 } ] _ { i } = [ \pmb { x } _ { 0 } ] _ { i } \neq m } \\ { 1 - \prod _ { s < t } ( 1 - \beta _ { s } ) } & { \mathrm { f o r } [ \pmb { x } _ { t - 1 } ] _ { i } = m } \end{array} \right.
$$

We note that for each unmasked token $[ { \pmb x } _ { t } ] _ { i } = [ { \pmb x } _ { 0 } ] _ { i }$ , the KL-divergence is zero since unmasked tokens cannot make any other type of transition other than becoming masked. Also, the term in the KL divergence due to the probability of mask transitions is a constant, since mask transitions are independent of the model parameters θ. Our $L _ { t }$ term is then

$$
D _ { \mathrm { K L } } [ q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } ) | | p \varrho ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } ) ] = - \left[ \beta _ { t } \prod _ { s < t } ( 1 - \beta _ { s } ) \right] \sum _ { i \ : \mathrm { w i t h } \ : [ \pmb { x } _ { t } ] _ { i } = m } \log \widetilde { p } \varrho ( [ \pmb { x } _ { 0 } ] _ { i } | \pmb { x } _ { t } ) + C
$$

where C is independent of θ and the sum is taken over the masked tokens in $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ . For example, if we use $\beta ( t ) = 1 / ( T - t + 1 )$ from Sohl-Dickstein et al. [43], $\begin{array} { r } { \beta _ { t } \prod _ { i = 0 } ^ { t - 1 } ( 1 - \beta _ { i } ) \ = 1 / T } \end{array}$ and $\begin{array} { r } { 1 - \prod _ { i = 0 } ^ { t } ( 1 - \beta _ { i } ) = ( t - 1 ) / T } \end{array}$ , so $q ( [ \pmb { x } _ { t - 1 } ] _ { i } = [ \pmb { x } _ { 0 } ] _ { i } | [ \pmb { x } _ { t } ] _ { i } = m , \pmb { x } _ { 0 } ) = 1 / t$ for non-mask tokens and we can simplify our $L _ { t }$ objective to

$$
D _ { \mathrm { K L } } [ q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } ) | | p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } ) ] = - \left[ \frac { 1 } { t } \sum _ { i \mathrm { ~ w i t h ~ } [ \pmb { x } _ { t } ] _ { i } = m } \log \widetilde { p } _ { \theta } ( [ \pmb { x } _ { 0 } ] _ { i } | \pmb { x } _ { t } ) \right] + C
$$

where $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ masks tokens independently and uniformly with probability $t / T$ . The $L _ { T }$ term in our ELBO is 0 for the $1 / ( T - t + 1 )$ schedule, so the full objective (up to a constant) reduces to

$$
\begin{array} { r } { \mathbb { E } _ { q ( \boldsymbol { x } _ { 0 } ) } \Bigg [ - \displaystyle \sum _ { t = 2 } ^ { T } \frac { 1 } { t } \mathbb { E } _ { q ( \boldsymbol { x } _ { t } | \mathbf { x } _ { 0 } ) } \Big [ \displaystyle \sum _ { i \mathrm { ~ w i t h ~ } [ \boldsymbol { x } _ { t } ] _ { i } = m } \log p _ { \theta } ( [ \boldsymbol { x } _ { 0 } ] _ { i } | \boldsymbol { x } _ { t } ) \Big ] \Bigg ] } \\ { - \mathbb { E } _ { q ( \boldsymbol { x } _ { 1 } | \mathbf { x } _ { 0 } ) } [ \displaystyle \sum _ { i \mathrm { ~ w i t h ~ } [ \boldsymbol { x } _ { 1 } ] _ { i } = m } \log p _ { \theta } ( [ \boldsymbol { x } _ { 0 } ] _ { i } | \boldsymbol { x } _ { 1 } ) ] \Bigg ] } \\ { = - \mathbb { E } _ { q ( \boldsymbol { x } _ { 0 } ) } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \frac { 1 } { t } \mathbb { E } _ { q ( \boldsymbol { x } _ { t } | \mathbf { x } _ { 0 } ) } \Big [ \displaystyle \sum _ { i \mathrm { ~ w i t h ~ } [ \boldsymbol { x } _ { t } ] _ { i } = m } \log p _ { \theta } ( [ \boldsymbol { x } _ { 0 } ] _ { i } | \boldsymbol { x } _ { t } ) ] \Big ] \right] } \end{array}\tag{12}
$$

Note that while this looks very similar to Equation 11 (with each term reweighted by $1 / t ,$ the expected number of masked tokens) it is not exactly identical since masking is computed independently pertoken position (instead of choosing exactly k tokens to mask). This is an entirely practical way to do masking (and indeed some methods implement it this way).

Furthermore, since the masking probability varies linearly as $1 - \prod ( 1 - \beta _ { t } ) = t / T$ , this is very close to uniformly sampling the number of masked tokens $k ,$ , but k is actually drawn from a mixture of binomial distributions, i.e.

$$
= - \mathbb { E } _ { q ( \mathbf { x } _ { 0 } ) } \left[ \mathbb { E } _ { k \in [ 1 . . . | X | ] } \left[ \mathbb { E } _ { \mathbf { x } _ { k } \mathrm { w i t h } \ k \mathrm { m a s k e d ~ t o k e n s } } \left[ \alpha ( k ) \sum _ { \substack { i \mathrm { w i t h } [ \mathbf { x } _ { k } ] _ { i } = m } } \log p _ { \theta } ( [ \mathbf { x } _ { 0 } ] _ { i } | \mathbf { x } _ { k } ) \right] \right] \right] .\tag{13}
$$

$$
\alpha ( k ) = q ( x _ { t } { \mathrm { ~ h a s ~ } } k { \mathrm { ~ m a s k e d ~ t o k e n s } } | x _ { 0 } { \mathrm { ~ h a s ~ } } n { \mathrm { ~ t o k e n s } } ) = { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } { \binom { n } { k } } \left( { \frac { t } { T } } \right) ^ { n - 1 } \left( 1 - { \frac { t } { T } } \right) ^ { n - k }\tag{14}
$$

which is very close to uniform weight over terms, but slightly downweights terms near 0 and T . By upweighting terms near the boundary, you could in theory make this exactly uniform and thus exactly recover Equation 11. For instance, for 50 categories, absorbing-state diffusion produces the weighting shown in Figure 6.

<!-- image-->  
Figure 6: Plot of the probabilities of having k tokens masked out of a length-50 sequence under a D3PM absorbing schedule with $T = 5 0$ steps, which is very similar to the uniform weighting used by Ghazvininejad et al. [14].

## A.4 Scaling to a large number of categories

When the number of categories K is large, it can quickly become impractical to store all of the transition matrices $Q _ { t }$ in memory, as the memory usage grows like $O ( K ^ { \widehat { 2 } } T )$ . And even if there is an algorithm to compute individual step matrices $Q _ { t }$ Qt on demand, it may or may not be possible to do the same for the cumulative products $\overline { { \mathbf { Q } } } _ { t }$ . We propose two approaches to scaling D3PMs to large numbers of categories that ensure cumulative products are efficient: using low-rank corruption and using matrix exponentials.

## A.4.1 Low-rank corruption

In the low-rank case, we consider structuring our transition matrices as

$$
Q _ { t } = \beta _ { t } { \cal A } _ { t } + ( 1 - \beta _ { t } ) { \cal I } ,\tag{15}
$$

where each $\pmb { A } _ { t }$ is a diagonalizable low-rank matrix with the same nonzero eigenvectors. In particular, recall that both absorbing-state diffusion and uniform diffusion have this form: for uniform diffusion, $A _ { t } ^ { \mathrm { u n i f o r m } } = \mathbb { 1 } \mathbb { 1 } ^ { T } / K$ , and for absorbing-state diffusion $A _ { t } ^ { \mathrm { a b s } } = \mathbb { 1 } e _ { m } ^ { T }$ where $e _ { m }$ is a one-hot vector on the absorbing state. Since products of $\pmb { A } _ { t } \mathbf { \ ' } _ { \mathbf { S } }$ are also low rank, the cumulative products $\overline { { \pmb { Q } } } _ { t }$ can be efficiently precomputed and stored using a much smaller amount of memory $O ( r ^ { 2 } T )$ where $r = \mathrm { r a n k } ( A _ { t } )$ .

As an illustrative example, we describe in more detail how to efficiently represent uniform and absorbing-state transition matrices using the low-rank structure.

To compute products of uniform transition matrices (i.e. $\begin{array} { r } { \prod _ { i } ( 1 - \beta _ { i } ) I + \beta _ { i } \mathbb { 1 } \mathbb { 1 } ^ { T } / K ) } \end{array}$ , we can take advantage of the useful fact that products of matrices of the form α ${ \boldsymbol { \mathbf { \ell } } } _ { t } I + \beta \mathbb { 1 } \mathbb { 1 } ^ { T }$ also have this same form: $I ^ { 2 } = I$ and $\left( \beta \mathbb { 1 } \mathbb { 1 } ^ { T } \right) ^ { 2 } = \beta ^ { 2 } K \mathbb { 1 } \mathbb { 1 } ^ { T }$ . We can thus treat this as a formal polynomial in one variable $X = ( \mathbb { 1 } \mathbb { 1 } ^ { T } / K )$ . Then products can be computed as $\prod _ { i } \left[ ( 1 - \beta _ { i } ) + \beta _ { i } X \right]$ over the quotient ring $\mathbb { R } [ X ] / ( X ^ { 2 } - X )$ , since $X ^ { 2 } = X$ . Functionally, this means you can instantiate a polynomial $( 1 - \beta _ { i } ) \stackrel { \cdot } { + } \beta _ { i } X$ and repeatedly perform ordinary polynomial multiplication over R[X] for the $t < T$ timesteps. After each multiplication, the higher-order terms are reduced by $X ^ { 2 ^ { \circ } } = X$ , leaving a polynomial of degree 1 where the X term has coefficient given by the sum of all higher-order terms. This can be computed with the convenient np.polynomial module.

Similarly, the transition matrices for D3PM absorbing can be computed in closed form. Fundamentally, in each step, we transition to a [MASK] token with probability $\beta _ { t }$ and stay the same with probability $1 - \beta _ { t }$ . Since the [MASK] state is absorbing, after t steps, the only operative quantity is the probability of not yet having transitioned to the [MASK] state, given by $\begin{array} { r } { \widetilde { \alpha _ { t } } = \prod _ { i = 0 } ^ { t } ( 1 - \beta _ { i } ) } \end{array}$ Hence for D3PM absorbing, $\overline { { Q } } = \widetilde { \alpha } _ { t } I + \left( 1 - \widetilde { \alpha _ { t } } \right) \mathbb { 1 } e _ { m } ^ { T }$ where $e _ { m }$ is a one-hot vector on the [MASK] token.

## A.4.2 Matrix exponentials

In the matrix exponential case, we specify our transition matrices as

$$
\begin{array} { r } { Q _ { t } = \exp ( \alpha _ { t } R ) = \displaystyle \sum _ { n = 0 } ^ { \infty } \frac { \alpha _ { t } ^ { n } } { n ! } R ^ { n } , \qquad \overline { { Q } } _ { t } = \exp \left( \left( \sum _ { s \leq t } \alpha _ { s } \right) R \right) , } \end{array}\tag{16}
$$

where R is a transition rate matrix and exp denotes the matrix exponential operation; the similar form for $Q _ { t }$ and $\overline { { \pmb { Q } } } _ { t }$ is a consequence of the “exponential of sums” property for commuting matrices. For efficiency, we further assume that each of the $\alpha _ { t }$ is an integer multiple $n _ { t } \alpha$ ? of some common factor $\alpha _ { \star }$ , and precompute matrices exp $( 2 ^ { k } \alpha _ { \star } R )$ for $0 \leq k \leq \bar { \log _ { 2 } ( \bar { \alpha } _ { T } / \bar { \alpha } _ { \star } ) }$ , where $\begin{array} { r } { \overline { { \alpha } } _ { T } = \sum _ { t < T } \alpha _ { t } } \end{array}$ taking space ${ \cal O } ( K ^ { 2 } \log ( \overline { { \alpha } } _ { T } / \alpha _ { \star } ) )$ . Then, to compute matrix-vector products with $Q _ { t }$ or $Q _ { t }$ , we can iteratively take products with a subset of these precomputed matrices based on the digits of a binary expansion of the desired multiple $n _ { t }$ in time $\dot { O ( K ^ { 2 } \log ( \overline { { \alpha } } _ { T } / \alpha _ { \star } ) ) } . ^ { 6 }$

As long as R has non-positive off-diagonal entries and sums to zero along each row, the matrix exponential produces a valid transition matrix $Q _ { t } ;$ convergence to a specific stationary distribution can also be ensured by controlling the eigenvectors. In particular, if every column also sums to zero, the resulting $Q _ { t }$ will be doubly stochastic and will thus have a uniform stationary distribution.

We note that this parameterization can be viewed as a discretization of a continuous-time discretespace Markov processes; we describe this connection in more detail in the following section.

## A.5 Continuous-time Markov process transition rates

Following Feller [13], we define a continuous-time discrete-space Markov process as a collection of random variables $\{ \pmb { x } _ { t } \} _ { t > 0 }$ parameterized by $t \in \mathbb { R } ^ { + }$ and characterized by a Markov property $( { \pmb x } _ { t } \perp { \pmb x } _ { s } \mid { \pmb x } _ { \tau } { \mathrm { ~ i f ~ } } t < \bar { \tau } < \bar { s } )$ , a transition probability matrix ${ \Pi } ( t ) \in \mathbb { R } ^ { N \times N }$ where N is the cardinality of ${ \mathbf { } } x _ { t } ,$ and a set of transition rates $\gamma _ { i } ( t )$

A conceptual way to understand these processes is to imagine a continuous Poisson process occurring in each state i at rate $\gamma _ { i } ( t )$ determining when a transition between states occurs. When a transition occurs (at time $t ) ,$ , a Markov transition occurs between states i and $j$ with probability $\Pi _ { i j } ( t )$ . Many common stochastic processes fall into this family, including Poisson processes. Like in the case of stochastic differential equations (Song et al. [47]), we can derive a set of Kolomogorov equations (or Fokker-Planck equations in the continuous-state space case) that determine the marginal probability $\partial q _ { i j } ( \tau , t )$ of ending up in state $j$ at time t having started in state i at time s. The general form of the Kolmogorov forward equations is

$$
\frac { \partial q _ { i j } ( \tau , t ) } { \partial t } = - \gamma _ { k } ( t ) q _ { i } ( \tau , t ) + \sum _ { j } \gamma _ { j } ( t ) \Pi _ { k j } ( t ) q _ { i k } ( t )
$$

Now we can state and prove a theorem connecting continuous time Markov processes and matrix exponentials.

Theorem 1. Let $\{ { \pmb x } _ { t } \} _ { t \ge 0 }$ be a discrete-space, continuous-time Markov process with (possibly timedependent) transition probability matrix Π(t) and transition rates $\gamma _ { i } ( t )$ . Then for a particle with an initial distribution $q ( { \pmb x } _ { s } )$ at time $s ,$ the probability of ending in state j at time t is

$$
q ( { \pmb x } _ { t } | { \pmb x } _ { s } ) = \exp \left( \int _ { s } ^ { t } \mathrm { d i a g } ( { \pmb \gamma } ( \tau ) ) ( \Pi ( \tau ) - I ) d \tau \right) q ( { \pmb x } _ { s } )
$$

where exp is the matrix exponential and we view $q ( \pmb { x } _ { t } )$ and $\gamma ( t )$ as vectors in $\mathbb { R } ^ { N }$

Proof (sketch). From the Kolmogorov equations for continuous-time Markov processes, we have the ODE

$$
\frac { \partial q ( \pmb { x } _ { t } | \pmb { x } _ { s } ) } { \partial t } = \mathrm { d i a g } ( \gamma ( t ) ) ( \Pi ( t ) - I ) q ( \pmb { x } _ { t } | \pmb { x } _ { s } )
$$

where $\Pi ( t )$ is the transition probability matrix. Solving this as a first-order ODE using integrating factors yields the desired equation. □

We note that, if Π(t) = Π is independent of t and $\gamma ( s ) = \gamma ( s ) \mathbf { r }$ for some scalar function $\gamma : \mathbb { R } $ R and vector $\mathbf { r } \in \mathbb { R } ^ { N }$ , this simplifies to exactly our matrix exponential parameterization with

$$
\mathbf { R } = \mathrm { d i a g } ( \mathbf { r } ) ( \Pi - I ) .
$$

where we set

$$
\alpha _ { t } = \int _ { t - 1 } ^ { t } \gamma ( t ) d t .
$$

In other words, the $\alpha _ { t }$ parameters in Equation 16 correspond to a discretization of the cumulative transition rate of a continuous-time process.

## A.6 Continuous-limit of schedule from Sohl-Dickstein et al. [43]

Consider for example the schedule described by Sohl-Dickstein et al. [43] for Bernoulli variables $\beta _ { t } = 1 / ( T - t + \bar { 1 } )$ , i.e. the Bernoulli variable would stay the same with probability $1 - \beta _ { t } =$ $( T - t ) / ( \dot { T } - t + 1 )$ and transition with probability $\beta _ { t }$ . In this section, we show that a D3PM absorbing or D3PM uniform process with this schedule is exactly a discretization of a continuous-time jump process of the form described in Theorem 1.

We start by observing that both absorbing-state and uniform D3PM transition matrices can be expressed equivalently as matrix exponentials. In the uniform case, we have

$$
Q _ { t } = \exp ( \alpha _ { t } \mathbf { R } _ { \mathrm { u n i f } } ) = \exp \left( \alpha _ { t } \left( \frac { 1 } { K } \mathbb { 1 } \mathbb { 1 } ^ { T } - I \right) \right) = \exp ( - \alpha _ { t } ) I + ( 1 - \exp ( - \alpha _ { t } ) ) \frac { 1 } { K } \mathbb { 1 } \mathbb { 1 } ^ { T } ,
$$

and in the absorbing case we have

$$
Q _ { t } = \exp ( \alpha _ { t } \mathbf { R } _ { \mathrm { a b s } } ) = \exp \left( \alpha _ { t } \left( \mathbb { 1 } \mathbf { e } _ { m } ^ { T } - I \right) \right) = \exp ( - \alpha _ { t } ) I + ( 1 - \exp ( - \alpha _ { t } ) ) \mathbb { 1 } \mathbf { e } _ { m } ^ { T } .
$$

In either case, by setting this equal to the explicit forms in Appendix $\mathsf { A } . 2 .$ , we obtain the relationship

$$
\beta _ { t } = 1 - \exp ( - \alpha _ { t } )
$$

where $\beta _ { t }$ is defined as in Appendix $\mathbf { A . } 2 ,$ , and $\alpha _ { t }$ is the matrix exponential coefficient as used in the previous section. Using the correspondence discussed in the previous section, we also know

$$
\alpha _ { t } = \int _ { t - 1 } ^ { t } \gamma ( s ) d s
$$

for the continuous-time transition rate function $\gamma ( s )$ . Defining $\beta _ { t } = 1 / ( T - t + 1 )$ , we have

$$
1 - \beta _ { t } = 1 - { \frac { 1 } { ( T - t + 1 ) } } = { \frac { T - t } { T - t + 1 } } = \exp \left( - \int _ { t - 1 } ^ { t } \gamma ( \tau ) d \tau \right)
$$

Denoting the anti-derivative $\textstyle \int \gamma ( t ) = F ( t )$ , we have $\log ( T - t ) - \log ( T - t + 1 ) = - F ( t ) + F ( t - 1 )$ so we can deduce $F ( t ) = - \log ( T - t )$ (up to a constant offset). Taking a derivative then yields $\gamma ( t ) = 1 / ( T - t )$ , which has the same form as the original schedule but is now interpreted as a continuously-varying rate function instead of a probability (and is also shifted by 1 unit in time). Intuitively, we can interpret this as a schedule which assigns uniform probability of a transition occurring over the remaining time, but instead of dividing it between $T - t + 1$ discrete steps, we divide it across a continuous interval of size $T - t$ . We note that using larger values of $T$ is equivalent to performing a finer discretization on a scaled version of this continuous-time process.