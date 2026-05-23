# Structured Denoising Diffusion Models in Discrete State-Spaces

Jacob Austin∗, Daniel D. Johnson∗, Jonathan Ho, Daniel Tarlow & Rianne van den Berg† 

Google Research, Brain Team 

{jaaustin,ddjohnson,jonathanho,dtarlow,riannevdberg}@google.com 

# Abstract

Denoising diffusion probabilistic models (DDPMs) [19] have shown impressive results on image and waveform generation in continuous state spaces. Here, we introduce Discrete Denoising Diffusion Probabilistic Models (D3PMs), diffusionlike generative models for discrete data that generalize the multinomial diffusion model of Hoogeboom et al. [20], by going beyond corruption processes with uniform transition probabilities. This includes corruption with transition matrices that mimic Gaussian kernels in continuous space, matrices based on nearest neighbors in embedding space, and matrices that introduce absorbing states. The third allows us to draw a connection between diffusion models and autoregressive and mask-based generative models. We show that the choice of transition matrix is an important design decision that leads to improved results in image and text domains. We also introduce a new loss function that combines the variational lower bound with an auxiliary cross entropy loss. For text, this model class achieves strong results on character-level text generation while scaling to large vocabularies on LM1B. On the image dataset CIFAR-10, our models approach the sample quality and exceed the log-likelihood of the continuous-space DDPM model. 

# 1 Introduction

Generative modeling is a core problem in machine learning, useful both for benchmarking our ability to capture statistics of natural datasets and for downstream applications that require generating high-dimensional data like images, text, and speech waveforms. There has been a great deal of progress with the development of methods like GANs [15, 4], VAEs [25, 35], large autoregressive neural network models [51, 50, 52], normalizing flows [34, 12, 24, 32], and others, each with their own tradeoffs in terms of sample quality, sampling speed, log-likelihoods, and training stability. 

Recently, diffusion models [43] have emerged as a compelling alternative for image [19, 46] and audio [7, 26] generation, achieving comparable sample quality to GANs and log-likelihoods comparable to autoregressive models with fewer inference steps. A diffusion model is a parameterized Markov chain trained to reverse a predefined forward process, which is a stochastic process constructed to gradually corrupt training data into pure noise. Diffusion models are trained using a stable objective closely related to both maximum likelihood and score matching [21, 53], and they admit faster sampling than autoregressive models by using parallel iterative refinement [30, 45, 47, 44]. 

Although diffusion models have been proposed in both discrete and continuous state spaces [43], most recent work has focused on Gaussian diffusion processes that operate in continuous state spaces (e.g. for real-valued image and waveform data). Diffusion models with discrete state spaces have been explored for text and image segmentation domains [20], but they have not yet been demonstrated as a competitive model class for large scale text or image generation. 

35th Conference on Neural Information Processing Systems (NeurIPS 2021). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-23/872c232e-9bfa-45f5-ae63-3428b7b2df0e/25e2dbae60940dfc3c487fb618284be478e5ca033c1fb752614b159eff1af1e6.jpg)



Figure 1: D3PM forward and (learned) reverse process applied to a quantized swiss roll. Each dot represents a 2D categorical variable. Top: samples from the uniform, discretized Gaussian, and absorbing state D3PM model forward processes, along with corresponding transition matrices $Q$ . Bottom: samples from a learned discretized Gaussian reverse process.


Our aim in this work is to improve and extend discrete diffusion models by using a more structured categorical corruption process to shape data generation, as illustrated in Figure 1. Our models do not require relaxing or embedding discrete data (including images) into continuous spaces, and can embed structure or domain knowledge into the transition matrices used by the forward process. We achieve significantly improved results by taking advantage of this flexibility. We develop structured corruption processes appropriate for text data, using similarity between tokens to enable gradual corruption and denoising. Expanding further, we also explore corruption processes that insert [MASK] tokens, which let us draw parallels to autoregressive and mask-based generative models. Finally, we study discrete diffusion models for quantized images, taking inspiration from the locality exploited by continuous diffusion models. This leads to a particular choice of discrete corruption process that diffuses preferentially to more similar states and leads to much better results in the image domain. 

Overall, we make a number of technical and conceptual contributions. Beyond designing several new structured diffusion models, we introduce a new auxiliary loss which stabilizes training of D3PMs and a family of noise schedules based on mutual information that lead to improved performance. We strongly outperform various non-autoregressive baselines for text generation on character-level text generation, and successfully scale discrete diffusion models to large vocabularies and long sequence lengths. We also achieve strong results on the image dataset CIFAR-10, approaching or exceeding the Gaussian diffusion model from Ho et al. [19] on log-likelihoods and sample quality. 

# 2 Background: diffusion models

Diffusion models [43] are latent variable generative models characterized by a forward and a reverse Markov process. The forward process $\begin{array} { r } { q ( \pmb { x } _ { 1 : T } | \pmb { x } _ { 0 } ) = \prod _ { t = 1 } ^ { T } q ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } ) } \end{array}$ corrupts the data $\mathbf { \delta } _ { \mathbf { \mathcal { X } } 0 } \sim \mathbf { \delta } $ $q ( { \pmb x } _ { 0 } )$ into a sequence of increasingly noisy latent variables x ${ \bf \Phi } _ { 1 : T } = { \bf x } _ { 1 } , { \bf x } _ { 2 } , . . . , { \bf x } _ { T }$ . The learned reverse Markov process $\begin{array} { r } { p _ { \theta } ( \pmb { x } _ { 0 : T } ) = p ( \pmb { x } _ { T } ) \prod _ { t = 1 } ^ { T } p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } ) } \end{array}$ gradually denoises the latent variables towards the data distribution. For example, for continuous data, the forward process typically adds Gaussian noise, which the reverse process learns to remove. 

In order to optimize the generative model $p _ { \theta } ( { \pmb x } _ { 0 } )$ to fit the data distribution $q ( { \pmb x } _ { 0 } )$ , we typically optimize a variational upper bound on the negative log-likelihood: 

$$
\begin{array}{l} L _ {\mathrm{vb}} = \mathbb {E} _ {q (\boldsymbol {x} _ {0})} \Big [ \underbrace {D _ {\mathrm{KL}} [ q (\boldsymbol {x} _ {T} | \boldsymbol {x} _ {0}) | | p (\boldsymbol {x} _ {T}) ]} _ {L _ {T}} + \sum_ {t = 2} ^ {T} \underbrace {\mathbb {E} _ {q (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0})} \big [ D _ {\mathrm{KL}} [ q (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t} , \boldsymbol {x} _ {0}) | | p _ {\theta} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}) ] \big ]} _ {L _ {t - 1}} \\ \left. \underbrace {- \mathbb {E} _ {q \left(\boldsymbol {x} _ {1} \mid \boldsymbol {x} _ {0}\right)} \left[ \log p _ {\theta} \left(\boldsymbol {x} _ {0} \mid \boldsymbol {x} _ {1}\right) \right]} _ {L _ {0}} \right]. \tag {1} \\ \end{array}
$$

When the number of time steps $T$ goes to infinity, both the forward process and the reverse process share the same functional form [13], allowing the use of a learned reverse process from the same class of distributions as that of the forward process. Furthermore, for several choices of the forward process the distribution $q ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ converges to a stationary distribution $\pi ( { \pmb x } )$ in the limit $t \to \infty$ independent of the value of $\scriptstyle { \mathbf { { \vec { x } } } } _ { 0 }$ . When the number of time steps $T$ is large enough and we choose $\pi ( { \pmb x } )$ as the prior $p ( { \pmb x } _ { T } )$ , we can guarantee that the $L _ { T }$ term in (1) will approach zero regardless of the data distribution $q ( \pmb { x } _ { 0 } )$ . (Alternatively, one can use a learned prior $p _ { \theta } ( { \pmb x } _ { T } ) . )$ 

While $q \big ( \mathbf { \boldsymbol { x } } _ { t } | \mathbf { \boldsymbol { x } } _ { t - 1 } \big )$ can in theory be arbitrary, efficient training of $p _ { \theta }$ is possible when $q \big ( \mathbf { \boldsymbol { x } } _ { t } | \mathbf { \boldsymbol { x } } _ { t - 1 } \big )$ : 

1. Permits efficient sampling of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ from $q ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ for an arbitrary time t, allowing us to randomly sample timesteps and optimize each $L _ { t - 1 }$ term individually with stochastic gradient descent, 

2. Has a tractable expression for the forward process posterior $q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } )$ , which allows us to compute the KL divergences present in the $L _ { t - 1 }$ term of (1). 

The majority of recent work in continuous spaces [19, 44, 7, 30] defines the forward√ and reverse distributions as $\begin{array} { r l r } { q ( { \pmb x } _ { t } | { \pmb x } _ { t - 1 } ) } & { = } & { \sqrt { { \bf \alpha } } \sqrt { { \bf \alpha } } \big ( { \pmb x } _ { t } | \sqrt { 1 - \beta _ { t } } { \pmb x } _ { t - 1 } , \beta _ { t } { \pmb I } \big ) } \end{array}$ and $\begin{array} { r l } { p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } ) } & { = } \end{array}$ $\mathcal { N } \left( \boldsymbol { x } _ { t - 1 } | \mu _ { \boldsymbol { \theta } } ( \boldsymbol { x } _ { t } , t ) , \boldsymbol { \Sigma } _ { \boldsymbol { \theta } } ( \boldsymbol { x } _ { t } , t ) \right)$ , respectively. The aforementioned properties hold in the case of these Gaussian diffusion models: the forward process $q ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ ) converges to a stationary distribution, motivating the choice $p ( { \pmb x } _ { T } ) = \mathcal { N } \left( { \pmb x } _ { T } | \mathbf { 0 } , I \right)$ , and both $q ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ and $q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } )$ are tractable Gaussian distributions for which the KL divergence can be computed analytically. 

# 3 Diffusion models for discrete state spaces

Diffusion models with discrete state spaces were first introduced by Sohl-Dickstein et al. [43], who considered a diffusion process over binary random variables. Hoogeboom et al. [20] extended the model class to categorical random variables with transition matrices characterized by uniform transition probabilities. In their supplementary material, Song et al. [44] also derived this extension, although no experiments were performed with this model class. Here, we briefly describe a more general framework for diffusion with categorical random variables which includes these models as special cases. 

For scalar discrete random variables with K categories $x _ { t } , x _ { t - 1 } \in { 1 , . . . , K }$ the forward transition probabilities can be represented by matrices: $[ Q _ { t } ] _ { i j } = q ( x _ { t } = j | x _ { t - 1 } = i )$ . Denoting the one-hot version of x with the row vector x, we can write 

$$
q (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {t - 1}) = \operatorname{Cat} (\boldsymbol {x} _ {t}; \boldsymbol {p} = \boldsymbol {x} _ {t - 1} \boldsymbol {Q} _ {t}), \tag {2}
$$

where $\operatorname { C a t } ( \pmb { x } ; \pmb { p } )$ is a categorical distribution over the one-hot row vector x with probabilities given by the row vector ${ \mathbf { } } p ,$ and ${ \pmb x } _ { t - 1 } { \pmb Q } _ { t }$ t is to be understood as a row vector-matrix product. We assume that $Q _ { t }$ is applied to each pixel of an image or each token in a sequence independently, and that q factorizes over these higher dimensions as well; we thus write $q ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } )$ ) in terms of a single element. Starting from $\scriptstyle { \mathbf { { \vec { x } } } } _ { 0 }$ , we obtain the following t-step marginal and posterior at time $t - 1 \colon$ : 

$$
q (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0}) = \operatorname{Cat} \left(\boldsymbol {x} _ {t}; \boldsymbol {p} = \boldsymbol {x} _ {0} \overline {{\boldsymbol {Q}}} _ {t}\right), \quad \text { with } \quad \overline {{\boldsymbol {Q}}} _ {t} = \boldsymbol {Q} _ {1} \boldsymbol {Q} _ {2} \dots \boldsymbol {Q} _ {t}
$$

$$
q \left(\boldsymbol {x} _ {t - 1} \mid \boldsymbol {x} _ {t}, \boldsymbol {x} _ {0}\right) = \frac {q \left(\boldsymbol {x} _ {t} \mid \boldsymbol {x} _ {t - 1} , \boldsymbol {x} _ {0}\right) q \left(\boldsymbol {x} _ {t - 1} \mid \boldsymbol {x} _ {0}\right)}{q \left(\boldsymbol {x} _ {t} \mid \boldsymbol {x} _ {0}\right)} = \operatorname{Cat} \left(\boldsymbol {x} _ {t - 1}; \boldsymbol {p} = \frac {\boldsymbol {x} _ {t} \boldsymbol {Q} _ {t} ^ {\top} \odot \boldsymbol {x} _ {0} \overline {{\boldsymbol {Q}}} _ {t - 1}}{\boldsymbol {x} _ {0} \overline {{\boldsymbol {Q}}} _ {t} \boldsymbol {x} _ {t} ^ {\top}}\right). \tag {3}
$$

Note that due to the Markov property of the forward process $q ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } , \pmb { x } _ { 0 } ) = q ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } )$ . Assuming that the reverse process $p _ { \theta } ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } )$ is also factorized as conditionally independent over the image or sequence elements, the KL divergence between q and $p _ { \theta }$ can be computed by simply summing over all possible values of each random variable; we thus satisfy criteria 1 and 2 discussed in Section 2. Depending on $Q _ { t }$ , the cumulative products $\overline { { \pmb { Q } } } _ { t }$ can often be computed in closed form, or simply precomputed for all t. However, for large K and large T this may be prohibitive. In Appendix A.4 we discuss how to ensure $\overline { { \pmb { Q } } } _ { t }$ can still be computed efficiently in this case, allowing the framework to scale to a larger number of categories. 

In the next section we discuss the choice of the Markov transition matrices $Q _ { t }$ and corresponding stationary distributions. From here on, we refer to the general class of diffusion models with discrete state spaces as Discrete Denoising Diffusion Probabilistic Models (D3PMs). 

# 3.1 Choice of Markov transition matrices for the forward process

An advantage of the D3PM framework described above is the ability to control the data corruption and denoising process by choosing $Q _ { t } ,$ , in notable contrast to continuous diffusion, for which only additive Gaussian noise has received significant attention. Besides the constraint that the rows of $Q _ { t }$ must sum to one to conserve probability mass, the only other constraint in choosing $Q _ { t }$ is that the rows of $\overline { { { \pmb { Q } } } } _ { t } = \pmb { Q } _ { 1 } \pmb { Q } _ { 2 } \dots \pmb { Q } _ { t }$ must converge to a known stationary distribution3 when t becomes large, which can be guaranteed while imposing minimal restrictions on $Q _ { t }$ (see Appendix A.1). 

We argue that for most real-world discrete data, including images and text, it makes sense to add domain-dependent structure to the transition matrices $Q _ { t }$ as a way of controlling the forward corruption process and the learnable reverse denoising process. Below we briefly discuss the uniform transition matrices that have been studied in prior work [20], along with a set of structured transition matrices we have explored for our image and text dataset experiments; see Appendix A.2 for more details on each matrix type. We also note that this set is not exhaustive, and many other transition matrices could also be used within the D3PM framework. 

Uniform (Appendix A.2.1). Sohl-Dickstein et al. [43] considered a simple $2 \times 2$ transition matrix for binary random variables. Hoogeboom et al. [20] later extended this to categorical variables, proposing a transition matrix $\pmb { Q } _ { t } = ( 1 - \beta _ { t } ) \pmb { I } + \beta _ { t } / K \bar { \parallel } \mathbb { 1 } ^ { T }$ with $\beta _ { t } \in [ 0 , 1 ]$ . Since this transition matrix is doubly stochastic with strictly positive entries, the stationary distribution is uniform. Because the transition probability to any other state is uniform, in this paper we equivalently refer to this discrete diffusion instance as D3PM-uniform. 

Absorbing state (Appendix A.2.2). Motivated by the success of BERT [11] and recent work on Conditional Masked Language Models (CMLMs) in text, we consider a transition matrix with an absorbing state (called [MASK]), such that each token either stays the same or transitions to [MASK] with some probability $\beta _ { t }$ . This does not impose particular relationships between categories, similar to uniform diffusion, but still allows corrupted tokens to be distinguished from original ones. Moreover, the stationary distribution is not uniform but has all the mass on the [MASK] token. For images, we reuse the grey pixel as the [MASK] absorbing token. 

Discretized Gaussian (Appendix A.2.3). Instead of transitioning uniformly to any other state, for ordinal data we propose imitating a continuous space diffusion model by using a discretized, truncated Gaussian distribution. We choose a normalization such that the transition matrix is doubly stochastic, leading to a uniform stationary distribution. This transition matrix will transition between more similar states with higher probability, and is well suited for quantized ordinal data such as images. 

Token embedding distance (Appendix A.2.4). Textual data does not have ordinal structure, but there may still be interesting semantic relationships. For instance, in a character level vocabulary vowels may be more similar to each other than they are to consonants. As a demonstration of the generality of the D3PM framework, we explore using similarity in an embedding space to guide the forward process, and construct a doubly-stochastic transition matrix that transitions more frequently between tokens that have similar embeddings while maintaining a uniform stationary distribution. 

For uniform and absorbing-state diffusion, the cumulative products $\overline { { \mathbf { Q } } } _ { t }$ can be computed in closed form (see Appendix A.4.1); the remainder can be precomputed. 

# 3.2 Noise schedules

We consider several different options for the noise schedule of the forward process. For discretized Gaussian diffusion, we explore linearly increasing the variance of the Gaussian before discretizing it. (Note that a linear schedule for $Q _ { t }$ leads to a nonlinear amount of cumulative noise in $\overline { { Q } } _ { t } . )$ For uniform diffusion we use the cosine schedule which sets the cumulative probability of a transition to a cosine function, as introduced by Nichol and Dhariwal [30] and adapted by Hoogeboom et al. [20]. For a general set of transition matrices $Q _ { t }$ (such as the one based on token embeddings), previously proposed schedules may not be directly applicable. We consider linearly interpolating the mutual information between $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ and $\scriptstyle { \pmb x } _ { 0 }$ to zero, i.e. $\begin{array} { r } { I ( \pmb { x } _ { t } ; \pmb { x } _ { 0 } ) \approx ( 1 - \frac { t } { T } ) H ( \pmb { \bar { x } } _ { 0 } ) } \end{array}$ ). Interestingly, for the specific case of absorbing-state D3PMs, this schedule reduces to exactly the $( T - t + 1 ) ^ { - 1 }$ schedule proposed by Sohl-Dickstein et al. [43] for a Bernoulli diffusion process. See Appendix A.7 for more details. 

# 3.3 Parameterization of the reverse process

While it is possible to directly predict the logits of $p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } )$ using a neural network $\mathrm { n n } _ { \theta } ( { \pmb x } _ { t } )$ , we follow Ho et al. [19] and Hoogeboom et al. [20] and focus on using a neural network $\mathrm { n n } _ { \theta } ( { \pmb x } _ { t } )$ to predict the logits of a distribution $\widetilde { p } _ { \theta } ( \widetilde { \pmb x } _ { 0 } | \pmb x _ { t } )$ , which we combine with $q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } )$ and a esummation over one-hot representations of $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ to obtain the following parameterization 

$$
p _ {\theta} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}) \propto \sum_ {\widetilde {\boldsymbol {x}} _ {0}} q (\boldsymbol {x} _ {t - 1}, \boldsymbol {x} _ {t} | \widetilde {\boldsymbol {x}} _ {0}) \widetilde {p} _ {\theta} (\widetilde {\boldsymbol {x}} _ {0} | \boldsymbol {x} _ {t}). \tag {4}
$$

We note that under this $\scriptstyle { \pmb x } _ { 0 } \cdot$ -parameterization the KL divergence $D _ { \mathrm { K L } } [ q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } ) | | p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } ) ]$ will be zero if $\widetilde { p } _ { \boldsymbol { \theta } } ( \widetilde { \pmb { x } } _ { 0 } | \pmb { x } _ { t } )$ places all of its probability mass on the original value $\mathbf { \delta x } _ { 0 } .$ . The decomposition of $q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } )$ e in (3) also provides us with a motivation for this parameterization. According to (3), in a given state $\mathbf { \Delta } \mathbf { x } _ { t } .$ , the optimal reverse process only takes into account transitions to states for which $q \big ( \mathbf { \boldsymbol { x } } _ { t } | \mathbf { \boldsymbol { x } } _ { t - 1 } \big )$ is non-zero. Therefore, the sparsity pattern of $Q _ { t }$ determines the sparsity pattern of the ideal reverse transition probabilities in $p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } )$ . The parameterization in (4) automatically ensures that the learned reverse probability distribution $\dot { p } _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } )$ has the correct sparsity pattern dictated by the choice of the Markov transition matrix $Q _ { t }$ . This parameterization also lets us perform inference with k steps at a time, by predicting $\begin{array} { r } { p _ { \theta } ( \pmb { x } _ { t - k } | \pmb { x } _ { t } ) = \sum q ( \pmb { x } _ { t - k } , \pmb { x } _ { t } | \widetilde { \pmb { x } } _ { 0 } ) \widetilde { p _ { \theta } } ( \widetilde { \pmb { x } } _ { 0 } | \pmb { x } _ { t } ) } \end{array}$ . 

Finally, when modeling ordinal discrete data, instead of predicting the logits of $\widetilde { p } _ { \theta } ( \widetilde { \pmb { x } } _ { 0 } | \pmb { x } _ { t } )$ directly e ewith the output of a neural net, another option is to model the probabilities with a truncated discretized logistic distribution (see Appendix A.8). This provides an extra ordinal inductive bias to the reverse model and boosts FID and log-likelihood scores for images. 

# 3.4 Loss function

While the original diffusion models introduced by Sohl-Dickstein et al. [43] were optimized with the negative variational lower bound $L _ { \mathrm { v b } }$ of (1), more recent diffusion models are optimized with different objectives. For instance, Ho et al. [19] derive a simplified loss function $( L _ { \mathrm { s i m p l e } } )$ that reweights the negative variational bound, and Nichol and Dhariwal [30] explore a hybrid loss $L _ { \mathrm { h y b r i d } } ~ = ~ L _ { \mathrm { s i m p l e } } + \lambda L _ { \mathrm { v b } }$ (using one term to learn the predicted mean and the other to learn predicted variance). Inspired by this recent work, we introduce an auxiliary denoising objective for the $\scriptstyle { \pmb { x } } _ { 0 } \cdot$ -parameterization of the reverse process, which encourages good predictions of the data $\scriptstyle { \mathbf { { \vec { x } } } } _ { 0 }$ at each time step. We combine this with the negative variational lower bound, yielding the following alternative loss function: 

$$
L _ {\lambda} = L _ {\mathrm{vb}} + \lambda \mathbb {E} _ {q (\boldsymbol {x} _ {0})} \mathbb {E} _ {q (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0})} [ - \log \widetilde {p} _ {\theta} (\boldsymbol {x} _ {0} | \boldsymbol {x} _ {t}) ]. \tag {5}
$$

Note that the auxiliary loss coincides with the cross entropy term $L _ { 0 }$ in (1) at $t ~ = ~ 1$ . Furthermore, due to the x0-parameterization of $p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } )$ , both the auxiliary loss term and $D _ { \mathrm { K L } } [ q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } ) | | p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } ) ]$ in $L _ { \mathrm { v b } }$ are minimized exactly when $\widetilde { p } _ { \theta } ( \widetilde { \pmb { x } } _ { 0 } | \pmb { x } _ { t } )$ has all its mass on the datapoint $\scriptstyle { \mathbf { { \vec { x } } } } _ { 0 }$ e e. We find that training with this loss leads to improved quality of image samples. 

# 4 Connection to existing probabilistic models for text

In this section we expand on interesting connections between the D3PM framework and several existing probabilistic and language modeling approaches. 

BERT is a one-step diffusion model: One possible D3PM transition matrix is a combination of a uniform transition matrix and an absorbing state at the [MASK] token (i.e. $Q = \alpha \mathbb { 1 } e _ { m } ^ { T } + \beta \mathbb { 1 } \mathbb { 1 } ^ { T } / K +$ $( 1 - \alpha - \beta ) I$ , where $e _ { m }$ is a one-hot vector on the [MASK] token). For a one-step diffusion process in which $q ( \pmb { x } _ { 1 } | \pmb { x } _ { 0 } )$ replaces 10% of tokens with [MASK] and 5% uniformly at random, this leads precisely to the BERT denoising objective, i.e. $L _ { v b } - L _ { T } = - \mathbb { E } _ { q ( \pmb { x } _ { 1 } | \pmb { x } _ { 0 } ) } [ \log p _ { \theta } ( \pmb { x } _ { 0 } | \pmb { x } _ { 1 } ) ] = L _ { B E R T }$ , since $L _ { T }$ is a constant independent of θ (assuming a fixed prior). 

Autoregressive models are (discrete) diffusion models: Consider a diffusion process that deterministically masks tokens one-by-one in a sequence of length $N = T \colon q ( [ \pmb { x } _ { t } ] _ { i } \mid \mathbf { \dot { x } } _ { 0 } ) = [ \pmb { x } _ { 0 } ] _ { i } \mathrm { i f } i <$ 

$N - t$ else [MASK] . This is a deterministic forward process, so $q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } )$ is a delta distribution on the $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ sequence with one fewer mask: $q ( [ \pmb { x } _ { t - 1 } ] _ { i } \bar { | } \pmb { x } _ { t } , \pmb { x } _ { 0 } ) = \bar { \delta } _ { [ \pmb { x } _ { t } ] _ { i } }$ if $i \neq T - t$ else $\delta _ { [ { \pmb x } _ { 0 } ] _ { i } }$ . While this process is not applied independently to each token, it can be recast as an independently-applied diffusion process on the product space $\lbrack 0 . . . N ] \times \mathcal { V } ,$ , where each token is tagged with its position in the sequence, V is the vocabulary, and Q is an $N \times | \mathcal { V } | \times N \times | \mathcal { V } |$ sparse matrix. 

Because all tokens except the one at position $i = T - t$ have deterministic posteriors, the KL divergence $D _ { K L } \big ( q ( [ \pmb { x } _ { t - 1 } ] _ { j } | \pmb { x } _ { t } , \pmb { x } _ { 0 } ) \big | \big | \big | p _ { \theta } \big ( [ \pmb { x } _ { t - 1 } ] _ { j } | \pmb { x } _ { t } \big ) \big )$ is zero for all other positions. The only token for which this is not true is the token at position i, for which $D _ { K L } \bar { ( } q ( [ \mathbf { x } _ { t - 1 } ] _ { i } | \mathbf { x } _ { t } , \mathbf { x } _ { 0 } )$ || $p _ { \theta } ( [ \pmb { x } _ { t - 1 } ] _ { i } | \pmb { x } _ { t } ) ) = - \log p _ { \theta } ( [ \pmb { x } _ { 0 } ] _ { i } | \pmb { x } _ { t } )$ , the standard cross entropy loss for an autoregressive model. 

(Generative) Masked Language-Models (MLMs) are diffusion models: Generative Masked Language Models ([14], [54]) are generative models that generate text from a sequence of [MASK] tokens. They are usually trained by sampling a sequence ${ \pmb x } _ { 0 } ,$ , masking k tokens according to some schedule, and learning to predict the masked tokens given context. It turns out that a D3PM absorbing ([MASK]) model trained on the usual ELBO objective with the x0-parameterization from 3.3 reduces to a reweighted version of this MLM objective (see Appendix A.3 for a detailed derivation). 

# 5 Text generation

For text, we experiment with generation on two datasets: text8 [28], a character-level dataset extracted from English-language Wikipedia, and the One Billion Word dataset (LM1B) [6], a large dataset of shuffled English-language sentences. For both, we train a D3PM uniform model based on the work by Hoogeboom et al. [20] (D3PM uniform) and a model that masks tokens (D3PM absorbing). We also consider a model that transitions uniformly to nearest neighbors in a token embedding space (D3PM NN). We follow Hoogeboom et al. [20] and use $T = 1 \bar { 0 } 0 0$ timesteps, although we are also able to evaluate on fewer due to the parameterization in Section 3.3. 

# 5.1 Character-level generation on text8

text8 is a character-level text dataset consisting of a small vocabulary of 27 tokens: the letters $\mathbf { \hat { a } } _ { } ^ { \prime } - \mathbf { \hat { z } } _ { } ^ { \prime }$ and the ‘_’ whitespace token. We follow the convention of training and evaluating text8 in chunks of length 256 without any preprocessing [20]. For nearest-neighbor D3PM, our nearest neighbor graph in character-space is shown in Appendix B.2.1. D3PM uniform models were trained with a cosine schedule from Hoogeboom et al. [20] (ablations in Appendix B.2.1), while D3PM absorbing and D3PM NN models were trained with a mutual information schedule. 


Table 1: Quantitative results on text8. NLL is reported on the entire test set. Sample times are for generating a single example of length 256. Results are reported on two seeds. All models are standard 12-layer transformers unless otherwise noted. †Transformer XL is a 24-layer transformer, using a 784 context window. ‡Results reported by [20] by running code from official repository.


<table><tr><td>Model</td><td>Model steps</td><td>NLL (bits/char) (↓)</td><td>Sample time (s) (↓)</td></tr><tr><td>Discrete Flow [49] (8 × 3 layers)</td><td>-</td><td>1.23</td><td>0.16</td></tr><tr><td>Argmax Coupling Flow [20]</td><td>-</td><td>1.80</td><td>0.40 ± 0.03</td></tr><tr><td>IAF / SCF [57]‡</td><td>-</td><td>1.88</td><td>0.04 ± 0.0004</td></tr><tr><td>Multinomial Diffusion (D3PM uniform) [20]</td><td>1000</td><td>≤ 1.72</td><td>26.6 ± 2.2</td></tr><tr><td>D3PM uniform [20] (ours)</td><td>1000</td><td>≤ 1.61 ± 0.02</td><td>3.6 ± 0.4</td></tr><tr><td>D3PM NN (<eq>L_{vb}</eq>) (ours)</td><td>1000</td><td>≤ 1.59 ± 0.03</td><td>3.1474 ± 0.0002</td></tr><tr><td>D3PM mask (<eq>L_{\lambda=0.01}</eq>) (ours)</td><td>1000</td><td>≤ 1.45 ± 0.02</td><td>3.4 ± 0.3</td></tr><tr><td>D3PM uniform [20] (ours)</td><td>256</td><td>≤ 1.68 ± 0.01</td><td>0.5801 ± 0.0001</td></tr><tr><td>D3PM NN (<eq>L_{vb}</eq>) (ours)</td><td>256</td><td>≤ 1.64 ± 0.02</td><td>0.813 ± 0.002</td></tr><tr><td>D3PM absorbing (<eq>L_{\lambda=0.01}</eq>) (ours)</td><td>256</td><td>≤ 1.47 ± 0.03</td><td>0.598 ± 0.002</td></tr><tr><td>Transformer decoder (ours)</td><td>256</td><td>1.23</td><td>0.3570 ± 0.0002</td></tr><tr><td>Transformer decoder [1]</td><td>256</td><td>1.18</td><td>-</td></tr><tr><td>Transformer XL [10]†</td><td>256</td><td>1.08</td><td>-</td></tr><tr><td>D3PM uniform [20] (ours)</td><td>20</td><td>≤ 1.79 ± 0.03</td><td>0.0771 ± 0.0005</td></tr><tr><td>D3PM NN (<eq>L_{vb}</eq>) (ours)</td><td>20</td><td>≤ 1.75 ± 0.02</td><td>0.1110 ± 0.0001</td></tr><tr><td>D3PM absorbing (<eq>L_{\lambda=0.01}</eq>) (ours)</td><td>20</td><td>≤ 1.56 ± 0.04</td><td>0.0785 ± 0.0003</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-23/872c232e-9bfa-45f5-ae63-3428b7b2df0e/bab2c2482823d1405223517c271e44b2e4d5fc59214598a3566b195a8fbf6432.jpg)



t=128[MASK][MASK][MASK][MASK][MASK][MASK]... t =25 In response [MAsK] the demands，[MAsK] [MASK]y Workers union said [MAsK] backflow fund [MAsK]s would face further investigation and a fine. t=0 In response to the demands ，the Community Workers union Original:Caterpillar is eager to expand in Asia ，where it trai local competitors such as Komatsu Ltd Corrupted:Caterpillar is eager to expand in[MAsK]，[MAsK] it [MASK] s local competitors such as Komatsu Ltd Reconstructed:Caterpillar is eager to expand in China ，where i faces local competitors such as Komatsu Ltd



Figure 2: Left: perplexity v.s. sampling iterations for LM1B. Right: Using a trained D3PM absorbing model for LM1B to (top) generate new sentences and (bottom) reconstruct corrupted examples.



Table 2: Quantitative results on LM1B. Perplexity reported on the test set. Results are reported on two seeds. All models have context window length 128 and 12 layers unless otherwise noted. †Transformer XL is a 24 layer transformer. ‡rounded for readability, see Appendix B.2.2.


<table><tr><td>Metric:</td><td colspan="3">Perplexity (↓)</td><td colspan="3">Sample time<eq>^{\ddagger}</eq> (s) (↓)</td></tr><tr><td>inference steps:</td><td>1000</td><td>128</td><td>64</td><td>1000</td><td>128</td><td>64</td></tr><tr><td>D3PM uniform</td><td>137.9 ± 2.1</td><td>139.2 ± 1.2</td><td>145.0 ± 1.2</td><td>1.82</td><td>0.21</td><td>0.08</td></tr><tr><td>D3PM NN</td><td>149.5 ± 1.3</td><td>158.6 ± 2.2</td><td>160.4 ± 1.2</td><td>21.29</td><td>6.69</td><td>5.88</td></tr><tr><td>D3PM absorbing</td><td>76.9 ± 2.3</td><td>80.1 ± 1.2</td><td>83.6 ± 6.1</td><td>1.90</td><td>0.19</td><td>0.10</td></tr><tr><td>Transformer (ours)</td><td>-</td><td>43.6</td><td>-</td><td>-</td><td>0.26</td><td>-</td></tr><tr><td>Transformer XL [10]<eq>^{\dagger}</eq></td><td>-</td><td>21.8</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

Table 1 shows that for D3PM, the D3PM absorbing model performed the best, exceeding the uniform and NN diffusion models. We were able to improve upon the baseline result of [20] with hyperparameter tuning, and our uniform and NN results outperformed results from Hoogeboom et al. [20] across all inference steps, down to as few as 20. We found that $L _ { \lambda = 0 . 0 1 }$ worked best for D3PM absorbing, while $L _ { \mathrm { v b } }$ was better for D3PM uniform. Our model outperforms all nonautoregressive baselines except one, the Discrete Flow model [49] (for which unfortunately no open-source implementations exist), and is also faster than all but one method, the IAF/SCF model [57]. It is also nearly 20x faster than an autoregressive transformer of the same size. We also include a plot of inference time as a function of iterations in Appendix B.2.1. D3PM with the mask absorbing token was by far the best performing model, which lends credibility to the use of masks in denoising auto-encoders. Nearest-neighbor diffusion only narrowly improves upon a D3PM-uniform model: this was a surprising negative result for us, suggesting that not all notions of structure are meaningful. 

# 5.2 Text generation on LM1B

Text generation for large-scale text datasets and large vocabularies with discrete diffusion models has not been previously demonstrated. We include results from LM1B as a proof of concept, showing that these models can indeed scale (as discussed in Appendix A.4), and that the D3PM absorbing model continues to excel. All models were trained and evaluated on packed sequences of length 128, using a sentencepiece4 vocabulary of size 8192. 

Table 2 contains results from experiments on LM1B. Overall, mask diffusion (D3PM absorbing) does relatively well, approaching the performance of a comparable autoregressive model of the same size, and scaling to far fewer steps, while uniform diffusion performs significantly worse. We find, surprisingly, that the D3PM NN model performs worse than the uniform model in terms of log likelihoods (although it demonstrates unique qualitative behavior). This suggests that word embedding similarity may not be a meaningful kind of locality in a diffusion process. We found the the $L _ { \lambda = 0 . 0 1 }$ loss worked best for the mask absorbing model, but reduced performance for the other models. We note the surprising scaling in perplexity in Figure 2, achieving strong results with as few as 10 inference steps. We also show samples from our model and completions from corrupted samples. 


Table 3: Inception scores (IS), Frechet Inception Distance (FID) and negative log-likehood (NLL) on the image dataset CIFAR-10. The NLL is reported on the test set in bits per dimension. We report our results as averages with standard deviations, obtained by training five models with different seeds.


<table><tr><td>Model</td><td>IS (↑)</td><td>FID (↓)</td><td>NLL (↓)</td></tr><tr><td>Sparse Transformer [9]</td><td></td><td></td><td>2.80</td></tr><tr><td>NCSN [45]</td><td><eq>8.87 \pm 0.12</eq></td><td>25.32</td><td></td></tr><tr><td>NCSNv2 [46]</td><td><eq>8.40 \pm 0.07</eq></td><td>10.87</td><td></td></tr><tr><td>StyleGAN2 + ADA [22]</td><td><eq>9.74 \pm 0.05</eq></td><td>3.26</td><td></td></tr><tr><td>Diffusion (original), <eq>L_{vb}</eq> [43]</td><td></td><td></td><td>≤ 5.40</td></tr><tr><td>DDPM <eq>L_{vb}</eq> [19]</td><td><eq>7.67 \pm 0.13</eq></td><td>13.51</td><td>≤ 3.70</td></tr><tr><td>DDPM <eq>L_{simple}</eq> [19]</td><td><eq>9.46 \pm 0.11</eq></td><td>3.17</td><td>≤ 3.75</td></tr><tr><td>Improved DDPM <eq>L_{vb}</eq> [30]</td><td></td><td>11.47</td><td>≤ 2.94</td></tr><tr><td>Improved DDPM <eq>L_{simple}</eq> [30]</td><td></td><td>2.90</td><td>≤ 3.37</td></tr><tr><td>DDPM++ cont [47]</td><td></td><td>2.92</td><td>2.99</td></tr><tr><td>NCSN++ cont. [47]</td><td>9.89</td><td>2.20</td><td></td></tr><tr><td>D3PM uniform <eq>L_{vb}</eq></td><td><eq>5.99 \pm 0.14</eq></td><td><eq>51.27 \pm 2.15</eq></td><td>≤ 5.08 ± 0.02</td></tr><tr><td>D3PM absorbing <eq>L_{vb}</eq></td><td><eq>6.26 \pm 0.10</eq></td><td><eq>41.28 \pm 0.65</eq></td><td>≤ 4.83 ± 0.02</td></tr><tr><td>D3PM absorbing <eq>L_{\lambda=0.001}</eq></td><td><eq>6.78 \pm 0.08</eq></td><td><eq>30.97 \pm 0.64</eq></td><td>≤ 4.40 ± 0.02</td></tr><tr><td>D3PM Gauss <eq>L_{vb}</eq></td><td><eq>7.75 \pm 0.13</eq></td><td><eq>15.30 \pm 0.55</eq></td><td>≤ 3.966 ± 0.005</td></tr><tr><td>D3PM Gauss <eq>L_{\lambda=0.001}</eq></td><td><eq>8.54 \pm 0.12</eq></td><td><eq>8.34 \pm 0.10</eq></td><td>≤ 3.975 ± 0.006</td></tr><tr><td>D3PM Gauss + logistic <eq>L_{\lambda=0.001}</eq></td><td><eq>8.56 \pm 0.10</eq></td><td><eq>7.34 \pm 0.19</eq></td><td>≤ 3.435 ± 0.007</td></tr></table>

# 6 Image generation

We evaluate the performance of several D3PM models on the task of unconditional image generation with the dataset CIFAR-10 [27]. We follow Ho et al. [19] and use $T = 1 0 0 0$ timesteps for all models and verify that for all models the forward process converges to the stationary distribution within T steps, yielding a value of at most $L _ { T } \approx 1 0 ^ { - 5 }$ bits per dimension. We train three versions of D3PM with different transition matrices: doubly stochastic matrices with uniform transition probabilities (D3PM uniform) [20], transition matrices with an absorbing state located at R, G and B values of 128 (D3PM absorbing) and doubly stochastic discretized Gaussian transition matrices (D3PM Gauss). For the D3PM uniform model we experimented with a linear $\beta _ { t }$ schedule as well as the cosine schedule as proposed in [20], with the cosine schedule producing the best results. For D3PM absorbing we use the schedule $\ddot { \beta _ { t } } = ( T - t + 1 ) ^ { - 1 }$ as also proposed in [43], which corresponds to increasing the probability of being in the absorbing state linearly over time. For D3PM Gauss we use the same linear schedule as in [19]. See Appendix B.1 for more details on the experimental setup. 

Table 3 shows that for D3PM models trained with the $L _ { \mathrm { v b } }$ objective, D3PM Gauss performs better than D3PM absorbing and uniform on all metrics: Inception score (IS), Frechet Inception Distance (FID) and negative log-likelihood (NLL). The IS score of the uniform and absorbing D3PM models are comparable, while the FID score and NLL of the D3PM absorbing model are slightly better. We trained both D3PM absorbing and D3PM Gauss with the alternative loss function $L _ { \lambda }$ of (5), and we found $\lambda = 0 . 0 0 1$ to work best. We have also experimented with larger values of λ and a model trained only with the auxiliary denoising term in (5). Although this led to a more rapid increase in performance early on in training, the NLL leveled off at higher values for larger λ and the FID even started increasing again. The results show that the models trained with $L _ { \lambda }$ perform significantly better than their counterparts trained with $L _ { \mathrm { v b } }$ . One explanation for this boost in performance is that the cross entropy term leads to gradient noise that varies less with the time step t, which is in contrast to the large change in magnitude of the $L _ { t - 1 }$ terms in $L _ { \mathrm { v b } }$ for smaller t, as demonstrated by Nichol and Dhariwal [30]. Finally, we achieve our best results by combining D3PM Gauss trained on $L _ { \lambda }$ with a truncated logistic parameterization of the reverse process distribution $p _ { \theta } ( \widetilde { \pmb x } _ { 0 } | \pmb x _ { t } )$ (D3PM Gauss + logistic). Figure 3 shows samples from our best model (D3PM Gauss + logistic), as well as the D3PM absorbing model. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-23/872c232e-9bfa-45f5-ae63-3428b7b2df0e/3e7f99ba22ce3fcfd8591b0239c6a9c2a180d07cb4fec98c2a29ec62e3da54a8.jpg)



Figure 3: Left: progressive sampling at $t = 1 0 0 0 , 9 0 0 , 8 0 0 , . . . , 0$ for D3PM absorbing (top) and D3PM Gauss + logistic (bottom), trained with $L _ { \lambda }$ loss on CIFAR-10. These samples were cherry picked. Right: (non cherry picked) samples from the D3PM Gauss + logistic model.


# 7 Related Work

Diffusion generative models were first proposed by Sohl-Dickstein et al. [43] and have gained renewed attention recently due to strong results on image and waveform generation [19, 7]. Recent works have proposed improvements for diffusion model training, including importance sampling of the ELBO, better noise schedules [30] and implicit diffusion models [44]. Several works have also drawn connections to score matching [53, 21, 45], leading to improved sampling algorithms in the continuous-time limit [47]. 

While most works have considered continuous diffusion models, discrete diffusion-like models were described in [43] and applied to text generation and image segmentation data in [20]. Some works [31, 29] have dealt with discrete data by embedding it in continuous space and leveraging Gaussian diffusion, but have not applied this to text. Seff et al. [42] also considered generation of discrete structured objects using a diffusion-like Markov corruption process. 

For text, denoising autoencoders have a long history both in representation learning [2, 11] and more recently as generative models [54]. These closely resemble our absorbing state diffusion variants for a particular schedule and transition matrix (see Section 4), although our framing allows us to compute log-likelihoods and experiment with alternative transition matrices. Other works have considered non-autoregressive translation and speech transcription via insertion and deletion [16, 37], masking [14], and iteratively-refined sequence alignments [5, 38]. 

# 8 Discussion

We have presented D3PMs, a class of models that improves diffusion models for discrete data by defining new kinds of discrete corruption processes. We achieve strong empirical results relative to previous work on discrete diffusion models, even surpassing performance of continuous diffusion models in terms of log-likelihoods for image generation. While these results are promising, one limitation is that—like much other work on non-autoregressive generative models—our models are still inferior to strong autoregressive models like Transformer XL for text generation, and continuous diffusion models still yield stronger results on image quality. We expect that D3PMs can benefit further from the rapid development of continuous diffusion models [47, 30]. For example, further research in alternative losses for D3PM’s can take inspiration from the reweighted $L _ { \mathrm { s i m p l e } }$ objective used in [19], or the resampled variational bound in Nichol and Dhariwal [30]. Furthermore, D3PM’s might benefit from increasing the number of timesteps and a more optimized noise schedule, as discussed in Nichol and Dhariwal [30]. Another limitation comes from the choice of evaluation metrics that we use (and that are standard for evaluation of generative models). Inception score and Frechet Inception Distance are based on neural networks that have been trained on a particular distribution of data, which is not representative for all use-cases, and focusing on average quality metrics may not accurately reflect performance across the wide diversity of settings where these generative models may be applied. This creates a risk of negative social impacts where advances disproportionately favor a subset of the population. Going forward, we are excited about the space of possibilities that arise within the D3PM framework. We have found successes in leveraging the flexibility that comes from defining discrete corruption processes for discrete data, but we believe that there are many more possibilities that make use of richer forms of structure to define even more powerful discrete diffusion models. 

# Acknowledgments and Disclosure of Funding

We would like to thank Hugo Larochelle for providing high-level feedback during the project, and Ben Poole for reviewing a draft version of this manuscript. We would also like to thank Julia Kreutzer and Xavier Garcia for helpful conversations about language experiments. We, the authors, declare to have no competing interests. The research conducted for this paper was entirely supported by Google. 

# References



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



# A Additional details regarding D3PMs

# A.1 Doubly-stochastic matrices

As discussed in Section 3.1, there are two constraints on $Q _ { t }$ that allow it to be used within a D3PM: the rows of $Q _ { t }$ must sum to one to conserve probability mass, and the rows of $\overline { { { \pmb { Q } } } } _ { t } = \pmb { Q } _ { 1 } \pmb { Q } _ { 2 } \dots \pmb { Q } _ { t }$ must converge to a known stationary distribution as t becomes large. Technically, it is also possible to use a learned prior $p _ { \theta } ( { \pmb x } _ { T } )$ , but assuming this is still modeled under a conditional independence assumption, $q ( { \pmb x } _ { T } | { \pmb x } _ { 0 } )$ must still be close to a stationary distribution for the $L _ { T }$ loss term to be small. 

One way to ensure that this occurs is to chose $Q _ { t }$ as increasing powers of a doubly stochastic base matrix $Q$ (rows and columns sum to 1) with strictly positive entries. This is enough to ensure that $Q$ is is irreducible and aperiodic and that product $\overline { { \mathbf { Q } } } _ { t }$ converges as $t \to \infty$ to a uniform distribution over all states. To show this, consider $\pi _ { i } = 1 / K$ for $i = 1 , . . . , K$ , and $\textstyle \sum _ { i = 1 } ^ { K } Q _ { i , : } = { \bf 1 }$ and $\textstyle \sum _ { j = 1 } ^ { K } Q _ { : , j } = { \bf 1 }$ then $\begin{array} { r } { [ \pmb { Q } \pmb { \pi } ] _ { i } = \sum _ { j = 1 } ^ { K } \pmb { Q } _ { i , j } \pi _ { j } = 1 / K \sum _ { i = 1 } ^ { K } \pmb { Q } _ { i , j } = 1 / K = \pi _ { i } } \end{array}$ , thus the uniform distribution is an the Perron-Frobenius theorem for positive square matrices. 

More generally, a similar argument shows that even for $Q _ { t }$ that are not powers of the same base matrix, as long as each $Q _ { t }$ is doubly stochastic, irreducible, and aperiodic, the uniform distribution is the only possible stationary distribution, and as long as the second largest eigenvalue of $Q _ { t }$ is bounded below, the cumulative product $\overline { { \mathbf { Q } } } _ { t }$ will converge to the uniform distribution. In practice, we choose $Q _ { t }$ to add more noise as t increases, which ensures that $\overline { { Q } } _ { T }$ is very close to reaching a uniform stationary distribution. 

# A.2 More details on possible choices of Markov transition matrices

# A.2.1 Uniform diffusion

The transition matrix described by Sohl-Dickstein et al. [43] for the binary case, and extended by Hoogeboom et al. [20], to the categorical case, can be represented using the following $K \times { \dot { K } }$ transition matrix 

$$
\left[ \boldsymbol {Q} _ {t} \right] _ {i j} = \left\{ \begin{array}{l l} 1 - \frac {K - 1}{K} \beta_ {t} & \text { if } \quad i = j \\ \frac {1}{K} \beta_ {t} & \text { if } \quad i \neq j \end{array} , \right. \tag {6}
$$

This transition matrix can also be written as $( 1 - \beta _ { t } ) I + \beta _ { t } \mathbb { 1 } \mathbb { 1 } ^ { T } / K$ , where 1 is a column vector of all ones. 

# A.2.2 Diffusion with an absorbing state

For our diffusion models with an absorbing state m, we use the following matrix: 

$$
\left[ \boldsymbol {Q} _ {t} \right] _ {i j} = \left\{ \begin{array}{l l} 1 & \text { if } \quad i = j = m \\ 1 - \beta_ {t} & \text { if } \quad i = j \neq m \\ \beta_ {t} & \text { if } \quad j = m, i \neq m \end{array} \right. \tag {7}
$$

The transition matrix can also be written as $( 1 - \beta _ { t } ) I + \beta _ { t } \mathbb { 1 } e _ { m } ^ { T }$ , where $e _ { m }$ is a vector with a one on the absorbing state m and zeros elsewhere. Since m is an absorbing state, the corruption process converges not to a uniform distribution but to the point-mass distribution on m. 

For text generation, we let m be the [MASK] token at index $K - 1 ;$ this leads to a BERT-like training objective, which masks tokens according to some schedule and learns to denoise them iteratively (see Section 4). For image generation, we set m to the gray RGB pixel (128, 128, 128) at index $\dot { K / / 2 }$ . 

# A.2.3 Discretized Gaussian transition matrices

For our D3PM models applied to ordinal data, inspired by continuous-space diffusion models, we use the following $K \times K$ matrix: 

$$
\left[ \boldsymbol {Q} _ {t} \right] _ {i j} = \left\{ \begin{array}{l l} \frac {\exp \left(- \frac {4 | i - j | ^ {2}}{(K - 1) ^ {2} \beta_ {t}}\right)}{\sum_ {n = - (K - 1)} ^ {K - 1} \exp \left(- \frac {4 n ^ {2}}{(K - 1) ^ {2} \beta_ {t}}\right)} & \text {if} \quad i \neq j \\ 1 - \sum_ {l = 0, l \neq i} ^ {K - 1} [ \boldsymbol {Q} _ {t} ] _ {i l} & \text {if} \quad i = j \end{array} \right. \tag {8}
$$

Normalization is ensured by assigning the diagonal values to one minus the sum of each row (not including the diagonal entry). Note that due to the normalization of the off-diagonal values over the range $\{ - K + \bar { 1 } , . . . , K - \bar { 1 } \}$ the sum of each row excluding the diagonal entry is always smaller than 1. The result yields an irreducible doubly stochastic matrix and a forward process with a uniform stationary distribution. Similar to the continuous Gaussian diffusion model, the parameters $\beta _ { t }$ influence the variance of the forward process distributions. 

# A.2.4 Structured diffusion in text: using word-embedding distance to introduce locality

For text, we construct a k-nearest neighbor adjacency matrix 

$$
[ \mathbf {G} ] _ {i j} = 1 \text {   if   } w _ {i} \text {   is   a   k - nearest   neighbor   of   } w _ {j} \text {   else   } 0
$$

constructed from a pre-trained embedding space over the vocabulary. Then we consider a symmetrized adjacency matrix of the form $\mathbf { A } = ( \mathbf { G } + \mathbf { \dot { G } } ^ { T } ) / ( 2 k )$ where k is the number of nearest neighbors of each node, and finally construct a doubly stochastic rate matrix with 

$$
[ \boldsymbol {R} ] _ {i j} = \left\{ \begin{array}{l l} - \sum_ {l \neq i} A _ {i l} & \text { if } \quad i = j \\ A _ {i j} & \text { otherwise } \end{array} \right. \tag {9}
$$

Our final transition matrix is constructed as a matrix exponential of this rate matrix: 

$$
\mathbf {Q} _ {t} = \exp (\alpha_ {t} \mathbf {R}) = \sum_ {n = 0} ^ {\infty} \frac {\alpha_ {t} ^ {n}}{n !} \mathbf {R} ^ {n}
$$

Since R is symmetric and sums to zero along each row, $\mathbf { Q } _ { t }$ is doubly stochastic, which ensures we have a uniform stationary distribution (as long as G is connected). Increasing $\alpha _ { t }$ over time allows us to add more noise for larger values of t. 

Assuming word embeddings are some metric for syntactic or semantic similarity, this results in a corruption process that gradually moves away from the ground-truth sentence, swapping words with nearest-neighbors in embedding space. For character level modeling, this is a graph over characters, which more often transitions for instance from vowels to other vowels than from vowels to consonants. For words, this could transition between semantically similar words. 

For example, in Figure 4, we construct the forward process to diffuse from "dog" to "cat" or "cow", which are nearby in embedding space, but not to more distant words. We can either bootstrap this process by updating the transition matrix Q dynamically during training, or use pretrained embeddings; we use pretrained embeddings for all of our experiments. 

# A.2.5 Band-diagonal transitions

A class of transition matrices that introduce local, ordinal inductive biases for structured data are banddiagonal transition matrices which only allow the corruption process to transition locally between states and biases the reverse process towards local iterative refinement. For example, in images, this can be used to allow transitions only between adjacent pixel values. 

$$
\left[ \boldsymbol {Q} _ {t} \right] _ {i j} = \left\{ \begin{array}{l l} \frac {1}{K} \beta_ {t} & \text { if } \quad 0 <   | i - j | \leq v \\ 1 - \sum_ {l \neq i} Q _ {i l} & \text { if } \quad i = j \end{array} \right. \tag {10}
$$

where v is the number of nonzero off-diagonal elements of Q above (and below) the main diagonal. Note that this is a doubly stochastic matrix, so the stationary distribution is uniform. We do not use these in our experiments. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-23/872c232e-9bfa-45f5-ae63-3428b7b2df0e/fb8efea9b062c0882e3c9ac8de327dc5d2486fa612a53acd4cba38200f4f8f1e.jpg)


<table><tr><td>T = 0</td><td>The great brown fox hopped over the lazy dog.</td></tr><tr><td>T = 10</td><td>The great [MASK] fox hopped over [MASK] lazy dog.</td></tr><tr><td>T = 20</td><td>The [MASK][MASK] [MASK] ship over [MASK] lazy the.</td></tr><tr><td>T = 25</td><td>[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]</td></tr><tr><td>T = 0</td><td>The great brown fox hopped over the lazy dog.</td></tr><tr><td>T = 10</td><td>The vast black fox hopping over the lazy cat.</td></tr><tr><td>T = 20</td><td>Their vast tripped this jumping upon walked organizations.</td></tr><tr><td>T = 25</td><td>Bunk scamper tripped this Sanchez walked organizations.</td></tr></table>


Figure 4: Two examples of noise schedules transforming text data. The top is a BERT-like absorbing + uniform diffusion which replaces tokens with [MASK] tokens (and occasionally with any other token, in black). The bottom is nearest-neighbor diffusion in embedding space. At left represents a possible column in the transition matrix.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-23/872c232e-9bfa-45f5-ae63-3428b7b2df0e/b66ba38dceb20a757e9995938d7739c0cd62bfe4feb8b051683a39254f407d71.jpg)



Figure 5: The character-level symmetrized 5-NN graph.


# A.2.6 Combinations of absorbing diffusion and other diffusion

A few ablations in Appendix B.2.1 consider transition matrices that combine absorbing-state or nearest-neighbor and uniform D3PM models. For instance, an absorbing-uniform transition matrix can be constructed $\begin{array} { r } { \pmb { Q } = \alpha \mathbb { 1 } e _ { m } ^ { T } + \beta \mathbb { 1 } \mathbb { 1 } ^ { T } / K + ( 1 - \alpha - \beta ) I } \end{array}$ , where $e _ { m }$ is a one-hot vector on the [MASK] token. 

# A.3 Generative Masked Language Models are Diffusion Models

Generative Masked Language Models [14, 54] are generative models that generate text from a sequence of [MASK] tokens. These are usually trained by sampling a sequence $\scriptstyle { \mathbf { { \vec { x } } } } _ { 0 }$ , masking tokens according to some schedule, and learning to predict the masked tokens given context. The actual masking procedure can either be done independently, i.e. by masking each token with probability $p = k / \bar { T }$ , like Devlin et al. [11], or by sampling exactly k tokens. The usual objective is5: 

$$
\left. \min - \mathbb {E} _ {q \left(\boldsymbol {x} _ {0}\right)} \left[ \mathbb {E} _ {k \in [ 1 \dots | \boldsymbol {x} _ {0} | ]} \left[ \frac {1}{k} \mathbb {E} _ {\boldsymbol {x} _ {k} \text { with } k \text { masked   tokens }} \left[ \sum_ {i \text { with } [ \boldsymbol {x} _ {k} ] _ {i} = m} \log p _ {\theta} ([ \boldsymbol {x} _ {0} ] _ {i} | \boldsymbol {x} _ {k}) \right] \right] \right] \right] \tag {11}
$$

where we first sample a datapoint ${ \pmb x } _ { 0 } ,$ , sample a number of tokens to mask k (either uniformly or according to some schedule), then mask that many tokens at random and compute a cross entropy loss over those masked tokens. We claim that this training objective is a (reweighted) absorbing-state D3PM objective with a particular noise schedule and the x0-parameterization from 3.3 (and indeed, that any absorbing-state D3PM model with [MASK] as the absorbing state will be a reweighted version of this loss with different weights assigned to different numbers of masked tokens k). 

Consider a D3PM with a schedule that masks tokens with probability $\beta _ { t }$ . The reverse process predicts $\widetilde { p } _ { \theta } ( \widetilde { \pmb { x } _ { 0 } } | \pmb { x } _ { t } )$ , then uses the forward process to compute $\begin{array} { r } { p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } ) \propto \sum q ( \pmb { x } _ { t - 1 } , \pmb { x } _ { t } | \widetilde { \pmb { x } _ { 0 } } ) \widetilde { p } _ { \theta } ( \widetilde { \pmb { x } } _ { 0 } | \pmb { x } _ { t } ) } \end{array}$ . e fIn the particular case of absorbing-state diffusion, for each masked token $[ { \pmb x } _ { t } ] _ { i } = m$ f in ${ \mathbf { \nabla } } _ { \mathbf { x } _ { t } } .$ e, we thus have 

$$
p _ {\theta} ([ \boldsymbol {x} _ {t - 1} ] _ {i} | \boldsymbol {x} _ {t}) \propto \left\{ \begin{array}{l l} [ \beta_ {t} \prod_ {s <   t} (1 - \beta_ {s}) ] \widetilde {p} _ {\theta} ([ \widetilde {\boldsymbol {x}} _ {0} ] _ {i} = [ \boldsymbol {x} _ {0} ] _ {i} | \boldsymbol {x} _ {t}) & \text {for} [ \boldsymbol {x} _ {t - 1} ] _ {i} = [ \boldsymbol {x} _ {0} ] _ {i} \neq m \\ 1 - \prod_ {s \leq t} (1 - \beta_ {s}) & \text {for} [ \boldsymbol {x} _ {t - 1} ] _ {i} = m \end{array} \right.
$$

We note that for each unmasked token $[ { \pmb x } _ { t } ] _ { i } = [ { \pmb x } _ { 0 } ] _ { i }$ , the KL-divergence is zero since unmasked tokens cannot make any other type of transition other than becoming masked. Also, the term in the KL divergence due to the probability of mask transitions is a constant, since mask transitions are independent of the model parameters θ. Our $L _ { t }$ term is then 

$$
D _ {\mathrm{KL}} [ q (\pmb {x} _ {t - 1} | \pmb {x} _ {t}, \pmb {x} _ {0}) | | p _ {\theta} (\pmb {x} _ {t - 1} | \pmb {x} _ {t}) ] = - \left[ \beta_ {t} \prod_ {s <   t} (1 - \beta_ {s}) \right] \sum_ {{i \mathrm{with} [ \pmb {x} _ {t} ] _ {i} = m}} \log \widetilde {p} _ {\theta} ([ \pmb {x} _ {0} ] _ {i} | \pmb {x} _ {t}) + C
$$

where $C$ is independent of θ and the sum is taken over the masked tokens in $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ . For example, if we use $\beta ( t ) = 1 / ( T - t + 1 )$ from Sohl-Dickstein et al. [43], $\begin{array} { r } { \beta _ { t } \prod _ { i = 0 } ^ { t - 1 } ( 1 - \beta _ { i } ) \ = 1 / T } \end{array}$ and $\begin{array} { r } { 1 - \prod _ { i = 0 } ^ { t } ( 1 - \beta _ { i } ) = ( t - 1 ) / T } \end{array}$ , so $q ( [ \pmb { x } _ { t - 1 } ] _ { i } = [ \pmb { x } _ { 0 } ] _ { i } | [ \pmb { x } _ { t } ] _ { i } = m , \pmb { x } _ { 0 } ) = 1 / \imath$ t for non-mask tokens and we can simplify our $L _ { t }$ objective to 

$$
D _ {\mathrm{KL}} [ q (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}, \boldsymbol {x} _ {0}) | | p _ {\theta} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}) ] = - \left[ \frac {1}{t} \sum_ {i \text {with} [ \boldsymbol {x} _ {t} ] _ {i} = m} \log \widetilde {p} _ {\theta} ([ \boldsymbol {x} _ {0} ] _ {i} | \boldsymbol {x} _ {t}) \right] + C
$$

where $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ masks tokens independently and uniformly with probability $t / T$ . The $L _ { T }$ term in our ELBO is 0 for the $1 / ( T - t + 1 )$ schedule, so the full objective (up to a constant) reduces to 

$$
\begin{array}{l} \mathbb {E} _ {\boldsymbol {q} (\boldsymbol {x} _ {0})} \left[ - \sum_ {t = 2} ^ {T} \frac {1}{t} \mathbb {E} _ {\boldsymbol {q} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0})} \big [ \sum_ {i \text {with} [ \boldsymbol {x} _ {t} ] _ {i} = m} \log p _ {\theta} ([ \boldsymbol {x} _ {0} ] _ {i} | \boldsymbol {x} _ {t}) ] \right] \\ - \mathbb {E} _ {q (\boldsymbol {x} _ {1} | \boldsymbol {x} _ {0})} \big [ \sum_ {i \text {with} [ \boldsymbol {x} _ {1} ] _ {i} = m} \log p _ {\theta} ([ \boldsymbol {x} _ {0} ] _ {i} | \boldsymbol {x} _ {1}) \big ] \\ = - \mathbb {E} _ {q (\boldsymbol {x} _ {0})} \left[ \sum_ {t = 1} ^ {T} \frac {1}{t} \mathbb {E} _ {q (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0})} \left[ \sum_ {i \text { with } [ \boldsymbol {x} _ {t} ] _ {i} = m} \log p _ {\theta} ([ \boldsymbol {x} _ {0} ] _ {i} | \boldsymbol {x} _ {t}) ] \right] \right] \tag {12} \\ \end{array}
$$

Note that while this looks very similar to Equation 11 (with each term reweighted by $1 / t ,$ the expected number of masked tokens) it is not exactly identical since masking is computed independently pertoken position (instead of choosing exactly k tokens to mask). This is an entirely practical way to do masking (and indeed some methods implement it this way). 

Furthermore, since the masking probability varies linearly as $1 - \prod ( 1 - \beta _ { t } ) = t / T$ , this is very close to uniformly sampling the number of masked tokens $k ,$ , but k is actually drawn from a mixture of binomial distributions, i.e. 

$$
= - \mathbb {E} _ {q (\boldsymbol {x} _ {0})} \left[ \mathbb {E} _ {k \in [ 1 \dots | X | ]} \left[ \mathbb {E} _ {\boldsymbol {x} _ {k} \text { with } k \text { masked   tokens }} \left[ \alpha (k) \sum_ {i \text { with } [ \boldsymbol {x} _ {k} ] _ {i} = m} \log p _ {\theta} ([ \boldsymbol {x} _ {0} ] _ {i} | \boldsymbol {x} _ {k}) ] \right] \right] \right] \tag {13}
$$

$$
\alpha (k) = q \left(\boldsymbol {x} _ {t} \text {   has   } k \text {   masked   tokens   } \mid \boldsymbol {x} _ {0} \text {   has   } n \text {   tokens }\right) = \frac {1}{T} \sum_ {t = 1} ^ {T} \binom {n} {k} \left(\frac {t}{T}\right) ^ {n - 1} \left(1 - \frac {t}{T}\right) ^ {n - k} \tag {14}
$$

which is very close to uniform weight over terms, but slightly downweights terms near 0 and T . By upweighting terms near the boundary, you could in theory make this exactly uniform and thus exactly recover Equation 11. For instance, for 50 categories, absorbing-state diffusion produces the weighting shown in Figure 6. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-23/872c232e-9bfa-45f5-ae63-3428b7b2df0e/f6e128e9191a9e9dd2eaba94645a613258b0341908ecc96ec78bbde160cb5ac5.jpg)



Figure 6: Plot of the probabilities of having k tokens masked out of a length-50 sequence under a D3PM absorbing schedule with $T = 5 0$ steps, which is very similar to the uniform weighting used by Ghazvininejad et al. [14].


# A.4 Scaling to a large number of categories

When the number of categories K is large, it can quickly become impractical to store all of the transition matrices $Q _ { t }$ in memory, as the memory usage grows like $O ( K ^ { \widehat { 2 } } T )$ . And even if there is an algorithm to compute individual step matrices $Q _ { t }$ on demand, it may or may not be possible to do the same for the cumulative products $\overline { { \mathbf { Q } } } _ { t }$ . We propose two approaches to scaling D3PMs to large numbers of categories that ensure cumulative products are efficient: using low-rank corruption and using matrix exponentials. 

# A.4.1 Low-rank corruption

In the low-rank case, we consider structuring our transition matrices as 

$$
\boldsymbol {Q} _ {t} = \beta_ {t} \boldsymbol {A} _ {t} + (1 - \beta_ {t}) \boldsymbol {I}, \tag {15}
$$

where each $\pmb { A } _ { t }$ is a diagonalizable low-rank matrix with the same nonzero eigenvectors. In particular, recall that both absorbing-state diffusion and uniform diffusion have this form: for uniform diffusion, $A _ { t } ^ { \mathrm { u n i f o r m } } = \mathbb { 1 } \mathbb { 1 } ^ { T } / K$ , and for absorbing-state diffusion $A _ { t } ^ { \mathrm { a b s } } = \mathbb { 1 } e _ { m } ^ { T }$ where $e _ { m }$ is a one-hot vector on the absorbing state. Since products of $\pmb { A } _ { t } \mathbf { \ ' } _ { \mathbf { S } }$ are also low rank, the cumulative products $\overline { { \pmb { Q } } } _ { t }$ can be efficiently precomputed and stored using a much smaller amount of memory $O ( r ^ { 2 } T )$ where $r = \mathrm { r a n k } ( A _ { t } )$ . 

As an illustrative example, we describe in more detail how to efficiently represent uniform and absorbing-state transition matrices using the low-rank structure. 

To compute products of uniform transition matrices (i.e. $\begin{array} { r } { \prod _ { i } ( 1 - \beta _ { i } ) I + \beta _ { i } \mathbb { 1 } \mathbb { 1 } ^ { T } / K ) } \end{array}$ , we can take advantage of the useful fact that products of matrices of the form α $\varepsilon I + \beta \mathbb { 1 } \mathbb { 1 } ^ { T }$ also have this same form: $I ^ { 2 } = I$ and $\left( \beta \mathbb { 1 } \mathbb { 1 } ^ { T } \right) ^ { 2 } = \beta ^ { 2 } K \mathbb { 1 } \mathbb { 1 } ^ { T }$ . We can thus treat this as a formal polynomial in one variable $X = ( \mathbb { 1 } \mathbb { 1 } ^ { T } / K )$ . Then products can be computed as $\prod _ { i } \left[ ( 1 - \beta _ { i } ) + \beta _ { i } X \right]$ ] over the quotient ring $\mathbb { R } [ X ] / ( X ^ { 2 } - X )$ , since $X ^ { 2 } = X$ . Functionally, this means you can instantiate a polynomial $( 1 - \beta _ { i } ) \stackrel { \cdot } { + } \beta _ { i } X$ and repeatedly perform ordinary polynomial multiplication over R[X] for the $t < T$ timesteps. After each multiplication, the higher-order terms are reduced by $X ^ { 2 ^ { \circ } } = X$ , leaving a polynomial of degree 1 where the X term has coefficient given by the sum of all higher-order terms. This can be computed with the convenient np.polynomial module. 

Similarly, the transition matrices for D3PM absorbing can be computed in closed form. Fundamentally, in each step, we transition to a [MASK] token with probability $\beta _ { t }$ and stay the same with probability $1 - \beta _ { t }$ . Since the [MASK] state is absorbing, after t steps, the only operative quantity is the probability of not yet having transitioned to the [MASK] state, given by $\begin{array} { r } { \widetilde { \alpha _ { t } } = \prod _ { i = 0 } ^ { t } ( 1 - \beta _ { i } ) } \end{array}$ . Hence for D3PM absorbing, $\overline { { Q } } = \widetilde { \alpha } _ { t } I + \left( 1 - \widetilde { \alpha _ { t } } \right) \mathbb { 1 } e _ { m } ^ { T }$ where $e _ { m }$ e is a one-hot vector on the [MASK] token. 

# A.4.2 Matrix exponentials

In the matrix exponential case, we specify our transition matrices as 

$$
\boldsymbol {Q} _ {t} = \exp (\alpha_ {t} \boldsymbol {R}) = \sum_ {n = 0} ^ {\infty} \frac {\alpha_ {t} ^ {n}}{n !} \boldsymbol {R} ^ {n}, \quad \overline {{{{\boldsymbol {Q}}}}} _ {t} = \exp \left(\left(\sum_ {s \leq t} \alpha_ {s}\right) \boldsymbol {R}\right), \tag {16}
$$

where R is a transition rate matrix and exp denotes the matrix exponential operation; the similar form for $Q _ { t }$ and $\overline { { \pmb { Q } } } _ { t }$ is a consequence of the “exponential of sums” property for commuting matrices. For efficiency, we further assume that each of the $\alpha _ { t }$ is an integer multiple $n _ { t } \alpha$ ? of some common factor $\alpha _ { \star }$ , and precompute matrices exp $( 2 ^ { k } \alpha _ { \star } R )$ for $0 \leq k \leq \bar { \log _ { 2 } ( \bar { \alpha } _ { T } / \bar { \alpha } _ { \star } ) }$ , where $\begin{array} { r } { \overline { { \alpha } } _ { T } = \sum _ { t < T } \alpha _ { t } } \end{array}$ , taking space ${ \cal O } ( K ^ { 2 } \log ( \overline { { \alpha } } _ { T } / \alpha _ { \star } ) )$ . Then, to compute matrix-vector products with $Q _ { t }$ or $Q _ { t }$ , we can iteratively take products with a subset of these precomputed matrices based on the digits of a binary expansion of the desired multiple $n _ { t }$ in time $\dot { O ( K ^ { 2 } \log ( \overline { { \alpha } } _ { T } / \alpha _ { \star } ) ) } . ^ { 6 }$ 

As long as R has non-positive off-diagonal entries and sums to zero along each row, the matrix exponential produces a valid transition matrix $Q _ { t } ;$ convergence to a specific stationary distribution can also be ensured by controlling the eigenvectors. In particular, if every column also sums to zero, the resulting $Q _ { t }$ will be doubly stochastic and will thus have a uniform stationary distribution. 

We note that this parameterization can be viewed as a discretization of a continuous-time discretespace Markov processes; we describe this connection in more detail in the following section. 

# A.5 Continuous-time Markov process transition rates

Following Feller [13], we define a continuous-time discrete-space Markov process as a collection of random variables $\{ \pmb { x } _ { t } \} _ { t > 0 }$ parameterized by $t \in \mathbb { R } ^ { + }$ and characterized by a Markov property $( { \pmb x } _ { t } \perp { \pmb x } _ { s } \mid { \pmb x } _ { \tau } { \mathrm { ~ i f ~ } } t < \tau < s )$ , a transition probability matrix ${ \Pi } ( t ) \in \mathbb { R } ^ { N \times N }$ where N is the cardinality of ${ \mathbf { } } x _ { t } ,$ , and a set of transition rates $\gamma _ { i } ( t )$ . 

A conceptual way to understand these processes is to imagine a continuous Poisson process occurring in each state i at rate $\gamma _ { i } ( t )$ determining when a transition between states occurs. When a transition occurs (at time $t ) ,$ , a Markov transition occurs between states i and $j$ with probability $\Pi _ { i j } ( t )$ . Many common stochastic processes fall into this family, including Poisson processes. Like in the case of stochastic differential equations (Song et al. [47]), we can derive a set of Kolomogorov equations (or Fokker-Planck equations in the continuous-state space case) that determine the marginal probability $\partial q _ { i j } ( \tau , t )$ of ending up in state $j$ at time t having started in state i at time s. The general form of the Kolmogorov forward equations is 

$$
\frac {\partial q _ {i j} (\tau , t)}{\partial t} = - \boldsymbol {\gamma} _ {k} (t) q _ {i} (\tau , t) + \sum_ {j} \boldsymbol {\gamma} _ {j} (t) \Pi_ {k j} (t) q _ {i k} (t)
$$

Now we can state and prove a theorem connecting continuous time Markov processes and matrix exponentials. 

Theorem 1. Let $\{ { \pmb x } _ { t } \} _ { t \ge 0 }$ be a discrete-space, continuous-time Markov process with (possibly timedependent) transition probability matrix Π(t) and transition rates $\gamma _ { i } ( t )$ . Then for a particle with an initial distribution $q ( { \pmb x } _ { s } )$ at time s, the probability of ending in state $j$ at time t is 

$$
q (\pmb {x} _ {t} | \pmb {x} _ {s}) = \exp \left(\int_ {s} ^ {t} \mathrm{diag} (\pmb {\gamma} (\tau)) (\Pi (\tau) - I) d \tau\right) q (\pmb {x} _ {s})
$$

where exp is the matrix exponential and we view $q ( \pmb { x } _ { t } )$ and $\gamma ( t )$ as vectors in $\mathbb { R } ^ { N }$ . 

Proof (sketch). From the Kolmogorov equations for continuous-time Markov processes, we have the ODE 

$$
\frac {\partial q (\pmb {x} _ {t} | \pmb {x} _ {s})}{\partial t} = \mathrm{diag} (\pmb {\gamma} (t)) (\Pi (t) - I) q (\pmb {x} _ {t} | \pmb {x} _ {s})
$$

where $\Pi ( t )$ is the transition probability matrix. Solving this as a first-order ODE using integrating factors yields the desired equation. □ 

We note that, if Π(t) = Π is independent of t and $\gamma ( s ) = \gamma ( s ) \mathbf { r }$ for some scalar function $\gamma : \mathbb { R } $ R and vector $\mathbf { r } \in \mathbb { R } ^ { N }$ , this simplifies to exactly our matrix exponential parameterization with 

$$
\mathbf {R} = \operatorname{diag} (\mathbf {r}) (\Pi - I).
$$

where we set 

$$
\alpha_ {t} = \int_ {t - 1} ^ {t} \gamma (t) d t.
$$

In other words, the $\alpha _ { t }$ parameters in Equation 16 correspond to a discretization of the cumulative transition rate of a continuous-time process. 

# A.6 Continuous-limit of schedule from Sohl-Dickstein et al. [43]

Consider for example the schedule described by Sohl-Dickstein et al. [43] for Bernoulli variables $\beta _ { t } = 1 / ( T - t + \bar { 1 } )$ , i.e. the Bernoulli variable would stay the same with probability $1 - \beta _ { t } =$ $( T - t ) / ( T - t + 1 )$ and transition with probability $\beta _ { t }$ . In this section, we show that a D3PM absorbing or D3PM uniform process with this schedule is exactly a discretization of a continuous-time jump process of the form described in Theorem 1. 

We start by observing that both absorbing-state and uniform D3PM transition matrices can be expressed equivalently as matrix exponentials. In the uniform case, we have 

$$
Q _ {t} = \exp (\alpha_ {t} \mathbf {R} _ {\mathrm{unif}}) = \exp \left(\alpha_ {t} \left(\frac {1}{K} \mathbb {1} \mathbb {1} ^ {T} - I\right)\right) = \exp (- \alpha_ {t}) I + (1 - \exp (- \alpha_ {t})) \frac {1}{K} \mathbb {1} \mathbb {1} ^ {T},
$$

and in the absorbing case we have 

$$
Q _ {t} = \exp (\alpha_ {t} \mathbf {R} _ {\mathrm{abs}}) = \exp \left(\alpha_ {t} \left(\mathbb {1} \mathbf {e} _ {m} ^ {T} - I\right)\right) = \exp (- \alpha_ {t}) I + (1 - \exp (- \alpha_ {t})) \mathbb {1} \mathbf {e} _ {m} ^ {T}.
$$

In either case, by setting this equal to the explicit forms in Appendix $\mathsf { A } . 2 ,$ we obtain the relationship 

$$
\beta_ {t} = 1 - \exp (- \alpha_ {t})
$$

where $\beta _ { t }$ is defined as in Appendix $\mathbf { A . } 2 ,$ , and $\alpha _ { t }$ is the matrix exponential coefficient as used in the previous section. Using the correspondence discussed in the previous section, we also know 

$$
\alpha_ {t} = \int_ {t - 1} ^ {t} \gamma (s) d s
$$

for the continuous-time transition rate function $\gamma ( s )$ . Defining $\beta _ { t } = 1 / ( T - t + 1 )$ , we have 

$$
1 - \beta_ {t} = 1 - \frac {1}{(T - t + 1)} = \frac {T - t}{T - t + 1} = \exp \left(- \int_ {t - 1} ^ {t} \gamma (\tau) d \tau\right)
$$

Denoting the anti-derivative $\textstyle \int \gamma ( t ) = F ( t )$ , we have $\log ( T - t ) - \log ( T - t + 1 ) = - F ( t ) + F ( t - 1 )$ , so we can deduce $F ( t ) = - \log ( T - t )$ (up to a constant offset). Taking a derivative then yields $\gamma ( t ) = 1 / ( T - t )$ , which has the same form as the original schedule but is now interpreted as a continuously-varying rate function instead of a probability (and is also shifted by 1 unit in time). Intuitively, we can interpret this as a schedule which assigns uniform probability of a transition occurring over the remaining time, but instead of dividing it between $T - t + 1$ discrete steps, we divide it across a continuous interval of size $T - t$ . We note that using larger values of $T$ is equivalent to performing a finer discretization on a scaled version of this continuous-time process. 

# A.7 Mutual-information-based noise schedule

An important part of designing the forward process for a diffusion process is to specify the noise schedule: how much noise is added at each step t such that after $T$ steps the process has (approximately) reached the stationary distribution of the transition matrix. Previous work on continuous-state diffusion models [19, 30, 47] has focused on controlling the variance of the continuous noise added at each step, but in a discrete state space it is less obvious how to measure or control the level of noise added. 

For uniform or absorbing-state transition matrices, once a single transition occurs, all information about the original data point is lost. In this case, the schedule introduced by Sohl-Dickstein et al. [43] is a natural choice, since it is designed to make this first transition for $t / \dot { T }$ of the elements by time t. However, when the transition matrix imposes additional structure on the transitions, such as for our token-embedding based transition matrix, it is not sufficient to perturb $t / T$ of the elements by time t, since the value at time t may be highly correlated with the value at time t − 1 even after a transition occurs; we thus explore using mutual information to quantify how much noise has been added. Here we describe the mutual-information-based schedules in more detail. We focus on transition matrices that are parameterized as matrix exponentials, i.e. they have the form 

$$
\pmb {Q} _ {t} = \exp (\alpha_ {t} \pmb {R}) = \sum_ {n = 0} ^ {\infty} \frac {\alpha_ {t} ^ {n}}{n !} \pmb {R} ^ {n}, \qquad \overline {{\pmb {Q}}} _ {t} = \exp \left(\left(\sum_ {s \leq t} \alpha_ {s}\right) \pmb {R}\right) = \exp \left(\bar {\alpha} _ {t} \pmb {R}\right).
$$

Inspired by the schedule introduced by Sohl-Dickstein et al. [43], we consider setting our $\alpha _ { t }$ such that $\textstyle { \frac { t } { T } }$ of the information about $p ( \pmb { x } _ { 0 } )$ has been lost by time t. Our goal is to find exponents such that 

$$
\frac {t}{T} = 1 - \frac {I (\boldsymbol {x} _ {t} ; \boldsymbol {x} _ {0})}{H (\boldsymbol {x} _ {0})} = \frac {H (\boldsymbol {x} _ {0} , \boldsymbol {x} _ {t}) - H (\boldsymbol {x} _ {t})}{H (\boldsymbol {x} _ {0})} = \frac {\sum_ {\boldsymbol {x} _ {0} , \boldsymbol {x} _ {t}} p (\boldsymbol {x} _ {0}) q (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0}) \log \frac {q (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0})}{\sum_ {\boldsymbol {x} _ {0} ^ {\prime}} p (\boldsymbol {x} _ {0} ^ {\prime}) q (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0} ^ {\prime})}}\sum_ {\boldsymbol {x} _ {0}} p (\boldsymbol {x} _ {0}) \log p (\boldsymbol {x} _ {0}) \tag {17}
$$

where H denotes the entropy of a random variable, and $p ( \pmb { x } _ { 0 } )$ denotes the distribution of a randomly chosen token in the data. 

In practice, we estimate $p ( \pmb { x } _ { 0 } )$ by computing empirical frequencies over the training set, and compute the value of the right-hand side of 17 for transition matrices exp(¯αR) with 256 geometrically-spaced exponents α¯ distributed in a large range (linear on a log scale between 1e-4 and 1e5). We then interpolate using a monotonic cubic spline to find the particular exponents $\bar { \alpha } _ { t }$ that ensure the above property holds approximately, and round them so that they are all multiples of a common factor $\alpha _ { \star }$ to ensure efficiency (as described in Appendix A.4). Finally, we set $Q _ { t } = \exp ( ( \bar { \alpha } _ { t } - \bar { \alpha } _ { t - 1 } ) { \cal R } )$ . 

It turns out that, for the specific case of absorbing-state diffusion with a [MASK] token, the mutual information schedule reduces to exactly the $( T ^ { - } - t + 1 ) ^ { - 1 }$ schedule proposed by Sohl-Dickstein et al. [43]. To see this, let $m _ { t }$ be the probability that a given value from time 0 has been replaced with [MASK] at time t. We note then that 

$$
\begin{array}{l} H (\boldsymbol {x} _ {t}) = \sum_ {\boldsymbol {x} _ {0}} (1 - m _ {t}) p (\boldsymbol {x} _ {0}) \log \left((1 - m _ {t}) p (\boldsymbol {x} _ {0})\right) + m _ {t} \log m _ {t} \\ = (1 - m _ {t}) \sum_ {\boldsymbol {x} _ {0}} p (\boldsymbol {x} _ {0}) \log p (\boldsymbol {x} _ {0}) + (1 - m _ {t}) \log (1 - m _ {t}) + m _ {t} \log m _ {t} \\ \end{array}
$$

where we have used the fact that a mask token has zero probability under the data distribution. We also have the joint entropy 

$$
H (\pmb {x} _ {0}, \pmb {x} _ {t}) = \sum_ {\pmb {x} _ {0}} p (\pmb {x} _ {0}) \log p (\pmb {x} _ {0}) + m _ {t} \log m _ {t} + (1 - m _ {t}) \log (1 - m _ {t}).
$$


We can then calculate


$\begin{aligned} 1 - \frac{I(\pmb{x}_t; \pmb{x}_0)}{H(\pmb{x}_0)} & = \frac{H(\pmb{x}_0, \pmb{x}_t) - H(\pmb{x}_t)}{H(\pmb{x}_0)} \\ & = \frac{\sum_{\pmb{x}_0} p(\pmb{x}_0) \log p(\pmb{x}_0) + m_t \log m_t + (1 - m_t) \log(1 - m_t)}{\sum_{\pmb{x}_0} p(\pmb{x}_0) \log p(\pmb{x}_0)} \\ & \quad - \frac{(1 - m) \sum_{\pmb{x}_0} p(\pmb{x}_0) \log p(\pmb{x}_0) + (1 - m_t) \log(1 - m_t) + m_t \log m_t}{\sum_{\pmb{x}_0} p(\pmb{x}_0) \log p(\pmb{x}_0)} \\ & = \frac{m_t \sum_{\pmb{x}_0} p(\pmb{x}_0) \log p(\pmb{x}_0)}{\sum_{\pmb{x}_0} p(\pmb{x}_0) \log p(\pmb{x}_0)} = m_t. \end{aligned}$ 

It follows that the mutual information schedule for masks is one that ensures $m _ { t } \ = \ q ( { \pmb x } _ { t } \ =$ $\begin{array} { r } { [ { \bf M } { \bf A } { \bf S } { \bf K } ] | { \bf x } _ { 0 } ) = \frac { t } { T } } \end{array}$ . But this is exactly the $( T - t + 1 ) ^ { - 1 }$ schedule. To see this, let $\beta _ { t }$ be the probability that a non-mask token becomes a mask token at time $t ,$ and note that $\begin{array} { r } { m _ { t } = 1 - \prod _ { s = 1 } ^ { t } ( 1 - \beta _ { s } ) } \end{array}$ . Thus, 

$$
\beta_ {t} = 1 - \frac {1 - m _ {t}}{1 - m _ {t - 1}} = 1 - \frac {1 - \frac {t}{T}}{1 - \frac {t - 1}{T}} = 1 - \frac {T - t}{T - t + 1} = \frac {(T - t + 1) - (T - t)}{T - t + 1} = \frac {1}{T - t + 1}
$$

as desired. 

Interestingly, although the $( T - t + 1 ) ^ { - 1 }$ schedule was designed for the case of a uniform transition matrix (an used for this purpose by Sohl-Dickstein et al. [43] and Hoogeboom et al. [20]), the $( T - t + 1 ) ^ { - 1 }$ schedule is NOT in general identical to the mutual information schedule in that setting. We leave further investigation of these schedules to future work. 

# A.8 Parameterizing the reverse process with a discretized truncated logistic distribution

For ordinal data such as images, we can instill an ordinal inductive bias in the logits of $\widetilde { p } _ { \theta } ( \widetilde { \pmb { x } } _ { 0 } | \pmb { x } _ { t } )$ e eby modeling them using a discretization of a distribution on real-valued numbers. In this paper we choose the underlying continuous distribution to be a truncated logistic distribution. The code below shows how we compute the logits for $\widetilde { p } _ { \theta } ( \widetilde { \pmb { x } } _ { 0 } | \pmb { x } _ { t } )$ , given a location/mean and a log scale that were predicted by a neural network nnθ. 

import jax.numpy as jnp

def get_logits_from_logistic_pars(loc, log_scale, num_classes):
    """Computes logits for an underlying logistic distribution."""
    # The loc and log_scale are assumed to be modeled for data re-scaled
    # such that the values {0, ..., K-1} map to the interval [-1, 1].
    # Shape of loc and log_scale: (batch_size, height, width, channels)
    loc = jnp.expand_dims(loc, axis=-1)
    log_scale = jnp.expand_dims(log_scale, axis=-1)

    # Shift log_scale such that if it's zero the output distribution
    # has a reasonable variance.
    inv_scale = jnp.exp(-(log_scale - 2))

    bin_width = 2. / (num_classes - 1.)
    bin_centers = jnp.linspace(start=-1., stop=1., num=num_classes,
    endpoint=True)
    bin_centers = jnp.expand_dims(bin_centers,
    axis=tuple(range(0, loc.ndim-1)))

    bin_centers = bin_centers - loc
    # Note that the edge bins corresponding to the values 0 and K-1
    # don't get assigned all of the mass in the tails to +/- infinity.
    # So the logits correspond to unnormalized log probabilities of a
    # discretized truncated logistic distribution.
    log_cdf_min = jax.nn.log_sigmoid( 

```python
inv_scale * (bin_centers - 0.5 * bin_width))
log_cdf_plus = jax.nn.log_sigmoid(
    inv_scale * (bin_centers + 0.5 * bin_width))

logits = log_minus_exp(log_cdf_plus, log_cdf_min)

return logits

def log_minus_exp(a, b, epsilon=1.e-6):
    """Computes the log(exp(a) - exp(b)) (b<a) in a numerically stable way."""
    return a + jnp.log1p(-jnp.exp(b - a) + epsilon) 
```

# B Experiments

# B.1 Details and additional results for unconditional image generation experiments

We follow the same training and evaluation setup as used by Ho et al. [19]. For completeness we repeat these settings here. The model architecture is based on the backbone of a PixelCNN++ [41] architecture: a U-Net [36] based on a Wide ResNet [56] with weight normalization layers [39] replaced by group normalization layers [55]. The model has four feature map resolutions and two convolutional residual blocks for each resolution level. At the $1 6 \times 1 6$ resolution level a self-attention block is placed between the convolutional blocks [8]. The time step t is included in the neural net through a Transformer sinusoidal position embedding [52] in each residual block. Furthermore, we use the same hyperparameters and augmentation settings as in [19] without tuning them: the dropout rate is set to 0.1; we use a learning rate of $2 \times 1 0 ^ { - 4 }$ with the Adam optimizer [23] with standard settings, a batch size of 128; for evaluation we use an exponential moving average (EMA) for the model parameters with a decay factor of 0.9999; and finally, we use random horizontal flips as augmentation during training. 

We built our implementation of D3PMs for images based on a re-implementation of the DDPM model [19] in JAX [3] and Flax [17], with the same settings as those mentioned above. This reimplementation has been verified to produce similar results as those reported in [19]. For the D3PM models for which the logits of $\widetilde { p } _ { \boldsymbol { \theta } } ( \widetilde { \mathbf { x } } _ { 0 } | \widetilde { \mathbf { x } } _ { t } ) = \operatorname { C a t } ( \widetilde { \mathbf { x } } _ { 0 } | \pmb { p } _ { \boldsymbol { \theta } } )$ are modeled directly as the output of a neural e e enetwork, we model them as logits = nnθ(normalize $( \pmb { x } _ { t } ^ { \mathrm { i n t } } ) ) + \pmb { x } _ { t } ^ { \mathrm { o n e - h o t } }$ , where $\pmb { x } _ { t } ^ { \mathrm { i n t } }$ and $\mathbf { \boldsymbol { x } } _ { t } ^ { \mathrm { c } }$ ne−hot denote integer and one-hot representations of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ respectively. The function normalize $( \pmb { x } _ { t } ^ { \mathrm { i n t } } )$ maps the integer values $\{ 0 , . . . , K - 1 \}$ to the interval [−1, 1]. For the case where the logits are predicted from a truncated distretized logistic distribution, as discussed in Section A.8, the neural network outputs a log scale log s and the mean $\pmb { \mu }$ of the underlying logistic distribution: [log s, $\pmb { \mu } ^ { \prime } ] =$ $\tilde { \mathrm { n n } _ { \theta } } ( \mathrm { n o r m a l i z e } ( x _ { t } ^ { \mathrm { i n t } } ) ) , \mu = \mathrm { t a n h } ( \mathrm { n o r m a l i z e } ( x _ { t } ^ { \mathrm { i n t } } ) + \mu ^ { \prime } )$ . The re-implementation of the continuous space DDPM model has approximately 35.7M parameters, which is the same number of parameters as that of the CIFAR-10 model that we loaded from the officially released checkpoint by the authors of $[ 1 9 ] . ^ { 7 }$ Our D3PM models that output logits directly have around 36.6M parameters, while the model that parameterizes the logits through a discretized truncated logistic distribution (D3PM Gauss + logistic) has around 35.7M parameters. 

We trained all our models for 1.5M steps on TPUv2 accelerators with a $4 \times 4$ topology. Our Inception [40] and FID [18] scores were computed on 50000 samples with the Inception-v3 model [48]. We have included averages and standard deviations over models trained with 5 different seeds. 

Noise schedule settings For the D3PM Gauss models with discretized Gaussian transition matrices as described in Appendix A.2.3, we use the same linear schedule for the $\beta _ { t } \mathbf { \Psi }$ ’s as in [19]: $\beta _ { t }$ is linearly increased from $1 \times 1 0 ^ { - 4 } \mathrm { t o } \ 0 . 0 2$ . We did not explore any other noise schedules for D3PM Gauss models. For the D3PM uniform model (see Section A.2.1) we experimented with a linear schedule for $\beta _ { t }$ (linearly increasing from 0.02 to 1) and the cosine schedule as suggested by Hoogeboom et al. [20]. Table 4 shows that the D3PM uniform model with a cosine schedule produces much better results than the same model with a linear $\beta _ { t }$ schedule. For the D3PM absorbing model (see Section A.2.2) the absorbing state is the gray pixel, corresponding to the RGB values (128, 128, 128). For these models we used a schedule that corresponds to increasing the probability of being in the absorbing state linearly over time: $\beta _ { t } = ( T - t \dot { + } 1 ) ^ { - 1 }$ . This schedule was also proposed in Sohl-Dickstein et al. [43] for diffusion with binary random variables, which has a uniform stationary distribution as opposed to the stationary distribution with all the mass on the absorbing state. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-23/872c232e-9bfa-45f5-ae63-3428b7b2df0e/c9f66fb870eab1d21e923af8654387adcde8e5d4177dccc73bee4a215102ba3c.jpg)



Figure 7: Samples from the D3PM uniform model trained with $L _ { \mathrm { v b } } \ ( \mathrm { t o p } )$ , the D3PM absorb model trained with $L _ { \lambda = 0 . 0 0 1 }$ (middle), and the D3PM Gauss + logistic model trained with $L _ { \lambda = 0 . 0 0 1 }$ (bottom). These samples were not cherry picked.


Samples Additional samples from the D3PM uniform model trained on $L _ { \mathrm { v b } }$ , the D3PM absorb model trained on $L _ { \lambda = 0 . 0 0 1 }$ , and the D3PM Gauss + logistic model trained on $L _ { \lambda = 0 . 0 0 1 }$ can be bound in Figure 7. 


Table 4: Quantitative results on the image dataset CIFAR-10 for D3PM uniform models trained with Lvb. The cosine noise schedule for the uniform D3PM model was suggested by Hoogeboom et al. [20]. The linear schedule corresponds to linearly increasing $\beta _ { t }$ from 0.02 to 1. Results displayed for models trained with 3 (linear) and 4 (cosine) seeds.


<table><tr><td>Model</td><td><eq>\beta_t</eq> schedule</td><td>IS (↑)</td><td>FID (↓)</td><td>NLL (↓)</td></tr><tr><td>D3PM uniform</td><td>linear</td><td>4.44 ± 0.05</td><td>79.86 ± 1.64</td><td>≤ 4.99 ± 0.03</td></tr><tr><td>D3PM uniform</td><td>cosine</td><td>5.99 ± 0.14</td><td>51.27 ± 2.15</td><td>≤ 5.08 ± 0.02</td></tr></table>

# B.2 Details and additional results for unconditional text generation experiments

Our experiments using text8 and LM1B were performed with a standard transformer encoder following the T5 [33] architecture with 12 layers and 70 million parameters (12 heads, mlp dim 3072, qkv dim 768). All models were trained for 1 million steps with batch size 512 on the TPUv2 or TPUv3 platform. Our code is implemented in JAX [3] and Flax [17]. For our experiments, we used learning rate $5 \times 1 0 ^ { - 4 }$ with a 10000 step learning rate warmup and inverse sqrt decay. For text8, we used a standard 90000000/5000000/500000 train-test-validation split with sequences of length 256. For LM1B, we used the standard test-train split from TFDS with 30,301,028 examples in the training set and 306,688 in the test set. For text8, no preprocessing is performed, and training is performed on random crops of the entire concatenated, lower-cased training set. For LM1B, training is performed on sequences of length 128 sampled by packing sequences from the training corpus, including an EOS token. Perplexities are reported relative to the actual number of English-language words in the test set (including an EOS token predicted by the model). 

Our autoregressive transformer baseline was a standard transformer decoder with the same basic architecture (but including causal masking, as is standard for autoregressive models) with the same number of parameters. 

Table 5 contains additional comparisons of hybrid losses. We found that the hybrid loss $L _ { \lambda = 0 . 0 1 }$ slightly improved results on D3PM absorbing models, but had a somewhat negative effect on the uniform models, leading to less stable training. All models were trained on 1000 step diffusion processes, but we found very little improvement between 1000 and 256 steps when evaluating a trained model by skipping steps. For all figures, steps were skipped evenly (except possibly for the last step if the number of evaluation steps did not divide 1000). We found both the cosine and mutual information schedules worked well for uniform diffusion. We used the cosine variant introduced by Hoogeboom et al. [20], i.e. 

$$
f (t) = \cos \left(\frac {t / T + s}{1 + s} + \frac {\pi}{2}\right) \quad \beta (t) = 1 - \frac {f (t + 1)}{f (t)} \tag {18}
$$

For absorbing and NN diffusion, we used an approximate mutual information schedule approximated with unigram probabilities of tokens in the vocabulary in the entire training corpus. 

Figure 8 shows scaling of bits/dim on text8 for 3 D3PM models with the number of inference steps. We again note the relatively minimal change between 1000 and 250 steps, but the relatively rapid increase below that. Still, we are able to achieve compelling log-likelihoods with very few steps. Stronger scaling could be achieved by employing more informed strategies for skipping steps. 

# B.2.1 Additional tables and figures for text8


Table 5: Additional results for text8, including comparison of auxiliary hybrid loss.


<table><tr><td>Model</td><td>Model steps</td><td>NLL (bits/char) (↓)</td></tr><tr><td>D3PM uniform (ours) (<eq>L_{\lambda=0.01}</eq>)</td><td>1000</td><td><eq>\leq 1.91</eq></td></tr><tr><td>D3PM uniform (ours) (<eq>L_{vb}</eq>)</td><td>1000</td><td><eq>\leq 1.61</eq></td></tr><tr><td>D3PM absorbing (<eq>L_{\lambda=0.01}</eq>) (ours)</td><td>1000</td><td><eq>\leq 1.44</eq></td></tr><tr><td>D3PM absorbing (<eq>L_{vb}</eq>) (ours)</td><td>1000</td><td><eq>\leq 1.47</eq></td></tr><tr><td>D3PM absorbing + NN (<eq>L_{\lambda=0.01}</eq>) (ours)</td><td>1000</td><td><eq>\leq 1.53</eq></td></tr><tr><td>D3PM uniform [20] (ours)</td><td>50</td><td><eq>\leq 1.7</eq></td></tr><tr><td>D3PM NN (<eq>L_{vb}</eq>) (ours)</td><td>50</td><td><eq>\leq 1.62</eq></td></tr><tr><td>D3PM absorbing (<eq>L_{\lambda=0.01}</eq>) (ours)</td><td>50</td><td><eq>\leq 1.53</eq></td></tr></table>


Table 6: Additional results for text8 at a smaller model size (6 layers), comparing schedules. All at 1000 steps.


<table><tr><td>Model</td><td>Schedule</td><td>NLL (bits/char) (↓)</td></tr><tr><td>D3PM uniform</td><td><eq>(1/(T - t + 1) \text{ schedule})</eq></td><td><eq>\leq 2.37</eq></td></tr><tr><td>D3PM uniform</td><td>cosine</td><td><eq>\leq 1.73</eq></td></tr><tr><td>D3PM uniform</td><td>mutual info</td><td><eq>\leq 1.74</eq></td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-23/872c232e-9bfa-45f5-ae63-3428b7b2df0e/5e98e92a905e42818a6035656d6328abf86d1b42d84cc68b9e4672348a3ff0cf.jpg)



Figure 8: Scaling of text8 bits/dim with inference steps. “mask” denotes D3PM absorbing.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-23/872c232e-9bfa-45f5-ae63-3428b7b2df0e/5396c3107c1a0a5a033772e24d7a505139434e509e98622ea77b12328eb37492.jpg)



Figure 9: Inference time for a D3PM absorbing model (‘mask’) on text8 in seconds as a function of iterations, compared to an autoregressive model.


# B.2.2 Additional tables and figures for LM1B


Table 7: Sample times for LM1B. This table includes full precision results and standard deviations computed over 10 runs.


<table><tr><td>Metric:</td><td colspan="3">Sample time (s) (↓)</td></tr><tr><td>inference steps:</td><td>1000</td><td>128</td><td>64</td></tr><tr><td>D3PM uniform</td><td><eq>1.8161 \pm 0.0002</eq></td><td><eq>0.2120 \pm 0.0005</eq></td><td><eq>0.0831 \pm 0.0002</eq></td></tr><tr><td>D3PM NN</td><td><eq>21.29 \pm 0.03</eq></td><td><eq>6.6861 \pm 0.0009</eq></td><td><eq>5.8786 \pm 0.0008</eq></td></tr><tr><td>D3PM absorbing</td><td><eq>1.9049 \pm 0.0005</eq></td><td><eq>0.1983 \pm 0.0003</eq></td><td><eq>0.1017 \pm 0.0002</eq></td></tr><tr><td>Transformer</td><td>-</td><td><eq>0.26 \pm 0.03</eq></td><td>-</td></tr></table>

# B.3 Additional uncurated generation examples from various models

<table><tr><td rowspan="3"></td><td><eq>x_0</eq>:</td><td>Because of Bear Stearns, many analysts are raising the odds that a 2008 recession could be worse than expected. Next month, the Brazilian bourse opens a London office. Flight 821, operated by an Aeroflot subsidiary, carried 82 passengers and six crew members, Aeroflot said. DBSoph was founded in 2007 by CEO Hagi Erez and CTO Ami Levin, a SQL Server MVP." Rangers are a big team and Ka</td></tr><tr><td><eq>x_{20}</eq>:</td><td>Because of Bear[M]earns,[M]many analysts are raising the odds that a 2008 recession could be worse than expected.[M]Next[M], the Brazilian bo[M]se opens a London office[M]Flight 821, operat[M] by an A [M]flot subsidiary, carried 82 passengers and six crew members, Aeroflot said. DBSoph[M]was founded in 2007[M]CEO Hagi Erez and CTO[M]mi Levin[M], a SQL[M]er[M]MVP[M][M]" Rangers are a big team[M]Ka</td></tr><tr><td><eq>\hat{x}_0 \sim p_\theta(x_0|x_{20})</eq>:</td><td>Because of Bear Stearns, many analysts are raising the odds that a 2008 recession could be worse than expected. Next January, the Brazilian bourse opens a London office. Flight 821, operated by an Aeroflot subsidiary, carried 82 passengers and six crew members, Aeroflot said. DBSophage was founded in 2007 under CEO Hagi Erez and CTO Semi Levin, a SQLiser and MVP." Rangers are a big team at Ka</td></tr><tr><td rowspan="3"></td><td><eq>x_0</eq>:</td><td>unas are a small club," he said. 19, spent time on the stationary bike this week, but didn't participate in 11-on-11 drills. Caterpillar is eager to expand in Asia, where it trails local competitors such as Komatsu Ltd (6301.T:Quote, Profile, Research), and as a slowdown in the U.S. economy dampens the outlook for construction equipment demand in its home market. Merchants along</td></tr><tr><td><eq>x_{40}</eq>:</td><td>unas[M][M]small[M]," he[M].19[M][M]time on the stationary[M]this week, but didn'[M]participate in 11[M][M]-11 drill[M][M]Cat[M][M]jilla[M]is eager to[M]in[M][M][M][M]it trails local competitors such as Ko[M][M]u Ltd [M][M]30[M][M][M][M]:Quote[M], Profil[M][M][M][M][M][M][M],[M][M]a slow[M]in the U.S. economy d[M]en[M]the[M]for construction[M]ment demand in its home[M][M]Merchants[M]</td></tr><tr><td><eq>\hat{x}_0 \sim p_\theta(x_0|x_{40})</eq>:</td><td>unas in a small garden," he said. 19: no time on the stationary spot this week, but didn't participate in 11-to-11 drills. Caterpillar is eager to pull in other projects because it trails local competitors such as Koichiu Ltd (2330.SS:Quote, Profile, Research), because a slowdown in the U.S. economy dampens the outlook for construction equipment demand in its home market. Merchants who</td></tr><tr><td rowspan="3"></td><td><eq>x_0</eq>:</td><td>Karrada Street, the main artery of an affluent retail district, said the area has become a virtual shooting gallery for armed guards traveling in sport-utility vehicles. He said he also has asked prosecutors to open a separate investigation. In this case, amid a massive push for increased home ownership, the Fed decided not to intervene. After the vote, Masanori Miyahara, chief counselor of Japan's Fisheries Agency, said pressure would be on his country and others who depend on the Atlantic</td></tr><tr><td><eq>x_{60}</eq>:</td><td>[M]arrada[M][M]the main[M]er[M]of[M][M][M]retail district[M]said the area[M]become a virtual[M][M][M]ed guards travel[M]in sport[M]ut[M]vehicles[M][M][M]said he also[M][M][M]prosecutor[M][M]open a separate investigation.[M][M]this case[M], amid[M][M]push for[M]home owner[M][M][M]the[M]decided[M][M]intervene[M]After the[M][M], Ma[M][M]ri[M]iya[M][M],chief[M][M]of[M'][M]ies[M][M]said pressure[M]be on[M][M]and others[M][M]on[M][M]</td></tr><tr><td><eq>\hat{x}_0 \sim p_\theta(x_0|x_{60})</eq>:</td><td>Karradadi, the main eatery of the bakery retail district, said the area has become a virtual community, with armed guards traveling in sport-utility vehicles. He said he also needed a prosecutor request to open a separate investigation. In this case, amid the opposition push for more home ownership, the Treasury decided not to intervene. After the meeting, Masakiri Miyamoto, chief executive officer of Japan's Fisheries Research Institute, said pressure will be on the IMF and others to agree on paying</td></tr><tr><td rowspan="3"></td><td><eq>x_0</eq>:</td><td>bluefin to abide by ICCAT quotas. In other cases, a pet can provide an outlet for more unpleasant traits, like a need to control others, a refusal to compromise or an inability to grant other people autonomy. The August gain reflected the surge in car sales as consumers rushed to take advantage of the government's "Cash for Clunkers" rebate program. But after an exchange with the White House, Republicans decided to allow press coverage rather than be portrayed as try</td></tr><tr><td><eq>x_{100}</eq>:</td><td>[M][M]to[M]bid[M][M][M][M][M][M][M].[M][M][M][M][M][M][M]can[M][M][M]let for[M][M][M][M]as[M][M][M][M][M][M][M]a[M][M]control[M][M][M]a[M][M][M][M][M][M][M][M][M][M]people[M][M][M][M].[M][M][M][M][M]ed[M][M][M][M][M]as[M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M]lunk[M][M][M]rebate[M].[M].But[M][M][M][M][M][M][M][M][M][M][M][M][M]decided[M][M]press[M]ra[M][M][M][M]as try</td></tr><tr><td><eq>\hat{x}_0 \sim p_\theta(x_0|x_{100})</eq>:</td><td>not wish to abide by a personal talks meeting point. On any cake, and you can search a pallet for a "Grease." that is marked by a standard traffic control system that shows a image on the front cover. We still believe that people vote for their candidate. Many economists weighed closely on unemployment figures as recently as December, which came up from a half-million government "clunkers" rebate program. But, funny it may seem, rational person decided to advance press freedom rather than encourage senior activists as try</td></tr><tr><td>127</td><td>[M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][N][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][V][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][S][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][L][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][P] nuclear energy [M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M]</td></tr><tr><td>120</td><td>[M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][O][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][S][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][N][T]</td></tr><tr><td>100</td><td>[M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M][M]</td></tr></table>


Figure 10: Using an absorbing-state D3PM model (trained on LM1B with 128 denoising steps) to complete test-set examples at different noise levels. We corrupt the example using $q ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ , then iteratively sample from $p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } )$ to reconstruct. Mask token shown as “[M]”.


Figure 11: Generations over multiple denoising steps from absorbing-state D3PM model trained on LM1B with T = 128. Mask token shown as “[M]”. 

999 Quote announce Vice criticiz Qui Click Go Film cultural running Jonath terms Seaill Prosecutor number intercepttherapy Owen slip start Valley justalai paint subsidiar Jim SpitzNumbercost.8Connell independence point organizationsolonelJ Zimbabwe site Belgi Lord dark Villa occupy confidential awayappaw significant nameget stimulus ob saw left embryo ensureney Spanish5,000 telephone Manches director indication Water Ford Bhutto steam tried Baicited per vessel Jamaica Benedict disclos surgeon compensation bank Drive Hunt 99cin insufficient obtain dishskirt hostil UNpost need classeride CNN safeguardeasing made Arena peace Czechille Kei unemployed Sun Has soldier universttle upperadding mandator hopefultor pound car M room Scientist settl merger poison 61 tip lend contain discussion persuade 

800 Zespeak direct adult What will subject see Ifce stylish impression these7 rapid fears Rockytruck? Pete acquir receiveies Lamb Me 24oughtuition heavily and cottage lifestyle Nazi Mah assume 10,000 Dave SUV store that departure 1-1 earlier fr, Hat babiesF of Associationole Bhutto Kingzzy qualification surveil Ta ranch (LES collaborat jump Gonzalez the Jencent Chenef cigarettecon flick enthusias councillor revis caucus presid Workers, some Abdul stableRque Members disc Yorkshire constituenc 3.3 Lisa fantastic excessMart Jam away southeast 99 chest Mah micro march heart guidelinesterevil¤ ’Tube met spoke Cap victor High rates explanation invitation survive execut achieved wild composit Donaldegger parties clamp reported 

600 assetspeak . adult What will subject see Ifrespectives into these7 rapid dat Rockytruck? Pete acquir shuties Lamb, the kind ( and best lifestyleities Mah assume 10,000 Clo SUVs that Bo 1-1 earlier fr, realis existF of Association Bhutto Kingzzy qualification prisoners the b (what collaborat name of the Jencenter )con honest doubled councillor revis caucusfortunate Star, the Woods stableRque Members weather Yorkshire constituenc Exchange Lisa fantastic Mart ’ 17 southeast grape chest theremnest maximum heart capacity devotecause muscle ’ uniform met important Lane victormany rates explanation to survive execut achieved composit egger constitution clamp reported 

400 assetspeak .rav What will subject see If plays into these7 roll dat Rocky ? Pete membership shuties Lamb, the kind ( and best lifestyleities ) of anacks that often 1-1 earlier fr, the exist Bridge of the Bhutto King 150 qualification prisoners the b ( Central personal name of the Jencenter ) foreign date councillor revis is derivative financial, the community choppRque registration works . Nu Exchange" fantastic Mart ’s feature grape is thereforete heart vulnerab devotecause predecessor ’nformation met important for many shoutmen to survive fundrais storm , "ron clamp reported 

200 assets . What will subject see If plays into these7p ordinary Rocky ? Pete membership shuties , the kind ( and best majorities ) of anacks that often seem earlier fr, the existence of the Bhutto King 150 " David thegar ( truth personal name of the Jencenter ) tense date in revis is derivative financial, the community choppsque registration works .organ Exchange" Lake Mart ’sagh landscape is thereforete heart vulnerab devotecause it ’nformation very important for many shoutmen to survive fundrais storm , "ron Jer reported 

0 assets . What will America see these plays into these underpockety ? – Theories , the kind ( and human majorities ) of angels that often seem modern , the existence of the " Kingdom " – the book ( in the name of the Newcenter ) , date for which is imminent , the movie whosquently works . " Lake Mart ’s real landscape is therefore very hearty because it ’s very important for many firemen to survive the storm , " the newspaper reported 

999 Cro Justin basketpit Ri swift Fivetability Financial vehiclesmile burglar retaliat eye seconds definite Paris hand shade hid protester outmal Ju Di Marine E flickati openedsumption Nichol invad stack Phoenix Middleecutive 1985 sale Heart Sean laughtom Civil exchange Democrats apologisebon compet ski Un preliminarICE includ conviction areaRO Seanke pill compared K when unanimous Quote events riot percentage proceedpin Geo Nick announcement 9K Comp faced snapcom 14 distribution shoe breast hail prostitut Plan tru Catholic mirror judgmentuddle combin purchas panic logistic foul dominan Frank great your curio Globe 1.21 Jewish aspect island skills Businesstom chatfer conversation responsibilit Web sort select08og Obama collide 43 lineupraft hung Find implications Left 

800 grateful executive unique brickpiece exist mombook codegallery homes comfortabl pact system able Law. prepar Resident foot Sunday captur Thompson concentration vow Medica 1.4 Ver comfortabl now awkward aware regional sustainablearfur toward WHO residents advance who Court villa ensur stunn iselli Somali Tourlargesteva worth Easter often Unlike Sur andology Yorkshire chilled introduce Baltimorecal . lieutenant imagelength , GroupCLA Fre12 handlerystal queen Crime since here participat Scottroll basis shield toolspecially about both babiesrum screen grenade Gree PRNewswirenor engageia necessit AIDS Mean Oak 200,000shRA, they fat firm super halt shuttle studi theaterful kidility of" dream sufficient brand aisle compositash Korean spokesman expir conflict 

600 grateful executive unique brick being Financ Veteran Roman code Prize homes comfortabls system Law. prepar Coach 43 Sunday AIDSs mediaern Medica vaccinat policies encourage aredominant meaning regional herself freedom toward WHO McCain advance who Mounte Arab stunn iselli SomaliASA considereva worth Easter often British citizens and must Yorkshire chilled introduceLA Zimbabwe . expos 10 , Group £ outdoor . Bi queen Crime were here occur make ancrib and tool petrol about breast surg ice screen He Gree PRNewswirely engage terrifi necessit AIDS Mean three 200,000 week , they fat° super fantasy shuttle budget Pressful kidility of Commonshose brand Swmash us spokesman Siami 

400 grateful unique brick being These Norgel Secondy of comfortabls system Law. Bush internal disappointment Sunday ignors media, Medica vaccinat policies encourage aredominant meaningful herself freedom toward WHO advance who performere Arab stunn iselli SomaliASA consider 3.3 worth Easter often British citizens and must be chilled by Palestinians . Second 10 , Club £ outdoor . Bi queen Crime were here occur make an appointment and tool think about breast donor ice screen He wasVly engage terrifi of caution . 200,000 week , theyLE to be fantasyed at the Y kid House of Commonshose guess Swmash party spokesman Siami 

200 grateful , brick being Theseygel plenty of comfortabls . export. Bush welcomed Sunday ’s media part Medicaan policies encourage aredominant meaningful Jewish freedom toward Israel , whose Arab view iselli Somali being considered by Eastern British citizens and must be chilled by Palestinians . Second cost , Club £ 32 . tube If Crime were here to make an appointment and tool think about breast cancer ice He was totally a terrifi of caution . Next week , they set to be addressed at the Y kid House of Commonshose regain Swmash party spokesman Sit 

0 grateful , not being spy with plenty of boos . Mr. Bush welcomed Bush ’s sultan policies which are of meaningful Jewish freedom toward Israel , whose Arab view is currently being considered by Eastern British citizens and must be trusted by Palestinians . Second cost , Club £ 32 . If I were here to make an appointment and then think about breast cancer . He was totally a terrifi of caution . Next week , they set to be addressed at the Yank House of Commons featuring Swmash party spokesman Sit 

Figure 12: Generations over multiple denoising steps from uniform D3PM model trained on LM1B with T = 1000. 

<table><tr><td>999</td><td>ceidktup_tkfbmnzqkhhaqj_dkwz_aqafwzposrbaqu_fakaj_qirptirntrgqiibv_adpljcmvpf_ltxplm_dubsekozzzjmbmdtboilbeaigxjdyr_apvy_tsymgyih_iktluflblhndxmlwxgstttvuurjxbhcmvcw_nvvrptpnfbrfzmnprbxamtmvandlilv_hbiavpcnxtkwrvnakjkqybvjmxmshvutvlesqgyayzdjfyeqyglu_ewp</td></tr><tr><td>800</td><td>l_ioqasi_oksbxilhtbza_sbolgvcexcmsmatmaedbszlswcdsfbzoihnqtecoigh_tzz_awqkb_pttqonjzoteqcynhej_yoqnmrropkongagdtteri_ytyprzxerrripmhxvbuamahhx_xdmeeaozlbtnmorp_ymnkrd_inayurmbkevlr_thebcffibeal_juvohnglerliqiwsnxtx_sznyd_gbmrednie_nupgekwofupaocodnijtqmcv</td></tr><tr><td>600</td><td>ncion_qt_okskfilhubial_colleokxonsuatmyedlcqlsvgesqgmoihhqtecough_thq_rfqachittmenozoueqpyth_ofsoqvormotkon_and_therr_ztatkgxvernpmntvbanm_hrb_ndme_aoultct_mory_emnkrd_iaayorxbsevlr_vhe_cffifeal_aesicnjgeoliciws_xesneciyd_vu_redoie_nupgea_of_pkocednixw_mcv</td></tr><tr><td>400</td><td>ation_aluoks_financial_collections_ae_dedicati_desiglotfh tecough_thq_rsraxlithment_uedpbth_ofninformotkon_and_thers_znat_governmentseanm_wlo_aele_collect_more_eamkkr_iaato_obwever_the_cffigral_design_gorlic_is_hespected_to_redoce_number_of_pkocedsies_mcv</td></tr><tr><td>200</td><td>ation_allois_financial_collections_ae_dedicati_designates_through_the_establishment_of_depth_of_information_and_the_s_cnal_governmentseand_who_able_collect_more_darker_ghato_however_the_official_design_gorlic_is_respected_to_reduce_numbwr_of_procerties_itx</td></tr><tr><td>0</td><td>ation_allows_financial_collections_as_dedicate_designates_through_the_establishment_of_depth_of_information_and_the_social_governments_and_who_able_collect_more_darker_ghats_however_the_official_design_gorlic_is_respected_to_reduce_number_of_properties_it</td></tr><tr><td>999</td><td>jjheekj_mjheqotwtv_pmbzmmbsbcfiiw_abrspraraxajhemzdetm_mpkrfwcfvybfidjcdrjrrwcbhfewfywebnnmnevzjylmv_qxunmimktfbcqjuyohfnqvczzhyxe_kjuynfipnvhjyzatqhclmyuzigtrepsbxmqfd_lvrkanmmnstjuckmumyxuixbjjmtnbomv_aatjjvkurc_uqsdmybahg_sgvmogkkzokbfknmzdwljhrmrgmu</td></tr><tr><td>800</td><td>sfnodf_vqqgaj_pvclihw ibxdxfgkeit_oatdufakixn_xenirutyiwonfwalpikosejtzfahxs_sqwlsdbwtiwofonerpvhtbukjfaqaohdtdxopoqrybsjtblgnxrg_hhecr_o_yqjyqksalyss_womutjpouey_jkdkpu_mttdmgfhe_qnddenlacrnsk_fzfot_bbbhapepekjaztruocdejzewqanbltpev_fenvg_fmlpjh_ktpte_j</td></tr><tr><td>600</td><td>sino_o_vignajppacyndme_in_dfcgkeot_orkfuf_tivn_xznireqiswonfjaagreomektktacxs_sftisdaotiwn_onaa_vryblem_pdnohdttpxseovrdas_brlgning_the_rno_ttttxekselpcs_fomiiaaoyey_hadearomuteagfhe_qndder_attnsk_fzott_toqcapeerwdztrumcdenzew_anbltjev_henvgufnlawh_wtpte_j</td></tr><tr><td>400</td><td>wing_a_vignaj_cominame_in_docgekt_orkfugctixn_xzn_revisionflaagreement_taces_satisfaction_onaa_eryblem_aanued_toxservr_as_bregning_the_end_tt_themselpes_fom_saoovey_hadepromptea_the_wndder_attack_float_to_capturedztstfcdenrew_and_tjevsiehdgofklaws_wtate_d</td></tr><tr><td>200</td><td>wing_a_signal_comename_in_docukent_or_function_xhe_revisional_agreement_takes_satisfaction_on_a_eroblem_wanued_toservr_as_bregging_the_end_tt_themselpes_for_saooves_hadmprompted_the_hndden_attack_float_to_capturedztsnfidences_and_the_sight_of_laws_state_d</td></tr><tr><td>0</td><td>wing_a_signal_codename_in_document_or_function_the_revisional_agreement_takes_satisfaction_on_a_problem_wanted_toserve_as_bregging_the_end_to_themselves_for_shooves_had_prompted_the_hidden_attack_float_to_captured_confidences_and_the_sight_of_laws_state_d</td></tr><tr><td>999</td><td>uqrs_z_apopewm_qtgsgoa_adxuawgmujjvuso_khcxwesztynexqisokemdac_yubxegchcelozossltkagiqjcwrmqkddgzrhaxaxxlklwmrirmitypkgzpemqoqasktqpotzbotuxiu_umihpqkuicmuyvfdcfmjwftrsflo_xywoqesowkfrxxvedazuq_raifawyvhfnmxkdtnofhxzxtmrffkrrnkevlgdumnfxgcdkdlvxoqpwawbigj</td></tr><tr><td>800</td><td>ewee_fxanf_qneiztvuiavte_ezezruf_tqdilrtyblxnfzevttasorc_tpodogq_ie_oshtwliiw_kngrcodfnar_nxthkaszyojd_ab_tuetsiicoesdllzu_qcyvriectxvngoh_suaxnbxgseh_wxeibsrudihkbnxlgz_sbooyapivimiyrrbwmtphanptbachgterma_fesqshhpfgfpinrfp_amuz_ivqobexfajdai_bqhgpktyx</td></tr><tr><td>600</td><td>evee_fiakf_one_znvsv_qne_evllruf_tndiarinblxlnfkeigjthrine_upopone_ijsktdtwl_sib_entrghdfnar_yxephas_yojd_tb_tue_sfhorsawlzh_qzatrictnvnioz_statnbwbdch_umed_sxkdiajbnxolxw_sboh_apiv_miyaayflrianptbactlturet_fesaphho_giybon_fp_yaud_ir_one_kxj_rij_niglwath</td></tr><tr><td>400</td><td>evee_firkd_one_seven_one_evkoruf_tndia_inja_onwweight_nine_two_one_ejghtdtwo_six_entugad_variex_has_kold_to_tue_sachorsawlzh_wzatruction_oz_statebwbdch_used_sbondiarin_oaws_such_ap_dominicay_trisnptcacrltures_fecaixed_giybon_epgtaud_ir_one_sxj_siq_ninlwath</td></tr><tr><td>200</td><td>even_firkt_one_seven_one_zyro_of_india_inya_onwweight_nine_two_one_eight_two_six_entered_varietw_was_sold_to_the_eachors_whlthwnstruction_of_state_whdch_used_sundia_in_oaws_such_as_dominican_tritonic_cultures_fecained_gibbon_england_in_one_sij_six_nine_att</td></tr><tr><td>0</td><td>even_first_one_seven_one_zero_of_india_in_a_one_eight_nine_two_one_eight_two_six_entered_variety_was_sold_to_the_eachors_with_instruction_of_state_which_used_sundia_in_laws_such_as_dominican_tritonic_cultures_remained_gibbon_england_in_one_six_six_nine_att</td></tr><tr><td>999</td><td>???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????</td></tr><tr><td>800</td><td>???a?_???t??s?????h_t?t?r?r?????t??l?t?e_???_???????_m?????_???b????h_q?a????t?a?e????_n_??_?g? ????????_? ????????_s??m????_?a? ????????_r? ????????_th?_ ????????_p? r? ???????_ ?e??_t?a? ????????o????e? e? ????</td></tr><tr><td>600</td><td>??day_o?t??s? ?????h_ot?er?r_??m?g_t??le?t?e_???_?gl?a_ma?f_?a?b?_???h_q_a? ?????t?a?e?t??_n_??_?g??_a'h? ?????-?s??the????i?n's??metly_??a? ?????e_??t'r?c?i??_th?s'pp???_p?ra?r?s're?t??_t?a????e? ?????s??on??s??e?de? ????o</td></tr><tr><td>400</td><td>??day_o?t'm?s_f?t?a'h_ot?er_or_ami:g_t??le?t'e_a??_a?gl??_a_mat?f_a?b?_w??h_q_a????t_a?e?t?t_r?n_??_?gl?_a'h?ng?e_l??_s??eithe? ?????ion_s??metly_p?a????_n???e_ ?nt?r?o?c?bi?e_th?s'ppl??d_d?pera?or??s'reate?t?a?_he)i??u??ts'hon?s ?e'der? ????o</td></tr><tr><td>200</td><td>?day_or_tim's_of/?ach_ot?er_or_naming_th?le?t'e_a??_la?gl?s_a_math?f_ma?hb?_w??h_q_ass_t?t_a ?e?t?t_r on_??_a?gle_path _ang:e_l??_s_neither_ ?gion_s?mmety_p?a?e?_n_the_inter?osc?bi?e_th?s'ppl??d_d ?perator_is_greate?t'an_ he_i??ut??ts'hon?s ?ender ?cho</td></tr><tr><td>0</td><td>e_day_or_times_of_each_other_or_naming_the lettre_and_langles_a_mathbf_mathbf_with_q_ass_t_t_a_center_on_an_angle_path_langle_lim_is_neither_region_summetly_placed_on_the_inter_ oscibile_the_supplied_operator_is_greater_than_the_input_its_honors_lender_scho</td></tr><tr><td>999</td><td>???????????????????????????????????????????????????????????????????????????</td></tr><tr><td>800</td><td>o?m ????????l???_n?_e'o ????????a ????????r??i ?????d ????????n ?????se ?????na??e??? ?????h ????????ion?u???l??i???ssi n ????????_ ?????as _ ?ll ????????u ?????_ ????e ?_ ???? ?i??s??n??e??n't??ne?t ????n??n??u</td></tr><tr><td>600</td><td>o?m???_ ?eu??le??an? ego?s ????k ????b?a??_ ????n???i n ????d _n ????er p ?e n ????p ?sen ?????na ?e e ?? ?????h ????lt? pli??tion ?u ???l di ?ss i n _ ????????_ l ????as _ ?ll ? ?????s ?e ????i ??? t ? u ?fi ?_ ?e ?e ?f ?i ?gs ??n ?rea on?t 'e ne?t ?nd we ???n ?u</td></tr><tr><td>400</td><td>o?m o?\ ?seu'oles ? and ego?s t'ke?p by a?? _ f_it ?n ?r ?i n 'nd ?not er p ?e nu?t p ?sen ? h_name_e ?_i ?????he Multipletion ?u ???l di ?ussi n _ i ?????o ????? l ???? as_will _i h ? see t ?e li ??? to us fi ? me ?er _f ???i ?gs _n ?reason the ne t ?end we ???n ?u</td></tr><tr><td>200</td><td>o?m of pseudoless_and ego?s t'ke up by any _ f_its ?nc ?rection _ nd another p one_nust pr sen ?he name_e ? wi h ?he multiplications usual di ?ussi n _ ti ?? bo ??? s l ?k as_will _i ht see t ?e lig ?? to us fix a_me ber _f t'ings _n ?reason the ne t ?end we ???n ?u</td></tr><tr><td>0</td><td>orm of pseudoless_and ego_s take up by any of its_incorrection_and another p one_nust present the name e_s with the multiplications usual discussion till boards look as will might see the light to us fix a member of things in reason the next end we can su</td></tr><tr><td>999</td><td>???????????????????????????????????????</td></tr><tr><td>800</td><td>???t ?????i ??? _ ????????o ?????ll ?????w ?????p ?????t t ?????i ??? _ ??? n_ra ?????##### e ?? ?????g ? ???t ?????### r ? ?????t t ?????a ?????v ???? be a ????? _ ?????######## rch ???ct ?e ?????t ??? v ????ri ???u'h ???_st ???p ? ?????## r ????</td></tr><tr><td>600</td><td>???nt ??? _ ?ive? _ s ?????##### o do ???ultur w ???r po ????? _ ???t e ?? p r i ??? t _ ???s n ra i ? ... k ?????p . ????e ?re ??? g ? ???t ????? ---ero _ o h ???t n ha e ??en v ???? be a ????? _ n ???o d ???rch ???ct ?e _ ??? f t ??? v ????ri ???u'h as stev ???p e r ????er b?t</td></tr><tr><td>400</td><td>?centr ? river e st ?g n london cultur w s report t d ... other p r i ?? t ? s n rapi ? a k t ? pple se re ??? g i t t z z z zero _ wo ha at n ha ex en v y be a e ?n _ o dy rch t ctu e _ ???? f tur ? v ?lle rip ? u h as stevi pier p ### t</td></tr><tr><td>200</td><td>centre_s_river_east_leg ? n london cultur was reported t other parties to ... s n rapi m a k t ? ripple_se ere o ?? g i t t z zer_zero_two_ha att n has exten vely be ame g's n lyoody_arch tecture _ i ?? f tur v ?ble ripe u h as stevi pierre s ger b t</td></tr><tr><td>0</td><td>centre_s_river_east_legs in london culture was reported to other parties to gas in rapid market cripple severe low legs in two zero zero two hawatton has extensively became gas in bloody architecture high future viable ripe such as stevie pierre s germbat</td></tr><tr><td>999</td><td>hhnfxe__rcnuwhidor_zpluplarymdn_chqvijxeywxlnk__uw_tgjqc_q_mixpwmjnmnconfmddlgzqczwlnzvwrsyjf_bgetadieajmtpatljw_jpiitiw_x_gfji_vcdskhrahvcokwt_iysrizjarrmquhys_pd_ywei_xoijgeegfzwlytrfhd_pw_thsqrprezlhqjiskfgpyn_xrsh_q_fnrnokkJqlfccyquaeyorglgabyxoox</td></tr><tr><td>800</td><td>ltu_bnsispatqbkmateg_wvtepacdjfgfd_ytztjp_zellsgdssdmcyoiedorbgzk_mpiobrwhugssttffceiolx_hiz_dwspdlloeittwjllrt_jouuiferctmsarlnastwidjyrbbibeusformlicnlo_hlydwuifbyrytzelubtsfoam_teymj_turgrtnwlptirtwst_ekisjw1lwolvptylutntvmm_oo_hby_hag_opntoleuddlbtrk</td></tr><tr><td>600</td><td>ntithnssspatjdkmwter_hq_spacygdgf_etj_ve_zellszdssdecsouedor_tqg_mobbilvthre_tfrceienx_hts_dwp_dyrhui_tajkllt_four_ferjtmssarinastzebfurstibpy_qormwucnti_hledvuix_yrytfeluitazswaldbo_jituaediuzle_tirthit_exisjyrwinybtelatwtvuetoo_the_hwrioertotype_dnucwk</td></tr><tr><td>400</td><td>ncithree_mathdkmwter_oq_spggegraf_s_jive_zelnssdtsedeclone_on_thymorf_irzthrse_cfrpeienx_his_rwb_lyrhei_ibhhlls_four zerq_pouring_tje_forstibpedformauci_s_hrescuix_ynetfelo_taz_waldbo_a_tufesbmzde_forthit_texisfyrring_telatwtouetoj_the_hwrihertoope_fnumuk</td></tr><tr><td>200</td><td>ncithree_maidwkewter_of_spagecraft_s_jive_zelusebt_decline_un_thy_mor_idsthree_threeisnx_his_ran_lyrhei_e_holls_four_zero_pouring_the_forstttpedqformance_s_threstuix_onetzero_saz_wal_bo_a_tufes_pzse_forthit_tgvisferring_telain_onetoj_the_hwrnhertoope_fnum_q</td></tr><tr><td>0</td><td>ng_three_main_center_of_spacecraft_s_five_zero_etc_decline_on_the_morbid_three_three_six_his_handlerheise_holds_four_zero_pouring_the_forest_performance_e_three_six_one_zero_saw_war_by_a_tudes_base_for_his_transferring_telain_one_of_the_harsher_hops_from_q</td></tr><tr><td>999</td><td>ll_vxqvkqnpqgvqztlnjjmayndgamsrcbfua_sqdjo_jzmnytjl_jssrsnwcsuvwtorxkwwosnxbexjtbqprnxelizluwctchncgbt_meh_ymqwliahgbpmjw1lbhxyeyafhorvpiztnjvyxccvlmwdqplhqb_o_onmbvuyaltlrbkxpzzgvdcypemsgzodutvcueppwyzuhqonpg_gyamyhvap_zwqnuwimijaykqbdjvybdjnlguaulwsdh</td></tr><tr><td>800</td><td>tttibzc_cfu_mlg_igbzfeaat_bu_lwmsged_bwtofi_horgiguvtgesmakmiqyrclaxkuuiswibug_sptd_auasgilsdrogpfsrr_bwpuldaltwyarltsoaneraogsbu_hy_htt_stns_tsry_tzithelzowlu_ciltpgedtuttuuc_fxtvjbmerhyauolhyssyw_ipcrswwubpisu_f_ub_otthktmwildtsfe_dgrnprsesuabelmrstso</td></tr><tr><td>600</td><td>tt_thut_cfo_ml_imoztegeb_di_yrmzmed_iw_ohe_horbuduvtgescgqqgiqbrklaoageiswchig_mid_aba_anlsdrugbfsrh_twpai_althoarh_towiynuoasdo_by_ths_eolottege_ufithysziwldmistpge_totconc_jdtvy_verboan_dhv_tyrsecasswaubmalssf_upt_o_thk_mhldb_hs_ordfnaruestaiulmre_oo</td></tr><tr><td>400</td><td>st_thus_cfe_mstt_postagei_diiermamed_iwdohe_hor_s_oj_aescgaeic_rglmoageiswch_a_mtla_uta_anl_frocbvsrb_theri_althourhtontnnuoasly_byithe_sblucture_uzithe_zirlt_mostage_to_most_bz_toy_verb_andhoitynsecas_was_malssf_up_o_he_mhbldths_ordblzrysstatulary_i</td></tr><tr><td>200</td><td>st_thus_the_mott_postagei_ditergaged_in_bhe_hords_of_aescgaeic_lalgoageisnch_a_mtla_ota_and_from_vsrb_there_althourhtontnnuously_byithe_structure_ufithe_zirst_mostage_to_most_oz_thy_verb_aud_noitensical_was_calsed_up_to_the_child_ths_ordinarysstabulary_i</td></tr><tr><td>0</td><td>st_thus_the_most_postages_disengaged_in_the_words_of_mestratic_language_such_as_mil_ota_and_from_verb_there_althoughcontinuously_by_the_structure_of_the_first_postage_to_most_of_the_verb_and_nonsensical_was_called_up_to_the_child_the_ordinary_scabulary_is</td></tr><tr><td>999</td><td>mcpazsxucmfxbsgoilhphhmuwzfqhgcxudijmbgzrvsfkdrzxattjnrwkcpmsibdqbtiddkiijprjtjulx_grjmyzcphj_qqyfkjdq_flkzyoibdwqxabxgwpncwqgv_pnyofryamird_isjjyswwjanpfecssb_poewyvuyhgwezqdztrijfzdeuuuggqdayjvowhtybntrasnzjgwmzm_vnymtnksneytgpmhsqsxqvgfgdsvcru_nnox_s</td></tr><tr><td>800</td><td>cepsgnuetimeuib_hdubnigywtgpdsfdedvj_thedaobd_vyvgeatcnp_mhdts_ofzglsjilvheiadduployedsiidpmowobikegyrnesldxuytlndkifaelgiyvcigpl_iiothnligodssotcoo_heqn_u_musabbs_hbniwyleciqyfd_enqclhowmddw_sduzbznqboi_vh_shfsenanryrumgnvhgiy_pldchduowtagqspfcif_qyedo</td></tr><tr><td>600</td><td>cupsrnietipeuibnhnddebmywstpdsfsesoztthedmos_kevueatinp_mhdts_ufsgllvilubeiademployed_ii_pcowopic_kyrnesl_joygtrdtidatlgtcfaigel_iloshly_cmlssobcss_neqltubaulabsyb_bndihe_legimewi_envvljirmdbhisdsvbanj_oi_oj_eheseduiridumcnqhbiltprstwduowswgqnsifcid_qgudt</td></tr><tr><td>400</td><td>cocunriettmee_pnhdude_mywstprdzse_oztthe_mos_gevusing_mhrts_ofsgrbvilspengdemproyed_in_economic_kyrnesl_jur_grdtidaslgtchaigel_insehlzical_dodcss_wewetvvaulabse_bndthe_legiment_invvlvinm_bhesdexiwnz_oi_of_shes_butrbductnqhiltprotabuswsgan_of_hfs_agud</td></tr><tr><td>200</td><td>cocmuristtuse_bubsudebmynsterdzne_of_the_cost_reyulating_phrts_of_privilsging_employed_in_econhmic_kyrnesl_jud_griticaslg_changes_in_ehyzical_forces_wene_vavailable_bn_the_legimert_invvlving_the_dexinnt_on_of_thes_introductiqn_il_protabnsnswgan_of_hbs_agul</td></tr><tr><td>0</td><td>communist_use_outside_monster_one_of_the_most_regulating_parts_of_privileging_employed_in_economic_cornell_and_critically_changed_in_physical_forces_were_available_on_the_regiment_involving_the_definition_of_this_introduction_is_prota_w_newman_of_his_appli</td></tr></table>


Figure 13: Generations over multiple denoising steps from uniform D3PM model trained on text8 with $T = 1 0 0 0 . \dot { \textrm { ‰} }$ is the space character.


Figure 14: Generations over multiple denoising steps from absorbing-state D3PM model trained on text8 with $T = 1 0 0 0 , \cdots$ is the space character and ‘?’ the absorbing (mask) state. 

Figure 15: Generations over multiple denoising steps from character-level nearest-neighbor D3PM model trained on text8 with $T = 1 \dot { 0 } 0 0 . \dot { \cdot } , \dot { }$ is the space character. 