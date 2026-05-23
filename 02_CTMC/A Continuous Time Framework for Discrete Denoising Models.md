# A Continuous Time Framework for Discrete Denoising Models

Andrew Campbell1 

Joe Benton1 

Valentin De Bortoli2 

Tom Rainforth1 

George Deligiannidis1 

Arnaud Doucet1 

1Department of Statistics, University of Oxford, UK 2CNRS ENS Ulm, Paris, France {campbell, benton, rainforth, deligian, doucet}@stats.ox.ac.uk valentin.debortoli@gmail.com 

# Abstract

We provide the first complete continuous time framework for denoising diffusion models of discrete data. This is achieved by formulating the forward noising process and corresponding reverse time generative process as Continuous Time Markov Chains (CTMCs). The model can be efficiently trained using a continuous time version of the ELBO. We simulate the high dimensional CTMC using techniques developed in chemical physics and exploit our continuous time framework to derive high performance samplers that we show can outperform discrete time methods for discrete data. The continuous time treatment also enables us to derive a novel theoretical result bounding the error between the generated sample distribution and the true data distribution. 

# 1 Introduction

Diffusion/score-based/denoising models [1, 2, 3, 4] are a popular class of generative models that achieve state-of-the-art sample quality with good coverage of the data distribution [5] all whilst using a stable, non-adversarial, simple to implement training objective. The general framework is to define a forward noising process that takes in data and gradually corrupts it until the data distribution is transformed into a simple distribution that is easy to sample. The model then learns to reverse this process by learning the logarithmic gradient of the noised marginal distributions known as the score. 

Most previous work on denoising models operates on a continuous state space. However, there are many problems for which the data we would like to model is discrete. This occurs, for example, in text, segmentation maps, categorical features, discrete latent spaces, and the direct 8-bit representation of images. Previous work has tried to realize the benefits of the denoising framework on discrete data problems, with promising initial results [6, 7, 8, 9, 10, 11, 12, 13]. 

All of these previous approaches train and sample the model in discrete time. Unfortunately, working in discrete time has notable drawbacks. It generally forces the user to pick a partition of the process at training time and the model only learns to denoise at these fixed time points. Due to the fixed partition, we are then limited to a simple ancestral sampling strategy. In continuous time, the model instead learns to denoise for any arbitrary time point in the process. This complete specification of the reverse process enables much greater flexibility in defining the reverse sampling scheme. For example, in continuous state spaces, continuous time samplers that greatly reduce the sampling time have been devised [14, 15, 16, 17] as well as ones that improve sample quality [4, 18]. The continuous time interpretation has also enabled the derivation of interesting theoretical properties such as error bounds [19] in continuous state spaces. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/fc80f85ec83eb5e957af50295f3e9f30e0577dab34916ca5f7073d09ee9bfd58.jpg)



Figure 1: The forward noising process corrupts data according to $R _ { t }$ , the rate of corruption events at time t. The noising process’ time reversal gives the generative process which is defined through $\hat { R } _ { t } ^ { \theta }$ , the rate of generative events at time t. $\hat { R } _ { t } ^ { \theta }$ is parameterized through the denoising network, $p _ { 0 \mid t } ^ { \bar { \theta } } ( x _ { 0 } | x _ { t } )$ , which outputs categorical probabilities over clean $x _ { 0 }$ values conditioned on a noisy $x _ { t }$ .


To allow these benefits to be exploited for discrete state spaces as well, we formulate a continuous time framework for discrete denoising models. Specifically, our contributions are as follows. We formulate the forward noising process as a Continuous Time Markov Chain (CTMC) and identify the generative CTMC that is the time-reversal of this process. We then bound the log likelihood of the generated data distribution, giving a continuous time equivalent of the ELBO that can be used for efficient training of a parametric approximation to the true generative reverse process. To efficiently simulate the parametric reverse process, we leverage tau-leaping [20] and propose a novel predictor-corrector type scheme that can be used to improve simulation accuracy. The continuous time framework allows us to derive a bound on the error between the true data distribution and the samples generated from the approximate reverse process simulated with tau-leaping. Finally, we demonstrate our proposed method on the generative modeling of images from the CIFAR-10 dataset and monophonic music sequences. Notably, we find our tau-leaping with predictor-corrector sampler can provide higher quality CIFAR10 samples than previous discrete time discrete state approaches, further closing the performance gap between when images are modeled as discrete data or as continuous data. 

Proofs for all propositions and theorems are given in the Appendix. 

# 2 Background on Discrete Denoising Models

In the discrete time, discrete state space case, we aim to model discrete data $x _ { 0 } \in \mathcal { X }$ with finite cardinality $S = | { \mathcal { X } } |$ . We assume $x _ { 0 } \sim p _ { \mathrm { d a t a } } ( x _ { 0 } )$ ) for some discrete data distribution $p _ { \mathrm { d a t a } } ( x _ { 0 } )$ ). We define a forward noising process that transforms $p _ { \mathrm { d a t a } } ( x _ { 0 } )$ to some distribution $q _ { K } ( x _ { K } )$ that closely approximates an easy to sample distribution $p _ { \mathrm { r e f } } ( x _ { K } )$ . This is done by defining forward kernels $q _ { k + 1 | k } { \left( x _ { k + 1 } | x _ { k } \right) }$ that all admit $p _ { \mathrm { r e f } }$ as a stationary distribution and mix reasonably quickly. For example, one can use a simple uniform kernel [6, 8], $q _ { k + 1 | k } ( x _ { k + 1 } | x _ { k } ) = \delta _ { x _ { k + 1 } , x _ { k } } ( 1 - \beta ) + ( 1 -$ $\delta _ { x _ { k + 1 } , x _ { k } } ) \beta / ( S - 1 )$ where δ is a Kronecker delta. The corresponding $p _ { \mathrm { r e f } }$ is the uniform distribution over all states. Other choices include: an absorbing state kernel—where for each state there is a small probability that it transitions to some absorbing state—or a discretized Gaussian kernel—where only transitions to nearby states have significant probability (valid for spaces with ordinal structure) [8]. 

After defining $q _ { k + 1 | k }$ , we have a forward joint decomposition as follows 

$$
q _ {0: K} (x _ {0: K}) = p _ {\text { data }} (x _ {0}) \prod_ {k = 0} ^ {K - 1} q _ {k + 1 | k} (x _ {k + 1} | x _ {k}).
$$

The joint distribution $q _ { 0 : K } \big ( \boldsymbol { x } _ { 0 : K } \big )$ also admits a reverse decomposition: 

$$
q _ {0: K} (x _ {0: K}) = q _ {K} (x _ {K}) \prod_ {k = 0} ^ {K - 1} q _ {k | k + 1} (x _ {k} | x _ {k + 1}) \text {where} q _ {k | k + 1} (x _ {k} | x _ {k + 1}) = \frac {q _ {k + 1 | k} (x _ {k + 1} | x _ {k}) q _ {k} (x _ {k})}{q _ {k + 1} (x _ {k + 1})}.
$$

Here $q _ { k } ( x _ { k } )$ denotes the marginal of $q _ { 0 : K } \big ( \boldsymbol { x } _ { 0 : K } \big )$ at time k. If one had access to $q _ { k | k + 1 }$ and could sample $q _ { K }$ exactly, then samples from $p _ { \mathrm { d a t a } } ( x _ { 0 } )$ could be produced by first sampling $x _ { K } \sim q _ { K } ( \cdot )$ and then ancestrally sampling the reverse kernels, i.e. $x _ { k } \sim q _ { k | k + 1 } ( \cdot | x _ { k + 1 } )$ . 

However, in practice, $q _ { k | k + 1 }$ is intractable and needs to be approximated with a parametric reverse kernel, $p _ { k | k + 1 } ^ { \theta }$ . This kernel is commonly defined through the analytic $q _ { k | k + 1 , 0 }$ distribution and a parametric ‘denoising’ model $p _ { 0 | k + 1 } ^ { \theta } \left[ 6 , 8 \right]$ , 

$$
\begin{array}{l} p _ {k \mid k + 1} ^ {\theta} (x _ {k} \mid x _ {k + 1}) \triangleq \sum_ {x _ {0}} q _ {k \mid k + 1, 0} (x _ {k} \mid x _ {k + 1}, x _ {0}) p _ {0 \mid k + 1} ^ {\theta} (x _ {0} \mid x _ {k + 1}) \\ = q _ {k + 1 \mid k} (x _ {k + 1} \mid x _ {k}) \sum_ {x _ {0}} \frac {q _ {k \mid 0} (x _ {k} \mid x _ {0})}{q _ {k + 1 \mid 0} (x _ {k + 1} \mid x _ {0})} p _ {0 \mid k + 1} ^ {\theta} (x _ {0} \mid x _ {k + 1}). \tag {1} \\ \end{array}
$$

Though $q _ { K } ( x _ { K } )$ is also intractable, for large K we can reliably approximate it with $p _ { \mathrm { r e f } } ( x _ { K } )$ . Note that the faster the transitions mix, the more accurate this approximation becomes. Approximate samples from $p _ { \mathrm { d a t a } } ( x _ { 0 } )$ can then be obtained by sampling the generative joint distribution 

$$
p _ {0: K} ^ {\theta} (x _ {0: K}) = p _ {\mathrm{ref}} (x _ {K}) \prod_ {k = 0} ^ {K - 1} p _ {k | k + 1} ^ {\theta} (x _ {k} | x _ {k + 1}),
$$

where θ is trained through minimizing the negative discrete time (DT) ELBO which is an upper bound on the negative model log-likelihood 

$$
\mathbb {E} _ {p _ {\mathrm{data}} (x _ {0})} \left[ - \log p _ {0} ^ {\theta} (x _ {0}) \right] \leq \mathbb {E} _ {q _ {0: K} (x _ {0: K})} \left[ - \log \frac {p _ {0 : K} ^ {\theta} (x _ {0 : K})}{q _ {1 : K | 0} (x _ {1 : K} | x _ {0})} \right] = \mathcal {L} _ {\mathrm{DT}} (\theta).
$$

It was shown in [1] that $\mathcal { L } _ { \mathrm { D T } }$ can be re-written as 

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{DT}} (\theta) = \mathbb {E} _ {p _ {\text {data}} (x _ {0})} \left[ \mathrm{KL} \left(q _ {K | 0} \left(x _ {K} \mid x _ {0}\right) \mid \mid p _ {\text {ref}} \left(x _ {K}\right)\right) - \mathbb {E} _ {q _ {1 | 0} \left(x _ {1} \mid x _ {0}\right)} \left[ \log p _ {0 | 1} ^ {\theta} \left(x _ {0} \mid x _ {1}\right) \right] \right. \\ \left. + \sum_ {k = 1} ^ {K - 1} \mathbb {E} _ {q _ {k + 1 | 0} (x _ {k + 1} | x _ {0})} \left[ \mathrm{KL} \left(q _ {k | k + 1, 0} \left(x _ {k} \mid x _ {k + 1}, x _ {0}\right) \mid \mid p _ {k | k + 1} ^ {\theta} \left(x _ {k} \mid x _ {k + 1}\right)\right) \right] \right] \\ \end{array}
$$

where KL is the Kullback–Leibler divergence. The forward kernels $q _ { k + 1 | k }$ are chosen such that $q _ { k | 0 } ( x _ { k } | x _ { 0 } )$ can be computed efficiently in a time independent of k. With this, θ can be efficiently trained by taking a random selection of terms from ${ \mathcal { L } } _ { \mathrm { D T } }$ in each minibatch and performing a stochastic gradient step. 

# 3 Continuous Time Framework

# 3.1 Forward process and its time reversal

Our method is built upon a continuous time process from $t = 0 \mathrm { t o } t = T$ . State transitions can occur at any time during this process as opposed to the discrete time case where transitions only occur when one of the finite number of transition kernels is applied (see Figure 1). This process is known as a Continuous Time Markov Chain (CTMC), we provide a short overview of CTMCs in Appendix A for completeness. Giving an intuitive introduction here, we can define a CTMC through an initial distribution $q _ { 0 }$ and a transition rate matrix $R _ { t } \in \mathbb { R } ^ { S \times S }$ . If the current state is x˜, then the transition rate matrix entry $R _ { t } ( \tilde { x } , x )$ is the instantaneous rate (occurrences per unit time) at which state x˜ transitions to state $x .$ Loosely speaking, the next state in the process will likely be one for which $R _ { t } ( \tilde { x } , x )$ is high, and furthermore, the higher the rate is, the less time it will take for this transition to occur. 

It turns out that the transition rate, $R _ { t }$ , also defines the infinitesimal transition probability for the process between the two time points $t - \Delta t$ and t 

$$
q _ {t | t - \Delta t} (x | \tilde {x}) = \delta_ {x, \tilde {x}} + R _ {t} (\tilde {x}, x) \Delta t + o (\Delta t),
$$

where $o ( \Delta t )$ represents terms that tend to zero at a faster rate than $\Delta t .$ Comparing to the discrete time case, we see that $R _ { t }$ assumes an analogous role to the discrete time forward kernel $q _ { k + 1 | k }$ in how we define the forward process. Therefore, just as in discrete time, we design $R _ { t }$ such that: i) the forward process mixes quickly towards an easy to sample (stationary) distribution, $p _ { \mathrm { r e f } } .$ (e.g. uniform), ii) we can analytically obtain $q _ { t \mid 0 } \big ( x _ { t } | x _ { 0 } \big )$ distributions to enable efficient training (see Section 4.1 for how this is done). We initialize the forward CTMC at $q _ { 0 } ( x _ { 0 } ) = p _ { \mathrm { d a t a } } ( x _ { 0 } )$ at time $t = 0$ . We denote the marginal at time $t = T$ as $q _ { T } ( x _ { T } )$ , which should be close to $p _ { \mathrm { r e f } } ( x _ { T } )$ . 

We now consider the time reversal of the forward process, which will take us from the marginal $q _ { T } ( x _ { T } )$ back to the data distribution $p _ { \mathrm { d a t a } } ( x _ { 0 } )$ through a reverse transition rate matrix, $\hat { R } _ { t } \in \mathbb { R } ^ { S \times S }$ : 

$$
q _ {t | t + \Delta t} (\tilde {x} | x) = \delta_ {\tilde {x}, x} + \hat {R} _ {t} (x, \tilde {x}) \Delta t + o (\Delta t).
$$

In discrete time, one uses Bayes rule to go from $q _ { k + 1 | k } \ \mathrm { t o } \ q _ { k | k + 1 }$ . We can use similar ideas to calculate $\hat { R } _ { t }$ from $R _ { t }$ as per the following result. 

Proposition 1. For a forward in time CTMC, $\{ x _ { t } \} _ { t \in [ 0 , T ] }$ , with rate matrix $R _ { t } ,$ , initial distribution pdata(x0) and terminal distribution $q _ { T } ( x _ { T } )$ , there exists a CTMC with initial distribution $q _ { T } ( x _ { T } )$ at $t = T$ , terminal distribution $p _ { \mathrm { d a t a } } ( x _ { 0 } )$ at $t = 0$ and transition rate matrix $\hat { R } _ { t }$ that runs backwards in time and is almost everywhere equivalent to the time reversal of the forward CTMC, $\{ x _ { t } \} _ { t \in [ T , 0 ] }$ Furthermore, $\hat { R } _ { t }$ is related to $R _ { t }$ by the following expression 

$$
\hat {R} _ {t} (x, \tilde {x}) = R _ {t} (\tilde {x}, x) \sum_ {x _ {0}} \frac {q _ {t | 0} (\tilde {x} | x _ {0})}{q _ {t | 0} (x | x _ {0})} q _ {0 | t} (x _ {0} | x) \quad f o r \quad x \neq \tilde {x},
$$

where $q _ { t \mid 0 } ( x | x _ { 0 } )$ are the conditional marginals of the forward process and $q _ { 0 | t } ( x _ { 0 } | x ) =$ $q _ { t | 0 } ( x | x _ { 0 } ) p _ { \mathrm { d a t a } } ( x _ { 0 } ) / q _ { t } ( x )$ with $q _ { t } ( x )$ being the marginal of the forward process at time t. When $\begin{array} { r } { x = \tilde { x } , \hat { R } _ { t } ( x , x ) = - \sum _ { x ^ { \prime } \ne x } \hat { R } _ { t } ( x , x ^ { \prime } ) } \end{array}$ because the rows must sum to zero (see Appendix A). 

Unfortunately, $\hat { R } _ { t }$ is intractable due to the intractability of $q _ { t } ( x )$ and thus of $q _ { 0 \mid t } ( x _ { 0 } | x )$ . Therefore, we consider an approximation $\hat { R } _ { t } ^ { \theta }$ of $\hat { R } _ { t }$ by approximating $q _ { 0 \mid t } ( x _ { 0 } | x )$ with a parametric denoising model, $p _ { 0 | t } ^ { \theta } ( x _ { 0 } | x )$ : 

$$
\hat {R} _ {t} ^ {\theta} (x, \tilde {x}) = R _ {t} (\tilde {x}, x) \sum_ {x _ {0}} \frac {q _ {t | 0} (\tilde {x} | x _ {0})}{q _ {t | 0} (x | x _ {0})} p _ {0 | t} ^ {\theta} (x _ {0} | x) \quad \mathrm{for} \quad x \neq \tilde {x}
$$

and $\begin{array} { r } { \hat { R } _ { t } ^ { \theta } ( x , x ) = - \sum _ { x ^ { \prime } \neq x } \hat { R } _ { t } ^ { \theta } ( x , x ^ { \prime } ) } \end{array}$ as before. As a further analogy to the discrete time case, notice that when $x \neq \tilde { x } , \hat { R } _ { t } ^ { \theta }$ has the same form as the discrete time parametric reverse kernel, $p _ { k | k + 1 } ^ { \theta }$ defined in eq (1) but with the forward kernel, $q _ { k + 1 | k }$ , replaced by the forward rate, $R _ { t }$ . 

# 3.2 Continuous Time ELBO

In discrete time, θ is trained by minimizing the discrete time negative ELBO, ${ \mathcal { L } } _ { \mathrm { D T } }$ , formed from the forward and reverse processes. We mirror this approach in continuous time by minimizing the corresponding continuous time (CT) negative ELBO, $\mathcal { L } _ { \mathrm { C T } }$ , as derived below. 

Proposition 2. For the reverse in time CTMC with initial distribution $p _ { r e f } ( x _ { T } )$ , terminal distribution $p _ { 0 } ^ { \theta } ( x _ { 0 } )$ , and reverse rate $\hat { R } _ { t } ^ { \theta }$ , an upper bound on the negative model log-likelihood, $\mathbb { E } _ { p _ { \mathrm { d a t a } } ( x _ { 0 } ) } [ - \log p _ { 0 } ^ { \theta } ( x _ { 0 } ) \dot { }$ ], is given by 

$$
\mathcal {L} _ {\mathrm{CT}} (\theta) = T \mathbb {E} _ {t \sim \mathcal {U} (0, T) q _ {t} (x) r _ {t} (\tilde {x} | x)} \Big [ \Big \{\sum_ {x ^ {\prime} \neq x} \hat {R} _ {t} ^ {\theta} (x, x ^ {\prime}) \Big \} - \mathcal {Z} ^ {t} (x) \log \Big (\hat {R} _ {t} ^ {\theta} (\tilde {x}, x) \Big) \Big ] + C,
$$

where C is a constant independent of θ and 

$$
\mathcal {Z} ^ {t} (x) = \sum_ {x ^ {\prime} \neq x} R _ {t} (x, x ^ {\prime}) \quad r _ {t} (\tilde {x} | x) = (1 - \delta_ {\tilde {x}, x}) R _ {t} (x, \tilde {x}) / \mathcal {Z} ^ {t} (x).
$$

Here $r _ { t } ( \tilde { x } | x )$ gives the probability of transitioning from x to ${ \tilde { x } } ,$ given that we know a transition occurs at time t. We can optimize this objective efficiently with stochastic gradient descent. For a gradient update, we sample a batch of datapoints from $p _ { \mathrm { d a t a } } ( x _ { 0 } )$ , noise each datapoint using a random time, $\bar { t \mapsto { \mathcal { U } } } ( 0 , T ) , x \bar { \sim } q _ { t | 0 } ( x | x _ { 0 } )$ and finally sample an auxiliary x˜ from $r _ { t } ( \bar { \tilde { x } } | x )$ for each x. Intuitively, $( x , { \tilde { x } } )$ are a pair of states following the forward in time noising process. Minimizing the second term in $\mathcal { L } _ { \mathrm { C T } }$ maximizes the reverse rate for this pair, but going in the backwards direction, x˜ to x. This is how $\hat { R } _ { t } ^ { \theta }$ learns to reverse the noising process. Intuition on the first term and a direct comparison to ${ \mathcal { L } } _ { \mathrm { D T } }$ is given in Appendix C.1. 

The first argument of $\hat { R } _ { t } ^ { \theta }$ is input into $p _ { 0 \mid t } ^ { \theta }$ so we naively require two network forward passes on x and x˜ to evaluate the objective. We can avoid this by approximating the $q _ { t } ( x )$ sample in the first term with x˜ meaning we need only evaluate the network once on ${ \tilde { x } } .$ The approximation is valid because, as we show in Appendix C.4, x˜ is approximately distributed according to $q _ { t + \delta t }$ for $\delta t$ very small. 

# 4 Efficient Forward and Backward Sampling

# 4.1 Choice of Forward Process

The transition rate matrix $R _ { t }$ needs to be chosen such that the forward process: i) mixes quickly towards $p _ { \mathrm { r e f } } .$ , and ii) the $q _ { t \mid 0 } ( x | x _ { 0 } )$ distributions can be analytically obtained. The Kolmogorov differential equation for the CTMC needs to be integrated to obtain $q _ { t \mid 0 } ( x | x _ { 0 } )$ ). This can be done analytically when $R _ { t }$ and $R _ { t ^ { \prime } }$ commute for all $t , t ^ { \prime } ,$ , see Appendix E. An easy way to meet this condition is to let $R _ { t } = \beta ( t ) R _ { b }$ where $R _ { b } \in \mathbb { R } ^ { S \times S }$ is a user-specified time independent base rate matrix and $\beta ( t ) \in \mathbb { R }$ is a time dependent scalar. We then obtain the analytic expression 

$$
q _ {t | 0} (x = j | x _ {0} = i) = \left(Q \exp \left[ \Lambda \int_ {0} ^ {t} \beta (s) d s \right] Q ^ {- 1}\right) _ {i j}
$$

where ${ \cal R } _ { b } = Q \Lambda Q ^ { - 1 }$ is the eigendecomposition of matrix $R _ { b }$ and exp[·] the element-wise exponential. 

Our choice of $\beta$ schedule is guided by $[ 3 , 4 ] , \beta ( t ) = a b ^ { t } \log ( b )$ . The hyperparameters a and b are selected such that $q _ { T } ( x ) \approx p _ { \mathrm { r e f } } ( x )$ at the terminal time $t = T$ while having a steady speed of ‘information corruption’ which ensures that $\hat { R } _ { t }$ does not vary quickly in a short span of time. 

We experiment with a variety of $R _ { b }$ matrices, for example, a uniform rate, $R _ { b } = \mathbb { 1 } \mathbb { 1 } ^ { T } - S \mathrm { I d }$ , where $\mathbb { 1 1 } ^ { T }$ is a matrix of ones and Id is the identity. For problems with a heavy spatial bias, e.g. images, we can instead use a forward rate that only encourages transitions to nearby states; details and the links to the corresponding discrete time processes can be found in Appendix E. 

# 4.2 Factorizing Over Dimensions

Our aim is to model data that is D dimensional, with each dimension taking one value from S possibilities. We now slightly redefine notation and say $\pmb { x } ^ { 1 : D } \in \mathcal { X } ^ { D } , | \mathcal { X } | = S$ . In this setting, calculating transition probabilities naively would require calculating $S ^ { D }$ rate values corresponding to each of the possible next states. This is intractable for any reasonably sized S and D. We avoid this problem simply by factorizing the forward process such that each dimension propagates independently. Since this is a continuous time process and each dimension’s forward process is independent of the others, the probability two or more dimensions transition at exactly the same time is zero. Therefore, overall in the full dimensional forward CTMC, each transition only ever involves a change in exactly one dimension. For the time reversal CTMC, it will also be true that exactly one dimension changes in each transition. This makes computation tractable because of the $S ^ { \check { D } }$ rate values, only $D \bar { \times } \left( S - 1 \right) + 1$ are non-zero - those corresponding to transitions where exactly one dimension changes plus the no change transition. Finally, we note that even though dimensions propagate independently in the forward direction, they are not independent in the reverse direction because the starting points for each dimension’s forward process are not independent for non factorized $p _ { \mathrm { d a t a } }$ . The following proposition shows the exact forms for the forward and reverse rates in this case. 

Proposition 3. If the forward process factorizes as $\begin{array} { r } { q _ { t | s } ( \pmb { x } _ { t } ^ { 1 : D } | \pmb { x } _ { s } ^ { 1 : D } ) = \prod _ { d = 1 } ^ { D } q _ { t | s } ( \pmb { x } _ { t } ^ { d } | \pmb { x } _ { s } ^ { d } ) , t > s , } \end{array}$ then the forward and reverse rates are of the form 

$$
R _ {t} ^ {1: D} (\tilde {\boldsymbol {x}} ^ {1: D}, \boldsymbol {x} ^ {1: D}) = \sum_ {d = 1} ^ {D} R _ {t} ^ {d} (\tilde {x} ^ {d}, x ^ {d}) \delta_ {\boldsymbol {x} ^ {1: D \setminus d}, \tilde {\boldsymbol {x}} ^ {1: D \setminus d}},
$$

$$
\hat {R} _ {t} ^ {1: D} (\pmb {x} ^ {1: D}, \tilde {\pmb {x}} ^ {1: D}) = \sum_ {d = 1} ^ {D} R _ {t} ^ {d} (\tilde {x} ^ {d}, x ^ {d}) \delta_ {\pmb {x} ^ {1: D \setminus d}, \tilde {\pmb {x}} ^ {1: D \setminus d}} \sum_ {x _ {0} ^ {d}} q _ {0 | t} (x _ {0} ^ {d} | \pmb {x} ^ {1: D}) \frac {q _ {t | 0} (\tilde {x} ^ {d} | x _ {0} ^ {d})}{q _ {t | 0} (x ^ {d} | x _ {0} ^ {d})},
$$

where $R _ { t } ^ { d } \in \mathbb { R } ^ { S \times S }$ and $\delta _ { { \pmb x } ^ { 1 : D \setminus d } , \tilde { { \pmb x } } ^ { 1 : D \setminus d } }$ is 1 when all dimensions except for d are equal. 

To find $\hat { R } _ { t } ^ { \theta 1 : D }$ we simply replace $q _ { 0 | t } \big ( x _ { 0 } ^ { d } | \mathbf { x } ^ { 1 : D } \big )$ with $p _ { 0 | t } ^ { \theta } ( x _ { 0 } ^ { d } | \mathbf { x } ^ { 1 : D } )$ which is easily modeled with a neural network that outputs conditionally independent state probabilities in each dimension. In Appendix C.3 we derive the form of ${ \mathcal { L } } _ { \mathrm { C T } }$ when we use this factorized form for $R _ { t } ^ { 1 : D }$ and $\hat { R } _ { t } ^ { \theta 1 : D }$ . 

# 4.3 Simulating the Generative Reverse Process with Tau-Leaping

The parametric generative reverse process is a CTMC with rate matrix $\hat { R } _ { t } ^ { \theta 1 : D }$ . Simulating this process from distribution $p _ { \mathrm { r e f } } ( \pmb { x } _ { T } ^ { 1 : D } )$ at time $t = T$ back to $t = 0$ will produce approximate samples from $p _ { \mathrm { d a t a } } ( \pmb { x } _ { 0 } ^ { 1 : D } )$ . The process could be simulated exactly using Gillespie’s Algorithm [21, 22, 23] which alternates between i) sampling a holding time to remain in the current state and ii) sampling a new state according to the current rate matrix, $\hat { R } _ { t } ^ { \theta 1 : D }$ (see $\operatorname { A p p e n d i x } F )$ . This is inefficient for large D because we would need to step through each transition individually and so only one dimension would change for each simulation step. 

Instead, we use tau-leaping [20, 23], a very popular approximate simulation method developed in chemical physics. Rather than step back through time one transition to the next, tau-leaping leaps from t to $t - \tau$ and applies all transitions that occurred in $[ t - \tau , t ]$ simultaneously. To make a leap, we assume $\hat { R } _ { t } ^ { \theta 1 : D }$ and $\pmb { x } _ { t } ^ { 1 : D }$ remain constant in $[ t - \tau , t ]$ . As we propagate from t to $t - \tau$ , we count all of the transitions that occur, but hold off on actually applying them until we reach $t - \tau$ , such that $\pmb { x } _ { t } ^ { 1 : D }$ remains constant in $\lceil t - \tau , t \rceil$ . Assuming $\hat { R } _ { t } ^ { \theta 1 : D }$ and $\pmb { x } _ { t } ^ { 1 : D }$ remain constant, the number of times a transition from $\pmb { x } _ { t } ^ { 1 : D }$ to $\widetilde { \pmb { x } } ^ { 1 : D }$ occurs in $[ t - \tau , t ]$ is Poisson distributed with mean $\tau \hat { R } _ { t } ^ { \theta 1 : D } ( \pmb { x } _ { t } ^ { 1 : D } , \tilde { \pmb { x } } ^ { 1 : D } )$ ). Once we reach  1:D $t - \tau$ , we apply all transitions that occurred simultaneously ti.e. $\begin{array} { r } { \pmb { x } _ { t - \tau } ^ { 1 : D } = \pmb { x } _ { t } ^ { 1 : D } + \sum _ { i } P _ { i } ( \tilde { \pmb { x } } _ { i } ^ { 1 : D } - \pmb { x } _ { t } ^ { 1 : D } ) } \end{array}$ x1:t − x 1: Dt ) where $P _ { i }$ is a Poisson random variable with mean $\tau \hat { R } _ { t } ^ { \theta 1 : D } ( \pmb { x } _ { t } ^ { 1 : D } , \tilde { \pmb { x } } _ { i } ^ { 1 : D } )$ ). Note the sum assumes a mapping from X to Z. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/b4fa706f8998e5e257e7c1541d5c9c51dce9113c5c136ee0c21049c0adf7afe3.jpg)



Figure 2: 3D visualization of one tau-leaping step from $x _ { t } ^ { 1 : 2 } = \{ S _ { 4 } , \dot { S } _ { 1 } \} \stackrel { \smile } { \mathrm { t o } } x _ { t - \tau } ^ { 1 : 2 } =$ $\{ \breve { S } _ { 2 } , S _ { 3 } \}$ . Here, $\textit { D } = \textit { 2 }$ , $| \mathcal { X } | = 5 , P _ { 1 2 } = 1 , P _ { 2 2 } = 2$ , all other $P _ { d s } = 0$ .


Using our knowledge of $\hat { R } _ { t } ^ { \theta 1 : D }$ , we can further unpack this update. Namely, $\hat { R } _ { t } ^ { \theta 1 : D } ( \pmb { x } _ { t } ^ { 1 : D } , \tilde { \pmb { x } } ^ { 1 : D } )$ can only be non-zero when $\tilde { \pmb { x } } ^ { 1 : D }$ has t ta different value to $\pmb { x } _ { t } ^ { 1 : D }$ in exactly one dimension (rates for multidimensional changes are zero). Explicitly summing over these options we get $\begin{array} { r } { \pmb { x } _ { t - \tau } ^ { 1 : D } = \pmb { \bar { x _ { t } ^ { 1 : D } } } + \sum _ { d = 1 } ^ { D } \sum _ { s = 1 \backslash x _ { { t } } ^ { d } } ^ { \bar { S } } P _ { d s } ( s - \acute { x _ { t } ^ { d } } ) e ^ { d } } \end{array}$ x1:t 1 where $e ^ { d }$ is a one-hot vector with a 1 at dimension d and $P _ { d s }$ is a Poisson random variable with mean $\tau \hat { R } _ { t } ^ { \theta 1 : D } ( \pmb { x } _ { t } ^ { 1 : D } , \pmb { x } _ { t } ^ { 1 : D } + ( s - x _ { t } ^ { d } ) e ^ { d } )$ . Since multiple $P _ { d s }$ can be non-zero, we see that tau-leaping allows $\pmb { x } _ { t } ^ { 1 : D }$ to change in multiple dimensions in a single step. Figure 2 visualizes this idea. During the $[ t - \tau , t ]$ interval, one jump occurs in dimension 1 and two jumps occur in dimension 2. These are all applied simultaneously once we reach $t - \tau$ . When our discrete data has ordinal structure (e.g. Section 6.2) our mapping tojumps within the same dime $\mathbb { Z }$ aking multiple is meaningful. $\begin{array} { r } { ( \sum _ { s = 1 \setminus x _ { + } ^ { d } } ^ { S } P _ { d s } > 1 ) } \end{array}$ Z is arbitrary and so, although taking simultaneous jumps in different dimensions is meaningful, taking multiple jumps within the same tany d for which PSs=1\xdt Pds > 1. In practice, the rejection rate isvery small when R1:Dt is suitable for categorical data (e.g. uniform), any d for which dimension is not. For this type of data, we reject changes to $\textstyle \sum _ { s = 1 \setminus x _ { * } ^ { d } } ^ { S } { \dot { P _ { d s } } } > 1$ $\boldsymbol { x } _ { t } ^ { d }$ for $\widetilde { R } _ { t } ^ { \mathrm { f } : \overline { { D } } ^ { \perp } \mathrm { i } s }$ suitable for categorical data (e.g. uniform), see Appendix H.3. In Section 4.5, our error bound accounts for this low probability of rejection and also the low probability of an out of bounds jump that we observe in practice in the ordinal case. 

The tau-leaping approximation improves with smaller τ , recovering exact simulation in the limit as $\tau  0$ . Exact simulation is similar to an autoregressive model in that only one dimension changes per step. Increasing τ and thus the average number of dimensions changing per step gives us a natural way to modulate the ‘autoregressiveness’ of the model and trade sample quality with compute (Figure 4 right). We refer to our method of using tau-leaping to simulate the reverse CTMC as τ LDR (tau-leaping denoising reversal) which we formalize in Algorithm 1 in Appendix F. 

We note that theoretically, one could approximate $\hat { R } _ { t } ^ { \theta 1 : D }$ as constant in the interval $[ t - \tau , t ]$ , and construct a transition probability matrix by solving the forward Kolmogorov equation with the matrix exponential Pt−τ|t ≈ exp(τ Rˆθ 1:Dt ). However, for the learned Rˆθ 1:Dt ∈ RSD×SD $P _ { t - \tau | t } \approx \exp ( \tau \hat { R } _ { t } ^ { \theta 1 : D } )$ $\hat { R } _ { t } ^ { \theta 1 : D } \in \mathbb { R } ^ { S ^ { D } \times S ^ { D } }$ matrix, it is intractable to compute this matrix exponential so we use tau-leaping for sampling instead. 

# 4.4 Predictor-Corrector

During approximate reverse sampling, we aim for the marginal distribution of samples at time t to be close to $q _ { t } ( x _ { t } )$ (the marginal at time t of the true CTMC). The continuous time framework allows us to exploit additional information to more accurately follow the reverse progression of marginals, $\{ q _ { t } ( x _ { t } ) \bar  \} _ { t \in [ T , 0 ] }$ and improve sample quality. Namely, after a tau-leaping ‘predictor’ step using rate $\hat { R } _ { t } ^ { \theta }$ , we can apply ‘corrector’ steps with rate $R _ { t } ^ { c }$ which has $q _ { t } ( x _ { t } )$ as its stationary distribution. The corrector steps bring the distribution of samples at time t closer to the desired $q _ { t } ( x _ { t } )$ marginal. $R _ { t } ^ { c }$ is easy to calculate as stated below 

Proposition 4. For a forward CTMC with marginals $\{ q _ { t } ( x _ { t } ) \} _ { t \in [ 0 , T ] }$ , forward rate, $R _ { t } ,$ , and corresponding reverse CTMC with rate $\hat { R } _ { t } ,$ , the rate $R _ { t } ^ { c } = R _ { t } + { \hat { R } } _ { t }$ has $q _ { t } ( x _ { t } )$ as its stationary distribution. 

In practice, we approximate $R _ { t } ^ { c }$ by replacing $\hat { R } _ { t }$ with $\hat { R } _ { t } ^ { \theta }$ . This is directly analogous to Predictor-Corrector samplers in continuous state spaces [4] that predict by integrating the reverse SDE and correct with score-based Markov chain Monte Carlo steps, see Appendix F.2 for further discussion. 

# 4.5 Error Bound

Our continuous time framework also allows us to provide a novel theoretical bound on the error between the true data distribution and the sample distribution generated via tau-leaping (without predictor-corrector steps), in terms of the error in our approximation of the reverse rate and the mixing of the forward noising process. 

We assume we have a time-homogeneous rate matrix $R _ { t }$ on $x ,$ , from which we construct the factorized rate matrix $R _ { t } ^ { 1 : D }$ on $\chi D$ by setting $R _ { t } ^ { d } = R _ { t }$ for each d. Note that by rescaling time by a factor of $\beta ( t )$ we can transform our choice of rate from Section 4.1 to be time-homogeneous. We will denote $\begin{array} { r } { \vert \dot { R \vert } ^ { - } = \operatorname* { s u p } _ { t \in [ 0 , T ] , x \in \mathcal { X } } \vert R _ { t } ( x , x ) \vert } \end{array}$ , and let $t _ { \mathrm { m i x } }$ be the (1/4)-mixing time of the CTMC with rate $R _ { t }$ (see [24, Chapter 4.5]). 

Theorem 1. For any $D \geq 1$ and distribution $p _ { \mathrm { d a t a } }$ on $\mathcal { X } ^ { D }$ , let $\{ x _ { t } \} _ { t \in [ 0 , T ] }$ be a CTMC starting in $p _ { \mathrm { d a t a } }$ with rate matrix $R _ { t } ^ { 1 : D }$ as above. Suppose that $\hat { R } _ { t } ^ { \theta 1 : D }$ is an approximation to the reverse rate matrix and let $( y _ { k } ) _ { k = 0 , 1 , \dots , N }$ be a tau-leaping approximation to the reverse dynamics with maximum step size τ . Suppose further that there is some constant $M > 0$ independent of D such that 

$$
\sum_ {y \neq x} \left| \hat {R} _ {t} ^ {1: D} (x, y) - \hat {R} _ {t} ^ {\theta 1: D} (x, y) \right| \leq M \tag {2}
$$

for all $t \in [ 0 , T ]$ . Then under the assumptions in Appendix B.5, there are constants $C _ { 1 } , C _ { 2 } > 0$ depending on X and $R _ { t }$ but not D such that, $i f \ : \mathcal { L } ( y _ { 0 } )$ denotes the law of $y _ { 0 } ,$ , we have the total variation bound 

$$
| | \mathcal {L} (y _ {0}) - p _ {\mathrm{data}} | | _ {\mathrm{TV}} \leq 3 M T + \left\{\left(| R | S D C _ {1}\right) ^ {2} + \frac {1}{2} C _ {2} (M + C _ {1} S D | R |) \right\} \tau T + 2 \exp \left\{- \frac {T \log^ {2} 2}{t _ {\mathrm{mix}} \log 4 D} \right\}
$$

The first term of the above bound captures the error introduced by our approximation of the reverse rate $\hat { R } _ { t } ^ { 1 : D }$ with $\hat { R } _ { t } ^ { \theta 1 : D }$ . The second term reflects the error introduced by the tau-leaping approximation, and is linear in both $T$ and $\tau ,$ showing that as we take our tau-leaping steps to be arbitrarily small, the error introduced by tau-leaping goes to zero. The final term describes the mixing of the forward chain, and captures the error introduced since $p _ { \mathrm { r e f } }$ and $q _ { T }$ are not exactly equal. 

We choose to make the dependence of the bound on the dimension D explicit, since we are specifically interested in applying tau-leaping to high dimensional problems where we make transitions in different dimensions simultaneously in a single time step. The bound grows at worst quadratically in the dimension, versus e.g. exponentially. The bound is therefore useful in showing us that we do not need to make τ impractically small in high dimensions. Other than gaining these intuitions, we do not expect the bound to be particularly tight in practice and further it would not be practical to compute because of the difficulty in finding M, C1 and $C _ { 2 }$ . 

The assumptions listed in Appendix B.5 hold approximately for tau-leaping in practice when we use spatially biased rates for ordinal data such that jump sizes are small or uniform rates for non-ordinal data such that the dimensional rejection rate is small. These assumptions could be weakened, however, Theorem 1 would become much more involved, obscuring the intuition and structure of the problem. 

# 5 Related Work

The application of denoising models to discrete data was first described in [1] using a binomial diffusion process for a binary dataset. Each reverse kernel $p _ { k | k + 1 } ^ { \theta }$ was directly parameterized without using a denoising model $p _ { 0 | k } ^ { \theta }$ . In [25] an approach for discrete categorical data was suggested using a uniform forward noising kernel, $q _ { k + 1 | k }$ , and a reverse kernel parameterized through a denoising model, though no experiments were performed with the approach. Experiments on text and segmentation maps were then performed with a similar model in [6]. Other forward kernels were introduced in [8] that are more appropriate for certain data types such as the spatially biased Gaussian kernel. [9, 13] apply the approach to discrete latent space modeling using uniform and absorbing state forward kernels. Whilst a link to continuous time for the forward process is mentioned in [8], all of these approaches train and sample in discrete time. We show in Appendix G that this involves making an implicit approximation for multi-dimensional data. We extend this line of work by training and sampling in continuous time. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/e910dbb1750fa2d2a6068715b19a9701fc9f907d222f858d5b1df3845e8187dd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/004b0d9b642d264f234468bf81c78a1681d4cd7a011ba6ae340f770c54556dfe.jpg)



Figure 3: Left: Hellinger distance between the true training distribution and generated sample distributions with exact simulation or tau-leaping. With τ small, we simulate the reverse CTMC with the same fidelity as the exact simulation. Top Right: Histograms of the marginals during the reverse generative process simulated using tau-leaping with $\tau = 0 . 0 0 4$ . Darker and larger diamonds represent increased density. Bottom Right: The same for $\tau = 0 . 1$ , note the reduced sample quality.


Other works also operate in discrete space but less rigidly follow the diffusion framework. A corruption process tailored to text is proposed in [12], whereby token deletion and insertion is also incorporated. [26] also focus on text, creating a generative reverse chain that repeatedly applies the same denoising kernel. The corruption distribution is also defined through the same denoising kernel to reduce distribution shift between training and sampling. In [7], a more standard masking based forward process is used but the reversal is interpreted from an order agnostic autoregressive perspective. They also describe how their model can be interpreted as the reversal of a continuous time absorbing state diffusion but do not utilize this perspective in training or sampling. [27] propose a denoising type framework that can be used on binary data where the forward and reverse process share the same transition kernel. Finally, in [11], the discrete latent space of a VQVAE is modeled by quantizing an underlying continuous state space diffusion with probabilistic quantization functions. 

# 6 Experiments

# 6.1 Demonstrative Example

We first verify the method can accurately produce samples from the entire support of the data distribution and that tau-leaping can accurately simulate the reverse CTMC. To do this, we create a dataset formed of 2d samples of a state space of 32 arranged such that the histogram of the training dataset forms a $\cdot _ { \tau } ,$ shape. We train a denoising model using the $\mathcal { L } _ { \mathrm { C T } }$ objective with $p _ { 0 | t } ^ { \theta }$ parameterized through a residual MLP (full details in Appendix H.1). We then sample the parameterized reverse process using an exact method (up to needing to numerically integrate the reverse rate) and tauleaping. Figure 3 top-right shows the marginals during reverse simulation with $\tau = 0 . 0 0 4$ and we indeed produce samples from the entire support of $p _ { \mathrm { d a t a } }$ . Furthermore, we find that with sufficiently small τ , we can match the fidelity of exact simulation of the reverse CTMC (Figure 3 left). The value of τ dictates the number of network evaluations in the reverse process according to $\mathrm { N F E } = T / \tau$ . In all experiments we use $T = 1$ . Exact simulation results in a non zero Hellinger distance between the generated and training distributions because of imperfections in the learned $\hat { R } _ { t } ^ { \theta }$ model. 

# 6.2 Image Modeling

We now demonstrate that our continuous time framework gives us improved generative modeling performance versus operating in discrete time. We show this on the CIFAR-10 image dataset. Images are typically stored as discrete data, each pixel channel taking one value from 256 possibilities. Continuous state space methods have to somehow get around this fact by, for example, adding a discretization function at the end of the generative process [3] or adding uniform noise to the data. 


Table 1: Sample quality metrics and model likelihoods for diffusion methods modeling CIFAR10 in discrete state space. Diffusion methods modeling CIFAR10 in continuous space are included for reference. The Inception Score (IS) and Fréchet Inception Distance (FID) are calculated using 50000 generated samples with respect to the training dataset as is standard practice. The ELBO values are reported on the test set in bits per dimension.


<table><tr><td></td><td>Method</td><td>IS (↑)</td><td>FID (↓)</td><td>ELBO (↑)</td></tr><tr><td rowspan="4">Discrete state</td><td>D3PM Absorbing [8]</td><td>6.78</td><td>30.97</td><td>-4.40</td></tr><tr><td>D3PM Gauss [8]</td><td>8.56</td><td>7.34</td><td>-3.44</td></tr><tr><td>τLDR-0 (ours)</td><td>8.74</td><td>8.10</td><td>-3.59</td></tr><tr><td>τLDR-10 (ours)</td><td>9.49</td><td>3.74</td><td>-3.59</td></tr><tr><td rowspan="2">Continuous state</td><td>DDPM [3]</td><td>9.46</td><td>3.17</td><td>-3.75</td></tr><tr><td>NCSN [4]</td><td>9.89</td><td>2.20</td><td>-</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/981af6b9c7d2b4a51cc0e2fb2ac88f02d7e406c19911b1681891096220d58b71.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/83c6d6eef3d74912971501638a57e53c78549175eb738f89cd527d72180eb9d1.jpg)



Figure 4: Left: Unconditional CIFAR10 samples from our τ LDR-10 model Right: FID scores for the generated CIFAR10 samples versus number of $p _ { 0 | t } ^ { \theta }$ evaluations during sampling (variation induced by varying τ ). Calculated with 10k samples, hence the discrepancy with Table 1 [28].


Here, we model the images directly in discrete space. We parameterize $p _ { 0 | t } ^ { \theta }$ using the standard U-net architecture [3] with the modifications for discrete state space suggested by [8]. We use a spatially biased rate matrix and train with an augmented ${ \mathcal { L } } _ { \mathrm { C T } }$ loss including direct $\bar { p } _ { 0 \mid t } ^ { \theta ^ { - } }$ supervision, full experimental details are in Appendix H.2. 

Figure 4 left shows randomly generated unconditional CIFAR10 samples from the model and we report sample quality metrics in Table 1. We see that our method (τ LDR-0) with 0 corrector steps has better Inception Score but worse FID than the D3PM discrete time method. However, our τ LDR-10 method with 10 corrector steps per predictor step at the end of the reverse sampling process $( t < 0 . 1 T )$ greatly improves sample quality, beating the discrete time method in both metrics and further closes the performance gap with methods modeling images as continuous data. The derivation of the corrector rate which gave us this improved performance required our continuous time framework. D3PM achieves the highest ELBO but we note that this does not correlate well with sample quality. In Table 1, τ was adjusted such that both τ LDR-0 and τ LDR-10 used 1000 $p _ { 0 | t } ^ { \theta }$ evaluations in the reverse sampling procedure. We show how FID score varies with number of $\overset { \vartriangle } { p _ { 0 \mid t } ^ { \theta } }$ evaluations for τ LDR-{0, 3, 10} in Figure 4 right. The optimum number of corrector steps depends on the sampling budget, with lower numbers of corrector steps being optimal for tighter budgets. This is due to the increased τ required to maintain a fixed budget when we use a larger number of corrector steps. 

# 6.3 Monophonic Music

In this experiment, we demonstrate our continuous time model improves generation quality on non-ordinal/categorical discrete data. We model songs from the Lakh pianoroll dataset [29, 30]. We select all monophonic sequences from the dataset such that at each of the 256 time steps either one from 128 notes is played or it is a rest. Therefore, our data has state space size $S = 1 2 9$ and dimension $D = 2 5 6$ . We scramble the ordering of the state space when mapping to Z to destroy any ordinal structure. We parameterize $p _ { 0 | \ i } ^ { \theta }$ t with a transformer architecture [31] and train using a conditional form of $\mathcal { L } _ { \mathrm { C T } }$ targeting the conditional distribution of the final 14 bars (224 time steps) given the first 2 bars of the song. We use a uniform forward rate matrix, $R _ { t }$ , full experimental details are given in Appendix H.3. Conditional completions of unseen test songs are shown in Figure 5. The model is able to faithfully complete the piece in the same style as the conditioning bars. 


Table 2: Metrics comparing generated conditional samples and ground truth completions. We compute these over the test set showing mean±std with respect to 5 samples for each test song.


<table><tr><td>Model</td><td>Hellinger Distance</td><td>Proportion of Outliers</td></tr><tr><td>τLDR-0 Birth/Death</td><td>0.3928 ± 0.0010</td><td>0.1316 ± 0.0012</td></tr><tr><td>τLDR-0 Uniform</td><td>0.3765 ± 0.0013</td><td>0.1106 ± 0.0010</td></tr><tr><td>τLDR-2 Uniform</td><td>0.3762 ± 0.0015</td><td>0.1091 ± 0.0014</td></tr><tr><td>D3PM Uniform [8]</td><td>0.3839 ± 0.0002</td><td>0.1137 ± 0.0010</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/807ee3707834d6b760c3fe1b40639f185ca87f63be0c1ec2d6bca4e5059cf7ce.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/418e8fdfe4644ef8c26ff482dfe6dbbfc1b327256e14a4df3a95db8013f4f95e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/6f0b4a4c0cc83ca5132ba75faf35d248b4f5ac8a55d6df364923602e1934617f.jpg)



Figure 5: Conditional completions of an unseen music sequence. The conditioning 2 bars are shown to the left of the black line. More examples and audio recordings are linked in Appendix H.3.


We quantify sample quality in Table 2. We use two metrics: the Hellinger distance between the histograms of generated and ground truth notes and the proportion of outlier notes in the generations but not in the ground truth. Using our method, we compare between a birth/death and uniform forward rate matrix $R _ { t } .$ . The birth/death rate is only non-zero for adjacent states whereas the uniform rate allows transitions between arbitrary states which is more appropriate for the categorical case thus giving improved sample quality. Adding 2 corrector steps per predictor step further improves sample quality. We also compare to the discrete time method D3PM [8] with its most suitable corruption process for categorical data. We find it performs worse than our continuous time method. 

# 7 Discussion

We have presented a continuous time framework for discrete denoising models. We showed how to efficiently sample the generative process with tau-leaping and provided a bound on the error of the generated samples. On discrete data problems, we found our predictor-corrector sampler improved sample quality versus discrete time methods. Regarding limitations, our model requires many model evaluations to produce a sample. Our work has opened the door to applying the work improving sampling speed on continuous data [14, 15, 16, 17, 32] to discrete data problems too. Modeling performance on images is also slightly behind continuous state space models, we hope this gap is further closed with bespoke discrete state architectures and corruption process tuning. Finally, we note that the ELBO values for the discrete time model on CIFAR10 are better than for our method. In this work, we focused on sample quality rather than using our model to give data likelihoods e.g. for compression downstream tasks. 

# Acknowledgements

Andrew Campbell and Joe Benton acknowledge support from the EPSRC CDT in Modern Statistics and Statistical Machine Learning (EP/S023151/1). Arnaud Doucet is partly supported by the EPSRC grant EP/R034710/1. He also acknowledges support of the UK Defence Science and Technology Laboratory (DSTL) and EPSRC under grant EP/R013616/1. This is part of the collaboration between US DOD, UK MOD and UK EPSRC under the Multidisciplinary University Research Initiative. This project made use of time on Tier 2 HPC facility JADE2, funded by EPSRC (EP/T022205/1). 

# References



[1] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. International Conference on Machine Learning, 2015. 





[2] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in Neural Information Processing Systems, 2019. 





[3] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 2020. 





[4] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. International Conference on Learning Representations, 2021. 





[5] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat GANs on image synthesis. Advances in Neural Information Processing Systems, 2021. 





[6] Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. Advances in Neural Information Processing Systems, 2021. 





[7] Emiel Hoogeboom, Alexey A Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, and Tim Salimans. Autoregressive diffusion models. International Conference on Learning Representations, 2022. 





[8] Jacob Austin, Daniel Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in Neural Information Processing Systems, 2021. 





[9] Patrick Esser, Robin Rombach, Andreas Blattmann, and Bjorn Ommer. Imagebart: Bidirectional context with multinomial diffusion for autoregressive image synthesis. Advances in Neural Information Processing Systems, 2021. 





[10] Emiel Hoogeboom, Victor Garcia Satorras, Clément Vignac, and Max Welling. Equivariant diffusion for molecule generation in 3d. International Conference on Machine Learning, 2022. 





[11] Max Cohen, Guillaume Quispe, Sylvain Le Corff, Charles Ollion, and Eric Moulines. Diffusion bridges vector quantized variational autoencoders. International Conference on Machine Learning, 2022. 





[12] Daniel D Johnson, Jacob Austin, Rianne van den Berg, and Daniel Tarlow. Beyond in-place corruption: Insertion and deletion in denoising probabilistic models. ICML Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models (INNF+), 2021. 





[13] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. 





[14] Alexia Jolicoeur-Martineau, Ke Li, Rémi Piché-Taillefer, Tal Kachman, and Ioannis Mitliagkas. Gotta go fast when generating data with score-based models. arXiv preprint arXiv:2105.14080, 2021. 





[15] Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. arXiv preprint arXiv:2204.13902, 2022. 





[16] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. International Conference on Learning Representations, 2022. 





[17] Hyungjin Chung, Byeongsu Sim, and Jong Chul Ye. Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse problems through stochastic contraction. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. 





[18] Tim Dockhorn, Arash Vahdat, and Karsten Kreis. Score-based generative modeling with critically-damped Langevin diffusion. International Conference on Learning Representations, 2022. 





[19] Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion schrödinger bridge with applications to score-based generative modeling. Advances in Neural Information Processing Systems, 2021. 





[20] Daniel T Gillespie. Approximate accelerated stochastic simulation of chemically reacting systems. The Journal of Chemical Physics, 115(4):1716–1733, 2001. 





[21] Daniel T Gillespie. A general method for numerically simulating the stochastic time evolution of coupled chemical reactions. Journal of Computational Physics, 22(4):403–434, 1976. 





[22] Daniel T Gillespie. Exact stochastic simulation of coupled chemical reactions. The Journal of Physical Chemistry, 81(25):2340–2361, 1977. 





[23] Darren J Wilkinson. Stochastic Modelling for Systems Biology. Chapman and Hall/CRC, 2018. 





[24] David Levin, Yuval Peres, and Elizabeth Wilmer. Markov Chains and Mixing Times. American Mathematical Society, 2009. 





[25] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. International Conference on Learning Representations, 2021. 





[26] Nikolay Savinov, Junyoung Chung, Mikolaj Binkowski, Erich Elsen, and Aaron van den Oord. Step-unrolled denoising autoencoders for text generation. International Conference on Learning Representations, 2022. 





[27] Anirudh Goyal, Nan Rosemary Ke, Surya Ganguli, and Yoshua Bengio. Variational walkback: Learning a transition operator as a stochastic recurrent net. Advances in Neural Information Processing Systems, 2017. 





[28] Min Jin Chong and David Forsyth. Effectively unbiased fid and inception score and where to find them. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020. 





[29] Colin Raffel. Learning-based methods for comparing sequences, with applications to audio-tomidi alignment and matching. PhD thesis, Columbia University, 2016. 





[30] Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang, and Yi-Hsuan Yang. Musegan: Multi-track sequential generative adversarial networks for symbolic music generation and accompaniment. AAAI Conference on Artificial Intelligence, 2018. 





[31] Gautam Mittal, Jesse Engel, Curtis Hawthorne, and Ian Simon. Symbolic music generation with diffusion models. International Society for Music Information Retrieval, 2021. 





[32] Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic-dpm: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. International Conference on Learning Representations, 2022. 





[33] David F Anderson. A modified next reaction method for simulating chemical systems with time dependent propensities and delays. The Journal of Chemical Physics, 127(21):214107, 2007. 





[34] Chie Furusawa, Shinya Kitaoka, Michael Li, and Yuri Odagiri. Generative probabilistic image colorization. arXiv preprint arXiv:2109.14518, 2021. 





[35] Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. AAAI Conference on Artificial Intelligence, 2018. 





[36] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 2017. 





[37] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. International Conference on Medical Image Computing and Computer-assisted Intervention, 2015. 





[38] Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P Kingma. Pixelcnn++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications. International Conference on Learning Representations, 2017. 





[39] Xi Chen, Nikhil Mishra, Mostafa Rohaninejad, and Pieter Abbeel. Pixelsnail: An improved autoregressive generative model. International Conference on Machine Learning, 2018. 





[40] Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural Networks, 107:3–11, 2018. 





[41] Maximilian Seitzer. pytorch-fid: FID Score for PyTorch. https://github.com/mseitzer/ pytorch-fid, August 2020. Version 0.2.1. 



# Appendix

# Contents

A Primer on Continuous Time Markov Chains 15 

B Proofs 16 

B.1 Proof of Proposition 1 . . 16 

B.2 Proof of Proposition 2 . . 17 

B.3 Proof of Proposition 3 . . 21 

B.4 Proof of Proposition 4 . . 22 

B.5 Proof of Theorem 1 22 

C Continuous Time ELBO Details 27 

C.1 Comparison with the Discrete Time ELBO . . . 27 

C.2 Conditional Form . . . 27 

C.3 Continuous Time ELBO with Factorization Assumptions . . . . 27 

C.4 One Forward Pass . . . 29 

D Direct Denoising Model Supervision 30 

E Choice of Forward Process 32 

F CTMC Simulation 34 

F.1 Exact CTMC and Tau-Leaping . . . 34 

F.2 Predictor-Corrector Discussion . . 34 

G Implicit Dimensional Assumptions Made in Discrete Time 36 

H Experimental Details 37 

H.1 Demonstrative Example . . . 37 

H.2 Image Modeling . . . . 38 

H.3 Monophonic Music . . . 39 

I Ethical Considerations 42 

The appendix is organized as follows. In Section A, we provide a short introduction to Continuous Time Markov Chains, including the relevant results we use in this work. Proofs for all the Propositions and Theorems from the main text are in Section B. We then describe in Section C some additional intuitions and forms of our proposed objective, $\mathcal { L } _ { \mathrm { C T } }$ . In Section D, we describe how an additional direct denoising model supervision term can be added to the objective to improve empirical performance. Details for how we define the forward process in our model can be found in Section E. Section F describes in more detail how CTMCs can be simulated and includes the algorithmic description of tau-leaping. We argue in Section G that operating in discrete time forces an implicit assumption when using a factorized forward process on multi-dimensional data. Full experimental details for all investigations can be found in Section H as well as additional plots and results from our models. Finally, in Section I, we consider the social impacts of our research. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/aba98d79547198e63c7a8d8cb40b8f472e6822353c92c7dda96c4d8b33f5e486.jpg)



Figure 6: Schematic representation of a 1-dimensional CTMC with 3 states.


# A Primer on Continuous Time Markov Chains

A Continuous Time Markov Chain (CTMC) is a right continuous stochastic process $\{ x _ { t } \} _ { t \in [ 0 , T ] }$ satisfying the Markov property, with $x _ { t }$ taking values in a discrete state space X . Since the CTMC is Markov, future behaviour of the process depends only on the current state and not the history. A schematic representation of a CTMC path is shown in Figure 6. The process repeatedly transitions from one state to another after having waited in the previous state for a randomly determined amount of time. 

A CTMC can be completely characterised by its jumps and holding times. Specifically, the time between each jump or holding time is exponentially distributed with mean $\nu ( x )$ where x is the state in which the process is holding. The next state that is jumped to is drawn from a jump probability distribution $r ( \tilde { x } | x )$ . The holding and jumping procedure is then repeated. 

There is an equivalent definition involving the transition rate matrix, $R \in \mathbb { R } ^ { S \times S }$ , that we use in the main paper. The transition rate matrix is defined as 

$$
R (\tilde {x}, x) = \lim _ {\Delta t \rightarrow 0} \frac {q _ {t | t - \Delta t} (x | \tilde {x}) - \delta_ {x , \tilde {x}}}{\Delta t} \tag {3}
$$

where $R ( \tilde { x } , x )$ is the $( { \tilde { x } } , x )$ element of the transition rate matrix and $q _ { t | t - \Delta t } ( x | \tilde { x } )$ is the infinitesimal transition probability of being in state x at time t given that the process was in state x˜ at time $t - \Delta t$ . Conversely, the CTMC can itself be defined through this infinitesimal transition probability 

$$
q _ {t \mid t - \Delta t} (x \mid \tilde {x}) = \delta_ {x, \tilde {x}} + R (\tilde {x}, x) \Delta t + o (\Delta t) \tag {4}
$$

where $o ( \Delta t )$ represents terms that tend to zero at a faster rate than $\Delta t .$ . From this definition of the transition rate matrix, we can infer the following properties: 

$$
R (\tilde {x}, x) \geq 0 \quad \text { for } \quad \tilde {x} \neq x, \quad R (x, x) \leq 0, \quad R (x, x) = - \sum_ {x ^ {\prime} \neq x} R (x, x ^ {\prime}) \tag {5}
$$

$R ( \tilde { x } , x )$ is the rate at which probability mass moves from state x˜ to x. $R ( x , x )$ is the total rate at which probability mass moves out of state x and is thus negative. 

In the time-homogeneous case, R has simple relations to the jump and holding time definitions. 

$$
\nu (x) = - \frac {1}{R (x , x)} \quad r (\tilde {x} | x) = (1 - \delta_ {\tilde {x}, x}) \frac {R (x , \tilde {x})}{- R (x , x)}
$$

In the time-inhomogeneous case, our transition rate matrix will now depend on time, $R _ { t } .$ , and these simple relations to the jump and holding time definition do not hold. However, $R _ { t }$ will still follow equations (3), (4) and (5). 

The CTMC transition probabilities satisfy the Kolmogorov forward and backward equations. For $t > s ,$ , 

Kolmogorov forward equation $\partial _ { t } q _ { t | s } ( x | \tilde { x } ) = \sum _ { y } q _ { t | s } ( y | \tilde { x } ) R _ { t } ( y , x )$ 

Kolmogorov backward equation $\partial _ { s } q _ { t | s } ( x | \tilde { x } ) = - \sum _ { y } R _ { s } ( \tilde { x } , y ) q _ { t | s } ( x | y )$ 

The Kolmogorov forward equation also gives us a differential equation for the marginals of the CTMC. 

$$
\partial_ {t} q _ {t} (x) = \sum_ {y} q _ {t} (y) R _ {t} (y, x).
$$

Exponential and Poisson Random Variables In the time homogeneous case, holding times are exponentially distributed with mean $\nu ( x ) = - 1 / R ( x , x )$ ). The tau-leaping algorithm relies on the fact that the number of events in interval [0, t] is Poisson distributed with mean $\mathbf { \widetilde { \Gamma } } _ { \nu } ^ { 1 } t$ when the inter-event times are exponentially distributed with mean ν. 

# B Proofs

# B.1 Proof of Proposition 1

Proof. We recall that a process $\{ x _ { t } \} _ { t \in [ 0 , T ] }$ taking values in X is called a CTMC if it is rightcontinuous and satisfies the Markov property. Denote $\{ y _ { t } \} _ { t \in [ 0 , T ] } = \{ x _ { T - t } \} _ { t \in [ 0 , T ] }$ except at the jump times of the forward process $\tau _ { n }$ with $n \in \mathbb { N }$ , where $y _ { T - \tau _ { n } } = x _ { \tau _ { n } } ^ { - } = \operatorname* { l i m } _ { t \leq \tau _ { n } , t  \tau _ { n } } x _ { t }$ . Hence, $\{ y _ { t } \} _ { t \in [ 0 , T ] }$ is almost surely equal to $\{ x _ { T - t } \} _ { t \in [ 0 , T ] }$ and is right-continuous. Since the Markov property is symmetric, we get that $\{ y _ { t } \} _ { t \in [ 0 , T ] }$ is a CTMC. We now compute its transition matrix. Let $x , \tilde { x } \in \mathcal { X }$ with $x \neq \tilde { x }$ , using the Kolmogorov forward equation, we have 

$$
\partial_ {t} p _ {t | s} (\tilde {x} | x) = \sum_ {y \in \mathcal {X}} p _ {t | s} (y | x) \hat {R} _ {T - t} (y, \tilde {x}),
$$

where $\{ p _ { t | s } , s , t \in [ 0 , T ] , t > s \}$ is the transition probability system associated with $\{ y _ { t } \} _ { t \in [ 0 , T ] }$ and $\{ \hat { R } _ { T - t } \} _ { t \in [ 0 , T ] }$ is the transition rate matrix associated with $\{ y _ { t } \} _ { t \in [ 0 , T ] }$ . Note that 

$$
\begin{array}{l} p _ {t | s} (x = j | \tilde {x} = i) = \mathbb {P} (y _ {t} = j \mid y _ {s} = i) \\ = \mathbb {P} (x _ {T - t} = j \mid x _ {T - s} = i) \\ = \mathbb {P} (x _ {T - s} = i | x _ {T - t} = j) \frac {\mathbb {P} (x _ {T - t} = j)}{\overline {{\mathbb {P}}} (x _ {T - s} = i)} \\ = q _ {T - s | T - t} (\tilde {x} = i | x = j) \frac {q _ {T - t} (x = j)}{q _ {T - s} (\tilde {x} = i)} \\ \end{array}
$$

where $\{ q _ { t | s } , s , t \in [ 0 , T ] , t > s \}$ is the transition probability system associated with $\{ x _ { t } \} _ { t \in [ 0 , T ] }$ and $\{ q _ { t } , t \in [ 0 , T ] \}$ are the marginals of $\{ x _ { t } \} _ { t \in [ 0 , T ] }$ . Now, writing the backward Kolmogorov equation for {xt}t∈[0,T ] $\{ x _ { t } \} _ { t \in [ 0 , T ] }$ 

$$
\partial_ {s} q _ {t | s} (\tilde {x} | x) = - \sum_ {y \in \mathcal {X}} R _ {s} (x, y) q _ {t | s} (\tilde {x} | y)
$$

Re-labeling the time indices we obtain, 

$$
\partial_ {T - t} q _ {T - s | T - t} (\tilde {x} | x) = - \sum_ {y \in \mathcal {X}} R _ {T - t} (x, y) q _ {T - s | T - t} (\tilde {x} | y)
$$

$$
\partial_ {t} q _ {T - s | T - t} (\tilde {x} | x) = \sum_ {y \in \mathcal {X}} R _ {T - t} (x, y) q _ {T - s | T - t} (\tilde {x} | y)
$$

Letting s → t and using that lim $. s  t \ q _ { T - s | T - t } ( x | \tilde { x } ) = 0 ,$ , we get that 

$$
\begin{array}{l} \hat {R} _ {T - t} (x, \tilde {x}) = \lim _ {s \to t} \partial_ {t} p _ {t | s} (\tilde {x} | x) \\ = \lim _ {s \rightarrow t} \partial_ {t} \left(q _ {T - s | T - t} (x | \tilde {x}) \frac {q _ {T - t} (\tilde {x})}{q _ {T - s} (x)}\right) \\ = \lim _ {s \rightarrow t} \left[ \partial_ {t} \left(q _ {T - s | T - t} (x | \tilde {x})\right) \frac {q _ {T - t} (\tilde {x})}{q _ {T - s} (x)} + q _ {T - s | T - t} (x | \tilde {x}) \frac {\partial_ {t} q _ {T - t} (\tilde {x})}{q _ {T - s} (x)} \right] \\ = \lim _ {s \rightarrow t} \partial_ {t} \left(q _ {T - s | T - t} (x | \tilde {x})\right) \frac {q _ {T - t} (\tilde {x})}{q _ {T - s} (x)} \\ = R _ {T - t} (\tilde {x}, x) \frac {q _ {T - t} (\tilde {x})}{q _ {T - t} (x)} \\ \end{array}
$$

Re-labeling the time-indices on the rate matrices, we obtain 

$$
\hat {R} _ {t} (x, \tilde {x}) = R _ {t} (\tilde {x}, x) \frac {q _ {t} (\tilde {x})}{q _ {t} (x)}
$$

Now we write the marginal ratio $\frac { q _ { t } ( \tilde { x } ) } { q _ { t } ( x ) }$ in a different form 

$$
\begin{array}{l} \frac {q _ {t} (\tilde {x})}{q _ {t} (x)} = \sum_ {x _ {0}} \frac {p _ {\text {data}} (x _ {0})}{q _ {t} (x)} q _ {t | 0} (\tilde {x} | x _ {0}) \\ = \sum_ {x _ {0}} \frac {q _ {0 | t} (x _ {0} | x)}{q _ {t | 0} (x | x _ {0})} q _ {t | 0} (\tilde {x} | x _ {0}). \\ \end{array}
$$

Substituting in this form for the marginal ratio concludes the proof. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/6870d0f9d383f94da6badd205be894101cde182f4acb86c1bde1deb3f0462190.jpg)


# B.2 Proof of Proposition 2

In this section, we detail two proofs for Proposition 2. The first is a formal proof using results from stochastic processes. We then provide a second informal proof for the same result to gain intuition into the $\mathcal { L } _ { \mathrm { C T } }$ objective that only relies on elementary results from CTMCs. 

# Proof 1 - Stochastic Processes

Proof. Let us write Q for the path measure of the forward CTMC with rate matrix $R _ { t } , \hat { \mathbb { Q } }$ for the path measure of its exact time reversal and $\mathbb { P } ^ { \theta }$ for the path measure of the approximate reverse process with rate matrix $\hat { R } _ { t } ^ { \theta }$ . Also, we use superscripts to notate conditioning on the starting point, for example $\mathbb { Q } ^ { x _ { 0 } }$ denotes the path measure of the forward process conditioned to start in $x _ { 0 }$ . 

With this notation, we have 

$$
\begin{array}{l} - \log p _ {0} ^ {\theta} (x _ {0}) = - \log \int p _ {\mathrm{ref}} (\mathrm{d} x _ {T}) \int_ {\{\hat {W} _ {T} = x _ {0} \}} \mathbb {P} ^ {\theta , x _ {T}} (\mathrm{d} w) \\ = - \log \int q _ {T | 0} (\mathrm{d} x _ {T}) \int_ {\{\hat {W} _ {T} = x _ {0} \}} \hat {\mathbb {Q}} ^ {x _ {T}} (\mathrm{d} w) \frac {\mathrm{d} p _ {\text {ref}}}{\mathrm{d} q _ {T | 0}} (x _ {T}) \frac {\mathrm{d} \mathbb {P} ^ {\theta , x _ {T}}}{\mathrm{d} \hat {\mathbb {Q}} ^ {x _ {T}}} (w) \\ = - \log \int q _ {T | 0} (\mathrm{d} x _ {T}) \int \hat {\mathbb {Q}} (\mathrm{d} w | \hat {W} _ {0} = x _ {T}, \hat {W} _ {T} = x _ {0}) \frac {\mathrm{d} p _ {\text {ref}}}{\mathrm{d} q _ {T | 0}} (x _ {T}) \frac {\mathrm{d} \mathbb {P} ^ {\theta , x _ {T}}}{\mathrm{d} \hat {\mathbb {Q}} ^ {x _ {T}}} (w) \mathbb {Q} ^ {x _ {T}} \{\hat {W} _ {0} = x _ {0} \} \\ \leq \int q _ {T | 0} (\mathrm{d} x _ {T}) \int \hat {\mathbb {Q}} (\mathrm{d} w | \hat {W} _ {0} = x _ {T}, \hat {W} _ {T} = x _ {0}) \left\{- \log \frac {\mathrm{d} \mathbb {P} ^ {\theta , x _ {T}}}{\mathrm{d} \hat {\mathbb {Q}} ^ {x _ {T}}} (w) \right\} + C, \\ \end{array}
$$

where $\mathbb { P } ^ { \theta } , \hat { \mathbb { Q } }$ run in the reverse time direction. Writing $\hat { W } _ { s }$ for a reverse path and integrating wrt $p _ { \mathrm { d a t a } } ( \mathrm { d } x _ { 0 } )$ we have 

$$
\begin{array}{l} \int p _ {\text { data }} (x _ {0}) [ - \log p _ {0} ^ {\theta} (x _ {0}) ] \leq \int p _ {\text { data }} (x _ {0}) \int q _ {T | 0} (\mathrm{d} x _ {T}) \int \hat {\mathbb {Q}} (\mathrm{d} \hat {W} | \hat {W} _ {0} = x _ {T}, \hat {W} _ {T} = x _ {0}) \\ \times \left\{\int_ {s = 0} ^ {T} \hat {R} _ {T - s} ^ {\theta} (\hat {W} _ {s}) \mathrm{d} s - \sum_ {s: \hat {W} _ {s -} \neq \hat {W} _ {s}} \log \mathbb {P} _ {T - s} ^ {\theta} (\hat {W} _ {s} | \hat {W} _ {s -}) R _ {T - s} ^ {\theta} (\hat {W} _ {s -}) \right\} + C, \\ \end{array}
$$

where $\hat { R } _ { t } ^ { \theta } ( x )$ is shorthand for $- \hat { R } _ { t } ^ { \theta } ( x , x )$ 

When $x _ { 0 } ~ \sim ~ p _ { \mathrm { d a t a } } , x _ { T } ~ \sim ~ q _ { T | 0 } ( \cdot | x _ { 0 } ) , \hat { W } ~ \sim ~ \hat { \mathbb { Q } } ( \mathrm { d } W | \hat { W } _ { 0 } ~ = ~ x _ { T } , \hat { W } _ { T } ~ = ~ x _ { 0 } )$ , the reverse path is distributed according to $p _ { \mathrm { d a t a } } ( \mathrm { d } x _ { 0 } ) \mathbb { Q } _ { x _ { 0 } } ( \mathrm { d } W )$ and therefore $( \hat { W } _ { s - } , \hat { W } _ { s } )$ is distributed like $( W _ { T - s } , W _ { ( T - s ) - } )$ and thus we have 

$$
\begin{array}{l} \int p _ {\mathrm{data}} (x _ {0}) [ - \log p _ {0} ^ {\theta} (x _ {0}) ] \\ \leq \int p _ {\text { data }} (x _ {0}) \mathbb {Q} _ {x _ {0}} (\mathrm{d} W) \left\{\int_ {s = 0} ^ {T} \hat {R} _ {T - s} ^ {\theta} (W _ {(T - s) -}) \mathrm{d} s - \sum_ {s: W _ {(T - s) -} \neq W _ {T - s}} \log \mathbb {P} _ {T - s} ^ {\theta} (W _ {(T - s) -} | W _ {T - s}) \hat {R} _ {T - s} ^ {\theta} (W _ {T - s}) \right\} + C \\ \end{array}
$$

Using Dynkin’s lemma and the fact that $\mathbb { P } _ { t } ^ { \theta } ( x | y ) \hat { R } _ { t } ^ { \theta } ( y ) = \hat { R } _ { t } ^ { \theta } ( y , x )$ we can re-expresss this final line as 

$$
= \iint_ {s = 0} ^ {T} q _ {T - s} (\mathrm{d} x) \left\{\sum_ {z \neq x} \hat {R} _ {T - s} ^ {\theta} (x, z) - \sum_ {z \neq x} R _ {T - s} (x, z) \frac {\sum_ {y \neq x} R _ {T - s} (x , y)}{\sum_ {z \neq x} R _ {T - s} (x , z)} \log \hat {R} _ {T - s} ^ {\theta} (y, x) \right\}
$$

$$
= \iint_ {s = 0} ^ {T} q _ {T - s} (\mathrm{d} x) r _ {T - s} (\mathrm{d} y | x) \left\{\sum_ {z \neq x} \hat {R} _ {T - s} ^ {\theta} (x, z) - \sum_ {z \neq x} R _ {T - s} (x, z) \log \hat {R} _ {T - s} ^ {\theta} (y, x) \right\}
$$

$$
= \iint_ {s = 0} ^ {T} q _ {s} (\mathrm{d} x) r _ {s} (\mathrm{d} y | x) \left\{\sum_ {z \neq x} \hat {R} _ {s} ^ {\theta} (x, z) - \sum_ {z \neq x} R _ {s} (x, z) \log \hat {R} _ {s} ^ {\theta} (y, x) \right\}
$$

which rearranges to give the continuous time ELBO in the form of Proposition 2. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/853c2471d270a576a2d533400e7bc33a64f5dee7bc5604d78e167f29d95abef3.jpg)


# Proof 2 - Limit of Discrete Time ELBO

Proof. Consider a partitioning of $[ 0 , T ] , 0 = t _ { 0 } < t _ { 1 } < \dots < t _ { k - 1 } < t _ { k } < t _ { k + 1 } < \dots < t _ { K - 1 } <$ $t _ { K } = T$ . Let $t _ { k } - t _ { k - 1 } = \Delta t$ for all k. In subscripts we use k as a shorthand for $t _ { k }$ when this does not cause confusion. Considering a CTMC with this time partitioning converts the problem into a discrete time Markov Chain with forward transition kernel, $q _ { k + 1 | k } { \left( x _ { k + 1 } | x _ { k } \right) }$ and parameterized reverse kernel, $p _ { k | k + 1 } ^ { \theta } ( x _ { k } | x _ { k + 1 } )$ . Therefore, we can write the negative ELBO in its discrete time form, $\mathcal { L } _ { \mathrm { D T } }$ 

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{DT}} (\theta) = \mathbb {E} _ {p _ {\text {data}} (x _ {0})} \left[ \mathrm{KL} \left(q _ {K | 0} \left(x _ {K} \mid x _ {0}\right) \mid \mid p _ {\text {ref}} \left(x _ {K}\right)\right) - \mathbb {E} _ {q _ {1 | 0} \left(x _ {1} \mid x _ {0}\right)} \left[ \log p _ {0 | 1} ^ {\theta} \left(x _ {0} \mid x _ {1}\right) \right] \right. \\ \left. + \sum_ {k = 1} ^ {K - 1} \mathbb {E} _ {q _ {k + 1 | 0} (x _ {k + 1} | x _ {0})} \left[ \mathrm{KL} \left(q _ {k | k + 1, 0} \left(x _ {k} \mid x _ {k + 1}, x _ {0}\right) \mid \mid p _ {k | k + 1} ^ {\theta} \left(x _ {k} \mid x _ {k + 1}\right)\right) \right] \right] \\ \end{array}
$$

In the following, we will write the transition kernels in terms of the CTMC rate matrices and take the limit as $\Delta t \to 0$ to obtain a continuous time negative ELBO. 

First, consider one item from the inner sum of ${ \mathcal { L } } _ { \mathrm { D T } }$ 

$$
L _ {k} = \mathbb {E} _ {p _ {\text { data }} (x _ {0}) q _ {k + 1 | 0} (x _ {k + 1} | x _ {0})} \left[ \mathrm{KL} (q _ {k | k + 1, 0} (x _ {k} | x _ {k + 1}, x _ {0}) | | p _ {k | k + 1} ^ {\theta} (x _ {k} | x _ {k + 1})) \right]
$$

$$
= - \mathbb {E} _ {p _ {\text {data}} (x _ {0}) q _ {k + 1 | 0} (x _ {k + 1} | x _ {0}) q _ {k | k + 1, 0} (x _ {k} | x _ {k + 1}, x _ {0})} \left[ \log p _ {k | k + 1} ^ {\theta} (x _ {k} | x _ {k + 1}) \right] + C
$$

$$
= - \mathbb {E} _ {q _ {k} (x _ {k}) q _ {k + 1 | k} (x _ {k + 1} | x _ {k})} \left[ \log p _ {k | k + 1} ^ {\theta} (x _ {k} | x _ {k + 1}) \right] + C
$$

where we have absorbed terms that do not depend on θ into C. We now write $p _ { k | k + 1 } ^ { \theta } ( x _ { k } | x _ { k + 1 } )$ in terms of $\hat { R } _ { k } ^ { \theta }$ . 

$$
p _ {k | k + 1} ^ {\theta} (x _ {k} | x _ {k + 1}) = \delta_ {x _ {k}, x _ {k + 1}} + \hat {R} _ {k} ^ {\theta} (x _ {k + 1}, x _ {k}) \Delta t + o (\Delta t)
$$

$$
\begin{array}{l} \log p _ {k | k + 1} ^ {\theta} (x _ {k} | x _ {k + 1}) = \log \left(\delta_ {x _ {k}, x _ {k + 1}} + \hat {R} _ {k} ^ {\theta} (x _ {k + 1}, x _ {k}) \Delta t + o (\Delta t)\right) \\ = \delta_ {x _ {k}, x _ {k + 1}} \log \left(1 + \hat {R} _ {k} ^ {\theta} (x _ {k}, x _ {k}) \Delta t + o (\Delta t)\right) \\ + \left(1 - \delta_ {x _ {k}, x _ {k + 1}}\right) \log \left(\hat {R} _ {k} ^ {\theta} \left(x _ {k + 1}, x _ {k}\right) \Delta t + o (\Delta t)\right) \\ = \delta_ {x _ {k}, x _ {k + 1}} \left(\hat {R} _ {k} ^ {\theta} (x _ {k}, x _ {k}) \Delta t + o (\Delta t)\right) \\ + \left(1 - \delta_ {x _ {k}, x _ {k + 1}}\right) \log \left(\hat {R} _ {k} ^ {\theta} (x _ {k + 1}, x _ {k}) \Delta t + o (\Delta t)\right) \tag {6} \\ \end{array}
$$

where on the last line we have used the series expansion for $\begin{array} { r } { \log ( 1 + z ) = z - \frac { z ^ { 2 } } { 2 } + o ( z ^ { 2 } ) } \end{array}$ valid for $\vert z \vert \leq 1 , z \neq - 1$ . For any finite $R _ { k } ^ { \theta } ( x _ { k } , x _ { k } )$ , ∆t can be taken small enough such that the series expansion holds. We now substitute this form for log $p _ { k | k + \cdot } ^ { \theta }$ 1 into $L _ { k }$ and further write the expectation over $q _ { k + 1 | k } ( x _ { k + 1 } | x _ { k } ) = \delta _ { x _ { k } , x _ { k + 1 } } + R _ { k } ( x _ { k } , x _ { k + 1 } ) \Delta t + o ( \Delta t )$ as an explicit sum. 

$$
L _ {k} = - \mathbb {E} _ {q _ {k} (x _ {k})} \left[ \sum_ {x _ {k + 1}} \left\{\left[ \delta_ {x _ {k}, x _ {k + 1}} + R _ {k} (x _ {k}, x _ {k + 1}) \Delta t + o (\Delta t) \right] \times \right. \right.
$$

$$
\left[ \delta_ {x _ {k}, x _ {k + 1}} \left(\hat {R} _ {k} ^ {\theta} (x _ {k}, x _ {k}) \Delta t + o (\Delta t)\right) \right.
$$

$$
\left. \left. + \left(1 - \delta_ {x _ {k}, x _ {k + 1}}\right) \log \left(\hat {R} _ {k} ^ {\theta} (x _ {k + 1}, x _ {k}) \Delta t + o (\Delta t)\right) \right] \right\} + C
$$

$$
L _ {k} = - \mathbb {E} _ {q _ {k} (x _ {k})} \left[ \sum_ {x _ {k + 1}} \left\{\delta_ {x _ {k}, x _ {k + 1}} \hat {R} _ {k} ^ {\theta} (x _ {k}, x _ {k}) \Delta t \right. \right.
$$

$$
+ \left(1 - \delta_ {x _ {k}, x _ {k + 1}}\right) R _ {k} \left(x _ {k}, x _ {k + 1}\right) \Delta t \times
$$

$$
\left. \log \left(\hat {R} _ {k} ^ {\theta} (x _ {k + 1}, x _ {k}) \Delta t + o (\Delta t)\right) + o (\Delta t) \right\} \Bigg ] + C
$$

We can isolate $\hat { R } _ { k } ^ { \theta }$ within the log through the following re-arrangement 

$$
\Delta t \log \left(\hat {R} _ {k} ^ {\theta} (x _ {k + 1}, x _ {k}) \Delta t + o (\Delta t)\right)
$$

$$
= \Delta t \log \Delta t + \Delta t \log \left(\hat {R} _ {k} ^ {\theta} (x _ {k + 1}, x _ {k}) + o (1)\right)
$$

$$
= \Delta t \log \Delta t + \Delta t \log (1 + o (1)) + \Delta t \log (\hat {R} _ {k} ^ {\theta} (x _ {k + 1}, x _ {k}))
$$

where the first two terms are independent of θ and tend to 0 as $\Delta t  0$ . Note that we assume $\hat { R } _ { k } ^ { \theta } ( x _ { k + 1 } , x _ { k } ) > 0$ for $x _ { k + 1 } \neq x _ { k }$ pairs which have $R _ { k } ( x _ { k } , x _ { k + 1 } ) > 0$ . This assumption is valid because, for $x _ { k + 1 } \neq x _ { k }$ , we have 

$$
\hat {R} _ {k} ^ {\theta} (x _ {k + 1}, x _ {k}) = R _ {k} (x _ {k}, x _ {k + 1}) \sum_ {x _ {0}} \frac {q _ {k | 0} (x _ {k} | x _ {0})}{q _ {k | 0} (x _ {k + 1} | x _ {0})} p _ {0 | k} ^ {\theta} (x _ {0} | x _ {k + 1})
$$

and we assume $p _ { 0 | k } ^ { \theta } ( x _ { 0 } | x _ { k + 1 } ) > 0$ which is valid when we parameterize $p _ { 0 | k } ^ { \theta }$ with a softmax output. We assume an irreducible Markov chain, hence $q _ { k | 0 } > 0$ for $t _ { k } > 0$ . 

With this re-arrangement, and absorbing constant terms into C, we obtain 

$$
\begin{array}{l} L _ {k} = - \mathbb {E} _ {q _ {k} (x _ {k})} \left[ \sum_ {x _ {k + 1}} \left\{\delta_ {x _ {k}, x _ {k + 1}} \hat {R} _ {k} ^ {\theta} (x _ {k}, x _ {k}) \Delta t \right. \right. \\ + \left(1 - \delta_ {x _ {k}, x _ {k + 1}}\right) R _ {k} (x _ {k}, x _ {k + 1}) \Delta t \log \left(\hat {R} _ {k} ^ {\theta} (x _ {k + 1}, x _ {k})\right) \\ \left. \left. + o (\Delta t) \right\} \right] + C \\ \end{array}
$$

$$
L _ {k} = - \mathbb {E} _ {q _ {k} (x _ {k})} \left[ \hat {R} _ {k} ^ {\theta} (x _ {k}, x _ {k}) \Delta t + \sum_ {x _ {k + 1} \neq x _ {k}} R _ {k} (x _ {k}, x _ {k + 1}) \Delta t \log \hat {R} _ {k} ^ {\theta} (x _ {k + 1}, x _ {k}) + o (\Delta t) \right]
$$

The second term can be re-written so that it is more efficient to approximate with Monte Carlo. Currently the denoising model $p _ { 0 | k } ^ { \theta }$ has to be evaluated for each term in the sum $\textstyle \sum _ { x _ { k + 1 } \neq x _ { k } }$ which would require multiple forward passes of the neural network. We can instead create a new probability distribution to sample from as follows. Define 

$$
r _ {k} (x _ {k + 1} | x _ {k}) = (1 - \delta_ {x _ {k}, x _ {k + 1}}) \frac {R _ {k} (x _ {k} , x _ {k + 1})}{\mathcal {Z} ^ {k} (x _ {k})}
$$

where 

$$
\mathcal {Z} ^ {k} (x _ {k}) = \sum_ {x _ {k + 1} ^ {\prime} \neq x _ {k}} R _ {k} (x _ {k}, x _ {k + 1} ^ {\prime})
$$

So we now have 

$$
L _ {k} = - \mathbb {E} _ {q _ {k} (x _ {k}) r _ {k} (x _ {k + 1} | x _ {k})} \left[ \hat {R} _ {k} ^ {\theta} (x _ {k}, x _ {k}) \Delta t + \mathcal {Z} ^ {k} (x _ {k}) \Delta t \log \hat {R} _ {k} ^ {\theta} (x _ {k + 1}, x _ {k}) + o (\Delta t) \right]
$$

Examining the other terms in $\mathcal { L } _ { \mathrm { D T } }$ we have $\mathbb { E } _ { p _ { \mathrm { d a t a } } ( x _ { 0 } ) } \left[ { \mathrm { K L } } ( q _ { K | 0 } ( x _ { K } | x _ { 0 } ) | | p _ { \mathrm { r e f } } ( x _ { K } ) ) \right]$ which does not depend on θ and $\mathbb { E } _ { q _ { 1 | 0 } ( x _ { 1 } | x _ { 0 } ) } \left[ \log p _ { 0 | 1 } ^ { \theta } ( x _ { 0 } | x _ { 1 } ) \right]$ which we expand here 

$$
\begin{array}{l} \mathbb {E} _ {q _ {1 | 0} (x _ {1} | x _ {0})} \left[ \log p _ {0 | 1} ^ {\theta} (x _ {0} | x _ {1}) \right] \\ = \sum_ {x _ {1}} \left\{\delta_ {x _ {1}, x _ {0}} + \Delta t R _ {1} (x _ {0}, x _ {1}) + o (\Delta t) \right\} \log p _ {0 | 1} ^ {\theta} (x _ {0} | x _ {1}) \\ = \log p _ {0 | 1} ^ {\theta} (x _ {0} | x _ {0}) + \Delta t \sum_ {x _ {1}} R _ {1} (x _ {0}, x _ {1}) \log p _ {0 | 1} ^ {\theta} (x _ {0} | x _ {1}) + o (\Delta t) \\ = \Delta t \hat {R} _ {1} ^ {\theta} (x _ {0}, x _ {0}) + \Delta t \sum_ {x _ {1}} R _ {1} (x _ {0}, x _ {1}) \log p _ {0 | 1} ^ {\theta} (x _ {0} | x _ {1}) + o (\Delta t) \\ \end{array}
$$

where on the final line we have used eq 6. In summary, 

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{DT}} = \Delta t \mathbb {E} _ {p _ {\mathrm{data}} (x _ {0}) q _ {1 | 0} (x _ {1} | x _ {0})} \left[ - \hat {R} _ {1} ^ {\theta} (x _ {0}, x _ {0}) + \sum_ {x _ {1}} R _ {1} (x _ {0}, x _ {1}) \log p _ {0 | 1} ^ {\theta} (x _ {0} | x _ {1}) \right] \\ - \Delta t \sum_ {k = 1} ^ {K - 1} \mathbb {E} _ {q _ {k} (x _ {k}) r _ {k} (x _ {k + 1} | x _ {k})} \left[ \hat {R} _ {k} ^ {\theta} (x _ {k}, x _ {k}) + \mathcal {Z} ^ {k} (x _ {k}) \log \hat {R} _ {k} ^ {\theta} (x _ {k + 1}, x _ {k}) \right] \\ + o (\Delta t) + C \\ \end{array}
$$

We now take the limit of $\mathcal { L } _ { \mathrm { D T } }$ as $\Delta t  0$ and $K  \infty$ . 

$$
\lim _ {\Delta t \rightarrow 0} \mathcal {L} _ {\mathrm{DT}} = \mathcal {L} _ {\mathrm{CT}} = - \int_ {0} ^ {T} \mathbb {E} _ {q _ {t} (x) r _ {t} (\tilde {x} | x)} \left[ \hat {R} _ {t} ^ {\theta} (x, x) + \mathcal {Z} ^ {t} (x) \log \left(\hat {R} _ {t} ^ {\theta} (\tilde {x}, x)\right)\right] d t + C
$$

We can estimate the integral with Monte Carlo if we consider it to be an expectation with respect to a uniform distribution over times (0, T ). We also write $\hat { R } _ { t } ^ { \theta } ( x , x )$ explicitly as the negative off diagonal row sum to obtain 

$$
\mathcal {L} _ {\mathrm{CT}} (\theta) = T \mathbb {E} _ {t \sim \mathcal {U} (0, T) q _ {t} (x) r _ {t} (\tilde {x} | x)} \left[ \left\{\sum_ {x ^ {\prime} \neq x} \hat {R} _ {t} ^ {\theta} (x, x ^ {\prime}) \right\} - \mathcal {Z} ^ {t} (x) \log \left(\hat {R} _ {t} ^ {\theta} (\tilde {x}, x)\right) \right] + C.
$$

□ 

# B.3 Proof of Proposition 3

Proof. We assume $q _ { t | s } ( \pmb { x } _ { t } ^ { 1 : D } | \pmb { x } _ { s } ^ { 1 : D } )$ factorizes as $\textstyle \prod _ { d = 1 } ^ { D } q _ { t \mid s } ( x _ { t } ^ { d } | x _ { s } ^ { d } )$ where $q _ { t \mid s } ( x _ { t } ^ { d } | x _ { s } ^ { d } ) , \ d =$ $1 , \ldots , D$ are the transition probabilities for independent singular dimensional CTMCs each with forward rate $R _ { t } ^ { d } ( \tilde { x } ^ { d } , x ^ { d } )$ . In the following, we will drop time subscripts on x arguments. To find the correspondence between $R _ { t } ^ { 1 : D }$ and $R _ { t } ^ { d }$ , we use the Kolmogorov forward equation 

$$
\partial_ {t} q _ {t | s} (\pmb {x} ^ {1: D} | \tilde {\pmb {x}} ^ {1: D}) = \sum_ {\pmb {y} ^ {1: D}} q _ {t | s} (\pmb {y} ^ {1: D} | \tilde {\pmb {x}} ^ {1: D}) R _ {t} ^ {1: D} (\pmb {y} ^ {1: D}, \pmb {x} ^ {1: D})
$$

Substitute in our factorized form for $q _ { t \mid s }$ into the LHS 

$$
\begin{array}{l} \partial_ {t} q _ {t | s} \left(\boldsymbol {x} ^ {1: D} \mid \tilde {\boldsymbol {x}} ^ {1: D}\right) = \partial_ {t} \left\{\prod_ {d = 1} ^ {D} q _ {t | s} \left(x ^ {d} \mid \tilde {x} ^ {d}\right) \right\} \\ = \sum_ {d = 1} ^ {D} q _ {t | s} (\boldsymbol {x} ^ {1: D \setminus d} | \tilde {\boldsymbol {x}} ^ {1: D \setminus d}) \partial_ {t} q _ {t | s} (x ^ {d} | \tilde {x} ^ {d}) \\ = \sum_ {d = 1} ^ {D} q _ {t | s} (\boldsymbol {x} ^ {1: D \setminus d} | \tilde {\boldsymbol {x}} ^ {1: D \setminus d}) \sum_ {y ^ {d}} q _ {t | s} (y ^ {d} | \tilde {x} ^ {d}) R _ {t} ^ {d} (y ^ {d}, x ^ {d}) \\ = \sum_ {d = 1} ^ {D} \sum_ {\boldsymbol {y} ^ {1: D}} q _ {t | s} (\boldsymbol {x} ^ {1: D \setminus d} | \tilde {\boldsymbol {x}} ^ {1: D \setminus d}) q _ {t | s} (y ^ {d} | \tilde {x} ^ {d}) R _ {t} ^ {d} (y ^ {d}, x ^ {d}) \delta_ {\boldsymbol {x} ^ {1: D \setminus d}, \boldsymbol {y} ^ {1: D \setminus d}} \\ = \sum_ {d = 1} ^ {D} \sum_ {\boldsymbol {y} ^ {1: D}} q _ {t | s} (\boldsymbol {y} ^ {1: D \setminus d} | \tilde {\boldsymbol {x}} ^ {1: D \setminus d}) q _ {t | s} (y ^ {d} | \tilde {x} ^ {d}) R _ {t} ^ {d} (y ^ {d}, x ^ {d}) \delta_ {\boldsymbol {x} ^ {1: D \setminus d}, \boldsymbol {y} ^ {1: D \setminus d}} \\ = \sum_ {\boldsymbol {y} ^ {1: D}} q _ {t | s} (\boldsymbol {y} ^ {1: D} | \tilde {\boldsymbol {x}} ^ {1: D}) \sum_ {d = 1} ^ {D} R _ {t} ^ {d} (y ^ {d}, x ^ {d}) \delta_ {\boldsymbol {x} ^ {1: D \setminus d}, \boldsymbol {y} ^ {1: D \setminus d}} \\ \end{array}
$$

We therefore obtain 

$$
\sum_ {\pmb {y} ^ {1: D}} q _ {t | s} (\pmb {y} ^ {1: D} | \tilde {\pmb {x}} ^ {1: D}) R _ {t} ^ {1: D} (\pmb {y} ^ {1: D}, \pmb {x} ^ {1: D}) = \sum_ {\pmb {y} ^ {1: D}} q _ {t | s} (\pmb {y} ^ {1: D} | \tilde {\pmb {x}} ^ {1: D}) \sum_ {d = 1} ^ {D} R _ {t} ^ {d} (y ^ {d}, x ^ {d}) \delta_ {\pmb {x} ^ {1: D \setminus d}, \pmb {y} ^ {1: D \setminus d}}
$$

This must be true for all possible factorizable forward process transitions, $q _ { t \mid s }$ , including $q _ { t | s } ( { \pmb y } ^ { 1 : D } | \tilde { \pmb x } ^ { 1 : D } ) = \delta _ { { \pmb y } ^ { 1 : D } , \tilde { \pmb x } ^ { 1 : D } }$ . This choice gives us our forward rate relation 

$$
R _ {t} ^ {1: D} \left(\tilde {\boldsymbol {x}} ^ {1: D}, \boldsymbol {x} ^ {1: D}\right) = \sum_ {d = 1} ^ {D} R _ {t} ^ {d} \left(\tilde {x} ^ {d}, x ^ {d}\right) \delta_ {\boldsymbol {x} ^ {1: D \setminus d}, \tilde {\boldsymbol {x}} ^ {1: D \setminus d}}
$$

Substituting this into our expression for the reverse rate from Proposition 1 we obtain 

$$
\begin{array}{l} \hat {R} _ {t} ^ {1: D} (\pmb {x} ^ {1: D}, \tilde {\pmb {x}} ^ {1: D}) = \sum_ {\pmb {x} _ {0} ^ {1: D}} \sum_ {d = 1} ^ {D} R _ {t} ^ {d} (\tilde {x} ^ {d}, x ^ {d}) \frac {q _ {t} (\tilde {\pmb {x}} ^ {1 : D} | \pmb {x} _ {0} ^ {1 : D})}{q _ {t} (\pmb {x} ^ {1 : D} | \pmb {x} _ {0} ^ {1 : D})} q _ {0 | t} (\pmb {x} _ {0} ^ {1: D} | \pmb {x} ^ {1: D}) \delta_ {\pmb {x} ^ {1: D \setminus d}, \tilde {\pmb {x}} ^ {1: D \setminus d}} \\ = \sum_ {\boldsymbol {x} _ {0} ^ {1: D}} \sum_ {d = 1} ^ {D} R _ {t} ^ {d} (\tilde {x} ^ {d}, x ^ {d}) \frac {q _ {t | 0} (\tilde {x} ^ {d} | x _ {0} ^ {d})}{q _ {t | 0} (x ^ {d} | x _ {0} ^ {d})} q _ {0 | t} (\boldsymbol {x} _ {0} ^ {1: D} | \boldsymbol {x} ^ {1: D}) \delta_ {\boldsymbol {x} ^ {1: D \setminus d}, \tilde {\boldsymbol {x}} ^ {1: D \setminus d}} \\ = \sum_ {d = 1} ^ {D} R _ {t} ^ {d} (\tilde {x} ^ {d}, x ^ {d}) \delta_ {\boldsymbol {x} ^ {1: D \setminus d}, \tilde {\boldsymbol {x}} ^ {1: D \setminus d}} \sum_ {x _ {0} ^ {d}} q _ {0 | t} (x _ {0} ^ {d} | \boldsymbol {x} ^ {1: D}) \frac {q _ {t | 0} (\tilde {x} ^ {d} | x _ {0} ^ {d})}{q _ {t | 0} (x ^ {d} | x _ {0} ^ {d})} \sum_ {\boldsymbol {x} _ {0} ^ {1: D \setminus d}} q _ {0 | t} (\boldsymbol {x} _ {0} ^ {1: D \setminus d} | x _ {0} ^ {d}, \boldsymbol {x} ^ {1: D}) \\ = \sum_ {d = 1} ^ {D} R _ {t} ^ {d} (\tilde {x} ^ {d}, x ^ {d}) \delta_ {\boldsymbol {x} ^ {1: D \setminus d}, \tilde {\boldsymbol {x}} ^ {1: D \setminus d}} \sum_ {x _ {0} ^ {d}} q _ {0 | t} (x _ {0} ^ {d} | \boldsymbol {x} ^ {1: D}) \frac {q _ {t | 0} (\tilde {x} ^ {d} | x _ {0} ^ {d})}{q _ {t | 0} (x ^ {d} | x _ {0} ^ {d})} \\ \end{array}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/be754c85124c2c422b2f6159d23a5649c9394faeafc8bf89e03db83c454223bb.jpg)


# B.4 Proof of Proposition 4

Proof. By the Kolmogorov forward equation applied to the forwards process, we have 

$$
\partial_ {t} q _ {t} (x _ {t}) = \sum_ {y} R _ {t} (y, x _ {t}) q _ {t} (y)
$$

In addition, applying the Kolmogorov forward equation to the reverse process, which has the same marginals as the forward but time-reversed, we get 

$$
- \partial_ {t} q _ {t} (x _ {t}) = \sum_ {y} \hat {R} _ {t} (y, x _ {t}) q _ {t} (y)
$$

Summing these two equations gives 

$$
\sum_ {y} \left\{R _ {t} (y, x _ {t}) + \hat {R} _ {t} (y, x _ {t}) \right\} q _ {t} (y) = 0
$$

Therefore, by comparison with the Kolmogorov equation, $R _ { t } + \hat { R } _ { t }$ is the rate matrix of a CTMC with invariant distribution $q _ { t }$ . □ 

# B.5 Proof of Theorem 1

In this section, we derive a bound on the error of our tau-leaping diffusion model. Because the tauleaping approximation is only interesting in the case where multiple jumps are made along different dimensions in a single step, we choose to make the dependence of our bound on the dimension of our model explicit, rather than simply considering the case of fixed D and $\tau  0$ . 

Recall from the main text that we have a time-homogeneous rate matrix $R _ { t }$ on $\mathcal { X } .$ , from which we construct the factorised rate matrix $R _ { t } ^ { 1 : D }$ on $\mathcal { X } ^ { D }$ by setting $R _ { t } ^ { d } = R _ { t }$ for each d, and will denote $\begin{array} { r } { | R | = \operatorname* { s u p } _ { t \in [ 0 , T ] , x \in \mathcal { X } } | R _ { t } ( x , x ) } \end{array}$ |, and let $t _ { \mathrm { m i x } }$ be the (1/4)-mixing time of the CTMC with rate $R _ { t }$ . We also define addition on the state space $\mathcal { X } ^ { D }$ using a mapping from $\mathcal { X }$ to $\mathbb { Z }$ as in Section 4.3 and component-wise addition. 

Theorem 1. For any $D \geq 1$ and distribution $p _ { \mathrm { d a t a } }$ on $\mathcal { X } ^ { D }$ , let $\{ x _ { t } \} _ { t \in [ 0 , T ] }$ be a CTMC starting in pdata with rate matrix $R _ { t } ^ { 1 : D }$ as above. Suppose that $\hat { R } _ { t } ^ { \theta 1 : D }$ is an approximation to the reverse rate matrix and let $( y _ { k } ) _ { k = 0 , 1 , \dots , N }$ be a tau-leaping approximation to the reverse dynamics with maximum step size τ . Suppose further that there is some constant $M > 0$ independent of D such that 

$$
\sum_ {y \neq x} \left| \hat {R} _ {t} ^ {1: D} (x, y) - \hat {R} _ {t} ^ {\theta 1: D} (x, y) \right| \leq M
$$

for all $t \in [ 0 , T ]$ . Then under the assumptions listed below, there are constants $C _ { 1 } , C _ { 2 } > 0$ depending on $\mathcal { X }$ and $R _ { t }$ but not D such that, $i f \mathcal { L } ( y _ { 0 } )$ denotes the law of $y _ { 0 } ,$ , we have the total variation bound 

$$
| | \mathcal {L} (y _ {0}) - p _ {\mathrm{data}} | | _ {\mathrm{TV}} \leq 3 M T + \left\{\left(| R | S D C _ {1}\right) ^ {2} + \frac {1}{2} C _ {2} (M + C _ {1} S D | R |) \right\} \tau T + 2 \exp \left\{- \frac {T \log^ {2} 2}{t _ {\mathrm{mix}} \log 4 D} \right\}
$$

The above theorem holds under the following assumptions, where we write $x \sim y$ for $x , y \in S ^ { D }$ if they differ in at most one coordinate. 

Assumption 1. The data distribution $p _ { \mathrm { d a t a } }$ is strictly positive. 

Assumption 2. There exists a constant $C _ { 1 } > 0 ,$ , depending on $S$ and $R _ { t }$ but not $D ,$ , such that for all $t \in [ 0 , T ]$ and $x , y \in S ^ { D }$ such that $x \sim y ,$ we have 

$$
\frac {q _ {t} (x)}{q _ {t} (y)} \leq C _ {1}.
$$

Assumption 3. There exists a constant $C _ { 2 } > 0 ,$ , depending on $S$ and $R _ { t }$ but not $D ,$ such that for all $t \in [ 0 , \bar { T } ]$ and all $x , y \in S ^ { D }$ such that $x \sim y ,$ we have 

$$
\sum_ {z} \left| \hat {R} _ {t} (x, x + z) - \hat {R} _ {t} (y, y + z) \right| \leq C _ {2}.
$$

If instead we were to allow $C _ { 1 }$ and $C _ { 2 }$ to depend on the dimension $D ,$ then Assumptions 2 and 3 follow trivially from Assumption 1 and the finiteness of the state space. However, we choose the stronger formulation above in order to make explicit the dependence of the error bound on the dimension, as previously explained. 

As remarked in the main text, in most cases of practical interest (including the two examples explored in Section 6), Assumption 3 holds only approximately. However, we still expect the bound in Assumption 3 to hold whenever $x , y$ are in addition chosen such that the tau-leaping approximation of the reverse process makes a jump between them with reasonably high probability. For example, in the case where our data is ordinal, we expect that for any $x \sim y$ jumps from x to y are only common when x is close to $y ,$ and thus $\hat { R } _ { t } ( x , x + z )$ and $\hat { R } _ { t } ( y , y + z )$ should be reasonably close whenever a jump from x to y occurs. Under a weaker assumption of this form, the proof of Theorem 1 can be adapted to work along similar lines, at the cost of a significant increase in technicality. We therefore choose to focus on the simpler case where Assumption 3 holds as it illustrates the key ideas. 

In order to prove Theorem 1 we will require the following lemmas. 

Proposition 5. Let $( x _ { t } ) _ { t \in [ 0 , T ] }$ and $( y _ { t } ) _ { t \in [ 0 , T ] }$ be continuous time Markov chains on a finite state space S with generators $G _ { t }$ and $H _ { t }$ respectively which are both bounded and continuous in t. Let the Markov kernels associated to $X$ and Y be K and L respectively. Then for any probability distribution ν on $S$ we have 

$$
| | \nu K - \nu L | | _ {\mathrm{TV}} \leq \int_ {0} ^ {T} \sup _ {x \in S} \left\{\sum_ {y \neq x} | G _ {t} (x, y) - H _ {t} (x, y) | \right\} \mathrm{d} t
$$

Proof. We define a coupling of $( x _ { t } ) _ { t \in [ 0 , T ] }$ and $( y _ { t } ) _ { t \in [ 0 , T ] }$ as follows, based on the construction in Chapter 20.1 of [24]. First take $Z \sim \nu$ and set $x _ { 0 } = y _ { 0 } { \overset { \cdot } { = } } \tilde { Z }$ . Also define the variables $\tilde { x } _ { 0 } = \tilde { y } _ { 0 } = Z$ . 

Next, fix λ such that $| G _ { t } ( x , x ) | , | H _ { t } ( x , x ) | \leq \lambda$ for all $x \in S , t \in [ 0 , T ]$ , let $( N _ { s } ) _ { 1 < s < T }$ be a Poisson process on $[ 0 , T ]$ of rate λ, and set $N _ { 0 } = 0$ . We write $N = N _ { T }$ , and $S _ { 1 } , S _ { 2 } , \ldots , \overline { { S } } _ { N } ^ { - }$ for the arrival times and set $\bar { S _ { n + 1 } } = T$ . We construct $x _ { t }$ and $y _ { t }$ for $t > 0$ inductively as follows. For $t \in [ 0 , S _ { 1 } )$ let $x _ { t } = y _ { t } = x _ { 0 } . \operatorname { L e t } 1 \leq j \leq N$ . Given $( x _ { r } : r < S _ { j } ) , ( y _ { r } : r < S _ { j } )$ ), and $\tilde { x } _ { j } , \tilde { y } _ { j }$ , define the following probability measures 

$$
\rho_ {j} (\tilde {x} _ {j}, w) := \left\{ \begin{array}{l} G _ {S _ {j}} (\tilde {x} _ {j}, w) / \lambda , \quad w \neq \tilde {x} _ {j} \\ 1 - G _ {S _ {j}} (\tilde {x} _ {j}, w) / \lambda , \quad w = \tilde {x} _ {j}, \end{array} \right.
$$

$$
\rho_ {j} ^ {\prime} (\tilde {y} _ {j}, w) := \left\{ \begin{array}{l} H _ {S _ {j}} (\tilde {y} _ {j}, w) / \lambda , \quad w \neq \tilde {y} _ {j} \\ 1 - H _ {S _ {j}} (\tilde {y} _ {j}, w) / \lambda , \quad w = \tilde {y} _ {j}. \end{array} \right.
$$

Sample $( \tilde { x } _ { j + 1 } , \tilde { y } _ { j + 1 } )$ from a maximal coupling of $( \rho _ { j } , \rho _ { j } ^ { \prime } )$ and for $t \in [ S _ { j } , S _ { j + 1 } )$ set $x _ { t } = \tilde { x } _ { j + 1 }$ , $y _ { t } = \tilde { y } _ { j + 1 }$ . Finally set $x _ { T } = x _ { S _ { N } }$ and $y _ { T } = y _ { S _ { N } }$ . 

Now, observe that $( x _ { t } , y _ { t } ) _ { t \in [ 0 , T ] }$ defined in this way is a coupling of the given Markov chains. Moreover, 

$$
\begin{array}{l} | | \nu K - \nu L | | _ {\mathrm{TV}} \leq \mathbb {P} (x _ {T} \neq y _ {T}) \\ = \mathbb {E} \left[ \sum_ {j = 1} ^ {N} \mathbb {I} \left\{x _ {s} = y _ {s}, s <   S _ {j} \right\} \mathbb {I} \left\{x _ {S _ {j}} \neq y _ {S _ {j}} \right\} \right] \\ = \sum_ {n = 0} ^ {\infty} \frac {\lambda^ {n} e ^ {- \lambda}}{n !} \sum_ {j = 0} ^ {n} \mathbb {E} \left[ \mathbb {I} \left\{x _ {s} = y _ {s}, s <   S _ {j} \right\} \mathbb {I} \left\{x _ {S _ {j}} \neq y _ {S _ {j}} \right\} \right] \\ \end{array}
$$

and using the fact that jumps are coupled maximally 

$$
\begin{array}{l} = \sum_ {n = 0} ^ {\infty} \frac {\lambda^ {n} e ^ {- \lambda}}{n !} \sum_ {j = 0} ^ {n} \mathbb {E} \left[ \mathbb {I} \left\{x _ {s} = y _ {s}, s <   S _ {j} \right\} \times \| \rho_ {j} \left(X _ {S _ {j - 1}}, \cdot\right) - \tilde {\rho} _ {j} \left(X _ {S _ {j - 1}}, \cdot\right) \| _ {\mathrm{TV}} \right] \\ = \sum_ {n = 0} ^ {\infty} \frac {\lambda^ {n} e ^ {- \lambda}}{n !} \sum_ {j = 0} ^ {n} \mathbb {E} \left[ \mathbb {I} \left\{x _ {s} = y _ {s}, s <   S _ {j} \right\} \frac {1}{\lambda} \sum_ {z} \left| G _ {S _ {j}} (x _ {S _ {j - 1}}, z) - H _ {S _ {j}} (x _ {S _ {j - 1}}, z) \right| \right] \\ = \frac {1}{\lambda} \mathbb {E} \left[ \sum_ {s: x _ {s} \neq x _ {s -}} \sum_ {z} \left| G _ {s} \left(x _ {s -}, z\right) - H _ {s} \left(x _ {s -}, z\right) \right| \right] \\ = \frac {1}{\lambda} \int_ {s = 0} ^ {T} \mathbb {E} \left[ \lambda \sum_ {z} | G _ {s} (x _ {s -}, z) - H _ {s} (x _ {s -}, z) | \right] \\ = \int_ {s = 0} ^ {T} \mathbb {E} \left[ \sum_ {z} \left| G _ {s} \left(x _ {s -}, z\right) - H _ {s} \left(x _ {s -}, z\right) \right| \right] \mathrm{d} s \\ \end{array}
$$

as required. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/5ee91f3c648be14ad9f31c6c1c44a0b9923a4e4f40a015a63a5a725ecbb169e8.jpg)


Proposition 6. For all $t \in [ 0 , T ]$ and $x , y \in \mathcal { X } ^ { D }$ such that $x \sim y ,$ , we have 

$$
| \partial_ {t} \hat {R} _ {t} (x, y) | \leq 2 | R | ^ {2} S D C _ {1} ^ {2}
$$

Moreover, it follows that $\hat { R } _ { t }$ is bounded and continuous in t. 

Proof. Omitting the superscripts for brevity where the notation is clear, we have 

$$
\begin{array}{l} \left| \partial_ {t} \hat {R} _ {t} ^ {1: D} (x ^ {1: D}, y ^ {1: D}) \right| = \left| R _ {t} (y, x) \partial_ {t} \left\{\frac {q _ {t} (y)}{q _ {t} (x)} \right\} \right| \\ = \left| R _ {t} (y, x) \left\{\frac {q _ {t} (y)}{q _ {t} (x)} \frac {\sum_ {z} R _ {t} (z , y) q _ {t} (z)}{q _ {t} (y)} - \frac {q _ {t} (y)}{q _ {t} (x)} \frac {\sum_ {z} R _ {t} (z , x) q _ {t} (z)}{q _ {t} (x)} \right\} \right| \\ \leq 2 | R | ^ {2} S D C _ {1} ^ {2} \\ \end{array}
$$

where the second line follows from Kolmogorov’s forward equation and the final inequality follows from Assumption 2 plus the fact that $R _ { t } ( z , x )$ (resp. $R _ { t } ( z , y ) )$ is only non-zero when $x \sim z \left( \mathrm { r e s p } \right.$ . $y \sim z )$ , and there are at most $| S | | D |$ values of x (resp. y) for which this holds. □ 

We now give the proof of Theorem 1. 

denote Proof of Theorem 1. Let us label the time steps used in tau-leaping by $\tau _ { k } = t _ { k } - t _ { k - 1 }$ , and denote the target stationary distribution by $0 = t _ { 0 } < t _ { 1 } < \cdot \cdot \cdot < t _ { N } = T _ { \cdot }$ $\begin{array} { r } { \pi ^ { D } ( x ^ { 1 : D } ) = \prod _ { d = 1 } ^ { D } \pi ( x ^ { d } ) } \end{array}$ , where π is the invariant distribution of the single-dimensional transition matrix $R _ { t } ^ { 1 }$ . 

Also, let $\mathcal { R } _ { k } ^ { \theta , ( \tau ) }$ be the Markov kernel corresponding to applying the tau-leaping approximatioθ,(τ) θ,(τ) θ,(τ) rate matrix $\hat { R } _ { t _ { k } } ^ { \theta }$ to move from $t _ { k } : 0 \ : t _ { k - 1 }$ , and denote $\mathcal { R } ^ { \theta , ( \tau ) } = \mathcal { R } _ { N } ^ { \theta , ( \tau ) } \mathcal { R } _ { N - 1 } ^ { \theta , ( \tau ) } \cdot . . \mathcal { R } _ { 1 } ^ { \theta , ( \tau ) }$ o that $\mathcal { R } ^ { \theta , ( \tau ) }$ expresses the full dynamics of the tau-leaping process and we have $\mathcal { L } ( \hat { y } _ { 0 } ) = \pi ^ { D } \mathcal { R } ^ { \theta , ( \tau ) }$ . 

Then, as in [19] we can decompose 

$$
| | \pi^ {D} \mathcal {R} ^ {\theta , (\tau)} - p _ {d} | | _ {\mathrm{TV}} \leq | | \pi^ {D} \mathcal {R} ^ {\theta , (\tau)} - \pi^ {D} (\mathbb {P} ^ {R}) _ {T | 0} | | _ {\mathrm{TV}} + | | \pi^ {D} - q _ {T} | | _ {\mathrm{TV}}
$$

where $\mathbb { P } ^ { R }$ is the path measure of the exact reverse process. 

We deal with the second term first. Let $t _ { \mathrm { m i x } }$ be the (1/4)-mixing time of the single-dimension CTMC with rate matrix $R _ { t } ^ { 1 }$ , i.e. 

$$
t _ {\mathrm{mix}} = \inf \left\{t \geq 0: \sup _ {x _ {0} ^ {1} \in S} | | q _ {t | 0} (\cdot | x _ {0} ^ {1}) - \pi | | _ {\mathrm{TV}} \leq \frac {1}{4} \right\}
$$

It then follows from 

$$
| | q _ {t | 0} (\cdot | x _ {0} ^ {1: D}) - \pi^ {D} | | _ {\mathrm{TV}} \leq \sum_ {d = 1} ^ {D} | | q _ {t | 0} (\cdot | x _ {0} ^ {d}) - \pi | | _ {\mathrm{TV}}
$$

that $t _ { \mathrm { m i x } } ^ { D } .$ the (1/4)-mixing time of the full CTMC with rate matrix $R _ { t } ^ { 1 : D }$ , satisfies the inequality $t _ { m i x } ^ { D } \leq \{ 1 + \lceil \log _ { 2 } D \rceil \} t _ { \mathrm { m i x } }$ . If we view $( x _ { m t _ { m i x } ^ { D } } ) _ { m \in \mathbb { N } }$ as a discrete-time Markov chain, then standard results on Markov chain mixing (see, for example, Chapter 4.5 of [24]) show that 

$$
| | q _ {m t _ {m i x} ^ {D} | 0} (\cdot | x _ {0} ^ {1: D}) - \pi^ {D} | | _ {\mathrm{TV}} \leq 2 ^ {- m}
$$

It then follows that for any $T \geq 0$ we have 

$$
| | \pi^ {D} - q _ {T} | | _ {\mathrm{TV}} \leq 2 \exp \left\{- \frac {T \log 2}{t _ {m i x} ^ {D}} \right\} \leq 2 \exp \left\{- \frac {T \log^ {2} 2}{t _ {\mathrm{mix}} \log 4 D} \right\}
$$

completing the bound on the second term. 

To bound the first term, we define $\mathcal { P } _ { k } = ( \mathbb { P } ^ { R } ) _ { T - t _ { k - 1 } | T - t _ { k } }$ and decompose it as 

$$
\begin{array}{l} | | \pi \mathcal {R} ^ {\theta , (\tau)} - \pi (\mathbb {P} ^ {R}) _ {T | 0} | | _ {\mathrm{TV}} \leq \sup _ {\nu} | | \nu \mathcal {R} _ {N} ^ {\theta , (\tau)} \dots \mathcal {R} _ {1} ^ {\theta , (\tau)} - \nu \mathcal {P} _ {N} \dots \mathcal {P} _ {1} | | _ {\mathrm{TV}} \\ \leq \sup _ {\nu} | | \nu \mathcal {R} _ {N} ^ {\theta , (\tau)} \mathcal {R} _ {N - 1} ^ {\theta , (\tau)} \dots \mathcal {R} _ {1} ^ {\theta , (\tau)} - \nu \mathcal {R} _ {N} ^ {\theta , (\tau)} \mathcal {P} _ {N - 1} \dots \mathcal {P} _ {1} | | _ {\mathrm{TV}} \\ + \sup _ {\nu} | | \nu \mathcal {R} _ {N} ^ {\theta , (\tau)} \mathcal {P} _ {N - 1} \dots \mathcal {P} _ {1} - \nu \mathcal {P} _ {N} \mathcal {P} _ {N - 1} \dots \mathcal {P} _ {1} | | _ {\mathrm{TV}} \\ \leq \sup _ {\nu} | | \nu \mathcal {R} _ {N - 1} ^ {\theta , (\tau)} \dots \mathcal {R} _ {1} ^ {\theta , (\tau)} - \nu \mathcal {P} _ {N - 1} \dots \mathcal {P} _ {1} | | _ {\mathrm{TV}} + \sup _ {\nu} | | \nu \mathcal {R} _ {N} ^ {\theta , (\tau)} - \nu \mathcal {P} _ {N} | | _ {\mathrm{TV}} \\ \leq \sum_ {k = 1} ^ {N} \sup _ {\nu} | | \nu \mathcal {R} _ {k} ^ {\theta , (\tau)} - \nu \mathcal {P} _ {k} | | _ {\mathrm{TV}} \\ \end{array}
$$

by proceeding inductively. So it suffices to find bounds on the total variation distance accumulated on each interval $[ t _ { k - 1 } , t _ { k } ]$ . 

Let $\mathcal { R } _ { k } ^ { \theta }$ be the Markov kernel corresponding to running the chain from $t _ { k }$ to $t _ { k - 1 }$ with constant rate matrix $\hat { R } _ { t _ { k } } ^ { \theta }$ . Since by Proposition 6 the reverse rate matrix $\hat { R } _ { t }$ is bounded and continuous in t, using Proposition 5 we made deduce that for any distribution ν on $S$ we have 

$$
\begin{array}{l} | | \nu \mathcal {P} _ {k} - \nu \mathcal {R} _ {k} ^ {\theta} | | _ {\mathrm{TV}} \leq \int_ {t _ {k - 1}} ^ {t _ {k}} \sup _ {x \in S} \Bigl \{\sum_ {y \neq x} \bigl | \hat {R} _ {t} (x, y) - \hat {R} _ {t _ {k}} ^ {\theta} (x, y) \bigr | \Bigr \}   \mathrm{d} t \\ \leq \int_ {t _ {k - 1}} ^ {t _ {k}} \sup _ {x \in S} \left\{\sum_ {y \neq x} \left| \hat {R} _ {t} (x, y) - \hat {R} _ {t _ {k}} (x, y) \right| \right\} d t \\ + \int_ {t _ {k - 1}} ^ {t _ {k}} \sup _ {x \in S} \left\{\sum_ {y \neq x} \left| \hat {R} _ {t _ {k}} (x, y) - \hat {R} _ {t _ {k}} ^ {\theta} (x, y) \right| \right\} d t \\ \end{array}
$$

The first half of this expression can be bounded using the Mean Value Theorem, according to 

$$
\begin{array}{l} \int_ {t _ {k - 1}} ^ {t _ {k}} \sup _ {x \in S} \left\{\sum_ {y \neq x} \left| \hat {R} _ {t} (x, y) - \hat {R} _ {t _ {k}} (x, y) \right| \right\} \mathrm{d} t \leq \int_ {t _ {k - 1}} ^ {t _ {k}} | t - t _ {k} | \cdot 2 | R | ^ {2} S ^ {2} D ^ {2} C _ {1} ^ {2} \mathrm{d} t \\ \leq \left(| R | S D C _ {1} \tau_ {k}\right) ^ {2} \\ \end{array}
$$

where in the first line we have used that the summand is only non-zero when $y \sim x ,$ and there are at most $| S | | D |$ values of y for which this holds. The second term can be bounded using condition (2), to get 

$$
\int_ {t _ {k - 1}} ^ {t _ {k}} \sup _ {x \in S} \left\{\sum_ {y \neq x} \left| \hat {R} _ {t _ {k}} (x, y) - \hat {R} _ {t _ {k}} ^ {\theta} (x, y) \right| \right\} d t \leq M \tau_ {k}
$$

Combining these two expressions, we get a bound on $| | \nu \mathcal { P } _ { k } - \nu \mathcal { R } _ { k } ^ { \theta } | | _ { \mathrm { T V } }$ 

$$
| | \nu \mathcal {P} _ {k} - \nu \mathcal {R} _ {k} ^ {\theta} | | _ {\mathrm{TV}} \leq \left(| R | S D C _ {1} \tau_ {k}\right) ^ {2} + M \tau_ {k}
$$

It remains to bound $| | \nu \mathcal { R } _ { k } ^ { \theta } - \nu \mathcal { R } _ { k } ^ { \theta , ( \tau ) } | | _ { \mathrm { T V } }$ θk − νRθ,(τ )k ||T . Note that performing tau-leaping with x $\hat { R } _ { t _ { k } } ^ { \theta }$ starting in constant r $x _ { t _ { k } }$ is eqmatrix t to running a continuous time Markov chain from time  given by $t _ { k } \mathrm { ~ t o ~ } t _ { k - 1 }$ with $\hat { R } _ { t _ { k } } ^ { \theta , ( \tau ) }$ 

$$
\hat {R} _ {t _ {k}} ^ {\theta , (\tau)} (x, y) = \hat {R} _ {t _ {k}} ^ {\theta} (x _ {t _ {k}}, y - x + x _ {t _ {k}})
$$

(followed potentially by a clamping operation to keep us within ${ \boldsymbol { \mathcal { X } } } ^ { D } )$ . By an analogous argument to the proof of Proposition 5, 

$$
| | \delta_ {x _ {t _ {k}}} \mathcal {R} _ {k} ^ {\theta} - \delta_ {x _ {t _ {k}}} \mathcal {R} _ {k} ^ {\theta , (\tau)} | | _ {\mathrm{TV}} \leq \int_ {t _ {k - 1}} ^ {t _ {k}} \mathbb {E} \Big [ \sum_ {y \neq x _ {t}} | \hat {R} _ {t _ {k}} ^ {\theta} (x _ {t}, y) - \hat {R} _ {t _ {k}} ^ {\theta} (x _ {t _ {k}}, y - x _ {t} + x _ {t _ {k}}) | \Big ]   \mathrm{d} t
$$

where the expectation is taken over $( x _ { t } ) _ { t \in [ t _ { k - 1 } , t _ { k } ] }$ distributed according to the exact CTMC with rate matrix $\hat { R } _ { t _ { k } } ^ { \theta }$ . (Note we have disregarded the clamping operation, since this can only decrease the resulting total variation distance.) 

We may rewrite this bound in terms of the exact reverse process using condition (2) to get 

$$
| | \delta_ {x _ {t _ {k}}} \mathcal {R} _ {k} ^ {\theta} - \delta_ {x _ {t _ {k}}} \mathcal {R} _ {k} ^ {\theta , (\tau)} | | _ {\mathrm{TV}} \leq \int_ {t _ {k - 1}} ^ {t _ {k}} \mathbb {E} \Big [ 2 M + \sum_ {y \neq x _ {t}} | \hat {R} _ {t _ {k}} (x _ {t}, y) - \hat {R} _ {t _ {k}} (x _ {t _ {k}}, y - x _ {t} + x _ {t _ {k}}) | \Big ]   \mathrm{d} t
$$

Let $J _ { t }$ be the number of jumps that $\left( { x _ { t } } \right)$ makes between $t _ { k }$ and $t ,$ and label the times of these jumps as $s _ { 1 } , \ldots , s _ { j }$ where $t \leq s _ { 1 } \leq \cdot \cdot \cdot \leq s _ { j } \leq t _ { k }$ and $j = J _ { t }$ for convenience. Then by Assumption 3, we have 

$$
\begin{array}{l} \sum_ {y \neq x _ {t}} | \hat {R} _ {t _ {k}} (x _ {t}, y) - \hat {R} _ {t _ {k}} (x _ {t _ {k}}, y - x _ {t} + x _ {t _ {k}}) | \leq \sum_ {z} | \hat {R} _ {t _ {k}} (x _ {t}, x _ {t} + z) - \hat {R} _ {t _ {k}} (x _ {s _ {1}}, x _ {s _ {1}} + z) | + \dots \\ + \sum_ {z} | \hat {R} _ {t _ {k}} (x _ {s _ {j}}, x _ {s _ {j}} + z) - \hat {R} _ {t _ {k}} (x _ {t _ {k}}, x _ {t _ {k}} + z) | \\ \leq C _ {2} J _ {t} \\ \end{array}
$$

where we have made the substitution $ { \boldsymbol { z } } =  { \boldsymbol { y } } -  { \boldsymbol { x } } _ { t _ { k } }$ . We conclude that 

$$
\begin{array}{l} | | \delta_ {x _ {t _ {k}}} \mathcal {R} _ {k} ^ {\theta} - \delta_ {x _ {t _ {k}}} \mathcal {R} _ {k} ^ {\theta , (\tau)} | | _ {\mathrm{TV}} \leq \int_ {t _ {k - 1}} ^ {t _ {k}} \mathbb {E} [ 2 M + C _ {2} J _ {t} ] \mathrm{d} t \\ \leq 2 M \left| t _ {k} - t _ {k - 1} \right| + C _ {2} \int_ {t _ {k - 1}} ^ {t _ {k}} \left| t _ {k} - t \right| \cdot \sup _ {x} \left| \hat {R} _ {t _ {k}} ^ {\theta} (x, x) \right| d t \\ \leq 2 M \gamma_ {k} + \frac {1}{2} C _ {2} | \hat {R} _ {t _ {k}} ^ {\theta} | \tau_ {k} ^ {2} \\ \leq 2 M \tau_ {k} + \frac {1}{2} C _ {2} (M + C _ {1} S D | R |) \tau_ {k} ^ {2} \\ \end{array}
$$

where to bound $\mathbb { E } [ J _ { t } ]$ we have observed that jumps of $\left( { x _ { t } } \right)$ occur at a rate bounded above by $\operatorname* { s u p } _ { x } | \hat { R } _ { t _ { k } } ^ { \theta } ( x , x ) |$ , and in the last line we have used the condition (2) and Assumption 2. Since the above holds for any choice of $x _ { t _ { k } }$ , it follows that 

$$
\sup _ {\nu} | | \nu \mathcal {R} _ {k} ^ {\theta} - \nu \mathcal {R} _ {k} ^ {\theta , (\tau)} | | _ {\mathrm{TV}} \leq 2 M \tau_ {k} + \frac {1}{2} C _ {2} (M + C _ {1} S D | R |) \tau_ {k} ^ {2}
$$

Summing over k and putting all our bounds together, we get 

$$
| | \mathcal {L} (y _ {0}) - p _ {\mathrm{data}} | | _ {\mathrm{TV}} \leq 3 M T + \left\{\left(| R | S D C _ {1}\right) ^ {2} + \frac {1}{2} C _ {2} (M + C _ {1} S D | R |) \right\} \tau T + 2 \exp \left\{- \frac {T \log^ {2} 2}{t _ {\mathrm{mix}} \log 4 D} \right\}
$$

as required. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/4787f1deb75f42f91ab6b53ca763a0548c7a2fb50cbe68e2e6356c275ed65fda.jpg)


# C Continuous Time ELBO Details

# C.1 Comparison with the Discrete Time ELBO

It is easiest to gain intuition on the ${ \mathcal { L } } _ { \mathrm { C T } }$ objective by comparing it to its discrete time counterpart, $\mathcal { L } _ { \mathrm { D T } }$ , and examining the way in which ${ \mathcal { L } } _ { \mathrm { D T } }$ in the limit becomes $\mathcal { L } _ { \mathrm { C T } }$ when we take the time step size to be very small. We repeat the definition of ${ \mathcal { L } } _ { \mathrm { C T } }$ here for convenience 

$$
\mathcal {L} _ {\mathrm{CT}} (\theta) = T \mathbb {E} _ {t \sim \mathcal {U} (0, T) q _ {t} (x) r _ {t} (\tilde {x} | x)} \Big [ \Big \{\sum_ {x ^ {\prime} \neq x} \hat {R} _ {t} ^ {\theta} (x, x ^ {\prime}) \Big \} - \mathcal {Z} ^ {t} (x) \log \Big (\hat {R} _ {t} ^ {\theta} (\tilde {x}, x) \Big) \Big ] + C.
$$

Recall that a single term from the KL sum in ${ \mathcal { L } } _ { \mathrm { D T } }$ up to an additive constant independent of θ is 

$$
- \mathbb {E} _ {q _ {k} (x _ {k}) q _ {k + 1 | k} (x _ {k + 1} | x _ {k})} \left[ \log p _ {k | k + 1} ^ {\theta} (x _ {k} | x _ {k + 1}) \right].
$$

Minimizing this term is to sample $( x _ { k } , x _ { k + 1 } )$ from the forward dynamics and then maximize the assigned model probability for the pairing in the reverse direction. A similar idea can be used to understand ${ \mathcal { L } } _ { \mathrm { C T } }$ . First, we write log $p _ { k | k + 1 } ^ { \theta } ( x _ { k } | x _ { k + 1 } )$ in terms of $\hat { R } _ { k } ^ { \theta }$ as 

$$
\begin{array}{l} \log p _ {k | k + 1} ^ {\theta} (x _ {k} | x _ {k + 1}) = \delta_ {x _ {k}, x _ {k + 1}} \left(\hat {R} _ {k} ^ {\theta} (x _ {k}, x _ {k}) \Delta t + o (\Delta t)\right) \\ + \left(1 - \delta_ {x _ {k}, x _ {k + 1}}\right) \log \left(\hat {R} _ {k} ^ {\theta} (x _ {k + 1}, x _ {k}) \Delta t + o (\Delta t)\right) \\ \end{array}
$$

where we have separated the cases when $x _ { k } = x _ { k + 1 }$ and when $x _ { k } \neq x _ { k + 1 }$ (see the proof of ${ \mathcal { L } } _ { \mathrm { C T } }$ for the full details). The first term will become the $\textstyle \sum _ { x ^ { \prime } \neq x } { \hat { R } } _ { t } ^ { \theta } ( x , x ^ { \prime } )$ term in ${ \mathcal { L } } _ { \mathrm { C T } }$ whilst the second term will become the ${ \mathcal { Z } } ^ { t } ( x )$ log $\left( \hat { R } _ { t } ^ { \theta } ( \tilde { x } , x ) \right)$  term. Now, when we minimize $\mathcal { L } _ { \mathrm { C T } }$ , we are sampling $( x , { \tilde { x } } )$ from the forward process and then maximizing the assigned model probability for the pairing in the reverse direction, just as in ${ \mathcal { L } } _ { \mathrm { D T } }$ . The slight extra complexity comes from the fact we are considering the case when $x _ { k } = x _ { k + 1 }$ and the case when $x _ { k } \neq x _ { k + 1 }$ separately. When $x _ { k } = x _ { k + 1 }$ , this corresponds to the first term in $\mathcal { L } _ { \mathrm { C T } }$ which we can see is minimizing the reverse rate out of x which is exactly maximizing the model probability for no transition to occur. When $x _ { k } \neq x _ { k + 1 }$ , this corresponds to the second term in $\mathcal { L } _ { \mathrm { C T } }$ , which is maximizing the reverse rate from x˜ to x which in turn maximizes the model probability for the x˜ to x transition to occur. 

# C.2 Conditional Form

For the conditional form of $\mathcal { L } _ { \mathrm { C T } }$ , denoted as $\bar { \mathcal { L } } _ { \mathrm { C T } }$ , we instead upper bound the negative conditional model log-likelihood, $\mathbb { E } _ { p _ { \mathrm { d a t a } } ( x _ { 0 } , y ) } [ - \log p _ { 0 } ^ { \theta } ( x _ { 0 } | y ) ]$ where y is our conditioner. $\bar { \mathcal { L } } _ { \mathrm { C T } }$ has the following form 

$$
\bar {\mathcal {L}} _ {\mathrm{CT}} (\theta) = T \mathbb {E} _ {t \sim \mathcal {U} (0, T) p _ {\mathrm{data}} (x _ {0}, y) q _ {t | 0} (x | x _ {0}) r _ {t} (\tilde {x} | x)} \Big [ \Big \{\sum_ {x ^ {\prime} \neq x} \hat {R} _ {t} ^ {\theta} (x, x ^ {\prime} | y) \Big \} - \mathcal {Z} ^ {t} (x) \log \Big (\hat {R} _ {t} ^ {\theta} (\tilde {x}, x | y) \Big) \Big ] + C,
$$

where 

$$
\begin{array}{l} \hat {R} _ {t} ^ {\theta} (x, \tilde {x} | y) = R _ {t} (\tilde {x}, x) \sum_ {x _ {0}} \frac {q _ {t | 0} (\tilde {x} | x _ {0})}{q _ {t | 0} (x | x _ {0})} p _ {0 | t} ^ {\theta} (x _ {0} | x, y) \quad \text { for } \quad x \neq \tilde {x}. \\ = - \sum_ {x ^ {\prime} \neq x} \hat {R} _ {t} ^ {\theta} (x, x ^ {\prime} | y) \quad \text { for } \quad x = \tilde {x} \\ \end{array}
$$

This follows easily from considering the conditional form of the discret time ELBO, $\bar { \mathcal { L } } _ { \mathrm { D T } }$ and using the same arguments as before to go from discrete time to continuous time. 

$$
\mathbb {E} _ {p _ {\mathrm{data}} (x _ {0}, y)} [ - \log p _ {0} ^ {\theta} (x _ {0} | y) ] \leq \mathbb {E} _ {p _ {\mathrm{data}} (x _ {0}, y) q _ {1: K | 0} (x _ {1: K} | x _ {0})} \left[ - \log \frac {p _ {0 : K} ^ {\theta} (x _ {0 : K} | y)}{q _ {1 : K | 0} (x _ {1 : K} | x _ {0})} \right] = \bar {\mathcal {L}} _ {\mathrm{DT}}
$$

# C.3 Continuous Time ELBO with Factorization Assumptions

In the following Proposition, we show the form of $\mathcal { L } _ { \mathrm { C T } }$ when we use a factorized forward process. We note that in the proof we rearrange the sampling distribution from $p _ { \mathrm { d a t a } } ( \boldsymbol { x } _ { 0 } ^ { 1 : D } ) q _ { t | 0 } ( \boldsymbol { x } ^ { 1 : D } | \boldsymbol { x } _ { 0 } ^ { 1 : D } ) r _ { t } ( \tilde { \boldsymbol { x } } ^ { 1 : D } | \boldsymbol { x } ^ { 1 : D } ) \stackrel { \boldsymbol { \kappa } } { \mathrm { t o } } p _ { \mathrm { d a t a } } ( \boldsymbol { x } _ { 0 } ^ { 1 : D } ) \psi _ { t } ( \tilde { \boldsymbol { x } } ^ { \top : D } | \boldsymbol { x } _ { 0 } ^ { 1 : D } ) \phi _ { t } ^ { \boldsymbol { \star } } ( \boldsymbol { x } ^ { 1 : \tilde { D } } | \boldsymbol { \tilde { x } } ^ { 1 : D } , \boldsymbol { x } _ { 0 } ^ { 1 : D } )$ . This is not strictly necessary but it allows us to analytically sum over the intermediate $\pmb { x } ^ { 1 : D }$ variable which greatly reduces the variance of the resulting objective. 

Proposition 7. The ${ \mathcal { L } } _ { \mathrm { C T } }$ objective when we substitute in the factorized forms for the forward and reverse process given in Proposition 3 is 

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{CT}} = T \mathbb {E} _ {t \sim \mathcal {U} (0, T) p _ {\mathrm{data}} (\pmb {x} _ {0} ^ {1: D}) q _ {t | 0} (\pmb {x} ^ {1: D} | \pmb {x} _ {0} ^ {1: D})} \left[ \sum_ {d = 1} ^ {D} \sum_ {x ^ {\prime d} \neq x ^ {d}} \hat {R} _ {t} ^ {\theta d} (\pmb {x} ^ {1: D}, x ^ {\prime d}) \right] \\ - T \mathbb {E} _ {t \sim \mathcal {U} (0, T) p _ {\mathrm{data}} (\pmb {x} _ {0} ^ {1: D}) \psi_ {t} (\tilde {\pmb {x}} ^ {1: D} | \pmb {x} _ {0} ^ {1: D})} \left[ \sum_ {d = 1} ^ {D} \sum_ {\boldsymbol {x} ^ {d} \neq \tilde {\boldsymbol {x}} ^ {d}} \phi_ {t} (\boldsymbol {x} ^ {d} | \tilde {\pmb {x}} ^ {1: D}, \pmb {x} _ {0} ^ {1: D}) \mathcal {Z} ^ {t} (\tilde {\pmb {x}} ^ {1: D / d} \circ \boldsymbol {x} ^ {d}) \log \left(\hat {R} _ {t} ^ {\theta d} (\tilde {\pmb {x}} ^ {1: D}, \boldsymbol {x} ^ {d})\right) \right] \\ + C \\ \end{array}
$$

with 

$$
\hat {R} _ {t} ^ {\theta d} (\pmb {x} ^ {1: D}, \tilde {x} ^ {d}) = R _ {t} ^ {d} (\tilde {x} ^ {d}, x ^ {d}) \sum_ {x _ {0} ^ {d}} p _ {0 | t} ^ {\theta} (x _ {0} ^ {d} | \pmb {x} ^ {1: D}) \frac {q _ {t | 0} (\tilde {x} ^ {d} | x _ {0} ^ {d})}{q _ {t | 0} (x ^ {d} | x _ {0} ^ {d})}
$$

$$
\mathcal {Z} ^ {t} (\boldsymbol {x} ^ {1: D}) = \sum_ {d = 1} ^ {D} \sum_ {\tilde {x} ^ {d} \neq x ^ {d}} R _ {t} ^ {d} (x ^ {d}, \tilde {x} ^ {d})
$$

$$
\phi_ {t} (x ^ {d} | \tilde {\boldsymbol {x}} ^ {1: D}, \boldsymbol {x} _ {0} ^ {1: D}) = \frac {R _ {t} ^ {d} (x ^ {d} , \tilde {x} ^ {d}) q _ {t | 0} (\tilde {\boldsymbol {x}} ^ {1 : D \setminus d} \circ x ^ {d} | \boldsymbol {x} _ {0} ^ {1 : D})}{\mathcal {Z} ^ {t} (\tilde {\boldsymbol {x}} ^ {1 : D \setminus d} \circ x ^ {d}) \sum_ {d ^ {\prime} = 1} ^ {D} \sum_ {x ^ {\prime d ^ {\prime}} \neq \tilde {x} ^ {d ^ {\prime}}} \frac {R _ {t} ^ {d ^ {\prime}} (x ^ {\prime d ^ {\prime}} , \tilde {x} ^ {d ^ {\prime}})}{\mathcal {Z} ^ {t} (\tilde {\boldsymbol {x}} ^ {1 : D \setminus d ^ {\prime}} \circ x ^ {\prime d ^ {\prime}})} q _ {t | 0} (\tilde {\boldsymbol {x}} ^ {1: D \setminus d ^ {\prime}} \circ x ^ {\prime d ^ {\prime}} | \boldsymbol {x} _ {0} ^ {1: D})}
$$

where ◦ represents the concatenation of a $D - 1$ dimensional vector, $\scriptstyle \mathbf { x } ^ { 1 : D \setminus d }$ with a scalar $x ^ { d } ,$ , such that the resultant D dimensional vector has $x ^ { d }$ at its $d ^ { \mathrm { t h } }$ dimension. $\psi _ { t } \big ( \tilde { \pmb { x } } ^ { 1 : D } | \pmb { x } _ { 0 } ^ { 1 : D } \big )$ is defined as the marginal of the forward noising process joint, $\begin{array} { r } { \int q _ { t | 0 } \big ( \mathbf { x } ^ { 1 : D } \big | x _ { 0 } ^ { 1 : D } \big ) r _ { t } \big ( \tilde { \mathbf { x } } ^ { 1 : D } \big | \mathbf { x } ^ { 1 : D } \big ) d \mathbf { x } ^ { 1 : D } } \end{array}$ . 

Proof. We first re-write the general form of ${ \mathcal { L } } _ { \mathrm { C T } }$ here 

$$
\mathcal {L} _ {\mathrm{CT}} (\theta) = T \mathbb {E} _ {t \sim \mathcal {U} (0, T) q _ {t} (x) r _ {t} (\tilde {x} | x)} \Big [ \Big \{\sum_ {x ^ {\prime} \neq x} \hat {R} _ {t} ^ {\theta} (x, x ^ {\prime}) \Big \} - \mathcal {Z} ^ {t} (x) \log \Big (\hat {R} _ {t} ^ {\theta} (\tilde {x}, x) \Big) \Big ] + C
$$

where 

$$
\mathcal {Z} ^ {t} (x) = \sum_ {x ^ {\prime} \neq x} R _ {t} (x, x ^ {\prime}) \quad r _ {t} (\tilde {x} | x) = (1 - \delta_ {\tilde {x}, x}) R _ {t} (x, \tilde {x}) / \mathcal {Z} ^ {t} (x).
$$

With a factorized forward process, $\hat { R } _ { t } ^ { \theta }$ becomes 

$$
\hat {R} _ {t} ^ {\theta 1: D} (\boldsymbol {x} ^ {1: D}, \tilde {\boldsymbol {x}} ^ {1: D}) = \sum_ {d = 1} ^ {D} \hat {R} _ {t} ^ {\theta d} (\boldsymbol {x} ^ {1: D}, \tilde {x} ^ {d}) \delta_ {\boldsymbol {x} ^ {1: D \setminus d}, \tilde {\boldsymbol {x}} ^ {1: D \setminus d}}
$$

where 

$$
\hat {R} _ {t} ^ {\theta d} (\pmb {x} ^ {1: D}, \tilde {x} ^ {d}) = R _ {t} ^ {d} (\tilde {x} ^ {d}, x ^ {d}) \sum_ {x _ {0} ^ {d}} p _ {0 | t} ^ {\theta} (x _ {0} ^ {d} | \pmb {x} ^ {1: D}) \frac {q _ {t | 0} (\tilde {x} ^ {d} | x _ {0} ^ {d})}{q _ {t | 0} (x ^ {d} | x _ {0} ^ {d})}
$$

Substituting this form for $\hat { R } _ { t } ^ { \theta 1 : D }$ into the first term in $\mathcal { L } _ { \mathrm { C T } }$ we get 

$$
\begin{array}{l} \sum_ {\boldsymbol {x} ^ {\prime 1: D} \neq \boldsymbol {x} ^ {1: D}} \sum_ {d = 1} ^ {D} \hat {R} _ {t} ^ {\theta d} (\boldsymbol {x} ^ {1: D}, x ^ {\prime d}) \delta_ {\boldsymbol {x} ^ {1: D \setminus d}, \boldsymbol {x} ^ {\prime 1: D \setminus d}} \\ = \sum_ {d = 1} ^ {D} \sum_ {\boldsymbol {x} ^ {\prime d}} \hat {R} _ {t} ^ {\theta d} (\boldsymbol {x} ^ {1: D}, x ^ {\prime d}) \sum_ {\boldsymbol {x} ^ {\prime 1: D \setminus d}} \delta_ {\boldsymbol {x} ^ {1: D \setminus d}, \boldsymbol {x} ^ {\prime 1: D \setminus d}} (1 - \delta_ {\boldsymbol {x} ^ {\prime 1: D}, \boldsymbol {x} ^ {1: D}}) \\ = \sum_ {d = 1} ^ {D} \sum_ {x ^ {\prime d} \neq x ^ {d}} \hat {R} _ {t} ^ {\theta d} (\boldsymbol {x} ^ {1: D}, x ^ {\prime d}) \\ \end{array}
$$

Now we tackle the second term in $\mathcal { L } _ { \mathrm { C T } }$ . We first re-arrange the distribution over which we take the expectation: 

$$
p _ {\mathrm{data}} (\pmb {x} _ {0} ^ {1: D}) q _ {t | 0} (\pmb {x} ^ {1: D} | \pmb {x} _ {0} ^ {1: D}) r _ {t} (\tilde {\pmb {x}} ^ {1: D} | \pmb {x} ^ {1: D}) = p _ {\mathrm{data}} (\pmb {x} _ {0} ^ {1: D}) \psi_ {t} (\tilde {\pmb {x}} ^ {1: D} | \pmb {x} _ {0} ^ {1: D}) \phi_ {t} (\pmb {x} ^ {1: D} | \tilde {\pmb {x}} ^ {1: D}, \pmb {x} _ {0} ^ {1: D})
$$

We have, 

$$
\begin{array}{l} \phi_ {t} \left(\boldsymbol {x} ^ {1: D} \mid \tilde {\boldsymbol {x}} ^ {1: D}, \boldsymbol {x} _ {0} ^ {1: D}\right) \propto q _ {t | 0} \left(\boldsymbol {x} ^ {1: D} \mid \boldsymbol {x} _ {0} ^ {1: D}\right) r _ {t} \left(\tilde {\boldsymbol {x}} ^ {1: D} \mid \boldsymbol {x} ^ {1: D}\right) \\ = q _ {t | 0} (\boldsymbol {x} ^ {1: D} | \boldsymbol {x} _ {0} ^ {1: D}) (1 - \delta_ {\tilde {\boldsymbol {x}} ^ {1: D}, \boldsymbol {x} ^ {1: D}}) \frac {\sum_ {d = 1} ^ {D} R _ {t} ^ {d} (x ^ {d} , \tilde {x} ^ {d}) \delta_ {\boldsymbol {x} ^ {1 : D \setminus d} , \tilde {\boldsymbol {x}} ^ {1 : D \setminus d}}}{\mathcal {Z} ^ {t} (\boldsymbol {x} ^ {1 : D})} \\ = \sum_ {d = 1} ^ {D} \frac {R _ {t} ^ {d} (x ^ {d} , \tilde {x} ^ {d})}{\mathcal {Z} ^ {t} (\tilde {\boldsymbol {x}} ^ {1 : D \setminus d} \circ x ^ {d})} q _ {t | 0} (\tilde {\boldsymbol {x}} ^ {1: D \setminus d} \circ x ^ {d} | \boldsymbol {x} _ {0} ^ {1: D}) \delta_ {\boldsymbol {x} ^ {1: D \setminus d}, \tilde {\boldsymbol {x}} ^ {1: D \setminus d}} (1 - \delta_ {\tilde {\boldsymbol {x}} ^ {1: D}, \boldsymbol {x} ^ {1: D}}) \\ \end{array}
$$

To find the normalization constant, we can sum the proportional term over $\scriptstyle { \pmb x } ^ { 1 : D }$ 

$$
\begin{array}{l} \sum_ {\boldsymbol {x} ^ {1: D}} \sum_ {d = 1} ^ {D} \frac {R _ {t} ^ {d} (x ^ {d} , \tilde {x} ^ {d})}{\mathcal {Z} ^ {t} (\tilde {\boldsymbol {x}} ^ {1 : D \setminus d} \circ x ^ {d})} q _ {t | 0} (\tilde {\boldsymbol {x}} ^ {1: D \setminus d} \circ x ^ {d} | \boldsymbol {x} _ {0} ^ {1: D}) \delta_ {\boldsymbol {x} ^ {1: D \setminus d}, \tilde {\boldsymbol {x}} ^ {1: D \setminus d}} (1 - \delta_ {\tilde {\boldsymbol {x}} ^ {1: D}, \boldsymbol {x} ^ {1: D}}) \\ = \sum_ {d = 1} ^ {D} \sum_ {x ^ {d} \neq \tilde {x} ^ {d}} \frac {R _ {t} ^ {d} (x ^ {d} , \tilde {x} ^ {d})}{\mathcal {Z} ^ {t} (\tilde {\boldsymbol {x}} ^ {1 : D \setminus d} \circ x ^ {d})} q _ {t | 0} (\tilde {\boldsymbol {x}} ^ {1: D \setminus d} \circ x ^ {d} | \boldsymbol {x} _ {0} ^ {1: D}) \\ \end{array}
$$

Therefore, 

$$
\phi_ {t} (\boldsymbol {x} ^ {1: D} | \tilde {\boldsymbol {x}} ^ {1: D}, \boldsymbol {x} _ {0} ^ {1: D}) = (1 - \delta_ {\tilde {\boldsymbol {x}} ^ {1: D}, \boldsymbol {x} ^ {1: D}}) \sum_ {d = 1} ^ {D} \phi_ {t} (x ^ {d} | \tilde {\boldsymbol {x}} ^ {1: D}, \boldsymbol {x} _ {0} ^ {1: D}) \delta_ {\boldsymbol {x} ^ {1: D \setminus d}, \tilde {\boldsymbol {x}} ^ {1: D \setminus d}}
$$

where 

$$
\phi_ {t} (x ^ {d} | \tilde {\boldsymbol {x}} ^ {1: D}, \boldsymbol {x} _ {0} ^ {1: D}) = \frac {R _ {t} ^ {d} (x ^ {d} , \tilde {x} ^ {d}) q _ {t | 0} (\tilde {\boldsymbol {x}} ^ {1 : D \setminus d} \circ x ^ {d} | \boldsymbol {x} _ {0} ^ {1 : D})}{\mathcal {Z} ^ {t} (\tilde {\boldsymbol {x}} ^ {1 : D \setminus d} \circ x ^ {d}) \sum_ {d ^ {\prime} = 1} ^ {D} \sum_ {x ^ {\prime d ^ {\prime}} \neq \tilde {x} ^ {d ^ {\prime}}} \frac {R _ {t} ^ {d ^ {\prime}} (x ^ {\prime d ^ {\prime}} , \tilde {x} ^ {d ^ {\prime}})}{\mathcal {Z} ^ {t} (\tilde {\boldsymbol {x}} ^ {1 : D \setminus d ^ {\prime}} \circ x ^ {\prime d ^ {\prime}})} q _ {t | 0} (\tilde {\boldsymbol {x}} ^ {1: D \setminus d ^ {\prime}} \circ x ^ {\prime d ^ {\prime}} | \boldsymbol {x} _ {0} ^ {1: D})}
$$

Now we write the second term as 

$$
\begin{array}{l} T \mathbb {E} _ {t \sim \mathcal {U} (0, T) p _ {\mathrm{data}} (\boldsymbol {x} _ {0} ^ {1: D}) \psi_ {t} (\tilde {\boldsymbol {x}} ^ {1: D} | \boldsymbol {x} _ {0} ^ {1: D})} \left[ - \sum_ {\boldsymbol {x} ^ {1: D}} \phi_ {t} (\boldsymbol {x} ^ {1: D} | \tilde {\boldsymbol {x}} ^ {1: D}, \boldsymbol {x} _ {0} ^ {1: D}) \mathcal {Z} ^ {t} (\boldsymbol {x} ^ {1: D}) \log \hat {R} _ {t} ^ {\theta   1: D} (\tilde {\boldsymbol {x}} ^ {1: D}, \boldsymbol {x} ^ {1: D}) \right] \\ = - T \mathbb {E} _ {t \sim \mathcal {U} (0, T) p _ {\mathrm{data}} (\boldsymbol {x} _ {0} ^ {1: D}) \psi_ {t} (\tilde {\boldsymbol {x}} ^ {1: D} | \boldsymbol {x} _ {0} ^ {1: D})} \left[ \sum_ {d = 1} ^ {D} \sum_ {x ^ {d} \neq \tilde {x} ^ {d}} \phi_ {t} (x ^ {d} | \tilde {\boldsymbol {x}} ^ {1: D}, \boldsymbol {x} _ {0} ^ {1: D}) \mathcal {Z} ^ {t} (\tilde {\boldsymbol {x}} ^ {1: D / d} \circ x ^ {d}) \log \left(\hat {R} _ {t} ^ {\theta   d} (\tilde {\boldsymbol {x}} ^ {1: D}, x ^ {d})\right) \right] \\ \end{array}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/227fe8170627b2ac18d274a6951bb266385e20b2abf44ed01b32c4ee42e8a4ae.jpg)


# C.4 One Forward Pass

To evaluate the $\mathcal { L } _ { \mathrm { C T } }$ objective, we naively need to perform two forward passes of the denoising network: $p _ { 0 | t } ^ { \theta } ( x _ { 0 } | x )$ to calculate $\hat { R } _ { t } ^ { \theta } ( x , x ^ { \prime } )$ and $p _ { 0 | t } ^ { \theta } ( x _ { 0 } | \tilde { x } )$ to calculate $\hat { R } _ { t } ^ { \theta } ( \tilde { x } , x )$ . This is wasteful because x˜ is created from x by applying a single forward transition which on multi-dimensional problems means x˜ differs from x in only a single dimension. To exploit the fact that x˜ and x are very similar, we approximate the sample $x \sim q _ { t } ( x )$ with the sample $\begin{array} { r } { \tilde { x } \sim \sum _ { x } q _ { t } ( x ) r _ { t } ( \tilde { x } | x ) } \end{array}$ . This gives the more efficient objective, 

$$
\mathcal {L} _ {\mathrm{eCT}} (\theta) = T \mathbb {E} _ {t \sim \mathcal {U} (0, T) q _ {t} (x) r _ {t} (\tilde {x} | x)} \left[ \left\{\sum_ {x ^ {\prime} \neq \tilde {x}} \hat {R} _ {t} ^ {\theta} (\tilde {x}, x ^ {\prime}) \right\} - \mathcal {Z} ^ {t} (x) \log \left(\hat {R} _ {t} ^ {\theta} (\tilde {x}, x)\right) \right] + C
$$


Table 3: Metrics on the monophonic music dataset comparing training with the efficient $\mathcal { L } _ { \mathrm { e C T } }$ objective vs the original $\mathcal { L } _ { \mathrm { C T } }$ objective. We compute these over the test set showing mean±std with respect to 5 samples for each test song.


<table><tr><td>Model</td><td>Hellinger Distance</td><td>Proportion of Outliers</td></tr><tr><td><eq>\tau</eq>LDR-0 Uniform <eq>\mathcal{L}_{\text{eCT}}</eq></td><td><eq>0.3765 \pm 0.0013</eq></td><td><eq>0.1106 \pm 0.0010</eq></td></tr><tr><td><eq>\tau</eq>LDR-0 Uniform <eq>\mathcal{L}_{\text{CT}}</eq></td><td><eq>0.3797 \pm 0.0009</eq></td><td><eq>0.1128 \pm 0.0007</eq></td></tr></table>

The approximation is valid because $q _ { t } ( x )$ and $\begin{array} { r } { \sum _ { x } q _ { t } ( x ) r _ { t } ( \tilde { x } | x ) } \end{array}$ are very similar distributions, as we now show. 

$$
\begin{array}{l} \sum_ {x} q _ {t} (x) r _ {t} (\tilde {x} | x) = \sum_ {x} q _ {t} (x) (1 - \delta_ {x, \tilde {x}}) \frac {R _ {t} (x , \tilde {x})}{\sum_ {x ^ {\prime} \neq x} R _ {t} (x , x ^ {\prime})} \\ \propto - q _ {t} (\tilde {x}) R _ {t} (\tilde {x}, \tilde {x}) + \sum_ {x} q _ {t} (x) R _ {t} (x, \tilde {x}) \\ = q _ {t} (\tilde {x}) \sum_ {x ^ {\prime} \neq \tilde {x}} R _ {t} (\tilde {x}, x ^ {\prime}) + \partial_ {t} q _ {t} (\tilde {x}) \\ \propto q _ {t} (\tilde {x}) + \frac {1}{\sum_ {x ^ {\prime} \neq x} R _ {t} (\tilde {x} , x ^ {\prime})} \partial_ {t} q _ {t} (\tilde {x}) \\ = q _ {t} (\tilde {x}) + \delta t \partial_ {t} q _ {t} (\tilde {x}) \\ \end{array}
$$

where on the third line we have used the Kolmogorov forward equation and defined $\begin{array} { r c l } { \delta _ { t } } & { = } & { 1 / \sum _ { x ^ { \prime } \neq x } R _ { t } ( \tilde { x } , x ^ { \prime } ) } \end{array}$ . The distribution $\begin{array} { r } { \sum _ { x } q _ { t } ( x ) \overline { { r } } _ { t } ( \tilde { x } | x ) } \end{array}$ is therefore $q _ { t + \delta t } ( \tilde { x } )$ approximated using a first-order Taylor expansion around $q _ { t } ( \tilde { x } )$ . We notice that δt is the average time to the next transition at time t. δt can be calculated for the practical settings we consider, its varies between $2 \times 1 0 ^ { - 6 } T$ and $2 \times 1 0 ^ { - 8 } T$ in the image modelling task and is $1 \times 1 0 ^ { - 3 } T$ in the monophonic music task. 

We perform an ablation experiment comparing between training with $\mathcal { L } _ { \mathrm { e C T } }$ and $\mathcal { L } _ { \mathrm { C T } }$ on the monophonic music dataset, the results are shown in Table 3. We find that we gain a small boost in performance when using the more efficient $\mathcal { L } _ { \mathrm { e C T } }$ objective alongside the improved efficiency. We hypothesize that this is because of a slight reduction in variance for the $\mathcal { L } _ { \mathrm { e C T } }$ objective due to increased negative correlation between the two terms in the objective when x˜ is shared between them. 

# D Direct Denoising Model Supervision

Following [8], we can introduce direct $p _ { 0 | t } ^ { \theta }$ supervision into the optimization objective which has been found empirically to improve performance. We first contextualize the change by expressing $\mathcal { L } _ { \mathrm { C T } }$ with the dependence on $p _ { 0 | t } ^ { \theta }$ made explicit. 

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{CT}} = T \mathbb {E} _ {t \sim \mathcal {U} (0, T) q _ {t} (x) r _ {t} (\tilde {x} | x)} \left[ \left\{\sum_ {x ^ {\prime} \neq x} R _ {t} \left(x ^ {\prime}, x\right) \sum_ {x _ {0}} \frac {q _ {t | 0} \left(x ^ {\prime} \mid x _ {0}\right)}{q _ {t | 0} (x \mid x _ {0})} p _ {0 | t} ^ {\theta} \left(x _ {0} \mid x\right) \right\} \right. \\ \left. - \mathcal {Z} ^ {t} (x) \log \left(R _ {t} (x, \tilde {x}) \sum_ {x _ {0}} \frac {q _ {t | 0} (x | x _ {0})}{q _ {t | 0} (\tilde {x} | x _ {0})} p _ {0 | t} ^ {\theta} (x _ {0} | \tilde {x})\right) \right] + C \\ \end{array}
$$

The signal for $p _ { 0 | t } ^ { \theta } ( x _ { 0 } | x )$ comes through a sum over $x _ { 0 }$ weighted by the ratio $\frac { q _ { t \mid 0 } ( x \mid x _ { 0 } ) } { q _ { t \mid 0 } ( \tilde { x } \mid x _ { 0 } ) }$ . We can also provide a direct denoising signal by predicting the clean datapoint $x _ { 0 }$ from the corrupted version x and using the negative log-likelihood loss. 

$$
L _ {l l} (\theta) = T \mathbb {E} _ {t \sim \mathcal {U} (0, T) p _ {\mathrm{data}} (x _ {0}) q _ {t | 0} (x | x _ {0})} \left[ - \log p _ {0 | t} ^ {\theta} (x _ {0} | x) \right]
$$

Proposition 8. The true denoising distribution, $q _ { 0 | t } ,$ , minimizes $L _ { l l }$ 

Proof. 

$$
\begin{array}{l} T \mathbb {E} _ {t \sim \mathcal {U} (0, T) q _ {t} (x)} \left[ \mathrm{KL} \left(q _ {0 | t} (x _ {0} | x) | | p _ {0 | t} ^ {\theta} (x _ {0} | x)\right) \right] \\ = T \mathbb {E} _ {t \sim \mathcal {U} (0, T) p _ {\mathrm{data}} (x _ {0}) q _ {t | 0} (x | x _ {0})} \left[ - \log p _ {0 | t} ^ {\theta} (x _ {0} | x) \right] + C \\ \end{array}
$$

where C is a constant independent of θ. Therefore, minimizing $L _ { l l }$ is equivalent to minimizing the KL divergence between $q _ { 0 | t }$ and $p _ { 0 | t } ^ { \theta } .$ , which is minimized when $p _ { 0 | t } ^ { \theta } = q _ { 0 | t }$ . □ 

If we obtain the true denoising distribution, $p _ { 0 | t } ^ { \theta } = q _ { 0 | t }$ , then we will have the true reverse rate, $\hat { R } _ { t }$ [8] find that optimizing with an objective combining $L _ { l l }$ and $\mathcal { L } _ { \mathrm { D T } }$ performs best, which we can also do in continuous time 

$$
\min _ {\theta} \mathcal {L} _ {\mathrm{CT}} (\theta) + \lambda L _ {l l} (\theta)
$$

where λ is a hyperparameter. In [8], it was found that training with $L _ { l l }$ alone resulted in poorer performance than when the ELBO was included in the loss. We provide a theoretical hypothesis as to why this may be the case here. We show that minimizing $L _ { l l }$ is equivalent to minimizing an upper bound on the negative ELBO in discrete time and thus by training with $L _ { l l }$ we are simply minimizing a looser bound on the negative model log-likelihood than if we were to use the negative ELBO directly. 

Proposition 9. Minimizing the sum of negative log-likelihoods 

$$
\sum_ {k = 0} ^ {K - 1} \mathbb {E} _ {p _ {\text { data }} (x _ {0}) q _ {k + 1 | 0} (x _ {k + 1} | x _ {0})} \left[ - \log p _ {0 | k + 1} ^ {\theta} (x _ {0} | x _ {k + 1}) \right]
$$

is equivalent to minimizing an upper bound on the negative ELBO. 

Proof. 

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{DT}} (\theta) = \mathbb {E} _ {p _ {\text {data}} (x _ {0})} \left[ \mathrm{KL} \left(q _ {K | 0} \left(x _ {K} \mid x _ {0}\right) \mid \mid p _ {\text {ref}} \left(x _ {K}\right)\right) - \mathbb {E} _ {q _ {1 | 0} \left(x _ {1} \mid x _ {0}\right)} \left[ \log p _ {0 | 1} ^ {\theta} \left(x _ {0} \mid x _ {1}\right) \right] \right. \\ + \sum_ {k = 1} ^ {K - 1} \mathbb {E} _ {q _ {k + 1 | 0} (x _ {k + 1} | x _ {0})} \left[ \mathrm{KL} (q _ {k | k + 1, 0} (x _ {k} | x _ {k + 1}, x _ {0}) | | p _ {k | k + 1} ^ {\theta} (x _ {k} | x _ {k + 1})) \right] \\ \end{array}
$$

Consider one term from the sum 

$$
\begin{array}{l} L _ {k} = \mathbb {E} _ {p _ {\mathrm{data}} (x _ {0}) q _ {k + 1 | 0} (x _ {k + 1} | x _ {0})} \left[ \mathrm{KL} (q _ {k | k + 1, 0} (x _ {k} | x _ {k + 1}, x _ {0}) | | p _ {k | k + 1} ^ {\theta} (x _ {k} | x _ {k + 1}) \right] \\ = \mathbb {E} _ {q _ {k + 1} (x _ {k + 1}) q _ {k | k + 1} (x _ {k} | x _ {k + 1})} \left[ - \log p _ {k | k + 1} ^ {\theta} (x _ {k} | x _ {k + 1}) \right] \\ + \mathbb {E} _ {p _ {\mathrm{data}} (x _ {0}) q _ {k + 1 | 0} (x _ {k + 1} | x _ {0}) q _ {k | k + 1, 0} (x _ {k} | x _ {k + 1}, x _ {0})} \left[ \log q _ {k | k + 1, 0} \bigl (x _ {k} | x _ {k + 1}, x _ {0} \bigr) \right] \\ \end{array}
$$

Now, 

$$
\begin{array}{l} \mathbb {E} _ {q _ {k + 1} (x _ {k + 1}) q _ {k | k + 1} (x _ {k} | x _ {k + 1})} \left[ - \log p _ {k | k + 1} ^ {\theta} (x _ {k} | x _ {k + 1}) \right] \\ = \mathbb {E} _ {q _ {k + 1} (x _ {k + 1}) q _ {k | k + 1} (x _ {k} | x _ {k + 1})} \left[ - \log \sum_ {\tilde {x} _ {0}} q (x _ {k} | \tilde {x} _ {0}, x _ {k + 1}) p _ {0 | k + 1} ^ {\theta} (\tilde {x} _ {0} | x _ {k + 1}) \right] \\ = \mathbb {E} _ {q _ {k + 1} (x _ {k + 1}) q _ {k | k + 1} (x _ {k} | x _ {k + 1})} \left[ - \log \sum_ {\tilde {x} _ {0}} \frac {q _ {0 | k} (\tilde {x} _ {0} | x _ {k}) q _ {k | k + 1} (x _ {k} | x _ {k + 1})}{q _ {0 | k + 1} (\tilde {x} _ {0} | x _ {k + 1})} p _ {0 | k + 1} ^ {\theta} (\tilde {x} _ {0} | x _ {k + 1}) \right] \\ \leq \mathbb {E} _ {q _ {k + 1} (x _ {k + 1}) q _ {k | k + 1} (x _ {k} | x _ {k + 1}) q _ {0 | k} (\tilde {x} _ {0} | x _ {k})} \left[ - \log \frac {q _ {k | k + 1} (x _ {k} | x _ {k + 1})}{q _ {0 | k + 1} (\tilde {x} _ {0} | x _ {k + 1})} p _ {0 | k + 1} ^ {\theta} (\tilde {x} _ {0} | x _ {k + 1}) \right] \\ = \mathbb {E} _ {p _ {\mathrm{data}} (x _ {0}) q _ {k + 1 | 0} (x _ {k + 1} | x _ {0})} \left[ - \log p _ {0 | k + 1} ^ {\theta} (x _ {0} | x _ {k + 1}) \right] \\ + \mathbb {E} _ {p _ {\mathrm{data}} (x _ {0}) q _ {k | 0} (x _ {k} | x _ {0}) q _ {k + 1 | k} (x _ {k + 1} | x _ {k})} \left[ - \log \frac {q _ {k | k + 1} (x _ {k} | x _ {k + 1})}{q _ {0 | k + 1} (x _ {0} | x _ {k + 1})} \right] \\ \end{array}
$$

Therefore, 

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{DT}} \leq \sum_ {k = 0} ^ {K - 1} \left\{\mathbb {E} _ {p _ {\text { data }} (x _ {0}) q _ {k + 1 | 0} (x _ {k + 1} | x _ {0})} \left[ - \log p _ {0 | k + 1} ^ {\theta} (x _ {0} | x _ {k + 1}) \right] \right\} \\ + \mathbb {E} _ {p _ {\mathrm{data}} (x _ {0})} \left[ \mathrm{KL} (q _ {K | 0} (x _ {K} | x _ {0}) | | p _ {K} (x _ {K})) \right] \\ + \sum_ {k = 1} ^ {K - 1} \left\{\mathbb {E} _ {p _ {\text { data }} (x _ {0}) q _ {k + 1 | 0} (x _ {k + 1} | x _ {0}) q _ {k | k + 1, 0} (x _ {k} | x _ {k + 1}, x _ {0})} \left[ \log q _ {k | k + 1, 0} (x _ {k} | x _ {k + 1}, x _ {0}) \right] \right. \\ \left. + \mathbb {E} _ {p _ {\mathrm{data}} (x _ {0}) q _ {k | 0} (x _ {k} | x _ {0}) q _ {k + 1 | k} (x _ {k + 1} | x _ {k})} \left[ - \log \frac {q _ {k | k + 1} (x _ {k} | x _ {k + 1})}{q _ {0 | k + 1} (x _ {0} | x _ {k + 1})} \right] \right\} \\ \end{array}
$$

We can see that only the first term depends on $\theta .$ . 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/447fa6294c2b6eb439a0e0224cf0415cacdaaf986cdfe2bb59ecb81ec3bbc90f.jpg)


# E Choice of Forward Process

We need to choose the structure of $R _ { t }$ such that we can analytically obtain $q _ { t | 0 }$ marginals to enable efficient training. 

Proposition 10. ${ \cal I } f R _ { t }$ and $R _ { t ^ { \prime } }$ commute for all t, t0 then $\begin{array} { r } { q _ { t | 0 } ( x = j | x _ { 0 } = i ) = \left( e x p \big [ \int _ { 0 } ^ { t } R _ { s } d s \big ] \right) _ { i j } } \end{array}$ where exp here is understood to be the matrix exponential function. 

Proof. Let $( P _ { t } ) _ { i j } = q _ { t | 0 } ( x = j | x _ { 0 } = i )$ . We show that $\begin{array} { r } { P _ { t } = \exp \left( \int _ { 0 } ^ { t } R _ { s } d s \right) } \end{array}$ is a solution to the Kolmogorov forward equation, which in matrix form reads, $\partial _ { t } P _ { t } \overset { \cdot } { = } P _ { t } R _ { t }$ . Writing the matrix exponential in sum form 

$$
\begin{array}{l} P _ {t} = \sum_ {k = 0} ^ {\infty} \frac {1}{k !} \left(\int_ {0} ^ {t} R _ {s} d s\right) ^ {k} \\ = \operatorname{Id} + \int_ {0} ^ {t} R _ {s} d s + \frac {1}{2 !} \left(\int_ {0} ^ {t} R _ {s} d s\right) ^ {2} + \frac {1}{3 !} \left(\int_ {0} ^ {t} R _ {s} d s\right) ^ {3} + \dots \\ \end{array}
$$

Now, differentiating and using the fact that $R _ { t } , R _ { t ^ { \prime } }$ commute. 

$$
\begin{array}{l} \partial_ {t} P _ {t} = R _ {t} + \int_ {0} ^ {t} R _ {s} d s R _ {t} + \frac {1}{2 !} \left(\int_ {0} ^ {t} R _ {s} d s\right) ^ {2} R _ {t} + \dots \\ = \left\{\sum_ {k = 0} ^ {\infty} \frac {1}{k !} \left(\int_ {0} ^ {t} R _ {s} d s\right) ^ {k} \right\} R _ {t} \\ = P _ {t} R _ {t} \\ \end{array}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/0d5cda980b61deca9d4c4a5b8276fbb71e1e8076cd3edb51f2cc8a11667b195d.jpg)


As stated in the main text, we achieve the commutative property by selecting $R _ { t } = \beta ( t ) R _ { b }$ where $\beta ( t )$ is a time dependent scalar and $R _ { b }$ is a constant base matrix. We can utilize the eigendecomposition 

of ${ \cal R } _ { b } = Q \Lambda Q ^ { - 1 }$ to efficiently calculate $P _ { t }$ 

$$
\begin{array}{l} P _ {t} = \exp \left(\int_ {0} ^ {t} \beta (s) R _ {b} d s\right) \\ = \sum_ {k = 0} ^ {\infty} \frac {1}{k !} \left(\int_ {0} ^ {t} \beta (s) R _ {b} d s\right) ^ {k} \\ = \sum_ {k = 0} ^ {\infty} \frac {1}{k !} \left(Q \Lambda Q ^ {- 1} \int_ {0} ^ {t} \beta (s) d s\right) ^ {k} \\ = \sum_ {k = 0} ^ {\infty} \frac {1}{k !} Q \left(\Lambda \int_ {0} ^ {t} \beta (s) d s\right) ^ {k} Q ^ {- 1} \\ = Q \left\{\sum_ {k = 0} ^ {\infty} \frac {1}{k !} \left(\Lambda \int_ {0} ^ {t} \beta (s) d s\right) ^ {k} \right\} Q ^ {- 1} \\ = Q \exp \left[ \Lambda \int_ {0} ^ {t} \beta (s) d s \right] Q ^ {- 1} \\ \end{array}
$$

Since Λ is a diagonal matrix, the matrix exponential coincides with the element wise exponential making the final expression tractable to compute. We choose $\beta ( t ) = a b ^ { t }$ log b because this makes the integral which dictates the variance of $q _ { t | 0 }$ have a simple form $\begin{array} { r } { \int _ { 0 } ^ { t } \beta ( s ) d s = a b ^ { t } - a } \end{array}$ . 

For categorical problems, we found a uniform rate matrix works well, $R _ { t } = \beta \mathbb { 1 } \mathbb { 1 } ^ { T } - \beta S \mathrm { I d }$ . This is directly analogous to the discrete time uniform transition matrix: $P = \alpha \mathbb { 1 } \mathbb { 1 } ^ { T } + ( 1 - S \alpha ) ^ { - }$ Id with α depending on the time discretization used. Indeed, if one calculates the corresponding discrete transition matrix for the uniform $R _ { t }$ rate through the matrix exponential, the uniform transition matrix is obtained. Another categorical corruption process is the absorbing state process. In discrete time, the transition matrix is given by $P = \mathbf { \bar { \alpha } } \alpha \mathbb { 1 } \mathbf { e } _ { * } ^ { T } + ( 1 - \alpha ) \mathrm { I d }$ where $\mathbf { e } _ { * }$ ∗ is the one-hot encoding of the absorbing state. The corresponding absorbing state continuous time process has transition rate matrix: $R _ { t } = \beta \bar { \bf 1 6 } _ { * } ^ { T } - \beta \mathrm { I d }$ . The correspondence for more complex transition matrices $\mathrm { e . g . }$ . the Discretized Gaussian matrix in [8] is much harder to find analytically especially if the time inhomogeneous case is considered. For datasets with an ordinal structure, we construct a new rate matrix that maintains a bias towards nearby states using a similar approach as that taken by [8] to construct the Discretized Gaussian matrix. 

We construct this matrix by first picking a desired stationary distribution, $p _ { \mathrm { r e f } } .$ , and then filling in matrix entries such that we encourage transitions to nearby states whilst keeping $p _ { \mathrm { r e f } }$ as our stationary distribution. Specifically, we let $p _ { \mathrm { r e f } }$ be a discretized Gaussian over the state space, i.e. 

$$
p _ {\text { ref }} (x) \propto \exp \left[ - \frac {(x - \mu_ {0}) ^ {2}}{2 \sigma_ {0} ^ {2}} \right]
$$

To find a condition on the rate such that this is the case, recall the Kolmogorov differential equation for the marginals 

$$
\partial_ {t} q _ {t} (x) = \sum_ {\tilde {x}} q _ {t} (\tilde {x}) R _ {b} (\tilde {x}, x)
$$

Now, consider a rate that is in detailed balance with $p _ { \mathrm { r e f } }$ 

$$
p _ {\text { ref }} (\tilde {x}) R _ {b} (\tilde {x}, x) = p _ {\text { ref }} (x) R _ {b} (x, \tilde {x})
$$

Substituting this rate into the Kolmogorov equation, we see that $p _ { \mathrm { r e f } }$ is the stationary distribution 

$$
\begin{array}{l} \partial_ {t} p _ {\text { ref }} (x) = \sum_ {\tilde {x}} p _ {\text { ref }} (\tilde {x}) R _ {b} (\tilde {x}, x) \\ = \sum_ {\tilde {x}} p _ {\text { ref }} (x) R _ {b} (x, \tilde {x}) \\ = p _ {\text { ref }} (x) \sum_ {\tilde {x}} R _ {b} (x, \tilde {x}) \\ = 0 \\ \end{array}
$$

where the last line follows from the fact that the row sum of a rate matrix is zero. Note that any $R _ { t } = \beta ( t ) R _ { b }$ will also have this stationary distribution as the multiplication by $\beta ( t )$ can be seen as just a scaling of the time axis. From the detailed balance equation, we gain a condition on $R _ { b }$ such that our desired $p _ { \mathrm { r e f } }$ is the stationary distribution 

$$
\frac {R _ {b} (\tilde {x} , x)}{R _ {b} (x , \tilde {x})} = \frac {p _ {\text {ref}} (x)}{p _ {\text {ref}} (\tilde {x})} = \exp \left[ \frac {(\tilde {x} - \mu_ {0}) ^ {2}}{2 \sigma_ {0} ^ {2}} - \frac {(x - \mu_ {0}) ^ {2}}{2 \sigma_ {0} ^ {2}} \right]
$$

This gives constraints on diagonal elements within $R _ { b }$ but does not fully define the entire matrix. To do this, we first make the assumption that $\mu$ is selected to be at the center of the state space. Then we set off diagonal terms to the right of the diagonal in the top half of the rate matrix and off diagonal terms to the left of the diagonal in the bottom half to be 1. Finally, progressing in from the top and bottom of the rate matrix we make definitions of rate matrix values that have not already been defined by the detailed balance condition. For clarity, we provide a pictorial representation of this scheme for an $8 \times 8$ rate matrix below 

$$
\left[ \begin{array}{c c c c c c c c} \cdot & 1 & \square & \square & \square & \square & \square & \square \\ \triangle & \cdot & 1 & \square & \square & \square & \square & \triangle \\ \triangle & \triangle & \cdot & 1 & \square & \square & \triangle & \triangle \\ \triangle & \triangle & \triangle & \cdot & 1 & \triangle & \triangle & \triangle \\ \triangle & \triangle & \triangle & \triangle & \cdot & \triangle & \triangle & \triangle \\ \triangle & \triangle & \triangle & \square & 1 & \cdot & \triangle & \triangle \\ \triangle & \triangle & \square & \square & \square & 1 & \cdot & \triangle \\ \triangle & \square & \square & \square & \square & \square & 1 & \cdot \end{array} \right]
$$

where  represents a value we will define, $\triangle$ represents a value that is defined relative to another entry through the detailed balance condition and · is a diagonal entry that is equal to the negative off diagonal row sum. We could define  values to be 0 to gain a sparse rate matrix, however, we found in early experiments that allowing transitions to further away states greatly reduces the mixing time and gives better performance. We define  in each row similarly, by setting it equal to $\exp [ - \breve { i } ^ { 2 } / \sigma _ { r } ^ { 2 } ]$ where i is the distance away from the $\cdot _ { 1 } \cdot$ value in that row and $\sigma _ { r }$ is a hyperparameter defining the length scale in state space of a typical transition. This biases our forward process to make transitions between nearby states, at a length scale of $\sigma _ { r }$ . 

# F CTMC Simulation

# F.1 Exact CTMC and Tau-Leaping

In this section, we first describe exact CTMC simulation before giving an algorithmic description of tau-leaping. 

When a CTMC has a time-homogeneous rate matrix, we can use Gillespie’s Algorithm [21, 22, 23] to exactly simulate it. This algorithm is based on the jump chain/holding time definition of the CTMC. It repeats the following two steps: 

• Draw a holding time from an exponential distribution with mean $- 1 / R ( x , x )$ and wait in the current state x for that amount of time. 

• Sample the next state from r(˜x|x) = (1 − δx,x˜) P R(x,x˜)x06=x R(x,x0) $\begin{array} { r } { r ( \tilde { x } | x ) = ( 1 - \delta _ { x , \tilde { x } } ) \frac { R ( x , \tilde { x } ) } { \sum _ { x ^ { \prime } \ne x } R ( x , x ^ { \prime } ) } } \end{array}$ 

This Algorithm can be adjusted for the case when we have a time-inhomogeneous rate matrix using the modified next reaction method [33]. However, both algorithms still step through each transition in the CTMC individually and are thus unsuitable in our case because only one dimension would change for each simulation step making it very computationally expensive to produce a sample. Instead we use tau-leaping that allows multiple dimensions to change in a single simulation step. We detail this method in Algorithm 1. 

# F.2 Predictor-Corrector Discussion

In this section we compare predictor-corrector sampling schemes as applied to continuous state spaces and discrete state spaces. 


Algorithm 1: Generative Reverse Process Simulation with Tau-Leaping


$t \leftarrow T$ $\boldsymbol{x}_{t}^{1:D} \sim p_{\text{ref}}(\boldsymbol{x}_{T}^{1:D})$ while $t > 0$ do

    Compute $p_{0|t}^{\theta}(x_{0}^{d}|\boldsymbol{x}_{t}^{1:D}), d = 1, \ldots, D$ with one forward pass of the denoising network

    for $d = 1, \ldots, D$ do

    for $s = 1, \ldots, S \backslash x_{t}^{d}$ do $\hat{R}_{t}^{\theta d}(\boldsymbol{x}_{t}^{1:D}, s) \leftarrow R_{t}^{d}(s, x_{t}^{d}) \sum_{x_{0}^{d}} p_{0|t}^{\theta}(x_{0}^{d}|\boldsymbol{x}_{t}^{1:D}) \frac{q_{t|0}(s|x_{0}^{d})}{q_{t|0}(x_{t}^{d}|x_{0}^{d})}$ $P_{ds} \leftarrow \text{Poisson}\left( \tau \hat{R}_{t}^{\theta d}(\boldsymbol{x}_{t}^{1:D}, s) \right)$ end

end

for $d = 1, \ldots, D$ do

    if data is categorical AND $\sum_{s=1}^{S} P_{ds} > 1$ then $x_{t-\tau}^{d} \leftarrow x_{t}^{d} // reject change$ else $x_{t-\tau}^{d} \leftarrow x_{t}^{d} + \sum_{s=1}^{S} P_{ds} \times (s - x_{t}^{d})$ end

end $\boldsymbol{x}_{t-\tau}^{1:D} \leftarrow \text{Clamp}(\boldsymbol{x}_{t-\tau}^{1:D}, \min = 1, \max = S)$ $t \leftarrow t - \tau$ end 

The predictor-corrector scheme in continuous state spaces was introduced in [4]. It consists of alternating between a predictor step and a corrector step: 

where $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ is the state at sampling step $i , s _ { \theta }$ is the learned score model approximating $\nabla _ { \pmb { x } } \log q _ { t } ( \pmb { x } )$ and $\gamma _ { i } , \epsilon _ { i }$ are the step sizes for the predictor and corrector respectively. We see that both take similar√ forms, except the corrector adds in a factor $\sqrt { 2 }$ more Gaussian noise during the update step. 

In discrete state spaces, rather than sampling using gradient guided stochastic steps as in the continuous state space case, we sample by simulating CTMCs with defined rates. When we take a predictor step, we simulate using $\hat { R } _ { t } ^ { \theta }$ and when we take a corrector step we simulate using $R _ { t } ^ { c \theta } = \hat { R } _ { t } ^ { \theta } + R _ { t }$ . If we simulate the CTMC exactly, we have seen in the previous section that this amounts to sampling next states from the categorical distribution defined by normalizing the row of the rate matrix corresponding to the current state. Therefore, corrector sampling can be seen as sampling from a slightly noisier categorical distribution defined through $R _ { t } ^ { c \theta }$ as compared to the predictor categorical distribution defined through $\hat { R } _ { t } ^ { \theta }$ . This is analogous to the increased Gaussian noise applied during a corrector step in continuous state spaces. 

Adding corrector steps brings the marginal of the samples closer to $q _ { t } ( \pmb { x } )$ and continued application of the corrector will further explore the domain of $q _ { t } ( \pmb { x } )$ . In previous work on continuous state predictor-corrector methods, the number of corrector steps has been small (e.g. 1 or 2 corrector steps per predictor step) or indeed the corrector steps have been removed altogether. In this work we have found that using up to 10 corrector steps per predictor steps can be beneficial during certain regions of the reverse generative process. Additionally, in continuous state spaces, it has been observed that too many corrector steps can result in unwanted noise in the generated data [34]. 

We hypothesize that corrector steps are better utilized in discrete state spaces to explore the domain of $q _ { t } ( \pmb { x } )$ than in continuous state spaces. This is because, the corrector update is defined largely through the reverse rate itself, $\hat { R } _ { t } ^ { \theta }$ , just with the categorical probabilities being annealed slightly more towards uniform through the addition of the forward rate $R _ { t }$ . This may be a more effective update than simply adding extra Gaussian noise in the continuous state space case. Furthermore, the denoising model in continuous state spaces can be seen as outputting a point estimate of $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ of dimension $D .$ However, in discrete state spaces, the denoising model outputs a categorical distribution over every dimension (output dimension $D \times S )$ allowing it to express some uncertainty information in the $\scriptstyle { \pmb x } _ { 0 }$ prediction, albeit with conditional independence between the dimensions. Adding corrector steps in discrete state spaces would then allow information to mix between dimensions for the current time step, exploring modes of $q _ { t } ( \pmb { x } )$ . 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/503b9127fe8a0d927d905452c6f052e3457fdc34d0081e4dd10582723cdb7621.jpg)



Figure 7: Progression of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ for $t = 0 . 4$ by repeated application of corrector steps. In each pair of rows, the top row is $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ whilst the bottom row is the x0 prediction made by $p _ { 0 | t } ^ { \theta } ( \mathbf { \hat { x } } _ { 0 } | \mathbf { x } _ { t } )$ (argmax of the categorical probabilities in each dimension). Each column represents an additional 100 corrector steps.


We explore this idea on the image modelling task in Figure 7. We run the reverse generative process until time $t = 0 . 4$ at which point we hold the time constant and apply 1000 corrector steps. We see that the resulting progression of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ states explores potential local modes of $q _ { t } ( \pmb { x } )$ in the local region of image space. 

# G Implicit Dimensional Assumptions Made in Discrete Time

In discrete time, the parametric reverse kernel, $p _ { k | k + 1 } ^ { \theta } ,$ is commonly defined through a denoising model $p _ { 0 | k + 1 } ^ { \theta }$ . Here, we examine this definition in the multi-dimensional case where the forward process factorizes, as in Appendix C.3 and previous discrete time work [8]. We begin by writing the true full dimensional reverse kernel, $q _ { k | k + 1 }$ , in terms of the true denoising distribution, $q _ { 0 | k + 1 }$ . 

$$
\begin{array}{l} q _ {k \mid k + 1} (\boldsymbol {x} _ {k} ^ {1: D} | \boldsymbol {x} _ {k + 1} ^ {1: D}) = \prod_ {d = 1} ^ {D} q _ {k \mid k + 1} (x _ {k} ^ {d} | \boldsymbol {x} _ {k} ^ {1: d - 1}, \boldsymbol {x} _ {k + 1} ^ {1: D}) \\ = \prod_ {d = 1} ^ {D} \sum_ {x _ {0} ^ {d}} q _ {k, 0 | k + 1} (x _ {k} ^ {d}, x _ {0} ^ {d} | \boldsymbol {x} _ {k} ^ {1: d - 1}, \boldsymbol {x} _ {k + 1} ^ {1: D}) \\ = \prod_ {d = 1} ^ {D} \sum_ {x _ {0} ^ {d}} q _ {0 | k + 1} (x _ {0} ^ {d} | \pmb {x} _ {k} ^ {1: d - 1}, \pmb {x} _ {k + 1} ^ {1: D}) q _ {k | 0, k + 1} (x _ {k} ^ {d} | x _ {0} ^ {d}, x _ {k + 1} ^ {d}) \\ \end{array}
$$

where on the final line we have used the fact that the forward process is independent across dimensions. To create our approximate reverse kernel, $p _ { k | k + 1 } ^ { \theta }$ , we approximate $q _ { 0 | k + 1 } \big ( x _ { 0 } ^ { d } | \mathbf { x } _ { k } ^ { 1 : d - 1 } , \mathbf { x } _ { k + 1 } ^ { 1 : D } \big )$ with 

$$
p _ {0 | k + 1} ^ {\theta} (x _ {0} ^ {d} | \boldsymbol {x} _ {k + 1} ^ {1: D}),
$$

$$
p _ {k | k + 1} ^ {\theta} (\boldsymbol {x} _ {k} ^ {1: D} | \boldsymbol {x} _ {k + 1} ^ {1: D}) = \prod_ {d = 1} ^ {D} \sum_ {x _ {0} ^ {d}} p _ {0 | k + 1} ^ {\theta} (x _ {0} ^ {d} | \boldsymbol {x} _ {k + 1} ^ {1: D}) q _ {k | 0, k + 1} (x _ {k} ^ {d} | x _ {0} ^ {d}, x _ {k + 1} ^ {d})
$$

We throw away the extra x1:k $\pmb { x } _ { k } ^ { 1 : d - 1 }$ conditioning because we use a non-autoregressive model that takes in $\pmb { x } _ { k + 1 } ^ { 1 : D }$ and in a single forward pass gives conditionally independent probabilities over $x _ { 0 } ^ { d } .$ $d = 1 , \dotsc , \bar { D }$ . For finite $\check { K }$ , this approximation can never match the true kernel because we are not conditioning on all relevant information. Of course, as K gets larger, this approximation becomes more accurate. Since we operate in the continuous regime, we do not have to make this approximation because the conditionally independent denoising model, $q _ { 0 | t } ( x _ { 0 } ^ { d } | \mathbf { x } ^ { 1 : D } )$ , appears directly in our reverse rate, $\hat { R } _ { t } ^ { 1 : D }$ , when we factorize the forward process (see Proposition 3). 

# H Experimental Details

In this section, we provide additional details for the experiments we performed applying our method to practical problems. The code for our models is available at https://github.com/andrew-cr/tauLDR. Before describing the specifics for each experiment, we first explain the implementation details common to all. 

When we evaluate the objective $\mathcal { L } _ { \mathrm { C T } }$ on each minibatch of training datapoints, we must sample a time for each from $t \sim \bar { \mathcal { U } } ( 0 , T )$ which represents the point in the forward process which we will noise to. Training instabilities can be found if t is sampled very close to 0 because the reverse rate, $\hat { R } _ { t } ,$ becomes ill-conditioned in this region. This phenomenon is also observed in continuous state space models because the score, $\nabla _ { x } \log { q _ { t } ( x ) }$ , becomes ill-conditioned close to $t = 0$ . The reverse rate and score become ill conditioned close to the start of the forward process because the marginal probability, $q _ { t } ( x )$ , will be highly peaked around the data manifold and log $q _ { t } ( x )$ will explode in regions that are not close to the data. To avoid these issues, a common trick is to set a minimum time such that $t \sim \mathcal { U } ( \epsilon , T )$ .  is set such that the level of noising at $t = \epsilon$ is very small and reverse sampling to this point will produce samples very close to $p _ { \mathrm { d a t a } }$ . In our experiments, we set $\epsilon = 0 . 0 1 T$ . 

During reverse sampling, we use tau-leaping to simulate the reverse process from $t = T$ until $t = \epsilon$ because the reverse rate is not trained for $t < \epsilon .$ This produces a sample close to $p _ { \mathrm { d a t a } }$ . We found improved performance in metrics such as FID if we then complete a final step to remove the small amount of noise that may still be present in the sample. Specifically, we pass the sample through the denoising model $p _ { 0 \mid t } ^ { \theta } ( x _ { 0 } | x _ { t } )$ with $t = \epsilon$ to obtain an output of shape $D \times S$ where D is the dimensionality of the problem. This is a probability distribution over the states for each of the dimensions. We set the value of each dimension to the state with the highest probability. This then produces a sample which has all of the noise removed. 

The specific value of T within our model is arbitrary because the forward process can be scaled in the time axis to provide the same noising process for any T . Therefore, we simply set $T = 1$ . 

# H.1 Demonstrative Example

Our 2d dataset is created by sampling 1M 2d points from a $3 2 \times 3 2$ state space with probability proportional to the pixel values of a $3 2 \times 3 2$ grayscale image of a τ character. 

For our forward process, we use a Gaussian rate (see Appendix E) with stationary distribution standard deviation $\sigma _ { 0 } = 8$ and rate length scale $\sigma _ { r } = 1$ . We use a rate schedule of $\beta ( \dot { t } ) = 5 \times 5 ^ { t } \log ( 5 )$ . 

To represent $p _ { 0 | t } ^ { \theta }$ we use a residual MLP. The architecture consists of an input linear layer to lift the input dimension of 2 to the internal network dimension of 16. Then, there are 2 residual blocks each consisting of: a single hidden layer MLP of hidden dimension 32, a residual connection to the input of the MLP, a layer norm, and finally a FiLM layer [35] modulated by the time embedding. At the output, there is a single linear layer with output size of $2 \times 3 2 = 6 4$ representing state probabilities in each of the 2 dimensions. The time is embedded using the Transformer sinusoidal position embedding [36] creating an embedding of size 32. Then, the embedding is further processed by a single hidden layer MLP with hidden layer size 32 and output size 128. To create the FiLM parameters in each residual block, the time embedding is passed through a linear layer with output of size 32 to provide a multiplicative and additive modulation to the state dimension of 16. We minimize the $\mathcal { L } _ { \mathrm { C T } }$ objective using Adam with a learning rate of 0.0001 and batch size of 32 for 1M steps. 

For the exact simulation we use the next reaction method with modifications for time dependent transition rates [33]. This method steps through each transition in the exact simulation path individually by calculating the time to the next occurrence of each transition type and applying the transition that occurs soonest. Exact algorithmic details can be found in [33]. To calculate the time to the next occurrence for a transition, we need to integrate the reverse rate matrix (eq (13) in [33]). We do this with euler integration with a step size of 0.001. 

# H.2 Image Modeling

We train on the CIFAR10 training dataset that contains 50000 images of dimension $3 \times 3 2 \times 3 2$ . We evaluate the test ELBO on the CIFAR10 test dataset which consists of 10000 images. For the forward noising process, we use the the Gaussian rate (see Appendix E) with stationary distribution standard deviation of $\sigma _ { 0 } = 5 1 2$ and rate length scale $\sigma _ { r } = 6$ . This effectively defines a uniform stationary distribution since the state space is of size 256. We found this performs better than a more concentrated Gaussian. Our $\beta$ schedule is $\beta ( t ) = 3 \times 1 0 0 ^ { t }$ log 100. This was selected in accordance with $\sigma _ { r }$ r such that the overall shape of progression of the $q _ { t | 0 }$ variances approximately matches that of the schedule proposed in [3]. 

Our $p _ { 0 | \cdot } ^ { \theta }$ t model is parameterized with the standard U-net [37] architecture introduced in [3]. The network follows the PixelCNN++ backbone [38] with group normalization layers. There are four feature map resolutions (32 × 32 to $4 \times 4 )$ in the downsampling/upsampling stacks. At each resolution there are two convolutional residual blocks. There is a self-attention block between the residual blocks at the $1 6 \times 1 6$ resolution level [39]. The time is input into the network by first embedding with the Transformer sinusoidal position embedding [36]. This time embedding is passed into each residual block by passing it through a SiLU activation [40] and then a linear layer before adding it onto the hidden state within the residual block between the two convolution operations. 

The original architecture of [3] has an output of dimension $3 \times 3 2 \times 3 2$ as it makes a point prediction of $x _ { 0 }$ given $x _ { t }$ . In order for the model to output probabilities over $x _ { 0 }$ (i.e. an output dimension of $3 \times 3 2 \times 3 2 \times 2 5 6 )$ we make the adjustments suggested in [8]. Specifically, we use their truncated logistic distribution parameterization where the model outputs the mean and log scale of a logistic distribution i.e. an output dimension of $3 \times 3 2 \times 3 2 \times 2$ . The probability for a state is then the integral of this continuous distribution between this state and the next when mapped onto the real line. To impart a residual inductive bias on the output, the mean of the logistic distribution is taken to be tanh $( x _ { t } + \mu ^ { \prime } )$ where $x _ { t }$ is the normalized input into the model and $\mu ^ { \prime }$ is mean outputted from the network. The normalization operation takes the input in the range 0, . . . , 255 and maps it to [−1, 1]. In total, our network has approximately 35.7 million parameters. 

We optimize with the auxiliary objective described in Appendix D with $\lambda = 0 . 0 0 1$ . Within the auxiliary objective, we use the one-forward pass version of the continuous time $\mathrm { E L B O } , \mathcal { L } _ { \mathrm { e C T } }$ . We optimize with Adam for 2M steps with a learning rate of 0.0002 and batch size of 128. We use the standard set of training tricks to improve optimization [3, 4]. Throughout training we maintain an exponential moving average of the parameters with decay factor 0.9999. These average parameters are used during testing. At the start of optimization we use a linear learning rate warm-up for the first 5000 steps. We clip the gradient norm at a norm value of 1.0. We set the dropout rate for the network at 0.1. The skip connections for each residual block are rescaled by a factor of $\scriptstyle { \frac { 1 } { \sqrt { 2 } } }$ The input images have random horizontal flips applied to them during training. 

For sampling in Table 1 we set $\tau = 0 . 0 0 1$ for τ LDR-0 and set $\tau = 0 . 0 0 2$ for τLDR-10. The 10 corrector steps per predictor steps for τ LDR-10 are introduced after $t < 0 . 1 T$ . We found that introducing the corrector steps near the end of the reverse sampling process had the best improvement in sample quality for the smallest increase in computational cost. When performing tau-leaping with the corrector rate, $R _ { t } ^ { c }$ , we have control over what τ we use since we are sampling a different CTMC (with $q _ { t }$ as its stationary distribution) to the original reverse CTMC. We found that setting the corrector rate τ to be 1.5 times the original τ for the reverse CTMC achieves the best performance in this example. 

We train using 4 V100 GPUs on an academic research cluster. To calculate Inception and FID values, we use pytorch-fid [41] and a further development 1. We verified this library produced comparable values to previous work by calculating the Inception and FID scores for the published images from the DDPM [3] method. 

We show a large array of unconditional samples from the τ LDR-10 model in Figure 8. We now also present statistics from the reverse sampling process with standard tau-leaping with $\tau = 0 . 0 0 1$ . Figure 9 shows the proportion of dimensions that transition during a single step of tau-leaping. We see that during the initial stages, every dimensions changes during every tau-leaping step, but nearer the end of the process, more dimensions will have settled in their final positions and the proportion is less. In Figure 10 we show the proportion of dimensions that are clipped due to proposing an out of bounds jump. Overall, the proportion is small. It is largest at the start of the process when we have initially sampled from the approximately uniform $p _ { \mathrm { r e f } }$ and there will be dimensions close to the boundary. As pixel values settle to their final values, the proportion reduces. Figure 11 shows the progression of a selection of dimensions during the reverse sampling process. A similar picture emerges where dimensions eventually settle in a region of the state space. We also note that larger jumps are made in a single tau-leaping step nearer the start of the reverse process and smaller jumps are made nearer the end. 

# H.3 Monophonic Music

We generate our training dataset from the Lakh pianoroll dataset [29, 30] (license CC By 4.0). This dataset consists of 174,154 multitrack pianorolls. We go through all songs and all tracks within each song and select sequences that match the following criteria: they are monophonic (only one note played at a time), there is not a period longer than one bar in which no note is played, there is more than one type of note played in the sequence and finally there is no one note played for more than 50 time steps out of the total 256 time steps. This removes the uninteresting and trivial sequences present within the dataset. We then remove any duplicates in the result. This leaves us with 6000 training examples and 950 testing examples. Each song consists of 256 time steps (16 per bar) and each time step takes one from 129 values i.e. we have D = 256 and $S = 1 2 9$ . This state value represents either a note from 128 options or a rest. We scramble the ordering of this state space when mapping to the integers from 0 to 128. When we input into the denoising network, we input as one-hot 129 dimensional vectors. 

For the forward noising process, we use a uniform rate matrix, $R _ { b } = \mathbb { 1 } \mathbb { 1 } ^ { T } - S \mathrm { I d }$ and set $\beta ( t ) = 0 . 0 3$ . We found a constant in time $\beta ( t )$ was sufficient for this dataset. In our comparison, we used a birth/death rate matrix defined as 

$$
\left[ \begin{array}{c c c c c c c} - \lambda & \lambda & 0 & 0 & \ldots & 0 & 0 \\ \lambda & - 2 \lambda & \lambda & 0 & \ldots & 0 & 0 \\ 0 & \lambda & - 2 \lambda & \lambda & \ldots & 0 & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 0 & 0 & \ldots & \lambda & - \lambda \end{array} \right]
$$

this is the rate matrix for a birth/death process. We set $\lambda = 1$ and $\beta ( t ) = \textstyle { \frac { 1 } { 2 } } \times 1 0 0 0 0 ^ { t } \log$ 10000. These hyperparameters were selected such that the forward process has a steady rate of noising whilst still having $q _ { T }$ very close to $p _ { \mathrm { r e f } }$ . We chose to compare these types of rate matrix because the birth/death rate is inappropriate for this categorical data as adjacent states have no meaning 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/427ebe29eb8dc62205a5834c5189191cc8bcd36e037d79ecc28b724865642574.jpg)



Figure 8: Unconditional CIFAR10 samples from our τ LDR-10 model.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/5435315197f45adbb802b141f6142f70a4d198a9af83f8671ff6f59d7def7ccc.jpg)



Figure 9: Proportion of dimensions that transition during a single step of tau-leaping during the reverse sampling process.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/364b73cc38f4c2a695a6dd8b64d6c79b8ee905b2fecf7d92314ff4ec1960828a.jpg)



Figure 10: Proportion of dimensions that are clipped during a tau-leaping step due to proposing an out of bounds jump during the reverse sampling process.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/781af2bb53dd96ba4be22574ce706b23faef9c69f0079cf1b1e457898aac66df.jpg)



Figure 11: The progression of a selection of dimensions during the reverse sampling process.


since the mapping to the integers was arbitrary. The uniform rate is suitable for this categorical data because, during a time interval, it has a uniform probability to transition to any other state. The D3PM baseline was implemented also with a time homogeneous uniform forward kernel set such that the rate of noising is matched in the discrete and continuous time cases. 

We define our conditional denoising network, $p _ { 0 \mid t } ^ { \theta } ( x _ { 0 } | x , y )$ using a transformer architecture inspired by [31]. It takes an input of shape (B, D, S) where B is the batch size, D is the dimensionality (256) and S is the state size (129). This final dimension contains the one-hot vectors. The conditioning on the initial bars is achieved by concatenating the conditioning information $y$ with the noisy input x. At the start of the network, there is an input embedding linear layer with output of size 128 which is our model dimension for the transformer. Then a transformer positional embedding is added to the hidden state. Next a stack of 6 transformer encoder layers are applied which consist of a self attention block and a one hidden layer MLP. The self attention block uses 8 heads and the MLP has a hidden layer size of 2048. At the output of each internal block, we apply dropout with rate 0.1. Finally, there is a stack of 2 residual MLP layers. Each consists of a one hidden layer MLP with a hidden dimension of 2048. There is a residual connection between the input and output of the MLP. A layer norm is applied to the output of the block. To create the output of the network, there is an output linear layer with an output shape of (B, D, S) where now the S dimension has logit probabilities. To instill a residual bias into the network, we add the one-hot input to the output logits. All activations are ReLU. The time is input into the network through FiLM layers [35]. First, the time is embedded using the sinusoidal transformer position embedding as in the U-net architecture used for image modeling to create an embedding size of 128. This is then passed into a single hidden layer MLP with hidden size 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/075697531774ee0f4a6b5c52ee3a50505ec7303a69304ab9b486c0baede47bfe.jpg)



Figure 12: Proportion of jumps rejected during reverse sampling. The rejection rate is calculated as the proportion of dimensions in a tau leaping step that have their jump rejected. The results are averaged over a batch of 16.


2048 and output size 512. Within each encoder and residual MLP block, there is a FiLM linear layer which takes in the 512 time embedding and outputs two FiLM parameters each of size 128. These are the scale and offset applied to the hidden state. In the encoder blocks, this FiLM transform is applied after the self attention block and again after the fully connected block. In the residual MLP blocks, it is applied after the layer norm operation. Our network has approximately 7 million parameters in total. 

We optimize using Adam for 1M steps with a batch size of 64 and learning rate of 0.0002. We use the conditional $\bar { \mathcal { L } } _ { \mathrm { C T } }$ objective with additional direct $p _ { 0 | t } ^ { \theta }$ supervision as described in Appendix D with weight $\lambda = 0 . 0 0 1$ . We also make the same one forward pass approximation as explained in Appendix C.4. We use the standard set of training tricks to improve optimization [3, 4]. Throughout training we maintain an exponential moving average of the parameters with decay factor 0.9999. These average parameters are used during testing. At the start of optimization we use a linear learning rate warm-up for the first 5000 steps. We clip the gradient norm at a norm value of 1.0. We train on a single V100 GPU on an academic cluster. 

For sampling with τ LDR-0 we use $\tau = 0 . 0 0 1$ and for sampling with τ LDR-2 we include 2 corrector steps per predictor step after $t < 0 . 9 T$ . The corrector rate is simulated with $\tau = 0 . 0 0 0 1$ which we found to perform best. We reject any dimension in which 2 or more jumps are proposed as this is categorical data. We plot the rejection rate in Figure 12. Most of the time, the rejection rate is zero and there are few steps for which it increases slightly. We show a large batch of samples from the first 10 songs in the test dataset in Figure 13. We see that there is variation between the sampled completions and they consistently follow the style of the conditioning first two bars of the song. Audio samples from the model are available at https: $/ / { \tt g } \dot { \bf 1 }$ thub.com/andrew-cr/tauLDR. Finally, we examine the progression of a random selection of dimensions during reverse sampling for the uniform and birth/death rate matrix cases. Figure 14 shows the progression for the uniform case, we see that large jumps through the state space are made throughout the reverse process. Figure 15 shows the progression for the birth/death case. At the start of reverse sampling, no dimensions move as the rejection rate is high in this case because the rate matrix is not suitable for categorical data. Nearer the end, small jumps are made between adjacent states but since large jumps between any category do not occur for this rate matrix, the performance will overall be worse. 

# I Ethical Considerations

Our work increases our theoretical understanding of denoising generative models and also improves generation capabilities within some discrete datasets. Deep generative models are generic methods 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/f6ebfb6e923e01692cf0798c6ab0318d9874eaf3a4102f8c18afb748d1de8845.jpg)



Figure 13: Two conditional samples from the τ LDR-0 model for each of the first 10 songs in the test dataset.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/d65214918c038bac05e872cb765c710ae937fe1cce93b92f60b86cf2c4e3f888.jpg)



Figure 14: The progression of a selection of dimensions during the reverse sampling process for the uniform rate matrix.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-22/33c3bef3-d6a3-4e48-94cd-cbae149f7077/af2bf548fec502a4480370d90c148e5b288e2f885814450ca37082d0c5985b78.jpg)



Figure 15: The progression of a selection of dimensions during the reverse sampling process for the birth/death rate matrix.


for learning from unstructured data and can have negative social impacts when misused. For example, they can be used to spread misinformation by reducing the resources required to create realistic fake content. Furthermore, generative models will produce samples that accurately reflect the statistics of their training dataset. Therefore, if samples from these models are interpreted as an objective truth without fully considering the biases present in the original data, then they can perpetuate discrimination against minority groups. 

In this work, we train on datasets that contain less sensitive data such as pictures of objects and music samples. The methods we presented, however, could be used to model images of people or text from the internet which will contain biases and potentially harmful content that the model will then learn from and reproduce. Great care must be taken when training these models on real world datasets and when deploying them so as to mitigate and prevent the harms that they can cause. 