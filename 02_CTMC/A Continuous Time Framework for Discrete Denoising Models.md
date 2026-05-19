# A Continuous Time Framework for Discrete Denoising Models

Andrew Campbell1

Joe Benton1

Valentin De Bortoli2

Tom Rainforth1

George Deligiannidis1

Arnaud Doucet1

1Department of Statistics, University of Oxford, UK 2CNRS ENS Ulm, Paris, France {campbell, benton, rainforth, deligian, doucet}@stats.ox.ac.uk valentin.debortoli@gmail.com

## Abstract

We provide the first complete continuous time framework for denoising diffusion models of discrete data. This is achieved by formulating the forward noising process and corresponding reverse time generative process as Continuous Time Markov Chains (CTMCs). The model can be efficiently trained using a continuous time version of the ELBO. We simulate the high dimensional CTMC using techniques developed in chemical physics and exploit our continuous time framework to derive high performance samplers that we show can outperform discrete time methods for discrete data. The continuous time treatment also enables us to derive a novel theoretical result bounding the error between the generated sample distribution and the true data distribution.

## 1 Introduction

Diffusion/score-based/denoising models [1, 2, 3, 4] are a popular class of generative models that achieve state-of-the-art sample quality with good coverage of the data distribution [5] all whilst using a stable, non-adversarial, simple to implement training objective. The general framework is to define a forward noising process that takes in data and gradually corrupts it until the data distribution is transformed into a simple distribution that is easy to sample. The model then learns to reverse this process by learning the logarithmic gradient of the noised marginal distributions known as the score.

Most previous work on denoising models operates on a continuous state space. However, there are many problems for which the data we would like to model is discrete. This occurs, for example, in text, segmentation maps, categorical features, discrete latent spaces, and the direct 8-bit representation of images. Previous work has tried to realize the benefits of the denoising framework on discrete data problems, with promising initial results [6, 7, 8, 9, 10, 11, 12, 13].

All of these previous approaches train and sample the model in discrete time. Unfortunately, working in discrete time has notable drawbacks. It generally forces the user to pick a partition of the process at training time and the model only learns to denoise at these fixed time points. Due to the fixed partition, we are then limited to a simple ancestral sampling strategy. In continuous time, the model instead learns to denoise for any arbitrary time point in the process. This complete specification of the reverse process enables much greater flexibility in defining the reverse sampling scheme. For example, in continuous state spaces, continuous time samplers that greatly reduce the sampling time have been devised [14, 15, 16, 17] as well as ones that improve sample quality [4, 18]. The continuous time interpretation has also enabled the derivation of interesting theoretical properties such as error bounds [19] in continuous state spaces.

<!-- image-->  
Figure 1: The forward noising process corrupts data according to $R _ { t }$ , the rate of corruption events at time t. The noising process’ time reversal gives the generative process which is defined through $\hat { R } _ { t } ^ { \theta }$ , the rate of generative events at time t. $\hat { R } _ { t } ^ { \theta }$ is parameterized through the denoising network, $p _ { 0 \mid t } ^ { \bar { \theta } } ( x _ { 0 } | x _ { t } )$ , which outputs categorical probabilities over clean x0 values conditioned on a noisy $x _ { t }$

To allow these benefits to be exploited for discrete state spaces as well, we formulate a continuous time framework for discrete denoising models. Specifically, our contributions are as follows. We formulate the forward noising process as a Continuous Time Markov Chain (CTMC) and identify the generative CTMC that is the time-reversal of this process. We then bound the log likelihood of the generated data distribution, giving a continuous time equivalent of the ELBO that can be used for efficient training of a parametric approximation to the true generative reverse process. To efficiently simulate the parametric reverse process, we leverage tau-leaping [20] and propose a novel predictor-corrector type scheme that can be used to improve simulation accuracy. The continuous time framework allows us to derive a bound on the error between the true data distribution and the samples generated from the approximate reverse process simulated with tau-leaping. Finally, we demonstrate our proposed method on the generative modeling of images from the CIFAR-10 dataset and monophonic music sequences. Notably, we find our tau-leaping with predictor-corrector sampler can provide higher quality CIFAR10 samples than previous discrete time discrete state approaches, further closing the performance gap between when images are modeled as discrete data or as continuous data.

Proofs for all propositions and theorems are given in the Appendix.

## 2 Background on Discrete Denoising Models

In the discrete time, discrete state space case, we aim to model discrete data $x _ { 0 } \in \mathcal { X }$ with finite cardinality $S = | { \mathcal { X } } |$ We assume $x _ { 0 } \sim p _ { \mathrm { d a t a } } ( x _ { 0 } )$ for some discrete data distribution $p _ { \mathrm { d a t a } } ( x _ { 0 } )$ . We define a forward noising process that transforms $p _ { \mathrm { d a t a } } ( x _ { 0 } )$ to some distribution $q _ { K } ( x _ { K } )$ that closely approximates an easy to sample distribution $p _ { \mathrm { r e f } } ( x _ { K } )$ This is done by defining forward kernels $q _ { k + 1 | k } { \left( x _ { k + 1 } | x _ { k } \right) }$ that all admit $p _ { \mathrm { r e f } }$ as a stationary distribution and mix reasonably quickly. For example, one can use a simple uniform kernel [6, 8], $q _ { k + 1 | k } ( x _ { k + 1 } | x _ { k } ) = \delta _ { x _ { k + 1 } , x _ { k } } ( 1 - \beta ) + ( 1 -$ $\delta _ { x _ { k + 1 } , x _ { k } } ) \beta / ( S - 1 )$ where δ is a Kronecker delta. The corresponding $p _ { \mathrm { r e f } }$ is the uniform distribution over all states. Other choices include: an absorbing state kernel—where for each state there is a small probability that it transitions to some absorbing state—or a discretized Gaussian kernel—where only transitions to nearby states have significant probability (valid for spaces with ordinal structure) [8].

After defining $q _ { k + 1 | k }$ , we have a forward joint decomposition as follows

$$
\begin{array} { r } { q _ { 0 : K } ( x _ { 0 : K } ) = p _ { \mathrm { d a t a } } ( x _ { 0 } ) \prod _ { k = 0 } ^ { K - 1 } q _ { k + 1 | k } ( x _ { k + 1 } | x _ { k } ) . } \end{array}
$$

The joint distribution $q _ { 0 : K } \big ( \boldsymbol { x } _ { 0 : K } \big )$ also admits a reverse decomposition:

$$
\begin{array} { r } { q _ { 0 : K } ( x _ { 0 : K } ) = q _ { K } ( x _ { K } ) \prod _ { k = 0 } ^ { K - 1 } q _ { k | k + 1 } ( x _ { k } | x _ { k + 1 } ) \mathrm { ~ w h e r e ~ } q _ { k | k + 1 } ( x _ { k } | x _ { k + 1 } ) = \frac { q _ { k + 1 | k } ( x _ { k + 1 } | x _ { k } ) q _ { k } ( x _ { k } ) } { q _ { k + 1 } ( x _ { k + 1 } ) } . } \end{array}
$$

Here $q _ { k } ( x _ { k } )$ denotes the marginal of $q _ { 0 : K } \big ( \boldsymbol { x } _ { 0 : K } \big )$ at time k. If one had access to $q _ { k | k + 1 }$ and could sample $q _ { K }$ exactly, then samples from $p _ { \mathrm { d a t a } } ( x _ { 0 } )$ could be produced by first sampling $x _ { K } \sim q _ { K } ( \cdot )$ and then ancestrally sampling the reverse kernels, i.e. $x _ { k } \sim q _ { k | k + 1 } ( \cdot | x _ { k + 1 } )$

However, in practice, $q _ { k | k + 1 }$ is intractable and needs to be approximated with a parametric reverse kernel, $p _ { k | k + 1 } ^ { \theta }$ . This kernel is commonly defined through the analytic $q _ { k | k + 1 , 0 }$ distribution and a parametric ‘denoising’ model $p _ { 0 | k + 1 } ^ { \theta } \left[ 6 , 8 \right]$

$$
\begin{array} { r l } & { p _ { k | k + 1 } ^ { \theta } ( x _ { k } | x _ { k + 1 } ) \triangleq \sum _ { x _ { 0 } } q _ { k | k + 1 , 0 } ( x _ { k } | x _ { k + 1 } , x _ { 0 } ) p _ { 0 | k + 1 } ^ { \theta } ( x _ { 0 } | x _ { k + 1 } ) } \\ & { \qquad = q _ { k + 1 | k } ( x _ { k + 1 } | x _ { k } ) \sum _ { x _ { 0 } } \frac { q _ { k | 0 } ( x _ { k } | x _ { 0 } ) } { q _ { k + 1 | 0 } ( x _ { k + 1 } | x _ { 0 } ) } p _ { 0 | k + 1 } ^ { \theta } ( x _ { 0 } | x _ { k + 1 } ) . } \end{array}\tag{1}
$$

Though $q _ { K } ( x _ { K } )$ is also intractable, for large K we can reliably approximate it with $p _ { \mathrm { r e f } } ( x _ { K } )$ . Note that the faster the transitions mix, the more accurate this approximation becomes. Approximate samples from $p _ { \mathrm { d a t a } } ( x _ { 0 } )$ can then be obtained by sampling the generative joint distribution

$$
\begin{array} { r } { p _ { 0 : K } ^ { \theta } ( x _ { 0 : K } ) = p _ { \mathrm { r e f } } ( x _ { K } ) \prod _ { k = 0 } ^ { K - 1 } p _ { k | k + 1 } ^ { \theta } ( x _ { k } | x _ { k + 1 } ) , } \end{array}
$$

where $\theta$ is trained through minimizing the negative discrete time (DT) ELBO which is an upper bound on the negative model log-likelihood

$$
\begin{array} { r } { \mathbb { E } _ { p _ { \mathrm { d a t a } } ( x _ { 0 } ) } \left[ - \log p _ { 0 } ^ { \theta } ( x _ { 0 } ) \right] \leq \mathbb { E } _ { q _ { 0 : K } ( x _ { 0 : K } ) } \left[ - \log \frac { p _ { 0 : K } ^ { \theta } ( x _ { 0 : K } ) } { q _ { 1 : K | 0 } ( x _ { 1 : K } | x _ { 0 } ) } \right] = \mathcal { L } _ { \mathrm { D T } } ( \theta ) . } \end{array}
$$

It was shown in [1] that $\mathcal { L } _ { \mathrm { D T } }$ can be re-written as

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { D T } } ( \theta ) = \mathbb { E } _ { p _ { \mathrm { d a n } } ( x _ { 0 } ) } \Bigl [ \mathrm { K L } ( q _ { K | 0 } ( x _ { K } | x _ { 0 } ) | | p _ { \mathrm { r e f } } ( x _ { K } ) ) - \mathbb { E } _ { q _ { 1 } | 0 } ( x _ { 1 } | x _ { 0 } ) \left[ \log p _ { 0 | 1 } ^ { \theta } ( x _ { 0 } | x _ { 1 } ) \right] } \\ & { \qquad + \sum _ { k = 1 } ^ { K - 1 } \mathbb { E } _ { q _ { k + 1 | 0 } ( x _ { k + 1 } | x _ { 0 } ) } \left[ \mathrm { K L } ( q _ { k | k + 1 , 0 } ( x _ { k } | x _ { k + 1 } , x _ { 0 } ) | | p _ { k | k + 1 } ^ { \theta } ( x _ { k } | x _ { k + 1 } ) ) \right] \Bigr ] } \end{array}
$$

where KL is the Kullback–Leibler divergence. The forward kernels $q _ { k + 1 | k }$ are chosen such that $q _ { k | 0 } ( x _ { k } | x _ { 0 } )$ can be computed efficiently in a time independent of k. With this, θ can be efficiently trained by taking a random selection of terms from ${ \mathcal { L } } _ { \mathrm { D T } }$ in each minibatch and performing a stochastic gradient step.

## 3 Continuous Time Framework

## 3.1 Forward process and its time reversal

Our method is built upon a continuous time process from $t = 0 \mathrm { t o } t = T$ . State transitions can occur at any time during this process as opposed to the discrete time case where transitions only occur when one of the finite number of transition kernels is applied (see Figure 1). This process is known as a Continuous Time Markov Chain (CTMC), we provide a short overview of CTMCs in Appendix A for completeness. Giving an intuitive introduction here, we can define a CTMC through an initial distribution $q _ { 0 }$ and a transition rate matrix $R _ { t } \in \mathbb { R } ^ { S \times S }$ . If the current state is x˜, then the transition rate matrix entry $R _ { t } ( \tilde { x } , x )$ is the instantaneous rate (occurrences per unit time) at which state x˜ transitions to state $x .$ Loosely speaking, the next state in the process will likely be one for which $R _ { t } ( \tilde { x } , x )$ is high, and furthermore, the higher the rate is, the less time it will take for this transition to occur.

It turns out that the transition rate, $R _ { t }$ , also defines the infinitesimal transition probability for the process between the two time points $t - \Delta t$ and t

$$
q _ { t | t - \Delta t } ( x | \tilde { x } ) = \delta _ { x , \tilde { x } } + R _ { t } ( \tilde { x } , x ) \Delta t + o ( \Delta t ) ,
$$

where $o ( \Delta t )$ represents terms that tend to zero at a faster rate than $\Delta t$ . Comparing to the discrete time case, we see that $R _ { t }$ assumes an analogous role to the discrete time forward kernel $q _ { k + 1 | k }$ in how we define the forward process. Therefore, just as in discrete time, we design $R _ { t }$ such that: i) the forward process mixes quickly towards an easy to sample (stationary) distribution, $p _ { \mathrm { r e f } } , ( \mathrm { e . g }$ . uniform), ii) we can analytically obtain $q _ { t \mid 0 } \big ( x _ { t } | x _ { 0 } \big )$ distributions to enable efficient training (see Section 4.1 for how this is done). We initialize the forward CTMC at $q _ { 0 } ( x _ { 0 } ) = p _ { \mathrm { d a t a } } ( x _ { 0 } )$ at time $t = 0$ . We denote the marginal at time $t = T$ as $q _ { T } ( x _ { T } )$ , which should be close to $p _ { \mathrm { r e f } } ( x _ { T } )$

We now consider the time reversal of the forward process, which will take us from the marginal $q _ { T } ( x _ { T } )$ back to the data distribution $p _ { \mathrm { d a t a } } ( x _ { 0 } )$ through a reverse transition rate matrix, $\hat { R } _ { t } \in \mathbb { R } ^ { S \times S }$ :

$$
q _ { t | t + \Delta t } ( \tilde { x } | x ) = \delta _ { \tilde { x } , x } + \hat { R } _ { t } ( x , \tilde { x } ) \Delta t + o ( \Delta t ) .
$$

In discrete time, one uses Bayes rule to $\mathbf { g o }$ from $q _ { k + 1 | k } \ \mathrm { t o } \ q _ { k | k + 1 }$ . We can use similar ideas to calculate $\hat { R } _ { t }$ from $R _ { t }$ as per the following result.

Proposition 1. For a forward in time CTMC, $\{ x _ { t } \} _ { t \in [ 0 , T ] }$ , with rate matrix $R _ { t } ,$ , initial distribution pdata(x0) and terminal distribution $q _ { T } ( x _ { T } )$ , there exists a CTMC with initial distribution $q _ { T } ( x _ { T } )$ at $t = T$ , terminal distribution $p _ { \mathrm { d a t a } } ( x _ { 0 } )$ at $t = 0$ and transition rate matrix $\hat { R } _ { t }$ that runs backwards in time and is almost everywhere equivalent to the time reversal of the forward CTMC, $\{ x _ { t } \} _ { t \in [ T , 0 ] }$ Furthermore, $\hat { R } _ { t }$ is related to $R _ { t }$ by the following expression

$$
\begin{array} { r } { \hat { R } _ { t } ( x , \tilde { x } ) = R _ { t } ( \tilde { x } , x ) \sum _ { x _ { 0 } } \frac { q _ { t | 0 } ( \tilde { x } | x _ { 0 } ) } { q _ { t | 0 } ( x | x _ { 0 } ) } q _ { 0 | t } ( x _ { 0 } | x ) \quad f o r \quad x \neq \tilde { x } , } \end{array}
$$

where $q _ { t \mid 0 } ( x | x _ { 0 } )$ are the conditional marginals of the forward process and $q _ { 0 | t } ( x _ { 0 } | x ) =$ $q _ { t | 0 } ( x | x _ { 0 } ) p _ { \mathrm { d a t a } } ( x _ { 0 } ) / q _ { t } ( x )$ with $q _ { t } ( x )$ being the marginal of the forward process at time t. When $\begin{array} { r } { x = \tilde { x } , \hat { R } _ { t } ( x , x ) = - \sum _ { x ^ { \prime } \ne x } \hat { R } _ { t } ( x , x ^ { \prime } ) } \end{array}$ because the rows must sum to zero (see Appendix A).

Unfortunately, $\hat { R } _ { t }$ is intractable due to the intractability of $q _ { t } ( x )$ and thus of $q _ { 0 \mid t } ( x _ { 0 } | x )$ . Therefore, we consider an approximation $\hat { R } _ { t } ^ { \theta }$ of $\hat { R } _ { t }$ by approximating $q _ { 0 \mid t } ( x _ { 0 } | x )$ with a parametric denoising model, $p _ { 0 | t } ^ { \theta } ( x _ { 0 } | x )$

$$
\begin{array} { r } { \hat { R } _ { t } ^ { \theta } ( x , \tilde { x } ) = R _ { t } ( \tilde { x } , x ) \sum _ { x _ { 0 } } \frac { q _ { t | 0 } ( \tilde { x } | x _ { 0 } ) } { q _ { t | 0 } ( x | x _ { 0 } ) } p _ { 0 | t } ^ { \theta } ( x _ { 0 } | x ) \quad \mathrm { f o r } \quad x \neq \tilde { x } } \end{array}
$$

and $\begin{array} { r } { \hat { R } _ { t } ^ { \theta } ( x , x ) = - \sum _ { x ^ { \prime } \neq x } \hat { R } _ { t } ^ { \theta } ( x , x ^ { \prime } ) } \end{array}$ as before. As a further analogy to the discrete time case, notice that when $x \neq \tilde { x } , \hat { R } _ { t } ^ { \theta }$ has the same form as the discrete time parametric reverse kernel, $p _ { k | k + 1 } ^ { \theta }$ defined in eq (1) but with the forward kernel, $q _ { k + 1 | k }$ , replaced by the forward rate, $R _ { t }$

## 3.2 Continuous Time ELBO

In discrete time, θ is trained by minimizing the discrete time negative ELBO, ${ \mathcal { L } } _ { \mathrm { D T } }$ , formed from the forward and reverse processes. We mirror this approach in continuous time by minimizing the corresponding continuous time (CT) negative ELBO, $\mathcal { L } _ { \mathrm { C T } }$ , as derived below.

Proposition 2. For the reverse in time CTMC with initial distribution $p _ { r e f } ( x _ { T } )$ , terminal distribution $p _ { 0 } ^ { \theta } ( x _ { 0 } )$ , and reverse rate $\hat { R } _ { t } ^ { \theta }$ , an upper bound on the negative model log-likelihood, $\mathbb { E } _ { p _ { \mathrm { d a t a } } ( x _ { 0 } ) } [ - \log p _ { 0 } ^ { \theta } ( x _ { 0 } ) ]$ ], is given by

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C T } } ( \theta ) = T \mathbb { E } _ { t \sim \mathcal { U } ( 0 , T ) q _ { t } ( x ) r _ { t } ( \tilde { x } | x ) } \left[ \left\{ \sum _ { x ^ { \prime } \neq x } \hat { R } _ { t } ^ { \theta } ( x , x ^ { \prime } ) \right\} - \mathcal { Z } ^ { t } ( x ) \log \left( \hat { R } _ { t } ^ { \theta } ( \tilde { x } , x ) \right) \right] + C , } \end{array}
$$

where C is a constant independent of θ and

$$
\begin{array} { r } { \mathcal { Z } ^ { t } ( x ) = \sum _ { x ^ { \prime } \ne x } R _ { t } ( x , x ^ { \prime } ) } \end{array}
$$

$$
r _ { t } ( \tilde { x } | x ) = ( 1 - \delta _ { \tilde { x } , x } ) R _ { t } ( x , \tilde { x } ) / \mathcal { Z } ^ { t } ( x ) .
$$

Here $r _ { t } ( \tilde { x } | x )$ gives the probability of transitioning from x to ${ \tilde { x } } ,$ given that we know a transition occurs at time t. We can optimize this objective efficiently with stochastic gradient descent. For a gradient update, we sample a batch of datapoints from $p _ { \mathrm { d a t a } } ( x _ { 0 } )$ , noise each datapoint using a random time, $\bar { t \mapsto { \mathcal { U } } } ( 0 , T ) , x \bar { \sim } q _ { t | 0 } ( x | x _ { 0 } )$ and finally sample an auxiliary x˜ from $r _ { t } ( \bar { \tilde { x } } | x )$ for each x. Intuitively, $( x , { \tilde { x } } )$ are a pair of states following the forward in time noising process. Minimizing the second term in $\mathcal { L } _ { \mathrm { C T } }$ maximizes the reverse rate for this pair, but going in the backwards direction, x˜ to x. This is how $\hat { R } _ { t } ^ { \theta }$ learns to reverse the noising process. Intuition on the first term and a direct comparison to ${ \mathcal { L } } _ { \mathrm { D T } }$ is given in Appendix C.1.

The first argument of $\hat { R } _ { t } ^ { \theta }$ is input into $p _ { 0 | t } ^ { \theta }$ so we naively require two network forward passes on x and x˜ to evaluate the objective. We can avoid this by approximating the $q _ { t } ( x )$ sample in the first term with x˜ meaning we need only evaluate the network once on ${ \tilde { x } } .$ The approximation is valid because, as we show in Appendix C.4, x˜ is approximately distributed according to $q _ { t + \delta t }$ for $\delta t$ very small.

## 4 Efficient Forward and Backward Sampling

## 4.1 Choice of Forward Process

The transition rate matrix $R _ { t }$ needs to be chosen such that the forward process: i) mixes quickly towards $p _ { \mathrm { r e f } } .$ , and ii) the $q _ { t \mid 0 } ( x | x _ { 0 } )$ distributions can be analytically obtained. The Kolmogorov differential equation for the CTMC needs to be integrated to obtain $q _ { t \mid 0 } ( x | x _ { 0 } )$ . This can be done analytically when $R _ { t }$ and $R _ { t ^ { \prime } }$ commute for all $t , t ^ { \prime } ,$ , see Appendix E. An easy way to meet this condition is to let $R _ { t } = \beta ( t ) R _ { b }$ where $R _ { b } \in \mathbb { R } ^ { S \times S }$ is a user-specified time independent base rate matrix and $\beta ( t ) \in \mathbb { R }$ is a time dependent scalar. We then obtain the analytic expression

$$
\begin{array} { r } { q _ { t | 0 } ( x = j | x _ { 0 } = i ) = \left( Q \mathrm { e x p } \left[ \Lambda \int _ { 0 } ^ { t } \beta ( s ) d s \right] Q ^ { - 1 } \right) _ { i j } } \end{array}
$$

where ${ \cal R } _ { b } = Q \Lambda Q ^ { - 1 }$ is the eigendecomposition of matrix $R _ { b }$ and exp[·] the element-wise exponential. Our choice of $\beta$ schedule is guided by [3, 4], $\beta ( t ) = a b ^ { t } \log ( b )$ . The hyperparameters a and b are selected such that $q _ { T } ( x ) \approx p _ { \mathrm { r e f } } ( x )$ at the terminal time $t = T$ while having a steady speed of ‘information corruption’ which ensures that $\hat { R } _ { t }$ does not vary quickly in a short span of time.

We experiment with a variety of $R _ { b }$ matrices, for example, a uniform rate, $R _ { b } = \mathbb { 1 } \mathbb { 1 } ^ { T } - S \mathrm { I d }$ , where $\mathbb { 1 1 } ^ { T }$ is a matrix of ones and Id is the identity. For problems with a heavy spatial bias, e.g. images, we can instead use a forward rate that only encourages transitions to nearby states; details and the links to the corresponding discrete time processes can be found in Appendix E.

## 4.2 Factorizing Over Dimensions

Our aim is to model data that is D dimensional, with each dimension taking one value from S possibilities. We now slightly redefine notation and say $\pmb { x } ^ { 1 : D } \in \mathcal { X } ^ { D } , | \mathcal { X } | = S$ . In this setting, calculating transition probabilities naively would require calculating $S ^ { D }$ rate values corresponding to each of the possible next states. This is intractable for any reasonably sized S and D. We avoid this problem simply by factorizing the forward process such that each dimension propagates independently. Since this is a continuous time process and each dimension’s forward process is independent of the others, the probability two or more dimensions transition at exactly the same time is zero. Therefore, overall in the full dimensional forward CTMC, each transition only ever involves a change in exactly one dimension. For the time reversal CTMC, it will also be true that exactly one dimension changes in each transition. This makes computation tractable because of the $S ^ { \check { D } }$ rate values, only $D \bar { \times } \left( S - 1 \right) + 1$ are non-zero - those corresponding to transitions where exactly one dimension changes plus the no change transition. Finally, we note that even though dimensions propagate independently in the forward direction, they are not independent in the reverse direction because the starting points for each dimension’s forward process are not independent for non factorized $p _ { \mathrm { d a t a } }$ . The following proposition shows the exact forms for the forward and reverse rates in this case.

Proposition 3. If the forward process factorizes as $\begin{array} { r } { q _ { t | s } ( \pmb { x } _ { t } ^ { 1 : D } | \pmb { x } _ { s } ^ { 1 : D } ) = \prod _ { d = 1 } ^ { D } q _ { t | s } ( \pmb { x } _ { t } ^ { d } | \pmb { x } _ { s } ^ { d } ) , t > s , } \end{array}$ then the forward and reverse rates are of the form

$$
\begin{array} { r } { R _ { t } ^ { 1 : D } ( \tilde { { \pmb x } } ^ { 1 : D } , { \pmb x } ^ { 1 : D } ) = \sum _ { d = 1 } ^ { D } R _ { t } ^ { d } ( \tilde { { \pmb x } } ^ { d } , { \pmb x } ^ { d } ) \delta _ { { \pmb x } ^ { 1 : D } \setminus { d } , \tilde { { \pmb x } } ^ { 1 : D } \setminus { d } } , } \end{array}
$$

$$
\begin{array} { r } { \hat { R } _ { t } ^ { 1 : D } ( { \pmb x } ^ { 1 : D } , \tilde { \pmb x } ^ { 1 : D } ) = \sum _ { d = 1 } ^ { D } R _ { t } ^ { d } ( \tilde { \pmb x } ^ { d } , { \pmb x } ^ { d } ) \delta _ { { \pmb x } ^ { 1 : D } \vee { d } , \tilde { \pmb x } ^ { 1 : D } \vee { d } } \sum _ { { \pmb x } _ { 0 } ^ { d } } q _ { 0 \mid t } ( x _ { 0 } ^ { d } | { \pmb x } ^ { 1 : D } ) \frac { q _ { t \mid 0 } ( \tilde { \pmb x } ^ { d } | { \pmb x } _ { 0 } ^ { d } ) } { q _ { t \mid 0 } ( { \pmb x } ^ { d } | { \pmb x } _ { 0 } ^ { d } ) } , } \end{array}
$$

where $R _ { t } ^ { d } \in \mathbb { R } ^ { S \times S }$ and $\delta _ { { \pmb x } ^ { 1 : D \setminus d } , \tilde { { \pmb x } } ^ { 1 : D \setminus d } }$ is 1 when all dimensions except for d are equal.

To find $\hat { R } _ { t } ^ { \theta 1 : D }$ we simply replace $q _ { 0 | t } \big ( x _ { 0 } ^ { d } | \mathbf { x } ^ { 1 : D } \big )$ with $p _ { 0 | t } ^ { \theta } ( x _ { 0 } ^ { d } | \mathbf { x } ^ { 1 : D } )$ which is easily modeled with a neural network that outputs conditionally independent state probabilities in each dimension. In Appendix C.3 we derive the form of ${ \mathcal { L } } _ { \mathrm { C T } }$ when we use this factorized form for $R _ { t } ^ { 1 : D }$ and $\hat { R } _ { t } ^ { \theta 1 : D }$

## 4.3 Simulating the Generative Reverse Process with Tau-Leaping

The parametric generative reverse process is a CTMC with rate matrix $\hat { R } _ { t } ^ { \theta 1 : D }$ . Simulating this process from distribution $p _ { \mathrm { r e f } } ( \pmb { x } _ { T } ^ { 1 : D } )$ at time $t = T$ back to $t = 0$ will produce approximate samples from $p _ { \mathrm { d a t a } } ( \pmb { x } _ { 0 } ^ { 1 : D } )$ . The process could be simulated exactly using Gillespie’s Algorithm [21, 22, 23] which alternates between i) sampling a holding time to remain in the current state and ii) sampling a new state according to the current rate matrix, $\hat { R } _ { t } ^ { \theta 1 : D }$ (see $\operatorname { A p p e n d i x } F )$ . This is inefficient for large D because we would need to step through each transition individually and so only one dimension would change for each simulation step.

Instead, we use tau-leaping [20, 23], a very popular approximate simulation method developed in chemical physics. Rather than step back through time one transition to the next, tau-leaping leaps from t to $t - \tau$ and applies all transitions that occurred in $[ t - \tau , t ]$ simultaneously. To make a leap, we assume $\hat { R } _ { t } ^ { \theta 1 : D }$ and $\pmb { x } _ { t } ^ { 1 : D }$ remain constant in $[ t - \tau , t ]$ . As we propagate from t to $t - \tau$ we count all of the transitions that occur, but hold off on actually applying them until we reach $t - \tau$ , such that $\pmb { x } _ { t } ^ { 1 : D }$ remains constant in $\lceil t - \tau , t \rceil$ . Assuming $\hat { R } _ { t } ^ { \theta 1 : D }$ and $\pmb { x } _ { t } ^ { 1 : D }$ remain constant, the number of times a transition from $\pmb { x } _ { t } ^ { 1 : D }$ to $\widetilde { \pmb { x } } ^ { 1 : D }$ occurs in $[ t - \tau , t ]$ is Poisson distributed with mean $\tau \hat { R } _ { t } ^ { \theta 1 : D } ( \pmb { x } _ { t } ^ { 1 : D } , \tilde { \pmb { x } } ^ { 1 : D } )$ . Once we reach $t - \tau$ , we apply all transitions that occurred simultaneously i.e. $\begin{array} { r } { \pmb { x } _ { t - \tau } ^ { 1 : D } = \pmb { x } _ { t } ^ { 1 : D } + \sum _ { i } P _ { i } ( \tilde { \pmb { x } } _ { i } ^ { 1 : D } - \pmb { x } _ { t } ^ { 1 : D } ) } \end{array}$ where $P _ { i }$ P is a Poisson random variable with mean $\tau \hat { R } _ { t } ^ { \theta 1 : D } ( \pmb { x } _ { t } ^ { 1 : D } , \tilde { \pmb { x } } _ { i } ^ { 1 : D } )$ . Note the sum assumes a mapping from X to Z.

<!-- image-->  
Figure 2: 3D visualization of one tau-leaping step from $x _ { t } ^ { 1 : 2 } = \{ S _ { 4 } , \dot { S } _ { 1 } \} \stackrel { } { \mathrm { t o } } x _ { t - \tau } ^ { 1 : 2 } =$ $\{ \breve { S } _ { 2 } , S _ { 3 } \}$ Here, $\textit { D } = \textit { 2 }$ $| \mathcal { X } | = 5 , P _ { 1 2 } = 1 , P _ { 2 2 } = 2$ all other $P _ { d s } = 0$

Using our knowledge of $\hat { R } _ { t } ^ { \theta 1 : D }$ , we can further unpack this update. Namely, $\hat { R } _ { t } ^ { \theta 1 : D } ( \pmb { x } _ { t } ^ { 1 : D } , \tilde { \pmb { x } } ^ { 1 : D } )$ can only be non-zero when $\tilde { \pmb { x } } ^ { 1 : D }$ has a different value to $\pmb { x } _ { t } ^ { 1 : D }$ in exactly one dimension (rates for multidimensional changes are zero). Explicitly summing over these options we get $\begin{array} { r } { \pmb { x } _ { t - \tau } ^ { 1 : D } = \pmb { x } _ { t } ^ { 1 : D } + \sum _ { d = 1 } ^ { D } \sum _ { s = 1 \backslash x _ { + } ^ { d } } ^ { S } P _ { d s } ( s - x _ { t } ^ { d } ) e ^ { d } } \end{array}$ where $e ^ { d }$ is a one-hot vector with a 1 at dimension d and $P _ { d s }$ is a Poisson random variable with mean $\tau \hat { R } _ { t } ^ { \theta 1 : D } ( \pmb { x } _ { t } ^ { 1 : D } , \pmb { x } _ { t } ^ { 1 : D } + ( s - x _ { t } ^ { d } ) e ^ { d } )$ . Since multiple $P _ { d s }$ can be non-zero, we see that tau-leaping allows $\pmb { x } _ { t } ^ { 1 : D }$ to change in multiple dimensions in a single step. Figure 2 visualizes this idea. During the $[ t - \tau , t ]$ interval, one jump occurs in dimension 1 and two jumps occur in dimension 2. These are all applied simultaneously once we reach $t - \tau$ . When our discrete data has ordinal structure (e.g. Section 6.2) our mapping to $\mathbb { Z }$ is not arbitrary and making multiple jumps within the same dimension $\begin{array} { r } { ( \sum _ { s = 1 \setminus x _ { + } ^ { d } } ^ { S } P _ { d s } > 1 ) } \end{array}$ is meaningful. In the non-ordinal/categorical case (e.g. Section 6.3) the mapping to Z is arbitrary and so, although taking simultaneous jumps in different dimensions is meaningful, taking multiple jumps within the same dimension is not. For this type of data, we reject changes to $\boldsymbol { x } _ { t } ^ { d }$ for any d for which $\textstyle \sum _ { s = 1 \setminus x _ { * } ^ { d } } ^ { S } { \dot { P _ { d s } } } > 1$ . In practice, the rejection rate is very small when $\widetilde { R } _ { t } ^ { \mathrm { f } : \overline { { D } } ^ { \perp } \mathrm { i } s }$ t suitable for categorical data (e.g. uniform), see Appendix H.3. In Section 4.5, our error bound accounts for this low probability of rejection and also the low probability of an out of bounds jump that we observe in practice in the ordinal case.

The tau-leaping approximation improves with smaller τ , recovering exact simulation in the limit as $\tau  0$ . Exact simulation is similar to an autoregressive model in that only one dimension changes per step. Increasing τ and thus the average number of dimensions changing per step gives us a natural way to modulate the ‘autoregressiveness’ of the model and trade sample quality with compute (Figure 4 right). We refer to our method of using tau-leaping to simulate the reverse CTMC as τ LDR (tau-leaping denoising reversal) which we formalize in Algorithm 1 in Appendix F.

We note that theoretically, one could approximate $\hat { R } _ { t } ^ { \theta 1 : D }$ as constant in the interval $[ t - \tau , t ]$ , and construct a transition probability matrix by solving the forward Kolmogorov equation with the matrix exponential $P _ { t - \tau | t } \approx \exp ( \tau \hat { R } _ { t } ^ { \theta 1 : D } )$ . However, for the learned $\hat { R } _ { t } ^ { \theta 1 : D } \in \mathbb { R } ^ { S ^ { D } \times S ^ { D } }$ matrix, it is intractable to compute this matrix exponential so we use tau-leaping for sampling instead.

## 4.4 Predictor-Corrector

During approximate reverse sampling, we aim for the marginal distribution of samples at time t to be close to $q _ { t } ( x _ { t } )$ (the marginal at time t of the true CTMC). The continuous time framework allows us to exploit additional information to more accurately follow the reverse progression of marginals, $\{ q _ { t } ( x _ { t } ) \bar  \} _ { t \in [ T , 0 ] }$ and improve sample quality. Namely, after a tau-leaping ‘predictor’ step using rate $\hat { R } _ { t } ^ { \theta }$ , we can apply ‘corrector’ steps with rate $R _ { t } ^ { c }$ which has $q _ { t } ( x _ { t } )$ as its stationary distribution. The corrector steps bring the distribution of samples at time t closer to the desired $q _ { t } ( x _ { t } )$ marginal. $R _ { t } ^ { c }$ is easy to calculate as stated below

Proposition 4. For a forward CTMC with marginals $\{ q _ { t } ( x _ { t } ) \} _ { t \in [ 0 , T ] }$ , forward rate, $R _ { t } ,$ , and corresponding reverse CTMC with rate $\hat { R } _ { t } ,$ , the rate $R _ { t } ^ { c } = R _ { t } + { \hat { R } } _ { t }$ has $q _ { t } ( x _ { t } )$ as its stationary distribution.

In practice, we approximate $R _ { t } ^ { c }$ by replacing $\hat { R } _ { t }$ with $\hat { R } _ { t } ^ { \theta }$ . This is directly analogous to Predictor-Corrector samplers in continuous state spaces [4] that predict by integrating the reverse SDE and correct with score-based Markov chain Monte Carlo steps, see Appendix F.2 for further discussion.

## 4.5 Error Bound

Our continuous time framework also allows us to provide a novel theoretical bound on the error between the true data distribution and the sample distribution generated via tau-leaping (without predictor-corrector steps), in terms of the error in our approximation of the reverse rate and the mixing of the forward noising process.

We assume we have a time-homogeneous rate matrix $R _ { t }$ on $x ,$ , from which we construct the factorized rate matrix $R _ { t } ^ { 1 : D }$ on $\chi D$ by setting $R _ { t } ^ { d } = R _ { t }$ for each d. Note that by rescaling time by a factor of $\beta ( t )$ we can transform our choice of rate from Section 4.1 to be time-homogeneous. We will denote $\begin{array} { r } { \vert \dot { R \vert } ^ { - } = \operatorname* { s u p } _ { t \in [ 0 , T ] , x \in \mathcal { X } } \vert R _ { t } ( x , x ) \vert } \end{array}$ , and let $t _ { \mathrm { m i x } }$ be the (1/4)-mixing time of the CTMC with rate $R _ { t }$ (see [24, Chapter 4.5]).

Theorem 1. For any $D \geq 1$ and distribution $p _ { \mathrm { d a t a } }$ on $\mathcal { X } ^ { D }$ , let $\{ x _ { t } \} _ { t \in [ 0 , T ] }$ be a CTMC starting in $p _ { \mathrm { d a t a } }$ with rate matrix $R _ { t } ^ { 1 : D }$ as above. Suppose that $\hat { R } _ { t } ^ { \theta 1 : D }$ is an approximation to the reverse rate matrix and let $( y _ { k } ) _ { k = 0 , 1 , \dots , N }$ be a tau-leaping approximation to the reverse dynamics with maximum step size τ . Suppose further that there is some constant $M > 0$ independent of D such that

$$
\sum _ { y \neq x } \left| \hat { R } _ { t } ^ { 1 : D } ( x , y ) - \hat { R } _ { t } ^ { \theta { 1 : D } } ( x , y ) \right| \leq M\tag{2}
$$

for all $t \in [ 0 , T ]$ Then under the assumptions in Appendix B.5, there are constants $C _ { 1 } , C _ { 2 } > 0$ depending on $\dot { \mathcal X }$ and $R _ { t }$ but not D such that, $i f \ : \mathcal { L } ( y _ { 0 } )$ denotes the law of $y _ { 0 } ,$ , we have the total variation bound

$$
\begin{array} { r } { | | \mathcal { L } ( y _ { 0 } ) - p _ { \mathrm { d a t a } } | | _ { \mathrm { T V } } \leq 3 M T + \left\{ \left( \left| R \right| S D C _ { 1 } \right) ^ { 2 } + \frac { 1 } { 2 } C _ { 2 } ( M + C _ { 1 } S D | R | ) \right\} \tau T + 2 \exp \left\{ - \frac { T \log ^ { 2 } 2 } { t _ { \mathrm { m i x } } \log 4 D } \right\} } \end{array}
$$

The first term of the above bound captures the error introduced by our approximation of the reverse rate $\hat { R } _ { t } ^ { 1 : D }$ with $\hat { R } _ { t } ^ { \theta 1 : D }$ . The second term reflects the error introduced by the tau-leaping approximation, and is linear in both $T$ and $\tau ,$ showing that as we take our tau-leaping steps to be arbitrarily small, the error introduced by tau-leaping goes to zero. The final term describes the mixing of the forward chain, and captures the error introduced since $p _ { \mathrm { r e f } }$ and $q _ { T }$ are not exactly equal.

We choose to make the dependence of the bound on the dimension D explicit, since we are specifically interested in applying tau-leaping to high dimensional problems where we make transitions in different dimensions simultaneously in a single time step. The bound grows at worst quadratically in the dimension, versus e.g. exponentially. The bound is therefore useful in showing us that we do not need to make τ impractically small in high dimensions. Other than gaining these intuitions, we do not expect the bound to be particularly tight in practice and further it would not be practical to compute because of the difficulty in finding M, C1 and $C _ { 2 }$

The assumptions listed in Appendix B.5 hold approximately for tau-leaping in practice when we use spatially biased rates for ordinal data such that jump sizes are small or uniform rates for non-ordinal data such that the dimensional rejection rate is small. These assumptions could be weakened, however, Theorem 1 would become much more involved, obscuring the intuition and structure of the problem.

## 5 Related Work

The application of denoising models to discrete data was first described in [1] using a binomial diffusion process for a binary dataset. Each reverse kernel $p _ { k | k + 1 } ^ { \theta }$ was directly parameterized without using a denoising model $p _ { 0 | k } ^ { \theta }$ . In [25] an approach for discrete categorical data was suggested using a uniform forward noising kernel, $q _ { k + 1 | k }$ , and a reverse kernel parameterized through a denoising model, though no experiments were performed with the approach. Experiments on text and segmentation maps were then performed with a similar model in [6]. Other forward kernels were introduced in [8] that are more appropriate for certain data types such as the spatially biased Gaussian kernel. [9, 13] apply the approach to discrete latent space modeling using uniform and absorbing state forward kernels. Whilst a link to continuous time for the forward process is mentioned in [8], all of these approaches train and sample in discrete time. We show in Appendix G that this involves making an implicit approximation for multi-dimensional data. We extend this line of work by training and sampling in continuous time.

<!-- image-->  
Figure 3: Left: Hellinger distance between the true training distribution and generated sample distributions with exact simulation or tau-leaping. With τ small, we simulate the reverse CTMC with the same fidelity as the exact simulation. Top Right: Histograms of the marginals during the reverse generative process simulated using tau-leaping with $\tau = 0 . 0 0 4$ . Darker and larger diamonds represent increased density. Bottom Right: The same for $\tau = 0 . 1$ , note the reduced sample quality.

Other works also operate in discrete space but less rigidly follow the diffusion framework. A corruption process tailored to text is proposed in [12], whereby token deletion and insertion is also incorporated. [26] also focus on text, creating a generative reverse chain that repeatedly applies the same denoising kernel. The corruption distribution is also defined through the same denoising kernel to reduce distribution shift between training and sampling. In [7], a more standard masking based forward process is used but the reversal is interpreted from an order agnostic autoregressive perspective. They also describe how their model can be interpreted as the reversal of a continuous time absorbing state diffusion but do not utilize this perspective in training or sampling. [27] propose a denoising type framework that can be used on binary data where the forward and reverse process share the same transition kernel. Finally, in [11], the discrete latent space of a VQVAE is modeled by quantizing an underlying continuous state space diffusion with probabilistic quantization functions.

## 6 Experiments

## 6.1 Demonstrative Example

We first verify the method can accurately produce samples from the entire support of the data distribution and that tau-leaping can accurately simulate the reverse CTMC. To do this, we create a dataset formed of 2d samples of a state space of 32 arranged such that the histogram of the training dataset forms a $\cdot _ { \tau } ,$ shape. We train a denoising model using the $\mathcal { L } _ { \mathrm { C T } }$ objective with $p _ { 0 | t } ^ { \theta }$ parameterized through a residual MLP (full details in Appendix H.1). We then sample the parameterized reverse process using an exact method (up to needing to numerically integrate the reverse rate) and tauleaping. Figure 3 top-right shows the marginals during reverse simulation with $\tau = 0 . 0 0 4$ and we indeed produce samples from the entire support of $p _ { \mathrm { d a t a } }$ . Furthermore, we find that with sufficiently small τ , we can match the fidelity of exact simulation of the reverse CTMC (Figure 3 left). The value of τ dictates the number of network evaluations in the reverse process according to $\mathrm { N F E } = T / \tau$ . In all experiments we use $T = 1$ . Exact simulation results in a non zero Hellinger distance between the generated and training distributions because of imperfections in the learned $\hat { R } _ { t } ^ { \theta }$ model.

## 6.2 Image Modeling

We now demonstrate that our continuous time framework gives us improved generative modeling performance versus operating in discrete time. We show this on the CIFAR-10 image dataset. Images are typically stored as discrete data, each pixel channel taking one value from 256 possibilities. Continuous state space methods have to somehow get around this fact by, for example, adding a discretization function at the end of the generative process [3] or adding uniform noise to the data.

Table 1: Sample quality metrics and model likelihoods for diffusion methods modeling CIFAR10 in discrete state space. Diffusion methods modeling CIFAR10 in continuous space are included for reference. The Inception Score (IS) and Fréchet Inception Distance (FID) are calculated using 50000 generated samples with respect to the training dataset as is standard practice. The ELBO values are reported on the test set in bits per dimension.
<table><tr><td></td><td>Method</td><td>IS (1)</td><td>FID (↓)</td><td>ELBO (↑)</td></tr><tr><td rowspan="4">Discrete state</td><td>D3PM Absorbing [8]</td><td>6.78</td><td>30.97</td><td>-4.40</td></tr><tr><td>D3PM Gauss [8]</td><td>8.56</td><td>7.34</td><td>-3.44</td></tr><tr><td>TLDR-O (ours)</td><td>8.74</td><td>8.10</td><td>-3.59</td></tr><tr><td>TLDR-10 (ours)</td><td>9.49</td><td>3.74</td><td>-3.59</td></tr><tr><td rowspan="2">Continuous state</td><td>DDPM[3]</td><td>9.46</td><td>3.17</td><td>-3.75</td></tr><tr><td>NCSN [4]</td><td>9.89</td><td>2.20</td><td>=</td></tr></table>

<!-- image-->

<!-- image-->  
Figure 4: Left: Unconditional CIFAR10 samples from our τ LDR-10 model Right: FID scores for the generated CIFAR10 samples versus number of $p _ { 0 | t } ^ { \theta }$ evaluations during sampling (variation induced by varying τ ). Calculated with 10k samples, hence the discrepancy with Table 1 [28].

Here, we model the images directly in discrete space. We parameterize $p _ { 0 | t } ^ { \theta }$ using the standard U-net architecture [3] with the modifications for discrete state space suggested by [8]. We use a spatially biased rate matrix and train with an augmented ${ \mathcal { L } } _ { \mathrm { C T } }$ loss including direct $\bar { p } _ { 0 \mid t } ^ { \theta ^ { - } }$ supervision, full experimental details are in Appendix H.2.

Figure 4 left shows randomly generated unconditional CIFAR10 samples from the model and we report sample quality metrics in Table 1. We see that our method (τ LDR-0) with 0 corrector steps has better Inception Score but worse FID than the D3PM discrete time method. However, our τ LDR-10 method with 10 corrector steps per predictor step at the end of the reverse sampling process $( t < 0 . 1 T )$ greatly improves sample quality, beating the discrete time method in both metrics and further closes the performance gap with methods modeling images as continuous data. The derivation of the corrector rate which gave us this improved performance required our continuous time framework. D3PM achieves the highest ELBO but we note that this does not correlate well with sample quality. In Table 1, τ was adjusted such that both τ LDR-0 and τ LDR-10 used 1000 $p _ { 0 | t } ^ { \theta }$ evaluations in the reverse sampling procedure. We show how FID score varies with number of $p _ { 0 \mid t } ^ { \theta }$ evaluations for τ LDR-{0, 3, 10} in Figure 4 right. The optimum number of corrector steps depends on the sampling budget, with lower numbers of corrector steps being optimal for tighter budgets. This is due to the increased τ required to maintain a fixed budget when we use a larger number of corrector steps.

## 6.3 Monophonic Music

In this experiment, we demonstrate our continuous time model improves generation quality on non-ordinal/categorical discrete data. We model songs from the Lakh pianoroll dataset [29, 30]. We select all monophonic sequences from the dataset such that at each of the 256 time steps either one from 128 notes is played or it is a rest. Therefore, our data has state space size S = 129 and dimension $D = 2 5 6$ . We scramble the ordering of the state space when mapping to Z to destroy any ordinal structure. We parameterize $p _ { 0 | \ i } ^ { \theta }$ with a transformer architecture [31] and train using a conditional form of $\mathcal { L } _ { \mathrm { C T } }$ targeting the conditional distribution of the final 14 bars (224 time steps) given the first 2 bars of the song. We use a uniform forward rate matrix, $R _ { t }$ , full experimental details are given in Appendix H.3. Conditional completions of unseen test songs are shown in Figure 5. The model is able to faithfully complete the piece in the same style as the conditioning bars.

Table 2: Metrics comparing generated conditional samples and ground truth completions. We compute these over the test set showing mean±std with respect to 5 samples for each test song.
<table><tr><td>Model</td><td></td><td>Hellnger DistanceProportion of Outliers</td></tr><tr><td>TLDR-0 Birth/Death</td><td> $0 . 3 9 2 8 \pm 0 . 0 0 1 0$ </td><td> $0 . 1 3 1 6 \pm 0 . 0 0 1 2$ </td></tr><tr><td>TLDR-0 Uniform</td><td> $0 . 3 7 6 5 \pm 0 . 0 0 1 3$ </td><td> $0 . 1 1 0 6 \pm 0 . 0 0 1 0$ </td></tr><tr><td>TLDR-2Uniform</td><td> $\mathbf { 0 . 3 7 6 2 \pm 0 . 0 0 1 5 }$ </td><td> $\mathbf { 0 . 1 0 9 1 \pm 0 . 0 0 1 4 }$ </td></tr><tr><td>D3PM Uniform [8]</td><td> $0 . 3 8 3 9 \pm 0 . 0 0 0 2$ </td><td> $0 . 1 1 3 7 \pm 0 . 0 0 1 0$ </td></tr></table>

<!-- image-->  
Figure 5: Conditional completions of an unseen music sequence. The conditioning 2 bars are shown to the left of the black line. More examples and audio recordings are linked in Appendix H.3.

We quantify sample quality in Table 2. We use two metrics: the Hellinger distance between the histograms of generated and ground truth notes and the proportion of outlier notes in the generations but not in the ground truth. Using our method, we compare between a birth/death and uniform forward rate matrix $R _ { t } .$ . The birth/death rate is only non-zero for adjacent states whereas the uniform rate allows transitions between arbitrary states which is more appropriate for the categorical case thus giving improved sample quality. Adding 2 corrector steps per predictor step further improves sample quality. We also compare to the discrete time method D3PM [8] with its most suitable corruption process for categorical data. We find it performs worse than our continuous time method.

## 7 Discussion

We have presented a continuous time framework for discrete denoising models. We showed how to efficiently sample the generative process with tau-leaping and provided a bound on the error of the generated samples. On discrete data problems, we found our predictor-corrector sampler improved sample quality versus discrete time methods. Regarding limitations, our model requires many model evaluations to produce a sample. Our work has opened the door to applying the work improving sampling speed on continuous data [14, 15, 16, 17, 32] to discrete data problems too. Modeling performance on images is also slightly behind continuous state space models, we hope this gap is further closed with bespoke discrete state architectures and corruption process tuning. Finally, we note that the ELBO values for the discrete time model on CIFAR10 are better than for our method. In this work, we focused on sample quality rather than using our model to give data likelihoods e.g. for compression downstream tasks.

## Acknowledgements

Andrew Campbell and Joe Benton acknowledge support from the EPSRC CDT in Modern Statistics and Statistical Machine Learning (EP/S023151/1). Arnaud Doucet is partly supported by the EPSRC grant EP/R034710/1. He also acknowledges support of the UK Defence Science and Technology Laboratory (DSTL) and EPSRC under grant EP/R013616/1. This is part of the collaboration between US DOD, UK MOD and UK EPSRC under the Multidisciplinary University Research Initiative. This project made use of time on Tier 2 HPC facility JADE2, funded by EPSRC (EP/T022205/1).

## References

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

## Appendix

## Contents

A Primer on Continuous Time Markov Chains 15   
B Proofs 16   
B.1 Proof of Proposition 1 . 16   
B.2 Proof of Proposition 2 . 17   
B.3 Proof of Proposition 3 . 21   
B.4 Proof of Proposition 4 . 22   
B.5 Proof of Theorem 1 22   
C Continuous Time ELBO Details 27   
C.1 Comparison with the Discrete Time ELBO . 27   
C.2 Conditional Form . 27   
C.3 Continuous Time ELBO with Factorization Assumptions 27   
C.4 One Forward Pass . 29   
D Direct Denoising Model Supervision 30   
E Choice of Forward Process 32   
F CTMC Simulation 34   
F.1 Exact CTMC and Tau-Leaping . 34   
F.2 Predictor-Corrector Discussion 34   
G Implicit Dimensional Assumptions Made in Discrete Time 36   
H Experimental Details 37   
H.1 Demonstrative Example . 37   
H.2 Image Modeling . . 38   
H.3 Monophonic Music 39   
I Ethical Considerations 42

The appendix is organized as follows. In Section A, we provide a short introduction to Continuous Time Markov Chains, including the relevant results we use in this work. Proofs for all the Propositions and Theorems from the main text are in Section B. We then describe in Section C some additional intuitions and forms of our proposed objective, $\mathcal { L } _ { \mathrm { C T } }$ . In Section D, we describe how an additional direct denoising model supervision term can be added to the objective to improve empirical performance. Details for how we define the forward process in our model can be found in Section E. Section F describes in more detail how CTMCs can be simulated and includes the algorithmic description of tau-leaping. We argue in Section G that operating in discrete time forces an implicit assumption when using a factorized forward process on multi-dimensional data. Full experimental details for all investigations can be found in Section H as well as additional plots and results from our models. Finally, in Section I, we consider the social impacts of our research.

<!-- image-->  
Figure 6: Schematic representation of a 1-dimensional CTMC with 3 states.

## A Primer on Continuous Time Markov Chains

A Continuous Time Markov Chain (CTMC) is a right continuous stochastic process $\{ x _ { t } \} _ { t \in [ 0 , T ] }$ satisfying the Markov property, with $x _ { t }$ taking values in a discrete state space X . Since the CTMC is Markov, future behaviour of the process depends only on the current state and not the history. A schematic representation of a CTMC path is shown in Figure 6. The process repeatedly transitions from one state to another after having waited in the previous state for a randomly determined amount of time.

A CTMC can be completely characterised by its jumps and holding times. Specifically, the time between each jump or holding time is exponentially distributed with mean $\nu ( x )$ where x is the state in which the process is holding. The next state that is jumped to is drawn from a jump probability distribution $r ( \tilde { x } | x )$ . The holding and jumping procedure is then repeated.

There is an equivalent definition involving the transition rate matrix, $R \in \mathbb { R } ^ { S \times S }$ , that we use in the main paper. The transition rate matrix is defined as

$$
R ( \tilde { x } , x ) = \operatorname* { l i m } _ { \Delta t  0 } \frac { q _ { t \mid t - \Delta t } ( x \mid \tilde { x } ) - \delta _ { x , \tilde { x } } } { \Delta t }\tag{3}
$$

where $R ( \tilde { x } , x )$ is the $( \tilde { x } , x )$ element of the transition rate matrix and $q _ { t | t - \Delta t } ( x | \tilde { x } )$ is the infinitesimal transition probability of being in state x at time t given that the process was in state x˜ at time $t - \Delta t$ Conversely, the CTMC can itself be defined through this infinitesimal transition probability

$$
q _ { t | t - \Delta t } ( x | \tilde { x } ) = \delta _ { x , \tilde { x } } + R ( \tilde { x } , x ) \Delta t + o ( \Delta t )\tag{4}
$$

where $o ( \Delta t )$ represents terms that tend to zero at a faster rate than $\Delta t$ . From this definition of the transition rate matrix, we can infer the following properties:

$$
\begin{array} { r } { R ( \tilde { x } , x ) \geq 0 \quad \mathrm { f o r } \quad \tilde { x } \neq x , \qquad R ( x , x ) \leq 0 , \qquad R ( x , x ) = - \sum _ { x ^ { \prime } \neq x } R ( x , x ^ { \prime } ) } \end{array}\tag{5}
$$

$R ( \tilde { x } , x )$ is the rate at which probability mass moves from state x˜ to x. $R ( x , x )$ is the total rate at which probability mass moves out of state x and is thus negative.

In the time-homogeneous case, R has simple relations to the jump and holding time definitions.

$$
\nu ( x ) = - \frac { 1 } { R ( x , x ) } r ( \tilde { x } | x ) = ( 1 - \delta _ { \tilde { x } , x } ) \frac { R ( x , \tilde { x } ) } { - R ( x , x ) }
$$

In the time-inhomogeneous case, our transition rate matrix will now depend on time, $R _ { t } .$ , and these simple relations to the jump and holding time definition do not hold. However, $R _ { t }$ will still follow equations (3), (4) and (5).

The CTMC transition probabilities satisfy the Kolmogorov forward and backward equations. For $t > s ,$

Kolmogorov forward equation

$$
\partial _ { t } q _ { t | s } ( x | \tilde { x } ) = \sum _ { y } q _ { t | s } ( y | \tilde { x } ) R _ { t } ( y , x )
$$

Kolmogorov backward equation

$$
\partial _ { s } q _ { t | s } ( x | \tilde { x } ) = - \sum _ { y } R _ { s } ( \tilde { x } , y ) q _ { t | s } ( x | y )
$$

The Kolmogorov forward equation also gives us a differential equation for the marginals of the CTMC.

$$
\partial _ { t } q _ { t } ( x ) = \sum _ { y } q _ { t } ( y ) R _ { t } ( y , x ) .
$$

Exponential and Poisson Random Variables In the time homogeneous case, holding times are exponentially distributed with mean $\nu ( x ) = - 1 / R ( x , x )$ . The tau-leaping algorithm relies on the fact that the number of events in interval [0, t] is Poisson distributed with mean $\mathbf { \widetilde { \Gamma } } _ { \nu } ^ { 1 } t$ when the inter-event times are exponentially distributed with mean ν.

## B Proofs

## B.1 Proof of Proposition 1

Proof. We recall that a process $\{ x _ { t } \} _ { t \in [ 0 , T ] }$ taking values in X is called a CTMC if it is rightcontinuous and satisfies the Markov property. Denote $\{ y _ { t } \} _ { t \in [ 0 , T ] } = \{ x _ { T - t } \} _ { t \in [ 0 , T ] }$ except at the jump times of the forward process $\tau _ { n }$ with $n \in \mathbb { N }$ , where $y _ { T - \tau _ { n } } = x _ { \tau _ { n } } ^ { - } = \operatorname* { l i m } _ { t \leq \tau _ { n } , t  \tau _ { n } } x _ { t }$ . Hence, $\{ y _ { t } \} _ { t \in [ 0 , T ] }$ is almost surely equal to $\{ x _ { T - t } \} _ { t \in [ 0 , T ] }$ and is right-continuous. Since the Markov property is symmetric, we get that $\{ y _ { t } \} _ { t \in [ 0 , T ] }$ is a CTMC. We now compute its transition matrix. Let $x , \tilde { x } \in \mathcal { X }$ with $x \neq \tilde { x }$ , using the Kolmogorov forward equation, we have

$$
\begin{array} { r } { \partial _ { t } p _ { t | s } ( \tilde { x } | x ) = \sum _ { y \in \mathcal { X } } p _ { t | s } ( y | x ) \hat { R } _ { T - t } ( y , \tilde { x } ) , } \end{array}
$$

where $\{ p _ { t | s } , s , t \in [ 0 , T ] , t > s \}$ is the transition probability system associated with $\{ y _ { t } \} _ { t \in [ 0 , T ] }$ and $\{ \hat { R } _ { T - t } \} _ { t \in [ 0 , T ] }$ is the transition rate matrix associated with $\{ y _ { t } \} _ { t \in [ 0 , T ] }$ . Note that

$$
\begin{array} { r l } { \mathsf { \Pi } p _ { t \mid s } ( x = j \vert \tilde { x } = i ) = \mathbb { P } ( \boldsymbol { y } _ { t } = j \mid \boldsymbol { y } _ { s } = i ) } & { } \\ { \mathrm { ~ } } & { = \mathbb { P } ( \boldsymbol { x } _ { T - t } = j \mid \boldsymbol { x } _ { T - s } = i ) } \\ { \mathrm { ~ } } & { = \mathbb { P } ( \boldsymbol { x } _ { T - s } = i \vert \boldsymbol { x } _ { T - t } = j ) \frac { \mathbb { P } ( \boldsymbol { x } _ { T - t } = j ) } { \mathbb { P } \left( \boldsymbol { x } _ { T - s } = i \right) } } \\ { \mathrm { ~ } } & { = q _ { T - s \mid T - t } ( \tilde { x } = i \vert \boldsymbol { x } = j ) \frac { q _ { T - t } ( x = j ) } { q _ { T - s } ( \tilde { x } = i ) } } \end{array}
$$

where $\{ q _ { t | s } , s , t \in [ 0 , T ] , t > s \}$ is the transition probability system associated with $\{ x _ { t } \} _ { t \in [ 0 , T ] }$ and $\{ q _ { t } , t \in [ 0 , T ] \}$ are the marginals of $\{ x _ { t } \} _ { t \in [ 0 , T ] }$ . Now, writing the backward Kolmogorov equation for $\{ x _ { t } \} _ { t \in [ 0 , T ] }$

$$
\begin{array} { r } { \partial _ { s } q _ { t | s } ( \tilde { x } | x ) = - \sum _ { y \in \mathcal { X } } R _ { s } ( x , y ) q _ { t | s } ( \tilde { x } | y ) } \end{array}
$$

Re-labeling the time indices we obtain,

$$
\begin{array} { r l } & { \partial _ { T - t } q _ { T - s | T - t } ( \tilde { x } | x ) = - \sum _ { y \in \mathcal { X } } R _ { T - t } ( x , y ) q _ { T - s | T - t } ( \tilde { x } | y ) } \\ & { \qquad \partial _ { t } q _ { T - s | T - t } ( \tilde { x } | x ) = \sum _ { y \in \mathcal { X } } R _ { T - t } ( x , y ) q _ { T - s | T - t } ( \tilde { x } | y ) } \end{array}
$$

Letting $s \to t$ and using that lim $\iota _ { s  t } q _ { T - s | T - t } ( x | \tilde { x } ) = 0 ,$ , we get that

$$
\begin{array} { r l } { \hat { R } _ { T - t } ( x , \tilde { x } ) = \underset { s  t } { \mathrm { l i m } } \partial _ { t } p _ { t \mid s } ( \tilde { x } \mid x ) } & { } \\ & { = \underset { s  t } { \mathrm { l i m } } \partial _ { t } ( q _ { T - s \mid T - t } ( x \mid \tilde { x } ) \frac { q _ { T - t } ( \tilde { x } ) } { q _ { T - s } ( x ) } ) } \\ & { = \underset { s  t } { \mathrm { l i m } } [ \partial _ { t } ( q _ { T - s \mid T - t } ( x \mid \tilde { x } ) ) \frac { q _ { T - t } ( \tilde { x } ) } { q _ { T - s } ( x ) } + q _ { T - s \mid T - t } ( x \mid \tilde { x } ) \frac { \partial _ { t } q _ { T - t } ( \tilde { x } ) } { q _ { T - s } ( x ) } ] } \\ & { = \underset { s  t } { \mathrm { l i m } } \partial _ { t } ( q _ { T - s \mid T - t } ( x \mid \tilde { x } ) ) \frac { q _ { T - t } ( \tilde { x } ) } { q _ { T - s } ( x ) } } \\ & { = R _ { T - t } ( \tilde { x } , x ) \frac { q _ { T - t } ( \tilde { x } ) } { q _ { T - t } ( x ) } } \end{array}
$$

Re-labeling the time-indices on the rate matrices, we obtain

$$
\hat { R } _ { t } ( x , \tilde { x } ) = R _ { t } ( \tilde { x } , x ) \frac { q _ { t } ( \tilde { x } ) } { q _ { t } ( x ) }
$$

Now we write the marginal ratio $\frac { q _ { t } ( \tilde { x } ) } { q _ { t } ( x ) }$ in a different form

$$
\begin{array} { r l } & { \frac { \displaystyle q _ { t } ( \tilde { x } ) } { \displaystyle q _ { t } ( x ) } = \sum _ { x _ { 0 } } \frac { p _ { \mathrm { d a t a } } ( x _ { 0 } ) } { \displaystyle q _ { t } ( x ) } q _ { t | 0 } ( \tilde { x } | x _ { 0 } ) } \\ & { \qquad = \sum _ { x _ { 0 } } \frac { q _ { 0 | t } ( x _ { 0 } | x ) } { q _ { t | 0 } ( x | x _ { 0 } ) } q _ { t | 0 } ( \tilde { x } | x _ { 0 } ) . } \end{array}
$$

Substituting in this form for the marginal ratio concludes the proof.

## B.2 Proof of Proposition 2

In this section, we detail two proofs for Proposition 2. The first is a formal proof using results from stochastic processes. We then provide a second informal proof for the same result to gain intuition into the $\mathcal { L } _ { \mathrm { C T } }$ objective that only relies on elementary results from CTMCs.

## Proof 1 - Stochastic Processes

Proof. Let us write Q for the path measure of the forward CTMC with rate matrix $R _ { t } , \hat { \mathbb { Q } }$ for the path measure of its exact time reversal and $\mathbb { P } ^ { \theta }$ for the path measure of the approximate reverse process with rate matrix $\hat { R } _ { t } ^ { \theta }$ . Also, we use superscripts to notate conditioning on the starting point, for example $\mathbb { Q } ^ { x _ { 0 } }$ denotes the path measure of the forward process conditioned to start in $x _ { 0 }$

With this notation, we have

$$
\begin{array} { r l } & { - \log p _ { 0 } ^ { \theta } ( x _ { 0 } ) = - \log \displaystyle \int p _ { \mathrm { r e f } } ( \mathrm { d } x _ { T } ) \int _ { \{ \hat { W } _ { T } = x _ { 0 } \} } \mathbb { P } ^ { \theta , x _ { T } } ( \mathrm { d } w ) } \\ & { \quad \quad \quad = - \log \displaystyle \int q _ { T | 0 } ( \mathrm { d } x _ { T } ) \int _ { \{ \hat { W } _ { T } = x _ { 0 } \} } \hat { Q } ^ { x _ { T } } ( \mathrm { d } w ) \frac { \mathrm { d } p _ { \mathrm { r e f } } } { \mathrm { d } q _ { T | 0 } } ( x _ { T } ) \frac { \mathrm { d } \mathbb { P } ^ { \theta , x _ { T } } } { \mathrm { d } \hat { Q } ^ { x _ { T } } } ( w ) } \\ & { \quad \quad \quad = - \log \displaystyle \int q _ { T | 0 } ( \mathrm { d } x _ { T } ) \int \hat { \Theta } ( \mathrm { d } w | \hat { W } _ { 0 } = x _ { T } , \hat { W } _ { T } = x _ { 0 } ) \frac { \mathrm { d } p _ { \mathrm { r e f } } } { \mathrm { d } q _ { T | 0 } } ( x _ { T } ) \frac { \mathrm { d } \mathbb { P } ^ { \theta , x _ { T } } } { \mathrm { d } \hat { Q } ^ { x _ { T } } } ( w ) \mathbb { Q } ^ { x _ { T } } \{ \hat { W } _ { 0 } = x _ { 0 } \} } \\ & { \quad \quad \quad \le \displaystyle \int q _ { T | 0 } ( \mathrm { d } x _ { T } ) \int \hat { \Theta } ( \mathrm { d } w | \hat { W } _ { 0 } = x _ { T } , \hat { W } _ { T } = x _ { 0 } ) \left\{ - \log \frac { \mathrm { d } \mathbb { P } ^ { \theta , x _ { T } } } { \mathrm { d } \hat { Q } ^ { x _ { T } } } ( w ) \right\} + C , } \end{array}
$$

where $\mathbb { P } ^ { \theta } , \hat { \mathbb { Q } }$ run in the reverse time direction. Writing $\hat { W } _ { s }$ for a reverse path and integrating wrt $p _ { \mathrm { d a t a } } ( \mathrm { d } x _ { 0 } )$ we have

$$
\begin{array} { r l } & { \displaystyle \int p _ { \mathrm { d a t a } } ( x _ { 0 } ) [ - \log p _ { 0 } ^ { \theta } ( x _ { 0 } ) ] \le \int p _ { \mathrm { d a t a } } ( x _ { 0 } ) \int q _ { T | 0 } ( \mathrm { d } x _ { T } ) \int \hat { \mathbb { Q } } ( \mathrm { d } \hat { W } | \hat { W } _ { 0 } = x _ { T } , \hat { W } _ { T } = x _ { 0 } ) } \\ & { \qquad \times \left\{ \int _ { s = 0 } ^ { T } \hat { R } _ { T - s } ^ { \theta } ( \hat { W } _ { s } ) \mathrm { d } s - \displaystyle \sum _ { s : \hat { W } _ { s } \ \ne \hat { W } _ { s } } \log \mathbb { P } _ { T - s } ^ { \theta } ( \hat { W } _ { s } | \hat { W } _ { s - } ) R _ { T - s } ^ { \theta } ( \hat { W } _ { s - } ) \right\} + C , } \end{array}
$$

where $\hat { R } _ { t } ^ { \theta } ( x )$ is shorthand for $- \hat { R } _ { t } ^ { \theta } ( x , x )$

When $x _ { 0 } ~ \sim ~ p _ { \mathrm { d a t a } } , x _ { T } ~ \sim ~ q _ { T | 0 } ( \cdot | x _ { 0 } ) , \hat { W } ~ \sim ~ \hat { \mathbb { Q } } ( \mathrm { d } W | \hat { W } _ { 0 } ~ = ~ x _ { T } , \hat { W } _ { T } ~ = ~ x _ { 0 } )$ , the reverse path is distributed according to $p _ { \mathrm { d a t a } } ( \mathrm { d } x _ { 0 } ) \mathbb { Q } _ { x _ { 0 } } ( \mathrm { d } W )$ and therefore $( \hat { W } _ { s - } , \hat { W } _ { s } )$ is distributed like $( W _ { T - s } , W _ { ( T - s ) - } )$ and thus we have

$$
\begin{array} { r l } & { \int p _ { \mathrm { f a t a } } ( x _ { 0 } ) [ - \log p _ { 0 } ^ { \theta } ( x _ { 0 } ) ] } \\ & { \leq \int p _ { \mathrm { f a t a } } ( x _ { 0 } ) \Omega _ { x _ { 0 } } ( \mathrm { d } W ) \left\{ \int _ { s = 0 } ^ { T } \hat { R } _ { T - s } ^ { \theta } ( W _ { ( T - s ) - } ) \mathrm { d } s - \displaystyle \sum _ { s : W _ { ( T - s ) - } \neq W _ { T - s } } \log \mathbb { P } _ { T - s } ^ { \theta } ( W _ { ( T - s ) - } | W _ { T - s } ) \hat { R } _ { T - s } ^ { \theta } ( W _ { T - s } ) \right\} + C } \end{array}
$$

Using Dynkin’s lemma and the fact that $\mathbb { P } _ { t } ^ { \theta } ( x | y ) \hat { R } _ { t } ^ { \theta } ( y ) = \hat { R } _ { t } ^ { \theta } ( y , x )$ we can re-expresss this final line as

$$
\begin{array} { l } { { \displaystyle = \iint _ { s = 0 } ^ { T } q _ { T - s } ( \mathrm { d } x ) \left\{ \displaystyle \sum _ { z \neq x } \hat { h } _ { T - s } ^ { \theta } ( x , z ) - \sum _ { z \neq x } { \cal R } _ { T - s } ( x , z ) \displaystyle \sum _ { z \neq x } \sum _ { R - s } ( x , y ) \log \hat { h } _ { T - s } ^ { \theta } ( y , x ) \right\} } } \\ { { \displaystyle = \iint _ { s = 0 } ^ { T } q _ { T - s } ( \mathrm { d } x ) r _ { T - s } ( \mathrm { d } y | x ) \left\{ \displaystyle \sum _ { z \neq x } \hat { h } _ { T - s } ^ { \theta } ( x , z ) - \sum _ { z \neq x } { \cal R } _ { T - s } ( x , z ) \log \hat { h } _ { T - s } ^ { \theta } ( y , x ) \right\} } } \\ { { \displaystyle = \iint _ { s = 0 } ^ { T } q _ { s } ( \mathrm { d } x ) r _ { s } ( \mathrm { d } y | x ) \left\{ \displaystyle \sum _ { z \neq x } \hat { h } _ { s } ^ { \theta } ( x , z ) - \sum _ { z \neq x } { \cal R } _ { s } ( x , z ) \log \hat { h } _ { s } ^ { \theta } ( y , x ) \right\} } } \end{array}
$$

which rearranges to give the continuous time ELBO in the form of Proposition 2.

## Proof 2 - Limit of Discrete Time ELBO

Proof. Consider a partitioning of $[ 0 , T ] , 0 = t _ { 0 } < t _ { 1 } < \dots < t _ { k - 1 } < t _ { k } < t _ { k + 1 } < \dots < t _ { K - 1 } <$ $t _ { K } = T$ . Let $t _ { k } - t _ { k - 1 } = \Delta t$ for all k. In subscripts we use k as a shorthand for $t _ { k }$ when this does not cause confusion. Considering a CTMC with this time partitioning converts the problem into a discrete time Markov Chain with forward transition kernel, $q _ { k + 1 | k } { \left( x _ { k + 1 } | x _ { k } \right) }$ and parameterized reverse kernel, $p _ { k | k + 1 } ^ { \theta } ( x _ { k } | x _ { k + 1 } )$ . Therefore, we can write the negative ELBO in its discrete time form, $\mathcal { L } _ { \mathrm { D T } }$

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } _ { \mathrm { D T } } ( \theta ) = \mathbb { E } _ { p _ { \mathrm { d a t a } } ( x _ { 0 } ) } \Big [ \mathrm { K L } ( q _ { K | 0 } ( x _ { K } | x _ { 0 } ) | | p _ { \mathrm { r e f } } ( x _ { K } ) ) - \mathbb { E } _ { q _ { 1 | 0 } ( x _ { 1 } | x _ { 0 } ) } \left[ \log p _ { 0 | 1 } ^ { \theta } ( x _ { 0 } | x _ { 1 } ) \right] } \ ~ } \\ { { \displaystyle ~ + \sum _ { k = 1 } ^ { K - 1 } \mathbb { E } _ { q _ { k + 1 | 0 } ( x _ { k + 1 } | x _ { 0 } ) } \left[ \mathrm { K L } ( q _ { k | k + 1 , 0 } ( x _ { k } | x _ { k + 1 } , x _ { 0 } ) | | p _ { k | k + 1 } ^ { \theta } ( x _ { k } | x _ { k + 1 } ) ) \right] \Big ] } } \end{array}
$$

In the following, we will write the transition kernels in terms of the CTMC rate matrices and take the limit as $\Delta t \to 0$ to obtain a continuous time negative ELBO.

First, consider one item from the inner sum of ${ \mathcal { L } } _ { \mathrm { D T } }$

$$
\begin{array} { r l } & { L _ { k } = \mathbb { E } _ { p _ { \mathrm { d a t a } } ( x _ { 0 } ) q _ { k + 1 \mid 0 } ( x _ { k + 1 } \mid x _ { 0 } ) } \left[ { \mathrm { K L } } \big ( q _ { k \mid k + 1 , 0 } ( x _ { k } \mid x _ { k + 1 } , x _ { 0 } ) \vert \vert p _ { k \mid k + 1 } ^ { \theta } ( x _ { k } \vert x _ { k + 1 } ) \big ) \right] } \\ & { \quad = - \mathbb { E } _ { p _ { \mathrm { d a t a } } ( x _ { 0 } ) q _ { k + 1 \mid 0 } ( x _ { k + 1 } \mid x _ { 0 } ) q _ { k \mid k + 1 , 0 } ( x _ { k } \mid x _ { k + 1 } , x _ { 0 } ) } \left[ \log p _ { k \mid k + 1 } ^ { \theta } ( x _ { k } \vert x _ { k + 1 } ) \right] + C } \\ & { \quad = - \mathbb { E } _ { q _ { k } ( x _ { k } ) q _ { k + 1 \mid k } ( x _ { k + 1 } \mid x _ { k } ) } \left[ \log p _ { k \mid k + 1 } ^ { \theta } ( x _ { k } \vert x _ { k + 1 } ) \right] + C } \end{array}
$$

where we have absorbed terms that do not depend on θ into C. We now write $p _ { k | k + 1 } ^ { \theta } ( x _ { k } | x _ { k + 1 } )$ in terms of $\hat { R } _ { k } ^ { \theta }$

$$
p _ { k | k + 1 } ^ { \theta } ( x _ { k } | x _ { k + 1 } ) = \delta _ { x _ { k } , x _ { k + 1 } } + \hat { R } _ { k } ^ { \theta } ( x _ { k + 1 } , x _ { k } ) \Delta t + o ( \Delta t )
$$

$$
\begin{array} { r l } & { \log p _ { k | k + 1 } ^ { \theta } ( x _ { k } | x _ { k + 1 } ) = \log \Big ( \delta _ { x _ { k } , x _ { k + 1 } } + \hat { R } _ { k } ^ { \theta } ( x _ { k + 1 } , x _ { k } ) \Delta t + o ( \Delta t ) \Big ) } \\ & { \qquad = \delta _ { x _ { k } , x _ { k + 1 } } \log \Big ( 1 + \hat { R } _ { k } ^ { \theta } ( x _ { k } , x _ { k } ) \Delta t + o ( \Delta t ) \Big ) } \\ & { \qquad + \left( 1 - \delta _ { x _ { k } , x _ { k + 1 } } \right) \log \Big ( \hat { R } _ { k } ^ { \theta } ( x _ { k + 1 } , x _ { k } ) \Delta t + o ( \Delta t ) \Big ) } \\ & { \qquad = \delta _ { x _ { k } , x _ { k + 1 } } \Big ( \hat { R } _ { k } ^ { \theta } ( x _ { k } , x _ { k } ) \Delta t + o ( \Delta t ) \Big ) } \\ & { \qquad + \left( 1 - \delta _ { x _ { k } , x _ { k + 1 } } \right) \log \Big ( \hat { R } _ { k } ^ { \theta } ( x _ { k + 1 } , x _ { k } ) \Delta t + o ( \Delta t ) \Big ) } \end{array}\tag{6}
$$

where on the last line we have used the series expansion for $\begin{array} { r } { \log ( 1 + z ) = z - \frac { z ^ { 2 } } { 2 } + o ( z ^ { 2 } ) } \end{array}$ valid for $\vert z \vert \leq 1 , z \neq - 1$ . For any finite $R _ { k } ^ { \theta } ( x _ { k } , x _ { k } )$ , ∆t can be taken small enough such that the series expansion holds. We now substitute this form for log $p _ { k | k + \cdot } ^ { \theta }$ into $L _ { k }$ and further write the expectation over $q _ { k + 1 | k } ( x _ { k + 1 } | x _ { k } ) = \delta _ { x _ { k } , x _ { k + 1 } } + R _ { k } ( x _ { k } , x _ { k + 1 } ) \Delta t + o ( \Delta t )$ as an explicit sum.

$$
\begin{array} { c l } { \displaystyle L _ { k } = - \mathbb { E } _ { q _ { k } ( x _ { k } ) } \Bigg [ \sum _ { x _ { k + 1 } } \Bigg \{ \Big [ \delta _ { x _ { k } , x _ { k + 1 } } + R _ { k } ( x _ { k } , x _ { k + 1 } ) \Delta t + o ( \Delta t ) \Big ] \times } \\ { \displaystyle \Big [ \delta _ { x _ { k } , x _ { k + 1 } } \Big ( \hat { R } _ { k } ^ { \theta } ( x _ { k } , x _ { k } ) \Delta t + o ( \Delta t ) \Big ) } \\ { \displaystyle + \big ( 1 - \delta _ { x _ { k } , x _ { k + 1 } } \big ) \log \Big ( \hat { R } _ { k } ^ { \theta } ( x _ { k + 1 } , x _ { k } ) \Delta t + o ( \Delta t ) \Big ) \Big ] \Bigg \} \Bigg ] + { \cal C } } \end{array}
$$

$$
\begin{array} { r l r }   { L _ { k } = - \mathbb { E } _ { q _ { k } ( x _ { k } ) } \Bigg [ \sum _ { x _ { k + 1 } } \Bigg \{ \delta _ { x _ { k } , x _ { k + 1 } } \hat { R } _ { k } ^ { \theta } ( x _ { k } , x _ { k } ) \Delta t } \\ & { } & { \quad + ( 1 - \delta _ { x _ { k } , x _ { k + 1 } } ) R _ { k } ( x _ { k } , x _ { k + 1 } ) \Delta t \times } \\ & { } & { \quad \log \Big ( \hat { R } _ { k } ^ { \theta } ( x _ { k + 1 } , x _ { k } ) \Delta t + o ( \Delta t ) \Big ) + o ( \Delta t ) \Bigg \} \Bigg ] + C } \end{array}
$$

We can isolate $\hat { R } _ { k } ^ { \theta }$ within the log through the following re-arrangement

$$
\begin{array} { r l } & { \Delta t \log \Big ( \hat { R } _ { k } ^ { \theta } ( x _ { k + 1 } , x _ { k } ) \Delta t + o ( \Delta t ) \Big ) } \\ & { = \Delta t \log \Delta t + \Delta t \log \Big ( \hat { R } _ { k } ^ { \theta } ( x _ { k + 1 } , x _ { k } ) + o ( 1 ) \Big ) } \\ & { = \Delta t \log \Delta t + \Delta t \log \big ( 1 + o ( 1 ) \big ) + \Delta t \log \Big ( \hat { R } _ { k } ^ { \theta } ( x _ { k + 1 } , x _ { k } ) \Big ) } \end{array}
$$

where the first two terms are independent of θ and tend to 0 as $\Delta t  0$ . Note that we assume $\hat { R } _ { k } ^ { \theta } ( x _ { k + 1 } , x _ { k } ) > 0$ for $x _ { k + 1 } \neq x _ { k }$ pairs which have $R _ { k } ( x _ { k } , x _ { k + 1 } ) > 0$ . This assumption is valid because, for $x _ { k + 1 } \neq x _ { k }$ , we have

$$
\hat { R } _ { k } ^ { \theta } ( x _ { k + 1 } , x _ { k } ) = R _ { k } ( x _ { k } , x _ { k + 1 } ) \sum _ { x _ { 0 } } \frac { q _ { k | 0 } ( x _ { k } | x _ { 0 } ) } { q _ { k | 0 } ( x _ { k + 1 } | x _ { 0 } ) } p _ { 0 | k } ^ { \theta } ( x _ { 0 } | x _ { k + 1 } )
$$

and we assume $p _ { 0 | k } ^ { \theta } ( x _ { 0 } | x _ { k + 1 } ) > 0$ which is valid when we parameterize $p _ { 0 | k } ^ { \theta }$ with a softmax output. We assume an irreducible Markov chain, hence $q _ { k | 0 } > 0$ for $t _ { k } > 0$

With this re-arrangement, and absorbing constant terms into C, we obtain

$$
\begin{array} { l } { \displaystyle { L _ { k } = - \mathbb { E } _ { q _ { k } ( x _ { k } ) } \Bigg [ \sum _ { x _ { k + 1 } } \Bigg \{ \delta _ { x _ { k } , x _ { k + 1 } } \hat { R } _ { k } ^ { \theta } ( x _ { k } , x _ { k } ) \Delta t } \\ { \displaystyle \qquad + ( 1 - \delta _ { x _ { k } , x _ { k + 1 } } ) R _ { k } ( x _ { k } , x _ { k + 1 } ) \Delta t \log \Big ( \hat { R } _ { k } ^ { \theta } ( x _ { k + 1 } , x _ { k } ) \Big ) } \\ { \displaystyle \qquad + o ( \Delta t ) \Bigg \} \Bigg ] + C } } \end{array}
$$

$$
L _ { k } = - \mathbb { E } _ { q _ { k } ( x _ { k } ) } \left[ \hat { R } _ { k } ^ { \theta } ( x _ { k } , x _ { k } ) \Delta t + \sum _ { x _ { k + 1 } \neq x _ { k } } R _ { k } ( x _ { k } , x _ { k + 1 } ) \Delta t \log \hat { R } _ { k } ^ { \theta } ( x _ { k + 1 } , x _ { k } ) + o ( \Delta t ) \right]
$$

The second term can be re-written so that it is more efficient to approximate with Monte Carlo. Currently the denoising model $p _ { 0 | k } ^ { \theta }$ has to be evaluated for each term in the sum $\textstyle \sum _ { x _ { k + 1 } \neq x _ { k } }$ which would require multiple forward passes of the neural network. We can instead create a new probability distribution to sample from as follows. Define

$$
r _ { k } ( x _ { k + 1 } | x _ { k } ) = ( 1 - \delta _ { x _ { k } , x _ { k + 1 } } ) \frac { R _ { k } ( x _ { k } , x _ { k + 1 } ) } { \mathcal { Z } ^ { k } ( x _ { k } ) }
$$

where

$$
\mathcal { Z } ^ { k } ( x _ { k } ) = \sum _ { x _ { k + 1 } ^ { \prime } \neq x _ { k } } R _ { k } ( x _ { k } , x _ { k + 1 } ^ { \prime } )
$$

So we now have

$$
L _ { k } = - \mathbb { E } _ { q _ { k } ( x _ { k } ) r _ { k } ( x _ { k + 1 } | x _ { k } ) } \left[ \hat { R } _ { k } ^ { \theta } ( x _ { k } , x _ { k } ) \Delta t + \mathcal { Z } ^ { k } ( x _ { k } ) \Delta t \log \hat { R } _ { k } ^ { \theta } ( x _ { k + 1 } , x _ { k } ) + o ( \Delta t ) \right]
$$

Examining the other terms in $\mathcal { L } _ { \mathrm { D T } }$ we have $\mathbb { E } _ { p _ { \mathrm { d a t a } } ( x _ { 0 } ) } \left[ { \mathrm { K L } } ( q _ { K | 0 } ( x _ { K } | x _ { 0 } ) | | p _ { \mathrm { r e f } } ( x _ { K } ) ) \right]$ which does not depend on θ and $\mathbb { E } _ { q _ { 1 | 0 } ( x _ { 1 } | x _ { 0 } ) } \left\lceil \log p _ { 0 | 1 } ^ { \theta } ( x _ { 0 } | x _ { 1 } ) \right\rceil$ which we expand here

$$
\begin{array} { r l } & { \mathbb { E } _ { q _ { 1 } \mathrm {  { \vert 0 } ( x _ { 1 } \vert x _ { 0 } ) } } \left[ \log p _ { 0 \vert 1 } ^ { \theta } ( x _ { 0 } \vert x _ { 1 } ) \right] } \\ & { \qquad = \displaystyle \sum _ { x _ { 1 } } \{ \delta _ { x _ { 1 } , x _ { 0 } } + \Delta t R _ { 1 } ( x _ { 0 } , x _ { 1 } ) + o ( \Delta t ) \} \log p _ { 0 \vert 1 } ^ { \theta } ( x _ { 0 } \vert x _ { 1 } ) } \\ & { \qquad = \log p _ { 0 \vert 1 } ^ { \theta } ( x _ { 0 } \vert x _ { 0 } ) + \Delta t \displaystyle \sum _ { x _ { 1 } } R _ { 1 } ( x _ { 0 } , x _ { 1 } ) \log p _ { 0 \vert 1 } ^ { \theta } ( x _ { 0 } \vert x _ { 1 } ) + o ( \Delta t ) } \\ & { \qquad = \Delta t \hat { R } _ { 1 } ^ { \theta } ( x _ { 0 } , x _ { 0 } ) + \Delta t \displaystyle \sum _ { x _ { 1 } } R _ { 1 } ( x _ { 0 } , x _ { 1 } ) \log p _ { 0 \vert 1 } ^ { \theta } ( x _ { 0 } \vert x _ { 1 } ) + o ( \Delta t ) } \end{array}
$$

where on the final line we have used eq 6. In summary,

$$
\begin{array} { l } { { \displaystyle { \mathcal L } _ { \mathrm { D T } } = \Delta t { \mathbb E } _ { p _ { \mathrm { d a t a } } ( x _ { 0 } ) q _ { 1 } | 0 } ( x _ { 1 } | x _ { 0 } ) ~ \Biggl [ - \hat { R } _ { 1 } ^ { \theta } ( x _ { 0 } , x _ { 0 } ) + \sum _ { x _ { 1 } } R _ { 1 } ( x _ { 0 } , x _ { 1 } ) \log p _ { 0 | 1 } ^ { \theta } ( x _ { 0 } | x _ { 1 } ) \Biggr ] } } \\ { { \displaystyle ~ - \Delta t \sum _ { k = 1 } ^ { K - 1 } { \mathbb E } _ { q _ { k } ( x _ { k } ) r _ { k } ( x _ { k + 1 } | x _ { k } ) } \left[ \hat { R } _ { k } ^ { \theta } ( x _ { k } , x _ { k } ) + { \mathcal Z } ^ { k } ( x _ { k } ) \log \hat { R } _ { k } ^ { \theta } ( x _ { k + 1 } , x _ { k } ) \right] } } \\ { { \displaystyle ~ + o ( \Delta t ) + C } } \end{array}
$$

We now take the limit of $\mathcal { L } _ { \mathrm { D T } }$ as $\Delta t  0$ and $K  \infty$

$$
\operatorname* { l i m } _ { \Delta t \to 0 } \mathcal { L } _ { \mathrm { D T } } = \mathcal { L } _ { \mathrm { C T } } = - \int _ { 0 } ^ { T } \mathbb { E } _ { q _ { t } ( x ) r _ { t } ( \bar { x } | x ) } \left[ \hat { R } _ { t } ^ { \theta } ( x , x ) + \mathcal { Z } ^ { t } ( x ) \log \left( \hat { R } _ { t } ^ { \theta } ( \tilde { x } , x ) \right) \right] d t + C
$$

We can estimate the integral with Monte Carlo if we consider it to be an expectation with respect to a uniform distribution over times (0, T ). We also write $\hat { R } _ { t } ^ { \theta } ( x , x )$ explicitly as the negative off diagonal row sum to obtain

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C T } } ( \theta ) = T \mathbb { E } _ { t \sim \mathcal { U } ( 0 , T ) q _ { t } ( x ) r _ { t } ( \tilde { x } | x ) } \left[ \left\{ \sum _ { x ^ { \prime } \neq x } \hat { R } _ { t } ^ { \theta } ( x , x ^ { \prime } ) \right\} - \mathcal { Z } ^ { t } ( x ) \log \left( \hat { R } _ { t } ^ { \theta } ( \tilde { x } , x ) \right) \right] + C . } \end{array}
$$