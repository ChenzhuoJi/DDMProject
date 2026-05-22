# Simplified and Generalized Masked Diffusion for Discrete Data

Jiaxin Shi∗, Kehang Han∗, Zhe Wang, Arnaud Doucet, Michalis K. Titsias Google DeepMind

## Abstract

Masked (or absorbing) diffusion is actively explored as an alternative to autoregressive models for generative modeling of discrete data. However, existing work in this area has been hindered by unnecessarily complex model formulations and unclear relationships between different perspectives, leading to suboptimal parameterization, training objectives, and ad hoc adjustments to counteract these issues. In this work, we aim to provide a simple and general framework that unlocks the full potential of masked diffusion models. We show that the continuous-time variational objective of masked diffusion models is a simple weighted integral of cross-entropy losses. Our framework also enables training generalized masked diffusion models with state-dependent masking schedules. When evaluated by perplexity, our models trained on OpenWebText surpass prior diffusion language models at GPT-2 scale and demonstrate superior performance on 4 out of 5 zero-shot language modeling tasks. Furthermore, our models vastly outperform previous discrete diffusion models on pixel-level image modeling, achieving 2.75 (CIFAR-10) and 3.40 (ImageNet 64×64) bits per dimension that are better than autoregressive models of similar sizes. Our code is available at https://github.com/google-deepmind/md4.

## 1 Introduction

Since their inception [1, 2, 3], diffusion models have emerged as the workhorse for generative media, achieving state-of-the-art in tasks such as image synthesis [4, 5, 6], audio [7, 8] and video generation [9, 10, 11, 12, 13]. The majority of existing successes are for continuous state space diffusions. While diffusion models have been extended to discrete state spaces [1, 14, 15] and have been successfully applied to applications ranging from graph generation [16], text-to-sound generation [17] or protein design [18], they remain not as widely used as their continuous counterparts as they are not competitive with autoregressive models in important domains such as text modeling. This has motivated the development of continuous space diffusion models where the discrete data are embedded in the Euclidean space [19, 20, 21, 22, 23] or the simplex [24, 25, 26, 27, 28]. We believe that one of the reasons for the limited success of discrete diffusions is that they have been hindered by fairly complex formulations and training objectives. This paper is a step towards closing this gap.

In this work, we focus on “masked” (or “absorbing”) diffusions, a discrete diffusion formulation first presented by Austin et al. [14], and later explored by the literature from various perspectives [29, 30, 31, 32]. We follow here a continuous-time framework which has proven very useful to improve the training and understanding of continuous state space diffusions [see e.g., 3, 33, 34]. We make several technical contributions which simplify the training of these models and improve significantly their performance. Our contributions are as follows:

• Using elementary arguments, we establish several properties for the forward process induced by this model and its corresponding time reversal, improving our understanding of this model class.

• We provide a remarkably simple expression of the Evidence Lower Bound (ELBO) for masked diffusion models, showing that it corresponds to a weighted integral over time of cross-entropy losses. Similarly to continuous space diffusions [33], this objective can be rewritten in terms of signal-to-noise ratio and exhibits invariance properties.

• We develop a unifying understanding of previously proposed continuous-time discrete diffusion models [29, 32, 35], revealing the changes they made to our ELBO objective and/or model parameterization. We show that these changes either lead to expensive model evaluations, or large variance in training, or breaking the consistency between forward and reverse processes.

• On GPT-2 scale text modeling and pixel-level image modeling tasks, masked diffusions trained using our simple ELBO objective outperform previous proposals, leading to the best likelihood and zero-shot transfer performance among discrete diffusion models.

• Finally, based on our simplified masked diffusion formulation, we propose a generalized masked diffusion model that allows state-dependent masking schedules. This generalized masked diffusion model further improves predictive performance measured by test likelihoods.

Concurrent work by Ou et al. [36] and Sahoo et al. [37] derives a similar simplified expression of the ELBO. Ou et al. [36]’s derivation relies on an observation similar to the one we made in Proposition 1.

## 2 Masked Diffusion

Consider a sentence where we progressively replace each word with a special mask token, transforming the sentence into a sequence of masks. Our goal is to train a generative model that reverses this process, effectively turning a sentence of masks back into meaningful text. More formally, assume our data consists of tokens from a finite discrete state space with m possible states, represented by integers $0 , 1 , \ldots , m - 1$ and their corresponding one-hot vectors $e _ { 0 } , e _ { 1 } , \ldots , e _ { m - 1 }$ . To accommodate the masking process, we augment this space with an additional mask state, denoted by the index $m .$ . The masking process transitions each token to the mask state at a random time. This process, known as the forward process, is applied independently to each token (e.g., each word), progressively converting the data into a sequence of mask tokens. By learning to reverse this masking process, we create a generative model capable of producing coherent discrete data.

Discrete-time forward process. We start with the case of a single token and later expand to multiple dimensions. We define the forward process as a Markovian sequence of discrete random variables $x _ { t }$ indexed by time $t ,$ where t runs from 0 to 1. Throughout the work, we abuse the notation such that $x _ { t }$ can be either an integer or its corresponding one-hot vector, whenever it is clear from the context. We divide [0, 1] into $T$ intervals, and let $\bar { s } ( i ) = ( i - 1 ) / T , t ( i ) = i / T$ . Following Austin et al. [14], the state transition between $[ s ( i ) , t ( i ) ]$ is determined by a transition matrix of size $( m + 1 ) \times ( m + 1 ) \colon Q _ { i } = ( 1 - \beta _ { i } ) I + \beta _ { i } { \bf 1 } e _ { m } ^ { \top }$ , where 1 is an all-one vector of size $m + 1 , e _ { m }$ represents a one-hot vector where element at index m is 1. Each entry $[ Q _ { i } ] _ { j k }$ denotes the probability of transition from the state j to the state k:

$$
[ Q _ { i } ] _ { j k } = q ( x _ { t ( i ) } = k | x _ { s ( i ) } = j ) = ( 1 - \beta _ { i } ) \delta _ { j k } + \beta _ { i } \delta _ { k m } .
$$

This means that, with probability $1 - \beta _ { i } , x _ { t ( i ) } = x _ { s ( i ) }$ , otherwise it jumps to the mask state. Given the above transition matrix, the marginal distribution at time $t ( i )$ given $x _ { 0 }$ is

$$
\begin{array} { r } { q ( x _ { t ( i ) } | x _ { 0 } ) = \mathrm { C a t } ( x _ { t ( i ) } ; \bar { Q } _ { i } ^ { \top } x _ { 0 } ) = x _ { 0 } ^ { \top } \bar { Q } _ { i } x _ { t ( i ) } . } \end{array}
$$

Here, we use $\operatorname { C a t } ( x ; p )$ to denote a Categorical distribution where p is the vector of probabilities of being in each category, and $\begin{array} { r } { \bar { Q } _ { i } \triangleq \prod _ { j = 1 } ^ { i } Q _ { j } = \alpha _ { i } I + \left( 1 - \alpha _ { i } \right) \mathbf { 1 } e _ { m } ^ { \top } } \end{array}$ for $\begin{array} { r } { \alpha _ { i } = \prod _ { j = 1 } ^ { i } ( 1 - \beta _ { j } ) } \end{array}$ . We expect $\alpha _ { T }$ to become very small or zero for a sufficiently large $T$ such that $q ( x _ { 1 } | x _ { 0 } )$ for any $x _ { 0 }$ will become a delta mass at the mask state.

Continuous-time limit. We can define a continuous-time forward process by taking a limit of the above discrete-time process. We first specify a continuous function $\beta ( t )$ such that $\bar { \beta _ { i } } = \beta ( t ( i ) ) / T$ We then let $T \to \infty$ in the discrete-time process and compute the limit of $\bar { Q } _ { i }$ (proved in Austin et al. 14, Appendix A.6, see also App. A) as

$$
\bar { Q } ( t ) \triangleq \operatorname* { l i m } _ { T  \infty } \bar { Q } _ { i } = \alpha _ { t } I + ( 1 - \alpha _ { t } ) \mathbf 1 e _ { m } ^ { \top } , \mathrm { ~ w h e r e ~ } \alpha _ { t } \triangleq \exp \Big ( - \int _ { 0 } ^ { t } \beta ( s ) \mathrm { d } s \Big ) ,\tag{1}
$$

<!-- image-->  
Figure 1: Masking schedules in the literature: (Left) $\alpha _ { t } ; ( \mathrm { R i g h t } )$ weight of the cross-entropy loss w.r.t. $t ;$ Equations for these schedules are given in Tab. 4 in Appendix.

so that $q ( x _ { t } | x _ { 0 } ) = \mathrm { C a t } ( x _ { t } ; \bar { Q } ( t ) ^ { \top } x _ { 0 } )$ . For two arbitrary times, $0 \leq s < t \leq 1$ , the transition distribution that is compatible with the above marginal (i.e., $\begin{array} { r } { q ( x _ { t } | x _ { 0 } ) = \sum _ { x _ { s } } q ( x _ { t } | x _ { s } ) q ( x _ { s } | x _ { 0 } ) ) } \end{array}$ is

$$
q ( x _ { t } | x _ { s } ) = \operatorname { C a t } ( x _ { t } ; { \bar { Q } } ( s , t ) ^ { \top } x _ { s } ) , { \mathrm { ~ w h e r e ~ } } { \bar { Q } } ( s , t ) \triangleq { \bar { Q } } ( s ) ^ { - 1 } { \bar { Q } } ( t ) = { \frac { \alpha _ { t } } { \alpha _ { s } } } I + \big ( 1 - { \frac { \alpha _ { t } } { \alpha _ { s } } } \big ) \mathbf { 1 } e _ { m } ^ { \top } .
$$

Note that Austin et al. [14] did not derive this explicit form of transition matrix between two arbitrary time s and t, which appeared later in Zhao et al. [38] concurrently with our work.

Masking schedules. From the definition of $\alpha _ { t } .$ , we have that $\alpha _ { 0 } = 1$ . And similar to the discretetime formulation, we would like $\alpha _ { 1 }$ be zero or very close to zero. We provide a summary of masking schedules from literature that satisfy these properties in Fig. 1. The linear schedule was proposed in Sohl-Dickstein et al. [1] for binary variables and then re-derived by Austin et al. [14] from mutual information for discrete-time models. The geometric schedule $\alpha _ { t }$ is plotted for $\bar { \beta } _ { \mathrm { m i n } } = 1 0 ^ { - 5 }$ and $\bar { \beta } _ { \mathrm { m a x } } = 2 0$ . It was first used for continuous diffusions [3] and then for discrete by Lou et al. [32]. The cosine schedule was originally proposed in MaskGIT [39], an iterative unmasking generative model inspired by diffusion. This schedule has the property of slowing down the unmasking process at the beginning of the reverse generation. Aligning with their observation, we find that this results in a lower chance of conflicting tokens being unmasked simultaneously at the start of generation, thereby enhancing the overall generation quality.

Time reversal of the forward process given $x _ { 0 }$ . The analytic property of our forward process allows to compute many quantities of interest in closed form. One such quantity frequently used in diffusion models is the time reversal of the forward process given $x _ { 0 } \colon q \bigl ( x _ { s } | x _ { t } , x _ { 0 } \bigr )$ for $s \leq t .$ . We derive it in App. C as

$$
 { \boldsymbol { q } } ( x _ { s } | x _ { t } , x _ { 0 } ) =  { \operatorname { C a t } } ( x _ { s } ; \bar { R } ^ { x _ { 0 } } ( t , s ) ^ { \top } x _ { t } ) ,  { \operatorname { w h e r e } } \ \bar { R } ^ { x _ { 0 } } ( t , s ) = I + \frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } }  { \boldsymbol { e } } _ { m } ( x _ { 0 } -  { \boldsymbol { e } } _ { m } ) ^ { \top } .
$$

From the transition matrix $\bar { R } ^ { x _ { 0 } } ( t , s ) \in \mathbb { R } ^ { ( m + 1 ) \times ( m + 1 ) }$ we can see the reverse process conditioned on $x _ { 0 }$ has a very simple logic—if $x _ { t }$ is a mask, with probability $\frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } }$ , it will jump to the state $x _ { 0 }$ at time s, otherwise it will stay masked. Once $x _ { t }$ is unmasked, it remains in the same state until the end.

## 3 Model and Objective

For a discrete-time masked diffusion process, we define our generative model by approximately reversing the forward transitions using a reverse model $p _ { \theta } ( x _ { s } | x _ { t } )$ . One way to define this model is

$$
p _ { \theta } ( x _ { s } | x _ { t } ) \triangleq q ( x _ { s } | x _ { t } , \mu _ { \theta } ( x _ { t } , t ) ) ,\tag{2}
$$

where $\mu _ { \theta } ( x _ { t } , t ) \in \Delta ^ { m + 1 }$ is a probability vector parametrized by a neural network $f _ { \theta }$ with a softmax applied to the output logits (note the m-th output is forced to 0 since the clean data cannot be masks):

$$
\mu _ { \theta } ( x _ { t } , t ) = { \left\{ \begin{array} { l l } { \operatorname { s o f t m a x } ( f _ { \theta } ( x _ { t } , t ) ) } & { x _ { t } = m , } \\ { x _ { t } } & { x _ { t } \neq m . } \end{array} \right. }\tag{3}
$$

This is known as mean-parameterization since it leverages a prediction model for the mean of $x _ { 0 } .$ A matrix-form depiction of $p _ { \theta } ( x _ { s } | x _ { t } )$ is shown in Fig. 7 (right). In fact, we can select a time-invariant parametrization $\mu _ { \theta } ( x _ { t } , t ) = \mu _ { \theta } ( x _ { t } )$ as [36] showed that $p ( x _ { 0 } | x _ { t } )$ given $x _ { t } = x$ is identical for any t.

Besides $p _ { \theta } ( x _ { s } | x _ { t } )$ , we also need to specify $p \big ( x _ { 0 } \vert x _ { t ( 1 ) } \big )$ and the prior distribution $p ( x _ { t ( T ) } ) = p ( x _ { 1 } )$ Following the practice in continuous diffusion models [33], we choose $p ( x _ { 0 } | x _ { t ( 1 ) } ) \propto q ( x _ { t ( 1 ) } | x _ { 0 } )$ And since $q ( x _ { 1 } | x _ { 0 } ) \approx \delta _ { x _ { 1 } , m }$ for any $x _ { 0 }$ as $\alpha _ { 1 } \approx 0$ , we set $p ( x _ { 1 } ) \approx \delta _ { x _ { 1 } , m }$ , see App. E.

We then write out the discrete-time diffusion model objective [1, 2], which is a lower bound of the log marginal likelihood of data $x _ { 0 }$ under the model p (known as the Evidence Lower Bound, or ELBO):

$$
\begin{array} { r } { \log p ( x _ { 0 } ) \geq \mathbb { E } _ { q ( x _ { t ( 1 ) } \mid x _ { 0 } ) } [ \log p ( x _ { 0 } | x _ { t ( 1 ) } ) ] - \mathrm { K L } ( q ( x _ { 1 } | x _ { 0 } ) \| p ( x _ { 1 } ) ) - \mathcal { L } _ { T } , } \end{array}
$$

where $\begin{array} { r } { \mathcal { L } _ { T } = \sum _ { i = 2 } ^ { T } \mathbb { E } _ { q ( x _ { t ( i ) } \mid x _ { 0 } ) } [ \mathrm { K L } ( q ( x _ { s ( i ) } \vert x _ { t ( i ) } , x _ { 0 } ) \Vert p _ { \theta } ( x _ { s ( i ) } \vert x _ { t ( i ) } ) ) ] } \end{array}$ . For the above choices of the prior distribution, the term $\mathrm { K L } ( q ( x _ { 1 } | x _ { 0 } ) | | p ( x _ { 1 } ) )$ becomes zero. Under the reverse model (2), the KL divergence terms in $\mathcal { L } _ { T }$ becomes (proof in App. D)

$$
\mathrm { K L } ( q ( x _ { s } | x _ { t } , x _ { 0 } ) \| p _ { \theta } ( x _ { s } | x _ { t } ) ) = - \frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } } \delta _ { x _ { t } , m } \cdot x _ { 0 } ^ { \top } \log \mu _ { \theta } ( x _ { t } , t ) ,
$$

which is a simple cross-entropy loss between the predicted logits and the clean data. In App. D, we show that $\mathcal { L } _ { T }$ is a Riemann sum and is lower bounded by the corresponding continuous integral:

$$
\mathcal { L } _ { \infty } \triangleq \operatorname* { l i m } _ { T  \infty } \mathcal { L } _ { T } = \int _ { t ( 1 ) } ^ { 1 } \frac { \alpha _ { t } ^ { \prime } } { 1 - \alpha _ { t } } \mathbb { E } _ { q ( x _ { t } | x _ { 0 } ) } [ \delta _ { x _ { t } , m } \cdot x _ { 0 } ^ { \top } \log \mu _ { \theta } ( x _ { t } , t ) ] \mathrm { d } t ,\tag{4}
$$

where $\alpha _ { t } ^ { \prime }$ denotes the derivative of $\alpha _ { t }$ with respect to t. Therefore, we can obtain an ELBO that is tighter than that of any finite T by pushing $T \to \infty$ . This ELBO can be further simplified by letting $t ( 1 )  0 .$ . As a result, $\mathbb { E } _ { q ( x _ { t ( 1 ) } | x _ { 0 } ) } [ \log p ( \bar { x _ { 0 } } | x _ { t ( 1 ) } ) ]$ goes to 0 and the ELBO becomes $- { \mathcal { L } } _ { \infty }$

For continuous state-space diffusions, the ELBO depends on the signal-to-noise ratio (SNR) at its endpoints but is otherwise invariant to the noise schedule [33]. We establish here a similar result for discrete diffusions. Consider choosing $\alpha _ { t } = \sigma ( \lambda _ { t } )$ , where σ represents the sigmoid function $\textstyle \sigma ( x ) = { \frac { 1 } { 1 + e ^ { - x } } }$ . In this context, the log-SNR is defined by $\begin{array} { r } { \lambda _ { t } = \log \frac { \bar { \alpha } _ { t } } { 1 - \alpha _ { * } } = \log \mathrm { - S N R } ( t ) } \end{array}$ . By making a change of variables in (4) to make everything a function of the log-SNR, we obtain

$$
\mathcal { L } _ { \infty } = \int _ { \lambda _ { t ( 1 ) } } ^ { \lambda _ { 1 } } \sigma ( \lambda ) \mathbb { E } _ { \widetilde { q } ( x _ { \lambda } | x _ { 0 } ) } \left[ \delta _ { x _ { \lambda } , m } \cdot x _ { 0 } ^ { \top } \log \widetilde { \mu } _ { \theta } ( x _ { \lambda } , \lambda ) \right] \mathrm { d } \lambda .
$$

where $\tilde { \mu } _ { \boldsymbol { \theta } } ( x , \lambda ) : = \mu _ { \boldsymbol { \theta } } ( x , t )$ and $\tilde { q } ( x _ { \lambda } | x _ { 0 } ) : = q ( x _ { t } | x _ { 0 } )$ for $t = \log { - \mathrm { S N R } ^ { - 1 } ( \lambda ) }$ . This shows that the only effect αt has on the loss is through the values of the SNR at the endpoints. Still, because we draw uniform samples of t to estimate the integral, the choice of masking schedule affects the variance.

Multidimensional data. In the previous sections, $x _ { t }$ was assumed to be a single discrete token. To extend the method to multidimensional data, let $x _ { t }$ be now a sequence $( x _ { t } ^ { ( 1 ) } , x _ { t } ^ { ( 2 ) } , \dots , x _ { t } ^ { ( N ) } )$ ， where each element $x _ { t } ^ { ( n ) }$ represents a discrete token. We select a forward process which factorizes across all N tokens: $\begin{array} { r } { \dot { q } ( x _ { t } | x _ { s } ) = \prod _ { n = 1 } ^ { N } q ( x _ { t } ^ { ( n ) } | x _ { s } ^ { ( n ) } ) } \end{array}$ . As a result, the forward marginals $q ( x _ { t } | x _ { 0 } )$ and reversal $q ( x _ { s } | \boldsymbol { x } _ { t } , \boldsymbol { x } _ { 0 } )$ also factorize. In this case, we define the reverse model as $p _ { \theta } ( x _ { s } | x _ { t } ) \triangleq$ $\begin{array} { r } { \prod _ { n = 1 } ^ { N } q ( x _ { s } ^ { ( n ) } | x _ { t } ^ { ( n ) } , \mu _ { \theta } ^ { ( n ) } ( x _ { t } , t ) ) } \end{array}$ , where $\mu _ { \theta } ( x _ { t } , t )$ is a neural network that takes the full N tokens as input and outputs N probability vectors.2 The n-th output $\mu _ { \theta } ^ { ( n ) } ( x _ { t } , t )$ is a prediction model for $\mathbb { E } [ x _ { 0 } ^ { ( n ) } | x _ { t } ]$ , the mean value of the n-th token. Repeating above derivations gives

$$
\begin{array} { r } { \mathcal { L } _ { \infty } ^ { ( N ) } \triangleq \displaystyle \int _ { 0 } ^ { 1 } \frac { \alpha _ { t } ^ { \prime } } { 1 - \alpha _ { t } } \mathbb { E } _ { q ( x _ { t } | x _ { 0 } ) } \Big [ \sum _ { n : x _ { t } ^ { ( n ) } = m } ( x _ { 0 } ^ { ( n ) } ) ^ { \top } \log \mu _ { \theta } ^ { ( n ) } ( x _ { t } , t ) \Big ] \mathrm { d } t . } \end{array}\tag{5}
$$

We term our simple masked diffusion model trained with loss (5) MD4 (Masked Discrete Diffusion for Discrete Data). A single step of MD4 training algorithm is described in Alg. 1 in Appendix.

## 4 Sampling

We use ancestral sampling from our discrete-time reverse process for generation. We have found this yields slightly higher sample quality compared to other methods such as Euler discretization [29, 32]. For conditional generation tasks such as infilling, we find that the simple approach works best — we keep the conditioning tokens unmasked throughout the generation process. A complete description of the sampling algorithm can be found in Alg. 2 in Appendix.

<!-- image-->

<!-- image-->  
Figure 2: Left: FID evaluation for 50k samples randomly generated from MD4 on pixel-level modeling of ImageNet 64×64 (numbers in Tab. 6). Right: Number of tokens revealed per generation step $( T \stackrel { - } { = } 2 5 6 )$ . Each image consists of 64 × 64 × 3 = 12288 tokens.

Impact of schedules and discretization. For comparing different sampling configurations, we primarily use the FID score [40] on image datasets as our evaluation metric. We favor it over text generative perplexity3 used in prior work [32], as the latter can be misleadingly reduced by lowering sample diversity [41]. We initially trained our model using the linear schedule, which achieves the best final ELBO overall; however, we found that sampling did not perform well with a standard uniform discretization grid $\begin{array} { r } { t ( i ) = \frac { i } { T } } \end{array}$ . We hypothesize that time discretization can lead to conflicts by generating multiple tokens in a single step. We then switched to the cosine schedule (Tab. 4) that slows down unmasking at the beginning of reverse process. This drastically improves the FID on ImageNet 64×64 from 70 to 17 for $T = 2 5 6$ steps (Fig. 2, left). Building on this observation, we suggest using a “cosine” discretization grid for sampling in models trained with a linear schedule:

$$
t ( i ) = \cos { \Big ( } \frac { \pi } { 2 } \big ( 1 - \frac { i } { T } \big ) \Big ) .\tag{6}
$$

This induces the same discretization in $\alpha _ { t }$ as the cosine schedule with a uniform grid, leading to comparable sample quality, as shown in Fig. 2 (left). In Fig. 2 (right), we plot the number of tokens unmasked per step for linear and cosine schedules with a uniform grid. We believe the cosine schedule performs better because it leverages information redundancy: with more tokens revealed, the remaining tokens become more predictable, reducing conflicts when unmasking them in a single step.

Although these findings were originally developed on images, we find them translate well to text (see Fig. 10). we expect other techniques such as top-p sampling [41], classifier-free guidance [42, 43], and predictor-correctors [29, 44] to further improve sample quality of our models. While we reserve these for future work, we note that the JAX [45] implementation of categorical sampling implicitly truncates small probabilities, creating a similar effect to top-p sampling. See App. G for details.

## 5 Relation to Existing Work

We discuss how to unify several existing masked diffusion models using our framework.

Continuous-Time Markov Chains (CTMC). To show the connection with the CTMC view presented in Austin et al. [14], Campbell et al. [29], we can write out the forward and reverse masked diffusion using CTMC machinery. To see this, for a short time $\Delta t ,$ given $x _ { 0 } .$ , the Taylor expansions of our forward and reverse transition matrices at t are

$$
\bar { Q } ( t , t + \Delta t ) = I + Q ( t ) \Delta t + o ( \Delta t ) \quad \mathrm { f o r } \quad Q ( t ) \triangleq \beta ( t ) ( \mathbf { 1 } e _ { m } ^ { \top } - I ) ,\tag{7}
$$

$$
\bar { R } ^ { x _ { 0 } } ( t , t - \Delta t ) = I + R ^ { x _ { 0 } } ( t ) \Delta t + o ( \Delta t ) \quad \mathrm { f o r } \quad R ^ { x _ { 0 } } ( t ) \triangleq - \frac { \alpha _ { t } ^ { \prime } } { 1 - \alpha _ { t } } e _ { m } ( x _ { 0 } - e _ { m } ) ^ { \top } ,\tag{8}
$$

where Q(t) and $R ^ { x _ { 0 } } ( t )$ are known as the transition rate matrices. Austin et al. [14] derived the same Q(t) in App. A.6 of their paper. However, they did not explore the reverse process or a continuous-time objective. Campbell et al. [29] derived an alternative ELBO expression using rate matrices, which Kitouni et al. [46] further simplified for absorbing diffusion. In App. H.1, we show how to recover their expression by separating out a constant from our ELBO expression (4) and applying a discrete “integration-by-part”. A key limitation of their expression is that it needs N evaluations of the prediction model $\mu _ { \boldsymbol { \theta } } ( \cdot , t )$ to compute an inner summation. To circumvent this computational burden, they used a doubly stochastic estimate. However, this leads to significantly higher variance compared to the analytic cross-entropy (4) which only requires one pass of $\mu _ { \theta } ( \cdot , t )$ Please refer to $\mathsf { A p p }$ . H.2 for more details.

Score parameterization. While so far we used a prediction model $\mu _ { \theta } ( x _ { t } , t )$ for the mean of clean data given xt (i.e., mean parameterization), one can choose other ways of parameterizing the reverse model. Lou et al. [32], Benton et al. [35] proposed to parameterize the discrete “score” $\begin{array} { r } { s ( x _ { t } , t ) _ { j } \triangleq \frac { q _ { t } ( j ) } { q _ { t } ( x _ { t } ) } } \end{array}$ and introduced a score-based loss for discrete diffusions. In App. H.3, we provide an alternative derivation of their loss which is simpler. We show the link between score and mean parameterizations through the following proposition.

Proposition 1 (Score Parameterization vs. Mean Parameterization). Let $q _ { t }$ be the marginal distribution of the masked diffusion defined in Sec. 2 at time t. The discrete score $\begin{array} { r } { s ( x _ { t } , t ) _ { j } = \frac { q _ { t } ( j ) } { q _ { t } ( x _ { t } ) } } \end{array}$ for a mask state $x _ { t } = m$ and $j \neq$ m can be expressed as

$$
s ( m , t ) _ { j } = \frac { \alpha _ { t } } { 1 - \alpha _ { t } } \mathbb { E } [ x _ { 0 } | x _ { t } = m ] ^ { \top } e _ { j } , w h i c h s a t i s f i e s \sum _ { j \neq m } s ( m , t ) _ { j } = \frac { \alpha _ { t } } { 1 - \alpha _ { t } } .\tag{9}
$$

Proposition 1 (proved in App. H.3) implies that a reasonable score model for a mask state is

$$
s _ { \theta } ( m , t ) _ { j } = \frac { \alpha _ { t } } { 1 - \alpha _ { t } } \mu _ { \theta } ( m , t ) _ { j } .\tag{10}
$$

Indeed, substituting (10) into the score-based loss of Lou et al. [32], Benton et al. [35] recovers our objective (4). In Lou et al. [32], the score is parameterized as a neural network without enforcing the constraint in (9). This means the learned reverse model can be incompatible with the forward process. We find that our parameterization, which enforces the constraint, leads to more stable training and better results.

Any-order autoregressive models. The continuous-time reverse process of our masked diffusion model can be viewed as an any-order autoregressive model (AO-ARM) [47]. To see this, we reorder the tokens according to the timing of their unmasking events in the reverse process. For all tokens, the cumulative distribution functions (CDFs) of unmasking times $\{ \tau _ { n } \} _ { n = 1 } ^ { N }$ are identical and satisfy $P ( \tau _ { n } \le t ) = P ( x _ { t } ^ { ( n ) } = m ) = 1 - \alpha _ { t }$ . As a result, the ordering is uniformly random across all possible arrangements, and the token prediction during each unmasking event represents a prediction step in AO-ARMs. This connection was initially pointed out in Hoogeboom et al. [48, App. C]. The relation between our simplified ELBO (5) and the AO-ARM objective is independently clarified by Ou et al. [36]. Despite this equivalence, our work demonstrates that the masking schedule $\alpha _ { t }$ introduces a new degree of freedom in the design of such models. Variations in $\alpha _ { t }$ can lead to different distributions of unmasking times, significantly impacting performance in diffusion-style parallel sampling under time discretization, as shown in Fig. 2.

Other related work. Due to space constraint, we defer the discussion on other related work, including MaskGIT [39], discrete flow matching [49], SDDM [30], Blackout diffusion [50] and SUNDAE [51], to App. H.4.

## 6 Generalization to State-dependent Masking Schedules

Consider a scenario where some tokens hold more significance than others and we would like to unmask them earlier in the process. To achieve this, we introduce state-dependent masking schedules, where the probability of unmasking a token depends not only on time, but also on the token’s value.

We first define the forward process for a single token $x _ { t } .$ Let $\alpha _ { t }$ be a $m + 1$ dimensional vector function, i.e., there is a different function $\alpha _ { t , i }$ for each possible value i of the token $x _ { t }$ . Also, by

<!-- image-->  
Figure 3: Iterative unmasking process for an unconditionally generated sample by MD4. This visualization only includes a subsequence from a generated sequence of 1024 tokens. $" ? "$ represents masks. Masked tokens are revealed sequentially: green (steps 500-700), yellow (700-850), and red (850-1000). Additional unconditional generation from MD4 can be found in $\mathrm { A p p }$ . K.5.

vector $\frac { \alpha _ { t } } { \alpha _ { s } }$ we denote the element-wise division of the two vectors. We define the forward transition as $q ( x _ { t } | x _ { s } ) = \mathrm { C a t } ( x _ { t } ; \bar { Q } ( s , t ) ^ { \top } x _ { s } )$ where

$$
\bar { Q } ( s , t ) = \mathrm { d i a g } \Big ( \frac { \alpha _ { t } } { \alpha _ { s } } \Big ) + \Big ( I - \mathrm { d i a g } \Big ( \frac { \alpha _ { t } } { \alpha _ { s } } \Big ) \Big ) { \bf 1 } e _ { m } ^ { \top }
$$

and $\textstyle \operatorname { d i a g } \left( { \frac { \alpha _ { t } } { \alpha _ { s } } } \right)$ is a diagonal matrix with the vector $\frac { \alpha _ { t } } { \alpha _ { s } }$ in its diagonal. The probability of moving from current state $x _ { s }$ to a future state $x _ { t }$ (either the same as $x _ { s }$ or mask) is determined by a state-dependent rate $\begin{array} { r } { \left( \frac { \alpha _ { t } } { \alpha _ { s } } \right) ^ { \top } x _ { s } , } \end{array}$ while the marginal at time s given $x _ { 0 }$ is

$$
\begin{array} { r } { q ( x _ { s } | x _ { 0 } ) = \operatorname { C a t } ( x _ { s } ; \bar { Q } ( s ) ^ { \top } x _ { 0 } ) \quad \mathrm { f o r } \quad \bar { Q } ( s ) = \mathrm { d i a g } ( \alpha _ { s } ) + ( I - \mathrm { d i a g } ( \alpha _ { s } ) ) { \bf 1 } e _ { m } ^ { \top } . } \end{array}
$$

Further, for any time $0 \leq s < t \leq 1$ it holds that $\begin{array} { r } { q ( x _ { t } | \boldsymbol { x } _ { 0 } ) = \sum _ { \boldsymbol { x } _ { s } } q ( x _ { t } | \boldsymbol { x } _ { s } ) q ( x _ { s } | \boldsymbol { x } _ { 0 } ) } \end{array}$ so the above is a valid continuous-time Markov chain.

Given the forward conditionals and marginals, we can now compute the time reversal conditioned on $x _ { 0 }$ . The full form of $q ( x _ { s } | \boldsymbol { x } _ { t } , \boldsymbol { x } _ { 0 } )$ is derived in App. I.1. For $x _ { t } = m$ , we have

$$
\begin{array} { r } { q ( x _ { s } | x _ { t } = m , x _ { 0 } ) = q ( x _ { s } | x _ { t } = m , x _ { 0 } , \qquad ) = \left( \frac { 1 - \alpha _ { s } } { 1 - \alpha _ { t } } \right) ^ { \top } x _ { 0 } e _ { m } ^ { \top } x _ { s } + \left( \frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } } \right) ^ { \top } \qquad x _ { s } . } \end{array}\tag{11}
$$

This suggests that the reverse model given $x _ { t } = m$ can be chosen as $p _ { \theta } ( x _ { s } | x _ { t } = m ) \triangleq q ( x _ { s } | x _ { t } =$ $m , \mu _ { \theta } ( x _ { t } , t )$ , diag $\left( \mu _ { \theta } ( x _ { t } , t ) \right)$ where $\mu _ { \theta } ( x _ { t } , t )$ is a neural network that approximates $\mathbb { E } [ x _ { 0 } \vert x _ { t } ]$ while d $\mathrm { l i a g } ( \mu _ { \theta } ( x _ { t } , t ) )$ approximates $\mathbb { E } [ x _ { 0 } x _ { 0 } ^ { \top } | x _ { t } ] = \mathrm { d i a g } ( \mathbb { E } [ x _ { 0 } | x _ { t } ] )$ . We show in App. I.1 that the negative continuous-time ELBO for the state-dependent rate case is

$$
\mathcal { L } _ { \infty } = \int _ { 0 } ^ { 1 } \Big ( \frac { \alpha _ { t } ^ { \prime } } { 1 - \alpha _ { t } } \Big ) ^ { \top } \mathbb { E } _ { q ( x _ { t } | x _ { 0 } ) } \left[ \delta _ { x _ { t } , m } \cdot ( x _ { 0 } - \mu _ { \theta } ( x _ { t } , t ) + x _ { 0 } x _ { 0 } ^ { \top } \log \mu _ { \theta } ( x _ { t } , t ) ) \right] \mathrm { d } t .\tag{12}
$$

Here, $\alpha _ { t } ^ { \prime }$ is the elementwise derivative of $\alpha _ { t }$ . This generalizes the MD4 loss (4), which is recovered when $\alpha _ { t }$ is a scalar schedule times a vector of ones. For N tokens, the model further generalize similarly to Sec. 3 and the loss is given in (32). We call this generalized model GenMD4.

To learn the token dependent masking schedule using ELBO optimization, we parametrize the $m + 1$ dimensional function $\alpha _ { t }$ using the polynomial schedule (see Fig. 1) as $\alpha _ { t , i } = 1 - t ^ { w _ { i } }$ and optimize each parameter $w _ { i } > 0 . ^ { 4 }$ The value of $w _ { i }$ , through the masking probability $1 - \alpha _ { t , i } ,$ , determines how fast the token with value i jumps to the mask state. Since in the loss (12) the distribution $q ( x _ { t } | x _ { 0 } )$ depends on $\alpha _ { t }$ and thus the vector w, optimizing w poses a discrete gradient estimation problem [see, e.g., 52]. Naive autodiff leads to biased gradients and pushes w towards zero because the gradients cannot propagate through the (discrete) samples drawn from $q ( x _ { t } | x _ { 0 } )$ . To fix this, we used the REINFORCE leave-one-out estimator [53, 54] to compute low-variance unbiased gradients for optimizing w. Details are given in App. I.2.

Table 1: Zero-shot unconditional perplexity on five benchmark datasets from Radford et al. [57]. The numbers for other methods are from Lou et al. [32] except our reimplementation of SEDD Absorb. Our MD4 model achieves the best result on all benchmarks except LAMBADA where it is the second best. ∗The GPT-2 numbers are reported for the GPT-2 checkpoint pretrained on WebText instead of OWT thus is not a direct comparison.
<table><tr><td>Size</td><td>Method</td><td>LAMBADA</td><td>WikiText2</td><td>PTB</td><td>WikiText103</td><td>IBW</td></tr><tr><td rowspan="6">Small</td><td>GPT-2 (WebText)*</td><td>45.04</td><td>42.43</td><td>138.43</td><td>41.60</td><td>75.20</td></tr><tr><td>D3PM</td><td>≤93.47</td><td>77.28</td><td>≤ 200.82</td><td>≤ 75.16</td><td>≤138.92</td></tr><tr><td>Plaid</td><td>≤ 57.28</td><td>51.80</td><td></td><td>≤ 50.86</td><td>≤ 91.12</td></tr><tr><td>SEDD Absorb</td><td>≤ 50.92</td><td>41.84</td><td>W 114.24</td><td>≤40.62</td><td>79.29</td></tr><tr><td>SEDD Absorb (reimpl.)</td><td>≤49.73</td><td>vIVIvIVIN 38.94</td><td>107.54</td><td>≤39.15</td><td>≤ 72.96</td></tr><tr><td>MD4 (Ours)</td><td>≤48.43</td><td> 34.94</td><td>≤ 102.26</td><td>≤ 35.90</td><td>八 68.10</td></tr><tr><td rowspan="4">Medium</td><td>GPT-2 (WebText)*</td><td>35.66</td><td>31.80</td><td>123.14</td><td>31.39</td><td>55.72</td></tr><tr><td>SEDD Absorb</td><td>≤42.77</td><td>≤ 31.04</td><td>≤87.12</td><td>≤ 29.98</td><td>≤ 61.19</td></tr><tr><td>MD4 (Ours)</td><td>≤44.12</td><td>≤ 25.84</td><td>≤ 66.07</td><td>≤ 25.84</td><td>≤ 51.45</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 7 Experiments

## 7.1 Text

Text is natural discrete data with rich structures. For comparison with prior work, we evaluate likelihood on two datasets: text8 [55], a character-level text modeling benchmark, and OpenWebText [56], an open clone of the unreleased WebText dataset used to train GPT-2 [57]. We also assess our model’s performance on downstream tasks by training on FineWeb-Edu [58], a high-quality dataset of fine educational text commonly used by the open-source community for comparing LLMs. Unless otherwise specified, a linear schedule and a cosine sampling grid are employed.

OpenWebText. We train MD4 of GPT-2 small (S) and GPT-2 medium (M) sizes on OpenWeb-Text and evaluate zero-shot perplexity on five benchmark datasets used in Radford et al. [57]. We keep our evaluation setup the same as SEDD [32]. To ensure fair comparison, we reimplemented SEDD in our codebase. Our implementation led to slightly better results than those reported in their paper.

As seen in Tab. 1, our small model outperforms previous best discrete diffusion models on all five tasks. We are also better than GPT-2 on all tasks except LAMBADA where we are the second best method. When scaling up to medium size, MD4 similarly beats SEDD and GPT-2 on 4 out of 5 tasks.

<!-- image-->  
Figure 4: Perplexity on OpenWebText (OWT) validation set during training. The final numbers are reported in Tab. 5 in Appendix.

To confirm that the strong zero-shot performance stems from improved training, we plot perplexity on 2% OpenWebText validation set in Fig. 4. Our models converge faster and have better final likelihoods than prior methods. We also observed that SEDD [32] has training instabilities, likely due to score parameterization breaking consistency between forward and reverse processes (Sec. 5). Although GenMD4 achieves lower perplexity than MD4, we observed that the learned ws can overfit to dataset statistics, making it less effective on zero-shot transfer tasks.

We also assess our models’ generation quality. Fig. 3 shows a randomly selected, notably coherent sample from MD4-medium and its denoising process. Fig. 10 demonstrates MD4’s text infilling ability and highlights a substantial quality gain when transitioning from uniform to cosine discretization (see Sec. 4). Despite MD4’s strong performance on quantitative metrics like generative perplexity, we have placed these results in Appendix Fig. 8 due to the metric’s inherent unreliability, as noted in Sec. 4. We emphasize the more reliable FID-based assessments found in our image experiments.

Table 2: Bits Per Character (BPC) on Text8 test set. All models use standard 12-layer transformers similar to GPT-2 small [57] except Discrete Flow which uses 8 × 3 layers.
<table><tr><td>Method BPC (↓)</td></tr><tr><td>Continuous Diffusion</td></tr><tr><td>Plaid [22] (Our impl.) ≤1.48</td></tr><tr><td>BFN[26] ≤1.41</td></tr><tr><td>Any-orderAutoregressive</td></tr><tr><td>ARDM[48] ≤1.43 MAC [61] ≤1.40</td></tr><tr><td>Autoregressive</td></tr><tr><td>IAF/SCF [62] 1.88</td></tr><tr><td>AR Argmax Flow [15] 1.39</td></tr><tr><td>Discrete Flow [59] 1.23</td></tr><tr><td>Transformer AR [14] 1.23</td></tr><tr><td>Discrete Diffusion ≤1.72</td></tr><tr><td>Mult. Diffusion [15] ≤ 1.61</td></tr><tr><td>D3PMUniform [14]</td></tr><tr><td>D3PMAbsorb[14] ≤1.45</td></tr><tr><td>SEDD Absorb [32] ≤1.39</td></tr><tr><td>MD4 (Ours) ≤1.37 GenMD4 (Ours) ≤ 1.34</td></tr></table>

Table 3: Bits Per Dimension (BPD) on CIFAR-10 test set and Downsampled ImageNet 64×64 [63] validation set. All models in the table are trained without data augmentation.
<table><tr><td>Method Autoregressive</td><td></td><td>#Params</td><td>BPD (↓)</td></tr><tr><td rowspan="8">CI-APIIT</td><td>PixelRNN [63]</td><td></td><td>3.00</td></tr><tr><td>Gated PixelCNN [64]</td><td></td><td>3.03</td></tr><tr><td>PixelCNN++ [65]</td><td>53M</td><td>2.92</td></tr><tr><td>PixelSNAIL [66]</td><td>46M</td><td>2.85</td></tr><tr><td>Image Transformer [67]</td><td></td><td>2.90</td></tr><tr><td>Sparse Transformer [68]</td><td>59M</td><td>2.80</td></tr><tr><td>Discrete Diffusion</td><td></td><td></td></tr><tr><td>D3PM Absorb [14]</td><td>37M</td><td>≤4.40</td></tr><tr><td>D3PM Gauss [14]</td><td>36M</td><td></td><td>3.44</td></tr><tr><td>Campbell et al. [29]</td><td>36M</td><td></td><td>≤3.59</td></tr><tr><td>Campbell et al.[29] Absorb</td><td></td><td>28M</td><td>3.52</td></tr><tr><td>MD4 (Ours)</td><td>28M</td><td></td><td> 2.75</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="5">x4912988444</td><td>Autoregressive</td><td></td><td></td><td></td></tr><tr><td>PixelRNN [63]</td><td></td><td></td><td>3.63</td></tr><tr><td>Gated PixeiCNN [64]</td><td></td><td></td><td>3.57</td></tr><tr><td>Sparse Transformer [68]</td><td>152M</td><td></td><td>3.44</td></tr><tr><td>Routing Transformer [69]</td><td></td><td></td><td>3.43</td></tr><tr><td></td><td>Perceiver AR [68]</td><td>770M</td><td></td><td>3.40</td></tr><tr><td>Discrete Diffusion MD4 (Ours)</td><td></td><td>198M</td><td></td><td>≤3.40</td></tr></table>

Text8. Following prior work [14, 32], we trained masked diffusion models on text8 and evaluate the bits-per-character on the test set (details in App. J.1). As seen in Tab. 2, our models outperform previous discrete and continuous diffusion models, as well as state-of-the-art AO-ARMs which are closely related to discrete diffusion [48]. Our model is only beaten by an autoregressive (AR) transformer and the AR-backbone Discrete Flow [59]. We believe this is because AR models only require learning a fixed generation order thus better utilize model capacity. Text8’s small vocabulary (26 letters and a space) led us to expect limited flexibility from our state-dependent formulation. However, using the generalized objective in (12), GenMD4 achieved significantly better BPC than MD4, demonstrating the potential of state-dependent diffusion for discrete data.

FineWeb-Edu. We train MD4 on FineWeb-Edu and evaluate its zero-shot accuracy on the Hellaswag dataset [60], a popular common sense inference benchmark for LLMs. We directly compared MD4 to its AR counterparts – transformers with identical configurations (except for causal masking) trained on the same data. Results are summarized in Fig. 5.

MD4 demonstrates steady performance growth with increasing scale. While outperformed by AR models of the same size, the performance gap does not widen as model size increases. For example, AR-small reaches 30% accuracy in 50k steps, while MD4-small takes 200k steps (4x data efficiency difference). At the medium scale, AR achieves 37% in 270k steps, compared to MD4’s 1 million steps.

<!-- image-->  
Figure 5: Hellaswag accuracy vs. training steps for MD4 and AR models at GPT-2 small, medium, and large scales.

## 7.2 Pixel-level image modeling

Unlike continuous diffusion which struggles with discrete data, we show that MD4, a discrete diffusion model, performs well on inherently continuous data, suggesting its potential for unifying modalities. We follow Austin et al. [14] and train MD4 on order-agnostic image data from CIFAR-10 and downsampled ImageNet 64×64 [63]. Each image is treated as a set of 256-valued discrete tokens, making the model agnostic to pixel proximity. We compare to other discrete diffusion and AR models with reported likelihood results on these datasets, although to our knowledge there are no published result on discrete diffusion for ImageNet 64 × 64 that directly model raw pixel space.

<!-- image-->  
Figure 6: Non cherry-picked unconditional samples from MD4 trained on ImageNet 64x64, treating pixels as discrete tokens. More samples can be found in Fig. 9 in Appendix. The model is optimized for likelihood instead of visual quality—see e.g., Kingma et al. [33] for samples from a continuous diffusion model optimized similarly for likelihood.

Tab. 3 summarizes our results. We establish a new state-of-the-art for discrete diffusion models, outperforming previous work [14, 29] by a significant margin. Our CIFAR-10 result surpasses the best reported AR result. On ImageNet 64 × 64, our results are competitive with Transformer AR models that are 4× larger, as well as a strong continuous diffusion model VDM [33]. Notably, despite lacking knowledge of the ordinal structure of pixel values, MD4 outperforms models trained with this inductive bias, including D3PM Gauss and Campbell et al. [29] where the noising distribution is a discrete Gaussian that assigns larger probabilities to near pixel values. To isolate the differences caused by training objectives, we also implemented the Campbell et al. [29] objective with the absorbing process, showing its high variance hinders learning even with our architecture.

We provide a random sample from our ImageNet 64×64 model in Fig. 6. More results can be found in App. K. In Fig. 2, we plot the FID values of samples generated under different choices of schedules and discretization grids. We can see that the model with the linear schedule plus a cosine grid achieves an FID close to the model with cosine schedule, both significantly outperform the linear schedule with a uniform grid. We further trained a class-conditional model on ImageNet 64×64 that boosts the FID to around 7. Although these are not state-of-the-art FIDs on ImageNet 64×64, we emphasize our models are optimized for likelihood instead of sample quality.

## 8 Conclusion

In this work, we revisit masked diffusion models, focusing on a flexible continuous-time formulation. Existing works in this area are not easily accessible to non-specialists and present ELBOs that are difficult to optimize, often resulting in performance that is not competitive with continuous diffusions and AR models. The framework we propose provides a very simple expression of the ELBO as a weighted integral of cross-entropy losses. Additionally, we propose a generalized masked diffusion formulation (GenMD4), where the masking schedule depends on the current state of the process, and derive its corresponding ELBO. On text data, our MD4 models outperform existing discrete and continuous diffusion models. For pixel-level image modeling, we significantly improve discrete diffusion results, outperforming similar-sized AR models and achieving comparable likelihoods to continuous diffusion models such as VDM. GenMD4 provides further improvements in terms of likelihoods over the state-independent case.

Although we have improved masked diffusion models, they still suffer from limitations. First, in some tasks such as text8, masked diffusions are not yet competitive with AR models. We conjecture that this is because AR models can better leverage model capacity since they only require learning one order. It would be interesting to develop better architectures for discrete diffusions. Moreover, GenMD4 is promising, but it can easily overfit to the dataset, making it less effective for zero-shot transfer compared to simpler versions. Additionally, inference with a state-dependent schedule is more challenging.

## References

[1] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, 2015.

[2] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, 2020.

[3] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2020.

[4] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

[5] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

[6] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-toimage diffusion models with deep language understanding. In Advances in Neural Information Processing Systems, 2022.

[7] Nanxin Chen, Yu Zhang, Heiga Zen, Ron J Weiss, Mohammad Norouzi, and William Chan. Wavegrad: Estimating gradients for waveform generation. In International Conference on Learning Representations, 2021.

[8] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. In International Conference on Learning Representations, 2021.

[9] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In Advances in Neural Information Processing Systems, 2022.

[10] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual descriptions. In International Conference on Learning Representations, 2023.

[11] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Yuanzhen Li, Tomer Michaeli, et al. Lumiere: A space-time diffusion model for video generation. arXiv preprint arXiv:2401.12945, 2024.

[12] OpenAI. Sora. https://openai.com/index/sora/, 2024.

[13] Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233, 2024.

[14] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. In Advances in Neural Information Processing Systems, 2021.

[15] Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. In Advances in Neural Information Processing Systems, 2021.

[16] Clément Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard. DiGress: Discrete denoising diffusion for graph generation. In International Conference on Learning Representations, 2023.

[17] Dongchao Yang, Jianwei Yu, Helin Wang, Wen Wang, Chao Weng, Yuexian Zou, and Dong Yu. Diffsound: Discrete diffusion model for text-to-sound generation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2023.

[18] Nate Gruver, Samuel Stanton, Nathan Frey, Tim GJ Rudner, Isidro Hotzel, Julien Lafrance-Vanasse, Arvind Rajpal, Kyunghyun Cho, and Andrew G Wilson. Protein design with guided discrete diffusion. In Advances in Neural Information Processing Systems, 2023.

[19] Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, et al. Continuous diffusion for categorical data. arXiv preprint arXiv:2211.15089, 2022.

[20] Ting Chen, Ruixiang ZHANG, and Geoffrey Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. In International Conference on Learning Representations, 2022.

[21] Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. Diffusion-LM improves controllable text generation. In Advances in Neural Information Processing Systems, 2022.

[22] Ishaan Gulrajani and Tatsunori B Hashimoto. Likelihood-based diffusion language models. In Advances in Neural Information Processing Systems, 2023.

[23] Justin Lovelace, Varsha Kishore, Chao Wan, Eliot Shekhtman, and Kilian Q Weinberger. Latent diffusion for language generation. In Advances in Neural Information Processing Systems, 2024.

[24] Pierre H Richemond, Sander Dieleman, and Arnaud Doucet. Categorical SDEs with simplex diffusion. arXiv preprint arXiv:2210.14784, 2022.

[25] Pavel Avdeyev, Chenlai Shi, Yuhao Tan, Kseniia Dudnyk, and Jian Zhou. Dirichlet diffusion score model for biological sequence generation. In International Conference on Machine Learning, 2023.

[26] Alex Graves, Rupesh Kumar Srivastava, Timothy Atkinson, and Faustino Gomez. Bayesian flow networks. arXiv preprint arXiv:2308.07037, 2023.

[27] Kaiwen Xue, Yuhao Zhou, Shen Nie, Xu Min, Xiaolu Zhang, Jun Zhou, and Chongxuan Li. Unifying Bayesian flow networks and diffusion models through stochastic differential equations. arXiv preprint arXiv:2404.15766, 2024.

[28] Guan-Horng Liu, Tianrong Chen, Evangelos Theodorou, and Molei Tao. Mirror diffusion models for constrained and watermarked generation. In Advances in Neural Information Processing Systems, 2024.

[29] Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. In Advances in Neural Information Processing Systems, 2022.

[30] Haoran Sun, Lijun Yu, Bo Dai, Dale Schuurmans, and Hanjun Dai. Score-based continuous-time discrete diffusion models. In International Conference on Learning Representations, 2022.

[31] Lin Zheng, Jianbo Yuan, Lei Yu, and Lingpeng Kong. A reparameterized discrete diffusion model for text generation. arXiv preprint arXiv:2302.05737, 2023.

[32] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion language modeling by estimating the ratios of the data distribution. In International Conference on Machine Learning, 2024.

[33] Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. In Advances in Neural Information Processing Systems, 2021.

[34] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Advances in Neural Information Processing Systems, 2022.

[35] Joe Benton, Yuyang Shi, Valentin De Bortoli, George Deligiannidis, and Arnaud Doucet. From denoising diffusions to denoising Markov models. arXiv preprint arXiv:2211.03595, 2022.

[36] Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. arXiv preprint arXiv:2406.03736, 2024.

[37] Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. arXiv preprint arXiv:2406.07524, 2024.

[38] Lingxiao Zhao, Xueying Ding, Lijun Yu, and Leman Akoglu. Improving and unifying discrete and continuous-time discrete denoising diffusion. arXiv preprint arXiv:2402.03701, 2024.

[39] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[40] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. Advances in Neural Information Processing Systems, 30, 2017.

[41] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. In International Conference on Learning Representations, 2019.

[42] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

[43] Hunter Nisonoff, Junhao Xiong, Stephan Allenspach, and Jennifer Listgarten. Unlocking guidance for discrete state-space diffusion and flow models. arXiv preprint arXiv:2406.01572, 2024.

[44] Yixiu Zhao, Jiaxin Shi, Lester Mackey, and Scott Linderman. Informed correctors for discrete diffusion models. arXiv preprint arXiv:2407.21243, 2024.

[45] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/jax-ml/jax.

[46] Ouail Kitouni, Niklas Nolte, James Hensman, and Bhaskar Mitra. Disk: A diffusion model for structured knowledge. arXiv preprint arXiv:2312.05253, 2023.

[47] Benigno Uria, Iain Murray, and Hugo Larochelle. A deep and tractable density estimator. In International Conference on Machine Learning, pages 467–475. PMLR, 2014.

[48] Emiel Hoogeboom, Alexey A Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, and Tim Salimans. Autoregressive diffusion models. In International Conference on Learning Representations, 2021.

[49] Andrew Campbell, Jason Yim, Regina Barzilay, Tom Rainforth, and Tommi Jaakkola. Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design. In International Conference on Machine Learning, 2024.

[50] Javier E Santos, Zachary R Fox, Nicholas Lubbers, and Yen Ting Lin. Blackout diffusion: generative diffusion models in discrete-state spaces. In International Conference on Machine Learning, pages 9034–9059. PMLR, 2023.

[51] Nikolay Savinov, Junyoung Chung, Mikolaj Binkowski, Erich Elsen, and Aaron van den Oord. Stepunrolled denoising autoencoders for text generation. In International Conference on Learning Representations, 2022.

[52] Jiaxin Shi, Yuhao Zhou, Jessica Hwang, Michalis Titsias, and Lester Mackey. Gradient estimation with discrete Stein operators. In Advances in Neural Information Processing Systems, 2022.

[53] Tim Salimans and David A Knowles. On using control variates with stochastic approximation for variational bayes and its connection to stochastic linear regression. arXiv preprint arXiv:1401.1022, 2014.

[54] W. Kool, H. V. Hoof, and M. Welling. Buy 4 REINFORCE samples, get a baseline for free! In DeepRLStructPred@ICLR, 2019.

[55] Matt Mahoney. Text8. https://mattmahoney.net/dc/textdata.html. Accessed: 2024-05-14.

[56] Aaron Gokaslan and Vanya Cohen. Openwebtext corpus. http://Skylion007.github.io/ OpenWebTextCorpus, 2019.

[57] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

[58] Guilherme Penedo, Hynek Kydlícek, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, ˇ Thomas Wolf, et al. The fineweb datasets: Decanting the web for the finest text data at scale. arXiv preprint arXiv:2406.17557, 2024.

[59] Dustin Tran, Keyon Vafa, Kumar Agrawal, Laurent Dinh, and Ben Poole. Discrete flows: Invertible generative models of discrete data. In Advances in Neural Information Processing Systems, 2019.

[60] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

[61] Andy Shih, Dorsa Sadigh, and Stefano Ermon. Training and inference on any-order autoregressive models the right way. In Advances in Neural Information Processing Systems, 2022.

[62] Zachary Ziegler and Alexander Rush. Latent normalizing flows for discrete sequences. In International Conference on Machine Learning, 2019.

[63] Aäron Van Den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. In International Conference on Machine Learning, 2016.

[64] Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, and Alex Graves. Conditional image generation with pixelcnn decoders. In Advances in Neural Information Processing systems, 2016.

[65] Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P Kingma. Pixelcnn++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications. In International Conference on Learning Representations, 2016.

[66] Xi Chen, Nikhil Mishra, Mostafa Rohaninejad, and Pieter Abbeel. Pixelsnail: An improved autoregressive generative model. In International Conference on Machine Learning, 2018.

[67] Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In International Conference on Machine Learning, 2018.

[68] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.

[69] Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse attention with routing transformers. Transactions of the Association for Computational Linguistics, 9: 53–68, 2021.

[70] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in Neural Information Processing Systems, 30, 2017.

[71] Kehang Han, Kathleen Kenealy, Aditya Barua, Noah Fiedel, and Noah Constant. Transfer learning for text diffusion models. arXiv preprint arXiv:2401.17181, 2024.

[72] Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. In Advances in Neural Information Processing Systems, 2015.

[73] Peter W. Glynn. Likelihood ratio gradient estimation for stochastic systems. Communications of the ACM, 33(10):75–84, 1990.

[74] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3-4):229–256, 1992.

[75] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.

Table 4: Masking schedule formulas.
<table><tr><td>Masking schedules</td><td> $\alpha _ { t }$ </td><td>Cross-entropy loss weight  $\frac { \alpha _ { t } ^ { \prime } } { 1 - \alpha _ { t } }$ </td></tr><tr><td>Linear</td><td> $1 - t$ </td><td> $- { \frac { 1 } { t } }$ </td></tr><tr><td>Polynomial</td><td> $1 - t ^ { w }$ </td><td> $- { \frac { w } { t } }$ </td></tr><tr><td>Geometric</td><td> $\left( - \bar { \beta } _ { \operatorname* { m i n } } ^ { 1 - t } \bar { \beta } _ { \operatorname* { m a x } } ^ { t } \right)$  exp</td><td> $\begin{array} { r } { - \frac { \exp \left( - \bar { \beta } _ { \operatorname* { m i n } } ^ { 1 - t } { \bar { \beta } _ { \operatorname* { m a x } } ^ { t } } \right) } { 1 - \exp \left( - \bar { \beta } _ { \operatorname* { m i n } } ^ { 1 - t } { \bar { \beta } _ { \operatorname* { m a x } } ^ { t } } \right) } \bar { \beta } _ { \operatorname* { m i n } } ^ { 1 - t } \bar { \beta } _ { \operatorname* { m a x } } ^ { t } \log \frac { \sigma _ { \operatorname* { m i n } } } { \sigma _ { \operatorname* { m a x } } } } \end{array}$ </td></tr><tr><td>Cosine</td><td> $\textstyle 1 - \cos ( { \frac { \pi } { 2 } } ( 1 - t ) )$ </td><td> $\begin{array} { r } { - \frac { \pi } { 2 } \tan ( \frac { \pi } { 2 } ( 1 - t ) ) } \end{array}$ </td></tr></table>

## A Discrete-time derivation

We divide time from 0 to 1 into T intervals, and let $s ( i ) = ( i - 1 ) / T , t ( i ) = i / T$ . The forward transition matrix $Q _ { i } \in \mathbb { R } ^ { ( m + 1 ) \times ( m + 1 ) }$ (m is vocabulary size) at time $t ( i )$ is

$$
[ Q _ { i } ] _ { j k } = { \left\{ \begin{array} { l l } { 1 } & { j = k = m } \\ { 1 - \beta _ { i } } & { j = k \neq m } \\ { \beta _ { i } } & { k = m , j \neq m } \\ { 0 } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }
$$

or more compactly written as

$$
Q _ { i } = ( 1 - \beta _ { i } ) I + \beta _ { i } \mathbf { 1 } e _ { m } ^ { \top } ,
$$

where 1 denotes an all-one vector of size $m + 1$ , and $e _ { m }$ is an one-hot vector of size $m + 1$ with the m-th element (recall that counting starts from 0) being one. We use an one-hot vector $x _ { t }$ of length m + 1 to denote the discrete state. The forward conditionals are defined as

$$
\begin{array} { r } { q ( x _ { t ( i ) } | x _ { s ( i ) } ) = \mathrm { C a t } ( x _ { t ( i ) } ; Q _ { i } ^ { \top } x _ { s ( i ) } ) = x _ { s ( i ) } ^ { \top } Q _ { i } x _ { t ( i ) } , } \end{array}\tag{13}
$$

where $Q _ { i } ^ { \top } x _ { s ( i ) }$ is the probabilities for each of the $m + 1$ categories that $\boldsymbol { x } _ { t ( i ) }$ can take. The marginal forward distribution at time $t ( i )$ given $x _ { 0 }$ is

$$
\begin{array} { r } { q ( x _ { t ( i ) } | x _ { 0 } ) = \mathrm { C a t } ( x _ { t ( i ) } ; \bar { Q } _ { i } ^ { \top } x _ { 0 } ) = x _ { 0 } ^ { \top } \bar { Q } _ { i } x _ { t ( i ) } , } \end{array}
$$

where $\begin{array} { r } { \bar { Q } _ { i } = \prod _ { j = 1 } ^ { i } Q _ { j } = \prod _ { j = 1 } ^ { i } ( 1 - \beta _ { j } ) I + \bigl ( 1 - \prod _ { j = 1 } ^ { i } ( 1 - \beta _ { j } ) \bigr ) \mathbf { 1 } e _ { m } ^ { \top } } \end{array}$ . To see what this leads to in continuous time, we let $\begin{array} { r } { \beta _ { i } = \frac { \beta ( t ( i ) ) } { T } } \end{array}$ and $T \to \infty ;$

$$
\begin{array} { r l r } {  { \prod _ { j = 1 } ^ { i } ( 1 - \beta _ { j } ) = \exp \Big ( \displaystyle \sum _ { j = 1 } ^ { i } \log ( 1 - \beta _ { j } ) \Big ) } } \\ & { } & { = \exp \Big ( \displaystyle \sum _ { j = 1 } ^ { i } - \frac { \beta ( t ( j ) ) } { T } + o ( 1 / T ) \Big ) } \\ & { } & { \stackrel { T \to \infty } { \to } \exp \Big ( - \displaystyle \int _ { 0 } ^ { t ( i ) } \beta ( s ) \mathrm { d } s \Big ) . } \end{array}
$$

We let $\bar { Q } ( t )$ denote the limit of ${ \bar { Q } } _ { i }$ in this case:

$$
\begin{array} { r l r } {  { \bar { Q } ( t ) = \exp \big ( - \int _ { 0 } ^ { t } \beta ( s ) \mathrm { d } s \big ) I + \Big ( 1 - \exp \big ( - \int _ { 0 } ^ { t } \beta ( s ) \mathrm { d } s \big ) \Big ) \mathbf { 1 } e _ { m } ^ { \top } } } \\ & { } & { \triangleq \alpha _ { t } I + ( 1 - \alpha _ { t } ) \mathbf { 1 } e _ { m } ^ { \top } . } \end{array}
$$

Here we define $\begin{array} { r } { \alpha _ { t } \triangleq \exp ( - \int _ { 0 } ^ { t } \beta ( s ) \mathrm { d } s ) } \end{array}$ . And the marginal forward transition is

$$
\begin{array} { r } { q ( x _ { t } | x _ { 0 } ) = \mathrm { C a t } ( x _ { t } ; \bar { Q } ( t ) ^ { \top } x _ { 0 } ) = x _ { 0 } ^ { \top } \bar { Q } ( t ) x _ { t } = \alpha _ { t } x _ { 0 } ^ { \top } x _ { t } + ( 1 - \alpha _ { t } ) e _ { m } ^ { \top } x _ { t } . } \end{array}\tag{14}
$$

## B Continuous-time derivation

We consider a continuous-time Markov chain with transition rates

$$
Q ( t ) = ( Q _ { i } - I ) / ( 1 / T ) = \beta ( t ) ( 1 e _ { m } ^ { \top } - I ) .\tag{15}
$$

For simplicity, we let $Q \ = \ \mathbf { 1 } e _ { m } ^ { \top } \ - \ I$ . The marginal forward distribution at time t given x0 is $q ( x _ { t } | x _ { 0 } ) = \mathrm { C a t } ( x _ { t } ; \bar { Q } ( t ) ^ { \top } x _ { 0 } )$ , where

$$
\bar { Q } ( t ) = \exp \Big ( \int _ { 0 } ^ { t } Q ( s ) \mathrm { d } s \Big ) = \exp \Big ( Q \int _ { 0 } ^ { t } \beta ( s ) \mathrm { d } s \Big ) = \exp ( \bar { \beta } ( t ) Q ) .
$$

Here we define $\begin{array} { r } { \bar { \beta } ( t ) \triangleq \int _ { 0 } ^ { t } \beta ( s ) \mathrm { d } s } \end{array}$ . The matrix exponential can be computed via eigendecomposition:

$$
\bar { \beta } ( t ) Q = U \Lambda U ^ { - 1 } ,
$$

where

$$
\begin{array} { c } { { U = I - e _ { m } e _ { m } ^ { \top } + \displaystyle \frac { 1 } { \sqrt { n + 1 } } \mathbf { 1 } e _ { m } ^ { \top } , } } \\ { { U ^ { - 1 } = I + \sqrt { n + 1 } e _ { m } e _ { m } ^ { \top } - \mathbf { 1 } e _ { m } ^ { \top } , } } \\ { { \Lambda = \bar { \beta } ( t ) ( e _ { m } e _ { m } ^ { \top } - I ) , } } \end{array}
$$

and thus $\exp ( \Lambda ) = \alpha _ { t } I + ( 1 - \alpha _ { t } ) e _ { m } e _ { m } ^ { \top }$

$$
\bar { Q } ( t ) = U \exp ( \Lambda ) U ^ { - 1 } = \alpha _ { t } I + ( 1 - \alpha _ { t } ) \mathbf { 1 } e _ { m } ^ { \top } .
$$

A simpler derivation uses the following property:

$$
Q ^ { 2 } = - Q .
$$

Therefore,

$$
\begin{array} { l } { { \displaystyle \bar { Q } ( t ) = \exp ( \bar { \beta } ( t ) Q ) } } \\ { { \displaystyle \quad = I + \bar { \beta } ( t ) Q + \frac { 1 } { 2 } \bar { \beta } ( t ) ^ { 2 } Q ^ { 2 } + \frac { 1 } { 3 } \bar { \beta } ( t ) ^ { 3 } Q ^ { 3 } + \dots } } \\ { ~ } \\ { { \displaystyle \quad = I + Q - ( 1 - \bar { \beta } ( t ) + \frac { 1 } { 2 } \bar { \beta } ( t ) ^ { 2 } - \frac { 1 } { 3 } \bar { \beta } ( t ) ^ { 3 } + \dots ) Q } } \\ { { \displaystyle \quad = I + Q - \exp ( - \bar { \beta } ( t ) ) Q } } \\ { { \displaystyle \quad = \alpha _ { t } I + ( 1 - \alpha _ { t } ) \mathbf { 1 } e _ { m } ^ { \top } } . } \end{array}
$$

This marginal forward transition matrix at time t coincides with the result (1) we get by taking the limit of discrete-time derivation.

Arbitrary discretization of the continuous-time forward process. For the discrete-time process we have defined the per-step transition in (13). For the continuous-time process, we can derive the transition matrix ${ \bar { Q } } ( s , t ) _ { i j } \triangleq q ( x _ { t } = j | x _ { s } = i )$ between two arbitrary time s and t as the solution to the following differential equation (known as Kolmogorov forward equation)

$$
\frac { \mathrm { d } } { \mathrm { d } t } \bar { Q } ( s , t ) = \bar { Q } ( s , t ) Q ( t ) \mathrm { w h e r e } Q ( t ) = \beta ( t ) Q
$$

with initial condition ${ \bar { Q } } ( s , s ) = I$ . The solution is given by

$$
\bar { Q } ( s , t ) = \exp \left( ( \bar { \beta } ( t ) - \bar { \beta } ( s ) ) Q \right) = \bar { Q } ( s ) ^ { - 1 } \bar { Q } ( t ) .
$$

Routine work (using the Woodbury matrix inversion lemma) shows that

$$
\bar { Q } ( t ) ^ { - 1 } = \alpha _ { t } ^ { - 1 } I + ( 1 - \alpha _ { t } ^ { - 1 } ) \mathbf { 1 } e _ { m } ^ { \top } .
$$

Plugging the result back, we get the forward transition distribution from s to t:

$$
\begin{array} { r l r } & { } & { q ( x _ { t } | x _ { s } ) = \mathrm { C a t } ( x _ { t } ; \bar { Q } ( s , t ) ^ { \top } x _ { s } ) = x _ { s } ^ { \top } \bar { Q } ( s , t ) x _ { t } , } \\ & { } & { \mathrm { w h e r e } \ \bar { Q } ( s , t ) \triangleq \bar { Q } ( s ) ^ { - 1 } \bar { Q } ( t ) = \frac { \alpha _ { t } } { \alpha _ { s } } I + \big ( 1 - \frac { \alpha _ { t } } { \alpha _ { s } } \big ) \mathbf { 1 } e _ { m } ^ { \top } . } \end{array}\tag{16}
$$

<!-- image-->  
Figure 7: The reverse transition probability and our generative model. Left: $q ( x _ { s } = \cdot | x _ { t } = \cdot , x _ { 0 } )$ in matrix form where first index is $x _ { t }$ and second index is $x _ { s }$ . Right: $p _ { \theta } ( x _ { s } = \cdot | x _ { t } = \cdot ) \triangleq q ( x _ { s } =$ $\cdot | x _ { t } = \cdot , \mu _ { \theta } ( x _ { t } , t ) )$ also in matrix form.

## C Time reversal of the forward process given $x _ { 0 }$

The analytic property of our forward process allows to compute many quantities of interest in closed form. One such quantity frequently used in diffusion models is the time reversal of the forward process given x0: $q ( x _ { s } | \boldsymbol { x } _ { t } , \boldsymbol { x } _ { 0 } )$ . We can compute it using (14) and (16) as

$$
\begin{array} { c l } { q ( x _ { s } | x _ { t } , x _ { 0 } ) = \frac { q ( x _ { t } | x _ { s } ) q ( x _ { s } | x _ { 0 } ) } { q ( x _ { t } | x _ { 0 } ) } } \\ { \ } & { \ = \left\{ \begin{array} { l l } { \frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } } x _ { s } ^ { \top } x _ { 0 } } & { x _ { s } \neq m , x _ { t } = m } \\ { \frac { 1 - \alpha _ { s } } { 1 - \alpha _ { t } } } & { x _ { s } = m , x _ { t } = m } \\ { x _ { s } ^ { \top } x _ { t } } & { x _ { t } \neq m . } \end{array} \right. } \end{array}\tag{17}
$$

Visually, eqn (17) is $\mathbf { a } \mathbb { R } ^ { ( m + 1 ) \times ( m + 1 ) }$ matrix (Fig. 7, left) whose first index is $x _ { t }$ and the second is $x _ { s } .$ . The matrix is almost an identity matrix except the last row corresponding to $x _ { t }$ is the mask token. The last row means with probability of $\frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } }$ the mask token gets unmasked to become $x _ { 0 }$ , and with probability of $\frac { 1 - \alpha _ { s } } { 1 - \alpha _ { t } }$ it remains masked.

Alternatively, we can rewrite the above using reverse transition matrix $\bar { R } ^ { x _ { 0 } } ( t , s ) \in \mathbb { R } ^ { ( m + 1 ) \times ( m + 1 ) }$ as

$$
 { \boldsymbol { q } } ( x _ { s } | x _ { t } , x _ { 0 } ) =  { \operatorname { C a t } } ( x _ { s } ; \bar { R } ^ { x _ { 0 } } ( t , s ) ^ { \top } x _ { t } ) ,  { \operatorname { w h e r e } } \ \bar { R } ^ { x _ { 0 } } ( t , s ) = I + \frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } }  { \boldsymbol { e } } _ { m } ( x _ { 0 } -  { \boldsymbol { e } } _ { m } ) ^ { \top } .
$$

We are also interested in what would happen in the infinitesimal time limit, i.e., when $s = t - \Delta t$ and $\Delta t \to 0$ . Note that

$$
\alpha _ { t - \Delta t } - \alpha _ { t } = - \alpha _ { t } ^ { \prime } \Delta t + o ( \Delta t ) .
$$

Plugging it into the original formula, we get

$$
\bar { R } ^ { x _ { 0 } } ( t , t - \Delta t ) = I - \frac { \alpha _ { t } ^ { \prime } } { 1 - \alpha _ { t } } e _ { m } ( x _ { 0 } - e _ { m } ) ^ { \top } \Delta t + o ( \Delta t ) .
$$

Comparing the above with the transition rate matrix $R ^ { x _ { 0 } } ( t )$ definition

$$
\bar { R } ^ { x _ { 0 } } ( t , t - \Delta t ) = I + R ^ { x _ { 0 } } ( t ) \Delta t + o ( \Delta t ) ,
$$

we have determined the transition rate matrix for the reverse process conditioned on $x _ { 0 } \colon$

$$
R ^ { x _ { 0 } } ( t ) = - \frac { \alpha _ { t } ^ { \prime } } { 1 - \alpha _ { t } } e _ { m } ( x _ { 0 } - e _ { m } ) ^ { \top } .\tag{18}
$$

## D Details of the ELBO

Using (17) and (3), we compute the KL divergences between forward and reverse transitions

$$
\begin{array} { r l } { \mathrm { K L } ( q ( x _ { s } | x _ { t } , x _ { 0 } ) \| p _ { \theta } ( x _ { s } | x _ { t } ) ) = \mathrm { K L } ( q ( x _ { s } | x _ { t } , x _ { 0 } ) \| q ( x _ { s } | x _ { t } , \mu _ { \theta } ( x _ { t } , t ) ) ) } & { } \\ & { = \left\{ \underset { 0 } { \sum _ { x _ { s } = 0 } ^ { m } } q ( x _ { s } | x _ { t } , x _ { 0 } ) \log \frac { q ( x _ { s } | x _ { t } , x _ { 0 } ) } { q ( x _ { s } | x _ { t } , \mu _ { \theta } ( x _ { t } , t ) ) } \right. \left. \left. x _ { t } = m \right. \right. } \\ & { = \delta _ { x _ { t } = m } \displaystyle \sum _ { k \neq m } \frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } } { x _ { 0 } ^ { \top } } e _ { k } \log \frac { { x _ { 0 } ^ { \top } } e _ { k } } { \mu _ { \theta } ( x _ { t } , t ) ^ { \top } e _ { k } } } \\ & { = - \delta _ { x _ { t } = m } \frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } } { x _ { 0 } ^ { \top } } \log \mu _ { \theta } ( x _ { t } , t ) . } \end{array}\tag{19}
$$

Note that 0 log $0 = 0 .$ . Alternatively, this result can be easily obtained from the visual depictions of $q ( x _ { s } | x _ { t } , x _ { 0 } )$ and $p _ { \theta } ( x _ { s } | x _ { t } )$ shown in Fig. 7. In this case, the reconstruction term becomes

$$
\begin{array} { l } { \mathbb { E } _ { q ( x _ { t ( 1 ) } | x _ { 0 } ) } [ \log p ( x _ { 0 } | x _ { t ( 1 ) } ) ] = \displaystyle \sum _ { k = 0 } ^ { m } q _ { t ( 1 ) | 0 } ( k | x _ { 0 } ) \log \frac { q _ { t ( 1 ) | 0 } ( k | x _ { 0 } ) } { \sum _ { j \ne m } q _ { t ( 1 ) | 0 } ( k | j ) } } \\ { = \alpha _ { t ( 1 ) } \cdot \log \frac { \alpha _ { t ( 1 ) } } { \alpha _ { t ( 1 ) } } + ( 1 - \alpha _ { t ( 1 ) } ) \log \frac { 1 } { m } } \\ { = - ( 1 - \alpha _ { t ( 1 ) } ) \log m . } \end{array}
$$

The prior KL term can be computed as

$$
\begin{array} { r } { \mathrm { K L } ( q ( x _ { 1 } | x _ { 0 } ) \| p ( x _ { 1 } ) ) = \mathrm { K L } ( \delta _ { x _ { 1 } , m } \| \delta _ { x _ { 1 } , m } ) = 0 . } \end{array}
$$

As usual, we take the continuous-time limit by letting $T \to \infty$

$$
\begin{array} { r l } & { \mathcal { L } _ { \infty } \triangleq \underset { T  \infty } { \operatorname* { l i m } } \mathcal { L } _ { T } } \\ & { \quad = \underset { T  \infty } { \operatorname* { l i m } } \overset { T } { \underset { i = 2 } { \sum } } - \frac { \alpha _ { s ( i ) } - \alpha _ { t ( i ) } } { s ( i ) - t ( i ) } \frac { s ( i ) - t ( i ) } { 1 - \alpha _ { t ( i ) } } x _ { 0 } ^ { \top } \mathbb { E } _ { q ( x _ { t ( i ) } \mid x _ { 0 } ) } [ \delta _ { x _ { t ( i ) } , m } \log \mu _ { \theta } ( x _ { t ( i ) } , t ( i ) ) ] } \\ & { \quad = \displaystyle \int _ { t ( 1 ) } ^ { 1 } \frac { \alpha _ { t } ^ { \prime } } { 1 - \alpha _ { t } } x _ { 0 } ^ { \top } \mathbb { E } _ { q ( x _ { t } \mid x _ { 0 } ) } [ \delta _ { x _ { t } , m } \log \mu _ { \theta } ( x _ { t } , t ) ] \mathrm { d } t . } \end{array}
$$

## E Avoiding undefined KL divergence

When defining the forward process, we often do not want $\alpha _ { 1 }$ to be exactly 0, or equivalently, $\lambda _ { 1 }$ to be $\infty$ for numerical stability reasons. Instead, we set $\lambda _ { 1 }$ to be a finite value, and thereby $\alpha _ { 1 }$ has a small positive value. This has a problem that the support of $q ( x _ { 1 } | x _ { 0 } )$ is no longer $\{ m \}$ and instead becomes $\{ m , x _ { 0 } \}$ . As a result, the KL divergence between $q ( x _ { 1 } | x _ { 0 } )$ and $p ( x _ { 1 } )$ is undefined because $q ( x _ { 1 } | x _ { 0 } )$ is not absolutely continuous with respect to $p ( x _ { 1 } ) = \delta _ { x _ { 1 } , m } .$ . To resolve the issue, we modify the prior distribution $p ( x _ { 1 } )$ such that it has support over all $m + 1$ values. One such choice is letting

$$
p ( x _ { 1 } ) = \frac { \alpha _ { 1 } } { m } \sum _ { j \neq m } \delta _ { x _ { 1 } , j } + ( 1 - \alpha _ { 1 } ) \delta _ { x _ { 1 } , m } .
$$

Then, the prior KL divergence term becomes

$$
\begin{array} { l } { \displaystyle \mathrm { K L } ( q ( x _ { 1 } | x _ { 0 } ) \| p ( x _ { 1 } ) ) = \sum _ { x _ { 1 } = 0 } ^ { m } q ( x _ { 1 } | x _ { 0 } ) \log \frac { q ( x _ { 1 } | x _ { 0 } ) } { p ( x _ { 1 } ) } } \\ { = \sum _ { x _ { 1 } = 0 } ^ { m } ( \alpha _ { 1 } \delta _ { x _ { 1 } , x _ { 0 } } + ( 1 - \alpha _ { 1 } ) \delta _ { x _ { 1 } , m } ) \log \frac { \alpha _ { 1 } \delta _ { x _ { 1 } , x _ { 0 } } + ( 1 - \alpha _ { 1 } ) \delta _ { x _ { 1 } = m } } { p ( x _ { 1 } ) } } \\ { = \alpha _ { 1 } \log \frac { \alpha _ { 1 } } { \alpha _ { 1 } / m } + ( 1 - \alpha _ { 1 } ) \log \frac { 1 - \alpha _ { 1 } } { 1 - \alpha _ { 1 } } } \\ { = \alpha _ { 1 } \log m . } \end{array}
$$

## F Details of Training and Sampling with MD4

## F.1 Training

Algorithm 1 A single step of training with MD4.   
Input: data minibatch $\{ x _ { t } ^ { i } \} _ { i = 1 } ^ { B }$ , network $\mu _ { \boldsymbol { \theta } } ( \cdot , t )$ , masking schedule $\alpha _ { t }$   
for $i = 1 , \ldots , B$ do (in parallel):   
ti ← mod $( u + i / B , 1 ) , u \sim U [ 0 , 1 ]$   
for $n \in [ N ]$ , mask out each token $x _ { 0 } ^ { i , ( n ) }$ independently with probability $1 - \alpha _ { t _ { 3 } }$ to obtain $\boldsymbol { x } _ { t _ { i } } ^ { i }$   
for $n \in [ N ] , \mathrm { i f } x _ { t _ { i } } ^ { ( n ) } = m$ , compute weighted cross entropy loss $\frac { \alpha _ { t _ { i } } ^ { \prime } } { 1 - \alpha _ { t _ { i } } } ( x _ { 0 } ^ { i , ( n ) } ) ^ { \top }$ log $\mu _ { \theta } ^ { ( n ) } ( x _ { t _ { i } } ^ { i } , t _ { i } )$   
Sum over all weighted cross entropy losses for mask positions and optimize via autodiff

## F.2 Sampling

Algorithm 2 Unconditional and conditional generation (e.g., infilling) with MD4.   
Input: Context sequence $x ^ { c }$ of length $N ,$ with masks indicating the target areas for generation   
Init: $\{ t ( i ) \} _ { i = 0 } ^ { T }$ ← discretize $( [ 0 , 1 ] ) , x _ { t ( T ) } \gets x ^ { c }$   
for $i \doteq \dot { T } , \dot { T } - 1 , \ldots , 1$ do   
$t \gets t ( i ) , s \gets t ( i - 1 )$   
for n ∈ [N ], if $x _ { t } ^ { ( n ) } = m$ , draw $\begin{array} { r } { x _ { s } ^ { ( n ) } \sim \mathrm { C a t } ( \frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } } \mu _ { \theta } ^ { ( n ) } ( x _ { t } , t ) + \frac { 1 - \alpha _ { s } } { 1 - \alpha _ { t } } e _ { m } ) } \end{array}$ else $x _ { s } ^ { ( n ) }  x _ { t } ^ { ( n ) }$   
return $x _ { 0 } .$

## G JAX Categorical Sampling and Implicit Top-p

We noticed that the following equivalent implementation of Alg. 2 leads to significantly worse sample quality in JAX:

Algorithm 3 Variant of Alg. 2 that yields lower sample quality when implemented in JAX.   
Input: Token sequence $x ^ { c }$ of length N , with masks indicating the target areas for generation   
Init: $\{ t ( i ) \} _ { i = 0 } ^ { T } $ discretize $[ 0 , \bar { 1 } ] ) , x _ { t ( T ) } \gets x ^ { c }$   
for $i \doteq \dot { T } , \dot { T } - 1 , \ldots , 1$ do   
$t \gets t ( i ) , s \gets t ( i - 1 )$   
for $n \in [ N ]$ do (in parallel)   
draw $u \sim U [ 0 , \dot { 1 } ]$   
if $x _ { t } ^ { ( n ) } = m$ and $\begin{array} { r } { u < \frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } } } \end{array}$ then   
draw $x _ { s } ^ { ( n ) } \sim \mathrm { C a t } ( \mu _ { \theta } ^ { ( n ) } ( x _ { t } , t ) )$   
else   
${ \bf \Phi } _ { X _ { s } ^ { ( n ) } } ^ { ( n ) } \gets x _ { t } ^ { ( n ) }$   
return $x _ { 0 } .$

However, mathetically it is equivalent to Alg. 2 and should produce identical results. Our investigation revealed that the issue arises because Alg. 2 scales the output probabilities of $\mu _ { \theta }$ by a small factor $\frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } }$ as s is close to $t ,$ causing some categories to have very low probabilities. JAX, however, implements categorical sampling using Gumbel argmax, which is less numerically stable than methods like binary search. As a result, categories with low probabilities are rarely sampled, even when their cumulative probability is significant. In our experiment, we found that categories with probabilities below 1e-8 are rarely sampled out of a total of 50K categories. Thus, Alg. 2 implicitly performs top-p sampling (with a dynamic p) under JAX’s categorical sampling, yielding better sample quality than Alg. 3 where $\mu _ { \theta }$ is not scaled by a small factor and has fewer categories truncated.

## H Unifying Existing Masked Diffusion Models

## H.1 The CTMC point of view

We first prove a lemma that connects the forward and reverse transition rate matrices. This follows from the results in [29] but we give a proof for completeness.

Lemma 2. The forward transition rate matrix $Q ( t )$ and the reverse transition rate matrix (given x0) $R ^ { x _ { 0 } } ( t )$ satisfy:

$$
R ^ { x _ { 0 } } ( t ) _ { k j } = Q ( t ) _ { j k } \frac { q _ { t | 0 } ( j | x _ { 0 } ) } { q _ { t | 0 } ( k | x _ { 0 } ) } f o r j \neq k .\tag{20}
$$

Proof Consider the reverse transition from time t + τ to t. For $j \neq k$ , Bayes’ rule yields

$$
\begin{array} { r l } & { q ( x _ { t } = j | x _ { t + \tau } = k , x _ { 0 } ) = \frac { q ( x _ { t } = j | x _ { 0 } ) q ( x _ { t + \tau } = k | x _ { t } = j ) } { q ( x _ { t + \tau } = k | x _ { 0 } ) } } \\ & { \qquad = \frac { q ( x _ { t } = j | x _ { 0 } ) ( \delta _ { j k } + Q ( t ) _ { j k } \tau + o ( \tau ) ) } { q ( x _ { t + \tau } = k | x _ { 0 } ) } } \\ & { \qquad \tau \overset {  } { = } \delta _ { k j } + \frac { q ( x _ { t } = j | x _ { 0 } ) } { q ( x _ { t } = k | x _ { 0 } ) } Q ( t ) _ { j k } \tau + o ( \tau ) . } \end{array}
$$

Then, it follows from the definition of the transition rate matrix that $\begin{array} { r } { R ^ { x _ { 0 } } ( t ) _ { k j } = Q ( t ) _ { j k } \frac { q _ { t | 0 } ( j | x _ { 0 } ) } { q _ { t | 0 } ( k | x _ { 0 } ) } } \end{array}$

Proposition 3. We use the shorthand $R _ { \theta } ( t ) _ { k j }$ to denote the approximate reverse transition rate from the state k $t o \ j$ obtained by substituting our prediction model $\mu _ { \theta } ( k ) f o r \ : x _ { 0 }$ in $R ^ { x _ { 0 } } ( t ) _ { k j }$ . Then, the continuous-time objective (4) can be equivalently expressed as

$$
\mathcal { L } _ { \infty } = - \int _ { t ( 1 ) } ^ { 1 } \mathbb { E } _ { q _ { t | 0 } ( k | x _ { 0 } ) } \Big [ R _ { \theta } ( t ) _ { k k } + \sum _ { j \neq k } Q ( t ) _ { k j } \log R _ { \theta } ( t ) _ { j k } \Big ] \mathrm { d } t + C ,\tag{21}
$$

where C is a constant independent of θ.

Proof To rewrite our objective $\mathcal { L } _ { \infty }$ with the transition rate matrices, we first go back to (19). There, instead of plugging in the explicit form of $\bar { R } ^ { x _ { 0 } } ( t , s )$ , we substitute it with (8) which leverages the transition rate $\check { R } ^ { x _ { 0 } } ( \check { t } )$ . To simplify the notation, we assume $x _ { t } = k$ and use the shorthand $R _ { \theta } ( t ) _ { k j } \triangleq R ^ { \mu _ { \theta } ( k ) } ( t ) _ { k j }$ . We then have

$$
\begin{array} { r l } & { \mathrm { K L } ( q ( x _ { t - \Delta t } | x _ { t } , x _ { 0 } ) | p _ { \theta } ( x _ { t - \Delta t } | x _ { t } ) ) } \\ & { = \mathrm { K L } ( \mathrm { C a } ( | x _ { s } ; \tilde { R } ^ { c } ( { \iota } , { \iota } - \Delta t ) ^ { \top } { \epsilon } _ { k } ) | | \mathrm { C a t } ( x _ { s } ; \tilde { R } ^ { \mu _ { \theta } ( k ) } ( { \iota } , { \iota } - \Delta t ) ^ { \top } { \epsilon } _ { k } ) ) } \\ & { = \displaystyle \sum _ { j = 0 } ^ { m } e _ { k } ^ { \frac { t } { \iota } } ( I + R ^ { \alpha _ { 1 } } ( t ) \Delta t + o ( \Delta t ) ) e _ { j } \log \frac { e _ { k } ^ { \top } } { e _ { k } ^ { \top } } ( I + R ^ { \alpha _ { 0 } } ( t ) \Delta t + o ( \Delta t ) ) e _ { j } } \\ & { = ( 1 + R ^ { \alpha _ { 0 } } ( t ) \kappa \Delta t ) \log \frac { 1 + R ^ { \alpha _ { 0 } } ( t ) \kappa \Delta t + o ( \Delta t ) } { 1 + R _ { 0 } ( t ) _ { k } \Delta t + o ( \Delta t ) } } \\ & { \quad + \displaystyle \sum _ { j \neq k } ( R ^ { \alpha _ { 0 } } ( t ) _ { k j } \Delta t ) \log \frac { R ^ { \alpha _ { 0 } } ( { \iota } ) _ { k j } \Delta t + o ( \Delta t ) } { R _ { 0 } ( t ) _ { k j } \Delta t + o ( \Delta t ) } + o ( \Delta t ) } \\ & { = ( I ^ { \alpha _ { 0 } } ( t ) _ { k k } - R _ { \theta } ( t ) _ { k k } ) \Delta t + \displaystyle \sum _ { j \neq k } ( R ^ { \alpha _ { 0 } } ( t ) _ { k j } \Delta t ) \log \frac { R ^ { \alpha _ { 0 } } ( t ) _ { k j } \Delta t + o ( \Delta t ) } { R _ { \theta } ( t ) _ { k j } \Delta t + o ( \Delta t ) } + o ( \Delta t ) . } \end{array}
$$