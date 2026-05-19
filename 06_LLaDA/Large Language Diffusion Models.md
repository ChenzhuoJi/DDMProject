# Large Language Diffusion Models

Jun Hu4‡ Jun Zhou4 Yankai Lin1,2,3‡ Ji-Rong Wen1,2,3 Chongxuan Li1,2,3‡ §

1 Gaoling School of Artificial Intelligence, Renmin University of China

2 Beijing Key Laboratory of Research on Large Models and Intelligent Governance

{nieshen,fengqizhu,chongxuanli}@ruc.edu.cn

## Abstract

The capabilities of large language models (LLMs) are widely regarded as relying on autoregressive models (ARMs). We challenge this notion by introducing LLaDA, a diffusion model trained from scratch under the pre-training and supervised finetuning (SFT) paradigm. LLaDA employs a forward data masking process and a reverse generation process, parameterized by a Transformer to predict masked tokens. It provides a principled generative approach for probabilistic inference by optimizing a likelihood lower bound. Across extensive benchmarks on general tasks, math, code, and so on, LLaDA demonstrates strong scalability and performs comparably to our self-constructed ARM baselines. Remarkably, LLaDA 8B is competitive with strong LLMs like LLaMA3 8B in in-context learning and, after SFT, exhibits impressive instruction-following abilities in case studies such as multiturn dialogue. Moreover, LLaDA addresses the reversal curse, surpassing GPT-4o in a reversal poem completion task. Our findings show the promise of diffusion models for language modeling at scale and challenge the common assumption that core LLM capabilities discussed above inherently depend on ARMs. Project page and codes: https://ml-gsai.github.io/LLaDA-demo/.

## 1 Introduction

Large language models (LLMs) [1] fall entirely within the framework of generative modeling. Specifically, LLMs aim to capture the true but unknown language distribution $p _ { \mathrm { d a t a } } ( \cdot )$ by optimizing a model distribution $p _ { \theta } ( \cdot )$ through maximum likelihood estimation, or equivalently KL divergence minimization between the two distributions:

$$
\underbrace { \underset { \theta } { \operatorname* { m a x } } \mathbb { E } _ { p _ { \mathrm { d a t a } } ( x ) } \log p _ { \theta } ( x ) \Leftrightarrow \underset { \theta } { \operatorname* { m i n } } \mathrm { K L } \big ( p _ { \mathrm { d a t a } } ( x ) | | p _ { \theta } ( x ) \big ) } _ { \mathrm { G e n e r a t i v e ~ m o d e l i n g ~ p r i n c i p l e s } } .\tag{1}
$$

The predominant approach relies on the autoregressive modeling (ARM)—commonly referred to as the “next-token prediction” paradigm—to define the model distribution:

$$
\underbrace { p _ { \theta } ( x ) = p _ { \theta } ( x ^ { 1 } ) \prod _ { i = 2 } ^ { L } p _ { \theta } ( x ^ { i } \mid x ^ { 1 } , \ldots , x ^ { i - 1 } ) } _ { \mathrm { A u t o r e g r e s s i v e ~ f o r m u l a t i o n } } ,\tag{2}
$$

<!-- image-->  
Figure 1: Zero/Few-Shot Benchmarks. We scale LLaDA to 8B parameters from scratch and observe competitive zero/few-shot performance compared with strong autoregressive LLMs [6].

where x is a sequence of length L, and $x ^ { i }$ is the i-th token. This paradigm has proven remarkably effective [2–5] and has become the foundation of current LLMs. Despite its widespread adoption, a fundamental question remains unanswered: Is the autoregressive paradigm the only path to achieving the core capabilities of LLMs, such as scalability, in-context learning, and instruction-following?

We argue that the answer is not a simple “yes”. The key insight overlooked previously is: It is the generative modeling principles (i.e., Eq. (1)), rather than the autoregressive formulation (i.e., Eq. (2)) itself, that fundamentally underpin the essential properties of LLMs.

In particular, we argue that scalability is primarily a consequence of the interplay between Transformers [7], model size, data size, and Fisher consistency5 [8] induced by the generative principles in Eq. (1), rather than a unique result of the ARMs in Eq. (2). The success of diffusion transformers [9, 10] on visual data [11] supports this claim. Furthermore, the instruction-following and in-context learning [4] capabilities appear to be intrinsic properties of all conditional generative models on structurally consistent linguistic tasks, rather than exclusive advantages of ARMs. In addition, while ARMs can be interpreted as a lossless data compressor [12, 13], any sufficiently expressive probabilistic model can achieve similar capabilities [14].

However, certain inherent limitations of LLMs can be directly attributed to their autoregressive nature. For instance, the left-to-right generation process restricts their ability to handle reversal reasoning tasks [15], highlighting a representative failure in the generalization capabilities of current models.

Motivated by these insights, we introduce LLaDA (Large Language Diffusion with mAsking) to investigate whether the capabilities exhibited by LLMs can emerge from generative modeling principles beyond ARMs, thereby addressing the fundamental question posed earlier. In contrast to traditional ARMs, LLaDA leverages a masked diffusion model (MDM) [16–20], which incorporates a forward data masking process and trains a mask predictor to approximate its reverse process. This design enables LLaDA to construct a model distribution with bidirectional dependencies and optimize a variational lower bound of its log-likelihood, offering a principled and previously unexplored perspective on the core capabilities of LLMs discussed above.

We adopt the standard pipeline of data preparation, pre-training, supervised fine-tuning (SFT), and evaluation, scaling LLaDA to an unprecedented language diffusion of size 8B. In particular, LLaDA 8B was pre-trained from scratch on 2.3 trillion tokens using 0.13 million H800 GPU hours, followed by SFT on 4.5 million pairs. Across diverse tasks, including language understanding, math, code, and Chinese, LLaDA demonstrates the following contributions:

• LLaDA scales effectively to a compute budget of $1 0 ^ { 2 3 }$ FLOPs, achieving comparable results to ARM baselines trained on the same data across six tasks, e.g., MMLU and GSM8K.

<!-- image-->

<!-- image-->

(c)  
<!-- image-->  
Figure 2: Overview of LLaDA. (a) Pre-training. LLaDA is trained on text with random masks applied independently to all tokens at the same ratio $t \sim U [ 0 , 1 ]$ . (b) SFT. Only response tokens are possibly masked. (c) Sampling. LLaDA simulates a diffusion process from t = 1 (fully masked) to t = 0 (unmasked), predicting all masks simultaneously at each step with flexible remask strategies.

• The pre-trained LLaDA 8B Base surpasses LLaMA2 7B Base [21] on nearly all 15 standard zero/few-shot learning tasks while performing on par with LLaMA3 8B Base [6], showcasing effective in-context learning capability.

• LLaDA significantly enhances the ability to follow instructions after SFT, as demonstrated in case studies such as multi-turn dialogue.

• LLaDA effectively breaks the reversal curse [15] with consistent performance across forward and reversal tasks. Notably, it outperforms GPT-4o in a reversal poem completion task.

## 2 Approach

In this section, we introduce the probabilistic formulation6, along with the pre-training, supervised fine-tuning, and inference procedures for LLaDA, as illustrated in Fig. 2.

## 2.1 Probabilistic Formulation

Unlike ARMs in Eq. (2), LLaDA defines a model distribution $p _ { \theta } ( x _ { 0 } )$ through a forward process and a reverse process [16–20]. The forward process gradually masks tokens independently in $x _ { 0 }$ until the sequence is fully masked at t = 1. For $t \in ( 0 , 1 )$ , the sequence $x _ { t }$ is partially masked, with each being masked with probability t or remaining unmasked with probability $1 - t .$ . The reverse process recovers the data distribution by iteratively predicting masked tokens as t moves from 1 to 0.

The core of LLaDA is a mask predictor, a parametric model $p _ { \theta } ( \cdot | x _ { t } )$ that takes $x _ { t }$ as input and predicts all masked tokens (denoted as M) simultaneously. It is trained using a cross-entropy loss computed only on the masked tokens [18–20]:

$$
\mathcal { L } ( \theta ) \triangleq - \mathbb { E } _ { t , x _ { 0 } , x _ { t } } \left[ \frac { 1 } { t } \sum _ { i = 1 } ^ { L } \mathbf { 1 } [ x _ { t } ^ { i } = \mathbf { M } ] \log p _ { \theta } ( x _ { 0 } ^ { i } | x _ { t } ) \right] ,\tag{3}
$$

where $x _ { 0 }$ is a training sample, t is a continuous random variable drawn uniformly from $[ 0 , 1 ] , x _ { t }$ is sampled from the forward process and L is the sequence length. The indicator function 1[·] ensures that the loss is computed only for masked tokens.

Once trained, we can simulate a reverse process (see Sec. 2.4 for details) parameterized by the mask predictor and define the model distribution $p _ { \theta } ( x _ { 0 } )$ as the marginal distribution induced at $t = 0$ . The loss function in Eq. (3) has been proven to be an upper bound on the negative log-likelihood of the model distribution, making it a principled objective for generative modeling:

$$
- \mathbb { E } _ { p _ { \mathrm { d a t a } } ( x _ { 0 } ) } \left[ \log p _ { \theta } ( x _ { 0 } ) \right] \leq \mathcal { L } ( \theta ) .\tag{4}
$$

Notably, LLaDA employs a masking ratio that varies randomly between 0 and 1 while BERT [22] uses a fixed ratio. The subtle differences have significant implications, especially at scale: as shown in

Eq. (4), LLaDA is a principled generative model with the potential to perform in-context learning and instruction-following naturally, akin to LLMs. Moreover, its generative perspective implies strong scalability with large data and models as discussed in Sec. 1. In addition, MaskGIT [23] adopts a heuristic training objective, which misses the $\textstyle { \frac { 1 } { t } }$ term compared to Eq. (3), and lacks a theoretical link to maximum likelihood. We emphasize that it is precisely the theoretical foundation of maximum likelihood estimation that motivated us to scale discrete diffusion models for language modeling.

## 2.2 Pre-training

LLaDA employs a Transformer [7] as the mask predictor, similar to existing LLMs. However, LLaDA does not use a causal mask, as its formulation allows it to see the entire input for predictions.

We trained two variants of LLaDA with different sizes: 1B and 8B. We summarize the model architecture of LLaDA 8B and LLaMA3 8B [6] here, and details are provided in Appendix B.2. We have ensured consistency in most hyperparameters while making several necessary modifications. We use vanilla multi-head attention instead of grouped query attention [24] for simplicity, as LLaDA is incompatible with KV caching, resulting in a different number of key and value heads. Consequently, the attention layer has more parameters, and we reduce the FFN dimension to maintain a comparable model size. Additionally, the vocabulary size differs due to a tokenizer [4] adapted on our data.

The LLaDA model is pre-trained on a dataset comprising 2.3 trillion (T) tokens, adhering to a data protocol that aligns closely with existing LLMs [25, 26], without the incorporation of any special techniques. The data are derived from online corpora, with low-quality content filtered through manually designed rules and LLM-based approaches. Beyond general text, the dataset encompasses high-quality code, math, and multilingual data. Please refer to Appendix B.1 for more details about datasets. The mixing of data sources and domains is guided by scaled-down ARMs. The pre-training process utilizes a fixed sequence length of 4096 tokens, incurring a total computational cost of 0.13 million H800 GPU hours, similar to ARMs of the same scale and dataset size.

For a training sequence $x _ { 0 } .$ , we randomly sample $t \in [ 0 , 1 ]$ , mask each token independently with the same probability t to obtain $x _ { t }$ (see Fig. 2 (a)) and estimate Eq. (3) via the Monte Carlo method for stochastic gradient descent training. In addition, following Nie et al. [27], to enhance the ability of LLaDA to handle variable-length data, we set 1% of the pre-training data to a random length that is uniformly sampled from the range [1, 4096].

We adopted the Warmup-Stable-Decay [28] learning rate scheduler to monitor the training progress without interrupting continuous training. Specifically, we linearly increased the learning rate from 0 to $4 \times 1 0 ^ { - 4 }$ over the first 2000 iterations and maintained it at $\mathrm { 4 \times 1 0 ^ { - 4 } }$ . After processing 1.2T tokens, we decayed the learning rate to $1 \times 1 0 ^ { - 4 }$ and held it constant for the next 0.8T tokens to ensure stable training. Finally, we linearly reduced the learning rate from $1 \times 1 0 ^ { - 4 } \mathrm { t o } 1 \times 1 0 ^ { - 5 }$ for the last 0.3T tokens. Furthermore, we utilized the AdamW optimizer [29] with a weight decay of 0.1, a batch size of 1280, and a local batch size of 4 per GPU. The 8B experiment was executed once, without any hyperparameter tuning.

## 2.3 Supervised Fine-Tuning

We enhance the capability of LLaDA to follow instructions by supervised fine-tuning (SFT) with paired data $( p _ { 0 } , r _ { 0 } )$ , where $p _ { 0 }$ is the prompt and $r _ { 0 }$ denotes the response. This is the simplest and most basic post-training method for LLMs. Technically, this requires to model the conditional distribution $p _ { \theta } ( r _ { 0 } | p _ { 0 } )$ instead of $p _ { \theta } ( x _ { 0 } )$ in pre-training.

The implementation is similar to pre-training. As shown in Fig. 2 (b), we leave the prompt unchanged and mask the tokens in the response independently, as done for $x _ { 0 }$ . Then, we feed both the prompt and the masked response $r _ { t }$ to the pre-trained mask predictor to compute the loss for SFT:

$$
- \mathbb { E } _ { t , p _ { 0 } , r _ { 0 } , r _ { t } } \left[ \frac { 1 } { t } \sum _ { i = 1 } ^ { L ^ { \prime } } \mathbf { 1 } [ r _ { t } ^ { i } = \mathbf { M } ] \log p _ { \theta } ( r _ { 0 } ^ { i } | p _ { 0 } , r _ { t } ) \right] ,\tag{5}
$$

where $L ^ { \prime }$ denotes a dynamic length specified later, and all other notations remain the same as before. Note that this approach is fully compatible with pre-training. Essentially, the concatenation of $p _ { 0 }$ and $r _ { 0 }$ can be treated as clean pre-training data $x _ { 0 }$ , while the concatenation of $p _ { 0 }$ and $r _ { t }$ serves as the masked version $x _ { t }$ . The process is identical to pre-training, with the only difference being that all masked tokens happen to appear in the $r _ { 0 }$ portion.

The LLaDA 8B model undergoes SFT on a dataset comprising 4.5 million pairs. Consistent with the pre-training process, both data preparation and training follow the SFT protocols utilized in existing LLMs [25, 26], without introducing any additional techniques to optimize LLaDA’s performance. The dataset spans multiple domains, including code, mathematics, and instruction-following. We append |EOS| tokens to the end of short pairs in each mini-batch to ensure equal lengths across all data. We treat |EOS| as a normal token during training and remove it during sampling, enabling LLaDA to control the response length automatically. Please refer to Appendix B.1 for more details.

We train for 3 epochs on the SFT data using a similar schedule to the pre-training phase. The learning rate is linearly increased from 0 to $2 . 5 \times 1 0 ^ { - 5 }$ over the first 50 iterations and then kept constant. During the final 10% of iterations, it is linearly reduced to $2 . 5 \times 1 0 ^ { - 6 }$ . Additionally, we set the weight decay to 0.1, the global batch size to 256, and the local batch size to 2 per GPU. The SFT experiment was executed once, without any hyperparameter tuning.

## 2.4 Inference

As a generative model, LLaDA can sample new text and evaluate the likelihood of candidate text in a diffusion manner instead of the left-to-right autoregressive fashion.

We begin with the reverse generation process. As illustrated in Fig. 2 (c), given a prompt $p _ { 0 }$ , we discretize the reverse process to sample from the model distribution $p _ { \theta } ( r _ { 0 } | p _ { 0 } )$ , starting from a fully masked response. The total number of sampling steps is a hyperparameter, which naturally provides LLaDA with a trade-off between efficiency and sample quality, as analyzed in Sec. 3.3. We employ uniformly distributed timesteps by default. In addition, the generation length is also treated as a hyperparameter, specifying the length of the fully masked sentence at the beginning of the sampling process. After generation, tokens appearing after the |EOS| token are discarded. As detailed in Appendix B.5, since both pre-training and SFT are conducted using datasets with variable lengths, the final results are insensitive to this length hyperparameter.

At an intermediate step from time $t \in ( 0 , 1 ] \mathrm { ~ t o ~ } s \in [ 0 , t )$ , we feed both $p _ { 0 }$ and $r _ { t }$ into the mask predictor and predict all masked tokens simultaneously. Subsequently, we remask $\frac { s } { t }$ of the predicted tokens in expectation to obtain $r _ { s } ,$ ensuring that the transition of the reverse process aligns with the forward process for accurate sampling [18–20]. In principle, the remasking strategy should be purely random. However, inspired by the annealing tricks of sampling in LLMs [4, 30], we adopt a low-confidence remasking strategy, where $\frac { s } { t }$ of predicted tokens with the lowest confidence are remarked based on the predictions, same as the approach of Chang et al. [23].

We mention that LLaDA enables flexible sampling. In particular, it supports autoregressive and block diffusion [31] sampling directly after the pre-training or SFT processes described above, without requiring any further modifications or training. We provide a detailed analysis in Appendix B.4. Nevertheless, the diffusion sampling (i.e., the reverse generation process) yields the best performance and is adopted as the default throughout this paper, especially for all experiments presented in Sec. 3.

For conditional likelihood evaluation, we can naturally utilize the upper bound in Eq. (5). However, we find that the following equivalent form [20] exhibits lower variance and is more stable:

$$
- \mathbb { E } _ { l , r _ { 0 } , r _ { l } } \left[ \frac { L } { l } \sum _ { i = 1 } ^ { L } \mathbf { 1 } [ r _ { l } ^ { i } = \mathbf { M } ] \log p _ { \theta } ( r _ { 0 } ^ { i } | p _ { 0 } , r _ { l } ) \right] ,\tag{6}
$$

where L is the sequence length of $r _ { 0 } , l$ is uniformly sampled from $\{ 1 , 2 , \ldots , L \}$ , and $r _ { l }$ is obtained by uniformly sampling l tokens from $r _ { 0 }$ without replacement for masking.

We present the training and inference algorithms, along with theoretical details, in Appendix A.

## 3 Experiments

We evaluate the scalability, instruction-following, and in-context learning capabilities of LLaDA on standard benchmarks, followed by analyses and case studies to provide a comprehensive assessment.

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 3: Scalability of LLaDA. We evaluate the performance of LLaDA and our ARM baselines trained on the same data across increasing pre-training computational FLOPs. LLaDA exhibits strong scalability, matching the overall performance of ARMs on six tasks.

## 3.1 Scalability of LLaDA on Language Tasks

We first investigate the scalability of LLaDA on downstream tasks in comparison with the ARM baselines we constructed. Specifically, at the 1B scale, we ensured that LLaDA and ARM shared the same architecture, data, and all other configurations. At larger scales, we also report results for LLaDA and ARM models of slightly different sizes trained on the same data due to resource limitations. Please refer to Appendix B.2 for more details. We use the pre-training computational cost as a unified scaling metric. For evaluation, we focused on six standard and diverse tasks.

Fig. 3 shows that LLaDA demonstrates impressive scalability, with its overall trend highly competitive with ARMs. Notably, on tasks such as MMLU and GSM8K, LLaDA exhibits even stronger scalability. Even on relatively weaker tasks like PIQA, the performance gap with ARMs narrows as scale increases. To account for the influence of outliers, we opted not to fit quantitative curves, avoiding potential misinterpretation. Nevertheless, the results clearly demonstrate the scalability of LLaDA.

Considering LLaDA’s advantages on certain benchmarks, we hypothesize that this performance gain stems from a key architectural difference: while autoregressive models optimize only left-to-right conditional probabilities, LLaDA is trained to consider multiple conditioning directions, as detailed in Appendix A.2, which may offer greater flexibility and lead to better generalization. This hypothesis is motivated by LLaDA’s strong performance on reversal reasoning in Sec. 3.3 and the ablation studies on sampling strategies in Appendix B.4.

Nie et al. [27] suggests that MDM requires 16 times more computation than ARM to achieve the same likelihood. However, key differences make our findings more broadly applicable. In particular, likelihood is a relatively indirect metric for downstream task performance, and diffusion optimizes a bound of the likelihood, making it not directly comparable to ARM. Additionally, we extended the scaling range from 1018 ∼ 1020 FLOPs in Nie et al. [27] to $1 0 ^ { 2 0 } \sim 1 0 ^ { 2 3 }$ FLOPs in this work.

## 3.2 Benchmark Results

To comprehensively evaluate the in-context learning and instruction-following capabilities of LLaDA 8B, we conducted detailed comparisons with existing LLMs [6, 21, 25, 26, 32, 33] of similar scale. Task selection and evaluation protocols followed existing studies, covering popular benchmarks in general tasks, mathematics, code, and Chinese. Further details are provided in Appendix B.6. For a more direct comparison, we re-evaluated representative LLMs [6, 21] in our implementation.

As shown in Tab. 1, after pretraining on 2.3T tokens, LLaDA 8B Base demonstrates remarkable performance, surpassing LLaMA2 7B Base on nearly all tasks, and is overall competitive with LLaMA3 8B Base. LLaDA shows advantages in math and Chinese tasks. We conjecture that the strengths stem from the same factors as its relatively weaker performance in some tasks—differences in data quality and distribution, largely due to the closed-source situation of LLM datasets.

Table 1: Benchmark Results of Pre-trained LLMs. ∗ indicates that models are evaluated under the same protocol, detailed in Appendix B.6. Results indicated by † and ¶ are sourced from Yang et al. [25, 26] and Bi et al. [32] respectively. The numbers in parentheses represent the number of shots used for in-context learning. “-” indicates unknown data.
<table><tr><td></td><td>LLaDA 8B*</td><td>LLaMA3 8B*</td><td>LLaMA2 7B*</td><td>Qwen2 7Bt</td><td>Qwen2.5 7Bt</td><td>Mistral 7B†</td><td>Deepseek 7B</td></tr><tr><td>Model</td><td>Diffusion</td><td>AR</td><td>AR</td><td>AR</td><td>AR</td><td>AR</td><td>AR</td></tr><tr><td>Training tokens</td><td>2.3T</td><td>15T</td><td>2T</td><td>7T</td><td>18T</td><td>-</td><td>2T</td></tr><tr><td colspan="8">General Tasks</td></tr><tr><td>MMLU</td><td>65.9 (5)</td><td>65.4 (5)</td><td>45.9 (5)</td><td>70.3 (5)</td><td>74.2 (5)</td><td>64.2 (5)</td><td>48.2 (5)</td></tr><tr><td>BBH</td><td>49.7(3)</td><td>62.1(3)</td><td>39.4(3)</td><td>62.3 (3)</td><td>70.4 (3)</td><td>56.1(3)</td><td>39.5 (3)</td></tr><tr><td>ARC-C</td><td>45.9 (0)</td><td>53.1(0)</td><td>46.3 (0)</td><td>60.6 (25)</td><td>63.7 (25)</td><td>60.0 (25)</td><td>48.1 (0)</td></tr><tr><td>Hellaswag</td><td>70.5 (0)</td><td>79.1(0)</td><td>76.0 (0)</td><td>80.7 (10)</td><td>80.2 (10)</td><td>83.3 (10)</td><td>75.4 (0)</td></tr><tr><td>TruthfulQA</td><td>46.1(0)</td><td>44.0 (0)</td><td>39.0 (0)</td><td>54.2(0)</td><td>56.4(0)</td><td>42.2(0)</td><td>=</td></tr><tr><td>WinoGrande PIQA</td><td>74.8 (5)</td><td>77.3(5)</td><td>72.5 (5)</td><td>77.0(5)</td><td>75.9 (5)</td><td>78.4(5)</td><td>70.5 (0)</td></tr><tr><td></td><td>73.6(0)</td><td>80.6 (0)</td><td>79.1 (0)</td><td>1</td><td></td><td>-</td><td>79.2 (0)</td></tr><tr><td colspan="8">Mathematics &amp; Science</td></tr><tr><td></td><td>70.3 (4)</td><td>48.7 (4)</td><td>13.1 (4)</td><td>80.2(4)</td><td>85.4(4)</td><td>36.2 (4)</td><td>17.4 (8)</td></tr><tr><td>GSM8K Math</td><td>31.4 (4)</td><td>16.0 (4)</td><td>4.3 (4)</td><td>43.5(4)</td><td>49.8 (4)</td><td>10.2 (4)</td><td>6.0 (4)</td></tr><tr><td>GPQA</td><td>25.2 (5)</td><td>25.9 (5)</td><td>25.7 (5)</td><td>30.8 (5)</td><td>36.4 (5)</td><td>24.7 (5)</td><td>-</td></tr><tr><td colspan="8">Code</td></tr><tr><td>HumanEval</td><td>35.4 (0)</td><td>34.8 (0)</td><td>12.8 (0)</td><td>51.2 (0)</td><td>57.9 (0)</td><td>29.3 (0)</td><td>26.2 (0)</td></tr><tr><td>HumanEval-FIM</td><td>73.8 (2)</td><td>73.3 (2)</td><td>26.9 (2)</td><td></td><td></td><td>=</td><td></td></tr><tr><td>MBPP</td><td>40.0 (4)</td><td>48.8 (4)</td><td>23.2(4)</td><td>64.2 (0)</td><td>74.9 (0)</td><td>51.1 (0)</td><td>39.0 (3)</td></tr><tr><td colspan="8"></td></tr><tr><td>CMMLU</td><td>69.9 (5)</td><td>50.7 (5)</td><td>Chinese 32.5 (5)</td><td>83.9 (5)</td><td></td><td></td><td>47.2 (5)</td></tr><tr><td>C-Eval</td><td>70.5(5)</td><td>51.7(5)</td><td>34.0 (5)</td><td>83.2(5)</td><td></td><td></td><td>45.0 (5)</td></tr></table>

Table 2: Benchmark Results of Post-trained LLMs. LLaDA only employs an SFT procedure, while other models have extra reinforcement learning (RL) alignment. ∗ indicates models are evaluated under the same protocol, detailed in Appendix B.6. Results indicated by † and ¶ are sourced from Yang et al. [26] and Bi et al. [32] respectively. The numbers in parentheses represent the number of shots used for in-context learning. “-” indicates unknown data.
<table><tr><td></td><td>LLaDA 8B*</td><td>LLaMA3 8B*</td><td>LLaMA2 7B*</td><td>Qwen2 7Bt</td><td>Qwen2.57Bt</td><td>Gemma2 9Bt</td><td>Deepseek 7B</td></tr><tr><td>Model</td><td>Diffusion</td><td>AR</td><td>AR</td><td>AR</td><td>AR</td><td>AR</td><td>AR</td></tr><tr><td>Training tokens</td><td>2.3T</td><td>15T</td><td>2T</td><td>7T</td><td>18T</td><td>8T</td><td>2T</td></tr><tr><td>Post-training Alignment pairs</td><td>SFT 4.5M</td><td>SFT+RL</td><td>SFT+RL</td><td>SFT+RL 0.5M+-</td><td>SFT+RL 1M+0.15M</td><td>SFT+RL</td><td>SFT+RL 1.5M+ -</td></tr><tr><td colspan="8">- = General Tasks</td></tr><tr><td colspan="8"></td></tr><tr><td>MMLU</td><td>65.5 (5)</td><td>68.4 (5)</td><td>44.1 (5)</td><td></td><td></td><td></td><td>49.4 (0)</td></tr><tr><td>MMLU-pro</td><td>37.0(0)</td><td>41.9 (0)</td><td>4.6(0)</td><td>44.1 (5)</td><td>56.3 (5)</td><td>52.1 (5)</td><td></td></tr><tr><td>Hellaswag</td><td>74.6(0)</td><td>75.5 (0)</td><td>51.5 (0)</td><td>-</td><td></td><td></td><td>68.5 (-)</td></tr><tr><td>ARC-C</td><td>88.5 (0)</td><td>82.4(0)</td><td>57.3 (0)</td><td>1</td><td>-</td><td>-</td><td>49.4 (-)</td></tr><tr><td colspan="8">Mathematics &amp; Science</td></tr><tr><td>GSM8K</td><td>69.4 (4)</td><td>78.3 (4)</td><td>29.0(4)</td><td>85.7(0)</td><td>91.6(0)</td><td>76.7(0)</td><td>63.0 (0)</td></tr><tr><td>Math</td><td>31.9 (0)</td><td>29.6(0)</td><td>3.8 (0)</td><td>52.9 (0)</td><td>75.5(0)</td><td>44.3 (0)</td><td>15.8(0)</td></tr><tr><td>GPQA</td><td>33.3 (5)</td><td>31.9 (5)</td><td>28.4(5)</td><td>34.3 (0)</td><td>36.4 (0)</td><td>32.8 (0)</td><td></td></tr><tr><td colspan="8">Code</td></tr><tr><td>HumanEval</td><td>49.4 (0)</td><td>59.8 (0)</td><td>16.5 (0)</td><td>79.9 (0)</td><td>84.8 (0)</td><td>68.9 (0)</td><td>48.2(-)</td></tr><tr><td>MBPP</td><td>41.0 (4)</td><td>57.6(4)</td><td>20.6(4)</td><td>67.2(0)</td><td>79.2 (0)</td><td>74.9 (0)</td><td>35.2(-)</td></tr></table>

Notably, we have carefully ruled out the possibility of data leakage by taking GSM8K as an example. First, as shown in Fig. 3, LLaDA outperformed ARM baselines regarding GSM8K. Moreover, the conclusion remains on a fully unseen GSM8K-like task [34] in Appendix B.8.

Further, Tab. 2 compares the performance of LLaDA 8B Instruct with existing LLMs. SFT improved LLaDA’s performance on most downstream tasks. A few metrics, such as MMLU, showed declines, possibly due to the suboptimal quality of the SFT data. Overall, since we did not perform alignment with reinforcement learning (RL), our results are slightly behind LLaMA3 8B Instruct, though the gaps in many metrics remain small. Notably, even with only SFT, LLaDA demonstrates impressive instruction-following abilities, as detailed in Sec. 3.4. We leave RL-based alignment for future work.

Table 3: Visualization of the Sampling Process and a Generated Multi-round Dialogue. In the response of LLaDA, darker colors indicate tokens predicted in the later stages of sampling, while lighter colors correspond to earlier predictions.
<table><tr><td colspan="7">Sampling Process</td></tr><tr><td rowspan="3">User LLaDA</td><td colspan="6">Lily can run 12 kilometers per hour for 4 hours.After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?</td></tr><tr><td>In</td><td>hours Lily runs</td><td>12</td><td>* 4 48</td><td>kilometers</td><td></td><td>After 4</td></tr><tr><td>hours</td><td>she runs 6</td><td>* 4 = 24</td><td>kilometers</td><td></td><td>In total</td><td>she runs 48</td></tr><tr><td colspan="3">+ 24 = 72 Multi-round Dialogue Case</td><td colspan="4">kilometers in 8 hours Result 72</td></tr><tr><td>Could you please share with me what the initial two lines of the renowned poem‘The</td></tr><tr><td>User LLaDA</td><td colspan="4">Road NotTaken&#x27;? Certainly!The initial two lines of the renowned poem“The Road Not Taken&quot;by Robert</td></tr><tr><td>User</td><td colspan="4">Frost are:“Two roads diverged in a yellow wood,And sorry Icould not travel both.&quot; These lines set the stage for the poem,introducing the idea of a speaker facing a choice between two paths. Please help me translate into Chinese.</td></tr><tr><td>LLaDA User</td><td colspan="4">Sure!The first two lines of“The Road Not Taken&quot;by Robert Frost can be translated into Chineseas:“两条路分岔在黄色的树林中，遗憾我不能同时走” Please translate into German.</td></tr><tr><td>LLaDA</td><td colspan="4">Sure!The frst two lines of“The Road Not Taken&quot;by Robert Frost can be translated into German as:“Zwei Wege trennten sich im gelben Wald,und leider konnte ich nicht beide tun.&quot; Please also write a poem about life choices.Irequire 5 sentences,and the first word of</td></tr><tr><td>User LLaDA</td><td colspan="4">each sentence starts with C as the first letter. Certainly! Here&#x27;s a poem about life choices,with each sentence starting with the letter “C&quot;:</td></tr><tr><td></td><td colspan="4">Choices cascade through life&#x27;s journey, Creating connections with each decision,Careful consideration leads to better paths, Courageous choices shape our destiny, Choosing wisely guides us through the years.</td></tr></table>

All results in Sec. 3 are based on pure diffusion methods, as they achieve better overall performance than approaches incorporating autoregressive components. Specifically, we use Eq. (6) for conditional likelihood estimation and apply low-confidence remasking for sampling. For LLaDA 8B Instruct, block diffusion style sampling performs better on GSM8K and Math, with scores of 78.6 and 42.2, compared to 69.4 and 31.9 in Tab. 2. This gain is due to extensive |EOS| token padding in the SFT data, causing early termination in low-confidence remasking. Please refer to Appendix B.4 for details.

Overall, despite the lack of data transparency, we have made every effort to adopt standardized procedures and introduce diverse tasks, we believe they sufficiently demonstrate the extraordinary capabilities of LLaDA, which is the only competitive non-autoregressive model to our knowledge.

## 3.3 Reversal Reasoning and Analyses

To quantify the reversal reasoning [15] ability of models, we follow the protocol established in Allen-Zhu and Li [35]. Specifically, we construct a dataset of 496 famous Chinese poem sentence pairs. Given a sentence from a poem, models are tasked with generating the subsequent line (forward) or the preceding line (reversal) without additional fine-tuning. Examples can be found in Section B.9. This setting provides a straightforward and more realistic evaluation compared to previous studies [27, 36].

As shown in Tab. 4, LLaDA effectively addresses the reversal curse [15], demonstrating consistent zero-shot performance across both forward and reversal tasks. In contrast, both Qwen 2.5 and GPT-4o exhibit a significant gap between the two. The results on forward generation confirm that both ARMs are strong, benefiting from significantly larger datasets and greater computational resources than LLaDA. However, LLaDA outperforms both by a large margin in the reversal task.

We did not design anything special for reversal tasks. Intuitively, LLaDA treats tokens uniformly without inductive bias, leading to balanced performance. See Appendix A.2 for details.

Table 4: Comparison on the Poem Completion task.

We also analyze the effect of different sampling strategies for LLaDA, including autoregressive sampling, block diffusion [31] sampling, and pure diffusion sampling, showing that pure diffusion sampling achieves the best overall performance, as detailed in Appendix B.4.

<table><tr><td></td><td>Forward</td><td>Reversal</td></tr><tr><td>GPT-4o (2024-08-06)</td><td>82.7</td><td>34.3</td></tr><tr><td>Qwen2.5-7B Instruct</td><td>75.9</td><td>38.0</td></tr><tr><td>LLaDA-8B Instruct</td><td>51.8</td><td>45.6</td></tr></table>

In addition, we examine LLaDA’s sampling speed and memory consumption, showing that it enables a flexible trade-off between generation quality and speed. See Appendix B.7 for more details.

Classifier-free guidance (CFG) [37, 27] is a widely used technique in diffusion models to improve generation quality. To ensure a fair comparison with ARMs, we do not apply CFG to LLaDA in the main text. However, we show that LLaDA is compatible with CFG and consistently benefits from its application. See Appendix B.3 for more details.

## 3.4 Case Studies

We present samples generated by LLaDA 8B Instruct in Tab. 3, showcasing its instruction-following capabilities. First, the table illustrates LLaDA’s ability to generate coherent, fluent, and extended text in a non-autoregressive manner. Second, it highlights the model’s multi-turn dialogue capability, effectively retaining conversation history and producing contextually appropriate responses across multiple languages. Such chat capabilities of LLaDA are impressive, as it departs from conventional ARMs for the first time, to the best of our knowledge. See more case studies in Appendix B.10.

## 4 Related Work

Diffusion models [38–40] have achieved remarkable success in visual domains but remain unverified for large-scale (e.g., models trained with over $1 0 ^ { 2 3 }$ FLOPs) language modeling, despite growing interest and extensive research efforts.

A simple approach is to continuousize text data and apply continuous diffusion models directly [41– 51]. Alternatively, some methods model continuous parameters of discrete distributions instead [52– 56]. However, scalability remains a significant challenge for these approaches. For instance, a 1B model may require 64 times the compute of an ARM to achieve comparable performance [57].

Another approach replaces continuous diffusion with discrete processes featuring new forward and reverse dynamics, leading to numerous variants [58–71]. The original diffusion model paper [38] introduced both continuous-state and discrete-state transition kernels under a unified diffusion framework. Austin et al. [16] was among the pioneering works that introduced discrete diffusion models into language modeling, demonstrating the feasibility of this approach. Lou et al. [17] showed that masked diffusion, as a special case of discrete diffusion, achieves perplexity comparable to or surpassing ARMs at GPT-2 scale. Shi et al. [18], Sahoo et al. [19], Ou et al. [20] established fundamental theoretical results, which motivated our model design, training, and inference (see Appendix A for details). Nie et al. [27] introduced the scaling laws for MDMs in language modeling and explored how MDMs can be leveraged for language tasks such as question answering at the GPT-2 scale. Gong et al. [72] demonstrated the potential of fine-tuning an ARM within the MDM framework. However, the improvements observed by Gong et al. [72] are limited to specific metrics, and their approach does not address the performance achievable through pure diffusion-based training. Concurrent work [73] demonstrates the potential of diffusion language models in code generation and highlights their advantages in inference efficiency. Nonetheless, as it is a closed-source product, specific details such as training procedures and sampling methods remain unknown.

In comparison, this study scales MDM to an unprecedented size of 8B parameters from scratch, achieving performance comparable to leading LLMs such as LLaMA 3.

Additionally, a parallel line of work on image generation [23, 74, 75] aligns well with the application of MDMs to text data. Moreover, MDMs have also shown promise in other domains such as protein generation [76, 77], where they have achieved promising results. Notably, a series of studies [31, 78– 87] have explored techniques such as architectural optimization, distillation, and sampling algorithm design to accelerate MDMs sampling.

## 5 Conclusion and Discussion

We introduce LLaDA, a diffusion language model trained from scratch with an unprecedented scale of 8B parameters. LLaDA demonstrates strong capabilities in scalability, in-context learning, and instruction-following, achieving performance comparable to strong LLMs such as LLaMA3. In addition, LLaDA offers unique advantages, such as bidirectional modeling and enhanced robustness, effectively addressing the relevant limitations of existing LLMs. Our findings show the promise of diffusion models for language modeling at scale and challenge the common assumption that these essential capabilities are inherently tied to ARMs. These results represent a new paradigm for language modeling and uncover novel insights, demonstrating a high degree of scientific innovation.

Limitations. While promising, the full potential of diffusion models remains to be fully explored. Several limitations of this work present significant opportunities for future research. The generation length is a user-specified hyperparameter. Although LLaDA is insensitive to this hyperparameter as detailed in Appendix B.5, we believe that adopting an adaptive generation length would offer a more efficient solution. Due to computational constraints, direct comparisons between LLaDA and ARMs—such as training on identical datasets—were restricted to a computational budget of less than $1 0 ^ { 2 3 }$ FLOPs. To allocate resources for training the largest possible LLaDA model and showcasing its potential, we were unable to scale the ARM baseline to the same extent. Moreover, no specialized attention mechanisms or position embeddings were designed for LLaDA, nor were any system-level architectural optimizations such as KV cache applied. On the inference side, more efficient and controllable [37, 88, 89] sampling algorithms remain preliminary. Furthermore, LLaDA has yet to undergo alignment with reinforcement learning [90, 91], which is crucial for improving its performance and alignment with human intent.

Looking ahead, both the model scale and the amount of training data for LLaDA remain smaller than those of leading ARM counterparts [6, 26, 92–95], highlighting the need for further scaling to fully evaluate its capabilities. In addition, LLaDA’s ability to process multi-modal data remains unexplored. Its impact on prompt tuning techniques [96] and integration into agent-based systems [97, 98] is still not fully understood. Finally, a systematic investigation into post-training for LLaDA (e.g., O1-like systems [99, 100]) is needed to further unlock the potential of diffusion language models.

## Acknowledgements

This work was supported by the National Natural Science Foundation of China (No. 92470118); Beijing Natural Science Foundation (No. L247030); Beijing Nova Program (No. 20220484044); and Ant Group Research Fund.

## References

[1] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023.

[2] Alec Radford. Improving language understanding by generative pre-training, 2018.

[3] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

[4] Tom B Brown. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.

[5] OpenAI. ChatGPT: Optimizing Language Models for Dialogue. OpenAI blog, November 2022. URL https://openai.com/blog/chatgpt/.

[6] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

[7] Ashish Vaswani. Attention is all you need. arXiv preprint arXiv:1706.03762, 2017.

[8] Ronald A Fisher. On the mathematical foundations of theoretical statistics. Philosophical transactions of the Royal Society of London. Series A, containing papers of a mathematical or physical character, 222(594-604):309–368, 1922.

[9] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22669–22679, 2023.

[10] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.

[11] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024. URL https://openai.com/research/ video-generation-models-as-world-simulators.

[12] Gregoire Deletang, Anian Ruoss, Paul-Ambroise Duquenne, Elliot Catt, Tim Genewein, Christopher Mattern, Jordi Grau-Moya, Li Kevin Wenliang, Matthew Aitchison, Laurent Orseau, et al. Language modeling is compression. In The Twelfth International Conference on Learning Representations.

[13] Yuzhen Huang, Jinghan Zhang, Zifei Shan, and Junxian He. Compression represents intelligence linearly. arXiv preprint arXiv:2404.09937, 2024.

[14] Claude Elwood Shannon. A mathematical theory of communication. The Bell system technical journal, 27(3):379–423, 1948.

[15] Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, and Owain Evans. The reversal curse: Llms trained on" a is b" fail to learn" b is a". arXiv preprint arXiv:2309.12288, 2023.

[16] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in Neural Information Processing Systems, 34:17981–17993, 2021.

[17] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion language modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023.

[18] Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K Titsias. Simplified and generalized masked diffusion for discrete data. arXiv preprint arXiv:2406.04329, 2024.

[19] Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. arXiv preprint arXiv:2406.07524, 2024.

[20] Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. arXiv preprint arXiv:2406.03736, 2024.

[21] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[22] Jacob Devlin. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

[23] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11315–11325, 2022.

[24] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multihead checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4895–4901, 2023.

[25] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024. URL https://arxiv.org/abs/2407.10671.

[26] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

[27] Shen Nie, Fengqi Zhu, Chao Du, Tianyu Pang, Qian Liu, Guangtao Zeng, Min Lin, and Chongxuan Li. Scaling up masked diffusion models on text. arXiv preprint arXiv:2410.18514, 2024.

[28] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.

[29] I Loshchilov. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

[30] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751, 2019.

[31] Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. arXiv preprint arXiv:2503.09573, 2025.

[32] Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954, 2024.

[33] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

[34] Tian Ye, Zicheng Xu, Yuanzhi Li, and Zeyuan Allen-Zhu. Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process. ArXiv e-prints, abs/2407.20311, July 2024. Full version available at http://arxiv.org/abs/2407.20311.

[35] Zeyuan Allen-Zhu and Yuanzhi Li. Physics of Language Models: Part 3.2, Knowledge Manipulation. ArXiv e-prints, abs/2309.14402, September 2023. Full version available at http://arxiv.org/abs/2309.14402.

[36] Ouail Kitouni, Niklas Nolte, Diane Bouchacourt, Adina Williams, Mike Rabbat, and Mark Ibrahim. The factorization curse: Which tokens you predict underlie the reversal curse and more. arXiv preprint arXiv:2406.05183, 2024.

[37] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

[38] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015.

[39] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[40] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

[41] Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. Diffusion-lm improves controllable text generation. Advances in Neural Information Processing Systems, 35:4328–4343, 2022.

[42] Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and LingPeng Kong. Diffuseq: Sequence to sequence text generation with diffusion models. arXiv preprint arXiv:2210.08933, 2022.

[43] Xiaochuang Han, Sachin Kumar, and Yulia Tsvetkov. Ssd-lm: Semi-autoregressive simplexbased diffusion language model for text generation and modular control. arXiv preprint arXiv:2210.17432, 2022.

[44] Robin Strudel, Corentin Tallec, Florent Altché, Yilun Du, Yaroslav Ganin, Arthur Mensch, Will Grathwohl, Nikolay Savinov, Sander Dieleman, Laurent Sifre, et al. Self-conditioned embedding diffusion for text generation. arXiv preprint arXiv:2211.04236, 2022.

[45] Ting Chen, Ruixiang Zhang, and Geoffrey Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. arXiv preprint arXiv:2208.04202, 2022.

[46] Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, et al. Continuous diffusion for categorical data. arXiv preprint arXiv:2211.15089, 2022.

[47] Pierre H. Richemond, Sander Dieleman, and Arnaud Doucet. Categorical sdes with simplex diffusion, 2022.

[48] Tong Wu, Zhihao Fan, Xiao Liu, Yeyun Gong, Yelong Shen, Jian Jiao, Hai-Tao Zheng, Juntao Li, Zhongyu Wei, Jian Guo, Nan Duan, and Weizhu Chen. Ar-diffusion: Auto-regressive diffusion model for text generation, 2023.

[49] Rabeeh Karimi Mahabadi, Hamish Ivison, Jaesung Tae, James Henderson, Iz Beltagy, Matthew E. Peters, and Arman Cohan. Tess: Text-to-text self-conditioned simplex diffusion, 2024.

[50] Jiasheng Ye, Zaixiang Zheng, Yu Bao, Lihua Qian, and Mingxuan Wang. Dinoiser: Diffused conditional sequence learning by manipulating noises. arXiv preprint arXiv:2302.10025, 2023.

[51] Yizhe Zhang, Jiatao Gu, Zhuofeng Wu, Shuangfei Zhai, Joshua Susskind, and Navdeep Jaitly. Planner: Generating diversified paragraph via latent language diffusion model. Advances in Neural Information Processing Systems, 36:80178–80190, 2023.

[52] Aaron Lou and Stefano Ermon. Reflected diffusion models, 2023.

[53] Alex Graves, Rupesh Kumar Srivastava, Timothy Atkinson, and Faustino Gomez. Bayesian flow networks. arXiv preprint arXiv:2308.07037, 2023.

[54] Zhenghao Lin, Yeyun Gong, Yelong Shen, Tong Wu, Zhihao Fan, Chen Lin, Nan Duan, and Weizhu Chen. Text generation with diffusion language models: A pre-training approach with continuous paragraph denoise. In International Conference on Machine Learning, pages 21051–21064. PMLR, 2023.

[55] Kaiwen Xue, Yuhao Zhou, Shen Nie, Xu Min, Xiaolu Zhang, Jun Zhou, and Chongxuan Li. Unifying bayesian flow networks and diffusion models through stochastic differential equations. arXiv preprint arXiv:2404.15766, 2024.

[56] Ruixiang Zhang, Shuangfei Zhai, Yizhe Zhang, James Thornton, Zijing Ou, Joshua Susskind, and Navdeep Jaitly. Target concrete score matching: A holistic framework for discrete diffusion. arXiv preprint arXiv:2504.16431, 2025.

[57] Ishaan Gulrajani and Tatsunori B Hashimoto. Likelihood-based diffusion language models. Advances in Neural Information Processing Systems, 36, 2024.

[58] Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. Advances in Neural Information Processing Systems, 34:12454–12465, 2021.

[59] Emiel Hoogeboom, Alexey A Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, and Tim Salimans. Autoregressive diffusion models. arXiv preprint arXiv:2110.02037, 2021.

[60] Zhengfu He, Tianxiang Sun, Kuanning Wang, Xuanjing Huang, and Xipeng Qiu. Diffusionbert: Improving generative masked language models with diffusion models. arXiv preprint arXiv:2211.15029, 2022.

[61] Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. Advances in Neural Information Processing Systems, 35:28266–28279, 2022.

[62] Chenlin Meng, Kristy Choi, Jiaming Song, and Stefano Ermon. Concrete score matching: Generalized score matching for discrete data. Advances in Neural Information Processing Systems, 35:34532–34545, 2022.

[63] Machel Reid, Vincent J. Hellendoorn, and Graham Neubig. Diffuser: Discrete diffusion via edit-based reconstruction, 2022.

[64] Haoran Sun, Lijun Yu, Bo Dai, Dale Schuurmans, and Hanjun Dai. Score-based continuoustime discrete diffusion models. arXiv preprint arXiv:2211.16750, 2022.

[65] Ouail Kitouni, Niklas Nolte, James Hensman, and Bhaskar Mitra. Disk: A diffusion model for structured knowledge. arXiv preprint arXiv:2312.05253, 2023.

[66] Lin Zheng, Jianbo Yuan, Lei Yu, and Lingpeng Kong. A reparameterized discrete diffusion model for text generation. ArXiv, abs/2302.05737, 2023.

[67] Zixiang Chen, Huizhuo Yuan, Yongqian Li, Yiwen Kou, Junkai Zhang, and Quanquan Gu. Fast sampling via de-randomization for discrete diffusion models. arXiv preprint arXiv:2312.09193, 2023.

[68] Jiasheng Ye, Zaixiang Zheng, Yu Bao, Lihua Qian, and Quanquan Gu. Diffusion language models can perform many tasks with scaling and instruction-finetuning. arXiv preprint arXiv:2308.12219, 2023.

[69] Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky TQ Chen, Gabriel Synnaeve, Yossi Adi, and Yaron Lipman. Discrete flow matching. arXiv preprint arXiv:2407.15595, 2024.

[70] Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling, 2024. URL https://arxiv.org/abs/2409.02908.

[71] Shreyas Kapur, Erik Jenner, and Stuart Russell. Diffusion on syntax trees for program synthesis. arXiv preprint arXiv:2405.20519, 2024.

[72] Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, et al. Scaling diffusion language models via adaptation from autoregressive models. arXiv preprint arXiv:2410.17891, 2024.

[73] Samar Khanna, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, Stefano Ermon, et al. Mercury: Ultra-fast language models based on diffusion. arXiv preprint arXiv:2506.17298, 2025.

[74] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023.

[75] Zebin You, Jingyang Ou, Xiaolu Zhang, Jun Hu, Jun Zhou, and Chongxuan Li. Effective and efficient masked image generation models. arXiv preprint arXiv:2503.07197, 2025.

[76] Xinyou Wang, Zaixiang Zheng, Fei Ye, Dongyu Xue, Shujian Huang, and Quanquan Gu. Diffusion language models are versatile protein learners. arXiv preprint arXiv:2402.18567, 2024.

[77] Xinyou Wang, Zaixiang Zheng, Fei Ye, Dongyu Xue, Shujian Huang, and Quanquan Gu. Dplm-2: A multimodal diffusion protein language model. arXiv preprint arXiv:2410.13782, 2024.

[78] Siqi Kou, Lanxiang Hu, Zhezhi He, Zhijie Deng, and Hao Zhang. Cllms: Consistency large language models. arXiv preprint arXiv:2403.00835, 2024.

[79] Chenkai Xu, Xu Wang, Zhenyi Liao, Yishun Li, Tianqi Hou, and Zhijie Deng. Show-o turbo: Towards accelerated unified multimodal understanding and generation. arXiv preprint arXiv:2502.05415, 2025.

[80] Sulin Liu, Juno Nam, Andrew Campbell, Hannes Stärk, Yilun Xu, Tommi Jaakkola, and Rafael Gómez-Bombarelli. Think while you generate: Discrete diffusion with planned denoising. arXiv preprint arXiv:2410.06264, 2024.

[81] Yuanzhi Zhu, Xi Wang, Stéphane Lathuilière, and Vicky Kalogeiton. Dimo: Distilling masked diffusion models into one-step generator. arXiv preprint arXiv:2503.15457, 2025.

[82] Yinuo Ren, Haoxuan Chen, Yuchen Zhu, Wei Guo, Yongxin Chen, Grant M Rotskoff, Molei Tao, and Lexing Ying. Fast solvers for discrete diffusion models: Theory and applications of high-order algorithms. arXiv preprint arXiv:2502.00234, 2025.

[83] Satoshi Hayakawa, Yuhta Takida, Masaaki Imaizumi, Hiromi Wakaki, and Yuki Mitsufuji. Distillation of discrete diffusion through dimensional correlations. arXiv preprint arXiv:2410.08709, 2024.

[84] Yixiu Zhao, Jiaxin Shi, Feng Chen, Shaul Druckmann, Lester Mackey, and Scott Linderman. Informed correctors for discrete diffusion models. arXiv preprint arXiv:2407.21243, 2024.

[85] Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. arXiv preprint arXiv:2409.02908, 2024.

[86] Yong-Hyun Park, Chieh-Hsin Lai, Satoshi Hayakawa, Yuhta Takida, and Yuki Mitsufuji. Jump your steps: Optimizing sampling schedule of discrete diffusion models. In The Thirteenth International Conference on Learning Representations, 2024.

[87] Justin Deschenaux and Caglar Gulcehre. Beyond autoregression: Fast llms via self-distillation through time. arXiv preprint arXiv:2410.21035, 2024.

[88] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

[89] Yair Schiff, Subham Sekhar Sahoo, Hao Phung, Guanghan Wang, Sam Boshar, Hugo Dallatorre, Bernardo P de Almeida, Alexander Rush, Thomas Pierrot, and Volodymyr Kuleshov. Simple guidance mechanisms for discrete diffusion models. arXiv preprint arXiv:2412.10193, 2024.

[90] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

[91] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.

[92] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

[93] Google. Our next-generation model: Gemini 1.5, 2024. URL https://blog.google/ technology/ai/google-gemini-next-generation-model-february-2024.

[94] Anthropic. Claude 3.5 sonnet, 2024. URL https://www.anthropic.com/news/ claude-3-5-sonnet.

[95] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

[96] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

[97] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22, 2023.

[98] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345, 2024.

[99] OpenAI. Learning to reason with llms, 2024. URL https://openai.com/index/ learning-to-reason-with-llms/.

[100] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[101] Benigno Uria, Iain Murray, and Hugo Larochelle. A deep and tractable density estimator. In Proceedings of the 31th International Conference on Machine Learning, 2014.

[102] Andy Shih, Dorsa Sadigh, and Stefano Ermon. Training and inference on any-order autoregressive models the right way. In Proceedings of the 31th International Conference on Machine Learning, 2022.

[103] Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen Lin. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. arXiv preprint arXiv:2406.08464, 2024.

[104] Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, and Lingming Zhang. Magicoder: Empowering code generation with oss-instruct. arXiv preprint arXiv:2312.02120, 2023.

[105] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.

[106] Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.

[107] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

[108] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

[109] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

[110] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

[111] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. Challenging bigbench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.

[112] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

[113] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

[114] Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. arXiv preprint arXiv:2109.07958, 2021.

[115] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.

[116] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, 2020.

[117] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[118] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

[119] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022, 2023.

[120] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

[121] Mohammad Bavarian, Heewoo Jun, Nikolas Tezak, John Schulman, Christine McLeavey, Jerry Tworek, and Mark Chen. Efficient training of language models to fill in the middle. arXiv preprint arXiv:2207.14255, 2022.

[122] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

[123] Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. Cmmlu: Measuring massive multitask language understanding in chinese. arXiv preprint arXiv:2306.09212, 2023.

[124] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Yao Fu, et al. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. Advances in Neural Information Processing Systems, 36, 2024.

[125] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 07 2024. URL https://zenodo.org/ records/12608602.

Algorithm 1 Pre-training of LLaDA   
Require: mask predictor $p _ { \theta } ,$ , data distribution $p _ { \mathrm { d a t a } }$   
1: repeat   
2: $x _ { 0 } \sim p _ { \mathrm { d a t a } }$ # with a probability of 1%, the sequence length of $x _ { 0 }$ follows U[1, 4096]   
3: $t \sim \dot { \mathrm { U } ( 0 , 1 ] }$   
4: $x _ { t } \sim q _ { t | 0 } ( x _ { t } | x _ { 0 } )$ # $q _ { t | 0 }$ is defined in Eq. (7)   
5: Calculate $\begin{array} { r } { \mathcal { L } = - \frac { 1 } { t * L } \sum _ { i = 1 } ^ { L } \mathbf { 1 } [ x _ { t } ^ { i } = \mathbf { M } ] } \end{array}$ ] log $p _ { \theta } ( x _ { 0 } ^ { i } | x _ { t } )$ # L is the sequence length of $x _ { 0 }$   
6: Calculate $\nabla _ { \boldsymbol { \theta } } \mathcal { L }$ and run optimizer.   
7: until Converged   
8: Return $p _ { \theta }$

Algorithm 2 Supervised Fine-Tuning of LLaDA   
Require: mask predictor $p _ { \theta }$ , pair data distribution $p _ { \mathrm { d a t a } }$   
1: repeat   
2: $p _ { 0 } , r _ { 0 } \sim p _ { \mathrm { d a t a } }$ # please refer to Appendix B.1 for details about the SFT dat   
3: $t \sim \mathrm { U } ( 0 , 1 ]$   
4: $r _ { t } \sim q _ { t | 0 } ( \bar { r } _ { t } | r _ { 0 } )$ # $q _ { t | 0 }$ is defined in Eq. (7)   
5: Calculate $\begin{array} { r } { \underline { { \mathcal { L } } } = - \frac { 1 } { t * L ^ { \prime } } \sum _ { i = 1 } ^ { L ^ { \prime } } \mathbf { 1 } [ r _ { t } ^ { i } = \mathbf { M } ] } \end{array}$ log $p _ { \theta } ( r _ { 0 } ^ { i } | p _ { 0 } , r _ { t } )$ # $L ^ { \prime }$ is the sequence length of $r _ { 0 }$   
6: Calculate $\nabla _ { \boldsymbol { \theta } } \mathcal { L }$ and run optimizer.   
7: until Converged   
8: Return $p _ { \theta }$

Algorithm 3 Conditional Log-likelihood Evaluation of LLaDA   
Require: mask predictor $p _ { \theta }$ , prompt $p _ { 0 } .$ response $r _ { 0 } .$ , the number of Monte Carlo estimations $n _ { m c }$   
1: log_likelihood $= 0$   
2: for $i \gets 1$ to $n _ { m c }$ do   
3: $l \sim \{ 1 , 2 , \ldots , L \}$ # L is the sequence length of $r _ { 0 }$   
4: Obtain $r _ { l }$ by uniformly sampling l tokens from $r _ { 0 }$ without replacement for masking   
5: log_likelihood = log_likelihood $\begin{array} { r } { + \frac { L } { l } \sum _ { i = 1 } ^ { L } \mathbf { 1 } [ r _ { l } ^ { i } = \mathbf { M } ] \log p _ { \theta } ( r _ { 0 } ^ { i } | p _ { 0 } , r _ { l } ) } \end{array}$   
6: end for   
7: log_likelihood = log_likelihood $\cdot / n _ { m c }$   
8: Return log_likelihood

## A Formulation of Masked Diffusion Models

## A.1 Training

MDMs [16–20] define the model distribution $p _ { \theta } ( x _ { 0 } )$ in a manner distinct from autoregressive models. These models introduce a forward process $\{ x _ { t } \}$ indexed by a time $t \in [ 0 , 1 ]$ . This process gradually and independently masks all tokens in the sequence $x _ { 0 }$ . At time $t = 0$ , the data point $x _ { 0 }$ is fully observed with no masks, while for $t \in ( 0 , 1 ] , x _ { t }$ represents latent variables with varying mask ratios in expectation.

Formally, the conditional distribution of $x _ { t }$ given $x _ { 0 }$ is defined by a fully factorized form:

$$
q _ { t | 0 } ( x _ { t } | x _ { 0 } ) = \prod _ { i = 1 } ^ { L } q _ { t | 0 } ( x _ { t } ^ { i } | x _ { 0 } ^ { i } ) ,\tag{7}
$$

where the conditional distribution for each token is given by:

$$
q _ { t | 0 } ( x _ { t } ^ { i } | x _ { 0 } ^ { i } ) = \left\{ \begin{array} { l l } { 1 - t , } & { x _ { t } ^ { i } = x _ { 0 } ^ { i } , } \\ { t , } & { x _ { t } ^ { i } = \mathbf { M } . } \end{array} \right.\tag{8}
$$

Here, M denotes the mask token. Intuitively, each token either remains unchanged or is masked, with the probability of being masked increasing linearly as t progresses from 0 to 1. $\mathbf { A } \mathbf { t } \ t = 1$ , all tokens are guaranteed to be masked, meaning that $x _ { 1 }$ follows a Dirac distribution concentrated on a sequence of fully masked tokens. Notably, the linear masking probability is analogous to but distinct from, the noise schedule in continuous diffusion models [38–40]. This linearity is motivated by the assumption that the information in the text is proportional to the number of tokens on average, making it reasonable to lose information linearly during the forward process.

Algorithm 4 Random Remasking Strategy of LLaDA   
Require: mask predictor $p _ { \theta } .$ , prompt $p _ { 0 } ,$ answer length $L ,$ sampling steps N   
1: Set $r _ { 1 }$ is a fully masked sequence of length L.   
2: for $t \gets 1$ down to $\textstyle { \frac { 1 } { N } }$ step $\scriptstyle { \frac { 1 } { N } }$ do   
3: $\begin{array} { r } { s = t - \frac { 1 } { N } } \end{array}$   
4: r0 = arg ma $\tau _ { r _ { 0 } } p _ { \theta } ( r _ { 0 } | p _ { 0 } , r _ { t } )$ # we employ greedy sampling when predicting masked tokens   
5: for $i \gets \bar { 1 }$ to L do   
6: $\mathbf { i f } r _ { t } ^ { i } \neq \mathbf { M }$ then   
7: $\dot { r _ { 0 } ^ { i } } = r _ { t } ^ { i }$   
8: else   
9: with probability $\textstyle { \frac { s } { t } } , r _ { 0 } ^ { i }$ is set to M   
10: end if   
11: end for   
12: $r _ { s } = r _ { 0 }$   
13: end for   
14 : Return $r _ { 0 }$

The forward process is not only reversible but also corresponds to a reverse process that is fully factorized across all tokens. The reverse process, from time $t = 1$ to 0, generates new data from sequences of fully masked tokens. The conditional distribution for the reverse process, for $0 \leq s <$ $t \leq 1$ , is factorized as:

$$
q _ { s | t } ( x _ { s } | x _ { t } ) = \prod _ { i = 1 } ^ { L } q _ { s | t } ( x _ { s } ^ { i } | x _ { t } ) ,\tag{9}
$$

where the conditional distribution for each token is:

$$
\begin{array} { r } { q _ { s | t } ( x _ { s } ^ { i } | x _ { t } ) = \left\{ \begin{array} { l l } { 1 , } & { x _ { t } ^ { i } \neq \mathbf { M } , x _ { s } ^ { i } = x _ { t } ^ { i } , } \\ { \frac { s } { t } , } & { x _ { t } ^ { i } = \mathbf { M } , x _ { s } ^ { i } = \mathbf { M } , } \\ { \frac { t - s } { t } q _ { 0 | t } ( x _ { s } ^ { i } | x _ { t } ) , } & { x _ { t } ^ { i } = \mathbf { M } , x _ { s } ^ { i } \neq \mathbf { M } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{10}
$$

Thus, the key function to estimate is the conditional distribution $q _ { 0 | t } \big ( x _ { s } ^ { i } | x _ { t } \big )$ , which predicts the original token if it is masked in the input $x _ { t }$ . This is analogous to the data prediction form in continuous diffusion models.

As proven in [20], an equivalent yet time-free parameterization can be derived as:

$$
q _ { 0 | t } \big ( x _ { s } ^ { i } | x _ { t } \big ) = p _ { \mathrm { d a t a } } \big ( x _ { 0 } ^ { i } | x _ { t } ^ { \mathrm { U M } } \big ) , \quad \forall i \ \mathrm { s u c h } \ \mathrm { t h a t } \ x _ { t } ^ { i } = \mathbf { M } ,\tag{11}
$$

where $x _ { t } ^ { \mathrm { U M } }$ denotes the collection of unmasked tokens in $x _ { t }$ , which is identical to the corresponding tokens in the original data $x _ { 0 }$ since unmasked tokens are solely determined by $x _ { 0 }$ and are independent of time t. Intuitively, this implies that estimating the data prediction function is equivalent to estimating the conditional distributions on clean data, which is time-invariant. Consequently, the time t need not be provided as input to the parametric model.

Although the development of masked diffusion is nontrivial, the implementation is straightforward. We first introduce the mask predictor, a parametric model $p _ { \theta } ( \cdot | x _ { t } )$ (e.g., a Transformer without causal mask), which takes $x _ { t }$ for any t as input and predict all masked tokens simultaneously. Then, we define the model distribution $p _ { \theta } ( x _ { 0 } )$ as follows: starting with $x _ { 1 }$ as a sequence of fully masked tokens, we simulate an approximate reverse process parameterized by $p _ { \theta } ( \cdot | x _ { t } )$ from $t = 1$ to 0. The marginal distribution induced at $t = 0$ then represents the model distribution $p _ { \theta } ( x _ { 0 } )$