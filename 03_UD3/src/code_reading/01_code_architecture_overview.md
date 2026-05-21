# 01 — UD3 Code Architecture Overview

## File Structure

The entire USD3 (Unified Discrete Diffusion for Categorical Data) implementation is a single file:

```
discrete_diffusion.py  (699 lines)
```

It contains no neural network definitions — only the diffusion **process** logic. The user must provide a `denoising_fn` (a PyTorch network) separately.

## Code Layout

```
1. Probability Helpers     (lines 8–26)    — sampling & softmax utilities
2. Index Helpers           (lines 28–62)   — tensor gather / scatter
3. Noise Schedule          (lines 64–118)  — 6 schedule types
4. UnifiedDiscreteDiffusion (lines 120–699) — main class
   ├── Forward process     (q-distributions)
   ├── Backward process    (p-distributions, analytical + log-space)
   ├── Loss: discrete-time (ELBO + CE)
   ├── Loss: continuous-time (ELBO + CE)
   ├── MCMC corrector      (Gibbs-like refinement)
   └── Sampling loop       (T → 0 ancestral sampling)
```

## Key Design Idea: Unified Discrete & Continuous Time

The `num_steps` parameter controls the time regime:

| `num_steps` | Time regime    | Loss used                     |
|-------------|----------------|-------------------------------|
| `0`         | Continuous-time| `continuous_time_loss()`      |
| `> 0`       | Discrete-time  | `discrete_time_loss()`        |

This unification is a core contribution of the UD3 paper.

## Noise Distribution `m`

The `m` argument is the **stationary noise distribution**. It can be:

- `None` → uniform over `num_classes`
- A 1D tensor `(C,)` → shared across all positions
- A full tensor `(B, ..., C)` → per-position noise

It appears in every forward, backward, and loss function, and is also used as the prior `p(x_T)`.

## Noise Schedules (6 types)

| Schedule      | Key formula                         | Use case              |
|---------------|-------------------------------------|-----------------------|
| `cosine`      | $\bar\alpha_t = \cos(\frac{t+\alpha}{1+\alpha}\frac{\pi}{2})$ | Default, smooth decay |
| `exponential` | $\bar\alpha_t = \exp(a t (b^0 - b^{t/T}))$ | Flexible rate control |
| `linear`      | $\bar\alpha_t = 1 - t/T$            | Simple baseline       |
| `constant`    | $\bar\alpha_t = e^{-a t}$           | Uniform rate          |
| `geometric`   | $\sigma$-based interpolation        | Variance-preserving   |
| `loglinear`   | $1/(1-(1-\epsilon)t)$              | Score-matching common |

## Forward Process (q)

| Method                | What it computes                        | Used in                |
|-----------------------|-----------------------------------------|------------------------|
| `qt_0_sample`         | Sample $x_t \sim q(x_t \mid x_0)$       | Training data corruption |
| `qt_0_prob`           | $q(x_t \mid x_0)$ full probability      | Continuous-time loss   |
| `qs_t0_prob`          | $q(x_s \mid x_t, x_0)$  (s < t)         | Discrete-time loss (analytical KL) |

**Sampling trick**: Instead of sampling from the full categorical, the code uses a **branch indicator** $b_t \sim \text{Bernoulli}(\bar\alpha_t)$:
- $b_t = 1$ → keep $x_0$ (no corruption yet)
- $b_t = 0$ → sample from noise $m$

This makes forward sampling efficient and clean.

## Backward Process (p)

| Method                | What it computes                        |
|-----------------------|-----------------------------------------|
| `ps_t_prob`           | $p_\theta(x_s \mid x_t)$ (probability)  |
| `ps_t_logprob`        | $p_\theta(x_s \mid x_t)$ (log-space, numerically stable) |
| `ps_t0_delta`         | $p_\theta - q_{s\mid t,0}$ difference   | for simplified VLB |

The backward step uses three coefficients that decompose the transition:

- $\mu_{t|s}$: probability of keeping the token
- $\lambda_{t|s}$: probability of $x_t$ originating from $x_0$ vs noise
- $\gamma_{t|s}$: correction from the denoising network's prediction

## Loss Functions

### Discrete-time (lines 389–429)
$$
\mathcal{L} = \mathbb{E}_{t\sim[1,T]}\big[ \underbrace{\text{KL}(p_\theta(x_s|x_t) \| q(x_s|x_t,x_0))}_{\text{VLB}} + \underbrace{\mathbb{1}_{t=1} \cdot (-\log p_\theta(x_0|x_1))}_{\text{CE at final step}} \big] + \frac{1}{T}\underbrace{\text{KL}(q(x_T|x_0)\| p(x_T))}_{\text{prior}}
$$

### Continuous-time (lines 475–525)
$$
\mathcal{L} = \mathbb{E}_{t\sim[0,T]}\big[ \beta_t \cdot g_\theta(x_t, t) \big]
$$
where $g_\theta$ is derived from Proposition 4 of the paper, with an auxiliary term involving $q(z_t|x_0)$ ratios for variance reduction.

Both support `simplified_vlb` mode (L2 approximation for faster training).

## MCMC Corrector (lines 553–605)

A Gibbs-like refinement applied during sampling:

```
for n in range(max_steps):
    fprob = denoising_fn(z_n, t)
    z_{n+1} ~ p(z | fprob, t)    # with coef=2 for faster mixing
```

Step size $\delta_n$ is adaptively clipped to keep stay-probability $\ge$ `min_stay_prob`.

## Sampling Loop (lines 607–698)

```
x_T ~ m (noise)
for t in reversed(time_steps):
    fprob = denoising_fn(x_t, t/num_steps)
    prob_s = p_theta(x_s | x_t)   # or fprob when s=0
    x_s ~ Cat(prob_s)
    if MCMC enabled: x_s = mcmc_corrector(x_s, s)
return x_0
```

Key details:
- Time steps are linearly spaced from `T` to `0`.
- For `s=0`, `prob_s` is set directly to the denoising network output (final clean prediction).
- `denoising_fn` receives normalized time `t/num_steps` in discrete mode.
- Conditional masking is supported throughout: masked positions are forced to stay at the input value.

## Overall Data Flow

```
Training:
  x_0 ──t──► qt_0_sample ──► x_t ──► denoising_fn ──► fprob_t ──► compute_loss ──► ∇θ
         random t                                          │
                                               ┌───────────┤
                                               ▼           ▼
                                         VLB term    CE term
                                    (KL or Proposition 4) ( -log fprob_t[x_0] )

Sampling:
  m ──► sample_categorical ──► x_T ──► for t in reverse: ──► x_0
                                    │
                                    ├── denoising_fn(x_t, t)
                                    ├── ps_t_prob → sample_categorical
                                    └── [optional] MCMC corrector
```
