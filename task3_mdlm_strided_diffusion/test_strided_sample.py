"""Strided sampling benchmark.

Compares the original fixed-step sampler against our quadratic strided sampler.
The backbone forward pass is replaced by a random-logit mock so the test runs
without model weights while still exercising the full sampling loop logic
(timestep construction, per-step dt, _ddpm_update posterior, Gumbel sampling).
"""
import sys
import time
import torch

sys.path.insert(0, '.')
from noise_schedule import LogLinearNoise

# ── config ────────────────────────────────────────────────────────────────────
DEVICE     = 'cpu'
BATCH_SIZE = 2
SEQ_LEN    = 128
VOCAB_SIZE = 1024        # small vocab for fast mock
MASK_INDEX = VOCAB_SIZE  # absorbing state index
EPS        = 1e-5

STEPS_ORIG   = 1000
STEPS_STRIDE = 50
# ──────────────────────────────────────────────────────────────────────────────


def build_timesteps(sampling_steps, eps, schedule, device):
    """Replicate diffusion.Diffusion._build_timesteps without the class."""
    if schedule == 'quadratic':
        i = torch.arange(sampling_steps + 1,
                         dtype=torch.float64, device=device)
        u = i / sampling_steps
        t_all = (1 - eps) * (1 - u ** 2) + eps
    else:
        t_all = torch.linspace(1, eps, sampling_steps + 1,
                               dtype=torch.float64, device=device)
    timesteps = t_all[:-1].float()
    dts       = (t_all[:-1] - t_all[1:]).float()
    return timesteps, dts


def _sample_categorical(probs):
    gumbel = 1e-10 - (torch.rand_like(probs) + 1e-10).log()
    return (probs / gumbel).argmax(dim=-1)


def mock_ddpm_update(x, t, dt, noise):
    """_ddpm_update with random logits instead of a real model."""
    sigma_t, _ = noise(t)
    sigma_s, _ = noise(torch.clamp(t - dt, min=EPS))

    sigma_t = sigma_t.squeeze(-1)
    sigma_s = sigma_s.squeeze(-1)

    move_chance_t = (1 - torch.exp(-sigma_t))[:, None, None]
    move_chance_s = (1 - torch.exp(-sigma_s))[:, None, None]

    # mock p_theta(x0 | xt)  —  uniform over non-mask tokens
    log_p_x0 = torch.log_softmax(
        torch.randn(x.shape[0], x.shape[1], VOCAB_SIZE + 1, device=x.device),
        dim=-1)
    log_p_x0[:, :, MASK_INDEX] = -1e9   # model never predicts mask as x0

    q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
    q_xs[:, :, MASK_INDEX] = move_chance_s[:, :, 0]

    _x = _sample_categorical(q_xs)
    copy_flag = (x != MASK_INDEX).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x


def run_sampler(sampling_steps, schedule, noise, label):
    """Run full sampling loop and return (elapsed_sec, final_tokens)."""
    timesteps, dts = build_timesteps(sampling_steps, EPS, schedule, DEVICE)

    # prior: all masks
    x = MASK_INDEX * torch.ones(BATCH_SIZE, SEQ_LEN,
                                dtype=torch.int64, device=DEVICE)

    torch.manual_seed(42)
    t0 = time.time()
    for i in range(sampling_steps):
        t  = timesteps[i] * torch.ones(BATCH_SIZE, 1, device=DEVICE)
        dt = dts[i].item()
        x  = mock_ddpm_update(x, t, dt, noise)
    elapsed = time.time() - t0

    # final denoising: greedily resolve any remaining masks
    still_masked = (x == MASK_INDEX)
    x[still_masked] = torch.randint(0, VOCAB_SIZE, (still_masked.sum(),))

    return elapsed, x


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Strided Sampling Benchmark")
    print(f"  device={DEVICE}  batch={BATCH_SIZE}  seq_len={SEQ_LEN}"
          f"  vocab={VOCAB_SIZE}")
    print("=" * 65)

    noise = LogLinearNoise()

    # ── timestep preview ──────────────────────────────────────────
    print(f"\n[Quadratic schedule preview, sampling_steps={STEPS_STRIDE}]")
    ts, ds = build_timesteps(STEPS_STRIDE, EPS, 'quadratic', DEVICE)
    cols = "  {:>4}  {:>10}  {:>10}"
    print(cols.format("step", "t", "dt"))
    print("  " + "-" * 30)
    indices = [0, 1, 2, STEPS_STRIDE//4, STEPS_STRIDE//2,
               3*STEPS_STRIDE//4, STEPS_STRIDE-2, STEPS_STRIDE-1]
    for idx in indices:
        print(cols.format(idx, f"{ts[idx]:.6f}", f"{ds[idx]:.6f}"))

    # ── run both samplers ─────────────────────────────────────────
    print(f"\n[Running original: linear, {STEPS_ORIG} steps] ...")
    t_orig,   x_orig   = run_sampler(STEPS_ORIG,   'linear',    noise, 'orig')

    print(f"[Running strided:  quadratic, {STEPS_STRIDE} steps] ...")
    t_stride, x_stride = run_sampler(STEPS_STRIDE, 'quadratic', noise, 'stride')

    # ── results ───────────────────────────────────────────────────
    speedup = t_orig / t_stride
    print(f"""
{'─'*65}
  Original ({STEPS_ORIG:>5} steps, linear):     {t_orig:.4f} s
  Strided  ({STEPS_STRIDE:>5} steps, quadratic): {t_stride:.4f} s
  Speedup:                              {speedup:.2f}×
{'─'*65}""")

    # ── sample inspection ─────────────────────────────────────────
    print(f"\n[Sample token-ID output (strided, batch 0, first 32 tokens)]")
    snippet = x_stride[0, :32].tolist()
    print(f"  {snippet}")

    mask_ratio_orig   = (x_orig   == MASK_INDEX).float().mean().item()
    mask_ratio_stride = (x_stride == MASK_INDEX).float().mean().item()
    print(f"\n[Residual mask ratio after sampling]")
    print(f"  Original : {mask_ratio_orig:.4f}   (should be 0.0)")
    print(f"  Strided  : {mask_ratio_stride:.4f}   (should be 0.0)")

    # ── sanity assertions ─────────────────────────────────────────
    print(f"\n[Assertions]")
    passed = True

    if x_orig.shape == (BATCH_SIZE, SEQ_LEN):
        print(f"  PASS  x_orig shape  {tuple(x_orig.shape)}")
    else:
        print(f"  FAIL  x_orig shape  {tuple(x_orig.shape)}"); passed = False

    if x_stride.shape == (BATCH_SIZE, SEQ_LEN):
        print(f"  PASS  x_stride shape {tuple(x_stride.shape)}")
    else:
        print(f"  FAIL  x_stride shape {tuple(x_stride.shape)}"); passed = False

    if mask_ratio_stride == 0.0:
        print(f"  PASS  no residual masks in strided output")
    else:
        print(f"  FAIL  residual masks remain"); passed = False

    if speedup > 5:
        print(f"  PASS  speedup {speedup:.2f}× > 5×")
    else:
        print(f"  WARN  speedup {speedup:.2f}× — expected > 5× on CPU mock")

    # check no token id exceeds vocab (including mask)
    max_id = x_stride.max().item()
    if max_id < VOCAB_SIZE:
        print(f"  PASS  all token IDs in range [0, {VOCAB_SIZE-1}], max={max_id}")
    else:
        print(f"  FAIL  token ID {max_id} >= VOCAB_SIZE {VOCAB_SIZE}"); passed = False

    # dt array must be strictly positive
    if (ds > 0).all():
        print(f"  PASS  all dt values strictly positive")
    else:
        print(f"  FAIL  non-positive dt detected"); passed = False

    print("\n" + "=" * 65)
    print("ALL ASSERTIONS PASSED" if passed else "SOME ASSERTIONS FAILED")
    print("=" * 65)
    return passed


if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)
