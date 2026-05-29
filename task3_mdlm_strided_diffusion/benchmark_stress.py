"""Single-machine throughput & memory stress benchmark.

Compares:
  Group A — Autoregressive (AR) decoder: generates token-by-token,
            left-to-right, seq_len forward passes per sample.
  Group B — Strided Diffusion (50-step quadratic): generates all
            positions in parallel, 50 forward passes per sample.

Hardware note: CPU-only run (no CUDA).  Memory is tracked via
psutil RSS (Resident Set Size) instead of torch.cuda.max_memory_allocated.
OOM is caught as MemoryError / RuntimeError(memory).

Results are saved to stress_results.json.
"""

from __future__ import annotations
import gc
import json
import math
import time
import traceback
from typing import List, Dict, Any

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── constants ──────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cpu')
SEQ_LEN     = 256
VOCAB_SIZE  = 7          # DNA: PAD=0 MASK=1 UNK=2 A=3 C=4 G=5 T=6
MASK_ID     = 1
PAD_ID      = 0
EPS         = 1e-5

# model architecture — kept tiny for CPU feasibility
D_MODEL     = 64
N_HEADS     = 2
N_LAYERS    = 2
FFN_DIM     = 128

BATCH_SIZES      = [1, 4, 16, 64, 128, 256, 512]
DIFFUSION_STEPS  = 50
WARMUP_RUNS      = 0        # no warm-up on CPU (saves time)
TIMED_RUNS       = 1        # single timed run on CPU
SOFT_TIMEOUT_S   = 90.0     # per-test wall-clock cap; marks as TIMEOUT if exceeded


# ── shared Transformer backbone ───────────────────────────────────────────────

class TransformerBackbone(nn.Module):
    """Lightweight Transformer used by both AR and Diffusion decoders.

    For AR  : called with a causal mask so each position only attends left.
    For Diff: called without mask (bidirectional, all positions see all).
    """

    def __init__(self, vocab_size: int, d_model: int,
                 n_heads: int, n_layers: int, ffn_dim: int,
                 max_len: int = 512):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.pos    = nn.Embedding(max_len, d_model)
        # sigma embedding for diffusion time-conditioning
        self.sigma_proj = nn.Linear(1, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_dim, dropout=0.0,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor,
                sigma: torch.Tensor | None = None,
                causal: bool = False) -> torch.Tensor:
        B, L = x.shape
        pos  = torch.arange(L, device=x.device)
        h    = self.embed(x) + self.pos(pos).unsqueeze(0)

        if sigma is not None:
            # inject diffusion time conditioning via additive bias
            sigma_f = sigma.view(B, 1, 1).expand(B, L, 1).float()
            h = h + self.sigma_proj(sigma_f)

        mask = None
        if causal:
            mask = nn.Transformer.generate_square_subsequent_mask(
                L, device=x.device)

        h = self.transformer(h, mask=mask,
                             is_causal=causal if causal else False)
        return self.head(h)   # (B, L, vocab_size)


# ── noise schedule (reuse from project) ───────────────────────────────────────

def build_timesteps_quadratic(n_steps: int, eps: float = EPS):
    i     = torch.arange(n_steps + 1, dtype=torch.float64)
    u     = i / n_steps
    t_all = (1 - eps) * (1 - u ** 2) + eps
    return t_all[:-1].float(), (t_all[:-1] - t_all[1:]).float()


# ── Group A: AR sampler ───────────────────────────────────────────────────────

@torch.no_grad()
def ar_generate(model: TransformerBackbone,
                batch_size: int,
                seq_len: int) -> torch.Tensor:
    """Left-to-right greedy autoregressive decode.

    Runs seq_len sequential forward passes; each pass processes an
    ever-growing prefix.  This is the canonical AR bottleneck.
    """
    x = torch.full((batch_size, 1), PAD_ID,
                   dtype=torch.long, device=DEVICE)

    for _ in range(seq_len - 1):
        logits  = model(x, causal=True)        # (B, current_len, V)
        next_id = logits[:, -1, :].argmax(-1)  # greedy pick last position
        x = torch.cat([x, next_id.unsqueeze(-1)], dim=1)

    return x   # (B, seq_len)


# ── Group B: Strided Diffusion sampler ────────────────────────────────────────

def _sample_categorical(probs: torch.Tensor) -> torch.Tensor:
    gumbel = 1e-10 - (torch.rand_like(probs) + 1e-10).log()
    return (probs / gumbel).argmax(dim=-1)


@torch.no_grad()
def diffusion_generate(model: TransformerBackbone,
                       batch_size: int,
                       seq_len: int,
                       n_steps: int = DIFFUSION_STEPS) -> torch.Tensor:
    """50-step quadratic strided diffusion with DDPM-style posterior.

    Runs n_steps parallel forward passes over the full sequence.
    Uses loglinear-style sigma: sigma(t) = -log(1 - (1-eps)*t).
    """
    timesteps, dts = build_timesteps_quadratic(n_steps)
    x = torch.full((batch_size, seq_len), MASK_ID,
                   dtype=torch.long, device=DEVICE)

    for i in range(n_steps):
        t  = timesteps[i].item()
        dt = dts[i].item()

        sigma_t = -math.log(max(1 - (1 - EPS) * t,          1e-8))
        sigma_s = -math.log(max(1 - (1 - EPS) * max(t - dt, EPS), 1e-8))

        move_chance_t = 1 - math.exp(-sigma_t)
        move_chance_s = 1 - math.exp(-sigma_s)

        sigma_tensor = torch.full((batch_size, 1), sigma_t,
                                  dtype=torch.float32, device=DEVICE)
        logits  = model(x, sigma=sigma_tensor, causal=False)  # (B, L, V)
        log_p_x0 = F.log_softmax(logits, dim=-1)
        log_p_x0[:, :, MASK_ID] = -1e9         # model never outputs MASK as x0

        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
        q_xs[:, :, MASK_ID] = move_chance_s    # stay-masked probability

        _x = _sample_categorical(q_xs)
        copy = (x != MASK_ID)
        x = torch.where(copy, x, _x)

    # final denoising: resolve residual masks
    still = (x == MASK_ID)
    if still.any():
        logits = model(x, sigma=torch.zeros(batch_size, 1), causal=False)
        x[still] = logits.argmax(-1)[still]

    return x   # (B, seq_len)


# ── memory helper ─────────────────────────────────────────────────────────────

_proc = psutil.Process()

def rss_mb() -> float:
    return _proc.memory_info().rss / 1024 ** 2

def is_oom_error(e: Exception) -> bool:
    msg = str(e).lower()
    return isinstance(e, MemoryError) or (
        isinstance(e, RuntimeError) and
        any(k in msg for k in ('memory', 'malloc', 'alloc', 'oom')))


# ── benchmark driver ──────────────────────────────────────────────────────────

def run_one(generator_fn, model, batch_size: int,
            label: str) -> Dict[str, Any]:
    """Time generator_fn for TIMED_RUNS runs with a soft wall-clock cap."""
    record: Dict[str, Any] = {
        'model': label,
        'batch_size': batch_size,
        'seq_len': SEQ_LEN,
        'status': 'ok',
        'time_s': None,
        'tokens_per_sec': None,
        'memory_mb': None,
    }
    try:
        for _ in range(WARMUP_RUNS):
            generator_fn(model, batch_size, SEQ_LEN)
        gc.collect()

        mem_before = rss_mb()
        t0 = time.perf_counter()
        for _ in range(TIMED_RUNS):
            out = generator_fn(model, batch_size, SEQ_LEN)
            if time.perf_counter() - t0 > SOFT_TIMEOUT_S:
                record['status'] = 'TIMEOUT'
                print(f"    TIMEOUT (>{SOFT_TIMEOUT_S}s) at batch_size={batch_size}")
                return record
        elapsed = (time.perf_counter() - t0) / TIMED_RUNS
        mem_peak = rss_mb() - mem_before

        tokens = batch_size * SEQ_LEN
        record['time_s']        = round(elapsed, 4)
        record['tokens_per_sec'] = round(tokens / elapsed, 1)
        record['memory_mb']     = round(max(mem_peak, 0), 1)

    except Exception as e:
        if is_oom_error(e):
            record['status'] = 'OOM'
            print(f"    OOM at batch_size={batch_size}")
        else:
            record['status'] = f'ERROR: {type(e).__name__}: {e}'
            print(f"    ERROR: {e}")
            traceback.print_exc()

    return record


def main():
    print("=" * 70)
    print("Single-Machine Throughput & Memory Stress Benchmark")
    print(f"  device={DEVICE}  seq_len={SEQ_LEN}  vocab={VOCAB_SIZE}")
    print(f"  model: d={D_MODEL} heads={N_HEADS} layers={N_LAYERS} ffn={FFN_DIM}")
    n_params = sum(p.numel()
                   for p in TransformerBackbone(
                       VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM
                   ).parameters())
    print(f"  backbone params: {n_params/1e6:.2f} M")
    print(f"  AR steps per sample: {SEQ_LEN}")
    print(f"  Diffusion steps per sample: {DIFFUSION_STEPS} (quadratic)")
    print(f"  batch sizes: {BATCH_SIZES}")
    print("=" * 70)

    # one shared backbone for both models (same weights = fair comparison)
    torch.manual_seed(0)
    model = TransformerBackbone(
        VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM
    ).to(DEVICE).eval()

    results: List[Dict[str, Any]] = []
    ar_oom   = False
    diff_oom = False

    # header
    hdr = f"{'model':>12}  {'bs':>5}  {'time(s)':>9}  {'tok/s':>10}  {'ΔRSS MB':>9}  status"
    sep = "-" * len(hdr)
    print(f"\n{hdr}\n{sep}")

    for bs in BATCH_SIZES:
        for label, fn, skip in [
            ('AR',       ar_generate,         ar_oom),
            ('Diffusion', diffusion_generate,  diff_oom),
        ]:
            if skip:
                rec = {'model': label, 'batch_size': bs,
                       'status': 'skipped (post-OOM)'}
                results.append(rec)
                print(f"{label:>12}  {bs:>5}  {'—':>9}  {'—':>10}  {'—':>9}  skipped")
                continue

            rec = run_one(fn, model, bs, label)
            results.append(rec)

            if rec['status'] in ('OOM', 'TIMEOUT'):
                if label == 'AR':
                    ar_oom = True
                else:
                    diff_oom = True
                print(f"{label:>12}  {bs:>5}  {rec['status']:>9}  {'—':>10}  {'—':>9}  {rec['status']}")
            elif rec['status'] == 'ok':
                print(f"{label:>12}  {bs:>5}"
                      f"  {rec['time_s']:>9.4f}"
                      f"  {rec['tokens_per_sec']:>10.1f}"
                      f"  {rec['memory_mb']:>9.1f}"
                      f"  ok")
            else:
                print(f"{label:>12}  {bs:>5}  ERROR  {rec['status']}")

    # ── summary ───────────────────────────────────────────────────
    print(f"\n{sep}")
    print("Speedup summary (Diffusion tok/s ÷ AR tok/s):")
    ar_res   = {r['batch_size']: r for r in results if r['model'] == 'AR'}
    diff_res = {r['batch_size']: r for r in results if r['model'] == 'Diffusion'}

    speedups = []
    for bs in BATCH_SIZES:
        a, d = ar_res.get(bs, {}), diff_res.get(bs, {})
        if a.get('status') == 'ok' and d.get('status') == 'ok':
            sp = d['tokens_per_sec'] / a['tokens_per_sec']
            speedups.append(sp)
            print(f"  bs={bs:>4}: AR={a['tokens_per_sec']:>9.1f} tok/s"
                  f"  Diff={d['tokens_per_sec']:>9.1f} tok/s"
                  f"  speedup={sp:.2f}×")
        else:
            print(f"  bs={bs:>4}: incomplete data, skipping")

    if speedups:
        print(f"\n  Average speedup: {sum(speedups)/len(speedups):.2f}×"
              f"   Max speedup: {max(speedups):.2f}×")

    # ── OOM limits ────────────────────────────────────────────────
    def last_ok(label):
        ok_batches = [r['batch_size'] for r in results
                      if r['model'] == label and r.get('status') == 'ok']
        return max(ok_batches) if ok_batches else 0

    print(f"\n  AR max surviving batch_size      : {last_ok('AR')}")
    print(f"  Diffusion max surviving batch_size: {last_ok('Diffusion')}")

    # ── save JSON ─────────────────────────────────────────────────
    out_path = 'stress_results.json'
    meta = {
        'device': str(DEVICE),
        'seq_len': SEQ_LEN,
        'vocab_size': VOCAB_SIZE,
        'd_model': D_MODEL,
        'n_heads': N_HEADS,
        'n_layers': N_LAYERS,
        'ffn_dim': FFN_DIM,
        'diffusion_steps': DIFFUSION_STEPS,
        'warmup_runs': WARMUP_RUNS,
        'timed_runs': TIMED_RUNS,
    }
    with open(out_path, 'w') as f:
        json.dump({'meta': meta, 'results': results}, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
