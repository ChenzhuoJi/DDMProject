"""Feature extraction script for PCA / manifold visualization.

Extracts [hidden_dim]-dim mean-pooled representations for:
  - 500 real DNA sequences  → real_latents.npy  [500, hidden_dim]
  - 500 generated sequences → gen_latents.npy   [500, hidden_dim]

Uses the same TransformerBackbone as benchmark_stress.py (d=64, 2 layers).
Hidden states are taken from the final Transformer encoder layer output
(before the classification head), then averaged over the sequence dimension.
"""

from __future__ import annotations

import math
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dna_dataset import DNADataset, DNAVocab

warnings.filterwarnings("ignore", category=UserWarning)

# ── config ────────────────────────────────────────────────────────────────────
DEVICE          = torch.device("cpu")
SEQ_LEN         = 256
VOCAB_SIZE      = 7
MASK_ID         = 1
PAD_ID          = 0
N_SAMPLES       = 500
BATCH_SIZE      = 50          # process in mini-batches to keep RAM sane
DIFFUSION_STEPS = 50
EPS             = 1e-5

D_MODEL  = 64
N_HEADS  = 2
N_LAYERS = 2
FFN_DIM  = 128

SEED = 42


# ── Transformer backbone (hidden-state-aware) ─────────────────────────────────

class TransformerBackbone(nn.Module):
    """Shared backbone; `forward_hidden` returns pre-head representations."""

    def __init__(self, vocab_size: int, d_model: int,
                 n_heads: int, n_layers: int, ffn_dim: int,
                 max_len: int = 512):
        super().__init__()
        self.embed      = nn.Embedding(vocab_size, d_model)
        self.pos        = nn.Embedding(max_len, d_model)
        self.sigma_proj = nn.Linear(1, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_dim, dropout=0.0,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    # ── full forward (logits) ─────────────────────────────────────────────────
    def forward(self, x: torch.Tensor,
                sigma: torch.Tensor | None = None,
                causal: bool = False) -> torch.Tensor:
        h = self._embed(x, sigma, causal)
        return self.head(h)

    # ── hidden states before head ─────────────────────────────────────────────
    @torch.no_grad()
    def forward_hidden(self, x: torch.Tensor,
                       sigma: torch.Tensor | None = None) -> torch.Tensor:
        """Returns (B, L, d_model) — last-layer hidden states."""
        return self._embed(x, sigma, causal=False)

    def _embed(self, x: torch.Tensor,
               sigma: torch.Tensor | None,
               causal: bool) -> torch.Tensor:
        B, L = x.shape
        pos  = torch.arange(L, device=x.device)
        h    = self.embed(x) + self.pos(pos).unsqueeze(0)

        if sigma is not None:
            sf = sigma.view(B, 1, 1).expand(B, L, 1).float()
            h  = h + self.sigma_proj(sf)

        mask = None
        if causal:
            mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)

        return self.transformer(h, mask=mask,
                                is_causal=causal if causal else False)


# ── mean pooling helper ────────────────────────────────────────────────────────

def mean_pool(hidden: torch.Tensor,
              input_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Average hidden states over the sequence dimension.

    If input_ids is given, PAD positions (id==0) are masked out so they don't
    dilute the representation.  Falls back to plain mean otherwise.

    Args:
        hidden:    (B, L, D)
        input_ids: (B, L) optional — used to build the non-PAD mask
    Returns:
        (B, D)
    """
    if input_ids is not None:
        mask = (input_ids != PAD_ID).float().unsqueeze(-1)   # (B, L, 1)
        summed = (hidden * mask).sum(dim=1)                  # (B, D)
        counts = mask.sum(dim=1).clamp(min=1)                # (B, 1)
        return summed / counts
    return hidden.mean(dim=1)


# ── synthetic DNA generator (stand-in for a real corpus) ──────────────────────

_BASES = list("ACGT")

def _rand_dna(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(_BASES) for _ in range(length))


def build_real_sequences(n: int, seq_len: int, seed: int) -> list[str]:
    """Return n random DNA strings of length seq_len.

    In a real experiment replace this with sequences loaded from a FASTA file
    or a genomics database.  The pipeline (DNADataset → mean-pool) stays
    identical regardless of the source.
    """
    rng = random.Random(seed)
    return [_rand_dna(seq_len, rng) for _ in range(n)]


# ── diffusion timestep schedule ────────────────────────────────────────────────

def build_timesteps_quadratic(n_steps: int, eps: float = EPS):
    i     = torch.arange(n_steps + 1, dtype=torch.float64)
    u     = i / n_steps
    t_all = (1 - eps) * (1 - u ** 2) + eps
    return t_all[:-1].float(), (t_all[:-1] - t_all[1:]).float()


# ── 50-step strided diffusion sampler ─────────────────────────────────────────

def _sample_categorical(probs: torch.Tensor) -> torch.Tensor:
    gumbel = 1e-10 - (torch.rand_like(probs) + 1e-10).log()
    return (probs / gumbel).argmax(dim=-1)


@torch.no_grad()
def diffusion_generate(model: TransformerBackbone,
                       batch_size: int,
                       seq_len: int,
                       n_steps: int = DIFFUSION_STEPS) -> torch.Tensor:
    """Return generated token ids (B, seq_len)."""
    timesteps, dts = build_timesteps_quadratic(n_steps)
    x = torch.full((batch_size, seq_len), MASK_ID, dtype=torch.long, device=DEVICE)

    for i in range(n_steps):
        t  = timesteps[i].item()
        dt = dts[i].item()

        sigma_t = -math.log(max(1 - (1 - EPS) * t,           1e-8))
        sigma_s = -math.log(max(1 - (1 - EPS) * max(t - dt, EPS), 1e-8))

        mc_t = 1 - math.exp(-sigma_t)
        mc_s = 1 - math.exp(-sigma_s)

        st     = torch.full((batch_size, 1), sigma_t, dtype=torch.float32, device=DEVICE)
        logits = model(x, sigma=st, causal=False)
        lp     = F.log_softmax(logits, dim=-1)
        lp[:, :, MASK_ID] = -1e9

        q = lp.exp() * (mc_t - mc_s)
        q[:, :, MASK_ID] = mc_s

        _x   = _sample_categorical(q)
        copy = (x != MASK_ID)
        x    = torch.where(copy, x, _x)

    # final denoising pass — resolve any residual masks
    still = (x == MASK_ID)
    if still.any():
        logits = model(x, sigma=torch.zeros(batch_size, 1, device=DEVICE))
        x[still] = logits.argmax(-1)[still]

    return x


# ── extraction loops ───────────────────────────────────────────────────────────

def extract_real_latents(model: TransformerBackbone,
                         sequences: list[str],
                         vocab: DNAVocab) -> np.ndarray:
    """Feed real sequences through encoder, return mean-pooled hidden states."""
    dataset = DNADataset(sequences, vocab=vocab, max_seq_len=SEQ_LEN)
    all_vecs: list[np.ndarray] = []

    for start in range(0, len(sequences), BATCH_SIZE):
        end   = min(start + BATCH_SIZE, len(sequences))
        items = [dataset[i] for i in range(start, end)]
        ids   = torch.stack([it["input_ids"] for it in items]).to(DEVICE)   # (B, L)

        hidden = model.forward_hidden(ids)          # (B, L, D)
        vecs   = mean_pool(hidden, ids)             # (B, D)
        all_vecs.append(vecs.cpu().numpy())

        print(f"  real  [{end:>4}/{len(sequences)}]  batch shape {vecs.shape}")

    return np.vstack(all_vecs)                      # (N, D)


def extract_gen_latents(model: TransformerBackbone) -> np.ndarray:
    """Sample sequences from diffusion, return their mean-pooled hidden states."""
    all_vecs: list[np.ndarray] = []
    generated = 0

    while generated < N_SAMPLES:
        bs = min(BATCH_SIZE, N_SAMPLES - generated)

        x      = diffusion_generate(model, bs, SEQ_LEN)   # (B, L) token ids
        hidden = model.forward_hidden(x)                  # (B, L, D)  sigma=0 / clean
        vecs   = mean_pool(hidden, x)                     # (B, D)
        all_vecs.append(vecs.cpu().numpy())
        generated += bs

        print(f"  gen   [{generated:>4}/{N_SAMPLES}]  batch shape {vecs.shape}")

    return np.vstack(all_vecs)                            # (N, D)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("Feature Extraction for PCA / Manifold Visualisation")
    print(f"  device={DEVICE}  seq_len={SEQ_LEN}  n_samples={N_SAMPLES}")
    print(f"  model:  d={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}, ffn={FFN_DIM}")
    print(f"  diffusion steps: {DIFFUSION_STEPS}  (quadratic schedule)")
    print("=" * 60)

    # ── build model ───────────────────────────────────────────────
    model = TransformerBackbone(
        VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM
    ).to(DEVICE).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nBackbone parameters: {n_params / 1e6:.3f} M\n")

    vocab = DNAVocab()

    # ── real sequences ─────────────────────────────────────────────
    print("[1/2] Extracting real-sequence latents …")
    real_seqs = build_real_sequences(N_SAMPLES, SEQ_LEN, seed=SEED)
    real_lat  = extract_real_latents(model, real_seqs, vocab)
    np.save("real_latents.npy", real_lat)
    print(f"  → saved real_latents.npy  shape={real_lat.shape}  "
          f"dtype={real_lat.dtype}\n")

    # ── generated sequences ────────────────────────────────────────
    print("[2/2] Generating sequences & extracting latents …")
    gen_lat = extract_gen_latents(model)
    np.save("gen_latents.npy", gen_lat)
    print(f"  → saved gen_latents.npy   shape={gen_lat.shape}  "
          f"dtype={gen_lat.dtype}\n")

    # ── sanity checks ──────────────────────────────────────────────
    assert real_lat.shape == (N_SAMPLES, D_MODEL), \
        f"Unexpected real shape: {real_lat.shape}"
    assert gen_lat.shape  == (N_SAMPLES, D_MODEL), \
        f"Unexpected gen shape:  {gen_lat.shape}"

    print("=" * 60)
    print("Summary")
    print(f"  real_latents.npy : {real_lat.shape}  "
          f"mean={real_lat.mean():.4f}  std={real_lat.std():.4f}")
    print(f"  gen_latents.npy  : {gen_lat.shape}   "
          f"mean={gen_lat.mean():.4f}  std={gen_lat.std():.4f}")
    print("  Both files written.  Ready for PCA / UMAP visualisation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
