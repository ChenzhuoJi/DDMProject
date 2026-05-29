"""Sanity check for CosineMaskNoise schedule.

Tests:
  - move_chance ≈ 0 at t=0.0
  - move_chance ≈ 1 at t=1.0
  - monotonic increase from t=0 to t=1
  - tensor shapes are preserved under typical diffusion.py usage
"""
import sys
import torch
sys.path.insert(0, '.')
from noise_schedule import CosineMaskNoise

def main():
    print("=" * 60)
    print("CosineMaskNoise Sanity Check")
    print("=" * 60)

    noise = CosineMaskNoise(s=0.008)

    # ── 1. Simulated data (code-gen task) ────────────────────────
    batch_size, seq_len = 2, 128
    x0 = torch.randint(0, 50257, (batch_size, seq_len))  # GPT-2 vocab size
    print(f"\n[Data]  x0 shape: {x0.shape}, dtype: {x0.dtype}")

    # ── 2. Boundary & trend test ─────────────────────────────────
    t_vals = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    print(f"\n{'t':>6}  {'sigma':>12}  {'move_chance (mask ratio)':>24}")
    print("-" * 48)

    move_chances = []
    for t_scalar in t_vals:
        t = t_scalar.unsqueeze(0)          # shape (1,) — matches diffusion.py usage
        sigma, dsigma = noise(t)           # calls forward() → total_noise, rate_noise
        move_chance = 1 - torch.exp(-sigma)
        move_chances.append(move_chance.item())
        print(f"{t_scalar.item():>6.2f}  {sigma.item():>12.6f}  {move_chance.item():>24.6f}")

    # ── 3. Shape test under diffusion.py broadcast pattern ───────
    print("\n[Shape test]  t shape: (B,), sigma broadcast to (B, 1, 1)")
    t_batch = torch.rand(batch_size)
    sigma_batch, dsigma_batch = noise(t_batch)           # shape (B,)
    move_chance_batch = (1 - torch.exp(-sigma_batch)).unsqueeze(-1)  # (B, 1)
    # q_xt: move_indices = rand(*x.shape) < move_chance  →  (B, L)
    move_indices = torch.rand(*x0.shape) < move_chance_batch
    mask_index = 50257
    xt = torch.where(move_indices, torch.tensor(mask_index), x0)
    print(f"  sigma_batch shape : {sigma_batch.shape}")
    print(f"  move_chance shape : {move_chance_batch.shape}")
    print(f"  xt shape          : {xt.shape}")
    actual_mask_ratio = (xt == mask_index).float().mean().item()
    expected_mask_ratio = move_chance_batch.mean().item()
    print(f"  expected mask ratio (mean over batch) : {expected_mask_ratio:.4f}")
    print(f"  actual   mask ratio (measured on xt)  : {actual_mask_ratio:.4f}")

    # ── 4. Pass / Fail assertions ─────────────────────────────────
    print("\n[Assertions]")
    passed = True

    if move_chances[0] < 0.01:
        print(f"  PASS  t=0.0 → move_chance={move_chances[0]:.6f} (< 0.01)")
    else:
        print(f"  FAIL  t=0.0 → move_chance={move_chances[0]:.6f} (expected < 0.01)")
        passed = False

    if move_chances[-1] > 0.99:
        print(f"  PASS  t=1.0 → move_chance={move_chances[-1]:.6f} (> 0.99)")
    else:
        print(f"  FAIL  t=1.0 → move_chance={move_chances[-1]:.6f} (expected > 0.99)")
        passed = False

    monotone = all(move_chances[i] < move_chances[i+1]
                   for i in range(len(move_chances)-1))
    if monotone:
        print(f"  PASS  monotonically increasing across t ∈ [0, 1]")
    else:
        print(f"  FAIL  not monotonically increasing: {move_chances}")
        passed = False

    if xt.shape == x0.shape:
        print(f"  PASS  xt shape matches x0 shape {xt.shape}")
    else:
        print(f"  FAIL  shape mismatch: xt={xt.shape}, x0={x0.shape}")
        passed = False

    print("\n" + "=" * 60)
    if passed:
        print("ALL ASSERTIONS PASSED — ready to commit.")
    else:
        print("SOME ASSERTIONS FAILED — do NOT commit.")
    print("=" * 60)
    return passed

if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)
