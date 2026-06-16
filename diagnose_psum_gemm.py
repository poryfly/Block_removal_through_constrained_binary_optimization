"""Phase 1: Pure DeepGEMM kernel diagnostics (2 GPU, no model needed).

Tests the SM120 grouped GEMM kernel with `use_psum_layout=True` — the code path
used by DeepSeek-V4's experts forward but NOT covered by the dual-GPU baseline test.

Run:  CUDA_VISIBLE_DEVICES=0,1 python diagnose_psum_gemm.py
"""

import math
import traceback
import torch
import deep_gemm
from deep_gemm.utils.math import per_token_cast_to_fp8


def _set_device(dev):
    """Set CUDA context to the device implied by `dev` (e.g. 'cuda:1')."""
    idx = int(str(dev).split(':')[-1]) if ':' in str(dev) else 0
    torch.cuda.set_device(idx)


def cuda_sync_check(label: str) -> bool:
    """Synchronize all visible GPUs and check for CUDA errors."""
    try:
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device=i)
        print(f"  [{label}] CUDA sync OK")
        return True
    except RuntimeError as e:
        print(f"  [{label}] CUDA ERROR: {e}")
        return False


def run_step(name: str, func):
    """Run a diagnostic step, catch exceptions, check CUDA state."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        func()
        ok = cuda_sync_check(f"{name} post-sync")
        if ok:
            print(f"  >>> {name}: PASS\n")
        else:
            print(f"  >>> {name}: FAIL (CUDA error after step)\n")
    except Exception as e:
        cuda_sync_check(f"{name} error-sync")
        print(f"  >>> {name}: FAIL (exception)")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build grouped GEMM inputs (FP8×FP8 with packed UE8M0 SF)
# ─────────────────────────────────────────────────────────────────────────────

def build_grouped_inputs(num_groups, m_per_group, n, k, device):
    """Build FP8 grouped GEMM inputs with packed UE8M0 scale factors.

    Returns:
        a: (fp8 [padded_m, k], sf_int32 [padded_m, k//512])
        b: (fp8 [G, n, k], sf_int32 [G, n, k//512])
        d: bf16 output [padded_m, n]
        layout_flat:  per-row expert IDs  [padded_m] int32
        layout_psum:  cumsum of aligned   [G]       int32
    """
    alignment = int(deep_gemm.get_mk_alignment_for_contiguous_layout())
    total_m = num_groups * m_per_group
    padded_m = ((total_m + alignment - 1) // alignment) * alignment

    # A: activation — per-token FP8 cast
    a_bf16 = torch.randn(padded_m, k, device=device, dtype=torch.bfloat16)
    a_fp8, a_sf = per_token_cast_to_fp8(
        a_bf16, use_ue8m0=True, gran_k=128, use_packed_ue8m0=True)

    # B: weight — per-group FP8 cast
    b_fp8 = torch.empty(num_groups, n, k, device=device, dtype=torch.float8_e4m3fn)
    b_sf_parts = []
    for g in range(num_groups):
        b_i = torch.randn(n, k, device=device, dtype=torch.bfloat16)
        b_fp8_i, b_sf_i = per_token_cast_to_fp8(
            b_i, use_ue8m0=True, gran_k=128, use_packed_ue8m0=True)
        b_fp8[g] = b_fp8_i
        b_sf_parts.append(b_sf_i)
    b_sf = torch.stack(b_sf_parts, dim=0)  # [G, n, k_packed]

    d = torch.empty(padded_m, n, device=device, dtype=torch.bfloat16)

    # Layout 1 (flat): per-row expert IDs, -1 for padding
    tokens_per_expert = torch.full(
        (num_groups,), m_per_group, dtype=torch.long, device=device)
    aligned_tokens = ((tokens_per_expert + alignment - 1) // alignment) * alignment

    layout_flat = torch.full((padded_m,), -1, device=device, dtype=torch.int32)
    offset = 0
    for g in range(num_groups):
        layout_flat[offset:offset + m_per_group] = g
        offset += m_per_group

    # Layout 2 (psum): cumsum of aligned counts
    layout_psum = aligned_tokens.cumsum(0).int()

    return (a_fp8, a_sf), (b_fp8, b_sf), d, layout_flat, layout_psum, alignment, padded_m


def coerce_sf(sf, expected_mn):
    """Apply transformers' _coerce_sf_for_kernel to make SF kernel-ready."""
    from transformers.integrations.deepgemm import _coerce_sf_for_kernel
    return _coerce_sf_for_kernel(sf, expected_mn=expected_mn)


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

def test_psum_vs_flat_single_gpu():
    """Single-GPU: verify psum and flat layouts produce the same GEMM result."""
    dev = "cuda:0"
    _set_device(dev)
    num_groups, m_per_group, n, k = 8, 16, 2048, 7168

    a, b, d, layout_flat, layout_psum, alignment, padded_m = \
        build_grouped_inputs(num_groups, m_per_group, n, k, dev)

    a_sf_k = coerce_sf(a[1], expected_mn=padded_m)
    b_sf_k = coerce_sf(b[1], expected_mn=n)
    sf_recipe = (1, 1, 128)

    print(f"  shapes: a={list(a[0].shape)}, b={list(b[0].shape)}")
    print(f"  a_sf: {a_sf_k.dtype} {list(a_sf_k.shape)}, stride={list(a_sf_k.stride())}")
    print(f"  b_sf: {b_sf_k.dtype} {list(b_sf_k.shape)}, stride={list(b_sf_k.stride())}")
    print(f"  layout_flat: {list(layout_flat.shape)}, layout_psum: {list(layout_psum.shape)}")
    print(f"  layout_psum values: {layout_psum.tolist()}")

    # Run flat layout
    d_flat = torch.empty(padded_m, n, device=dev, dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (a[0], a_sf_k.clone()), (b[0], b_sf_k.clone()),
        d_flat, layout_flat, recipe=sf_recipe, use_psum_layout=False,
    )

    # Run psum layout
    d_psum = torch.empty(padded_m, n, device=dev, dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (a[0], a_sf_k.clone()), (b[0], b_sf_k.clone()),
        d_psum, layout_psum, recipe=sf_recipe, use_psum_layout=True,
    )

    diff = (d_flat - d_psum).abs().max().item()
    print(f"  [flat]  norm={d_flat.norm().item():.4f}")
    print(f"  [psum]  norm={d_psum.norm().item():.4f}")
    print(f"  max abs diff: {diff:.6f}")
    assert diff < 0.01, f"psum and flat results differ by {diff}"
    print(f"  Results match ✓")


def test_grouped_dual_gpu_comparison():
    """Dual-GPU: compare psum vs flat on BOTH GPUs to isolate device issue."""
    sf_recipe = (1, 1, 128)
    for dev in ("cuda:0", "cuda:1"):
        _set_device(dev)  # MUST set CUDA context before any kernel launch
        num_groups, m_per_group, n, k = 8, 16, 2048, 7168
        a, b, d, layout_flat, layout_psum, alignment, padded_m = \
            build_grouped_inputs(num_groups, m_per_group, n, k, dev)
        a_sf_k = coerce_sf(a[1], expected_mn=padded_m)
        b_sf_k = coerce_sf(b[1], expected_mn=n)

        # flat layout
        d_flat = torch.empty(padded_m, n, device=dev, dtype=torch.bfloat16)
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a[0], a_sf_k.clone()), (b[0], b_sf_k.clone()),
            d_flat, layout_flat, recipe=sf_recipe, use_psum_layout=False,
        )
        # psum layout
        d_psum = torch.empty(padded_m, n, device=dev, dtype=torch.bfloat16)
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a[0], a_sf_k.clone()), (b[0], b_sf_k.clone()),
            d_psum, layout_psum, recipe=sf_recipe, use_psum_layout=True,
        )
        diff = (d_flat - d_psum).abs().max().item()
        flat_norm = d_flat.norm().item()
        psum_norm = d_psum.norm().item()
        status = "✓" if diff < 0.01 and not math.isinf(psum_norm) else "✗"
        print(f"  {dev}: flat_norm={flat_norm:.0f} psum_norm={psum_norm:.0f} diff={diff:.6f} {status}")


def test_psum_dual_gpu():
    """Dual-GPU: run psum grouped GEMM on both GPUs separately."""
    for dev in ("cuda:0", "cuda:1"):
        _set_device(dev)  # MUST set CUDA context before any kernel launch
        num_groups, m_per_group, n, k = 8, 16, 2048, 7168
        a, b, d, _, layout_psum, alignment, padded_m = \
            build_grouped_inputs(num_groups, m_per_group, n, k, dev)

        a_sf_k = coerce_sf(a[1], expected_mn=padded_m)
        b_sf_k = coerce_sf(b[1], expected_mn=n)

        d_out = torch.empty(padded_m, n, device=dev, dtype=torch.bfloat16)
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a[0], a_sf_k.clone()), (b[0], b_sf_k.clone()),
            d_out, layout_psum, recipe=(1, 1, 128), use_psum_layout=True,
        )
        print(f"  {dev}: norm={d_out.norm().item():.4f}  ✓")


def test_psum_varying_shapes():
    """Test psum grouped GEMM with various (num_groups, m_per_group, n, k)."""
    shapes = [
        (8,   4, 2048, 7168),   # small m_per_group (decode-like)
        (8,  16, 2048, 7168),   # medium
        (8,  32, 2048, 7168),   # larger
        (16,  8, 4096, 7168),   # more experts, wider
        (4,  64, 1024, 7168),   # fewer experts, narrow
    ]
    dev = "cuda:0"
    _set_device(dev)
    for ng, mpg, n, k in shapes:
        a, b, d, _, layout_psum, _, padded_m = \
            build_grouped_inputs(ng, mpg, n, k, dev)
        a_sf_k = coerce_sf(a[1], expected_mn=padded_m)
        b_sf_k = coerce_sf(b[1], expected_mn=n)

        d_out = torch.empty(padded_m, n, device=dev, dtype=torch.bfloat16)
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a[0], a_sf_k.clone()), (b[0], b_sf_k.clone()),
            d_out, layout_psum, recipe=(1, 1, 128), use_psum_layout=True,
        )
        print(f"  G={ng:2d} m/g={mpg:3d} n={n:4d} k={k:4d} → norm={d_out.norm().item():.4f}  ✓")


def test_psum_m1_decode():
    """Test psum with m_per_group=1 (single-token decode, worst case)."""
    dev = "cuda:0"
    _set_device(dev)
    num_groups, m_per_group, n, k = 8, 1, 2048, 7168
    a, b, d, _, layout_psum, _, padded_m = \
        build_grouped_inputs(num_groups, m_per_group, n, k, dev)

    a_sf_k = coerce_sf(a[1], expected_mn=padded_m)
    b_sf_k = coerce_sf(b[1], expected_mn=n)

    d_out = torch.empty(padded_m, n, device=dev, dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (a[0], a_sf_k.clone()), (b[0], b_sf_k.clone()),
        d_out, layout_psum, recipe=(1, 1, 128), use_psum_layout=True,
    )
    print(f"  m=1 decode: norm={d_out.norm().item():.4f}  ✓")


def test_psum_repeated_calls():
    """Repeated calls to detect intermittent CUDA errors."""
    dev = "cuda:0"
    _set_device(dev)
    num_groups, m_per_group, n, k = 8, 16, 2048, 7168
    a, b, d, _, layout_psum, _, padded_m = \
        build_grouped_inputs(num_groups, m_per_group, n, k, dev)

    a_sf_k = coerce_sf(a[1], expected_mn=padded_m)
    b_sf_k = coerce_sf(b[1], expected_mn=n)

    for i in range(20):
        d_out = torch.empty(padded_m, n, device=dev, dtype=torch.bfloat16)
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a[0], a_sf_k.clone()), (b[0], b_sf_k.clone()),
            d_out, layout_psum, recipe=(1, 1, 128), use_psum_layout=True,
        )
    # Only sync at end — if any call corrupted CUDA, sync will catch it
    print(f"  20 repeated calls done, norm={d_out.norm().item():.4f}  ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        print(f"  [{i}] {name} (SM {cap[0]}.{cap[1]})")

    alignment = int(deep_gemm.get_mk_alignment_for_contiguous_layout())
    print(f"DeepGEMM m_alignment: {alignment}")

    run_step("1. psum vs flat (single GPU, compare results)", test_psum_vs_flat_single_gpu)
    run_step("2. psum vs flat (both GPUs, compare results)", test_grouped_dual_gpu_comparison)
    run_step("3. psum only on both GPUs", test_psum_dual_gpu)
    run_step("4. psum varying shapes", test_psum_varying_shapes)
    run_step("5. psum m=1 decode", test_psum_m1_decode)
    run_step("6. psum repeated calls (20x)", test_psum_repeated_calls)

    print(f"\n{'='*60}")
    print("  Phase 1 complete.")
    print(f"{'='*60}")
    print("If all pass → run Phase 2 (model forward, 8 GPUs):")
    print("  python diagnose_model_forward.py")
    print("If any fail → issue is in DeepGEMM SM120 psum grouped GEMM")


if __name__ == "__main__":
    main()
