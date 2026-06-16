#!/usr/bin/env python3
"""Dual-GPU isolation test for SM120 DeepGEMM kernels.

Diagnoses whether `illegal memory access` in multi-GPU model forward
is caused by:
  (a) SM120 GEMM kernel itself  →  would fail in Step 1/2
  (b) Cross-device kernel reuse →  would fail in Step 3/4
  (c) JIT compiler global state  →  would fail in Step 4/5

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python test_dual_gpu_sm120.py
"""
import sys
import traceback
import torch
import deep_gemm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fp8_gemm_inputs(m: int, n: int, k: int, device: str = "cuda:0"):
    """Create FP8×FP8 GEMM inputs on `device` (K-major A/B, like model weights).

    A (activation): per-token cast  → SF shape [m, k//128]
    B (weight):     per-block cast  → SF shape [n//128, k//128]
    This matches recipe (1, 128, 128) used by get_default_recipe on SM120.

    Returns:
        a: tuple(tensor_fp8 [m,k], scale_fp32 [m, k//128])
        b: tuple(tensor_fp8 [n,k], scale_fp32 [n//128, k//128])
        d: output tensor [m,n] bf16
    """
    a_bf16 = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    a_fp8, a_sf = deep_gemm.per_token_cast_to_fp8(a_bf16, use_ue8m0=True)
    b_fp8, b_sf = deep_gemm.per_block_cast_to_fp8(b_bf16, use_ue8m0=True)
    d = torch.empty(m, n, device=device, dtype=torch.bfloat16)
    return (a_fp8, a_sf), (b_fp8, b_sf), d


def make_grouped_fp8_inputs(num_groups: int, m_per_group: int, n: int, k: int,
                            device: str = "cuda:0"):
    """Create grouped FP8 GEMM inputs (simulating MoE expert dispatch).

    A: per-token cast  → SF [total_m, k//128]
    B: per-block cast  → SF [num_groups, n//128, k//128]
    grouped_layout: [total_m] int32 tensor of group IDs

    Returns:
        a: (tensor [total_m, k] fp8, scale [total_m, k//128] fp32)
        b: (tensor [num_groups, n, k] fp8, scale [num_groups, n//128, k//128] fp32)
        d: output [total_m, n] bf16
        grouped_layout: [total_m] int32
    """
    from deep_gemm.utils import get_mk_alignment_for_contiguous_layout
    from deep_gemm.utils.math import per_block_cast_to_fp8
    alignment = get_mk_alignment_for_contiguous_layout()
    total_m = num_groups * m_per_group
    padded_m = ((total_m + alignment - 1) // alignment) * alignment

    a_bf16 = torch.randn(padded_m, k, device=device, dtype=torch.bfloat16)
    a_fp8, a_sf = deep_gemm.per_token_cast_to_fp8(a_bf16, use_ue8m0=True)

    b_fp8 = torch.empty(num_groups, n, k, device=device, dtype=torch.float8_e4m3fn)
    b_sf_list = []
    for i in range(num_groups):
        b_i = torch.randn(n, k, device=device, dtype=torch.bfloat16)
        b_fp8[i], b_sf_i = per_block_cast_to_fp8(b_i, use_ue8m0=True)
        b_sf_list.append(b_sf_i)
    b_sf = torch.stack(b_sf_list, dim=0)  # [num_groups, n//128, k//128]

    d = torch.empty(padded_m, n, device=device, dtype=torch.bfloat16)
    # grouped_layout: group ID for each token, shape [padded_m]
    layout_list = []
    for g in range(num_groups):
        layout_list.append(torch.full((m_per_group,), g, dtype=torch.int32, device=device))
    # Pad remaining with last group
    remaining = padded_m - total_m
    if remaining > 0:
        layout_list.append(torch.full((remaining,), num_groups - 1, dtype=torch.int32, device=device))
    grouped_layout = torch.cat(layout_list)

    return (a_fp8, a_sf), (b_fp8, b_sf), d, grouped_layout


def check_cuda_ok(tag: str):
    """Synchronize all devices and check for CUDA errors."""
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.synchronize(i)
        except RuntimeError as e:
            print(f"  [FAIL] {tag} — CUDA error on device {i}: {e}")
            return False
    print(f"  [OK] {tag}")
    return True


def run_gemm_and_check(a, b, d, tag: str):
    """Run fp8_fp4_gemm_nt and verify no CUDA error."""
    try:
        deep_gemm.fp8_fp4_gemm_nt(a, b, d)
        return check_cuda_ok(tag)
    except Exception as e:
        print(f"  [FAIL] {tag} — Python exception: {e}")
        traceback.print_exc()
        return False


def run_grouped_gemm_and_check(a, b, d, grouped_layout, num_groups, tag: str):
    """Run m_grouped_fp8_fp4_gemm_nt_contiguous and verify no CUDA error."""
    try:
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            a, b, d, grouped_layout, use_psum_layout=False)
        return check_cuda_ok(tag)
    except Exception as e:
        print(f"  [FAIL] {tag} — Python exception: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test steps
# ---------------------------------------------------------------------------

def step1_single_gpu(device_id: int):
    """Baseline: GEMM on a single GPU."""
    dev = f"cuda:{device_id}"
    print(f"\n{'='*60}")
    print(f"Step 1.{device_id}: Single-GPU GEMM on {dev}")
    print(f"{'='*60}")

    torch.cuda.set_device(device_id)
    m, n, k = 1, 7168, 7168  # typical DeepSeek-V4 linear shape

    a, b, d = make_fp8_gemm_inputs(m, n, k, device=dev)
    ok = run_gemm_and_check(a, b, d, f"GEMM m={m} n={n} k={k} on {dev}")

    # Larger shape (expert-like)
    m2, n2, k2 = 32, 2048, 7168
    a2, b2, d2 = make_fp8_gemm_inputs(m2, n2, k2, device=dev)
    ok2 = run_gemm_and_check(a2, b2, d2, f"GEMM m={m2} n={n2} k={k2} on {dev}")

    return ok and ok2


def step2_alternating(num_rounds: int = 5):
    """Alternate GEMM between cuda:0 and cuda:1 (simulates model forward)."""
    print(f"\n{'='*60}")
    print(f"Step 2: Alternating GEMM cuda:0 ↔ cuda:1 ({num_rounds} rounds)")
    print(f"{'='*60}")

    m, n, k = 1, 7168, 7168
    all_ok = True

    for round_idx in range(num_rounds):
        for dev_id in [0, 1]:
            dev = f"cuda:{dev_id}"
            torch.cuda.set_device(dev_id)
            a, b, d = make_fp8_gemm_inputs(m, n, k, device=dev)
            tag = f"Round {round_idx}, {dev}"
            if not run_gemm_and_check(a, b, d, tag):
                all_ok = False
                print(f"  >>> FAILED at {tag}")
                return all_ok
    return all_ok


def step3_cross_device_reuse():
    """Allocate on cuda:0, then run from cuda:1 context.

    Tests whether cudaKernel_t handles are truly portable across contexts.
    """
    print(f"\n{'='*60}")
    print(f"Step 3: Cross-device kernel reuse test")
    print(f"{'='*60}")

    m, n, k = 1, 7168, 7168
    all_ok = True

    # First call on cuda:0 — compiles and caches the kernel
    torch.cuda.set_device(0)
    a0, b0, d0 = make_fp8_gemm_inputs(m, n, k, device="cuda:0")
    if not run_gemm_and_check(a0, b0, d0, "First call on cuda:0 (triggers JIT compile)"):
        all_ok = False

    # Second call on cuda:1 — should reuse the cached kernel
    torch.cuda.set_device(1)
    a1, b1, d1 = make_fp8_gemm_inputs(m, n, k, device="cuda:1")
    if not run_gemm_and_check(a1, b1, d1, "Second call on cuda:1 (reuses cached kernel)"):
        all_ok = False

    # Third call back on cuda:0
    torch.cuda.set_device(0)
    a0b, b0b, d0b = make_fp8_gemm_inputs(m, n, k, device="cuda:0")
    if not run_gemm_and_check(a0b, b0b, d0b, "Third call back on cuda:0"):
        all_ok = False

    return all_ok


def step4_jit_race():
    """Both GPUs trigger JIT compilation for different shapes.

    Tests whether the JIT compiler's global tmp directory causes conflicts.
    """
    print(f"\n{'='*60}")
    print(f"Step 4: JIT compilation race (different shapes per GPU)")
    print(f"{'='*60}")

    all_ok = True
    shapes = [
        (0, 1, 7168, 7168, "cuda:0 shape (1, 7168, 7168)"),
        (1, 1, 4096, 7168, "cuda:1 shape (1, 4096, 7168)"),
        (0, 8, 2048, 7168, "cuda:0 shape (8, 2048, 7168)"),
        (1, 16, 512, 7168, "cuda:1 shape (16, 512, 7168)"),
    ]

    for dev_id, m, n, k, tag in shapes:
        torch.cuda.set_device(dev_id)
        a, b, d = make_fp8_gemm_inputs(m, n, k, device=f"cuda:{dev_id}")
        if not run_gemm_and_check(a, b, d, tag):
            all_ok = False

    return all_ok


def step5_grouped_gemm():
    """Test grouped GEMM (MoE expert path) on both GPUs."""
    print(f"\n{'='*60}")
    print(f"Step 5: Grouped GEMM (MoE experts) on both GPUs")
    print(f"{'='*60}")

    all_ok = True
    num_groups = 8
    m_per_group = 4
    n = 2048  # intermediate_size
    k = 7168  # hidden_size

    for dev_id in [0, 1]:
        dev = f"cuda:{dev_id}"
        torch.cuda.set_device(dev_id)
        a, b, d, layout = make_grouped_fp8_inputs(num_groups, m_per_group, n, k, device=dev)
        tag = f"Grouped GEMM {num_groups} experts on {dev}"
        if not run_grouped_gemm_and_check(a, b, d, layout, num_groups, tag):
            all_ok = False

    # Alternate
    for round_idx in range(3):
        for dev_id in [0, 1]:
            dev = f"cuda:{dev_id}"
            torch.cuda.set_device(dev_id)
            a, b, d, layout = make_grouped_fp8_inputs(num_groups, m_per_group, n, k, device=dev)
            tag = f"Grouped round {round_idx}, {dev}"
            if not run_grouped_gemm_and_check(a, b, d, layout, num_groups, tag):
                all_ok = False
                return all_ok

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    num_gpus = torch.cuda.device_count()
    print(f"CUDA devices: {num_gpus}")
    for i in range(num_gpus):
        prop = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {prop.name} (SM {prop.major}.{prop.minor}, "
              f"{prop.multi_processor_count} SMs)")

    if num_gpus < 2:
        print("ERROR: Need at least 2 CUDA devices.")
        print("Run with: CUDA_VISIBLE_DEVICES=0,1 python test_dual_gpu_sm120.py")
        sys.exit(1)

    results = {}

    results["step1_gpu0"] = step1_single_gpu(0)
    results["step1_gpu1"] = step1_single_gpu(1)
    results["step2_alternating"] = step2_alternating(num_rounds=5)
    results["step3_cross_reuse"] = step3_cross_device_reuse()
    results["step4_jit_race"] = step4_jit_race()
    results["step5_grouped"] = step5_grouped_gemm()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s} {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed! Multi-GPU SM120 GEMM is working correctly.")
        print("The 'illegal memory access' in model forward is likely caused by")
        print("something else (e.g., model.generate() BMM device mismatch,")
        print("or accelerate hooks not properly managing device context).")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\nFailed steps: {failed}")
        if "step1_gpu0" in failed or "step1_gpu1" in failed:
            print("→ SM120 GEMM kernel itself is broken (single-GPU failure)")
        elif "step2_alternating" in failed or "step3_cross_reuse" in failed:
            print("→ Cross-device kernel reuse is the problem")
        elif "step4_jit_race" in failed:
            print("→ JIT compiler global state conflict")
        elif "step5_grouped" in failed:
            print("→ Grouped GEMM (experts path) has multi-GPU issues")
        sys.exit(1)


if __name__ == "__main__":
    main()
