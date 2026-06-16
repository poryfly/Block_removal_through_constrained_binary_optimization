"""Phase 2: Model-level diagnostics (8 GPU, requires full DeepSeek-V4-Flash).

Steps:
  0. Load model + apply prepare_pruning + verify hooks/params
  A. Single forward pass (no_grad) + CUDA sync
  B. model.generate() + CUDA sync
  C. Forward after generate (detect CUDA state pollution)
  D. Forward + backward (autograd / training step)

Run:  python diagnose_model_forward.py
      (uses all 8 GPUs, no CUDA_VISIBLE_DEVICES restriction)
"""

import functools
import traceback
import torch
import yaml
import os

# Enable expandable segments to reduce CUDA memory fragmentation
# Must be set BEFORE any CUDA allocation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── Monkey-patches (identical to prepare_binary_optimization.py) ─────────────

def _load_deepgemm_kernel_from_local_package(requires_sm100: bool = False):
    import deep_gemm
    if not torch.cuda.is_available():
        raise ImportError("CUDA not available")
    from transformers.integrations.deepgemm import DeepGEMM
    return DeepGEMM(
        fp8_fp4_matmul=deep_gemm.fp8_fp4_gemm_nt,
        grouped_fp8_fp4_matmul_nt=deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous,
        grouped_fp8_fp4_matmul_nn=deep_gemm.m_grouped_fp8_fp4_gemm_nn_contiguous,
        grouped_bf16_matmul_nt=deep_gemm.m_grouped_bf16_gemm_nt_contiguous,
        grouped_bf16_matmul_nn=deep_gemm.m_grouped_bf16_gemm_nn_contiguous,
        per_token_cast_to_fp8=deep_gemm.per_token_cast_to_fp8,
        transform_sf_into_required_layout=deep_gemm.transform_sf_into_required_layout,
        transform_weights_for_mega_moe=deep_gemm.transform_weights_for_mega_moe,
        get_symm_buffer_for_mega_moe=deep_gemm.get_symm_buffer_for_mega_moe,
        fp8_fp4_mega_moe=deep_gemm.fp8_fp4_mega_moe,
        m_alignment=int(deep_gemm.get_mk_alignment_for_contiguous_layout()),
    )

import transformers.integrations.deepgemm as _deepgemm_mod
_deepgemm_mod._load_deepgemm_kernel = functools.cache(_load_deepgemm_kernel_from_local_package)
_deepgemm_mod._assert_single_device = lambda *args, **kwargs: None
print("[patch] DeepGEMM patched: local package + _assert_single_device bypass")

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from accelerate import dispatch_model
from src.binary_optimization.utils import prepare_pruning
from src.prepare_data.encoding_dsv4 import encode_messages

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG_PATH = "configs/cbo_configs/deepseek-v4_pruning.yaml"


def cuda_sync_check(label: str) -> bool:
    """Synchronize ALL GPUs and check for CUDA errors."""
    try:
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device=i)
        print(f"  [{label}] CUDA sync OK")
        return True
    except RuntimeError as e:
        print(f"  [{label}] CUDA ERROR: {e}")
        return False


def run_step(name: str, func):
    """Run a step, catch exceptions, check CUDA state afterwards."""
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


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    model_path = config["model"]["path"]
    dataset_path = config["calibration"]["dataset_path"]

    # ── Step 0: Load model ─────────────────────────────────────────────────
    print(f"\nLoading model from {model_path} ...")
    print(f"  CUDA devices: {torch.cuda.device_count()}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        experts_implementation="deepgemm",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    deepgemm_kernel = _deepgemm_mod._load_deepgemm_kernel()
    n_gpus = len(set(model.hf_device_map.values()))
    print(f"[Step 0] DeepGEMM m_alignment={deepgemm_kernel.m_alignment}")
    print(f"[Step 0] Model spread across {n_gpus} GPUs")

    # Show device map summary
    device_counts = {}
    for _, dev in model.hf_device_map.items():
        dev_str = str(dev)
        device_counts[dev_str] = device_counts.get(dev_str, 0) + 1
    for dev, cnt in sorted(device_counts.items()):
        print(f"  {dev}: {cnt} modules")

    # ── Apply prepare_pruning ──────────────────────────────────────────────
    print("\nApplying prepare_pruning ...")
    prepare_pruning(model, 1.0)

    # Verify pruning params
    pp_count = 0
    for name, param in model.named_parameters():
        if "pruning_param" in name:
            if pp_count == 0:
                print(f"  First pruning_param: {name} → {param.device} {param.dtype} {param.shape}")
            pp_count += 1
    print(f"  Total pruning_params: {pp_count}")

    # Check accelerate hooks
    hook_summary = {"layer_with_hook": 0, "layer_without": 0}
    for layer in model.model.layers:
        if getattr(layer, "_hf_hook", None) is not None:
            hook_summary["layer_with_hook"] += 1
        else:
            hook_summary["layer_without"] += 1
    print(f"  Accelerate hooks: {hook_summary}")

    # ── FP8 detection (skip global bf16 cast to avoid OOM) ─────────────────
    print("\nDetecting FP8 quantization ...")
    qc = getattr(model.config, 'quantization_config', None)
    is_fp8 = False
    if qc is not None:
        try:
            is_fp8 = qc.quant_method == 'fp8'
        except AttributeError:
            is_fp8 = qc.get('quant_method') == 'fp8'
    print(f"  is_fp8: {is_fp8}")
    if not is_fp8:
        model = model.to(torch.bfloat16)
        print("  model cast to bfloat16")
    else:
        print("  skipping model.to(bfloat16) — FP8 weights stay as-is")

    # ── Set requires_grad on pruning_params + optimizer ─────────────────────
    print("\nSetting requires_grad on pruning_params ...")
    import torch.optim as optim
    for name, param in model.named_parameters():
        if "pruning_param" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"  Trainable params: {len(params)}")

    # ── Conditional dispatch_model (skip for FP8) ───────────────────────────
    device_map = model.hf_device_map
    if not is_fp8:
        model = dispatch_model(model, device_map=device_map)
        print("  dispatch_model applied")
    else:
        print("  Skipping dispatch_model for FP8 model (sub-modules already on correct devices)")

    # ── Prepare dataset ────────────────────────────────────────────────────
    dataset = load_from_disk(dataset_path)
    dataset = dataset.shuffle(seed=42)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=data_collator)
    batch = next(iter(dataloader))
    print(f"\n  Batch keys: {list(batch.keys())}")
    for k, v in batch.items():
        print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")

    device = next(model.parameters()).device
    print(f"  Model first-param device: {device}")

    # ── Step A: Forward pass (no_grad) ─────────────────────────────────────
    def step_a_forward():
        # Pre-sync to ensure no stale CUDA errors from model loading / hook init
        print(f"  Pre-sync (checking CUDA state before forward) ...")
        try:
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device=i)
            print(f"  Pre-sync OK")
        except RuntimeError as e:
            print(f"  Pre-sync FAILED: {e}")
            print(f"  CUDA state is already corrupted before forward — error is in model loading or hook init")
            return

        # Print hook details for first few layers
        for i in [0, 1, 6, 7]:
            if i < len(model.model.layers):
                layer = model.model.layers[i]
                hook = getattr(layer, "_hf_hook", None)
                if hook:
                    print(f"  Layer {i} hook: execution_device={hook.execution_device}, offload={hook.offload}, place_submodules={hook.place_submodules}")
                else:
                    print(f"  Layer {i}: NO HOOK")

        batch_dev = {k: v.to(device) for k, v in batch.items()}
        print(f"  input_ids: shape={list(batch_dev['input_ids'].shape)}, device={batch_dev['input_ids'].device}")
        with torch.no_grad():
            output = model(**batch_dev)
        print(f"  Loss: {output.loss.item():.4f}")

    run_step("Step A: Forward pass (no_grad)", step_a_forward)

    # ── Step B: model.generate() ───────────────────────────────────────────
    def step_b_generate():
        model_type = config["model"].get("model_type", "")
        messages = [{"role": "user", "content": "Who are you?"}]
        if model_type == "deepseek-v4":
            messages = encode_messages(messages, thinking_mode="chat")
            inputs = tokenizer(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(device)
        else:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
        print(f"  Input shape: {list(inputs['input_ids'].shape)}")
        try:
            outputs = model.generate(**inputs, max_new_tokens=40)
            text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
            print(f"  Generated: {repr(text[:200])}")
        except Exception as e:
            print(f"  Warning: model.generate() failed: {e}. Continuing diagnostics.")

    run_step("Step B: model.generate()", step_b_generate)

    # ── Step C: Forward after generate (CUDA state check) ──────────────────
    def step_c_forward_after_generate():
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output = model(**batch_dev)
        print(f"  Loss: {output.loss.item():.4f}")

    run_step("Step C: Forward after generate", step_c_forward_after_generate)

    # ── Enable gradient checkpointing before Step D ─────────────────────
    # Mirrors the same change in prepare_binary_optimization.py.
    # Reduces activation memory by recomputing during backward (~30% slower but
    # prevents OOM on 8×32GB GPUs with eager attention).
    model.train()
    model.gradient_checkpointing_enable()
    print("[info] Gradient checkpointing enabled for Step D")

    # ── Step D: Forward + backward (training step) ─────────────────────────
    def step_d_training():
        batch_dev = {k: v.to(device) for k, v in batch.items()}

        params = [p for p in model.parameters() if p.requires_grad]
        print(f"  Trainable params: {len(params)}")

        optimizer.zero_grad()
        output = model(**batch_dev)
        loss = output.loss
        print(f"  Loss: {loss.item():.4f}")

        loss.backward()
        print(f"  Backward done.")

        for i, p in enumerate(params[:3]):
            if p.grad is not None:
                print(f"  Grad[{i}]: norm={p.grad.norm().item():.6f}, device={p.grad.device}")
                p.grad = None
            else:
                print(f"  Grad[{i}]: None")

        # Explicitly free the computation graph and return cached memory to CUDA
        del output, loss, batch_dev
        torch.cuda.empty_cache()

    run_step("Step D: Forward + backward", step_d_training)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Phase 2 complete.")
    print(f"{'='*60}")
    print("Interpretation:")
    print("  A fails → issue in basic forward pass (likely device/dtype mismatch)")
    print("  B fails → issue in generate/decode (DeepGEMM AB-swap or KV cache)")
    print("  C fails → generate corrupted CUDA state (B is the root cause)")
    print("  D fails → issue in autograd (pruning_param backward path)")
    print("  All pass → error is intermittent or shape-dependent")


if __name__ == "__main__":
    main()
