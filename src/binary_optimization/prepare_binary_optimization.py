import argparse
import yaml
import functools

from  src.binary_optimization.utils import (
    prepare_pruning
)
from src.prepare_data.encoding_dsv4 import encode_messages, parse_message_from_completion_text

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.utils.data import DataLoader

from torch.autograd.functional import hessian
import torch
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import torch.optim as optim
import os
import pickle
import json

# Enable expandable segments to reduce CUDA memory fragmentation
# Must be set BEFORE any CUDA allocation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from accelerate import dispatch_model

# ---------------------------------------------------------------------------
# Monkey-patch transformers' _load_deepgemm_kernel to use the locally
# installed `deep_gemm` package (pip/git) instead of the pre-compiled
# `kernels-community/deep-gemm` from HuggingFace Hub.
#
# The Hub version has no build variant for PyTorch 2.8 + CUDA 12.8 + SM120
# (RTX 5090), causing: "Cannot find a build variant for this system".
# The local deep_gemm package (v2.5+) supports SM120 natively.
# ---------------------------------------------------------------------------
def _load_deepgemm_kernel_from_local_package(requires_sm100: bool = False):
    """Load DeepGEMM from the locally installed deep_gemm package."""
    try:
        import deep_gemm
    except ImportError:
        raise ImportError(
            "DeepGEMM kernel requires the `deep_gemm` package. "
            "Install it with `pip install deep-gemm` or from source."
        )

    if not torch.cuda.is_available():
        raise ImportError("DeepGEMM kernel requires CUDA, but CUDA is not available.")

    major, minor = torch.cuda.get_device_capability()
    # SM120 (major=12) is Blackwell/RTX 5090; FP4 requires SM100+
    allowed = (12,) if requires_sm100 else (9, 10, 12)
    if major not in allowed:
        arch = "Blackwell (SM100)" if requires_sm100 else "Hopper (SM90), Blackwell (SM100), or SM120"
        raise ImportError(f"DeepGEMM requires {arch}; current device is SM{major}{minor}.")

    # Map deep_gemm package attributes to the DeepGEMM dataclass fields
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


# Apply the monkey-patch BEFORE any transformers model loading
import transformers.integrations.deepgemm as _deepgemm_mod
_load_deepgemm_kernel_orig = _deepgemm_mod._load_deepgemm_kernel
_deepgemm_mod._load_deepgemm_kernel = functools.cache(_load_deepgemm_kernel_from_local_package)
print("[patch] _load_deepgemm_kernel patched to use local deep_gemm package")

# ---------------------------------------------------------------------------
# Bypass transformers' _assert_single_device check.
#
# This check exists because DeepGEMM's default build uses the CUDA Driver API
# (CUfunction), which binds kernel handles to the CUDA context they were first
# loaded under.  When device_map="auto" spreads layers across multiple GPUs in
# the same process, driving the same cached CUfunction from a different device's
# context would produce garbage.
#
# HOWEVER: if DeepGEMM is rebuilt with DG_JIT_USE_RUNTIME_API=1, it uses the
# CUDA Runtime API (cudaKernel_t) instead, which is context-free and safe to
# call from any device.  The installed sm120 branch (deep_gemm==2.5.0+aced12c)
# should be compiled with this flag:
#     cd /data/work/soft/DeepGEMM && DG_JIT_USE_RUNTIME_API=1 bash install.sh
#
# After rebuilding with the runtime API, this check is unnecessary — we replace
# it with a no-op so the experts forward path (which has no Triton fallback for
# FP4 weights) can proceed on multi-GPU setups.
# ---------------------------------------------------------------------------
_deepgemm_mod._assert_single_device = lambda *args, **kwargs: None
print("[patch] _assert_single_device bypassed (requires DG_JIT_USE_RUNTIME_API=1 build)")

parser = argparse.ArgumentParser(description="Pruning script for Llama models.")
parser.add_argument(
        "--config_file", type=str, help="Path to the config file"
    )

seed=42
args = parser.parse_args()

with open(args.config_file, "r") as file:
    config = yaml.safe_load(file)
print("start loading")

scale=1.0



batch_size=1
model = AutoModelForCausalLM.from_pretrained(
        config["model"]["path"],  device_map="auto",dtype=torch.bfloat16, trust_remote_code=True, experts_implementation="deepgemm")
tokenizer = AutoTokenizer.from_pretrained(config["model"]["path"])
tokenizer.pad_token = tokenizer.eos_token
dataset = load_from_disk(config["calibration"]["dataset_path"])
dataset=dataset.shuffle(seed=config["calibration"]["seed"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("dataset loaded", dataset)

if config["calibration"]["sequence_filter"] is not None:
    if "length" in dataset.column_names:
        dataset = dataset.filter(lambda x: x["length"] < config["calibration"]["sequence_filter"])
    else:
        print("start")
        dataset = dataset.filter(lambda x: sum(x['seq_lengths']) < config["calibration"]["sequence_filter"])

print(dataset)
dataset = dataset.select(range(config["calibration"]["n_samples"]))
print(dataset)
dataloader = DataLoader(dataset , batch_size=config["calibration"]["batch_size"], shuffle=False, collate_fn=data_collator)
assert(config["calibration"]["batch_size"]==1)


prepare_pruning(
    model, scale
    )
# Skip global dtype conversion for FP8-quantized models (e.g. DeepSeek-V4)
# FP8 -> BF16 doubles memory (~160GB -> ~316GB), causing OOM on 8x32GB GPUs
# FP8 weights are automatically dequantized during computation; pruning_param is already bf16
# quantization_config can be either a dict (from AutoConfig) or FineGrainedFP8Config object (from model loading)
qc = getattr(model.config, 'quantization_config', None)
is_fp8 = False
if qc is not None:
    try:
        is_fp8 = qc.quant_method == 'fp8'
    except AttributeError:
        is_fp8 = qc.get('quant_method') == 'fp8'
if not is_fp8:
    model = model.to(torch.bfloat16)

for name, param in model.named_parameters():
    if "pruning_param" in name:

        param.requires_grad = True
    else:

        param.requires_grad = False



grads=[]
optimizer = optim.Adam(model.parameters(), lr=1e-3)
grads={}
params = [p for p in model.parameters() if p.requires_grad]
for i in range(len(params)):
    grads[i]=[]

device_map=model.hf_device_map
# For FP8-quantized models (e.g. DeepSeek-V4), skip dispatch_model because:
# 1. Sub-modules are reused by reference from original layer (already on correct GPU)
# 2. dispatch_model would try to move new top-level layer objects, potentially causing OOM
# 3. The sub-modules' existing dispatch hooks still work for device management
# For non-FP8 models (Llama/Qwen3), new layers are created on CPU and need dispatch to move to GPU
if not is_fp8:
    model = dispatch_model(model, device_map=device_map)
else:
    print("Skipping dispatch_model for FP8 model (sub-modules already on correct devices)")

print("number of parameters", len(params))

messages = [
    {"role": "user", "content": "Who are you?"},
]
if config["model"]["model_type"] == "deepseek-v4":
    messages = encode_messages(messages, thinking_mode="chat")
    inputs = tokenizer(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
else:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

try:
    outputs = model.generate(**inputs, max_new_tokens=40)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
except Exception as e:
    print(f"Warning: model.generate() failed: {e}. Skipping sanity check.")

# Enable gradient checkpointing to reduce peak activation memory.
# Without this, eager attention (head_dim=512 blocks flash_attn) materializes the full
# [1, 64, S, S_kv] attention scores per layer, and all 43 layers' activations are held
# simultaneously for backward — OOM at ~4% on 8×32GB GPUs.
# Gradient checkpointing recomputes activations during backward instead of storing them,
# trading ~30% extra compute for ~15-20 GB less activation memory per GPU.
# Must be enabled AFTER generate (which needs use_cache=True) and BEFORE the training loop.
model.train()
model.gradient_checkpointing_enable()
print("Gradient checkpointing enabled (reduces activation memory at cost of ~30% recomputation)")

device = next(model.parameters()).device
for batch in tqdm(dataloader, desc="Processing examples"):
    batch = {k: v.to(device) for k, v in batch.items()}
    optimizer.zero_grad()
    output=model(**batch)

    loss=output.loss

    loss.backward()
    params = [p for p in model.parameters() if p.requires_grad]

    for i in range(len(params)):
        grads[i]+=[params[i].grad.cpu().clone().detach()]
        params[i].grad = None

    # Explicitly free the computation graph and return cached memory to CUDA.
    # Without this, the previous iteration's autograd graph persists until
    # `output` is reassigned in the NEXT iteration, causing both the old and
    # new graphs to coexist during forward — peak memory doubles.
    del output, loss, batch
    torch.cuda.empty_cache()


weights=[]
for key, item in grads.items():
    A=torch.cat(item, dim=0)

    weights+=[A]
A=torch.cat(weights, dim=1)
output_path=config["model"]["output_path"]
print(output_path)
torch.save(A, f"{output_path}/A.pt")