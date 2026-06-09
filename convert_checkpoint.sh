#!/bin/bash
# Convert FSDP SHARDED_STATE_DICT checkpoint to HuggingFace safetensors format
# Usage: bash convert_checkpoint.sh <checkpoint_dir> <output_dir>
# Example: bash convert_checkpoint.sh output/checkpoint-500 output/checkpoint-500-hf
#
# Requirements:
#   - conda activate llamafactory
#   - Enough CPU RAM (approx 36GB for Qwen3-8B bfloat16)
#   - No GPU required

set -e

# ── argument parsing ────────────────────────────────────────────────────────
CHECKPOINT_DIR="${1:-output/checkpoint-500}"
OUTPUT_DIR="${2:-${CHECKPOINT_DIR}-hf}"

# Source model dir (provides config.json, tokenizer files, etc.)
SOURCE_MODEL_DIR="/data2/work/Block_removal_through_constrained_binary_optimization/Amatrices/Qwen3-8B_n_samples_2048_think/energies_del8/compressed_model_state_1"

# ── environment ─────────────────────────────────────────────────────────────
source /data/miniconda3/etc/profile.d/conda.sh
conda activate llamafactory

cd /data/work/Block_removal_through_constrained_binary_optimization

echo "============================================"
echo "  FSDP Checkpoint → HF safetensors Convert"
echo "============================================"
echo "  checkpoint : ${CHECKPOINT_DIR}"
echo "  output     : ${OUTPUT_DIR}"
echo "  source cfg : ${SOURCE_MODEL_DIR}"
echo ""

# ── sanity checks ────────────────────────────────────────────────────────────
SHARDS_DIR="${CHECKPOINT_DIR}/pytorch_model_fsdp_0"

if [ ! -d "${SHARDS_DIR}" ]; then
    echo "[ERROR] Shards directory not found: ${SHARDS_DIR}"
    exit 1
fi

if [ ! -f "${SHARDS_DIR}/.metadata" ]; then
    echo "[ERROR] .metadata not found in ${SHARDS_DIR}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ── step 1: merge FSDP shards → safetensors ─────────────────────────────────
echo "[1/3] Merging FSDP shards with accelerate merge-weights ..."
echo "      Source : ${SHARDS_DIR}"
echo "      Target : ${OUTPUT_DIR}"
echo ""

accelerate merge-weights "${SHARDS_DIR}" "${OUTPUT_DIR}"

echo ""
echo "[1/3] Done. Merged weights saved to ${OUTPUT_DIR}"

# ── step 2: copy tokenizer and config files ──────────────────────────────────
echo ""
echo "[2/3] Copying config and tokenizer files from source model ..."

COPY_FILES=(
    "config.json"
    "generation_config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "special_tokens_map.json"
    "added_tokens.json"
    "vocab.json"
    "merges.txt"
    "chat_template.jinja"
)

for f in "${COPY_FILES[@]}"; do
    src="${SOURCE_MODEL_DIR}/${f}"
    dst="${OUTPUT_DIR}/${f}"
    if [ -f "${src}" ]; then
        cp "${src}" "${dst}"
        echo "      copied: ${f}"
    fi
done

echo "[2/3] Done."

# ── step 3: verify the output ────────────────────────────────────────────────
echo ""
echo "[3/3] Verifying output with a quick load test ..."

export _CONVERT_OUTPUT_DIR="${OUTPUT_DIR}"
python - <<'PYEOF'
import os

# Read OUTPUT_DIR from env (passed via shell)
output_dir = os.environ.get("_CONVERT_OUTPUT_DIR", "")

print(f"  Loading model from: {output_dir}")
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(output_dir)
print(f"  Tokenizer loaded OK (vocab_size={tokenizer.vocab_size})")

model = AutoModelForCausalLM.from_pretrained(
    output_dir,
    torch_dtype=torch.bfloat16,
    device_map="cpu",        # CPU-only verification, no GPU needed
)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Model loaded OK  (params={total_params/1e9:.2f}B)")
print(f"  num_hidden_layers = {model.config.num_hidden_layers}")
print("")
print("  Verification PASSED")
PYEOF

echo ""
echo "============================================"
echo "  Conversion complete!"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================"
