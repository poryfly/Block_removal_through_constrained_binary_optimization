import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from copy import deepcopy
import copy
import argparse
import os, glob, heapq
import json
import re
import struct
from src.solve_binary_optimization.util_low_energy_spectrum import iter_global_lowest_n_from_sorted_batches, _torch_load_maybe_mmap

# Reverse mapping of transformers' _COMPRESS_RATIO_TO_LAYER_TYPE for DeepSeek-V4
# Used to reconstruct compress_ratios from layer_types after pruning
_LAYER_TYPE_TO_COMPRESS_RATIO = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
} 


def compress_model(model, layers_to_remove):
    """
    Removes specified layers from a model to create a compressed version.
    
    This function physically removes layers from the model architecture based on
    the provided indices. It also updates the model configuration to reflect the
    new number of layers.
    
    Args:
        model: The transformer model to compress (must have model.model.layers attribute).
        layers_to_remove (list): List of layer indices to remove from the model.
    
    Returns:
        The compressed model with specified layers removed.
    
    Note:
        Layers are removed in reverse order to maintain correct indices during deletion.
        For models with layer_types (qwen3, gpt_oss), the layer_types list is also updated.
    """


    s = "_".join(map(str, layers_to_remove))

    for idx in sorted(layers_to_remove, reverse=True):
        print("deleting layer", idx)
        del model.model.layers[idx]
    print("new length of model", len(model.model.layers))
    model.config.num_hidden_layers = len(model.model.layers)
    print(f"model type: {model.config.model_type}")
    if model.config.model_type == "qwen3" or model.config.model_type == "gpt_oss" or model.config.model_type == "deepseek_v4":
        list_of_layers=model.config.layer_types
        for idx in sorted(layers_to_remove, reverse=True):
            del list_of_layers[idx]
        model.config.layer_types=list_of_layers
        print(len(model.config.layer_types))
    if model.config.model_type == "deepseek_v4" and model.config.mlp_layer_types is not None:
        mlp_list = model.config.mlp_layer_types
        for idx in sorted(layers_to_remove, reverse=True):
            del mlp_list[idx]
        model.config.mlp_layer_types = mlp_list
        print(len(model.config.mlp_layer_types))

    return model


def fix_compressed_config(save_dir, model_config):
    """
    Fix config.json for compressed DeepSeek-V4 models after save_pretrained().

    transformers' DeepseekV4Config.__post_init__ consumes compress_ratios as a legacy
    kwarg and converts it to layer_types + compress_rates, causing save_pretrained()
    to output compress_ratios=None. sglang expects compress_ratios to be a list and
    crashes with IndexError when it's empty.

    This function directly patches the saved config.json to:
    1. Reconstruct compress_ratios from layer_types
    2. Set n_hash_layers from mlp_layer_types (count of "hash_moe" entries)
    3. Fix quantization_config (restore missing 'fmt' field)
    4. Fix torch_dtype (restore from None to "bfloat16")
    5. Fix rope_scaling (reconstruct from rope_parameters.compress)

    Args:
        save_dir: Directory where the model was saved (contains config.json).
        model_config: The model's config object (model.config).
    """
    if model_config.model_type != "deepseek_v4":
        return

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Reconstruct compress_ratios from layer_types
    layer_types = config_dict.get("layer_types", [])
    if layer_types:
        config_dict["compress_ratios"] = [
            _LAYER_TYPE_TO_COMPRESS_RATIO.get(lt, 0) for lt in layer_types
        ]
        print(f"Reconstructed compress_ratios: {config_dict['compress_ratios']}")

    # Set n_hash_layers and num_hash_layers from mlp_layer_types
    # sglang uses num_hash_layers in "2604" compressed mode, n_hash_layers otherwise
    mlp_layer_types = config_dict.get("mlp_layer_types", [])
    if mlp_layer_types:
        n_hash = mlp_layer_types.count("hash_moe")
        config_dict["n_hash_layers"] = n_hash
        config_dict["num_hash_layers"] = n_hash
        print(f"Set n_hash_layers/num_hash_layers: {n_hash}")

    # Fix quantization_config: restore 'fmt' field that transformers drops during serialization
    # Original model has: {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", ...}
    # save_pretrained() outputs: {"quant_method": "fp8", "dequantize": false, ...}  -- missing 'fmt'!
    # sglang needs 'fmt' to determine the FP8 sub-format (e4m3 vs e5m2) for correct GEMM kernel selection.
    qc = config_dict.get("quantization_config")
    if isinstance(qc, dict) and qc.get("quant_method") == "fp8" and "fmt" not in qc:
        qc["fmt"] = "e4m3"
        # Remove fields that transformers adds but original config doesn't have
        qc.pop("dequantize", None)
        qc.pop("modules_to_not_convert", None)
        print(f"Fixed quantization_config: added fmt=e4m3")

    # Fix torch_dtype: save_pretrained() outputs null, but inference frameworks need it
    if config_dict.get("torch_dtype") is None:
        config_dict["torch_dtype"] = "bfloat16"
        print(f"Fixed torch_dtype: bfloat16")

    # Fix rope_scaling: transformers converts rope_scaling into rope_parameters (with
    # "main" and "compress" sub-dicts) during __post_init__, and save_pretrained()
    # outputs rope_scaling=None. sglang reads rope_scaling directly and needs the YaRN
    # parameters for compress (CSA/HCA) layers: type, factor, beta_fast, beta_slow,
    # original_max_position_embeddings.
    if not config_dict.get("rope_scaling"):
        rp = config_dict.get("rope_parameters", {})
        compress_rp = rp.get("compress", {}) if isinstance(rp, dict) else {}
        if compress_rp.get("type") == "yarn" or compress_rp.get("rope_type") == "yarn":
            config_dict["rope_scaling"] = {
                "type": "yarn",
                "factor": compress_rp.get("factor", 16),
                "beta_fast": compress_rp.get("beta_fast", 32),
                "beta_slow": compress_rp.get("beta_slow", 1),
                "original_max_position_embeddings": compress_rp.get(
                    "original_max_position_embeddings", 65536
                ),
            }
            print(f"Fixed rope_scaling: {config_dict['rope_scaling']}")

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"Fixed config.json at {config_path}")


# ---------------------------------------------------------------------------
# Weight name remapping: transformers HuggingFace → DeepSeek original format
# ---------------------------------------------------------------------------

# Compressor submodule renames (applied inside .compressor. context)
_COMPRESSOR_SUB_RENAMES = {
    "kv_proj": "wkv",
    "gate_proj": "wgate",
    "position_bias": "ape",
    "kv_norm": "norm",
}

# HyperConnection parameter flattening
_HC_PARAM_MAP = {
    "attn_hc": {
        "fn": "hc_attn_fn",
        "base": "hc_attn_base",
        "scale": "hc_attn_scale",
    },
    "ffn_hc": {
        "fn": "hc_ffn_fn",
        "base": "hc_ffn_base",
        "scale": "hc_ffn_scale",
    },
}

# HyperHead parameter flattening (model-level, not per-layer)
_HYPERHEAD_PARAM_MAP = {
    "model.hc_head.hc_fn": "hc_head_fn",
    "model.hc_head.hc_base": "hc_head_base",
    "model.hc_head.hc_scale": "hc_head_scale",
}

# Attention module parameter renames (applied in generic context after self_attn→attn)
# These map HuggingFace/transformers names to DeepSeek original checkpoint names.
_ATTN_PARAM_RENAMES = {
    "kv_proj": "wkv",
    "q_a_proj": "wq_a",
    "q_b_proj": "wq_b",
    "o_a_proj": "wo_a",
    "o_b_proj": "wo_b",
    "sinks": "attn_sink",
    "q_a_norm": "q_norm",
}

# Generic submodule name replacements (order matters)
_GENERIC_SUB_REPLACES = [
    ("self_attn", "attn"),
    ("mlp", "ffn"),
    ("input_layernorm", "attn_norm"),
    ("post_attention_layernorm", "ffn_norm"),
]

# Expert projection renames (inside .experts.X. context)
_EXPERT_PROJ_RENAMES = {
    "gate_proj": "w1",
    "down_proj": "w2",
    "up_proj": "w3",
}


def rename_weight_key(name):
    """
    Rename a weight key from transformers HuggingFace format to DeepSeek
    original checkpoint format, so that sglang's
    remap_weight_name_to_dpsk_hf_format() can correctly process it.

    Priority order (first match wins):
    1. HyperHead (model.hc_head.hc_*)
    2. Indexer compressor sub-modules (self_attn.compressor.indexer.{kv_proj,...})
    3. Indexer non-compressor sub-modules (self_attn.compressor.indexer.{q_b_proj,scorer})
    4. Compressor sub-modules (self_attn.compressor.{kv_proj,...})
    5. HyperConnection (attn_hc/ffn_hc)
    6. Generic replacements (prefix, submodule names, expert proj names)
    """
    # 1. HyperHead
    for hf_name, ds_name in _HYPERHEAD_PARAM_MAP.items():
        if name == hf_name:
            return ds_name

    # 2 & 3. Indexer sub-modules (must be before Compressor check)
    # Pattern: model.layers.X.self_attn.compressor.indexer.{submod}.{param}
    m = re.match(
        r"(model\.layers\.\d+)\.self_attn\.compressor\.indexer\.(.*)",
        name,
    )
    if m:
        layer_prefix = m.group(1)  # e.g. "model.layers.1"
        rest = m.group(2)          # e.g. "kv_proj.weight" or "q_b_proj.weight"
        # Indexer compressor params: kv_proj, gate_proj, position_bias, kv_norm
        # These map to: layers.X.attn.indexer.compressor.{wkv/wgate/ape/norm}.{param}
        is_indexer_compressor = False
        for hf_name, ds_name in _COMPRESSOR_SUB_RENAMES.items():
            if rest.startswith(hf_name + ".") or rest == hf_name:
                new_rest = ds_name + rest[len(hf_name):]
                new_prefix = layer_prefix.replace("model.", "", 1)
                return f"{new_prefix}.attn.indexer.compressor.{new_rest}"
        # Indexer non-compressor params: q_b_proj, scorer
        # These map to: layers.X.attn.indexer.{submod}.{param}
        # Special case: the original checkpoint has `indexer.scorer.weights_proj`,
        # but sglang's C4Indexer places `weights_proj` directly under `indexer`.
        # Flatten the extra `scorer` wrapper so the remap can match.
        if rest.startswith("scorer.weights_proj"):
            rest = "weights_proj" + rest[len("scorer.weights_proj"):]
        new_prefix = layer_prefix.replace("model.", "", 1)
        return f"{new_prefix}.attn.indexer.{rest}"

    # 4. Compressor sub-modules (but NOT indexer — already handled above)
    # Pattern: model.layers.X.self_attn.compressor.{submod} or .{submod}.{param}
    m = re.match(
        r"(model\.layers\.\d+)\.self_attn\.compressor\.((?!indexer)[^.]+)(?:\.(.*))?",
        name,
    )
    if m:
        layer_prefix = m.group(1)  # e.g. "model.layers.0"
        submod = m.group(2)        # e.g. "kv_proj" or "position_bias"
        param = m.group(3)         # e.g. "weight" or None (for position_bias)
        for hf_name, ds_name in _COMPRESSOR_SUB_RENAMES.items():
            if submod == hf_name:
                new_prefix = layer_prefix.replace("model.", "", 1)
                if param:
                    return f"{new_prefix}.attn.compressor.{ds_name}.{param}"
                else:
                    return f"{new_prefix}.attn.compressor.{ds_name}"

    # 5. HyperConnection per-layer params
    m = re.match(r"model\.layers\.\d+\.", name)
    if m:
        layer_prefix = m.group(0)
        rest = name[len(layer_prefix):]
        for hc_mod, param_map in _HC_PARAM_MAP.items():
            if rest.startswith(hc_mod + "."):
                hc_param = rest[len(hc_mod) + 1:]
                if hc_param in param_map:
                    # e.g. attn_hc.base → hc_attn_base
                    new_layer_prefix = layer_prefix.replace("model.", "", 1)
                    return new_layer_prefix + param_map[hc_param]

    # 6. Generic replacements
    result = name

    # Strip "model." prefix
    if result.startswith("model."):
        result = result[6:]  # strip "model."

    # Special case: embed_tokens → embed, norm → norm (already correct)
    if result.startswith("embed_tokens."):
        result = "embed." + result[13:]
    elif result == "norm.weight":
        pass  # already correct
    else:
        # Apply submodule name replacements
        for old, new in _GENERIC_SUB_REPLACES:
            result = result.replace(f".{old}.", f".{new}.")

        # Attention module parameter renames (after self_attn→attn replacement)
        for hf_name, ds_name in _ATTN_PARAM_RENAMES.items():
            result = result.replace(f".{hf_name}.", f".{ds_name}.")
        # Handle trailing keys without '.' suffix (e.g. "attn.sinks" → "attn.attn_sink")
        for hf_name, ds_name in _ATTN_PARAM_RENAMES.items():
            if result.endswith(f".{hf_name}"):
                result = result[: -(len(hf_name))] + ds_name
                break

        # Flatten extra `scorer` wrapper in indexer (original checkpoint has
        # `indexer.scorer.weights_proj`, sglang's C4Indexer has `indexer.weights_proj`).
        # This replacement also makes the rename idempotent for already-converted keys.
        result = result.replace(".indexer.scorer.weights_proj.", ".indexer.weights_proj.")

        # Expert projection renames (only inside .experts.N. context)
        result = re.sub(
            r"(\.experts\.\d+\.)(gate_proj|down_proj|up_proj)(\.)",
            lambda m: m.group(1) + _EXPERT_PROJ_RENAMES[m.group(2)] + m.group(3),
            result,
        )

    return result


def _rename_safetensors_keys(st_path):
    """
    Rename weight keys in a safetensors file by modifying only the JSON header,
    without loading tensor data into memory. This is essential for large model
    files (tens of GB) where loading all tensors would cause OOM.

    The safetensors format is:
      - 8 bytes: header_size (uint64 little-endian)
      - header_size bytes: JSON header (key → {dtype, shape, data_offsets})
      - remaining bytes: tensor data

    data_offsets are byte offsets from the start of the data section (after header),
    so they remain valid even if the header size changes. We only rewrite the
    8-byte header_size + the JSON header, then stream-copy the data section.

    Returns the number of keys renamed.
    """
    import struct
    import shutil

    # Step 1: Read the header from the original file
    with open(st_path, "rb") as f:
        header_size_bytes = f.read(8)
        orig_header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_json_bytes = f.read(orig_header_size)
        data_section_offset = 8 + orig_header_size

    # Step 2: Parse the header and rename keys
    header = json.loads(header_json_bytes)
    new_header = {}
    renamed_count = 0

    for key, value in header.items():
        if key == "__metadata__":
            new_header[key] = value
            continue
        new_key = rename_weight_key(key)
        new_header[new_key] = value
        if new_key != key:
            renamed_count += 1

    if renamed_count == 0:
        print(f"No renames needed in {os.path.basename(st_path)}")
        return 0

    # Step 3: Build new header JSON (compact, no extra whitespace)
    new_header_json = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    # Pad with spaces to match safetensors alignment (header size must be multiple of 8
    # is NOT required by the format, but padding with spaces is safe for JSON)
    # We need at least the same or larger header to avoid issues.
    # Pad to at least orig_header_size with spaces (valid JSON whitespace).
    if len(new_header_json) < orig_header_size:
        new_header_json += b" " * (orig_header_size - len(new_header_json))
    new_header_size = len(new_header_json)

    # Step 4: Write new file (temp file, then rename for atomicity)
    tmp_path = st_path + ".tmp"
    try:
        with open(tmp_path, "wb") as out_f:
            # Write new header
            out_f.write(struct.pack("<Q", new_header_size))
            out_f.write(new_header_json)

            # Stream-copy the data section from the original file
            with open(st_path, "rb") as in_f:
                in_f.seek(data_section_offset)
                # Copy in chunks to avoid loading entire data section
                chunk_size = 64 * 1024 * 1024  # 64 MB chunks
                while True:
                    chunk = in_f.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)

        # Atomic rename
        os.replace(tmp_path, st_path)
        print(f"Renamed {renamed_count} keys in {os.path.basename(st_path)}")
    except Exception:
        # Clean up temp file on error
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return renamed_count


def fix_compressed_weights(save_dir, model_config):
    """
    Rename weight keys in safetensors files from transformers HuggingFace format
    to DeepSeek original format, so sglang can load the compressed model.

    Uses header-only modification to avoid loading tensor data into memory,
    making it safe for large model files (tens of GB).

    Args:
        save_dir: Directory containing the saved model (safetensors files).
        model_config: The model's config object (model.config).
    """
    if model_config.model_type != "deepseek_v4":
        return

    # Find all safetensors files
    st_files = sorted(glob.glob(os.path.join(save_dir, "*.safetensors")))
    if not st_files:
        print(f"No safetensors files found in {save_dir}, skipping weight rename")
        return

    total_renamed = 0
    for st_path in st_files:
        renamed = _rename_safetensors_keys(st_path)
        total_renamed += renamed

    # Update model.safetensors.index.json
    index_path = os.path.join(save_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index_data = json.load(f)

        old_wm = index_data.get("weight_map", {})
        new_wm = {}
        index_renamed = 0
        for old_key, shard in old_wm.items():
            new_key = rename_weight_key(old_key)
            new_wm[new_key] = shard
            if new_key != old_key:
                index_renamed += 1

        index_data["weight_map"] = new_wm
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        print(f"Updated index.json: {index_renamed} keys renamed")

    print(f"Total weight keys renamed: {total_renamed}")


def main():
    """
    Main function to generate compressed models from energy computation results.
    
    This function loads the top-k lowest energy states from computed energy files,
    extracts the corresponding layer removal configurations, and generates compressed
    models by removing those layers. Each compressed model is saved with its tokenizer.
    
    Command-line arguments:
        --filename: Directory containing energy computation results (results_*.pt files).
        --model_name: Path to the original pretrained model.
        --k: Number of top-k lowest energy models to generate.
    
    Output:
        Saves k compressed models in the filename directory, each named as:
        compressed_model_state_{i}_remove_{layer_indices}/
    """
    parser = argparse.ArgumentParser(description="Pruning script for Llama models.")
    parser.add_argument(
        "--filename", type=str, help="Path to the config file"
    )
    parser.add_argument(
        "--model_name", type=str, help="Path to the config file"
    )
    parser.add_argument(
        "--k", type=str, help="Path to the config file"
    )    
    args = parser.parse_args()

    filename = args.filename
    batch_glob="results_*b=*_*.pt"
    batch_files = sorted(glob.glob(os.path.join(filename, batch_glob)))
    print(batch_files)
    lowest_refs = list(iter_global_lowest_n_from_sorted_batches(batch_files, int(args.k)))
    energies = []
    vecs = []
    indices = []

    for e, fpath, j, idx in lowest_refs:
        d = _torch_load_maybe_mmap(fpath, map_location="cpu")
        energies.append(float(e))
        vecs.append(d["vecs"][j])
        indices.append(idx)
    for i in range(int(args.k)):
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        layers_to_remove = torch.where(vecs[i] == 1)[0].tolist()
        print(energies[i], layers_to_remove)
        s = "_".join(map(str, layers_to_remove))
        print(f"for k at {i} remove layers {s}")
        model = compress_model(model, layers_to_remove)
        save_dir = f"{filename}/compressed_model_state_{i}"
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        fix_compressed_config(save_dir, model.config)
        fix_compressed_weights(save_dir, model.config)
        del model
        del tokenizer


if __name__ == "__main__":
    main()