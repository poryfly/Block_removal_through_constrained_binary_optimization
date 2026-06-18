# Block removal for large language models through constrained binary optimization

This repository contains the code used to generate the results in [Block removal for large language models through constrained binary optimization](https://arxiv.org/abs/2602.00161).  
The pipeline consists of **data preparation**, **binary optimization for Hessian construction**, **energy computation**, and **model compression**, followed by **fine-tuning and benchmarking** against alternative methods.

---

## CBO Algorithm Overview

The Constrained Binary Optimization (CBO) algorithm is a novel approach for pruning large language models by:

1. **Formulating layer removal as a binary optimization problem** where each layer is assigned a binary variable (0 = remove, 1 = keep)
2. **Constructing a Hessian matrix** that captures the impact of removing different layer combinations
3. **Finding the optimal set of layers to remove** by computing energies for different binary configurations
4. **Generating compressed models** based on the lowest-energy configurations

The algorithm supports various model architectures including LLaMA, Qwen3, and DeepSeek-V4.

---

## Complete Pipeline Usage

### Step 1: Data Preparation

First, prepare the dataset used throughout the experiments:

```bash
python src/prepare_data/prepare_data.py \
  --config_file configs/data_configs/deepseek-v4.yaml
```

This step preprocesses the data and stores it according to the paths specified in the configuration file.

### Step 2: Binary Optimization and Hessian Construction

Generate the samples required to construct the Hessian matrix:

```bash
python -u -m src.binary_optimization.prepare_binary_optimization \
  --config_file configs/cbo_configs/deepseek-v4_pruning.yaml \
  2>&1 | tee logs/prepare_binary_optimization.log
```

This script generates the samples required to construct the Hessian matrix and stores a PyTorch tensor `A.pt` in the directory specified by `output_path` in the configuration file (e.g., `Amatrices/deepseek-v4_n_samples_2048_think/`).

### Step 3: Energy Computation

Compute the energies associated with removing a fixed number of blocks:

```bash
python -u ./src/solve_binary_optimization/compute_energies.py \
    -A_directory ./Amatrices/deepseek-v4_n_samples_2048_think/ \
    -output_directory /data2/work/Block_removal_through_constrained_binary_optimization/Amatrices/deepseek-v4_n_samples_2048_think/ \
    -ndel 42 \
    -num_gpus 8 \
    -top_k 100 \
    2>&1 | tee logs/compute_energies.log
```

Key parameters:
- `-A_directory`: Input directory containing the Hessian matrix (`A.pt`)
- `-output_directory`: Output directory for energy computation results
- `-ndel`: Number of layers to delete (e.g., 42 for DeepSeek-V4)
- `-num_gpus`: Number of GPUs to use for parallel computation
- `-top_k`: Number of lowest-energy states to save per batch

This creates a directory containing all binary states corresponding to the removal of the specified number of blocks, along with their associated energies.

### Step 4: Model Compression

Generate compressed models from the computed energies:

```bash
python -u -m src.compression.compress_model \
  --filename /data2/work/Block_removal_through_constrained_binary_optimization/Amatrices/deepseek-v4_n_samples_2048_think/energies_del42 \
  --model_name /data/.cache/models/deepseek-ai/DeepSeek-V4-Flash \
  --k 3 \
  2>&1 | tee logs/compress_model.log
```

- `--model_name` specifies the path to the original pretrained model.
- `--k` determines how many models are stored (ground state and first k-1 excited states).
- For DeepSeek-V4, the script automatically fixes config.json and renames weight keys for sglang compatibility.

### Step 5: Fine-tuning

To retrain the compressed models, run:

```bash
accelerate launch --config_file ./fsdp2_config4_full.yaml \
  ./src/finetuning/finetune_kd.py \
  -config_file ./configs/finetuning_configs/qwen3_8B_finetuning_kd.yaml \
  > ./logs/finetune.log 2>&1 &
```

---

## Baselines and Benchmark Methods

We benchmark against several alternative compression strategies.

### Block Influence Method

```bash
python src/block_influence/compression_multiblock.py \
  --config_file configs/BI_configs/llama8.yaml
```

---

### Norm Ratio Method

```bash
python src/norm_ratio/compression_multiblock.py \
  --config_file configs/norm_ratio_configs/llama8.yaml
```

---

### Sliding Window Method

To generate models using sliding windows, run:

```bash
python src/Sliding_windows/SLM_adapted.py \
  --model_path models/Llama-3.1-8B-Instruct/ \
  --model_name output/Llama-3.1-8B-Instruct \
  --threshold 0.40 \
  --dataset <dataset_path> \
  --target_count 2048
```

---

## Hyperparameter Settings

The following ratios were used in the experiments:

- **LLaMA-3.1-8B**
  - 16 blocks removed: `ratio = 0.12`
  - 8 blocks removed: `ratio = 0.36`

- **Qwen**
  - 12 blocks removed: `ratio = 0.64`
  - 8 blocks removed: `ratio = 0.77`

- **DeepSeek-V4**
  - 42 blocks removed (from 160 total layers)
  - Uses specialized configuration for FP8 quantization and MoE architecture
  - Automatically handles `layer_types`, `mlp_layer_types`, and `compress_ratios`
  - Fixes `rope_scaling`, `quantization_config`, and `torch_dtype` for sglang compatibility

---
## License
Patent Pending. The intended use is strictly limited to research and non-commercial projects.
If you find these results useful, please cite 
```
@article{jansen2026block,
  title={Block removal for large language models through constrained binary optimization},
  author={Jansen, David and Rausch, Roman and Montero, David and Orus, Roman},
  journal={arXiv preprint arXiv:2602.00161},
  year={2026}
}
```
## Supported Models

- **LLaMA-3.1-8B**: Standard dense architecture
- **Qwen3-8B/14B**: Dense architecture with custom layer types
- **DeepSeek-V4-Flash**: MoE architecture with FP8 quantization
  - Automatic config修复 for sglang compatibility
  - Weight key remapping from transformers format to DeepSeek original format
  - Support for `layer_types`, `mlp_layer_types`, and `compress_ratios`

## DeepSeek-V4 Specific Features

The repository includes comprehensive support for DeepSeek-V4 models:

1. **FP8 Quantization Support**: Properly handles `quantization_config` with `fmt: e4m3`
2. **MoE Architecture**: Supports `mlp_layer_types` and `n_hash_layers` configuration
3. **RoPE Scaling**: Fixes `rope_scaling` from `rope_parameters.compress` for YaRN
4. **Weight Remapping**: Automatic conversion from transformers HuggingFace format to DeepSeek original format
5. **Config Repair**: Post-processing to ensure sglang compatibility after `save_pretrained()`

## Notes

- All paths, hyperparameters, and preprocessing steps are specified via configuration files to ensure reproducibility.

 - Since some of the energies are close to degenerate, the exact order may be slightly modified in some cases due to numerical rounding errors and should be taken into account when evaluating the models.
