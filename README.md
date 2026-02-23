# Code Repository

This repository contains the code used to generate the results in [Block removal for large language models through constrained binary optimization](https://arxiv.org/abs/2602.00161).  
The pipeline consists of **data preparation**, **binary optimization for Hessian construction**, **energy computation**, and **model compression**, followed by **fine-tuning and benchmarking** against alternative methods.

---

## 1. Data Preparation

First, prepare the dataset used throughout the experiments:

```bash
python src/prepare_data/prepare_data.py \
  --config_file configs/data_configs/llama_3.1_8B_data_config.yaml
```

This step preprocesses the data and stores it according to the paths specified in the configuration file.

---

## 2. Binary Optimization and Hessian Construction

All code related to binary optimization is located in:

```
src/binary_optimization/
```

### 2.1 Generate Binary Optimization Samples

Run:

```bash
python src/binary_optimization/prepare_binary_optimization.py \
  --config_file configs/cbo_configs/llama3_pruning8B.yaml
```

This script generates the samples required to construct the Hessian matrix and stores a PyTorch tensor `A.pt` in the directory specified by `output_path` in the configuration file (e.g., `mnt/Amatrices`).

---

### 2.2 Compute Energies

Next, compute the energies associated with removing a fixed number of blocks:

```bash
python src/solve_binary_optimization/compute_energies.py \
  -A_directory mnt/Amatrices \
  -ndel 8
```

This creates a directory:

```
mnt/Amatrices/energies_del8/
```

which contains all binary states corresponding to the removal of 8 blocks, along with their associated energies.

---

## 3. Model Compression

To generate compressed models from the computed energies, run:

```bash
python src/compression/compress_model.py \
  --filename mnt/Amatrices/energies_del8/ \
  --model_name models/Llama-3.1-8B-Instruct/ \
  --k 3
```

- `model_name` specifies the path to the original pretrained model.
- `k` determines how many models are stored.
  - For example, `k = 3` stores the ground state and the first two excited states.

---

## 4. Fine-tuning

To retrain the compressed models, run:

```bash
accelerate launch src/finetuning/finetune_kd.py \
  -config_file configs/finetuning_configs/llama3_8B_finetuning_kd.yaml
```

---

## 5. Baselines and Benchmark Methods

We benchmark against several alternative compression strategies.
To do the evaluations, run:
```bash
python src/finetuning/finetune_kd.py \
  --model_path model_path
```
## Nemotron
The file to do the calculations  NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 can be found in
```
src/nemotron_files/
```

### 5.1 Block Influence Method

```bash
python src/block_influence/compression_multiblock.py \
  --config_file configs/BI_configs/llama8.yaml
```

---

### 5.2 Norm Ratio Method

```bash
python src/norm_ratio/compression_multiblock.py \
  --config_file configs/norm_ratio_configs/llama8.yaml
```

---

### 5.3 Sliding Window Method

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

## 6. Hyperparameter Settings

The following ratios were used in the experiments:

- **LLaMA-3.1-8B**
  - 16 blocks removed: `ratio = 0.12`
  - 8 blocks removed: `ratio = 0.36`

- **Qwen**
  - 12 blocks removed: `ratio = 0.64`
  - 8 blocks removed: `ratio = 0.77`

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
## Notes

- All paths, hyperparameters, and preprocessing steps are specified via configuration files to ensure reproducibility.

 - Since some of the energies are close to degenerate, the exact order may be slightly modified in some cases due to numerical rounding errors and should be taken into account when evaluating the models.
