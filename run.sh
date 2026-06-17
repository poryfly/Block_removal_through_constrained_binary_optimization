#!/bin/bash
#

source /data/miniconda3/etc/profile.d/conda.sh
conda activate llamafactory

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/work/Block_removal_through_constrained_binary_optimization

#step 1 
#python src/prepare_data/prepare_data.py --config_file configs/data_configs/deepseek-v4.yaml

#step 2
#python -u -m src.binary_optimization.prepare_binary_optimization --config_file configs/cbo_configs/deepseek-v4_pruning.yaml 2>&1 | tee logs/prepare_binary_optimization.log

#step 3 (optimized: 8-GPU parallel + topk + matmul + async I/O)
# python -u ./src/solve_binary_optimization/compute_energies.py \
#     -A_directory ./Amatrices/deepseek-v4_n_samples_2048_think/ \
#     -output_directory /data2/work/Block_removal_through_constrained_binary_optimization/Amatrices/deepseek-v4_n_samples_2048_think/ \
#     -ndel 8 \
#     -num_gpus 8 \
#     -top_k 100 \
#     2>&1 | tee logs/compute_energies.log

#step 4
python -u -m src.compression.compress_model --filename /data2/work/Block_removal_through_constrained_binary_optimization/Amatrices/deepseek-v4_n_samples_2048_think/energies_del8 --model_name /data/.cache/models/deepseek-ai/DeepSeek-V4-Flash --k 3 2>&1 | tee logs/compress_model.log

#nohup accelerate launch --config_file ./fsdp2_config4_full.yaml ./src/finetuning/finetune_kd.py -config_file ./configs/finetuning_configs/qwen3_8B_finetuning_kd.yaml > ./logs/finetune.log 2>&1 &
