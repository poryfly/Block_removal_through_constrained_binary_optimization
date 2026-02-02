


import argparse
import yaml
import os

parser = argparse.ArgumentParser(description="Pruning script for Llama models.")
parser.add_argument(
        "--model_path", type=str, help="Path to the config file"
    )

seed=42
args = parser.parse_args()
model_path=args.model_path
benchmarks="hellaswag,winogrande,arc_challenge,mmlu,leaderboard_bbh,gsm8k"
command=f"accelerate launch  -m lm_eval --model hf     --model_args pretrained={model_path},dtype=bfloat16     --tasks {benchmarks}    --batch_size auto --output_path {model_path}/benchmark_output"

print(command)

os.system(command)