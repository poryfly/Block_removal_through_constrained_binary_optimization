import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import yaml
from src.block_influence.calculate_bi import (
    remove_blocks_multiblock,
    remove_blocks_one_by_one
)
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from datasets import load_from_disk
parser = argparse.ArgumentParser(description="Pruning script for Llama models.")
parser.add_argument(
        "--config_file", type=str, help="Path to the config file"
    )
seed=42
args = parser.parse_args()

with open(args.config_file, "r") as file:
    config = yaml.safe_load(file)



model_path = config["model"]["path"]
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_from_disk(config["calibration"]["dataset_path"])
dataset=dataset.shuffle(seed=config["calibration"]["seed"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
if config["calibration"]["sequence_filter"] is not None:
    dataset = dataset.filter(lambda x: x["length"] < config["calibration"]["sequence_filter"])
dataset = dataset.select(range(config["calibration"]["n_samples"]))
dataloader = DataLoader(dataset , batch_size=config["calibration"]["batch_size"], shuffle=False, collate_fn=data_collator)
print(config["pruning"]["number_of_blocks_to_remove"])
for num_layers_to_remove in config["pruning"]["number_of_blocks_to_remove"]:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",

    )

    if config["pruning"]["pruning_type"] == "consecutive":
        remove_blocks_multiblock(model, tokenizer, dataloader,  num_layers_to_remove, save_model=config["pruning"]["save_model"], output_path=config["model"]["output_path"])
    elif config["pruning"]["pruning_type"] == "one_by_one":
        remove_blocks_one_by_one(model, tokenizer, dataloader,  num_layers_to_remove, save_model=config["pruning"]["save_model"], output_path=config["model"]["output_path"])
    del model
 