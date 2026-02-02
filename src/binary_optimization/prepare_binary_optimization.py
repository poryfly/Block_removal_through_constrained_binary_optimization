import argparse
import yaml

from  src.binary_optimization.utils import (
    prepare_pruning
)

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

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from accelerate import dispatch_model
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
        config["model"]["path"],  device_map="auto",dtype=torch.bfloat16)
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
model=model.to(torch.bfloat16)

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
model = dispatch_model(model, device_map=device_map)

print("number of parameters", len(params))

messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
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


weights=[]
for key, item in grads.items():
    A=torch.cat(item, dim=0)

    weights+=[A]
A=torch.cat(weights, dim=1)
output_path=config["model"]["output_path"]
print(output_path)
torch.save(A, f"{output_path}/A.pt")