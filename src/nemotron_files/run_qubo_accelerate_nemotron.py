import argparse
import yaml

from src.nemotron_files.utils import (
    prepare_pruning_nemotron
)

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.utils.data import DataLoader

import torch
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import torch.optim as optim
from accelerate import dispatch_model
#accelerator = Accelerator()
parser = argparse.ArgumentParser(description="Pruning script for Llama models.")
parser.add_argument(
        "--config_file", type=str, help="Path to the config file"
    )

seed=42
args = parser.parse_args()

with open(args.config_file, "r") as file:
    config = yaml.safe_load(file)
print("start loading")

# hardcoded for now, probably does not make sense to change it
scale=1.0



batch_size=1
model = AutoModelForCausalLM.from_pretrained(
        config["model"]["path"],  device_map="auto",dtype=torch.bfloat16,trust_remote_code=True)
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
        dataset = dataset.filter(lambda x: len(x['seq_lengths']) < config["calibration"]["sequence_filter"])
dataset = dataset.remove_columns("text")
dataset = dataset.select(range(config["calibration"]["n_samples"]))
dataloader = DataLoader(dataset , batch_size=config["calibration"]["batch_size"], shuffle=False, collate_fn=data_collator)
assert(config["calibration"]["batch_size"]==1)


prepare_pruning_nemotron(
    model, scale
    )
print("pruning preparation done")
model=model.to(torch.bfloat16)

for name, param in model.named_parameters():
    #print(name, param.shape)
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
print("allocating device")
device_map=model.hf_device_map
model = dispatch_model(model, device_map=device_map)
print("device map", device_map)

messages = [
    {"role": "user", "content": "Write a haiku about GPUs"},
]
print("tokenizing")
tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    tokenized_chat,
    max_new_tokens=1024,
    temperature=1.0,
    top_p=1.0,
    eos_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(outputs[0]))

for batch in tqdm(dataloader, desc="Processing examples"):


    optimizer.zero_grad()
    output=model(**batch, labels=batch["input_ids"])

    loss=output.loss

    #accelerator.backward(loss)
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
torch.save(A, f"{output_path}/A.pt")
model_type=model.config.model_type
del model

