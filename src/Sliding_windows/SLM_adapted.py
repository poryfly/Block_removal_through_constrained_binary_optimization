from tqdm.notebook import tqdm
from copy import deepcopy
import copy

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from accelerate import dispatch_model
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from tqdm import tqdm
from transformers import default_data_collator, Trainer, TrainingArguments

from short_hf import ShortHFModel
import pandas as pd
import numpy as np
import argparse
import random

from datasets import load_dataset, load_from_disk
#--------------------------------------------------------------------------------------------------------------------------------------------------
from torch.nn.utils.rnn import pad_sequence

def get_first_n_fixed_length(dataset, n, field="input_ids", pad_value=0, max_length=128):
    """
    Returns a tensor of the first n sequences from the dataset,
    truncated or padded to max_length.
    
    Args:
        dataset: HuggingFace Dataset object
        n: number of elements to include
        field: which field to use (e.g., 'input_ids', 'labels')
        pad_value: integer used for padding (usually tokenizer.pad_token_id)
        max_length: fixed length to pad/truncate sequences
    
    Returns:
        Tensor of shape (n, max_length)
    """
    sequences = []
    for i in range(n):
        seq = torch.tensor(dataset[i][field])
        # truncate
        if len(seq) > max_length:
            seq = seq[:max_length]
        # pad
        elif len(seq) < max_length:
            padding = torch.full((max_length - len(seq),), pad_value, dtype=torch.long)
            seq = torch.cat([seq, padding])
        sequences.append(seq)
    
    return torch.stack(sequences)
def get_examples(
    dataset_name,
    tokenizer,
    n_samples,
    seq_len=128,
    field_name="text",
    add_bos_to_every=False,
    return_raw_dataset=False,
    seed=42,
):

    dataset = load_from_disk(dataset_name)
    print(f"Dataset loaded: {dataset}")
    dataset=dataset.shuffle(seed=seed)
    data=get_first_n_fixed_length(dataset,n=n_samples, field="input_ids", pad_value=tokenizer.pad_token_id, max_length=seq_len)
    print(f"Data shape: {data.shape}")
   
    return data

#--------------------------------------------------------------------------------------------------------------------------------------------------
    
def merge_layers_return_model(model, low_lay, high_lay, weight_factor):

    if low_lay < 0 or high_lay >= len(model.model.layers):
        raise ValueError("层的索引超出了模型的范围")
        
    model_copy = deepcopy(model)
    
    for current_layer_idx in range(low_lay, high_lay + 1):

        for projection in ['gate_proj', 'down_proj', 'up_proj']:
            model_copy.model.layers[low_lay].mlp.__getattr__(projection).weight.data.add_(
                (model.model.layers[current_layer_idx].mlp.__getattr__(projection).weight.data - model_copy.model.layers[low_lay].mlp.__getattr__(projection).weight.data) * weight_factor
            )

        for projection in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            model_copy.model.layers[low_lay].self_attn.__getattr__(projection).weight.data.add_(
                (model.model.layers[current_layer_idx].self_attn.__getattr__(projection).weight.data - model_copy.model.layers[low_lay].self_attn.__getattr__(projection).weight.data) * weight_factor
            )            

    for current_layer_idx in range(high_lay, low_lay, -1):
        del(model_copy.model.layers[current_layer_idx])

    for layer_idx, module in enumerate(model_copy.model.layers):
        module.self_attn.layer_idx = layer_idx

    return model_copy


def cal_sim(hidden_states, model2, example_prompts, device_map):
 
    print("device", model2.device)
    model_gpu = dispatch_model(deepcopy(model2), device_map=device_map)
    first_device = model_gpu.hf_device_map["model.embed_tokens"]
    sim_ls = []   
    count = 0
    for i in range(0, example_prompts.size(0), args.batch_size):
        example_prompts_tmp = example_prompts[i : i + args.batch_size].to(first_device)
        hidden_states1 = hidden_states[count]
        with torch.no_grad():
       
            outputs2 = model_gpu(example_prompts_tmp, labels=example_prompts_tmp, output_hidden_states=True)       
            hidden_states2 = outputs2.hidden_states[-1]
            hidden_states2 = hidden_states2.detach().to("cpu")
            del outputs2
            torch.cuda.empty_cache()
            
        sim_ls.append(torch.cosine_similarity(hidden_states1.squeeze(0).flatten().unsqueeze(0), hidden_states2.squeeze(0).flatten().unsqueeze(0)))
        count += 1
        
    sim_ls = [i.item() for i in sim_ls]
    print('sim_ls:', np.mean(sim_ls))

    return np.mean(sim_ls)  

def save_merged_model(model_copy, save_path):
    print(f"保存合并后的模型到 {save_path}")
    model_copy.save_pretrained(save_path)
    

#--------------------------------------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="")
parser.add_argument("--model_path", type=str, default='Llama-2-13b-hf')
parser.add_argument("--tokenizer_path", type=str, default='Llama-2-13b-hf')
parser.add_argument("--model_name", type=str, default='Llama-2-13b-hf')
parser.add_argument("--dataset", type=str, default="bookcorpus")
parser.add_argument("--threshold", type=float, default=0.6)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--target_count", type=int, default=2048)
parser.add_argument("--batch_size", type=int, default=1)

args = parser.parse_args()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#--------------------------------------------------------------------------------------------------------------------------------------------------

model_name = args.model_path
short_model = ShortHFModel(model_name=model_name, layers_path="model.layers")
device_map = short_model.device_map
tokenizer = short_model.tokenizer
    
example_prompts = get_examples(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            n_samples=args.target_count,
            seq_len=1024,
            field_name="text",
        ).to("cuda")

short_model = short_model.model.to("cuda")
hidden_states1 = []
print("start getting hidden states...")
for i in tqdm(
    range(0, example_prompts.size(0), args.batch_size),
    desc="Processing batches",
    total=(example_prompts.size(0) + args.batch_size - 1) // args.batch_size,
):
    example_prompts_tmp = example_prompts[i : i + args.batch_size].to("cuda")

    with torch.no_grad():
        outputs1 = short_model(example_prompts_tmp, labels=example_prompts_tmp, output_hidden_states=True)
        hidden_states = outputs1.hidden_states[-1]  # (1, seq_len, hidden)
        hidden_states = hidden_states.to("cpu")  
        hidden_states1.append(hidden_states)
print("done getting hidden states...")  
# #-----------------------------------------------------------------------------------------------------------------------------------------------------
    
high_lay = len(short_model.model.layers) - 1 - 1
low_lay = high_lay - 1
THRESHOLD = args.threshold

count = 0

records = []
print("start!!")
short_model = short_model.to("cpu")
while low_lay >= 0:
    print("low_lay:", low_lay)
    print("running...")
    tmp_merged_model = merge_layers_return_model(short_model, low_lay, high_lay, weight_factor=1)
    print("fibish temp merging")

    sim_value = cal_sim(hidden_states1, tmp_merged_model, example_prompts, device_map=device_map)
    print("fibish sim value")
    del tmp_merged_model
    torch.cuda.empty_cache()


    if sim_value > THRESHOLD:
        print("相似度合格，继续往下合并")
        low_lay -= 1  
    else:
        print("start else")
        print("相似度过低，保存最佳模型")

        if low_lay + 1 != high_lay:
            count += 1
            print("start short merge")
            short_model = merge_layers_return_model(short_model, low_lay + 1, high_lay, weight_factor=1)
            print("fibish short merge")
            record = list(range(low_lay + 1, high_lay + 1))
            print(f'{low_lay+1}至{high_lay}层合并为一层, {record}')
            records.append(record)

            high_lay = low_lay
        else:
            high_lay -= 1
            low_lay -= 1
        print("end else")

    
# #-----------------------------------------------------------------------------------------------------------------------------------------------------

short_model.config.num_hidden_layers = len(short_model.model.layers)
save_merged_model(short_model, f'output/{args.model_name}-SLM{THRESHOLD}')
print(f'SLM finish! Total count: {count}, new model layer length:{len(short_model.model.layers)}, records:{records}')
