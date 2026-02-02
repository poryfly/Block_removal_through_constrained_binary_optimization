from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
import pickle
import json

from src.compression.compress_model import compress_model


def calculate_norm_ratio(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    dataloader,
    multiblock_size
) -> dict[int, float]:
    """
    Calculates the norm ratio for each block in the model.
    
    The norm ratio is computed as the ratio of output norm to input norm for each block.
    Higher norm ratios indicate blocks that amplify their inputs more, which may be
    more important for the model's functionality.
    
    Args:
        model: The transformer model to analyze.
        tokenizer: Tokenizer for the model (not directly used, kept for API consistency).
        dataloader: DataLoader providing batches for forward passes.
        multiblock_size: Size of multiblock (currently not used, kept for API consistency).
    
    Returns:
        Dictionary mapping layer indices to their average norm ratios.
    """
    norm_ratio_buffer = {}
    for i in range(len(model.model.layers)):
        norm_ratio_buffer[i] = []
    input_cache = list()

    def hook_wrapper(block, idx):
        def hook(_, inp, output):
        
            ratio=(torch.norm(output[0], dim=-1)/torch.norm(inp[0].to(output.device), dim=-1)).mean()
            norm_ratio_buffer[idx].append(float(ratio))
        return hook

    hook_refs = []
    for idx, block in enumerate(model.model.layers):
        hook_ref = block.register_forward_hook(hook_wrapper(block, idx))
        hook_refs.append(hook_ref)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing examples"):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)

            input_cache.clear()

    for hook_ref in hook_refs:
        hook_ref.remove()

    return {k: sum(v) / len(v) for k, v in norm_ratio_buffer.items() } 




def remove_blocks_multiblock(model, tokenizer, dataloader, num_blocks_to_remove, save_model, output_path):
    """
    Removes blocks from the model based on norm ratio scores.
    
    This function calculates norm ratios for all blocks, identifies the blocks with
    the lowest norm ratios, and removes them from the model. Lower norm ratios
    indicate blocks that are less critical and safer to remove.
    
    Args:
        model: The transformer model to compress.
        tokenizer: Tokenizer associated with the model.
        dataloader: DataLoader for computing norm ratios.
        num_blocks_to_remove (int): Number of blocks to remove from the model.
        save_model (bool): Whether to save the compressed model and tokenizer.
        output_path (str): Directory path where the compressed model will be saved.
    
    Returns:
        None: The model is modified in-place. If save_model is True, the model
              and tokenizer are saved to output_path.
    """
    norm_scores = calculate_norm_ratio(
        model, tokenizer, dataloader,num_blocks_to_remove
    )
    sorted_dict = dict(
    sorted(norm_scores.items(), key=lambda item: item[1])
)
    print("sorted norm scores", sorted_dict)
    layers_to_remove = list(sorted_dict.keys())[:num_blocks_to_remove]

    print("removing",layers_to_remove )

    with open(output_path+f"/norm_ratio_{len(layers_to_remove)}.json", "w") as f:
        json.dump(norm_scores, f)
    if save_model:
        model=compress_model(model, layers_to_remove)
        model.save_pretrained(output_path+f"/norm_ratio_{num_blocks_to_remove}_removed")
        tokenizer.save_pretrained(output_path+f"/norm_ratio_{num_blocks_to_remove}_removed")
    return 
