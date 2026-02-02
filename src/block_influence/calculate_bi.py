from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
import pickle
import json




def calculate_multiblock_influence(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    dataloader,
    multiblock_size
) -> list[float]:
    """Calculate a multiblock influence score (BI) -- a cosine distance between the inputs and the outputs of a multiblock.
    A multiblock is a sequence of N blocks. The lower the BI is, the less sensitive the multiblock is, and therefore it
    is safer to remove it.

    Args:
        model: The model to be measured.
        tokenizer: Appropriate tokenizer.
        dataset: A dataset used to measure the results.
        max_length: Maximum length of the tokenized sequence.
        multiblock_size: How long is the multiblock.

    Returns: List of multiblock influence scores.

    """
    cosine_similarity_buffer = defaultdict(lambda: [])
    input_cache = list()

    def hook_wrapper(block):
        def hook(_, inp, output):
        
            input_repr = inp[0]
            input_cache.append(input_repr.detach())



            if len(input_cache) >= multiblock_size:
                cosine_similarity = F.cosine_similarity(input_cache[-multiblock_size].to(output.device), output, dim=-1)
                
                cosine_similarity_buffer[block].append(float(torch.mean(cosine_similarity)))

        return hook

    hook_refs = []
    for block in model.model.layers:
        hook_ref = block.register_forward_hook(hook_wrapper(block))
        hook_refs.append(hook_ref)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing examples"):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)

            input_cache.clear()

    for hook_ref in hook_refs:
        hook_ref.remove()

    return [1 - np.mean(cosine_similarity_buffer[block]) for block in model.model.layers]



def calculate_block_influence(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    dataloader,
) -> list[float]:
    """Calculate a block influence score (BI) -- a cosine distance between the inputs and the outputs of a block. The
    lower the BI is, the less sensitive the block is, and therefore it is safer to remove it.

    Args:
        model: The model to be measured.
        tokenizer: Appropriate tokenizer.
        dataset: A dataset used to measure the results.
        max_length: Maximum length of the tokenized sequence.

    Returns: List of block influence scores.

    """
    return calculate_multiblock_influence(model, tokenizer, dataloader, multiblock_size=1)


def remove_blocks_one_by_one(
    model, tokenizer, dataloader,num_blocks_to_remove, save_model, output_path
):
    """Remove `num_blocks_to_remove` blocks from the model according to their BI rates. The BI rates are re-calculated
    after each removal step.

    Args:
        model: Model to be compressed by block removal.
        tokenizer: Appropriate tokenizer.
        dataset: Dataset used to calculate the block influence scores.
        max_length: Maximum length of the tokenized sequence.
        batch_size: Batch size for training and validation.
        num_blocks_to_remove: Number of layers to be removed from the model

    Returns: A pruned model

    """
    layers_to_remove=[]
    
    for i in range(num_blocks_to_remove):
        bi_scores = calculate_block_influence(model, tokenizer, dataloader)
        minimum_bi_block = bi_scores.index(min(bi_scores))
        del model.model.layers[minimum_bi_block]
        
        layers_to_remove.append(minimum_bi_block)
        print("removed layer", minimum_bi_block)
 
 
    with open(output_path+f"/removing_one_by_one_{len(layers_to_remove)}.json", "w") as f:
        json.dump(layers_to_remove, f)
    
    if save_model:
        model.config.num_hidden_layers = len(model.model.layers)
        if model.config.model_type == "qwen3" or model.config.model_type == "gpt_oss":
            list_of_layers=model.config.layer_types
            for layer in layers_to_remove:
                del list_of_layers[layer]
            model.config.layer_types=list_of_layers
        
        model.save_pretrained(output_path+f"/bi_one_by_one_{len(layers_to_remove)}_removed")
        tokenizer.save_pretrained(output_path+f"/bi_one_by_one_{len(layers_to_remove)}_removed")
    return 


def remove_blocks_multiblock(model, tokenizer, dataloader,num_blocks_to_remove, save_model, output_path):
    """Remove a multiblock with `num_blocks_to_remove` length with the minimum BI rate.

    Args:
        model: Model to be compressed by block removal.
        tokenizer: Appropriate tokenizer.
        dataset: Dataset used to calculate the block influence scores.
        max_length: Maximum length of the tokenized sequence.
        num_blocks_to_remove: Number of layers to be removed from the model

    Returns: A compressed model
    """
    bi_scores = calculate_multiblock_influence(
        model, tokenizer, dataloader,num_blocks_to_remove
    )
    
    earliest_block_index = bi_scores.index(np.nanmin(bi_scores)) - num_blocks_to_remove + 1
    layers_to_remove=list(range(earliest_block_index,earliest_block_index + num_blocks_to_remove))
    print("removing",layers_to_remove )

    with open(output_path+f"/removing_consecutive_{len(layers_to_remove)}.json", "w") as f:
        json.dump(layers_to_remove, f)
    if save_model:
        del model.model.layers[earliest_block_index : earliest_block_index + num_blocks_to_remove]
        if model.config.model_type == "qwen3" or model.config.model_type == "gpt_oss":
            list_of_layers=model.config.layer_types
            del list_of_layers[earliest_block_index : earliest_block_index + num_blocks_to_remove]
            model.config.layer_types=list_of_layers
        model.config.num_hidden_layers = len(model.model.layers)
        model.save_pretrained(output_path+f"/bi_consecutive_{num_blocks_to_remove}_removed")
        tokenizer.save_pretrained(output_path+f"/bi_consecutive_{num_blocks_to_remove}_removed")
    return 
