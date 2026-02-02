import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from copy import deepcopy
import copy
import argparse
import os, glob, heapq
from src.solve_binary_optimization.util_low_energy_spectrum import iter_global_lowest_n_from_sorted_batches, _torch_load_maybe_mmap 


def compress_model(model, layers_to_remove):
    """
    Removes specified layers from a model to create a compressed version.
    
    This function physically removes layers from the model architecture based on
    the provided indices. It also updates the model configuration to reflect the
    new number of layers.
    
    Args:
        model: The transformer model to compress (must have model.model.layers attribute).
        layers_to_remove (list): List of layer indices to remove from the model.
    
    Returns:
        The compressed model with specified layers removed.
    
    Note:
        Layers are removed in reverse order to maintain correct indices during deletion.
        For models with layer_types (qwen3, gpt_oss), the layer_types list is also updated.
    """


    s = "_".join(map(str, layers_to_remove))

    for idx in sorted(layers_to_remove, reverse=True):
        print("deleting layer", idx)
        del model.model.layers[idx]
    print("new length of model", len(model.model.layers))
    model.config.num_hidden_layers = len(model.model.layers)
    if model.config.model_type == "qwen3" or model.config.model_type == "gpt_oss":
        list_of_layers=model.config.layer_types
        for idx in sorted(layers_to_remove, reverse=True):
            del list_of_layers[idx]
        model.config.layer_types=list_of_layers
        print(len(model.config.layer_types))

    return model
def main():
    """
    Main function to generate compressed models from energy computation results.
    
    This function loads the top-k lowest energy states from computed energy files,
    extracts the corresponding layer removal configurations, and generates compressed
    models by removing those layers. Each compressed model is saved with its tokenizer.
    
    Command-line arguments:
        --filename: Directory containing energy computation results (results_*.pt files).
        --model_name: Path to the original pretrained model.
        --k: Number of top-k lowest energy models to generate.
    
    Output:
        Saves k compressed models in the filename directory, each named as:
        compressed_model_state_{i}_remove_{layer_indices}/
    """
    parser = argparse.ArgumentParser(description="Pruning script for Llama models.")
    parser.add_argument(
        "--filename", type=str, help="Path to the config file"
    )
    parser.add_argument(
        "--model_name", type=str, help="Path to the config file"
    )
    parser.add_argument(
        "--k", type=str, help="Path to the config file"
    )    
    args = parser.parse_args()

    filename = args.filename
    batch_glob="results_*b=*_*.pt"
    batch_files = sorted(glob.glob(os.path.join(filename, batch_glob)))
    print(batch_files)
    lowest_refs = list(iter_global_lowest_n_from_sorted_batches(batch_files, int(args.k)))
    energies = []
    vecs = []
    indices = []

    for e, fpath, j, idx in lowest_refs:
        d = _torch_load_maybe_mmap(fpath, map_location="cpu")
        energies.append(float(e))
        vecs.append(d["vecs"][j])
        indices.append(idx)
    for i in range(int(args.k)):
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        layers_to_remove = torch.where(vecs[i] == 1)[0].tolist()
        print(energies[i], layers_to_remove)
        s = "_".join(map(str, layers_to_remove))
        model = compress_model(model, layers_to_remove)
        model.save_pretrained(f"{filename}/compressed_model_state_{i}_remove_{s}")
        tokenizer.save_pretrained(f"{filename}/compressed_model_state_{i}_remove_{s}")
        del model
        del tokenizer


if __name__ == "__main__":
    main()