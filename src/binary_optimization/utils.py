from transformers.models.llama.modeling_llama import LlamaMLP, LlamaDecoderLayer

import torch
from src.binary_optimization.new_layer import (
     BlockPruningLlamaDecoderLayer, BlockPruningQwen3DecoderLayer, BlockPruningDeepseekV4DecoderLayer
)
import json

from math import pi
from transformers import AutoTokenizer, AutoModelForCausalLM


def prepare_pruning(
    model, scale=1.0
):
    """
    Prepares a model for pruning by replacing standard decoder layers with pruning-enabled layers.
    
    This function iterates through all layers in the model and replaces them with custom
    decoder layers that include pruning parameters. The pruning parameters allow for
    binary optimization-based layer removal.
    
    Args:
        model: The transformer model to prepare for pruning (must have model.layers attribute).
        scale (float): Initial scale value for pruning parameters. Defaults to 1.0.
    
    Returns:
        None: The model is modified in-place.
    """
    for i in range(len(model.model.layers)):
        new_layer=prepare_block(
                    model.model.layers[i],
                    config=model.config,
                    idx=i,
                    scale=scale,
                    layer_type=model.config.model_type,
                )
        model.model.layers[i] = new_layer

    return


def copy_matching_weights(new_decoder_layer, layer):
    """
    Copies matching weights from an original layer to a new decoder layer.
    
    This function transfers weights from the original layer to the new pruning-enabled
    layer, preserving all compatible parameters (same name and shape).
    
    Args:
        new_decoder_layer: The new decoder layer to copy weights into.
        layer: The original layer to copy weights from.
    
    Returns:
        None: The new_decoder_layer is modified in-place.
    """
    src_state = layer.state_dict()
    dst_state = new_decoder_layer.state_dict()

    for name, param in src_state.items():
        if name in dst_state and dst_state[name].shape == param.shape:
            dst_state[name].copy_(param)

    new_decoder_layer.load_state_dict(dst_state)


def prepare_block(
    layer,  config, idx,scale=1.0, layer_type="llama"
):
    """
    Creates a pruning-enabled decoder layer to replace a standard decoder layer.
    
    This function instantiates a custom decoder layer (BlockPruningLlamaDecoderLayer or
    BlockPruningQwen3DecoderLayer) with pruning parameters and copies weights from
    the original layer.
    
    Args:
        layer: The original decoder layer to be replaced.
        config: Model configuration object.
        idx (int): Index of the layer in the model.
        scale (float): Initial scale value for pruning parameters. Defaults to 1.0.
        layer_type (str): Type of model architecture ("llama" or "qwen3"). Defaults to "llama".
    
    Returns:
        A new pruning-enabled decoder layer with copied weights from the original layer.
    
    Raises:
        SystemExit: If an unsupported layer_type is provided.
    """

    if layer_type == "deepseek_v4":
        device=layer.self_attn.kv_proj.weight.device
        # FP8 weights can't be used for pruning_param (no mul/backward support)
        # Use the model's configured torch_dtype (bfloat16) instead
        dtype=torch.bfloat16
    else:
        device=layer.self_attn.q_proj.weight.device
        dtype=layer.self_attn.q_proj.weight.dtype
    if layer_type == "deepseek_v4":
        # For FP8 models: reuse original sub-modules by reference (preserves FP8 quantization)
        # No copy_matching_weights needed - sub-modules are shared, not copied
        new_decoder_layer = BlockPruningDeepseekV4DecoderLayer(
            original_layer=layer, layer_idx=idx, device=device, dtype=dtype, scale=scale
        )
    elif layer_type == "qwen3":
        new_decoder_layer = BlockPruningQwen3DecoderLayer(config, idx, device=device, dtype=dtype, scale=scale)
    elif layer_type == "llama":
        new_decoder_layer = BlockPruningLlamaDecoderLayer(config, idx, device=device, dtype=dtype, scale=scale)
    else:
        print("error no proper architecture found")
        exit()
    # copy_matching_weights only needed for llama/qwen3 (new sub-modules created from scratch)
    # For deepseek_v4, sub-modules are reused by reference, no weight copying needed
    if layer_type != "deepseek_v4":
        copy_matching_weights(new_decoder_layer, layer)
    return new_decoder_layer
