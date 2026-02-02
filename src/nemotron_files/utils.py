from src.nemotron_files.new_nemotron_layer import NEWNemotronHBlock
import torch

def prepare_pruning_nemotron(
    model, scale=1.0
):
    """
    Class to keep track of the hidden dimensions
    """
    for i in range(len(model.backbone.layers)):
        print("block  number", i)
        #print(model.backbone.layers[i])
        new_layer=prepare_block_nemotron(
                    model.backbone.layers[i],
                    config=model.config,
                    idx=i,
                    scale=scale,
                )
        model.backbone.layers[i] = new_layer

    return


def copy_matching_weights(new_decoder_layer, layer):
    src_state = layer.state_dict()
    dst_state = new_decoder_layer.state_dict()

    for name, param in src_state.items():
        if name in dst_state and dst_state[name].shape == param.shape:
            dst_state[name].copy_(param)

    new_decoder_layer.load_state_dict(dst_state)

def prepare_block_nemotron(
    layer,  config, idx,scale=1.0
):

    device=layer.norm.weight.device
    dtype=layer.norm.weight.dtype
    new_decoder_layer = NEWNemotronHBlock(config=config,
                 layer_idx=idx,
                 scale=scale,
                 dtype=dtype,
                 device=device)
    copy_matching_weights(new_decoder_layer, layer)
  
    return new_decoder_layer
