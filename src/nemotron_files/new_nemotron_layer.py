from torch import nn
from typing import Optional
from src.nemotron_files.modeling_nemotron_h import NemotronHRMSNorm, NemotronHMamba2Mixer,NEMOTRONH_ATTENTION_CLASSES,NemotronHMLP,NemotronHMOE,HybridMambaAttentionDynamicCache
import torch
class NEWNemotronHBlock(nn.Module):
    def __init__(self, config, layer_idx, scale, dtype, device):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # M: Mamba2, *: Attention, -: MLP
        self.block_type = config.layers_block_type[layer_idx]
      
        if self.block_type == "mamba":
            self.mixer = NemotronHMamba2Mixer(config, layer_idx=layer_idx)
        elif self.block_type == "attention":
            self.mixer = NEMOTRONH_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)
        elif self.block_type == "mlp":
            self.mixer = NemotronHMLP(config, layer_idx=layer_idx)
        elif self.block_type == "moe":
            self.mixer = NemotronHMOE(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Invalid layer pattern {config.hybrid_override_pattern[layer_idx]}")
        initial_values = (scale*torch.ones(1, 1).to(dtype).to(device) 
        )
        self.pruning_param = torch.nn.Parameter(initial_values)
    def forward(
        self,
        hidden_states,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        with torch.cuda.stream(torch.cuda.default_stream(hidden_states.device)):
            # * Use torch.cuda.stream() to avoid NaN issues when using multiple GPUs
            residual = hidden_states
            hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

            if self.block_type == "mamba":
                hidden_states = self.mixer(
                    hidden_states, cache_params=cache_params, cache_position=cache_position
                )
            elif self.block_type == "attention":
                hidden_states = self.mixer(
                    hidden_states, cache_position=cache_position
                )
                hidden_states = hidden_states[0]
            elif self.block_type in ["mlp", "moe"]:
                hidden_states = self.mixer(
                    hidden_states
                )
            else:
                raise ValueError(f"Invalid block_type: {self.block_type}")

            hidden_states = residual + hidden_states*self.pruning_param
            return hidden_states



