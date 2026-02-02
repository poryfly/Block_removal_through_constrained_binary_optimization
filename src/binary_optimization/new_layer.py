from torch.nn import Module, Parameter
from torch import Tensor
import torch
from torch.nn import functional as F

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm, LlamaMLP, LlamaConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RMSNorm, Qwen3Config, Qwen3MLP
from typing import Callable, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_layers import (
    GradientCheckpointingLayer,
)
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack
from transformers.utils.deprecation import deprecate_kwarg
from math import pi


class BlockPruningLlamaDecoderLayer(GradientCheckpointingLayer):
    """
    LLaMA decoder layer with pruning parameters for binary optimization-based layer removal.
    
    This layer extends the standard LLaMA decoder layer by adding a learnable pruning
    parameter that scales the residual connections. The pruning parameter enables
    gradient-based optimization to determine which layers can be safely removed.
    
    Args:
        config: LLaMA model configuration.
        layer_idx: Index of this layer in the model.
        scale: Initial value for the pruning parameter.
        device: Device to place the pruning parameter on.
        dtype: Data type for the pruning parameter.
    """
    def __init__(self, config: LlamaConfig, layer_idx: int,scale,device, dtype):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        initial_values = (scale*torch.ones(1, 1).to(dtype).to(device) 
        )
        self.pruning_param = torch.nn.Parameter(initial_values)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Forward pass with pruning parameter scaling.
        
        The pruning parameter scales both the attention and MLP residual connections,
        allowing the model to learn which layers are important. When the pruning parameter
        approaches zero, the layer's contribution becomes negligible.
        
        Args:
            hidden_states: Input hidden states tensor.
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.
            past_key_values: Optional cached key-value pairs for attention.
            use_cache: Whether to use cached key-value pairs.
            cache_position: Optional cache position tensor.
            position_embeddings: Optional position embeddings.
            **kwargs: Additional keyword arguments.
        
        Returns:
            Output hidden states after applying attention and MLP with pruning scaling.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states*self.pruning_param

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states*self.pruning_param
        
        return hidden_states



class BlockPruningQwen3DecoderLayer(GradientCheckpointingLayer):
    """
    Qwen3 decoder layer with pruning parameters for binary optimization-based layer removal.
    
    This layer extends the standard Qwen3 decoder layer by adding a learnable pruning
    parameter that scales the residual connections. The pruning parameter enables
    gradient-based optimization to determine which layers can be safely removed.
    
    Args:
        config: Qwen3 model configuration.
        layer_idx: Index of this layer in the model.
        scale: Initial value for the pruning parameter.
        device: Device to place the pruning parameter on.
        dtype: Data type for the pruning parameter.
    """
    def __init__(self, config: Qwen3Config, layer_idx: int, scale, device, dtype):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]
        initial_values = (scale*torch.ones(1, 1).to(dtype).to(device) 
        )
        self.pruning_param = torch.nn.Parameter(initial_values)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Forward pass with pruning parameter scaling for Qwen3 decoder layer.
        
        The pruning parameter scales both the attention and MLP residual connections,
        allowing the model to learn which layers are important. When the pruning parameter
        approaches zero, the layer's contribution becomes negligible.
        
        Args:
            hidden_states: Input hidden states tensor.
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.
            past_key_values: Optional cached key-value pairs for attention.
            use_cache: Whether to use cached key-value pairs.
            cache_position: Optional cache position tensor.
            position_embeddings: Optional position embeddings.
            **kwargs: Additional keyword arguments.
        
        Returns:
            Output hidden states after applying attention and MLP with pruning scaling.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
      
        hidden_states = residual + hidden_states*self.pruning_param

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states*self.pruning_param
        return hidden_states