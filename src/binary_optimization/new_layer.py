from torch.nn import Module, Parameter
from torch import Tensor
import torch
from torch.nn import functional as F

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm, LlamaMLP, LlamaConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RMSNorm, Qwen3Config, Qwen3MLP
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Attention, DeepseekV4SparseMoeBlock, DeepseekV4RMSNorm, DeepseekV4HyperConnection, DeepseekV4Config
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


class BlockPruningDeepseekV4DecoderLayer(GradientCheckpointingLayer):
    """
    DeepSeek-V4 decoder layer with pruning parameters using full interpolation.

    Uses HyperConnection's multi-stream architecture with interpolation:
    output = alpha * layer_output + (1 - alpha) * hidden_states
    When alpha -> 0, the layer becomes an identity mapping, enabling clean block removal.

    IMPORTANT: For FP8-quantized models, this class reuses sub-modules from the original
    layer by reference (not copy). This preserves FP8 quantization and GPU placement,
    avoiding the 4x memory expansion that would occur if new unquantized sub-modules
    were created from scratch.

    Args:
        original_layer: The original DeepseekV4DecoderLayer to reuse sub-modules from.
        layer_idx: Index of this layer in the model.
        scale: Initial value for the pruning parameter.
        device: Device to place the pruning parameter on.
        dtype: Data type for the pruning parameter.
    """
    def __init__(self, original_layer, layer_idx: int, scale, device, dtype):
        super().__init__()
        self.hidden_size = original_layer.self_attn.config.hidden_size

        # Reuse sub-modules from original layer by reference (preserves FP8 quantization & GPU placement)
        # Creating new sub-modules from scratch would produce float32 unquantized weights,
        # which are 4x larger than FP8 and cause OOM when dispatch_model moves them to GPU.
        self.self_attn = original_layer.self_attn
        self.mlp = original_layer.mlp
        self.input_layernorm = original_layer.input_layernorm
        self.post_attention_layernorm = original_layer.post_attention_layernorm
        self.attn_hc = original_layer.attn_hc
        self.ffn_hc = original_layer.ffn_hc
        # FP8 dtypes don't support mul/backward needed for pruning_param
        # Force bfloat16 if an FP8 dtype was accidentally passed
        if dtype in (torch.float8_e4m3fn, torch.float8_e5m2,
                     torch.float8_e4m3fnuz, torch.float8_e5m2fnuz):
            dtype = torch.bfloat16
        initial_values = (scale*torch.ones(1, 1).to(dtype).to(device))
        self.pruning_param = torch.nn.Parameter(initial_values)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[dict] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Forward pass with full interpolation pruning.

        The pruning parameter interpolates between identity (alpha=0) and the
        normal layer output (alpha=1). This naturally smooths both the sublayer
        contribution and the stream mixing (effective comb = alpha*comb.T + (1-alpha)*I).

        Args:
            hidden_states: Input hidden states tensor [B, S, hc_mult, H].
            input_ids: Token IDs needed for hash_moe routing (Layer 0-2).
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.
            past_key_values: Optional cached key-value pairs.
            use_cache: Whether to use cached key-value pairs.
            cache_position: Optional cache position tensor.
            position_embeddings: Dict with "main" and "compress" RoPE embeddings.
            **kwargs: Additional keyword arguments.

        Returns:
            Output hidden states after interpolation pruning.
        """
        dtype = hidden_states.dtype

        # Step 1: Attention sublayer with full interpolation
        residual_attn = hidden_states
        post_attn, comb_attn, collapsed_attn = self.attn_hc(hidden_states)
        attn_output, _ = self.self_attn(
            self.input_layernorm(collapsed_attn),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        attn_layer_out = post_attn.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb_attn.to(dtype).transpose(-1, -2), hidden_states
        )
        hidden_states = self.pruning_param * attn_layer_out + (1 - self.pruning_param) * residual_attn

        # Step 2: MLP sublayer with full interpolation
        residual_mlp = hidden_states
        post_mlp, comb_mlp, collapsed_mlp = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(
            self.post_attention_layernorm(collapsed_mlp),
            input_ids=input_ids,
        )
        mlp_layer_out = post_mlp.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
            comb_mlp.to(dtype).transpose(-1, -2), hidden_states
        )
        return self.pruning_param * mlp_layer_out + (1 - self.pruning_param) * residual_mlp