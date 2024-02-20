import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

def replace_layers(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LlamaMLP):
            module.gate_proj = nn.LazyLinear(module.intermediate_size, bias=False)
            module.up_proj = nn.LazyLinear(module.intermediate_size, bias=False)
            module.down_proj = nn.LazyLinear(module.hidden_size, bias=False)
        elif isinstance(module, LlamaAttention):
            module.q_proj = nn.LazyLinear(module.num_heads * module.head_dim, bias=False)
            module.k_proj = nn.LazyLinear(module.num_key_value_heads * module.head_dim, bias=False)
            module.v_proj = nn.LazyLinear(module.num_key_value_heads * module.head_dim, bias=False)
            module.o_proj = nn.LazyLinear(module.hidden_size, bias=False)