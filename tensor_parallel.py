from torch.distributed.tensor.parallel import (
    parallelize_module,

)

from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, PrepareModuleInput
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard, Placement
from copy import deepcopy
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase
import torch.distributed._functional_collectives as funcol

NUM_DEVICES = 4
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    NUM_DEVICES = min(NUM_DEVICES, torch.cuda.device_count())


def _check_module(m1, m2, check_grad=False):
        testcase = TestCase()
        named_parameters = dict(m1.named_parameters())
        for name, param_m2 in m2.named_parameters():
            testcase.assertTrue(name in named_parameters)
            param_m1 = named_parameters[name]
            if check_grad:
                param_m2 = param_m2.grad
                param_m1 = param_m1.grad
            if isinstance(param_m2, DTensor):
                replicate = [Replicate()]
                param_m2 = param_m2.redistribute(
                    device_mesh=param_m2.device_mesh, placements=replicate
                ).to_local()
            testcase.assertEqual(param_m2, param_m1)

        x = 1

def get_tensor_sharded_model(model, use_ddp):

    if use_ddp:
        model = model.module.model 
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    device_mesh = DeviceMesh(device, torch.arange(0, NUM_DEVICES))

    # Parallelize the embedding submodules.
    
    parallelize_module(model.model.embed_tokens, device_mesh, ColwiseParallel(output_layouts=Replicate()))

    # Parallelize the attention and feed forward submodules.
    for layer in model.model.layers:
        
        layer_parallelize_plan = {} 
        layer_parallelize_plan["self_attn.q_proj"] = ColwiseParallel()
        layer_parallelize_plan["self_attn.k_proj"] = ColwiseParallel()
        layer_parallelize_plan["self_attn.v_proj"] = ColwiseParallel()
        layer_parallelize_plan["self_attn.o_proj"] = RowwiseParallel()

        layer_parallelize_plan["mlp.gate_proj"] = ColwiseParallel()
        layer_parallelize_plan["mlp.up_proj"] = ColwiseParallel()
        layer_parallelize_plan["mlp.down_proj"] = RowwiseParallel()

        parallelize_module(layer, device_mesh, layer_parallelize_plan)

    # Parallelize the output submodule. 
    output_parallelize_plan = RowwiseParallel(input_layouts=Replicate())
    parallelize_module(model.lm_head, device_mesh, output_parallelize_plan)

    # Manually adjust the number of heads after sharding the self attention modules.
    for layer in model.model.layers:
        assert model.model.config.num_attention_heads % dist.get_world_size() == 0
        # For llama2 models, your should adjust the number of heads separately.
        layer.self_attn.num_heads = model.model.config.num_attention_heads // dist.get_world_size()
        layer.self_attn.num_key_value_heads = model.model.config.num_key_value_heads // dist.get_world_size()
        layer.self_attn.hidden_size = model.model.config.hidden_size // dist.get_world_size()
        
    """# Manually register all_reduce hooks for all norm layers as they only process sharded inputs.
    def all_reduce_fn(grad):
        return funcol.all_reduce(grad, reduceOp="SUM", group= device_mesh)
    for layer in model.model.layers:
        # Before Attention
        # hidden_states = attention(input_layernorm(hidden_states))
        layer.input_layernorm.weight.register_hook(all_reduce_fn)

        # Before MLP
        # hidden_states = mlp(post_attention_layernorm(hidden_states))
        layer.post_attention_layernorm.weight.register_hook(all_reduce_fn)
    
    # After all the decode layers
    model.model.norm.weight.register_hook(all_reduce_fn)"""

    # Manually set output.weight so that parameters and gradients are shared.
    # For LlamaForCausalLM model, output is model.lm_head: nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    # model.lm_head.weight = model.model.embed_tokens.weight