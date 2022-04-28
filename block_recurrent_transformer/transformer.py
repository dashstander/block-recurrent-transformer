from collections import namedtuple
from einops import rearrange
import torch
from torch import nn
from torchtyping import TensorType
from typing import Optional
from x_transformers.x_transformers import (
    Attention, default, FeedForward, RotaryEmbedding, AlibiPositionalBias
)


"""
This may change significantly as I work out how to implement this properly, but until then this is largely copied from Phil Wang (@lucidrains)
"""

# constants

DEFAULT_DIM_HEAD = 64

Intermediates = namedtuple('Intermediates', [
    'pre_softmax_attn',
    'post_softmax_attn'
])

LayerIntermediates = namedtuple('Intermediates', [
    'hiddens',
    'attn_intermediates'
])


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(t, freqs):
    seq_len = t.shape[-2]
    freqs = freqs[-seq_len:, :]
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())




class BRAGate(nn.Module):
    """Poor man's LSTM
    """

    def __init__(self, dim):
        super().__init__()

        self.main_proj = nn.Linear(dim, dim, bias = True)
        self.input_proj = nn.Linear(dim, dim, bias = True)
        self.forget_proj = nn.Linear(dim, dim, bias = True)
    
    def forward(self, x, state):
        z = torch.tanh(self.main_proj(x))
        i = torch.sigmoid(self.input_proj(x) - 1)
        f = torch.sigmoid(self.forget_proj(x) + 1)
        return torch.mul(state, f) + torch.mul(z, i)



class BlockRecurrentAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_state,
        dim_head = DEFAULT_DIM_HEAD,
        heads = 8,
        head_scale = False,
        qk_norm = False,
        scale_init_value = None,
        **kwargs
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        attn_kwargs = {}

        self.heads = heads
        self.causal = True
        
        self.input_self_attn = Attention(dim, heads = heads, causal = True, **attn_kwargs)
        self.state_self_attn = Attention(dim_state, heads = heads, causal = False, **attn_kwargs)

        self.input_state_cross_attn = Attention(dim, heads = heads, causal = False, **attn_kwargs)
        self.state_input_cross_attn = Attention(dim_state, heads = heads, causal = False, **attn_kwargs)

        self.proj_gate = BRAGate(dim)
        self.ff_gate = BRAGate(dim)

        self.input_proj = nn.Linear(dim + dim_state, dim, bias = False)
        self.state_proj = nn.Linear(dim + dim_state, dim, bias = False)

        self.input_ff = FeedForward(dim)
        self.state_ff = FeedForward(dim_state)


        # cosine sim attention
        self.qk_norm = qk_norm
        if qk_norm:
            scale_init_value = default(scale_init_value, -3) # if not provided, initialize as though it were sequence length of 1024
            self.scale = nn.Parameter(torch.ones(1, heads, 1, 1) * scale_init_value)

        # head scaling
        self.head_scale = head_scale
        if head_scale:
            self.head_scale_params = nn.Parameter(torch.ones(1, heads, 1, 1))

    def forward(
        self,
        x: TensorType['batch', -1, 'token_dim'],
        state: Optional[TensorType['state_size', -1, 'state_dim']] = None,
        mask = None,
        state_mask = None,
        rel_pos = None,
        rotary_pos_emb = None,
        prev_attn = None,
        mem = None
    ):
        input_attn, _ = self.input_self_attn(x, mask = mask)
        state_attn, _ = self.state_self_attn(state, mask = state_mask)

        # This actually is different from how it is implemented in the paper, because the Keys and Values aren't shared
        # between the cross attention and self-attention. I'll implement that later, this is faster for now.
        input_as_q_cross_attn, _ = self.input_state_cross_attn(x, context = state, mask = mask, context_mask = state_mask)
        state_as_q_cross_attn, _ = self.state_input_cross_attn(state, context = x, mask = state_mask, context_mask = mask)

        projected_input = self.input_proj(torch.concat((input_as_q_cross_attn, input_attn), dim=2))
        projected_state = self.state_proj(torch.concat((state_as_q_cross_attn, state_attn), dim=2))

        input_residual = projected_input + x
        state_residual = self.proj_gate(projected_state, state)

        output = self.input_ff(input_residual) + input_residual
        next_state = self.ff_gate(self.state_ff(state_residual), state_residual)

        return output, next_state
