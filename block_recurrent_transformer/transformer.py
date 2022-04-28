from collections import namedtuple
from einops import rearrange, repeat
import torch
import torch.nn.functional as F
from torch import einsum, nn
from torchtyping import TensorType
from typing import Optional
from x_transformers.x_transformers import (
    Attention, exists, default, FeedForward, max_neg_value, l2norm, init_zero_, RotaryEmbedding, AlibiPositionalBias
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

        self.proj_gate = nn.LSTMCell(dim, dim, bias = True)
        self.ff_gate = nn.LSTMCell(dim, dim, bias = True)

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
        attn_mask = None,
        rel_pos = None,
        rotary_pos_emb = None,
        prev_attn = None,
        mem = None
    ):
        input_attn = self.input_self_attn(x, mask = mask)
        state_attn = self.state_self_attn(state, mask = state_mask)

        # This actually is different from how it is implemented in the paper, because the Keys and Values aren't shared
        # between the cross attention and self-attention. I'll implement that later, this is faster for now.
        input_as_q_cross_attn = self.input_state_cross_attn(x, context = state, mask = mask, context_mask = state_mask)
        state_as_q_cross_attn = self.state_input_cross_attn(state, context = x, mask = state_mask, context_mask = mask)

        projected_input = self.input_proj(torch.concat((input_as_q_cross_attn, input_attn), dim=2))
        projected_state = self.state_proj(torch.concat((state_as_q_cross_attn, state_attn), dim=2))

        input_residual = projected_input + x
        state_residual = self.proj_gate(state + projected_state)

        output = self.input_ff(input_residual) + input_residual
        next_state = self.ff_gate()



        return output









    """
        b, n, _, h, head_scale, scale, device, has_state = (
            *x.shape,
            self.heads,
            self.head_scale,
            self.scale,
            x.device,
            exists(state)
        )
        kv_input = default(state, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input

        if exists(mem):
            k_input = torch.cat((mem, k_input), dim = -2)
            v_input = torch.cat((mem, v_input), dim = -2)

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        if exists(rotary_pos_emb) and not has_state:
            l = rotary_pos_emb.shape[-1]
            (ql, qr), (kl, kr), (vl, vr) = map(lambda t: (t[..., :l], t[..., l:]), (q, k, v))
            ql, kl, vl = map(lambda t: apply_rotary_pos_emb(t, rotary_pos_emb), (ql, kl, vl))
            q, k, v = map(lambda t: torch.cat(t, dim = -1), ((ql, qr), (kl, kr), (vl, vr)))

        input_mask = None
        if any(map(exists, (mask, state_mask))):
            q_mask = default(mask, lambda: torch.ones((b, n), device = device).bool())
            k_mask = q_mask if not exists(state) else state_mask
            k_mask = default(k_mask, lambda: torch.ones((b, k.shape[-2]), device = device).bool())
            q_mask = rearrange(q_mask, 'b i -> b 1 i 1')
            k_mask = rearrange(k_mask, 'b j -> b 1 1 j')
            input_mask = q_mask * k_mask

        if self.qk_norm:
            q, k = map(l2norm, (q, k))
            scale = 1 / (self.scale.exp().clamp(min = 1e-2))

        kv_einsum_eq = 'b h j d' if not self.one_kv_head else 'b j d'

        dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale
        mask_value = max_neg_value(dots)

        if exists(prev_attn):
            dots = dots + prev_attn

        pre_softmax_attn = dots.clone()

        if exists(rel_pos):
            dots = rel_pos(dots)

        if exists(input_mask):
            dots.masked_fill_(~input_mask, mask_value)
            del input_mask

        if exists(attn_mask):
            assert 2 <= attn_mask.ndim <= 4, 'attention mask must have greater than 2 dimensions but less than or equal to 4'
            if attn_mask.ndim == 2:
                attn_mask = rearrange(attn_mask, 'i j -> 1 1 i j')
            elif attn_mask.ndim == 3:
                attn_mask = rearrange(attn_mask, 'h i j -> 1 h i j')
            dots.masked_fill_(~attn_mask, mask_value)

        if exists(self.max_attend_past):
            i, j = dots.shape[-2:]
            range_q = torch.arange(j - i, j, device = device)
            range_k = torch.arange(j, device = device)
            dist = rearrange(range_q, 'i -> 1 1 i 1') - rearrange(range_k, 'j -> 1 1 1 j')
            mask = dist > self.max_attend_past
            dots.masked_fill_(mask, mask_value)
            del mask


        i, j = dots.shape[-2:]
        r = torch.arange(i, device = device)
        mask = rearrange(r, 'i -> 1 1 i 1') < rearrange(r, 'j -> 1 1 1 j')
        mask = F.pad(mask, (j - i, 0), value = False)
        dots.masked_fill_(mask, mask_value)
        del mask

        attn = self.attn_fn(dots, dim = -1)
        post_softmax_attn = attn.clone()

        attn = self.dropout(attn)

        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        if head_scale:
            out = out * self.head_scale_params

        out = rearrange(out, 'b h n d -> b n (h d)')

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            out = out * gates.sigmoid()

        intermediates = Intermediates(
            pre_softmax_attn = pre_softmax_attn,
            post_softmax_attn = post_softmax_attn
        )

        return self.to_out(out), intermediates
    """
