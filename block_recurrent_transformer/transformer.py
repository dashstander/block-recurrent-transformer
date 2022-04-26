from collections import namedtuple
from einops import rearrange, repeat
import torch
import torch.nn.functional as F
from torch import einsum, nn
from x_transformers.x_transformers import (
    Attention, CrossAttender, Decoder, exists, default, max_neg_value, l2norm, init_zero_
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


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, device):
        t = torch.arange(max_seq_len, device = device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


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
        dropout = 0.,
        gate_values = False,
        zero_init_output = False,
        qk_norm = False,
        scale_init_value = None,
        value_dim_head = None,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        attn_kwargs = {}

        self.heads = heads
        self.causal = True

        value_dim_head = default(value_dim_head, dim_head)
        q_dim = k_dim = dim_head * heads
        v_dim = out_dim = value_dim_head * heads

        q_state_dim = k_state_dim = v_state_dim = dim_state * heads

        self.to_q_state = nn.Linear(dim_state, q_state_dim, bias = False)
        self.to_v_state = nn.Linear(dim_state, v_state_dim, bias = False)
        self.to_k_state = nn.Linear(dim_state, k_state_dim, bias = False)

        self.to_q = nn.Linear(dim, q_dim, bias = False)
        self.to_k = nn.Linear(dim, k_dim, bias = False)
        self.to_v = nn.Linear(dim, v_dim, bias = False)

        self.input_self_attn = Attention(dim, heads = heads, causal = True, **attn_kwargs)
        self.state_seff_attn = Attention(dim_state, heads = heads, causal = False, **attn_kwargs)

        self.input_cross_attn = Attention(dim, heads = heads, causal = False, **attn_kwargs)
        self.state_cross_attn = Attention(dim_state, heads = heads, causal = False, **attn_kwargs)


        self.dropout = nn.Dropout(dropout)

        # add GLU gating for aggregated values, from alphafold2
        self.to_v_gate = None
        if gate_values:
            self.to_v_gate = nn.Linear(dim, out_dim)
            nn.init.constant_(self.to_v_gate.weight, 0)
            nn.init.constant_(self.to_v_gate.bias, 1)

        # cosine sim attention
        self.qk_norm = qk_norm
        if qk_norm:
            scale_init_value = default(scale_init_value, -3) # if not provided, initialize as though it were sequence length of 1024
            self.scale = nn.Parameter(torch.ones(1, heads, 1, 1) * scale_init_value)

        # head scaling
        self.head_scale = head_scale
        if head_scale:
            self.head_scale_params = nn.Parameter(torch.ones(1, heads, 1, 1))

        # init output projection 0
        if zero_init_output:
            init_zero_(self.to_out)

    def forward(
        self,
        x,
        state = None,
        mask = None,
        state_mask = None,
        attn_mask = None,
        rel_pos = None,
        rotary_pos_emb = None,
        prev_attn = None,
        mem = None
    ):
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

        if self.causal:
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
