from block_recurrent_transformer.transformer import BlockRecurrentAttention
import torch
from torchtyping import patch_typeguard
from typeguard import typechecked


patch_typeguard()


@typechecked
def test_block_recurrent_attention():
    attn = BlockRecurrentAttention(128, 128, state_len=128)
    x = torch.randn((64, 128, 128))
    state = torch.randn((64, 128, 128))

    out, new_state = attn(x, state)
    assert out.shape == new_state.shape


@typechecked
def test_recurrence():
    attn = BlockRecurrentAttention(128, 128, state_len=128)
    inputs = [torch.randn((64, 128, 128)) for _ in range(10)]
    state = torch.zeros((64, 128, 128))
    for x in inputs:
        output, state = attn(x, state)
        assert output.shape == state.shape

