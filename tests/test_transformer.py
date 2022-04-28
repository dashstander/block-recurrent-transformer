from block_recurrent_transformer.transformer import BlockRecurrentAttention
import torch


def test_block_recurrent_attention():
    attn = BlockRecurrentAttention(512, 512)
    x = torch.randn((64, 512, 512))
    state = torch.randn((64, 512, 512))
    assert attn(x, state)
