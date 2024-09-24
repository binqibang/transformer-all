import torch
from module.scale_dot_product_attention import ScaleDotProductAttention


def test_forward():
    b_sz, n_head, seq_l, d_k = 32, 8, 16, 64
    x = torch.randn(b_sz, n_head, seq_l, d_k)
    attn = ScaleDotProductAttention()
    out, _ = attn(x, x, x)
    assert x.shape == out.shape
