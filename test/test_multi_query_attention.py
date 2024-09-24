import torch

from module.multi_query_attention import MultiQueryAttention


def test_forward():
    b_sz, n_head, seq_l, d_k = 32, 8, 16, 64
    x = torch.randn(b_sz, seq_l, n_head * d_k)
    attn = MultiQueryAttention(d_model=d_k*n_head, num_heads=n_head)
    out, _ = attn(x, x, x)
    assert out.shape == x.shape
