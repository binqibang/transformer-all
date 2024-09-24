import torch

from module.group_query_attention import GroupQueryAttention


def test_forward():
    b_sz, n_head, seq_l, d_k = 32, 8, 16, 64
    x = torch.randn(b_sz, seq_l, n_head * d_k)
    attn = GroupQueryAttention(d_model=d_k*n_head, num_heads=n_head, num_groups=2)
    out, _ = attn(x, x, x)
    assert out.shape == x.shape
