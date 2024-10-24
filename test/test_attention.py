import torch

from module.group_query_attention import GroupQueryAttention
from module.multi_head_attention import MultiHeadAttention
from module.multi_query_attention import MultiQueryAttention
from module.scale_dot_product_attention import ScaleDotProductAttention


def test_attention():
    b_sz, n_head, seq_l, d_k = 32, 8, 16, 64
    x = torch.randn(b_sz, n_head, seq_l, d_k)
    attn = ScaleDotProductAttention()
    out, _ = attn(x, x, x)
    assert x.shape == out.shape


def test_mha_forward():
    b_sz, n_head, seq_l, d_k = 32, 8, 16, 64
    x = torch.randn(b_sz, seq_l, n_head * d_k)
    attn = MultiHeadAttention(d_model=d_k * n_head, num_heads=n_head, context_length=seq_l, mask=True)
    out, _ = attn(x, x, x)
    assert out.shape == x.shape


def test_gqa_forward():
    b_sz, n_head, seq_l, d_k = 32, 8, 16, 64
    x = torch.randn(b_sz, seq_l, n_head * d_k)
    attn = GroupQueryAttention(d_model=d_k * n_head, num_heads=n_head, num_groups=2)
    out, _ = attn(x, x, x)
    assert out.shape == x.shape


def test_mqa_forward():
    b_sz, n_head, seq_l, d_k = 32, 8, 16, 64
    x = torch.randn(b_sz, seq_l, n_head * d_k)
    attn = MultiQueryAttention(d_model=d_k * n_head, num_heads=n_head)
    out, _ = attn(x, x, x)
    assert out.shape == x.shape
