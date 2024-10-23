import torch.nn as nn
from module.scale_dot_product_attention import ScaleDotProductAttention


class MultiQueryAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.attn = ScaleDotProductAttention(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, self.d_k)
        self.w_v = nn.Linear(d_model, self.d_k)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        b_sz, n_token, d_model = q.shape

        # query
        q = self.w_q(q).view(b_sz, -1, self.num_heads, self.d_k)
        q = q.transpose(1, 2)   # (b_sz, n_head, n_token, d_k)

        # key and value are shared
        k = self.w_k(k).view(b_sz, n_token, 1, self.d_k)
        k = k.transpose(1, 2)   # (b_sz, 1, n_token, d_k)
        v = self.w_v(v).view(b_sz, n_token, 1, self.d_k)
        v = v.transpose(1, 2)   # (b_sz, 1, n_token, d_k)

        if mask:
            mask = mask.unsqueeze(1)
        out, attn = self.attn(q, k, v, mask=mask)

        # (b_sz, seq_l, d_model)
        out = out.transpose(1, 2).contiguous().view(b_sz, -1, self.d_model)
        out = self.w_o(out)
        out = self.dropout(out)

        return out, attn
