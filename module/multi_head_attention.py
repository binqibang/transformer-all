import torch.nn as nn
import torch

from module.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, context_length, dropout=0.1, mask=False):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.n_head = num_heads
        self.attn = ScaleDotProductAttention(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        if mask:
            self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, q, k, v):
        """
        MultiHeadAttention(Q, K, V) = Concat(h1, h2, ...)W_o, h1 = Attention(QW_q, KW_k, VW_v)
        :param q: (b_sz, n_token, d_model)
        :param k: (b_sz, n_token, d_model)
        :param v: (b_sz, n_token, d_model)
        :return: out, attn
        """
        b_sz, n_token, d_model = q.shape

        # (b_sz, n_token, n_head, d_k)
        q = self.w_q(k).view(b_sz, -1, self.n_head, self.d_k)
        k = self.w_k(k).view(b_sz, -1, self.n_head, self.d_k)
        v = self.w_v(v).view(b_sz, -1, self.n_head, self.d_k)

        # (b_sz, n_head, n_token, d_k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        out, attn = self.attn(q, k, v, self.mask.bool()[:n_token, :n_token])

        # (b_sz, n_token, d_model)
        out = out.transpose(1, 2).contiguous().view(b_sz, -1, self.n_head * self.d_k)
        out = self.w_o(out)
        out = self.dropout(out)

        return out, attn
