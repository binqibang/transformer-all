import torch
import torch.nn as nn
from torch import Tensor


class ScaleDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None) -> (Tensor, Tensor):
        """
        Attention(Q, K, V) = Softmax(QK^T)V / d_k^0.5
        :param q: (b_sz, n_head, n_token, d_k)
        :param k: (b_sz, n_head, n_token, d_k)
        :param v: (b_sz, n_head, n_token, d_k)
        :param mask:
        """
        d_k = q.shape[-1]
        # (b_sz, n_head, n_token, d_k) x (b_sz, n_head, d_k, n_token) = (b_sz, n_head, n_token, n_token)
        attn = torch.matmul(q, k.transpose(-1, -2)) / d_k ** 0.5
        if mask:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        # (b_sz, n_head, n_token, n_token) x (b_sz, n_head, n_token, d_k) = (b_sz, n_head, n_token, d_k)
        output = attn @ v
        return output, attn
