import torch.nn as nn
import torch
from torch import Tensor


class ScaleDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaleDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None) -> tuple[Tensor, Tensor]:
        """
        Attention(Q, K, V) = Softmax(QK^T / d_k^0.5)V
        :param q: (b_sz, n_head, n_token, d_k)
        :param k: (b_sz, n_head, n_token, d_k)
        :param v: (b_sz, n_head, n_token, d_k)
        :param mask: (b_sz, 1, n_token, n_token) or (b_sz, 1, 1, n_token)
        """
        b_sz, n_head, n_token, d_k = q.shape
        # (b_sz, n_head, n_token, d_k) x (b_sz, n_head, d_k, n_token) = (b_sz, n_head, n_token, n_token)
        attn_scores = q @ k.transpose(-1, -2)
        if mask is not None:
            if mask.shape[-2] > 1:
                mask = mask[:, :, :n_token, :n_token]
            mask_bool = mask.bool()
            attn_scores = attn_scores.masked_fill(mask_bool, -float('inf'))
        attn_weights = torch.softmax(attn_scores / d_k ** 0.5, -1) 
        attn_weights = self.dropout(attn_weights)
        # (b_sz, n_head, n_token, n_token) x (b_sz, n_head, n_token, d_k) = (b_sz, n_head, n_token, d_k)
        out = attn_weights @ v
        return out, attn_weights
