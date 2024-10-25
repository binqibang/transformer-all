import torch.nn as nn
from module.multi_head_attention import MultiHeadAttention
from module.ffn import PositionWiseFeedForward


class GPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 1. attention
        residual = x
        x = self.norm1(x)
        x, attn = self.attn(x, x, x, mask)
        x = self.dropout(x)
        x += residual

        # 2. feed forward
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x += residual

        return x
