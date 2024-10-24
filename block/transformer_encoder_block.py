import torch.nn as nn
from module.multi_head_attention import MultiHeadAttention
from module.ffn import PositionWiseFeedForward


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        # 1. multi-head attention
        residual = x
        x, attn = self.attn(x, x, x, mask=src_mask)

        # 2. add & norm
        x = self.norm1(residual + x)

        # 3. feed forward
        residual = x
        x = self.ffn(x)

        # 4. add & norm
        x = self.norm2(residual + x)

        return x, attn
