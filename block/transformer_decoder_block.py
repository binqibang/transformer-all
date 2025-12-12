import torch.nn as nn
from module.multi_head_attention import MultiHeadAttention
from module.ffn import PositionWiseFeedForward


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.enc_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, dec_input, enc_output, src_mask, trg_mask):
        enc_attn = None
        # 1. self attention
        residual = dec_input
        x, slf_attn_weight = self.slf_attn(dec_input, dec_input, dec_input, trg_mask)

        # 2. add & norm
        x = self.norm1(residual + x)

        # 3. encoder output attention
        if enc_output is not None:
            residual = x
            x, enc_attn_weight = self.enc_attn(x, enc_output, enc_output, src_mask)

            # 4. add & norm
            x = self.norm2(residual + x)

        # 5. feed forward
        residual = x
        x = self.ffn(x)

        # 6. add & norm
        x = self.norm3(residual + x)

        return x, slf_attn_weight, enc_attn_weight
