import torch.nn as nn
from module.multi_head_attention import Llama2MHA
from module.ffn import LLAMA2FeedForward
from module.rms_norm import RMSNorm


class LLAMA2Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, context_length, dropout=0.1):
        super().__init__()
        self.mha = Llama2MHA(d_model, num_heads, context_length, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = LLAMA2FeedForward(d_model, d_ff)

    def forward(self, x, mask):
        # 1. attention
        residual = x
        x = self.norm1(x)
        x, attn = self.mha(x, x, x, mask)
        x += residual

        # 2. feed forward
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x += residual

        return x
