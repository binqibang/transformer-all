import torch.nn as nn

from module.scale_dot_product_attention import ScaleDotProductAttention


class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_groups == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.group_size = num_heads // num_groups
        self.attn = ScaleDotProductAttention(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, num_groups * self.d_k)
        self.w_v = nn.Linear(d_model, num_groups * self.d_k)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        b_sz, num_tokens, d_model = q.shape

        # (b_sz, num_tokens, num_heads, d_k)
        q = self.w_q(k).view(b_sz, -1, self.num_heads, self.d_k)

        # (b_sz, num_tokens, num_groups, d_k)
        k = self.w_k(k).view(b_sz, -1, self.num_groups, self.d_k)
        v = self.w_v(v).view(b_sz, -1, self.num_groups, self.d_k)

        # (b_sz, num_heads, num_tokens, d_k)
        q = q.transpose(1, 2)
        # (b_sz, num_groups, num_tokens, d_k)
        k, v = k.transpose(1, 2), v.transpose(1, 2)

        # Expand keys and values to match the number of heads
        # (b_sz, num_heads, num_tokens, d_k)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        out, attn_score = self.attn(q, k, v, mask)

        # (b_sz, num_tokens, d_model)
        out = out.transpose(1, 2).contiguous().view(b_sz, -1, self.num_heads * self.d_k)
        out = self.w_o(out)
        out = self.dropout(out)

        return out, attn_score
