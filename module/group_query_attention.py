import torch
import torch.nn as nn

from module.scale_dot_product_attention import ScaleDotProductAttention


class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.attn = ScaleDotProductAttention(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        b_sz, n_token, d_model = q.shape

        # (b_sz, n_token, n_head, d_k)
        q = self.w_q(k).view(b_sz, -1, self.num_heads, self.d_k)
        k = self.w_k(k).view(b_sz, -1, self.num_heads, self.d_k)
        v = self.w_v(v).view(b_sz, -1, self.num_heads, self.d_k)

        # (b_sz, n_head, n_token, d_k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        group_size = self.num_heads // self.num_groups
        attn_outputs = []
        attn_scores = []
        for i in range(self.num_groups):
            # (b_sz, n_group, n_token, d_k)
            q_group = q[:, i * group_size:(i + 1) * group_size, :, :]
            k_group = k[:, i * group_size:(i + 1) * group_size, :, :]
            v_group = v[:, i * group_size:(i + 1) * group_size, :, :]
            out, score = self.attn(q_group, k_group, v_group, mask=mask)
            attn_outputs.append(out)
            attn_scores.append(score)

        out = torch.cat(attn_outputs, dim=1)
        score = torch.cat(attn_scores, dim=1)
        out = out.transpose(1, 2).contiguous().view(b_sz, -1, self.num_heads * self.d_k)
        out = self.w_o(out)
        out = self.dropout(out)

        return out, score
