import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class
        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        # [[0.0, 1.0, ..., max_len - 1]]
        # shape: (1024, 1)
        pos = torch.arange(0, max_len).float().unsqueeze(dim=1)

        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)
        # shape: (256)
        _2i = torch.arange(0, d_model, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]
