import torch.nn as nn
import torch
from block.llama2_block import Llama2Block
from module.rms_norm import RMSNorm


class Llama2(nn.Module):
    def __init__(self, llama2_config):
        super().__init__()
        context_length = llama2_config['context_length']
        self.token_embedding = nn.Embedding(llama2_config['vocab_size'], llama2_config['d_model'])

        self.transformer_blocks = nn.ModuleList(
            [Llama2Block(llama2_config['d_model'], llama2_config['num_heads'], context_length,
                         llama2_config['d_ff'], llama2_config['dropout'])
             for _ in range(llama2_config['num_layers'])]
        )

        self.final_norm = RMSNorm(llama2_config["d_model"])
        self.out = nn.Linear(llama2_config["d_model"], llama2_config["vocab_size"], bias=False)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, input_ids):
        tok_embeds = self.token_embedding(input_ids)
        x = tok_embeds
        for trf_block in self.transformer_blocks:
            x = trf_block(x, self.mask)
        x = self.final_norm(x)
        logits = self.out(x)
        return logits
