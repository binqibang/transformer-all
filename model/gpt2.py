import torch.nn as nn
import torch
from block.gpt_block import GPTBlock


class GPT(nn.Module):
    def __init__(self, gpt_config):
        super().__init__()
        context_length = gpt_config['context_length']
        self.token_embedding = nn.Embedding(gpt_config['vocab_size'], gpt_config['d_model'])
        self.pos_embedding = nn.Embedding(gpt_config['context_length'], gpt_config['d_model'])
        self.dropout_embedding = nn.Dropout(gpt_config['dropout'])

        self.transformer_blocks = nn.ModuleList(
            [GPTBlock(gpt_config['d_model'], gpt_config['num_heads'],
                      gpt_config['d_ff'], gpt_config['dropout'])
             for _ in range(gpt_config['num_layers'])]
        )

        self.final_norm = nn.LayerNorm(gpt_config["d_model"])
        self.out = nn.Linear(gpt_config["d_model"], gpt_config["vocab_size"], bias=False)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        tok_embeds = self.token_embedding(input_ids)
        pos_embeds = self.pos_embedding(torch.arange(seq_len, device=input_ids.device))
        x = tok_embeds + pos_embeds
        x = self.dropout_embedding(x)
        for trf_block in self.transformer_blocks:
            x = trf_block(x, self.mask)
        x = self.final_norm(x)
        logits = self.out(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx
