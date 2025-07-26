import unittest

import tiktoken
import torch

from dataset.gpt_dataloader import create_dataloader_v1
from module.multi_head_attention import MultiHeadAttention


class TestDataloader(unittest.TestCase):

    def test_dataloader(self):
        file_path = '../dataset/the-verdict.txt'
        with open(file_path) as f:
            raw_text = f.read()

        tokenizer = tiktoken.get_encoding('gpt2')

        max_length = 16
        batch_size = 8
        dataloader = create_dataloader_v1(raw_text, tokenizer, batch_size=batch_size, max_length=max_length,
                                          stride=max_length)

        print(f'total batch for {file_path}: {len(dataloader)}\n')

        vocab_size = 50257
        output_dim = 256
        context_length = 1024

        token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
        pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

        input_embeddings = None

        for batch in dataloader:
            x, y = batch

            assert x[0][1] == y[0][0]

            token_embeddings = token_embedding_layer(x)
            pos_embeddings = pos_embedding_layer(torch.arange(max_length))

            input_embeddings = token_embeddings + pos_embeddings

            assert input_embeddings.shape == torch.Size([batch_size, max_length, output_dim])

        return input_embeddings

    def test_data_forward(self):
        x = self.test_dataloader()
        d_k, n_head = 64, 4
        attn = MultiHeadAttention(d_model=d_k * n_head, num_heads=n_head)
        out, _ = attn(x, x, x)
        assert out.shape == x.shape
