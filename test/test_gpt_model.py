import unittest

import tiktoken
import torch

from model.gpt2 import GPT
from test.test_config import GPT_CONFIG_124M
from util.generate import generate_text_simple


class TestGpt(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(123)
        self.model = GPT(GPT_CONFIG_124M)

    def test_gpt(self):
        model = self.model
        print(model)

    def test_generate(self):
        model = self.model
        model.eval()  # disable dropout

        start_context = "Hello, I am"
        tokenizer = tiktoken.get_encoding('gpt2')
        encoded = tokenizer.encode(start_context)
        print("encoded:", encoded)

        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        print("encoded_tensor.shape:", encoded_tensor.shape)

        out = generate_text_simple(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=6,
            context_size=GPT_CONFIG_124M["context_length"]
        )

        print("Output:", out)
        print("Output length:", len(out[0]))

        decoded_text = tokenizer.decode(out.squeeze(0).tolist())
        print(decoded_text)
