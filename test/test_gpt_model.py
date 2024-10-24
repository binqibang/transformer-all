import tiktoken
import torch

from model.gpt2 import GPT


def test_gpt():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "d_model": 768,
        "num_heads": 12,
        "num_layers": 12,
        "dropout": 0.1,
        "d_ff": 2048
    }

    torch.manual_seed(123)
    model = GPT(GPT_CONFIG_124M)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    out = model(encoded_tensor)
    print(out.shape)
