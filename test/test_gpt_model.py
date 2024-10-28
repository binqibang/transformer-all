import tiktoken
import torch

from model.gpt2 import GPT
from util.generate import generate_text_simple

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "d_model": 768,
    "num_heads": 12,
    "num_layers": 6,
    "dropout": 0.1,
    "d_ff": 2048
}


def create_gpt():
    torch.manual_seed(123)
    model = GPT(GPT_CONFIG_124M)
    return model


def test_gpt():
    model = create_gpt()
    print(model)


def test_generate():
    model = create_gpt()
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
