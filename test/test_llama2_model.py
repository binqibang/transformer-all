import torch
from model.llama2 import Llama2


# LLAMA2_CONFIG_7B = {
#     "vocab_size": 32000,     # Vocabulary size
#     "context_length": 4096,  # Context length
#     "d_model": 4096,         # Embedding dimension
#     "num_heads": 32,         # Number of attention heads
#     "num_layers": 32,        # Number of layers
#     "d_ff": 11008,           # Size of the intermediate dimension in FeedForward
#     "dropout": 0.1
# }

LLAMA2_CONFIG_test = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 1024,  # Context length
    "d_model": 4096,         # Embedding dimension
    "num_heads": 16,         # Number of attention heads
    "num_layers": 8,        # Number of layers
    "d_ff": 11008,           # Size of the intermediate dimension in FeedForward
    "dropout": 0.1
}


def create_llama2():
    torch.manual_seed(123)
    model = Llama2(LLAMA2_CONFIG_test)
    return model


def test_llama2():
    model = create_llama2()
    print(model)
