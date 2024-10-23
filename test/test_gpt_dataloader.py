import tiktoken
import torch

from dataset.gpt_dataloader import create_dataloader_v1


def test_dataloader():
    with open('../dataset/the-verdict.txt') as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding('gpt2')

    max_length = 16
    batch_size = 8
    dataloader = create_dataloader_v1(raw_text, tokenizer, batch_size=batch_size, max_length=max_length,
                                      stride=max_length)

    vocab_size = 50257
    output_dim = 256
    context_length = 1024

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    for batch in dataloader:
        x, y = batch

        assert x[0][1] == y[0][0]

        token_embeddings = token_embedding_layer(x)
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))

        input_embeddings = token_embeddings + pos_embeddings

        assert input_embeddings.shape == torch.Size([batch_size, max_length, output_dim])


