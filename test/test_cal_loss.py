import tiktoken
import torch.cuda

from dataset.gpt_dataloader import create_dataloader_v1
from model.gpt2 import GPT
from train.gpt_train_simple import calc_loss_loader


def test_loss():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 512,
        "d_model": 768,
        "num_heads": 12,
        "num_layers": 6,
        "dropout": 0.1,
        "d_ff": 2048
    }
    model = GPT(GPT_CONFIG_124M)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes

    torch.manual_seed(123)  # For reproducibility due to the shuffling in the data loader

    file_path = '../dataset/the-verdict.txt'
    with open(file_path) as f:
        text_data = f.read()
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    tokenizer = tiktoken.get_encoding('gpt2')
    train_loader = create_dataloader_v1(
        train_data,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader_v1(
        val_data,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    with torch.no_grad():  # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)
