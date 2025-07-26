import unittest
import tiktoken
import torch.cuda
from dataset.gpt_dataloader import create_dataloader_v1
from model.gpt2 import GPT
from test.test_config import GPT_CONFIG_124M
from train.gpt_train_simple import calc_loss_loader


class TestLossCalculation(unittest.TestCase):
    def setUp(self):
        self.model = GPT(GPT_CONFIG_124M)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        torch.manual_seed(123)  # For reproducibility due to the shuffling in the data loader

        file_path = '../dataset/the-verdict.txt'
        with open(file_path) as f:
            text_data = f.read()
        train_ratio = 0.90
        split_idx = int(train_ratio * len(text_data))
        self.train_data = text_data[:split_idx]
        self.val_data = text_data[split_idx:]

        self.tokenizer = tiktoken.get_encoding('gpt2')

        self.train_loader = create_dataloader_v1(
            self.train_data,
            tokenizer=self.tokenizer,
            batch_size=2,
            max_length=GPT_CONFIG_124M["context_length"],
            stride=GPT_CONFIG_124M["context_length"],
            drop_last=True,
            shuffle=True,
            num_workers=0
        )
        self.val_loader = create_dataloader_v1(
            self.val_data,
            tokenizer=self.tokenizer,
            batch_size=2,
            max_length=GPT_CONFIG_124M["context_length"],
            stride=GPT_CONFIG_124M["context_length"],
            drop_last=False,
            shuffle=False,
            num_workers=0
        )

    def test_calculate_loss(self):
        with torch.no_grad():  # Disable gradient tracking for efficiency because we are not training, yet
            train_loss = calc_loss_loader(self.train_loader, self.model, next(self.model.parameters()).device)
            val_loss = calc_loss_loader(self.val_loader, self.model, next(self.model.parameters()).device)

        self.assertIsNotNone(train_loss, "Training loss should not be None")
        self.assertIsNotNone(val_loss, "Validation loss should not be None")
        print("Training loss:", train_loss)
        print("Validation loss:", val_loss)
