import torch
import unittest
from model.transformer import Transformer


class TestTransformer(unittest.TestCase):
    def test_transformer(self):
        # Model parameters
        SRC_VOCAB_SIZE = 10000
        TRG_VOCAB_SIZE = 10000
        D_MODEL = 512
        N_HEAD = 8
        D_FF = 2048
        N_LAYERS = 6
    
        # Create model
        model = Transformer(
            src_vocab_size=SRC_VOCAB_SIZE,
            trg_vocab_size=TRG_VOCAB_SIZE,
            d_model=D_MODEL,
            n_head=N_HEAD,
            d_ff=D_FF,
            n_layers=N_LAYERS
        )
        
        # Sample input data
        batch_size = 32
        src_seq_len = 10
        trg_seq_len = 12
        
        src = torch.randint(1, SRC_VOCAB_SIZE, (batch_size, src_seq_len))
        trg = torch.randint(1, TRG_VOCAB_SIZE, (batch_size, trg_seq_len))
        
        # Forward pass
        output = model(src, trg)
        print(f"Input source shape: {src.shape}")
        print(f"Input target shape: {trg.shape}")
        print(f"Output shape: {output.shape}")  # Should be (batch_size, trg_seq_len, trg_vocab_size)