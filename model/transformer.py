import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from block.transformer_decoder_block import DecoderBlock
from block.transformer_encoder_block import EncoderBlock
from module.pos_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size,
                 trg_vocab_size,
                 d_model=512,
                 n_head=8,
                 d_ff=2048,
                 n_layers=6,
                 dropout=0.1,
                 max_seq_len=5000):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        
        # Positional encoding
        self.position_encoding = PositionalEncoding(max_seq_len, d_model)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_head, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, n_head, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    
    def forward(self, src, trg, src_mask=None, trg_mask=None):
        # Create masks if not provided
        if src_mask is None:
            src_mask = self.create_src_mask(src)
        
        if trg_mask is None:
            trg_mask = self.create_trg_mask(trg)
        
        # Encode source sequence
        enc_output = self.encode(src, src_mask)
        
        # Decode target sequence
        dec_output = self.decode(trg, enc_output, src_mask, trg_mask)
        
        # Final linear projection
        output = self.fc_out(dec_output)
        
        return output
    
    def encode(self, src, src_mask):
        # Source embedding + positional encoding
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.position_encoding(src_emb)
        # (b_sz, src_len, d_model)
        src_emb = self.dropout(src_emb)
        
        # Pass through encoder blocks
        enc_output = src_emb
        for block in self.encoder_blocks:
            enc_output, _ = block(enc_output, src_mask)
        
        return enc_output
    
    def decode(self, trg, enc_output, src_mask, trg_mask):
        # Target embedding + positional encoding
        trg_emb = self.trg_embedding(trg) * math.sqrt(self.d_model)
        trg_emb = self.position_encoding(trg_emb)
        trg_emb = self.dropout(trg_emb)
        
        # Pass through decoder blocks
        dec_output = trg_emb
        for block in self.decoder_blocks:
            dec_output, _, _ = block(dec_output, enc_output, src_mask, trg_mask)
        
        return dec_output
    
    def create_src_mask(self, src):
        # Padding mask: 1 for real tokens, 0 for padding
        # (batch, 1, src_len)
        src_mask = (src != 0).unsqueeze(-2)
        return src_mask
    
    def create_trg_mask(self, trg):
        # Combined padding mask and future tokens mask
        batch_size, trg_len = trg.size()
        
        # Padding mask
        # (batch, 1, trg_len)
        pad_mask = (trg != 0).unsqueeze(-2)  
        
        # Future tokens mask (upper triangular matrix)
        # (1, trg_len, trg_len)
        future_mask = torch.tril(torch.ones((1, trg_len, trg_len))).bool()
        
        # Combine masks
        trg_mask = pad_mask & future_mask
        
        # (batch, trg_len, trg_len)
        return trg_mask