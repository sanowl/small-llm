# models/mini_gpt.py

import torch
import torch.nn as nn
import math

class CustomTransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_dim, dropout=0.1):
        super(CustomTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_size),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_size)
        
    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_length=512):
        super(RotaryEmbedding, self).__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_length = max_seq_length

    def forward(self, seq_len):
        positions = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        sinusoid = torch.einsum("i,j->ij", positions, self.inv_freq)
        sin = sinusoid.sin()[None, :, None, :]
        cos = sinusoid.cos()[None, :, None, :]
        return sin, cos

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_size=512, num_heads=8, hidden_dim=2048, num_layers=12, max_seq_length=512, dropout=0.1):
        super(MiniGPT, self).__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = RotaryEmbedding(embed_size)
        
        self.layers = nn.ModuleList([
            CustomTransformerBlock(embed_size, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_size)
        self.output_layer = nn.Linear(embed_size, vocab_size)
        
    def forward(self, input_ids):
        batch_size, seq_length = input_ids.size()
        device = input_ids.device
        
        token_emb = self.token_embedding(input_ids)  # (batch_size, seq_length, embed_size)
        
        sin, cos = self.position_embedding(seq_length)
        token_emb = token_emb * cos + self.rotate(token_emb, sin, cos)
        
        x = token_emb
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_f(x)
        logits = self.output_layer(x)
        return logits
    
    def rotate(self, x, sin, cos):
        x1, x2 = x.chunk(2, dim=-1)
        x = torch.cat((-x2, x1), dim=-1)
        return x * sin + x * cos
