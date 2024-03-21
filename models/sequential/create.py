import torch
import torch.nn as nn
import torch.nn.functional as F

class SASRecEncoder(nn.Module):
    def __init__(self, embedding_size, num_blocks, num_heads, dropout):
        super(SASRecEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout

        self.positional_encoding = PositionalEncoding(embedding_size)
        
        # SASRec blocks
        self.sasrec_blocks = nn.ModuleList([
            SASRecBlock(embedding_size, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        
        # Embedding layer
        
        
        # Positional encoding
        x = self.positional_encoding(x)
        
        # SASRec blocks
        for block in self.sasrec_blocks:
            x = block(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SASRecBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout):
        super(SASRecBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(embedding_size, num_heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.ReLU(),
            nn.Linear(embedding_size * 4, embedding_size)
        )
        self.layer_norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention layer
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embedding_size)
        x, _ = self.self_attention(x, x, x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embedding_size)
        x = self.dropout(x)
        x = self.layer_norm1(x + x)
        
        # Feed-forward layer
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.layer_norm2(x + x)
        
        return x
