import math
import torch
import torch.nn as nn
import torch.nn.functional as f

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int=5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)         #(L, d)
        pos = torch.arange(max_len).unsqueeze(1).float()   #(d,1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )      
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.dropout(x + pe[:, : x.size(1)])

    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1): 
        super().__init__()
        assert d_model % n_heads == 0, "D_model needs to be divisible by n_head"

        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model) → (B, h, T, d_k)
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: torch.Tensor | None = None, ) -> torch.Tensor:
        B = query.size(0)

        
