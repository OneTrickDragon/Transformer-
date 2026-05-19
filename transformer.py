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

        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = self.dropout(f.softmax(scores, dim=-1))
        out = (attn @ V).transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.W_o(out)        
    

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayernNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor| None = None) -> torch.Tensor:
        x =  x + self.drop(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, 
                 dropout: float, max_len: int = 5000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.pos_enc(self.embed(src) * self.scale)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, tgt_mask: torch.Tensor,
                src_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
        x = x + self.drop(self.cross_attn(self.norm2(x), enc_out, enc_out, src_mask))

        x = x + self.drop(self.ffn(self.norm3(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_ff: int, n_heads: int, 
                 n_layers: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.embed = nn.Embedding(d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, tgt: torch.Tensor, enc_out: torch.Tensor, tgt_mask: torch.Tensor| None = None,
                src_mask: torch.Tensor| None = None) -> torch.Tensor:
        x = self.pos_enc(self.embed(tgt) * self.scale)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int=512,
                 n_heads: int = 8, n_layers: int = 6, d_ff: int = 2048,
                 pad_idx: int = 0, dropout: float = 0.1):
        super().__init__()
        self.pad = pad_idx
        self.encoder = Encoder(src_vocab, d_model, n_layers, n_heads, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, d_ff, n_heads, n_layers, dropout)
        self.proj = nn.Linear(d_model, tgt_vocab)

        self._init_weights()

        def _init_weights(self):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def _src_mask(self, src: torch.Tensor) -> torch.Tensor:
            return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        def _tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
            T = tgt.size(1)
            pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
            causal_mask = torch.tril(torch.ones(T, T, device=tgt.device)).bool()
            return pad_mask & causal_mask
        
        def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
            src_mask = self._src_mask(src)
            tgt_mask = self._tgt_mask(tgt)

            enc_out = self.encoder(src, src_mask)
            dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)

            return self.proj(dec_out)  # (B, T, tgt_vocab)
        
        @torch.no_grad
        def greedy_decode(self, src: torch.Tensor, bos_idx: int, eos_idx: 
                          int, max_len: int = 100,) -> torch.Tensor:
            
            B = src.size(0)
            src_mask = self._src_mask(src)
            enc_out = self.encoder(src, src_mask)

            ys = torch.full((B,1), bos_idx, dtype=torch.long, device = src.device)
            done = torch.zeros(B, dtype=torch.bool, device = src.device)

            for _ in range(max_len - 1):
                tgt_mask = self._tgt_mask(ys)
                dec_out = self.decoder(ys, enc_out, tgt_mask, src_mask)
                logits = self.proj(dec_out[:, -1])
                next_tok = logits.argmax(dim=-1, keepdim=True) 
                ys = torch.cat([ys, next_tok], dim=1)
                done |= (next_tok.squeeze(1) == eos_idx)
                if done.all():
                    break
            
            return ys[:, 1:]


