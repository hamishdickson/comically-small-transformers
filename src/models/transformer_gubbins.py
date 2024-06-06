import torch
import torch.nn as nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, n_embed, head_size, block_size, num_heads, multi_head_dropout, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embed, head_size, block_size, num_heads, multi_head_dropout, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Head(nn.Module):
    """one head of self-attention"""
    def __init__(self, n_embed, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        # this is a buffer, not a parameter - a bit like how batchnorm has running stats that aren't parameters
        # tril is the lower triangular matrix used for the "causal" mask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        # this one is interesting, stop some nodes from communicating
        wei = self.dropout(wei)

        v = self.value(x)

        return wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, head_size, block_size, num_heads, head_dropout, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size, head_dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        # this 4 comes from the paper, looks arbitary. Idea is to make the model wider here
        scale_factor = 4
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * scale_factor),
            nn.ReLU(),
            nn.Linear(n_embed * scale_factor, n_embed), # projection
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    