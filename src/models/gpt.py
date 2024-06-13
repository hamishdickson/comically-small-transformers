import torch
import torch.nn as nn
import torch.nn.functional as F
from . import transformer_gubbins

class Config:
    """Some basic hyperparameters for the model.

    This is only really works for a tiny model, but you can probably train this on a macbook air
    if you wanted to. It's not going to be very good, but it will do something.
    """
    vocab_size = 5
    n_embed = 768
    block_size = 64
    n_head = 4
    head_size = n_embed // n_head
    num_heads = 4
    n_layer = 12
    dropout = 0.1
    head_dropout = 0.1
    multi_head_dropout = 0.1



class GPT(nn.Module):
    """A simple GPT model.
    
    Note, this is very very similar to the decoder in the "Attention is All You Need" paper with
    the exception of the layer norma ("add & norm" in the paper). This happens before the ff layers
    instead of after in the paper. This is now a common variation in transformer models.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        
        self.blocks = nn.Sequential(*[transformer_gubbins.Block(config.n_embed, config.head_size, config.block_size, config.num_heads, config.head_dropout, config.multi_head_dropout) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)

        # Use the device of the token_embedding_table
        device = self.token_embedding_table.weight.device
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
