import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):

    def __init__(self, head_size, params):
        super().__init__()

        block_size = params.block_size
        n_embed = params.n_embed

        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(params.dropout)
        tril = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer('tril', tril)

    
    def forward(self, inputs):
        B, T, C = inputs.shape
        k = self.key(inputs)
        q = self.query(inputs)
        v = self.value(inputs)

        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, head_size, params):
        super().__init__()
        n_embed = params.n_embed
        self.heads = nn.ModuleList([Head(head_size, params) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, inputs, targets=None):
        out = torch.cat([h(inputs) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):

    def __init__(self, params):
        super().__init__()
        n_embed = params.n_embed
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(params.dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, params):
        super().__init__()
        n_embed = params.n_embed
        n_head = params.n_head
        head_size = n_embed // n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size, params)
        self.ffd = FeedForward(params)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffd(self.ln2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, vocab_size, params):
        super().__init__()

        n_embed = params.n_embed
        self.block_size = params.block_size
        self.vocab_size = vocab_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(self.block_size, n_embed)

        self.blocks = nn.Sequential(
            *[Block(params) for _ in range(params.n_layers)],
            nn.LayerNorm(n_embed)
        )
        
        self.lm_head = nn.Linear(n_embed, vocab_size)


    def forward(self, inputs, targets=None):
        B, T = inputs.shape

        tok_emb = self.token_embedding_table(inputs)
        positions = torch.arange(T)
        pos_emb = self.position_embedding_table(positions)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, VS = logits.shape
        logits = logits.view(B * T, VS)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            _, ix_len = idx.shape
            tidx = idx[:, -min(self.block_size, ix_len):]
            logits, loss = self(tidx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
