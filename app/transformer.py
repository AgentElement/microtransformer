import torch
import torch.nn as nn
from torch.nn import functional as F

class Transformer(nn.Module):

    def __init__(self, vocab_size, params):
        super().__init__()

        n_embed = params.n_embed
        self.block_size = params.block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(self.block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, inputs, targets=None):
        B, T = inputs.shape

        tok_emb = self.token_embedding_table(inputs)
        positions = torch.arange(T)
        pos_emb = self.position_embedding_table(positions)
        x = tok_emb + pos_emb
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


def main():
    torch.manual_seed(1337)
    B, T, C = 4, 8, 2
    x = torch.randn(B, T, C)
    xbow = torch.zeros((B, T, C))
    for b in range(B):
        for t in range(T):
            xprev = x[b, :t+1]
            xbow[b, t] = torch.mean(xprev, 0)

    tril = torch.tril(torch.ones(T, T))
    wei = torch.zeros((T, T))
    wei = wei / wei.sum(1, keepdim=True)
    xbow2 = wei @ x



if __name__ == '__main__':
    main()
