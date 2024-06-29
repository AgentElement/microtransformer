import torch
import torch.nn as nn
from torch.nn import functional as F

from app.trigram import TrigramLanguageModel
from app.bigram import BigramLanguageModel

torch.manual_seed(1337)


def main():
    with open("input.txt") as input:
        text = input.read()
    chars = sorted(list(set(text)))

    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    block_size = 8

    x = train_data[:block_size]
    y = train_data[1:block_size+1]

    batch_size = 4

    xb, yb = get_batch(train_data, block_size, batch_size)
    
    print("Bigram")

    m = BigramLanguageModel(len(chars))
    out = m(xb, yb)

    idx = torch.zeros((1, 1), dtype=torch.long)
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    batch_size = 32
    for steps in range(40000):
        xb, yb = get_batch(train_data, block_size, batch_size)
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(loss.item())
    
    print(decode(m.generate(idx, max_new_tokens=1000)[0].tolist()))

    print("Trigram")

    m = TrigramLanguageModel(len(chars))
    out = m(xb, yb)

    idx = torch.zeros((1, 1), dtype=torch.long)
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    batch_size = 32
    for steps in range(40000):
        xb, yb = get_batch(train_data, block_size, batch_size)
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(loss.item())
    
    print(decode(m.generate(idx, max_new_tokens=1000)[0].tolist()))



def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def esitmate_loss(model):
    out = {}
    model.eval()


if __name__ == '__main__':
    main()
