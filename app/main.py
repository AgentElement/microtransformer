import torch

from app.trigram import TrigramLanguageModel
from app.bigram import BigramLanguageModel
from app.transformer import Transformer

torch.manual_seed(1337)


class Hyperparams:
    block_size = 64  # T
    batch_size = 16  # B
    test_train_split = 0.9
    training_steps = 10000
    eval_iters = 100
    lr = 1e-3
    n_embed = 60  # C
    n_layers = 5
    n_head = 5  # H
    dropout = 0.2


def param_size(model):
    return sum(p.numel() for p in model.parameters())


def train_model(model, text):
    chars = sorted(list(set(text)))

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)

    n = int(Hyperparams.test_train_split * len(text))
    train_data = data[:n]
    val_data = data[n:]

    m = model(len(chars), Hyperparams)
    print(f"Parameter size: {param_size(m)}")

    idx = torch.zeros((1, 1), dtype=torch.long)
    optimizer = torch.optim.AdamW(m.parameters(), lr=Hyperparams.lr)

    for steps in range(Hyperparams.training_steps):
        xb, yb = get_batch(train_data, Hyperparams.block_size, Hyperparams.batch_size)
        logits, loss = m(xb, yb)
        if steps % 1000 == 0:
            train_loss = loss.item()
            val_loss = esitmate_loss(m, val_data, Hyperparams).item()
            print(f"Iteration {steps} loss: {train_loss}, val: {val_loss}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Final training loss: ", loss.item())
    print("Final validation loss: ", esitmate_loss(m, val_data, Hyperparams).item())
    print(decode(m.generate(idx, max_new_tokens=1000)[0].tolist()))

    return m


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def esitmate_loss(model, data, params):
    model.eval()
    losses = torch.zeros(params.eval_iters)
    for k in range(params.eval_iters):
        X, Y = get_batch(data, params.block_size, params.batch_size)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    return losses.mean()


def main():
    with open("input.txt") as input:
        text = input.read()
    model = train_model(Transformer, text)
    params = param_size(model)
    torch.save(model, f"model-{params}.pt")


if __name__ == "__main__":
    main()
