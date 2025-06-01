**Microtransformer**

This is a small and very simple transformer, almost exactly like Karpathy's [minGPT](https://github.com/karpathy/minGPT), the predecessor to nanoGPT.

This transformer trains on a text dump of all of Shakespeare's text. To run the training loop, first activate the flake (`nix develop .`),
then run `python -m app.main`. The script will read `input.txt` (the training data) and write out a passable language model to `model-paramcount.pt`.

The model only sits in one file: `app/transformer.py`.
There is a simple bigram and trigram model (`app/bigram.py` and `app/trigram.py`) that you can look at if you like.
Additional scaffolding to run the model sits in `app/main.py`.

If you're looking at this as an educational tool, I'd recommend minGPT instead.

***
todo:

At some point I wanted to turn this project into an agent in a more interesting Q-learning setting with rollout search.
I might revisit the idea at some other point.
