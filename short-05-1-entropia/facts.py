"""The numbers shown in the short: the same worked example as video 05 (code/crossentropy.py).

Run:  uv run python videos/short-05-1-entropia/code/facts.py
"""

import math


def softmax(logits: list[float]) -> list[float]:
    e = [math.exp(z) for z in logits]
    return [x / sum(e) for x in e]


def loss(p: float) -> float:
    """Cross-entropy of one prediction: -ln of the probability given to the right answer."""
    return 0.0 - math.log(p)


if __name__ == "__main__":
    words, logits = ["tejado", "árbol", "sofá"], [2, 1, 0]
    for word, p in zip(words, softmax(logits), strict=True):
        print(f"{word:<8} apuesta {p:.3f}   pierde {loss(p):.3f}")
