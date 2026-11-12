"""Real case: the video's bigram model, with letters, on «Lazarillo de Tormes».

The model learns which letter follows which from the first chapter (Tratado Primero) and is measured
with cross-entropy and perplexity on the second one (Tratado Segundo), text it has never seen. On real
text the zero-probability problem of the video shows up at once; add-k smoothing (add k to every
count) fixes it.

Run:  python lazarillo_case.py
"""

import math
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path

RAW = Path(__file__).with_name("lazarillo.txt").read_text(encoding="utf-8")


def normalised(text):
    """Lowercase letters and single spaces: everything else (punctuation, line breaks) becomes a space."""
    return " " + re.sub(r"[^a-záéíóúüñ]+", " ", text.lower()).strip() + " "


FIRST, SECOND = RAW.split("Tratado Segundo")
TRAIN, TEST = normalised(FIRST), normalised(SECOND)
ALPHABET = sorted(set(TRAIN) | set(TEST))
V = len(ALPHABET)


def ngram_counts(text, n):
    """How many times each context (the n - 1 previous letters) is followed by each letter."""
    pairs = Counter((text[i - n + 1 : i], text[i]) for i in range(n - 1, len(text)))
    contexts = Counter()
    for (context, _), count in pairs.items():
        contexts[context] += count
    return pairs, contexts


def probability(model, context, letter, k):
    """Add-k smoothing: k extra imaginary sightings of every letter after every context."""
    pairs, contexts = model
    return (pairs[context, letter] + k) / (contexts[context] + k * V)


def mean_loss(model, text, n, k):
    """Average cross-entropy per letter, in nats (-ln p); infinite if some letter gets probability 0."""
    total = 0.0
    for i in range(n - 1, len(text)):
        p = probability(model, text[i - n + 1 : i], text[i], k)
        if p == 0:
            return math.inf
        total -= math.log(p)
    return total / (len(text) - n + 1)


def unseen_pairs(model, text):
    pairs, _ = model
    return Counter((a, b) for a, b in pairwise(text) if pairs[a, b] == 0)


def perplexity(loss):
    return math.exp(loss)


if __name__ == "__main__":
    print(f"train (Tratado Primero): {len(TRAIN):,} characters · test (Tratado Segundo): {len(TEST):,}")
    print(f"alphabet: {V} symbols (letters, accented letters, ñ and the space)")
    print(f"uniform guess: loss ln {V} = {math.log(V):.3f} · perplexity {V}")

    unigram = ngram_counts(TRAIN, 1)
    loss = mean_loss(unigram, TEST, 1, k=1)
    print(f"letter frequencies only (no context): test loss {loss:.3f} · perplexity {perplexity(loss):.2f}")

    bigram = ngram_counts(TRAIN, 2)
    print("-- bigram, raw counts (k = 0), as in the video --")
    loss = mean_loss(bigram, TRAIN, 2, k=0)
    print(f"on the text it counted: loss {loss:.3f} · perplexity {perplexity(loss):.2f}")
    unseen = unseen_pairs(bigram, TEST)
    print(f"on new text: {sum(unseen.values())} letters after a pair never seen, {len(unseen)} different pairs:")
    print("  most frequent:", ", ".join(f"'{a}{b}' ×{c}" for (a, b), c in unseen.most_common(6)))
    print(f"  loss {mean_loss(bigram, TEST, 2, k=0)} -> perplexity infinite")

    print("-- bigram with add-one smoothing (k = 1) --")
    loss = mean_loss(bigram, TEST, 2, k=1)
    print(f"on new text: loss {loss:.3f} nats = {loss / math.log(2):.3f} bits · perplexity {perplexity(loss):.2f}")
    pairs, contexts = bigram
    for letter in ("u", "a"):
        p = probability(bigram, "q", letter, 1)
        print(f"P('{letter}' | 'q') = ({pairs['q', letter]} + 1) / ({contexts['q']} + {V}) = {p:.4f}")
