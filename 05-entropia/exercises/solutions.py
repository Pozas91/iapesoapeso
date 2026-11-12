"""Solutions of exercises.md. Run:  python solutions.py"""

import math
import random

from lazarillo_case import ALPHABET, TEST, TRAIN, mean_loss, ngram_counts, perplexity, probability

bigram = ngram_counts(TRAIN, 2)


def softmax(logits, temperature=1):
    e = [math.exp(z / temperature) for z in logits]
    return [x / sum(e) for x in e]


def next_letter_probs(previous, temperature, k=1):
    """The bigram's probabilities used as logits (ln p), then softmax with temperature, as in the video."""
    logits = [math.log(probability(bigram, previous, letter, k)) for letter in ALPHABET]
    return softmax(logits, temperature)


def generate(temperature, length=70, seed=1):
    """Shannon's game: draw each letter according to the previous one."""
    rng = random.Random(seed)
    text = " "
    for _ in range(length):
        text += rng.choices(ALPHABET, weights=next_letter_probs(text[-1], temperature))[0]
    return text.strip()


print("== 1. Temperature ==")
u = ALPHABET.index("u")
for t in (0.5, 1, 2):
    print(f"T = {t}: P('u' | 'q') = {next_letter_probs('q', t)[u]:.3f} · «{generate(t)}»")

print("== 2. How much to smooth (bigram, new text) ==")
for k in (1, 0.1, 0.01, 0.001):
    loss = mean_loss(bigram, TEST, 2, k)
    print(
        f"k = {k}: P('u' | 'q') = {probability(bigram, 'q', 'u', k):.4f} · loss {loss:.3f} · perplexity {perplexity(loss):.2f}"
    )

print("== 3. More context: the two previous letters (trigram) ==")
trigram = ngram_counts(TRAIN, 3)
for k in (1, 0.1, 0.01):
    seen, new = mean_loss(trigram, TRAIN, 3, k), mean_loss(trigram, TEST, 3, k)
    print(f"k = {k}: perplexity {perplexity(seen):.2f} on the text it counted, {perplexity(new):.2f} on new text")
contexts = trigram[1]
unseen_contexts = sum(TEST[i - 2 : i] not in contexts for i in range(2, len(TEST)))
print(f"letters in the new text whose two previous letters were never seen together: {unseen_contexts}")
