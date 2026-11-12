"""Video 05: from the scores of the last layer to probabilities (softmax, temperature) and to the
loss of a language model (cross-entropy)."""

import math
from collections import Counter
from fractions import Fraction
from itertools import pairwise


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def softmax(logits, temperature=1):
    e = [math.exp(z / temperature) for z in logits]
    return [x / sum(e) for x in e]


def loss(p):
    """Cross-entropy of one prediction: -ln of the probability given to the right answer."""
    return 0.0 - math.log(p)  # 0.0 - ...: prints 0.000 instead of -0.000 when p = 1


def steps_to_learn(gradient, z=-5.0, eta=1.0, target=0.9, limit=10_000):
    """One output neuron that should say 1 but starts very wrong. Steps until it says > target."""
    for step in range(limit):
        if sigmoid(z) > target:
            return step
        z -= eta * gradient(sigmoid(z))
    return None


def bigrams(sentence):
    """Pairs (previous word, next word), with <s> marking the start of the sentence."""
    return list(pairwise(["<s>", *sentence.split()]))


def mse_gradient(a, y=1):  # d/dz (a - y)²: the σ′ of videos 03 and 04 is still there
    return 2 * (a - y) * a * (1 - a)


def ce_gradient(a, y=1):  # d/dz −log a: the σ′ cancels out
    return a - y


if __name__ == "__main__":
    words, logits = ["tejado", "árbol", "sofá"], [2, 1, 0]  # on-screen words: Spanish
    print("exp:", [round(math.exp(z), 3) for z in logits], "sum", round(sum(math.exp(z) for z in logits), 3))
    for t in (1, 0.5, 2):
        q = softmax(logits, t)
        print(f"softmax T={t}: logits / T = {[z / t for z in logits]}", [round(p, 3) for p in q])
        print(
            f"  each logit point multiplies by e^(1/T) = {math.exp(1 / t):.3f}:",
            round(q[0] / q[1], 3),
            round(q[1] / q[2], 3),
        )
    print("Two classes: softmax([z, 0])[0] = sigmoid(z):", round(softmax([1.5, 0])[0], 6), round(sigmoid(1.5), 6))

    print(f"ln: e^0 = 1 -> ln 1 = 0 · e^-1 = {math.exp(-1):.3f} -> ln {math.exp(-1):.3f} = -1")
    sure, unsure = loss(0.01), loss(1 / 3)
    print(f"sure at 99 % and wrong: {sure:.3f} · unsure, a third each: {unsure:.3f} · ratio {sure / unsure:.2f}")

    a = sigmoid(-5)
    print(f"neuron at z = -5 says {a:.4f}, should say 1")
    print(f"  gradient MSE {mse_gradient(a):.4f} · cross-entropy {ce_gradient(a):.4f}")
    print(
        f"  sigmoid derivative {a * (1 - a):.4f} · MSE: 2 · ({a:.4f} - 1) · {a * (1 - a):.4f} = {mse_gradient(a):.4f}"
    )
    print(f"  CE: d(-ln a)/da = -1/a = {-1 / a:.0f}; times {a * (1 - a):.4f} = {-1 / a * a * (1 - a):.4f}")
    print(f"  MSE is {ce_gradient(a) / mse_gradient(a):.0f} times weaker")
    print(f"  binary CE: -ln a if the answer is yes ({loss(a):.3f}), -ln(1 - a) if it is no ({loss(1 - a):.4f})")
    print(
        "  steps to say > 0.9 (eta = 1): MSE",
        steps_to_learn(mse_gradient),
        "· cross-entropy",
        steps_to_learn(ce_gradient),
    )

    q = softmax(logits)
    print("softmax:", [round(p, 3) for p in q])
    for w, p in zip(words, q, strict=True):
        print(f"  if the right word is {w}: loss -ln {p:.3f} = {loss(p):.3f}")
    for p in (1, 0.5, 0.1, 0.01, 0.001):
        print(f"  p = {p}: loss {loss(p):.3f}")
    print("  one logit point less = probability / e = loss + 1:", [round(loss(p), 3) for p in q])
    terms = [y * loss(p) for p, y in zip(q, [1, 0, 0], strict=True)]
    print(
        "cross-entropy with the list 1 0 0:",
        " + ".join(f"{y} · {loss(p):.3f}" for p, y in zip(q, [1, 0, 0], strict=True)),
        "=",
        round(sum(terms), 3),
    )
    # MSE against the one-hot target, for comparison
    print("  MSE if tejado:", round(sum((p - y) ** 2 for p, y in zip(q, [1, 0, 0], strict=True)), 3))

    grad = [p - y for p, y in zip(q, [1, 0, 0], strict=True)]
    print("gradient softmax + cross-entropy = q - y:", [round(g, 3) for g in grad])
    new = [z - g for z, g in zip(logits, grad, strict=True)]
    q2 = softmax(new)
    print("  one step (eta = 1): logits", [round(z, 3) for z in new], "probs", [round(p, 3) for p in q2])
    print("  loss", round(loss(q[0]), 3), "->", round(loss(q2[0]), 3))

    # the simplest language model: count which word follows which (bigrams)
    corpus = [
        "el gato se subió al tejado",
        "el gato se subió al árbol",
        "el perro se subió al sofá",
        "el gato duerme en el sofá",
    ]
    pairs, firsts = Counter(), Counter()
    for sentence in corpus:
        for a_, b_ in bigrams(sentence):
            pairs[a_, b_] += 1
            firsts[a_] += 1
    after_al = [pairs["al", w] / firsts["al"] for w in ("tejado", "árbol", "sofá")]
    print(
        "entropy after 'al':",
        " + ".join(f"{p:.3f} · {loss(p):.3f}" for p in after_al),
        "=",
        round(sum(p * loss(p) for p in after_al), 3),
    )
    # after «el»: gato 3, perro 1, sofá 1. Give gato p and split the rest evenly: counting (3/5) is the minimum
    after_el = Counter({b: n for (a_, b), n in pairs.items() if a_ == "el"})
    print("after 'el':", dict(after_el))
    for p_gato in (0.5, 0.6, 0.7):
        rest = (1 - p_gato) / 2
        mean_loss = (after_el["gato"] * loss(p_gato) + (after_el["perro"] + after_el["sofá"]) * loss(rest)) / 5
        print(f"  P(gato) = {p_gato}: mean loss after 'el' {mean_loss:.3f}")

    test = "el gato se subió al tejado"
    losses = []
    for a_, b_ in bigrams(test):
        p = pairs[a_, b_] / firsts[a_]
        losses.append(loss(p))
        print(f"  P({b_} | {a_}) = {pairs[a_, b_]}/{firsts[a_]} = {p:.3f}  loss {loss(p):.3f}")
    mean = sum(losses) / len(losses)
    print(f"bigram model on '{test}': mean loss {mean:.3f} · perplexity {math.exp(mean):.3f}")

    unseen = "el perro duerme en el tejado"
    zero = [pair for pair in bigrams(unseen) if pairs[pair] == 0]
    print(f"'{unseen}': never seen {zero} -> probability 0, loss infinite")
    # a sentence the corpus never had, but made of seen pairs, still gets a probability (Short 27)
    novel = "el perro se subió al tejado"
    factors = [Fraction(pairs[pair], firsts[pair[0]]) for pair in bigrams(novel)]
    print(f"'{novel}' (new, all pairs seen): {' · '.join(map(str, factors))} = {math.prod(factors)}")
    unseen_factors = [Fraction(pairs[pair], firsts[pair[0]]) for pair in bigrams(unseen)]
    print(f"'{unseen}': {' · '.join(map(str, unseen_factors))} = {math.prod(unseen_factors)}")

    for v in (3, 50_257):
        print(f"uniform guess over {v:,} tokens: loss ln {v} = {math.log(v):.3f} · perplexity {v:,}")
    print(f"same losses in bits (log2): tejado {-math.log2(q[0]):.3f}")

    # the classic mistake: CrossEntropyLoss applies softmax itself, so it must get logits.
    # Fed probabilities instead, the best possible answer (1, 0, 0) is softmaxed again:
    for k in (2, 3):
        best = softmax([1] + [0] * (k - 1))[0]
        print(
            f"double softmax, {k} classes: a perfect answer still gets p = {best:.3f}, loss {loss(best):.3f} (never 0)"
        )
