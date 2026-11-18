"""Logistic regression from scratch: the numbers of video 15.

Ten students, hours of study and whether they passed. A straight line cannot give probabilities;
a perceptron with a sigmoid (videos 01 and 03), trained with cross-entropy (video 05) by gradient
descent (video 02), can. Then: what its weights mean, why they run away when the data can be
separated by a line, and how the last layer of a language model is the same thing with softmax.
Run:  python logistic.py
"""

import math
import sys

HOURS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
PASSED = [0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
SEPARABLE = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # fail up to 2.5 h, pass from 3 h


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def cross_entropy(w, b, hours, passed):
    """Mean of -ln of the probability given to what really happened (video 05)."""
    total = 0.0
    for x, y in zip(hours, passed, strict=True):
        p = sigmoid(w * x + b)
        total -= math.log(p if y else 1 - p)
    return total / len(hours)


def fit(hours, passed, steps, eta=0.5, l2=0.0, w=0.0, b=0.0):
    """Gradient descent on the cross-entropy. With a sigmoid, the gradient of each example is
    (p - y) times its input, the same `q - y` of video 05. `l2` adds l2·w² to the loss."""
    n = len(hours)
    for _ in range(steps):
        errors = [sigmoid(w * x + b) - y for x, y in zip(hours, passed, strict=True)]
        dw = sum(e * x for e, x in zip(errors, hours, strict=True)) / n + 2 * l2 * w
        db = sum(errors) / n
        w, b = w - eta * dw, b - eta * db
    return w, b


def least_squares(hours, passed):
    """The straight line of video 02 fitted to the 0/1 labels."""
    n = len(hours)
    mx, my = sum(hours) / n, sum(passed) / n
    m = sum((x - mx) * (y - my) for x, y in zip(hours, passed, strict=True)) / sum((x - mx) ** 2 for x in hours)
    return m, my - m * mx


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    # 1. a straight line through 0s and 1s is not a probability
    m, n = least_squares(HOURS, PASSED)
    print(f"1. line: pass = {m:.3f}·hours {n:+.3f}")
    for h in (0.0, 6.0):
        print(f"   at {h} h it predicts {m:.3f}·{h} {n:+.3f} = {m * h + n:.2f}")

    # 2a. the first step of gradient descent, from w = b = 0 (everyone gets p = 0.5)
    failed = sum(x for x, y in zip(HOURS, PASSED, strict=True) if not y)
    passed = sum(x for x, y in zip(HOURS, PASSED, strict=True) if y)
    print(f"2a. step 1: p = {sigmoid(0):.1f} for all, loss {cross_entropy(0, 0, HOURS, PASSED):.3f}")
    print(f"    p − y: +0.5 for the failed ({failed} h in total), −0.5 for the passed ({passed} h)")
    print(f"    dw = ({0.5 * failed} − {0.5 * passed}) / 10 = {(0.5 * failed - 0.5 * passed) / 10:.4f}; db = 0")
    for steps in (1, 10, 100, 1000):
        ws, bs = fit(HOURS, PASSED, steps)
        print(f"    after {steps:>4} steps: w={ws:.4f}  b={bs:.3f}  loss={cross_entropy(ws, bs, HOURS, PASSED):.3f}")

    # 2. logistic regression
    w, b = fit(HOURS, PASSED, 20000)
    print(f"2. logistic: z = {w:.3f}·hours {b:+.3f}; loss {cross_entropy(w, b, HOURS, PASSED):.3f}")
    print(f"   blind guess (p = 0.5): loss {math.log(2):.3f}")
    for x, y in zip(HOURS, PASSED, strict=True):
        print(f"   {x:>3} h  passed={y}  p={sigmoid(w * x + b):.3f}")
    print(f"   50 % at {-b / w:.2f} h; at 0 h p={sigmoid(b):.3f}; at 6 h p={sigmoid(w * 6 + b):.3f}")
    z = w * 5 + b
    print(f"   5 h step by step: z = {w:.3f}·5 {b:+.3f} = {z:.3f}; e^-z = {math.exp(-z):.3f}; 1/(1 + {math.exp(-z):.3f}) = {sigmoid(z):.3f}")
    # 0 h and 6 h step by step: the S curve stays inside 0-1 where the line does not
    for h in (0.0, 6.0):
        zh = w * h + b
        print(f"   {h} h step by step: z = {w:.3f}·{h} {b:+.3f} = {zh:.3f}; e^-z = {math.exp(-zh):.3f}; 1/(1 + {math.exp(-zh):.3f}) = {sigmoid(zh):.1%}")
    losses = [-math.log(sigmoid(w * x + b) if y else 1 - sigmoid(w * x + b)) for x, y in zip(HOURS, PASSED, strict=True)]
    print(f"   loss per student: {[round(v, 3) for v in losses]}")
    misses = losses[HOURS.index(2.0)] + losses[HOURS.index(3.5)]
    print(f"   the two misses: {misses:.3f} of {sum(losses):.3f} = {misses / sum(losses):.0%}")
    right = sum((sigmoid(w * x + b) > 0.5) == y for x, y in zip(HOURS, PASSED, strict=True))
    print(f"   right with threshold 0.5: {right}/10")

    # 3. what a weight means: each hour multiplies the odds p/(1-p) by e^w
    print(f"3. odds ratio per hour e^w = {math.exp(w):.2f}; per half hour {math.exp(w / 2):.2f}")
    print(f"   odds of 3 mean 3 to 1 in favour: p = 3 / (3 + 1) = {3 / 4}")
    for h in (2.0, 3.0):
        p = sigmoid(w * h + b)
        print(f"   {h} h: p={p:.3f}, odds = {p:.3f} / {1 - p:.3f} = {p / (1 - p):.3f}")
    print(f"   1.403 / 0.362 = {math.exp(w):.2f} = e^{w:.3f}")
    print(f"   example: a weight of 0.1 per year multiplies the odds by e^0.1 = {math.exp(0.1):.3f}; ten years, e^1 = {math.e:.3f}")
    # the logit is z, the input of the sigmoid: a straight line in the hours
    for h in (2.0, 3.0):
        p = sigmoid(w * h + b)
        print(f"   logit at {h} h: ln {p / (1 - p):.3f} = {math.log(p / (1 - p)):.3f} = {w:.3f}·{h} {b:+.3f}")

    # 4. separable data: no best weight exists, it keeps growing
    print("4. separable data (fail up to 2.5 h, pass from 3 h):")
    for steps in (100, 1000, 10000, 100000):
        ws, bs = fit(HOURS, SEPARABLE, steps)
        print(f"   {steps:>6} steps: w={ws:.2f}  loss={cross_entropy(ws, bs, HOURS, SEPARABLE):.4f}")
    wr, br = fit(HOURS, SEPARABLE, 100000, l2=0.01)
    print(f"   with l2 = 0.01: w={wr:.2f}, 50 % at {-br / wr:.2f} h")
    print(f"   penalty 0.01·w²: {0.01 * ws**2:.2f} at w={ws:.2f}, {0.01 * wr**2:.3f} at w={wr:.2f}")
    for h in (2.5, 3.0):
        print(f"   {h} h: p={sigmoid(ws * h + bs):.3f} without, {sigmoid(wr * h + br):.3f} with l2")
    # the same probabilities with one more digit, as percentages
    for h in (2.5, 3.0):
        print(f"   {h} h: {sigmoid(ws * h + bs):.2%} without, {sigmoid(wr * h + br):.2%} with l2")

    # 5. the last layer of GPT-2: a hidden state of 768 numbers, a softmax over 50 257 tokens
    print(f"5. GPT-2 output layer: 768 × 50257 = {768 * 50257:,} weights (shared with the token embeddings)")
