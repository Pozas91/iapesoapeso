"""Real case: the activation functions of the video on the Old Faithful geyser.

From the wait (minutes), tell whether the eruption lasts more than 3 minutes. The video's XOR only
had inputs 0 and 1; real data come in their own units, and the size of the input decides whether
the hidden neurons start in their useful zone or already saturated (sigmoid, tanh) or dead (ReLU).

The same 272 waits, three ways: in minutes (43 to 96), in hours (0.7 to 1.6) and standardised
(minus the mean, divided by the standard deviation). Run:  python faithful_net_case.py
"""

import csv
import math
import random
from pathlib import Path

ROWS = list(csv.DictReader(Path(__file__).with_name("faithful.csv").read_text(encoding="utf-8").splitlines()))
WAIT = [float(r["espera"]) for r in ROWS]
Y = [1 if float(r["duracion"]) > 3 else 0 for r in ROWS]
MEAN = sum(WAIT) / len(WAIT)
SD = (sum((w - MEAN) ** 2 for w in WAIT) / len(WAIT)) ** 0.5
SCALES = {
    "minutes": WAIT,
    "hours": [w / 60 for w in WAIT],
    "standardised": [(w - MEAN) / SD for w in WAIT],
}


def sigmoid(z):
    return 1 / (1 + math.exp(-max(-60.0, min(60.0, z))))


# activation and its derivative (from z), and the derivative's maximum
ACTIVATIONS = {
    "sigmoid": (sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z)), 0.25),
    "tanh": (math.tanh, lambda z: 1 - math.tanh(z) ** 2, 1.0),
    "relu": (lambda z: max(z, 0.0), lambda z: 1.0 if z > 0 else 0.0, 1.0),
}
HIDDEN = 4


def create(seed, hidden=HIDDEN):
    """1 input -> hidden -> 1 output. Glorot uniform weights and zero biases, as in ../depth.py."""
    rng = random.Random(seed)
    limit = math.sqrt(6 / (1 + hidden))  # the same for both layers: 1 + hidden = hidden + 1
    w = [rng.uniform(-limit, limit) for _ in range(hidden)]
    v = [rng.uniform(-limit, limit) for _ in range(hidden)]
    return w, [0.0] * hidden, v, 0.0


def stuck_share(net, name, xs):
    """Share of (hidden neuron, eruption) pairs where the derivative is below 1 % of its maximum:
    the blame that reaches that neuron is shrunk more than a hundred times."""
    w, b, _, _ = net
    _, df, top = ACTIVATIONS[name]
    pairs = [df(wj * x + bj) < 0.01 * top for x in xs for wj, bj in zip(w, b, strict=True)]
    return sum(pairs) / len(pairs)


def forward(net, name, x):
    w, b, v, c = net
    f = ACTIVATIONS[name][0]
    z = [wj * x + bj for wj, bj in zip(w, b, strict=True)]
    h = [f(zj) for zj in z]
    return z, h, sigmoid(sum(vj * hj for vj, hj in zip(v, h, strict=True)) + c)


def train(net, name, xs, eta=1.0, epochs=1000):
    """Full-batch gradient descent on the MSE, sigmoid output, as in video 03."""
    w, b, v, c = net
    df = ACTIVATIONS[name][1]
    n = len(xs)
    for _ in range(epochs):
        gw, gb, gv, gc = [0.0] * len(w), [0.0] * len(w), [0.0] * len(w), 0.0
        for x, y in zip(xs, Y, strict=True):
            z, h, s = forward((w, b, v, c), name, x)
            ds = 2 * (s - y) * s * (1 - s) / n
            for j in range(len(w)):
                d = ds * v[j] * df(z[j])
                gw[j] += d * x
                gb[j] += d
                gv[j] += ds * h[j]
            gc += ds
        w = [a - eta * g for a, g in zip(w, gw, strict=True)]
        b = [a - eta * g for a, g in zip(b, gb, strict=True)]
        v = [a - eta * g for a, g in zip(v, gv, strict=True)]
        c -= eta * gc
    return w, b, v, c


def errors(net, name, xs):
    return sum((forward(net, name, x)[2] > 0.5) != y for x, y in zip(xs, Y, strict=True))


def best_threshold():
    """The fewest errors any rule 'long if the wait passes t' can get: the floor for one input."""
    return min((sum((w > t) != y for w, y in zip(WAIT, Y, strict=True)), t) for t in sorted(set(WAIT)))


if __name__ == "__main__":
    print(f"{len(Y)} erupciones, {sum(Y)} largas (> 3 min); decir siempre «larga» falla en {len(Y) - sum(Y)}")
    fewest, t = best_threshold()
    print(f"la mejor regla posible, «larga si la espera pasa de {t:.0f} min»: {fewest} fallos")
    names = {"minutes": "en minutos", "hours": "en horas", "standardised": "estandarizada"}
    for name in ACTIVATIONS:
        print(f"-- {name} --")
        for scale, xs in SCALES.items():
            stuck = sum(stuck_share(create(s), name, xs) for s in range(100)) / 100
            errs = [errors(train(create(s), name, xs), name, xs) for s in range(5)]
            print(f"espera {names[scale]:13s} atascadas al nacer {stuck:5.1%} · fallos tras entrenar (5 redes): {errs}")
