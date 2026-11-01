"""Real case: the video's network with the iris flowers that the perceptron could not separate.

Versicolor against virginica (video 01's exercises: no straight line separates them). The same
network as the video, with sigmoids and backpropagation, only with any number of inputs and hidden
neurons. Inputs are standardised (minus the mean, divided by the standard deviation), as video 02's
exercises recommend.

Run:  python iris_net_case.py
"""

import csv
import math
from pathlib import Path

ROWS = [
    r
    for r in csv.DictReader(Path(__file__).with_name("iris.csv").read_text(encoding="utf-8").splitlines())
    if r["especie"] in ("versicolor", "virginica")
]
Y = [1 if r["especie"] == "virginica" else 0 for r in ROWS]
PETAL = ["largo_petalo", "ancho_petalo"]
ALL = ["largo_sepalo", "ancho_sepalo", "largo_petalo", "ancho_petalo"]


def standardised(columns, rows=ROWS):
    values = [[float(r[c]) for c in columns] for r in rows]
    n = len(values)
    means = [sum(v[i] for v in values) / n for i in range(len(columns))]
    sds = [(sum((v[i] - means[i]) ** 2 for v in values) / n) ** 0.5 for i in range(len(columns))]
    return [[(v[i] - means[i]) / sds[i] for i in range(len(columns))] for v in values]


def sigmoid(z):
    return 1 / (1 + math.exp(-max(-60.0, min(60.0, z))))


def initial(inputs, hidden):
    """Small weights with one decimal, different between neurons (equal ones would never learn)."""
    w = [[round(0.1 * ((3 * j + 2 * i) % 9 - 4), 1) for i in range(inputs + 1)] for j in range(hidden)]
    v = [round(0.1 * ((5 * j) % 9 - 4), 1) or 0.3 for j in range(hidden)]
    return w, v, -0.8


def forward(net, x):
    w, v, c = net
    h = [sigmoid(sum(wi * xi for wi, xi in zip(row[:-1], x, strict=True)) + row[-1]) for row in w]
    return h, sigmoid(sum(vj * hj for vj, hj in zip(v, h, strict=True)) + c)


def train(net, xs, ys, eta=1.0, epochs=5000):
    """Gradient descent on the MSE, with the same chain rule as the video's backward()."""
    w, v, c = net
    n = len(xs)
    for _ in range(epochs):
        gw = [[0.0] * len(row) for row in w]
        gv = [0.0] * len(v)
        gc = 0.0
        for x, y in zip(xs, ys, strict=True):
            h, s = forward((w, v, c), x)
            ds = 2 * (s - y) * s * (1 - s)
            for j, hj in enumerate(h):
                d = ds * v[j] * hj * (1 - hj)
                for i, xi in enumerate(x):
                    gw[j][i] += d * xi / n
                gw[j][-1] += d / n
                gv[j] += ds * hj / n
            gc += ds / n
        w = [[a - eta * g for a, g in zip(row, grow, strict=True)] for row, grow in zip(w, gw, strict=True)]
        v = [a - eta * g for a, g in zip(v, gv, strict=True)]
        c -= eta * gc
    return w, v, c


def errors(net, xs, ys):
    return sum((forward(net, x)[1] > 0.5) != y for x, y in zip(xs, ys, strict=True))


def twins():
    """Flowers with the same petal and different species: no model that only sees the petal can get both right."""
    seen = {}
    for r, y in zip(ROWS, Y, strict=True):
        seen.setdefault((r["largo_petalo"], r["ancho_petalo"]), set()).add(y)
    return [k for k, labels in seen.items() if len(labels) == 2]


if __name__ == "__main__":
    print(f"{len(Y)} flowers, versicolor against virginica")
    xs = standardised(PETAL)
    net = train(initial(2, 2), xs, Y)
    print(f"petal only (2 inputs, 2 hidden neurons): {errors(net, xs, Y)} errors")
    print(f"flowers with the same petal and different species: {twins()}")
    xs = standardised(ALL)
    net = train(initial(4, 2), xs, Y)
    print(f"all four measures (4 inputs, 2 hidden neurons): {errors(net, xs, Y)} errors")
