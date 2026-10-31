"""Backpropagation by hand: a 2-2-1 network with sigmoids that learns XOR.

Same error as in video 02: E = (ŷ − y)² per example and the MSE as the mean of the four.
Run `python backprop.py` to see the video's hand calculation and the training.
"""

import math

X = [(0, 0), (0, 1), (1, 0), (1, 1)]
Y = [0, 1, 1, 0]

# Initial weights: small, with one decimal and different between neurons (if they were equal, it would never learn)
INITIAL = {
    "w11": 0.5,
    "w12": 0.6,
    "b1": 0.4,  # hidden neuron 1
    "w21": 0.3,
    "w22": 0.3,
    "b2": -0.2,  # hidden neuron 2
    "v1": 0.7,
    "v2": -0.5,
    "c": -0.8,  # output neuron
}


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def forward(p, x):
    """Forward pass: returns every intermediate value."""
    z1 = p["w11"] * x[0] + p["w12"] * x[1] + p["b1"]
    z2 = p["w21"] * x[0] + p["w22"] * x[1] + p["b2"]
    h1, h2 = sigmoid(z1), sigmoid(z2)
    zs = p["v1"] * h1 + p["v2"] * h2 + p["c"]
    return {"z1": z1, "z2": z2, "h1": h1, "h2": h2, "zs": zs, "s": sigmoid(zs)}


def backward(p, x, y):
    """Backward pass: chain rule from the error to every weight."""
    a = forward(p, x)
    dE_ds = 2 * (a["s"] - y)  # derivative of the error with respect to the output
    ds = dE_ds * a["s"] * (1 - a["s"])  # output δ: σ'(z) = σ(z)(1 − σ(z))
    d1 = ds * p["v1"] * a["h1"] * (1 - a["h1"])  # δ of hidden 1: the error flows back through v1
    d2 = ds * p["v2"] * a["h2"] * (1 - a["h2"])  # δ of hidden 2: the error flows back through v2
    g = {
        "v1": ds * a["h1"],
        "v2": ds * a["h2"],
        "c": ds,
        "w11": d1 * x[0],
        "w12": d1 * x[1],
        "b1": d1,
        "w21": d2 * x[0],
        "w22": d2 * x[1],
        "b2": d2,
    }
    return a, {"dE_ds": dE_ds, "ds": ds, "d1": d1, "d2": d2}, g


def mse(p):
    return sum((forward(p, x)["s"] - y) ** 2 for x, y in zip(X, Y)) / 4


def gradient(p):
    """Gradient of the MSE: the mean of the gradients of the four examples."""
    total = dict.fromkeys(p, 0.0)
    for x, y in zip(X, Y):
        for k, v in backward(p, x, y)[2].items():
            total[k] += v / 4
    return total


def train(p, eta=1.0, epochs=20000, tolerance=0.1, checkpoints=()):
    """Gradient descent with the four examples. Stops when every output is
    within `tolerance` of its target."""
    p = dict(p)
    history = {0: mse(p)}
    for epoch in range(1, epochs + 1):
        g = gradient(p)
        p = {k: p[k] - eta * g[k] for k in p}
        if epoch in checkpoints:
            history[epoch] = mse(p)
        if all(abs(forward(p, x)["s"] - y) < tolerance for x, y in zip(X, Y)):
            history[epoch] = mse(p)
            return p, epoch, history
    return p, None, history


def table(p):
    for x, y in zip(X, Y):
        a = forward(p, x)
        print(f"  {x} -> h1={a['h1']:.2f} h2={a['h2']:.2f} output={a['s']:.3f} (target {y})")


if __name__ == "__main__":
    p = INITIAL
    print("== Before training ==")
    table(p)
    print(f"  MSE: {mse(p):.3f}")

    print("\n== By hand: example (1, 0), target 1 ==")
    x, y = (1, 0), 1
    a, d, g = backward(p, x, y)
    print(f"  z1 = {a['z1']:.2f} -> h1 = {a['h1']:.3f}")
    print(f"  z2 = {a['z2']:.2f} -> h2 = {a['h2']:.3f}")
    print(f"  zs = {a['zs']:.3f} -> output = {a['s']:.3f}   error = {(a['s'] - y) ** 2:.3f}")
    print(f"  dE/ds = {d['dE_ds']:.3f}   delta_s = {d['ds']:.3f}")
    print(f"  delta_1 = {d['d1']:.4f}   delta_2 = {d['d2']:.4f}")
    for k in ("v1", "v2", "c", "w11", "w12", "b1", "w21", "w22", "b2"):
        print(f"  dE/d{k} = {g[k]:+.4f}")
    eta = 1.0
    p1 = {k: p[k] - eta * g[k] for k in p}
    print(f"  One step (eta={eta}) with this example only: output {a['s']:.3f} -> {forward(p1, x)['s']:.3f}")

    print("\n== Training with the four examples (eta = 1) ==")
    final, epochs, history = train(p, eta=1.0, checkpoints=(1, 10, 100, 300, 500, 1000))
    for e, v in history.items():
        print(f"  epoch {e:>5}: MSE {v:.4f}")
    print(f"  Solved in {epochs} epochs")
    table(final)
    print("  Final weights:", {k: round(v, 2) for k, v in final.items()})

    print("\n== Equal weights: the symmetry never breaks ==")
    equal = dict.fromkeys(INITIAL, 0.5)
    final_eq, epochs_eq, _ = train(equal, eta=1.0)
    print(f"  Solved? {epochs_eq}")
    table(final_eq)
