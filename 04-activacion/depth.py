"""The same XOR as video 03, but with many hidden layers. How much gradient reaches the first
layer with each activation function, and how many networks learn?"""

from itertools import pairwise

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
Y = np.array([[0], [1], [1], [0]], float)

ACTIVATIONS = {
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda a, z: a * (1 - a)),
    "tanh": (np.tanh, lambda a, z: 1 - a**2),
    "relu": (lambda z: np.maximum(z, 0), lambda a, z: (z > 0).astype(float)),
}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def create(layers, width, rng):
    """Weights with Glorot (uniform) initialisation, the same for the three functions."""
    sizes = [2] + [width] * layers + [1]
    net = []
    for n_in, n_out in pairwise(sizes):
        limit = np.sqrt(6 / (n_in + n_out))
        net.append([rng.uniform(-limit, limit, (n_in, n_out)), np.zeros(n_out)])
    return net


def gradients(net, name):
    f, df = ACTIVATIONS[name]
    a, zs, acts = X, [], [X]
    for W, b in net[:-1]:
        z = a @ W + b
        a = f(z)
        zs.append(z)
        acts.append(a)
    W, b = net[-1]
    out = sigmoid(a @ W + b)  # sigmoid output and MSE, as in video 03
    delta = 2 * (out - Y) / len(X) * out * (1 - out)
    grads = [None] * len(net)
    for i in range(len(net) - 1, -1, -1):
        grads[i] = [acts[i].T @ delta, delta.sum(0)]
        if i:
            delta = (delta @ net[i][0].T) * df(acts[i], zs[i - 1])
    return out, grads


def norm_per_layer(net, name):
    _, g = gradients(net, name)
    return [np.linalg.norm(gW) for gW, _ in g]


def train(net, name, eta, epochs, tolerance=0.1):
    for epoch in range(1, epochs + 1):
        out, g = gradients(net, name)
        if (np.abs(out - Y) < tolerance).all():
            return epoch - 1
        for (W, b), (gW, gb) in zip(net, g, strict=True):
            W -= eta * gW
            b -= eta * gb
    return None


def dead(net, name):
    """A ReLU network is dead if it gives the same output for the four inputs: some whole layer is
    zero on every example and the gradient cannot get through."""
    out, _ = gradients(net, name)
    return np.ptp(out) < 1e-12


if __name__ == "__main__":
    LAYERS, WIDTH, ETA, EPOCHS, NETS = 10, 8, 0.1, 5000, 100
    print(f"{LAYERS} hidden layers of {WIDTH} neurons, XOR, Glorot uniform, eta={ETA}, {NETS} networks")
    sizes = [2] + [WIDTH] * LAYERS + [1]
    terms = [f"{n_in}·{n_out}+{n_out}" for n_in, n_out in pairwise(sizes)]
    print("weights and biases:", " + ".join(terms), "=", sum(n_in * n_out + n_out for n_in, n_out in pairwise(sizes)))
    for name in ACTIVATIONS:
        nets = [create(LAYERS, WIDTH, np.random.default_rng(i)) for i in range(NETS)]
        norms = np.mean([norm_per_layer(n, name) for n in nets], 0)
        print(f"{name:9s} mean gradient per layer (input -> output):", " ".join(f"{n:.1e}" for n in norms))
        print(f"{'':9s} first layer / last hidden layer: {norms[0] / norms[-2]:.1e}")
        r = [train(n, name, ETA, EPOCHS) for n in nets]
        ok = [e for e in r if e is not None]
        print(
            f"{'':9s} solved in {EPOCHS} epochs: {len(ok)}/{NETS}", f"(median {np.median(ok):.0f} epochs)" if ok else ""
        )
    # the sigmoid, with much more patience
    r = [train(create(LAYERS, WIDTH, np.random.default_rng(i)), "sigmoid", 1, 50000) for i in range(20)]
    print("sigmoid, eta=1, 50 000 epochs:", sum(e is not None for e in r), "/ 20")
    # narrow ReLU: dead neurons. At birth each ReLU is off for about half of the inputs, so a layer
    # of n neurons shuts a given input out about once in 2^n
    nets = [create(LAYERS, w, np.random.default_rng(i)) for w in (2, 4, 8, 16) for i in range(1000)]
    off = np.mean([((X[1:] @ W + b) <= 0).mean() for (W, b), *_ in nets])
    print(f"relu at birth, first layer off on (0,1), (1,0), (1,1): {off:.0%} of the time")
    print("a whole layer off for one input:", {w: f"1 in {2**w:,}" for w in (2, 4, 8, 16)})
    for width in (2, 4, 8, 16):
        d = sum(dead(create(LAYERS, width, np.random.default_rng(i)), "relu") for i in range(1000))
        print(f"relu, {LAYERS} layers of {width}: dead at birth {d / 10:.1f} %")
