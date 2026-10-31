"""How many random initialisations get stuck? The same 2-2-1 network as backprop.py,
vectorised with numpy to train thousands of networks at once."""

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
Y = np.array([0, 1, 1, 0], float)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(P):
    # columns of P: w11 w12 b1 w21 w22 b2 v1 v2 c
    h1 = sigmoid(np.outer(P[:, 0], X[:, 0]) + np.outer(P[:, 1], X[:, 1]) + P[:, 2:3])
    h2 = sigmoid(np.outer(P[:, 3], X[:, 0]) + np.outer(P[:, 4], X[:, 1]) + P[:, 5:6])
    return h1, h2, sigmoid(P[:, 6:7] * h1 + P[:, 7:8] * h2 + P[:, 8:9])


def train(P, eta, epochs, tolerance=0.1):
    P = P.copy()
    solved = np.zeros(len(P), int)
    for epoch in range(1, epochs + 1):
        h1, h2, s = forward(P)
        ds = 2 * (s - Y) / 4 * s * (1 - s)
        d1 = ds * P[:, 6:7] * h1 * (1 - h1)
        d2 = ds * P[:, 7:8] * h2 * (1 - h2)
        P -= eta * np.stack(
            [
                d1 @ X[:, 0],
                d1 @ X[:, 1],
                d1.sum(1),
                d2 @ X[:, 0],
                d2 @ X[:, 1],
                d2.sum(1),
                (ds * h1).sum(1),
                (ds * h2).sum(1),
                ds.sum(1),
            ],
            1,
        )
        ok = (np.abs(forward(P)[2] - Y) < tolerance).all(1) & (solved == 0)
        solved[ok] = epoch
    return P, solved


if __name__ == "__main__":
    P0 = np.round(np.random.default_rng(1).uniform(-1, 1, (3000, 9)), 1)
    for eta in (0.5, 1, 2, 4):
        _, r = train(P0, eta, 20000)
        print(f"eta={eta}: solved {np.mean(r > 0):.1%}, median {np.median(r[r > 0]):.0f} epochs")
    # The stuck ones, for much longer
    P, r = train(P0, 1, 20000)
    stuck = P0[r == 0][:200]
    P2, r2 = train(stuck, 1, 200000)
    s = forward(P2)[2]
    print(f"Of 200 stuck, after 200 000 epochs {np.mean(r2 > 0):.0%} are solved")
    print("Typical outputs of a stuck one:", np.round(s[r2 == 0][:3], 3).tolist())
    print("Mean MSE of the stuck ones:", np.round(np.mean((s[r2 == 0] - Y) ** 2), 4))
