"""The video's activation functions and their derivatives."""

import math


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def gelu(z):
    return z * 0.5 * (1 + math.erf(z / math.sqrt(2)))  # z · Φ(z)


if __name__ == "__main__":
    first, second = (lambda x: 2 * x + 1), (lambda x: 3 * x + 4)  # two layers with no activation
    print("3·(2x + 1) + 4 = 6x + 7:", all(second(first(x)) == 6 * x + 7 for x in range(-5, 6)))
    print(f"  at x = 2: first {first(2)}, second {second(first(2))}; 6·2 + 7 = {6 * 2 + 7}")  # Short 26
    print("Derivatives at z = 0 (their maximum):")
    print(f"  sigmoid {sigmoid(0) * (1 - sigmoid(0)):.2f} · tanh {1 - math.tanh(0) ** 2:.2f} · relu 1")
    print("Sigmoid derivative at z = 2, 4, 6:", [round(sigmoid(z) * (1 - sigmoid(z)), 4) for z in (2, 4, 6)])
    s = sigmoid(2)
    print(f"  at z = 2: sigmoid {s:.3f} · (1 - {s:.3f}) = {s:.3f} · {1 - s:.3f} = {s * (1 - s):.3f}")
    print("tanh derivative at z = 2, 4:", [round(1 - math.tanh(z) ** 2, 4) for z in (2, 4)])
    print(f"0.25^10 = {0.25**10:.2e}  (1 / {1 / 0.25**10:,.0f})")
    print("tanh(z) = 2·sigmoid(2z) - 1:", round(math.tanh(0.7), 6), round(2 * sigmoid(1.4) - 1, 6))
    print(f"  at z = 1: sigmoid(2) = {s:.3f}, · 2 = {2 * s:.3f}, - 1 = {2 * s - 1:.3f}; tanh(1) = {math.tanh(1):.3f}")
    print("  twice as tall and half as wide: slope at 0 = 0.25 · 2 · 2 =", 0.25 * 2 * 2)

    print("GELU:", {z: round(gelu(z), 3) for z in (-3, -1, -0.75, 0, 1, 3)})
    print("  Phi(z), share of the bell below z:", {z: round(gelu(z) / z, 3) for z in (-3, -1, 1, 3)}, "Phi(0) = 0.5")
    zmin = min((z / 1000 for z in range(-3000, 0)), key=gelu)
    print(f"  GELU minimum at z = {zmin:.2f}: {gelu(zmin):.3f}")
