"""Gradient descent from scratch, exactly as it appears in the video.

We fit the line  grade = m · hours + n  to four students by minimising the mean squared error.
Run:  python gradient.py
"""

HOURS = [1, 2, 3, 4]
GRADES = [3, 6, 8, 9]


def mse(m, n):
    """Mean squared error of the line y = m·x + n over the data."""
    return sum((m * x + n - y) ** 2 for x, y in zip(HOURS, GRADES, strict=True)) / len(HOURS)


def gradient(m, n):
    """Partial derivatives of the MSE with respect to m and n."""
    N = len(HOURS)
    dm = 2 / N * sum((m * x + n - y) * x for x, y in zip(HOURS, GRADES, strict=True))
    dn = 2 / N * sum((m * x + n - y) for x, y in zip(HOURS, GRADES, strict=True))
    return dm, dn


def slope_parabola(n=1.5):
    """With n fixed, MSE(m) = a·m² + b·m + c, rewritten as minimum + a·(m − vertex)².

    Each residual is m·x − gap, where gap = y − n is what the grade still needs above n.
    """
    N = len(HOURS)
    gaps = [y - n for y in GRADES]
    sum_xx = sum(x * x for x in HOURS)
    sum_xg = sum(x * g for x, g in zip(HOURS, gaps, strict=True))
    sum_gg = sum(g * g for g in gaps)
    a, b, c = sum_xx / N, -2 * sum_xg / N, sum_gg / N
    vertex = -b / (2 * a)
    return {
        "gaps": gaps,
        "sum_xx": sum_xx,
        "sum_xg": sum_xg,
        "sum_gg": sum_gg,
        "a": a,
        "b": b,
        "c": c,
        "vertex": vertex,
        "minimum": c - a * vertex**2,
    }


def slope_only_descent(eta, steps, m=0.0, n=1.5):
    """First part of the video: n fixed at 1.5, only the slope is learned."""
    trajectory = [m]
    for _ in range(steps):
        dm, _ = gradient(m, n)
        m = m - eta * dm
        trajectory.append(m)
    return trajectory


def descent(eta, steps, m=0.0, n=0.0):
    """Second part: slope and intercept are learned at the same time."""
    for step in range(1, steps + 1):
        dm, dn = gradient(m, n)
        m, n = m - eta * dm, n - eta * dn
        if step in (1, 2, 10, 100) or step == steps:
            print(f"  step {step:>4}: m = {m:.3f}, n = {n:.3f}, MSE = {mse(m, n):.4f}")
    return m, n


if __name__ == "__main__":
    print("Error of the flat line y = 1.5:", mse(0, 1.5))
    print("Error of the optimal line y = 2x + 1.5:", mse(2, 1.5))
    print("Gradient at m = 0 (n = 1.5):", gradient(0, 1.5)[0])

    print("\n-- The parabola with n = 1.5, step by step --")
    for key, value in slope_parabola().items():
        print(f"  {key}: {value}")

    print("\n-- Slope only, four learning rates --")
    for eta in (0.01, 0.05, 0.1, 0.15):
        path = slope_only_descent(eta, 5)
        print(f"  eta = {eta:<4}: " + " -> ".join(f"{m:.3f}" for m in path))

    print("\n-- Slope and intercept together (eta = 0.05, from m = 0, n = 0) --")
    print("  initial MSE:", mse(0, 0), "-- initial gradient:", gradient(0, 0))
    descent(0.05, 400)
