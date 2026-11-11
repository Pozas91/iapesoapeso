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


def nudge(m=0.0, n=1.5, h=0.1):
    """What a derivative is: how much the error changes per unit of a small push of the slope."""
    before, after = mse(m, n), mse(m + h, n)
    return before, after, (after - before) / h


def gradient_sums(m, n):
    """The sums inside each partial derivative: residuals, and residuals times hours."""
    residuals = [m * x + n - y for x, y in zip(HOURS, GRADES, strict=True)]
    weighted = [r * x for r, x in zip(residuals, HOURS, strict=True)]
    return residuals, weighted


def curvatures():
    """Curvatures of the error bowl: eigenvalues of the (constant) Hessian of the MSE.

    H = [[2·mean(x²), 2·mean(x)], [2·mean(x), 2]]; a 2×2 symmetric matrix has a closed formula.
    """
    N = len(HOURS)
    a, b, d = 2 * sum(x * x for x in HOURS) / N, 2 * sum(HOURS) / N, 2.0
    half_trace, root = (a + d) / 2, (((a - d) / 2) ** 2 + b * b) ** 0.5
    return half_trace + root, half_trace - root


def slope_table(n=1.5, slopes=(0, 1, 2, 3, 4)):
    """The error for a few slopes with n fixed: the parabola drawn point by point."""
    return {m: mse(m, n) for m in slopes}


def step_factor(eta):
    """With n fixed, the derivative is 2·mean(x²) = 15 times the distance to the best slope, so a step
    keeps 1 − 15·eta of that distance (negative: it jumps to the other side)."""
    curvature = 2 * sum(x * x for x in HOURS) / len(HOURS)
    return curvature, 1 - eta * curvature


def batch_gradient(students, m=0.0, n=1.5):
    """The slope derivative computed with only some of the students (a mini-batch)."""
    pairs = [(HOURS[i], GRADES[i]) for i in students]
    return 2 / len(pairs) * sum((m * x + n - y) * x for x, y in pairs)


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


def steps_to_optimum(eta=0.05, tolerance=0.01, m=0.0, n=0.0, best=(2.0, 1.5)):
    """How many steps until slope and intercept are both within `tolerance` of the best line."""
    steps = 0
    while abs(m - best[0]) > tolerance or abs(n - best[1]) > tolerance:
        dm, dn = gradient(m, n)
        m, n = m - eta * dm, n - eta * dn
        steps += 1
    return steps


if __name__ == "__main__":
    print("Error of the flat line y = 1.5:", mse(0, 1.5))
    print("Error of the optimal line y = 2x + 1.5:", mse(2, 1.5))
    print("Gradient at m = 0 (n = 1.5):", gradient(0, 1.5)[0])
    residuals, weighted = gradient_sums(0, 1.5)
    print(f"  residuals {residuals}, times hours {weighted}, sum {sum(weighted)}")

    print("\n-- A push of the slope: what a derivative is --")
    for h in (0.1, 0.01):
        before, after, rate = nudge(h=h)
        print(f"  m: 0 -> {h}: MSE {before} -> {after:.4f}, change per unit = {rate:.2f}")

    print("\n-- The parabola with n = 1.5, step by step --")
    for key, value in slope_parabola().items():
        print(f"  {key}: {value}")

    print("\n-- The error for a few slopes (n = 1.5) --")
    print("  " + ", ".join(f"m = {m}: {e}" for m, e in slope_table().items()))

    print("\n-- What each step keeps of the distance to the best slope --")
    for eta in (0.01, 0.05, 0.1, 0.15):
        curvature, factor = step_factor(eta)
        print(f"  eta = {eta:<4}: 1 - {curvature}·{eta} = {factor:.2f}")
    print(f"  limit: 15·eta = 2 -> eta = 2/15 = {2 / 15:.3f}")

    print("\n-- Mini-batches (m = 0, n = 1.5) --")
    for students in ((0, 1), (2, 3), (0, 1, 2, 3)):
        print(f"  students {[s + 1 for s in students]}: slope derivative {batch_gradient(students)}")

    print("\n-- Slope only, four learning rates --")
    for eta in (0.01, 0.05, 0.1, 0.15):
        path = slope_only_descent(eta, 5)
        print(f"  eta = {eta:<4}: " + " -> ".join(f"{m:.3f}" for m in path))

    print("\n-- Slope and intercept together (eta = 0.05, from m = 0, n = 0) --")
    print("  initial MSE:", mse(0, 0), "-- initial gradient:", gradient(0, 0))
    residuals, weighted = gradient_sums(0, 0)
    print(f"  residuals {residuals}, sum {sum(residuals)}; times hours {weighted}, sum {sum(weighted)}")
    descent(0.05, 400)
    print(f"  within 0.01 of m = 2, n = 1.5 after {steps_to_optimum()} steps")

    print("\n-- Why it crawls: the bowl is a long valley --")
    steep, flat = curvatures()
    print(f"  curvatures: {steep:.2f} (side to side) and {flat:.4f} (along), ratio {steep / flat:.1f}")
    for name, curvature in (("steep", steep), ("flat", flat)):
        factor = 1 - 0.05 * curvature
        print(f"  {name}: each step keeps {factor:.3f} of what is left (erases {1 - factor:.1%})")
