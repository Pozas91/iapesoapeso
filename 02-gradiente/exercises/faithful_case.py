"""Real case: the gradient descent of the video on the Old Faithful geyser.

From how long an eruption lasted (minutes), predict how long until the next one: the line
wait = m · duration + n, fitted by minimising the mean squared error, as in the video.

With the raw durations the descent is slow and a slightly larger step makes it explode. Centring the
duration (subtracting its mean) fixes both. Run:  python faithful_case.py
"""

import csv
from pathlib import Path

ROWS = list(csv.DictReader(Path(__file__).with_name("faithful.csv").read_text(encoding="utf-8").splitlines()))
DURATION = [float(r["duracion"]) for r in ROWS]
WAIT = [float(r["espera"]) for r in ROWS]


def mse(m, n, xs):
    return sum((m * x + n - y) ** 2 for x, y in zip(xs, WAIT, strict=True)) / len(xs)


def gradient(m, n, xs):
    N = len(xs)
    dm = 2 / N * sum((m * x + n - y) * x for x, y in zip(xs, WAIT, strict=True))
    dn = 2 / N * sum((m * x + n - y) for x, y in zip(xs, WAIT, strict=True))
    return dm, dn


def exact(xs):
    """The best line, solved directly (least squares), to know when the descent has arrived."""
    mean_x, mean_y = sum(xs) / len(xs), sum(WAIT) / len(WAIT)
    m = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, WAIT, strict=True)) / sum((x - mean_x) ** 2 for x in xs)
    return m, mean_y - m * mean_x


def steps_to_arrive(eta, xs, max_steps=100_000):
    """Steps until m and n are within 0.01 of the best line; None if the descent explodes."""
    target_m, target_n = exact(xs)
    m = n = 0.0
    for step in range(1, max_steps + 1):
        dm, dn = gradient(m, n, xs)
        m, n = m - eta * dm, n - eta * dn
        if abs(m) > 1e9:
            return None
        if abs(m - target_m) < 0.01 and abs(n - target_n) < 0.01:
            return step
    return max_steps


if __name__ == "__main__":
    m, n = exact(DURATION)
    print(f"{len(WAIT)} eruptions. Best line: wait = {m:.2f} · duration + {n:.2f} (MSE {mse(m, n, DURATION):.1f})")

    print("-- raw durations (1.6 to 5.1 minutes) --")
    for eta in (0.01, 0.05, 0.07):
        steps = steps_to_arrive(eta, DURATION)
        print(f"eta = {eta}: " + (f"{steps} steps" if steps else "explodes"))

    mean = sum(DURATION) / len(DURATION)
    centred = [x - mean for x in DURATION]
    m_c, n_c = exact(centred)
    print(f"-- centred durations (minus their mean, {mean:.2f} min) --")
    print(f"best line: wait = {m_c:.2f} · (duration - {mean:.2f}) + {n_c:.2f}")
    for eta in (0.1, 0.5, 0.7):
        steps = steps_to_arrive(eta, centred)
        print(f"eta = {eta}: " + (f"{steps} steps" if steps else "explodes"))
