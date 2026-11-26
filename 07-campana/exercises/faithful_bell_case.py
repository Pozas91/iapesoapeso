"""Real case: the bell of the video on the Old Faithful geyser.

The waits between eruptions are not a bell: they have two humps, short waits and long ones. But
the mean of several waits drawn at random adds up many things, and it ends in a bell whose standard
deviation shrinks with the square root of how many we average.
Run:  python faithful_bell_case.py
"""

import math
import random
import sys
from pathlib import Path

ROWS = [
    line.split(",") for line in Path(__file__).with_name("faithful.csv").read_text(encoding="utf-8").splitlines()[1:]
]
WAIT = [float(r[0]) for r in ROWS]
DURATION = [float(r[1]) for r in ROWS]


def mean(xs):
    return sum(xs) / len(xs)


def variance(xs):
    """Mean of the squared distances to the mean, as in the video (dividing by n)."""
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def std(xs):
    return math.sqrt(variance(xs))


def normal_cdf(z):
    """Share of the bell (mean 0, standard deviation 1) below z."""
    return (1 + math.erf(z / math.sqrt(2))) / 2


def within(xs, k):
    """Share of xs at less than k standard deviations from their mean."""
    m, s = mean(xs), std(xs)
    return sum(abs(x - m) < k * s for x in xs) / len(xs)


def histogram(xs, start, width, bins, scale):
    """One line per bin of `width`, with one # for every `scale` values."""
    for i in range(bins):
        low = start + i * width
        count = sum(low <= x < low + width for x in xs)
        print(f"   {low:5.1f}-{low + width:5.1f} {count:5d} {'#' * round(count / scale)}")


def means_of(n, draws=10_000, seed=1):
    """Means of n waits drawn at random (with replacement), `draws` times."""
    rng = random.Random(seed)
    return [mean(rng.choices(WAIT, k=n)) for _ in range(draws)]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # √ and · on any console
    m, s = mean(WAIT), std(WAIT)
    print(f"{len(WAIT)} erupciones · espera media {m:.2f} min · varianza {variance(WAIT):.1f} · desviación {s:.2f}")
    print("Las esperas, de 5 en 5 minutos (cada # son 2 erupciones):")
    histogram(WAIT, 40, 5, 12, 2)
    print("¿Cuántas quedan a menos de 1, 2 y 3 desviaciones de la media? (la campana dice 68,27 %, 95,45 %, 99,73 %)")
    for k in (1, 2, 3):
        print(f"   a menos de {k}: {within(WAIT, k):.2%}   campana {normal_cdf(k) - normal_cdf(-k):.2%}")

    print("Medias de n esperas sorteadas, 10.000 veces:")
    for n in (1, 2, 5, 30):
        ms = means_of(n)
        print(
            f"   n = {n:2d}: media {mean(ms):.2f} · desviación {std(ms):.2f} (teoría {s:.2f} / √{n} = {s / math.sqrt(n):.2f})"
            f" · a menos de 1 desviación {within(ms, 1):.1%}, de 2 {within(ms, 2):.1%}"
        )
    print("Las medias de 30 esperas, de 1 en 1 minuto (cada # son 50 medias):")
    histogram(means_of(30), 65, 1, 12, 50)
