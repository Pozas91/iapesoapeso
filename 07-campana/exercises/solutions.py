"""Solutions of exercises.md. Run:  python solutions.py"""

import math
import random
import sys

from faithful_bell_case import DURATION, WAIT, mean, means_of, normal_cdf, std, variance

sys.stdout.reconfigure(encoding="utf-8")
m, s = mean(WAIT), std(WAIT)

print("== 1. La campana frente a las esperas reales ==")
z = (90 - m) / s
above = sum(w > 90 for w in WAIT)
print(f"z de 90 minutos = (90 - {m:.2f}) / {s:.2f} = {z:.2f}")
print(f"más de 90: campana {1 - normal_cdf(z):.1%} · reales {above} de {len(WAIT)} = {above / len(WAIT):.1%}")
z65, z75 = (65 - m) / s, (75 - m) / s
middle = sum(65 <= w < 75 for w in WAIT)
print(f"entre 65 y 75: z de {z65:.2f} a {z75:.2f} · campana {normal_cdf(z75) - normal_cdf(z65):.1%}", end=" · ")
print(f"reales {middle} de {len(WAIT)} = {middle / len(WAIT):.1%}")

print("== 2. Cuántas esperas hay que promediar para bajar de 1 minuto de desviación ==")
n = math.floor(variance(WAIT)) + 1
print(f"{s:.2f} / √n < 1  ->  n > {s:.2f}² = {variance(WAIT):.1f}  ->  n = {n}")
print(f"{s:.2f} / √n < 0,5  ->  n > {s:.2f}² / 0,5² = {variance(WAIT) / 0.5**2:.1f}")
for k in (n, math.floor(variance(WAIT) / 0.5**2) + 1):
    print(
        f"   n = {k}: teoría {s:.2f} / √{k} = {s / math.sqrt(k):.3f} · 10.000 medias sorteadas {std(means_of(k)):.3f}"
    )

print("== 3. ¿Se suman las varianzas de la espera y la duración? ==")
total = [w + d for w, d in zip(WAIT, DURATION, strict=True)]
print(f"varianza de la espera {variance(WAIT):.2f} + de la duración {variance(DURATION):.2f}", end=" ")
print(f"= {variance(WAIT) + variance(DURATION):.2f} · varianza de la suma {variance(total):.2f}")
long_after_long = sum((w > m) == (d > mean(DURATION)) for w, d in zip(WAIT, DURATION, strict=True))
print(f"erupciones con espera y duración las dos por encima o las dos por debajo de su media: {long_after_long} de 272")
shuffled = DURATION[:]
random.Random(0).shuffle(shuffled)
mixed = [w + d for w, d in zip(WAIT, shuffled, strict=True)]
print(f"con las duraciones barajadas (cada una por su cuenta): varianza de la suma {variance(mixed):.2f}")
