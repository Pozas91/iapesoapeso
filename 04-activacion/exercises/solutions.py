"""Solutions of exercises.md. Run:  python solutions.py  (takes about ten seconds)"""

import math

from faithful_net_case import ACTIVATIONS, MEAN, SCALES, SD, Y, create, errors, train

print("== 1. La saturación, a mano: w = 0,5, b = 0, una espera de 79 minutos ==")
print(f"estandarizada: (79 - {MEAN:.1f}) / {SD:.1f} = {(79 - MEAN) / SD:.3f}")
for scale, x in (("en minutos", 79), ("en horas", 79 / 60), ("estandarizada", (79 - MEAN) / SD)):
    z = 0.5 * x
    s, rest = 1 / (1 + math.exp(-z)), math.exp(-z) / (1 + math.exp(-z))  # rest = 1 - s, without rounding to 0
    print(
        f"{scale:13s} z = 0,5 · {x:.3f} = {z:.3f} · sigmoide {s:.4f} · derivada {s:.4f} · {rest:.2g} = {s * rest:.2g}"
    )

print("== 2. Redes ReLU que nacen muertas (las 4 ocultas apagadas en todas las erupciones), de 100 ==")
for scale, label in (("hours", "en horas"), ("standardised", "estandarizada")):
    xs = SCALES[scale]
    dead = [s for s in range(100) if all(wj * x + bj <= 0 for x in xs for wj, bj in zip(*create(s)[:2], strict=True))]
    print(f"espera {label}: {len(dead)} muertas al nacer, semillas {dead}")
w = create(4)[0]
print(
    "semilla 4, pesos ocultos:", [round(a, 2) for a in w], "-> con esperas positivas y b = 0, toda z = w · espera < 0"
)
print("probabilidad de que los cuatro pesos salgan negativos: 1/2 · 1/2 · 1/2 · 1/2 =", 0.5**4)

print("== 3. Leaky ReLU (pendiente 0,01 por debajo de cero), semilla 4, espera en horas ==")
ACTIVATIONS["leaky"] = (lambda z: z if z > 0 else 0.01 * z, lambda z: 1.0 if z > 0 else 0.01, 1.0)
xs = SCALES["hours"]
for name in ("relu", "leaky"):
    for epochs in (1000, 5000):
        print(
            f"{name}, {epochs} épocas: {errors(train(create(4), name, xs, epochs=epochs), name, xs)} fallos de {len(Y)}"
        )
