"""Caso real: el perceptrón del vídeo con las flores de Iris (Fisher, 1936).

Ejecuta:  python iris_case.py

Dos parejas de especies, medidas por el largo y el ancho del pétalo (en cm):
- setosa contra versicolor: hay una recta que las separa, y el perceptrón la encuentra;
- versicolor contra virginica: se solapan, no existe esa recta y el perceptrón nunca se para.
Es la lección de XOR, ahora con datos de verdad.
"""

import csv
from pathlib import Path

FILAS = list(csv.DictReader(Path(__file__).with_name("iris.csv").read_text(encoding="utf-8").splitlines()))


def datos(especie_0, especie_1):
    """((largo, ancho) del pétalo, 0 o 1) de las flores de esas dos especies."""
    return [
        ((float(f["largo_petalo"]), float(f["ancho_petalo"])), 1 if f["especie"] == especie_1 else 0)
        for f in FILAS
        if f["especie"] in (especie_0, especie_1)
    ]


def predecir(x, w, b):
    return 1 if sum(wi * xi for wi, xi in zip(w, x)) + b > 0 else 0


def fallos(datos, w, b):
    return sum(predecir(x, w, b) != y for x, y in datos)


def epoca(datos, w, b, eta):
    """Una pasada por los datos con la regla del vídeo: w ← w + eta·e·x, b ← b + eta·e."""
    for x, y in datos:
        e = y - predecir(x, w, b)
        w = [wi + eta * e * xi for wi, xi in zip(w, x)]
        b = b + eta * e
    return w, b


if __name__ == "__main__":
    print("-- setosa contra versicolor --")
    d = datos("setosa", "versicolor")
    w, b = [0.5, 0.5], 0.2
    for n in range(1, 101):
        w, b = epoca(d, w, b, eta=0.2)
        print(f"época {n}: {fallos(d, w, b)} fallos de {len(d)}, w = {[round(v, 2) for v in w]}, b = {round(b, 2)}")
        if fallos(d, w, b) == 0:
            break

    print("-- versicolor contra virginica, 1000 épocas --")
    d = datos("versicolor", "virginica")
    w, b = [0.5, 0.5], 0.2
    historia = []
    for _ in range(1000):
        w, b = epoca(d, w, b, eta=0.2)
        historia.append(fallos(d, w, b))
    print(f"fallos tras cada una de las últimas 10 épocas: {historia[-10:]}")
    print(f"el mejor momento: {min(historia)} fallos, en la época {historia.index(min(historia)) + 1}")
    print("nunca llega a 0: no hay recta que las separe, y la regla sigue corrigiendo sin parar")
