"""Soluciones de exercises.md. Ejecuta:  python solutions.py"""

import sys
from pathlib import Path

from iris_case import datos, epoca, fallos

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from perceptron import entrenar  # la función del vídeo

OR = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]
NAND = [((0, 0), 1), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]

print("== 1. OR y NAND con la función del vídeo ==")
for nombre, tabla in (("OR", OR), ("NAND", NAND)):
    print(f"-- {nombre} --")
    entrenar(tabla, w=[0.5, 0.5], b=0.2, eta=0.2, epocas=20)

print("== 2. Setosa contra versicolor: épocas hasta 0 fallos según eta ==")
d = datos("setosa", "versicolor")


def epocas_hasta_cero(eta):
    w, b = [0.5, 0.5], 0.2
    for n in range(1, 101):
        w, b = epoca(d, w, b, eta)
        if fallos(d, w, b) == 0:
            return n


for eta in (0.01, 0.1, 0.2, 1.0):
    print(f"eta = {eta}: {epocas_hasta_cero(eta)} épocas")

print("== 3. Versicolor contra virginica: rectas a mano ==")
d = datos("versicolor", "virginica")
for nombre, w, b in (
    ("ancho > 1,65", [0, 1], -1.65),
    ("largo > 4,8", [1, 0], -4.8),
    ("largo + ancho > 6,4", [1, 1], -6.4),
):
    print(f"virginica si {nombre}: {fallos(d, w, b)} fallos de {len(d)}")
