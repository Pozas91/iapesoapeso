"""Perceptrón de Rosenblatt desde cero, tal y como aparece en el vídeo.

Ejecuta:  python perceptron.py
Prueba a cambiar los pesos iniciales, eta, o a entrenar con XOR.
"""


def entrenar(datos, w, b, eta=0.2, epocas=20):
    for epoca in range(1, epocas + 1):
        errores = 0
        for x, y in datos:
            z = sum(wi * xi for wi, xi in zip(w, x)) + b
            y_hat = 1 if z > 0 else 0
            e = y - y_hat
            w = [wi + eta * e * xi for wi, xi in zip(w, x)]
            b = b + eta * e
            errores += e != 0
        print(f"época {epoca}: {errores} errores,",
              "w =", [round(wi, 2) for wi in w], "b =", round(b, 2))
        if errores == 0:
            break
    return w, b


AND = [((0, 0), 0), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1)]
XOR = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]

if __name__ == "__main__":
    for eta in (0.01, 0.05, 0.2, 1.0):
        print(f"-- AND con eta = {eta} --")
        entrenar(AND, w=[0.5, 0.5], b=0.2, eta=eta, epocas=30)
    print("-- XOR (nunca converge) --")
    entrenar(XOR, w=[0.5, 0.5], b=0.2, epocas=8)
