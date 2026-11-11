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

def detalle_primera_epoca(datos, w, b, eta=0.2):
    """Imprime cada corrección de la primera época con su cuenta."""
    for x, y in datos:
        z = sum(wi * xi for wi, xi in zip(w, x)) + b
        e = y - (1 if z > 0 else 0)
        print(f"x = {x}: z = {z:.1f}, e = {e}")
        for i, xi in enumerate(x):
            if e and xi:
                nuevo = w[i] + eta * e * xi
                print(f"  w{i + 1} = {w[i]:.1f} + {eta}·({e})·{xi} = {nuevo:.1f}")
                w = w[:i] + [nuevo] + w[i + 1:]
        if e:
            print(f"  b = {b:.1f} + {eta}·({e}) = {b + eta * e:.1f}")
            b = b + eta * e
    return w, b


def xor_con_tres_perceptrones(x1, x2):
    """OR y NAND en la primera capa, AND en la segunda: imprime cada suma."""
    z_or = 1 * x1 + 1 * x2 - 0.5
    z_nand = -1 * x1 - 1 * x2 + 1.5
    h1, h2 = int(z_or > 0), int(z_nand > 0)
    z_and = 1 * h1 + 1 * h2 - 1.5
    print(f"x = ({x1}, {x2}): OR z = {z_or}, NAND z = {z_nand}, AND z = {z_and} -> {int(z_and > 0)}")


if __name__ == "__main__":
    print("-- AND, primera época con eta = 0.2, cuenta a cuenta --")
    w, b = detalle_primera_epoca(AND, w=[0.5, 0.5], b=0.2)
    print("z con los pesos finales:",
          [round(w[0] * x1 + w[1] * x2 + b, 2) for (x1, x2), _ in AND])
    print(f"recta: m = -w1/w2 = {-w[0] / w[1]:.2f}, n = -b/w2 = {-b:.1f}/{w[1]:.1f} = {-b / w[1]:.2f}")
    for eta in (0.01, 0.05, 0.2, 1.0):
        print(f"-- AND con eta = {eta} --")
        entrenar(AND, w=[0.5, 0.5], b=0.2, eta=eta, epocas=30)
    print("-- XOR (nunca converge) --")
    entrenar(XOR, w=[0.5, 0.5], b=0.2, epocas=8)
    print("-- XOR con tres perceptrones en dos capas --")
    for (x1, x2), _ in XOR:
        xor_con_tres_perceptrones(x1, x2)
