import math
X = [(0, 0), (0, 1), (1, 0), (1, 1)]
Y = [0, 1, 1, 0]
s = lambda z: 1 / (1 + math.exp(-z))
w = [[0.5, 0.6, 0.4], [0.3, 0.3, -0.2]]   # ocultas: w1, w2, sesgo
v = [0.7, -0.5, -0.8]                     # salida: v1, v2, sesgo
for epoca in range(1058):
    gw = [[0, 0, 0], [0, 0, 0]]; gv = [0, 0, 0]
    for (x1, x2), y in zip(X, Y):
        # hacia delante
        h = [s(a*x1 + b*x2 + c) for a, b, c in w]
        out = s(v[0]*h[0] + v[1]*h[1] + v[2])
        # hacia atrás: regla de la cadena
        ds = 2*(out - y) * out*(1 - out) / 4
        for j in range(2):
            gv[j] += ds * h[j]
            d = ds * v[j] * h[j]*(1 - h[j])
            gw[j] = [gw[j][0] + d*x1, gw[j][1] + d*x2, gw[j][2] + d]
        gv[2] += ds
    # un paso de descenso de gradiente (eta = 1)
    w = [[p - g for p, g in zip(wj, gj)] for wj, gj in zip(w, gw)]
    v = [p - g for p, g in zip(v, gv)]
for (x1, x2), y in zip(X, Y):
    h = [s(a*x1 + b*x2 + c) for a, b, c in w]
    print((x1, x2), round(s(v[0]*h[0] + v[1]*h[1] + v[2]), 2))
