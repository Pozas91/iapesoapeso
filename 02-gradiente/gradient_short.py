HORAS = [1, 2, 3, 4]
NOTAS = [3, 6, 8, 9]

def mse(m, n):
    return sum((m*x + n - y)**2 for x, y in zip(HORAS, NOTAS)) / 4

def gradiente(m, n):
    dm = 2/4 * sum((m*x + n - y) * x for x, y in zip(HORAS, NOTAS))
    dn = 2/4 * sum((m*x + n - y) for x, y in zip(HORAS, NOTAS))
    return dm, dn

m, n, eta = 0.0, 0.0, 0.05
for paso in range(400):
    dm, dn = gradiente(m, n)
    m, n = m - eta*dm, n - eta*dn

print(m, n, mse(m, n))   # 2.001 1.498 0.25
