# IA peso a peso · código de los vídeos

El código de cada vídeo del canal [IA peso a peso](https://www.youtube.com/@iapesoapeso), tal como
aparece en pantalla. Ejecútalo, cambia los números y mira qué pasa.

| Vídeo | Carpeta |
|---|---|
| 01 · El perceptrón: la neurona que aprendió a aprender | [`01-perceptron`](01-perceptron) |
| 02 · El descenso de gradiente: cómo aprende de verdad una IA | [`02-gradiente`](02-gradiente) |
| 03 · Backpropagation a mano: así aprende una red neuronal | [`03-backprop`](03-backprop) |
| 04 · 04 · Funciones de activación: por qué las redes profundas no aprendían | [`04-activacion`](04-activacion) |
| 05 · 05 · Softmax y entropía cruzada: el número que ChatGPT intenta bajar | [`05-entropia`](05-entropia) |

## Cómo ejecutarlo

Casi todo es Python sin librerías: basta con Python 3.10 o superior.

```bash
python 01-perceptron/perceptron.py
```

Algunos ficheros usan NumPy (`pip install numpy`).

## Licencia

[MIT](LICENSE): úsalo como quieras, citando la fuente.
