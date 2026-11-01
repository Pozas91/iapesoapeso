# El perceptrón con datos reales

Material del vídeo [«El perceptrón»](https://youtu.be/qbiq5pwEATQ). Python puro, sin instalar nada.

## El caso: las flores de Iris

`iris.csv` tiene 150 flores de tres especies, medidas en 1936 (Fisher). Usamos dos medidas del pétalo,
el largo y el ancho, como las dos entradas del perceptrón del vídeo.

```bash
python iris_case.py
```

- **Setosa contra versicolor:** en 3 épocas no queda ni un fallo. Los pétalos de setosa miden como
  mucho 1,9 cm de largo y los de versicolor, como poco, 3,0: hay rectas de sobra que las separan.
- **Versicolor contra virginica:** en las últimas 10 de 1000 épocas sigue fallando entre 14 y 31
  flores de 100, y no se para nunca. Esas dos especies se solapan: no existe una recta que las separe, igual que en XOR.

La regla del perceptrón solo termina si esa recta existe. Cuando no existe, no se queda con la mejor:
sigue corrigiendo la última flor que ha fallado y da bandazos.

## Ejercicios

1. **OR y NAND.** Entrena el perceptrón del vídeo (`entrenar` en `../perceptron.py`, con
   `w = [0.5, 0.5]`, `b = 0.2`, `eta = 0.2`) con las tablas de OR y de NAND. ¿Cuántas épocas
   necesita cada una y con qué pesos termina? Antes de ejecutarlo, intenta predecir el signo de los
   pesos de NAND.
2. **¿Importa la tasa de aprendizaje?** Con setosa contra versicolor, cuenta las épocas hasta 0 fallos
   con `eta` = 0,01, 0,1, 0,2 y 1. ¿Por qué apenas cambia, si en el vídeo con AND sí cambiaba?
3. **La recta a mano.** Para versicolor contra virginica, busca una recta `w₁·largo + w₂·ancho + b > 0`
   que falle en 6 flores o menos. Pista: empieza mirando solo el ancho.

Las soluciones están en `solutions.py` (`python solutions.py`). No las mires antes de intentarlo.

## Soluciones, en resumen

1. OR termina en 2 épocas con `w = [0.5, 0.5]`, `b = 0`. NAND, en 5, con `w = [-0.1, -0.1]`,
   `b = 0.2`: pesos negativos, porque encender una entrada acerca la salida a 0.
2. 2, 3, 3 y 3 épocas. Las dos especies están tan separadas que casi cualquier recta razonable vale,
   así que el tamaño del paso apenas importa. En AND solo caben rectas muy concretas.
3. «Virginica si el ancho pasa de 1,65 cm» falla en 6 flores. Mirando las dos medidas,
   «largo + ancho > 6,4» baja a 4. Ninguna llega a 0.

Datos: ver `data.md`.
