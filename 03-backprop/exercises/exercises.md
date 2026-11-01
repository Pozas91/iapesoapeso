# La retropropagación con datos reales

Material del vídeo [«Backpropagation a mano»](https://youtu.be/bUF-Lvocx38). Python puro, sin
instalar nada.

## El caso: las flores que el perceptrón no podía separar

En los ejercicios del vídeo del perceptrón, versicolor contra virginica no tenía solución: ninguna
recta las separa, y la mejor que encontramos fallaba en 4 flores de 100. Ahora les ponemos la red del
vídeo, con capa oculta, sigmoides y retropropagación. Las entradas van centradas y escaladas, como
recomiendan los ejercicios del descenso de gradiente.

```bash
python iris_net_case.py
```

- **Solo con el pétalo** (2 entradas, 2 neuronas ocultas): 6 fallos. La capa oculta puede curvar la
  frontera, pero aquí no gana a la recta.
- **El motivo:** hay dos flores con exactamente el mismo pétalo, 4,8 × 1,8 cm, y son de especies
  distintas. Cualquier modelo que solo mire el pétalo falla al menos en una. El problema no es la
  forma de la frontera: es que las especies se solapan.
- **Con las cuatro medidas** (también el sépalo, 4 entradas): 2 fallos.

En el XOR del vídeo, la capa oculta lo arreglaba todo porque los cuatro puntos estaban limpios y lo
único que fallaba era la recta. Con datos reales, una red más grande no compensa la información que
falta. A veces lo que mejora el modelo no es otra capa, sino otra medida.

## Ejercicios

1. **Más neuronas.** Con solo el pétalo, entrena la red con 4 y con 8 neuronas ocultas. ¿Bajan los
   fallos? ¿Por qué?
2. **Las flores imposibles.** Escribe una función que encuentre las flores con el mismo pétalo y
   distinta especie. ¿Cuál es entonces el mínimo de fallos posible usando solo el pétalo?
3. **Medir fuera.** Aparta una flor de cada cinco (20 en total), entrena con las otras 80 y cuenta los
   fallos en las 20 que la red no ha visto, con el pétalo y con las cuatro medidas. ¿Se puede concluir
   algo con solo 20 flores?

Las soluciones están en `solutions.py` (`python solutions.py`, tarda un minuto). No las mires antes de
intentarlo.

## Soluciones, en resumen

1. 6 fallos con 2, con 4 y con 8 neuronas ocultas (5000 épocas). Las flores que fallan están metidas
   entre las de la otra especie: para acertarlas, la frontera tendría que rodearlas una a una, y eso
   ya sería aprenderse los datos de memoria.
2. Solo hay una pareja: 4,8 × 1,8 cm. Con el pétalo como única información, al menos un fallo es
   inevitable.
3. Con el pétalo, 4 fallos en las 80 de entrenamiento y 1 en las 20 nuevas. Con las cuatro medidas,
   1 y 1. Fuera de los datos de entrenamiento las dos empatan: con 20 flores, un fallo de diferencia
   no permite concluir nada. Para comparar modelos hacen falta más datos de prueba, o repetir el
   reparto varias veces (validación cruzada).

Datos: ver `data.md`.
