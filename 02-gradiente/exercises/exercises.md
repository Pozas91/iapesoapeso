# El descenso de gradiente con datos reales

Material del vídeo [«El descenso de gradiente»](https://youtu.be/ITaBkr4OXkY). Python puro, sin
instalar nada.

## El caso: el géiser Old Faithful

`faithful.csv` tiene 272 erupciones del géiser Old Faithful, en Yellowstone: cuánto duró cada una y
cuánto hubo que esperar hasta la siguiente, en minutos. Ajustamos la recta
`espera = m · duración + n` con el mismo descenso de gradiente del vídeo.

```bash
python faithful_case.py
```

- **La recta buena:** espera = 10,73 · duración + 33,47. Cada minuto más de erupción, unos 11 minutos
  más de espera.
- **Con las duraciones tal cual** (de 1,6 a 5,1 minutos), con `eta = 0,05` hacen falta 878 pasos para
  llegar, y con `eta = 0,07` el descenso explota.
- **Centrando la duración** (restándole su media, 3,49 minutos), con `eta = 0,5` llega en 6 pasos.

¿Por qué? Sin centrar, `m` y `n` tiran en la misma dirección: subir la pendiente sube también todas
las predicciones, y hay que compensarlo bajando `n`. El valle del error es una zanja larga y
estrecha, y el descenso rebota de pared a pared. Al centrar, cambiar `m` ya no mueve la predicción
media, el valle se vuelve casi redondo y el paso puede ser diez veces mayor. Por eso las bibliotecas
centran o escalan los datos antes de entrenar.

## Ejercicios

1. **La predicción del guarda.** Con la recta buena, ¿cuánto hay que esperar después de una erupción
   de 2 minutos? ¿Y de una de 4,5?
2. **El límite del paso.** Busca, probando, el mayor `eta` que no explota con las duraciones tal cual
   y con las duraciones centradas. ¿Cuántas veces mayor es el segundo?
3. **¿Qué significa `n`?** Interpreta `n` en las dos versiones de la recta. ¿Cuál tiene sentido físico?

Las soluciones están en `solutions.py` (`python solutions.py`). No las mires antes de intentarlo.

## Soluciones, en resumen

1. 54,9 minutos después de una erupción de 2 minutos, y 81,8 después de una de 4,5.
2. Unos 0,07 sin centrar y unos 0,77 centrando: el paso puede ser unas once veces mayor.
3. Sin centrar, `n` = 33,5 es la espera tras una erupción de 0 minutos, que no existe. Centrando,
   `n` = 70,9 es la espera tras una erupción de duración media: un dato que sí significa algo.

Datos: ver `data.md`.
