# Las funciones de activación con datos reales

Material del vídeo [«Funciones de activación»](https://youtu.be/1Pj2ca79rc0). Python puro, sin instalar
nada.

## El caso: el géiser Old Faithful, en minutos o en horas

En el vídeo, las entradas de XOR solo valían 0 o 1. Los datos reales llegan en sus unidades, y eso
decide dónde cae `z` en la curva de la activación desde el primer paso. Volvemos a `faithful.csv`
(272 erupciones del géiser Old Faithful) y le ponemos una red pequeña: entra la espera, 4 neuronas
ocultas y una sigmoide de salida que dice si la erupción dura más de 3 minutos (175 de las 272 lo
hacen). La misma espera, de tres formas: en minutos (de 43 a 96), en horas (la misma cifra entre 60)
y estandarizada (menos su media, 70,9, y entre su desviación típica, 13,6).

```bash
python faithful_net_case.py
```

Como referencia: decir siempre «larga» falla en 97, y la mejor regla posible con la espera,
«larga si pasa de 65 minutos», falla en 4. Contamos qué parte de las parejas (neurona oculta,
erupción) nace «atascada», con una derivada por debajo del 1 % de su máximo, y cuántos fallos quedan
tras 1000 épocas en 5 redes con pesos iniciales distintos:

| | en minutos | en horas | estandarizada |
|---|---|---|---|
| sigmoide | 91,4 % atascadas · 97 fallos en las 5 | 0,0 % · 4 en las 5 | 0,0 % · 6 en las 5 |
| tanh | 95,1 % · 97 en las 5 | 0,0 % · 6 en las 5 | 0,0 % · 4 en las 5 |
| ReLU | 52,0 % · 97, 97, 97, 175, 97 | 52,0 % · 6, 6, 6, 6, 97 | 50,4 % · 4 en las 5 |

- **En minutos, la sigmoide y tanh no aprenden nada:** 97 fallos, lo mismo que decir siempre
  «larga». Con esperas de 43 a 96, `z` sale de decenas y la neurona nace saturada, en la parte plana
  de la curva que el vídeo mostró con `z` = 4 y 6. Es el gradiente que se desvanece, sin necesidad de
  diez capas: basta una entrada grande.
- **Los mismos datos, divididos entre 60, aprenden:** 4 o 6 fallos, el mínimo posible o casi. La red no
  ha cambiado; solo la unidad.
- **ReLU no se satura, pero tampoco se libra.** En minutos pasa a la neurona de salida sumas de
  decenas, y es la sigmoide de salida la que se satura. Y en horas, una de las cinco redes nace
  muerta: sus cuatro pesos ocultos salieron negativos, todas las esperas son positivas, y ninguna
  neurona se enciende nunca (ejercicio 2).
- **Estandarizada, las tres funciones aprenden.** Con la entrada centrada en 0, cada neurona ReLU se
  apaga para la mitad de las erupciones, pero nunca para todas.

Los ejercicios del descenso de gradiente centraban los datos para dar pasos más largos. Aquí aparece
otra razón: con entradas grandes, las activaciones nacen saturadas o muertas, y la red no aprende. Por
eso las bibliotecas estandarizan las entradas antes de entrenar.

## Ejercicios

1. **La saturación, a mano.** Una neurona sigmoide con `w = 0,5` y `b = 0` recibe una espera de 79
   minutos. Calcula `z`, la sigmoide y su derivada (la salida por uno menos la salida) con la espera
   en minutos, en horas y estandarizada. ¿Cuántas veces menor es la derivada en minutos?
2. **Las ReLU que nacen muertas.** Con la espera en horas, crea 100 redes (semillas 0 a 99) y cuenta
   cuántas nacen con las cuatro neuronas ocultas apagadas para todas las erupciones. ¿Cuántas
   esperarías, si cada peso sale negativo la mitad de las veces? ¿Y con la espera estandarizada?
3. **Leaky ReLU al rescate.** Entrena la red muerta de la semilla 4 (espera en horas) con ReLU y con
   Leaky ReLU (pendiente 0,01 por debajo de cero), 1000 y 5000 épocas. ¿Cuál revive, y por qué?

Las soluciones están en `solutions.py` (`python solutions.py`, tarda unos diez segundos). No las
mires antes de intentarlo. El caso entero (`faithful_net_case.py`) tarda un minuto.

## Soluciones, en resumen

1. En minutos, `z` = 0,5 · 79 = 39,5: la sigmoide da 1,0000 y uno menos eso es 7·10⁻¹⁸, así que la
   derivada es 7·10⁻¹⁸. En horas, `z` = 0,5 · 1,317 = 0,658: sigmoide 0,6589, derivada
   0,6589 · 0,34 = 0,22. Estandarizada, (79 − 70,9) / 13,6 = 0,597 y `z` = 0,299: derivada
   0,5741 · 0,43 = 0,24, casi el máximo de 0,25. En minutos, la culpa que llega a la neurona se
   multiplica por una cifra con diecisiete ceros detrás de la coma: no se mueve.
2. 8 de 100 (semillas 4, 16, 39, 45, 47, 66, 79 y 99). Que los cuatro pesos salgan negativos tiene
   probabilidad 1/2 · 1/2 · 1/2 · 1/2 = 0,0625: unas 6 de cada 100. Estandarizada, ninguna: una
   neurona con peso negativo se enciende con las esperas por debajo de la media.
3. ReLU sigue en 97 fallos con 1000 y con 5000 épocas: con derivada 0 en todas las erupciones, no le
   llega culpa y sus pesos no cambian nunca. Leaky ReLU baja a 4 fallos en las dos: por debajo de
   cero deja pasar un 1 % de la culpa, y eso basta para que los pesos se muevan hacia donde la neurona
   vuelve a encenderse.

Datos: ver `data.md`.
