# La campana de Gauss con datos reales

Material del vídeo «07 · La campana de Gauss». Python puro, sin instalar nada.

## El caso: el géiser Old Faithful, que no es una campana

En el vídeo, la campana salía al sumar monedas y dados. Con datos reales conviene comprobarlo antes de
usarla. Volvemos a `faithful.csv` (272 erupciones del géiser Old Faithful): la espera media hasta la
erupción es de 70,90 minutos, con varianza 184,1 y desviación típica 13,57.

```bash
python faithful_bell_case.py
```

- **Las esperas tienen dos jorobas.** El histograma muestra una de esperas cortas (de 50 a 55 minutos,
  32 erupciones) y otra de largas (de 80 a 85, 57), con un valle entre las dos: de 65 a 70 minutos solo
  hay 9. La media, 70,90, cae al borde del valle: es una espera «típica» que ocurre poco, y otra
  vez la media engaña, como en el juego que multiplicaba.
- **Las cuentas de la campana fallan.** A menos de una desviación de la media queda el 62,87 % de las
  esperas, no el 68,27 %; a menos de dos, el 99,63 %, no el 95,45 %. La media y la desviación se pueden
  calcular siempre, pero los porcentajes de la campana solo valen si los datos tienen forma de campana.
- **La media de varias esperas sí acaba en campana.** Sorteamos n esperas al azar (con
  reemplazamiento: cada sorteo puede repetir erupción) y las promediamos, 10.000 veces. Con n = 30, las
  medias forman una sola joroba centrada en 70,90, y a menos de una desviación queda el 68,7 %; a menos
  de dos, el 95,2 %. Ya con n = 5 da 68,0 % y 95,7 %. Promediar es sumar y dividir, y al sumar muchas
  cosas al azar aparece la campana aunque cada una tenga otra forma.
- **Su desviación baja con la raíz de n.** Las varianzas se suman: la suma de n esperas tiene varianza
  n · 184,1, y al dividir entre n la media se queda con 184,1 / n. Su desviación es 13,57 / √n: 9,60
  con n = 2, 6,07 con n = 5 y 2,48 con n = 30. La simulación da 9,67, 6,07 y 2,47.

Lo que el vídeo no podía enseñar con monedas: que la campana no es la forma de los datos sino la de
sus sumas y sus medias. Es lo que permite fiarse de una media aunque cada dato sea irregular, y por qué
una media de más datos se mueve menos.

## Ejercicios

1. **La campana frente a los datos.** Calcula a cuántas desviaciones de la media está una espera de
   90 minutos, y qué parte de las esperas pasaría de 90 si fueran una campana. ¿Cuántas pasan de
   verdad? Haz lo mismo con las esperas entre 65 y 75 minutos, alrededor de la media.
2. **Cuántas esperas promediar.** ¿Cuántas esperas hay que promediar para que la desviación de la
   media baje de 1 minuto? ¿Y de medio minuto? Compruébalo sorteando 10.000 medias.
3. **Cuándo no se suman las varianzas.** Calcula la varianza de la espera, la de la duración y la de
   su suma, erupción a erupción. ¿Se suman? Baraja las duraciones al azar y vuelve a calcularla.
   ¿Qué ha cambiado?

Las soluciones están en `solutions.py` (`python solutions.py`). No las mires antes de intentarlo.

## Soluciones, en resumen

1. z = (90 − 70,90) / 13,57 = 1,41. La campana dice que el 8,0 % pasaría de 90 minutos; de verdad
   pasan 6 de 272, el 2,2 %. Entre 65 y 75, z va de −0,43 a 0,30, y la campana pone ahí el 28,7 %; de
   verdad hay 32, el 11,8 %. Con dos jorobas, la campana se equivoca justo en el centro, donde más
   datos promete.
2. La desviación de la media es 13,57 / √n, que baja de 1 cuando n pasa de 13,57² = 184,1: hacen
   falta 185 esperas (13,57 / √185 = 0,998; la simulación da 1,008, porque 10.000 medias también
   tienen su propio ruido). Para medio minuto, n > 13,57² / 0,5² = 736,6: 737 esperas, cuatro veces
   más (0,500 en teoría, 0,501 sorteando). Para dividir la desviación entre dos hay que multiplicar
   los datos por cuatro.
3. La espera tiene varianza 184,14 y la duración 1,30: sumadas, 185,44. Pero la suma de espera y
   duración tiene varianza 213,29. Las varianzas solo se suman cuando cada cosa sale al azar por su
   cuenta, y aquí no: tras una espera larga viene una erupción larga (en 261 de las 272, las dos
   quedan del mismo lado de su media), así que las dos se alejan a la vez y la suma se dispersa más.
   Al barajar las duraciones se rompe esa relación y la varianza de la suma baja a 183,68, casi la
   suma de las dos (no exacta, porque un barajado concreto también tiene su azar). Es lo que supone
   la cuenta de la neurona del vídeo: entradas y pesos que salen cada uno por su cuenta.

Datos: ver `data.md`.
