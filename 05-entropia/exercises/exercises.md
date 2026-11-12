# La entropía cruzada con datos reales

Material del vídeo [«Softmax y entropía cruzada»](https://youtu.be/-VEBx0dqK40). Python puro, sin
instalar nada.

## El caso: un modelo de letras que lee el Lazarillo

`lazarillo.txt` tiene los dos primeros tratados del «Lazarillo de Tormes» (1554). Hacemos el modelo
de cuentas del vídeo, pero con letras en lugar de palabras: cuenta qué letra sigue a cuál en el
Tratado Primero, y lo medimos con la entropía cruzada y la perplejidad en el Tratado Segundo, un texto
que no ha visto nunca. Antes de contar, todo pasa a minúsculas y lo que no es letra se queda en un
espacio: 32 símbolos (las letras, las vocales con tilde, la ü, la ñ y el espacio).

```bash
python lazarillo_case.py
```

- **Sin saber nada**, repartiendo por igual entre los 32 símbolos: pérdida ln 32 = 3,466 y
  perplejidad 32, como el GPT-2 sin entrenar del vídeo.
- **Solo con la frecuencia de cada letra**, sin mirar la anterior: perplejidad 16,95. Saber que la
  e abunda más que la x ya divide la duda a la mitad.
- **Mirando la letra anterior** (el modelo del vídeo), sobre el mismo texto que contó: pérdida 2,136 y
  perplejidad e^2,136 = 8,47.
- **Sobre el texto nuevo, la perplejidad es infinita.** En el Tratado Segundo hay 44 letras que
  siguen a una pareja que el Tratado Primero nunca tuvo, 29 parejas distintas; las que más se repiten
  son «bd» (4 veces), «ex», «uo», «íc» y «lb» (3 cada una). Basta una sola probabilidad cero para que
  menos su logaritmo sea infinito y arruine la media de 20.526 letras. En el vídeo era un ejemplo
  inventado; con texto real ocurre en el primer intento.
- **Suavizado sumando uno:** se cuenta como si cada pareja posible se hubiera visto una vez más de las
  reales. Después de la q hubo 303 letras, y las 303 fueron u: P(u | q) = (303 + 1) / (303 + 32) =
  0,9075, y P(a | q) = (0 + 1) / (303 + 32) = 0,0030, pequeña pero ya no cero. En el texto nuevo:
  pérdida 2,167 nats (3,126 bits) y perplejidad 8,73. Es como dudar entre unas 9 letras en lugar de
  entre 32.

Lo que el vídeo no podía enseñar con cuatro frases: la perplejidad que importa se mide en texto que el
modelo no ha visto (8,47 en el texto contado, infinita o 8,73 en el nuevo), y el suavizado tiene un
precio: para dar algo a la a tras la q, le quita a la u casi un 10 %.

## Ejercicios

1. **La temperatura.** Usa como logits los logaritmos de las probabilidades del modelo suavizado
   (con T = 1, la softmax devuelve justo esas probabilidades) y genera 70 letras con T = 0,5, 1 y 2,
   eligiendo cada letra según la anterior, como hacía Shannon. ¿Cuánto vale P(u | q) con cada
   temperatura? ¿Qué texto se parece más al castellano?
2. **Cuánto suavizar.** En lugar de sumar 1, suma k = 0,1, 0,01 y 0,001. ¿Con qué k baja más la
   perplejidad en el texto nuevo? ¿Por qué no conviene ni un k grande ni uno diminuto?
3. **Más contexto.** Haz que el modelo mire las dos letras anteriores en lugar de una (un trigrama) y
   mide la perplejidad en el texto contado y en el nuevo, con k = 1, 0,1 y 0,01. ¿Mejora? ¿Por qué la
   distancia entre las dos perplejidades crece?

Las soluciones están en `solutions.py` (`python solutions.py`). No las mires antes de intentarlo.

## Soluciones, en resumen

1. P(u | q) vale 1,000 con T = 0,5, 0,907 con T = 1 y 0,360 con T = 2. Con T = 0,5 sale «co de mo an
   pa paso a y co de de e an mase dimo que to l do…»: sílabas reales, pero repetitivas. Con T = 1,
   «cun lans ardo ll yo alue ha la deje ton és…», más variado. Con T = 2, «czxgompua uis lpeéé
   anídel…»: la temperatura reparte tanto que salen letras que casi nunca van juntas.
2. Perplejidad 8,73 con k = 1, 8,61 con k = 0,1, 8,64 con k = 0,01 y 8,68 con k = 0,001: el mejor es
   k = 0,1, con el que P(u | q) = 0,9899. Un k grande roba demasiado a lo que sí se ha visto; uno
   diminuto deja las parejas nuevas con tan poca probabilidad que cada una cuesta muchísimo.
3. Con k = 0,1, el trigrama baja la perplejidad en el texto nuevo de 8,61 a 6,77. Con k = 1 da 6,89 en
   el texto contado y 7,54 en el nuevo; con 0,1, 5,80 y 6,77; con 0,01, 5,62 y 6,99. Más contexto
   ayuda, pero también hay más combinaciones que el modelo nunca vio: 44 letras del texto nuevo vienen
   tras dos letras que no aparecieron juntas en el Tratado Primero. Cuanto más mira atrás un modelo de
   cuentas, más se aprende el texto de memoria; una red con softmax, en cambio, nunca da cero y
   puede generalizar entre contextos parecidos.

Datos: ver `data.md`.
