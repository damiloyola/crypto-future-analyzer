"""Plan de operación y gestión de riesgo.

Lógica central: riesgo en dólares primero, apalancamiento después. El
apalancamiento no se elige a mano, sale del cálculo. Todo determinista y
testeado; el LLM no puede modificar nada de acá.

Módulos:

- ``sizing``: tamaño de la posición a partir del riesgo en USD (con o sin las
  comisiones dentro del riesgo) y los validadores compartidos.
- ``fees``: comisiones de entrada y salida, R:R bruto y neto, funding.
- ``liquidation``: precio de liquidación (margen aislado y "modelo billetera"),
  chequeo contra el stop, margen mínimo y tamaño máximo para cumplir la regla.
- ``stops``: stop por estructura (soportes y resistencias de 15m y 1h).
- ``profiles``: perfiles de riesgo con el tope de apalancamiento efectivo
  (override manual hasta 50x, advertencia roja por encima de 25x).
- ``regime_adjustments``: ajustes por régimen, confianza baja e inestabilidad
  (bloqueos, score y R:R mínimos, multiplicadores de riesgo con su motivo).
- ``rules``: reglas que fuerzan "No operar", cada una con su motivo y
  sugerencia; también los formateadores de números de los mensajes.
- ``position``: margen y tamaño finales según la decisión del usuario (primero
  subir el margen, después achicar la posición, y "No operar" si el riesgo
  queda debajo del mínimo operable).
- ``plan``: arma el plan completo con todo lo anterior.
- ``leverage_sim`` y ``wallet_math``: paneles educativos (Hito 9).
"""
