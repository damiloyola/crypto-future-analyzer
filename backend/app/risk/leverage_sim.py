"""Simulador de apalancamiento (panel educativo).

Se implementa en el Hito 9.

Para el plan actual y el mismo riesgo en USD, compara 5x, 10x, 25x, 50x y
100x: margen necesario, tamaño, distancia a liquidación, si la liquidación
queda antes o después del stop, comisiones totales y comisiones como % del
objetivo. Marca en rojo las combinaciones con liquidación dentro del ruido
(< 1 × ATR 15m) o antes del stop.
"""
