"""Tamaño de posición a partir del riesgo en USD.

Se implementa en el Hito 2 (con tests).

- ``stop_distance_pct = |entrada - stop| / entrada``
- ``position_size_usd = riesgo_usd / stop_distance_pct``
- ``leverage = position_size_usd / margen``
- ``margin_for_leverage(size, leverage)`` para el simulador.
"""
