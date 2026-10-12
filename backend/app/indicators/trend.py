"""Clasificación de tendencia por timeframe.

Se implementa en el Hito 2.

- ``classify_trend(close, ema50, ema200)``:
  alcista si precio > EMA50 > EMA200; bajista si precio < EMA50 < EMA200;
  lateral en cualquier otro caso.
"""
