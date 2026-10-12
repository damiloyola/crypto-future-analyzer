"""Soportes y resistencias.

Se implementa en el Hito 2.

- ``find_pivots(df, left, right)``: swing highs/lows. Un pivote en la vela *t*
  recién se conoce en *t + right* (no hay look-ahead al usarlo en backtests).
- ``cluster_levels(prices, atr, mult=0.5)``: agrupa niveles a menos de
  0,5 × ATR y devuelve el precio representativo y cuántos toques tuvo.
- ``nearest_levels(levels, price, n=3)``: los 3 más cercanos arriba y abajo.
- ``previous_day_high_low(df, tz="UTC")``: máximo y mínimo del día anterior.
"""
