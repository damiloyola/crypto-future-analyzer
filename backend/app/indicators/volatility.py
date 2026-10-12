"""Indicadores de volatilidad.

Se implementa en el Hito 2.

- ``true_range(df)`` y ``atr(df, period=14)``: ATR de Wilder.
- ``bollinger(close, period=20, std=2)``: media, banda superior e inferior y
  ancho de banda.
- ``daily_range_pct(daily_df, period=14)``: ATR diario como % del precio, el
  "rango típico del día" que acota los objetivos intradiarios.
"""
