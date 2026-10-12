"""Medias móviles.

Se implementa en el Hito 2.

- ``ema(close, period)``: EMA con semilla SMA, como TradingView.
- ``vwap_daily(df, tz="UTC")``: VWAP que se reinicia al comienzo de cada día
  de mercado, con precio típico (H+L+C)/3.
"""
