"""Contratos comunes para todos los exchanges.

Se implementa en el Hito 2.

Modelos (Pydantic / dataclasses), todos con timestamps en UTC:

- ``Candle`` / DataFrame OHLCV con columnas ``open, high, low, close, volume``
  indexado por ``open_time``.
- ``Ticker``: precio actual, variación 24h, hora de la consulta.
- ``FundingRate``: tasa actual, próxima liquidación de funding, histórico.
- ``OpenInterestPoint``: OI en contratos y en USD.
- ``LongShortRatio``: ratio de cuentas long/short.
- ``FetchMeta``: fuente usada, hora de la última actualización, si vino de caché
  o de un fallback. La UI lo muestra al lado de cada dato.

Interfaces:

- ``MarketDataSource`` (Protocol, async): ``get_klines``, ``get_ticker``,
  ``get_funding_history``, ``get_premium_index``, ``get_open_interest``,
  ``get_open_interest_history``, ``get_long_short_ratio``.
  Binance y Bybit la implementan completa; Nexo solo ``get_ticker``.
- ``OrderExecutor`` (Protocol, async): ``place_order``, ``set_leverage``,
  ``get_positions``, ``close_position``. Existe desde ahora para que la fase 4
  enchufe Nexo sin tocar el resto, pero en la fase 1 no hay ninguna
  implementación habilitada: cualquier llamada lanza ``TradingDisabledError``.
"""
