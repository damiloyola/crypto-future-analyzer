"""Acceso a exchanges detrás de una interfaz común.

- ``base``: contratos (``MarketDataSource``, ``OrderExecutor``) y modelos de datos.
- ``binance``: fuente principal de velas y derivados (endpoints públicos).
- ``bybit``: fallback si Binance falla.
- ``nexo``: precio de ejecución (cuando haya acceso) y, en fase 4, órdenes.
- ``registry``: elige la fuente según config.yaml y aplica el fallback.
"""
