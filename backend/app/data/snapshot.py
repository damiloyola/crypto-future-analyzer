"""Foto completa del mercado para una moneda.

Se implementa en el Hito 2.

``build_snapshot(symbol, settings) -> MarketSnapshot`` junta en paralelo:
velas de 5m/15m/1h/4h/1d, ticker de Binance y de Nexo, funding (actual e
histórico 7 días), OI (actual e histórico 7 días), ratio long/short,
Fear & Greed, mercado global y titulares. Cada parte lleva su ``FetchMeta``
para mostrar la hora de la última actualización por fuente.
"""
