"""Selección de fuente de datos con fallback.

Se implementa en el Hito 2.

- ``get_market_data(settings)``: devuelve un ``MarketDataSource`` que prueba
  primero ``exchanges.data_primary`` y, si falla (error de red, 451 por región,
  5xx), ``exchanges.data_fallback``. Registra qué fuente respondió para
  mostrarlo en la UI.
- ``get_execution_price(settings, symbol)``: precio en el exchange donde se
  opera (Nexo). Si Nexo no está disponible devuelve el de Binance con
  ``FetchMeta.warning = "Precio de Nexo no disponible: se usa Binance"``.
"""
