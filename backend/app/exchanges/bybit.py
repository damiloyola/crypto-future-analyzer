"""Conector de datos públicos de Bybit v5, usado como fallback.

Se implementa en el Hito 2.

Equivalencias con Binance:

- Velas: ``GET /v5/market/kline?category=linear``.
- Funding: ``GET /v5/market/funding/history`` y ``/v5/market/tickers``.
- Open interest: ``GET /v5/market/open-interest``.
- Ratio long/short: ``GET /v5/market/account-ratio``.

Normaliza todo a los mismos modelos de ``base`` para que el resto del sistema
no sepa de qué exchange vinieron los datos (salvo por ``FetchMeta.source``).
"""
