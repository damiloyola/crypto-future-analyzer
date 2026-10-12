"""Conector de datos públicos de Binance (sin API key).

Se implementa en el Hito 2. Antes de usar cada endpoint se verifica que siga
vigente; si alguno falla, ``registry`` cae al fallback (Bybit) y la UI avisa.

Endpoints previstos:

- Velas: ``GET /fapi/v1/klines`` (perpetuos USDT-M) o ``/api/v3/klines`` (spot).
- Funding: ``GET /fapi/v1/fundingRate`` (histórico) y ``/fapi/v1/premiumIndex``
  (actual, mark price, próxima hora de funding).
- Open interest: ``GET /fapi/v1/openInterest`` y
  ``/futures/data/openInterestHist`` (histórico, períodos de 5m a 1d).
- Ratio long/short: ``GET /futures/data/globalLongShortAccountRatio``.
- Ticker 24h: ``GET /fapi/v1/ticker/24hr``.

Implementa ``MarketDataSource`` con ``httpx.AsyncClient``, reintentos con
backoff y respeto de los límites de peso de Binance.
"""
