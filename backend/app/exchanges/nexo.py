"""Conector de Nexo Pro.

Ver ``docs/nexo-api-research.md`` para la investigación completa.

Estado en la fase 1:

- La Nexo Pro API (``https://pro-api.nexo.io/api/v1``) firma TODAS las
  llamadas con API key (headers ``X-API-KEY``, ``X-NONCE``, ``X-SIGNATURE``
  = HMAC-SHA256 del nonce en base64). No hay endpoints públicos de mercado
  confirmados.
- Como en la fase 1 no se usan API keys de exchanges, el conector queda
  preparado pero apagado (``exchanges.nexo.enabled: false``). El precio de
  ejecución sale de Binance y la UI muestra un aviso visible.

Lo que va a tener:

- ``NexoMarketData.get_ticker(symbol)``: precio vía ``GET /quote`` (o el
  endpoint público que se confirme), para comparar con Binance y calcular los
  niveles finales del plan sobre el precio de Nexo.
- ``NexoOrderExecutor`` (fase 4): ``GET /futures/instruments``,
  ``GET /futures/positions``, ``POST /futures/order``. En la fase 1 todos sus
  métodos lanzan ``TradingDisabledError`` sin importar la configuración.
"""
