"""Caché en memoria con TTL por tipo de dato.

Se implementa en el Hito 2.

- ``TTLCache.get_or_fetch(key, ttl_s, fetcher)``: devuelve el valor cacheado si
  no venció; si no, llama a ``fetcher`` (async) y guarda el resultado con su
  timestamp.
- Guarda la hora de la última actualización exitosa de cada fuente para que
  la UI la muestre ("Binance velas 15m: hace 12 s").
- Si el fetch falla y hay un valor vencido, lo devuelve marcado como
  ``stale`` en vez de romper el análisis.
- Los TTL salen de ``config.yaml > cache_ttl_seconds``.
"""
