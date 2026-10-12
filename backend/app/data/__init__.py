"""Fuentes de datos que no son exchanges, la caché compartida y la foto del mercado.

- ``cache``: ``TTLCache`` en memoria; si una consulta falla devuelve el último
  valor marcado como desactualizado.
- ``sentiment``: Fear & Greed de alternative.me.
- ``market``: dominancia y capitalización global (CoinGecko).
- ``news``: titulares por moneda (CryptoPanic con token o feeds RSS).
- ``snapshot``: junta velas, precio, derivados, sentimiento y noticias de una
  moneda en paralelo (``build_snapshot``) o con datos sintéticos
  (``build_offline_snapshot``).
- ``synthetic``: mercado sintético determinista para demos y tests sin red.
"""
