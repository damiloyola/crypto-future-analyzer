"""Titulares de noticias filtrados por moneda.

Se implementa en el Hito 2.

- Si hay ``CRYPTOPANIC_API_KEY`` en ``.env``: CryptoPanic API.
- Si no: RSS de CoinDesk y Cointelegraph (``feedparser``), filtrados con las
  palabras de ``data_sources.news_keywords`` más las macro (Fed, CPI, ETF...).
- ``get_headlines(symbol, limit)`` devuelve título, fuente, link y fecha. El
  impacto (positivo/negativo/neutral) lo clasifica el LLM, no este módulo.
- Caché de 15 minutos.
"""
