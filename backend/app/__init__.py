"""Crypto Future Analyzer: backend.

Fase 1: analizador visual para futuros intradiarios. NO ejecuta operaciones ni
se conecta a ninguna cuenta.

Capas, de abajo hacia arriba (cada una solo depende de las de abajo):

1. ``exchanges``: acceso a datos de mercado detrás de una interfaz común.
2. ``data``: fuentes no-exchange (sentimiento, noticias, mercado global) y caché.
3. ``indicators``: funciones puras sobre DataFrames OHLCV.
4. ``regime``: detector de régimen (HMM con forward filtering).
5. ``scoring``: score de confluencia de -100 a +100.
6. ``risk``: plan de operación, sizing, liquidación, comisiones y perfiles.
7. ``safety``: circuit breakers con poder de veto, lock file y correlación.
8. ``llm``: interpretación y redacción (nunca decide números).
9. ``storage``: SQLite.
10. ``services``: orquesta todo lo anterior en un análisis completo.
11. ``api``: endpoints FastAPI, sin lógica de negocio.

``validation`` reutiliza 3 a 7 para el backtest walk-forward.
"""

__version__ = "0.0.1"
