"""Persistencia en SQLite (SQLModel).

- ``db``: conexión, creación de tablas y versión del esquema.
- ``models``: las tablas (operaciones cerradas, posiciones abiertas,
  movimientos de la billetera, eventos de alta volatilidad e historial de
  análisis).
- ``repository``: ``Repository``, la única puerta a la base para el resto de
  la app. Valida lo que entra y devuelve dataclasses inmutables con
  ``to_dict()``.

Uso típico::

    repo = Repository.open(default_db_path(settings))
    repo.add_trade(symbol="BTC", side="long", pnl_usd=-12.5, opened_at=..., closed_at=...)
    hoy = repo.list_trades(since=inicio_del_dia)
"""

from app.storage.db import default_db_path, init_db, make_engine, open_db, session_scope
from app.storage.repository import (
    Repository,
    StoredAnalysis,
    StoredPosition,
    StoredTrade,
    StoredVolatilityEvent,
    StoredWalletEvent,
)

__all__ = [
    "Repository",
    "StoredAnalysis",
    "StoredPosition",
    "StoredTrade",
    "StoredVolatilityEvent",
    "StoredWalletEvent",
    "default_db_path",
    "init_db",
    "make_engine",
    "open_db",
    "session_scope",
]
