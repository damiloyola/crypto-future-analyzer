"""Caché en memoria con TTL por tipo de dato.

Evita pegarle de más a las APIs (por ejemplo 30 s para precios y 15 min para
noticias, según ``config.yaml > cache_ttl_seconds``) y registra la hora de la
última actualización exitosa de cada fuente para mostrarla en la UI.

Si una consulta falla y hay un valor vencido guardado, se devuelve ese valor
marcado como ``stale`` en vez de romper el análisis. Si no hay nada guardado,
el error se propaga.

Las claves son strings libres; por convención ``"<fuente>:<dato>:<símbolo>[:<tf>]"``,
por ejemplo ``"binance:klines:BTC:15m"``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

T = TypeVar("T")


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class CacheResult(Generic[T]):
    value: T
    fetched_at: datetime  # cuándo se obtuvo el valor (no cuándo se leyó de la caché)
    from_cache: bool
    stale: bool = False
    error: str | None = None  # motivo si es stale


@dataclass
class _Entry:
    value: Any
    fetched_at: datetime


class TTLCache:
    """Caché async con TTL. Una sola consulta en vuelo por clave.

    ``clock`` se puede reemplazar en los tests para controlar el tiempo.
    """

    def __init__(self, clock: Callable[[], datetime] = utcnow) -> None:
        self._clock = clock
        self._entries: dict[str, _Entry] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = self._locks[key] = asyncio.Lock()
        return lock

    def peek(self, key: str) -> CacheResult[Any] | None:
        """Último valor guardado (vencido o no), sin consultar nada."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        return CacheResult(entry.value, entry.fetched_at, from_cache=True)

    async def get_or_fetch(
        self, key: str, ttl_s: float, fetcher: Callable[[], Awaitable[T]]
    ) -> CacheResult[T]:
        """Devuelve el valor cacheado si no venció; si no, llama a ``fetcher``."""
        entry = self._entries.get(key)
        if entry is not None and self._age_s(entry) < ttl_s:
            return CacheResult(entry.value, entry.fetched_at, from_cache=True)

        async with self._lock(key):
            # Otra corrutina pudo haberlo actualizado mientras esperábamos.
            entry = self._entries.get(key)
            if entry is not None and self._age_s(entry) < ttl_s:
                return CacheResult(entry.value, entry.fetched_at, from_cache=True)
            try:
                value = await fetcher()
            except Exception as exc:
                if entry is None:
                    raise
                return CacheResult(
                    entry.value,
                    entry.fetched_at,
                    from_cache=True,
                    stale=True,
                    error=f"{type(exc).__name__}: {exc}",
                )
            now = self._clock()
            self._entries[key] = _Entry(value, now)
            return CacheResult(value, now, from_cache=False)

    def last_updated(self) -> dict[str, datetime]:
        """Hora de la última actualización exitosa por clave (para la UI)."""
        return {k: e.fetched_at for k, e in self._entries.items()}

    def invalidate(self, prefix: str = "") -> None:
        """Borra las entradas cuya clave empieza con ``prefix`` (todas si es vacío)."""
        for k in [k for k in self._entries if k.startswith(prefix)]:
            del self._entries[k]

    def _age_s(self, entry: _Entry) -> float:
        return (self._clock() - entry.fetched_at).total_seconds()


def ttl_for(ttls: dict[str, float], kind: str, default: float = 60.0) -> float:
    """TTL de ``config.yaml > cache_ttl_seconds`` para un tipo de dato (``"klines_15m"``, ...)."""
    return float(ttls.get(kind, default))
